#!/usr/bin/env python3
"""Run one deterministic experimental fault schedule over four ROS 2 cameras."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Optional

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from camera_fault_injector import (
    CAMERA_NAMES,
    GAUSSIAN_NOISE_GENERATOR,
    CameraFaultConfig,
    CameraFaultModel,
    image_stamp_ns,
)


SCHEMA_VERSION = 1
SCHEMA_SCOPE = "experimental_local"
TIME_BASIS = "per_camera_first_input_header_stamp"
PIXEL_AND_HEADER_FAULTS = {
    "overexposure",
    "underexposure",
    "freeze",
    "timestamp_shift",
    "dropout",
    "occlusion",
    "gaussian_noise",
}
TRANSPORT_FAULTS = {
    "delivery_delay",
    "seeded_jitter",
    "bounded_reorder",
    "burst_dropout",
}
SCENARIO_FAULT_TYPES = PIXEL_AND_HEADER_FAULTS | TRANSPORT_FAULTS
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOPIC_PATTERN = re.compile(
    r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$"
)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_keys(
    value: dict[str, Any],
    label: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ValueError(f"{label} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)}")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be a safe nonempty identifier")
    return value


def _topic(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _TOPIC_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must be an absolute ROS topic")
    return value


def _derived_seed(scenario_seed: int, camera_id: int, window_id: str) -> int:
    identity = f"{scenario_seed}:{camera_id}:{window_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def _validated_parameters(
    fault_type: str,
    parameters_value: Any,
    scenario_seed: int,
    camera_id: int,
    window_id: str,
) -> dict[str, Any]:
    parameters = _mapping(parameters_value, f"window {window_id} parameters")

    if fault_type in {
        "overexposure",
        "underexposure",
        "freeze",
        "dropout",
    }:
        _exact_keys(parameters, f"window {window_id} parameters", set())
        return {}
    if fault_type == "timestamp_shift":
        _exact_keys(
            parameters,
            f"window {window_id} parameters",
            {"timestamp_shift_ms"},
        )
        shift_ms = _finite_number(
            parameters["timestamp_shift_ms"], "timestamp_shift_ms"
        )
        if abs(shift_ms) < 1.0e-12:
            raise ValueError("timestamp_shift_ms must be nonzero")
        return {"timestamp_shift_ms": shift_ms}
    if fault_type == "occlusion":
        _exact_keys(
            parameters,
            f"window {window_id} parameters",
            {"occlusion_mode"},
        )
        mode = parameters["occlusion_mode"]
        if mode not in {"black", "fixed_mask"}:
            raise ValueError("occlusion_mode must be black or fixed_mask")
        return {"occlusion_mode": mode}
    if fault_type == "gaussian_noise":
        _exact_keys(
            parameters,
            f"window {window_id} parameters",
            {"noise_stddev"},
        )
        stddev = _finite_number(
            parameters["noise_stddev"], "noise_stddev"
        )
        if stddev <= 0.0:
            raise ValueError("noise_stddev must be positive")
        return {
            "generator": GAUSSIAN_NOISE_GENERATOR,
            "noise_stddev": stddev,
            "seed": _derived_seed(scenario_seed, camera_id, window_id),
        }
    if fault_type == "delivery_delay":
        _exact_keys(
            parameters,
            f"window {window_id} parameters",
            {"delay_ms"},
        )
        delay_ms = _finite_number(parameters["delay_ms"], "delay_ms")
        if delay_ms <= 0.0:
            raise ValueError("delay_ms must be positive")
        return {"delay_ms": delay_ms}
    if fault_type == "seeded_jitter":
        _exact_keys(
            parameters,
            f"window {window_id} parameters",
            {"max_delay_ms"},
        )
        max_delay_ms = _finite_number(
            parameters["max_delay_ms"], "max_delay_ms"
        )
        if max_delay_ms <= 0.0:
            raise ValueError("max_delay_ms must be positive")
        return {
            "max_delay_ms": max_delay_ms,
            "seed": _derived_seed(scenario_seed, camera_id, window_id),
        }
    if fault_type == "bounded_reorder":
        _exact_keys(
            parameters,
            f"window {window_id} parameters",
            {"depth_frames"},
        )
        depth_frames = _positive_int(
            parameters["depth_frames"], "depth_frames"
        )
        if depth_frames < 2 or depth_frames > 64:
            raise ValueError("depth_frames must be in [2, 64]")
        return {
            "depth_frames": depth_frames,
            "seed": _derived_seed(scenario_seed, camera_id, window_id),
        }
    if fault_type == "burst_dropout":
        _exact_keys(
            parameters,
            f"window {window_id} parameters",
            {"period_frames", "drop_frames"},
        )
        period_frames = _positive_int(
            parameters["period_frames"], "period_frames"
        )
        drop_frames = _positive_int(
            parameters["drop_frames"], "drop_frames"
        )
        if drop_frames > period_frames:
            raise ValueError("drop_frames cannot exceed period_frames")
        return {
            "period_frames": period_frames,
            "drop_frames": drop_frames,
            "seed": _derived_seed(scenario_seed, camera_id, window_id),
        }
    raise ValueError(f"unsupported fault type: {fault_type}")


@dataclass(frozen=True)
class ScenarioWindow:
    window_id: str
    fault_type: str
    start_sec: float
    duration_sec: float
    parameters: dict[str, Any]

    @property
    def end_sec(self) -> float:
        return self.start_sec + self.duration_sec

    def to_effective_dict(self) -> dict[str, Any]:
        return {
            "id": self.window_id,
            "type": self.fault_type,
            "start_sec": self.start_sec,
            "duration_sec": self.duration_sec,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class CameraScenario:
    camera_id: int
    name: str
    input_topic: str
    output_topic: str
    windows: tuple[ScenarioWindow, ...]

    def to_effective_dict(self) -> dict[str, Any]:
        return {
            "id": self.camera_id,
            "name": self.name,
            "input_topic": self.input_topic,
            "output_topic": self.output_topic,
            "windows": [
                window.to_effective_dict() for window in self.windows
            ],
        }


@dataclass(frozen=True)
class FaultScenario:
    scenario_id: str
    seed: int
    duration_sec: float
    cameras: tuple[CameraScenario, ...]

    def to_effective_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "duration_sec": self.duration_sec,
            "cameras": [
                camera.to_effective_dict() for camera in self.cameras
            ],
        }


def parse_scenario(document: Any) -> FaultScenario:
    root = _mapping(document, "scenario")
    _exact_keys(
        root,
        "scenario",
        {
            "schema_version",
            "scenario_id",
            "seed",
            "duration_sec",
            "cameras",
        },
    )
    if (
        isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("schema_version must be integer 1")
    scenario_id = _identifier(root["scenario_id"], "scenario_id")
    seed = root["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > (1 << 63) - 1
    ):
        raise ValueError("seed must be a nonnegative signed 64-bit integer")
    duration_sec = _finite_number(root["duration_sec"], "duration_sec")
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")

    camera_values = root["cameras"]
    if not isinstance(camera_values, list) or len(camera_values) != 4:
        raise ValueError("cameras must contain exactly four entries")

    cameras: list[CameraScenario] = []
    camera_ids: set[int] = set()
    camera_names: set[str] = set()
    input_topics: set[str] = set()
    output_topics: set[str] = set()
    window_ids: set[str] = set()
    for camera_index, camera_value in enumerate(camera_values):
        camera = _mapping(camera_value, f"camera {camera_index}")
        _exact_keys(
            camera,
            f"camera {camera_index}",
            {"id", "name", "input_topic", "output_topic", "windows"},
        )
        camera_id = camera["id"]
        if (
            isinstance(camera_id, bool)
            or not isinstance(camera_id, int)
            or camera_id < 0
            or camera_id >= len(CAMERA_NAMES)
        ):
            raise ValueError("camera id must be in [0, 3]")
        name = camera["name"]
        if name != CAMERA_NAMES[camera_id]:
            raise ValueError("camera name must match canonical camera id")
        input_topic = _topic(
            camera["input_topic"], f"camera {camera_id} input_topic"
        )
        output_topic = _topic(
            camera["output_topic"], f"camera {camera_id} output_topic"
        )
        if input_topic == output_topic:
            raise ValueError("camera input and output topics must differ")
        if camera_id in camera_ids:
            raise ValueError(f"duplicate camera id: {camera_id}")
        if name in camera_names:
            raise ValueError(f"duplicate camera name: {name}")
        if input_topic in input_topics:
            raise ValueError(f"duplicate input topic: {input_topic}")
        if output_topic in output_topics:
            raise ValueError(f"duplicate output topic: {output_topic}")
        camera_ids.add(camera_id)
        camera_names.add(name)
        input_topics.add(input_topic)
        output_topics.add(output_topic)

        window_values = camera["windows"]
        if not isinstance(window_values, list):
            raise ValueError(f"camera {camera_id} windows must be an array")
        windows: list[ScenarioWindow] = []
        for window_index, window_value in enumerate(window_values):
            window = _mapping(
                window_value, f"camera {camera_id} window {window_index}"
            )
            _exact_keys(
                window,
                f"camera {camera_id} window {window_index}",
                {
                    "id",
                    "type",
                    "start_sec",
                    "duration_sec",
                    "parameters",
                },
            )
            window_id = _identifier(window["id"], "window id")
            if window_id in window_ids:
                raise ValueError(f"duplicate window id: {window_id}")
            fault_type = window["type"]
            if not isinstance(fault_type, str):
                raise ValueError("window type must be a string")
            if fault_type not in SCENARIO_FAULT_TYPES:
                raise ValueError(f"unsupported fault type: {fault_type}")
            start_sec = _finite_number(
                window["start_sec"], f"window {window_id} start_sec"
            )
            window_duration_sec = _finite_number(
                window["duration_sec"], f"window {window_id} duration_sec"
            )
            if start_sec < 0.0:
                raise ValueError("window start_sec must be nonnegative")
            if window_duration_sec <= 0.0:
                raise ValueError("window duration_sec must be positive")
            if start_sec + window_duration_sec > duration_sec:
                raise ValueError(
                    f"window {window_id} exceeds scenario duration"
                )
            parameters = _validated_parameters(
                fault_type,
                window["parameters"],
                seed,
                camera_id,
                window_id,
            )
            windows.append(
                ScenarioWindow(
                    window_id,
                    fault_type,
                    start_sec,
                    window_duration_sec,
                    parameters,
                )
            )
            window_ids.add(window_id)

        windows.sort(key=lambda item: (item.start_sec, item.window_id))
        for previous, current in zip(windows, windows[1:]):
            if current.start_sec < previous.end_sec:
                raise ValueError(
                    "same-camera fault windows must not overlap: "
                    f"{previous.window_id}, {current.window_id}"
                )
        cameras.append(
            CameraScenario(
                camera_id,
                name,
                input_topic,
                output_topic,
                tuple(windows),
            )
        )

    cameras.sort(key=lambda item: item.camera_id)
    if [camera.camera_id for camera in cameras] != list(range(4)):
        raise ValueError("camera ids must be exactly 0, 1, 2, 3")
    return FaultScenario(
        scenario_id, seed, duration_sec, tuple(cameras)
    )


def load_scenario(path: Path) -> FaultScenario:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load scenario: {error}") from error
    return parse_scenario(document)


@dataclass
class WindowCounters:
    matched: int = 0
    modified: int = 0
    dropped: int = 0
    scheduled: int = 0
    published: int = 0
    delayed_released: int = 0
    reordered_released: int = 0
    first_source_stamp_ns: Optional[int] = None
    last_source_stamp_ns: Optional[int] = None


@dataclass
class CameraCounters:
    received: int = 0
    published: int = 0
    dropped: int = 0
    delayed_released: int = 0
    reordered_publications: int = 0


@dataclass
class Publication:
    message: Image
    window_id: Optional[str]
    delayed: bool = False
    reordered: bool = False


@dataclass
class _WindowRuntime:
    config: ScenarioWindow
    camera: CameraScenario
    counters: WindowCounters = field(default_factory=WindowCounters)
    fault_model: Optional[CameraFaultModel] = None
    random_source: Optional[random.Random] = None
    reorder_buffer: list[Image] = field(default_factory=list)
    publication_receipts: list[dict[str, int]] = field(
        default_factory=list
    )
    burst_index: int = 0
    burst_phase: int = 0

    def __post_init__(self) -> None:
        parameters = self.config.parameters
        if self.config.fault_type in PIXEL_AND_HEADER_FAULTS:
            self.fault_model = CameraFaultModel(
                CameraFaultConfig(
                    camera_id=self.camera.camera_id,
                    camera_name=self.camera.name,
                    fault_type=self.config.fault_type,
                    start_sec=0.0,
                    duration_sec=1.0e12,
                    timestamp_shift_ms=parameters.get(
                        "timestamp_shift_ms", 0.0
                    ),
                    occlusion_mode=parameters.get(
                        "occlusion_mode", "black"
                    ),
                    noise_stddev=parameters.get("noise_stddev", 10.0),
                    noise_seed=parameters.get("seed", 0),
                )
            )
        if self.config.fault_type in {
            "seeded_jitter",
            "bounded_reorder",
            "burst_dropout",
        }:
            self.random_source = random.Random(parameters["seed"])
        if self.config.fault_type == "burst_dropout":
            assert self.random_source is not None
            self.burst_phase = self.random_source.randrange(
                parameters["period_frames"]
            )

    def observe(self, source_stamp_ns: int) -> None:
        self.counters.matched += 1
        if self.counters.first_source_stamp_ns is None:
            self.counters.first_source_stamp_ns = source_stamp_ns
        self.counters.last_source_stamp_ns = source_stamp_ns


class CameraScenarioModel:
    """Pure per-camera transformer and delivery scheduler."""

    def __init__(self, camera: CameraScenario) -> None:
        self.camera = camera
        self.counters = CameraCounters()
        self.first_source_stamp_ns: Optional[int] = None
        self.last_source_stamp_ns: Optional[int] = None
        self._last_published_source_stamp_ns: Optional[int] = None
        self._windows = [
            _WindowRuntime(window, camera) for window in camera.windows
        ]
        self._window_by_id = {
            runtime.config.window_id: runtime
            for runtime in self._windows
        }
        self._pending: list[tuple[int, int, Publication]] = []
        self._pending_sequence = 0

    def _active_window(
        self, relative_sec: float
    ) -> Optional[_WindowRuntime]:
        for runtime in self._windows:
            if (
                relative_sec >= runtime.config.start_sec
                and relative_sec < runtime.config.end_sec
            ):
                return runtime
        return None

    def _release_inactive_reorder(
        self, active_window_id: Optional[str]
    ) -> list[Publication]:
        publications: list[Publication] = []
        for runtime in self._windows:
            if (
                runtime.config.fault_type != "bounded_reorder"
                or not runtime.reorder_buffer
                or runtime.config.window_id == active_window_id
            ):
                continue
            publications.extend(self._release_reorder(runtime))
        return publications

    @staticmethod
    def _release_reorder(
        runtime: _WindowRuntime,
    ) -> list[Publication]:
        assert runtime.random_source is not None
        buffered = runtime.reorder_buffer
        if len(buffered) > 1:
            offset = 1 + runtime.random_source.randrange(len(buffered) - 1)
            buffered = buffered[offset:] + buffered[:offset]
        runtime.reorder_buffer = []
        runtime.counters.reordered_released += len(buffered)
        return [
            Publication(
                message,
                runtime.config.window_id,
                reordered=True,
            )
            for message in buffered
        ]

    def _schedule(
        self,
        message: Image,
        runtime: _WindowRuntime,
        due_monotonic_ns: int,
    ) -> None:
        publication = Publication(
            message, runtime.config.window_id, delayed=True
        )
        heapq.heappush(
            self._pending,
            (due_monotonic_ns, self._pending_sequence, publication),
        )
        self._pending_sequence += 1
        runtime.counters.scheduled += 1

    def poll(self, monotonic_ns: int) -> list[Publication]:
        publications: list[Publication] = []
        while self._pending and self._pending[0][0] <= monotonic_ns:
            _, _, publication = heapq.heappop(self._pending)
            publications.append(publication)
            self.counters.delayed_released += 1
            if publication.window_id is not None:
                self._window_by_id[
                    publication.window_id
                ].counters.delayed_released += 1
        return publications

    def process(
        self, message: Image, arrival_monotonic_ns: int
    ) -> list[Publication]:
        publications = self.poll(arrival_monotonic_ns)
        source_stamp_ns = image_stamp_ns(message)
        if self.first_source_stamp_ns is None:
            self.first_source_stamp_ns = source_stamp_ns
        self.last_source_stamp_ns = source_stamp_ns
        self.counters.received += 1
        relative_sec = (
            source_stamp_ns - self.first_source_stamp_ns
        ) * 1.0e-9
        runtime = self._active_window(relative_sec)
        active_window_id = (
            None if runtime is None else runtime.config.window_id
        )
        publications.extend(
            self._release_inactive_reorder(active_window_id)
        )
        if runtime is None:
            publications.append(Publication(message, None))
            return publications

        runtime.observe(source_stamp_ns)
        fault_type = runtime.config.fault_type
        if fault_type in PIXEL_AND_HEADER_FAULTS:
            assert runtime.fault_model is not None
            result = runtime.fault_model.process(message)
            if result.message is None:
                runtime.counters.dropped += 1
                self.counters.dropped += 1
            else:
                runtime.counters.modified += int(
                    fault_type != "dropout"
                )
                publications.append(
                    Publication(result.message, runtime.config.window_id)
                )
            return publications

        if fault_type == "delivery_delay":
            delay_ns = round(
                runtime.config.parameters["delay_ms"] * 1.0e6
            )
            self._schedule(
                message, runtime, arrival_monotonic_ns + delay_ns
            )
            return publications

        if fault_type == "seeded_jitter":
            assert runtime.random_source is not None
            delay_ns = round(
                runtime.random_source.uniform(
                    0.0, runtime.config.parameters["max_delay_ms"]
                )
                * 1.0e6
            )
            self._schedule(
                message, runtime, arrival_monotonic_ns + delay_ns
            )
            return publications

        if fault_type == "bounded_reorder":
            runtime.reorder_buffer.append(message)
            depth_frames = runtime.config.parameters["depth_frames"]
            if len(runtime.reorder_buffer) >= depth_frames:
                publications.extend(self._release_reorder(runtime))
            return publications

        if fault_type == "burst_dropout":
            period_frames = runtime.config.parameters["period_frames"]
            drop_frames = runtime.config.parameters["drop_frames"]
            should_drop = (
                runtime.burst_index + runtime.burst_phase
            ) % period_frames < drop_frames
            runtime.burst_index += 1
            if should_drop:
                runtime.counters.dropped += 1
                self.counters.dropped += 1
            else:
                publications.append(
                    Publication(message, runtime.config.window_id)
                )
            return publications

        raise AssertionError(f"unhandled fault type: {fault_type}")

    def record_publication(
        self,
        publication: Publication,
        published_monotonic_ns: Optional[int] = None,
    ) -> None:
        source_stamp_ns = image_stamp_ns(publication.message)
        if (
            self._last_published_source_stamp_ns is not None
            and source_stamp_ns < self._last_published_source_stamp_ns
        ):
            self.counters.reordered_publications += 1
        self._last_published_source_stamp_ns = source_stamp_ns
        self.counters.published += 1
        if publication.window_id is not None:
            runtime = self._window_by_id[publication.window_id]
            runtime.counters.published += 1
            if published_monotonic_ns is not None:
                runtime.publication_receipts.append(
                    {
                        "source_stamp_ns": source_stamp_ns,
                        "published_monotonic_ns": int(
                            published_monotonic_ns
                        ),
                    }
                )

    def pending_count(self) -> int:
        return len(self._pending) + sum(
            len(runtime.reorder_buffer) for runtime in self._windows
        )

    def summary(self) -> dict[str, Any]:
        windows = []
        for runtime in self._windows:
            source_origin_ns = self.first_source_stamp_ns
            windows.append(
                {
                    "id": runtime.config.window_id,
                    "type": runtime.config.fault_type,
                    "relative_sec": {
                        "start": runtime.config.start_sec,
                        "end": runtime.config.end_sec,
                    },
                    "source_stamp_ns": {
                        "start": (
                            None
                            if source_origin_ns is None
                            else source_origin_ns
                            + round(
                                runtime.config.start_sec * 1.0e9
                            )
                        ),
                        "end": (
                            None
                            if source_origin_ns is None
                            else source_origin_ns
                            + round(runtime.config.end_sec * 1.0e9)
                        ),
                    },
                    "parameters": runtime.config.parameters,
                    "counters": asdict(runtime.counters),
                    "publication_receipts": (
                        runtime.publication_receipts
                    ),
                }
            )
        counters = asdict(self.counters)
        counters["pending"] = self.pending_count()
        return {
            "id": self.camera.camera_id,
            "name": self.camera.name,
            "input_topic": self.camera.input_topic,
            "output_topic": self.camera.output_topic,
            "first_source_stamp_ns": self.first_source_stamp_ns,
            "last_source_stamp_ns": self.last_source_stamp_ns,
            "counters": counters,
            "windows": windows,
        }


class MultiCameraFaultModel:
    """Pure four-camera scenario model used by the ROS adapter."""

    def __init__(self, scenario: FaultScenario, run_id: str) -> None:
        self.scenario = scenario
        self.run_id = _identifier(run_id, "run_id")
        self.cameras = {
            camera.camera_id: CameraScenarioModel(camera)
            for camera in scenario.cameras
        }
        effective_scenario = scenario.to_effective_dict()
        canonical = json.dumps(
            effective_scenario,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.scenario_digest_sha256 = hashlib.sha256(canonical).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "schema_scope": SCHEMA_SCOPE,
            "run_id": self.run_id,
            "scenario_id": self.scenario.scenario_id,
            "scenario_seed": self.scenario.seed,
            "scenario_digest_sha256": self.scenario_digest_sha256,
            "time_basis": TIME_BASIS,
            "effective_scenario": self.scenario.to_effective_dict(),
            "cameras": [
                self.cameras[camera_id].summary()
                for camera_id in range(4)
            ],
        }


class MultiCameraFaultScenarioNode(Node):
    def __init__(
        self,
        model: MultiCameraFaultModel,
        metric_period_frames: int,
    ) -> None:
        super().__init__("multi_camera_fault_scenario")
        self.model = model
        self.metric_period_frames = metric_period_frames
        subscriber_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        publisher_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=200,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._camera_publishers = {}
        self._camera_subscriptions = []
        for camera in model.scenario.cameras:
            self._camera_publishers[
                camera.camera_id
            ] = self.create_publisher(
                Image, camera.output_topic, publisher_qos
            )
            self._camera_subscriptions.append(
                self.create_subscription(
                    Image,
                    camera.input_topic,
                    lambda message, camera_id=camera.camera_id: (
                        self._camera_callback(camera_id, message)
                    ),
                    subscriber_qos,
                )
            )
        self._delivery_timer = self.create_timer(
            0.001,
            self._delivery_callback,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def _publish(
        self, camera_id: int, publications: list[Publication]
    ) -> None:
        camera_model = self.model.cameras[camera_id]
        for publication in publications:
            self._camera_publishers[camera_id].publish(
                publication.message
            )
            camera_model.record_publication(
                publication,
                published_monotonic_ns=time.monotonic_ns(),
            )

    def _camera_callback(self, camera_id: int, message: Image) -> None:
        camera_model = self.model.cameras[camera_id]
        publications = camera_model.process(message, time.monotonic_ns())
        self._publish(camera_id, publications)
        if (
            camera_model.counters.received % self.metric_period_frames
            == 0
        ):
            counters = camera_model.counters
            self.get_logger().info(
                "[MULTI_CAMERA_FAULT_METRIC] "
                f"camera_id={camera_id} "
                f"received={counters.received} "
                f"published={counters.published} "
                f"dropped={counters.dropped} "
                f"pending={camera_model.pending_count()} "
                f"delayed_released={counters.delayed_released} "
                "reordered_publications="
                f"{counters.reordered_publications}"
            )

    def _delivery_callback(self) -> None:
        now_ns = time.monotonic_ns()
        for camera_id, camera_model in self.model.cameras.items():
            self._publish(camera_id, camera_model.poll(now_ns))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--metric-period-frames", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    parser = build_argument_parser()
    arguments, ros_arguments = parser.parse_known_args()
    if arguments.metric_period_frames <= 0:
        parser.error("metric-period-frames must be positive")
    try:
        scenario = load_scenario(arguments.scenario)
    except ValueError as error:
        parser.error(str(error))
    if arguments.validate_only:
        if arguments.run_id is not None or arguments.summary_json is not None:
            parser.error(
                "validate-only cannot be combined with run-id or summary-json"
            )
        print(
            json.dumps(
                scenario.to_effective_dict(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.run_id is None or arguments.summary_json is None:
        parser.error("run-id and summary-json are required for execution")
    try:
        model = MultiCameraFaultModel(scenario, arguments.run_id)
    except ValueError as error:
        parser.error(str(error))

    if arguments.summary_json.exists():
        parser.error("summary-json already exists")
    if not arguments.summary_json.parent.is_dir():
        parser.error("summary-json parent directory must exist")

    rclpy.init(args=ros_arguments)
    node = MultiCameraFaultScenarioNode(
        model, arguments.metric_period_frames
    )
    exit_code = 0
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as error:
        if rclpy.ok():
            node.get_logger().fatal(
                f"scenario injector terminated: {error}"
            )
            exit_code = 1
    finally:
        arguments.summary_json.write_text(
            json.dumps(model.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        node.destroy_node()
        rclpy.try_shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
