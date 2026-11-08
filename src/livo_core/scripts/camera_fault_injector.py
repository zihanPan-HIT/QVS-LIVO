#!/usr/bin/env python3
"""Inject one deterministic fault into one ROS 2 camera stream."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image


FAULT_TYPES = (
    "none",
    "overexposure",
    "underexposure",
    "freeze",
    "timestamp_shift",
    "dropout",
    "occlusion",
    "gaussian_noise",
)
CAMERA_NAMES = ("front", "left", "back", "right")
GAUSSIAN_NOISE_GENERATOR = "numpy-pcg64-v1"


@dataclass(frozen=True)
class CameraFaultConfig:
    camera_id: int
    fault_type: str
    start_sec: float
    duration_sec: float
    camera_name: str = ""
    timestamp_shift_ms: float = 0.0
    occlusion_mode: str = "black"
    noise_stddev: float = 10.0
    noise_seed: int = 0

    def validate(self) -> None:
        if self.camera_id < 0 or self.camera_id >= len(CAMERA_NAMES):
            raise ValueError("camera_id must identify front/left/back/right")
        expected_name = CAMERA_NAMES[self.camera_id]
        if self.camera_name and self.camera_name != expected_name:
            raise ValueError(
                "camera_name does not match camera_id: "
                f"expected {expected_name}, got {self.camera_name}"
            )
        if self.fault_type not in FAULT_TYPES:
            raise ValueError(f"unsupported fault_type: {self.fault_type}")
        if not math.isfinite(self.start_sec) or self.start_sec < 0.0:
            raise ValueError("start_sec must be finite and nonnegative")
        if not math.isfinite(self.duration_sec) or self.duration_sec <= 0.0:
            raise ValueError("duration_sec must be finite and positive")
        if not math.isfinite(self.timestamp_shift_ms):
            raise ValueError("timestamp_shift_ms must be finite")
        if self.occlusion_mode not in {"black", "fixed_mask"}:
            raise ValueError(
                f"unsupported occlusion_mode: {self.occlusion_mode}"
            )
        if not math.isfinite(self.noise_stddev) or self.noise_stddev <= 0.0:
            raise ValueError("noise_stddev must be finite and positive")
        if (
            self.fault_type == "timestamp_shift"
            and abs(self.timestamp_shift_ms) < 1.0e-12
        ):
            raise ValueError(
                "timestamp_shift requires a nonzero timestamp_shift_ms"
            )


@dataclass
class CameraFaultCounters:
    received: int = 0
    published: int = 0
    fault_window_frames: int = 0
    modified: int = 0
    dropped: int = 0
    first_stamp_ns: Optional[int] = None
    last_stamp_ns: Optional[int] = None
    first_fault_stamp_ns: Optional[int] = None
    last_fault_stamp_ns: Optional[int] = None


@dataclass
class FaultProcessResult:
    message: Optional[Image]
    in_fault_window: bool
    relative_sec: float


def image_stamp_ns(message: Image) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def set_image_stamp_ns(message: Image, timestamp_ns: int) -> None:
    timestamp_ns = max(0, timestamp_ns)
    message.header.stamp.sec = timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = timestamp_ns % 1_000_000_000


class CameraFaultModel:
    """Pure deterministic stream transformer used by the ROS adapter."""

    def __init__(self, config: CameraFaultConfig) -> None:
        config.validate()
        self.config = config
        self.counters = CameraFaultCounters()
        self._frozen_payload: Optional[bytes] = None
        self._frozen_shape: Optional[tuple[int, int, int, str, bool]] = None
        self._noise = np.random.Generator(
            np.random.PCG64(config.noise_seed)
        )

    def process(self, message: Image) -> FaultProcessResult:
        timestamp_ns = image_stamp_ns(message)
        if self.counters.first_stamp_ns is None:
            self.counters.first_stamp_ns = timestamp_ns
        self.counters.received += 1
        self.counters.last_stamp_ns = timestamp_ns

        relative_sec = (
            timestamp_ns - self.counters.first_stamp_ns
        ) * 1.0e-9
        in_fault_window = (
            relative_sec >= self.config.start_sec
            and relative_sec
            < self.config.start_sec + self.config.duration_sec
        )
        if not in_fault_window:
            self.counters.published += 1
            return FaultProcessResult(message, False, relative_sec)

        self.counters.fault_window_frames += 1
        if self.counters.first_fault_stamp_ns is None:
            self.counters.first_fault_stamp_ns = timestamp_ns
        self.counters.last_fault_stamp_ns = timestamp_ns

        if self.config.fault_type == "none":
            self.counters.published += 1
            return FaultProcessResult(message, True, relative_sec)
        if self.config.fault_type == "dropout":
            self.counters.dropped += 1
            return FaultProcessResult(None, True, relative_sec)

        output = copy.deepcopy(message)
        if self.config.fault_type == "overexposure":
            output.data = bytes([255]) * len(message.data)
        elif self.config.fault_type == "underexposure":
            output.data = bytes(len(message.data))
        elif self.config.fault_type == "freeze":
            shape = (
                int(message.height),
                int(message.width),
                int(message.step),
                str(message.encoding),
                bool(message.is_bigendian),
            )
            if self._frozen_payload is None:
                self._frozen_payload = bytes(message.data)
                self._frozen_shape = shape
            if shape != self._frozen_shape:
                raise ValueError(
                    "image layout changed during a freeze fault window"
                )
            output.data = self._frozen_payload
        elif self.config.fault_type == "timestamp_shift":
            shift_ns = round(self.config.timestamp_shift_ms * 1.0e6)
            set_image_stamp_ns(output, timestamp_ns + shift_ns)
        elif self.config.fault_type == "occlusion":
            fill_value = (
                0 if self.config.occlusion_mode == "black" else 127
            )
            output.data = bytes([fill_value]) * len(message.data)
        elif self.config.fault_type == "gaussian_noise":
            pixels = np.frombuffer(message.data, dtype=np.uint8)
            noise = self._noise.normal(
                0.0, self.config.noise_stddev, size=pixels.size
            )
            output.data = np.clip(
                np.rint(pixels.astype(np.float64) + noise),
                0.0,
                255.0,
            ).astype(np.uint8).tobytes()
        else:
            raise AssertionError(
                f"unhandled fault type: {self.config.fault_type}"
            )

        self.counters.modified += 1
        self.counters.published += 1
        return FaultProcessResult(output, True, relative_sec)

    def summary(
        self,
        input_topic: str = "",
        output_topic: str = "",
    ) -> dict:
        camera_name = (
            self.config.camera_name or CAMERA_NAMES[self.config.camera_id]
        )
        mode = self.config.fault_type
        if mode == "occlusion":
            mode = f"{mode}:{self.config.occlusion_mode}"
        manifest = {
            "camera": {
                "id": self.config.camera_id,
                "name": camera_name,
                "input_topic": input_topic,
                "output_topic": output_topic,
            },
            "mode": mode,
            "time_basis": "first_input_header_stamp",
            "start_sec": self.config.start_sec,
            "end_sec": (
                self.config.start_sec + self.config.duration_sec
            ),
        }
        if self.config.fault_type == "gaussian_noise":
            manifest["parameters"] = {
                "generator": GAUSSIAN_NOISE_GENERATOR,
                "seed": self.config.noise_seed,
                "stddev": self.config.noise_stddev,
            }
        source_origin_ns = self.counters.first_stamp_ns
        manifest["relative_sec"] = {
            "start": self.config.start_sec,
            "end": self.config.start_sec + self.config.duration_sec,
        }
        manifest["source_stamp_ns"] = {
            "start": (
                None
                if source_origin_ns is None
                else source_origin_ns
                + round(self.config.start_sec * 1.0e9)
            ),
            "end": (
                None
                if source_origin_ns is None
                else source_origin_ns
                + round(
                    (
                        self.config.start_sec
                        + self.config.duration_sec
                    )
                    * 1.0e9
                )
            ),
        }
        return {
            "schema_version": 3,
            "config": asdict(self.config),
            "input_topic": input_topic,
            "output_topic": output_topic,
            "manifest": manifest,
            "counters": asdict(self.counters),
        }


class CameraFaultInjectorNode(Node):
    def __init__(
        self,
        model: CameraFaultModel,
        input_topic: str,
        output_topic: str,
        metric_period_frames: int,
    ) -> None:
        super().__init__(
            f"camera_fault_injector_cam{model.config.camera_id}"
        )
        self.model = model
        self.input_topic = input_topic
        self.output_topic = output_topic
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
        self.publisher = self.create_publisher(
            Image, output_topic, publisher_qos
        )
        self.subscription = self.create_subscription(
            Image, input_topic, self._callback, subscriber_qos
        )

    def _callback(self, message: Image) -> None:
        try:
            result = self.model.process(message)
        except (ValueError, OverflowError) as error:
            self.get_logger().fatal(f"fault injection failed: {error}")
            raise

        if result.message is not None:
            self.publisher.publish(result.message)
        if (
            self.model.counters.received % self.metric_period_frames == 0
            or (
                result.in_fault_window
                and self.model.counters.fault_window_frames == 1
            )
        ):
            counters = self.model.counters
            self.get_logger().info(
                "[CAMERA_FAULT_METRIC] "
                f"camera_id={self.model.config.camera_id} "
                f"fault={self.model.config.fault_type} "
                f"relative_sec={result.relative_sec:.6f} "
                f"active={int(result.in_fault_window)} "
                f"received={counters.received} "
                f"published={counters.published} "
                f"modified={counters.modified} "
                f"dropped={counters.dropped}"
            )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=int, required=True)
    parser.add_argument(
        "--camera-name", choices=CAMERA_NAMES, required=True
    )
    parser.add_argument("--input-topic", required=True)
    parser.add_argument("--output-topic", required=True)
    parser.add_argument("--fault", choices=FAULT_TYPES, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--timestamp-shift-ms", type=float, default=0.0)
    parser.add_argument(
        "--occlusion-mode",
        choices=("black", "fixed_mask"),
        default="black",
    )
    parser.add_argument("--noise-stddev", type=float, default=10.0)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--metric-period-frames", type=int, default=100)
    return parser


def main() -> int:
    parser = build_argument_parser()
    arguments, ros_arguments = parser.parse_known_args()
    if arguments.input_topic == arguments.output_topic:
        parser.error("input-topic and output-topic must differ")
    if arguments.metric_period_frames <= 0:
        parser.error("metric-period-frames must be positive")

    config = CameraFaultConfig(
        camera_id=arguments.camera_id,
        fault_type=arguments.fault,
        start_sec=arguments.start_sec,
        duration_sec=arguments.duration_sec,
        camera_name=arguments.camera_name,
        timestamp_shift_ms=arguments.timestamp_shift_ms,
        occlusion_mode=arguments.occlusion_mode,
        noise_stddev=arguments.noise_stddev,
        noise_seed=arguments.noise_seed,
    )
    try:
        model = CameraFaultModel(config)
    except ValueError as error:
        parser.error(str(error))

    rclpy.init(args=ros_arguments)
    node = CameraFaultInjectorNode(
        model,
        arguments.input_topic,
        arguments.output_topic,
        arguments.metric_period_frames,
    )
    exit_code = 0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        node.get_logger().fatal(f"injector terminated: {error}")
        exit_code = 1
    finally:
        summary = model.summary(
            arguments.input_topic, arguments.output_topic
        )
        arguments.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
