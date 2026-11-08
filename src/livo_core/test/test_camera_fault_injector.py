"""Deterministic tests for the camera stream fault transformer."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INJECTOR_PATH = PACKAGE_ROOT / "scripts" / "camera_fault_injector.py"
SPEC = importlib.util.spec_from_file_location(
    "camera_fault_injector", INJECTOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
INJECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INJECTOR
SPEC.loader.exec_module(INJECTOR)

SCENARIO_PATH = (
    PACKAGE_ROOT / "scripts" / "multi_camera_fault_scenario.py"
)
SCENARIO_SPEC = importlib.util.spec_from_file_location(
    "multi_camera_fault_scenario", SCENARIO_PATH
)
assert SCENARIO_SPEC is not None and SCENARIO_SPEC.loader is not None
SCENARIO = importlib.util.module_from_spec(SCENARIO_SPEC)
sys.modules[SCENARIO_SPEC.name] = SCENARIO
SCENARIO_SPEC.loader.exec_module(SCENARIO)


def image(timestamp_sec: float, value: int) -> Image:
    message = Image()
    timestamp_ns = round(timestamp_sec * 1.0e9)
    message.header.stamp.sec = timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = timestamp_ns % 1_000_000_000
    message.height = 2
    message.width = 2
    message.encoding = "rgb8"
    message.is_bigendian = False
    message.step = 6
    message.data = bytes([value]) * 12
    return message


def config(
    fault_type: str,
    *,
    start_sec: float = 1.0,
    duration_sec: float = 2.0,
    timestamp_shift_ms: float = 0.0,
):
    return INJECTOR.CameraFaultConfig(
        camera_id=0,
        fault_type=fault_type,
        start_sec=start_sec,
        duration_sec=duration_sec,
        timestamp_shift_ms=timestamp_shift_ms,
    )


def scenario_document() -> dict:
    topics = (
        "/camera/camera/color/image_raw",
        "/left/camera/color/image_raw",
        "/back/camera/color/image_raw",
        "/right/camera/color/image_raw",
    )
    return {
        "schema_version": 1,
        "scenario_id": "unit-test",
        "seed": 20260731,
        "duration_sec": 20.0,
        "cameras": [
            {
                "id": camera_id,
                "name": camera_name,
                "input_topic": f"/fault/input/{camera_name}",
                "output_topic": topics[camera_id],
                "windows": [],
            }
            for camera_id, camera_name in enumerate(
                INJECTOR.CAMERA_NAMES
            )
        ],
    }


def window(
    window_id: str,
    fault_type: str,
    parameters: dict,
    *,
    start_sec: float = 0.0,
    duration_sec: float = 5.0,
) -> dict:
    return {
        "id": window_id,
        "type": fault_type,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "parameters": parameters,
    }


def process_and_record(model, message, arrival_ns: int):
    publications = model.process(message, arrival_ns)
    for publication in publications:
        model.record_publication(publication)
    return publications


@pytest.mark.parametrize(
    ("fault_type", "expected_value"),
    (("overexposure", 255), ("underexposure", 0)),
)
def test_intensity_faults_only_modify_frames_inside_window(
    fault_type: str,
    expected_value: int,
) -> None:
    model = INJECTOR.CameraFaultModel(config(fault_type))
    before = image(10.0, 20)
    active = image(11.0, 30)
    after = image(13.0, 40)

    before_result = model.process(before)
    active_result = model.process(active)
    after_result = model.process(after)

    assert before_result.message is before
    assert bytes(active_result.message.data) == bytes([expected_value]) * 12
    assert bytes(active.data) == bytes([30]) * 12
    assert after_result.message is after
    assert model.counters.modified == 1
    assert model.counters.published == 3


def test_freeze_repeats_pixels_but_preserves_incoming_timestamps() -> None:
    model = INJECTOR.CameraFaultModel(config("freeze"))
    model.process(image(10.0, 10))

    first = model.process(image(11.0, 20)).message
    second = model.process(image(12.0, 30)).message

    assert bytes(first.data) == bytes([20]) * 12
    assert bytes(second.data) == bytes([20]) * 12
    assert INJECTOR.image_stamp_ns(first) == 11_000_000_000
    assert INJECTOR.image_stamp_ns(second) == 12_000_000_000


def test_timestamp_shift_changes_only_published_header_stamp() -> None:
    model = INJECTOR.CameraFaultModel(
        config("timestamp_shift", timestamp_shift_ms=35.5)
    )
    model.process(image(10.0, 10))
    original = image(11.0, 20)

    shifted = model.process(original).message

    assert INJECTOR.image_stamp_ns(original) == 11_000_000_000
    assert INJECTOR.image_stamp_ns(shifted) == 11_035_500_000
    assert bytes(shifted.data) == bytes(original.data)


def test_dropout_suppresses_only_fault_window() -> None:
    model = INJECTOR.CameraFaultModel(config("dropout"))

    assert model.process(image(10.0, 10)).message is not None
    assert model.process(image(11.0, 20)).message is None
    assert model.process(image(12.9, 30)).message is None
    assert model.process(image(13.0, 40)).message is not None
    assert model.counters.received == 4
    assert model.counters.published == 2
    assert model.counters.dropped == 2
    assert model.counters.fault_window_frames == 2


def test_fault_window_is_relative_to_first_input_timestamp() -> None:
    model = INJECTOR.CameraFaultModel(
        config("overexposure", start_sec=0.5, duration_sec=0.5)
    )

    assert model.process(image(101.0, 10)).in_fault_window is False
    assert model.process(image(101.499, 20)).in_fault_window is False
    assert model.process(image(101.5, 30)).in_fault_window is True
    assert model.process(image(101.999, 40)).in_fault_window is True
    assert model.process(image(102.0, 50)).in_fault_window is False


def test_black_occlusion_replaces_full_frame_and_declares_manifest() -> None:
    fault_config = INJECTOR.CameraFaultConfig(
        camera_id=2,
        camera_name="back",
        fault_type="occlusion",
        start_sec=1.0,
        duration_sec=2.0,
        occlusion_mode="black",
    )
    model = INJECTOR.CameraFaultModel(fault_config)
    model.process(image(10.0, 10))

    occluded = model.process(image(11.0, 220)).message
    summary = model.summary("/fault/input", "/fault/output")

    assert bytes(occluded.data) == bytes(12)
    assert summary["manifest"] == {
        "camera": {
            "id": 2,
            "name": "back",
            "input_topic": "/fault/input",
            "output_topic": "/fault/output",
        },
        "mode": "occlusion:black",
        "time_basis": "first_input_header_stamp",
        "start_sec": 1.0,
        "end_sec": 3.0,
        "relative_sec": {"start": 1.0, "end": 3.0},
        "source_stamp_ns": {
            "start": 11_000_000_000,
            "end": 13_000_000_000,
        },
    }


def test_fixed_mask_occlusion_is_scene_independent() -> None:
    model = INJECTOR.CameraFaultModel(
        INJECTOR.CameraFaultConfig(
            camera_id=1,
            camera_name="left",
            fault_type="occlusion",
            start_sec=0.0,
            duration_sec=2.0,
            occlusion_mode="fixed_mask",
        )
    )

    first = model.process(image(10.0, 10)).message
    second = model.process(image(11.0, 240)).message

    assert bytes(first.data) == bytes([127]) * 12
    assert bytes(second.data) == bytes([127]) * 12


def test_gaussian_noise_is_reproducible_for_a_fixed_seed() -> None:
    fault_config = INJECTOR.CameraFaultConfig(
        camera_id=3,
        camera_name="right",
        fault_type="gaussian_noise",
        start_sec=0.0,
        duration_sec=1.0,
        noise_stddev=12.5,
        noise_seed=41,
    )
    first_model = INJECTOR.CameraFaultModel(fault_config)
    second_model = INJECTOR.CameraFaultModel(fault_config)
    original = image(10.0, 128)

    first = bytes(first_model.process(original).message.data)
    second = bytes(second_model.process(original).message.data)
    manifest = first_model.summary()["manifest"]

    assert first == second
    assert list(first) == [
        113,
        131,
        128,
        134,
        111,
        142,
        129,
        113,
        111,
        112,
        119,
        143,
    ]
    assert first != bytes(original.data)
    assert manifest["mode"] == "gaussian_noise"
    assert manifest["parameters"] == {
        "generator": "numpy-pcg64-v1",
        "seed": 41,
        "stddev": 12.5,
    }


def test_clean_relay_uses_the_same_relative_and_source_stamp_window() -> None:
    model = INJECTOR.CameraFaultModel(
        INJECTOR.CameraFaultConfig(
            camera_id=0,
            camera_name="front",
            fault_type="none",
            start_sec=1.0,
            duration_sec=2.0,
        )
    )
    model.process(image(10.0, 10))
    clean = image(11.0, 220)

    result = model.process(clean)
    summary = model.summary("/fault/input", "/fault/output")

    assert result.message is clean
    assert summary["counters"]["modified"] == 0
    assert summary["counters"]["dropped"] == 0
    assert summary["manifest"]["relative_sec"] == {
        "start": 1.0,
        "end": 3.0,
    }
    assert summary["manifest"]["source_stamp_ns"] == {
        "start": 11_000_000_000,
        "end": 13_000_000_000,
    }


def test_cli_exposes_manifest_and_determinism_parameters() -> None:
    arguments = INJECTOR.build_argument_parser().parse_args(
        [
            "--camera-id",
            "1",
            "--camera-name",
            "left",
            "--input-topic",
            "/in",
            "--output-topic",
            "/out",
            "--fault",
            "occlusion",
            "--occlusion-mode",
            "fixed_mask",
            "--noise-stddev",
            "7.5",
            "--noise-seed",
            "19",
            "--start-sec",
            "2",
            "--duration-sec",
            "3",
            "--summary-json",
            "/tmp/summary.json",
        ]
    )

    assert arguments.camera_name == "left"
    assert arguments.occlusion_mode == "fixed_mask"
    assert arguments.noise_stddev == 7.5
    assert arguments.noise_seed == 19


def test_timestamp_shift_requires_nonzero_offset() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        INJECTOR.CameraFaultModel(config("timestamp_shift"))


def test_four_camera_clean_scenario_is_exact_passthrough() -> None:
    scenario = SCENARIO.parse_scenario(scenario_document())
    model = SCENARIO.MultiCameraFaultModel(scenario, "clean-run")

    for camera_id in range(4):
        original = image(10.0 + camera_id, 20 + camera_id)
        publications = process_and_record(
            model.cameras[camera_id], original, 1_000_000_000
        )

        assert len(publications) == 1
        assert publications[0].message is original
        assert SCENARIO.image_stamp_ns(publications[0].message) == (
            10 + camera_id
        ) * 1_000_000_000

    summary = model.summary()
    assert summary["schema_version"] == 1
    assert summary["schema_scope"] == "experimental_local"
    assert summary["time_basis"] == (
        "per_camera_first_input_header_stamp"
    )
    assert [camera["counters"]["received"] for camera in summary["cameras"]] == [
        1,
        1,
        1,
        1,
    ]
    assert [camera["counters"]["published"] for camera in summary["cameras"]] == [
        1,
        1,
        1,
        1,
    ]


def test_scenario_identity_records_the_gaussian_generator() -> None:
    document = scenario_document()
    document["cameras"][3]["windows"] = [
        window(
            "right-noise",
            "gaussian_noise",
            {"noise_stddev": 9.5},
        )
    ]

    scenario = SCENARIO.parse_scenario(document)
    parameters = scenario.cameras[3].windows[0].parameters

    assert parameters == {
        "generator": "numpy-pcg64-v1",
        "noise_stddev": 9.5,
        "seed": SCENARIO._derived_seed(
            20260731, 3, "right-noise"
        ),
    }


def test_ros_adapter_constructs_four_publishers_and_subscriptions() -> None:
    scenario = SCENARIO.parse_scenario(scenario_document())
    model = SCENARIO.MultiCameraFaultModel(scenario, "node-smoke")
    node = None

    SCENARIO.rclpy.init(args=[])
    try:
        node = SCENARIO.MultiCameraFaultScenarioNode(
            model, metric_period_frames=100
        )

        assert set(node._camera_publishers) == {0, 1, 2, 3}
        assert len(node._camera_subscriptions) == 4
        assert all(
            publisher in node.publishers
            for publisher in node._camera_publishers.values()
        )
        assert all(
            subscription in node.subscriptions
            for subscription in node._camera_subscriptions
        )
    finally:
        if node is not None:
            node.destroy_node()
        if SCENARIO.rclpy.ok():
            SCENARIO.rclpy.shutdown()


class _LifecycleLogger:
    def __init__(self) -> None:
        self.fatal_messages: list[str] = []

    def fatal(self, message: str) -> None:
        self.fatal_messages.append(message)


class _LifecycleNode:
    def __init__(self) -> None:
        self.logger = _LifecycleLogger()
        self.destroy_count = 0

    def get_logger(self) -> _LifecycleLogger:
        return self.logger

    def destroy_node(self) -> None:
        self.destroy_count += 1


def _run_scenario_main_with_spin_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    *,
    context_ok: bool,
) -> tuple[
    tuple[int, dict[str, object], _LifecycleNode, int, int, int],
    int,
]:
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(scenario_document()), encoding="utf-8"
    )
    summary_path = tmp_path / "summary.json"
    node = _LifecycleNode()
    init_count = 0
    try_shutdown_count = 0
    shutdown_count = 0
    summary_write_count = 0
    original_write_text = Path.write_text

    def init(*, args) -> None:
        nonlocal init_count
        assert args == []
        init_count += 1

    def spin(spin_node) -> None:
        assert spin_node is node
        raise error

    def try_shutdown() -> None:
        nonlocal try_shutdown_count
        try_shutdown_count += 1

    def shutdown() -> None:
        nonlocal shutdown_count
        shutdown_count += 1

    def tracked_write_text(path, *args, **kwargs):
        nonlocal summary_write_count
        if path == summary_path:
            summary_write_count += 1
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(SCENARIO.rclpy, "init", init)
    monkeypatch.setattr(SCENARIO.rclpy, "spin", spin)
    monkeypatch.setattr(SCENARIO.rclpy, "ok", lambda: context_ok)
    monkeypatch.setattr(SCENARIO.rclpy, "try_shutdown", try_shutdown)
    monkeypatch.setattr(SCENARIO.rclpy, "shutdown", shutdown)
    monkeypatch.setattr(
        SCENARIO,
        "MultiCameraFaultScenarioNode",
        lambda model, metric_period_frames: node,
    )
    monkeypatch.setattr(Path, "write_text", tracked_write_text)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCENARIO_PATH),
            "--scenario",
            str(scenario_path),
            "--run-id",
            "lifecycle-run",
            "--summary-json",
            str(summary_path),
        ],
    )

    exit_code = SCENARIO.main()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return (
        exit_code,
        summary,
        node,
        init_count,
        try_shutdown_count,
        shutdown_count,
    ), summary_write_count


@pytest.mark.parametrize(
    ("error", "context_ok"),
    [
        (ExternalShutdownException(), True),
        (
            RCLError(
                "failed to initialize wait set: context is not valid"
            ),
            False,
        ),
    ],
)
def test_external_context_shutdown_is_a_clean_lifecycle_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    context_ok: bool,
) -> None:
    lifecycle, summary_write_count = _run_scenario_main_with_spin_error(
        monkeypatch,
        tmp_path,
        error,
        context_ok=context_ok,
    )
    (
        exit_code,
        summary,
        node,
        init_count,
        try_shutdown_count,
        shutdown_count,
    ) = lifecycle

    assert exit_code == 0
    assert summary["run_id"] == "lifecycle-run"
    assert summary["scenario_id"] == "unit-test"
    assert summary_write_count == 1
    assert node.logger.fatal_messages == []
    assert node.destroy_count == 1
    assert init_count == 1
    assert try_shutdown_count == 1
    assert shutdown_count == 0


def test_active_context_spin_failure_is_fatal_and_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lifecycle, summary_write_count = _run_scenario_main_with_spin_error(
        monkeypatch,
        tmp_path,
        RuntimeError("executor failed"),
        context_ok=True,
    )
    (
        exit_code,
        summary,
        node,
        init_count,
        try_shutdown_count,
        shutdown_count,
    ) = lifecycle

    assert exit_code == 1
    assert summary["run_id"] == "lifecycle-run"
    assert summary_write_count == 1
    assert node.logger.fatal_messages == [
        "scenario injector terminated: executor failed"
    ]
    assert node.destroy_count == 1
    assert init_count == 1
    assert try_shutdown_count == 1
    assert shutdown_count == 0


def test_delivery_delay_changes_release_time_without_changing_header() -> None:
    document = scenario_document()
    document["cameras"][0]["windows"] = [
        window("front-delay", "delivery_delay", {"delay_ms": 35.0})
    ]
    scenario = SCENARIO.parse_scenario(document)
    model = SCENARIO.CameraScenarioModel(scenario.cameras[0])
    original = image(10.0, 40)

    assert model.process(original, 1_000_000_000) == []
    assert model.poll(1_034_999_999) == []
    ready = model.poll(1_035_000_000)

    assert len(ready) == 1
    assert ready[0].message is original
    assert SCENARIO.image_stamp_ns(ready[0].message) == 10_000_000_000
    model.record_publication(ready[0])
    summary = model.summary()
    assert summary["counters"]["delayed_released"] == 1
    assert summary["windows"][0]["counters"]["scheduled"] == 1
    assert summary["windows"][0]["counters"]["delayed_released"] == 1


def test_delivery_timer_uses_steady_time_not_bag_clock() -> None:
    source = SCENARIO_PATH.read_text(encoding="utf-8")

    assert "ClockType.STEADY_TIME" in source
    assert "time.monotonic_ns()" in source


def _run_jitter_model(camera: SCENARIO.CameraScenario) -> list[int]:
    model = SCENARIO.CameraScenarioModel(camera)
    published_stamps = []
    for frame_index in range(12):
        publications = model.process(
            image(10.0 + frame_index * 0.01, 40 + frame_index),
            frame_index * 1_000_000,
        )
        for publication in publications:
            model.record_publication(publication)
            published_stamps.append(
                SCENARIO.image_stamp_ns(publication.message)
            )
    for publication in model.poll(100_000_000):
        model.record_publication(publication)
        published_stamps.append(
            SCENARIO.image_stamp_ns(publication.message)
        )
    return published_stamps


def test_seeded_jitter_is_reproducible_and_preserves_source_stamps() -> None:
    document = scenario_document()
    document["cameras"][1]["windows"] = [
        window(
            "left-jitter",
            "seeded_jitter",
            {"max_delay_ms": 30.0},
        )
    ]
    scenario = SCENARIO.parse_scenario(document)

    first = _run_jitter_model(scenario.cameras[1])
    second = _run_jitter_model(scenario.cameras[1])

    expected = [
        round((10.0 + frame_index * 0.01) * 1.0e9)
        for frame_index in range(12)
    ]
    assert first == second
    assert sorted(first) == expected
    assert first != expected


def _run_reorder_model(camera: SCENARIO.CameraScenario) -> list[int]:
    model = SCENARIO.CameraScenarioModel(camera)
    output_values = []
    for frame_index in range(6):
        publications = model.process(
            image(10.0 + frame_index * 0.1, frame_index),
            frame_index,
        )
        for publication in publications:
            model.record_publication(publication)
            output_values.append(bytes(publication.message.data)[0])
    return output_values


def test_bounded_reorder_is_seeded_reproducible_and_chunk_bounded() -> None:
    document = scenario_document()
    document["cameras"][2]["windows"] = [
        window(
            "back-reorder",
            "bounded_reorder",
            {"depth_frames": 3},
        )
    ]
    scenario = SCENARIO.parse_scenario(document)

    first = _run_reorder_model(scenario.cameras[2])
    second = _run_reorder_model(scenario.cameras[2])

    assert first == second
    assert first[:3] != [0, 1, 2]
    assert first[3:] != [3, 4, 5]
    assert sorted(first[:3]) == [0, 1, 2]
    assert sorted(first[3:]) == [3, 4, 5]


def _run_burst_model(camera: SCENARIO.CameraScenario) -> list[int]:
    model = SCENARIO.CameraScenarioModel(camera)
    output_values = []
    for frame_index in range(15):
        publications = process_and_record(
            model,
            image(10.0 + frame_index * 0.1, frame_index),
            frame_index,
        )
        output_values.extend(
            bytes(publication.message.data)[0]
            for publication in publications
        )
    assert model.counters.dropped == 6
    return output_values


def test_seeded_burst_dropout_is_reproducible() -> None:
    document = scenario_document()
    document["cameras"][3]["windows"] = [
        window(
            "right-burst",
            "burst_dropout",
            {"period_frames": 5, "drop_frames": 2},
        )
    ]
    scenario = SCENARIO.parse_scenario(document)

    first = _run_burst_model(scenario.cameras[3])
    second = _run_burst_model(scenario.cameras[3])

    assert first == second
    assert len(first) == 9


def test_manifest_records_camera_and_window_actual_counts_and_bounds() -> None:
    document = scenario_document()
    document["cameras"][0]["windows"] = [
        window(
            "front-overexposure",
            "overexposure",
            {},
            start_sec=1.0,
            duration_sec=2.0,
        )
    ]
    model = SCENARIO.MultiCameraFaultModel(
        SCENARIO.parse_scenario(document), "manifest-run"
    )
    camera_model = model.cameras[0]
    process_and_record(camera_model, image(10.0, 10), 0)
    process_and_record(camera_model, image(11.0, 20), 1)
    process_and_record(camera_model, image(14.0, 30), 2)

    summary = model.summary()
    front = summary["cameras"][0]
    fault_window = front["windows"][0]

    assert front["first_source_stamp_ns"] == 10_000_000_000
    assert front["last_source_stamp_ns"] == 14_000_000_000
    assert front["counters"] == {
        "received": 3,
        "published": 3,
        "dropped": 0,
        "delayed_released": 0,
        "reordered_publications": 0,
        "pending": 0,
    }
    assert fault_window["source_stamp_ns"] == {
        "start": 11_000_000_000,
        "end": 13_000_000_000,
    }
    assert fault_window["counters"]["matched"] == 1
    assert fault_window["counters"]["modified"] == 1
    assert fault_window["counters"]["published"] == 1


def test_fault_publication_receipt_binds_source_and_monotonic_time() -> None:
    document = scenario_document()
    document["cameras"][3]["windows"] = [
        window(
            "right-noise",
            "gaussian_noise",
            {"noise_stddev": 8.0},
        )
    ]
    model = SCENARIO.MultiCameraFaultModel(
        SCENARIO.parse_scenario(document), "receipt-run"
    )
    camera_model = model.cameras[3]
    publications = camera_model.process(
        image(10.0, 128), 1_000_000_000
    )

    assert len(publications) == 1
    camera_model.record_publication(
        publications[0], published_monotonic_ns=1_234_567_890
    )

    fault_window = model.summary()["cameras"][3]["windows"][0]
    assert fault_window["publication_receipts"] == [
        {
            "published_monotonic_ns": 1_234_567_890,
            "source_stamp_ns": 10_000_000_000,
        }
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"unknown": True}),
        lambda value: value.update({"schema_version": 1.0}),
        lambda value: value["cameras"][1].update({"id": 0}),
        lambda value: value["cameras"][1].update({"name": "front"}),
        lambda value: value["cameras"][0].update(
            {"input_topic": "relative/topic"}
        ),
        lambda value: value["cameras"][0]["windows"].append(
            window("unknown-fault", "not_a_fault", {})
        ),
        lambda value: value["cameras"][0]["windows"].append(
            window(
                "outside",
                "dropout",
                {},
                start_sec=19.0,
                duration_sec=2.0,
            )
        ),
        lambda value: value["cameras"][0]["windows"].extend(
            [
                window("first", "dropout", {}),
                window(
                    "overlap",
                    "overexposure",
                    {},
                    start_sec=4.0,
                ),
            ]
        ),
        lambda value: value["cameras"][0]["windows"].append(
            window(
                "bad-delay",
                "delivery_delay",
                {"delay_ms": -1.0},
            )
        ),
        lambda value: value["cameras"][0]["windows"].append(
            window(
                "extra-parameter",
                "dropout",
                {"unexpected": 1},
            )
        ),
    ],
)
def test_scenario_schema_rejects_invalid_or_ambiguous_config(
    mutate,
) -> None:
    document = copy.deepcopy(scenario_document())
    mutate(document)

    with pytest.raises(ValueError):
        SCENARIO.parse_scenario(document)
