from __future__ import annotations

import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "airy_d435_vio_ablation_ros2.yaml"

EXPECTED_INTRINSICS = {
    "front": (603.106567, 603.102905, 330.111053, 246.598862),
    "left": (604.419189, 604.081299, 327.659485, 250.951019),
    "back": (604.789978, 604.661682, 326.815796, 244.573853),
    "right": (604.105245, 603.948629, 328.195445, 247.374578),
}


def test_outdoor_projection_contract_is_four_zero_distortion_pinhole_models():
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    parameters = document["/**"]["ros__parameters"]
    cameras = parameters["multi_camera"]["cameras"]

    assert set(cameras) == set(EXPECTED_INTRINSICS)
    for name, expected_intrinsics in EXPECTED_INTRINSICS.items():
        camera = cameras[name]
        assert camera["model"] == "Pinhole"
        assert camera["width"] == 640
        assert camera["height"] == 480
        assert camera["scale"] == 1.0
        assert tuple(camera[key] for key in ("fx", "fy", "cx", "cy")) == (
            expected_intrinsics
        )
        assert all(math.isfinite(value) for value in expected_intrinsics)
        assert {key for key in camera if key.startswith("d")} == {
            "d0",
            "d1",
            "d2",
            "d3",
        }
        assert tuple(camera[f"d{index}"] for index in range(4)) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )

    single_camera = parameters["camera"]
    assert tuple(
        single_camera[key] for key in ("fx", "fy", "cx", "cy")
    ) == EXPECTED_INTRINSICS["front"]
    assert single_camera["model"] == "Pinhole"
    assert single_camera["width"] == 640
    assert single_camera["height"] == 480
    assert single_camera["scale"] == 1.0
    assert tuple(single_camera[f"d{index}"] for index in range(4)) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
