from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import struct
from typing import Any

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT / "config" / "airy_d435_vio_ablation_ros2.yaml"
)
CAMERA_NAMES = ("front", "left", "back", "right")
EXPECTED_RIG_FINGERPRINT = (
    "71e3d60274ed84df4237393f2d79db6f"
    "1d48ceef9d0f0bbfb97fb0fa84fcd048"
)
EXPECTED_RIG_REVISION = 8206638252039177439
EXPECTED_MODEL_REVISIONS = (
    1172720795579671296,
    14974416105296971232,
    10800054445785174959,
    11386666025749495921,
)
EXPECTED_T_IC = {
    "front": (
        (
            0.0031916438524545324,
            0.005008147203150214,
            -0.9999823657800725,
            0.999993488142341,
            -0.0017003358397726313,
            0.0031831636693519043,
            -0.0016843641034486483,
            -0.9999860135620017,
            -0.0050135414572385585,
        ),
        (
            -0.16056981004032952,
            -0.023585423545053538,
            -0.2725274098602011,
        ),
    ),
    "left": (
        (
            -0.9998279589765651,
            -0.014464221993487875,
            0.01161200804681417,
            -0.011402097122951621,
            -0.014511318976066556,
            -0.9998296923990474,
            0.0146302641792605,
            -0.9997900819190487,
            0.014343899969920915,
        ),
        (
            0.023585423545053535,
            -0.16056981004032955,
            -0.2725274098602011,
        ),
    ),
    "back": (
        (
            -0.0016205753677488634,
            0.004443169836950256,
            0.9999888159260971,
            -0.9997769766666371,
            0.021048989597706597,
            -0.0017137573440959807,
            -0.021056368699190493,
            -0.9997685723599817,
            0.0044080674323445646,
        ),
        (
            0.16056981004032955,
            0.023585423545053563,
            -0.27252740986020113,
        ),
    ),
    "right": (
        (
            0.9996940504216689,
            -0.0197784730625766,
            -0.014853200154535728,
            0.015445004958050595,
            0.030112931589163727,
            0.9994271675179499,
            -0.019319869910627285,
            -0.9993508009475048,
            0.03040919716495591,
        ),
        (
            -0.023585423545053545,
            0.16056981004032955,
            -0.2725274098602011,
        ),
    ),
}


Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Vector3 = tuple[float, float, float]


def _matrix3(values: list[float]) -> Matrix3:
    assert len(values) == 9
    return tuple(
        tuple(float(value) for value in values[offset : offset + 3])
        for offset in range(0, 9, 3)
    )


def _vector3(values: list[float]) -> Vector3:
    assert len(values) == 3
    return tuple(float(value) for value in values)


def _transpose(matrix: Matrix3) -> Matrix3:
    return tuple(
        tuple(matrix[column][row] for column in range(3))
        for row in range(3)
    )


def _matrix_product(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _matrix_vector_product(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        sum(matrix[row][index] * vector[index] for index in range(3))
        for row in range(3)
    )


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))


def _negate(vector: Vector3) -> Vector3:
    return tuple(-value for value in vector)


def _flatten(matrix: Matrix3) -> tuple[float, ...]:
    return tuple(value for row in matrix for value in row)


def _pack_unsigned(value: int) -> bytes:
    return struct.pack(">Q", value)


def _pack_signed(value: int) -> bytes:
    return struct.pack(">q", value)


def _pack_double(value: float) -> bytes:
    canonical_value = 0.0 if value == 0.0 else value
    return struct.pack(">d", canonical_value)


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return _pack_unsigned(len(encoded)) + encoded


def _model_fingerprint(camera: dict[str, Any]) -> bytes:
    canonical = bytearray(
        _pack_string("livo/four-camera-pinhole-model/v1")
    )
    canonical.extend(_pack_string(camera["model"]))
    canonical.extend(_pack_signed(camera["width"]))
    canonical.extend(_pack_signed(camera["height"]))
    for key in ("fx", "fy", "cx", "cy"):
        canonical.extend(_pack_double(float(camera[key])))
    for index in range(4):
        canonical.extend(_pack_double(float(camera[f"d{index}"])))
    return hashlib.sha256(canonical).digest()


def _revision(fingerprint: bytes) -> int:
    candidate = int.from_bytes(fingerprint[:8], byteorder="big")
    return candidate if candidate != 0 else 1


def _derive_rig(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    extrinsic = parameters["extrin_calib"]
    rotation_imu_from_lidar = _matrix3(extrinsic["extrinsic_R"])
    position_imu_from_lidar = _vector3(extrinsic["extrinsic_T"])
    rotation_lidar_from_imu = _transpose(rotation_imu_from_lidar)
    position_lidar_from_imu = _negate(
        _matrix_vector_product(
            rotation_lidar_from_imu,
            position_imu_from_lidar,
        )
    )

    multi_camera = parameters["multi_camera"]
    assert set(multi_camera["cameras"]) == set(CAMERA_NAMES)
    assert set(multi_camera["vio_extrinsics"]) == set(CAMERA_NAMES)
    rig: list[dict[str, Any]] = []
    for camera_id, name in enumerate(CAMERA_NAMES):
        projection = multi_camera["cameras"][name]
        lidar_extrinsic = multi_camera["vio_extrinsics"][name]
        rotation_camera_from_lidar = _matrix3(lidar_extrinsic["R"])
        position_camera_from_lidar = _vector3(lidar_extrinsic["T"])

        rotation_camera_from_imu = _matrix_product(
            rotation_camera_from_lidar,
            rotation_lidar_from_imu,
        )
        position_camera_from_imu = _add(
            _matrix_vector_product(
                rotation_camera_from_lidar,
                position_lidar_from_imu,
            ),
            position_camera_from_lidar,
        )
        rotation_imu_from_camera = _transpose(
            rotation_camera_from_imu
        )
        position_imu_from_camera = _negate(
            _matrix_vector_product(
                rotation_imu_from_camera,
                position_camera_from_imu,
            )
        )
        model_fingerprint = _model_fingerprint(projection)
        rig.append(
            {
                "camera_id": camera_id,
                "name": name,
                "projection": projection,
                "rotation_camera_from_imu": rotation_camera_from_imu,
                "position_camera_from_imu": position_camera_from_imu,
                "rotation_imu_from_camera": rotation_imu_from_camera,
                "position_imu_from_camera": position_imu_from_camera,
                "model_fingerprint": model_fingerprint,
                "model_revision": _revision(model_fingerprint),
            }
        )
    return rig


def _rig_fingerprint(rig: list[dict[str, Any]]) -> bytes:
    canonical = bytearray(
        _pack_string("livo/four-camera-pinhole-rig/v1")
    )
    canonical.extend(_pack_unsigned(len(rig)))
    for camera in sorted(rig, key=lambda value: value["camera_id"]):
        canonical.extend(struct.pack("B", camera["camera_id"]))
        canonical.extend(camera["model_fingerprint"])
        for value in _flatten(camera["rotation_camera_from_imu"]):
            canonical.extend(_pack_double(value))
        for value in camera["position_camera_from_imu"]:
            canonical.extend(_pack_double(value))
    return hashlib.sha256(canonical).digest()


def _parameters() -> dict[str, Any]:
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return document["/**"]["ros__parameters"]


def _assert_close(actual: tuple[float, ...], expected: tuple[float, ...]) -> None:
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=2.0e-15,
        )


def test_outdoor_yaml_independently_derives_t_ic_and_golden_fingerprint() -> None:
    rig = _derive_rig(_parameters())

    assert tuple(camera["model_revision"] for camera in rig) == (
        EXPECTED_MODEL_REVISIONS
    )
    for camera in rig:
        expected_rotation, expected_position = EXPECTED_T_IC[
            camera["name"]
        ]
        _assert_close(
            _flatten(camera["rotation_imu_from_camera"]),
            expected_rotation,
        )
        _assert_close(
            camera["position_imu_from_camera"],
            expected_position,
        )

    fingerprint = _rig_fingerprint(rig)
    assert fingerprint.hex() == EXPECTED_RIG_FINGERPRINT
    assert _revision(fingerprint) == EXPECTED_RIG_REVISION


@pytest.mark.parametrize(
    ("section", "camera_name", "field", "delta"),
    (
        ("projection", "front", "fx", 0.125),
        ("vio_translation", "left", 1, 0.001),
        ("lidar_translation", None, 2, 0.001),
    ),
)
def test_calibration_drift_changes_the_derived_rig_fingerprint(
    section: str,
    camera_name: str | None,
    field: str | int,
    delta: float,
) -> None:
    parameters = copy.deepcopy(_parameters())
    baseline = _rig_fingerprint(_derive_rig(parameters))

    if section == "projection":
        camera = parameters["multi_camera"]["cameras"][camera_name]
        camera[field] += delta
    elif section == "vio_translation":
        translation = parameters["multi_camera"]["vio_extrinsics"][
            camera_name
        ]["T"]
        translation[field] += delta
    else:
        translation = parameters["extrin_calib"]["extrinsic_T"]
        translation[field] += delta

    assert baseline.hex() == EXPECTED_RIG_FINGERPRINT
    assert _rig_fingerprint(_derive_rig(parameters)) != baseline
