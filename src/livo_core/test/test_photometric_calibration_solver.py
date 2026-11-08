"""Offline four-camera photometric calibration CLI contract tests."""

from __future__ import annotations

import errno
import fcntl
import base64
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
import numpy as np
from PIL import Image
import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOLVER_CLI = (
    PACKAGE_ROOT / "scripts" / "solve_photometric_calibration.py"
)
VALIDATOR_CLI = (
    PACKAGE_ROOT / "scripts" / "validate_photometric_calibration.py"
)
TEST_TRUST_DOMAIN = "livo-core-photometric-test-v1"
RUNTIME_PAYLOAD_FILENAME = (
    "photometric_calibration_runtime_payload.json"
)
RUNTIME_SIGNATURE_FILENAME = (
    "photometric_calibration_runtime_signature.json"
)
RECEIPT_HASH_SCHEME = (
    "sha256_runtime_payload_lf_ed25519_signature_v1"
)
RECEIPT_HASH_DOMAIN_SEPARATOR = (
    b"livo_core.photometric_calibration.receipt_hash.v1\x00"
)
SOLVER_MODULE_PATH = (
    PACKAGE_ROOT / "scripts" / "photometric_calibration_solver.py"
)
SOLVER_SPEC = importlib.util.spec_from_file_location(
    "photometric_calibration_solver_under_test",
    SOLVER_MODULE_PATH,
)
assert SOLVER_SPEC is not None and SOLVER_SPEC.loader is not None
SOLVER_MODULE = importlib.util.module_from_spec(SOLVER_SPEC)
sys.modules[SOLVER_SPEC.name] = SOLVER_MODULE
SOLVER_SPEC.loader.exec_module(SOLVER_MODULE)
SOLVER_CLI_SPEC = importlib.util.spec_from_file_location(
    "solve_photometric_calibration_under_test",
    SOLVER_CLI,
)
assert SOLVER_CLI_SPEC is not None and SOLVER_CLI_SPEC.loader is not None
SOLVER_CLI_MODULE = importlib.util.module_from_spec(SOLVER_CLI_SPEC)
script_directory = str(SOLVER_CLI.parent)
sys.path.insert(0, script_directory)
try:
    SOLVER_CLI_SPEC.loader.exec_module(SOLVER_CLI_MODULE)
finally:
    sys.path.remove(script_directory)


def _camera(camera_id: int) -> dict:
    return {
        "camera_id": camera_id,
        "name": ("front", "left", "back", "right")[camera_id],
        "black_level": 0.0,
        "white_level": 255.0,
        "black_white_provenance": "laboratory_measured",
        "source_encoding": "mono8",
        "grayscale_method": "identity",
    }


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "calibration_set_id": "synthetic-calibration",
        "source_id": "synthetic-lab-run",
        "frame_metadata": {
            "exposure_sec": {
                "field": "capture.exposure_sec",
                "provenance": "recorded_hardware",
            },
            "sensor_gain": {
                "field": "capture.sensor_gain",
                "provenance": "recorded_hardware",
            },
            "timestamp_semantics": "exposure_midpoint",
        },
        "cameras": [_camera(camera_id) for camera_id in range(4)],
        "captures": [],
    }


def _run_solver(
    manifest_path: Path,
    output_directory: Path,
    *extra_arguments: str,
    signed: bool = True,
) -> subprocess.CompletedProcess[str]:
    arguments = list(extra_arguments)
    if signed and "--signing-private-key" not in arguments:
        private_path, public_path = _write_signing_keypair(
            manifest_path.parent
        )
        arguments.extend(
            (
                "--signing-private-key",
                str(private_path),
                "--trusted-public-key",
                str(public_path),
                "--trust-domain",
                TEST_TRUST_DOMAIN,
            )
        )
    elif signed and "--trust-domain" not in arguments:
        arguments.extend(("--trust-domain", TEST_TRUST_DOMAIN))
    return subprocess.run(
        [
            sys.executable,
            str(SOLVER_CLI),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_directory),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _write_signing_keypair(tmp_path: Path) -> tuple[Path, Path]:
    private_path = tmp_path / "test-signing-private.pem"
    public_path = tmp_path / "test-trusted-public.pem"
    if private_path.exists() and public_path.exists():
        return private_path, public_path
    private_key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def _canonical_json_bytes(document: dict) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _receipt_hash_v1(
    payload_bytes: bytes,
    signature_bytes: bytes,
) -> str:
    preimage = (
        RECEIPT_HASH_DOMAIN_SEPARATOR
        + len(payload_bytes).to_bytes(8, "big")
        + payload_bytes
        + len(signature_bytes).to_bytes(8, "big")
        + signature_bytes
    )
    return hashlib.sha256(preimage).hexdigest()


def test_receipt_hash_v1_cross_language_golden() -> None:
    payload_bytes = (
        b'{"receipt_hash_scheme":'
        b'"sha256_runtime_payload_lf_ed25519_signature_v1"}\n'
    )

    assert _receipt_hash_v1(
        payload_bytes,
        bytes(range(64)),
    ) == (
        "dfe47714ab74419a2c7a32aa7507909b"
        "4c7f4f2a291aa82e5dced94175fae74f"
    )


def _write_recovery_staging_identity(
    staging: Path,
    output_directory: Path,
    transaction_id: str,
) -> None:
    config_bytes = b"status: available\n"
    (staging / "photometric_calibration.yaml").write_bytes(config_bytes)
    (staging / "photometric_calibration_quality.json").write_text(
        '{"calibrated": true}\n',
        encoding="utf-8",
    )
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    receipt_hash = None
    combined_hash = hashlib.sha256(
        _canonical_json_bytes(
            {
                "calibration_config_sha256": config_hash,
                "calibration_receipt_sha256": receipt_hash,
            }
        )
    ).hexdigest()
    marker = {
        "schema_version": 1,
        "marker_type": "photometric_calibration_publish_identity",
        "canonical_output_directory": str(
            output_directory.parent.resolve() / output_directory.name
        ),
        "transaction_id": transaction_id,
        "calibration_config_sha256": config_hash,
        "calibration_receipt_sha256": receipt_hash,
        "config_receipt_sha256": combined_hash,
    }
    (staging / "photometric_calibration_ownership.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_journal_document(
    output_directory: Path,
    transaction_id: str,
    *,
    phase: str,
    backup_name: str | None,
    staging_name: str | None = None,
    force: bool = True,
) -> dict:
    return {
        "schema_version": 2,
        "journal_type": "photometric_calibration_publish_transaction",
        "transaction_id": transaction_id,
        "canonical_output_directory": str(
            output_directory.parent.resolve() / output_directory.name
        ),
        "output_name": output_directory.name,
        "staging_name": (
            staging_name
            if staging_name is not None
            else f".{output_directory.name}.tmp-{transaction_id}"
        ),
        "backup_name": backup_name,
        "phase": phase,
        "force": force,
    }


def _assert_unavailable_result(
    completed: subprocess.CompletedProcess[str],
    output_directory: Path,
    blocker_fragment: str,
) -> None:
    assert completed.returncode == 3, completed.stderr
    calibration = yaml.safe_load(
        (output_directory / "photometric_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (output_directory / "photometric_calibration_quality.json").read_text(
            encoding="utf-8"
        )
    )
    assert calibration["status"] == "unavailable"
    assert report["calibrated"] is False
    assert any(
        blocker_fragment in blocker for blocker in report["blockers"]
    )
    assert not list(output_directory.glob("*_inverse_response_lut.txt"))
    assert not list(
        output_directory.glob("*_vignetting_transmission.txt")
    )


def _write_sensor_image(
    path: Path,
    irradiance: np.ndarray,
    exposure_sec: float,
    gamma: float,
    transmission: np.ndarray,
) -> None:
    linear_signal = np.clip(
        irradiance * exposure_sec * transmission,
        0.0,
        1.0,
    )
    raw = np.rint(255.0 * np.power(linear_signal, 1.0 / gamma))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(raw.astype(np.uint8), mode="L").save(path)


def _synthetic_calibration_manifest(
    tmp_path: Path,
    width: int = 8,
    height: int = 6,
) -> tuple[dict, dict]:
    yy, xx = np.mgrid[0:height, 0:width]
    radius_squared = (
        ((xx - 0.5 * (width - 1)) / width) ** 2
        + ((yy - 0.5 * (height - 1)) / height) ** 2
    )
    gammas = (1.8, 2.0, 2.2, 2.4)
    radiance_scales = (1.0, 1.12, 0.88, 1.20)
    radiance_offsets = (0.0, 0.18, -0.12, 0.10)
    transmissions = tuple(
        1.0 - (0.55 + 0.04 * camera_id) * radius_squared
        for camera_id in range(4)
    )
    manifest = _manifest()
    manifest["dataset_scope"] = []
    for camera in manifest["cameras"]:
        camera["image_width"] = width
        camera["image_height"] = height

    captures: list[dict] = []

    def add_frame(
        group: dict,
        camera_id: int,
        filename: str,
        irradiance: np.ndarray,
        exposure_sec: float,
    ) -> None:
        relative_path = Path("images") / filename
        _write_sensor_image(
            tmp_path / relative_path,
            irradiance,
            exposure_sec,
            gammas[camera_id],
            transmissions[camera_id],
        )
        group["frames"].append(
            {
                "camera_id": camera_id,
                "image_path": relative_path.as_posix(),
                "exposure_sec": exposure_sec,
                "sensor_gain": 1.0,
                "metadata_provenance": "recorded_hardware",
            }
        )
        if group["capture_kind"] == "cross_camera":
            group["frames"][-1]["gray_card_roi"] = [
                0,
                0,
                width,
                height,
            ]

    response_scenes = (
        0.20 + 2.0 * (xx + 2.0 * yy) / (width + 2.0 * height - 3.0),
        0.35 + 2.2 * (2.0 * xx + yy) / (2.0 * width + height - 3.0),
    )
    for camera_id in range(4):
        for scene_index, scene in enumerate(response_scenes):
            group = {
                "group_id": f"response-train-c{camera_id}-s{scene_index}",
                "capture_kind": "response",
                "split": "train",
                "static": True,
                "synchronized": False,
                "frames": [],
            }
            for exposure_index, exposure_sec in enumerate(
                (0.01, 0.03, 0.08, 0.20, 0.50)
            ):
                add_frame(
                    group,
                    camera_id,
                    (
                        f"response_train_c{camera_id}_s{scene_index}_"
                        f"e{exposure_index}.png"
                    ),
                    scene,
                    exposure_sec,
                )
            captures.append(group)
        held_out = {
            "group_id": f"response-held-c{camera_id}",
            "capture_kind": "response",
            "split": "held_out",
            "static": True,
            "synchronized": False,
            "frames": [],
        }
        held_scene = 0.28 + 1.9 * (
            3.0 * xx + 2.0 * yy
        ) / (3.0 * width + 2.0 * height - 5.0)
        for exposure_index, exposure_sec in enumerate(
            (0.015, 0.05, 0.15, 0.35)
        ):
            add_frame(
                held_out,
                camera_id,
                f"response_held_c{camera_id}_e{exposure_index}.png",
                held_scene,
                exposure_sec,
            )
        captures.append(held_out)

    for split, levels in (
        ("train", (5.0, 8.0, 11.0)),
        ("held_out", (7.0, 9.5)),
    ):
        for level_index, level in enumerate(levels):
            group = {
                "group_id": f"flatfield-{split}-{level_index}",
                "capture_kind": "flatfield",
                "split": split,
                "static": True,
                "synchronized": True,
                "frames": [],
            }
            for camera_id in range(4):
                add_frame(
                    group,
                    camera_id,
                    f"flat_{split}_{level_index}_c{camera_id}.png",
                    np.full((height, width), level),
                    0.05,
                )
            captures.append(group)

    for split, levels in (
        ("train", (4.0, 8.0, 13.0, 19.0, 24.0)),
        ("held_out", (6.0, 16.0)),
    ):
        for level_index, level in enumerate(levels):
            group = {
                "group_id": f"cross-{split}-{level_index}",
                "capture_kind": "cross_camera",
                "split": split,
                "static": True,
                "synchronized": True,
                "frames": [],
            }
            for camera_id in range(4):
                sensor_radiance = (
                    level - radiance_offsets[camera_id]
                ) / radiance_scales[camera_id]
                add_frame(
                    group,
                    camera_id,
                    f"cross_{split}_{level_index}_c{camera_id}.png",
                    np.full((height, width), sensor_radiance),
                    0.02,
                )
            captures.append(group)
    for group in captures:
        group["capture_session_id"] = f"session-{group['split']}"
        group["scene_id"] = group["group_id"]
    manifest["captures"] = captures
    truth = {
        "gammas": gammas,
        "radiance_scales": radiance_scales,
        "radiance_offsets": radiance_offsets,
        "transmissions": transmissions,
    }
    return manifest, truth


def test_missing_trusted_metadata_fails_with_unavailable_report(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    del manifest["frame_metadata"]["sensor_gain"]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 3
    calibration = yaml.safe_load(
        (output_directory / "photometric_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (output_directory / "photometric_calibration_quality.json").read_text(
            encoding="utf-8"
        )
    )
    assert calibration["status"] == "unavailable"
    assert all(
        camera["status"] == "unavailable"
        for camera in calibration["cameras"]
    )
    assert report["calibrated"] is False
    assert any(
        "frame_metadata.sensor_gain" in blocker
        for blocker in report["blockers"]
    )


def test_known_response_vignetting_and_cross_camera_affine_are_recovered(
    tmp_path: Path,
) -> None:
    manifest, truth = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 0, completed.stderr
    calibration = yaml.safe_load(
        (output_directory / "photometric_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (output_directory / "photometric_calibration_quality.json").read_text(
            encoding="utf-8"
        )
    )
    assert calibration["status"] == "available"
    assert report["calibrated"] is True
    assert report["held_out_quality"]["passed"] is True
    receipt_path = (
        output_directory / calibration["calibration_receipt_file"]
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload = dict(receipt)
    receipt_hash = receipt_payload.pop("receipt_sha256")
    signature = receipt_payload.pop("signature")
    runtime_payload_bytes = (
        output_directory / RUNTIME_PAYLOAD_FILENAME
    ).read_bytes()
    assert runtime_payload_bytes == _canonical_json_bytes(
        receipt_payload
    )
    assert receipt_payload["receipt_hash_scheme"] == (
        RECEIPT_HASH_SCHEME
    )
    assert receipt_hash == _receipt_hash_v1(
        runtime_payload_bytes,
        base64.b64decode(
            signature["signature_base64"],
            validate=True,
        ),
    )
    assert receipt["calibration_config_payload"] == calibration
    assert (
        calibration["source_bundle_sha256"]
        == receipt["source_bundle_sha256"]
    )
    config_path = output_directory / "photometric_calibration.yaml"
    marker = json.loads(
        (
            output_directory / "photometric_calibration_ownership.json"
        ).read_text(encoding="utf-8")
    )
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    receipt_file_hash = hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert marker["schema_version"] == 1
    assert marker["marker_type"] == (
        "photometric_calibration_publish_identity"
    )
    assert marker["canonical_output_directory"] == str(
        output_directory.resolve()
    )
    assert len(marker["transaction_id"]) == 32
    assert marker["calibration_config_sha256"] == config_hash
    assert marker["calibration_receipt_sha256"] == receipt_file_hash
    assert marker["config_receipt_sha256"] == hashlib.sha256(
        _canonical_json_bytes(
            {
                "calibration_config_sha256": config_hash,
                "calibration_receipt_sha256": receipt_file_hash,
            }
        )
    ).hexdigest()
    assert receipt["trust_boundary"] == (
        "ed25519_signature_requires_explicit_external_public_key_pin"
    )
    for camera in calibration["cameras"]:
        camera_id = camera["camera_id"]
        assert camera["status"] == "calibrated"
        lut = np.loadtxt(
            output_directory / camera["inverse_response_lut_file"]
        )
        assert lut.shape == (256,)
        assert np.all(np.diff(lut) >= -1.0e-12)
        expected_lut = np.power(
            np.linspace(0.0, 1.0, 256),
            truth["gammas"][camera_id],
        )
        assert np.mean(np.abs(lut - expected_lut)) < 0.07

        transmission = np.loadtxt(
            output_directory / camera["vignetting_transmission_file"]
        ).reshape(6, 8)
        expected_transmission = truth["transmissions"][camera_id]
        center = expected_transmission[2:4, 2:6]
        expected_transmission = expected_transmission / np.median(center)
        assert np.sqrt(
            np.mean((transmission - expected_transmission) ** 2)
        ) < 0.08
        assert abs(
            camera["radiance_scale"]
            - truth["radiance_scales"][camera_id]
        ) < 0.12
        assert abs(
            camera["radiance_offset"]
            - truth["radiance_offsets"][camera_id]
        ) < 0.20
        assert camera["covariance_scale"] >= 1.0
        assert (
            camera["source_bundle_sha256"]
            == calibration["source_bundle_sha256"]
        )
        assert camera["source_id"] == (
            "source-bundle-sha256:"
            f"{calibration['source_bundle_sha256']}"
        )

    validator = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_CLI),
            str(output_directory / "photometric_calibration.yaml"),
            "--trusted-public-key",
            str(tmp_path / "test-trusted-public.pem"),
            "--require-mode",
            "calibrated_direct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validator.returncode == 0, validator.stdout + validator.stderr


def test_vga_fixture_completes_inside_declared_resource_limits(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(
        tmp_path, width=640, height=480
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 0, completed.stderr
    calibration = yaml.safe_load(
        (output_directory / "photometric_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert calibration["status"] == "available"
    assert all(
        (
            output_directory
            / camera["vignetting_transmission_file"]
        ).stat().st_size
        > 0
        for camera in calibration["cameras"]
    )


def test_source_bundle_freezes_inputs_and_survives_source_replacement(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    manifest_snapshot = manifest_path.read_bytes()
    first_image_path = (
        tmp_path / manifest["captures"][0]["frames"][0]["image_path"]
    )
    output_directory = tmp_path / "result"
    completed = _run_solver(manifest_path, output_directory)
    assert completed.returncode == 0, completed.stderr
    config_path = output_directory / "photometric_calibration.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    bundle_path = output_directory / config["source_bundle_file"]
    bundle_bytes = bundle_path.read_bytes()
    bundle = json.loads(bundle_bytes)
    receipt = json.loads(
        (
            output_directory / config["calibration_receipt_file"]
        ).read_text(encoding="utf-8")
    )

    assert bundle["manifest_snapshot"]["raw_utf8"].encode(
        "utf-8"
    ) == manifest_snapshot
    assert bundle["manifest_snapshot"]["raw_sha256"] == hashlib.sha256(
        manifest_snapshot
    ).hexdigest()
    assert receipt["source_bundle_sha256"] == hashlib.sha256(
        bundle_bytes
    ).hexdigest()
    assert receipt["calibration_config_payload"] == config
    assert receipt["calibration_config_payload"]["dataset_scope"] == []
    assert receipt["calibration_config_payload"]["frame_metadata"][
        "timestamp_semantics"
    ] == "exposure_midpoint"
    capture_frames = [
        frame
        for group in bundle["captures"]
        for frame in group["frames"]
    ]
    assert all(
        len(frame["raw_image_sha256"]) == 64
        and len(frame["decoded_image_sha256"]) == 64
        for frame in capture_frames
    )

    manifest_path.write_text("replaced: true\n", encoding="utf-8")
    Image.fromarray(
        np.zeros((6, 8), dtype=np.uint8), mode="L"
    ).save(first_image_path)
    validator = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_CLI),
            str(config_path),
            "--trusted-public-key",
            str(tmp_path / "test-trusted-public.pem"),
            "--require-mode",
            "calibrated_direct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert validator.returncode == 0, validator.stdout + validator.stderr


def test_ed25519_signed_receipt_with_cli_pinned_key_is_trusted_ready(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    private_path, public_path = _write_signing_keypair(tmp_path)
    output_directory = tmp_path / "result"

    completed = _run_solver(
        manifest_path,
        output_directory,
        "--signing-private-key",
        str(private_path),
        "--trusted-public-key",
        str(public_path),
    )

    assert completed.returncode == 0, completed.stderr
    config_path = output_directory / "photometric_calibration.yaml"
    validator = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_CLI),
            str(config_path),
            "--trusted-public-key",
            str(public_path),
            "--require-mode",
            "calibrated_direct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validator.returncode == 0, validator.stdout + validator.stderr
    receipt = json.loads(
        (
            output_directory / "photometric_calibration_receipt.json"
        ).read_text(encoding="utf-8")
    )
    signature = receipt["signature"]
    assert signature["algorithm"] == "ed25519"
    assert len(signature["public_key_sha256"]) == 64
    rendered_receipt = json.dumps(receipt, sort_keys=True)
    assert str(private_path) not in rendered_receipt
    assert str(public_path) not in rendered_receipt


def test_signed_publish_emits_exact_runtime_payload_and_detached_signature(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    private_path, public_path = _write_signing_keypair(tmp_path)
    output_directory = tmp_path / "result"

    completed = _run_solver(
        manifest_path,
        output_directory,
        "--signing-private-key",
        str(private_path),
        "--trusted-public-key",
        str(public_path),
        "--trust-domain",
        TEST_TRUST_DOMAIN,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(
        (
            output_directory / "photometric_calibration_receipt.json"
        ).read_text(encoding="utf-8")
    )
    signed_payload = dict(receipt)
    signed_payload.pop("receipt_sha256")
    signature = signed_payload.pop("signature")
    payload_bytes = (
        output_directory / RUNTIME_PAYLOAD_FILENAME
    ).read_bytes()
    assert payload_bytes == _canonical_json_bytes(signed_payload)

    detached = json.loads(
        (
            output_directory / RUNTIME_SIGNATURE_FILENAME
        ).read_text(encoding="utf-8")
    )
    signature_bytes = base64.b64decode(
        signature["signature_base64"],
        validate=True,
    )
    expected_receipt_hash = _receipt_hash_v1(
        payload_bytes,
        signature_bytes,
    )
    assert receipt["receipt_hash_scheme"] == RECEIPT_HASH_SCHEME
    assert receipt["receipt_sha256"] == expected_receipt_hash
    assert detached == {
        "algorithm": "ed25519",
        "payload_file": RUNTIME_PAYLOAD_FILENAME,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "public_key_sha256": signature["public_key_sha256"],
        "receipt_hash_scheme": RECEIPT_HASH_SCHEME,
        "receipt_sha256": expected_receipt_hash,
        "schema_version": 1,
        "signature_base64": signature["signature_base64"],
        "signature_type": (
            "four_camera_photometric_calibration_runtime_signature"
        ),
        "solver": {
            "name": "livo_core_four_camera_photometric_calibration",
            "version": "2.0.0",
        },
        "trust_domain": TEST_TRUST_DOMAIN,
    }
    calibration = yaml.safe_load(
        (
            output_directory / "photometric_calibration.yaml"
        ).read_text(encoding="utf-8")
    )
    assert calibration["runtime_payload_file"] == (
        RUNTIME_PAYLOAD_FILENAME
    )
    assert calibration["runtime_signature_file"] == (
        RUNTIME_SIGNATURE_FILENAME
    )
    assert calibration["trust_root"] == {
        "public_key_sha256": signature["public_key_sha256"],
        "trust_domain": TEST_TRUST_DOMAIN,
    }


def test_validator_verifies_exact_runtime_payload_bytes(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    private_path, public_path = _write_signing_keypair(tmp_path)
    output_directory = tmp_path / "result"
    completed = _run_solver(
        manifest_path,
        output_directory,
        "--signing-private-key",
        str(private_path),
        "--trusted-public-key",
        str(public_path),
    )
    assert completed.returncode == 0, completed.stderr

    payload_path = output_directory / RUNTIME_PAYLOAD_FILENAME
    payload_bytes = payload_path.read_bytes()
    assert payload_bytes.endswith(b"\n")
    tampered_bytes = payload_bytes[:-1] + b" \n"
    payload_path.write_bytes(tampered_bytes)
    signature_path = output_directory / RUNTIME_SIGNATURE_FILENAME
    detached = json.loads(signature_path.read_text(encoding="utf-8"))
    detached["payload_sha256"] = hashlib.sha256(
        tampered_bytes
    ).hexdigest()
    signature_path.write_bytes(_canonical_json_bytes(detached))

    validator = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_CLI),
            str(
                output_directory / "photometric_calibration.yaml"
            ),
            "--trusted-public-key",
            str(public_path),
            "--require-mode",
            "calibrated_direct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert validator.returncode == 3
    assert (
        "runtime payload bytes are not canonical" in validator.stdout
        or "Ed25519 signature does not verify" in validator.stdout
    )


def test_unsigned_success_is_structural_but_never_trusted_ready(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(
        manifest_path,
        output_directory,
        signed=False,
    )

    assert completed.returncode == 4, completed.stderr
    receipt = json.loads(
        (
            output_directory / "photometric_calibration_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert "signature" not in receipt
    assert receipt["trust_boundary"] == (
        "unsigned_content_integrity_only_not_trusted"
    )
    structural = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_CLI),
            str(output_directory / "photometric_calibration.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert structural.returncode == 4
    required = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_CLI),
            str(output_directory / "photometric_calibration.yaml"),
            "--require-mode",
            "calibrated_direct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert required.returncode == 3


@pytest.mark.parametrize(
    ("calibrated", "trusted", "status", "expected"),
    (
        (True, True, "committed", 0),
        (True, False, "committed", 4),
        (False, False, "committed", 3),
        (True, True, "committed_with_warning", 5),
    ),
)
def test_solver_cli_exit_code_distinguishes_publish_outcomes(
    calibrated: bool,
    trusted: bool,
    status: str,
    expected: int,
) -> None:
    outcome = SOLVER_MODULE.PublishOutcome(
        committed=True,
        status=status,
    )

    assert SOLVER_CLI_MODULE._published_exit_code(
        calibrated=calibrated,
        trusted=trusted,
        outcome=outcome,
    ) == expected


@pytest.mark.parametrize("tamper_target", ("receipt", "lut"))
def test_validator_rejects_calibration_content_tampering(
    tmp_path: Path,
    tamper_target: str,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    completed = _run_solver(manifest_path, output_directory)
    assert completed.returncode == 0, completed.stderr
    calibration_path = output_directory / "photometric_calibration.yaml"
    calibration = yaml.safe_load(
        calibration_path.read_text(encoding="utf-8")
    )
    if tamper_target == "receipt":
        receipt_path = (
            output_directory / calibration["calibration_receipt_file"]
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["held_out_metrics"]["0"][
            "response_held_out_log_rmse"
        ] = 0.0
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        lut_path = (
            output_directory
            / calibration["cameras"][0]["inverse_response_lut_file"]
        )
        lut_path.write_bytes(lut_path.read_bytes() + b"\n")

    validator = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_CLI),
            str(calibration_path),
            "--trusted-public-key",
            str(tmp_path / "test-trusted-public.pem"),
            "--require-mode",
            "calibrated_direct",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert validator.returncode == 3
    if tamper_target == "receipt":
        assert (
            "runtime payload does not exactly match receipt signed payload"
            in validator.stdout
        )
    else:
        assert "hash does not match" in validator.stdout


@pytest.mark.parametrize("missing_field", ("exposure_sec", "sensor_gain"))
def test_frame_without_exposure_or_gain_fails_closed(
    tmp_path: Path,
    missing_field: str,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    del manifest["captures"][0]["frames"][0][missing_field]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        f"{missing_field} must be finite and positive",
    )


def test_unmeasured_black_white_levels_fail_before_image_loading(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["cameras"][2]["black_white_provenance"] = "unavailable"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "black/white provenance is not trusted"
    )


def test_timestamp_semantics_must_be_an_exposure_enum(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["frame_metadata"]["timestamp_semantics"] = "midpoint"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "timestamp_semantics must be one of"
    )


def test_declared_mono8_rejects_sixteen_bit_image_before_conversion(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    frame = manifest["captures"][0]["frames"][0]
    Image.fromarray(np.full((6, 8), 1024, dtype=np.uint16)).save(
        tmp_path / frame["image_path"]
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "must decode as uint8 before any mode conversion",
    )


def test_declared_mono8_rejects_float_nan_image_before_conversion(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    frame = manifest["captures"][0]["frames"][0]
    float_path = tmp_path / "images" / "float-nan.tiff"
    values = np.full((6, 8), 0.5, dtype=np.float32)
    values[0, 0] = np.nan
    Image.fromarray(values, mode="F").save(float_path)
    frame["image_path"] = "images/float-nan.tiff"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "must decode as uint8 before any mode conversion",
    )


def test_source_encoding_must_match_decoded_image_mode(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest["cameras"][0]["source_encoding"] = "rgb8"
    manifest["cameras"][0]["grayscale_method"] = "bt601_luma_uint8"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "does not match declared rgb8"
    )


def test_explicit_rgb8_bt601_capture_is_supported(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest["cameras"][0]["source_encoding"] = "rgb8"
    manifest["cameras"][0]["grayscale_method"] = "bt601_luma_uint8"
    for group in manifest["captures"]:
        for frame in group["frames"]:
            if frame["camera_id"] != 0:
                continue
            image_path = tmp_path / frame["image_path"]
            mono = np.asarray(Image.open(image_path))
            rgb = np.repeat(mono[:, :, None], 3, axis=2)
            Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(image_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 0, completed.stderr
    calibration = yaml.safe_load(
        (output_directory / "photometric_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert calibration["status"] == "available"


def test_lab_manifest_cannot_claim_outdoor_dataset_scope(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["dataset_scope"] = ["outdoor9"]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "laboratory calibration cannot claim an outdoor dataset scope",
    )


def test_declared_image_dimensions_over_resource_limit_fail_before_loading(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    for camera in manifest["cameras"]:
        camera["image_width"] = 100_000
        camera["image_height"] = 100_000
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "declared image resource limit"
    )


def test_manifest_file_size_limit_fails_closed(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "padding: " + "x" * (5 * 1024 * 1024),
        encoding="utf-8",
    )
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "manifest encoded size"
    )


def test_capture_frame_count_limit_fails_before_image_loading(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    prototype = dict(manifest["captures"][0]["frames"][0])
    manifest["captures"][0]["frames"] = [
        dict(prototype) for _ in range(513)
    ]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "capture frame count exceeds"
    )


def test_response_dense_matrix_budget_fails_closed(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    width = 100
    height = 100
    yy, xx = np.mgrid[0:height, 0:width]
    irradiance = 0.1 + 2.0 * (xx + yy) / (width + height - 2)
    for camera in manifest["cameras"]:
        camera["image_width"] = width
        camera["image_height"] = height
    frames = []
    for index, exposure_sec in enumerate(
        np.geomspace(0.01, 0.5, 16)
    ):
        path = Path("images") / f"dense-{index}.png"
        _write_sensor_image(
            tmp_path / path,
            irradiance,
            float(exposure_sec),
            2.0,
            np.ones((height, width)),
        )
        frames.append(
            {
                "camera_id": 0,
                "image_path": path.as_posix(),
                "exposure_sec": float(exposure_sec),
                "sensor_gain": 1.0,
                "metadata_provenance": "recorded_hardware",
            }
        )
    manifest["captures"] = [
        {
            "group_id": f"dense-response-{index}",
            "capture_session_id": "session-train",
            "scene_id": f"dense-scene-{index}",
            "capture_kind": "response",
            "split": "train",
            "static": True,
            "synchronized": False,
            "frames": [dict(frame) for frame in frames],
        }
        for index in range(8)
    ]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "response dense matrix resource limit exceeded",
    )


@pytest.mark.parametrize(
    "missing_field", ("capture_session_id", "scene_id")
)
def test_capture_group_requires_physical_identity(
    tmp_path: Path,
    missing_field: str,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    del manifest["captures"][0][missing_field]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, f"{missing_field} must be nonempty"
    )


@pytest.mark.parametrize(
    ("identity_field", "blocker"),
    (
        (
            "capture_session_id",
            "capture_session_id is reused across train and held_out",
        ),
        ("scene_id", "scene_id is reused across train and held_out"),
    ),
)
def test_train_and_held_out_require_distinct_physical_identity(
    tmp_path: Path,
    identity_field: str,
    blocker: str,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    train_group = next(
        group for group in manifest["captures"] if group["split"] == "train"
    )
    held_out_group = next(
        group
        for group in manifest["captures"]
        if group["split"] == "held_out"
    )
    held_out_group[identity_field] = train_group[identity_field]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(completed, output_directory, blocker)


def test_nonstatic_response_capture_fails_closed(tmp_path: Path) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest["captures"][0]["static"] = False
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "must declare static: true"
    )


def test_saturated_response_coverage_fails_closed(tmp_path: Path) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    for group in manifest["captures"]:
        if (
            group["capture_kind"] == "response"
            and group["split"] == "train"
            and group["frames"][0]["camera_id"] == 0
        ):
            for frame in group["frames"]:
                Image.fromarray(
                    np.full((6, 8), 255, dtype=np.uint8), mode="L"
                ).save(tmp_path / frame["image_path"])
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "fully saturated"
    )


def test_flatfield_held_out_rejects_half_missing_edge_ring(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    for group in manifest["captures"]:
        if (
            group["capture_kind"] == "flatfield"
            and group["split"] == "held_out"
        ):
            camera_zero = next(
                frame
                for frame in group["frames"]
                if frame["camera_id"] == 0
            )
            image_path = tmp_path / camera_zero["image_path"]
            image = np.asarray(Image.open(image_path)).copy()
            image[:, :4] = 255
            Image.fromarray(image.astype(np.uint8), mode="L").save(image_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "flatfield held-out edge coverage",
    )


def test_flatfield_held_out_rejects_interior_grid_hole(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    for group in manifest["captures"]:
        if (
            group["capture_kind"] == "flatfield"
            and group["split"] == "held_out"
        ):
            camera_zero = next(
                frame
                for frame in group["frames"]
                if frame["camera_id"] == 0
            )
            image_path = tmp_path / camera_zero["image_path"]
            image = np.asarray(Image.open(image_path)).copy()
            image[2:4, 2:4] = 255
            Image.fromarray(image.astype(np.uint8), mode="L").save(image_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "flatfield held-out grid coverage",
    )


def test_nonmonotonic_response_sequence_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    group = manifest["captures"][0]
    first_path = group["frames"][0]["image_path"]
    group["frames"][0]["image_path"] = group["frames"][-1]["image_path"]
    group["frames"][-1]["image_path"] = first_path
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "response sequence is non-monotonic"
    )


def test_cross_camera_held_out_failure_cannot_publish_calibrated(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    for group in manifest["captures"]:
        if (
            group["capture_kind"] == "cross_camera"
            and group["split"] == "held_out"
        ):
            camera_one = next(
                frame
                for frame in group["frames"]
                if frame["camera_id"] == 1
            )
            Image.fromarray(
                np.full((6, 8), 80, dtype=np.uint8), mode="L"
            ).save(tmp_path / camera_one["image_path"])
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "cross-camera held-out relative RMS"
    )


def test_cross_camera_tiny_gray_card_roi_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    cross_frame = next(
        group["frames"][0]
        for group in manifest["captures"]
        if group["capture_kind"] == "cross_camera"
    )
    cross_frame["gray_card_roi"] = [0, 0, 1, 1]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "gray-card area is below minimum"
    )


def test_cross_camera_frame_without_roi_or_mask_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    cross_frame = next(
        group["frames"][0]
        for group in manifest["captures"]
        if group["capture_kind"] == "cross_camera"
    )
    del cross_frame["gray_card_roi"]
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "must declare exactly one of gray_card_roi or gray_card_mask_path",
    )


def test_cross_camera_gray_card_requires_effective_pixel_ratio(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    cross_frame = next(
        group["frames"][0]
        for group in manifest["captures"]
        if group["capture_kind"] == "cross_camera"
    )
    image_path = tmp_path / cross_frame["image_path"]
    image = np.asarray(Image.open(image_path)).copy()
    image[:, :4] = 255
    Image.fromarray(image.astype(np.uint8), mode="L").save(image_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "insufficient valid gray-card pixels",
    )


def test_cross_camera_affine_ignores_background_outside_roi(
    tmp_path: Path,
) -> None:
    manifest, truth = _synthetic_calibration_manifest(tmp_path)
    roi = [2, 1, 4, 4]
    roi_mask = np.zeros((6, 8), dtype=bool)
    roi_mask[1:5, 2:6] = True
    for group in manifest["captures"]:
        if group["capture_kind"] != "cross_camera":
            continue
        for frame in group["frames"]:
            frame["gray_card_roi"] = roi
            image_path = tmp_path / frame["image_path"]
            image = np.asarray(Image.open(image_path)).copy()
            background = 15 + 45 * frame["camera_id"]
            image[~roi_mask] = background
            Image.fromarray(image.astype(np.uint8), mode="L").save(image_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 0, completed.stderr
    calibration = yaml.safe_load(
        (output_directory / "photometric_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )
    for camera in calibration["cameras"]:
        camera_id = camera["camera_id"]
        assert abs(
            camera["radiance_scale"]
            - truth["radiance_scales"][camera_id]
        ) < 0.12


def test_cross_camera_binary_mask_is_hashed_in_source_bundle(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    mask_values = np.zeros((6, 8), dtype=np.uint8)
    mask_values[1:5, 2:6] = 255
    mask_path = tmp_path / "masks" / "gray-card.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(mask_values, mode="L").save(mask_path)
    for group in manifest["captures"]:
        if group["capture_kind"] != "cross_camera":
            continue
        for frame in group["frames"]:
            del frame["gray_card_roi"]
            frame["gray_card_mask_path"] = "masks/gray-card.png"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 0, completed.stderr
    source_bundle = json.loads(
        (
            output_directory
            / "photometric_calibration_source_bundle.json"
        ).read_text(encoding="utf-8")
    )
    expected_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    cross_frames = [
        frame
        for group in source_bundle["captures"]
        if group["capture_kind"] == "cross_camera"
        for frame in group["frames"]
    ]
    assert cross_frames
    assert all(
        frame["raw_gray_card_mask_sha256"] == expected_hash
        and len(frame["decoded_gray_card_mask_sha256"]) == 64
        for frame in cross_frames
    )


def test_solver_hashes_and_decodes_each_physical_source_from_one_fd_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    mask_values = np.zeros((6, 8), dtype=np.uint8)
    mask_values[1:5, 2:6] = 255
    mask_path = tmp_path / "masks" / "gray-card.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(mask_values, mode="L").save(mask_path)
    for group in manifest["captures"]:
        if group["capture_kind"] != "cross_camera":
            continue
        for frame in group["frames"]:
            del frame["gray_card_roi"]
            frame["gray_card_mask_path"] = "masks/gray-card.png"
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    tracked_paths = {
        manifest_path.resolve(),
        mask_path.resolve(),
        *{
            (tmp_path / frame["image_path"]).resolve()
            for group in manifest["captures"]
            for frame in group["frames"]
        },
    }
    read_counts = {path: 0 for path in tracked_paths}
    original_os_read = SOLVER_MODULE.os.read

    def count_fd_reads(descriptor: int, byte_count: int) -> bytes:
        raw_bytes = original_os_read(descriptor, byte_count)
        if raw_bytes:
            resolved = Path(
                os.readlink(f"/proc/self/fd/{descriptor}")
            ).resolve()
            if resolved in read_counts:
                read_counts[resolved] += 1
        return raw_bytes

    monkeypatch.setattr(SOLVER_MODULE.os, "read", count_fd_reads)

    snapshot = SOLVER_MODULE.load_manifest(manifest_path)
    SOLVER_MODULE.load_capture_groups(snapshot.document, manifest_path)

    assert set(read_counts.values()) == {1}


def test_train_image_reused_as_held_out_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    train_group = next(
        group
        for group in manifest["captures"]
        if (
            group["capture_kind"] == "response"
            and group["split"] == "train"
            and group["frames"][0]["camera_id"] == 0
        )
    )
    held_out_group = next(
        group
        for group in manifest["captures"]
        if (
            group["capture_kind"] == "response"
            and group["split"] == "held_out"
            and group["frames"][0]["camera_id"] == 0
        )
    )
    held_out_group["frames"][0]["image_path"] = (
        train_group["frames"][0]["image_path"]
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "reused across train and held_out",
    )


def test_hardlink_alias_across_splits_is_rejected_by_inode(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    train_group = next(
        group
        for group in manifest["captures"]
        if group["capture_kind"] == "response"
        and group["split"] == "train"
    )
    held_out_group = next(
        group
        for group in manifest["captures"]
        if group["capture_kind"] == "response"
        and group["split"] == "held_out"
        and group["frames"][0]["camera_id"]
        == train_group["frames"][0]["camera_id"]
    )
    train_path = tmp_path / train_group["frames"][0]["image_path"]
    held_out_path = tmp_path / held_out_group["frames"][0]["image_path"]
    held_out_path.unlink()
    held_out_path.hardlink_to(train_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "capture source inode is reused by an alias",
    )


def test_image_path_replacement_during_fd_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    snapshot = SOLVER_MODULE.load_manifest(manifest_path)
    image_path = (
        tmp_path / manifest["captures"][0]["frames"][0]["image_path"]
    ).resolve()
    original_read = SOLVER_MODULE.os.read
    replaced = False

    def replace_path_after_first_read(
        descriptor: int,
        byte_count: int,
    ) -> bytes:
        nonlocal replaced
        raw_bytes = original_read(descriptor, byte_count)
        if (
            raw_bytes
            and not replaced
            and Path(
                os.readlink(f"/proc/self/fd/{descriptor}")
            ).resolve() == image_path
        ):
            replaced = True
            replacement = image_path.with_name(
                f".{image_path.name}.replacement"
            )
            Image.fromarray(
                np.zeros((6, 8), dtype=np.uint8),
                mode="L",
            ).save(replacement, format="PNG")
            SOLVER_MODULE.os.replace(replacement, image_path)
        return raw_bytes

    monkeypatch.setattr(
        SOLVER_MODULE.os,
        "read",
        replace_path_after_first_read,
    )

    with pytest.raises(
        SOLVER_MODULE.CalibrationInputError,
        match="capture image changed during snapshot",
    ):
        SOLVER_MODULE.load_capture_groups(
            snapshot.document,
            manifest_path,
        )

    assert replaced is True


def test_inode_cache_rejects_same_file_changed_between_snapshots(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"first-value")
    snapshot_cache: dict[tuple[int, int], object] = {}
    first_snapshot = SOLVER_MODULE._snapshot_regular_file(
        source_path,
        1024,
        "capture image",
        snapshot_cache,
    )
    source_path.write_bytes(b"other-value")
    changed_time = first_snapshot.mtime_ns + 1_000_000_000
    os.utime(
        source_path,
        ns=(changed_time, changed_time),
    )

    with pytest.raises(
        SOLVER_MODULE.CalibrationInputError,
        match="changed since its first snapshot",
    ):
        SOLVER_MODULE._snapshot_regular_file(
            source_path,
            1024,
            "capture image",
            snapshot_cache,
        )


def test_final_component_image_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    frame = manifest["captures"][0]["frames"][0]
    original_path = tmp_path / frame["image_path"]
    symlink_path = original_path.with_name("symlink.png")
    symlink_path.symlink_to(original_path.name)
    frame["image_path"] = symlink_path.relative_to(tmp_path).as_posix()
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "capture image must not be a symlink",
    )


def test_manifest_growth_after_open_is_capped_and_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, _manifest())
    original_read = SOLVER_MODULE.os.read
    injected = False

    def grow_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal injected
        data = original_read(descriptor, size)
        descriptor_path = Path(f"/proc/self/fd/{descriptor}")
        if (
            not injected
            and descriptor_path.exists()
            and descriptor_path.resolve() == manifest_path.resolve()
        ):
            injected = True
            with manifest_path.open("ab") as stream:
                stream.write(
                    b"x" * (SOLVER_MODULE.MAX_MANIFEST_BYTES + 1)
                )
        return data

    monkeypatch.setattr(SOLVER_MODULE.os, "read", grow_after_first_read)

    with pytest.raises(
        SOLVER_MODULE.CalibrationInputError,
        match="manifest encoded size exceeds resource limit",
    ):
        SOLVER_MODULE.load_manifest(manifest_path)


def test_train_content_copied_to_distinct_held_out_path_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    train_frame = next(
        group["frames"][0]
        for group in manifest["captures"]
        if group["split"] == "train"
    )
    held_out_frame = next(
        group["frames"][0]
        for group in manifest["captures"]
        if group["split"] == "held_out"
    )
    (tmp_path / held_out_frame["image_path"]).write_bytes(
        (tmp_path / train_frame["image_path"]).read_bytes()
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "content SHA-256 is reused across train and held_out",
    )


def test_different_png_compression_with_same_decoded_pixels_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    train_frame = next(
        group["frames"][0]
        for group in manifest["captures"]
        if group["split"] == "train"
    )
    held_out_frame = next(
        group["frames"][0]
        for group in manifest["captures"]
        if group["split"] == "held_out"
    )
    train_path = tmp_path / train_frame["image_path"]
    held_out_path = tmp_path / held_out_frame["image_path"]
    decoded = np.asarray(Image.open(train_path)).copy()
    Image.fromarray(decoded.astype(np.uint8), mode="L").save(
        held_out_path,
        compress_level=0,
    )
    assert train_path.read_bytes() != held_out_path.read_bytes()
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "decoded image SHA-256 is reused across train and held_out",
    )


def test_rgb_inputs_with_same_canonical_grayscale_cannot_cross_split(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    camera = manifest["cameras"][0]
    camera["source_encoding"] = "rgb8"
    camera["grayscale_method"] = "bt601_luma_uint8"
    camera_zero_frames = [
        frame
        for group in manifest["captures"]
        for frame in group["frames"]
        if frame["camera_id"] == 0
    ]
    for frame in camera_zero_frames:
        path = tmp_path / frame["image_path"]
        mono = np.asarray(Image.open(path))
        Image.fromarray(
            np.repeat(mono[:, :, None], 3, axis=2),
            mode="RGB",
        ).save(path)
    train_frame = next(
        frame
        for group in manifest["captures"]
        if group["capture_kind"] == "response"
        and group["split"] == "train"
        for frame in group["frames"]
        if frame["camera_id"] == 0
    )
    held_out_frame = next(
        frame
        for group in manifest["captures"]
        if group["capture_kind"] == "response"
        and group["split"] == "held_out"
        for frame in group["frames"]
        if frame["camera_id"] == 0
    )
    red = np.zeros((6, 8, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    green = np.zeros((6, 8, 3), dtype=np.uint8)
    green[:, :, 1] = 130
    Image.fromarray(red, mode="RGB").save(
        tmp_path / train_frame["image_path"]
    )
    Image.fromarray(green, mode="RGB").save(
        tmp_path / held_out_frame["image_path"]
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "decoded image SHA-256 is reused across train and held_out",
    )


def test_capture_path_escape_fails_before_calibration(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    outside = tmp_path.parent / "outside-calibration-image.png"
    Image.fromarray(np.full((6, 8), 100, dtype=np.uint8), mode="L").save(
        outside
    )
    manifest["captures"][0]["frames"][0]["image_path"] = (
        "../outside-calibration-image.png"
    )
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed, output_directory, "escapes manifest directory"
    )


def test_nonfinite_capture_metadata_fails_closed(tmp_path: Path) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest["captures"][0]["frames"][0]["exposure_sec"] = float("nan")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "exposure_sec must be finite and positive",
    )


@pytest.mark.parametrize(
    ("exposure_sec", "sensor_gain"),
    ((1.0e308, 1.0e308), (1.0e-320, 1.0e-320)),
)
def test_exposure_gain_product_uses_checked_log_domain(
    tmp_path: Path,
    exposure_sec: float,
    sensor_gain: float,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    frame = manifest["captures"][0]["frames"][0]
    frame["exposure_sec"] = exposure_sec
    frame["sensor_gain"] = sensor_gain
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "exposure_sec * sensor_gain is outside supported log-domain",
    )
    assert "Traceback" not in completed.stderr


def test_oversized_yaml_integer_metadata_fails_without_traceback(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest["captures"][0]["frames"][0]["sensor_gain"] = 10**1000
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"

    completed = _run_solver(manifest_path, output_directory)

    _assert_unavailable_result(
        completed,
        output_directory,
        "sensor_gain must be finite and positive",
    )
    assert "Traceback" not in completed.stderr


def test_existing_output_requires_force_and_replacement_is_complete(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    first = _run_solver(manifest_path, output_directory)
    assert first.returncode == 0, first.stderr
    sentinel = output_directory / "must-not-survive-force.txt"
    sentinel.write_text("preserve without force\n", encoding="utf-8")

    refused = _run_solver(manifest_path, output_directory)

    assert refused.returncode == 2
    assert sentinel.read_text(encoding="utf-8") == "preserve without force\n"

    replaced = _run_solver(manifest_path, output_directory, "--force")

    assert replaced.returncode == 0, replaced.stderr
    assert not sentinel.exists()
    assert (
        output_directory / "photometric_calibration.yaml"
    ).is_file()
    assert (
        output_directory / "photometric_calibration_ownership.json"
    ).is_file()
    assert not list(tmp_path.glob(".result.tmp-*"))
    backups = list(tmp_path.glob(".result.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / sentinel.name).read_text(
        encoding="utf-8"
    ) == "preserve without force\n"
    assert "previous output retained" in replaced.stderr
    assert not (tmp_path / ".result.publish-journal.json").exists()


def test_force_rejects_output_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    target = tmp_path / "symlink-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    output_directory = tmp_path / "result"
    output_directory.symlink_to(target, target_is_directory=True)

    completed = _run_solver(
        manifest_path, output_directory, "--force"
    )

    assert completed.returncode == 2
    assert "symlink" in completed.stderr
    assert output_directory.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"


def test_force_retains_unowned_existing_directory_as_backup(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    output_directory.mkdir()
    sentinel = output_directory / "user-data.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    completed = _run_solver(
        manifest_path, output_directory, "--force"
    )

    assert completed.returncode == 0, completed.stderr
    backups = list(tmp_path.glob(".result.backup-*"))
    assert len(backups) == 1
    assert (backups[0] / sentinel.name).read_text(
        encoding="utf-8"
    ) == "preserve\n"
    assert not sentinel.exists()


def test_parent_publish_lock_prevents_concurrent_writer(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    lock_path = tmp_path / ".result.publish.lock"
    with lock_path.open("w", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 2
    assert "publish lock is held" in completed.stderr
    assert not output_directory.exists()


def test_current_failure_cleans_only_its_staging_and_preserves_crash_orphan(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "result"
    crash_orphan = tmp_path / ".result.tmp-crash-orphan"
    crash_orphan.mkdir()
    sentinel = crash_orphan / "valuable.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid numeric output"):
        SOLVER_MODULE.publish_documents(
            output_directory,
            {"status": "available"},
            {"calibrated": True},
            {"../invalid.txt": np.asarray([1.0])},
            force=False,
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"
    assert list(tmp_path.glob(".result.tmp-*")) == [crash_orphan]
    assert not output_directory.exists()
    assert not (tmp_path / ".result.publish-journal.json").exists()


def test_atomic_publish_rename_never_replaces_existing_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_sentinel = source / "source.txt"
    source_sentinel.write_text("source\n", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    destination_sentinel = destination / "destination.txt"
    destination_sentinel.write_text("destination\n", encoding="utf-8")

    with pytest.raises(OSError):
        SOLVER_MODULE._rename_no_replace(source, destination)

    assert source_sentinel.read_text(encoding="utf-8") == "source\n"
    assert destination_sentinel.read_text(
        encoding="utf-8"
    ) == "destination\n"


@pytest.mark.parametrize("failure_errno", (errno.ENOSYS, errno.EXDEV))
def test_force_exchange_failure_is_precommit_and_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_errno: int,
) -> None:
    output_directory = tmp_path / "result"
    output_directory.mkdir()
    old_sentinel = output_directory / "old.txt"
    old_sentinel.write_text("old\n", encoding="utf-8")

    def fail_exchange(_first: Path, _second: Path) -> None:
        raise OSError(failure_errno, "injected rename exchange failure")

    monkeypatch.setattr(
        SOLVER_MODULE,
        "_rename_exchange",
        fail_exchange,
    )

    with pytest.raises(OSError) as error:
        SOLVER_MODULE.publish_documents(
            output_directory,
            {"status": "available"},
            {"calibrated": True},
            force=True,
        )

    assert error.value.errno == failure_errno
    assert old_sentinel.read_text(encoding="utf-8") == "old\n"
    assert not (
        output_directory / "photometric_calibration.yaml"
    ).exists()
    assert not list(tmp_path.glob(".result.backup-*"))
    assert not list(tmp_path.glob(".result.tmp-*"))
    assert not (tmp_path / ".result.publish-journal.json").exists()


def test_backup_moved_journal_never_grants_automatic_publish_authority(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    transaction_id = "a" * 32
    staging = tmp_path / f".result.tmp-{transaction_id}"
    staging.mkdir()
    _write_recovery_staging_identity(
        staging, output_directory, transaction_id
    )
    backup = tmp_path / f".result.backup-{transaction_id}"
    backup.mkdir()
    sentinel = backup / "old.txt"
    sentinel.write_text("old result\n", encoding="utf-8")
    journal = tmp_path / ".result.publish-journal.json"
    journal.write_text(
        json.dumps(
            _publish_journal_document(
                output_directory,
                transaction_id,
                phase="backup_moved",
                backup_name=backup.name,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 2
    assert "manual recovery required" in completed.stderr
    assert not output_directory.exists()
    assert staging.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "old result\n"
    assert journal.is_file()


def test_prepared_journal_never_grants_automatic_publish_authority(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    transaction_id = "b" * 32
    staging = tmp_path / f".result.tmp-{transaction_id}"
    staging.mkdir()
    _write_recovery_staging_identity(
        staging, output_directory, transaction_id
    )
    journal = tmp_path / ".result.publish-journal.json"
    journal.write_text(
        json.dumps(
            _publish_journal_document(
                output_directory,
                transaction_id,
                phase="prepared",
                backup_name=None,
                force=False,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 2
    assert "manual recovery required" in completed.stderr
    assert not output_directory.exists()
    assert staging.is_dir()
    assert journal.is_file()


def test_prepared_journal_preserves_ambiguous_output_and_staging(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    output_directory.mkdir()
    output_sentinel = output_directory / "old.txt"
    output_sentinel.write_text("old\n", encoding="utf-8")
    transaction_id = "c" * 32
    staging = tmp_path / f".result.tmp-{transaction_id}"
    staging.mkdir()
    _write_recovery_staging_identity(
        staging, output_directory, transaction_id
    )
    journal = tmp_path / ".result.publish-journal.json"
    journal.write_text(
        json.dumps(
            _publish_journal_document(
                output_directory,
                transaction_id,
                phase="prepared",
                backup_name=(
                    f".result.backup-{transaction_id}"
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 2
    assert "manual recovery required" in completed.stderr
    assert output_sentinel.read_text(encoding="utf-8") == "old\n"
    assert staging.is_dir()
    assert journal.is_file()


def test_recovery_rejects_backup_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    transaction_id = "d" * 32
    staging = tmp_path / f".result.tmp-{transaction_id}"
    staging.mkdir()
    _write_recovery_staging_identity(
        staging, output_directory, transaction_id
    )
    external = tmp_path / "external-backup-target"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("untouched\n", encoding="utf-8")
    backup = tmp_path / f".result.backup-{transaction_id}"
    backup.symlink_to(external, target_is_directory=True)
    journal = tmp_path / ".result.publish-journal.json"
    journal.write_text(
        json.dumps(
            _publish_journal_document(
                output_directory,
                transaction_id,
                phase="backup_moved",
                backup_name=backup.name,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 2
    assert "manual recovery required" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "untouched\n"
    assert backup.is_symlink()
    assert staging.is_dir()
    assert journal.is_file()
    assert not output_directory.exists()


def test_recovery_rejects_journal_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    external_journal = tmp_path / "external-journal.json"
    external_journal.write_text("valuable\n", encoding="utf-8")
    journal = tmp_path / ".result.publish-journal.json"
    journal.symlink_to(external_journal)

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 2
    assert "journal must not be a symlink" in completed.stderr
    assert external_journal.read_text(encoding="utf-8") == "valuable\n"
    assert journal.is_symlink()
    assert not output_directory.exists()


def test_recovery_journal_cannot_name_path_outside_exact_output_prefix(
    tmp_path: Path,
) -> None:
    manifest, _ = _synthetic_calibration_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, manifest)
    output_directory = tmp_path / "result"
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("untouched\n", encoding="utf-8")
    transaction_id = "e" * 32
    staging = (
        tmp_path / f".result.tmp-{transaction_id}-valuable-sentinel"
    )
    staging.mkdir()
    nested_sentinel = staging / "sentinel.txt"
    nested_sentinel.write_text("untouched\n", encoding="utf-8")
    journal = tmp_path / ".result.publish-journal.json"
    journal.write_text(
        json.dumps(
            _publish_journal_document(
                output_directory,
                transaction_id,
                phase="prepared",
                backup_name=None,
                staging_name=staging.name,
                force=False,
            )
        )
        + "\n",
        encoding="utf-8",
    )

    completed = _run_solver(manifest_path, output_directory)

    assert completed.returncode == 2
    assert "manual recovery required" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "untouched\n"
    assert nested_sentinel.read_text(encoding="utf-8") == "untouched\n"
    assert journal.is_file()
    assert not output_directory.exists()


def test_force_retains_two_markers_and_valuable_file_in_backup(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "result"
    output_directory.mkdir()
    (output_directory / "old.txt").write_text("old\n", encoding="utf-8")
    (output_directory / "photometric_calibration.yaml").write_text(
        "status: unavailable\n", encoding="utf-8"
    )
    (output_directory / "photometric_calibration_quality.json").write_text(
        '{"calibrated": false}\n', encoding="utf-8"
    )
    (output_directory / "photometric_calibration_ownership.json").write_text(
        '{"schema_version": 1}\n',
        encoding="utf-8",
    )

    warnings = SOLVER_MODULE.publish_documents(
        output_directory,
        {"status": "available"},
        {"calibrated": True},
        force=True,
    )

    assert (output_directory / "photometric_calibration.yaml").is_file()
    backups = list(tmp_path.glob(".result.backup-*"))
    assert len(backups) == 1
    backup = backups[0]
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old\n"
    assert (backup / "photometric_calibration.yaml").is_file()
    assert (backup / "photometric_calibration_quality.json").is_file()
    assert (backup / "photometric_calibration_ownership.json").is_file()
    assert any(str(backup) in warning for warning in warnings)
    assert not (tmp_path / ".result.publish-journal.json").exists()


def test_post_commit_journal_failure_returns_committed_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "result"
    output_directory.mkdir()
    old_sentinel = output_directory / "old.txt"
    old_sentinel.write_text("old\n", encoding="utf-8")
    journal_path = tmp_path / ".result.publish-journal.json"
    original_unlink = Path.unlink

    def fail_active_journal_unlink(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path == journal_path:
            raise OSError("injected journal retirement failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_active_journal_unlink)

    outcome = SOLVER_MODULE.publish_documents(
        output_directory,
        {"status": "available"},
        {"calibrated": True},
        force=True,
    )

    assert outcome.status == "committed_with_warning"
    assert outcome.committed is True
    assert any(
        "journal retirement failure" in warning
        for warning in outcome.warnings
    )
    assert (output_directory / "photometric_calibration.yaml").is_file()
    retained_locations = [
        path
        for path in tmp_path.iterdir()
        if (
            path.name.startswith(".result.backup-")
            or path.name.startswith(".result.tmp-")
        )
    ]
    assert any(
        (path / old_sentinel.name).is_file()
        for path in retained_locations
    )


@pytest.mark.parametrize(
    "injection_kind",
    ("extra", "symlink", "hardlink"),
)
def test_injected_staging_entry_never_reaches_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injection_kind: str,
) -> None:
    output_directory = tmp_path / "result"
    output_directory.mkdir()
    old_sentinel = output_directory / "old.txt"
    old_sentinel.write_text("old\n", encoding="utf-8")
    external = tmp_path / "external.txt"
    external.write_text("external\n", encoding="utf-8")
    journal_path = tmp_path / ".result.publish-journal.json"
    original_replace = SOLVER_MODULE.os.replace
    injected = False

    def inject_after_journal_write(
        source: Path,
        destination: Path,
    ) -> None:
        nonlocal injected
        original_replace(source, destination)
        if destination == journal_path and not injected:
            injected = True
            staging = next(tmp_path.glob(".result.tmp-*"))
            if injection_kind == "extra":
                (staging / "unexpected.txt").write_text(
                    "unexpected\n",
                    encoding="utf-8",
                )
            else:
                config_path = (
                    staging / "photometric_calibration.yaml"
                )
                config_path.unlink()
                if injection_kind == "symlink":
                    config_path.symlink_to(external)
                else:
                    config_path.hardlink_to(external)

    monkeypatch.setattr(
        SOLVER_MODULE.os,
        "replace",
        inject_after_journal_write,
    )

    with pytest.raises(OSError, match="staging"):
        SOLVER_MODULE.publish_documents(
            output_directory,
            {"status": "available"},
            {"calibrated": True},
            force=True,
        )

    assert old_sentinel.read_text(encoding="utf-8") == "old\n"
    assert external.read_text(encoding="utf-8") == "external\n"
