"""Photometric calibration contract validator tests."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    PACKAGE_ROOT / "scripts" / "validate_photometric_calibration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_photometric_calibration", VALIDATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)
PINNED_PUBLIC_KEYS: dict[Path, bytes] = {}
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


def _trusted_public_key(tmp_path: Path) -> bytes:
    return PINNED_PUBLIC_KEYS[tmp_path.resolve()]


def unavailable_document() -> dict:
    return {
        "schema_version": 1,
        "calibration_set_id": "unavailable-test",
        "status": "unavailable",
        "frame_metadata": {
            "exposure_sec": {"provenance": "unavailable", "field": ""},
            "sensor_gain": {"provenance": "unavailable", "field": ""},
        },
        "cameras": [
            {
                "camera_id": 0,
                "status": "unavailable",
            }
        ],
    }


def calibrated_document(
    tmp_path: Path,
    *,
    trusted_metadata: bool = False,
) -> dict:
    private_key = Ed25519PrivateKey.generate()
    public_key_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    raw_public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    public_key_sha256 = hashlib.sha256(raw_public_key).hexdigest()
    trust_root = {
        "trust_domain": TEST_TRUST_DOMAIN,
        "public_key_sha256": public_key_sha256,
    }
    (tmp_path / "response.txt").write_text(
        " ".join(str(index / 255.0) for index in range(256)) + "\n"
    )
    (tmp_path / "vignetting.txt").write_text("1.0 0.9 0.8 1.0\n")
    frame_metadata = (
        {
            "timestamp_semantics": "exposure_midpoint",
            "exposure_sec": {
                "provenance": "recorded_hardware",
                "field": "metadata.exposure_sec",
            },
            "sensor_gain": {
                "provenance": "recorded_driver",
                "field": "metadata.sensor_gain",
            },
        }
        if trusted_metadata
        else {
            "timestamp_semantics": "exposure_midpoint",
            "exposure_sec": {"provenance": "unavailable", "field": ""},
            "sensor_gain": {"provenance": "unavailable", "field": ""},
        }
    )
    manifest_raw = b'{"fixture":true}\n'
    source_bundle = {
        "schema_version": 1,
        "bundle_type": "photometric_calibration_source_snapshot",
        "solver": {
            "name": "livo_core_four_camera_photometric_calibration",
            "version": "2.0.0",
        },
        "snapshot_semantics": (
            "hashes_and_metadata_of_bytes_consumed_by_this_solver_run"
        ),
        "manifest_snapshot": {
            "encoding": "utf-8",
            "raw_utf8": manifest_raw.decode("utf-8"),
            "raw_size_bytes": len(manifest_raw),
            "raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "canonical_document_sha256": hashlib.sha256(
                _canonical_json_bytes({"fixture": True})
            ).hexdigest(),
        },
        "frame_metadata": frame_metadata,
        "cameras": [{"camera_id": 0}],
        "captures": [
            {
                "split": "train",
                "frames": [
                    {
                        "camera_id": 0,
                        "raw_image_sha256": "2" * 64,
                        "raw_image_size_bytes": 10,
                        "decoded_image_sha256": "3" * 64,
                        "decoded_image_mode": "L",
                    }
                ],
            },
            {
                "split": "held_out",
                "frames": [
                    {
                        "camera_id": 0,
                        "raw_image_sha256": "4" * 64,
                        "raw_image_size_bytes": 11,
                        "decoded_image_sha256": "5" * 64,
                        "decoded_image_mode": "L",
                    }
                ],
            },
        ],
        "source_counts": {
            "camera_count": 1,
            "capture_group_count": 2,
            "capture_frame_count": 2,
            "train_frame_count": 1,
            "held_out_frame_count": 1,
            "cumulative_encoded_bytes": 21,
        },
        "trust_boundary": (
            "snapshot_content_integrity_not_external_lab_identity"
        ),
    }
    source_bundle_bytes = _canonical_json_bytes(source_bundle)
    (tmp_path / "source_bundle.json").write_bytes(source_bundle_bytes)
    source_bundle_hash = hashlib.sha256(source_bundle_bytes).hexdigest()
    camera = {
        "camera_id": 0,
        "status": "calibrated",
        "provenance": "laboratory_measured",
        "source_id": f"source-bundle-sha256:{source_bundle_hash}",
        "source_bundle_sha256": source_bundle_hash,
        "image_width": 2,
        "image_height": 2,
        "black_level": 0.0,
        "white_level": 255.0,
        "inverse_response_lut_file": "response.txt",
        "vignetting_transmission_file": "vignetting.txt",
        "radiance_scale": 1.0,
        "radiance_offset": 0.0,
        "covariance_scale": 1.0,
    }
    document = {
        "schema_version": 1,
        "calibration_set_id": "calibrated-test",
        "dataset_scope": [],
        "status": "available",
        "source_bundle_file": "source_bundle.json",
        "source_bundle_sha256": source_bundle_hash,
        "calibration_receipt_file": "receipt.json",
        "runtime_payload_file": RUNTIME_PAYLOAD_FILENAME,
        "runtime_signature_file": RUNTIME_SIGNATURE_FILENAME,
        "trust_root": trust_root,
        "frame_metadata": frame_metadata,
        "cameras": [camera],
    }
    receipt_payload = {
        "schema_version": 1,
        "receipt_type": "four_camera_photometric_calibration",
        "solver": {
            "name": "livo_core_four_camera_photometric_calibration",
            "version": "2.0.0",
        },
        "calibration_set_id": "calibrated-test",
        "declared_source_id": "lab-run-1",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_bundle_file": "source_bundle.json",
        "source_bundle_sha256": source_bundle_hash,
        "source_counts": source_bundle["source_counts"],
        "frame_metadata": frame_metadata,
        "cameras": [
            {
                "camera_id": 0,
                "image_width": 2,
                "image_height": 2,
                "black_level": 0.0,
                "white_level": 255.0,
                "inverse_response_lut_file": "response.txt",
                "vignetting_transmission_file": "vignetting.txt",
                "radiance_scale": 1.0,
                "radiance_offset": 0.0,
                "covariance_scale": 1.0,
            }
        ],
        "calibration_config_payload": document,
        "calibration_config_sha256": hashlib.sha256(
            _canonical_json_bytes(document)
        ).hexdigest(),
        "held_out_metrics": {
            "0": {
                "response_held_out_log_rmse": 0.01,
                "flatfield_held_out_rms": 0.01,
                "cross_camera_held_out_relative_rms": 0.0,
            }
        },
        "thresholds": {
            "response_held_out_log_rmse_max": 0.14,
            "flatfield_held_out_rms_max": 0.10,
            "cross_camera_held_out_relative_rms_max": 0.08,
        },
        "artifacts_sha256": {
            "response.txt": hashlib.sha256(
                (tmp_path / "response.txt").read_bytes()
            ).hexdigest(),
            "vignetting.txt": hashlib.sha256(
                (tmp_path / "vignetting.txt").read_bytes()
            ).hexdigest(),
        },
        "integrity_scope": (
            "source_snapshot_config_solver_metrics_numeric_artifacts"
        ),
        "trust_boundary": (
            "ed25519_signature_requires_explicit_external_public_key_pin"
        ),
        "trust_root": trust_root,
        "receipt_hash_scheme": RECEIPT_HASH_SCHEME,
    }
    signed_payload_bytes = _canonical_json_bytes(receipt_payload)
    signature_bytes = private_key.sign(signed_payload_bytes)
    signature_base64 = base64.b64encode(signature_bytes).decode("ascii")
    receipt_hash = _receipt_hash_v1(
        signed_payload_bytes,
        signature_bytes,
    )
    receipt_payload["signature"] = {
        "algorithm": "ed25519",
        "public_key_sha256": public_key_sha256,
        "signature_base64": signature_base64,
    }
    (tmp_path / RUNTIME_PAYLOAD_FILENAME).write_bytes(
        signed_payload_bytes
    )
    (tmp_path / RUNTIME_SIGNATURE_FILENAME).write_bytes(
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "signature_type": (
                    "four_camera_photometric_calibration_runtime_signature"
                ),
                "algorithm": "ed25519",
                "solver": {
                    "name": (
                        "livo_core_four_camera_photometric_calibration"
                    ),
                    "version": "2.0.0",
                },
                "trust_domain": TEST_TRUST_DOMAIN,
                "payload_file": RUNTIME_PAYLOAD_FILENAME,
                "payload_sha256": hashlib.sha256(
                    signed_payload_bytes
                ).hexdigest(),
                "public_key_sha256": public_key_sha256,
                "receipt_hash_scheme": RECEIPT_HASH_SCHEME,
                "receipt_sha256": receipt_hash,
                "signature_base64": signature_base64,
            }
        )
    )
    PINNED_PUBLIC_KEYS[tmp_path.resolve()] = public_key_bytes
    receipt = {**receipt_payload, "receipt_sha256": receipt_hash}
    (tmp_path / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def _rewrite_receipt(
    tmp_path: Path,
    mutate: object,
) -> dict:
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    mutate(receipt)
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def _rewrite_source_bundle(
    tmp_path: Path,
    document: dict,
    mutate: object,
) -> None:
    bundle_path = tmp_path / "source_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    mutate(bundle)
    bundle_bytes = _canonical_json_bytes(bundle)
    bundle_path.write_bytes(bundle_bytes)
    bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
    document["source_bundle_sha256"] = bundle_hash
    for camera in document["cameras"]:
        camera["source_bundle_sha256"] = bundle_hash
        camera["source_id"] = f"source-bundle-sha256:{bundle_hash}"

    def update_receipt(receipt: dict) -> None:
        receipt["source_bundle_sha256"] = bundle_hash
        receipt["calibration_config_payload"] = document
        receipt["calibration_config_sha256"] = hashlib.sha256(
            _canonical_json_bytes(document)
        ).hexdigest()

    _rewrite_receipt(tmp_path, update_receipt)


def test_unavailable_contract_is_valid_but_blocks_both_modes(
    tmp_path: Path,
) -> None:
    report = VALIDATOR.validate_document(
        unavailable_document(), tmp_path / "calibration.yaml", (0,)
    )

    assert report["schema_valid"] is True
    assert report["mode_readiness"]["calibrated_direct"]["ready"] is False
    assert report["mode_readiness"]["robust_affine"]["ready"] is False
    assert (
        report["mode_readiness"]["robust_affine"][
            "cross_camera_comparable"
        ]
        is False
    )


def test_robust_affine_needs_calibration_but_not_exposure_metadata(
    tmp_path: Path,
) -> None:
    report = VALIDATOR.validate_document(
        calibrated_document(tmp_path),
        tmp_path / "calibration.yaml",
        (0,),
        _trusted_public_key(tmp_path),
    )

    assert report["schema_valid"] is True
    assert report["mode_readiness"]["robust_affine"]["ready"] is True
    assert report["mode_readiness"]["calibrated_direct"]["ready"] is False


def test_calibrated_direct_requires_trusted_recorded_fields(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)

    report = VALIDATOR.validate_document(
        document,
        tmp_path / "calibration.yaml",
        (0,),
        _trusted_public_key(tmp_path),
    )

    assert report["mode_readiness"]["calibrated_direct"]["ready"] is True
    assert (
        report["mode_readiness"]["calibrated_direct"][
            "cross_camera_comparable"
        ]
        is True
    )


def test_signed_receipt_hash_shape_and_value_are_fail_closed(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)
    receipt_path = tmp_path / "receipt.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    detached_path = tmp_path / RUNTIME_SIGNATURE_FILENAME
    original_detached = json.loads(
        detached_path.read_text(encoding="utf-8")
    )
    valid_hash = original["receipt_sha256"]
    invalid_hashes = (
        "0" * 64,
        "g" * 64,
        "0" * 63,
        "0" * 65,
        valid_hash[:-1] + ("0" if valid_hash[-1] != "0" else "1"),
    )

    for synchronize_detached in (False, True):
        for invalid_hash in invalid_hashes:
            tampered = dict(original)
            tampered["receipt_sha256"] = invalid_hash
            receipt_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tampered_detached = dict(original_detached)
            if synchronize_detached:
                tampered_detached["receipt_sha256"] = invalid_hash
            detached_path.write_text(
                json.dumps(
                    tampered_detached,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            report = VALIDATOR.validate_document(
                document,
                tmp_path / "calibration.yaml",
                (0,),
                _trusted_public_key(tmp_path),
            )

            blockers = report["mode_readiness"][
                "calibrated_direct"
            ]["blockers"]
            assert any(
                "structural self-hash does not match" in blocker
                for blocker in blockers
            )
            assert report["mode_readiness"][
                "calibrated_direct"
            ]["ready"] is False


def test_nonmonotonic_response_lut_blocks_all_modes(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    response = [index / 255.0 for index in range(256)]
    response[128], response[129] = response[129], response[128]
    (tmp_path / "response.txt").write_text(
        " ".join(str(value) for value in response) + "\n",
        encoding="utf-8",
    )

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    assert report["mode_readiness"]["robust_affine"]["ready"] is False
    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any("not monotonic" in blocker for blocker in blockers)


def test_calibrated_camera_without_receipt_blocks_all_modes(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    document.pop("calibration_receipt_file")
    (tmp_path / "receipt.json").unlink()

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    assert report["mode_readiness"]["robust_affine"]["ready"] is False
    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any("calibration receipt" in blocker for blocker in blockers)


def test_camera_solution_values_must_match_receipt(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    document["cameras"][0]["radiance_scale"] = 1.5

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    assert report["mode_readiness"]["robust_affine"]["ready"] is False
    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any(
        "radiance_scale does not match receipt" in blocker
        for blocker in blockers
    )


def test_dataset_scope_tamper_breaks_receipt_and_schema(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    document["dataset_scope"] = ["outdoor9"]

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    assert report["schema_valid"] is False
    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any("dataset_scope must be empty" in error for error in (
        report["schema_errors"]
    ))
    assert any(
        "config payload does not match" in blocker
        for blocker in blockers
    )


def test_validator_hashes_and_parses_each_artifact_from_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)
    lut_path = (tmp_path / "response.txt").resolve()
    tracked_paths = {
        (tmp_path / filename).resolve()
        for filename in (
            "receipt.json",
            "source_bundle.json",
            "response.txt",
            "vignetting.txt",
        )
    }
    read_counts = {path: 0 for path in tracked_paths}
    original_read_artifact_bytes = VALIDATOR._read_artifact_bytes

    def read_artifact_once(
        root: Path,
        file_name: object,
        label: str,
        maximum_bytes: int,
    ) -> tuple[bytes | None, str | None]:
        raw_bytes, error = original_read_artifact_bytes(
            root,
            file_name,
            label,
            maximum_bytes,
        )
        resolved = (root / str(file_name)).resolve()
        if resolved in read_counts:
            read_counts[resolved] += 1
        if resolved == lut_path and read_counts[resolved] == 1:
            lut_path.write_text("0.0 0.8 0.7\n", encoding="utf-8")
        return raw_bytes, error

    monkeypatch.setattr(
        VALIDATOR,
        "_read_artifact_bytes",
        read_artifact_once,
    )

    report = VALIDATOR.validate_document(
        document,
        tmp_path / "calibration.yaml",
        (0,),
        _trusted_public_key(tmp_path),
    )

    assert report["mode_readiness"]["calibrated_direct"]["ready"] is True
    assert set(read_counts.values()) == {1}
    assert lut_path.read_text(encoding="utf-8") == "0.0 0.8 0.7\n"


def test_validator_rejects_artifact_path_replacement_during_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)
    lut_path = (tmp_path / "response.txt").resolve()
    original_read = VALIDATOR.os.read
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
            ).resolve() == lut_path
        ):
            replaced = True
            replacement = lut_path.with_name(".response.replacement")
            replacement.write_text(
                " ".join("0" for _ in range(256)) + "\n",
                encoding="utf-8",
            )
            VALIDATOR.os.replace(replacement, lut_path)
        return raw_bytes

    monkeypatch.setattr(
        VALIDATOR.os,
        "read",
        replace_path_after_first_read,
    )

    report = VALIDATOR.validate_document(
        document,
        tmp_path / "calibration.yaml",
        (0,),
        _trusted_public_key(tmp_path),
    )

    assert replaced is True
    blockers = report["mode_readiness"]["calibrated_direct"]["blockers"]
    assert any("changed during read" in blocker for blocker in blockers)


def test_self_hashed_receipt_without_explicit_public_key_pin_is_not_ready(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt.pop("signature")
    receipt["trust_boundary"] = (
        "unsigned_content_integrity_only_not_trusted"
    )
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    assert report["schema_valid"] is True
    assert report["mode_readiness"]["calibrated_direct"]["ready"] is False
    blockers = report["mode_readiness"]["calibrated_direct"]["blockers"]
    assert any(
        "pinned Ed25519 public key" in blocker
        for blocker in blockers
    )


def test_recomputed_self_hash_cannot_hide_signed_payload_tampering(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt["held_out_metrics"]["0"][
        "response_held_out_log_rmse"
    ] = 0.02
    runtime_payload_bytes = (
        tmp_path / RUNTIME_PAYLOAD_FILENAME
    ).read_bytes()
    signature_bytes = base64.b64decode(
        receipt["signature"]["signature_base64"],
        validate=True,
    )
    receipt["receipt_sha256"] = _receipt_hash_v1(
        runtime_payload_bytes,
        signature_bytes,
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = VALIDATOR.validate_document(
        document,
        tmp_path / "calibration.yaml",
        (0,),
        _trusted_public_key(tmp_path),
    )

    blockers = report["mode_readiness"]["calibrated_direct"]["blockers"]
    assert any(
        (
            "runtime payload does not exactly match receipt signed payload"
            in blocker
        )
        for blocker in blockers
    )
    assert not any(
        "self-hash does not match" in blocker
        for blocker in blockers
    )


def test_non_allowlisted_solver_identity_is_never_trusted(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt["solver"]["version"] = "2.0.1"
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = VALIDATOR.validate_document(
        document,
        tmp_path / "calibration.yaml",
        (0,),
        _trusted_public_key(tmp_path),
    )

    blockers = report["mode_readiness"]["calibrated_direct"]["blockers"]
    assert any(
        "solver id/version is not allowlisted" in item
        for item in blockers
    )


def test_missing_crypto_backend_never_grants_trusted_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = calibrated_document(tmp_path, trusted_metadata=True)
    monkeypatch.setattr(VALIDATOR, "Ed25519PublicKey", None)

    report = VALIDATOR.validate_document(
        document,
        tmp_path / "calibration.yaml",
        (0,),
        _trusted_public_key(tmp_path),
    )

    blockers = report["mode_readiness"]["calibrated_direct"]["blockers"]
    assert any(
        "verification library is unavailable" in blocker
        for blocker in blockers
    )


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    (
        (
            lambda receipt: receipt["thresholds"].update(
                {"response_held_out_log_rmse_max": 1.0}
            ),
            "thresholds are incomplete or unexpected",
        ),
        (
            lambda receipt: receipt["held_out_metrics"]["0"].pop(
                "flatfield_held_out_rms"
            ),
            "held-out metric shape is invalid",
        ),
        (
            lambda receipt: receipt["source_counts"].update(
                {"capture_frame_count": 3}
            ),
            "source_counts do not match bundle",
        ),
    ),
)
def test_receipt_shape_gates_are_exact(
    tmp_path: Path,
    mutation: object,
    expected_blocker: str,
) -> None:
    document = calibrated_document(tmp_path)
    _rewrite_receipt(tmp_path, mutation)

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any(expected_blocker in blocker for blocker in blockers)


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    (
        (
            lambda bundle: bundle.update(
                {"solver": {"name": "unknown", "version": "2.0.0"}}
            ),
            "source bundle solver id/version is not allowlisted",
        ),
        (
            lambda bundle: bundle.update(
                {"cameras": [{"camera_id": 9}]}
            ),
            "source bundle camera identities do not match config",
        ),
        (
            lambda bundle: bundle.update(
                {"cameras": [{"camera_id": []}]}
            ),
            "source bundle camera identities do not match config",
        ),
    ),
)
def test_source_bundle_identity_shape_is_exact(
    tmp_path: Path,
    mutation: object,
    expected_blocker: str,
) -> None:
    document = calibrated_document(tmp_path)
    _rewrite_source_bundle(tmp_path, document, mutation)

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any(expected_blocker in blocker for blocker in blockers)


def test_response_lut_requires_exactly_256_values(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    (tmp_path / "response.txt").write_text("0.0 0.5 1.0\n")

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any(
        "LUT must contain exactly 256 values" in blocker
        for blocker in blockers
    )


def test_nonfinite_held_out_metric_fails_structural_gate(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt["held_out_metrics"]["0"][
        "response_held_out_log_rmse"
    ] = float("nan")
    receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt)
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any(
        "held-out metric must be finite" in blocker
        for blocker in blockers
    )


def test_unreferenced_old_numeric_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    old_artifact = tmp_path / "camera_9_inverse_response_lut.txt"
    old_artifact.write_text("0.0 1.0\n", encoding="utf-8")

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any(
        "unreferenced numeric artifact" in blocker
        for blocker in blockers
    )


def test_oversized_numeric_artifact_is_rejected_before_read(
    tmp_path: Path,
) -> None:
    document = calibrated_document(tmp_path)
    with (tmp_path / "response.txt").open("wb") as stream:
        stream.truncate(VALIDATOR.MAX_NUMERIC_ARTIFACT_BYTES + 1)

    report = VALIDATOR.validate_document(
        document, tmp_path / "calibration.yaml", (0,)
    )

    blockers = report["mode_readiness"]["robust_affine"]["blockers"]
    assert any(
        "exceeds resource limit" in blocker
        for blocker in blockers
    )
