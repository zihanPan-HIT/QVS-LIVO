#!/usr/bin/env python3
"""Validate the fail-closed multi-camera photometric calibration contract."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable

import yaml

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )
except ImportError:
    InvalidSignature = None
    serialization = None
    Ed25519PublicKey = None


TRUSTED_CALIBRATION_PROVENANCE = {
    "laboratory_measured",
    "manufacturer_provided",
}
TRUSTED_METADATA_PROVENANCE = {
    "recorded_hardware",
    "recorded_driver",
    "offline_synchronized",
}
SUPPORTED_MODES = ("calibrated_direct", "robust_affine")
SIGNED_RECEIPT_TRUST_BOUNDARY = (
    "ed25519_signature_requires_explicit_external_public_key_pin"
)
UNSIGNED_RECEIPT_TRUST_BOUNDARY = (
    "unsigned_content_integrity_only_not_trusted"
)
SOURCE_BUNDLE_TRUST_BOUNDARY = (
    "snapshot_content_integrity_not_external_lab_identity"
)
RECEIPT_INTEGRITY_SCOPE = (
    "source_snapshot_config_solver_metrics_numeric_artifacts"
)
TRUSTED_SOLVER = {
    "name": "livo_core_four_camera_photometric_calibration",
    "version": "2.0.0",
}
RUNTIME_PAYLOAD_FILENAME = (
    "photometric_calibration_runtime_payload.json"
)
RUNTIME_SIGNATURE_FILENAME = (
    "photometric_calibration_runtime_signature.json"
)
RUNTIME_SIGNATURE_TYPE = (
    "four_camera_photometric_calibration_runtime_signature"
)
RECEIPT_HASH_SCHEME = (
    "sha256_runtime_payload_lf_ed25519_signature_v1"
)
RECEIPT_HASH_DOMAIN_SEPARATOR = (
    b"livo_core.photometric_calibration.receipt_hash.v1\x00"
)
MAX_PUBLIC_KEY_BYTES = 64 * 1024
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_RUNTIME_SIGNATURE_BYTES = 64 * 1024
MAX_SOURCE_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_NUMERIC_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_TOTAL_NUMERIC_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_CAPTURE_FRAMES = 512
MAX_TOTAL_ENCODED_CAPTURE_BYTES = 512 * 1024 * 1024
EXPECTED_THRESHOLDS = {
    "response_held_out_log_rmse_max": 0.14,
    "flatfield_held_out_rms_max": 0.10,
    "cross_camera_held_out_relative_rms_max": 0.08,
}
HELD_OUT_METRIC_GATES = {
    "response_held_out_log_rmse": (
        "response_held_out_log_rmse_max"
    ),
    "flatfield_held_out_rms": "flatfield_held_out_rms_max",
    "cross_camera_held_out_relative_rms": (
        "cross_camera_held_out_relative_rms_max"
    ),
}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _parse_numeric_bytes(raw_bytes: bytes) -> list[float]:
    text = raw_bytes.decode("utf-8")
    tokens = [
        token
        for token in re.split(r"[\s,;]+", text.strip())
        if token
    ]
    values = [float(token) for token in tokens]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("contains a non-finite value")
    return values


def _canonical_json_bytes(document: dict[str, Any]) -> bytes:
    rendered = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (rendered + "\n").encode("utf-8")


def _receipt_sha256_v1(
    payload_bytes: bytes,
    signature_bytes: bytes,
) -> str | None:
    if (
        not payload_bytes.endswith(b"\n")
        or payload_bytes.endswith(b"\n\n")
        or len(signature_bytes) != 64
    ):
        return None
    preimage = (
        RECEIPT_HASH_DOMAIN_SEPARATOR
        + len(payload_bytes).to_bytes(8, "big")
        + payload_bytes
        + len(signature_bytes).to_bytes(8, "big")
        + signature_bytes
    )
    return hashlib.sha256(preimage).hexdigest()


def _load_unique_json_mapping(
    raw_bytes: bytes,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        loaded = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        return None, f"{label} is invalid: {error}"
    if not isinstance(loaded, dict):
        return None, f"{label} root must be a mapping"
    return loaded, None


def _safe_artifact_path(
    config_directory: Path,
    raw_filename: Any,
    label: str,
) -> tuple[Path | None, str | None]:
    if (
        not isinstance(raw_filename, str)
        or not raw_filename
        or Path(raw_filename).name != raw_filename
    ):
        return None, f"{label} must be a basename"
    path = config_directory / raw_filename
    if path.is_symlink():
        return None, f"{label} must not be a symlink"
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(config_directory.resolve(strict=True))
    except (OSError, ValueError) as error:
        return None, f"{label} is not a root-bound regular file: {error}"
    if not resolved.is_file():
        return None, f"{label} is not a regular file"
    return resolved, None


def _read_artifact_bytes(
    config_directory: Path,
    raw_filename: Any,
    label: str,
    maximum_bytes: int,
) -> tuple[bytes | None, str | None]:
    path, path_error = _safe_artifact_path(
        config_directory,
        raw_filename,
        label,
    )
    if path_error is not None:
        return None, path_error
    assert path is not None
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        return None, f"{label} cannot be read: {error}"
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            return None, f"{label} is not a regular file"
        if before.st_size > maximum_bytes:
            return None, f"{label} exceeds resource limit {maximum_bytes}"
        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            remaining = maximum_bytes + 1 - total_bytes
            if remaining <= 0:
                return None, (
                    f"{label} exceeds resource limit {maximum_bytes}"
                )
            chunk = os.read(
                descriptor,
                min(1024 * 1024, remaining),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total_bytes += len(chunk)
        raw_bytes = b"".join(chunks)
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError:
            return None, f"{label} path changed during read"
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (
            path_after.st_dev,
            path_after.st_ino,
        ) != (
            after.st_dev,
            after.st_ino,
        ) or stat.S_ISLNK(path_after.st_mode) or len(
            raw_bytes
        ) != after.st_size:
            return None, f"{label} changed during read"
        return raw_bytes, None
    except OSError as error:
        return None, f"{label} cannot be read: {error}"
    finally:
        os.close(descriptor)


def _validate_receipt_signature(
    receipt_payload: dict[str, Any],
    embedded_receipt_hash: Any,
    document: dict[str, Any],
    config_directory: Path,
    trusted_public_key_bytes: bytes | None,
) -> tuple[list[str], str | None]:
    blockers: list[str] = []
    calculated_receipt_hash: str | None = None
    if receipt_payload.get("solver") != TRUSTED_SOLVER:
        blockers.append(
            "calibration receipt solver id/version is not allowlisted"
        )
    signature = receipt_payload.get("signature")
    if not isinstance(signature, dict):
        blockers.append("calibration receipt Ed25519 signature is missing")
        if trusted_public_key_bytes is None:
            blockers.append(
                "trusted readiness requires an explicit pinned Ed25519 "
                "public key"
            )
        return blockers, calculated_receipt_hash
    if signature.get("algorithm") != "ed25519":
        blockers.append("calibration receipt signature algorithm is invalid")
    if document.get("runtime_payload_file") != RUNTIME_PAYLOAD_FILENAME:
        blockers.append(
            "runtime payload file must use the fixed trusted basename"
        )
    if (
        document.get("runtime_signature_file")
        != RUNTIME_SIGNATURE_FILENAME
    ):
        blockers.append(
            "runtime signature file must use the fixed trusted basename"
        )
    runtime_payload_bytes, payload_error = _read_artifact_bytes(
        config_directory,
        RUNTIME_PAYLOAD_FILENAME,
        "runtime signed payload",
        MAX_RECEIPT_BYTES,
    )
    runtime_signature_bytes, detached_error = _read_artifact_bytes(
        config_directory,
        RUNTIME_SIGNATURE_FILENAME,
        "runtime detached signature",
        MAX_RUNTIME_SIGNATURE_BYTES,
    )
    if payload_error is not None:
        blockers.append(payload_error)
    if detached_error is not None:
        blockers.append(detached_error)
    if (
        runtime_payload_bytes is None
        or runtime_signature_bytes is None
    ):
        return blockers, calculated_receipt_hash
    runtime_payload, payload_parse_error = _load_unique_json_mapping(
        runtime_payload_bytes,
        "runtime signed payload",
    )
    detached, detached_parse_error = _load_unique_json_mapping(
        runtime_signature_bytes,
        "runtime detached signature",
    )
    if payload_parse_error is not None:
        blockers.append(payload_parse_error)
    if detached_parse_error is not None:
        blockers.append(detached_parse_error)
    if runtime_payload is None or detached is None:
        return blockers, calculated_receipt_hash
    if runtime_payload_bytes != _canonical_json_bytes(runtime_payload):
        blockers.append("runtime payload bytes are not canonical")
    signed_payload = dict(receipt_payload)
    signed_payload.pop("signature", None)
    if runtime_payload != signed_payload:
        blockers.append(
            "runtime payload does not exactly match receipt signed payload"
        )
    trust_root = receipt_payload.get("trust_root")
    if not isinstance(trust_root, dict) or set(trust_root) != {
        "trust_domain",
        "public_key_sha256",
    }:
        blockers.append("signed trust root is missing or malformed")
        trust_root = {}
    if document.get("trust_root") != trust_root:
        blockers.append("config trust root does not match signed payload")
    receipt_hash_scheme = receipt_payload.get("receipt_hash_scheme")
    if receipt_hash_scheme != RECEIPT_HASH_SCHEME:
        blockers.append("signed receipt hash scheme is unknown")
    raw_signature = signature.get("signature_base64")
    try:
        signature_bytes = base64.b64decode(
            raw_signature,
            validate=True,
        )
    except (TypeError, ValueError) as error:
        blockers.append(
            f"calibration receipt signature is invalid: {error}"
        )
        signature_bytes = b""
    if len(signature_bytes) != 64:
        blockers.append(
            "calibration receipt Ed25519 signature length is invalid"
        )
    if receipt_hash_scheme == RECEIPT_HASH_SCHEME:
        calculated_receipt_hash = _receipt_sha256_v1(
            runtime_payload_bytes,
            signature_bytes,
        )
        if calculated_receipt_hash is None:
            blockers.append(
                "receipt hash inputs do not satisfy the v1 framing"
            )
        elif (
            not isinstance(embedded_receipt_hash, str)
            or re.fullmatch(
                r"[0-9a-f]{64}", embedded_receipt_hash
            )
            is None
            or not hmac.compare_digest(
                embedded_receipt_hash,
                calculated_receipt_hash,
            )
        ):
            blockers.append(
                "calibration receipt structural self-hash does not match"
            )
    expected_detached = {
        "schema_version": 1,
        "signature_type": RUNTIME_SIGNATURE_TYPE,
        "algorithm": "ed25519",
        "solver": TRUSTED_SOLVER,
        "trust_domain": trust_root.get("trust_domain"),
        "payload_file": RUNTIME_PAYLOAD_FILENAME,
        "payload_sha256": hashlib.sha256(
            runtime_payload_bytes
        ).hexdigest(),
        "public_key_sha256": signature.get("public_key_sha256"),
        "receipt_hash_scheme": receipt_hash_scheme,
        "receipt_sha256": embedded_receipt_hash,
        "signature_base64": signature.get("signature_base64"),
    }
    if detached != expected_detached:
        blockers.append(
            "runtime detached signature metadata does not exactly match"
        )
    if trusted_public_key_bytes is None:
        blockers.append(
            "trusted readiness requires an explicit pinned Ed25519 "
            "public key"
        )
        return blockers, calculated_receipt_hash
    if Ed25519PublicKey is None or serialization is None:
        blockers.append(
            "Ed25519 verification library is unavailable"
        )
        return blockers, calculated_receipt_hash
    try:
        loaded_key = serialization.load_pem_public_key(
            trusted_public_key_bytes
        )
    except (TypeError, ValueError) as error:
        blockers.append(f"pinned Ed25519 public key is invalid: {error}")
        return blockers, calculated_receipt_hash
    if not isinstance(loaded_key, Ed25519PublicKey):
        blockers.append("pinned public key is not Ed25519")
        return blockers, calculated_receipt_hash
    raw_public_key = loaded_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    fingerprint = hashlib.sha256(raw_public_key).hexdigest()
    if signature.get("public_key_sha256") != fingerprint:
        blockers.append(
            "calibration receipt key fingerprint does not match pin"
        )
        return blockers, calculated_receipt_hash
    if trust_root.get("public_key_sha256") != fingerprint:
        blockers.append(
            "signed trust-root fingerprint does not match pin"
        )
        return blockers, calculated_receipt_hash
    trust_domain = trust_root.get("trust_domain")
    if (
        not isinstance(trust_domain, str)
        or not trust_domain
        or len(trust_domain.encode("utf-8")) > 128
        or any(
            ord(character) < 0x21 or ord(character) > 0x7E
            for character in trust_domain
        )
    ):
        blockers.append("signed trust domain is invalid")
        return blockers, calculated_receipt_hash
    if len(signature_bytes) != 64:
        return blockers, calculated_receipt_hash
    try:
        loaded_key.verify(
            signature_bytes,
            runtime_payload_bytes,
        )
    except InvalidSignature:
        blockers.append(
            "calibration receipt Ed25519 signature does not verify"
        )
    return blockers, calculated_receipt_hash


def _validate_receipt(
    document: dict[str, Any],
    config_directory: Path,
    trusted_public_key_bytes: bytes | None,
) -> tuple[
    str | None,
    dict[str, Any] | None,
    dict[str, list[float]],
    str | None,
    list[str],
]:
    blockers: list[str] = []
    artifact_values: dict[str, list[float]] = {}
    receipt_bytes, path_error = _read_artifact_bytes(
        config_directory,
        document.get("calibration_receipt_file"),
        "calibration receipt",
        MAX_RECEIPT_BYTES,
    )
    if path_error is not None:
        blockers.append(path_error)
        return None, None, artifact_values, None, blockers
    assert receipt_bytes is not None
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        blockers.append(f"calibration receipt is invalid: {error}")
        return None, None, artifact_values, None, blockers
    if not isinstance(receipt, dict):
        blockers.append("calibration receipt root must be a mapping")
        return None, None, artifact_values, None, blockers
    embedded_hash = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if isinstance(payload.get("signature"), dict):
        signature_blockers, actual_hash = _validate_receipt_signature(
            payload,
            embedded_hash,
            document,
            config_directory,
            trusted_public_key_bytes,
        )
        blockers.extend(signature_blockers)
    else:
        actual_hash = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
        if (
            not isinstance(embedded_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", embedded_hash) is None
            or not hmac.compare_digest(embedded_hash, actual_hash)
        ):
            blockers.append(
                "calibration receipt structural self-hash does not match"
            )
        signature_blockers, _ = _validate_receipt_signature(
            payload,
            embedded_hash,
            document,
            config_directory,
            trusted_public_key_bytes,
        )
        blockers.extend(signature_blockers)
    if receipt.get("calibration_set_id") != document.get(
        "calibration_set_id"
    ):
        blockers.append("calibration receipt set id does not match config")
    if receipt.get("schema_version") != 1:
        blockers.append("calibration receipt schema_version must equal 1")
    if receipt.get("receipt_type") != (
        "four_camera_photometric_calibration"
    ):
        blockers.append("calibration receipt type is invalid")
    solver = receipt.get("solver")
    if (
        not isinstance(solver, dict)
        or not isinstance(solver.get("name"), str)
        or not solver["name"]
        or not isinstance(solver.get("version"), str)
        or not solver["version"]
    ):
        blockers.append("calibration receipt solver identity is missing")
    manifest_hash = receipt.get("manifest_sha256")
    if (
        not isinstance(manifest_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash)
    ):
        blockers.append("calibration receipt manifest hash is invalid")
    config_payload = receipt.get("calibration_config_payload")
    if config_payload != document:
        blockers.append(
            "calibration receipt config payload does not match config"
        )
    config_hash = hashlib.sha256(
        _canonical_json_bytes(document)
    ).hexdigest()
    if receipt.get("calibration_config_sha256") != config_hash:
        blockers.append(
            "calibration receipt config hash does not match config"
        )
    if receipt.get("frame_metadata") != document.get("frame_metadata"):
        blockers.append(
            "calibration receipt frame metadata does not match config"
        )
    held_out_metrics = receipt.get("held_out_metrics")
    if not isinstance(held_out_metrics, dict):
        blockers.append("calibration receipt held-out metrics are missing")
    thresholds = receipt.get("thresholds")
    if not isinstance(thresholds, dict):
        blockers.append("calibration receipt thresholds are missing")
    elif thresholds != EXPECTED_THRESHOLDS:
        blockers.append(
            "calibration receipt thresholds are incomplete or unexpected"
        )
    if isinstance(held_out_metrics, dict) and isinstance(
        thresholds, dict
    ):
        expected_metric_cameras = {
            str(camera.get("camera_id"))
            for camera in document.get("cameras", [])
            if (
                isinstance(camera, dict)
                and camera.get("status") == "calibrated"
            )
        }
        if set(held_out_metrics) != expected_metric_cameras:
            blockers.append(
                "calibration receipt held-out camera set is incomplete"
            )
        for camera_id, metrics in held_out_metrics.items():
            if (
                not isinstance(metrics, dict)
                or set(metrics) != set(HELD_OUT_METRIC_GATES)
            ):
                blockers.append(
                    f"camera[{camera_id}] held-out metric shape is invalid"
                )
                continue
            for metric_name, threshold_name in (
                HELD_OUT_METRIC_GATES.items()
            ):
                metric_value = metrics.get(metric_name)
                if not _finite_number(metric_value):
                    blockers.append(
                        f"camera[{camera_id}] held-out metric must be "
                        f"finite: {metric_name}"
                    )
                elif (
                    _finite_number(thresholds.get(threshold_name))
                    and float(metric_value)
                    > float(thresholds[threshold_name])
                ):
                    blockers.append(
                        f"camera[{camera_id}] held-out metric exceeds gate: "
                        f"{metric_name}"
                    )
    if receipt.get("integrity_scope") != RECEIPT_INTEGRITY_SCOPE:
        blockers.append("calibration receipt integrity scope is invalid")
    expected_trust_boundary = (
        SIGNED_RECEIPT_TRUST_BOUNDARY
        if isinstance(receipt.get("signature"), dict)
        else UNSIGNED_RECEIPT_TRUST_BOUNDARY
    )
    if receipt.get("trust_boundary") != expected_trust_boundary:
        blockers.append("calibration receipt trust boundary is invalid")

    expected_bundle_hash = document.get("source_bundle_sha256")
    if (
        not isinstance(expected_bundle_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_bundle_hash)
    ):
        blockers.append("source bundle SHA-256 is missing or invalid")
    if receipt.get("source_bundle_file") != document.get(
        "source_bundle_file"
    ):
        blockers.append(
            "calibration receipt source bundle file does not match config"
        )
    if receipt.get("source_bundle_sha256") != expected_bundle_hash:
        blockers.append(
            "calibration receipt source bundle hash does not match config"
        )
    bundle_bytes, bundle_error = _read_artifact_bytes(
        config_directory,
        document.get("source_bundle_file"),
        "source bundle",
        MAX_SOURCE_BUNDLE_BYTES,
    )
    source_bundle_hash: str | None = None
    source_bundle: dict[str, Any] | None = None
    if bundle_error is not None:
        blockers.append(bundle_error)
    else:
        assert bundle_bytes is not None
        source_bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
        if source_bundle_hash != expected_bundle_hash:
            blockers.append("source bundle hash does not match config")
        try:
            loaded_bundle = json.loads(bundle_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            blockers.append(f"source bundle is invalid: {error}")
        else:
            if isinstance(loaded_bundle, dict):
                source_bundle = loaded_bundle
            else:
                blockers.append("source bundle root must be a mapping")

    if source_bundle is not None:
        if source_bundle.get("schema_version") != 1:
            blockers.append("source bundle schema_version must equal 1")
        if source_bundle.get("solver") != TRUSTED_SOLVER:
            blockers.append(
                "source bundle solver id/version is not allowlisted"
            )
        if source_bundle.get("bundle_type") != (
            "photometric_calibration_source_snapshot"
        ):
            blockers.append("source bundle type is invalid")
        if source_bundle.get("snapshot_semantics") != (
            "hashes_and_metadata_of_bytes_consumed_by_this_solver_run"
        ):
            blockers.append("source bundle snapshot semantics are invalid")
        if source_bundle.get("trust_boundary") != (
            SOURCE_BUNDLE_TRUST_BOUNDARY
        ):
            blockers.append("source bundle trust boundary is invalid")
        if source_bundle.get("frame_metadata") != document.get(
            "frame_metadata"
        ):
            blockers.append(
                "source bundle frame metadata does not match config"
            )
        bundle_cameras = source_bundle.get("cameras")
        document_camera_ids = [
            camera.get("camera_id")
            for camera in document.get("cameras", [])
            if isinstance(camera, dict)
        ]
        bundle_camera_ids = (
            [
                camera.get("camera_id")
                for camera in bundle_cameras
                if isinstance(camera, dict)
            ]
            if isinstance(bundle_cameras, list)
            else []
        )
        valid_document_camera_ids = all(
            isinstance(camera_id, int)
            and not isinstance(camera_id, bool)
            and camera_id >= 0
            for camera_id in document_camera_ids
        )
        valid_bundle_camera_ids = all(
            isinstance(camera_id, int)
            and not isinstance(camera_id, bool)
            and camera_id >= 0
            for camera_id in bundle_camera_ids
        )
        if (
            not isinstance(bundle_cameras, list)
            or len(bundle_camera_ids) != len(bundle_cameras)
            or not valid_document_camera_ids
            or not valid_bundle_camera_ids
            or (
                valid_document_camera_ids
                and len(set(document_camera_ids))
                != len(document_camera_ids)
            )
            or (
                valid_bundle_camera_ids
                and len(set(bundle_camera_ids))
                != len(bundle_camera_ids)
            )
            or (
                valid_document_camera_ids
                and valid_bundle_camera_ids
                and set(bundle_camera_ids) != set(document_camera_ids)
            )
        ):
            blockers.append(
                "source bundle camera identities do not match config"
            )

        manifest_snapshot = source_bundle.get("manifest_snapshot")
        if not isinstance(manifest_snapshot, dict):
            blockers.append("source bundle manifest snapshot is missing")
        else:
            manifest_raw = manifest_snapshot.get("raw_utf8")
            if not isinstance(manifest_raw, str):
                blockers.append(
                    "source bundle manifest raw UTF-8 snapshot is invalid"
                )
            else:
                manifest_bytes = manifest_raw.encode("utf-8")
                if len(manifest_bytes) > MAX_MANIFEST_SNAPSHOT_BYTES:
                    blockers.append(
                        "source bundle manifest snapshot exceeds resource "
                        f"limit {MAX_MANIFEST_SNAPSHOT_BYTES}"
                    )
                manifest_actual_hash = hashlib.sha256(
                    manifest_bytes
                ).hexdigest()
                if manifest_snapshot.get(
                    "raw_size_bytes"
                ) != len(manifest_bytes):
                    blockers.append(
                        "source bundle manifest raw size does not match"
                    )
                if manifest_snapshot.get(
                    "raw_sha256"
                ) != manifest_actual_hash:
                    blockers.append(
                        "source bundle manifest raw hash does not match"
                    )
                if manifest_hash != manifest_actual_hash:
                    blockers.append(
                        "calibration receipt manifest hash does not match "
                        "source bundle"
                    )
                try:
                    manifest_document = yaml.safe_load(manifest_raw)
                except yaml.YAMLError as error:
                    blockers.append(
                        f"source bundle manifest snapshot is invalid: {error}"
                    )
                else:
                    if not isinstance(manifest_document, dict):
                        blockers.append(
                            "source bundle manifest root must be a mapping"
                        )
                    elif manifest_snapshot.get(
                        "canonical_document_sha256"
                    ) != hashlib.sha256(
                        _canonical_json_bytes(manifest_document)
                    ).hexdigest():
                        blockers.append(
                            "source bundle manifest canonical hash does "
                            "not match"
                        )

        captures = source_bundle.get("captures")
        if not isinstance(captures, list) or not captures:
            blockers.append("source bundle captures are missing")
        else:
            raw_by_split: dict[str, set[str]] = {
                "train": set(),
                "held_out": set(),
            }
            decoded_by_split: dict[str, set[str]] = {
                "train": set(),
                "held_out": set(),
            }
            frame_count = 0
            train_frame_count = 0
            held_out_frame_count = 0
            cumulative_encoded_bytes = 0
            for group in captures:
                if not isinstance(group, dict):
                    blockers.append(
                        "source bundle capture group must be a mapping"
                    )
                    continue
                split = group.get("split")
                frames = group.get("frames")
                if split not in raw_by_split or not isinstance(frames, list):
                    blockers.append(
                        "source bundle capture split or frames are invalid"
                    )
                    continue
                for frame in frames:
                    frame_count += 1
                    if split == "train":
                        train_frame_count += 1
                    else:
                        held_out_frame_count += 1
                    if not isinstance(frame, dict):
                        blockers.append(
                            "source bundle capture frame must be a mapping"
                        )
                        continue
                    raw_hash = frame.get("raw_image_sha256")
                    raw_size = frame.get("raw_image_size_bytes")
                    decoded_hash = frame.get("decoded_image_sha256")
                    if (
                        not isinstance(raw_hash, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", raw_hash)
                    ):
                        blockers.append(
                            "source bundle raw image hash is invalid"
                        )
                    else:
                        raw_by_split[split].add(raw_hash)
                    if (
                        not isinstance(raw_size, int)
                        or isinstance(raw_size, bool)
                        or raw_size < 0
                    ):
                        blockers.append(
                            "source bundle raw image size is invalid"
                        )
                    else:
                        cumulative_encoded_bytes += raw_size
                    if (
                        not isinstance(decoded_hash, str)
                        or not re.fullmatch(
                            r"[0-9a-f]{64}", decoded_hash
                        )
                    ):
                        blockers.append(
                            "source bundle decoded image hash is invalid"
                        )
                    else:
                        decoded_by_split[split].add(decoded_hash)
                    if not isinstance(
                        frame.get("decoded_image_mode"), str
                    ):
                        blockers.append(
                            "source bundle decoded image mode is invalid"
                        )
                    if "gray_card_mask_path" in frame:
                        for field_name in (
                            "raw_gray_card_mask_sha256",
                            "decoded_gray_card_mask_sha256",
                        ):
                            field_value = frame.get(field_name)
                            if (
                                not isinstance(field_value, str)
                                or not re.fullmatch(
                                    r"[0-9a-f]{64}", field_value
                                )
                            ):
                                blockers.append(
                                    f"source bundle {field_name} is invalid"
                                )
                        mask_size = frame.get(
                            "raw_gray_card_mask_size_bytes"
                        )
                        if (
                            not isinstance(mask_size, int)
                            or isinstance(mask_size, bool)
                            or mask_size < 0
                        ):
                            blockers.append(
                                "source bundle raw mask size is invalid"
                            )
                        else:
                            cumulative_encoded_bytes += mask_size
            if frame_count == 0:
                blockers.append("source bundle capture frames are missing")
            if frame_count > MAX_CAPTURE_FRAMES:
                blockers.append(
                    "source bundle frame count exceeds resource limit "
                    f"{MAX_CAPTURE_FRAMES}"
                )
            if (
                cumulative_encoded_bytes
                > MAX_TOTAL_ENCODED_CAPTURE_BYTES
            ):
                blockers.append(
                    "source bundle cumulative encoded bytes exceed "
                    f"resource limit {MAX_TOTAL_ENCODED_CAPTURE_BYTES}"
                )
            computed_source_counts = {
                "camera_count": len(
                    source_bundle.get("cameras", [])
                )
                if isinstance(source_bundle.get("cameras"), list)
                else -1,
                "capture_group_count": len(captures),
                "capture_frame_count": frame_count,
                "train_frame_count": train_frame_count,
                "held_out_frame_count": held_out_frame_count,
                "cumulative_encoded_bytes": cumulative_encoded_bytes,
            }
            if source_bundle.get(
                "source_counts"
            ) != computed_source_counts:
                blockers.append(
                    "source bundle source_counts do not match captures"
                )
            if receipt.get("source_counts") != computed_source_counts:
                blockers.append(
                    "calibration receipt source_counts do not match bundle"
                )
            if raw_by_split["train"] & raw_by_split["held_out"]:
                blockers.append(
                    "source bundle raw image hash is reused across splits"
                )
            if decoded_by_split["train"] & decoded_by_split["held_out"]:
                blockers.append(
                    "source bundle decoded image hash is reused across splits"
                )

    total_numeric_artifact_bytes = 0
    artifacts = receipt.get("artifacts_sha256")
    if not isinstance(artifacts, dict):
        blockers.append("calibration receipt artifact hashes are missing")
    else:
        referenced_artifacts = {
            filename
            for camera in document.get("cameras", [])
            if isinstance(camera, dict)
            for filename in (
                camera.get("inverse_response_lut_file"),
                camera.get("vignetting_transmission_file"),
            )
            if isinstance(filename, str)
        }
        artifact_names = set(artifacts)
        for filename in sorted(artifact_names - referenced_artifacts):
            blockers.append(
                f"unreferenced numeric artifact in receipt: {filename}"
            )
        for filename in sorted(referenced_artifacts - artifact_names):
            blockers.append(
                f"referenced numeric artifact is missing: {filename}"
            )
        for filename, artifact_hash in artifacts.items():
            artifact_bytes, artifact_error = _read_artifact_bytes(
                config_directory,
                filename,
                f"receipt artifact {filename}",
                MAX_NUMERIC_ARTIFACT_BYTES,
            )
            if artifact_error is not None:
                blockers.append(artifact_error)
                continue
            if (
                not isinstance(artifact_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", artifact_hash)
            ):
                blockers.append(
                    f"receipt artifact {filename} hash is invalid"
                )
                continue
            assert artifact_bytes is not None
            total_numeric_artifact_bytes += len(artifact_bytes)
            if (
                total_numeric_artifact_bytes
                > MAX_TOTAL_NUMERIC_ARTIFACT_BYTES
            ):
                blockers.append(
                    "cumulative numeric artifact bytes exceed resource "
                    f"limit {MAX_TOTAL_NUMERIC_ARTIFACT_BYTES}"
                )
            actual_artifact_hash = hashlib.sha256(
                artifact_bytes
            ).hexdigest()
            if actual_artifact_hash != artifact_hash:
                blockers.append(
                    f"receipt artifact {filename} hash does not match"
                )
            try:
                artifact_values[filename] = _parse_numeric_bytes(
                    artifact_bytes
                )
            except (UnicodeError, ValueError) as error:
                blockers.append(
                    f"receipt artifact {filename} is invalid: {error}"
                )
        for path in config_directory.iterdir():
            if (
                path.name.endswith("_inverse_response_lut.txt")
                or path.name.endswith("_vignetting_transmission.txt")
            ) and path.name not in referenced_artifacts:
                blockers.append(
                    f"unreferenced numeric artifact is present: {path.name}"
                )
    return (
        actual_hash,
        receipt,
        artifact_values,
        source_bundle_hash,
        blockers,
    )


def _validate_calibrated_camera(
    camera: dict[str, Any],
    receipt_hash: str | None,
    receipt: dict[str, Any] | None,
    artifact_values: dict[str, list[float]],
    source_bundle_hash: str | None,
) -> list[str]:
    blockers: list[str] = []
    camera_id = camera.get("camera_id", "?")
    prefix = f"camera[{camera_id}]"

    if camera.get("provenance") not in TRUSTED_CALIBRATION_PROVENANCE:
        blockers.append(f"{prefix}: untrusted calibration provenance")
    if receipt_hash is None:
        blockers.append(f"{prefix}: calibration receipt is unavailable")
    else:
        if source_bundle_hash is None:
            blockers.append(
                f"{prefix}: source bundle is unavailable"
            )
        if camera.get("source_bundle_sha256") != source_bundle_hash:
            blockers.append(
                f"{prefix}: source bundle hash does not match"
            )
        if camera.get("source_id") != (
            f"source-bundle-sha256:{source_bundle_hash}"
        ):
            blockers.append(
                f"{prefix}: source_id is not bound to source bundle"
            )
        receipt_cameras = (
            receipt.get("cameras") if isinstance(receipt, dict) else None
        )
        receipt_camera = (
            next(
                (
                    item
                    for item in receipt_cameras
                    if (
                        isinstance(item, dict)
                        and item.get("camera_id") == camera_id
                    )
                ),
                None,
            )
            if isinstance(receipt_cameras, list)
            else None
        )
        if receipt_camera is None:
            blockers.append(f"{prefix}: camera is missing from receipt")
        else:
            for field_name in (
                "image_width",
                "image_height",
                "black_level",
                "white_level",
                "inverse_response_lut_file",
                "vignetting_transmission_file",
                "radiance_scale",
                "radiance_offset",
                "covariance_scale",
            ):
                if receipt_camera.get(field_name) != camera.get(field_name):
                    blockers.append(
                        f"{prefix}: {field_name} does not match receipt"
                    )
        held_out_metrics = (
            receipt.get("held_out_metrics")
            if isinstance(receipt, dict)
            else None
        )
        if (
            not isinstance(held_out_metrics, dict)
            or str(camera_id) not in held_out_metrics
        ):
            blockers.append(
                f"{prefix}: held-out metrics are missing from receipt"
            )

    width = camera.get("image_width")
    height = camera.get("image_height")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        blockers.append(f"{prefix}: image_width must be a positive integer")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        blockers.append(f"{prefix}: image_height must be a positive integer")

    black = camera.get("black_level")
    white = camera.get("white_level")
    if (
        not _finite_number(black)
        or not _finite_number(white)
        or float(white) <= float(black)
    ):
        blockers.append(f"{prefix}: invalid black/white levels")

    radiance_scale = camera.get("radiance_scale")
    radiance_offset = camera.get("radiance_offset")
    covariance_scale = camera.get("covariance_scale")
    if not _finite_number(radiance_scale) or float(radiance_scale) <= 0.0:
        blockers.append(f"{prefix}: radiance_scale must be positive")
    if not _finite_number(radiance_offset):
        blockers.append(f"{prefix}: radiance_offset must be finite")
    if (
        not _finite_number(covariance_scale)
        or float(covariance_scale) < 1.0
    ):
        blockers.append(f"{prefix}: covariance_scale must be at least one")

    lut_path_value = camera.get("inverse_response_lut_file")
    if not isinstance(lut_path_value, str) or not lut_path_value:
        blockers.append(f"{prefix}: inverse response LUT is missing")
    else:
        receipt_artifacts = (
            receipt.get("artifacts_sha256")
            if isinstance(receipt, dict)
            else None
        )
        if (
            not isinstance(receipt_artifacts, dict)
            or lut_path_value not in receipt_artifacts
        ):
            blockers.append(f"{prefix}: LUT hash is missing from receipt")
        if lut_path_value not in artifact_values:
            blockers.append(f"{prefix}: LUT file was not validated")
        else:
            lut = artifact_values[lut_path_value]
            if len(lut) != 256:
                blockers.append(
                    f"{prefix}: LUT must contain exactly 256 values"
                )
            elif any(value < 0.0 for value in lut):
                blockers.append(f"{prefix}: LUT contains a negative value")
            elif any(
                right < left for left, right in zip(lut, lut[1:])
            ):
                blockers.append(f"{prefix}: LUT is not monotonic")
            elif lut[-1] <= lut[0]:
                blockers.append(f"{prefix}: LUT has no dynamic range")

    vignette_path_value = camera.get("vignetting_transmission_file")
    if not isinstance(vignette_path_value, str) or not vignette_path_value:
        blockers.append(f"{prefix}: vignetting map is missing")
    else:
        receipt_artifacts = (
            receipt.get("artifacts_sha256")
            if isinstance(receipt, dict)
            else None
        )
        if (
            not isinstance(receipt_artifacts, dict)
            or vignette_path_value not in receipt_artifacts
        ):
            blockers.append(
                f"{prefix}: vignetting hash is missing from receipt"
            )
        if vignette_path_value not in artifact_values:
            blockers.append(
                f"{prefix}: vignetting file was not validated"
            )
        else:
            vignetting = artifact_values[vignette_path_value]
            if isinstance(width, int) and isinstance(height, int):
                expected_size = width * height
                if len(vignetting) != expected_size:
                    blockers.append(
                        f"{prefix}: vignetting size {len(vignetting)} "
                        f"does not match {expected_size}"
                    )
            if any(value <= 0.0 for value in vignetting):
                blockers.append(
                    f"{prefix}: vignetting transmission must be positive"
                )

    return blockers


def validate_document(
    document: Any,
    config_path: Path,
    camera_ids: Iterable[int] = (0, 1, 2, 3),
    trusted_public_key_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Return schema errors and readiness for both supported modes."""

    schema_errors: list[str] = []
    if not isinstance(document, dict):
        return {
            "schema_valid": False,
            "schema_errors": ["document root must be a mapping"],
            "camera_status": {},
            "mode_readiness": {
                mode: {"ready": False, "blockers": ["invalid schema"]}
                for mode in SUPPORTED_MODES
            },
        }

    if document.get("schema_version") != 1:
        schema_errors.append("schema_version must equal 1")
    if not isinstance(document.get("calibration_set_id"), str):
        schema_errors.append("calibration_set_id must be a string")
    if document.get("status") not in {"available", "unavailable"}:
        schema_errors.append("status must be available or unavailable")

    raw_cameras = document.get("cameras")
    if not isinstance(raw_cameras, list):
        schema_errors.append("cameras must be a list")
        raw_cameras = []

    cameras: dict[int, dict[str, Any]] = {}
    for index, raw_camera in enumerate(raw_cameras):
        if not isinstance(raw_camera, dict):
            schema_errors.append(f"cameras[{index}] must be a mapping")
            continue
        camera_id = raw_camera.get("camera_id")
        if (
            not isinstance(camera_id, int)
            or isinstance(camera_id, bool)
            or camera_id < 0
        ):
            schema_errors.append(
                f"cameras[{index}].camera_id must be a nonnegative integer"
            )
            continue
        if camera_id in cameras:
            schema_errors.append(f"duplicate camera_id: {camera_id}")
            continue
        if raw_camera.get("status") not in {"calibrated", "unavailable"}:
            schema_errors.append(
                f"camera[{camera_id}].status must be calibrated or unavailable"
            )
        cameras[camera_id] = raw_camera

    metadata = document.get("frame_metadata")
    if not isinstance(metadata, dict):
        schema_errors.append("frame_metadata must be a mapping")
        metadata = {}
    elif (
        document.get("status") == "available"
        and metadata.get("timestamp_semantics") not in {
            "exposure_midpoint",
            "exposure_start",
            "exposure_end",
        }
    ):
        schema_errors.append(
            "frame_metadata.timestamp_semantics is invalid"
        )

    dataset_scope = document.get("dataset_scope")
    if document.get("status") == "available":
        if not isinstance(dataset_scope, list):
            schema_errors.append("dataset_scope must be a list")
        elif dataset_scope:
            schema_errors.append(
                "dataset_scope must be empty for laboratory calibration"
            )

    calibrated_present = any(
        camera.get("status") == "calibrated" for camera in cameras.values()
    )
    receipt_hash: str | None = None
    receipt: dict[str, Any] | None = None
    artifact_values: dict[str, list[float]] = {}
    source_bundle_hash: str | None = None
    receipt_blockers: list[str] = []
    if calibrated_present:
        (
            receipt_hash,
            receipt,
            artifact_values,
            source_bundle_hash,
            receipt_blockers,
        ) = _validate_receipt(
            document,
            config_path.parent,
            trusted_public_key_bytes,
        )

    camera_status: dict[str, dict[str, Any]] = {}
    calibration_blockers: list[str] = list(receipt_blockers)
    for camera_id in camera_ids:
        camera = cameras.get(camera_id)
        blockers: list[str]
        if camera is None:
            blockers = [f"camera[{camera_id}]: entry is missing"]
        elif camera.get("status") != "calibrated":
            blockers = [f"camera[{camera_id}]: calibration is unavailable"]
        else:
            blockers = _validate_calibrated_camera(
                camera,
                receipt_hash,
                receipt,
                artifact_values,
                source_bundle_hash,
            )
        camera_status[str(camera_id)] = {
            "ready": not blockers,
            "blockers": blockers,
        }
        calibration_blockers.extend(blockers)

    direct_blockers = list(calibration_blockers)
    for field_name in ("exposure_sec", "sensor_gain"):
        field = metadata.get(field_name)
        if not isinstance(field, dict):
            direct_blockers.append(
                f"frame_metadata.{field_name}: definition is missing"
            )
            continue
        if field.get("provenance") not in TRUSTED_METADATA_PROVENANCE:
            direct_blockers.append(
                f"frame_metadata.{field_name}: trusted provenance is missing"
            )
        if not isinstance(field.get("field"), str) or not field["field"]:
            direct_blockers.append(
                f"frame_metadata.{field_name}: recorded field is missing"
            )

    if schema_errors:
        calibration_blockers = ["invalid schema", *calibration_blockers]
        direct_blockers = ["invalid schema", *direct_blockers]

    return {
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "calibration_set_id": document.get("calibration_set_id", ""),
        "declared_status": document.get("status", ""),
        "camera_status": camera_status,
        "mode_readiness": {
            "calibrated_direct": {
                "ready": not direct_blockers,
                "cross_camera_comparable": not direct_blockers,
                "blockers": direct_blockers,
            },
            "robust_affine": {
                "ready": not calibration_blockers,
                "cross_camera_comparable": False,
                "blockers": calibration_blockers,
            },
        },
    }


def parse_camera_ids(value: str) -> tuple[int, ...]:
    try:
        camera_ids = tuple(int(token) for token in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "camera ids must be comma-separated integers"
        ) from error
    if not camera_ids or any(camera_id < 0 for camera_id in camera_ids):
        raise argparse.ArgumentTypeError(
            "camera ids must be nonnegative and nonempty"
        )
    if len(set(camera_ids)) != len(camera_ids):
        raise argparse.ArgumentTypeError("camera ids must be unique")
    return camera_ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--camera-ids",
        default=(0, 1, 2, 3),
        type=parse_camera_ids,
        help="comma-separated camera ids (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--require-mode",
        choices=SUPPORTED_MODES,
        help="return exit code 3 unless this mode is ready",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--trusted-public-key",
        type=Path,
        help=(
            "explicit Ed25519 public-key pin used for trusted readiness"
        ),
    )
    arguments = parser.parse_args()

    config_bytes, config_error = _read_artifact_bytes(
        arguments.config.parent,
        arguments.config.name,
        "calibration config",
        MAX_CONFIG_BYTES,
    )
    if config_error is not None:
        print(
            f"cannot load calibration config: {config_error}",
            file=sys.stderr,
        )
        return 2
    assert config_bytes is not None
    try:
        document = yaml.safe_load(config_bytes.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        print(f"cannot load calibration config: {error}", file=sys.stderr)
        return 2

    trusted_public_key_bytes = None
    if arguments.trusted_public_key is not None:
        (
            trusted_public_key_bytes,
            key_error,
        ) = _read_artifact_bytes(
            arguments.trusted_public_key.parent,
            arguments.trusted_public_key.name,
            "trusted public key",
            MAX_PUBLIC_KEY_BYTES,
        )
        if key_error is not None:
            print(
                f"cannot load trusted public key: {key_error}",
                file=sys.stderr,
            )
            return 2

    report = validate_document(
        document,
        arguments.config.resolve(),
        arguments.camera_ids,
        trusted_public_key_bytes,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.json_output:
        arguments.json_output.write_text(rendered + "\n", encoding="utf-8")

    if not report["schema_valid"]:
        return 2
    if (
        arguments.require_mode
        and not report["mode_readiness"][arguments.require_mode]["ready"]
    ):
        return 3
    if not arguments.require_mode and not any(
        readiness["ready"]
        for readiness in report["mode_readiness"].values()
    ):
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
