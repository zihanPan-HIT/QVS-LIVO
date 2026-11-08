#!/usr/bin/env python3
"""Pure support functions for offline photometric calibration."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import base64
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError
import yaml

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError:
    serialization = None
    Ed25519PrivateKey = None
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
CAMERA_IDS = (0, 1, 2, 3)
CONFIG_FILENAME = "photometric_calibration.yaml"
REPORT_FILENAME = "photometric_calibration_quality.json"
RECEIPT_FILENAME = "photometric_calibration_receipt.json"
SOURCE_BUNDLE_FILENAME = "photometric_calibration_source_bundle.json"
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
OWNERSHIP_MARKER_FILENAME = "photometric_calibration_ownership.json"
OWNERSHIP_MARKER_TYPE = "photometric_calibration_publish_identity"
PUBLISH_JOURNAL_TYPE = "photometric_calibration_publish_transaction"
SOLVER_NAME = "livo_core_four_camera_photometric_calibration"
SOLVER_VERSION = "2.0.0"
SIGNED_RECEIPT_TRUST_BOUNDARY = (
    "ed25519_signature_requires_explicit_external_public_key_pin"
)
UNSIGNED_RECEIPT_TRUST_BOUNDARY = (
    "unsigned_content_integrity_only_not_trusted"
)
TIMESTAMP_SEMANTICS = {
    "exposure_start",
    "exposure_midpoint",
    "exposure_end",
}
SOURCE_ENCODINGS = {
    "mono8": "identity",
    "rgb8": "bt601_luma_uint8",
}
LUT_SIZE = 256
MAX_RESPONSE_SAMPLES_PER_GROUP = 96
RESPONSE_LOG_RMSE_LIMIT = 0.14
FLATFIELD_RMS_LIMIT = 0.10
CROSS_CAMERA_RELATIVE_RMS_LIMIT = 0.08
MIN_FLATFIELD_PIXEL_SUPPORT = 2
MIN_FLATFIELD_COVERAGE = 0.98
MIN_FLATFIELD_GRID_COVERAGE = 0.85
MIN_FLATFIELD_EDGE_COVERAGE = 0.90
MIN_GRAY_CARD_PIXELS = 16
MIN_GRAY_CARD_AREA_FRACTION = 0.01
MIN_GRAY_CARD_VALID_FRACTION = 0.70
MIN_LOG_EXPOSURE_GAIN = -40.0
MAX_LOG_EXPOSURE_GAIN = 40.0
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ENCODED_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
MAX_PIXELS_PER_IMAGE = 16_777_216
MAX_CAPTURE_GROUPS = 128
MAX_FRAMES_PER_GROUP = 16
MAX_CAPTURE_FRAMES = 512
MAX_TOTAL_CAPTURE_PIXELS = 64_000_000
MAX_TOTAL_ENCODED_CAPTURE_BYTES = 512 * 1024 * 1024
MAX_DENSE_MATRIX_ELEMENTS = 8_000_000
MAX_KEY_FILE_BYTES = 64 * 1024
MAX_TRUST_DOMAIN_BYTES = 128


@dataclass(frozen=True)
class CalibrationFailure:
    """Fail-closed result that is still safe to publish for diagnostics."""

    blockers: tuple[str, ...]


class CalibrationInputError(ValueError):
    """Raised when calibration inputs cannot produce a trusted result."""

    def __init__(self, blockers: list[str] | tuple[str, ...]):
        self.blockers = tuple(blockers)
        super().__init__("; ".join(self.blockers))


@dataclass(frozen=True)
class PublishOutcome:
    """Typed result that never disguises a committed publication as failure."""

    committed: bool
    status: str
    warnings: tuple[str, ...] = ()
    retained_backup: Path | None = None

    def _messages(self) -> tuple[str, ...]:
        retained = (
            (f"previous output retained at {self.retained_backup}",)
            if self.retained_backup is not None
            else ()
        )
        return (*retained, *self.warnings)

    def __iter__(self):
        return iter(self._messages())

    def __len__(self) -> int:
        return len(self._messages())

    def __getitem__(self, index: int) -> str:
        return self._messages()[index]


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    raw_bytes: bytes
    raw_sha256: str
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class ManifestSnapshot:
    document: Any
    raw_bytes: bytes
    raw_sha256: str
    device: int
    inode: int
    size_bytes: int


@dataclass(frozen=True)
class SigningAuthority:
    """Validated signing material and its externally named trust domain."""

    private_key: Any
    public_key_sha256: str
    trust_domain: str


@dataclass(frozen=True)
class RuntimeSignedBundle:
    """Exact Ed25519 input bytes and detached runtime signature metadata."""

    payload_bytes: bytes
    signature_document: dict[str, Any]

    def files(self) -> dict[str, bytes]:
        return {
            RUNTIME_PAYLOAD_FILENAME: self.payload_bytes,
            RUNTIME_SIGNATURE_FILENAME: _canonical_json_bytes(
                self.signature_document
            ),
        }


@dataclass(frozen=True)
class CaptureFrame:
    camera_id: int
    image_path: Path
    image_sha256: str
    image_size_bytes: int
    image_device: int
    image_inode: int
    decoded_image_sha256: str
    decoded_image_mode: str
    exposure_sec: float
    sensor_gain: float
    metadata_provenance: str
    log_exposure_gain: float
    image: np.ndarray
    measurement_mask: np.ndarray | None
    mask_path: Path | None
    mask_sha256: str | None
    mask_size_bytes: int | None
    decoded_mask_sha256: str | None
    gray_card_roi: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class CaptureGroup:
    group_id: str
    capture_session_id: str
    scene_id: str
    capture_kind: str
    split: str
    static: bool
    synchronized: bool
    frames: tuple[CaptureFrame, ...]


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _checked_log_exposure_gain(
    exposure_sec: float,
    sensor_gain: float,
) -> float:
    try:
        result = math.log(exposure_sec) + math.log(sensor_gain)
    except (OverflowError, ValueError) as error:
        raise CalibrationInputError(
            [
                "exposure_sec * sensor_gain is outside supported "
                "log-domain"
            ]
        ) from error
    if (
        not math.isfinite(result)
        or result < MIN_LOG_EXPOSURE_GAIN
        or result > MAX_LOG_EXPOSURE_GAIN
    ):
        raise CalibrationInputError(
            [
                "exposure_sec * sensor_gain is outside supported "
                "log-domain"
            ]
        )
    return result


def _exposure_gain(frame: CaptureFrame) -> float:
    return math.exp(frame.log_exposure_gain)


def _decoded_image_sha256(
    mode: str,
    width: int,
    height: int,
    pixel_bytes: bytes,
) -> str:
    header = _canonical_json_bytes(
        {
            "schema_version": 1,
            "kind": "canonical_decoded_image",
            "mode": mode,
            "width": width,
            "height": height,
            "pixel_byte_count": len(pixel_bytes),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(pixel_bytes)
    return digest.hexdigest()


def _canonical_json_bytes(document: dict[str, Any]) -> bytes:
    rendered = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (rendered + "\n").encode("utf-8")


def _numeric_file_bytes(values: np.ndarray) -> bytes:
    stream = io.StringIO()
    np.savetxt(
        stream,
        np.asarray(values, dtype=np.float64).reshape(-1),
        fmt="%.17g",
    )
    return stream.getvalue().encode("utf-8")


def _snapshot_regular_file(
    path: Path,
    maximum_bytes: int,
    label: str,
    cache: dict[tuple[int, int], FileSnapshot] | None = None,
) -> FileSnapshot:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise CalibrationInputError(
                [f"{label} must not be a symlink: {path}"]
            ) from error
        raise CalibrationInputError(
            [f"cannot open {label} {path}: {error}"]
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationInputError(
                [f"{label} must be a regular file: {path}"]
            )
        if before.st_size > maximum_bytes:
            raise CalibrationInputError(
                [
                    f"{label} encoded size exceeds resource limit "
                    f"{maximum_bytes}: {path}"
                ]
            )
        identity = (before.st_dev, before.st_ino)
        cached = cache.get(identity) if cache is not None else None
        if cached is not None and (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            cached.size_bytes,
            cached.mtime_ns,
            cached.ctime_ns,
        ):
            raise CalibrationInputError(
                [f"{label} changed since its first snapshot: {path}"]
            )
        if cached is None:
            chunks: list[bytes] = []
            total_bytes = 0
            while True:
                remaining = maximum_bytes + 1 - total_bytes
                if remaining <= 0:
                    raise CalibrationInputError(
                        [
                            f"{label} encoded size exceeds resource limit "
                            f"{maximum_bytes}: {path}"
                        ]
                    )
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                total_bytes += len(chunk)
            raw_bytes = b"".join(chunks)
            if len(raw_bytes) > maximum_bytes:
                raise CalibrationInputError(
                    [
                        f"{label} encoded size exceeds resource limit "
                        f"{maximum_bytes}: {path}"
                    ]
                )
        else:
            raw_bytes = cached.raw_bytes
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise CalibrationInputError(
                [f"{label} path changed during snapshot: {path}"]
            ) from error
        stable_fields_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        stable_fields_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_identity = (path_after.st_dev, path_after.st_ino)
        if (
            stable_fields_before != stable_fields_after
            or path_identity != identity
            or stat.S_ISLNK(path_after.st_mode)
            or len(raw_bytes) != after.st_size
        ):
            raise CalibrationInputError(
                [f"{label} changed during snapshot: {path}"]
            )
        snapshot = FileSnapshot(
            path=path,
            raw_bytes=raw_bytes,
            raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            device=after.st_dev,
            inode=after.st_ino,
            size_bytes=len(raw_bytes),
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
        )
        if cache is not None and cached is None:
            cache[identity] = snapshot
        return snapshot
    finally:
        os.close(descriptor)


def load_key_file(path: Path, label: str) -> bytes:
    """Read one explicit key input through the hardened snapshot boundary."""

    return _snapshot_regular_file(
        path,
        MAX_KEY_FILE_BYTES,
        label,
    ).raw_bytes


def _load_signing_authority(
    private_key_bytes: bytes,
    pinned_public_key_bytes: bytes,
    trust_domain: str,
) -> SigningAuthority:
    if (
        serialization is None
        or Ed25519PrivateKey is None
        or Ed25519PublicKey is None
    ):
        raise CalibrationInputError(
            ["Ed25519 signing library is unavailable"]
        )
    if (
        not isinstance(trust_domain, str)
        or not trust_domain
        or len(trust_domain.encode("utf-8")) > MAX_TRUST_DOMAIN_BYTES
        or any(ord(character) < 0x21 or ord(character) > 0x7E
               for character in trust_domain)
    ):
        raise CalibrationInputError(
            ["photometric signing trust domain is invalid"]
        )
    try:
        private_key = serialization.load_pem_private_key(
            private_key_bytes,
            password=None,
        )
        public_key = serialization.load_pem_public_key(
            pinned_public_key_bytes
        )
    except (TypeError, ValueError) as error:
        raise CalibrationInputError(
            [f"Ed25519 signing key material is invalid: {error}"]
        ) from error
    if not isinstance(private_key, Ed25519PrivateKey):
        raise CalibrationInputError(
            ["signing private key is not Ed25519"]
        )
    if not isinstance(public_key, Ed25519PublicKey):
        raise CalibrationInputError(
            ["trusted public key is not Ed25519"]
        )
    private_public_raw = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    pinned_public_raw = public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if private_public_raw != pinned_public_raw:
        raise CalibrationInputError(
            ["signing private key does not match pinned public key"]
        )
    return SigningAuthority(
        private_key=private_key,
        public_key_sha256=hashlib.sha256(
            pinned_public_raw
        ).hexdigest(),
        trust_domain=trust_domain,
    )


def _sign_receipt_payload(
    payload_bytes: bytes,
    authority: SigningAuthority,
) -> tuple[dict[str, str], bytes]:
    signature = authority.private_key.sign(payload_bytes)
    return (
        {
            "algorithm": "ed25519",
            "public_key_sha256": authority.public_key_sha256,
            "signature_base64": base64.b64encode(signature).decode(
                "ascii"
            ),
        },
        signature,
    )


def _receipt_sha256_v1(
    payload_bytes: bytes,
    signature_bytes: bytes,
) -> str:
    if (
        not payload_bytes.endswith(b"\n")
        or payload_bytes.endswith(b"\n\n")
        or len(signature_bytes) != 64
    ):
        raise ValueError(
            "receipt hash v1 requires one payload LF and a "
            "64-byte Ed25519 signature"
        )
    preimage = (
        RECEIPT_HASH_DOMAIN_SEPARATOR
        + len(payload_bytes).to_bytes(8, "big")
        + payload_bytes
        + len(signature_bytes).to_bytes(8, "big")
        + signature_bytes
    )
    return hashlib.sha256(preimage).hexdigest()


def load_manifest(path: Path) -> ManifestSnapshot:
    """Load a YAML manifest without resolving any capture paths."""

    snapshot = _snapshot_regular_file(
        path,
        MAX_MANIFEST_BYTES,
        "manifest",
    )
    raw_bytes = snapshot.raw_bytes
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CalibrationInputError(
            [f"manifest must be UTF-8: {error}"]
        ) from error
    return ManifestSnapshot(
        document=yaml.safe_load(text),
        raw_bytes=raw_bytes,
        raw_sha256=snapshot.raw_sha256,
        device=snapshot.device,
        inode=snapshot.inode,
        size_bytes=snapshot.size_bytes,
    )


def validate_manifest_contract(document: Any) -> CalibrationFailure | None:
    """Validate fields that establish provenance before reading images."""

    blockers: list[str] = []
    if not isinstance(document, dict):
        return CalibrationFailure(("manifest root must be a mapping",))
    if document.get("schema_version") != 1:
        blockers.append("schema_version must equal 1")
    if not isinstance(document.get("calibration_set_id"), str) or not document[
        "calibration_set_id"
    ]:
        blockers.append("calibration_set_id must be a nonempty string")
    if not isinstance(document.get("source_id"), str) or not document[
        "source_id"
    ]:
        blockers.append("source_id must be a nonempty string")

    metadata = document.get("frame_metadata")
    if not isinstance(metadata, dict):
        blockers.append("frame_metadata must be a mapping")
        metadata = {}
    for field_name in ("exposure_sec", "sensor_gain"):
        field = metadata.get(field_name)
        prefix = f"frame_metadata.{field_name}"
        if not isinstance(field, dict):
            blockers.append(f"{prefix} definition is missing")
            continue
        if field.get("provenance") not in TRUSTED_METADATA_PROVENANCE:
            blockers.append(f"{prefix} provenance is not trusted")
        if not isinstance(field.get("field"), str) or not field["field"]:
            blockers.append(f"{prefix} recorded field is missing")
    if metadata.get("timestamp_semantics") not in TIMESTAMP_SEMANTICS:
        blockers.append(
            "frame_metadata.timestamp_semantics must be one of "
            + ",".join(sorted(TIMESTAMP_SEMANTICS))
        )

    dataset_scope = document.get("dataset_scope", [])
    if not isinstance(dataset_scope, list):
        blockers.append("dataset_scope must be a list")
    elif dataset_scope:
        blockers.append(
            "dataset_scope must be empty; laboratory calibration cannot "
            "claim an outdoor dataset scope"
        )

    raw_cameras = document.get("cameras")
    if not isinstance(raw_cameras, list):
        blockers.append("cameras must be a list")
        raw_cameras = []
    camera_ids: list[int] = []
    for index, camera in enumerate(raw_cameras):
        if not isinstance(camera, dict):
            blockers.append(f"cameras[{index}] must be a mapping")
            continue
        camera_id = camera.get("camera_id")
        if camera_id not in CAMERA_IDS:
            blockers.append(f"cameras[{index}].camera_id is invalid")
            continue
        camera_ids.append(camera_id)
        black = camera.get("black_level")
        white = camera.get("white_level")
        if (
            not _finite_number(black)
            or not _finite_number(white)
            or float(white) <= float(black)
        ):
            blockers.append(
                f"camera[{camera_id}] measured black/white levels are invalid"
            )
        if (
            camera.get("black_white_provenance")
            not in TRUSTED_CALIBRATION_PROVENANCE
        ):
            blockers.append(
                f"camera[{camera_id}] black/white provenance is not trusted"
            )
        source_encoding = camera.get("source_encoding")
        grayscale_method = camera.get("grayscale_method")
        if source_encoding not in SOURCE_ENCODINGS:
            blockers.append(
                f"camera[{camera_id}] source_encoding must be mono8 or rgb8"
            )
        elif grayscale_method != SOURCE_ENCODINGS[source_encoding]:
            blockers.append(
                f"camera[{camera_id}] grayscale_method must be "
                f"{SOURCE_ENCODINGS[source_encoding]} for {source_encoding}"
            )
        for dimension_name in ("image_width", "image_height"):
            dimension = camera.get(dimension_name)
            if (
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 1
            ):
                blockers.append(
                    f"camera[{camera_id}] {dimension_name} "
                    "must be an integer greater than one"
                )
        width = camera.get("image_width")
        height = camera.get("image_height")
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and isinstance(height, int)
            and not isinstance(height, bool)
            and (
                width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
                or width * height > MAX_PIXELS_PER_IMAGE
            )
        ):
            blockers.append(
                f"camera[{camera_id}] declared image resource limit exceeded"
            )
    if sorted(camera_ids) != list(CAMERA_IDS):
        blockers.append(
            "cameras must contain each camera_id 0,1,2,3 exactly once"
        )

    raw_captures = document.get("captures")
    if not isinstance(raw_captures, list):
        blockers.append("captures must be a list")
    else:
        if len(raw_captures) > MAX_CAPTURE_GROUPS:
            blockers.append(
                f"capture group count exceeds resource limit "
                f"{MAX_CAPTURE_GROUPS}"
            )
        dimensions_by_camera = {
            camera.get("camera_id"): (
                camera.get("image_width"),
                camera.get("image_height"),
            )
            for camera in raw_cameras
            if isinstance(camera, dict)
        }
        total_frames = 0
        total_pixels = 0
        for raw_group in raw_captures:
            if not isinstance(raw_group, dict):
                continue
            raw_frames = raw_group.get("frames")
            if not isinstance(raw_frames, list):
                continue
            if len(raw_frames) > MAX_FRAMES_PER_GROUP:
                blockers.append(
                    f"capture group {raw_group.get('group_id', '?')} frame "
                    f"count exceeds resource limit {MAX_FRAMES_PER_GROUP}"
                )
            total_frames += len(raw_frames)
            for raw_frame in raw_frames:
                if not isinstance(raw_frame, dict):
                    continue
                dimensions = dimensions_by_camera.get(
                    raw_frame.get("camera_id")
                )
                if (
                    isinstance(dimensions, tuple)
                    and len(dimensions) == 2
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value > 0
                        for value in dimensions
                    )
                ):
                    total_pixels += dimensions[0] * dimensions[1]
        if total_frames > MAX_CAPTURE_FRAMES:
            blockers.append(
                f"capture frame count exceeds resource limit "
                f"{MAX_CAPTURE_FRAMES}"
            )
        if total_pixels > MAX_TOTAL_CAPTURE_PIXELS:
            blockers.append(
                f"declared capture pixels exceed resource limit "
                f"{MAX_TOTAL_CAPTURE_PIXELS}"
            )

    return CalibrationFailure(tuple(blockers)) if blockers else None


def _safe_capture_path(manifest_path: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise CalibrationInputError(["capture image_path must be nonempty"])
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise CalibrationInputError(
            [f"capture image_path must be relative: {raw_path}"]
        )
    root = manifest_path.absolute().parent.resolve(strict=True)
    candidate = root / relative_path
    try:
        resolved_parent = candidate.parent.resolve(strict=True)
    except OSError as error:
        raise CalibrationInputError(
            [f"capture image does not exist: {raw_path}: {error}"]
        ) from error
    try:
        resolved_parent.relative_to(root)
    except ValueError as error:
        raise CalibrationInputError(
            [f"capture image_path escapes manifest directory: {raw_path}"]
        ) from error
    resolved = resolved_parent / candidate.name
    if resolved.is_symlink():
        raise CalibrationInputError(
            [f"capture image must not be a symlink: {raw_path}"]
        )
    if not _path_lexists(resolved) or not resolved.is_file():
        raise CalibrationInputError(
            [f"capture image is not a regular file: {raw_path}"]
        )
    return resolved


def _load_source_image(
    path: Path,
    source_encoding: str,
    grayscale_method: str,
    raw_bytes: bytes,
) -> tuple[np.ndarray, str, str, str]:
    if len(raw_bytes) > MAX_ENCODED_IMAGE_BYTES:
        raise CalibrationInputError(
            [f"capture image encoded size exceeds resource limit: {path}"]
        )
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            if (
                image.width > MAX_IMAGE_DIMENSION
                or image.height > MAX_IMAGE_DIMENSION
                or image.width * image.height > MAX_PIXELS_PER_IMAGE
            ):
                raise CalibrationInputError(
                    ["capture image decoded dimensions exceed resource limit"]
                )
            image.load()
            values = np.asarray(image)
            image_mode = image.mode
    except (OSError, UnidentifiedImageError) as error:
        raise CalibrationInputError(
            [f"cannot read capture image {path}: {error}"]
        ) from error
    if values.dtype != np.uint8:
        raise CalibrationInputError(
            [
                f"capture image {path} must decode as uint8 before any "
                "mode conversion"
            ]
        )
    if source_encoding == "mono8":
        if image_mode != "L" or values.ndim != 2:
            raise CalibrationInputError(
                [
                    f"capture image {path} encoding does not match "
                    "declared mono8"
                ]
            )
        grayscale = values.copy()
    elif source_encoding == "rgb8":
        if (
            image_mode != "RGB"
            or values.ndim != 3
            or values.shape[2] != 3
            or grayscale_method != "bt601_luma_uint8"
        ):
            raise CalibrationInputError(
                [
                    f"capture image {path} encoding or grayscale_method "
                    "does not match declared rgb8"
                ]
            )
        rgb = values.astype(np.float64)
        grayscale = np.rint(
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]
        ).astype(np.uint8)
    else:
        raise CalibrationInputError(
            [f"unsupported source_encoding for capture image {path}"]
        )
    if (
        grayscale.ndim != 2
        or grayscale.shape[0] <= 1
        or grayscale.shape[1] <= 1
    ):
        raise CalibrationInputError(
            [f"capture image is invalid: {path}"]
        )
    decoded_sha256 = _decoded_image_sha256(
        "L",
        int(grayscale.shape[1]),
        int(grayscale.shape[0]),
        np.ascontiguousarray(grayscale).tobytes(order="C"),
    )
    return grayscale, raw_sha256, decoded_sha256, "L"


def _load_binary_mask(
    path: Path,
    expected_shape: tuple[int, int],
    raw_bytes: bytes,
) -> tuple[np.ndarray, str, str]:
    if len(raw_bytes) > MAX_ENCODED_IMAGE_BYTES:
        raise CalibrationInputError(
            [f"gray-card mask encoded size exceeds resource limit: {path}"]
        )
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            if (
                image.width > MAX_IMAGE_DIMENSION
                or image.height > MAX_IMAGE_DIMENSION
                or image.width * image.height > MAX_PIXELS_PER_IMAGE
            ):
                raise CalibrationInputError(
                    ["gray-card mask dimensions exceed resource limit"]
                )
            image.load()
            values = np.asarray(image)
            image_mode = image.mode
    except (OSError, UnidentifiedImageError) as error:
        raise CalibrationInputError(
            [f"cannot read gray-card mask {path}: {error}"]
        ) from error
    if (
        image_mode != "L"
        or values.dtype != np.uint8
        or values.shape != expected_shape
    ):
        raise CalibrationInputError(
            [f"gray-card mask {path} must be an exact-size uint8 PIL L image"]
        )
    unique = np.unique(values)
    if not set(map(int, unique)).issubset({0, 255}):
        raise CalibrationInputError(
            [f"gray-card mask {path} must contain only 0 and 255"]
        )
    decoded_sha256 = _decoded_image_sha256(
        image_mode,
        int(values.shape[1]),
        int(values.shape[0]),
        np.ascontiguousarray(values).tobytes(order="C"),
    )
    return values == 255, raw_sha256, decoded_sha256


def _gray_card_roi_mask(
    raw_roi: Any,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if (
        not isinstance(raw_roi, list)
        or len(raw_roi) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_roi
        )
    ):
        raise CalibrationInputError(
            ["gray_card_roi must be [x, y, width, height] integers"]
        )
    x, y, width, height = map(int, raw_roi)
    image_height, image_width = image_shape
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise CalibrationInputError(
            ["gray_card_roi lies outside the capture image"]
        )
    mask = np.zeros(image_shape, dtype=bool)
    mask[y:y + height, x:x + width] = True
    return mask, (x, y, width, height)


def load_capture_groups(
    document: dict[str, Any],
    manifest_path: Path,
) -> tuple[CaptureGroup, ...]:
    """Validate and load all capture groups after provenance is trusted."""

    blockers: list[str] = []
    groups: list[CaptureGroup] = []
    seen_group_ids: set[str] = set()
    capture_path_splits: dict[Path, set[str]] = {}
    capture_hash_splits: dict[str, set[str]] = {}
    decoded_hash_splits: dict[str, set[str]] = {}
    source_snapshot_cache: dict[tuple[int, int], FileSnapshot] = {}
    source_path_by_inode: dict[tuple[int, int], Path] = {}
    source_inode_by_path: dict[Path, tuple[int, int]] = {}
    total_encoded_bytes = 0
    sessions_by_split: dict[str, set[str]] = {
        "train": set(),
        "held_out": set(),
    }
    scenes_by_split: dict[str, set[str]] = {
        "train": set(),
        "held_out": set(),
    }
    camera_settings = {
        camera["camera_id"]: {
            "dimensions": (
                int(camera["image_height"]),
                int(camera["image_width"]),
            ),
            "source_encoding": str(camera["source_encoding"]),
            "grayscale_method": str(camera["grayscale_method"]),
        }
        for camera in document["cameras"]
    }
    for group_index, raw_group in enumerate(document["captures"]):
        prefix = f"captures[{group_index}]"
        if not isinstance(raw_group, dict):
            blockers.append(f"{prefix} must be a mapping")
            continue
        group_id = raw_group.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            blockers.append(f"{prefix}.group_id must be nonempty")
            continue
        if group_id in seen_group_ids:
            blockers.append(f"duplicate group_id: {group_id}")
            continue
        seen_group_ids.add(group_id)
        capture_session_id = raw_group.get("capture_session_id")
        if (
            not isinstance(capture_session_id, str)
            or not capture_session_id
        ):
            blockers.append(f"{prefix}.capture_session_id must be nonempty")
        scene_id = raw_group.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            blockers.append(f"{prefix}.scene_id must be nonempty")
        capture_kind = raw_group.get("capture_kind")
        if capture_kind not in {"response", "flatfield", "cross_camera"}:
            blockers.append(f"{prefix}.capture_kind is invalid")
        split = raw_group.get("split")
        if split not in {"train", "held_out"}:
            blockers.append(f"{prefix}.split must be train or held_out")
        elif (
            isinstance(capture_session_id, str)
            and capture_session_id
            and isinstance(scene_id, str)
            and scene_id
        ):
            sessions_by_split[split].add(capture_session_id)
            scenes_by_split[split].add(scene_id)
        static = raw_group.get("static")
        if static is not True:
            blockers.append(f"{prefix} must declare static: true")
        synchronized = raw_group.get("synchronized")
        if not isinstance(synchronized, bool):
            blockers.append(f"{prefix}.synchronized must be boolean")
            synchronized = False
        if capture_kind in {"flatfield", "cross_camera"} and not synchronized:
            blockers.append(f"{prefix} must be synchronized")

        raw_frames = raw_group.get("frames")
        if not isinstance(raw_frames, list) or not raw_frames:
            blockers.append(f"{prefix}.frames must be a nonempty list")
            continue
        frames: list[CaptureFrame] = []
        for frame_index, raw_frame in enumerate(raw_frames):
            frame_prefix = f"{prefix}.frames[{frame_index}]"
            if not isinstance(raw_frame, dict):
                blockers.append(f"{frame_prefix} must be a mapping")
                continue
            camera_id = raw_frame.get("camera_id")
            if camera_id not in CAMERA_IDS:
                blockers.append(f"{frame_prefix}.camera_id is invalid")
                continue
            exposure_sec = raw_frame.get("exposure_sec")
            sensor_gain = raw_frame.get("sensor_gain")
            if not _finite_number(exposure_sec) or float(exposure_sec) <= 0.0:
                blockers.append(
                    f"{frame_prefix}.exposure_sec must be finite and positive"
                )
                continue
            if not _finite_number(sensor_gain) or float(sensor_gain) <= 0.0:
                blockers.append(
                    f"{frame_prefix}.sensor_gain must be finite and positive"
                )
                continue
            try:
                log_exposure_gain = _checked_log_exposure_gain(
                    float(exposure_sec), float(sensor_gain)
                )
            except CalibrationInputError as error:
                blockers.extend(
                    f"{frame_prefix}: {blocker}"
                    for blocker in error.blockers
                )
                continue
            if (
                raw_frame.get("metadata_provenance")
                not in TRUSTED_METADATA_PROVENANCE
            ):
                blockers.append(
                    f"{frame_prefix}.metadata_provenance is not trusted"
                )
                continue
            try:
                image_path = _safe_capture_path(
                    manifest_path, raw_frame.get("image_path")
                )
                settings = camera_settings[camera_id]
                image_snapshot = _snapshot_regular_file(
                    image_path,
                    MAX_ENCODED_IMAGE_BYTES,
                    "capture image",
                    source_snapshot_cache,
                )
                image_identity = (
                    image_snapshot.device,
                    image_snapshot.inode,
                )
                prior_path = source_path_by_inode.get(image_identity)
                if prior_path is not None and prior_path != image_path:
                    blockers.append(
                        f"{frame_prefix}: capture source inode is reused "
                        f"by an alias: {prior_path}, {image_path}"
                    )
                    continue
                prior_identity = source_inode_by_path.get(image_path)
                if (
                    prior_identity is not None
                    and prior_identity != image_identity
                ):
                    blockers.append(
                        f"{frame_prefix}: capture source path changed inode "
                        f"during load: {image_path}"
                    )
                    continue
                source_path_by_inode[image_identity] = image_path
                source_inode_by_path[image_path] = image_identity
                total_encoded_bytes += image_snapshot.size_bytes
                if total_encoded_bytes > MAX_TOTAL_ENCODED_CAPTURE_BYTES:
                    raise CalibrationInputError(
                        [
                            "cumulative capture encoded bytes exceed "
                            f"resource limit "
                            f"{MAX_TOTAL_ENCODED_CAPTURE_BYTES}"
                        ]
                    )
                (
                    image,
                    image_sha256,
                    decoded_image_sha256,
                    decoded_image_mode,
                ) = _load_source_image(
                    image_path,
                    settings["source_encoding"],
                    settings["grayscale_method"],
                    image_snapshot.raw_bytes,
                )
            except CalibrationInputError as error:
                blockers.extend(
                    f"{frame_prefix}: {blocker}"
                    for blocker in error.blockers
                )
                continue
            if image.shape != camera_settings[camera_id]["dimensions"]:
                blockers.append(
                    f"{frame_prefix} image shape {image.shape} does not match "
                    f"{camera_settings[camera_id]['dimensions']}"
                )
                continue
            measurement_mask: np.ndarray | None = None
            mask_path: Path | None = None
            mask_sha256: str | None = None
            mask_size_bytes: int | None = None
            decoded_mask_sha256: str | None = None
            gray_card_roi: tuple[int, int, int, int] | None = None
            if capture_kind == "cross_camera":
                raw_roi = raw_frame.get("gray_card_roi")
                raw_mask_path = raw_frame.get("gray_card_mask_path")
                if (raw_roi is None) == (raw_mask_path is None):
                    blockers.append(
                        f"{frame_prefix} must declare exactly one of "
                        "gray_card_roi or gray_card_mask_path"
                    )
                    continue
                try:
                    if raw_roi is not None:
                        measurement_mask, gray_card_roi = (
                            _gray_card_roi_mask(raw_roi, image.shape)
                        )
                    else:
                        mask_path = _safe_capture_path(
                            manifest_path, raw_mask_path
                        )
                        mask_snapshot = _snapshot_regular_file(
                            mask_path,
                            MAX_ENCODED_IMAGE_BYTES,
                            "gray-card mask",
                            source_snapshot_cache,
                        )
                        mask_identity = (
                            mask_snapshot.device,
                            mask_snapshot.inode,
                        )
                        prior_mask_path = source_path_by_inode.get(
                            mask_identity
                        )
                        if (
                            prior_mask_path is not None
                            and prior_mask_path != mask_path
                        ):
                            raise CalibrationInputError(
                                [
                                    "capture source inode is reused by an "
                                    f"alias: {prior_mask_path}, {mask_path}"
                                ]
                            )
                        source_path_by_inode[mask_identity] = mask_path
                        source_inode_by_path[mask_path] = mask_identity
                        mask_size_bytes = mask_snapshot.size_bytes
                        total_encoded_bytes += mask_snapshot.size_bytes
                        if (
                            total_encoded_bytes
                            > MAX_TOTAL_ENCODED_CAPTURE_BYTES
                        ):
                            raise CalibrationInputError(
                                [
                                    "cumulative capture encoded bytes exceed "
                                    "resource limit "
                                    f"{MAX_TOTAL_ENCODED_CAPTURE_BYTES}"
                                ]
                            )
                        (
                            measurement_mask,
                            mask_sha256,
                            decoded_mask_sha256,
                        ) = _load_binary_mask(
                            mask_path,
                            image.shape,
                            mask_snapshot.raw_bytes,
                        )
                except CalibrationInputError as error:
                    blockers.extend(
                        f"{frame_prefix}: {blocker}"
                        for blocker in error.blockers
                    )
                    continue
                selected_pixels = int(np.count_nonzero(measurement_mask))
                minimum_pixels = max(
                    MIN_GRAY_CARD_PIXELS,
                    int(math.ceil(image.size * MIN_GRAY_CARD_AREA_FRACTION)),
                )
                if selected_pixels < minimum_pixels:
                    blockers.append(
                        f"{frame_prefix} gray-card area is below minimum "
                        f"({selected_pixels} < {minimum_pixels})"
                    )
                    continue
            if split in {"train", "held_out"}:
                capture_path_splits.setdefault(image_path, set()).add(split)
                capture_hash_splits.setdefault(
                    image_sha256, set()
                ).add(split)
                decoded_hash_splits.setdefault(
                    decoded_image_sha256, set()
                ).add(split)
            frames.append(
                CaptureFrame(
                    camera_id=camera_id,
                    image_path=image_path,
                    image_sha256=image_sha256,
                    image_size_bytes=image_snapshot.size_bytes,
                    image_device=image_snapshot.device,
                    image_inode=image_snapshot.inode,
                    decoded_image_sha256=decoded_image_sha256,
                    decoded_image_mode=decoded_image_mode,
                    exposure_sec=float(exposure_sec),
                    sensor_gain=float(sensor_gain),
                    metadata_provenance=str(
                        raw_frame["metadata_provenance"]
                    ),
                    log_exposure_gain=log_exposure_gain,
                    image=image,
                    measurement_mask=measurement_mask,
                    mask_path=mask_path,
                    mask_sha256=mask_sha256,
                    mask_size_bytes=mask_size_bytes,
                    decoded_mask_sha256=decoded_mask_sha256,
                    gray_card_roi=gray_card_roi,
                )
            )
        if frames:
            camera_ids = [frame.camera_id for frame in frames]
            if capture_kind == "response" and len(set(camera_ids)) != 1:
                blockers.append(
                    f"{prefix} response group must contain one camera"
                )
            if capture_kind in {"flatfield", "cross_camera"} and sorted(
                camera_ids
            ) != list(CAMERA_IDS):
                blockers.append(
                    f"{prefix} must contain one frame from each camera"
                )
            groups.append(
                CaptureGroup(
                    group_id=group_id,
                    capture_session_id=str(capture_session_id),
                    scene_id=str(scene_id),
                    capture_kind=str(capture_kind),
                    split=str(split),
                    static=bool(static),
                    synchronized=synchronized,
                    frames=tuple(frames),
                )
            )
    for image_path, splits in capture_path_splits.items():
        if splits == {"train", "held_out"}:
            blockers.append(
                "capture image_path is reused across train and held_out: "
                f"{image_path}"
            )
    for image_sha256, splits in capture_hash_splits.items():
        if splits == {"train", "held_out"}:
            blockers.append(
                "capture image content SHA-256 is reused across train and "
                f"held_out: {image_sha256}"
            )
    for decoded_sha256, splits in decoded_hash_splits.items():
        if splits == {"train", "held_out"}:
            blockers.append(
                "decoded image SHA-256 is reused across train and held_out: "
                f"{decoded_sha256}"
            )
    shared_sessions = (
        sessions_by_split["train"] & sessions_by_split["held_out"]
    )
    if shared_sessions:
        blockers.append(
            "capture_session_id is reused across train and held_out: "
            + ",".join(sorted(shared_sessions))
        )
    shared_scenes = scenes_by_split["train"] & scenes_by_split["held_out"]
    if shared_scenes:
        blockers.append(
            "scene_id is reused across train and held_out: "
            + ",".join(sorted(shared_scenes))
        )
    if blockers:
        raise CalibrationInputError(blockers)
    return tuple(groups)


def _groups_for(
    groups: tuple[CaptureGroup, ...],
    capture_kind: str,
    split: str,
    camera_id: int | None = None,
) -> list[CaptureGroup]:
    result = [
        group
        for group in groups
        if group.capture_kind == capture_kind and group.split == split
    ]
    if camera_id is not None:
        result = [
            group
            for group in result
            if any(frame.camera_id == camera_id for frame in group.frames)
        ]
    return result


def _isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Pooled-adjacent-violators projection with unit weights."""

    block_values: list[float] = []
    block_weights: list[float] = []
    block_counts: list[int] = []
    for raw_value in np.asarray(values, dtype=np.float64):
        block_values.append(float(raw_value))
        block_weights.append(1.0)
        block_counts.append(1)
        while (
            len(block_values) >= 2
            and block_values[-2] > block_values[-1]
        ):
            total_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / total_weight
            merged_count = block_counts[-2] + block_counts[-1]
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [total_weight]
            block_counts[-2:] = [merged_count]
    return np.concatenate(
        [
            np.full(count, value, dtype=np.float64)
            for value, count in zip(block_values, block_counts)
        ]
    )


def _normalized_raw_bins(
    image: np.ndarray,
    black_level: float,
    white_level: float,
) -> np.ndarray:
    return np.clip(
        np.rint(
            (image - black_level)
            * (LUT_SIZE - 1)
            / (white_level - black_level)
        ),
        0,
        LUT_SIZE - 1,
    ).astype(np.int64)


def _response_order_violation_fraction(
    groups: list[CaptureGroup],
    camera_id: int,
) -> float:
    violations = 0
    comparisons = 0
    for group in groups:
        frames = sorted(
            (
                frame
                for frame in group.frames
                if frame.camera_id == camera_id
            ),
            key=lambda frame: frame.log_exposure_gain,
        )
        for first, second in zip(frames, frames[1:]):
            comparisons += first.image.size
            violations += int(np.count_nonzero(second.image < first.image))
    return float(violations) / float(max(comparisons, 1))


def solve_inverse_response(
    groups: list[CaptureGroup],
    camera_id: int,
    black_level: float,
    white_level: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(groups) < 2:
        raise CalibrationInputError(
            [f"camera[{camera_id}] needs at least two response train groups"]
        )
    if any(
        len([frame for frame in group.frames if frame.camera_id == camera_id])
        < 4
        for group in groups
    ):
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] response train groups need "
                "at least four exposures"
            ]
        )
    violation_fraction = _response_order_violation_fraction(
        groups, camera_id
    )
    if violation_fraction > 0.01:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] response sequence is non-monotonic "
                f"({violation_fraction:.6f})"
            ]
        )

    sampled: list[tuple[CaptureGroup, list[CaptureFrame], np.ndarray]] = []
    observed_bins: list[np.ndarray] = []
    irradiance_unknown_count = 0
    for group in groups:
        frames = [
            frame for frame in group.frames if frame.camera_id == camera_id
        ]
        pixel_count = frames[0].image.size
        sample_count = min(MAX_RESPONSE_SAMPLES_PER_GROUP, pixel_count)
        indices = np.unique(
            np.linspace(0, pixel_count - 1, sample_count, dtype=np.int64)
        )
        sampled.append((group, frames, indices))
        irradiance_unknown_count += len(indices)
        observed_bins.extend(
            _normalized_raw_bins(
                frame.image.ravel()[indices],
                black_level,
                white_level,
            )
            for frame in frames
        )
    all_observed_bins = np.concatenate(observed_bins)
    valid_observed = all_observed_bins[
        (all_observed_bins > 1) & (all_observed_bins < LUT_SIZE - 2)
    ]
    if valid_observed.size == 0:
        raise CalibrationInputError(
            [f"camera[{camera_id}] response captures are fully saturated"]
        )
    low_quantile, high_quantile = np.quantile(
        valid_observed / float(LUT_SIZE - 1), (0.02, 0.98)
    )
    unique_bins = int(np.unique(valid_observed).size)
    if low_quantile > 0.20 or high_quantile < 0.80 or unique_bins < 32:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] response saturation/dynamic coverage "
                f"is insufficient (q02={low_quantile:.3f}, "
                f"q98={high_quantile:.3f}, bins={unique_bins})"
            ]
        )

    column_count = LUT_SIZE + irradiance_unknown_count
    row_count_upper_bound = (
        sum(len(frames) * len(indices) for _, frames, indices in sampled)
        + 1
        + LUT_SIZE
        - 2
    )
    dense_elements = row_count_upper_bound * column_count
    if dense_elements > MAX_DENSE_MATRIX_ELEMENTS:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] response dense matrix resource limit "
                f"exceeded ({dense_elements} > "
                f"{MAX_DENSE_MATRIX_ELEMENTS})"
            ]
        )
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    irradiance_offset = 0
    for _, frames, indices in sampled:
        for frame in frames:
            bins = _normalized_raw_bins(
                frame.image.ravel()[indices],
                black_level,
                white_level,
            )
            log_exposure = frame.log_exposure_gain
            for local_index, intensity_bin in enumerate(bins):
                weight = 1.0 - abs(float(intensity_bin) - 127.5) / 127.5
                if weight <= 0.02:
                    continue
                row = np.zeros(column_count, dtype=np.float64)
                row[intensity_bin] = weight
                row[LUT_SIZE + irradiance_offset + local_index] = -weight
                rows.append(row)
                rhs.append(weight * log_exposure)
        irradiance_offset += len(indices)

    anchor = np.zeros(column_count, dtype=np.float64)
    anchor[LUT_SIZE // 2] = 1.0
    rows.append(anchor)
    rhs.append(0.0)
    smoothness = 12.0
    for intensity_bin in range(1, LUT_SIZE - 1):
        weight = 1.0 - abs(float(intensity_bin) - 127.5) / 127.5
        row = np.zeros(column_count, dtype=np.float64)
        row[intensity_bin - 1] = smoothness * weight
        row[intensity_bin] = -2.0 * smoothness * weight
        row[intensity_bin + 1] = smoothness * weight
        rows.append(row)
        rhs.append(0.0)
    matrix = np.vstack(rows)
    vector = np.asarray(rhs, dtype=np.float64)
    solution, _, _, _ = np.linalg.lstsq(matrix, vector, rcond=1.0e-10)
    log_response = _isotonic_non_decreasing(solution[:LUT_SIZE])
    log_response -= float(np.max(log_response))
    response = np.exp(np.clip(log_response, -700.0, 0.0))
    response -= response[0]
    dynamic_range = float(response[-1])
    if (
        not np.all(np.isfinite(response))
        or dynamic_range <= 1.0e-8
    ):
        raise CalibrationInputError(
            [f"camera[{camera_id}] response solution is degenerate"]
        )
    response /= dynamic_range
    response[0] = 0.0
    response[-1] = 1.0
    response = np.maximum.accumulate(response)
    return response, {
        "coverage_q02": float(low_quantile),
        "coverage_q98": float(high_quantile),
        "covered_bins": float(unique_bins),
        "order_violation_fraction": violation_fraction,
    }


def _apply_response(
    image: np.ndarray,
    response: np.ndarray,
    black_level: float,
    white_level: float,
) -> np.ndarray:
    coordinates = np.clip(
        (image - black_level)
        * (LUT_SIZE - 1)
        / (white_level - black_level),
        0.0,
        LUT_SIZE - 1,
    )
    return np.interp(
        coordinates.ravel(),
        np.arange(LUT_SIZE, dtype=np.float64),
        response,
    ).reshape(image.shape)


def response_held_out_log_rmse(
    groups: list[CaptureGroup],
    camera_id: int,
    response: np.ndarray,
    black_level: float,
    white_level: float,
) -> float:
    if not groups:
        raise CalibrationInputError(
            [f"camera[{camera_id}] response held-out group is missing"]
        )
    residuals: list[np.ndarray] = []
    for group in groups:
        frames = [
            frame for frame in group.frames if frame.camera_id == camera_id
        ]
        if len(frames) < 3:
            raise CalibrationInputError(
                [
                    f"camera[{camera_id}] response held-out groups "
                    "need at least three exposures"
                ]
            )
        corrected_stack = []
        valid_stack = []
        for frame in frames:
            corrected = _apply_response(
                frame.image, response, black_level, white_level
            ) / _exposure_gain(frame)
            valid = (
                (frame.image > black_level + 1.0)
                & (frame.image < white_level - 1.0)
                & (corrected > 0.0)
            )
            corrected_stack.append(corrected)
            valid_stack.append(valid)
        corrected_values = np.stack(corrected_stack)
        valid_values = np.stack(valid_stack)
        for pixel_index in range(corrected_values[0].size):
            pixel_valid = valid_values.reshape(
                valid_values.shape[0], -1
            )[:, pixel_index]
            if int(np.count_nonzero(pixel_valid)) < 2:
                continue
            logs = np.log(
                corrected_values.reshape(
                    corrected_values.shape[0], -1
                )[pixel_valid, pixel_index]
            )
            residuals.append(logs - np.median(logs))
    if not residuals:
        raise CalibrationInputError(
            [f"camera[{camera_id}] response held-out samples are invalid"]
        )
    flattened = np.concatenate(residuals)
    result = float(np.sqrt(np.mean(flattened * flattened)))
    if not math.isfinite(result) or result > RESPONSE_LOG_RMSE_LIMIT:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] response held-out log RMSE "
                f"{result:.6f} exceeds {RESPONSE_LOG_RMSE_LIMIT:.6f}"
            ]
        )
    return result


def _center_slice(height: int, width: int) -> tuple[slice, slice]:
    y_margin = max(1, height // 3)
    x_margin = max(1, width // 3)
    return (
        slice(y_margin, max(y_margin + 1, height - y_margin)),
        slice(x_margin, max(x_margin + 1, width - x_margin)),
    )


def solve_vignetting(
    groups: list[CaptureGroup],
    camera_id: int,
    response: np.ndarray,
    black_level: float,
    white_level: float,
) -> np.ndarray:
    frames = [
        frame
        for group in groups
        for frame in group.frames
        if frame.camera_id == camera_id
    ]
    if len(frames) < 3:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] needs at least three "
                "flatfield train frames"
            ]
        )
    normalized_fields: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []
    for frame in frames:
        linear = _apply_response(
            frame.image, response, black_level, white_level
        ) / _exposure_gain(frame)
        valid = (
            (frame.image > black_level + 1.0)
            & (frame.image < white_level - 1.0)
            & np.isfinite(linear)
            & (linear > 0.0)
        )
        center = _center_slice(*linear.shape)
        center_values = linear[center][valid[center]]
        if center_values.size < 2:
            raise CalibrationInputError(
                [f"camera[{camera_id}] flatfield center is invalid"]
            )
        center_level = float(np.median(center_values))
        field = np.where(valid, linear / center_level, np.nan)
        normalized_fields.append(field)
        support_masks.append(valid)
    support = np.sum(np.stack(support_masks), axis=0)
    if np.any(support < MIN_FLATFIELD_PIXEL_SUPPORT):
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] flatfield train pixels need at least "
                f"{MIN_FLATFIELD_PIXEL_SUPPORT} valid observations"
            ]
        )
    with np.errstate(all="ignore"):
        transmission = np.nanmedian(np.stack(normalized_fields), axis=0)
    if (
        not np.all(np.isfinite(transmission))
        or np.any(transmission <= 0.0)
    ):
        raise CalibrationInputError(
            [f"camera[{camera_id}] vignetting solution has invalid pixels"]
        )
    center_level = float(
        np.median(transmission[_center_slice(*transmission.shape)])
    )
    transmission /= center_level
    if (
        np.min(transmission) < 0.20
        or np.max(transmission) > 1.50
    ):
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] vignetting transmission "
                "is outside [0.20, 1.50]"
            ]
        )
    return transmission


def flatfield_held_out_rms(
    groups: list[CaptureGroup],
    camera_id: int,
    response: np.ndarray,
    transmission: np.ndarray,
    black_level: float,
    white_level: float,
) -> float:
    frames = [
        frame
        for group in groups
        for frame in group.frames
        if frame.camera_id == camera_id
    ]
    if len(frames) < MIN_FLATFIELD_PIXEL_SUPPORT:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] flatfield held-out needs at least "
                f"{MIN_FLATFIELD_PIXEL_SUPPORT} frames"
            ]
        )
    corrected_frames: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    for frame in frames:
        linear = _apply_response(
            frame.image, response, black_level, white_level
        ) / (
            _exposure_gain(frame) * transmission
        )
        valid = (
            (frame.image > black_level + 1.0)
            & (frame.image < white_level - 1.0)
            & np.isfinite(linear)
            & (linear > 0.0)
        )
        corrected_frames.append(linear)
        valid_masks.append(valid)
    support = np.sum(np.stack(valid_masks), axis=0)
    supported = support >= MIN_FLATFIELD_PIXEL_SUPPORT
    coverage = float(np.mean(supported))
    height, width = supported.shape
    ring_width = max(1, int(math.ceil(min(height, width) * 0.15)))
    edge_ring = np.zeros_like(supported, dtype=bool)
    edge_ring[:ring_width, :] = True
    edge_ring[-ring_width:, :] = True
    edge_ring[:, :ring_width] = True
    edge_ring[:, -ring_width:] = True
    edge_coverage = float(np.mean(supported[edge_ring]))
    if edge_coverage < MIN_FLATFIELD_EDGE_COVERAGE:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] flatfield held-out edge coverage "
                f"{edge_coverage:.6f} is below "
                f"{MIN_FLATFIELD_EDGE_COVERAGE:.6f}"
            ]
        )
    for y_indices in np.array_split(np.arange(height), min(4, height)):
        for x_indices in np.array_split(np.arange(width), min(4, width)):
            cell = supported[np.ix_(y_indices, x_indices)]
            cell_coverage = float(np.mean(cell))
            if cell_coverage < MIN_FLATFIELD_GRID_COVERAGE:
                raise CalibrationInputError(
                    [
                        f"camera[{camera_id}] flatfield held-out grid "
                        f"coverage {cell_coverage:.6f} is below "
                        f"{MIN_FLATFIELD_GRID_COVERAGE:.6f}"
                    ]
                )
    if coverage < MIN_FLATFIELD_COVERAGE:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] flatfield held-out pixel coverage "
                f"{coverage:.6f} is below {MIN_FLATFIELD_COVERAGE:.6f}"
            ]
        )

    residuals: list[np.ndarray] = []
    for linear, valid in zip(corrected_frames, valid_masks):
        values = linear[valid & supported]
        if values.size < max(4, linear.size // 2):
            raise CalibrationInputError(
                [f"camera[{camera_id}] flatfield held-out coverage is invalid"]
            )
        level = float(np.median(values))
        residuals.append(values / level - 1.0)
    flattened = np.concatenate(residuals)
    result = float(np.sqrt(np.mean(flattened * flattened)))
    if not math.isfinite(result) or result > FLATFIELD_RMS_LIMIT:
        raise CalibrationInputError(
            [
                f"camera[{camera_id}] flatfield held-out RMS "
                f"{result:.6f} exceeds {FLATFIELD_RMS_LIMIT:.6f}"
            ]
        )
    return result


def _frame_linear_radiance(
    frame: CaptureFrame,
    response: np.ndarray,
    transmission: np.ndarray,
    black_level: float,
    white_level: float,
) -> float:
    values = _apply_response(
        frame.image, response, black_level, white_level
    ) / (
        _exposure_gain(frame) * transmission
    )
    valid = (
        (frame.image > black_level + 1.0)
        & (frame.image < white_level - 1.0)
        & np.isfinite(values)
        & (values > 0.0)
    )
    if frame.measurement_mask is None:
        raise CalibrationInputError(
            [f"cross-camera frame {frame.image_path} has no gray-card mask"]
        )
    valid &= frame.measurement_mask
    valid_values = values[valid]
    selected_pixels = int(np.count_nonzero(frame.measurement_mask))
    minimum_valid = max(
        4,
        int(math.ceil(selected_pixels * MIN_GRAY_CARD_VALID_FRACTION)),
    )
    if valid_values.size < minimum_valid:
        raise CalibrationInputError(
            [
                f"cross-camera frame {frame.image_path} has insufficient "
                "valid gray-card pixels "
                f"({valid_values.size} < {minimum_valid})"
            ]
        )
    return float(np.median(valid_values))


def _robust_affine(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[float, float]:
    design = np.column_stack(
        [x_values, np.ones_like(x_values, dtype=np.float64)]
    )
    if (
        len(x_values) < 4
        or float(np.ptp(x_values)) <= 1.0e-6
        or np.linalg.cond(design) > 1.0e6
    ):
        raise CalibrationInputError(
            ["cross-camera radiance levels are insufficient or degenerate"]
        )
    weights = np.ones_like(x_values, dtype=np.float64)
    parameters = np.zeros(2, dtype=np.float64)
    for _ in range(8):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = y_values * np.sqrt(weights)
        parameters, _, _, _ = np.linalg.lstsq(
            weighted_design, weighted_target, rcond=None
        )
        residual = design @ parameters - y_values
        mad = float(
            1.4826 * np.median(np.abs(residual - np.median(residual)))
        )
        scale = max(mad, 1.0e-8)
        normalized = np.abs(residual) / (1.5 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / normalized)
    gain, bias = map(float, parameters)
    if not math.isfinite(gain) or gain <= 0.0 or not math.isfinite(bias):
        raise CalibrationInputError(
            ["cross-camera affine solution is invalid"]
        )
    return gain, bias


def solve_cross_camera_affine(
    train_groups: list[CaptureGroup],
    held_out_groups: list[CaptureGroup],
    camera_data: dict[int, dict[str, Any]],
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    if len(train_groups) < 4:
        raise CalibrationInputError(
            ["cross-camera calibration needs at least four train groups"]
        )
    if len(held_out_groups) < 2:
        raise CalibrationInputError(
            ["cross-camera calibration needs at least two held-out groups"]
        )

    def measurements(group: CaptureGroup) -> dict[int, float]:
        result: dict[int, float] = {}
        for frame in group.frames:
            data = camera_data[frame.camera_id]
            result[frame.camera_id] = _frame_linear_radiance(
                frame,
                data["response"],
                data["transmission"],
                data["black_level"],
                data["white_level"],
            )
        if sorted(result) != list(CAMERA_IDS):
            raise CalibrationInputError(
                [f"group {group.group_id} is missing a camera"]
            )
        return result

    train_measurements = [
        measurements(group) for group in train_groups
    ]
    held_out_measurements = [
        measurements(group) for group in held_out_groups
    ]
    affine: dict[int, tuple[float, float]] = {0: (1.0, 0.0)}
    held_out_rms: dict[int, float] = {0: 0.0}
    y_train = np.asarray(
        [measurement[0] for measurement in train_measurements]
    )
    for camera_id in CAMERA_IDS[1:]:
        x_train = np.asarray(
            [measurement[camera_id] for measurement in train_measurements]
        )
        gain, bias = _robust_affine(x_train, y_train)
        affine[camera_id] = (gain, bias)
        errors = []
        for measurement in held_out_measurements:
            reference = measurement[0]
            aligned = gain * measurement[camera_id] + bias
            errors.append(
                (aligned - reference) / max(abs(reference), 1.0e-6)
            )
        rms = float(np.sqrt(np.mean(np.square(errors))))
        if (
            not math.isfinite(rms)
            or rms > CROSS_CAMERA_RELATIVE_RMS_LIMIT
        ):
            raise CalibrationInputError(
                [
                    f"camera[{camera_id}] cross-camera held-out relative RMS "
                    f"{rms:.6f} exceeds "
                    f"{CROSS_CAMERA_RELATIVE_RMS_LIMIT:.6f}"
                ]
            )
        held_out_rms[camera_id] = rms
    return affine, held_out_rms


def _receipt_capture_groups(
    groups: tuple[CaptureGroup, ...],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    root = manifest_path.resolve().parent
    result: list[dict[str, Any]] = []
    for group in groups:
        frames: list[dict[str, Any]] = []
        for frame in group.frames:
            entry: dict[str, Any] = {
                "camera_id": frame.camera_id,
                "image_path": frame.image_path.relative_to(root).as_posix(),
                "image_sha256": frame.image_sha256,
                "raw_image_sha256": frame.image_sha256,
                "raw_image_size_bytes": frame.image_size_bytes,
                "source_device": frame.image_device,
                "source_inode": frame.image_inode,
                "decoded_image_sha256": frame.decoded_image_sha256,
                "decoded_image_mode": frame.decoded_image_mode,
                "exposure_sec": frame.exposure_sec,
                "sensor_gain": frame.sensor_gain,
                "metadata_provenance": frame.metadata_provenance,
            }
            if frame.gray_card_roi is not None:
                entry["gray_card_roi"] = list(frame.gray_card_roi)
            if frame.mask_path is not None:
                entry["gray_card_mask_path"] = (
                    frame.mask_path.relative_to(root).as_posix()
                )
                entry["gray_card_mask_sha256"] = frame.mask_sha256
                entry["raw_gray_card_mask_sha256"] = frame.mask_sha256
                entry["raw_gray_card_mask_size_bytes"] = (
                    frame.mask_size_bytes
                )
                entry["decoded_gray_card_mask_sha256"] = (
                    frame.decoded_mask_sha256
                )
            frames.append(entry)
        result.append(
            {
                "group_id": group.group_id,
                "capture_session_id": group.capture_session_id,
                "scene_id": group.scene_id,
                "capture_kind": group.capture_kind,
                "split": group.split,
                "static": group.static,
                "synchronized": group.synchronized,
                "frames": frames,
            }
        )
    return result


def _build_source_bundle(
    manifest_snapshot: ManifestSnapshot,
    document: dict[str, Any],
    manifest_path: Path,
    groups: tuple[CaptureGroup, ...],
) -> tuple[dict[str, Any], bytes, str]:
    manifest_utf8 = manifest_snapshot.raw_bytes.decode("utf-8")
    capture_frames = [
        frame for group in groups for frame in group.frames
    ]
    source_counts = {
        "camera_count": len(document["cameras"]),
        "capture_group_count": len(groups),
        "capture_frame_count": len(capture_frames),
        "train_frame_count": sum(
            len(group.frames) for group in groups if group.split == "train"
        ),
        "held_out_frame_count": sum(
            len(group.frames)
            for group in groups
            if group.split == "held_out"
        ),
        "cumulative_encoded_bytes": sum(
            frame.image_size_bytes + (frame.mask_size_bytes or 0)
            for frame in capture_frames
        ),
    }
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "bundle_type": "photometric_calibration_source_snapshot",
        "solver": {
            "name": SOLVER_NAME,
            "version": SOLVER_VERSION,
        },
        "snapshot_semantics": (
            "hashes_and_metadata_of_bytes_consumed_by_this_solver_run"
        ),
        "manifest_snapshot": {
            "encoding": "utf-8",
            "raw_utf8": manifest_utf8,
            "raw_size_bytes": len(manifest_snapshot.raw_bytes),
            "raw_sha256": manifest_snapshot.raw_sha256,
            "source_device": manifest_snapshot.device,
            "source_inode": manifest_snapshot.inode,
            "canonical_document_sha256": hashlib.sha256(
                _canonical_json_bytes(document)
            ).hexdigest(),
        },
        "frame_metadata": document["frame_metadata"],
        "cameras": document["cameras"],
        "captures": _receipt_capture_groups(groups, manifest_path),
        "source_counts": source_counts,
        "trust_boundary": (
            "snapshot_content_integrity_not_external_lab_identity"
        ),
    }
    bundle_bytes = _canonical_json_bytes(bundle)
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    return bundle, bundle_bytes, bundle_sha256


def _build_calibration_receipt(
    document: dict[str, Any],
    calibration: dict[str, Any],
    source_bundle_sha256: str,
    source_counts: dict[str, int],
    manifest_raw_sha256: str,
    output_cameras: list[dict[str, Any]],
    camera_quality: dict[str, dict[str, float]],
    thresholds: dict[str, float],
    numeric_files: dict[str, np.ndarray],
    signing_authority: SigningAuthority | None,
) -> tuple[
    dict[str, Any],
    str,
    RuntimeSignedBundle | None,
]:
    receipt_cameras = []
    solutions = {
        camera["camera_id"]: camera for camera in output_cameras
    }
    for camera in sorted(
        document["cameras"], key=lambda item: item["camera_id"]
    ):
        solution = solutions[camera["camera_id"]]
        receipt_cameras.append(
            {
                "camera_id": camera["camera_id"],
                "name": camera["name"],
                "image_width": camera["image_width"],
                "image_height": camera["image_height"],
                "source_encoding": camera["source_encoding"],
                "grayscale_method": camera["grayscale_method"],
                "black_level": camera["black_level"],
                "white_level": camera["white_level"],
                "black_white_provenance": camera[
                    "black_white_provenance"
                ],
                "inverse_response_lut_file": solution[
                    "inverse_response_lut_file"
                ],
                "vignetting_transmission_file": solution[
                    "vignetting_transmission_file"
                ],
                "radiance_scale": solution["radiance_scale"],
                "radiance_offset": solution["radiance_offset"],
                "covariance_scale": solution["covariance_scale"],
            }
        )
    held_out_metrics = {
        camera_id: {
            key: value
            for key, value in metrics.items()
            if "held_out" in key
        }
        for camera_id, metrics in camera_quality.items()
    }
    artifacts = {
        filename: hashlib.sha256(_numeric_file_bytes(values)).hexdigest()
        for filename, values in sorted(numeric_files.items())
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "four_camera_photometric_calibration",
        "solver": {
            "name": SOLVER_NAME,
            "version": SOLVER_VERSION,
        },
        "calibration_set_id": document["calibration_set_id"],
        "declared_source_id": document["source_id"],
        "manifest_sha256": manifest_raw_sha256,
        "source_bundle_file": SOURCE_BUNDLE_FILENAME,
        "source_bundle_sha256": source_bundle_sha256,
        "source_counts": source_counts,
        "frame_metadata": document["frame_metadata"],
        "cameras": receipt_cameras,
        "calibration_config_payload": calibration,
        "calibration_config_sha256": hashlib.sha256(
            _canonical_json_bytes(calibration)
        ).hexdigest(),
        "held_out_metrics": held_out_metrics,
        "thresholds": thresholds,
        "artifacts_sha256": artifacts,
        "integrity_scope": (
            "source_snapshot_config_solver_metrics_numeric_artifacts"
        ),
        "trust_boundary": (
            SIGNED_RECEIPT_TRUST_BOUNDARY
            if signing_authority is not None
            else UNSIGNED_RECEIPT_TRUST_BOUNDARY
        ),
    }
    runtime_bundle: RuntimeSignedBundle | None = None
    receipt_sha256: str
    if signing_authority is not None:
        payload["trust_root"] = {
            "trust_domain": signing_authority.trust_domain,
            "public_key_sha256": (
                signing_authority.public_key_sha256
            ),
        }
        payload["receipt_hash_scheme"] = RECEIPT_HASH_SCHEME
        payload_bytes = _canonical_json_bytes(payload)
        signature, signature_bytes = _sign_receipt_payload(
            payload_bytes,
            signing_authority,
        )
        receipt_sha256 = _receipt_sha256_v1(
            payload_bytes,
            signature_bytes,
        )
        payload["signature"] = signature
        runtime_bundle = RuntimeSignedBundle(
            payload_bytes=payload_bytes,
            signature_document={
                "schema_version": 1,
                "signature_type": RUNTIME_SIGNATURE_TYPE,
                "algorithm": signature["algorithm"],
                "solver": {
                    "name": SOLVER_NAME,
                    "version": SOLVER_VERSION,
                },
                "trust_domain": signing_authority.trust_domain,
                "payload_file": RUNTIME_PAYLOAD_FILENAME,
                "payload_sha256": hashlib.sha256(
                    payload_bytes
                ).hexdigest(),
                "public_key_sha256": (
                    signing_authority.public_key_sha256
                ),
                "receipt_hash_scheme": RECEIPT_HASH_SCHEME,
                "receipt_sha256": receipt_sha256,
                "signature_base64": (
                    signature["signature_base64"]
                ),
            },
        )
    else:
        receipt_sha256 = hashlib.sha256(
            _canonical_json_bytes(payload)
        ).hexdigest()
    return (
        {**payload, "receipt_sha256": receipt_sha256},
        receipt_sha256,
        runtime_bundle,
    )


def solve_calibration(
    document: dict[str, Any],
    manifest_path: Path,
    manifest_snapshot: ManifestSnapshot,
    signing_private_key_bytes: bytes | None = None,
    pinned_public_key_bytes: bytes | None = None,
    trust_domain: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, Any],
    RuntimeSignedBundle | None,
]:
    """Solve all four cameras and return schema, report, and numeric files."""

    signing_inputs_present = (
        signing_private_key_bytes is not None,
        pinned_public_key_bytes is not None,
        trust_domain is not None,
    )
    if any(signing_inputs_present) and not all(signing_inputs_present):
        raise CalibrationInputError(
            [
                "signing private key, trusted public key, and trust "
                "domain must be provided together"
            ]
        )
    signing_authority = (
        _load_signing_authority(
            signing_private_key_bytes,
            pinned_public_key_bytes,
            trust_domain,
        )
        if all(signing_inputs_present)
        else None
    )
    contract_failure = validate_manifest_contract(document)
    if contract_failure is not None:
        raise CalibrationInputError(contract_failure.blockers)
    groups = load_capture_groups(document, manifest_path)
    cameras = {
        camera["camera_id"]: camera for camera in document["cameras"]
    }
    camera_data: dict[int, dict[str, Any]] = {}
    camera_quality: dict[str, dict[str, float]] = {}
    for camera_id in CAMERA_IDS:
        camera = cameras[camera_id]
        black_level = float(camera["black_level"])
        white_level = float(camera["white_level"])
        response_train = _groups_for(
            groups, "response", "train", camera_id
        )
        response_held = _groups_for(
            groups, "response", "held_out", camera_id
        )
        response, response_quality = solve_inverse_response(
            response_train,
            camera_id,
            black_level,
            white_level,
        )
        response_rmse = response_held_out_log_rmse(
            response_held,
            camera_id,
            response,
            black_level,
            white_level,
        )
        flat_train = _groups_for(
            groups, "flatfield", "train", camera_id
        )
        flat_held = _groups_for(
            groups, "flatfield", "held_out", camera_id
        )
        transmission = solve_vignetting(
            flat_train,
            camera_id,
            response,
            black_level,
            white_level,
        )
        flat_rms = flatfield_held_out_rms(
            flat_held,
            camera_id,
            response,
            transmission,
            black_level,
            white_level,
        )
        camera_data[camera_id] = {
            "response": response,
            "transmission": transmission,
            "black_level": black_level,
            "white_level": white_level,
            "response_rmse": response_rmse,
            "flat_rms": flat_rms,
        }
        camera_quality[str(camera_id)] = {
            **response_quality,
            "response_held_out_log_rmse": response_rmse,
            "flatfield_held_out_rms": flat_rms,
        }

    affine, cross_rms = solve_cross_camera_affine(
        _groups_for(groups, "cross_camera", "train"),
        _groups_for(groups, "cross_camera", "held_out"),
        camera_data,
    )
    output_cameras: list[dict[str, Any]] = []
    numeric_files: dict[str, np.ndarray] = {}
    for camera_id in CAMERA_IDS:
        camera = cameras[camera_id]
        gain, bias = affine[camera_id]
        covariance_scale = min(
            100.0,
            1.0
            + (
                camera_data[camera_id]["response_rmse"]
                / RESPONSE_LOG_RMSE_LIMIT
            )
            ** 2
            + (
                camera_data[camera_id]["flat_rms"]
                / FLATFIELD_RMS_LIMIT
            )
            ** 2
            + (
                cross_rms[camera_id]
                / CROSS_CAMERA_RELATIVE_RMS_LIMIT
            )
            ** 2,
        )
        lut_filename = f"camera_{camera_id}_inverse_response_lut.txt"
        vignette_filename = (
            f"camera_{camera_id}_vignetting_transmission.txt"
        )
        numeric_files[lut_filename] = camera_data[camera_id]["response"]
        numeric_files[vignette_filename] = camera_data[camera_id][
            "transmission"
        ].reshape(-1)
        output_cameras.append(
            {
                "camera_id": camera_id,
                "name": camera["name"],
                "status": "calibrated",
                "provenance": "laboratory_measured",
                "source_id": "",
                "image_width": int(camera["image_width"]),
                "image_height": int(camera["image_height"]),
                "black_level": float(camera["black_level"]),
                "white_level": float(camera["white_level"]),
                "inverse_response_lut_file": lut_filename,
                "vignetting_transmission_file": vignette_filename,
                "radiance_scale": gain,
                "radiance_offset": bias,
                "covariance_scale": covariance_scale,
            }
        )
        camera_quality[str(camera_id)][
            "cross_camera_held_out_relative_rms"
        ] = cross_rms[camera_id]
        camera_quality[str(camera_id)]["radiance_scale"] = gain
        camera_quality[str(camera_id)]["radiance_offset"] = bias
        camera_quality[str(camera_id)][
            "covariance_scale"
        ] = covariance_scale

    source_bundle, _, source_bundle_sha256 = _build_source_bundle(
        manifest_snapshot,
        document,
        manifest_path,
        groups,
    )
    for camera in output_cameras:
        camera["source_id"] = (
            f"source-bundle-sha256:{source_bundle_sha256}"
        )
        camera["source_bundle_sha256"] = source_bundle_sha256
    calibration = {
        "schema_version": 1,
        "calibration_set_id": document["calibration_set_id"],
        "dataset_scope": document.get("dataset_scope", []),
        "status": "available",
        "frame_metadata": document["frame_metadata"],
        "source_bundle_file": SOURCE_BUNDLE_FILENAME,
        "source_bundle_sha256": source_bundle_sha256,
        "calibration_receipt_file": RECEIPT_FILENAME,
        "cameras": output_cameras,
    }
    if signing_authority is not None:
        calibration.update(
            {
                "runtime_payload_file": RUNTIME_PAYLOAD_FILENAME,
                "runtime_signature_file": RUNTIME_SIGNATURE_FILENAME,
                "trust_root": {
                    "trust_domain": signing_authority.trust_domain,
                    "public_key_sha256": (
                        signing_authority.public_key_sha256
                    ),
                },
            }
        )
    thresholds = {
        "response_held_out_log_rmse_max": RESPONSE_LOG_RMSE_LIMIT,
        "flatfield_held_out_rms_max": FLATFIELD_RMS_LIMIT,
        "cross_camera_held_out_relative_rms_max": (
            CROSS_CAMERA_RELATIVE_RMS_LIMIT
        ),
    }
    report = {
        "schema_version": 1,
        "calibration_set_id": document["calibration_set_id"],
        "source_id": document["source_id"],
        "calibrated": True,
        "blockers": [],
        "thresholds": thresholds,
        "camera_quality": camera_quality,
        "held_out_quality": {
            "passed": True,
            "response_groups": len(
                _groups_for(groups, "response", "held_out")
            ),
            "flatfield_groups": len(
                _groups_for(groups, "flatfield", "held_out")
            ),
            "cross_camera_groups": len(
                _groups_for(groups, "cross_camera", "held_out")
            ),
        },
    }
    receipt, receipt_sha256, runtime_bundle = _build_calibration_receipt(
        document,
        calibration,
        source_bundle_sha256,
        source_bundle["source_counts"],
        manifest_snapshot.raw_sha256,
        output_cameras,
        camera_quality,
        thresholds,
        numeric_files,
        signing_authority,
    )
    report["calibration_receipt_file"] = RECEIPT_FILENAME
    report["calibration_receipt_sha256"] = receipt_sha256
    report["source_bundle_file"] = SOURCE_BUNDLE_FILENAME
    report["source_bundle_sha256"] = source_bundle_sha256
    return (
        calibration,
        report,
        numeric_files,
        receipt,
        source_bundle,
        runtime_bundle,
    )


def unavailable_document(
    manifest: Any,
    blockers: tuple[str, ...],
) -> dict[str, Any]:
    """Build an existing-schema document that cannot enable either mode."""

    source = manifest if isinstance(manifest, dict) else {}
    raw_cameras = source.get("cameras")
    cameras_by_id = (
        {
            camera.get("camera_id"): camera
            for camera in raw_cameras
            if (
                isinstance(camera, dict)
                and camera.get("camera_id") in CAMERA_IDS
            )
        }
        if isinstance(raw_cameras, list)
        else {}
    )
    cameras = []
    for camera_id in CAMERA_IDS:
        camera = cameras_by_id.get(camera_id, {})
        cameras.append(
            {
                "camera_id": camera_id,
                "name": camera.get("name", f"camera_{camera_id}"),
                "status": "unavailable",
                "provenance": "unavailable",
                "source_id": "",
                "image_width": camera.get("image_width", 0),
                "image_height": camera.get("image_height", 0),
                "black_level": None,
                "white_level": None,
                "inverse_response_lut_file": "",
                "vignetting_transmission_file": "",
                "radiance_scale": None,
                "radiance_offset": None,
                "covariance_scale": None,
            }
        )
    return {
        "schema_version": 1,
        "calibration_set_id": source.get(
            "calibration_set_id", "photometric-calibration-unavailable"
        ),
        "dataset_scope": source.get("dataset_scope", []),
        "status": "unavailable",
        "blocking_reason": "; ".join(blockers),
        "frame_metadata": {
            "exposure_sec": {"provenance": "unavailable", "field": ""},
            "sensor_gain": {"provenance": "unavailable", "field": ""},
            "timestamp_semantics": "unknown",
        },
        "cameras": cameras,
    }


def quality_failure_report(
    blockers: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "calibrated": False,
        "blockers": list(blockers),
        "camera_quality": {},
        "held_out_quality": {},
    }


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _renameat2(source: Path, destination: Path, flags: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        operation = {
            1: "RENAME_NOREPLACE",
            2: "RENAME_EXCHANGE",
        }.get(flags, f"flags={flags}")
        raise OSError(
            errno.ENOTSUP,
            f"renameat2({operation}) is required for safe publishing",
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        flags,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename without overwriting a concurrently created path."""

    _renameat2(source, destination, 1)


def _rename_exchange(first: Path, second: Path) -> None:
    """Atomically exchange two existing paths."""

    _renameat2(first, second, 2)


def _journal_path(output_directory: Path) -> Path:
    return output_directory.parent / (
        f".{output_directory.name}.publish-journal.json"
    )


def _lock_path(output_directory: Path) -> Path:
    return output_directory.parent / f".{output_directory.name}.publish.lock"


def _canonical_output_directory(output_directory: Path) -> str:
    return str(
        output_directory.parent.resolve(strict=True)
        / output_directory.name
    )


def _config_receipt_sha256(
    config_sha256: str,
    receipt_sha256: str | None,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "calibration_config_sha256": config_sha256,
                "calibration_receipt_sha256": receipt_sha256,
            }
        )
    ).hexdigest()


def _build_ownership_marker(
    canonical_output_directory: str,
    transaction_id: str,
    config_bytes: bytes,
    receipt_bytes: bytes | None,
) -> dict[str, Any]:
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    receipt_sha256 = (
        hashlib.sha256(receipt_bytes).hexdigest()
        if receipt_bytes is not None
        else None
    )
    return {
        "schema_version": 1,
        "marker_type": OWNERSHIP_MARKER_TYPE,
        "canonical_output_directory": canonical_output_directory,
        "transaction_id": transaction_id,
        "calibration_config_sha256": config_sha256,
        "calibration_receipt_sha256": receipt_sha256,
        "config_receipt_sha256": _config_receipt_sha256(
            config_sha256,
            receipt_sha256,
        ),
    }


def _read_regular_file(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise OSError(f"{label} must be a regular non-symlink file: {path}")
    return path.read_bytes()


def _validate_staging_identity(
    staging: Path,
    canonical_output_directory: str,
    transaction_id: str,
) -> None:
    if staging.is_symlink() or not staging.is_dir():
        raise OSError(
            f"publish staging must be a non-symlink directory: {staging}"
        )
    marker_bytes = _read_regular_file(
        staging / OWNERSHIP_MARKER_FILENAME,
        "publish ownership marker",
    )
    try:
        marker = json.loads(marker_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise OSError(
            f"publish ownership marker is invalid: {error}"
        ) from error
    if (
        not isinstance(marker, dict)
        or marker.get("schema_version") != 1
        or marker.get("marker_type") != OWNERSHIP_MARKER_TYPE
        or marker.get("canonical_output_directory")
        != canonical_output_directory
        or marker.get("transaction_id") != transaction_id
    ):
        raise OSError("publish ownership marker identity does not match")

    config_bytes = _read_regular_file(
        staging / CONFIG_FILENAME,
        "staged calibration config",
    )
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    receipt_path = staging / RECEIPT_FILENAME
    receipt_sha256: str | None = None
    if _path_lexists(receipt_path):
        receipt_bytes = _read_regular_file(
            receipt_path,
            "staged calibration receipt",
        )
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if marker.get("calibration_config_sha256") != config_sha256:
        raise OSError("publish ownership marker config hash does not match")
    if marker.get("calibration_receipt_sha256") != receipt_sha256:
        raise OSError("publish ownership marker receipt hash does not match")
    if marker.get("config_receipt_sha256") != _config_receipt_sha256(
        config_sha256,
        receipt_sha256,
    ):
        raise OSError(
            "publish ownership marker combined hash does not match"
        )


def _cleanup_current_staging(
    staging: Path,
    created_identity: tuple[int, int],
    directory_descriptor: int,
) -> None:
    if not _path_lexists(staging):
        return
    opened_metadata = os.fstat(directory_descriptor)
    metadata = os.lstat(staging)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != created_identity
        or (
            opened_metadata.st_dev,
            opened_metadata.st_ino,
        )
        != created_identity
    ):
        raise OSError(
            "current-process staging identity changed; preserving it"
        )
    entry_names = os.listdir(directory_descriptor)
    for entry_name in entry_names:
        entry_metadata = os.stat(
            entry_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(entry_metadata.st_mode):
            raise OSError(
                "current-process staging contains a non-regular entry; "
                "preserving it"
            )
    for entry_name in entry_names:
        os.unlink(entry_name, dir_fd=directory_descriptor)
    staging.rmdir()
    _fsync_directory(staging.parent)


def _validate_staging_allowlist(
    staging_descriptor: int,
    expected_names: set[str],
) -> None:
    actual_names = set(os.listdir(staging_descriptor))
    if actual_names != expected_names:
        unexpected = sorted(actual_names - expected_names)
        missing = sorted(expected_names - actual_names)
        raise OSError(
            "staging allowlist mismatch; "
            f"unexpected={unexpected}, missing={missing}"
        )
    for entry_name in sorted(actual_names):
        metadata = os.stat(
            entry_name,
            dir_fd=staging_descriptor,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(
                f"staging entry must be a regular file: {entry_name}"
            )
        if metadata.st_nlink != 1:
            raise OSError(
                f"staging entry must have exactly one hard link: {entry_name}"
            )


def _write_publish_journal(
    path: Path,
    document: dict[str, Any],
) -> None:
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        payload = (
            json.dumps(document, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with os.fdopen(temporary_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise OSError(f"publish journal must not be a symlink: {path}")
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if _path_lexists(temporary_path):
            temporary_path.unlink()


def _retire_publish_journal(path: Path) -> None:
    if path.is_symlink():
        raise OSError(f"publish journal must not be a symlink: {path}")
    if _path_lexists(path):
        path.unlink()
        _fsync_directory(path.parent)


def _recover_publish_journal(output_directory: Path) -> list[str]:
    journal_path = _journal_path(output_directory)
    if not _path_lexists(journal_path):
        return []
    if journal_path.is_symlink():
        raise OSError(
            f"publish journal must not be a symlink: {journal_path}"
        )
    if not journal_path.is_file():
        raise OSError(
            f"publish journal must be a regular file: {journal_path}"
        )
    raise OSError(
        "manual recovery required: an active publish journal has no "
        "automatic rename authority; output, staging, backup, and journal "
        "were preserved"
    )


def _acquire_publish_lock(output_directory: Path) -> int:
    path = _lock_path(output_directory)
    if path.is_symlink():
        raise OSError(f"publish lock must not be a symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise OSError(f"publish lock is held: {path}") from error
    return descriptor


def publish_documents(
    output_directory: Path,
    calibration: dict[str, Any],
    quality_report: dict[str, Any],
    numeric_files: dict[str, np.ndarray] | None = None,
    receipt: dict[str, Any] | None = None,
    source_bundle: dict[str, Any] | None = None,
    runtime_bundle: RuntimeSignedBundle | None = None,
    *,
    force: bool,
) -> PublishOutcome:
    """Publish one exact result without deleting pre-existing directories."""

    output_directory = output_directory.absolute()
    if not output_directory.name:
        raise ValueError("output directory must name one exact result")
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink():
        raise OSError(
            f"output directory must not be a symlink: {output_directory}"
        )
    lock_descriptor = _acquire_publish_lock(output_directory)
    staging: Path | None = None
    staging_identity: tuple[int, int] | None = None
    staging_descriptor: int | None = None
    backup: Path | None = None
    transaction_id: str | None = None
    published = False
    journal_written = False
    journal_path = _journal_path(output_directory)
    try:
        _recover_publish_journal(output_directory)
        if _path_lexists(output_directory) and not force:
            raise FileExistsError(
                f"output directory already exists: {output_directory}"
            )
        if output_directory.is_symlink():
            raise OSError(
                f"output directory must not be a symlink: {output_directory}"
            )
        for _ in range(16):
            candidate_transaction_id = uuid.uuid4().hex
            candidate_staging = parent / (
                f".{output_directory.name}.tmp-"
                f"{candidate_transaction_id}"
            )
            candidate_backup = parent / (
                f".{output_directory.name}.backup-"
                f"{candidate_transaction_id}"
            )
            if (
                _path_lexists(candidate_staging)
                or _path_lexists(candidate_backup)
            ):
                continue
            try:
                candidate_staging.mkdir(mode=0o700)
            except FileExistsError:
                continue
            transaction_id = candidate_transaction_id
            staging = candidate_staging
            break
        if staging is None or transaction_id is None:
            raise OSError("cannot allocate a unique publish transaction")
        staging_open_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            staging_open_flags |= os.O_NOFOLLOW
        staging_descriptor = os.open(staging, staging_open_flags)
        staging_metadata = os.fstat(staging_descriptor)
        staging_identity = (
            staging_metadata.st_dev,
            staging_metadata.st_ino,
        )
        canonical_output_directory = _canonical_output_directory(
            output_directory
        )
        config_bytes = yaml.safe_dump(
            calibration,
            sort_keys=False,
        ).encode("utf-8")
        report_bytes = (
            json.dumps(quality_report, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        receipt_bytes = (
            (
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            if receipt is not None
            else None
        )
        _write_bytes_fsynced(
            staging / CONFIG_FILENAME,
            config_bytes,
        )
        _write_bytes_fsynced(
            staging / REPORT_FILENAME,
            report_bytes,
        )
        if receipt_bytes is not None:
            _write_bytes_fsynced(
                staging / RECEIPT_FILENAME,
                receipt_bytes,
            )
        if source_bundle is not None:
            _write_bytes_fsynced(
                staging / SOURCE_BUNDLE_FILENAME,
                _canonical_json_bytes(source_bundle),
            )
        runtime_files = (
            runtime_bundle.files()
            if runtime_bundle is not None
            else {}
        )
        for filename, contents in runtime_files.items():
            if filename not in {
                RUNTIME_PAYLOAD_FILENAME,
                RUNTIME_SIGNATURE_FILENAME,
            }:
                raise ValueError(
                    f"invalid runtime signed output: {filename}"
                )
            _write_bytes_fsynced(staging / filename, contents)
        for filename, values in (numeric_files or {}).items():
            if (
                not filename
                or Path(filename).name != filename
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(f"invalid numeric output: {filename}")
            _write_bytes_fsynced(
                staging / filename, _numeric_file_bytes(values)
            )
        ownership_marker = _build_ownership_marker(
            canonical_output_directory,
            transaction_id,
            config_bytes,
            receipt_bytes,
        )
        _write_bytes_fsynced(
            staging / OWNERSHIP_MARKER_FILENAME,
            (
                json.dumps(
                    ownership_marker,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _fsync_directory(staging)
        expected_staging_names = {
            CONFIG_FILENAME,
            REPORT_FILENAME,
            OWNERSHIP_MARKER_FILENAME,
            *(numeric_files or {}).keys(),
        }
        if receipt_bytes is not None:
            expected_staging_names.add(RECEIPT_FILENAME)
        if source_bundle is not None:
            expected_staging_names.add(SOURCE_BUNDLE_FILENAME)
        expected_staging_names.update(runtime_files)
        _validate_staging_allowlist(
            staging_descriptor,
            expected_staging_names,
        )
        _validate_staging_identity(
            staging,
            canonical_output_directory,
            transaction_id,
        )

        if _path_lexists(output_directory):
            backup = parent / (
                f".{output_directory.name}.backup-{transaction_id}"
            )
            if backup.is_symlink():
                raise OSError(
                    f"publish backup path must not be a symlink: {backup}"
                )
            if _path_lexists(backup):
                raise OSError(
                    f"publish backup path already exists: {backup}"
                )
        journal = {
            "schema_version": 2,
            "journal_type": PUBLISH_JOURNAL_TYPE,
            "transaction_id": transaction_id,
            "canonical_output_directory": canonical_output_directory,
            "output_name": output_directory.name,
            "staging_name": staging.name,
            "backup_name": backup.name if backup is not None else None,
            "phase": "prepared",
            "force": force,
        }
        _write_publish_journal(journal_path, journal)
        journal_written = True
        _validate_staging_allowlist(
            staging_descriptor,
            expected_staging_names,
        )
        if backup is not None:
            if output_directory.is_symlink():
                raise OSError(
                    "output directory became a symlink before replacement: "
                    f"{output_directory}"
                )
            if backup.is_symlink():
                raise OSError(
                    f"publish backup path must not be a symlink: {backup}"
                )
            _rename_exchange(staging, output_directory)
        else:
            _rename_no_replace(staging, output_directory)
        published = True
        post_commit_warnings: list[str] = []
        retained_backup: Path | None = None
        if backup is not None:
            try:
                _rename_no_replace(staging, backup)
                retained_backup = backup
            except OSError as error:
                retained_backup = staging
                post_commit_warnings.append(
                    "old output remains at exchanged staging path "
                    f"{staging}: {error}"
                )
        try:
            _fsync_directory(parent)
        except OSError as error:
            post_commit_warnings.append(
                f"parent fsync failed after commit: {error}"
            )
        journal["phase"] = "committed"
        try:
            _write_publish_journal(journal_path, journal)
        except OSError as error:
            post_commit_warnings.append(
                f"journal commit record failed after commit: {error}"
            )
        try:
            _retire_publish_journal(journal_path)
            journal_written = False
        except OSError as error:
            post_commit_warnings.append(
                f"journal retirement failed after commit: {error}"
            )
        return PublishOutcome(
            committed=True,
            status=(
                "committed_with_warning"
                if post_commit_warnings
                else "committed"
            ),
            warnings=tuple(post_commit_warnings),
            retained_backup=retained_backup,
        )
    except Exception as original_error:
        cleanup_errors: list[str] = []
        if (
            not published
            and staging is not None
            and staging_identity is not None
            and _path_lexists(staging)
        ):
            try:
                assert staging_descriptor is not None
                _cleanup_current_staging(
                    staging,
                    staging_identity,
                    staging_descriptor,
                )
            except OSError as error:
                cleanup_errors.append(f"staging cleanup failed: {error}")
        if journal_written and _path_lexists(journal_path):
            try:
                _retire_publish_journal(journal_path)
            except OSError as error:
                cleanup_errors.append(f"journal retirement failed: {error}")
        if cleanup_errors:
            raise OSError(
                f"{original_error}; " + "; ".join(cleanup_errors)
            ) from original_error
        if published:
            return PublishOutcome(
                committed=True,
                status="committed_with_warning",
                warnings=(f"post-commit failure: {original_error}",),
                retained_backup=(
                    backup
                    if backup is not None and _path_lexists(backup)
                    else staging
                    if staging is not None and _path_lexists(staging)
                    else None
                ),
            )
        raise
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(lock_descriptor)
