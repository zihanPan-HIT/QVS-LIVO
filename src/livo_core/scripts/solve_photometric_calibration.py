#!/usr/bin/env python3
"""Solve a fail-closed four-camera photometric calibration from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

from photometric_calibration_solver import (
    CalibrationInputError,
    PublishOutcome,
    load_key_file,
    load_manifest,
    publish_documents,
    quality_failure_report,
    solve_calibration,
    unavailable_document,
    validate_manifest_contract,
)


def _published_exit_code(
    *,
    calibrated: bool,
    trusted: bool,
    outcome: PublishOutcome,
) -> int:
    """Classify a publication without conflating commit and trust state."""

    if not outcome.committed:
        return 2
    if outcome.status == "committed_with_warning":
        return 5
    if not calibrated:
        return 3
    return 0 if trusted else 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "publish a replacement while retaining the exact previous "
            "output as a unique backup"
        ),
    )
    parser.add_argument(
        "--signing-private-key",
        type=Path,
        help="Ed25519 private key used only to sign the receipt payload",
    )
    parser.add_argument(
        "--trusted-public-key",
        type=Path,
        help="explicit Ed25519 public-key pin that must match the signer",
    )
    parser.add_argument(
        "--trust-domain",
        help=(
            "application-owned printable ASCII trust-domain identifier "
            "bound into the signed payload"
        ),
    )
    arguments = parser.parse_args()
    signing_inputs_present = (
        arguments.signing_private_key is not None,
        arguments.trusted_public_key is not None,
        arguments.trust_domain is not None,
    )
    if any(signing_inputs_present) and not all(signing_inputs_present):
        print(
            "signing private key, trusted public key, and trust domain "
            "must be provided together",
            file=sys.stderr,
        )
        return 2
    signing_private_key_bytes = None
    pinned_public_key_bytes = None
    if arguments.signing_private_key is not None:
        try:
            signing_private_key_bytes = load_key_file(
                arguments.signing_private_key,
                "signing private key",
            )
            assert arguments.trusted_public_key is not None
            pinned_public_key_bytes = load_key_file(
                arguments.trusted_public_key,
                "trusted public key",
            )
        except CalibrationInputError as error:
            print(
                f"cannot load signing material: {error}",
                file=sys.stderr,
            )
            return 2

    failure_blockers = None
    try:
        manifest_snapshot = load_manifest(arguments.manifest)
        manifest = manifest_snapshot.document
    except CalibrationInputError as error:
        manifest = {}
        failure_blockers = error.blockers
    except (MemoryError, OverflowError, RecursionError, ValueError) as error:
        manifest = {}
        failure_blockers = (
            "manifest numeric/resource failure: "
            f"{type(error).__name__}: {error}",
        )
    except (OSError, yaml.YAMLError) as error:
        print(f"cannot load manifest: {error}", file=sys.stderr)
        return 2

    failure = (
        validate_manifest_contract(manifest)
        if failure_blockers is None
        else None
    )
    numeric_files = None
    receipt = None
    source_bundle = None
    runtime_bundle = None
    calibrated = False
    if failure_blockers is None and failure is None:
        try:
            (
                calibration,
                quality_report,
                numeric_files,
                receipt,
                source_bundle,
                runtime_bundle,
            ) = solve_calibration(
                manifest,
                arguments.manifest,
                manifest_snapshot,
                signing_private_key_bytes,
                pinned_public_key_bytes,
                arguments.trust_domain,
            )
            calibrated = True
        except CalibrationInputError as error:
            failure_blockers = error.blockers
        except (
            ArithmeticError,
            MemoryError,
            OSError,
            RecursionError,
            ValueError,
        ) as error:
            failure_blockers = (
                "calibration numeric/resource failure: "
                f"{type(error).__name__}: {error}",
            )
    elif failure is not None:
        failure_blockers = failure.blockers
    if not calibrated:
        assert failure_blockers is not None
        calibration = unavailable_document(manifest, failure_blockers)
        quality_report = quality_failure_report(failure_blockers)
    try:
        publish_outcome = publish_documents(
            arguments.output_dir,
            calibration,
            quality_report,
            numeric_files,
            receipt,
            source_bundle,
            runtime_bundle,
            force=arguments.force,
        )
    except (OSError, ValueError) as error:
        print(f"cannot publish result: {error}", file=sys.stderr)
        return 2
    for warning in publish_outcome:
        print(f"publish warning: {warning}", file=sys.stderr)
    return _published_exit_code(
        calibrated=calibrated,
        trusted=(
            signing_private_key_bytes is not None
            and pinned_public_key_bytes is not None
            and arguments.trust_domain is not None
            and runtime_bundle is not None
        ),
        outcome=publish_outcome,
    )


if __name__ == "__main__":
    raise SystemExit(main())
