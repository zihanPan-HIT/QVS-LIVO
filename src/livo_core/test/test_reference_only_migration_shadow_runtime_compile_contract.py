from pathlib import Path
import re


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
VIO_SOURCE = PACKAGE_ROOT / "src" / "vio.cpp"


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace + 1 : index]
    raise AssertionError(f"unterminated function body for {signature}")


def test_runtime_seam_has_no_production_solver_or_store_write_surface() -> None:
    source = VIO_SOURCE.read_text(encoding="utf-8")
    runtime = function_body(
        source,
        "void VIOManager::runReferenceOnlyMigrationShadowRuntime(",
    )

    forbidden_identifiers = (
        "MigrationRequest",
        "WhitenedPatchContribution",
        "computeJointJacobianAndUpdateEKF",
        "bootstrapGeometry",
        "commitObservations",
        "findLandmark",
        "MultiViewLandmarkStore",
        "active_visual_feature_map_",
        "camera_feature_maps_",
    )
    for identifier in forbidden_identifiers:
        assert re.search(rf"\b{identifier}\b", runtime) is None, (
            f"metrics-only runtime named forbidden authority surface "
            f"{identifier}"
        )

    assert re.search(r"\bstate\s*->\s*(?!cov\b)", runtime) is None
    assert re.search(r"\bstate_propagat\s*->", runtime) is None
    assert re.search(r"\b(?:state|state_propagat)\s*->[^;]*=", runtime) is None
    assert "shared_landmark_shadow_session_.audit()" in runtime
    assert "shared_landmark_shadow_session_.configure" not in runtime
    assert "shared_landmark_shadow_session_.commit" not in runtime
    assert "Frame::w2f" not in runtime
    assert "->w2f(" not in runtime
    assert "ReferenceOnlyMigrationShadowRequest" in runtime
    assert "runReferenceOnlyMigrationShadowBatch" in runtime


def test_runtime_is_between_bootstrap_and_joint_solver() -> None:
    source = VIO_SOURCE.read_text(encoding="utf-8")
    process_frames = function_body(
        source,
        "void VIOManager::processFrames(",
    )
    bootstrap = process_frames.index("bootstrapSharedLandmarksShadow(")
    runtime = process_frames.index(
        "runReferenceOnlyMigrationShadowRuntime("
    )
    solver = process_frames.index("computeJointJacobianAndUpdateEKF(")
    assert bootstrap < runtime < solver


def test_bootstrap_seed_capture_does_not_read_back_shadow_store() -> None:
    source = VIO_SOURCE.read_text(encoding="utf-8")
    bootstrap = function_body(
        source,
        "VIOManager::bootstrapSharedLandmarksShadow(",
    )
    assert "resolveGeometryIdentity(" in bootstrap
    assert "findLandmark(" not in bootstrap
    assert "landmarkSnapshot(" not in bootstrap
    assert "storeSnapshot(" not in bootstrap
