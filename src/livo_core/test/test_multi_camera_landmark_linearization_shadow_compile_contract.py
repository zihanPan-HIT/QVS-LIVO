import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORY = PACKAGE_ROOT / "include"
MODULE_HEADER = (
    INCLUDE_DIRECTORY / "multi_camera_landmark_linearization_shadow.h"
)
MODULE_SOURCE = (
    PACKAGE_ROOT / "src" / "multi_camera_landmark_linearization_shadow.cpp"
)


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    configured_compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler = shutil.which(configured_compiler[0])
    assert compiler is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-multicam-selection-shadow-contract-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        source_path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                compiler,
                *configured_compiler[1:],
                "-std=c++17",
                "-fsyntax-only",
                f"-I{INCLUDE_DIRECTORY}",
                "-isystem",
                "/usr/include/eigen3",
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def assert_consumer_rejected(source: str, boundary_name: str) -> None:
    result = compile_translation_unit(source)
    assert result.returncode != 0, (
        f"ordinary consumer unexpectedly crossed {boundary_name}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    diagnostics = result.stdout + result.stderr
    assert boundary_name in diagnostics, (
        "consumer was rejected for an unrelated compilation error; "
        f"expected diagnostic for {boundary_name}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_owner_validated_public_seam_replaces_bare_targets() -> None:
    result = compile_translation_unit(
        """
#include "multi_camera_landmark_linearization_shadow.h"
#include <type_traits>

using Module = livo::MultiCameraLandmarkLinearizationShadow;
using Session = livo::SharedLandmarkShadowSession;
using BatchAuthority = livo::CameraObservationBatchAuthorityIdentity;
using RuntimeAuthority =
  livo::ReferenceOnlyMigrationRuntimePublisherIdentity;
using Policy = livo::AdaptiveVisualPolicy;
using Runtime = livo::ReferenceOnlyMigrationRuntimeBundle;
using Capability = livo::AdaptiveVisualPolicyPlanCapability;
using Config = livo::MultiCameraLandmarkLinearizationShadowConfig;
using Result = livo::MultiCameraLandmarkShadowBatchResult;
using Process = Result (Module::*)(
  const Runtime &,
  const Capability &);

static_assert(std::is_constructible_v<
  Module, Session &, const BatchAuthority &,
  const RuntimeAuthority &, const Policy &, Config>);
static_assert(!std::is_constructible_v<Module, Session &, Config>);
static_assert(std::is_same_v<
  decltype(static_cast<Process>(&Module::processBatch)), Process>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_selection_trace_and_results_are_sealed_non_production_values() -> None:
    result = compile_translation_unit(
        """
#include "multi_camera_landmark_linearization_shadow.h"
#include "cross_camera_patch_migration.h"
#include <type_traits>

using Trace = livo::MultiCameraLandmarkSelectionTrace;
using Decision = livo::MultiCameraLandmarkShadowDecision;
using Result = livo::MultiCameraLandmarkShadowBatchResult;

static_assert(!std::is_default_constructible_v<Trace>);
static_assert(!std::is_copy_constructible_v<Trace>);
static_assert(!std::is_copy_assignable_v<Trace>);
static_assert(!std::is_move_constructible_v<Trace>);
static_assert(!std::is_move_assignable_v<Trace>);
static_assert(!std::is_default_constructible_v<Decision>);
static_assert(!std::is_default_constructible_v<Result>);
static_assert(!std::is_convertible_v<Trace, livo::MigrationRequest>);
static_assert(
  !std::is_convertible_v<Trace, livo::WhitenedPatchContribution>);
static_assert(!Trace::hasLinearization());
static_assert(!Trace::consumable());
static_assert(!Trace::productionReady());
static_assert(!Trace::solverApplied());
static_assert(!Decision::hasLinearization());
static_assert(!Decision::consumable());
static_assert(!Decision::productionReady());
static_assert(!Decision::solverApplied());
static_assert(!Result::hasLinearization());
static_assert(!Result::consumable());
static_assert(!Result::productionReady());
static_assert(!Result::solverApplied());
"""
    )
    assert result.returncode == 0, result.stderr


def test_public_shadow_values_expose_no_solver_data_api() -> None:
    result = compile_translation_unit(
        """
#include "multi_camera_landmark_linearization_shadow.h"
#include "cross_camera_patch_migration.h"
#include <type_traits>
#include <utility>

using Trace = livo::MultiCameraLandmarkSelectionTrace;
using Decision = livo::MultiCameraLandmarkShadowDecision;
using Result = livo::MultiCameraLandmarkShadowBatchResult;

template<typename T, typename = void>
struct HasPublicWhitenedResidual : std::false_type {};
template<typename T>
struct HasPublicWhitenedResidual<
  T, std::void_t<decltype(std::declval<T &>().whitened_residual)>>
  : std::true_type {};

template<typename T, typename = void>
struct HasPublicJacobian : std::false_type {};
template<typename T>
struct HasPublicJacobian<
  T, std::void_t<decltype(std::declval<T &>().jacobian)>>
  : std::true_type {};

template<typename T, typename = void>
struct HasPublicInformation : std::false_type {};
template<typename T>
struct HasPublicInformation<
  T, std::void_t<decltype(std::declval<T &>().information)>>
  : std::true_type {};

template<typename T, typename = void>
struct HasPublicSolverAppliedState : std::false_type {};
template<typename T>
struct HasPublicSolverAppliedState<
  T, std::void_t<decltype(std::declval<T &>().solver_applied)>>
  : std::true_type {};

template<typename T, typename = void>
struct HasPublicSourcePatchGeneration : std::false_type {};
template<typename T>
struct HasPublicSourcePatchGeneration<
  T, std::void_t<decltype(
    std::declval<T &>().source_patch_generation)>>
  : std::true_type {};

template<typename T>
constexpr bool CanMintSolverData =
  std::is_convertible_v<const T &, livo::MigrationRequest> ||
  std::is_constructible_v<livo::MigrationRequest, const T &> ||
  std::is_convertible_v<
    const T &, livo::WhitenedPatchContribution> ||
  std::is_constructible_v<
    livo::WhitenedPatchContribution, const T &> ||
  std::is_convertible_v<const T &, livo::FrozenAcceptedSourcePose> ||
  std::is_constructible_v<
    livo::FrozenAcceptedSourcePose, const T &> ||
  std::is_convertible_v<
    const T &, livo::FrozenSourceAcceptanceReceipt> ||
  std::is_constructible_v<
    livo::FrozenSourceAcceptanceReceipt, const T &> ||
  std::is_convertible_v<
    const T &, livo::FrozenSourceIndependenceProof> ||
  std::is_constructible_v<
    livo::FrozenSourceIndependenceProof, const T &> ||
  std::is_convertible_v<
    const T &, livo::EndpointJointPoseCovariance> ||
  std::is_constructible_v<
    livo::EndpointJointPoseCovariance, const T &> ||
  std::is_convertible_v<const T &, livo::RawIntensityNoiseModel> ||
  std::is_constructible_v<
    livo::RawIntensityNoiseModel, const T &>;

static_assert(!HasPublicWhitenedResidual<Trace>::value);
static_assert(!HasPublicWhitenedResidual<Decision>::value);
static_assert(!HasPublicWhitenedResidual<Result>::value);
static_assert(!HasPublicJacobian<Trace>::value);
static_assert(!HasPublicJacobian<Decision>::value);
static_assert(!HasPublicJacobian<Result>::value);
static_assert(!HasPublicInformation<Trace>::value);
static_assert(!HasPublicInformation<Decision>::value);
static_assert(!HasPublicInformation<Result>::value);
static_assert(!HasPublicSolverAppliedState<Trace>::value);
static_assert(!HasPublicSolverAppliedState<Decision>::value);
static_assert(!HasPublicSolverAppliedState<Result>::value);
static_assert(!HasPublicSourcePatchGeneration<Trace>::value);
static_assert(!HasPublicSourcePatchGeneration<Decision>::value);
static_assert(!HasPublicSourcePatchGeneration<Result>::value);
static_assert(!CanMintSolverData<Trace>);
static_assert(!CanMintSolverData<Decision>);
static_assert(!CanMintSolverData<Result>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_private_session_snapshot_seam_is_not_public() -> None:
    assert_consumer_rejected(
        """
#include "shared_landmark_shadow_session.h"
int main() {
  livo::SharedLandmarkShadowSession session;
  return session
    .snapshotForMultiCameraLandmarkLinearizationShadow()
    .has_value() ? 0 : 1;
}
""",
        "snapshotForMultiCameraLandmarkLinearizationShadow",
    )


def test_selection_trace_payload_is_not_public() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void rewrite(livo::MultiCameraLandmarkSelectionTrace & trace) {
  trace.payload_ = {};
}
""",
        "payload_",
    )


def test_selection_trace_generation_is_not_mutable_public_state() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void rewrite(livo::MultiCameraLandmarkSelectionTrace & trace) {
  trace.accepted_target_generation = 7U;
}
""",
        "accepted_target_generation",
    )


def test_selection_trace_store_receipt_is_not_mutable_public_state() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void rewrite(livo::MultiCameraLandmarkSelectionTrace & trace) {
  trace.source_store_receipt = {};
}
""",
        "source_store_receipt",
    )


def test_decision_has_no_public_whitened_residual_state() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void mutate(livo::MultiCameraLandmarkShadowDecision & decision) {
  decision.whitened_residual.resize(64);
}
""",
        "whitened_residual",
    )


def test_decision_has_no_public_jacobian_state() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void mutate(livo::MultiCameraLandmarkShadowDecision & decision) {
  decision.jacobian.setZero();
}
""",
        "jacobian",
    )


def test_result_has_no_public_information_state() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void mutate(livo::MultiCameraLandmarkShadowBatchResult & result) {
  result.information.setIdentity();
}
""",
        "information",
    )


def test_result_consumable_marker_is_not_mutable_state() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void mutate(livo::MultiCameraLandmarkShadowBatchResult & result) {
  result.consumable = true;
}
""",
        "consumable",
    )


def test_result_solver_applied_marker_is_not_mutable_state() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void mutate(livo::MultiCameraLandmarkShadowBatchResult & result) {
  result.solver_applied = true;
}
""",
        "solver_applied",
    )


def test_legacy_bare_target_type_no_longer_exists() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void legacy() {
  livo::MultiCameraLandmarkShadowTarget target;
  (void)target;
}
""",
        "MultiCameraLandmarkShadowTarget",
    )


def test_legacy_selection_receipt_type_no_longer_exists() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void legacy() {
  livo::MultiCameraLandmarkSourceSelectionReceipt receipt;
  (void)receipt;
}
""",
        "MultiCameraLandmarkSourceSelectionReceipt",
    )


def test_legacy_single_argument_process_batch_overload_no_longer_exists() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
void legacy(
  livo::MultiCameraLandmarkLinearizationShadow & shadow)
{
  (void)shadow.processBatch({});
}
""",
        "processBatch",
    )


def test_legacy_mutable_batch_process_overload_no_longer_exists() -> None:
    assert_consumer_rejected(
        """
#include "multi_camera_landmark_linearization_shadow.h"
struct CameraObservationBatch;
void legacy(
  livo::MultiCameraLandmarkLinearizationShadow & shadow,
  const CameraObservationBatch & batch,
  const livo::ReferenceOnlyMigrationRuntimeBundle & runtime,
  const livo::AdaptiveVisualPolicyPlanCapability & capability)
{
  (void)shadow.processBatch(batch, runtime, capability);
}
""",
        "processBatch",
    )


def test_implementation_lexical_guard_supplements_public_api_contract() -> None:
    # The compile-time tests above own the public API guarantee. This lexical
    # scan is only a review tripwire for direct private implementation
    # dependencies; aliases or indirect construction require code review.
    source = "\n".join(
        (
            MODULE_HEADER.read_text(encoding="utf-8"),
            MODULE_SOURCE.read_text(encoding="utf-8"),
        )
    )
    forbidden_identifiers = (
        "MigrationRequest",
        "WhitenedPatchContribution",
        "FrozenAcceptedSourcePose",
        "FrozenSourceAcceptanceReceipt",
        "FrozenSourceIndependenceProof",
        "EndpointJointPoseCovariance",
        "RawIntensityNoiseModel",
        "StatesGroup",
        "JointVisualUpdate",
        "MultiCameraLandmarkShadowTarget",
        "MultiCameraLandmarkSourceSelectionReceipt",
        "ArchivedSource",
        "highest_observation_id_seen",
        "source_patch_generation",
        "maximum_archive_entries",
    )
    for identifier in forbidden_identifiers:
        assert re.search(rf"\b{identifier}\b", source) is None, (
            f"selection shadow named forbidden authority/state "
            f"surface {identifier}"
        )


def test_implementation_does_not_call_diagnostic_receipt_match() -> None:
    # AcceptedObservationStoreReceipt intentionally exposes this diagnostic
    # method, so only a source-level tripwire can guard its private use here.
    source = MODULE_SOURCE.read_text(encoding="utf-8")
    assert ".matchesObservationPayload(" not in source
