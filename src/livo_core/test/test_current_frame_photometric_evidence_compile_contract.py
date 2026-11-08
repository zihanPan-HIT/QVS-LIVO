from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORY = PACKAGE_ROOT / "include"


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    configured_compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler = shutil.which(configured_compiler[0])
    assert compiler is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-current-frame-photometric-contract-"
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


def assert_consumer_rejected(source: str, symbol: str) -> None:
    result = compile_translation_unit(source)
    assert result.returncode != 0, (
        f"ordinary consumer unexpectedly crossed {symbol}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    diagnostics = result.stdout + result.stderr
    assert symbol in diagnostics, (
        "consumer was rejected for an unrelated compilation error; "
        f"expected {symbol!r}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_public_seam_is_one_request_to_one_sealed_decision() -> None:
    result = compile_translation_unit(
        """
#include "current_frame_photometric_evidence.h"
#include <type_traits>

using Evaluator = livo::CurrentFramePhotometricEvidenceEvaluator;
using Request = livo::CurrentFramePhotometricEvidenceRequest;
using Decision = livo::CurrentFramePhotometricEvidenceDecision;
using Evaluate = Decision (Evaluator::*)(const Request &) const;

static_assert(std::is_same_v<
  decltype(static_cast<Evaluate>(&Evaluator::evaluate)), Evaluate>);
static_assert(!std::is_default_constructible_v<Decision>);
static_assert(!std::is_aggregate_v<Decision>);
static_assert(std::is_copy_constructible_v<Decision>);
static_assert(!Decision::hasLinearization());
static_assert(!Decision::consumable());
static_assert(!Decision::productionReady());
static_assert(!Decision::solverApplied());
"""
    )
    assert result.returncode == 0, result.stderr


def test_decision_cannot_mint_legacy_or_solver_authority() -> None:
    result = compile_translation_unit(
        """
#include "current_frame_photometric_evidence.h"
#include "cross_camera_patch_migration.h"
#include <type_traits>

using Decision = livo::CurrentFramePhotometricEvidenceDecision;

static_assert(!std::is_convertible_v<
  Decision, livo::RawIntensityNoiseModel>);
static_assert(!std::is_constructible_v<
  livo::RawIntensityNoiseModel, const Decision &>);
static_assert(!std::is_convertible_v<
  Decision, livo::PhotometricPatchResult>);
static_assert(!std::is_constructible_v<
  livo::PhotometricPatchResult, const Decision &>);
static_assert(!std::is_convertible_v<
  Decision, livo::MigrationRequest>);
static_assert(!std::is_constructible_v<
  livo::MigrationRequest, const Decision &>);
static_assert(!std::is_convertible_v<
  Decision, livo::WhitenedPatchContribution>);
static_assert(!std::is_constructible_v<
  livo::WhitenedPatchContribution, const Decision &>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_decision_exposes_no_solver_or_sigma_surface() -> None:
    result = compile_translation_unit(
        """
#include "current_frame_photometric_evidence.h"
#include <type_traits>
#include <utility>

using Decision = livo::CurrentFramePhotometricEvidenceDecision;

template<typename T, typename = void>
struct HasResidual : std::false_type {};
template<typename T>
struct HasResidual<
  T, std::void_t<decltype(std::declval<T &>().residual)>>
  : std::true_type {};

template<typename T, typename = void>
struct HasJacobian : std::false_type {};
template<typename T>
struct HasJacobian<
  T, std::void_t<decltype(std::declval<T &>().jacobian)>>
  : std::true_type {};

template<typename T, typename = void>
struct HasInformation : std::false_type {};
template<typename T>
struct HasInformation<
  T, std::void_t<decltype(std::declval<T &>().information)>>
  : std::true_type {};

template<typename T, typename = void>
struct HasStandardDeviation : std::false_type {};
template<typename T>
struct HasStandardDeviation<
  T, std::void_t<decltype(
    std::declval<T &>().standardDeviationCodeValue())>>
  : std::true_type {};

static_assert(!HasResidual<Decision>::value);
static_assert(!HasJacobian<Decision>::value);
static_assert(!HasInformation<Decision>::value);
static_assert(!HasStandardDeviation<Decision>::value);
"""
    )
    assert result.returncode == 0, result.stderr


def test_ordinary_consumer_cannot_construct_or_rewrite_decision() -> None:
    assert_consumer_rejected(
        """
#include "current_frame_photometric_evidence.h"
int main() {
  livo::CurrentFramePhotometricEvidenceDecision decision{};
  return decision.productionReady() ? 0 : 1;
}
""",
        "CurrentFramePhotometricEvidenceDecision",
    )
    assert_consumer_rejected(
        """
#include "current_frame_photometric_evidence.h"
void rewrite(livo::CurrentFramePhotometricEvidenceDecision & decision) {
  decision.trusted_noise_matched_ = true;
}
""",
        "trusted_noise_matched_",
    )


def test_public_metadata_cannot_be_added_to_evaluation_call() -> None:
    assert_consumer_rejected(
        """
#include "current_frame_photometric_evidence.h"
#include "photometric_calibration.h"
void evaluate(
  const livo::CurrentFramePhotometricEvidenceEvaluator & evaluator,
  const livo::CurrentFramePhotometricEvidenceRequest & request) {
  livo::PhotometricFrameMetadata metadata;
  metadata.exposure_sec = 0.01;
  metadata.sensor_gain = 1.0;
  evaluator.evaluate(request, metadata);
}
""",
        "evaluate",
    )


def test_mutable_batch_is_not_part_of_the_evaluation_request() -> None:
    assert_consumer_rejected(
        """
#include "current_frame_photometric_evidence.h"
struct CameraObservationBatch;
void evaluate(
  const CameraObservationBatch & batch,
  const livo::ReferenceOnlyMigrationRuntimeBundle & runtime,
  const livo::ReferenceOnlyMigrationFrameProvenance & frame)
{
  livo::CurrentFramePhotometricEvidenceRequest request{
    batch, runtime, frame};
  (void)request;
}
""",
        "ValidatedBatchRuntimeComposition",
    )
