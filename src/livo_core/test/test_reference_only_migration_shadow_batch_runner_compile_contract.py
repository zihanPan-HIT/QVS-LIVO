import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORY = PACKAGE_ROOT / "include"
RUNNER_HEADER = (
    INCLUDE_DIRECTORY / "reference_only_migration_shadow_batch_runner.h"
)
RUNNER_SOURCE = (
    PACKAGE_ROOT / "src" / "reference_only_migration_shadow_batch_runner.cpp"
)


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    configured_compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler = shutil.which(configured_compiler[0])
    assert compiler is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-reference-only-batch-contract-"
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


def assert_consumer_rejected(source: str) -> None:
    result = compile_translation_unit(source)
    assert result.returncode != 0, (
        "batch runner unexpectedly exposed a mutable or production path\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_batch_types_are_explicit_and_non_production() -> None:
    result = compile_translation_unit(
        """
#include "reference_only_migration_shadow_batch_runner.h"
#include <cstdint>
#include <type_traits>

using Candidate =
  livo::ReferenceOnlyMigrationShadowBatchCandidate;
using Request = livo::ReferenceOnlyMigrationShadowRequest;
using Result = livo::ReferenceOnlyMigrationShadowBatchResult;

static_assert(std::is_constructible_v<Candidate, Request>);
static_assert(!std::is_convertible_v<Request, Candidate>);
static_assert(
  !std::is_constructible_v<Candidate, Request, std::uint64_t>);
static_assert(!std::is_default_constructible_v<Result>);
static_assert(!std::is_convertible_v<Candidate, livo::MigrationRequest>);
static_assert(
  !std::is_convertible_v<
    Result, livo::WhitenedPatchContribution>);
static_assert(Result::experimental());
static_assert(!Result::consumable());
static_assert(!Result::productionReady());
static_assert(!Result::solverApplied());
static_assert(!Result::acceptanceReceiptIssued());
static_assert(!Result::independenceProofIssued());
static_assert(
  livo::kReferenceOnlyMigrationShadowPerPairBudget == 4U);
static_assert(
  livo::kReferenceOnlyMigrationShadowTotalBudget == 32U);
"""
    )
    assert result.returncode == 0, (
        "batch runner compile contract did not hold\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_runner_files_do_not_name_forbidden_authority_surfaces() -> None:
    source = "\n".join(
        (
            RUNNER_HEADER.read_text(encoding="utf-8"),
            RUNNER_SOURCE.read_text(encoding="utf-8"),
        )
    )
    forbidden_identifiers = (
        "StatesGroup",
        "VisualPoint",
        "MultiViewLandmarkStore",
        "LandmarkSnapshot",
        "MigrationRequest",
        "MigrationSource",
        "MigrationTarget",
        "WhitenedPatchContribution",
        "FrozenAcceptedSourcePose",
        "FrozenSourceAcceptanceReceipt",
        "FrozenSourceIndependenceProof",
    )
    for identifier in forbidden_identifiers:
        assert re.search(rf"\b{identifier}\b", source) is None, (
            f"runner named forbidden authority surface {identifier}"
        )


def test_candidate_request_is_read_only() -> None:
    assert_consumer_rejected(
        """
#include "reference_only_migration_shadow_batch_runner.h"

void mutate(
  livo::ReferenceOnlyMigrationShadowBatchCandidate & candidate)
{
  candidate.request().patch.patch_width = 16;
}
"""
    )


def test_candidate_has_no_state_store_or_selector_surface() -> None:
    assert_consumer_rejected(
        """
#include "reference_only_migration_shadow_batch_runner.h"

void mutate(
  livo::ReferenceOnlyMigrationShadowBatchCandidate & candidate)
{
  candidate.active_state_revision = 1U;
  candidate.store_revision = {};
  candidate.selector_snapshot_revision = 1U;
}
"""
    )


def test_result_has_no_mutable_capability_or_solver_surface() -> None:
    assert_consumer_rejected(
        """
#include "reference_only_migration_shadow_batch_runner.h"

void mutate(livo::ReferenceOnlyMigrationShadowBatchResult & result)
{
  result.consumable = true;
  result.solver_applied = true;
  result.information.setIdentity();
  result.acceptance_receipt = {};
}
"""
    )
