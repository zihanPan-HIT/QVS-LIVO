import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORY = PACKAGE_ROOT / "include"


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    configured_compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler = shutil.which(configured_compiler[0])
    assert compiler is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-reference-only-migration-contract-"
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
        "reference-only DTO unexpectedly exposed a production or write path\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_reference_only_types_are_not_production_convertible() -> None:
    result = compile_translation_unit(
        """
#include "reference_only_migration_shadow.h"
#include <type_traits>

static_assert(
  !std::is_convertible_v<
    livo::ReferenceOnlyMigrationShadowRequest,
    livo::MigrationRequest>);
static_assert(
  !std::is_constructible_v<
    livo::MigrationRequest,
    const livo::ReferenceOnlyMigrationShadowRequest &>);
static_assert(
  !std::is_convertible_v<
    livo::ReferenceOnlyMigrationShadowResult,
    livo::WhitenedPatchContribution>);
static_assert(
  !std::is_constructible_v<
    livo::WhitenedPatchContribution,
    const livo::ReferenceOnlyMigrationShadowResult &>);
static_assert(livo::ReferenceOnlyMigrationShadowResult::experimental());
static_assert(!livo::ReferenceOnlyMigrationShadowResult::consumable());
static_assert(!livo::ReferenceOnlyMigrationShadowResult::productionReady());
static_assert(!livo::ReferenceOnlyMigrationShadowResult::solverApplied());
static_assert(
  !livo::ReferenceOnlyMigrationShadowResult::acceptanceReceiptIssued());
static_assert(
  !livo::ReferenceOnlyMigrationShadowResult::independenceProofIssued());
"""
    )
    assert result.returncode == 0, (
        "public reference-only contract did not compile\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_request_cannot_be_implicitly_promoted_to_production() -> None:
    assert_consumer_rejected(
        """
#include "reference_only_migration_shadow.h"

livo::MigrationRequest promote(
  const livo::ReferenceOnlyMigrationShadowRequest & request)
{
  return request;
}
"""
    )


def test_result_cannot_be_implicitly_promoted_to_solver_information() -> None:
    assert_consumer_rejected(
        """
#include "reference_only_migration_shadow.h"

livo::WhitenedPatchContribution promote(
  const livo::ReferenceOnlyMigrationShadowResult & result)
{
  return result;
}
"""
    )


@pytest.mark.parametrize(
    "forbidden_statement",
    (
        "request.active_state_revision = 1U;",
        "request.accepted_batch_id = {};",
        "request.selector_snapshot_revision = 1U;",
        "request.store_revision = {};",
        "request.affine_gain = 1.0;",
    ),
)
def test_request_has_no_state_store_receipt_or_affine_write_surface(
    forbidden_statement: str,
) -> None:
    assert_consumer_rejected(
        f"""
#include "reference_only_migration_shadow.h"

void mutate(livo::ReferenceOnlyMigrationShadowRequest & request)
{{
  {forbidden_statement}
}}
"""
    )


@pytest.mark.parametrize(
    "forbidden_statement",
    (
        "result.information.setIdentity();",
        "result.whitened_jacobian.setZero();",
        "result.acceptance_receipt = {};",
        "result.independence_proof = {};",
        "result.consumable = true;",
        "result.solver_applied = true;",
    ),
)
def test_result_has_no_solver_receipt_or_mutable_capability_surface(
    forbidden_statement: str,
) -> None:
    assert_consumer_rejected(
        f"""
#include "reference_only_migration_shadow.h"

void mutate(livo::ReferenceOnlyMigrationShadowResult & result)
{{
  {forbidden_statement}
}}
"""
    )
