from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PACKAGE_ROOT.parents[1] / "build" / PACKAGE_ROOT.name
FLAGS_FILE = (
    BUILD_ROOT
    / "CMakeFiles"
    / "test_reference_only_migration_runtime_provenance.dir"
    / "flags.make"
)


def cmake_flags(variable: str) -> list[str]:
    prefix = f"{variable} = "
    for line in FLAGS_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return shlex.split(line.removeprefix(prefix))
    raise AssertionError(f"{variable} missing from {FLAGS_FILE}")


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("c++")
    assert compiler is not None
    assert FLAGS_FILE.is_file(), (
        "configure the focused CMake test target before running this "
        "compile contract"
    )
    with tempfile.TemporaryDirectory(
        prefix="livo-runtime-frame-provenance-contract-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        source_path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                compiler,
                *cmake_flags("CXX_DEFINES"),
                *cmake_flags("CXX_INCLUDES"),
                *cmake_flags("CXX_FLAGS"),
                "-fsyntax-only",
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def test_public_capability_contract_is_copyable_but_not_mintable() -> None:
    result = compile_translation_unit(
        """
#include "reference_only_migration_runtime_provenance.h"
#include <type_traits>

using Provenance = livo::ReferenceOnlyMigrationFrameProvenance;
using Composition = livo::ValidatedBatchRuntimeComposition;
static_assert(!std::is_default_constructible_v<Provenance>);
static_assert(!std::is_aggregate_v<Provenance>);
static_assert(std::is_copy_constructible_v<Provenance>);
static_assert(!std::is_default_constructible_v<Composition>);
static_assert(!std::is_aggregate_v<Composition>);
static_assert(std::is_copy_constructible_v<Composition>);
static_assert(Provenance::experimentalOnly());
static_assert(!Provenance::consumable());
static_assert(!Provenance::productionReady());
static_assert(!Provenance::solverApplied());
"""
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "forgery",
    (
        """
int main() {
  livo::ReferenceOnlyMigrationFrameProvenance forged{};
  return forged.valid() ? 0 : 1;
}
""",
        """
namespace livo {
class OrdinaryConsumer {
public:
  static ReferenceOnlyMigrationFrameProvenance forge() {
    return ReferenceOnlyMigrationFrameProvenance(
      std::shared_ptr<
        const ReferenceOnlyMigrationFrameProvenance::Binding>{});
  }
};
}
""",
        """
int main() {
  livo::ValidatedBatchRuntimeComposition forged{};
  return forged.valid() ? 0 : 1;
}
""",
        """
namespace livo {
class OrdinaryConsumer {
public:
  static ValidatedBatchRuntimeComposition forge() {
    return ValidatedBatchRuntimeComposition(
      std::shared_ptr<
        const ValidatedBatchRuntimeComposition::Binding>{});
  }
};
}
""",
    ),
)
def test_ordinary_consumer_cannot_construct_or_forge(
    forgery: str,
) -> None:
    result = compile_translation_unit(
        f"""
#include "reference_only_migration_runtime_provenance.h"
#include <memory>
{forgery}
"""
    )
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly minted frame provenance\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "private" in result.stderr or "no matching" in result.stderr


def test_consumer_cannot_trigger_batch_digest_validation() -> None:
    result = compile_translation_unit(
        """
#include "reference_only_migration_runtime_provenance.h"
void validate(
  const livo::CameraObservationBatchReceipt & receipt,
  const CameraObservationBatch & batch)
{
  (void)receipt.claimAndValidate(batch);
}
"""
    )
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly triggered batch hashing\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "private" in result.stderr

    legacy = compile_translation_unit(
        """
#include "reference_only_migration_runtime_provenance.h"
void validate(
  const livo::ReferenceOnlyMigrationRuntimeBundle & runtime,
  const CameraObservationBatch & batch,
  const livo::CameraObservationBatchAuthorityIdentity & identity)
{
  (void)runtime.acceptsBatch(batch, identity);
}
"""
    )
    assert legacy.returncode != 0, (
        "runtime unexpectedly retained public acceptsBatch\n"
        f"stdout:\n{legacy.stdout}\nstderr:\n{legacy.stderr}"
    )
