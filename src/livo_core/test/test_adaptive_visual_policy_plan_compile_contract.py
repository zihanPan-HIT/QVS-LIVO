import os
from pathlib import Path
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
        prefix="livo-adaptive-plan-contract-"
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


def assert_contract_rejects(source: str) -> None:
    result = compile_translation_unit(source)
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly forged or edited an adaptive "
        "visual plan capability\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_capability_is_not_default_constructible() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_policy.h"
#include <type_traits>
static_assert(!std::is_default_constructible_v<
  livo::AdaptiveVisualPolicyPlanCapability>);
static_assert(std::is_copy_constructible_v<
  livo::AdaptiveVisualPolicyPlanCapability>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_ordinary_consumer_cannot_construct_capability_payload() -> None:
    assert_contract_rejects(
        """
#include "adaptive_visual_policy.h"
int main() {
  livo::AdaptiveVisualPolicyPlanCapability forged;
  return forged.valid() ? 0 : 1;
}
"""
    )


def test_ordinary_consumer_cannot_replace_private_payload() -> None:
    assert_contract_rejects(
        """
#include "adaptive_visual_policy.h"
int main() {
  livo::AdaptiveVisualPolicyPlanCapability * capability = nullptr;
  capability->data_.reset();
  return 0;
}
"""
    )


def test_public_capability_surface_exposes_read_only_payload() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_policy.h"
#include <type_traits>
using Capability = livo::AdaptiveVisualPolicyPlanCapability;
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Capability &>().decisionPlan())>>);
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Capability &>().lidarPoseInformation())>>);
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Capability &>().observations())>>);
"""
    )
    assert result.returncode == 0, result.stderr
