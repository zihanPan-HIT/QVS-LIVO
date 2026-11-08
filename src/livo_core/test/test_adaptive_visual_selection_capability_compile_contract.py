from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = PACKAGE_ROOT.parents[1] / "build" / PACKAGE_ROOT.name
FLAGS_FILE = (
    BUILD_ROOT
    / "CMakeFiles"
    / "test_adaptive_visual_selection_capability.dir"
    / "flags.make"
)
MAIN_SELECTION_TEST = (
    BUILD_ROOT / "test_adaptive_visual_selection_capability"
)
POLICY_ALLOCATION_FAILURE_TEST = (
    BUILD_ROOT
    / "test_adaptive_visual_policy_reset_allocation_failure"
)
SELECTION_ALLOCATION_FAILURE_TEST = (
    BUILD_ROOT
    / "test_adaptive_visual_selection_receipt_allocation_failure"
)
PRODUCTION_POLICY_LIBRARY = (
    BUILD_ROOT / "libadaptive_visual_policy.a"
)
PRODUCTION_SELECTION_LIBRARY = (
    BUILD_ROOT / "libadaptive_visual_selection_authority.a"
)
SELECTION_HEADER = (
    PACKAGE_ROOT
    / "include"
    / "adaptive_visual_selection_authority.h"
)
SELECTION_SOURCE = (
    PACKAGE_ROOT
    / "src"
    / "adaptive_visual_selection_authority.cpp"
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
    assert FLAGS_FILE.is_file()
    with tempfile.TemporaryDirectory(
        prefix="livo-adaptive-selection-contract-"
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


def defined_symbols(binary: Path) -> str:
    nm = shutil.which("nm")
    assert nm is not None
    assert binary.is_file(), f"missing linkage-contract input: {binary}"
    result = subprocess.run(
        [nm, "-C", "--defined-only", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_failure_injection_hooks_are_isolated_to_dedicated_tests() -> None:
    policy_hook = "livo::test::failNextPolicyOwnerAllocation()"
    selection_hook = (
        "livo::test::failNextSelectionReceiptAllocation()"
    )

    main_symbols = defined_symbols(MAIN_SELECTION_TEST)
    assert policy_hook not in main_symbols
    assert selection_hook not in main_symbols

    production_policy_symbols = defined_symbols(
        PRODUCTION_POLICY_LIBRARY
    )
    production_selection_symbols = defined_symbols(
        PRODUCTION_SELECTION_LIBRARY
    )
    assert policy_hook not in production_policy_symbols
    assert selection_hook not in production_selection_symbols

    policy_test_symbols = defined_symbols(
        POLICY_ALLOCATION_FAILURE_TEST
    )
    assert policy_hook in policy_test_symbols
    assert selection_hook not in policy_test_symbols

    selection_test_symbols = defined_symbols(
        SELECTION_ALLOCATION_FAILURE_TEST
    )
    assert selection_hook in selection_test_symbols
    assert policy_hook not in selection_test_symbols


def test_capabilities_are_copyable_but_not_publicly_mintable() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_selection_authority.h"
#include <type_traits>

using Token = livo::AuthoritativeLidarGenerationToken;
using Capability = livo::AdaptiveVisualSelectionCapability;
using Receipt = livo::AdaptiveVisualSelectionReceipt;
using ConsumeResult = livo::AdaptiveVisualSelectionConsumeResult;
using Authority = livo::AdaptiveVisualSelectionAuthority;
using BatchIdentity =
  livo::CameraObservationBatchAuthorityIdentity;

static_assert(!std::is_default_constructible_v<Token>);
static_assert(!std::is_default_constructible_v<Capability>);
static_assert(!std::is_default_constructible_v<Receipt>);
static_assert(std::is_copy_constructible_v<Token>);
static_assert(std::is_copy_constructible_v<Capability>);
static_assert(std::is_copy_constructible_v<Receipt>);
static_assert(std::is_nothrow_move_constructible_v<Receipt>);
static_assert(std::is_nothrow_move_constructible_v<ConsumeResult>);
static_assert(!std::is_constructible_v<Authority, BatchIdentity>);
static_assert(!std::is_copy_constructible_v<Authority>);
static_assert(!std::is_move_constructible_v<Authority>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_runtime_input_owns_authority_lifetime_without_mutable_escape() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_selection_authority.h"
#include <memory>
#include <type_traits>
#include <utility>

using Authority = livo::AdaptiveVisualSelectionAuthority;
using Input = livo::AdaptiveVisualRuntimeAuthorityInput;
using Token = livo::AuthoritativeLidarGenerationToken;
using Permit = livo::AdaptiveVisualRuntimeAttemptPermit;
using AttemptResult = livo::AdaptiveVisualRuntimeAttemptResult;
using IssuedAuthority = decltype(
  std::declval<CameraObservationQueue &>().
  issueAdaptiveVisualSelectionAuthority());

template<typename T, typename = void>
struct HasMutableAuthorityAccessor : std::false_type {};

template<typename T>
struct HasMutableAuthorityAccessor<
  T,
  std::void_t<decltype(std::declval<const T &>().authority())>>
: std::true_type {};

template<typename T, typename = void>
struct HasIssueSelection : std::false_type {};

template<typename T>
struct HasIssueSelection<
  T,
  std::void_t<decltype(
    std::declval<T &>().issueSelection(
      std::declval<const livo::AdaptiveVisualPolicy &>(),
      std::declval<const CameraObservationBatch &>(),
      std::declval<
        const livo::AdaptiveVisualCandidatePlanCapability &>()))>>
: std::true_type {};

template<typename T, typename = void>
struct HasConsumeSelection : std::false_type {};

template<typename T>
struct HasConsumeSelection<
  T,
  std::void_t<decltype(
    std::declval<T &>().consumeSelection(
      std::declval<const livo::AdaptiveVisualPolicy &>(),
      std::declval<const CameraObservationBatch &>(),
      std::declval<
        const livo::AdaptiveVisualSelectionCapability &>()))>>
: std::true_type {};

static_assert(std::is_same_v<
  IssuedAuthority, std::shared_ptr<Authority>>);
static_assert(!std::is_default_constructible_v<Input>);
static_assert(std::is_copy_constructible_v<Input>);
static_assert(!std::is_constructible_v<
  Input, std::shared_ptr<Authority>, Token>);
static_assert(!HasMutableAuthorityAccessor<Input>::value);
static_assert(!HasIssueSelection<Input>::value);
static_assert(!HasConsumeSelection<Input>::value);
static_assert(!std::is_default_constructible_v<Permit>);
static_assert(!std::is_copy_constructible_v<Permit>);
static_assert(!std::is_copy_assignable_v<Permit>);
static_assert(std::is_move_constructible_v<Permit>);
static_assert(std::is_move_assignable_v<Permit>);
static_assert(std::is_same_v<
  std::remove_reference_t<
    decltype(std::declval<AttemptResult &>().permit)>,
  std::optional<Permit>>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_ordinary_consumer_cannot_construct_selection_authority() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_selection_authority.h"
void forge(
  livo::CameraObservationBatchAuthorityIdentity identity)
{
  livo::AdaptiveVisualSelectionAuthority authority(identity);
}
"""
    )
    assert result.returncode != 0
    assert "private" in result.stderr


def test_selected_camera_not_candidate_status_remains_defensive() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_selection_authority.h"

constexpr auto defensive_status =
  livo::AdaptiveVisualSelectionStatus::
  kSelectedCameraNotCandidate;
static_assert(
  defensive_status !=
  livo::AdaptiveVisualSelectionStatus::kReady);
"""
    )
    assert result.returncode == 0, result.stderr


def test_candidate_capability_replaces_partial_public_api() -> None:
    candidate = compile_translation_unit(
        """
#include "adaptive_visual_policy.h"
#include <type_traits>

using CandidateCapability =
  livo::AdaptiveVisualCandidatePlanCapability;
using Policy = livo::AdaptiveVisualPolicy;
static_assert(std::is_copy_constructible_v<CandidateCapability>);
static_assert(!std::is_convertible_v<
  CandidateCapability,
  livo::AdaptiveVisualPolicyPlanCapability>);
static_assert(std::is_same_v<
  decltype(&Policy::issueCandidatePlan),
  std::optional<CandidateCapability> (Policy::*)(
    const CameraObservationBatch &,
    const Policy::Matrix6 &,
    const Policy::CameraObservations &,
    bool)>);
"""
    )
    assert candidate.returncode == 0, candidate.stderr

    legacy_capability = (
        "AdaptiveVisual" + "Partial" + "PlanCapability"
    )
    legacy_issue = "issue" + "Partial" + "Plan"
    legacy = compile_translation_unit(
        f"""
#include "adaptive_visual_policy.h"
using LegacyCapability =
  livo::{legacy_capability};
auto legacy_issue = &livo::AdaptiveVisualPolicy::{legacy_issue};
"""
    )
    assert legacy.returncode != 0


def test_candidate_capability_has_no_full_four_runtime_bundle() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_policy.h"
#include <utility>
using CandidateCapability =
  livo::AdaptiveVisualCandidatePlanCapability;
using RuntimeBundle = decltype(
  std::declval<const CandidateCapability &>().runtimeBundle());
"""
    )
    assert result.returncode != 0
    assert "runtimeBundle" in result.stderr


def test_receipt_interface_is_read_only() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_selection_authority.h"
#include <type_traits>
#include <utility>

using Receipt = livo::AdaptiveVisualSelectionReceipt;
using Token = livo::AuthoritativeLidarGenerationToken;
using Capability = livo::AdaptiveVisualSelectionCapability;
using CandidateCapability =
  livo::AdaptiveVisualCandidatePlanCapability;
using Policy = livo::AdaptiveVisualPolicy;
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Receipt &>().candidateMask())>>);
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Receipt &>().selectedMask())>>);
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Token &>().lidarPoseInformation())>>);
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Capability &>().lidarPoseInformation())>>);
static_assert(std::is_const_v<std::remove_reference_t<
  decltype(std::declval<const Receipt &>().lidarPoseInformation())>>);
static_assert(!std::is_lvalue_reference_v<
  decltype(std::declval<const Receipt &>().selectedCameraId())>);
static_assert(!std::is_lvalue_reference_v<
  decltype(std::declval<const Receipt &>().candidateBudget())>);
static_assert(!std::is_lvalue_reference_v<
  decltype(std::declval<const Receipt &>().observationGeneration())>);
static_assert(std::is_same_v<
  decltype(std::declval<const Capability &>().selectionReason()),
  livo::AdaptiveSelectionReason>);
static_assert(std::is_same_v<
  decltype(std::declval<const Receipt &>().selectionReason()),
  livo::AdaptiveSelectionReason>);
static_assert(std::is_same_v<
  decltype(std::declval<Policy &>().commitHealthObservation(
    std::declval<const CandidateCapability &>(),
    std::declval<const Receipt &>(), true)),
  livo::AdaptiveCandidateHealthCommitStatus>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_bound_candidate_evidence_is_a_private_capability_only_module() -> None:
    header = SELECTION_HEADER.read_text(encoding="utf-8")
    source = SELECTION_SOURCE.read_text(encoding="utf-8")
    assert "BoundCandidateEvidence" not in header

    module_match = re.search(
        r"class BoundCandidateEvidence final(?P<body>.*?)"
        r"\n};\n\n}  // namespace",
        source,
        re.DOTALL,
    )
    assert module_match is not None
    module = module_match.group("body")
    capability_parameter = (
        "AdaptiveVisualCandidatePlanCapability "
        "input_policy_capability"
    )
    assert (
        "static std::optional<BoundCandidateEvidence> bind("
        in module
    )
    assert capability_parameter in module
    constructor_match = re.search(
        r"explicit BoundCandidateEvidence\((?P<parameters>.*?)\)"
        r"\n  :",
        module,
        re.DOTALL,
    )
    assert constructor_match is not None
    constructor_parameters = re.sub(
        r"\s+", " ", constructor_match.group("parameters")
    ).strip()
    assert constructor_parameters == capability_parameter
    for forbidden_parameter in (
        "CameraObservationBatchReceipt",
        "CandidateMask",
        "GenerationArray",
        "ExposureArray",
        "selected_camera",
        "candidate_budget",
        "patch_budget",
    ):
        assert forbidden_parameter not in constructor_parameters
    assert (
        "std::make_shared<\n    const BoundCandidateEvidence>"
        not in source
    )
    assert "const BoundCandidateEvidence evidence;" in source
    assert (
        "capability.data_, &capability.data_->evidence"
        in source
    )

    selection_valid = re.search(
        r"bool AdaptiveVisualSelectionCapability::valid\(\) "
        r"const noexcept\n\{(?P<body>.*?)\n\}",
        source,
        re.DOTALL,
    )
    assert selection_valid is not None
    assert (
        "data_->lidar_token.issuedBy("
        in selection_valid.group("body")
    )
    assert (
        "AdaptiveVisualSelectionAuthorityIdentity{"
        in selection_valid.group("body")
    )
    assert (
        "data_->owner"
        in selection_valid.group("body")
    )


def test_zero_selection_health_commit_requires_exact_move_only_authority() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_visual_selection_authority.h"
#include <type_traits>

using Policy = livo::AdaptiveVisualPolicy;
using Candidate =
  livo::AdaptiveVisualCandidatePlanCapability;
using Permit = livo::AdaptiveVisualRuntimeAttemptPermit;
using Receipt = livo::AdaptiveVisualSelectionReceipt;
using Status = livo::AdaptiveCandidateHealthCommitStatus;
using Method = Status (Policy::*)(
  const Candidate &, Permit &&);

static_assert(std::is_same_v<
  decltype(&Policy::commitZeroSelectionHealthObservation),
  Method>);
static_assert(!std::is_copy_constructible_v<Permit>);
static_assert(!std::is_copy_assignable_v<Permit>);
static_assert(std::is_move_constructible_v<Permit>);
static_assert(std::is_invocable_r_v<
  Status, Method, Policy &, const Candidate &, Permit &&>);
static_assert(!std::is_invocable_v<
  Method, Policy &, const Candidate &, const Permit &>);
static_assert(!std::is_invocable_v<
  Method, Policy &, const Candidate &, const Receipt &>);
static_assert(!std::is_invocable_v<
  Method, Policy &, std::uint64_t,
  livo::AdaptiveVisualSelectionStatus>);
"""
    )
    assert result.returncode == 0, result.stderr
