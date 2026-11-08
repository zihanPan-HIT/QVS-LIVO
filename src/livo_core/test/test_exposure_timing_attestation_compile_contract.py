from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = Path(
    os.environ.get(
        "LIVO_EXPOSURE_TIMING_BUILD_ROOT",
        PACKAGE_ROOT.parents[1] / "build" / PACKAGE_ROOT.name,
    )
)
HEADER = PACKAGE_ROOT / "include" / "exposure_timing_attestation.h"
SOURCE = PACKAGE_ROOT / "src" / "exposure_timing_attestation.cpp"
CMAKE_FILE = PACKAGE_ROOT / "CMakeLists.txt"
PRODUCTION_FLAGS = (
    BUILD_ROOT
    / "CMakeFiles"
    / "exposure_timing_attestation.dir"
    / "flags.make"
)
FAILURE_FLAGS = (
    BUILD_ROOT
    / "CMakeFiles"
    / "exposure_timing_attestation_failure_injection_test.dir"
    / "flags.make"
)
BEHAVIOR_FLAGS = (
    BUILD_ROOT
    / "CMakeFiles"
    / "test_exposure_timing_attestation.dir"
    / "flags.make"
)
PRODUCTION_LIBRARY = BUILD_ROOT / "libexposure_timing_attestation.a"
FAILURE_LIBRARY = (
    BUILD_ROOT
    / "libexposure_timing_attestation_failure_injection_test.a"
)
BEHAVIOR_TEST = BUILD_ROOT / "test_exposure_timing_attestation"


def cmake_flags(path: Path, variable: str) -> list[str]:
    prefix = f"{variable} = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return shlex.split(line.removeprefix(prefix))
    raise AssertionError(f"{variable} missing from {path}")


def cmake_cache_path(variable: str) -> str:
    cache = BUILD_ROOT / "CMakeCache.txt"
    prefix = f"{variable}:"
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            _, separator, value = line.partition("=")
            assert separator
            return value
    raise AssertionError(f"{variable} missing from {cache}")


def compile_translation_unit(
    source: str,
    extra_flags: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("c++")
    assert compiler is not None
    assert PRODUCTION_FLAGS.is_file()
    with tempfile.TemporaryDirectory(
        prefix="livo-exposure-timing-contract-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        source_path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                compiler,
                *cmake_flags(PRODUCTION_FLAGS, "CXX_DEFINES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_INCLUDES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_FLAGS"),
                *extra_flags,
                "-fsyntax-only",
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def link_and_run_translation_unit(
    source: str,
    library: Path,
) -> tuple[
    subprocess.CompletedProcess[str],
    subprocess.CompletedProcess[str] | None,
]:
    compiler = shutil.which("c++")
    assert compiler is not None
    assert PRODUCTION_FLAGS.is_file()
    assert library.is_file()
    with tempfile.TemporaryDirectory(
        prefix="livo-exposure-timing-link-contract-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        executable_path = Path(directory) / "consumer"
        source_path.write_text(source, encoding="utf-8")
        linked = subprocess.run(
            [
                compiler,
                *cmake_flags(PRODUCTION_FLAGS, "CXX_DEFINES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_INCLUDES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_FLAGS"),
                str(source_path),
                str(library),
                "-o",
                str(executable_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if linked.returncode != 0:
            return linked, None
        executed = subprocess.run(
            [str(executable_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        return linked, executed


def defined_symbols(binary: Path) -> str:
    nm = shutil.which("nm")
    assert nm is not None
    assert binary.is_file(), f"missing symbol input: {binary}"
    result = subprocess.run(
        [nm, "-C", "--defined-only", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def global_defined_symbols(binary: Path) -> str:
    nm = shutil.which("nm")
    assert nm is not None
    assert binary.is_file(), f"missing symbol input: {binary}"
    result = subprocess.run(
        [nm, "-g", "-C", "--defined-only", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def undefined_symbols(binary: Path) -> str:
    nm = shutil.which("nm")
    assert nm is not None
    assert binary.is_file(), f"missing symbol input: {binary}"
    result = subprocess.run(
        [nm, "-C", "--undefined-only", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def add_library_arguments(target: str) -> list[str]:
    declaration = re.search(
        rf"add_library\s*\(\s*{re.escape(target)}\b([^)]*)\)",
        CMAKE_FILE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert declaration is not None, f"missing add_library({target})"
    return declaration.group(1).split()


def test_opaque_tokens_and_solver_surface_have_exact_value_types() -> None:
    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

using Trusted = livo::TrustedExposureTimingCalibration;
using Source = livo::ExposureTimingSourceAuthority;
using SourceIdentity = livo::ExposureTimingSourceAuthorityIdentity;
using FrameReceipt = livo::ExposureFrameTimingReceipt;
using FrameIdentity = livo::ExposureFrameTimingIdentity;
using Effective = livo::EffectiveExposureTime;
using Audit = livo::ExposureTimingAuditView;
using Authority = livo::ExposureTimingAuthority;
using AuthorityIdentity = livo::ExposureTimingAuthorityIdentity;
using PayloadIdentity = livo::ExposureTimingAttestationIdentity;
using Attestation = livo::ExposureTimingAttestation;

static_assert(!std::is_default_constructible_v<Trusted>);
static_assert(!std::is_default_constructible_v<SourceIdentity>);
static_assert(!std::is_default_constructible_v<FrameReceipt>);
static_assert(!std::is_default_constructible_v<FrameIdentity>);
static_assert(!std::is_default_constructible_v<Effective>);
static_assert(!std::is_default_constructible_v<Audit>);
static_assert(!std::is_default_constructible_v<AuthorityIdentity>);
static_assert(!std::is_default_constructible_v<PayloadIdentity>);
static_assert(!std::is_default_constructible_v<Attestation>);

static_assert(std::is_copy_constructible_v<Trusted>);
static_assert(std::is_nothrow_move_constructible_v<Trusted>);
static_assert(std::is_copy_constructible_v<SourceIdentity>);
static_assert(std::is_nothrow_move_constructible_v<SourceIdentity>);
static_assert(std::is_copy_constructible_v<FrameReceipt>);
static_assert(std::is_nothrow_move_constructible_v<FrameReceipt>);
static_assert(std::is_copy_constructible_v<FrameIdentity>);
static_assert(std::is_nothrow_move_constructible_v<FrameIdentity>);
static_assert(std::is_copy_constructible_v<Effective>);
static_assert(std::is_nothrow_move_constructible_v<Effective>);
static_assert(std::is_copy_constructible_v<Audit>);
static_assert(std::is_nothrow_move_constructible_v<Audit>);
static_assert(std::is_copy_constructible_v<Attestation>);
static_assert(std::is_nothrow_move_constructible_v<Attestation>);
static_assert(!std::is_copy_constructible_v<Source>);
static_assert(!std::is_move_constructible_v<Source>);
static_assert(!std::is_copy_constructible_v<Authority>);
static_assert(!std::is_move_constructible_v<Authority>);

static_assert(std::is_same_v<
  decltype(std::declval<const Attestation &>().cameraId()),
  std::uint8_t>);
static_assert(std::is_same_v<
  decltype(std::declval<const Attestation &>().sourceSequence()),
  std::uint64_t>);
static_assert(std::is_same_v<
  decltype(
    std::declval<const Attestation &>().sourceAuthorityIdentity()),
  SourceIdentity>);
static_assert(std::is_same_v<
  decltype(
    std::declval<const Attestation &>().sourceFrameIdentity()),
  FrameIdentity>);
static_assert(std::is_same_v<
  decltype(
    std::declval<const Attestation &>().effectiveExposureTime()),
  std::optional<Effective>>);
static_assert(std::is_same_v<
  decltype(std::declval<const Attestation &>().audit()),
  std::optional<Audit>>);
static_assert(std::is_same_v<
  decltype(std::declval<const Audit &>().sourceId()),
  const std::string &>);
static_assert(std::is_same_v<
  decltype(std::declval<const Audit &>().exposureDurationSec()),
  std::optional<double>>);
static_assert(std::is_same_v<
  decltype(std::declval<const Audit &>().canonicalPayloadSha256()),
  livo::ExposureTimingArtifactDigest>);
static_assert(noexcept(std::declval<const Trusted &>().valid()));
"""
    )
    assert result.returncode == 0, result.stderr


def test_locking_operations_are_not_falsely_noexcept() -> None:
    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"
#include <cstdint>
#include <optional>
#include <utility>

using Source = livo::ExposureTimingSourceAuthority;
using Authority = livo::ExposureTimingAuthority;
using Receipt = livo::ExposureFrameTimingReceipt;

static_assert(!noexcept(
  std::declval<Source &>().issue(
    std::declval<std::uint8_t>(),
    std::declval<double>(),
    std::declval<std::optional<double>>())));
static_assert(!noexcept(
  std::declval<Authority &>().attest(
    std::declval<const Receipt &>())));
static_assert(noexcept(
  std::declval<const Source &>().configuredFor(
    std::declval<const
      livo::ExposureTimingSourceBinding &>())));
static_assert(noexcept(
  std::declval<const Authority &>().boundTo(
    std::declval<const
      livo::ExposureTimingSourceAuthorityIdentity &>())));
"""
    )
    assert result.returncode == 0, result.stderr


def test_failure_and_production_token_layouts_are_identical() -> None:
    layout_contract = """
#include "exposure_timing_attestation.h"
#include <memory>

using Shared = std::shared_ptr<const void>;

static_assert(
  sizeof(livo::TrustedExposureTimingCalibration) == sizeof(Shared));
static_assert(
  alignof(livo::TrustedExposureTimingCalibration) == alignof(Shared));
static_assert(
  sizeof(livo::ExposureTimingSourceAuthorityIdentity) ==
  sizeof(Shared));
static_assert(
  sizeof(livo::ExposureFrameTimingReceipt) == sizeof(Shared));
static_assert(
  sizeof(livo::ExposureFrameTimingIdentity) == sizeof(Shared));
static_assert(
  sizeof(livo::ExposureTimingAuditView) == sizeof(Shared));
static_assert(
  sizeof(livo::ExposureTimingAuthorityIdentity) == sizeof(Shared));
static_assert(
  sizeof(livo::ExposureTimingAttestationIdentity) == sizeof(Shared));
static_assert(
  sizeof(livo::ExposureTimingAttestation) == sizeof(Shared));
static_assert(sizeof(livo::EffectiveExposureTime) == sizeof(double));
"""
    production = compile_translation_unit(layout_contract)
    failure = compile_translation_unit(
        layout_contract,
        (
            "-DLIVO_EXPOSURE_TIMING_ATTESTATION_"
            "FAILURE_INJECTION_TESTING=1",
        ),
    )
    assert production.returncode == 0, production.stderr
    assert failure.returncode == 0, failure.stderr


def test_private_token_constructors_cannot_be_called_by_consumers() -> None:
    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"

void forge() {
  livo::TrustedExposureTimingCalibration trusted;
  livo::ExposureTimingSourceAuthorityIdentity source_identity;
  livo::ExposureFrameTimingReceipt receipt;
  livo::EffectiveExposureTime effective;
  (void)trusted;
  (void)source_identity;
  (void)receipt;
  (void)effective;
}
"""
    )
    assert result.returncode != 0
    assert "no matching" in result.stderr or "private" in result.stderr

    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"
#include <memory>

void forge() {
  std::shared_ptr<const livo::ExposureTimingAuditView::Data> data;
  livo::ExposureTimingAuditView audit(data);
}
"""
    )
    assert result.returncode != 0
    assert "private" in result.stderr


def test_public_document_cannot_self_sign_or_mint_trust() -> None:
    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"

void self_sign(
  const livo::ExposureTimingCalibrationDocument & document)
{
  auto created =
    livo::ExposureTimingAuthority::create(document);
  (void)created;
}
"""
    )
    assert result.returncode != 0

    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"

void mint(
  const livo::ExposureTimingCalibrationDocument & document)
{
  auto trusted =
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document);
  (void)trusted;
}
"""
    )
    assert result.returncode != 0

    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"
#include <memory>

namespace livo::test
{
class ExposureTimingCalibrationTestAccess
{
public:
  static TrustedExposureTimingCalibration forge()
  {
    std::shared_ptr<
      const TrustedExposureTimingCalibration::Data> data;
    return TrustedExposureTimingCalibration{data};
  }
};
}  // namespace livo::test
"""
    )
    assert result.returncode != 0
    assert "private" in result.stderr


def test_production_fixture_mint_returns_typed_unavailable() -> None:
    consumer = """
#include "exposure_timing_attestation.h"
#include <utility>

int main()
{
  static_assert(noexcept(
    livo::ExposureTimingCalibrationLoader::mintFixture(
      std::declval<const
        livo::ExposureTimingCalibrationDocument &>())));

  livo::ExposureTimingCalibrationDocument document;
  document.canonical_payload = "must-not-be-parsed";
  const auto result =
    livo::ExposureTimingCalibrationLoader::mintFixture(document);
  return
    result.status ==
      livo::ExposureTimingCalibrationStatus::
      FIXTURE_MINT_UNAVAILABLE &&
    !result.calibration.has_value() &&
    !result.ready() ?
    0 : 1;
}
"""
    production_link, production_run = link_and_run_translation_unit(
        consumer, PRODUCTION_LIBRARY
    )
    assert production_link.returncode == 0, production_link.stderr
    assert production_run is not None
    assert production_run.returncode == 0, production_run.stderr


def test_failure_archive_official_fixture_mint_succeeds() -> None:
    consumer = """
#include "exposure_timing_attestation.h"
#include <cstdint>
#include <string>

int main()
{
  livo::ExposureTimingCalibrationDocument document;
  document.mode =
    livo::ExposureTimingAuthorityMode::SIMULATION_TEST;
  document.provenance =
    livo::ExposureTimingProvenanceKind::TEST_FIXTURE;
  document.canonical_payload = "{fixture:true}";
  document.canonical_payload_sha256.fill(0xA5U);
  document.artifact_id = "official-fixture";
  document.calibration_revision = 1U;
  for (std::uint8_t camera_id = 0U;
    camera_id < livo::kExposureTimingCameraCount;
    ++camera_id)
  {
    auto & camera = document.cameras[camera_id];
    camera.source_binding.camera_id = camera_id;
    camera.source_binding.source_id =
      "source-" + std::to_string(camera_id);
    camera.source_binding.topic =
      "/camera/" + std::to_string(camera_id);
    camera.source_binding.sensor_serial =
      "serial-" + std::to_string(camera_id);
    camera.source_binding.frame_id =
      "frame-" + std::to_string(camera_id);
    camera.timestamp_convention =
      livo::ExposureTimestampConvention::MIDPOINT;
    camera.duration_source = "fixture-duration";
  }

  const auto result =
    livo::ExposureTimingCalibrationLoader::mintFixture(document);
  return
    result.status ==
      livo::ExposureTimingCalibrationStatus::READY &&
    result.calibration.has_value() &&
    result.calibration->valid() &&
    result.ready() ?
    0 : 1;
}
"""
    failure_link, failure_run = link_and_run_translation_unit(
        consumer, FAILURE_LIBRARY
    )
    assert failure_link.returncode == 0, failure_link.stderr
    assert failure_run is not None
    assert failure_run.returncode == 0, failure_run.stderr


def test_consumer_defined_loader_is_rejected_by_both_archives() -> None:
    malicious_consumer = """
#include "exposure_timing_attestation.h"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace livo
{

struct TrustedExposureTimingCalibration::Data final
{
  Data(
    const ExposureTimingCalibrationDocument & source,
    const void * source_issuer_identity)
  : issuer_identity(source_issuer_identity),
    document(source)
  {
    for (const auto & camera : source.cameras)
    {
      const auto camera_id = camera.source_binding.camera_id;
      document.cameras[camera_id] = camera;
      total_offsets_sec[camera_id] =
        source.global_offset_sec +
        camera.per_camera_offset_sec;
    }
  }

  const void * issuer_identity;
  ExposureTimingCalibrationDocument document;
  std::array<double, kExposureTimingCameraCount>
    total_offsets_sec{};
};

namespace
{
const TrustedExposureTimingCalibration *
  genuine_calibration_to_launder{nullptr};
}

TrustedExposureTimingCalibrationResult
ExposureTimingCalibrationLoader::mintFixture(
  const ExposureTimingCalibrationDocument & document) noexcept
{
  if (
    genuine_calibration_to_launder == nullptr ||
    !genuine_calibration_to_launder->data_)
  {
    return {
      ExposureTimingCalibrationStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  try
  {
    auto data =
      std::make_shared<const
      TrustedExposureTimingCalibration::Data>(
      document,
      genuine_calibration_to_launder->
        data_->issuer_identity);
    return {
      ExposureTimingCalibrationStatus::READY,
      TrustedExposureTimingCalibration{std::move(data)}};
  }
  catch (...)
  {
    return {
      ExposureTimingCalibrationStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
}

}  // namespace livo

int main()
{
  livo::ExposureTimingSourceAuthorityConfiguration configuration;
  for (std::uint8_t camera_id = 0U;
    camera_id < livo::kExposureTimingCameraCount;
    ++camera_id)
  {
    auto & binding = configuration.cameras[camera_id];
    binding.camera_id = camera_id;
    binding.source_id =
      "source-" + std::to_string(camera_id);
    binding.topic =
      "/camera/" + std::to_string(camera_id);
    binding.sensor_serial =
      "serial-" + std::to_string(camera_id);
    binding.frame_id =
      "frame-" + std::to_string(camera_id);
  }

  const auto source =
    livo::ExposureTimingSourceAuthority::create(
    configuration);
  return
    source.status ==
      livo::ExposureTimingSourceAuthorityCreateStatus::READY ?
    0 : 1;
}
"""
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        linked, executed = link_and_run_translation_unit(
            malicious_consumer, library
        )
        assert linked.returncode != 0, (
            f"consumer loader override linked against {library}"
        )
        assert executed is None
        assert "multiple definition" in linked.stderr.lower()
        assert "mintfixture" in linked.stderr.lower()


def test_raw_attest_input_and_public_snapshot_claim_are_absent() -> None:
    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"

void bypass(
  livo::ExposureTimingAuthority & authority,
  const livo::ExposureTimingAttestation & attestation)
{
  livo::ExposureTimingAttestInput raw;
  (void)authority.attest(raw);
  (void)authority.claimForSnapshot(attestation);
}
"""
    )
    assert result.returncode != 0

    header = HEADER.read_text(encoding="utf-8")
    for forbidden in (
        "ExposureTimingAttestInput",
        "ExposureTimingClaimStatus",
        "ExposureTimingClaimResult",
        "claimForSnapshot",
        "claimedForSnapshot",
    ):
        assert forbidden not in header


def test_attestation_has_no_parallel_raw_or_audit_getters() -> None:
    result = compile_translation_unit(
        """
#include "exposure_timing_attestation.h"
#include <type_traits>
#include <utility>

using Attestation = livo::ExposureTimingAttestation;

#define DECLARE_HAS_METHOD(Trait, Method)                         \
template<typename T, typename = void>                            \
struct Trait : std::false_type {};                              \
template<typename T>                                             \
struct Trait<T, std::void_t<                                    \
  decltype(std::declval<const T &>().Method())>>                 \
: std::true_type {}

DECLARE_HAS_METHOD(HasSourceId, sourceId);
DECLARE_HAS_METHOD(HasTopic, topic);
DECLARE_HAS_METHOD(HasSensorSerial, sensorSerial);
DECLARE_HAS_METHOD(HasFrameId, frameId);
DECLARE_HAS_METHOD(HasConvention, timestampConvention);
DECLARE_HAS_METHOD(HasRawHeader, rawHeaderTimestampSec);
DECLARE_HAS_METHOD(HasDuration, exposureDurationSec);
DECLARE_HAS_METHOD(HasDurationSource, durationSource);
DECLARE_HAS_METHOD(HasMidpoint, midpointTimestampSec);
DECLARE_HAS_METHOD(HasGlobalOffset, globalOffsetSec);
DECLARE_HAS_METHOD(HasCameraOffset, perCameraOffsetSec);
DECLARE_HAS_METHOD(HasTotalOffset, totalOffsetSec);
DECLARE_HAS_METHOD(HasRawEffective, effectiveTimestampSec);
DECLARE_HAS_METHOD(HasMode, authorityMode);
DECLARE_HAS_METHOD(HasProvenance, provenanceKind);
DECLARE_HAS_METHOD(HasRevision, calibrationRevision);
DECLARE_HAS_METHOD(HasArtifact, artifactId);
DECLARE_HAS_METHOD(HasPayload, canonicalPayload);
DECLARE_HAS_METHOD(HasDigest, canonicalPayloadSha256);
DECLARE_HAS_METHOD(HasClaim, claim);
DECLARE_HAS_METHOD(HasMutableData, data);

static_assert(!HasSourceId<Attestation>::value);
static_assert(!HasTopic<Attestation>::value);
static_assert(!HasSensorSerial<Attestation>::value);
static_assert(!HasFrameId<Attestation>::value);
static_assert(!HasConvention<Attestation>::value);
static_assert(!HasRawHeader<Attestation>::value);
static_assert(!HasDuration<Attestation>::value);
static_assert(!HasDurationSource<Attestation>::value);
static_assert(!HasMidpoint<Attestation>::value);
static_assert(!HasGlobalOffset<Attestation>::value);
static_assert(!HasCameraOffset<Attestation>::value);
static_assert(!HasTotalOffset<Attestation>::value);
static_assert(!HasRawEffective<Attestation>::value);
static_assert(!HasMode<Attestation>::value);
static_assert(!HasProvenance<Attestation>::value);
static_assert(!HasRevision<Attestation>::value);
static_assert(!HasArtifact<Attestation>::value);
static_assert(!HasPayload<Attestation>::value);
static_assert(!HasDigest<Attestation>::value);
static_assert(!HasClaim<Attestation>::value);
static_assert(!HasMutableData<Attestation>::value);
"""
    )
    assert result.returncode == 0, result.stderr


def test_public_header_is_dependency_free_and_test_layout_neutral() -> None:
    header = HEADER.read_text(encoding="utf-8")
    for forbidden in (
        "opencv",
        "cv::",
        "rclcpp",
        "sensor_msgs",
        "ros::",
        "issuer_identity",
        "issuer_seal",
        "LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING",
    ):
        assert forbidden not in header
    assert re.search(r"\b(?:void|char|unsigned char)\s*\*", header) is None
    assert "friend class test::" not in header
    for hook in (
        "failNextExposureTimingCalibrationAllocation",
        "failNextExposureTimingSourceAuthorityFactoryAllocation",
        "failNextExposureFrameTimingReceiptAllocation",
        "failNextExposureTimingAuthorityFactoryAllocation",
        "failNextExposureTimingAttestationAllocation",
        "exhaustNextExposureTimingSourceSequence",
        "exhaustNextExposureTimingAttestationSequence",
        "pauseNextExposureFrameTimingReceiptBeforeRevalidation",
        "waitForExposureFrameTimingReceiptPausedBeforeRevalidation",
        "releaseExposureFrameTimingReceiptBeforeRevalidation",
        "pauseNextExposureTimingAttestationBeforeRevalidation",
        "waitForExposureTimingAttestationPausedBeforeRevalidation",
        "releaseExposureTimingAttestationBeforeRevalidation",
    ):
        assert hook not in header


def test_archive_issuer_seals_have_internal_linkage() -> None:
    seal_names = (
        "trusted_calibration_issuer_seal",
        "timing_source_authority_issuer_seal",
    )
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        all_symbols = defined_symbols(library)
        global_symbols = global_defined_symbols(library)
        for seal_name in seal_names:
            assert all_symbols.count(seal_name) == 1
            assert "(anonymous namespace)" in next(
                line
                for line in all_symbols.splitlines()
                if seal_name in line
            )
            assert seal_name not in global_symbols


def test_source_relationship_proofs_have_strong_archive_definitions(
) -> None:
    expected_counts = {
        "ExposureTimingSourceAuthority::configuredFor": 1,
        "ExposureTimingAuthority::boundTo": 1,
    }
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        symbols = global_defined_symbols(library).splitlines()
        for symbol, expected_count in expected_counts.items():
            definitions = [
                line for line in symbols if symbol in line
            ]
            assert len(definitions) == expected_count, (
                library,
                symbol,
                definitions,
            )
            assert all(" T " in line for line in definitions)


def test_consumer_cannot_override_exact_source_binding_proof() -> None:
    malicious_consumers = (
        (
            "configuredfor",
            """
#include "exposure_timing_attestation.h"

namespace livo
{
bool ExposureTimingSourceAuthority::configuredFor(
  const ExposureTimingSourceBinding &) const noexcept
{
  return true;
}
}  // namespace livo

int main()
{
  livo::ExposureTimingSourceAuthorityConfiguration config;
  (void)livo::ExposureTimingSourceAuthority::create(config);
  return 0;
}
""",
        ),
        (
            "boundto",
            """
#include "exposure_timing_attestation.h"

namespace livo
{
bool ExposureTimingAuthority::boundTo(
  const ExposureTimingSourceAuthorityIdentity &) const noexcept
{
  return true;
}
}  // namespace livo

int main()
{
  livo::ExposureTimingSourceAuthorityConfiguration config;
  (void)livo::ExposureTimingSourceAuthority::create(config);
  return 0;
}
""",
        ),
    )
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        for symbol, malicious_consumer in malicious_consumers:
            linked, executed = link_and_run_translation_unit(
                malicious_consumer, library
            )
            assert linked.returncode != 0
            assert executed is None
            assert "multiple definition" in linked.stderr.lower()
            assert symbol in linked.stderr.lower()


def test_fixture_mint_has_one_strong_definition_per_archive() -> None:
    symbol = "ExposureTimingCalibrationLoader::mintFixture"
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        definitions = [
            line
            for line in global_defined_symbols(library).splitlines()
            if symbol in line
        ]
        assert len(definitions) == 1, (
            f"expected one global definition in {library}: "
            f"{definitions}"
        )
        assert " T " in definitions[0], definitions[0]

    assert symbol not in undefined_symbols(PRODUCTION_LIBRARY)


def test_failure_hooks_are_isolated_from_production() -> None:
    hooks = (
        "ExposureTimingCalibrationTestAccess::mintFixture",
        "failNextExposureTimingCalibrationAllocation",
        "failNextExposureTimingSourceAuthorityFactoryAllocation",
        "failNextExposureFrameTimingReceiptAllocation",
        "failNextExposureTimingAuthorityFactoryAllocation",
        "failNextExposureTimingAttestationAllocation",
        "exhaustNextExposureTimingSourceSequence",
        "exhaustNextExposureTimingAttestationSequence",
        "pauseNextExposureFrameTimingReceiptBeforeRevalidation",
        "waitForExposureFrameTimingReceiptPausedBeforeRevalidation",
        "releaseExposureFrameTimingReceiptBeforeRevalidation",
        "pauseNextExposureTimingAttestationBeforeRevalidation",
        "waitForExposureTimingAttestationPausedBeforeRevalidation",
        "releaseExposureTimingAttestationBeforeRevalidation",
    )
    production_symbols = defined_symbols(PRODUCTION_LIBRARY)
    failure_symbols = defined_symbols(FAILURE_LIBRARY)
    test_symbols = defined_symbols(BEHAVIOR_TEST)
    for hook in hooks:
        assert hook not in production_symbols
        assert hook in failure_symbols
        assert hook in test_symbols

    production_defines = cmake_flags(
        PRODUCTION_FLAGS, "CXX_DEFINES"
    )
    failure_defines = cmake_flags(
        FAILURE_FLAGS, "CXX_DEFINES"
    )
    behavior_defines = cmake_flags(
        BEHAVIOR_FLAGS, "CXX_DEFINES"
    )
    assert not any(
        "EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION"
        in item
        for item in production_defines
    )
    assert any(
        "EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION"
        in item
        for item in failure_defines
    )
    assert not any(
        "EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION"
        in item
        for item in behavior_defines
    )
    assert (
        "exposure_timing_attestation_failure_injection_test_support.h"
        not in SOURCE.read_text(encoding="utf-8")
    )


def test_module_objects_are_werror_clean_targets() -> None:
    for flags_path in (PRODUCTION_FLAGS, FAILURE_FLAGS):
        flags = cmake_flags(flags_path, "CXX_FLAGS")
        for required in (
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
        ):
            assert required in flags


def test_security_archives_remain_static_with_build_shared_libs_on(
) -> None:
    targets = (
        "exposure_timing_attestation",
        "exposure_timing_attestation_failure_injection_test",
    )
    for target in targets:
        arguments = add_library_arguments(target)
        assert arguments[0] == "STATIC", (
            f"{target} must explicitly select STATIC: {arguments}"
        )

    if shutil.disk_usage(PACKAGE_ROOT).free < 20 * 1024**3:
        return

    cmake = shutil.which("cmake")
    assert cmake is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-exposure-timing-shared-on-contract-"
    ) as directory:
        configured = subprocess.run(
            [
                cmake,
                "-S",
                str(PACKAGE_ROOT),
                "-B",
                directory,
                "-DBUILD_TESTING=ON",
                "-DBUILD_SHARED_LIBS=ON",
                "-DCMAKE_BUILD_TYPE=Release",
                "-Dlivox_ros_driver2_DIR="
                f"{cmake_cache_path('livox_ros_driver2_DIR')}",
                "-Dvikit_common_DIR="
                f"{cmake_cache_path('vikit_common_DIR')}",
                "-Dvikit_ros_DIR="
                f"{cmake_cache_path('vikit_ros_DIR')}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert configured.returncode == 0, configured.stderr
        built = subprocess.run(
            [
                cmake,
                "--build",
                directory,
                "--target",
                *targets,
                "--parallel",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert built.returncode == 0, built.stderr
        for target in targets:
            assert (
                Path(directory) / f"lib{target}.a"
            ).is_file()
            assert not tuple(
                Path(directory).glob(f"lib{target}.so*")
            )


def test_test_variant_does_not_colink_production_variant() -> None:
    link_file = (
        BUILD_ROOT
        / "CMakeFiles"
        / "test_exposure_timing_attestation.dir"
        / "link.txt"
    )
    link = shlex.split(link_file.read_text(encoding="utf-8"))
    names = {Path(item).name for item in link}
    assert (
        "libexposure_timing_attestation_failure_injection_test.a"
        in names
    )
    assert "libexposure_timing_attestation.a" not in names


def test_standalone_module_is_not_linked_to_runtime_or_existing_authorities(
) -> None:
    forbidden_dependencies = (
        "libadaptive_camera_snapshot_queue",
        "libadaptive_visual_selection_authority",
        "libcamera_observation_batch_authority",
        "liblaser_mapping",
        "librclcpp",
        "libopencv",
    )
    production_undefined = undefined_symbols(PRODUCTION_LIBRARY).lower()
    for dependency in ("rclcpp", "opencv", "cv::mat"):
        assert dependency not in production_undefined

    target_names = (
        "livo_mapping",
        "adaptive_camera_snapshot_queue",
        "adaptive_visual_selection_authority",
        "test_adaptive_camera_snapshot_queue",
        "test_adaptive_visual_selection_capability",
    )
    timing_libraries = (
        "libexposure_timing_attestation.a",
        "libexposure_timing_attestation_failure_injection_test.a",
    )
    for target in target_names:
        link_file = (
            BUILD_ROOT / "CMakeFiles" / f"{target}.dir" / "link.txt"
        )
        assert link_file.is_file(), f"missing link graph: {link_file}"
        link = link_file.read_text(encoding="utf-8")
        for timing_library in timing_libraries:
            assert timing_library not in link

    behavior_link = (
        BUILD_ROOT
        / "CMakeFiles"
        / "test_exposure_timing_attestation.dir"
        / "link.txt"
    ).read_text(encoding="utf-8").lower()
    for dependency in forbidden_dependencies:
        assert dependency not in behavior_link
