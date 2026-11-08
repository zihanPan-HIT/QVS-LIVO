import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORY = PACKAGE_ROOT / "include"
PRODUCTION_HEADERS = (
    INCLUDE_DIRECTORY / "camera_time_transfer.h",
    INCLUDE_DIRECTORY / "photometric_calibration.h",
    INCLUDE_DIRECTORY / "photometric_noise_authority.h",
    INCLUDE_DIRECTORY / "trusted_photometric_calibration_loader.h",
    INCLUDE_DIRECTORY / "cross_camera_patch_migration.h",
    INCLUDE_DIRECTORY / "camera_batch_time_transfer_receipt.h",
    INCLUDE_DIRECTORY / "configuration_generation.h",
    INCLUDE_DIRECTORY / "shared_landmark_shadow_session.h",
)


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    configured_compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler = shutil.which(configured_compiler[0])
    assert compiler is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-sealed-receipt-contract-"
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


def assert_private_contract_rejects(source: str) -> None:
    result = compile_translation_unit(source)
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly minted a sealed receipt\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "private" in result.stderr


def test_production_headers_do_not_declare_test_peers() -> None:
    for header in PRODUCTION_HEADERS:
        assert "TestPeer" not in header.read_text(encoding="utf-8")


def test_camera_transfer_receipt_rejects_ordinary_peer_definition() -> None:
    assert_private_contract_rejects(
        """
#include "camera_time_transfer.h"
namespace livo {
class CameraTimeTransferTestPeer {
public:
  static CameraPoseTransfer mint(const CameraPoseTransferRaw & raw) {
    return CameraPoseTransfer(raw);
  }
  static void seal(CameraPoseTransferData * data) {
    data->sealSuccessful();
  }
};
}
"""
    )


def test_photometric_receipts_reject_ordinary_peer_definition() -> None:
    assert_private_contract_rejects(
        """
#include "photometric_calibration.h"
namespace livo {
class PhotometricCalibrationTestPeer {
public:
  static void seal(PhotometricCameraCalibration * calibration) {
    calibration->sealMeasuredReceipt();
  }
  static void seal(PhotometricFrameMetadata * metadata) {
    metadata->sealHardwareReceipt();
  }
};
}
"""
    )


def test_trusted_loader_does_not_expose_a_caller_supplied_trust_root() -> None:
    result = compile_translation_unit(
        """
#include "trusted_photometric_calibration_loader.h"
int main() {
  return livo::TrustedPhotometricCalibrationLoader::load(
    ".", "caller-key", "caller-domain").trustedReady() ? 0 : 1;
}
"""
    )
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly supplied its own trust root"
    )


def test_noise_capability_rejects_ordinary_construction() -> None:
    assert_private_contract_rejects(
        """
#include "photometric_noise_authority.h"
int main() {
  livo::RawIntensityDomain domain;
  livo::TrustedRawIntensityNoiseCapability capability(
    0U, 1U, domain, 1.0, "source", "laboratory_measured",
    "constant_sigma_v1", "temporal_mad_p99_v1", 64U);
  return capability.valid() ? 0 : 1;
}
"""
    )


def test_noise_bundle_rejects_ordinary_construction() -> None:
    assert_private_contract_rejects(
        """
#include "photometric_noise_authority.h"
#include <utility>
#include <vector>
int main() {
  std::vector<livo::TrustedRawIntensityNoiseCapability> capabilities;
  livo::TrustedPhotometricCalibrationBundle bundle(
    1U, std::move(capabilities));
  return bundle.valid() ? 0 : 1;
}
"""
    )


def test_migration_receipts_reject_ordinary_peer_definition() -> None:
    assert_private_contract_rejects(
        """
#include "cross_camera_patch_migration.h"
namespace livo {
class CrossCameraPatchMigrationTestPeer {
public:
  static void sealEndpoint(MigrationEndpoint * endpoint) {
    endpoint->uncertainty.sealValidationReceipt(*endpoint);
  }
  static void sealFrozen(FrozenAcceptedSourcePose * frozen) {
    frozen->sealAcceptedSnapshot();
    frozen->sealIndependent(1U, 1U);
  }
  static void sealNoise(RawIntensityNoiseModel * noise) {
    noise->sealValidationReceipt();
  }
  static void sealDepth(MigrationTarget * target) {
    target->sealDepthFrameReceipt();
  }
  static EndpointNoiseConditionerGate gate() {
    return EndpointNoiseConditionerGate(1U, 1U, 1U, 1U);
  }
};
}
"""
    )


def test_public_receipts_remain_non_default_constructible() -> None:
    result = compile_translation_unit(
        """
#include "camera_time_transfer.h"
#include "camera_batch_time_transfer_receipt.h"
#include "configuration_generation.h"
#include "photometric_calibration.h"
#include "photometric_noise_authority.h"
#include "cross_camera_patch_migration.h"
#include <type_traits>
static_assert(!std::is_default_constructible_v<
  livo::CameraPoseTransfer>);
static_assert(!std::is_default_constructible_v<
  livo::PhotometricCalibrationReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::PhotometricFrameMetadataReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::FrozenSourceAcceptanceReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::FrozenSourceIndependenceProof>);
static_assert(!std::is_default_constructible_v<
  livo::EndpointJointPoseCovariance>);
static_assert(!std::is_default_constructible_v<
  livo::CameraEndpointUncertaintyReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::RawIntensityNoiseReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::TargetDepthFrameReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::CameraBatchTimeTransferReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::ConfigurationGenerationReceipt>);
static_assert(!std::is_default_constructible_v<
  livo::TrustedRawIntensityNoiseCapability>);
static_assert(!std::is_default_constructible_v<
  livo::TrustedPhotometricCalibrationBundle>);
static_assert(!std::is_copy_assignable_v<
  livo::TrustedRawIntensityNoiseCapability>);
static_assert(!std::is_move_assignable_v<
  livo::TrustedRawIntensityNoiseCapability>);
static_assert(!std::is_copy_assignable_v<
  livo::TrustedPhotometricCalibrationBundle>);
static_assert(!std::is_move_assignable_v<
  livo::TrustedPhotometricCalibrationBundle>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_observation_store_receipt_requires_expected_landmark() -> None:
    single_argument = compile_translation_unit(
        """
#include "multi_view_landmark.h"
void inspect(
    const livo::AcceptedObservationStoreReceipt & receipt,
    const livo::ObservationSnapshot & observation) {
  (void)receipt.accepts(observation);
}
"""
    )
    assert single_argument.returncode != 0, (
        "store receipt unexpectedly accepted an observation without "
        "the caller's expected landmark identity"
    )

    authoritative = compile_translation_unit(
        """
#include "multi_view_landmark.h"
#include <type_traits>
#include <utility>
using Receipt = livo::AcceptedObservationStoreReceipt;
using Observation = livo::ObservationSnapshot;
static_assert(std::is_same_v<
  decltype(std::declval<const Receipt &>().accepts(
    std::declval<livo::LandmarkId>(),
    std::declval<const Observation &>())),
  bool>);
static_assert(std::is_same_v<
  decltype(std::declval<const Receipt &>().
    matchesObservationPayload(
      std::declval<const Observation &>())),
  bool>);
"""
    )
    assert authoritative.returncode == 0, authoritative.stderr


def test_raw_time_status_cannot_authorize_shadow_observation() -> None:
    result = compile_translation_unit(
        """
#include "shared_landmark_shadow_session.h"
int main() {
  livo::SharedLandmarkShadowObservationSample sample;
  sample.time_status = livo::CameraTimeTransferStatus::kValid;
  return sample.time_status == livo::CameraTimeTransferStatus::kValid ? 0 : 1;
}
"""
    )
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly authorized a shadow observation "
        "with a raw diagnostic enum"
    )


def test_bound_camera_entry_rejects_ordinary_rewrapping() -> None:
    assert_private_contract_rejects(
        """
#include "camera_batch_time_transfer_receipt.h"
int main() {
  livo::CameraPoseTransferData raw;
  livo::CameraBatchTimeTransferEntry entry(
    1U, 10.0, 10.0, 1U,
    livo::kCameraTimeTransferActiveJacobianRevision,
    livo::kCameraTimeTransferActiveStateLayoutId,
    raw,
    std::shared_ptr<const void>{});
  return entry.has_value() ? 0 : 1;
}
"""
    )
