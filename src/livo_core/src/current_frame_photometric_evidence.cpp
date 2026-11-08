#include "current_frame_photometric_evidence.h"

#include "camera_observation.h"
#include "camera_observation_batch_authority.h"
#include "reference_only_migration_runtime_provenance.h"

#include <cstddef>
#include <utility>

namespace livo
{
namespace
{

enum class TrustedCalibrationValidationStatus
{
  kUnavailable,
  kRevisionMismatch,
  kNoiseUnavailable,
  kReady
};

}  // namespace

struct CurrentFramePhotometricEvidenceEvaluator::Implementation
{
  Implementation(
    CameraObservationBatchAuthorityIdentity
      input_batch_authority,
    ReferenceOnlyMigrationRuntimePublisherIdentity
      input_runtime_authority,
    TrustedPhotometricCalibrationLoadResult
      input_trusted_calibration,
    const CurrentFramePhotometricEvidenceConfig input_config)
  : expected_batch_authority(std::move(input_batch_authority)),
    expected_runtime_authority(std::move(input_runtime_authority)),
    trusted_calibration(std::move(input_trusted_calibration)),
    config(input_config)
  {
    authority_configuration_valid =
      expected_batch_authority.valid() &&
      expected_runtime_authority.valid() &&
      expected_runtime_authority.bindsBatchAuthority(
        expected_batch_authority);
    if (!trusted_calibration.trustedCalibrationReady())
    {
      return;
    }

    calibration_revision =
      trusted_calibration.calibration_revision;
    const PhotometricCalibration & calibration =
      *trusted_calibration.calibration;
    for (std::size_t camera_id = 0U;
         camera_id < 4U; ++camera_id)
    {
      if (
        calibration.cameraCalibrationStatus(camera_id) !=
          PhotometricStatus::kValid ||
        calibration.cameraCalibrationRevision(camera_id) !=
          calibration_revision)
      {
        calibration_validation =
          TrustedCalibrationValidationStatus::
          kRevisionMismatch;
        return;
      }
    }
    if (!trusted_calibration.trusted_noise_bundle.has_value())
    {
      calibration_validation =
        TrustedCalibrationValidationStatus::
        kNoiseUnavailable;
      return;
    }
    const TrustedPhotometricCalibrationBundle & bundle =
      *trusted_calibration.trusted_noise_bundle;
    if (bundle.calibrationRevision() != calibration_revision)
    {
      calibration_validation =
        TrustedCalibrationValidationStatus::
        kRevisionMismatch;
      return;
    }
    if (!bundle.valid() || !bundle.accepts(calibration))
    {
      calibration_validation =
        TrustedCalibrationValidationStatus::
        kNoiseUnavailable;
      return;
    }
    calibration_validation =
      TrustedCalibrationValidationStatus::kReady;
  }

  const CameraObservationBatchAuthorityIdentity
    expected_batch_authority;
  const ReferenceOnlyMigrationRuntimePublisherIdentity
    expected_runtime_authority;
  const TrustedPhotometricCalibrationLoadResult
    trusted_calibration;
  const CurrentFramePhotometricEvidenceConfig config;
  bool authority_configuration_valid{false};
  TrustedCalibrationValidationStatus calibration_validation{
    TrustedCalibrationValidationStatus::kUnavailable};
  std::uint64_t calibration_revision{0U};
};

bool ObservedRawIntensityDomain::valid() const noexcept
{
  return
    bit_depth == 8U &&
    unit == RawIntensityUnit::kCanonicalGrayCodeValue &&
    canonicalization !=
      ObservedRawIntensityCanonicalization::kInvalid &&
    image_width > 0 &&
    image_height > 0;
}

bool operator==(
  const ObservedRawIntensityDomain & lhs,
  const ObservedRawIntensityDomain & rhs) noexcept
{
  return
    lhs.bit_depth == rhs.bit_depth &&
    lhs.unit == rhs.unit &&
    lhs.canonicalization == rhs.canonicalization &&
    lhs.image_width == rhs.image_width &&
    lhs.image_height == rhs.image_height &&
    lhs.canonical_gray_owner_attested ==
      rhs.canonical_gray_owner_attested &&
    lhs.native_resolution_attested ==
      rhs.native_resolution_attested;
}

bool operator!=(
  const ObservedRawIntensityDomain & lhs,
  const ObservedRawIntensityDomain & rhs) noexcept
{
  return !(lhs == rhs);
}

CurrentFramePhotometricEvidencePath
CurrentFramePhotometricEvidenceDecision::path() const noexcept
{
  return path_;
}

CurrentFramePhotometricEvidenceStatus
CurrentFramePhotometricEvidenceDecision::status() const noexcept
{
  return status_;
}

CurrentFramePhotometricEvidenceStatus
CurrentFramePhotometricEvidenceDecision::
strictRejectionStatus() const noexcept
{
  return strict_rejection_status_;
}

std::uint8_t
CurrentFramePhotometricEvidenceDecision::cameraId() const noexcept
{
  return camera_id_;
}

std::uint64_t CurrentFramePhotometricEvidenceDecision::
cameraBatchSequence() const noexcept
{
  return camera_batch_sequence_;
}

std::uint64_t CurrentFramePhotometricEvidenceDecision::
runtimePublicationSequence() const noexcept
{
  return runtime_publication_sequence_;
}

std::uint64_t CurrentFramePhotometricEvidenceDecision::
observationGeneration() const noexcept
{
  return observation_generation_;
}

std::uint64_t CurrentFramePhotometricEvidenceDecision::
imageGeneration() const noexcept
{
  return image_generation_;
}

double CurrentFramePhotometricEvidenceDecision::
exposureTimestampSec() const noexcept
{
  return exposure_timestamp_sec_;
}

std::uint64_t CurrentFramePhotometricEvidenceDecision::
calibrationRevision() const noexcept
{
  return calibration_revision_;
}

const ObservedRawIntensityDomain &
CurrentFramePhotometricEvidenceDecision::
observedDomain() const noexcept
{
  return observed_domain_;
}

std::uint32_t CurrentFramePhotometricEvidenceDecision::
evidenceGapMask() const noexcept
{
  return evidence_gap_mask_;
}

bool CurrentFramePhotometricEvidenceDecision::hasEvidenceGap(
  const CurrentFramePhotometricEvidenceGap gap) const noexcept
{
  const std::uint32_t mask =
    currentFramePhotometricEvidenceGapMask(gap);
  return
    mask != 0U &&
    (evidence_gap_mask_ & mask) == mask;
}

bool CurrentFramePhotometricEvidenceDecision::
trustedNoiseMatched() const noexcept
{
  return trusted_noise_matched_;
}

bool CurrentFramePhotometricEvidenceDecision::
experimental() const noexcept
{
  return
    path_ ==
    CurrentFramePhotometricEvidencePath::
    kRobustAffineExperimental;
}

bool CurrentFramePhotometricEvidenceDecision::
crossCameraComparable() const noexcept
{
  return
    path_ ==
    CurrentFramePhotometricEvidencePath::
    kStrictCalibratedDirect;
}

CurrentFramePhotometricEvidenceEvaluator::
CurrentFramePhotometricEvidenceEvaluator(
  CameraObservationBatchAuthorityIdentity
    expected_batch_authority,
  ReferenceOnlyMigrationRuntimePublisherIdentity
    expected_runtime_authority,
  TrustedPhotometricCalibrationLoadResult trusted_calibration,
  const CurrentFramePhotometricEvidenceConfig config)
: implementation_(
    std::make_unique<const Implementation>(
      std::move(expected_batch_authority),
      std::move(expected_runtime_authority),
      std::move(trusted_calibration),
      config))
{
}

CurrentFramePhotometricEvidenceEvaluator::
~CurrentFramePhotometricEvidenceEvaluator() = default;

CurrentFramePhotometricEvidenceDecision
CurrentFramePhotometricEvidenceEvaluator::evaluate(
  const CurrentFramePhotometricEvidenceRequest & request) const
{
  CurrentFramePhotometricEvidenceDecision decision;
  if (!implementation_->config.enabled)
  {
    return decision;
  }

  const auto reject =
    [&decision](
    const CurrentFramePhotometricEvidenceStatus status) {
      decision.status_ = status;
      decision.strict_rejection_status_ = status;
      return decision;
    };
  if (!implementation_->authority_configuration_valid)
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kInvalidAuthorityConfiguration);
  }
  if (
    !request.composition.valid() ||
    !request.composition.issuedByBatch(
      implementation_->expected_batch_authority))
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kBatchAuthorityRejected);
  }
  if (
    !request.runtime.valid() ||
    !request.runtime.issuedBy(
      implementation_->expected_runtime_authority))
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kRuntimeAuthorityRejected);
  }
  if (
    !request.composition.issuedByRuntime(
      implementation_->expected_runtime_authority) ||
    !request.composition.accepts(request.runtime))
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kRuntimeCompositionRejected);
  }
  if (
    !request.frame.valid() ||
    !request.frame.issuedBy(
      implementation_->expected_runtime_authority))
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kFrameAuthorityRejected);
  }
  if (
    !request.composition.accepts(request.frame))
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kFrameCompositionMismatch);
  }

  decision.camera_id_ = request.frame.cameraId();
  decision.camera_batch_sequence_ =
    request.frame.cameraBatchSequence();
  decision.runtime_publication_sequence_ =
    request.frame.runtimePublicationSequence();
  decision.observation_generation_ =
    request.frame.observationGeneration();
  decision.image_generation_ =
    request.frame.imageGeneration();
  decision.exposure_timestamp_sec_ =
    request.frame.exposureTimestampSec();

  ObservedRawIntensityDomain observed;
  observed.bit_depth = 8U;
  observed.unit = RawIntensityUnit::kCanonicalGrayCodeValue;
  observed.image_width =
    request.composition.imageColumns(decision.camera_id_);
  observed.image_height =
    request.composition.imageRows(decision.camera_id_);
  const int observed_image_type =
    request.composition.imageType(decision.camera_id_);
  if (observed_image_type == CV_8UC1)
  {
    observed.canonicalization =
      ObservedRawIntensityCanonicalization::
      kMono8IdentityV1;
  }
  else if (observed_image_type == CV_8UC3)
  {
    observed.canonicalization =
      ObservedRawIntensityCanonicalization::
      kBgr8OpenCvBt601LumaUint8V1;
  }
  else
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kUnsupportedImageDomain);
  }
  decision.observed_domain_ = observed;
  decision.evidence_gap_mask_ =
    currentFramePhotometricEvidenceGapMask(
      CurrentFramePhotometricEvidenceGap::
      kCanonicalGrayAuthorityUnavailable) |
    currentFramePhotometricEvidenceGapMask(
      CurrentFramePhotometricEvidenceGap::
      kNativeResolutionAuthorityUnavailable);
  if (
    request.composition.photometricStatus(
      decision.camera_id_) ==
    PhotometricStatus::kMissingExposureGain)
  {
    decision.evidence_gap_mask_ |=
      currentFramePhotometricEvidenceGapMask(
      CurrentFramePhotometricEvidenceGap::
      kHardwareMetadataUnavailable);
  }
  else
  {
    // The status is public caller input. It can diagnose an unbound claim,
    // but can never authorize strict frame evidence.
    decision.evidence_gap_mask_ |=
      currentFramePhotometricEvidenceGapMask(
      CurrentFramePhotometricEvidenceGap::
      kHardwareMetadataUnbound);
  }
  decision.calibration_revision_ =
    implementation_->calibration_revision;
  if (
    implementation_->calibration_validation ==
    TrustedCalibrationValidationStatus::kUnavailable)
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kCalibrationUnavailable);
  }
  if (
    implementation_->calibration_validation ==
    TrustedCalibrationValidationStatus::kRevisionMismatch)
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kCalibrationRevisionMismatch);
  }

  const auto experimentalOrReject =
    [this, &decision](
    const CurrentFramePhotometricEvidenceStatus
      strict_rejection) {
      decision.strict_rejection_status_ =
        strict_rejection;
      if (
        implementation_->config.
        allow_robust_affine_experiment)
      {
        decision.path_ =
          CurrentFramePhotometricEvidencePath::
          kRobustAffineExperimental;
        decision.status_ =
          CurrentFramePhotometricEvidenceStatus::
          kRobustAffineExperimentalReady;
        decision.trusted_noise_matched_ = false;
      }
      else
      {
        decision.status_ = strict_rejection;
      }
      return decision;
    };

  if (
    implementation_->calibration_validation ==
    TrustedCalibrationValidationStatus::kNoiseUnavailable)
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kNoiseCapabilityUnavailable);
  }
  const TrustedPhotometricCalibrationBundle & bundle =
    *implementation_->trusted_calibration.
    trusted_noise_bundle;
  const TrustedRawIntensityNoiseCapability * const noise =
    bundle.noiseCapability(decision.camera_id_);
  if (noise == nullptr || !noise->valid())
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kNoiseCapabilityUnavailable);
  }

  const RawIntensityDomain & signed_domain = noise->domain();
  if (
    signed_domain.bit_depth != observed.bit_depth ||
    signed_domain.unit != observed.unit ||
    signed_domain.image_width != observed.image_width ||
    signed_domain.image_height != observed.image_height)
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kNoiseDomainMismatch);
  }
  if (
    observed.canonicalization !=
      ObservedRawIntensityCanonicalization::
      kMono8IdentityV1 ||
    signed_domain.canonicalization !=
      RawIntensityCanonicalization::kMono8IdentityV1)
  {
    return experimentalOrReject(
      CurrentFramePhotometricEvidenceStatus::
      kRuntimeCanonicalizationMismatch);
  }

  RawIntensityDomain observed_trusted_domain;
  observed_trusted_domain.bit_depth = observed.bit_depth;
  observed_trusted_domain.unit = observed.unit;
  observed_trusted_domain.canonicalization =
    RawIntensityCanonicalization::kMono8IdentityV1;
  observed_trusted_domain.image_width = observed.image_width;
  observed_trusted_domain.image_height = observed.image_height;
  if (
    !noise->validFor(
      decision.camera_id_,
      decision.calibration_revision_,
      observed_trusted_domain))
  {
    return reject(
      CurrentFramePhotometricEvidenceStatus::
      kNoiseDomainMismatch);
  }
  decision.trusted_noise_matched_ = true;
  return experimentalOrReject(
    CurrentFramePhotometricEvidenceStatus::
    kCanonicalGrayAuthorityUnavailable);
}

const char * currentFramePhotometricEvidencePathName(
  const CurrentFramePhotometricEvidencePath path) noexcept
{
  switch (path)
  {
    case CurrentFramePhotometricEvidencePath::kUnavailable:
      return "unavailable";
    case CurrentFramePhotometricEvidencePath::
      kStrictCalibratedDirect:
      return "strict_calibrated_direct";
    case CurrentFramePhotometricEvidencePath::
      kRobustAffineExperimental:
      return "robust_affine_experimental";
  }
  return "unknown";
}

const char * currentFramePhotometricEvidenceStatusName(
  const CurrentFramePhotometricEvidenceStatus status) noexcept
{
  switch (status)
  {
    case CurrentFramePhotometricEvidenceStatus::kDisabled:
      return "disabled";
    case CurrentFramePhotometricEvidenceStatus::
      kInvalidAuthorityConfiguration:
      return "invalid_authority_configuration";
    case CurrentFramePhotometricEvidenceStatus::
      kBatchAuthorityRejected:
      return "batch_authority_rejected";
    case CurrentFramePhotometricEvidenceStatus::
      kRuntimeAuthorityRejected:
      return "runtime_authority_rejected";
    case CurrentFramePhotometricEvidenceStatus::
      kRuntimeCompositionRejected:
      return "runtime_composition_rejected";
    case CurrentFramePhotometricEvidenceStatus::
      kFrameAuthorityRejected:
      return "frame_authority_rejected";
    case CurrentFramePhotometricEvidenceStatus::
      kFrameCompositionMismatch:
      return "frame_composition_mismatch";
    case CurrentFramePhotometricEvidenceStatus::
      kUnsupportedImageDomain:
      return "unsupported_image_domain";
    case CurrentFramePhotometricEvidenceStatus::
      kCalibrationUnavailable:
      return "calibration_unavailable";
    case CurrentFramePhotometricEvidenceStatus::
      kCalibrationRevisionMismatch:
      return "calibration_revision_mismatch";
    case CurrentFramePhotometricEvidenceStatus::
      kNoiseCapabilityUnavailable:
      return "noise_capability_unavailable";
    case CurrentFramePhotometricEvidenceStatus::
      kNoiseDomainMismatch:
      return "noise_domain_mismatch";
    case CurrentFramePhotometricEvidenceStatus::
      kRuntimeCanonicalizationMismatch:
      return "runtime_canonicalization_mismatch";
    case CurrentFramePhotometricEvidenceStatus::
      kCanonicalGrayAuthorityUnavailable:
      return "canonical_gray_authority_unavailable";
    case CurrentFramePhotometricEvidenceStatus::
      kRobustAffineExperimentalReady:
      return "robust_affine_experimental_ready";
  }
  return "unknown";
}

const char * observedRawIntensityCanonicalizationName(
  const ObservedRawIntensityCanonicalization
    canonicalization) noexcept
{
  switch (canonicalization)
  {
    case ObservedRawIntensityCanonicalization::kInvalid:
      return "invalid";
    case ObservedRawIntensityCanonicalization::
      kMono8IdentityV1:
      return "mono8_identity_v1";
    case ObservedRawIntensityCanonicalization::
      kBgr8OpenCvBt601LumaUint8V1:
      return "bgr8_opencv_bt601_luma_uint8_v1";
  }
  return "unknown";
}

}  // namespace livo
