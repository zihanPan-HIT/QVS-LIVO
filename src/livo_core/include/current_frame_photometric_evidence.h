#ifndef LIVO_CURRENT_FRAME_PHOTOMETRIC_EVIDENCE_H_
#define LIVO_CURRENT_FRAME_PHOTOMETRIC_EVIDENCE_H_

#include "photometric_noise_authority.h"
#include "trusted_photometric_calibration_loader.h"

#include <cstdint>
#include <memory>

namespace livo
{

class CameraObservationBatchAuthorityIdentity;
class ReferenceOnlyMigrationFrameProvenance;
class ReferenceOnlyMigrationRuntimeBundle;
class ReferenceOnlyMigrationRuntimePublisherIdentity;
class ValidatedBatchRuntimeComposition;

enum class CurrentFramePhotometricEvidencePath
{
  kUnavailable,
  // Output category retained by the seam contract. This evaluator cannot
  // select it until an owner attests the current-frame evidence gaps.
  kStrictCalibratedDirect,
  kRobustAffineExperimental
};

enum class CurrentFramePhotometricEvidenceStatus
{
  kDisabled,
  kInvalidAuthorityConfiguration,
  kBatchAuthorityRejected,
  kRuntimeAuthorityRejected,
  kRuntimeCompositionRejected,
  kFrameAuthorityRejected,
  kFrameCompositionMismatch,
  kUnsupportedImageDomain,
  kCalibrationUnavailable,
  kCalibrationRevisionMismatch,
  kNoiseCapabilityUnavailable,
  kNoiseDomainMismatch,
  kRuntimeCanonicalizationMismatch,
  kCanonicalGrayAuthorityUnavailable,
  kRobustAffineExperimentalReady
};

enum class CurrentFramePhotometricEvidenceGap
  : std::uint32_t
{
  kNone = 0U,
  // The exact gray canonicalization was not attested by its data owner.
  kCanonicalGrayAuthorityUnavailable = 1U << 0U,
  // Native acquisition dimensions and absence of resampling were not
  // attested by their data owner.
  kNativeResolutionAuthorityUnavailable = 1U << 1U,
  // Hardware exposure/gain metadata is explicitly absent.
  kHardwareMetadataUnavailable = 1U << 2U,
  // A public metadata claim exists but is not bound by an owner capability.
  kHardwareMetadataUnbound = 1U << 3U
};

constexpr std::uint32_t currentFramePhotometricEvidenceGapMask(
  const CurrentFramePhotometricEvidenceGap gap) noexcept
{
  return static_cast<std::uint32_t>(gap);
}

enum class ObservedRawIntensityCanonicalization
{
  kInvalid,
  kMono8IdentityV1,
  kBgr8OpenCvBt601LumaUint8V1
};

struct ObservedRawIntensityDomain
{
  std::uint8_t bit_depth{0U};
  RawIntensityUnit unit{RawIntensityUnit::kInvalid};
  ObservedRawIntensityCanonicalization canonicalization{
    ObservedRawIntensityCanonicalization::kInvalid};
  int image_width{0};
  int image_height{0};
  bool canonical_gray_owner_attested{false};
  bool native_resolution_attested{false};

  bool valid() const noexcept;
};

bool operator==(
  const ObservedRawIntensityDomain & lhs,
  const ObservedRawIntensityDomain & rhs) noexcept;

bool operator!=(
  const ObservedRawIntensityDomain & lhs,
  const ObservedRawIntensityDomain & rhs) noexcept;

struct CurrentFramePhotometricEvidenceConfig
{
  bool enabled{false};
  bool allow_robust_affine_experiment{false};
};

struct CurrentFramePhotometricEvidenceRequest
{
  const ValidatedBatchRuntimeComposition & composition;
  const ReferenceOnlyMigrationRuntimeBundle & runtime;
  const ReferenceOnlyMigrationFrameProvenance & frame;
};

class CurrentFramePhotometricEvidenceDecision
{
public:
  CurrentFramePhotometricEvidenceDecision(
    const CurrentFramePhotometricEvidenceDecision &) = default;
  CurrentFramePhotometricEvidenceDecision(
    CurrentFramePhotometricEvidenceDecision &&) = default;
  CurrentFramePhotometricEvidenceDecision & operator=(
    const CurrentFramePhotometricEvidenceDecision &) = default;
  CurrentFramePhotometricEvidenceDecision & operator=(
    CurrentFramePhotometricEvidenceDecision &&) = default;

  CurrentFramePhotometricEvidencePath path() const noexcept;
  CurrentFramePhotometricEvidenceStatus status() const noexcept;
  CurrentFramePhotometricEvidenceStatus
  strictRejectionStatus() const noexcept;
  std::uint8_t cameraId() const noexcept;
  std::uint64_t cameraBatchSequence() const noexcept;
  std::uint64_t runtimePublicationSequence() const noexcept;
  std::uint64_t observationGeneration() const noexcept;
  std::uint64_t imageGeneration() const noexcept;
  double exposureTimestampSec() const noexcept;
  std::uint64_t calibrationRevision() const noexcept;
  const ObservedRawIntensityDomain &
  observedDomain() const noexcept;
  std::uint32_t evidenceGapMask() const noexcept;
  bool hasEvidenceGap(
    CurrentFramePhotometricEvidenceGap gap) const noexcept;
  // Reports exact signed noise-domain identity only. It never exposes the
  // trusted sigma, and every experimental decision resets this to false.
  bool trustedNoiseMatched() const noexcept;
  bool experimental() const noexcept;
  bool crossCameraComparable() const noexcept;

  static constexpr bool hasLinearization() noexcept
  {
    return false;
  }

  static constexpr bool consumable() noexcept
  {
    return false;
  }

  static constexpr bool productionReady() noexcept
  {
    return false;
  }

  static constexpr bool solverApplied() noexcept
  {
    return false;
  }

private:
  CurrentFramePhotometricEvidenceDecision() = default;

  CurrentFramePhotometricEvidencePath path_{
    CurrentFramePhotometricEvidencePath::kUnavailable};
  CurrentFramePhotometricEvidenceStatus status_{
    CurrentFramePhotometricEvidenceStatus::kDisabled};
  CurrentFramePhotometricEvidenceStatus strict_rejection_status_{
    CurrentFramePhotometricEvidenceStatus::kDisabled};
  std::uint8_t camera_id_{0U};
  std::uint64_t camera_batch_sequence_{0U};
  std::uint64_t runtime_publication_sequence_{0U};
  std::uint64_t observation_generation_{0U};
  std::uint64_t image_generation_{0U};
  double exposure_timestamp_sec_{0.0};
  std::uint64_t calibration_revision_{0U};
  ObservedRawIntensityDomain observed_domain_;
  std::uint32_t evidence_gap_mask_{0U};
  bool trusted_noise_matched_{false};

  friend class CurrentFramePhotometricEvidenceEvaluator;
};

class CurrentFramePhotometricEvidenceEvaluator final
{
public:
  CurrentFramePhotometricEvidenceEvaluator(
    CameraObservationBatchAuthorityIdentity
      expected_batch_authority,
    ReferenceOnlyMigrationRuntimePublisherIdentity
      expected_runtime_authority,
    TrustedPhotometricCalibrationLoadResult
      trusted_calibration,
    CurrentFramePhotometricEvidenceConfig config);
  ~CurrentFramePhotometricEvidenceEvaluator();

  CurrentFramePhotometricEvidenceEvaluator(
    const CurrentFramePhotometricEvidenceEvaluator &) = delete;
  CurrentFramePhotometricEvidenceEvaluator & operator=(
    const CurrentFramePhotometricEvidenceEvaluator &) = delete;
  CurrentFramePhotometricEvidenceEvaluator(
    CurrentFramePhotometricEvidenceEvaluator &&) = delete;
  CurrentFramePhotometricEvidenceEvaluator & operator=(
    CurrentFramePhotometricEvidenceEvaluator &&) = delete;

  // Validates one exact owner-issued publication composition. It makes no
  // temporal freshness or anti-replay claim; upstream VIO owns that policy.
  CurrentFramePhotometricEvidenceDecision evaluate(
    const CurrentFramePhotometricEvidenceRequest & request) const;

private:
  struct Implementation;
  std::unique_ptr<const Implementation> implementation_;
};

const char * currentFramePhotometricEvidencePathName(
  CurrentFramePhotometricEvidencePath path) noexcept;

const char * currentFramePhotometricEvidenceStatusName(
  CurrentFramePhotometricEvidenceStatus status) noexcept;

const char * observedRawIntensityCanonicalizationName(
  ObservedRawIntensityCanonicalization canonicalization) noexcept;

}  // namespace livo

#endif  // LIVO_CURRENT_FRAME_PHOTOMETRIC_EVIDENCE_H_
