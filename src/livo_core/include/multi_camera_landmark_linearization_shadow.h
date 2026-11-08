#ifndef LIVO_MULTI_CAMERA_LANDMARK_LINEARIZATION_SHADOW_H_
#define LIVO_MULTI_CAMERA_LANDMARK_LINEARIZATION_SHADOW_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "multi_view_landmark.h"

namespace livo
{

class AdaptiveVisualPolicy;
class AdaptiveVisualPolicyPlanCapability;
class CameraObservationBatchAuthorityIdentity;
class ReferenceOnlyMigrationFrameProvenance;
class ReferenceOnlyMigrationRuntimeBundle;
class ReferenceOnlyMigrationRuntimePublisherIdentity;
class SharedLandmarkShadowSession;

// Immutable construction-time policy. There is deliberately no archive:
// every accepted target reads and validates the current shared-store
// snapshot.
struct MultiCameraLandmarkLinearizationShadowConfig
{
  bool enabled{false};
  bool allow_relative_affine_experiment{false};
  double maximum_source_age_sec{0.5};
  std::size_t maximum_candidate_evaluations_per_batch{16384U};
  std::uint64_t maximum_accepted_target_generations{
    std::numeric_limits<std::uint64_t>::max()};

  bool valid() const noexcept;
};

enum class MultiCameraLandmarkPhotometricPath
{
  kSameCameraLegacyUnattested,
  kCrossCameraCalibrated,
  kCrossCameraRelativeAffineExperimental
};

enum class MultiCameraLandmarkShadowBatchStatus
{
  kDisabled,
  kInvalidConfiguration,
  kInvalidAuthorityConfiguration,
  kTargetBatchAuthorityRejected,
  kRuntimeAuthorityRejected,
  kPolicyCapabilityRejected,
  kTargetCompositionRejected,
  kTargetOrderRejected,
  kSessionUnavailable,
  kMissingCurrentActivePlaneSnapshot,
  kStaleCurrentActivePlaneSnapshot,
  kStoreAuthorityViolation,
  kCandidateBudgetExceeded,
  kGenerationExhausted,
  kAccepted
};

enum class MultiCameraLandmarkShadowDecisionStatus
{
  kSelectionTracedLinearizationUnavailableMissingPhotometricNoise,
  kStrictCrossCameraEvidenceUnavailable
};

// Immutable historical trace of one source/target choice. It binds the
// owner-issued store receipt, owner-issued target frame, owner-sealed policy
// capability, actual policy decision values, and the immutable module
// configuration. It is not a linearization receipt or solver capability.
class MultiCameraLandmarkSelectionTrace
{
public:
  MultiCameraLandmarkSelectionTrace(
    const MultiCameraLandmarkSelectionTrace &) = delete;
  MultiCameraLandmarkSelectionTrace(
    MultiCameraLandmarkSelectionTrace &&) = delete;
  MultiCameraLandmarkSelectionTrace & operator=(
    const MultiCameraLandmarkSelectionTrace &) = delete;
  MultiCameraLandmarkSelectionTrace & operator=(
    MultiCameraLandmarkSelectionTrace &&) = delete;
  ~MultiCameraLandmarkSelectionTrace() = default;

  bool valid() const noexcept;
  std::uint64_t configurationGeneration() const noexcept;
  std::uint64_t acceptedTargetGeneration() const noexcept;
  LandmarkId landmarkId() const noexcept;
  ObservationId sourceObservationId() const noexcept;
  std::uint64_t sourceObservationRevision() const noexcept;
  const AcceptedObservationStoreReceipt &
  sourceStoreReceipt() const noexcept;
  MapRevision selectedStoreRevision() const noexcept;
  GeometryVersion sourceSamplingGeometryVersion() const noexcept;
  const ActivePlaneIdentityLease &
  sourceActivePlaneLease() const noexcept;
  std::uint8_t sourceCameraId() const noexcept;
  std::uint64_t sourceCameraBatchSequence() const noexcept;
  double sourceTimestampSec() const noexcept;
  const Eigen::Matrix<double, 6, 6> &
  sourcePoseCovariance() const noexcept;

  const ReferenceOnlyMigrationFrameProvenance &
  targetFrameProvenance() const noexcept;
  std::uint8_t targetCameraId() const noexcept;
  std::uint64_t targetCameraBatchSequence() const noexcept;
  std::uint64_t targetRuntimePublicationSequence() const noexcept;
  std::uint64_t targetActiveStateSnapshotRevision() const noexcept;
  std::uint64_t targetObservationGeneration() const noexcept;
  std::uint64_t targetImageGeneration() const noexcept;
  double targetTimestampSec() const noexcept;

  const AdaptiveVisualPolicyPlanCapability &
  policyCapability() const noexcept;
  std::uint64_t policyBaseRevision() const noexcept;
  double healthScore() const noexcept;
  double lidarComplementScore() const noexcept;
  std::size_t candidateBudget() const noexcept;
  std::size_t patchBudget() const noexcept;

  bool configuredRelativeAffineExperiment() const noexcept;
  double configuredMaximumSourceAgeSec() const noexcept;
  std::size_t
  configuredMaximumCandidateEvaluationsPerBatch() const noexcept;
  std::uint64_t
  configuredMaximumAcceptedTargetGenerations() const noexcept;

  MultiCameraLandmarkPhotometricPath
  photometricPath() const noexcept;
  const char * photometricPathLabel() const noexcept;
  bool crossCameraComparable() const noexcept;
  bool experimental() const noexcept;

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
  struct Payload;
  explicit MultiCameraLandmarkSelectionTrace(
    std::shared_ptr<const Payload> payload);

  std::shared_ptr<const Payload> payload_;

  friend class MultiCameraLandmarkLinearizationShadow;
};

class MultiCameraLandmarkShadowDecision
{
public:
  MultiCameraLandmarkShadowDecision(
    const MultiCameraLandmarkShadowDecision &) = default;
  MultiCameraLandmarkShadowDecision(
    MultiCameraLandmarkShadowDecision &&) = default;
  MultiCameraLandmarkShadowDecision & operator=(
    const MultiCameraLandmarkShadowDecision &) = default;
  MultiCameraLandmarkShadowDecision & operator=(
    MultiCameraLandmarkShadowDecision &&) = default;
  ~MultiCameraLandmarkShadowDecision() = default;

  MultiCameraLandmarkShadowDecisionStatus status() const noexcept;
  LandmarkId landmarkId() const noexcept;
  ObservationId sourceObservationId() const noexcept;
  std::uint8_t sourceCameraId() const noexcept;
  std::uint8_t targetCameraId() const noexcept;
  MultiCameraLandmarkPhotometricPath
  photometricPath() const noexcept;
  const char * photometricPathLabel() const noexcept;
  double healthScore() const noexcept;
  double lidarComplementScore() const noexcept;
  std::size_t candidateBudget() const noexcept;
  std::size_t patchBudget() const noexcept;
  bool hasSelectionTrace() const noexcept;
  const MultiCameraLandmarkSelectionTrace *
  selectionTrace() const noexcept;

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
  MultiCameraLandmarkShadowDecision(
    MultiCameraLandmarkShadowDecisionStatus status,
    LandmarkId landmark_id,
    ObservationId source_observation_id,
    std::uint8_t source_camera_id,
    std::uint8_t target_camera_id,
    MultiCameraLandmarkPhotometricPath photometric_path,
    double health_score,
    double lidar_complement_score,
    std::size_t candidate_budget,
    std::size_t patch_budget,
    std::shared_ptr<const MultiCameraLandmarkSelectionTrace>
      trace);

  MultiCameraLandmarkShadowDecisionStatus status_;
  LandmarkId landmark_id_{0U};
  ObservationId source_observation_id_{0U};
  std::uint8_t source_camera_id_{0U};
  std::uint8_t target_camera_id_{0U};
  MultiCameraLandmarkPhotometricPath photometric_path_{
    MultiCameraLandmarkPhotometricPath::
    kSameCameraLegacyUnattested};
  double health_score_{0.0};
  double lidar_complement_score_{0.0};
  std::size_t candidate_budget_{0U};
  std::size_t patch_budget_{0U};
  std::shared_ptr<const MultiCameraLandmarkSelectionTrace>
    trace_;

  friend class MultiCameraLandmarkLinearizationShadow;
};

class MultiCameraLandmarkShadowBatchResult
{
public:
  MultiCameraLandmarkShadowBatchResult(
    const MultiCameraLandmarkShadowBatchResult &) = default;
  MultiCameraLandmarkShadowBatchResult(
    MultiCameraLandmarkShadowBatchResult &&) = default;
  MultiCameraLandmarkShadowBatchResult & operator=(
    const MultiCameraLandmarkShadowBatchResult &) = default;
  MultiCameraLandmarkShadowBatchResult & operator=(
    MultiCameraLandmarkShadowBatchResult &&) = default;
  ~MultiCameraLandmarkShadowBatchResult() = default;

  MultiCameraLandmarkShadowBatchStatus status() const noexcept;
  bool accepted() const noexcept;
  std::uint64_t configurationGeneration() const noexcept;
  std::uint64_t acceptedTargetGeneration() const noexcept;
  std::uint64_t targetCameraBatchSequence() const noexcept;
  std::uint64_t targetRuntimePublicationSequence() const noexcept;
  MapRevision storeRevision() const noexcept;
  std::size_t inspectedSourceObservationCount() const noexcept;
  std::size_t storeAuthorityViolationCount() const noexcept;
  std::size_t skippedReferenceOnlySourceCount() const noexcept;
  std::size_t skippedInactivePlaneSourceCount() const noexcept;
  std::size_t skippedTombstonedSourceCount() const noexcept;
  std::size_t skippedSamplingGeometryCount() const noexcept;
  std::size_t skippedSourceOrderCount() const noexcept;
  std::size_t skippedSourceRuntimeAuthorityCount() const noexcept;
  std::size_t skippedSourceAgeCount() const noexcept;
  std::size_t candidateEvaluationCount() const noexcept;
  std::size_t strictPhotometricRejectedCount() const noexcept;
  const std::vector<MultiCameraLandmarkShadowDecision> &
  decisions() const noexcept;

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
  explicit MultiCameraLandmarkShadowBatchResult(
    MultiCameraLandmarkShadowBatchStatus status);

  MultiCameraLandmarkShadowBatchStatus status_;
  std::uint64_t configuration_generation_{0U};
  std::uint64_t accepted_target_generation_{0U};
  std::uint64_t target_camera_batch_sequence_{0U};
  std::uint64_t target_runtime_publication_sequence_{0U};
  MapRevision store_revision_;
  std::size_t inspected_source_observation_count_{0U};
  std::size_t store_authority_violation_count_{0U};
  std::size_t skipped_reference_only_source_count_{0U};
  std::size_t skipped_inactive_plane_source_count_{0U};
  std::size_t skipped_tombstoned_source_count_{0U};
  std::size_t skipped_sampling_geometry_count_{0U};
  std::size_t skipped_source_order_count_{0U};
  std::size_t skipped_source_runtime_authority_count_{0U};
  std::size_t skipped_source_age_count_{0U};
  std::size_t candidate_evaluation_count_{0U};
  std::size_t strict_photometric_rejected_count_{0U};
  std::vector<MultiCameraLandmarkShadowDecision> decisions_;

  friend class MultiCameraLandmarkLinearizationShadow;
};

class MultiCameraLandmarkLinearizationShadow
{
public:
  MultiCameraLandmarkLinearizationShadow(
    SharedLandmarkShadowSession & session,
    const CameraObservationBatchAuthorityIdentity &
      expected_batch_authority,
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      expected_runtime_authority,
    const AdaptiveVisualPolicy & expected_policy,
    MultiCameraLandmarkLinearizationShadowConfig config = {});
  ~MultiCameraLandmarkLinearizationShadow();

  MultiCameraLandmarkLinearizationShadow(
    const MultiCameraLandmarkLinearizationShadow &) = delete;
  MultiCameraLandmarkLinearizationShadow & operator=(
    const MultiCameraLandmarkLinearizationShadow &) = delete;
  MultiCameraLandmarkLinearizationShadow(
    MultiCameraLandmarkLinearizationShadow &&) = delete;
  MultiCameraLandmarkLinearizationShadow & operator=(
    MultiCameraLandmarkLinearizationShadow &&) = delete;

  bool active() const noexcept;
  std::uint64_t configurationGeneration() const noexcept;
  std::uint64_t acceptedTargetGeneration() const noexcept;
  MultiCameraLandmarkShadowBatchResult processBatch(
    const ReferenceOnlyMigrationRuntimeBundle & runtime,
    const AdaptiveVisualPolicyPlanCapability & capability);
  bool accepts(
    const MultiCameraLandmarkSelectionTrace & trace) const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

const char * multiCameraLandmarkShadowBatchStatusName(
  MultiCameraLandmarkShadowBatchStatus status) noexcept;
const char * multiCameraLandmarkShadowDecisionStatusName(
  MultiCameraLandmarkShadowDecisionStatus status) noexcept;
const char * multiCameraLandmarkPhotometricPathName(
  MultiCameraLandmarkPhotometricPath path) noexcept;

}  // namespace livo

#endif  // LIVO_MULTI_CAMERA_LANDMARK_LINEARIZATION_SHADOW_H_
