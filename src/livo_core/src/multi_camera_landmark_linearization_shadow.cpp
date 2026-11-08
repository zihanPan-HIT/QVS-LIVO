#include "multi_camera_landmark_linearization_shadow.h"

#include "adaptive_visual_policy.h"
#include "camera_observation.h"
#include "checked_increment.h"
#include "reference_only_migration_runtime_provenance.h"
#include "shared_landmark_shadow_session.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <utility>

namespace livo
{
namespace
{

bool sameGeometryVersion(
  const GeometryVersion & lhs,
  const GeometryVersion & rhs) noexcept
{
  return
    lhs.plane == rhs.plane &&
    lhs.revision == rhs.revision;
}

bool sameFrameBinding(
  const ReferenceOnlyMigrationFrameProvenance & lhs,
  const ReferenceOnlyMigrationFrameProvenance & rhs) noexcept
{
  return
    lhs.valid() &&
    rhs.valid() &&
    lhs.cameraBatchSequence() == rhs.cameraBatchSequence() &&
    lhs.runtimePublicationSequence() ==
      rhs.runtimePublicationSequence() &&
    lhs.activeStateSnapshotRevision() ==
      rhs.activeStateSnapshotRevision() &&
    lhs.cameraId() == rhs.cameraId() &&
    lhs.observationGeneration() == rhs.observationGeneration() &&
    lhs.imageGeneration() == rhs.imageGeneration() &&
    lhs.exposureTimestampSec() == rhs.exposureTimestampSec();
}

bool validPolicyDecision(
  const AdaptiveCameraDecision & decision) noexcept
{
  return
    std::isfinite(decision.health) &&
    decision.health >= 0.0 &&
    std::isfinite(decision.complement_score) &&
    decision.complement_score >= 0.0 &&
    decision.candidate_budget > 0U &&
    decision.patch_budget > 0U;
}

int pathPriority(
  const MultiCameraLandmarkPhotometricPath path) noexcept
{
  switch (path)
  {
    case MultiCameraLandmarkPhotometricPath::
      kSameCameraLegacyUnattested:
      return 0;
    case MultiCameraLandmarkPhotometricPath::
      kCrossCameraCalibrated:
      return 1;
    case MultiCameraLandmarkPhotometricPath::
      kCrossCameraRelativeAffineExperimental:
      return 2;
  }
  return 3;
}

}  // namespace

struct MultiCameraLandmarkSelectionTrace::Payload
{
  std::shared_ptr<const std::uint8_t> owner;
  std::uint64_t configuration_generation{0U};
  std::uint64_t accepted_target_generation{0U};
  MultiCameraLandmarkLinearizationShadowConfig config;

  LandmarkId landmark_id{0U};
  ObservationId source_observation_id{0U};
  std::uint64_t source_observation_revision{0U};
  std::optional<AcceptedObservationStoreReceipt>
    source_store_receipt;
  MapRevision selected_store_revision;
  GeometryVersion source_sampling_geometry_version;
  ActivePlaneIdentityLease source_active_plane_lease;
  std::uint8_t source_camera_id{0U};
  std::uint64_t source_camera_batch_sequence{0U};
  double source_timestamp_sec{0.0};
  Eigen::Matrix<double, 6, 6> source_pose_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};

  std::optional<ReferenceOnlyMigrationFrameProvenance>
    target_frame;
  std::uint8_t target_camera_id{0U};
  std::uint64_t target_camera_batch_sequence{0U};
  std::uint64_t target_runtime_publication_sequence{0U};
  std::uint64_t target_active_state_snapshot_revision{0U};
  std::uint64_t target_observation_generation{0U};
  std::uint64_t target_image_generation{0U};
  double target_timestamp_sec{0.0};

  std::optional<AdaptiveVisualPolicyPlanCapability>
    policy_capability;
  std::uint64_t policy_base_revision{0U};
  double health_score{0.0};
  double lidar_complement_score{0.0};
  std::size_t candidate_budget{0U};
  std::size_t patch_budget{0U};

  MultiCameraLandmarkPhotometricPath photometric_path{
    MultiCameraLandmarkPhotometricPath::
    kSameCameraLegacyUnattested};
  bool experimental{false};
};

struct MultiCameraLandmarkLinearizationShadow::Implementation
{
  struct TargetCamera
  {
    std::optional<ReferenceOnlyMigrationFrameProvenance> frame;
    double health{0.0};
    double complement{0.0};
    std::size_t candidate_budget{0U};
    std::size_t patch_budget{0U};
  };

  struct Source
  {
    const LandmarkSnapshot * landmark{nullptr};
    const ObservationSnapshot * observation{nullptr};
    ActivePlaneIdentityLease active_plane_lease;
  };

  struct Candidate
  {
    const LandmarkSnapshot * landmark{nullptr};
    const ObservationSnapshot * observation{nullptr};
    const TargetCamera * target{nullptr};
    ActivePlaneIdentityLease active_plane_lease;
    std::uint8_t target_camera_id{0U};
    MultiCameraLandmarkPhotometricPath path{
      MultiCameraLandmarkPhotometricPath::
      kSameCameraLegacyUnattested};
    double age_sec{0.0};
  };

  Implementation(
    SharedLandmarkShadowSession & session_value,
    const CameraObservationBatchAuthorityIdentity &
      batch_authority,
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      runtime_authority,
    const AdaptiveVisualPolicy & policy,
    MultiCameraLandmarkLinearizationShadowConfig config_value)
  : session(&session_value),
    expected_batch_authority(batch_authority),
    expected_runtime_authority(runtime_authority),
    expected_policy(&policy),
    config(std::move(config_value)),
    config_valid(config.valid()),
    authority_configuration_valid(
      expected_batch_authority.valid() &&
      expected_runtime_authority.valid() &&
      expected_runtime_authority.bindsBatchAuthority(
        expected_batch_authority)),
    owner(std::make_shared<const std::uint8_t>(0U))
  {
  }

  SharedLandmarkShadowSession * session{nullptr};
  CameraObservationBatchAuthorityIdentity
    expected_batch_authority;
  ReferenceOnlyMigrationRuntimePublisherIdentity
    expected_runtime_authority;
  const AdaptiveVisualPolicy * expected_policy{nullptr};
  MultiCameraLandmarkLinearizationShadowConfig config;
  bool config_valid{false};
  bool authority_configuration_valid{false};
  std::shared_ptr<const std::uint8_t> owner;
  std::uint64_t configuration_generation{1U};
  std::uint64_t accepted_target_generation{0U};
  std::uint64_t last_target_camera_batch_sequence{0U};
  std::uint64_t last_target_runtime_publication_sequence{0U};
  bool generation_exhausted{false};
};

bool MultiCameraLandmarkLinearizationShadowConfig::valid() const noexcept
{
  return
    std::isfinite(maximum_source_age_sec) &&
    maximum_source_age_sec >= 0.0 &&
    maximum_candidate_evaluations_per_batch > 0U &&
    maximum_accepted_target_generations > 0U;
}

MultiCameraLandmarkSelectionTrace::
MultiCameraLandmarkSelectionTrace(
  std::shared_ptr<const Payload> payload)
: payload_(std::move(payload))
{
}

bool MultiCameraLandmarkSelectionTrace::valid() const noexcept
{
  if (
    payload_ == nullptr ||
    payload_->owner == nullptr ||
    payload_->configuration_generation == 0U ||
    payload_->accepted_target_generation == 0U ||
    !payload_->config.enabled ||
    !payload_->config.valid() ||
    payload_->landmark_id == 0U ||
    payload_->source_observation_id == 0U ||
    payload_->source_observation_revision == 0U ||
    !payload_->source_store_receipt.has_value() ||
    !payload_->source_store_receipt->valid() ||
    payload_->source_store_receipt->landmarkId() !=
      payload_->landmark_id ||
    payload_->source_store_receipt->observationId() !=
      payload_->source_observation_id ||
    payload_->source_store_receipt->observationRevision() !=
      payload_->source_observation_revision ||
    payload_->selected_store_revision.value == 0U ||
    payload_->source_store_receipt->storeRevision().value >
      payload_->selected_store_revision.value ||
    !payload_->source_sampling_geometry_version.valid() ||
    !payload_->source_active_plane_lease.valid() ||
    !sameGeometryVersion(
      payload_->source_active_plane_lease.geometryVersion(),
      payload_->source_sampling_geometry_version) ||
    !sameGeometryVersion(
      payload_->source_store_receipt->
      samplingGeometryVersion(),
      payload_->source_sampling_geometry_version) ||
    payload_->source_camera_id >= 4U ||
    payload_->source_camera_batch_sequence == 0U ||
    payload_->source_store_receipt->
      acceptedCameraBatchSequence() !=
      payload_->source_camera_batch_sequence ||
    !std::isfinite(payload_->source_timestamp_sec) ||
    payload_->source_timestamp_sec <= 0.0 ||
    !payload_->source_pose_covariance.allFinite() ||
    payload_->source_pose_covariance.trace() <= 0.0 ||
    !payload_->target_frame.has_value() ||
    !payload_->target_frame->valid() ||
    payload_->target_camera_id >= 4U ||
    payload_->target_frame->cameraId() !=
      payload_->target_camera_id ||
    payload_->target_camera_batch_sequence == 0U ||
    payload_->target_frame->cameraBatchSequence() !=
      payload_->target_camera_batch_sequence ||
    payload_->target_runtime_publication_sequence == 0U ||
    payload_->target_frame->runtimePublicationSequence() !=
      payload_->target_runtime_publication_sequence ||
    payload_->target_active_state_snapshot_revision == 0U ||
    payload_->target_frame->activeStateSnapshotRevision() !=
      payload_->target_active_state_snapshot_revision ||
    payload_->target_observation_generation == 0U ||
    payload_->target_frame->observationGeneration() !=
      payload_->target_observation_generation ||
    payload_->target_image_generation == 0U ||
    payload_->target_frame->imageGeneration() !=
      payload_->target_image_generation ||
    !std::isfinite(payload_->target_timestamp_sec) ||
    payload_->target_frame->exposureTimestampSec() !=
      payload_->target_timestamp_sec ||
    payload_->target_timestamp_sec <
      payload_->source_timestamp_sec ||
    payload_->source_camera_batch_sequence >=
      payload_->target_camera_batch_sequence ||
    !payload_->policy_capability.has_value() ||
    !payload_->policy_capability->valid() ||
    payload_->policy_capability->basePolicyRevision() !=
      payload_->policy_base_revision ||
    payload_->policy_capability->
      runtimePublicationSequence() !=
      payload_->target_runtime_publication_sequence ||
    payload_->policy_capability->cameraBatchSequence() !=
      payload_->target_camera_batch_sequence ||
    !std::isfinite(payload_->health_score) ||
    payload_->health_score < 0.0 ||
    !std::isfinite(payload_->lidar_complement_score) ||
    payload_->lidar_complement_score < 0.0 ||
    payload_->candidate_budget == 0U ||
    payload_->patch_budget == 0U)
  {
    return false;
  }
  const AdaptiveCameraDecision & decision =
    payload_->policy_capability->
    decisionPlan().cameras[payload_->target_camera_id];
  if (
    !decision.enabled ||
    !decision.available ||
    decision.isolated ||
    decision.health != payload_->health_score ||
    decision.complement_score !=
      payload_->lidar_complement_score ||
    decision.candidate_budget !=
      payload_->candidate_budget ||
    decision.patch_budget != payload_->patch_budget)
  {
    return false;
  }
  const bool same_camera =
    payload_->source_camera_id ==
    payload_->target_camera_id;
  return
    (same_camera &&
     payload_->photometric_path ==
       MultiCameraLandmarkPhotometricPath::
       kSameCameraLegacyUnattested &&
     !payload_->experimental) ||
    (!same_camera &&
     payload_->config.allow_relative_affine_experiment &&
     payload_->photometric_path ==
       MultiCameraLandmarkPhotometricPath::
       kCrossCameraRelativeAffineExperimental &&
     payload_->experimental);
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
configurationGeneration() const noexcept
{
  return payload_->configuration_generation;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
acceptedTargetGeneration() const noexcept
{
  return payload_->accepted_target_generation;
}

LandmarkId
MultiCameraLandmarkSelectionTrace::landmarkId() const noexcept
{
  return payload_->landmark_id;
}

ObservationId
MultiCameraLandmarkSelectionTrace::
sourceObservationId() const noexcept
{
  return payload_->source_observation_id;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
sourceObservationRevision() const noexcept
{
  return payload_->source_observation_revision;
}

const AcceptedObservationStoreReceipt &
MultiCameraLandmarkSelectionTrace::
sourceStoreReceipt() const noexcept
{
  return *payload_->source_store_receipt;
}

MapRevision
MultiCameraLandmarkSelectionTrace::
selectedStoreRevision() const noexcept
{
  return payload_->selected_store_revision;
}

GeometryVersion
MultiCameraLandmarkSelectionTrace::
sourceSamplingGeometryVersion() const noexcept
{
  return payload_->source_sampling_geometry_version;
}

const ActivePlaneIdentityLease &
MultiCameraLandmarkSelectionTrace::
sourceActivePlaneLease() const noexcept
{
  return payload_->source_active_plane_lease;
}

std::uint8_t
MultiCameraLandmarkSelectionTrace::sourceCameraId() const noexcept
{
  return payload_->source_camera_id;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
sourceCameraBatchSequence() const noexcept
{
  return payload_->source_camera_batch_sequence;
}

double
MultiCameraLandmarkSelectionTrace::sourceTimestampSec() const noexcept
{
  return payload_->source_timestamp_sec;
}

const Eigen::Matrix<double, 6, 6> &
MultiCameraLandmarkSelectionTrace::
sourcePoseCovariance() const noexcept
{
  return payload_->source_pose_covariance;
}

const ReferenceOnlyMigrationFrameProvenance &
MultiCameraLandmarkSelectionTrace::
targetFrameProvenance() const noexcept
{
  return *payload_->target_frame;
}

std::uint8_t
MultiCameraLandmarkSelectionTrace::targetCameraId() const noexcept
{
  return payload_->target_camera_id;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
targetCameraBatchSequence() const noexcept
{
  return payload_->target_camera_batch_sequence;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
targetRuntimePublicationSequence() const noexcept
{
  return payload_->target_runtime_publication_sequence;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
targetActiveStateSnapshotRevision() const noexcept
{
  return payload_->target_active_state_snapshot_revision;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
targetObservationGeneration() const noexcept
{
  return payload_->target_observation_generation;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::
targetImageGeneration() const noexcept
{
  return payload_->target_image_generation;
}

double
MultiCameraLandmarkSelectionTrace::targetTimestampSec() const noexcept
{
  return payload_->target_timestamp_sec;
}

const AdaptiveVisualPolicyPlanCapability &
MultiCameraLandmarkSelectionTrace::
policyCapability() const noexcept
{
  return *payload_->policy_capability;
}

std::uint64_t
MultiCameraLandmarkSelectionTrace::policyBaseRevision() const noexcept
{
  return payload_->policy_base_revision;
}

double
MultiCameraLandmarkSelectionTrace::healthScore() const noexcept
{
  return payload_->health_score;
}

double
MultiCameraLandmarkSelectionTrace::
lidarComplementScore() const noexcept
{
  return payload_->lidar_complement_score;
}

std::size_t
MultiCameraLandmarkSelectionTrace::candidateBudget() const noexcept
{
  return payload_->candidate_budget;
}

std::size_t
MultiCameraLandmarkSelectionTrace::patchBudget() const noexcept
{
  return payload_->patch_budget;
}

bool MultiCameraLandmarkSelectionTrace::
configuredRelativeAffineExperiment() const noexcept
{
  return payload_->config.allow_relative_affine_experiment;
}

double MultiCameraLandmarkSelectionTrace::
configuredMaximumSourceAgeSec() const noexcept
{
  return payload_->config.maximum_source_age_sec;
}

std::size_t MultiCameraLandmarkSelectionTrace::
configuredMaximumCandidateEvaluationsPerBatch() const noexcept
{
  return
    payload_->config.maximum_candidate_evaluations_per_batch;
}

std::uint64_t MultiCameraLandmarkSelectionTrace::
configuredMaximumAcceptedTargetGenerations() const noexcept
{
  return
    payload_->config.maximum_accepted_target_generations;
}

MultiCameraLandmarkPhotometricPath
MultiCameraLandmarkSelectionTrace::photometricPath() const noexcept
{
  return payload_->photometric_path;
}

const char *
MultiCameraLandmarkSelectionTrace::photometricPathLabel() const noexcept
{
  return multiCameraLandmarkPhotometricPathName(
    payload_->photometric_path);
}

bool
MultiCameraLandmarkSelectionTrace::crossCameraComparable() const noexcept
{
  return false;
}

bool MultiCameraLandmarkSelectionTrace::experimental() const noexcept
{
  return payload_->experimental;
}

MultiCameraLandmarkShadowDecision::
MultiCameraLandmarkShadowDecision(
  const MultiCameraLandmarkShadowDecisionStatus status,
  const LandmarkId landmark_id,
  const ObservationId source_observation_id,
  const std::uint8_t source_camera_id,
  const std::uint8_t target_camera_id,
  const MultiCameraLandmarkPhotometricPath photometric_path,
  const double health_score,
  const double lidar_complement_score,
  const std::size_t candidate_budget,
  const std::size_t patch_budget,
  std::shared_ptr<const MultiCameraLandmarkSelectionTrace>
    trace)
: status_(status),
  landmark_id_(landmark_id),
  source_observation_id_(source_observation_id),
  source_camera_id_(source_camera_id),
  target_camera_id_(target_camera_id),
  photometric_path_(photometric_path),
  health_score_(health_score),
  lidar_complement_score_(lidar_complement_score),
  candidate_budget_(candidate_budget),
  patch_budget_(patch_budget),
  trace_(std::move(trace))
{
}

MultiCameraLandmarkShadowDecisionStatus
MultiCameraLandmarkShadowDecision::status() const noexcept
{
  return status_;
}

LandmarkId
MultiCameraLandmarkShadowDecision::landmarkId() const noexcept
{
  return landmark_id_;
}

ObservationId
MultiCameraLandmarkShadowDecision::sourceObservationId() const noexcept
{
  return source_observation_id_;
}

std::uint8_t
MultiCameraLandmarkShadowDecision::sourceCameraId() const noexcept
{
  return source_camera_id_;
}

std::uint8_t
MultiCameraLandmarkShadowDecision::targetCameraId() const noexcept
{
  return target_camera_id_;
}

MultiCameraLandmarkPhotometricPath
MultiCameraLandmarkShadowDecision::photometricPath() const noexcept
{
  return photometric_path_;
}

const char *
MultiCameraLandmarkShadowDecision::photometricPathLabel() const noexcept
{
  return multiCameraLandmarkPhotometricPathName(
    photometric_path_);
}

double MultiCameraLandmarkShadowDecision::healthScore() const noexcept
{
  return health_score_;
}

double
MultiCameraLandmarkShadowDecision::lidarComplementScore() const noexcept
{
  return lidar_complement_score_;
}

std::size_t
MultiCameraLandmarkShadowDecision::candidateBudget() const noexcept
{
  return candidate_budget_;
}

std::size_t
MultiCameraLandmarkShadowDecision::patchBudget() const noexcept
{
  return patch_budget_;
}

bool
MultiCameraLandmarkShadowDecision::hasSelectionTrace() const noexcept
{
  return trace_ != nullptr && trace_->valid();
}

const MultiCameraLandmarkSelectionTrace *
MultiCameraLandmarkShadowDecision::selectionTrace() const noexcept
{
  return hasSelectionTrace() ? trace_.get() : nullptr;
}

MultiCameraLandmarkShadowBatchResult::
MultiCameraLandmarkShadowBatchResult(
  const MultiCameraLandmarkShadowBatchStatus status)
: status_(status)
{
}

MultiCameraLandmarkShadowBatchStatus
MultiCameraLandmarkShadowBatchResult::status() const noexcept
{
  return status_;
}

bool MultiCameraLandmarkShadowBatchResult::accepted() const noexcept
{
  return status_ == MultiCameraLandmarkShadowBatchStatus::kAccepted;
}

std::uint64_t
MultiCameraLandmarkShadowBatchResult::
configurationGeneration() const noexcept
{
  return configuration_generation_;
}

std::uint64_t
MultiCameraLandmarkShadowBatchResult::
acceptedTargetGeneration() const noexcept
{
  return accepted_target_generation_;
}

std::uint64_t
MultiCameraLandmarkShadowBatchResult::
targetCameraBatchSequence() const noexcept
{
  return target_camera_batch_sequence_;
}

std::uint64_t
MultiCameraLandmarkShadowBatchResult::
targetRuntimePublicationSequence() const noexcept
{
  return target_runtime_publication_sequence_;
}

MapRevision
MultiCameraLandmarkShadowBatchResult::storeRevision() const noexcept
{
  return store_revision_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
inspectedSourceObservationCount() const noexcept
{
  return inspected_source_observation_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
storeAuthorityViolationCount() const noexcept
{
  return store_authority_violation_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
skippedReferenceOnlySourceCount() const noexcept
{
  return skipped_reference_only_source_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
skippedInactivePlaneSourceCount() const noexcept
{
  return skipped_inactive_plane_source_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
skippedTombstonedSourceCount() const noexcept
{
  return skipped_tombstoned_source_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
skippedSamplingGeometryCount() const noexcept
{
  return skipped_sampling_geometry_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
skippedSourceOrderCount() const noexcept
{
  return skipped_source_order_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
skippedSourceRuntimeAuthorityCount() const noexcept
{
  return skipped_source_runtime_authority_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
skippedSourceAgeCount() const noexcept
{
  return skipped_source_age_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
candidateEvaluationCount() const noexcept
{
  return candidate_evaluation_count_;
}

std::size_t
MultiCameraLandmarkShadowBatchResult::
strictPhotometricRejectedCount() const noexcept
{
  return strict_photometric_rejected_count_;
}

const std::vector<MultiCameraLandmarkShadowDecision> &
MultiCameraLandmarkShadowBatchResult::decisions() const noexcept
{
  return decisions_;
}

MultiCameraLandmarkLinearizationShadow::
MultiCameraLandmarkLinearizationShadow(
  SharedLandmarkShadowSession & session,
  const CameraObservationBatchAuthorityIdentity &
    expected_batch_authority,
  const ReferenceOnlyMigrationRuntimePublisherIdentity &
    expected_runtime_authority,
  const AdaptiveVisualPolicy & expected_policy,
  MultiCameraLandmarkLinearizationShadowConfig config)
: implementation_(
    std::make_unique<Implementation>(
      session, expected_batch_authority,
      expected_runtime_authority, expected_policy,
      std::move(config)))
{
}

MultiCameraLandmarkLinearizationShadow::
~MultiCameraLandmarkLinearizationShadow() = default;

bool MultiCameraLandmarkLinearizationShadow::active() const noexcept
{
  return
    implementation_ != nullptr &&
    implementation_->config.enabled &&
    implementation_->config_valid &&
    implementation_->authority_configuration_valid &&
    !implementation_->generation_exhausted;
}

std::uint64_t
MultiCameraLandmarkLinearizationShadow::
configurationGeneration() const noexcept
{
  return
    implementation_ != nullptr ?
    implementation_->configuration_generation : 0U;
}

std::uint64_t
MultiCameraLandmarkLinearizationShadow::
acceptedTargetGeneration() const noexcept
{
  return
    implementation_ != nullptr ?
    implementation_->accepted_target_generation : 0U;
}

MultiCameraLandmarkShadowBatchResult
MultiCameraLandmarkLinearizationShadow::processBatch(
  const ReferenceOnlyMigrationRuntimeBundle & runtime,
  const AdaptiveVisualPolicyPlanCapability & capability)
{
  const auto makeResult =
    [this](const MultiCameraLandmarkShadowBatchStatus status) {
      MultiCameraLandmarkShadowBatchResult result(status);
      if (implementation_ != nullptr)
      {
        result.configuration_generation_ =
          implementation_->configuration_generation;
        result.accepted_target_generation_ =
          implementation_->accepted_target_generation;
      }
      return result;
    };
  if (
    implementation_ == nullptr ||
    !implementation_->config.enabled)
  {
    return makeResult(
      MultiCameraLandmarkShadowBatchStatus::kDisabled);
  }
  if (!implementation_->config_valid)
  {
    return makeResult(
      MultiCameraLandmarkShadowBatchStatus::
      kInvalidConfiguration);
  }
  if (!implementation_->authority_configuration_valid)
  {
    return makeResult(
      MultiCameraLandmarkShadowBatchStatus::
      kInvalidAuthorityConfiguration);
  }
  if (
    implementation_->generation_exhausted ||
    implementation_->accepted_target_generation >=
      implementation_->config.
      maximum_accepted_target_generations)
  {
    implementation_->generation_exhausted = true;
    return makeResult(
      MultiCameraLandmarkShadowBatchStatus::
      kGenerationExhausted);
  }

  if (!runtime.valid())
  {
    return makeResult(
      MultiCameraLandmarkShadowBatchStatus::
      kRuntimeAuthorityRejected);
  }
  const ValidatedBatchRuntimeComposition & runtime_composition =
    runtime.validatedComposition();
  if (
    !runtime_composition.valid() ||
    !runtime_composition.issuedByBatch(
      implementation_->expected_batch_authority))
  {
    return makeResult(
      MultiCameraLandmarkShadowBatchStatus::
      kTargetBatchAuthorityRejected);
  }
  MultiCameraLandmarkShadowBatchResult result =
    makeResult(MultiCameraLandmarkShadowBatchStatus::kAccepted);
  result.target_camera_batch_sequence_ =
    runtime_composition.cameraBatchSequence();

  if (
    !runtime.issuedBy(
      implementation_->expected_runtime_authority) ||
    !runtime_composition.issuedByRuntime(
      implementation_->expected_runtime_authority) ||
    !runtime_composition.accepts(runtime))
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kRuntimeAuthorityRejected;
    return result;
  }
  result.target_runtime_publication_sequence_ =
    runtime.publicationSequence();

  if (
    !capability.valid() ||
    implementation_->expected_policy == nullptr ||
    !implementation_->expected_policy->accepts(capability))
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kPolicyCapabilityRejected;
    return result;
  }
  const AdaptiveVisualPolicyPlan & plan =
    capability.decisionPlan();
  if (
    !plan.adaptive_enabled ||
    !plan.lidar_information_valid ||
    plan.reason != AdaptivePolicyReason::kReady)
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kPolicyCapabilityRejected;
    return result;
  }
  const ReferenceOnlyMigrationRuntimeBundle & policy_runtime =
    capability.runtimeBundle();
  if (!policy_runtime.valid())
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kTargetCompositionRejected;
    return result;
  }
  const ValidatedBatchRuntimeComposition &
    policy_composition =
    policy_runtime.validatedComposition();
  if (
    !policy_runtime.issuedBy(
      implementation_->expected_runtime_authority) ||
    !policy_composition.issuedBy(
      implementation_->expected_batch_authority,
      implementation_->expected_runtime_authority) ||
    !policy_composition.accepts(policy_runtime) ||
    policy_runtime.publicationSequence() !=
      runtime.publicationSequence() ||
    policy_runtime.cameraBatchReceipt().batchSequence() !=
      runtime_composition.cameraBatchSequence() ||
    policy_runtime.activeStateSnapshotRevision() !=
      runtime.activeStateSnapshotRevision() ||
    capability.runtimePublicationSequence() !=
      runtime.publicationSequence() ||
    capability.cameraBatchSequence() !=
      runtime_composition.cameraBatchSequence())
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kTargetCompositionRejected;
    return result;
  }

  std::array<Implementation::TargetCamera, 4U> targets;
  for (std::size_t camera_id = 0U;
       camera_id < targets.size(); ++camera_id)
  {
    if (!runtime_composition.cameraPresent(
        static_cast<std::uint8_t>(camera_id)))
    {
      continue;
    }
    if (
      runtime_composition.photometricStatus(
        static_cast<std::uint8_t>(camera_id)) !=
      PhotometricStatus::kValid)
    {
      result.status_ =
        MultiCameraLandmarkShadowBatchStatus::
        kTargetCompositionRejected;
      return result;
    }
    const auto runtime_frame =
      runtime.frameProvenance(
      static_cast<std::uint8_t>(camera_id));
    const auto policy_frame =
      policy_runtime.frameProvenance(
      static_cast<std::uint8_t>(camera_id));
    if (
      !runtime_frame.has_value() ||
      !policy_frame.has_value() ||
      !runtime_frame->issuedBy(
        implementation_->expected_runtime_authority) ||
      !policy_frame->issuedBy(
        implementation_->expected_runtime_authority) ||
      !runtime_frame->accepts(runtime_composition) ||
      !policy_frame->accepts(policy_composition) ||
      !sameFrameBinding(*runtime_frame, *policy_frame) ||
      runtime_frame->cameraBatchSequence() !=
        result.target_camera_batch_sequence_ ||
      runtime_frame->runtimePublicationSequence() !=
        result.target_runtime_publication_sequence_ ||
      runtime_frame->activeStateSnapshotRevision() !=
        runtime.activeStateSnapshotRevision())
    {
      result.status_ =
        MultiCameraLandmarkShadowBatchStatus::
        kTargetCompositionRejected;
      return result;
    }
    const AdaptiveCameraDecision & decision =
      plan.cameras[camera_id];
    if (
      decision.enabled &&
      (!decision.available ||
       decision.isolated ||
       !validPolicyDecision(decision)))
    {
      result.status_ =
        MultiCameraLandmarkShadowBatchStatus::
        kTargetCompositionRejected;
      return result;
    }
    if (
      decision.enabled &&
      decision.available &&
      !decision.isolated)
    {
      targets[camera_id].frame = *runtime_frame;
      targets[camera_id].health = decision.health;
      targets[camera_id].complement =
        decision.complement_score;
      targets[camera_id].candidate_budget =
        decision.candidate_budget;
      targets[camera_id].patch_budget =
        decision.patch_budget;
    }
  }

  if (
    result.target_camera_batch_sequence_ <=
      implementation_->
      last_target_camera_batch_sequence ||
    result.target_runtime_publication_sequence_ <=
      implementation_->
      last_target_runtime_publication_sequence)
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kTargetOrderRejected;
    return result;
  }

  if (
    implementation_->session == nullptr ||
    !implementation_->session->active())
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kSessionUnavailable;
    return result;
  }
  const std::optional<ReadSnapshot> snapshot =
    implementation_->session->
    snapshotForMultiCameraLandmarkLinearizationShadow();
  if (!snapshot.has_value())
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kMissingCurrentActivePlaneSnapshot;
    return result;
  }
  if (!snapshot->active_planes.valid())
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kStaleCurrentActivePlaneSnapshot;
    return result;
  }
  result.store_revision_ = snapshot->store_revision;
  if (!snapshot->store_identity.valid())
  {
    result.store_authority_violation_count_ = 1U;
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kStoreAuthorityViolation;
    return result;
  }

  std::vector<Implementation::Source> sources;
  for (const LandmarkSnapshot & landmark : snapshot->landmarks)
  {
    const std::optional<ActivePlaneIdentityLease>
      active_plane_lease =
      snapshot->active_planes.leaseFor(
      landmark.geometry_version);
    for (const ObservationSnapshot & observation :
         landmark.observations)
    {
      ++result.inspected_source_observation_count_;
      if (
        !observation.store_receipt.has_value() ||
        !observation.store_receipt->valid() ||
        !observation.store_receipt->issuedBy(
          snapshot->store_identity) ||
        !observation.store_receipt->accepts(
          landmark.landmark_id, observation))
      {
        ++result.store_authority_violation_count_;
        continue;
      }
      if (observation.reference_only)
      {
        ++result.skipped_reference_only_source_count_;
        continue;
      }
      if (landmark.lifecycle == LandmarkLifecycle::kTombstoned)
      {
        ++result.skipped_tombstoned_source_count_;
        continue;
      }
      if (
        !sameGeometryVersion(
          observation.sampling_geometry_version,
          landmark.geometry_version) ||
        !sameGeometryVersion(
          observation.store_receipt->
          samplingGeometryVersion(),
          landmark.geometry_version))
      {
        ++result.skipped_sampling_geometry_count_;
        continue;
      }
      if (!active_plane_lease.has_value())
      {
        ++result.skipped_inactive_plane_source_count_;
        continue;
      }
      if (
        observation.accepted_camera_batch_sequence >=
        result.target_camera_batch_sequence_)
      {
        ++result.skipped_source_order_count_;
        continue;
      }
      if (
        observation.frame_provenance == nullptr ||
        !observation.frame_provenance->valid() ||
        !observation.frame_provenance->issuedBy(
          implementation_->expected_runtime_authority) ||
        observation.frame_provenance->
          cameraBatchSequence() !=
          observation.accepted_camera_batch_sequence ||
        observation.frame_provenance->
          runtimePublicationSequence() !=
          observation.pose_runtime_publication_sequence ||
        observation.frame_provenance->
          activeStateSnapshotRevision() !=
          observation.pose_active_state_snapshot_revision ||
        observation.frame_provenance->cameraId() !=
          observation.camera_id ||
        observation.frame_provenance->
          exposureTimestampSec() !=
          observation.exposure_timestamp_sec)
      {
        ++result.skipped_source_runtime_authority_count_;
        continue;
      }
      sources.push_back(
        Implementation::Source{
          &landmark, &observation, *active_plane_lease});
    }
  }
  if (result.store_authority_violation_count_ != 0U)
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kStoreAuthorityViolation;
    return result;
  }
  std::sort(
    sources.begin(), sources.end(),
    [](const Implementation::Source & lhs,
       const Implementation::Source & rhs) {
      if (
        lhs.landmark->landmark_id !=
        rhs.landmark->landmark_id)
      {
        return
          lhs.landmark->landmark_id <
          rhs.landmark->landmark_id;
      }
      if (
        lhs.observation->observation_id !=
        rhs.observation->observation_id)
      {
        return
          lhs.observation->observation_id <
          rhs.observation->observation_id;
      }
      return
        lhs.observation->camera_id <
        rhs.observation->camera_id;
    });

  std::size_t target_count = 0U;
  for (const Implementation::TargetCamera & target : targets)
  {
    target_count += target.frame.has_value() ? 1U : 0U;
  }
  if (
    target_count != 0U &&
    sources.size() >
      implementation_->config.
      maximum_candidate_evaluations_per_batch /
      target_count)
  {
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kCandidateBudgetExceeded;
    return result;
  }
  result.candidate_evaluation_count_ =
    sources.size() * target_count;

  const auto betterCandidate = [](
      const Implementation::Candidate & lhs,
      const Implementation::Candidate & rhs) {
      const int lhs_path = pathPriority(lhs.path);
      const int rhs_path = pathPriority(rhs.path);
      if (lhs_path != rhs_path)
      {
        return lhs_path < rhs_path;
      }
      if (lhs.target->health != rhs.target->health)
      {
        return lhs.target->health > rhs.target->health;
      }
      if (lhs.target->complement != rhs.target->complement)
      {
        return
          lhs.target->complement >
          rhs.target->complement;
      }
      if (lhs.age_sec != rhs.age_sec)
      {
        return lhs.age_sec < rhs.age_sec;
      }
      const double lhs_covariance =
        lhs.observation->pose_covariance.trace();
      const double rhs_covariance =
        rhs.observation->pose_covariance.trace();
      if (lhs_covariance != rhs_covariance)
      {
        return lhs_covariance < rhs_covariance;
      }
      if (
        lhs.observation->accepted_camera_batch_sequence !=
        rhs.observation->accepted_camera_batch_sequence)
      {
        return
          lhs.observation->accepted_camera_batch_sequence >
          rhs.observation->accepted_camera_batch_sequence;
      }
      if (
        lhs.observation->observation_id !=
        rhs.observation->observation_id)
      {
        return
          lhs.observation->observation_id <
          rhs.observation->observation_id;
      }
      if (
        lhs.observation->camera_id !=
        rhs.observation->camera_id)
      {
        return
          lhs.observation->camera_id <
          rhs.observation->camera_id;
      }
      return
        lhs.target_camera_id <
        rhs.target_camera_id;
    };

  std::map<LandmarkId, Implementation::Candidate>
    selected_by_landmark;
  for (const Implementation::Source & source : sources)
  {
    for (std::size_t camera_id = 0U;
         camera_id < targets.size(); ++camera_id)
    {
      const Implementation::TargetCamera & target =
        targets[camera_id];
      if (!target.frame.has_value())
      {
        continue;
      }
      const double age_sec =
        target.frame->exposureTimestampSec() -
        source.observation->exposure_timestamp_sec;
      if (
        !std::isfinite(age_sec) ||
        age_sec < 0.0 ||
        age_sec >
          implementation_->config.maximum_source_age_sec)
      {
        ++result.skipped_source_age_count_;
        continue;
      }
      Implementation::Candidate candidate;
      candidate.landmark = source.landmark;
      candidate.observation = source.observation;
      candidate.target = &target;
      candidate.active_plane_lease =
        source.active_plane_lease;
      candidate.target_camera_id =
        static_cast<std::uint8_t>(camera_id);
      candidate.age_sec = age_sec;
      if (
        source.observation->camera_id ==
        candidate.target_camera_id)
      {
        candidate.path =
          MultiCameraLandmarkPhotometricPath::
          kSameCameraLegacyUnattested;
      }
      else if (
        implementation_->config.
        allow_relative_affine_experiment)
      {
        candidate.path =
          MultiCameraLandmarkPhotometricPath::
          kCrossCameraRelativeAffineExperimental;
      }
      else
      {
        candidate.path =
          MultiCameraLandmarkPhotometricPath::
          kCrossCameraCalibrated;
      }
      const auto previous =
        selected_by_landmark.find(
        source.landmark->landmark_id);
      if (
        previous == selected_by_landmark.end() ||
        betterCandidate(candidate, previous->second))
      {
        selected_by_landmark[
          source.landmark->landmark_id] = candidate;
      }
    }
  }

  std::uint64_t next_generation =
    implementation_->accepted_target_generation;
  if (!checkedIncrement(next_generation))
  {
    implementation_->generation_exhausted = true;
    result.status_ =
      MultiCameraLandmarkShadowBatchStatus::
      kGenerationExhausted;
    return result;
  }

  result.decisions_.reserve(selected_by_landmark.size());
  for (const auto & selected : selected_by_landmark)
  {
    const Implementation::Candidate & candidate =
      selected.second;
    const ObservationSnapshot & observation =
      *candidate.observation;
    const Implementation::TargetCamera & target =
      *candidate.target;
    if (
      candidate.path ==
      MultiCameraLandmarkPhotometricPath::
      kCrossCameraCalibrated)
    {
      ++result.strict_photometric_rejected_count_;
      MultiCameraLandmarkShadowDecision decision(
        MultiCameraLandmarkShadowDecisionStatus::
        kStrictCrossCameraEvidenceUnavailable,
        candidate.landmark->landmark_id,
        observation.observation_id,
        observation.camera_id,
        candidate.target_camera_id,
        candidate.path,
        target.health,
        target.complement,
        target.candidate_budget,
        target.patch_budget,
        nullptr);
      result.decisions_.push_back(std::move(decision));
      continue;
    }

    auto payload =
      std::make_shared<
      MultiCameraLandmarkSelectionTrace::Payload>();
    payload->owner = implementation_->owner;
    payload->configuration_generation =
      implementation_->configuration_generation;
    payload->accepted_target_generation =
      next_generation;
    payload->config = implementation_->config;
    payload->landmark_id =
      candidate.landmark->landmark_id;
    payload->source_observation_id =
      observation.observation_id;
    payload->source_observation_revision =
      observation.observation_revision;
    payload->source_store_receipt =
      observation.store_receipt;
    payload->selected_store_revision =
      snapshot->store_revision;
    payload->source_sampling_geometry_version =
      observation.sampling_geometry_version;
    payload->source_active_plane_lease =
      candidate.active_plane_lease;
    payload->source_camera_id = observation.camera_id;
    payload->source_camera_batch_sequence =
      observation.accepted_camera_batch_sequence;
    payload->source_timestamp_sec =
      observation.exposure_timestamp_sec;
    payload->source_pose_covariance =
      observation.pose_covariance;
    payload->target_frame = target.frame;
    payload->target_camera_id =
      candidate.target_camera_id;
    payload->target_camera_batch_sequence =
      result.target_camera_batch_sequence_;
    payload->target_runtime_publication_sequence =
      result.target_runtime_publication_sequence_;
    payload->target_active_state_snapshot_revision =
      runtime.activeStateSnapshotRevision();
    payload->target_observation_generation =
      target.frame->observationGeneration();
    payload->target_image_generation =
      target.frame->imageGeneration();
    payload->target_timestamp_sec =
      target.frame->exposureTimestampSec();
    payload->policy_capability = capability;
    payload->policy_base_revision =
      capability.basePolicyRevision();
    payload->health_score = target.health;
    payload->lidar_complement_score =
      target.complement;
    payload->candidate_budget =
      target.candidate_budget;
    payload->patch_budget = target.patch_budget;
    payload->photometric_path = candidate.path;
    payload->experimental =
      candidate.path ==
      MultiCameraLandmarkPhotometricPath::
      kCrossCameraRelativeAffineExperimental;
    const auto trace =
      std::shared_ptr<
      const MultiCameraLandmarkSelectionTrace>(
      new MultiCameraLandmarkSelectionTrace(
        std::move(payload)));
    MultiCameraLandmarkShadowDecision decision(
      MultiCameraLandmarkShadowDecisionStatus::
      kSelectionTracedLinearizationUnavailableMissingPhotometricNoise,
      candidate.landmark->landmark_id,
      observation.observation_id,
      observation.camera_id,
      candidate.target_camera_id,
      candidate.path,
      target.health,
      target.complement,
      target.candidate_budget,
      target.patch_budget,
      trace);
    result.decisions_.push_back(std::move(decision));
  }

  implementation_->accepted_target_generation =
    next_generation;
  implementation_->last_target_camera_batch_sequence =
    result.target_camera_batch_sequence_;
  implementation_->last_target_runtime_publication_sequence =
    result.target_runtime_publication_sequence_;
  if (
    implementation_->accepted_target_generation >=
    implementation_->config.
      maximum_accepted_target_generations)
  {
    implementation_->generation_exhausted = true;
  }
  result.accepted_target_generation_ =
    implementation_->accepted_target_generation;
  result.status_ =
    MultiCameraLandmarkShadowBatchStatus::kAccepted;
  return result;
}

bool MultiCameraLandmarkLinearizationShadow::accepts(
  const MultiCameraLandmarkSelectionTrace & trace) const
{
  if (
    implementation_ == nullptr ||
    !trace.valid() ||
    trace.payload_->owner == nullptr ||
    implementation_->owner == nullptr ||
    trace.payload_->owner.get() !=
      implementation_->owner.get() ||
    trace.payload_->configuration_generation !=
      implementation_->configuration_generation ||
    trace.payload_->config.enabled !=
      implementation_->config.enabled ||
    trace.payload_->config.
      allow_relative_affine_experiment !=
      implementation_->config.
      allow_relative_affine_experiment ||
    trace.payload_->config.maximum_source_age_sec !=
      implementation_->config.maximum_source_age_sec ||
    trace.payload_->config.
      maximum_candidate_evaluations_per_batch !=
      implementation_->config.
      maximum_candidate_evaluations_per_batch ||
    trace.payload_->config.
      maximum_accepted_target_generations !=
      implementation_->config.
      maximum_accepted_target_generations ||
    !trace.payload_->target_frame->issuedBy(
      implementation_->expected_runtime_authority) ||
    implementation_->expected_policy == nullptr ||
    !implementation_->expected_policy->accepts(
      *trace.payload_->policy_capability))
  {
    return false;
  }
  const std::optional<ReadSnapshot> snapshot =
    implementation_->session->
    snapshotForMultiCameraLandmarkLinearizationShadow();
  if (
    !snapshot.has_value() ||
    !snapshot->store_identity.valid() ||
    !snapshot->active_planes.valid() ||
    snapshot->store_revision.value <
      trace.payload_->selected_store_revision.value)
  {
    return false;
  }
  const LandmarkSnapshot * const landmark =
    snapshot->find(trace.payload_->landmark_id);
  if (
    landmark == nullptr ||
    landmark->lifecycle == LandmarkLifecycle::kTombstoned ||
    !snapshot->active_planes.contains(
      trace.payload_->source_active_plane_lease) ||
    !sameGeometryVersion(
      landmark->geometry_version,
      trace.payload_->source_sampling_geometry_version))
  {
    return false;
  }
  const auto observation = std::find_if(
    landmark->observations.begin(),
    landmark->observations.end(),
    [&trace](const ObservationSnapshot & candidate) {
      return
        candidate.observation_id ==
        trace.payload_->source_observation_id;
    });
  return
    observation != landmark->observations.end() &&
    !observation->reference_only &&
    trace.payload_->source_store_receipt->issuedBy(
      snapshot->store_identity) &&
    trace.payload_->source_store_receipt->accepts(
      landmark->landmark_id, *observation) &&
    sameGeometryVersion(
      observation->sampling_geometry_version,
      landmark->geometry_version) &&
    observation->frame_provenance != nullptr &&
    observation->frame_provenance->issuedBy(
      implementation_->expected_runtime_authority) &&
    observation->accepted_camera_batch_sequence <
      trace.payload_->target_camera_batch_sequence;
}

const char * multiCameraLandmarkShadowBatchStatusName(
  const MultiCameraLandmarkShadowBatchStatus status) noexcept
{
  switch (status)
  {
    case MultiCameraLandmarkShadowBatchStatus::kDisabled:
      return "disabled";
    case MultiCameraLandmarkShadowBatchStatus::kInvalidConfiguration:
      return "invalid_configuration";
    case MultiCameraLandmarkShadowBatchStatus::
      kInvalidAuthorityConfiguration:
      return "invalid_authority_configuration";
    case MultiCameraLandmarkShadowBatchStatus::
      kTargetBatchAuthorityRejected:
      return "target_batch_authority_rejected";
    case MultiCameraLandmarkShadowBatchStatus::
      kRuntimeAuthorityRejected:
      return "runtime_authority_rejected";
    case MultiCameraLandmarkShadowBatchStatus::
      kPolicyCapabilityRejected:
      return "policy_capability_rejected";
    case MultiCameraLandmarkShadowBatchStatus::
      kTargetCompositionRejected:
      return "target_composition_rejected";
    case MultiCameraLandmarkShadowBatchStatus::
      kTargetOrderRejected:
      return "target_order_rejected";
    case MultiCameraLandmarkShadowBatchStatus::kSessionUnavailable:
      return "session_unavailable";
    case MultiCameraLandmarkShadowBatchStatus::
      kMissingCurrentActivePlaneSnapshot:
      return "missing_current_active_plane_snapshot";
    case MultiCameraLandmarkShadowBatchStatus::
      kStaleCurrentActivePlaneSnapshot:
      return "stale_current_active_plane_snapshot";
    case MultiCameraLandmarkShadowBatchStatus::
      kStoreAuthorityViolation:
      return "store_authority_violation";
    case MultiCameraLandmarkShadowBatchStatus::
      kCandidateBudgetExceeded:
      return "candidate_budget_exceeded";
    case MultiCameraLandmarkShadowBatchStatus::
      kGenerationExhausted:
      return "generation_exhausted";
    case MultiCameraLandmarkShadowBatchStatus::kAccepted:
      return "accepted";
  }
  return "unknown";
}

const char * multiCameraLandmarkShadowDecisionStatusName(
  const MultiCameraLandmarkShadowDecisionStatus status) noexcept
{
  switch (status)
  {
    case MultiCameraLandmarkShadowDecisionStatus::
      kSelectionTracedLinearizationUnavailableMissingPhotometricNoise:
      return
        "selection_traced_linearization_unavailable_"
        "missing_photometric_noise";
    case MultiCameraLandmarkShadowDecisionStatus::
      kStrictCrossCameraEvidenceUnavailable:
      return "strict_cross_camera_evidence_unavailable";
  }
  return "unknown";
}

const char * multiCameraLandmarkPhotometricPathName(
  const MultiCameraLandmarkPhotometricPath path) noexcept
{
  switch (path)
  {
    case MultiCameraLandmarkPhotometricPath::
      kSameCameraLegacyUnattested:
      return "legacy_unattested";
    case MultiCameraLandmarkPhotometricPath::
      kCrossCameraCalibrated:
      return "calibrated_evidence_unavailable";
    case MultiCameraLandmarkPhotometricPath::
      kCrossCameraRelativeAffineExperimental:
      return "experimental_non_radiometric";
  }
  return "unknown";
}

}  // namespace livo
