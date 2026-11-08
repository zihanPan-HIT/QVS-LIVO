#ifndef LIVO_ADAPTIVE_VISUAL_POLICY_H_
#define LIVO_ADAPTIVE_VISUAL_POLICY_H_

#include <Eigen/Core>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>

struct CameraObservationBatch;

namespace livo
{

constexpr std::size_t kAdaptiveVisualCameraCount = 4U;
constexpr std::uint32_t kAdaptiveVisualPolicyConfigRevision = 7U;

class ReferenceOnlyMigrationRuntimeBundle;
class AdaptiveVisualSelectionAuthority;
class AdaptiveVisualSelectionReceipt;
class AdaptiveVisualRuntimeAttemptPermit;
class CameraObservationBatchReceipt;

enum class AdaptiveHealthMetric : std::uint32_t
{
  kCandidateCount = 1U << 0,
  kTrackedCount = 1U << 1,
  kPhotometricResidual = 1U << 2,
  kSaturationFraction = 1U << 3,
  kDarkFraction = 1U << 4,
  kMeanGradient = 1U << 5,
  kSpatialCoverage = 1U << 6,
  kVisualInformation = 1U << 7,
  kInlierCount = 1U << 8,
  kInlierRatio = 1U << 9,
  kNis = 1U << 10,
  kNormalizedPoseCorrection = 1U << 11,
  kRobustInformationGain = 1U << 12,
};

constexpr std::uint32_t adaptiveMetricMask(const AdaptiveHealthMetric metric)
{
  return static_cast<std::uint32_t>(metric);
}

constexpr std::uint32_t kAdaptiveImageHealthMetricMask =
  adaptiveMetricMask(AdaptiveHealthMetric::kSaturationFraction) |
  adaptiveMetricMask(AdaptiveHealthMetric::kDarkFraction) |
  adaptiveMetricMask(AdaptiveHealthMetric::kMeanGradient);

constexpr std::uint32_t kAdaptiveTrackingHealthMetricMask =
  adaptiveMetricMask(AdaptiveHealthMetric::kCandidateCount) |
  adaptiveMetricMask(AdaptiveHealthMetric::kTrackedCount) |
  adaptiveMetricMask(AdaptiveHealthMetric::kPhotometricResidual) |
  adaptiveMetricMask(AdaptiveHealthMetric::kSpatialCoverage) |
  adaptiveMetricMask(AdaptiveHealthMetric::kVisualInformation);

constexpr std::uint32_t kAdaptiveFirstVersionMetricMask =
  kAdaptiveImageHealthMetricMask |
  kAdaptiveTrackingHealthMetricMask;
constexpr std::uint32_t kAdaptiveRobustAdmissionMetricMask =
  adaptiveMetricMask(AdaptiveHealthMetric::kInlierCount) |
  adaptiveMetricMask(AdaptiveHealthMetric::kInlierRatio) |
  adaptiveMetricMask(AdaptiveHealthMetric::kNis) |
  adaptiveMetricMask(AdaptiveHealthMetric::kNormalizedPoseCorrection) |
  adaptiveMetricMask(AdaptiveHealthMetric::kRobustInformationGain);

struct AdaptiveVisualPolicyConfig
{
  bool enabled{false};

  // L and C are normalized as S.transpose() * information * S. This keeps
  // rotation (rad) and translation (m) from being compared as equal units.
  double rotation_scale_rad{0.017453292519943295};
  double translation_scale_m{0.10};
  double weak_relative_eigenvalue{0.05};
  double weak_absolute_eigenvalue{1.0e-9};

  double health_off_threshold{0.25};
  double health_on_threshold{0.45};
  double hard_reject_health{0.08};
  std::size_t off_hysteresis_frames{3U};
  std::size_t on_hysteresis_frames{5U};
  double minimum_probation_duration_sec{0.5};

  double covariance_health_floor{0.10};
  double maximum_covariance_scale{10.0};
  double complement_score_gain{2.0};
  double selection_switch_margin{0.10};
  // One confirmation is one trusted, revision-matched candidate capability
  // commit with a strictly increasing authoritative LiDAR generation.
  std::size_t selection_challenger_frames{3U};

  std::size_t minimum_inliers{13U};
  double minimum_inlier_ratio{0.50};
  double nis_chi_square_scale{1.0};
  double maximum_normalized_pose_correction{0.04};

  // Effective scout floor derived from the downstream joint solver gate.
  // A camera that cannot satisfy this floor must not become a solver route.
  std::size_t minimum_selection_tracked_points{1U};

  std::size_t healthy_tracked_points{24U};
  double healthy_tracking_ratio{0.50};
  double healthy_residual_rmse{10.0};
  double maximum_saturation_fraction{0.25};
  double maximum_dark_fraction{0.35};
  double healthy_mean_gradient{12.0};
  double healthy_spatial_coverage{0.20};

  // These budgets bound the scout linearization and iterative visual update.
  // They do not bound visual-map lookup performed before the scout.
  std::size_t total_candidate_budget{240U};
  std::size_t minimum_candidate_budget{16U};
  std::size_t total_patch_budget{120U};
  std::size_t minimum_patch_budget{8U};
  std::size_t minimum_selected_cameras{2U};
  std::size_t maximum_selected_cameras{4U};
};

struct AdaptiveCameraObservation
{
  using Matrix6 = Eigen::Matrix<double, 6, 6>;

  std::uint8_t camera_id{0U};
  bool frame_received{false};
  double source_timestamp_sec{
    std::numeric_limits<double>::quiet_NaN()};
  // True preserves the established tracking-only eligibility contract for
  // callers that do not provide explicit visual-map lifecycle evidence.
  bool tracking_map_initialized{true};
  // Owner-side runtime state attests that this initialized camera has
  // completed the required consecutive trusted insufficient-tracking scouts
  // and may receive one LiDAR-anchored refresh.
  bool zero_map_reentry_ready{false};
  std::uint64_t observation_generation{0U};
  std::uint64_t image_generation{0U};
  std::uint32_t available_metrics{0U};
  std::size_t candidate_points{0U};
  std::size_t tracked_points{0U};
  std::size_t residual_count{0U};
  double squared_residual_sum{0.0};
  double saturation_fraction{0.0};
  double dark_fraction{0.0};
  double mean_gradient{0.0};
  double spatial_coverage{0.0};
  std::size_t inlier_count{0U};
  std::size_t total_patch_count{0U};
  double inlier_ratio{0.0};
  double nis{std::numeric_limits<double>::quiet_NaN()};
  std::size_t nis_dof{0U};
  double normalized_pose_correction{
    std::numeric_limits<double>::quiet_NaN()};
  double robust_information_gain{0.0};
  Matrix6 visual_information{Matrix6::Zero()};
};

using AdaptiveCameraObservations =
  std::array<AdaptiveCameraObservation, kAdaptiveVisualCameraCount>;

enum class AdaptiveCameraHealthState
{
  kHealthy,
  kDegraded,
  kIsolated,
  kProbation,
};

const char * adaptiveCameraHealthStateName(
  AdaptiveCameraHealthState state) noexcept;

struct AdaptiveCameraHysteresisState
{
  AdaptiveCameraHealthState state{AdaptiveCameraHealthState::kHealthy};
  // Compatibility mirror for existing consumers. kIsolated and kProbation
  // both remain ineligible until recovery is complete.
  bool isolated{false};
  std::size_t consecutive_unhealthy{0U};
  std::size_t consecutive_healthy{0U};
  double probation_started_at_sec{
    std::numeric_limits<double>::quiet_NaN()};
};

enum class AdaptiveCameraTransition
{
  kNone,
  kIsolated,
  kRecovered,
};

const char * adaptiveCameraTransitionName(AdaptiveCameraTransition transition);

enum class AdaptiveCameraRejectReason
{
  kNone,
  kFrameMissing,
  kImageHardFailure,
  kTrackingUnavailable,
  kIsolated,
  kHealthBelowHardReject,
  kRobustGateFailed,
};

enum class AdaptivePolicyReason
{
  kReady,
  kDisabled,
  kLidarNotReady,
  kLidarNonFinite,
  kLidarNonSymmetric,
  kLidarInsufficientInformation,
  kInsufficientCameras,
  kInvalidCameraObservations,
};

const char * adaptivePolicyReasonName(AdaptivePolicyReason reason);

enum class AdaptiveSelectionReason
{
  kNotApplicable,
  kSelectedBestHealthy,
  kRetainedHealthyPrimary,
  kSwitchedAfterChallengerConfirmation,
  kSwitchedAfterHardFault,
  kSwitchedAfterPrimaryUnavailable,
  kAwaitingChallengerConfirmation,
  kSelectedBestInformationGain,
};

const char * adaptiveSelectionReasonName(
  AdaptiveSelectionReason reason);

enum class AdaptiveCandidateHealthCommitStatus
{
  kCommitted,
  kUntrustedScout,
  kInvalidOrForeignCapability,
  kInvalidOrForeignReceipt,
  kBindingMismatch,
  kStalePolicyRevision,
  kNonMonotonicOrReplayedLidarGeneration,
  kRevisionExhausted,
};

const char * adaptiveCandidateHealthCommitStatusName(
  AdaptiveCandidateHealthCommitStatus status);

struct AdaptiveCameraDecision
{
  bool frame_received{false};
  bool available{false};
  // Camera-local gate before LiDAR readiness and top-k selection.
  bool eligible{false};
  // Final solver input after global readiness and top-k selection.
  bool enabled{false};
  AdaptiveCameraHealthState health_state{
    AdaptiveCameraHealthState::kHealthy};
  bool isolated{false};
  bool metrics_complete{false};
  bool health_evidence_complete{false};
  bool image_hard_failure{false};
  bool image_recovery_healthy{false};
  bool health_degraded{true};
  double health{0.0};
  double covariance_scale{1.0};
  double complement_score{0.0};
  double selection_score{0.0};
  bool robust_gate_passed{false};
  double robust_information_gain{0.0};
  std::size_t candidate_budget{0U};
  std::size_t patch_budget{0U};
  // kNone for eligible cameras, including healthy cameras pruned by top-k.
  AdaptiveCameraRejectReason reject_reason{
    AdaptiveCameraRejectReason::kNone};
  AdaptiveCameraTransition transition{AdaptiveCameraTransition::kNone};
};

struct AdaptiveSelectionChallengerState
{
  std::optional<std::uint8_t> camera_id;
  std::size_t consecutive_confirmations{0U};
  bool soft_gap_observed{false};
};

struct AdaptiveVisualPolicyPlan
{
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  using Vector6 = Eigen::Matrix<double, 6, 1>;

  bool adaptive_enabled{false};
  bool lidar_information_valid{false};
  AdaptivePolicyReason reason{AdaptivePolicyReason::kDisabled};
  std::uint64_t base_revision{0U};
  std::size_t weak_direction_count{0U};
  Vector6 lidar_eigenvalues{Vector6::Zero()};
  Matrix6 weak_direction_weight{Matrix6::Identity()};
  std::array<AdaptiveCameraDecision, kAdaptiveVisualCameraCount> cameras;
  std::array<AdaptiveCameraHysteresisState, kAdaptiveVisualCameraCount>
    proposed_hysteresis;
  std::optional<std::uint8_t> proposed_primary_camera;
  AdaptiveSelectionChallengerState proposed_selection_challenger;
  AdaptiveSelectionReason selection_reason{
    AdaptiveSelectionReason::kNotApplicable};
};

class AdaptiveVisualPolicy;

// Immutable capability minted only by one AdaptiveVisualPolicy owner. It
// shallow-copies the R2b runtime publication, so the original image/state
// publication remains shared rather than creating a second IMU/state path.
// The existing mutable AdaptiveVisualPolicyPlan remains a legacy-untrusted
// compatibility seam until VIO is migrated to this capability.
class AdaptiveVisualPolicyPlanCapability
{
public:
  AdaptiveVisualPolicyPlanCapability(
    const AdaptiveVisualPolicyPlanCapability &) = default;
  AdaptiveVisualPolicyPlanCapability(
    AdaptiveVisualPolicyPlanCapability &&) = default;
  AdaptiveVisualPolicyPlanCapability & operator=(
    const AdaptiveVisualPolicyPlanCapability &) = default;
  AdaptiveVisualPolicyPlanCapability & operator=(
    AdaptiveVisualPolicyPlanCapability &&) = default;

  bool valid() const noexcept;
  std::uint32_t configRevision() const noexcept;
  std::uint64_t basePolicyRevision() const noexcept;
  std::uint64_t runtimePublicationSequence() const noexcept;
  std::uint64_t cameraBatchSequence() const noexcept;
  bool lidarInformationReady() const noexcept;

  const AdaptiveVisualPolicyPlan & decisionPlan() const;
  const Eigen::Matrix<double, 6, 6> &
  lidarPoseInformation() const;
  const AdaptiveCameraObservations & observations() const;
  const ReferenceOnlyMigrationRuntimeBundle &
  runtimeBundle() const;

private:
  struct Data;

  explicit AdaptiveVisualPolicyPlanCapability(
    std::shared_ptr<const Data> data);

  std::shared_ptr<const Data> data_;

  friend class AdaptiveVisualPolicy;
};

// Owner-sealed candidate plan. Unlike the legacy/full-four plan capability
// above, this capability binds directly to one queue-issued candidate batch
// receipt and has no R2b runtime-publication dependency.
class AdaptiveVisualCandidatePlanCapability
{
public:
  AdaptiveVisualCandidatePlanCapability(
    const AdaptiveVisualCandidatePlanCapability &) = default;
  AdaptiveVisualCandidatePlanCapability(
    AdaptiveVisualCandidatePlanCapability &&) = default;
  AdaptiveVisualCandidatePlanCapability & operator=(
    const AdaptiveVisualCandidatePlanCapability &) = default;
  AdaptiveVisualCandidatePlanCapability & operator=(
    AdaptiveVisualCandidatePlanCapability &&) = default;

  bool valid() const noexcept;
  std::uint32_t configRevision() const noexcept;
  std::uint64_t basePolicyRevision() const noexcept;
  std::uint64_t cameraBatchSequence() const noexcept;
  bool lidarInformationReady() const noexcept;
  const std::array<bool, kAdaptiveVisualCameraCount> &
  candidateMask() const;
  std::size_t cameraCount() const noexcept;
  double batchTimestampSec() const noexcept;
  AdaptiveSelectionReason selectionReason() const noexcept;
  const AdaptiveVisualPolicyPlan & decisionPlan() const;
  const Eigen::Matrix<double, 6, 6> &
  lidarPoseInformation() const;
  const AdaptiveCameraObservations & observations() const;
  const CameraObservationBatchReceipt &
  cameraBatchReceipt() const;

private:
  struct Data;

  explicit AdaptiveVisualCandidatePlanCapability(
    std::shared_ptr<const Data> data);

  std::shared_ptr<const Data> data_;

  friend class AdaptiveVisualPolicy;
  friend class AdaptiveVisualSelectionAuthority;
  friend class AdaptiveVisualRuntimeAttemptPermit;
};

struct AdaptiveVisualMapMaintenanceDecision
{
  bool seed_new_points{false};
  bool update_existing_points{false};
};

enum class AdaptiveVisualMapLifecycle
{
  kEmpty,
  kSeededUnvalidated,
  kInitialized,
};

enum class AdaptiveVisualLioAnchoredMaintenanceRequest
{
  kBootstrap,
  kReseed,
};

AdaptiveVisualMapMaintenanceDecision
planAdaptiveLioAnchoredMapMaintenance(
  AdaptiveVisualLioAnchoredMaintenanceRequest request,
  AdaptiveVisualMapLifecycle lifecycle,
  bool camera_route_authorized, bool camera_isolated,
  bool image_seed_eligible, bool lidar_information_valid);

AdaptiveVisualMapMaintenanceDecision
planAdaptiveVisualSolverMapCommit(
  bool adaptive_enabled, bool camera_route_authorized,
  bool visual_update_accepted);

std::size_t adaptiveScoutCandidateBudget(
  std::size_t total_candidate_budget, std::size_t camera_id);

class AdaptiveVisualPolicy
{
public:
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  using CameraObservations = AdaptiveCameraObservations;

  explicit AdaptiveVisualPolicy(
    const AdaptiveVisualPolicyConfig & config = AdaptiveVisualPolicyConfig{});

  AdaptiveVisualPolicy(const AdaptiveVisualPolicy &) = delete;
  AdaptiveVisualPolicy & operator=(const AdaptiveVisualPolicy &) = delete;
  AdaptiveVisualPolicy(AdaptiveVisualPolicy &&) = delete;
  AdaptiveVisualPolicy & operator=(AdaptiveVisualPolicy &&) = delete;

  // The owning VIO transaction thread serializes plan, issue, reset, getters,
  // and non-candidate commit calls. The candidate capability-and-receipt
  // commit is internally safe against concurrent duplicate receipt copies, and
  // reset participates in that transaction lock. This is not blanket
  // thread safety for arbitrary AdaptiveVisualPolicy calls.
  const AdaptiveVisualPolicyConfig & config() const { return config_; }
  std::uint64_t revision() const { return revision_; }
  const std::array<AdaptiveCameraHysteresisState,
    kAdaptiveVisualCameraCount> & hysteresis() const
  {
    return hysteresis_;
  }

  AdaptiveVisualPolicyPlan plan(
    const Matrix6 & lidar_pose_information,
    const CameraObservations & observations,
    bool lidar_information_ready = true) const;

  // Owner-sealed foundation. The capability binds the immutable decision and
  // inputs to a strictly increasing R2b batch/publication pair. This owner
  // does not know the expected R2b publisher or batch authority; VIO/S2 must
  // verify that composition before treating the input as target-authoritative.
  std::optional<AdaptiveVisualPolicyPlanCapability> issuePlan(
    const ReferenceOnlyMigrationRuntimeBundle & runtime,
    const Matrix6 & lidar_pose_information,
    const CameraObservations & observations,
    bool lidar_information_ready = true);
  std::optional<AdaptiveVisualCandidatePlanCapability>
  issueCandidatePlan(
    const ::CameraObservationBatch & batch,
    const Matrix6 & lidar_pose_information,
    const CameraObservations & observations,
    bool lidar_information_ready = true);
  bool accepts(
    const AdaptiveVisualPolicyPlanCapability & capability) const noexcept;
  bool accepts(
    const AdaptiveVisualCandidatePlanCapability & capability)
    const noexcept;
  bool commitHealthObservation(
    const AdaptiveVisualPolicyPlanCapability & capability,
    bool scout_trusted);
  AdaptiveCandidateHealthCommitStatus commitHealthObservation(
    const AdaptiveVisualCandidatePlanCapability & capability,
    const AdaptiveVisualSelectionReceipt & receipt,
    bool scout_trusted);
  AdaptiveCandidateHealthCommitStatus
  commitZeroSelectionHealthObservation(
    const AdaptiveVisualCandidatePlanCapability & capability,
    AdaptiveVisualRuntimeAttemptPermit && permit);

  // Legacy-untrusted compatibility path used by the current VIO runtime.
  // It is deliberately non-exclusive until that runtime is migrated to the
  // owner-sealed overload above.
  bool commit(
    const AdaptiveVisualPolicyPlan & plan, bool visual_update_accepted);
  bool commitHealthObservation(
    const AdaptiveVisualPolicyPlan & plan, bool scout_trusted);

  static constexpr bool legacyMutablePlanCompatibilityEnabled() noexcept
  {
    return true;
  }

  static constexpr bool ownerSealedRuntimeExclusive() noexcept
  {
    return false;
  }

  static constexpr bool expectedRuntimeAuthorityPinned() noexcept
  {
    return false;
  }

  void reset();

private:
  bool ownsPlanCapability(
    const AdaptiveVisualPolicyPlanCapability & capability)
    const noexcept;
  bool ownsCandidatePlanCapability(
    const AdaptiveVisualCandidatePlanCapability & capability)
    const noexcept;

  AdaptiveVisualPolicyConfig config_;
  std::array<AdaptiveCameraHysteresisState, kAdaptiveVisualCameraCount>
    hysteresis_;
  std::optional<std::uint8_t> primary_camera_;
  AdaptiveSelectionChallengerState selection_challenger_;
  std::uint64_t revision_{0U};
  std::shared_ptr<const std::uint8_t> capability_owner_;
  std::uint64_t last_issued_batch_sequence_{0U};
  std::uint64_t last_issued_publication_sequence_{0U};
  std::uint64_t
    last_committed_candidate_lidar_generation_{0U};
  std::mutex candidate_commit_transaction_mutex_;

  friend class AdaptiveVisualSelectionAuthority;
};

}  // namespace livo

#endif  // LIVO_ADAPTIVE_VISUAL_POLICY_H_
