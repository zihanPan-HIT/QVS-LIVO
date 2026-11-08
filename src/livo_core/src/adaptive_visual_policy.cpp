#include "adaptive_visual_policy.h"
#include "camera_observation.h"
#include "reference_only_migration_runtime_provenance.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING
#include <atomic>
#include <new>
#endif

namespace livo
{

struct AdaptiveVisualPolicyPlanCapability::Data
{
  Data(
    std::shared_ptr<const std::uint8_t> input_owner,
    const ReferenceOnlyMigrationRuntimeBundle & input_runtime,
    const AdaptiveVisualPolicy::Matrix6 &
      input_lidar_pose_information,
    const AdaptiveCameraObservations & input_observations,
    const bool input_lidar_information_ready,
    AdaptiveVisualPolicyPlan input_decision_plan,
    const std::uint64_t input_runtime_publication_sequence,
    const std::uint64_t input_camera_batch_sequence)
  : owner(std::move(input_owner)),
    runtime(input_runtime),
    lidar_pose_information(input_lidar_pose_information),
    observations(input_observations),
    lidar_information_ready(input_lidar_information_ready),
    decision_plan(std::move(input_decision_plan)),
    base_policy_revision(decision_plan.base_revision),
    runtime_publication_sequence(
      input_runtime_publication_sequence),
    camera_batch_sequence(input_camera_batch_sequence)
  {
  }

  const std::shared_ptr<const std::uint8_t> owner;
  const ReferenceOnlyMigrationRuntimeBundle runtime;
  const AdaptiveVisualPolicy::Matrix6 lidar_pose_information;
  const AdaptiveCameraObservations observations;
  const bool lidar_information_ready;
  const AdaptiveVisualPolicyPlan decision_plan;
  const std::uint32_t config_revision{
    kAdaptiveVisualPolicyConfigRevision};
  const std::uint64_t base_policy_revision;
  const std::uint64_t runtime_publication_sequence;
  const std::uint64_t camera_batch_sequence;
};

struct AdaptiveVisualCandidatePlanCapability::Data
{
  Data(
    std::shared_ptr<const std::uint8_t> input_owner,
    CameraObservationBatchReceipt input_camera_batch_receipt,
    const std::array<bool, kAdaptiveVisualCameraCount> &
      input_candidate,
    const std::size_t input_camera_count,
    const double input_batch_timestamp_sec,
    const AdaptiveVisualPolicy::Matrix6 &
      input_lidar_pose_information,
    const AdaptiveCameraObservations & input_observations,
    const bool input_lidar_information_ready,
    AdaptiveVisualPolicyPlan input_decision_plan)
  : owner(std::move(input_owner)),
    camera_batch_receipt(
      std::move(input_camera_batch_receipt)),
    candidate(input_candidate),
    camera_count(input_camera_count),
    batch_timestamp_sec(input_batch_timestamp_sec),
    lidar_pose_information(input_lidar_pose_information),
    observations(input_observations),
    lidar_information_ready(input_lidar_information_ready),
    decision_plan(std::move(input_decision_plan)),
    base_policy_revision(decision_plan.base_revision),
    camera_batch_sequence(
      camera_batch_receipt.batchSequence())
  {
  }

  const std::shared_ptr<const std::uint8_t> owner;
  const CameraObservationBatchReceipt camera_batch_receipt;
  const std::array<bool, kAdaptiveVisualCameraCount> candidate;
  const std::size_t camera_count;
  const double batch_timestamp_sec;
  const AdaptiveVisualPolicy::Matrix6 lidar_pose_information;
  const AdaptiveCameraObservations observations;
  const bool lidar_information_ready;
  const AdaptiveVisualPolicyPlan decision_plan;
  const std::uint32_t config_revision{
    kAdaptiveVisualPolicyConfigRevision};
  const std::uint64_t base_policy_revision;
  const std::uint64_t camera_batch_sequence;
};

namespace
{

using Matrix6 = AdaptiveVisualPolicy::Matrix6;
using Vector6 = AdaptiveVisualPolicyPlan::Vector6;

#ifdef LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING
std::atomic<bool> fail_next_policy_owner_allocation{false};
#endif

bool sameOwner(
  const std::shared_ptr<const std::uint8_t> & left,
  const std::shared_ptr<const std::uint8_t> & right)
{
  return
    left &&
    right &&
    left.get() == right.get() &&
    !left.owner_before(right) &&
      !right.owner_before(left);
}

bool sameDouble(const double left, const double right);

bool isCanonicalAbsentObservation(
  const AdaptiveCameraObservation & observation,
  const std::uint8_t camera_id)
{
  AdaptiveCameraObservation canonical;
  canonical.camera_id = camera_id;
  return
    observation.camera_id == canonical.camera_id &&
    observation.frame_received ==
      canonical.frame_received &&
    observation.tracking_map_initialized ==
      canonical.tracking_map_initialized &&
    observation.zero_map_reentry_ready ==
      canonical.zero_map_reentry_ready &&
    observation.observation_generation ==
      canonical.observation_generation &&
    observation.image_generation ==
      canonical.image_generation &&
    observation.available_metrics ==
      canonical.available_metrics &&
    observation.candidate_points ==
      canonical.candidate_points &&
    observation.tracked_points ==
      canonical.tracked_points &&
    observation.residual_count ==
      canonical.residual_count &&
    observation.squared_residual_sum ==
      canonical.squared_residual_sum &&
    observation.saturation_fraction ==
      canonical.saturation_fraction &&
    observation.dark_fraction ==
      canonical.dark_fraction &&
    observation.mean_gradient ==
      canonical.mean_gradient &&
    observation.spatial_coverage ==
      canonical.spatial_coverage &&
    observation.inlier_count == canonical.inlier_count &&
    observation.total_patch_count == canonical.total_patch_count &&
    sameDouble(observation.inlier_ratio, canonical.inlier_ratio) &&
    sameDouble(observation.nis, canonical.nis) &&
    observation.nis_dof == canonical.nis_dof &&
    sameDouble(
      observation.normalized_pose_correction,
      canonical.normalized_pose_correction) &&
    sameDouble(
      observation.robust_information_gain,
      canonical.robust_information_gain) &&
    (observation.visual_information.array() ==
      canonical.visual_information.array()).all();
}

double clampUnit(const double value)
{
  return std::clamp(value, 0.0, 1.0);
}

bool sameDouble(const double left, const double right)
{
  return left == right ||
    (std::isnan(left) && std::isnan(right));
}

bool hasMetric(
  const AdaptiveCameraObservation & observation,
  const AdaptiveHealthMetric metric)
{
  return
    (observation.available_metrics & adaptiveMetricMask(metric)) != 0U;
}

bool hasAllMetrics(
  const AdaptiveCameraObservation & observation,
  const std::uint32_t required_metrics)
{
  return
    (observation.available_metrics & required_metrics) ==
    required_metrics;
}

// Wilson-Hilferty is sufficient for the admission threshold here: the exact
// quantile is not needed for ordering, while the dof remains bound to the
// final inlier patch count rather than the correlated pixel residual count.
double chiSquare99(const std::size_t degrees_of_freedom)
{
  if (degrees_of_freedom == 0U)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double dof = static_cast<double>(degrees_of_freedom);
  constexpr double normal_99 = 2.3263478740408408;
  const double root = 1.0 - 2.0 / (9.0 * dof) +
    normal_99 / std::sqrt(9.0 * dof);
  return dof * root * root * root;
}

Matrix6 normalizationMatrix(const AdaptiveVisualPolicyConfig & config)
{
  Matrix6 scale = Matrix6::Zero();
  scale.diagonal() <<
    config.rotation_scale_rad,
    config.rotation_scale_rad,
    config.rotation_scale_rad,
    config.translation_scale_m,
    config.translation_scale_m,
    config.translation_scale_m;
  return scale;
}

Matrix6 finiteSymmetricOrZero(const Matrix6 & information)
{
  if (!information.allFinite())
  {
    return Matrix6::Zero();
  }
  return 0.5 * (information + information.transpose());
}

double positiveLogDet(const Matrix6 & information)
{
  const Eigen::LDLT<Matrix6> factor(
    finiteSymmetricOrZero(information));
  if (factor.info() != Eigen::Success)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto diagonal = factor.vectorD();
  if (!diagonal.allFinite() ||
      (diagonal.array() <= 0.0).any())
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return diagonal.array().log().sum();
}

struct WeakDirectionResult
{
  bool valid{false};
  AdaptivePolicyReason reason{
    AdaptivePolicyReason::kLidarInsufficientInformation};
  std::size_t count{0U};
  Vector6 eigenvalues{Vector6::Zero()};
  Matrix6 weight{Matrix6::Zero()};
};

WeakDirectionResult computeWeakDirections(
  const Matrix6 & lidar_information,
  const AdaptiveVisualPolicyConfig & config)
{
  WeakDirectionResult result;
  if (!lidar_information.allFinite())
  {
    result.reason = AdaptivePolicyReason::kLidarNonFinite;
    return result;
  }
  const double information_norm = lidar_information.norm();
  const double asymmetry =
    (lidar_information - lidar_information.transpose()).norm() /
    std::max(1.0, information_norm);
  constexpr double kMaximumRelativeAsymmetry = 1.0e-6;
  if (!std::isfinite(asymmetry) ||
      asymmetry > kMaximumRelativeAsymmetry)
  {
    result.reason = AdaptivePolicyReason::kLidarNonSymmetric;
    return result;
  }

  const Matrix6 scale = normalizationMatrix(config);
  const Matrix6 normalized =
    0.5 *
    (scale.transpose() * lidar_information * scale +
     scale.transpose() * lidar_information.transpose() * scale);
  Eigen::SelfAdjointEigenSolver<Matrix6> solver(normalized);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite())
  {
    result.reason = AdaptivePolicyReason::kLidarInsufficientInformation;
    return result;
  }

  const Vector6 raw_eigenvalues = solver.eigenvalues();
  const double eigenvalue_scale =
    raw_eigenvalues.cwiseAbs().maxCoeff();
  const double negative_tolerance = std::max(
    config.weak_absolute_eigenvalue,
    1.0e-9 * eigenvalue_scale);
  if (raw_eigenvalues.minCoeff() < -negative_tolerance)
  {
    result.reason =
      AdaptivePolicyReason::kLidarInsufficientInformation;
    return result;
  }

  result.eigenvalues = raw_eigenvalues.cwiseMax(0.0);
  const double maximum_eigenvalue = result.eigenvalues.maxCoeff();
  if (!(maximum_eigenvalue > config.weak_absolute_eigenvalue))
  {
    result.reason = AdaptivePolicyReason::kLidarInsufficientInformation;
    return result;
  }

  const double threshold = std::max(
    config.weak_absolute_eigenvalue,
    config.weak_relative_eigenvalue * maximum_eigenvalue);
  Vector6 weights = Vector6::Zero();
  for (Eigen::Index index = 0; index < weights.rows(); ++index)
  {
    weights[index] = clampUnit(
      (threshold - result.eigenvalues[index]) /
      (threshold + config.weak_absolute_eigenvalue));
    if (weights[index] > 0.0)
    {
      ++result.count;
    }
  }
  result.weight =
    solver.eigenvectors() * weights.asDiagonal() *
    solver.eigenvectors().transpose();
  result.weight =
    0.5 * (result.weight + result.weight.transpose()).eval();
  result.valid = result.weight.allFinite();
  result.reason = result.valid ?
    AdaptivePolicyReason::kReady :
    AdaptivePolicyReason::kLidarInsufficientInformation;
  return result;
}

double residualRmse(const AdaptiveCameraObservation & observation)
{
  if (observation.residual_count == 0U ||
      !std::isfinite(observation.squared_residual_sum) ||
      observation.squared_residual_sum < 0.0)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return std::sqrt(
    observation.squared_residual_sum /
    static_cast<double>(observation.residual_count));
}

double computeHealth(
  const AdaptiveCameraObservation & observation,
  const AdaptiveVisualPolicyConfig & config)
{
  if (!observation.frame_received)
  {
    return 0.0;
  }
  double weighted_score = 0.0;
  double available_weight = 0.0;
  const auto add_score =
    [&weighted_score, &available_weight](
      const bool available, const double weight, const double score)
    {
      if (!available || !std::isfinite(score))
      {
        return;
      }
      weighted_score += weight * clampUnit(score);
      available_weight += weight;
    };

  add_score(
    hasMetric(observation, AdaptiveHealthMetric::kTrackedCount),
    0.25,
    static_cast<double>(observation.tracked_points) /
    static_cast<double>(config.healthy_tracked_points));

  const bool tracking_ratio_available =
    hasMetric(observation, AdaptiveHealthMetric::kCandidateCount) &&
    hasMetric(observation, AdaptiveHealthMetric::kTrackedCount);
  const double tracking_ratio =
    observation.candidate_points == 0U ? 0.0 :
    static_cast<double>(observation.tracked_points) /
    static_cast<double>(observation.candidate_points);
  add_score(
    tracking_ratio_available, 0.15,
    tracking_ratio / config.healthy_tracking_ratio);

  const double rmse = residualRmse(observation);
  const double normalized_rmse = rmse / config.healthy_residual_rmse;
  add_score(
    hasMetric(observation, AdaptiveHealthMetric::kPhotometricResidual),
    0.20, 1.0 / (1.0 + normalized_rmse * normalized_rmse));

  add_score(
    hasMetric(observation, AdaptiveHealthMetric::kSaturationFraction),
    0.10,
    1.0 -
    observation.saturation_fraction /
    config.maximum_saturation_fraction);
  add_score(
    hasMetric(observation, AdaptiveHealthMetric::kDarkFraction),
    0.10,
    1.0 -
    observation.dark_fraction / config.maximum_dark_fraction);
  add_score(
    hasMetric(observation, AdaptiveHealthMetric::kMeanGradient),
    0.10,
    observation.mean_gradient / config.healthy_mean_gradient);
  add_score(
    hasMetric(observation, AdaptiveHealthMetric::kSpatialCoverage),
    0.10,
    observation.spatial_coverage / config.healthy_spatial_coverage);

  if (!(available_weight > 0.0))
  {
    return 0.0;
  }
  return clampUnit(weighted_score / available_weight);
}

bool robustAdmissionPassed(
  const AdaptiveCameraObservation & observation,
  const AdaptiveVisualPolicyConfig & config);

bool measurementAvailable(
  const AdaptiveCameraObservation & observation,
  const AdaptiveVisualPolicyConfig & config)
{
  if (!observation.frame_received ||
      !hasMetric(observation, AdaptiveHealthMetric::kTrackedCount) ||
      !hasMetric(
        observation, AdaptiveHealthMetric::kPhotometricResidual) ||
      !hasMetric(
        observation, AdaptiveHealthMetric::kVisualInformation) ||
      observation.tracked_points <
        config.minimum_selection_tracked_points ||
      observation.residual_count == 0U ||
      !observation.visual_information.allFinite())
  {
    return false;
  }
  if (!(observation.visual_information.trace() > 0.0))
  {
    return false;
  }
  const bool robust_evidence_present =
    hasAllMetrics(
    observation, kAdaptiveRobustAdmissionMetricMask);
  const bool sealed_adaptive_runtime =
    config.enabled &&
    config.minimum_selected_cameras == 1U &&
    config.maximum_selected_cameras == 1U;
  return
    sealed_adaptive_runtime ||
    !robust_evidence_present ||
    robustAdmissionPassed(observation, config);
}

bool robustAdmissionPassed(
  const AdaptiveCameraObservation & observation,
  const AdaptiveVisualPolicyConfig & config)
{
  if (!hasAllMetrics(
      observation, kAdaptiveRobustAdmissionMetricMask) ||
      observation.inlier_count < config.minimum_inliers ||
      observation.total_patch_count == 0U ||
      !std::isfinite(observation.inlier_ratio) ||
      observation.inlier_ratio < config.minimum_inlier_ratio ||
      !std::isfinite(observation.spatial_coverage) ||
      observation.spatial_coverage < config.healthy_spatial_coverage ||
      !std::isfinite(observation.nis) ||
      observation.nis_dof == 0U ||
      observation.nis >
        chiSquare99(observation.nis_dof) * config.nis_chi_square_scale ||
      !std::isfinite(observation.normalized_pose_correction) ||
      observation.normalized_pose_correction >
        config.maximum_normalized_pose_correction ||
      !std::isfinite(observation.robust_information_gain) ||
      observation.robust_information_gain < 0.0)
  {
    return false;
  }
  return true;
}

bool imageHardFailure(
  const AdaptiveCameraObservation & observation,
  const AdaptiveVisualPolicyConfig & config)
{
  if (!observation.frame_received ||
      !hasAllMetrics(
      observation, kAdaptiveImageHealthMetricMask))
  {
    return false;
  }
  if (!std::isfinite(observation.saturation_fraction) ||
      !std::isfinite(observation.dark_fraction) ||
      !std::isfinite(observation.mean_gradient))
  {
    return true;
  }
  return
    observation.saturation_fraction >
      config.maximum_saturation_fraction ||
    observation.dark_fraction >
      config.maximum_dark_fraction;
}

bool imageRecoveryHealthy(
  const AdaptiveCameraObservation & observation,
  const AdaptiveVisualPolicyConfig & config)
{
  if (!observation.frame_received ||
      !hasAllMetrics(
      observation, kAdaptiveImageHealthMetricMask) ||
      !std::isfinite(observation.saturation_fraction) ||
      !std::isfinite(observation.dark_fraction) ||
      !std::isfinite(observation.mean_gradient))
  {
    return false;
  }
  return
    observation.saturation_fraction <=
      config.maximum_saturation_fraction &&
    observation.dark_fraction <=
      config.maximum_dark_fraction &&
    observation.mean_gradient >= config.healthy_mean_gradient;
}

bool trackingFailureEvidence(
  const AdaptiveCameraObservation & observation)
{
  // A zero-candidate cold start is ambiguous and must not accumulate
  // isolation. Any supported candidate set with zero tracks is actionable.
  return
    observation.frame_received &&
    hasMetric(observation, AdaptiveHealthMetric::kCandidateCount) &&
    hasMetric(observation, AdaptiveHealthMetric::kTrackedCount) &&
    observation.candidate_points > 0U &&
    observation.tracked_points == 0U;
}

bool zeroMapPreselectionEligible(
  const AdaptiveCameraObservation & observation,
  const bool image_recovery_healthy,
  const bool image_hard_failure)
{
  constexpr std::uint32_t kBootstrapTrackingMetricMask =
    adaptiveMetricMask(
    AdaptiveHealthMetric::kCandidateCount) |
    adaptiveMetricMask(
    AdaptiveHealthMetric::kTrackedCount);
  const bool bootstrap_ready =
    !observation.tracking_map_initialized &&
    observation.candidate_points == 0U &&
    observation.tracked_points == 0U;
  const bool refresh_ready =
    observation.tracking_map_initialized &&
    observation.zero_map_reentry_ready;
  return
    (bootstrap_ready || refresh_ready) &&
    observation.frame_received &&
    hasAllMetrics(
      observation, kBootstrapTrackingMetricMask) &&
    image_recovery_healthy &&
    !image_hard_failure;
}

bool maintenanceOnlyCameraCandidate(
  const AdaptiveCameraObservation & observation,
  const AdaptiveCameraDecision & decision)
{
  const bool bootstrap_only =
    !observation.tracking_map_initialized &&
    observation.candidate_points == 0U &&
    observation.tracked_points == 0U;
  const bool refresh_only =
    observation.tracking_map_initialized &&
    observation.zero_map_reentry_ready;
  return
    !decision.available &&
    (bootstrap_only || refresh_only);
}

AdaptiveCameraRejectReason cameraRejectReason(
  const AdaptiveCameraDecision & decision,
  const AdaptiveVisualPolicyConfig & config,
  const bool zero_map_preselection_eligible)
{
  if (!decision.frame_received)
  {
    return AdaptiveCameraRejectReason::kFrameMissing;
  }
  if (decision.image_hard_failure)
  {
    return AdaptiveCameraRejectReason::kImageHardFailure;
  }
  if (!decision.available &&
      !zero_map_preselection_eligible)
  {
    return AdaptiveCameraRejectReason::kTrackingUnavailable;
  }
  if (decision.isolated)
  {
    return AdaptiveCameraRejectReason::kIsolated;
  }
  if (decision.health < config.hard_reject_health)
  {
    return AdaptiveCameraRejectReason::kHealthBelowHardReject;
  }
  return AdaptiveCameraRejectReason::kNone;
}

AdaptiveCameraHysteresisState proposeHysteresis(
  const AdaptiveCameraHysteresisState & current,
  const double source_timestamp_sec,
  const bool frame_received,
  const bool health_evidence_complete,
  const double health,
  const bool image_hard_failure,
  const bool image_recovery_healthy,
  const bool tracking_failure_evidence,
  const AdaptiveVisualPolicyConfig & config)
{
  AdaptiveCameraHysteresisState proposed = current;
  if (!current.isolated)
  {
    proposed.consecutive_healthy = 0U;
    proposed.probation_started_at_sec =
      std::numeric_limits<double>::quiet_NaN();
    if (image_hard_failure)
    {
      proposed.state = AdaptiveCameraHealthState::kIsolated;
      proposed.isolated = true;
      proposed.consecutive_unhealthy = 0U;
    }
    else
    {
      const bool unhealthy_evidence =
        !frame_received ||
        tracking_failure_evidence ||
        (health_evidence_complete &&
         health < config.health_off_threshold);
      if (unhealthy_evidence)
      {
        proposed.state = AdaptiveCameraHealthState::kDegraded;
        ++proposed.consecutive_unhealthy;
        if (proposed.consecutive_unhealthy >=
            config.off_hysteresis_frames)
        {
          proposed.state = AdaptiveCameraHealthState::kIsolated;
          proposed.isolated = true;
          proposed.consecutive_unhealthy = 0U;
        }
      }
      else
      {
        proposed.consecutive_unhealthy = 0U;
        if (
          frame_received && !image_hard_failure &&
          !tracking_failure_evidence &&
          (image_recovery_healthy ||
           (health_evidence_complete &&
            health > config.health_on_threshold)))
        {
          proposed.state = AdaptiveCameraHealthState::kHealthy;
        }
      }
    }
  }
  else
  {
    proposed.consecutive_unhealthy = 0U;
    const bool healthy_recovery_evidence =
      frame_received && !image_hard_failure &&
      !tracking_failure_evidence &&
      (image_recovery_healthy ||
       (health_evidence_complete &&
        health > config.health_on_threshold));
    if (healthy_recovery_evidence)
    {
      if (current.state != AdaptiveCameraHealthState::kProbation)
      {
        proposed.state = AdaptiveCameraHealthState::kProbation;
        proposed.consecutive_healthy = 1U;
        proposed.probation_started_at_sec = source_timestamp_sec;
      }
      else
      {
        ++proposed.consecutive_healthy;
      }
      const bool duration_evidence_available =
        std::isfinite(source_timestamp_sec) &&
        std::isfinite(proposed.probation_started_at_sec);
      const bool duration_satisfied =
        !duration_evidence_available ||
        source_timestamp_sec - proposed.probation_started_at_sec >=
          config.minimum_probation_duration_sec;
      if (
        proposed.consecutive_healthy >=
          config.on_hysteresis_frames &&
        duration_satisfied)
      {
        proposed.state = AdaptiveCameraHealthState::kHealthy;
        proposed.isolated = false;
        proposed.consecutive_healthy = 0U;
        proposed.probation_started_at_sec =
          std::numeric_limits<double>::quiet_NaN();
      }
    }
    else
    {
      proposed.state = AdaptiveCameraHealthState::kIsolated;
      proposed.consecutive_healthy = 0U;
      proposed.probation_started_at_sec =
        std::numeric_limits<double>::quiet_NaN();
    }
  }
  return proposed;
}

double complementScore(
  const AdaptiveCameraObservation & observation,
  const Matrix6 & weak_direction_weight,
  const AdaptiveVisualPolicyConfig & config)
{
  if (!observation.frame_received ||
      !hasMetric(
      observation, AdaptiveHealthMetric::kVisualInformation) ||
      !observation.visual_information.allFinite())
  {
    return 0.0;
  }
  const Matrix6 scale = normalizationMatrix(config);
  const Matrix6 normalized_visual = finiteSymmetricOrZero(
    scale.transpose() * observation.visual_information * scale);
  const double score =
    (weak_direction_weight * normalized_visual).trace();
  return std::isfinite(score) ? std::max(0.0, score) : 0.0;
}

std::vector<std::size_t> enabledCameraIndices(
  const AdaptiveVisualPolicyPlan & plan)
{
  std::vector<std::size_t> indices;
  for (std::size_t camera_id = 0U;
       camera_id < plan.cameras.size(); ++camera_id)
  {
    if (plan.cameras[camera_id].enabled)
    {
      indices.push_back(camera_id);
    }
  }
  return indices;
}

struct AdaptivePrimarySelectionProposal
{
  std::optional<std::uint8_t> primary_camera;
  AdaptiveSelectionChallengerState challenger;
  AdaptiveSelectionReason reason{
    AdaptiveSelectionReason::kNotApplicable};
};

bool confirmSelectionChallenger(
  const std::size_t camera_id,
  const AdaptiveSelectionChallengerState & current_challenger,
  const std::size_t challenger_frames,
  AdaptivePrimarySelectionProposal & proposal)
{
  proposal.challenger.camera_id =
    static_cast<std::uint8_t>(camera_id);
  proposal.challenger.consecutive_confirmations =
    current_challenger.camera_id == proposal.challenger.camera_id ?
    current_challenger.consecutive_confirmations : 0U;
  proposal.challenger.soft_gap_observed = false;
  if (
    proposal.challenger.consecutive_confirmations <
    challenger_frames)
  {
    ++proposal.challenger.consecutive_confirmations;
  }
  return
    proposal.challenger.consecutive_confirmations >=
    challenger_frames;
}

bool retainChallengerAcrossOneSoftGap(
  const AdaptiveSelectionChallengerState & current_challenger,
  AdaptivePrimarySelectionProposal & proposal)
{
  if (
    current_challenger.camera_id &&
    current_challenger.consecutive_confirmations > 0U &&
    !current_challenger.soft_gap_observed)
  {
    proposal.challenger = current_challenger;
    proposal.challenger.soft_gap_observed = true;
    return true;
  }
  return false;
}

bool cameraRanksAhead(
  const AdaptiveVisualPolicyPlan & plan,
  const AdaptiveCameraObservations & observations,
  const std::size_t lhs, const std::size_t rhs)
{
  const double left_score = plan.cameras[lhs].selection_score;
  const double right_score = plan.cameras[rhs].selection_score;
  if (left_score != right_score)
  {
    return left_score > right_score;
  }

  const AdaptiveCameraObservation & left = observations[lhs];
  const AdaptiveCameraObservation & right = observations[rhs];
  if (left.observation_generation != right.observation_generation)
  {
    return left.observation_generation > right.observation_generation;
  }
  if (left.image_generation != right.image_generation)
  {
    return left.image_generation > right.image_generation;
  }
  const bool left_timestamp_finite =
    std::isfinite(left.source_timestamp_sec);
  const bool right_timestamp_finite =
    std::isfinite(right.source_timestamp_sec);
  if (left_timestamp_finite != right_timestamp_finite)
  {
    return left_timestamp_finite;
  }
  if (
    left_timestamp_finite &&
    left.source_timestamp_sec != right.source_timestamp_sec)
  {
    return left.source_timestamp_sec > right.source_timestamp_sec;
  }
  return false;
}

AdaptivePrimarySelectionProposal retainHighestScoringCameras(
  AdaptiveVisualPolicyPlan & plan, const std::size_t maximum_count,
  const AdaptiveCameraObservations & observations,
  const std::optional<std::uint8_t> primary_camera,
  const AdaptiveSelectionChallengerState & current_challenger,
  const double switch_margin,
  const std::size_t challenger_frames)
{
  AdaptivePrimarySelectionProposal proposal;
  std::vector<std::size_t> indices = enabledCameraIndices(plan);
  if (indices.empty())
  {
    if (
      maximum_count == 1U && primary_camera &&
      *primary_camera < plan.cameras.size())
    {
      proposal.primary_camera = primary_camera;
      proposal.reason =
        AdaptiveSelectionReason::kAwaitingChallengerConfirmation;
      retainChallengerAcrossOneSoftGap(
        current_challenger, proposal);
    }
    return proposal;
  }
  std::stable_sort(
    indices.begin(), indices.end(),
    [&plan, &observations](
      const std::size_t lhs, const std::size_t rhs)
    {
      return cameraRanksAhead(
        plan, observations, lhs, rhs);
    });
  if (maximum_count == 1U)
  {
    std::size_t selected_camera = indices.front();
    proposal.primary_camera =
      static_cast<std::uint8_t>(selected_camera);
    if (!primary_camera)
    {
      proposal.reason =
        AdaptiveSelectionReason::kSelectedBestHealthy;
    }
    else if (
      *primary_camera >= plan.cameras.size() ||
      !plan.cameras[*primary_camera].enabled)
    {
      const bool primary_image_hard_failure =
        *primary_camera < plan.cameras.size() &&
        plan.cameras[*primary_camera].image_hard_failure;
      const bool challenger_requires_maintenance =
        selected_camera < observations.size() &&
        maintenanceOnlyCameraCandidate(
        observations[selected_camera],
        plan.cameras[selected_camera]);
      if (
        challenger_requires_maintenance &&
        !primary_image_hard_failure)
      {
        // A soft tracking collapse must not rotate ownership merely because
        // another camera can only bootstrap or reseed. Keep scouting until
        // the primary recovers or a challenger has real tracking support.
        proposal.primary_camera = *primary_camera;
        proposal.reason =
          AdaptiveSelectionReason::kAwaitingChallengerConfirmation;
        retainChallengerAcrossOneSoftGap(
          current_challenger, proposal);
        for (const std::size_t camera_id : indices)
        {
          plan.cameras[camera_id].enabled = false;
        }
        return proposal;
      }
      const bool hard_failure =
        *primary_camera < plan.cameras.size() &&
        (primary_image_hard_failure ||
        plan.cameras[*primary_camera].isolated);
      if (hard_failure)
      {
        proposal.reason =
          AdaptiveSelectionReason::kSwitchedAfterHardFault;
      }
      else
      {
        proposal.primary_camera = *primary_camera;
        if (!confirmSelectionChallenger(
            selected_camera, current_challenger,
            challenger_frames, proposal))
        {
          proposal.reason =
            AdaptiveSelectionReason::
            kAwaitingChallengerConfirmation;
          for (const std::size_t camera_id : indices)
          {
            plan.cameras[camera_id].enabled = false;
          }
          return proposal;
        }
        proposal.primary_camera =
          static_cast<std::uint8_t>(selected_camera);
        proposal.challenger =
          AdaptiveSelectionChallengerState{};
        proposal.reason =
          AdaptiveSelectionReason::
          kSwitchedAfterPrimaryUnavailable;
      }
    }
    else
    {
      const bool primary_requires_maintenance =
        maintenanceOnlyCameraCandidate(
        observations[*primary_camera],
        plan.cameras[*primary_camera]);
      if (
        primary_requires_maintenance &&
        retainChallengerAcrossOneSoftGap(
          current_challenger, proposal))
      {
        proposal.primary_camera = *primary_camera;
        proposal.reason =
          AdaptiveSelectionReason::
          kAwaitingChallengerConfirmation;
        for (const std::size_t camera_id : indices)
        {
          plan.cameras[camera_id].enabled = false;
        }
        return proposal;
      }
      selected_camera = *primary_camera;
      proposal.primary_camera = *primary_camera;
      proposal.reason =
        AdaptiveSelectionReason::kRetainedHealthyPrimary;
      const std::size_t best_camera = indices.front();
      if (best_camera != *primary_camera)
      {
        const double retained_score =
          plan.cameras[*primary_camera].selection_score;
        const double switch_threshold =
          retained_score * (1.0 + switch_margin);
        if (
          plan.cameras[best_camera].selection_score >
          switch_threshold)
        {
          if (confirmSelectionChallenger(
              best_camera, current_challenger,
              challenger_frames, proposal))
          {
            selected_camera = best_camera;
            proposal.primary_camera =
              static_cast<std::uint8_t>(best_camera);
            proposal.challenger =
              AdaptiveSelectionChallengerState{};
            proposal.reason =
              AdaptiveSelectionReason::
              kSwitchedAfterChallengerConfirmation;
          }
        }
      }
    }
    for (const std::size_t camera_id : indices)
    {
      plan.cameras[camera_id].enabled =
        camera_id == selected_camera;
    }
    return proposal;
  }
  if (indices.size() <= maximum_count)
  {
    return proposal;
  }
  for (std::size_t index = maximum_count; index < indices.size(); ++index)
  {
    plan.cameras[indices[index]].enabled = false;
  }
  return proposal;
}

void allocateBudget(
  AdaptiveVisualPolicyPlan & plan, const std::size_t total_budget,
  const std::size_t minimum_budget, const bool candidate_budget)
{
  const std::vector<std::size_t> indices = enabledCameraIndices(plan);
  if (indices.empty())
  {
    return;
  }

  const std::size_t base_total = minimum_budget * indices.size();
  const std::size_t remainder =
    total_budget > base_total ? total_budget - base_total : 0U;
  double score_sum = 0.0;
  for (const std::size_t camera_id : indices)
  {
    score_sum += plan.cameras[camera_id].selection_score;
  }
  if (!(score_sum > 0.0))
  {
    score_sum = static_cast<double>(indices.size());
  }

  struct FractionalAllocation
  {
    std::size_t camera_id;
    double fraction;
  };
  std::vector<FractionalAllocation> fractional;
  fractional.reserve(indices.size());
  std::size_t allocated = base_total;
  for (const std::size_t camera_id : indices)
  {
    const double score =
      plan.cameras[camera_id].selection_score > 0.0 ?
      plan.cameras[camera_id].selection_score : 1.0;
    const double exact_extra =
      static_cast<double>(remainder) * score / score_sum;
    const std::size_t integral_extra =
      static_cast<std::size_t>(std::floor(exact_extra));
    const std::size_t budget = minimum_budget + integral_extra;
    if (candidate_budget)
    {
      plan.cameras[camera_id].candidate_budget = budget;
    }
    else
    {
      plan.cameras[camera_id].patch_budget = budget;
    }
    allocated += integral_extra;
    fractional.push_back(
      FractionalAllocation{
        camera_id, exact_extra - static_cast<double>(integral_extra)});
  }

  std::stable_sort(
    fractional.begin(), fractional.end(),
    [](const FractionalAllocation & lhs,
       const FractionalAllocation & rhs)
    {
      return lhs.fraction > rhs.fraction;
    });
  for (std::size_t index = 0U;
       allocated < total_budget && index < fractional.size();
       ++index, ++allocated)
  {
    AdaptiveCameraDecision & camera =
      plan.cameras[fractional[index].camera_id];
    if (candidate_budget)
    {
      ++camera.candidate_budget;
    }
    else
    {
      ++camera.patch_budget;
    }
  }
}

bool minimumBudgetProductFits(
  const std::size_t total_budget,
  const std::size_t minimum_budget,
  const std::size_t selected_camera_count)
{
  return
    selected_camera_count != 0U &&
    minimum_budget <= total_budget / selected_camera_count;
}

void validateConfig(const AdaptiveVisualPolicyConfig & config)
{
  const std::array<double, 21U> double_values{
    config.rotation_scale_rad,
    config.translation_scale_m,
    config.weak_relative_eigenvalue,
    config.weak_absolute_eigenvalue,
    config.health_off_threshold,
    config.health_on_threshold,
    config.hard_reject_health,
    config.minimum_probation_duration_sec,
    config.covariance_health_floor,
    config.maximum_covariance_scale,
    config.complement_score_gain,
    config.selection_switch_margin,
    config.healthy_tracking_ratio,
    config.healthy_residual_rmse,
    config.maximum_saturation_fraction,
    config.maximum_dark_fraction,
    config.healthy_mean_gradient,
    config.healthy_spatial_coverage,
    config.minimum_inlier_ratio,
    config.nis_chi_square_scale,
    config.maximum_normalized_pose_correction};
  const bool all_double_values_finite = std::all_of(
    double_values.begin(), double_values.end(),
    [](const double value)
    {
      return std::isfinite(value);
    });
  const bool invalid =
    !all_double_values_finite ||
    !(config.rotation_scale_rad > 0.0) ||
    !(config.translation_scale_m > 0.0) ||
    !(config.weak_relative_eigenvalue > 0.0) ||
    !(config.weak_absolute_eigenvalue > 0.0) ||
    !(config.health_off_threshold >= 0.0) ||
    !(config.health_on_threshold > config.health_off_threshold) ||
    !(config.health_on_threshold <= 1.0) ||
    !(config.hard_reject_health >= 0.0) ||
    !(config.hard_reject_health <= config.health_off_threshold) ||
    config.off_hysteresis_frames == 0U ||
    config.on_hysteresis_frames == 0U ||
    config.minimum_probation_duration_sec < 0.0 ||
    !(config.covariance_health_floor > 0.0) ||
    !(config.covariance_health_floor <= 1.0) ||
    !(config.maximum_covariance_scale >= 1.0) ||
    !(config.complement_score_gain >= 0.0) ||
    !(config.selection_switch_margin >= 0.0) ||
    !(config.selection_switch_margin <= 1.0) ||
    config.selection_challenger_frames == 0U ||
    config.minimum_selection_tracked_points == 0U ||
    config.healthy_tracked_points == 0U ||
    !(config.healthy_tracking_ratio > 0.0) ||
    !(config.healthy_residual_rmse > 0.0) ||
    !(config.maximum_saturation_fraction > 0.0) ||
    !(config.maximum_dark_fraction > 0.0) ||
    !(config.healthy_mean_gradient > 0.0) ||
    !(config.healthy_spatial_coverage > 0.0) ||
    config.minimum_inliers == 0U ||
    !(config.minimum_inlier_ratio > 0.0) ||
    !(config.minimum_inlier_ratio <= 1.0) ||
    !(config.nis_chi_square_scale > 0.0) ||
    !(config.maximum_normalized_pose_correction > 0.0) ||
    config.minimum_candidate_budget == 0U ||
    config.minimum_patch_budget == 0U ||
    config.minimum_selected_cameras == 0U ||
    config.minimum_selected_cameras > kAdaptiveVisualCameraCount ||
    config.maximum_selected_cameras == 0U ||
    config.maximum_selected_cameras > kAdaptiveVisualCameraCount ||
    config.minimum_selected_cameras > config.maximum_selected_cameras ||
    !minimumBudgetProductFits(
      config.total_candidate_budget,
      config.minimum_candidate_budget,
      config.minimum_selected_cameras) ||
    !minimumBudgetProductFits(
      config.total_patch_budget,
      config.minimum_patch_budget,
      config.minimum_selected_cameras);
  if (invalid)
  {
    throw std::invalid_argument("Invalid adaptive visual policy configuration");
  }
}

}  // namespace

#ifdef LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING
namespace test
{

void failNextPolicyOwnerAllocation()
{
  fail_next_policy_owner_allocation.store(
    true, std::memory_order_release);
}

}  // namespace test
#endif

AdaptiveVisualPolicyPlanCapability::
AdaptiveVisualPolicyPlanCapability(
  std::shared_ptr<const Data> data)
: data_(std::move(data))
{
}

bool AdaptiveVisualPolicyPlanCapability::valid() const noexcept
{
  if (
    !data_ ||
    !data_->owner ||
    data_->config_revision !=
      kAdaptiveVisualPolicyConfigRevision ||
    data_->base_policy_revision !=
      data_->decision_plan.base_revision ||
    data_->runtime_publication_sequence == 0U ||
    data_->camera_batch_sequence == 0U ||
    !data_->runtime.valid() ||
    data_->runtime.publicationSequence() !=
      data_->runtime_publication_sequence)
  {
    return false;
  }
  return
    data_->runtime.cameraBatchReceipt().batchSequence() ==
    data_->camera_batch_sequence;
}

std::uint32_t
AdaptiveVisualPolicyPlanCapability::configRevision() const noexcept
{
  return valid() ? data_->config_revision : 0U;
}

std::uint64_t
AdaptiveVisualPolicyPlanCapability::basePolicyRevision() const noexcept
{
  return valid() ? data_->base_policy_revision : 0U;
}

std::uint64_t
AdaptiveVisualPolicyPlanCapability::
runtimePublicationSequence() const noexcept
{
  return
    valid() ?
    data_->runtime_publication_sequence : 0U;
}

std::uint64_t
AdaptiveVisualPolicyPlanCapability::cameraBatchSequence() const noexcept
{
  return valid() ? data_->camera_batch_sequence : 0U;
}

bool
AdaptiveVisualPolicyPlanCapability::lidarInformationReady() const noexcept
{
  return valid() && data_->lidar_information_ready;
}

const AdaptiveVisualPolicyPlan &
AdaptiveVisualPolicyPlanCapability::decisionPlan() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual policy plan capability");
  }
  return data_->decision_plan;
}

const Eigen::Matrix<double, 6, 6> &
AdaptiveVisualPolicyPlanCapability::lidarPoseInformation() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual policy plan capability");
  }
  return data_->lidar_pose_information;
}

const AdaptiveCameraObservations &
AdaptiveVisualPolicyPlanCapability::observations() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual policy plan capability");
  }
  return data_->observations;
}

const ReferenceOnlyMigrationRuntimeBundle &
AdaptiveVisualPolicyPlanCapability::runtimeBundle() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual policy plan capability");
  }
  return data_->runtime;
}

AdaptiveVisualCandidatePlanCapability::
AdaptiveVisualCandidatePlanCapability(
  std::shared_ptr<const Data> data)
: data_(std::move(data))
{
}

bool AdaptiveVisualCandidatePlanCapability::valid() const noexcept
{
  if (
    !data_ ||
    !data_->owner ||
    !data_->camera_batch_receipt.valid() ||
    data_->config_revision !=
      kAdaptiveVisualPolicyConfigRevision ||
    data_->base_policy_revision !=
      data_->decision_plan.base_revision ||
    data_->camera_batch_sequence == 0U ||
    data_->camera_batch_sequence !=
      data_->camera_batch_receipt.batchSequence() ||
    data_->camera_count == 0U ||
    data_->camera_count > kAdaptiveVisualCameraCount ||
    !std::isfinite(data_->batch_timestamp_sec))
  {
    return false;
  }
  return
    static_cast<std::size_t>(
      std::count(
        data_->candidate.begin(), data_->candidate.end(),
        true)) == data_->camera_count;
}

std::uint32_t
AdaptiveVisualCandidatePlanCapability::configRevision() const noexcept
{
  return valid() ? data_->config_revision : 0U;
}

std::uint64_t
AdaptiveVisualCandidatePlanCapability::basePolicyRevision() const noexcept
{
  return valid() ? data_->base_policy_revision : 0U;
}

std::uint64_t
AdaptiveVisualCandidatePlanCapability::cameraBatchSequence() const noexcept
{
  return valid() ? data_->camera_batch_sequence : 0U;
}

bool
AdaptiveVisualCandidatePlanCapability::lidarInformationReady() const noexcept
{
  return valid() && data_->lidar_information_ready;
}

const std::array<bool, kAdaptiveVisualCameraCount> &
AdaptiveVisualCandidatePlanCapability::candidateMask() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual candidate plan capability");
  }
  return data_->candidate;
}

std::size_t
AdaptiveVisualCandidatePlanCapability::cameraCount() const noexcept
{
  return valid() ? data_->camera_count : 0U;
}

double
AdaptiveVisualCandidatePlanCapability::batchTimestampSec() const noexcept
{
  return
    valid() ?
    data_->batch_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

AdaptiveSelectionReason
AdaptiveVisualCandidatePlanCapability::selectionReason() const noexcept
{
  return
    valid() ?
    data_->decision_plan.selection_reason :
    AdaptiveSelectionReason::kNotApplicable;
}

const AdaptiveVisualPolicyPlan &
AdaptiveVisualCandidatePlanCapability::decisionPlan() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual candidate plan capability");
  }
  return data_->decision_plan;
}

const Eigen::Matrix<double, 6, 6> &
AdaptiveVisualCandidatePlanCapability::lidarPoseInformation() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual candidate plan capability");
  }
  return data_->lidar_pose_information;
}

const AdaptiveCameraObservations &
AdaptiveVisualCandidatePlanCapability::observations() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual candidate plan capability");
  }
  return data_->observations;
}

const CameraObservationBatchReceipt &
AdaptiveVisualCandidatePlanCapability::cameraBatchReceipt() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual candidate plan capability");
  }
  return data_->camera_batch_receipt;
}

const char * adaptiveCameraTransitionName(
  const AdaptiveCameraTransition transition)
{
  switch (transition)
  {
    case AdaptiveCameraTransition::kIsolated:
      return "isolated";
    case AdaptiveCameraTransition::kRecovered:
      return "recovered";
    case AdaptiveCameraTransition::kNone:
    default:
      return "none";
  }
}

const char * adaptiveCameraHealthStateName(
  const AdaptiveCameraHealthState state) noexcept
{
  switch (state)
  {
    case AdaptiveCameraHealthState::kHealthy:
      return "healthy";
    case AdaptiveCameraHealthState::kDegraded:
      return "degraded";
    case AdaptiveCameraHealthState::kIsolated:
      return "isolated";
    case AdaptiveCameraHealthState::kProbation:
      return "probation";
  }
  return "unknown";
}

const char * adaptivePolicyReasonName(const AdaptivePolicyReason reason)
{
  switch (reason)
  {
    case AdaptivePolicyReason::kReady:
      return "ready";
    case AdaptivePolicyReason::kDisabled:
      return "disabled";
    case AdaptivePolicyReason::kLidarNotReady:
      return "lidar_not_ready";
    case AdaptivePolicyReason::kLidarNonFinite:
      return "lidar_non_finite";
    case AdaptivePolicyReason::kLidarNonSymmetric:
      return "lidar_non_symmetric";
    case AdaptivePolicyReason::kLidarInsufficientInformation:
      return "lidar_insufficient_information";
    case AdaptivePolicyReason::kInsufficientCameras:
      return "insufficient_cameras";
    case AdaptivePolicyReason::kInvalidCameraObservations:
      return "invalid_camera_observations";
    default:
      return "unknown";
  }
}

const char * adaptiveSelectionReasonName(
  const AdaptiveSelectionReason reason)
{
  switch (reason)
  {
    case AdaptiveSelectionReason::kSelectedBestHealthy:
      return "selected_best_healthy";
    case AdaptiveSelectionReason::kRetainedHealthyPrimary:
      return "retained_healthy_primary";
    case AdaptiveSelectionReason::
      kSwitchedAfterChallengerConfirmation:
      return "switched_after_challenger_confirmation";
    case AdaptiveSelectionReason::kSwitchedAfterHardFault:
      return "switched_after_hard_fault";
    case AdaptiveSelectionReason::
      kSwitchedAfterPrimaryUnavailable:
      return "switched_after_primary_unavailable";
    case AdaptiveSelectionReason::
      kAwaitingChallengerConfirmation:
      return "awaiting_challenger_confirmation";
    case AdaptiveSelectionReason::kSelectedBestInformationGain:
      return "selected_best_information_gain";
    case AdaptiveSelectionReason::kNotApplicable:
    default:
      return "not_applicable";
  }
}

const char * adaptiveCandidateHealthCommitStatusName(
  const AdaptiveCandidateHealthCommitStatus status)
{
  switch (status)
  {
    case AdaptiveCandidateHealthCommitStatus::kCommitted:
      return "committed";
    case AdaptiveCandidateHealthCommitStatus::kUntrustedScout:
      return "untrusted_scout";
    case AdaptiveCandidateHealthCommitStatus::
      kInvalidOrForeignCapability:
      return "invalid_or_foreign_capability";
    case AdaptiveCandidateHealthCommitStatus::
      kInvalidOrForeignReceipt:
      return "invalid_or_foreign_receipt";
    case AdaptiveCandidateHealthCommitStatus::kBindingMismatch:
      return "binding_mismatch";
    case AdaptiveCandidateHealthCommitStatus::kStalePolicyRevision:
      return "stale_policy_revision";
    case AdaptiveCandidateHealthCommitStatus::
      kNonMonotonicOrReplayedLidarGeneration:
      return "nonmonotonic_or_replayed_lidar_generation";
    case AdaptiveCandidateHealthCommitStatus::kRevisionExhausted:
      return "revision_exhausted";
    default:
      return "unknown";
  }
}

AdaptiveVisualMapMaintenanceDecision
planAdaptiveLioAnchoredMapMaintenance(
  const AdaptiveVisualLioAnchoredMaintenanceRequest request,
  const AdaptiveVisualMapLifecycle lifecycle,
  const bool camera_route_authorized,
  const bool camera_isolated,
  const bool image_seed_eligible,
  const bool lidar_information_valid)
{
  if (!camera_route_authorized)
  {
    return {};
  }
  if (
    camera_isolated || !image_seed_eligible ||
    !lidar_information_valid)
  {
    return {};
  }

  const bool lifecycle_allows_request =
    request ==
      AdaptiveVisualLioAnchoredMaintenanceRequest::kBootstrap ?
    lifecycle == AdaptiveVisualMapLifecycle::kEmpty ||
      lifecycle == AdaptiveVisualMapLifecycle::kSeededUnvalidated :
    lifecycle != AdaptiveVisualMapLifecycle::kEmpty;
  if (!lifecycle_allows_request)
  {
    return {};
  }

  return AdaptiveVisualMapMaintenanceDecision{true, false};
}

AdaptiveVisualMapMaintenanceDecision
planAdaptiveVisualSolverMapCommit(
  const bool adaptive_enabled,
  const bool camera_route_authorized,
  const bool visual_update_accepted)
{
  if (!adaptive_enabled)
  {
    // Preserve the established joint-four behavior exactly when adaptive
    // routing is disabled.
    return AdaptiveVisualMapMaintenanceDecision{true, true};
  }
  if (!camera_route_authorized || !visual_update_accepted)
  {
    return {};
  }
  return AdaptiveVisualMapMaintenanceDecision{true, true};
}

std::size_t adaptiveScoutCandidateBudget(
  const std::size_t total_candidate_budget,
  const std::size_t camera_id)
{
  if (camera_id >= kAdaptiveVisualCameraCount)
  {
    return 0U;
  }
  const std::size_t base =
    total_candidate_budget / kAdaptiveVisualCameraCount;
  const std::size_t remainder =
    total_candidate_budget % kAdaptiveVisualCameraCount;
  return base + (camera_id < remainder ? 1U : 0U);
}

AdaptiveVisualPolicy::AdaptiveVisualPolicy(
  const AdaptiveVisualPolicyConfig & config)
: config_(config),
  capability_owner_(
    std::make_shared<const std::uint8_t>(0U))
{
  validateConfig(config_);
  hysteresis_.fill(AdaptiveCameraHysteresisState{});
}

AdaptiveVisualPolicyPlan AdaptiveVisualPolicy::plan(
  const Matrix6 & lidar_pose_information,
  const CameraObservations & observations,
  const bool lidar_information_ready) const
{
  AdaptiveVisualPolicyPlan result;
  result.adaptive_enabled = config_.enabled;
  result.base_revision = revision_;
  result.proposed_hysteresis = hysteresis_;
  result.proposed_primary_camera = primary_camera_;
  result.proposed_selection_challenger =
    selection_challenger_;
  if (!config_.enabled)
  {
    result.reason = AdaptivePolicyReason::kDisabled;
    for (AdaptiveCameraDecision & camera : result.cameras)
    {
      camera.frame_received = true;
      camera.available = true;
      camera.eligible = true;
      camera.enabled = true;
      camera.health_state = AdaptiveCameraHealthState::kHealthy;
      camera.isolated = false;
      camera.metrics_complete = true;
      camera.health_evidence_complete = true;
      camera.image_hard_failure = false;
      camera.image_recovery_healthy = true;
      camera.health_degraded = false;
      camera.health = 1.0;
      camera.covariance_scale = 1.0;
      camera.candidate_budget = std::numeric_limits<std::size_t>::max();
      camera.patch_budget = std::numeric_limits<std::size_t>::max();
    }
    return result;
  }

  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    if (
      observations[camera_id].camera_id !=
      static_cast<std::uint8_t>(camera_id))
    {
      result.reason =
        AdaptivePolicyReason::kInvalidCameraObservations;
      return result;
    }
  }

  WeakDirectionResult weak;
  if (lidar_information_ready)
  {
    weak = computeWeakDirections(
      lidar_pose_information, config_);
  }
  else
  {
    weak.reason = AdaptivePolicyReason::kLidarNotReady;
  }
  result.lidar_information_valid = weak.valid;
  result.reason = weak.reason;
  result.weak_direction_count = weak.valid ? weak.count : 0U;
  result.lidar_eigenvalues = weak.eigenvalues;
  result.weak_direction_weight =
    weak.valid ? weak.weight : Matrix6::Zero();

  for (std::size_t observation_index = 0U;
       observation_index < observations.size(); ++observation_index)
  {
    const AdaptiveCameraObservation & observation =
      observations[observation_index];
    if (observation.camera_id >= result.cameras.size())
    {
      continue;
    }
    AdaptiveCameraDecision & decision =
      result.cameras[observation.camera_id];
    decision.frame_received = observation.frame_received;
    const bool sealed_adaptive_runtime =
      config_.enabled &&
      config_.minimum_selected_cameras == 1U &&
      config_.maximum_selected_cameras == 1U;
    decision.metrics_complete = hasAllMetrics(
      observation, kAdaptiveFirstVersionMetricMask);
    decision.health = computeHealth(observation, config_);
    decision.available = measurementAvailable(observation, config_);
    decision.robust_information_gain =
      observation.robust_information_gain;
    decision.robust_gate_passed =
      !sealed_adaptive_runtime ||
      (decision.available &&
      robustAdmissionPassed(observation, config_));
    decision.health_evidence_complete =
      decision.metrics_complete && decision.available;
    decision.image_hard_failure =
      imageHardFailure(observation, config_);
    decision.image_recovery_healthy =
      imageRecoveryHealthy(observation, config_);
    const bool zero_map_preselection_eligible =
      zeroMapPreselectionEligible(
      observation, decision.image_recovery_healthy,
      decision.image_hard_failure);
    const bool tracking_failure_evidence =
      trackingFailureEvidence(observation);
    decision.complement_score = complementScore(
      observation, result.weak_direction_weight, config_);

    AdaptiveCameraHysteresisState & proposed =
      result.proposed_hysteresis[observation.camera_id];
    const bool was_isolated = proposed.isolated;
    proposed = proposeHysteresis(
      proposed, observation.source_timestamp_sec,
      decision.frame_received,
      decision.health_evidence_complete,
      decision.health, decision.image_hard_failure,
      decision.image_recovery_healthy,
      tracking_failure_evidence, config_);
    decision.isolated = proposed.isolated;
    if (!was_isolated && proposed.isolated)
    {
      decision.transition = AdaptiveCameraTransition::kIsolated;
    }
    else if (was_isolated && !proposed.isolated)
    {
      decision.transition = AdaptiveCameraTransition::kRecovered;
    }
    decision.health_degraded =
      !decision.health_evidence_complete ||
      decision.image_hard_failure ||
      decision.health < config_.health_on_threshold;
    if (!decision.isolated)
    {
      proposed.state = decision.health_degraded ?
        AdaptiveCameraHealthState::kDegraded :
        AdaptiveCameraHealthState::kHealthy;
    }
    decision.health_state = proposed.state;
    decision.covariance_scale = std::min(
      config_.maximum_covariance_scale,
      1.0 / std::max(
      decision.health, config_.covariance_health_floor));
    decision.reject_reason = cameraRejectReason(
      decision, config_,
      zero_map_preselection_eligible);
    decision.eligible =
      (decision.available ||
      zero_map_preselection_eligible) &&
      !decision.isolated &&
      !decision.image_hard_failure &&
      decision.health >= config_.hard_reject_health;
    if (sealed_adaptive_runtime)
    {
      // Health/hysteresis and the robust gates remain diagnostic tiers in
      // the sealed path. Basic measurement integrity is the route safety
      // boundary; robust admission only chooses the preferred candidate
      // pool below.
      decision.eligible =
        decision.frame_received &&
        decision.metrics_complete &&
        decision.available &&
        !decision.image_hard_failure;
      decision.reject_reason =
        decision.eligible ?
        AdaptiveCameraRejectReason::kNone :
        !decision.frame_received ?
        AdaptiveCameraRejectReason::kFrameMissing :
        decision.image_hard_failure ?
        AdaptiveCameraRejectReason::kImageHardFailure :
        AdaptiveCameraRejectReason::kTrackingUnavailable;
    }
    decision.enabled =
      result.lidar_information_valid && decision.eligible;
  }

  double maximum_complement = 0.0;
  for (const AdaptiveCameraDecision & decision : result.cameras)
  {
    if (decision.enabled)
    {
      maximum_complement =
        std::max(maximum_complement, decision.complement_score);
    }
  }
  for (AdaptiveCameraDecision & decision : result.cameras)
  {
    if (!decision.enabled)
    {
      decision.selection_score = 0.0;
      continue;
    }
    const double normalized_complement =
      maximum_complement > config_.weak_absolute_eigenvalue ?
      decision.complement_score / maximum_complement : 0.0;
    decision.selection_score =
      decision.health *
      (1.0 + config_.complement_score_gain * normalized_complement);
  }

  if (!result.lidar_information_valid)
  {
    return result;
  }

  const bool sealed_adaptive_runtime =
    config_.minimum_selected_cameras == 1U &&
    config_.maximum_selected_cameras == 1U;
  if (sealed_adaptive_runtime)
  {
    std::vector<std::size_t> base_indices;
    std::vector<std::size_t> robust_indices;
    for (std::size_t camera_id = 0U;
         camera_id < result.cameras.size(); ++camera_id)
    {
      AdaptiveCameraDecision & decision = result.cameras[camera_id];
      const double information_gain =
        observations[camera_id].robust_information_gain;
      if (decision.robust_gate_passed &&
          (!std::isfinite(information_gain) ||
          information_gain < 0.0))
      {
        decision.robust_gate_passed = false;
      }
      decision.robust_information_gain = information_gain;
      if (!decision.enabled)
      {
        continue;
      }
      base_indices.push_back(camera_id);
      if (decision.robust_gate_passed)
      {
        robust_indices.push_back(camera_id);
      }
    }

    if (base_indices.empty())
    {
      result.reason = AdaptivePolicyReason::kInsufficientCameras;
      result.selection_reason = AdaptiveSelectionReason::kNotApplicable;
      return result;
    }

    const bool robust_pool_available = !robust_indices.empty();
    for (const std::size_t camera_id : base_indices)
    {
      AdaptiveCameraDecision & decision = result.cameras[camera_id];
      if (robust_pool_available)
      {
        decision.enabled = decision.robust_gate_passed;
        decision.selection_score =
          decision.robust_gate_passed ?
          decision.robust_information_gain : 0.0;
      }
    }

    const AdaptivePrimarySelectionProposal selection_proposal =
      retainHighestScoringCameras(
      result, 1U, observations,
      primary_camera_, selection_challenger_,
      config_.selection_switch_margin,
      config_.selection_challenger_frames);
    result.proposed_primary_camera =
      selection_proposal.primary_camera;
    result.proposed_selection_challenger =
      selection_proposal.challenger;
    result.selection_reason =
      robust_pool_available &&
      selection_proposal.reason ==
      AdaptiveSelectionReason::kSelectedBestHealthy ?
      AdaptiveSelectionReason::kSelectedBestInformationGain :
      selection_proposal.reason;

    // If the old primary left the active tier, challenger confirmation must
    // not turn a valid candidate set into a zero-winner plan. Route the best
    // candidate from the active tier while retaining the proposed primary
    // and challenger state until confirmation completes.
    if (enabledCameraIndices(result).empty())
    {
      const std::vector<std::size_t> & active_indices =
        robust_pool_available ? robust_indices : base_indices;
      const auto selected = std::min_element(
        active_indices.begin(), active_indices.end(),
        [&result, &observations](
          const std::size_t lhs, const std::size_t rhs)
        {
          return cameraRanksAhead(
            result, observations, lhs, rhs);
        });
      result.cameras[*selected].enabled = true;
    }
    allocateBudget(
      result, config_.total_candidate_budget,
      config_.minimum_candidate_budget, true);
    allocateBudget(
      result, config_.total_patch_budget,
      config_.minimum_patch_budget, false);
    return result;
  }

  const std::size_t candidate_capacity =
    config_.total_candidate_budget /
    config_.minimum_candidate_budget;
  const std::size_t patch_capacity =
    config_.total_patch_budget / config_.minimum_patch_budget;
  const AdaptivePrimarySelectionProposal selection_proposal =
    retainHighestScoringCameras(
    result,
    std::min(
      config_.maximum_selected_cameras,
      std::min(candidate_capacity, patch_capacity)),
    observations,
    primary_camera_, selection_challenger_,
    config_.selection_switch_margin,
    config_.selection_challenger_frames);
  result.proposed_primary_camera =
    selection_proposal.primary_camera;
  result.proposed_selection_challenger =
    selection_proposal.challenger;
  result.selection_reason = selection_proposal.reason;
  if (enabledCameraIndices(result).size() <
      config_.minimum_selected_cameras)
  {
    result.reason = AdaptivePolicyReason::kInsufficientCameras;
    for (AdaptiveCameraDecision & decision : result.cameras)
    {
      decision.enabled = false;
    }
    return result;
  }
  allocateBudget(
    result, config_.total_candidate_budget,
    config_.minimum_candidate_budget, true);
  allocateBudget(
    result, config_.total_patch_budget,
    config_.minimum_patch_budget, false);
  return result;
}

std::optional<AdaptiveVisualPolicyPlanCapability>
AdaptiveVisualPolicy::issuePlan(
  const ReferenceOnlyMigrationRuntimeBundle & runtime,
  const Matrix6 & lidar_pose_information,
  const CameraObservations & observations,
  const bool lidar_information_ready)
{
  if (!runtime.valid())
  {
    return std::nullopt;
  }
  const std::uint64_t publication_sequence =
    runtime.publicationSequence();
  const std::uint64_t batch_sequence =
    runtime.cameraBatchReceipt().batchSequence();
  if (
    publication_sequence == 0U ||
    batch_sequence == 0U ||
    publication_sequence <=
      last_issued_publication_sequence_ ||
    batch_sequence <= last_issued_batch_sequence_)
  {
    return std::nullopt;
  }

  AdaptiveVisualPolicyPlan decision_plan =
    plan(
    lidar_pose_information, observations,
    lidar_information_ready);
  auto data =
    std::make_shared<
    const AdaptiveVisualPolicyPlanCapability::Data>(
    capability_owner_, runtime, lidar_pose_information,
    observations, lidar_information_ready,
    std::move(decision_plan), publication_sequence,
    batch_sequence);
  AdaptiveVisualPolicyPlanCapability capability{
    std::move(data)};
  if (!capability.valid())
  {
    return std::nullopt;
  }

  last_issued_publication_sequence_ =
    publication_sequence;
  last_issued_batch_sequence_ = batch_sequence;
  return capability;
}

std::optional<AdaptiveVisualCandidatePlanCapability>
AdaptiveVisualPolicy::issueCandidatePlan(
  const CameraObservationBatch & batch,
  const Matrix6 & lidar_pose_information,
  const CameraObservations & observations,
  const bool lidar_information_ready)
{
  const auto & receipt = batch.validatedReceipt();
  if (
    !receipt.has_value() ||
    !receipt->valid() ||
    receipt->sealedCameraCount() == 0U ||
    receipt->sealedCameraCount() >
      kAdaptiveVisualCameraCount ||
    std::count(
      receipt->sealedPresentMask().begin(),
      receipt->sealedPresentMask().end(),
      true) != static_cast<std::ptrdiff_t>(
      receipt->sealedCameraCount()) ||
    !std::isfinite(
      receipt->sealedBatchTimestampSec()))
  {
    return std::nullopt;
  }
  if (receipt->batchSequence() == 0U)
  {
    return std::nullopt;
  }
  for (std::size_t camera_id = 0U;
       camera_id < kAdaptiveVisualCameraCount;
       ++camera_id)
  {
    if (
      !receipt->sealedPresentMask()[camera_id] &&
      !isCanonicalAbsentObservation(
        observations[camera_id],
        static_cast<std::uint8_t>(camera_id)))
    {
      return std::nullopt;
    }
  }
  if (!receipt->matchesAndValidates(batch))
  {
    return std::nullopt;
  }

  AdaptiveVisualPolicyPlan decision_plan =
    plan(
    lidar_pose_information, observations,
    lidar_information_ready);
  auto data =
    std::make_shared<
    const AdaptiveVisualCandidatePlanCapability::Data>(
    capability_owner_, *receipt,
    receipt->sealedPresentMask(),
    receipt->sealedCameraCount(),
    receipt->sealedBatchTimestampSec(),
    lidar_pose_information, observations,
    lidar_information_ready,
    std::move(decision_plan));
  AdaptiveVisualCandidatePlanCapability capability{
    std::move(data)};
  if (!capability.valid()) {
    return std::nullopt;
  }

  return capability;
}

bool AdaptiveVisualPolicy::accepts(
  const AdaptiveVisualPolicyPlanCapability & capability)
  const noexcept
{
  return
    ownsPlanCapability(capability) &&
    capability.data_->config_revision ==
      kAdaptiveVisualPolicyConfigRevision &&
    capability.data_->base_policy_revision == revision_ &&
    capability.data_->runtime_publication_sequence ==
      last_issued_publication_sequence_ &&
    capability.data_->camera_batch_sequence ==
      last_issued_batch_sequence_ &&
    capability.data_->decision_plan.adaptive_enabled ==
      config_.enabled;
}

bool AdaptiveVisualPolicy::accepts(
  const AdaptiveVisualCandidatePlanCapability & capability)
  const noexcept
{
  return
    ownsCandidatePlanCapability(capability) &&
    capability.data_->config_revision ==
      kAdaptiveVisualPolicyConfigRevision &&
    capability.data_->base_policy_revision == revision_ &&
    capability.data_->decision_plan.adaptive_enabled ==
      config_.enabled;
}

bool AdaptiveVisualPolicy::ownsPlanCapability(
  const AdaptiveVisualPolicyPlanCapability & capability)
  const noexcept
{
  return
    capability.valid() &&
    sameOwner(
      capability.data_->owner, capability_owner_);
}

bool AdaptiveVisualPolicy::ownsCandidatePlanCapability(
  const AdaptiveVisualCandidatePlanCapability & capability)
  const noexcept
{
  return
    capability.valid() &&
    sameOwner(
      capability.data_->owner, capability_owner_);
}

bool AdaptiveVisualPolicy::commitHealthObservation(
  const AdaptiveVisualPolicyPlanCapability & capability,
  const bool scout_trusted)
{
  if (
    !scout_trusted ||
    !accepts(capability) ||
    !capability.data_->decision_plan.
      lidar_information_valid ||
    revision_ ==
      std::numeric_limits<std::uint64_t>::max())
  {
    return false;
  }
  const bool sealed_adaptive_runtime =
    config_.minimum_selected_cameras == 1U &&
    config_.maximum_selected_cameras == 1U;
  if (!sealed_adaptive_runtime)
  {
    hysteresis_ =
      capability.data_->decision_plan.proposed_hysteresis;
    primary_camera_ =
      capability.data_->decision_plan.proposed_primary_camera;
    selection_challenger_ =
      capability.data_->decision_plan.proposed_selection_challenger;
  }
  ++revision_;
  return true;
}

bool AdaptiveVisualPolicy::commit(
  const AdaptiveVisualPolicyPlan & plan,
  const bool visual_update_accepted)
{
  return commitHealthObservation(plan, visual_update_accepted);
}

bool AdaptiveVisualPolicy::commitHealthObservation(
  const AdaptiveVisualPolicyPlan & plan,
  const bool scout_trusted)
{
  if (!scout_trusted ||
      !plan.lidar_information_valid ||
      plan.base_revision != revision_ ||
      plan.adaptive_enabled != config_.enabled ||
      revision_ ==
        std::numeric_limits<std::uint64_t>::max())
  {
    return false;
  }
  const bool sealed_adaptive_runtime =
    config_.enabled &&
    config_.minimum_selected_cameras == 1U &&
    config_.maximum_selected_cameras == 1U;
  if (!sealed_adaptive_runtime)
  {
    hysteresis_ = plan.proposed_hysteresis;
    primary_camera_ = plan.proposed_primary_camera;
    selection_challenger_ =
      plan.proposed_selection_challenger;
  }
  ++revision_;
  return true;
}

void AdaptiveVisualPolicy::reset()
{
#ifdef LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING
  if (fail_next_policy_owner_allocation.exchange(
      false, std::memory_order_acq_rel))
  {
    throw std::bad_alloc();
  }
#endif
  auto new_capability_owner =
    std::make_shared<const std::uint8_t>(0U);
  const std::lock_guard<std::mutex> transaction_lock(
    candidate_commit_transaction_mutex_);
  hysteresis_.fill(AdaptiveCameraHysteresisState{});
  primary_camera_.reset();
  selection_challenger_ =
    AdaptiveSelectionChallengerState{};
  capability_owner_ =
    std::move(new_capability_owner);
  last_issued_batch_sequence_ = 0U;
  last_issued_publication_sequence_ = 0U;
  if (revision_ <
      std::numeric_limits<std::uint64_t>::max())
  {
    ++revision_;
  }
}

}  // namespace livo
