/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "vio.h"
#include "reference_only_migration_runtime_budget.h"
#include "structured_metric.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <optional>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace
{

struct ImageHealthStatistics
{
  double saturation_fraction{0.0};
  double dark_fraction{0.0};
  double mean_gradient{0.0};
};

ImageHealthStatistics computeImageHealthStatistics(const cv::Mat & gray_image)
{
  ImageHealthStatistics statistics;
  if (gray_image.empty() || gray_image.type() != CV_8UC1)
  {
    return statistics;
  }

  const double pixel_count =
    static_cast<double>(gray_image.rows) *
    static_cast<double>(gray_image.cols);
  statistics.saturation_fraction =
    static_cast<double>(cv::countNonZero(gray_image >= 250)) / pixel_count;
  statistics.dark_fraction =
    static_cast<double>(cv::countNonZero(gray_image <= 20)) / pixel_count;

  cv::Mat gradient_x;
  cv::Mat gradient_y;
  cv::Mat gradient_magnitude;
  cv::Sobel(gray_image, gradient_x, CV_32F, 1, 0, 3);
  cv::Sobel(gray_image, gradient_y, CV_32F, 0, 1, 3);
  cv::magnitude(gradient_x, gradient_y, gradient_magnitude);
  statistics.mean_gradient = cv::mean(gradient_magnitude)[0];
  return statistics;
}

double positiveLogDet6(
  const Eigen::Matrix<double, 6, 6> & information)
{
  const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> factor(
    0.5 * (information + information.transpose()));
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

bool isAdaptiveSingleCameraBatch(
  const livo::AdaptiveVisualPolicyConfig & config,
  const CameraObservationBatch & batch,
  const std::size_t prepared_camera_count)
{
  return
    config.enabled &&
    config.minimum_selected_cameras == 1U &&
    config.maximum_selected_cameras == 1U &&
    batch.partial &&
    batch.camera_count == 1U &&
    std::count(batch.present.begin(), batch.present.end(), true) == 1 &&
    prepared_camera_count == 1U;
}

bool hasAdaptiveVisualMapSupport(
  const livo::AdaptiveCameraObservation & observation,
  const livo::AdaptiveVisualPolicyConfig & config)
{
  return
    observation.tracked_points >=
      config.minimum_selection_tracked_points &&
    observation.inlier_count >= config.minimum_inliers;
}

template<typename ValueT>
std::vector<ValueT> selectValues(
  const std::vector<ValueT> & values,
  const std::vector<std::size_t> & selected_indices)
{
  std::vector<ValueT> selected;
  selected.reserve(selected_indices.size());
  for (const std::size_t index : selected_indices)
  {
    if (index < values.size())
    {
      selected.push_back(values[index]);
    }
  }
  return selected;
}

std::string fingerprintHex(
  const livo::FourCameraPinholeRigFingerprint & fingerprint)
{
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result;
  result.reserve(fingerprint.size() * 2U);
  for (const std::uint8_t value : fingerprint)
  {
    result.push_back(kHexDigits[value >> 4U]);
    result.push_back(kHexDigits[value & 0x0fU]);
  }
  return result;
}

bool sameRegisteredPinholeModel(
  const livo::CameraModelParameters & registered,
  const livo::CameraModelParameters & requested)
{
  return
    registered.model == "Pinhole" &&
    requested.model == registered.model &&
    requested.width == registered.width &&
    requested.height == registered.height &&
    requested.scale == registered.scale &&
    requested.fx == registered.fx &&
    requested.fy == registered.fy &&
    requested.cx == registered.cx &&
    requested.cy == registered.cy &&
    requested.skew == 0.0 &&
    requested.d0 == registered.d0 &&
    requested.d1 == registered.d1 &&
    requested.d2 == registered.d2 &&
    requested.d3 == registered.d3 &&
    requested.d4 == registered.d4;
}

template<std::size_t Count>
std::string commaSeparatedCounts(
  const std::array<std::size_t, Count> & counts)
{
  std::string result;
  result.reserve(Count * 3U);
  for (std::size_t index = 0U; index < Count; ++index)
  {
    if (index != 0U)
    {
      result.push_back(',');
    }
    result += std::to_string(counts[index]);
  }
  return result;
}

std::array<double, livo::kVioReadOnlyStateScalarCount>
stateScalars(const StatesGroup & state)
{
  std::array<double, livo::kVioReadOnlyStateScalarCount>
    scalars{};
  std::size_t offset = 0U;
  const auto append =
    [&scalars, &offset](
      const double * values, const std::size_t count)
    {
      std::copy_n(
        values, count, scalars.begin() +
        static_cast<std::ptrdiff_t>(offset));
      offset += count;
    };
  append(state.rot_end.data(), 9U);
  append(state.pos_end.data(), 3U);
  append(state.vel_end.data(), 3U);
  scalars[offset++] = state.inv_expo_time;
  append(state.bias_g.data(), 3U);
  append(state.bias_a.data(), 3U);
  append(state.gravity.data(), 3U);
  append(
    state.cov.data(),
    static_cast<std::size_t>(DIM_STATE * DIM_STATE));
  return scalars;
}

bool sameAdaptiveCameraDecision(
  const livo::AdaptiveCameraDecision & left,
  const livo::AdaptiveCameraDecision & right)
{
  return
    left.frame_received == right.frame_received &&
    left.available == right.available &&
    left.eligible == right.eligible &&
    left.enabled == right.enabled &&
    left.isolated == right.isolated &&
    left.metrics_complete == right.metrics_complete &&
    left.health_evidence_complete ==
      right.health_evidence_complete &&
    left.image_hard_failure == right.image_hard_failure &&
    left.image_recovery_healthy ==
      right.image_recovery_healthy &&
    left.health_degraded == right.health_degraded &&
    left.health == right.health &&
    left.covariance_scale == right.covariance_scale &&
    left.complement_score == right.complement_score &&
    left.selection_score == right.selection_score &&
    left.candidate_budget == right.candidate_budget &&
    left.patch_budget == right.patch_budget &&
    left.reject_reason == right.reject_reason &&
    left.transition == right.transition;
}

bool sameAdaptiveHysteresis(
  const livo::AdaptiveCameraHysteresisState & left,
  const livo::AdaptiveCameraHysteresisState & right)
{
  return
    left.isolated == right.isolated &&
    left.consecutive_unhealthy ==
      right.consecutive_unhealthy &&
    left.consecutive_healthy ==
      right.consecutive_healthy;
}

bool sameAdaptiveVisualPolicyPlan(
  const livo::AdaptiveVisualPolicyPlan & left,
  const livo::AdaptiveVisualPolicyPlan & right)
{
  if (
    left.adaptive_enabled != right.adaptive_enabled ||
    left.lidar_information_valid !=
      right.lidar_information_valid ||
    left.reason != right.reason ||
    left.base_revision != right.base_revision ||
    left.proposed_primary_camera !=
      right.proposed_primary_camera ||
    left.proposed_selection_challenger.camera_id !=
      right.proposed_selection_challenger.camera_id ||
    left.proposed_selection_challenger.consecutive_confirmations !=
      right.proposed_selection_challenger.consecutive_confirmations ||
    left.proposed_selection_challenger.soft_gap_observed !=
      right.proposed_selection_challenger.soft_gap_observed ||
    left.weak_direction_count !=
      right.weak_direction_count ||
    !(left.lidar_eigenvalues.array() ==
      right.lidar_eigenvalues.array()).all() ||
    !(left.weak_direction_weight.array() ==
      right.weak_direction_weight.array()).all())
  {
    return false;
  }
  for (std::size_t camera_id = 0U;
       camera_id < left.cameras.size(); ++camera_id)
  {
    if (
      !sameAdaptiveCameraDecision(
        left.cameras[camera_id],
        right.cameras[camera_id]) ||
      !sameAdaptiveHysteresis(
        left.proposed_hysteresis[camera_id],
        right.proposed_hysteresis[camera_id]))
    {
      return false;
    }
  }
  return true;
}

bool sameAdaptiveCameraObservation(
  const livo::AdaptiveCameraObservation & left,
  const livo::AdaptiveCameraObservation & right)
{
  return
    left.camera_id == right.camera_id &&
    left.frame_received == right.frame_received &&
    left.tracking_map_initialized ==
      right.tracking_map_initialized &&
    left.zero_map_reentry_ready ==
      right.zero_map_reentry_ready &&
    left.available_metrics == right.available_metrics &&
    left.candidate_points == right.candidate_points &&
    left.tracked_points == right.tracked_points &&
    left.residual_count == right.residual_count &&
    left.squared_residual_sum == right.squared_residual_sum &&
    left.saturation_fraction == right.saturation_fraction &&
    left.dark_fraction == right.dark_fraction &&
    left.mean_gradient == right.mean_gradient &&
    left.spatial_coverage == right.spatial_coverage &&
    left.inlier_count == right.inlier_count &&
    left.total_patch_count == right.total_patch_count &&
    (left.inlier_ratio == right.inlier_ratio ||
      (std::isnan(left.inlier_ratio) &&
       std::isnan(right.inlier_ratio))) &&
    (left.nis == right.nis ||
      (std::isnan(left.nis) && std::isnan(right.nis))) &&
    left.nis_dof == right.nis_dof &&
    (left.normalized_pose_correction ==
      right.normalized_pose_correction ||
      (std::isnan(left.normalized_pose_correction) &&
       std::isnan(right.normalized_pose_correction))) &&
    left.robust_information_gain ==
      right.robust_information_gain &&
    (left.visual_information.array() ==
      right.visual_information.array()).all();
}

bool sameAdaptiveCameraObservations(
  const livo::AdaptiveCameraObservations & left,
  const livo::AdaptiveCameraObservations & right)
{
  for (std::size_t camera_id = 0U;
       camera_id < left.size(); ++camera_id)
  {
    if (!sameAdaptiveCameraObservation(
        left[camera_id], right[camera_id]))
    {
      return false;
    }
  }
  return true;
}

bool sameReferenceOnlyMigrationFrameBinding(
  const livo::ReferenceOnlyMigrationFrameProvenance & left,
  const livo::ReferenceOnlyMigrationFrameProvenance & right)
{
  return
    left.valid() &&
    right.valid() &&
    left.cameraBatchSequence() == right.cameraBatchSequence() &&
    left.runtimePublicationSequence() ==
      right.runtimePublicationSequence() &&
    left.activeStateSnapshotRevision() ==
      right.activeStateSnapshotRevision() &&
    left.cameraId() == right.cameraId() &&
    left.observationGeneration() ==
      right.observationGeneration() &&
    left.imageGeneration() == right.imageGeneration() &&
    left.exposureTimestampSec() == right.exposureTimestampSec();
}

bool sameReferenceOnlyMigrationFrameBindings(
  const livo::ReferenceOnlyMigrationRuntimeBundle & left,
  const livo::ReferenceOnlyMigrationRuntimeBundle & right)
{
  const livo::ValidatedBatchRuntimeComposition &
    left_composition = left.validatedComposition();
  const livo::ValidatedBatchRuntimeComposition &
    right_composition = right.validatedComposition();

  for (std::uint8_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount;
       ++camera_id)
  {
    const auto left_frame = left.frameProvenance(camera_id);
    const auto right_frame = right.frameProvenance(camera_id);
    if (
      !left_frame.has_value() ||
      !right_frame.has_value() ||
      !left_frame->accepts(left_composition) ||
      !right_frame->accepts(right_composition) ||
      !sameReferenceOnlyMigrationFrameBinding(
        *left_frame, *right_frame))
    {
      return false;
    }
  }
  return true;
}

const char * adaptiveCameraRejectReasonName(
  const livo::AdaptiveCameraRejectReason reason) noexcept
{
  switch (reason)
  {
    case livo::AdaptiveCameraRejectReason::kNone:
      return "none";
    case livo::AdaptiveCameraRejectReason::kFrameMissing:
      return "frame_missing";
    case livo::AdaptiveCameraRejectReason::kImageHardFailure:
      return "image_hard_failure";
    case livo::AdaptiveCameraRejectReason::kTrackingUnavailable:
      return "tracking_unavailable";
    case livo::AdaptiveCameraRejectReason::kIsolated:
      return "isolated";
    case livo::AdaptiveCameraRejectReason::kHealthBelowHardReject:
      return "health_below_hard_reject";
    case livo::AdaptiveCameraRejectReason::kRobustGateFailed:
      return "robust_gate_failed";
  }
  return "unknown";
}

const char * adaptiveVisualMapActionName(
  const livo::AdaptiveVisualMapAction action) noexcept
{
  switch (action)
  {
    case livo::AdaptiveVisualMapAction::kNone:
      return "none";
    case livo::AdaptiveVisualMapAction::kColdStartSeed:
      return "cold_start_seed";
    case livo::AdaptiveVisualMapAction::kAcceptedVisualUpdate:
      return "accepted_visual_update";
    case livo::AdaptiveVisualMapAction::kLidarAnchoredReseed:
      return "lio_anchored_reseed";
  }
  return "unknown";
}

void emitAdaptiveCameraHealthMetric(
  const double timestamp,
  const double source_stamp_sec,
  const double batch_source_stamp_sec,
  const std::uint64_t lidar_feature_generation,
  const std::size_t camera_id,
  const std::string & camera_name,
  const livo::AdaptiveCameraObservation & observation,
  const livo::AdaptiveCameraDecision & decision,
  const bool frame_received,
  const bool eligible,
  const bool selected,
  const livo::AdaptiveCameraRejectReason reject_reason,
  const livo::AdaptiveVisualPolicyConfig & config)
{
  const double camera_rmse =
    observation.residual_count == 0U ?
    std::numeric_limits<double>::quiet_NaN() :
    std::sqrt(
    observation.squared_residual_sum /
    static_cast<double>(observation.residual_count));
  const bool sealed_adaptive_runtime =
    config.enabled &&
    config.minimum_selected_cameras == 1U &&
    config.maximum_selected_cameras == 1U;
  const std::size_t scout_candidate_cap =
    sealed_adaptive_runtime ? config.total_candidate_budget :
    livo::adaptiveScoutCandidateBudget(
      config.total_candidate_budget, camera_id);
  livo::emitStructuredMetric(
    "[ADAPTIVE_VIO_CAMERA_METRIC] t=%.9f source_stamp_sec=%.9f "
    "batch_source_stamp_sec=%.9f generation=%llu "
    "camera_id=%zu camera=%s metrics_mask=%u "
    "frame_received=%d available=%d eligible=%d selected=%d "
    "reject_reason=%s enabled=%d health_state=%s "
    "isolated=%d degraded=%d "
    "metrics_complete=%d health_evidence_complete=%d "
    "image_hard_failure=%d image_recovery_healthy=%d "
    "nis_available=%d blur_available=0 health=%.6f "
    "cov_scale=%.6f update_candidate_budget=%zu "
    "update_patch_budget=%zu candidate_budget=%zu patch_budget=%zu "
    "scout_candidate_cap=%zu budget_scope=update "
    "complement=%.9g selection=%.9g candidates=%zu tracked=%zu "
    "rmse=%.6f sat=%.6f dark=%.6f gradient=%.6f "
    "coverage=%.6f inliers=%zu patch_count=%zu inlier_ratio=%.6f "
    "nis=%.9g nis_dof=%zu normalized_pose_correction=%.9g "
    "robust_information_gain=%.9g robust_gate_passed=%d\n",
    timestamp, source_stamp_sec, batch_source_stamp_sec,
    static_cast<unsigned long long>(lidar_feature_generation),
    camera_id, camera_name.c_str(),
    observation.available_metrics,
    frame_received ? 1 : 0,
    decision.available ? 1 : 0,
    eligible ? 1 : 0,
    selected ? 1 : 0,
    adaptiveCameraRejectReasonName(reject_reason),
    decision.enabled ? 1 : 0,
    livo::adaptiveCameraHealthStateName(decision.health_state),
    decision.isolated ? 1 : 0,
    decision.health_degraded ? 1 : 0,
    decision.metrics_complete ? 1 : 0,
    decision.health_evidence_complete ? 1 : 0,
    decision.image_hard_failure ? 1 : 0,
    decision.image_recovery_healthy ? 1 : 0,
    observation.nis == observation.nis ? 1 : 0,
    decision.health, decision.covariance_scale,
    decision.candidate_budget, decision.patch_budget,
    decision.candidate_budget, decision.patch_budget,
    scout_candidate_cap,
    decision.complement_score, decision.selection_score,
    observation.candidate_points, observation.tracked_points,
    camera_rmse, observation.saturation_fraction,
    observation.dark_fraction, observation.mean_gradient,
    observation.spatial_coverage, observation.inlier_count,
    observation.total_patch_count, observation.inlier_ratio,
    observation.nis, observation.nis_dof,
    observation.normalized_pose_correction,
    decision.robust_information_gain,
    decision.robust_gate_passed ? 1 : 0);
}

}  // namespace

const char *
livo::referenceOnlyMigrationShadowRuntimeInitReasonName(
  const ReferenceOnlyMigrationShadowRuntimeInitReason reason)
  noexcept
{
  switch (reason)
  {
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kNotInitialized:
      return "not_initialized";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kDisabled:
      return "disabled";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRequiresFourCameraMode:
      return "requires_four_camera_mode";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingCrossCameraShadowOwner:
      return "missing_cross_camera_shadow_owner";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kCrossCameraShadowOwnerNotReady:
      return "cross_camera_shadow_owner_not_ready";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingSharedGeometryShadow:
      return "missing_shared_geometry_shadow";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingTimeTransferConfiguration:
      return "missing_time_transfer_configuration";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingBatchAuthority:
      return "missing_batch_authority";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingRuntimePublisherAuthority:
      return "missing_runtime_publisher_authority";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRuntimeAuthorityOwnerMismatch:
      return "runtime_authority_owner_mismatch";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRuntimeObservationAuthorityBindingRejected:
      return "runtime_observation_authority_binding_rejected";
    case ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMetricsOnlyReady:
      return "metrics_only_ready";
  }
  return "unknown";
}

const char *
livo::referenceOnlyMigrationShadowRuntimeBatchReasonName(
  const ReferenceOnlyMigrationShadowRuntimeBatchReason reason)
  noexcept
{
  switch (reason)
  {
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kNotRun:
      return "not_run";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kMissingTimeTransferBundle:
      return "missing_time_transfer_bundle";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kMissingRuntimeProvenance:
      return "missing_runtime_provenance";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kRuntimeProvenanceMismatch:
      return "runtime_provenance_mismatch";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidFullStateTimeTransferBundle:
      return "invalid_full_state_time_transfer_bundle";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kTimeTransferBatchBindingMismatch:
      return "time_transfer_batch_binding_mismatch";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidActiveStateSnapshot:
      return "invalid_active_state_snapshot";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidSourceCovariance:
      return "invalid_source_covariance";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kDependencyStale:
      return "dependency_stale";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kImageGeometryMismatch:
      return "image_geometry_mismatch";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kSequenceExhausted:
      return "sequence_exhausted";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kIncompleteFourCameraContexts:
      return "incomplete_four_camera_contexts";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kNoCandidate:
      return "no_candidate";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kNoEligibleDirectedPair:
      return "no_eligible_directed_pair";
    case ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kEvaluated:
      return "evaluated";
  }
  return "unknown";
}

const char *
livo::vioRuntimeObservationAuthorityStatusName(
  const VIORuntimeObservationAuthorityStatus status) noexcept
{
  switch (status)
  {
    case VIORuntimeObservationAuthorityStatus::kNotEvaluated:
      return "not_evaluated";
    case VIORuntimeObservationAuthorityStatus::
      kMissingRuntimeBundle:
      return "missing_runtime_bundle";
    case VIORuntimeObservationAuthorityStatus::
      kRuntimePublisherAuthorityMismatch:
      return "runtime_publisher_authority_mismatch";
    case VIORuntimeObservationAuthorityStatus::
      kCameraBatchAuthorityMismatch:
      return "camera_batch_authority_mismatch";
    case VIORuntimeObservationAuthorityStatus::
      kActiveStateSnapshotMismatch:
      return "active_state_snapshot_mismatch";
    case VIORuntimeObservationAuthorityStatus::
      kInvalidTimeTransferBundle:
      return "invalid_time_transfer_bundle";
    case VIORuntimeObservationAuthorityStatus::
      kReplayedRuntimeBundle:
      return "replayed_runtime_bundle";
    case VIORuntimeObservationAuthorityStatus::
      kInvalidFrameProvenance:
      return "invalid_frame_provenance";
    case VIORuntimeObservationAuthorityStatus::kReady:
      return "ready";
  }
  return "unknown";
}

const char *
livo::vioMultiCameraLandmarkShadowInvocationName(
  const VIOMultiCameraLandmarkShadowInvocation invocation) noexcept
{
  switch (invocation)
  {
    case VIOMultiCameraLandmarkShadowInvocation::kDisabled:
      return "disabled";
    case VIOMultiCameraLandmarkShadowInvocation::kNotEvaluated:
      return "not_evaluated";
    case VIOMultiCameraLandmarkShadowInvocation::kModuleUnavailable:
      return "module_unavailable";
    case VIOMultiCameraLandmarkShadowInvocation::kMissingRuntimeBundle:
      return "missing_runtime_bundle";
    case VIOMultiCameraLandmarkShadowInvocation::
      kRuntimeAuthorityNotReady:
      return "runtime_authority_not_ready";
    case VIOMultiCameraLandmarkShadowInvocation::
      kMissingPlanCapability:
      return "missing_plan_capability";
    case VIOMultiCameraLandmarkShadowInvocation::kProcessed:
      return "processed";
  }
  return "unknown";
}

VIOManager::VIOManager()
{
  active_visual_feature_map_ = &camera_feature_maps_[0U];
  has_ref_patch_cache = false;
  // downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
}

void VIOManager::initializeCrossCameraMigrationShadow(
  const livo::CrossCameraMigrationShadowInitializationConfig &
    config)
{
  cross_camera_migration_shadow_rig_.reset();
  cross_camera_migration_shadow_init_audit_ = {};
  if (!livo::checkedIncrement(
      cross_camera_migration_owner_generation_))
  {
    cross_camera_migration_shadow_init_audit_.enabled =
      config.enabled;
    cross_camera_migration_shadow_init_audit_.reason =
      "owner_generation_exhausted";
    return;
  }
  cross_camera_migration_shadow_init_audit_.owner_generation =
    cross_camera_migration_owner_generation_;
  cross_camera_migration_shadow_init_audit_.enabled =
    config.enabled;
  auto emit_init_metric = [this]()
    {
      const auto & audit =
        cross_camera_migration_shadow_init_audit_;
      const std::string fingerprint =
        audit.rig_fingerprint ?
        fingerprintHex(*audit.rig_fingerprint) : "none";
      std::array<std::uint64_t, 4U> model_revisions{};
      for (std::size_t camera_id = 0U;
           camera_id < model_revisions.size();
           ++camera_id)
      {
        if (audit.cameras[camera_id])
        {
          model_revisions[camera_id] =
            audit.cameras[camera_id]->model_revision;
        }
      }
      livo::emitStructuredMetric(
        "[VIO_CROSS_CAMERA_MIGRATION_SHADOW_INIT] "
        "enabled=%d ready=%d reason=%s rig_revision=%llu "
        "owner_generation=%llu "
        "fingerprint=%s model_revisions=%llu,%llu,%llu,%llu "
        "production_ready=0 solver_applied=0\n",
        audit.enabled ? 1 : 0,
        audit.ready ? 1 : 0,
        audit.reason.c_str(),
        static_cast<unsigned long long>(audit.rig_revision),
        static_cast<unsigned long long>(audit.owner_generation),
        fingerprint.c_str(),
        static_cast<unsigned long long>(model_revisions[0U]),
        static_cast<unsigned long long>(model_revisions[1U]),
        static_cast<unsigned long long>(model_revisions[2U]),
        static_cast<unsigned long long>(model_revisions[3U]));
    };

  if (!config.enabled)
  {
    cross_camera_migration_shadow_init_audit_.reason =
      "disabled";
    emit_init_metric();
    return;
  }
  if (!config.four_camera_mode)
  {
    cross_camera_migration_shadow_init_audit_.reason =
      "requires_four_camera_mode";
    emit_init_metric();
    return;
  }
  if (!imu_to_lidar_extrinsic_configured_)
  {
    cross_camera_migration_shadow_init_audit_.reason =
      "missing_imu_lidar_extrinsic";
    emit_init_metric();
    return;
  }

  constexpr std::array<const char *, 4U> kExpectedNames{
    {"front", "left", "back", "right"}};
  std::array<bool, 4U> seen{{false, false, false, false}};
  livo::FourCameraPinholeRigConfig rig_config;
  for (const auto & input : config.cameras)
  {
    const std::size_t camera_id =
      static_cast<std::size_t>(input.camera_id);
    if (camera_id >= seen.size() || seen[camera_id])
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "invalid_camera_id_set";
      emit_init_metric();
      return;
    }
    seen[camera_id] = true;
    if (!camera_configured_[camera_id])
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "camera_not_registered";
      emit_init_metric();
      return;
    }
    if (
      input.camera_name != kExpectedNames[camera_id] ||
      camera_names_[camera_id] != kExpectedNames[camera_id])
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "camera_name_mapping_mismatch";
      emit_init_metric();
      return;
    }
    if (!input.projection_calibration_present)
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "missing_camera_projection_calibration";
      emit_init_metric();
      return;
    }
    if (
      !registered_camera_parameters_[camera_id].has_value() ||
      !sameRegisteredPinholeModel(
        *registered_camera_parameters_[camera_id],
        input.projection))
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "canonical_camera_model_mismatch";
      emit_init_metric();
      return;
    }
    if (!input.extrinsic_calibration_present)
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "missing_camera_extrinsic_calibration";
      emit_init_metric();
      return;
    }
    if (input.projection.model != "Pinhole")
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "unsupported_projection_model";
      emit_init_metric();
      return;
    }
    if (
      !std::isfinite(input.projection.scale) ||
      input.projection.scale != 1.0)
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "unsupported_camera_scale";
      emit_init_metric();
      return;
    }
    if (
      input.projection.d0 != 0.0 ||
      input.projection.d1 != 0.0 ||
      input.projection.d2 != 0.0 ||
      input.projection.d3 != 0.0 ||
      input.projection.d4 != 0.0)
    {
      cross_camera_migration_shadow_init_audit_.reason =
        "unsupported_pinhole_distortion";
      emit_init_metric();
      return;
    }

    auto & output = rig_config.cameras[camera_id];
    output.camera_id = input.camera_id;
    output.projection.model = input.projection.model;
    output.projection.width = input.projection.width;
    output.projection.height = input.projection.height;
    output.projection.fx = input.projection.fx;
    output.projection.fy = input.projection.fy;
    output.projection.cx = input.projection.cx;
    output.projection.cy = input.projection.cy;
    output.projection.distortion_coefficients = {{
      input.projection.d0,
      input.projection.d1,
      input.projection.d2,
      input.projection.d3,
    }};
    output.rotation_camera_from_imu =
      lidar_to_camera_rotations_[camera_id] * Rli;
    output.position_camera_from_imu =
      lidar_to_camera_rotations_[camera_id] * Pli +
      lidar_to_camera_translations_[camera_id];
  }

  try
  {
    auto candidate =
      std::make_unique<livo::FourCameraPinholeRig>(
      rig_config);
    std::array<
      livo::CrossCameraMigrationShadowCameraAudit,
      4U> camera_audits;
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      const auto model = candidate->cameraModel(camera_id);
      if (!model)
      {
        throw std::runtime_error("missing shadow camera model");
      }
      const auto endpoint = candidate->shadowEndpoint(
        camera_id,
        livo::FourCameraPinholeRigRevisionExpectation{
          model->modelRevision(),
          candidate->rigRevision()});
      if (!endpoint.shadowReady() || !endpoint.endpoint)
      {
        throw std::runtime_error("shadow endpoint unavailable");
      }
      camera_audits[camera_id].camera_id = camera_id;
      camera_audits[camera_id].camera_name =
        kExpectedNames[camera_id];
      camera_audits[camera_id].model_revision =
        model->modelRevision();
      camera_audits[camera_id].rotation_imu_from_camera =
        endpoint.endpoint->rotation_imu_from_camera;
      camera_audits[camera_id].position_imu_from_camera =
        endpoint.endpoint->position_imu_from_camera;
    }

    cross_camera_migration_shadow_init_audit_.ready = true;
    cross_camera_migration_shadow_init_audit_.reason =
      "shadow_owner_ready";
    cross_camera_migration_shadow_init_audit_.rig_revision =
      candidate->rigRevision();
    cross_camera_migration_shadow_init_audit_.rig_fingerprint =
      candidate->configurationFingerprint();
    for (std::size_t camera_id = 0U;
         camera_id < camera_audits.size();
         ++camera_id)
    {
      cross_camera_migration_shadow_init_audit_.
        cameras[camera_id] = camera_audits[camera_id];
    }
    cross_camera_migration_shadow_rig_ = std::move(candidate);
  }
  catch (const std::exception &)
  {
    cross_camera_migration_shadow_rig_.reset();
    cross_camera_migration_shadow_init_audit_ = {};
    cross_camera_migration_shadow_init_audit_.enabled = true;
    cross_camera_migration_shadow_init_audit_.owner_generation =
      cross_camera_migration_owner_generation_;
    cross_camera_migration_shadow_init_audit_.reason =
      "invalid_rig_configuration";
  }
  emit_init_metric();
}

const livo::CrossCameraMigrationShadowInitAudit &
VIOManager::crossCameraMigrationShadowInitAudit() const noexcept
{
  return cross_camera_migration_shadow_init_audit_;
}

void VIOManager::configureReferenceOnlyMigrationShadowRuntime(
  const livo::ReferenceOnlyMigrationShadowRuntimeConfig &
    config)
{
  reference_only_migration_shadow_runtime_init_audit_ = {};
  reference_only_migration_shadow_runtime_batch_audit_ = {};
  reference_only_migration_shadow_runtime_init_audit_.enabled =
    config.enabled;
  const auto emit_init_metric = [this]()
    {
      const auto & audit =
        reference_only_migration_shadow_runtime_init_audit_;
      livo::emitStructuredMetric(
        "[VIO_REFERENCE_ONLY_MIGRATION_SHADOW_INIT] "
        "enabled=%d effective=%d reason=%s "
        "rig_revision=%llu rig_owner_generation=%llu "
        "geometry_configuration_generation=%llu "
        "time_transfer_revision=%llu authority_generation=%llu "
        "visibility_verified=0 "
        "experimental=1 consumable=0 production_ready=0 "
        "solver_applied=0 acceptance_receipt_issued=0 "
        "independence_proof_issued=0\n",
        audit.enabled ? 1 : 0,
        audit.effective ? 1 : 0,
        livo::referenceOnlyMigrationShadowRuntimeInitReasonName(
          audit.reason),
        static_cast<unsigned long long>(
          cross_camera_migration_shadow_init_audit_.
          rig_revision),
        static_cast<unsigned long long>(
          audit.bound_cross_camera_owner_generation),
        static_cast<unsigned long long>(
          audit.bound_geometry_configuration_generation),
        static_cast<unsigned long long>(
          audit.bound_time_transfer_revision),
        static_cast<unsigned long long>(
          audit.bound_authority_generation));
    };
  if (!config.enabled)
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kDisabled;
    emit_init_metric();
    return;
  }
  if (!config.four_camera_mode)
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRequiresFourCameraMode;
    emit_init_metric();
    return;
  }
  if (!cross_camera_migration_shadow_init_audit_.enabled)
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingCrossCameraShadowOwner;
    emit_init_metric();
    return;
  }
  if (!cross_camera_migration_shadow_init_audit_.ready)
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kCrossCameraShadowOwnerNotReady;
    emit_init_metric();
    return;
  }
  if (!shared_landmark_shadow_session_.audit().geometry_enabled)
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingSharedGeometryShadow;
    emit_init_metric();
    return;
  }
  if (
    shared_landmark_shadow_session_.audit().
    time_transfer_configuration_revision == 0U)
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingTimeTransferConfiguration;
    emit_init_metric();
    return;
  }
  if (
    !reference_only_migration_batch_authority_.valid())
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingBatchAuthority;
    emit_init_metric();
    return;
  }
  if (
    !reference_only_migration_publisher_authority_.valid())
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingRuntimePublisherAuthority;
    emit_init_metric();
    return;
  }
  if (
    !reference_only_migration_authorities_configured_ ||
    !reference_only_migration_authority_owner_match_)
  {
    reference_only_migration_shadow_runtime_init_audit_.reason =
      reference_only_migration_authority_owner_match_ ?
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRuntimeObservationAuthorityBindingRejected :
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRuntimeAuthorityOwnerMismatch;
    emit_init_metric();
    return;
  }
  const livo::SharedLandmarkShadowAudit dependency_audit =
    shared_landmark_shadow_session_.audit();
  reference_only_migration_shadow_runtime_init_audit_.
    bound_cross_camera_owner_generation =
    cross_camera_migration_shadow_init_audit_.owner_generation;
  reference_only_migration_shadow_runtime_init_audit_.
    bound_rig_revision =
    cross_camera_migration_shadow_init_audit_.rig_revision;
  reference_only_migration_shadow_runtime_init_audit_.
    bound_rig_fingerprint =
    cross_camera_migration_shadow_init_audit_.rig_fingerprint;
  reference_only_migration_shadow_runtime_init_audit_.
    bound_geometry_configuration_generation =
    dependency_audit.configuration_generation;
  reference_only_migration_shadow_runtime_init_audit_.
    bound_time_transfer_revision =
    dependency_audit.time_transfer_configuration_revision;
  reference_only_migration_shadow_runtime_init_audit_.
    bound_authority_generation =
    reference_only_migration_authority_generation_;
  reference_only_migration_shadow_runtime_init_audit_.effective =
    true;
  reference_only_migration_shadow_runtime_init_audit_.reason =
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
    kMetricsOnlyReady;
  emit_init_metric();
}

void VIOManager::configureReferenceOnlyMigrationRuntimeAuthorities(
  const livo::CameraObservationBatchAuthorityIdentity &
    batch_authority,
  const livo::ReferenceOnlyMigrationRuntimePublisherIdentity &
    publisher_authority)
{
  if (!livo::checkedIncrement(
      reference_only_migration_authority_generation_))
  {
    reference_only_migration_authorities_configured_ = false;
    reference_only_migration_authority_owner_match_ = false;
    reference_only_migration_batch_authority_ = {};
    reference_only_migration_publisher_authority_ = {};
    return;
  }
  reference_only_migration_batch_authority_ =
    batch_authority;
  reference_only_migration_publisher_authority_ =
    publisher_authority;
  reference_only_migration_authority_owner_match_ =
    publisher_authority.bindsBatchAuthority(batch_authority);
  reference_only_migration_authorities_configured_ =
    batch_authority.valid() &&
    publisher_authority.valid() &&
    reference_only_migration_authority_owner_match_ &&
    shared_landmark_shadow_session_.
    bindRuntimeObservationAuthority(
      batch_authority, publisher_authority);
}

const livo::ReferenceOnlyMigrationShadowRuntimeInitAudit &
VIOManager::referenceOnlyMigrationShadowRuntimeInitAudit()
  const noexcept
{
  return reference_only_migration_shadow_runtime_init_audit_;
}

const livo::ReferenceOnlyMigrationShadowRuntimeBatchAudit &
VIOManager::referenceOnlyMigrationShadowRuntimeBatchAudit()
  const noexcept
{
  return reference_only_migration_shadow_runtime_batch_audit_;
}

const livo::
  VIOMultiCameraLandmarkLinearizationShadowBatchAudit &
VIOManager::multiCameraLandmarkLinearizationShadowBatchAudit()
  const noexcept
{
  return
    multi_camera_landmark_linearization_shadow_batch_audit_;
}

const livo::AdaptiveVisualRuntimeAudit &
VIOManager::adaptiveVisualRuntimeAudit() const noexcept
{
  return adaptive_visual_runtime_audit_;
}

livo::VIOReadOnlyStructuralAudit
VIOManager::readOnlyStructuralAudit() const
{
  livo::VIOReadOnlyStructuralAudit audit;
  if (state != nullptr)
  {
    audit.active_state_available = true;
    audit.active_state = stateScalars(*state);
  }
  if (state_propagat != nullptr)
  {
    audit.propagated_state_available = true;
    audit.propagated_state =
      stateScalars(*state_propagat);
  }
  audit.shared_landmark_shadow =
    shared_landmark_shadow_session_.audit();
  for (std::size_t camera_id = 0U;
       camera_id < camera_feature_maps_.size();
       ++camera_id)
  {
    const VisualFeatureMap & feature_map =
      camera_feature_maps_[camera_id];
    audit.legacy_feature_voxel_counts[camera_id] =
      feature_map.size();
    for (const auto & entry : feature_map)
    {
      if (entry.second != nullptr)
      {
        audit.legacy_feature_point_counts[camera_id] +=
          entry.second->voxel_points.size();
      }
    }
  }
  audit.last_joint_solver = last_joint_solver_audit_;
  audit.active_camera_id = active_camera_id_;
  audit.adaptive_visual_map_lifecycle =
    adaptive_visual_map_lifecycle_;
  for (std::size_t camera_id = 0U;
       camera_id < adaptive_visual_map_lifecycle_.size();
       ++camera_id)
  {
    audit.adaptive_camera_initialized[camera_id] =
      adaptive_visual_map_lifecycle_[camera_id] ==
      livo::AdaptiveVisualMapLifecycle::kInitialized;
  }
  audit.adaptive_zero_map_reentry_healthy_scouts =
    adaptive_zero_map_reentry_healthy_scouts_;
  audit.adaptive_zero_map_reseed_latched =
    adaptive_zero_map_reseed_latched_;
  audit.adaptive_zero_map_last_trusted_generation =
    adaptive_zero_map_last_trusted_generation_;
  audit.frame_count =
    frame_count < 0 ? 0U :
    static_cast<std::size_t>(frame_count);
  audit.adaptive_policy_revision =
    adaptive_visual_policy_ ?
    adaptive_visual_policy_->revision() : 0U;
  audit.adaptive_prepare_call_count =
    adaptive_prepare_call_count_;
  audit.adaptive_scout_call_count =
    adaptive_scout_call_count_;
  audit.adaptive_finalize_call_count =
    adaptive_finalize_call_count_;
  audit.shared_landmark_shadow_entry_count =
    shared_landmark_shadow_entry_count_;
  audit.reference_only_migration_shadow_entry_count =
    reference_only_migration_shadow_entry_count_;
  audit.multi_camera_linearization_shadow_entry_count =
    multi_camera_linearization_shadow_entry_count_;
  return audit;
}

void VIOManager::configureSharedLandmarkShadow(
  const livo::LegacyLandmarkShadowConfig & config)
{
  shared_landmark_shadow_session_.configureGeometry(config);
  const livo::SharedLandmarkShadowAudit audit =
    shared_landmark_shadow_session_.audit();
  if (
    config.enabled &&
    audit.visual_revision_snapshot.
      rig_calibration_revision == 0U)
  {
    for (std::uint8_t camera_id = 0U;
         camera_id < kCameraCount; ++camera_id)
    {
      if (camera_configured_[camera_id])
      {
        shared_landmark_shadow_session_.
          recordCameraConfiguration(camera_id);
      }
    }
  }
}

void VIOManager::configureSharedLandmarkObservationShadow(
  const livo::LegacyLandmarkObservationShadowConfig & config)
{
  shared_landmark_shadow_session_.configureObservation(config);
}

void VIOManager::configureSharedLandmarkTombstoneShadow(
  const livo::SharedLandmarkTombstoneShadowConfig & config)
{
  shared_landmark_shadow_session_.configureTombstone(config);
}

void VIOManager::configureSharedLandmarkRefitShadow(
  const livo::SharedLandmarkGeometryRefitShadowConfig & config)
{
  shared_landmark_shadow_session_.configureRefit(config);
}

void VIOManager::configureSharedLandmarkTimeTransfer(
  const livo::CameraTimeTransferConfig & config)
{
  shared_landmark_shadow_session_.configureTimeTransfer(config);
}

void VIOManager::configureSharedLandmarkPhotometric(
  const std::uint8_t camera_id,
  const livo::ShadowPhotometricCapabilityEvidence & evidence)
{
  shared_landmark_shadow_session_.configurePhotometric(
    camera_id, evidence);
}

bool VIOManager::configureMultiCameraLandmarkLinearizationShadow(
  const livo::MultiCameraLandmarkLinearizationShadowConfig &
    config)
{
  if (
    adaptive_visual_policy_ != nullptr ||
    multi_camera_landmark_linearization_shadow_ != nullptr ||
    !config.valid())
  {
    return false;
  }
  multi_camera_landmark_linearization_shadow_config_ = config;
  multi_camera_landmark_linearization_shadow_batch_audit_ = {};
  multi_camera_landmark_linearization_shadow_batch_audit_.
    configured_enabled = config.enabled;
  multi_camera_landmark_linearization_shadow_batch_audit_.
    invocation =
    config.enabled ?
    livo::VIOMultiCameraLandmarkShadowInvocation::
      kNotEvaluated :
    livo::VIOMultiCameraLandmarkShadowInvocation::kDisabled;
  return true;
}

livo::SharedLandmarkShadowOperationResult
VIOManager::applySharedLandmarkTombstonesShadow(
  const std::vector<livo::PlaneTombstoneEvent> & events)
{
  const livo::SharedLandmarkShadowOperationResult result =
    shared_landmark_shadow_session_.applyTombstones(events);
  const livo::SharedLandmarkShadowAudit audit =
    shared_landmark_shadow_session_.audit();
  livo::emitStructuredMetric(
    "[VIO_LANDMARK_TOMBSTONE_SHADOW] geometry=%llu "
    "status=%s input_events=%zu matched_events=%zu "
    "ignored_events=%zu tombstoned=%zu acknowledge=%d "
    "consumable=0 solver_applied=0\n",
    static_cast<unsigned long long>(
      audit.geometry.value),
    livo::sharedLandmarkShadowOperationStatusName(result.status),
    result.input_count,
    result.matched_count,
    result.ignored_count,
    result.tombstoned_landmarks,
    result.acknowledge_events ? 1 : 0);
  return result;
}

livo::SharedLandmarkShadowOperationResult
VIOManager::applySharedLandmarkRefitsShadow(
  const std::vector<livo::PlaneRefitEvent> & events)
{
  const livo::SharedLandmarkShadowOperationResult result =
    shared_landmark_shadow_session_.applyRefitEvents(events);
  const livo::SharedLandmarkShadowAudit audit =
    shared_landmark_shadow_session_.audit();
  livo::emitStructuredMetric(
    "[VIO_LANDMARK_REFIT_SHADOW] geometry=%llu "
    "lio_revision=%llu status=%s input_events=%zu "
    "matched=%zu ignored=%zu geometry_refits=%zu "
    "acknowledge=%d consumable=0 solver_applied=0\n",
    static_cast<unsigned long long>(
      audit.geometry.value),
    static_cast<unsigned long long>(
      audit.lio_posterior_revision.value),
    livo::sharedLandmarkShadowOperationStatusName(result.status),
    result.input_count,
    result.matched_count,
    result.ignored_count,
    result.geometry_refits,
    result.acknowledge_events ? 1 : 0);
  return result;
}

VIOManager::~VIOManager()
{
  new_frame_.reset();
  delete visual_submap;
  for (auto& pair : warp_map) delete pair.second;
  warp_map.clear();
  for (auto &camera_map : camera_feature_maps_)
  {
    for (auto &pair : camera_map) delete pair.second;
    camera_map.clear();
  }
}

void VIOManager::emitVisualMapResources() const
{
  for (std::size_t camera_id = 0; camera_id < kCameraCount; ++camera_id)
  {
    std::size_t points = 0, observations = 0, image_bytes = 0;
    std::unordered_set<const unsigned char *> images;
    for (const auto &voxel : camera_feature_maps_[camera_id])
    {
      points += voxel.second->voxel_points.size();
      for (const auto *point : voxel.second->voxel_points)
      {
        observations += point->obs_.size();
        for (const auto *feature : point->obs_)
        {
          if (images.insert(feature->img_.data).second)
            image_bytes += feature->img_.total() * feature->img_.elemSize();
        }
      }
    }
    livo::emitStructuredMetric(
      "[VISUAL_MAP_RESOURCES] camera=%zu voxels=%zu points=%zu observations=%zu images=%zu image_bytes=%zu\n",
      camera_id, camera_feature_maps_[camera_id].size(), points,
      observations, images.size(), image_bytes);
  }
}

void VIOManager::setImuToLidarExtrinsic(const V3D &transl, const M3D &rot)
{
  Pli = -rot.transpose() * transl;
  Rli = rot.transpose();
  imu_to_lidar_extrinsic_configured_ = true;
}

std::optional<livo::MigrationEndpoint>
VIOManager::crossCameraMigrationShadowEndpoint(
  const std::uint8_t camera_id) const
{
  if (!cross_camera_migration_shadow_rig_)
  {
    return std::nullopt;
  }
  const auto model =
    cross_camera_migration_shadow_rig_->cameraModel(camera_id);
  if (!model)
  {
    return std::nullopt;
  }
  const auto result =
    cross_camera_migration_shadow_rig_->shadowEndpoint(
    camera_id,
    livo::FourCameraPinholeRigRevisionExpectation{
      model->modelRevision(),
      cross_camera_migration_shadow_rig_->rigRevision()});
  return result.shadowReady() ? result.endpoint : std::nullopt;
}

void VIOManager::setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P)
{
  Rcl << MAT_FROM_ARRAY(R);
  Pcl << VEC_FROM_ARRAY(P);
}

void VIOManager::registerCamera(
  const std::uint8_t camera_id, const std::string &name,
  vk::AbstractCamera *camera, const M3D &lidar_to_camera_rotation,
  const V3D &lidar_to_camera_translation)
{
  if (camera_id >= kCameraCount || camera == nullptr)
  {
    livo::destroyCameraModel(camera);
    throw std::invalid_argument("VIO camera registration received an invalid id or model");
  }

  if (camera_id != 0U && camera_configured_[0] &&
      (camera->width() != cameras_[0]->width() || camera->height() != cameras_[0]->height()))
  {
    livo::destroyCameraModel(camera);
    throw std::invalid_argument("All VIO cameras must use the same processed image dimensions");
  }

  std::optional<livo::CameraModelParameters>
    canonical_parameters;
  if (const auto * pinhole =
      dynamic_cast<const vk::PinholeCamera *>(camera))
  {
    livo::CameraModelParameters parameters;
    parameters.model = "Pinhole";
    parameters.width = pinhole->width();
    parameters.height = pinhole->height();
    parameters.scale = pinhole->scale();
    parameters.fx = pinhole->fx();
    parameters.fy = pinhole->fy();
    parameters.cx = pinhole->cx();
    parameters.cy = pinhole->cy();
    parameters.d0 = pinhole->d0();
    parameters.d1 = pinhole->d1();
    parameters.d2 = pinhole->d2();
    parameters.d3 = pinhole->d3();
    parameters.d4 = pinhole->d4();
    canonical_parameters = parameters;
  }

  cameras_[camera_id].reset(camera);
  registered_camera_parameters_[camera_id] =
    std::move(canonical_parameters);
  lidar_to_camera_rotations_[camera_id] = lidar_to_camera_rotation;
  lidar_to_camera_translations_[camera_id] = lidar_to_camera_translation;
  camera_names_[camera_id] = name;
  camera_configured_[camera_id] = true;
  shared_landmark_shadow_session_.recordCameraConfiguration(
    camera_id);

  if (camera_id == active_camera_id_ || cam == nullptr)
  {
    activateCamera(camera_id);
  }
}

bool VIOManager::activateCamera(const std::uint8_t camera_id)
{
  if (camera_id >= kCameraCount || !camera_configured_[camera_id])
  {
    return false;
  }

  active_camera_id_ = camera_id;
  cam = cameras_[camera_id].get();
  active_visual_feature_map_ = &camera_feature_maps_[camera_id];
  Rcl = lidar_to_camera_rotations_[camera_id];
  Pcl = lidar_to_camera_translations_[camera_id];
  updateActiveCameraGeometry();
  return true;
}

vk::AbstractCamera *VIOManager::cameraModel(const std::uint8_t camera_id) const
{
  if (camera_id >= kCameraCount || !camera_configured_[camera_id])
  {
    return nullptr;
  }
  return cameras_[camera_id].get();
}

bool VIOManager::cameraFromImuExtrinsic(
  const std::uint8_t camera_id,
  M3D & rotation_camera_from_imu,
  V3D & translation_camera_from_imu) const noexcept
{
  if (
    camera_id >= kCameraCount ||
    !camera_configured_[camera_id] ||
    !imu_to_lidar_extrinsic_configured_)
  {
    return false;
  }
  rotation_camera_from_imu =
    lidar_to_camera_rotations_[camera_id] * Rli;
  translation_camera_from_imu =
    lidar_to_camera_rotations_[camera_id] * Pli +
    lidar_to_camera_translations_[camera_id];
  return true;
}

const std::string &VIOManager::cameraName(const std::uint8_t camera_id) const
{
  static const std::string unknown_camera("unknown");
  if (camera_id >= kCameraCount || !camera_configured_[camera_id])
  {
    return unknown_camera;
  }
  return camera_names_[camera_id];
}

void VIOManager::updateActiveCameraGeometry()
{
  if (cam == nullptr)
  {
    return;
  }

  fx = cam->fx();
  fy = cam->fy();
  cx = cam->cx();
  cy = cam->cy();
  image_resize_factor = cam->scale();
  width = cam->width();
  height = cam->height();

  Rci = Rcl * Rli;
  Pci = Rcl * Pli + Pcl;

  V3D Pic = -Rci.transpose() * Pci;
  M3D tmp;
  Jdphi_dR = Rci;
  tmp << SKEW_SYM_MATRX(Pic);
  Jdp_dR = -Rci * tmp;
}

void VIOManager::initializeVIO()
{
  if (!activateCamera(0U))
  {
    throw std::runtime_error("Front VIO camera is not configured");
  }

  visual_submap = new SubSparseMap;
  const bool sealed_adaptive_single_camera =
    adaptive_policy_config.enabled &&
    adaptive_policy_config.minimum_selected_cameras == 1U &&
    adaptive_policy_config.maximum_selected_cameras == 1U;
  if (!sealed_adaptive_single_camera)
  {
    adaptive_policy_config.minimum_selection_tracked_points =
      std::max(
      adaptive_policy_config.minimum_selection_tracked_points,
      joint_update_config.min_tracked_points_per_camera);
  }
  if (sealed_adaptive_single_camera)
  {
    joint_update_config.min_joint_cameras = 1U;
    joint_update_config.min_joint_tracked_points = std::max(
      joint_update_config.min_joint_tracked_points,
      adaptive_policy_config.minimum_inliers);
  }
  multi_camera_landmark_linearization_shadow_.reset();
  multi_camera_landmark_linearization_shadow_batch_audit_ = {};
  multi_camera_landmark_linearization_shadow_batch_audit_.
    configured_enabled =
    multi_camera_landmark_linearization_shadow_config_.enabled;
  multi_camera_landmark_linearization_shadow_batch_audit_.
    invocation =
    multi_camera_landmark_linearization_shadow_config_.enabled ?
    livo::VIOMultiCameraLandmarkShadowInvocation::
      kNotEvaluated :
    livo::VIOMultiCameraLandmarkShadowInvocation::kDisabled;
  adaptive_visual_policy_.reset();
  adaptive_visual_policy_ =
    std::make_unique<livo::AdaptiveVisualPolicy>(adaptive_policy_config);
  adaptive_visual_map_lifecycle_.fill(
    livo::AdaptiveVisualMapLifecycle::kEmpty);
  adaptive_zero_map_reentry_healthy_scouts_.fill(0U);
  adaptive_zero_map_reseed_latched_.fill(false);
  adaptive_zero_map_last_trusted_generation_.fill(0U);
  adaptive_visual_runtime_audit_ = {};
  const livo::SharedLandmarkShadowAudit shadow_audit =
    shared_landmark_shadow_session_.audit();
  if (
    multi_camera_landmark_linearization_shadow_config_.enabled &&
    adaptive_policy_config.enabled &&
    reference_only_migration_authorities_configured_ &&
    reference_only_migration_authority_owner_match_ &&
    reference_only_migration_shadow_runtime_init_audit_.effective &&
    shadow_audit.geometry_enabled &&
    shadow_audit.observation_enabled)
  {
    multi_camera_landmark_linearization_shadow_ =
      std::make_unique<
      livo::MultiCameraLandmarkLinearizationShadow>(
      shared_landmark_shadow_session_,
      reference_only_migration_batch_authority_,
      reference_only_migration_publisher_authority_,
      *adaptive_visual_policy_,
      multi_camera_landmark_linearization_shadow_config_);
  }
  multi_camera_landmark_linearization_shadow_batch_audit_.
    configured_enabled =
    multi_camera_landmark_linearization_shadow_config_.enabled;
  multi_camera_landmark_linearization_shadow_batch_audit_.
    module_active =
    multi_camera_landmark_linearization_shadow_ != nullptr &&
    multi_camera_landmark_linearization_shadow_->active();
  if (
    multi_camera_landmark_linearization_shadow_config_.enabled &&
    !multi_camera_landmark_linearization_shadow_batch_audit_.
      module_active)
  {
    multi_camera_landmark_linearization_shadow_batch_audit_.
      invocation =
      livo::VIOMultiCameraLandmarkShadowInvocation::
        kModuleUnavailable;
  }

  printf("intrinsic: %.6lf, %.6lf, %.6lf, %.6lf\n", fx, fy, cx, cy);

  printf("width: %d, height: %d, scale: %f\n", width, height, image_resize_factor);

  if (grid_size > 10)
  {
    grid_n_width = ceil(static_cast<double>(width / grid_size));
    grid_n_height = ceil(static_cast<double>(height / grid_size));
  }
  else
  {
    grid_size = static_cast<int>(height / grid_n_height);
    grid_n_height = ceil(static_cast<double>(height / grid_size));
    grid_n_width = ceil(static_cast<double>(width / grid_size));
  }
  length = grid_n_width * grid_n_height;

  if(raycast_en)
  {
    // cv::Mat img_test = cv::Mat::zeros(height, width, CV_8UC1);
    // uchar* it = (uchar*)img_test.data;

    border_flag.resize(length, 0);

    std::vector<std::vector<V3D>>().swap(rays_with_sample_points);
    rays_with_sample_points.reserve(length);
    printf("grid_size: %d, grid_n_height: %d, grid_n_width: %d, length: %d\n", grid_size, grid_n_height, grid_n_width, length);

    float d_min = 0.1;
    float d_max = 3.0;
    float step = 0.2;
    for (int grid_row = 1; grid_row <= grid_n_height; grid_row++)
    {
      for (int grid_col = 1; grid_col <= grid_n_width; grid_col++)
      {
        std::vector<V3D> SamplePointsEachGrid;
        int index = (grid_row - 1) * grid_n_width + grid_col - 1;

        if (grid_row == 1 || grid_col == 1 || grid_row == grid_n_height || grid_col == grid_n_width) border_flag[index] = 1;

        int u = grid_size / 2 + (grid_col - 1) * grid_size;
        int v = grid_size / 2 + (grid_row - 1) * grid_size;
        // it[ u + v * width ] = 255;
        for (float d_temp = d_min; d_temp <= d_max; d_temp += step)
        {
          V3D xyz;
          xyz = cam->cam2world(u, v);
          xyz *= d_temp / xyz[2];
          // xyz[0] = (u - cx) / fx * d_temp;
          // xyz[1] = (v - cy) / fy * d_temp;
          // xyz[2] = d_temp;
          SamplePointsEachGrid.push_back(xyz);
        }
        rays_with_sample_points.push_back(SamplePointsEachGrid);
      }
    }
    // printf("rays_with_sample_points: %d, RaysWithSamplePointsCapacity: %d,
    // rays_with_sample_points[0].capacity(): %d, rays_with_sample_points[0]: %d\n",
    // rays_with_sample_points.size(), rays_with_sample_points.capacity(),
    // rays_with_sample_points[0].capacity(), rays_with_sample_points[0].size()); for
    // (const auto & it : rays_with_sample_points[0]) cout << it.transpose() << endl;
    // cv::imshow("img_test", img_test);
    // cv::waitKey(1);
  }

  if(colmap_output_en)
  {
    pinhole_cam = dynamic_cast<vk::PinholeCamera*>(cam);
    fout_colmap.open(DEBUG_FILE_DIR("Colmap/sparse/0/images.txt"), ios::out);
    fout_colmap << "# Image list with two lines of data per image:\n";
    fout_colmap << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    fout_colmap << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    fout_camera.open(DEBUG_FILE_DIR("Colmap/sparse/0/cameras.txt"), ios::out);
    fout_camera << "# Camera list with one line of data per camera:\n";
    fout_camera << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fout_camera << "1 PINHOLE " << width << " " << height << " "
        << std::fixed << std::setprecision(6)  // 控制浮点数精度为10位
        << fx << " " << fy << " "
        << cx << " " << cy << std::endl;
    fout_camera.close();
  }
  grid_num.resize(length);
  map_index.resize(length);
  map_dist.resize(length);
  update_flag.resize(length);
  scan_value.resize(length);

  patch_size_total = patch_size * patch_size;
  patch_size_half = static_cast<int>(patch_size / 2);
  patch_buffer.resize(patch_size_total);
  warp_len = patch_size_total * patch_pyrimid_level;
  border = (patch_size_half + 1) * (1 << patch_pyrimid_level);

  retrieve_voxel_points.reserve(length);
  append_voxel_points.reserve(length);

  sub_feat_map.clear();
}

void VIOManager::resetGrid()
{
  fill(grid_num.begin(), grid_num.end(), TYPE_UNKNOWN);
  fill(map_index.begin(), map_index.end(), 0);
  fill(map_dist.begin(), map_dist.end(), 10000.0f);
  fill(update_flag.begin(), update_flag.end(), 0);
  fill(scan_value.begin(), scan_value.end(), 0.0f);

  retrieve_voxel_points.clear();
  retrieve_voxel_points.resize(length);

  append_voxel_points.clear();
  append_voxel_points.resize(length);

  total_points = 0;
}

// void VIOManager::resetRvizDisplay()
// {
  // sub_map_ray.clear();
  // sub_map_ray_fov.clear();
  // visual_sub_map_cur.clear();
  // visual_converged_point.clear();
  // map_cur_frame.clear();
  // sample_points.clear();
// }

void VIOManager::computeJacobianAndUpdateEKF(cv::Mat img)
{
  if (total_points == 0) return;

  compute_jacobian_time = update_ekf_time = 0.0;

  for (int level = patch_pyrimid_level - 1; level >= 0; level--)
  {
    if (inverse_composition_en)
    {
      has_ref_patch_cache = false;
      updateStateInverse(img, level);
    }
    else
      updateState(img, level);
  }
  state->cov -= G * state->cov;
  updateFrameState(*state);
}

void VIOManager::updateStateInverse(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);
  V2D pc;
  MD(1, 2) Jimg;
  MD(2, 3) Jdpi;
  MD(1, 3) Jdphi, Jdp, JdR, Jdt;
  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();
  compute_jacobian_time = update_ekf_time = 0.0;
  M3D P_wi_hat;
  bool z_init = true;
  const int H_DIM = total_points * patch_size_total;

  z.resize(H_DIM);
  z.setZero();

  H_sub.resize(H_DIM, 6);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();
    double count_outlier = 0;
    if (has_ref_patch_cache == false) precomputeReferencePatches(level);
    int n_meas = 0;
    float error = 0.0;
    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    P_wi_hat << SKEW_SYM_MATRX(Pwi);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;

    M3D p_hat;

    for (int i = 0; i < total_points; i++)
    {
      float patch_error = 0.0;

      const int scale = (1 << level);

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      pc = cam->world2cam(pf);

      const float u_ref = pc[0];
      const float v_ref = pc[1];
      const int u_ref_i = floorf(pc[0] / scale) * scale;
      const int v_ref_i = floorf(pc[1] / scale) * scale;
      const float subpix_u_ref = (u_ref - u_ref_i) / scale;
      const float subpix_v_ref = (v_ref - v_ref_i) / scale;
      const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      const float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          double res = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] +
                       w_ref_br * img_ptr[scale * width + scale] - P[patch_size_total * level + x * patch_size + y];
          z(i * patch_size_total + x * patch_size + y) = res;
          patch_error += res * res;
          MD(1, 3) J_dR = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 0);
          MD(1, 3) J_dt = H_sub_inv.block<1, 3>(i * patch_size_total + x * patch_size + y, 3);
          JdR = J_dR * Rwi + J_dt * P_wi_hat * Rwi;
          Jdt = J_dt * Rwi;
          H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt;
          n_meas++;
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<6, 6>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      auto solution = -K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f)) { EKF_end = true; }
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
}

void VIOManager::updateState(cv::Mat img, int level)
{
  if (total_points == 0) return;
  StatesGroup old_state = (*state);

  VectorXd z;
  MatrixXd H_sub;
  bool EKF_end = false;
  float last_error = std::numeric_limits<float>::max();

  const int H_DIM = total_points * patch_size_total;
  z.resize(H_DIM);
  z.setZero();
  H_sub.resize(H_DIM, 7);
  H_sub.setZero();

  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double t1 = omp_get_wtime();

    M3D Rwi(state->rot_end);
    V3D Pwi(state->pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
    Jdp_dt = Rci * Rwi.transpose();

    float error = 0.0;
    int n_meas = 0;
    // int max_threads = omp_get_max_threads();
    // int desired_threads = std::min(max_threads, total_points);
    // omp_set_num_threads(desired_threads);

    #ifdef MP_EN
      omp_set_num_threads(MP_PROC_NUM);
      #pragma omp parallel for reduction(+:error, n_meas)
    #endif
    for (int i = 0; i < total_points; i++)
    {
      // printf("thread is %d, i=%d, i address is %p\n", omp_get_thread_num(), i, &i);
      MD(1, 2) Jimg;
      MD(2, 3) Jdpi;
      MD(1, 3) Jdphi, Jdp, JdR, Jdt;

      float patch_error = 0.0;
      int search_level = visual_submap->search_levels[i];
      int pyramid_level = level + search_level;
      int scale = (1 << pyramid_level);
      float inv_scale = 1.0f / scale;

      VisualPoint *pt = visual_submap->voxel_points[i];

      if (pt == nullptr) continue;

      V3D pf = Rcw * pt->pos_ + Pcw;
      V2D pc = cam->world2cam(pf);

      computeProjectionJacobian(pf, Jdpi);
      M3D p_hat;
      p_hat << SKEW_SYM_MATRX(pf);

      float u_ref = pc[0];
      float v_ref = pc[1];
      int u_ref_i = floorf(pc[0] / scale) * scale;
      int v_ref_i = floorf(pc[1] / scale) * scale;
      float subpix_u_ref = (u_ref - u_ref_i) / scale;
      float subpix_v_ref = (v_ref - v_ref_i) / scale;
      float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
      float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
      float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
      float w_ref_br = subpix_u_ref * subpix_v_ref;

      vector<float> P = visual_submap->warp_patch[i];
      double inv_ref_expo = visual_submap->inv_expo_list[i];
      // ROS_ERROR("inv_ref_expo: %.3lf, state->inv_expo_time: %.3lf\n", inv_ref_expo, state->inv_expo_time);

      for (int x = 0; x < patch_size; x++)
      {
        uint8_t *img_ptr = (uint8_t *)img.data + (v_ref_i + x * scale - patch_size_half * scale) * width + u_ref_i - patch_size_half * scale;
        for (int y = 0; y < patch_size; ++y, img_ptr += scale)
        {
          float du =
              0.5f *
              ((w_ref_tl * img_ptr[scale] + w_ref_tr * img_ptr[scale * 2] + w_ref_bl * img_ptr[scale * width + scale] +
                w_ref_br * img_ptr[scale * width + scale * 2]) -
               (w_ref_tl * img_ptr[-scale] + w_ref_tr * img_ptr[0] + w_ref_bl * img_ptr[scale * width - scale] + w_ref_br * img_ptr[scale * width]));
          float dv =
              0.5f *
              ((w_ref_tl * img_ptr[scale * width] + w_ref_tr * img_ptr[scale + scale * width] + w_ref_bl * img_ptr[width * scale * 2] +
                w_ref_br * img_ptr[width * scale * 2 + scale]) -
               (w_ref_tl * img_ptr[-scale * width] + w_ref_tr * img_ptr[-scale * width + scale] + w_ref_bl * img_ptr[0] + w_ref_br * img_ptr[scale]));

          Jimg << du, dv;
          Jimg = Jimg * state->inv_expo_time;
          Jimg = Jimg * inv_scale;
          Jdphi = Jimg * Jdpi * p_hat;
          Jdp = -Jimg * Jdpi;
          JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
          Jdt = Jdp * Jdp_dt;

          double cur_value =
              w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[scale] + w_ref_bl * img_ptr[scale * width] + w_ref_br * img_ptr[scale * width + scale];
          double res = state->inv_expo_time * cur_value - inv_ref_expo * P[patch_size_total * level + x * patch_size + y];

          z(i * patch_size_total + x * patch_size + y) = res;

          patch_error += res * res;
          n_meas += 1;

          if (exposure_estimate_en) { H_sub.block<1, 7>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt, cur_value; }
          else { H_sub.block<1, 6>(i * patch_size_total + x * patch_size + y, 0) << JdR, Jdt; }
        }
      }
      visual_submap->errors[i] = patch_error;
      error += patch_error;
    }

    error = error / n_meas;

    compute_jacobian_time += omp_get_wtime() - t1;

    // printf("\nPYRAMID LEVEL %i\n---------------\n", level);
    // std::cout << "It. " << iteration
    //           << "\t last_error = " << last_error
    //           << "\t new_error = " << error
    //           << std::endl;

    double t3 = omp_get_wtime();

    if (error <= last_error)
    {
      old_state = (*state);
      last_error = error;

      // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov; auto
      // vec = (*state_propagat) - (*state); G = K*H;
      // (*state) += (-K*z + vec - G*vec);

      auto &&H_sub_T = H_sub.transpose();
      H_T_H.setZero();
      G.setZero();
      H_T_H.block<7, 7>(0, 0) = H_sub_T * H_sub;
      MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
      auto &&HTz = H_sub_T * z;
      // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
      auto vec = (*state_propagat) - (*state);
      G.block<DIM_STATE, 7>(0, 0) = K_1.block<DIM_STATE, 7>(0, 0) * H_T_H.block<7, 7>(0, 0);
      MD(DIM_STATE, 1)
      solution = -K_1.block<DIM_STATE, 7>(0, 0) * HTz + vec - G.block<DIM_STATE, 7>(0, 0) * vec.block<7, 1>(0, 0);

      (*state) += solution;
      auto &&rot_add = solution.block<3, 1>(0, 0);
      auto &&t_add = solution.block<3, 1>(3, 0);

      auto &&expo_add = solution.block<1, 1>(6, 0);
      // if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f) && (expo_add.norm() < 0.001f)) EKF_end = true;
      if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))  EKF_end = true;
    }
    else
    {
      (*state) = old_state;
      EKF_end = true;
    }

    update_ekf_time += omp_get_wtime() - t3;

    if (iteration == max_iterations || EKF_end) break;
  }
  // if (state->inv_expo_time < 0.0)  {ROS_ERROR("reset expo time!!!!!!!!!!\n"); state->inv_expo_time = 0.0;}
}

void VIOManager::updateFrameState(StatesGroup state)
{
  M3D Rwi(state.rot_end);
  V3D Pwi(state.pos_end);
  Rcw = Rci * Rwi.transpose();
  Eigen::JacobiSVD<M3D> svd(Rcw, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Rcw = svd.matrixU() * svd.matrixV().transpose();
  if (Rcw.determinant() < 0.0)
  {
    M3D U = svd.matrixU();
    U.col(2) *= -1.0;
    Rcw = U * svd.matrixV().transpose();
  }
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  new_frame_->T_f_w_ = SE3<double>(Rcw, Pcw);
}

bool VIOManager::prepareJointCameraImage(
  const CameraObservation & observation, JointCameraContext & context) const
{
  if (observation.camera_id >= kCameraCount ||
      !camera_configured_[observation.camera_id] || observation.image.empty())
  {
    return false;
  }
  const auto * camera = cameras_[observation.camera_id].get();
  cv::Mat image = observation.image;
  if (camera->width() != image.cols || camera->height() != image.rows)
  {
    cv::resize(
      image, image, cv::Size(camera->width(), camera->height()), 0, 0, CV_INTER_LINEAR);
  }

  context.camera_id = observation.camera_id;
  context.timestamp = observation.timestamp;
  context.photometric_status = observation.photometric_status;
  context.color_image = image.clone();
  if (image.channels() == 3)
  {
    cv::cvtColor(image, context.gray_image, CV_BGR2GRAY);
  }
  else
  {
    context.gray_image = image.clone();
  }
  if (!context.gray_image.isContinuous())
  {
    context.gray_image = context.gray_image.clone();
  }
  const ImageHealthStatistics image_statistics =
    computeImageHealthStatistics(context.gray_image);
  context.saturation_fraction = image_statistics.saturation_fraction;
  context.dark_fraction = image_statistics.dark_fraction;
  context.mean_gradient = image_statistics.mean_gradient;

  return true;
}

bool VIOManager::prepareJointCamera(
  CameraObservation &observation, vector<pointWithVar> &pg,
  const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
  JointCameraContext &context)
{
  ++adaptive_prepare_call_count_;
  if (!activateCamera(observation.camera_id) || observation.image.empty())
  {
    return false;
  }

  if (context.gray_image.empty() && !prepareJointCameraImage(observation, context))
  {
    return false;
  }

  new_frame_.reset(new Frame(cam, context.gray_image));
  updateFrameState(*state);
  resetGrid();
  retrieveFromVisualSparseMap(
    context.gray_image, pg, feat_map,
    !adaptive_policy_config.enabled);

  context.frame = std::move(new_frame_);
  context.submap = std::move(*visual_submap);
  context.unbounded_candidate_points =
    context.submap.voxel_points.size();
  visual_submap->reset();
  return true;
}

VIOManager::JointCameraLinearization VIOManager::linearizeJointCamera(
  JointCameraContext &context, const int level,
  const std::size_t maximum_tracked_patches)
{
  JointCameraLinearization linearization;
  if (!activateCamera(context.camera_id) || context.gray_image.empty())
  {
    return linearization;
  }

  const cv::Mat &img = context.gray_image;
  const std::ptrdiff_t row_stride =
    static_cast<std::ptrdiff_t>(img.step[0]);
  const M3D Rwi(state->rot_end);
  const V3D Pwi(state->pos_end);
  Rcw = Rci * Rwi.transpose();
  Pcw = -Rci * Rwi.transpose() * Pwi + Pci;
  Jdp_dt = Rci * Rwi.transpose();

  return linearizeJointCameraAtState(context, level, maximum_tracked_patches, *state);
}

VIOManager::JointCameraLinearization VIOManager::linearizeJointCameraAtState(
  JointCameraContext & context, const int level,
  const std::size_t maximum_tracked_patches, const StatesGroup & prior) const
{
  JointCameraLinearization linearization;
  if (context.camera_id >= kCameraCount ||
      !camera_configured_[context.camera_id] || context.gray_image.empty())
  {
    return linearization;
  }
  const auto * camera = cameras_[context.camera_id].get();
  const M3D Rci = lidar_to_camera_rotations_[context.camera_id] * Rli;
  const V3D Pci = lidar_to_camera_rotations_[context.camera_id] * Pli +
    lidar_to_camera_translations_[context.camera_id];
  const V3D Pic = -Rci.transpose() * Pci;
  M3D cross;
  cross << SKEW_SYM_MATRX(Pic);
  const M3D Jdphi_dR = Rci;
  const M3D Jdp_dR = -Rci * cross;
  const M3D Rcw = Rci * prior.rot_end.transpose();
  const V3D Pcw = -Rci * prior.rot_end.transpose() * prior.pos_end + Pci;
  const M3D Jdp_dt = Rci * prior.rot_end.transpose();
  const cv::Mat & img = context.gray_image;
  const std::ptrdiff_t row_stride = static_cast<std::ptrdiff_t>(img.step[0]);

  const double max_patch_squared_error =
    joint_update_config.max_residual_rmse *
    joint_update_config.max_residual_rmse *
    static_cast<double>(patch_size_total);
  const double huber_delta =
    0.5 * joint_update_config.max_residual_rmse;
  // Coverage is a coarse image-space admission statistic. Keep it
  // independent from the tracker scratch grid (whose resolution changes
  // with camera/grid configuration) so the 0.20 gate has stable semantics.
  constexpr int kCoverageGridWidth = 4;
  constexpr int kCoverageGridHeight = 4;
  std::array<bool,
    kCoverageGridWidth * kCoverageGridHeight> occupied_grid_cells{};
  struct PatchLinearization
  {
    JointVisualNormalEquation::Matrix6 hessian;
    JointVisualNormalEquation::Vector6 gradient;
    double squared_error{0.0};
    double rmse{0.0};
    std::size_t point_index{0U};
    std::size_t grid_cell{0U};
  };
  std::vector<PatchLinearization> patches;
  patches.reserve(context.submap.voxel_points.size());

  for (std::size_t point_index = 0U;
       point_index < context.submap.voxel_points.size(); ++point_index)
  {
    VisualPoint *pt = context.submap.voxel_points[point_index];
    if (pt == nullptr)
    {
      continue;
    }

    const int search_level = context.submap.search_levels[point_index];
    const int pyramid_level = level + search_level;
    const int scale = 1 << pyramid_level;
    const double inv_scale = 1.0 / static_cast<double>(scale);
    const int sampling_border = (patch_size_half + 1) * scale;

    const V3D pf = Rcw * pt->pos_ + Pcw;
    if (!pf.allFinite() || pf.z() <= 0.0)
    {
      continue;
    }
    const V2D pc = camera->world2cam(pf);
    if (!pc.allFinite() ||
        !camera->isInFrame(pc.cast<int>(), sampling_border))
    {
      continue;
    }

    const std::vector<float> &reference_patch =
      context.submap.warp_patch[point_index];
    if (reference_patch.size() <
        static_cast<std::size_t>(patch_size_total * (level + 1)))
    {
      continue;
    }

    MD(2, 3) Jdpi;
    const double z_inv = 1. / pf[2];
    const double z_inv_2 = z_inv * z_inv;
    Jdpi(0, 0) = camera->fx() * z_inv;
    Jdpi(0, 1) = 0.0;
    Jdpi(0, 2) = -camera->fx() * pf[0] * z_inv_2;
    Jdpi(1, 0) = 0.0;
    Jdpi(1, 1) = camera->fy() * z_inv;
    Jdpi(1, 2) = -camera->fy() * pf[1] * z_inv_2;
    M3D p_hat;
    p_hat << SKEW_SYM_MATRX(pf);

    const double u_ref = pc[0];
    const double v_ref = pc[1];
    const int u_ref_i =
      static_cast<int>(std::floor(pc[0] / scale)) * scale;
    const int v_ref_i =
      static_cast<int>(std::floor(pc[1] / scale)) * scale;
    const double subpix_u_ref = (u_ref - u_ref_i) * inv_scale;
    const double subpix_v_ref = (v_ref - v_ref_i) * inv_scale;
    const double w_ref_tl =
      (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    const double w_ref_tr =
      subpix_u_ref * (1.0 - subpix_v_ref);
    const double w_ref_bl =
      (1.0 - subpix_u_ref) * subpix_v_ref;
    const double w_ref_br = subpix_u_ref * subpix_v_ref;

    JointVisualNormalEquation::Matrix6 patch_hessian =
      JointVisualNormalEquation::Matrix6::Zero();
    JointVisualNormalEquation::Vector6 patch_gradient =
      JointVisualNormalEquation::Vector6::Zero();
    double patch_squared_error = 0.0;
    bool patch_is_finite = true;
    const double inv_ref_expo =
      context.submap.inv_expo_list[point_index];

    for (int x = 0; x < patch_size && patch_is_finite; ++x)
    {
      const int row =
        v_ref_i + x * scale - patch_size_half * scale;
      const int column =
        u_ref_i - patch_size_half * scale;
      const std::uint8_t *img_ptr =
        img.data + static_cast<std::ptrdiff_t>(row) * row_stride + column;

      for (int y = 0; y < patch_size; ++y, img_ptr += scale)
      {
        const double du =
          0.5 *
          ((w_ref_tl * img_ptr[scale] +
            w_ref_tr * img_ptr[scale * 2] +
            w_ref_bl * img_ptr[scale * row_stride + scale] +
            w_ref_br * img_ptr[scale * row_stride + scale * 2]) -
           (w_ref_tl * img_ptr[-scale] +
            w_ref_tr * img_ptr[0] +
            w_ref_bl * img_ptr[scale * row_stride - scale] +
            w_ref_br * img_ptr[scale * row_stride]));
        const double dv =
          0.5 *
          ((w_ref_tl * img_ptr[scale * row_stride] +
            w_ref_tr * img_ptr[scale * row_stride + scale] +
            w_ref_bl * img_ptr[scale * row_stride * 2] +
            w_ref_br * img_ptr[scale * row_stride * 2 + scale]) -
           (w_ref_tl * img_ptr[-scale * row_stride] +
            w_ref_tr * img_ptr[-scale * row_stride + scale] +
            w_ref_bl * img_ptr[0] +
            w_ref_br * img_ptr[scale]));

        MD(1, 2) Jimg;
        Jimg << du, dv;
        Jimg *= prior.inv_expo_time * inv_scale;
        const MD(1, 3) Jdphi = Jimg * Jdpi * p_hat;
        const MD(1, 3) Jdp = -Jimg * Jdpi;
        const MD(1, 3) JdR =
          Jdphi * Jdphi_dR + Jdp * Jdp_dR;
        const MD(1, 3) Jdt = Jdp * Jdp_dt;

        JointVisualNormalEquation::Vector6 jacobian;
        jacobian << JdR.transpose(), Jdt.transpose();

        const double current_value =
          w_ref_tl * img_ptr[0] +
          w_ref_tr * img_ptr[scale] +
          w_ref_bl * img_ptr[scale * row_stride] +
          w_ref_br * img_ptr[scale * row_stride + scale];
        const std::size_t patch_offset =
          static_cast<std::size_t>(
            patch_size_total * level + x * patch_size + y);
        const double residual =
          prior.inv_expo_time * current_value -
          inv_ref_expo * reference_patch[patch_offset];

        if (!std::isfinite(residual) || !jacobian.allFinite())
        {
          patch_is_finite = false;
          break;
        }
        patch_hessian.noalias() += jacobian * jacobian.transpose();
        patch_gradient.noalias() += jacobian * residual;
        patch_squared_error += residual * residual;
      }
    }

    if (!patch_is_finite || !std::isfinite(patch_squared_error))
    {
      continue;
    }
    const int grid_column = std::clamp(
      static_cast<int>(
        pc[0] * static_cast<double>(kCoverageGridWidth) /
        static_cast<double>(img.cols)),
      0, kCoverageGridWidth - 1);
    const int grid_row = std::clamp(
      static_cast<int>(
        pc[1] * static_cast<double>(kCoverageGridHeight) /
        static_cast<double>(img.rows)),
      0, kCoverageGridHeight - 1);
    patches.push_back(
      PatchLinearization{
        patch_hessian,
        patch_gradient,
        patch_squared_error,
        std::sqrt(
          patch_squared_error /
          static_cast<double>(patch_size_total)),
        point_index,
        static_cast<std::size_t>(
          grid_row * kCoverageGridWidth + grid_column)});
    if (patches.size() >= maximum_tracked_patches)
    {
      break;
    }
  }

  std::vector<double> patch_errors;
  patch_errors.reserve(patches.size());
  for (const PatchLinearization & patch : patches)
  {
    patch_errors.push_back(patch.rmse);
  }
  const auto median = [](std::vector<double> values)
    {
      if (values.empty())
      {
        return std::numeric_limits<double>::quiet_NaN();
      }
      const std::size_t middle = values.size() / 2U;
      std::nth_element(
        values.begin(), values.begin() + middle, values.end());
      double value = values[middle];
      if ((values.size() % 2U) == 0U)
      {
        const auto lower = std::max_element(
          values.begin(), values.begin() + middle);
        value = 0.5 * (value + *lower);
      }
      return value;
    };
  const double median_error = median(patch_errors);
  std::vector<double> deviations;
  deviations.reserve(patch_errors.size());
  for (const double error : patch_errors)
  {
    deviations.push_back(std::abs(error - median_error));
  }
  const double mad = median(deviations);
  const double measurement_scale = std::max(
    1.0e-6, std::sqrt(std::max(1.0, joint_img_point_cov)));
  const double robust_scale = std::max(
    1.4826 * std::max(0.0, mad), measurement_scale);
  const double robust_cutoff = std::min(
    joint_update_config.max_residual_rmse,
    median_error + 3.0 * robust_scale);
  for (const PatchLinearization & patch : patches)
  {
    if (!std::isfinite(robust_cutoff) ||
        patch.rmse > robust_cutoff ||
        patch.squared_error > max_patch_squared_error)
    {
      continue;
    }
    const double patch_rmse = std::sqrt(
      patch.squared_error / static_cast<double>(patch_size_total));
    const double robust_weight =
      patch_rmse > huber_delta && patch_rmse > 0.0 ?
      huber_delta / patch_rmse : 1.0;
    linearization.hessian.noalias() +=
      robust_weight * patch.hessian;
    linearization.gradient.noalias() +=
      robust_weight * patch.gradient;
    linearization.squared_residual_sum +=
      robust_weight * patch.squared_error;
    linearization.residual_count +=
      static_cast<std::size_t>(patch_size_total);
    ++linearization.tracked_points;
    ++linearization.inlier_count;
    linearization.accepted_point_indices.push_back(
      patch.point_index);
    occupied_grid_cells[patch.grid_cell] = true;
    context.submap.errors[patch.point_index] =
      static_cast<float>(patch.squared_error);
  }

  linearization.total_patch_count = patches.size();

  const std::size_t occupied_cells =
    static_cast<std::size_t>(std::count(
      occupied_grid_cells.begin(), occupied_grid_cells.end(), true));
  linearization.spatial_coverage =
    occupied_grid_cells.empty() ? 0.0 :
    static_cast<double>(occupied_cells) /
    static_cast<double>(occupied_grid_cells.size());
  return linearization;
}

livo::AdaptiveCameraObservations
VIOManager::buildAdaptiveObservations(
  const CameraObservationBatch & batch,
  std::vector<JointCameraContext> & contexts,
  const bool bind_candidate_generations,
  const livo::AdaptiveVisualPolicy::Matrix6 * lidar_pose_information,
  const bool lidar_information_valid)
{
  static_cast<void>(lidar_pose_information);
  static_cast<void>(lidar_information_valid);
  livo::AdaptiveCameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id].camera_id =
      static_cast<std::uint8_t>(camera_id);
  }

  if (state == nullptr)
  {
    return observations;
  }
  const StatesGroup lio_prior = *state;
  struct DryRunSolution
  {
    MD(DIM_STATE, 1) delta{MD(DIM_STATE, 1)::Zero()};
    StatesGroup state;
    JointVisualNormalEquation::Matrix6 physical_information{
      JointVisualNormalEquation::Matrix6::Zero()};
    double nis{std::numeric_limits<double>::quiet_NaN()};
    double normalized_pose_correction{
      std::numeric_limits<double>::quiet_NaN()};
    double information_gain{
      std::numeric_limits<double>::quiet_NaN()};
  };
  const auto solve_dry_run =
    [this, &lio_prior](
      const JointCameraLinearization & linearization,
      const MD(DIM_STATE, 1) * prior_delta)
      -> std::optional<DryRunSolution>
    {
      const double normalization_denominator =
        static_cast<double>(patch_size_total) *
        std::max(1.0e-12, joint_img_point_cov);
      if (
        !linearization.hessian.allFinite() ||
        !linearization.gradient.allFinite() ||
        !std::isfinite(linearization.squared_residual_sum) ||
        !(normalization_denominator > 0.0))
      {
        return std::nullopt;
      }

      const JointVisualNormalEquation::Matrix6 physical_information =
        linearization.hessian / normalization_denominator;
      const JointVisualNormalEquation::Vector6 physical_b =
        linearization.gradient / normalization_denominator;
      MD(DIM_STATE, DIM_STATE) full_information =
        MD(DIM_STATE, DIM_STATE)::Zero();
      full_information.block<6, 6>(0, 0) = physical_information;
      MD(DIM_STATE, 1) full_b = MD(DIM_STATE, 1)::Zero();
      full_b.block<6, 1>(0, 0) = physical_b;

      const MD(DIM_STATE, DIM_STATE) prior_covariance =
        0.5 * (lio_prior.cov + lio_prior.cov.transpose());
      Eigen::LDLT<MD(DIM_STATE, DIM_STATE)> prior_factor(
        prior_covariance);
      if (
        prior_factor.info() != Eigen::Success ||
        !prior_factor.vectorD().allFinite() ||
        (prior_factor.vectorD().array() <= 0.0).any())
      {
        return std::nullopt;
      }
      const MD(DIM_STATE, DIM_STATE) prior_information =
        prior_factor.solve(MD(DIM_STATE, DIM_STATE)::Identity());
      if (
        !prior_information.allFinite() ||
        !prior_information.isApprox(
          prior_information.transpose(), 1.0e-9))
      {
        return std::nullopt;
      }

      const MD(DIM_STATE, DIM_STATE) system =
        prior_information + full_information;
      Eigen::LDLT<MD(DIM_STATE, DIM_STATE)> system_factor(system);
      if (
        system_factor.info() != Eigen::Success ||
        !system_factor.vectorD().allFinite() ||
        (system_factor.vectorD().array() <= 0.0).any())
      {
        return std::nullopt;
      }
      MD(DIM_STATE, 1) rhs = full_b;
      if (prior_delta != nullptr)
      {
        rhs += prior_information * *prior_delta;
      }
      const MD(DIM_STATE, 1) correction =
        system_factor.solve(-rhs);
      const MD(DIM_STATE, 1) delta =
        (prior_delta == nullptr ?
          MD(DIM_STATE, 1)::Zero() : *prior_delta) + correction;
      const MD(DIM_STATE, DIM_STATE) posterior_covariance =
        system_factor.solve(MD(DIM_STATE, DIM_STATE)::Identity());
      if (
        !rhs.allFinite() || !correction.allFinite() ||
        !delta.allFinite() || !posterior_covariance.allFinite())
      {
        return std::nullopt;
      }

      const JointVisualNormalEquation::Matrix6 prior_pose_covariance =
        0.5 * (lio_prior.cov.block<6, 6>(0, 0) +
        lio_prior.cov.block<6, 6>(0, 0).transpose());
      const JointVisualNormalEquation::Matrix6 posterior_pose_covariance =
        0.5 * (posterior_covariance.block<6, 6>(0, 0) +
        posterior_covariance.block<6, 6>(0, 0).transpose());
      const double prior_logdet = positiveLogDet6(
        prior_pose_covariance);
      const double posterior_logdet = positiveLogDet6(
        posterior_pose_covariance);
      const double information_gain =
        prior_logdet - posterior_logdet;
      if (
        !std::isfinite(prior_logdet) ||
        !std::isfinite(posterior_logdet) ||
        !std::isfinite(information_gain) ||
        information_gain < 0.0)
      {
        return std::nullopt;
      }

      const double q = linearization.squared_residual_sum /
        normalization_denominator;
      const double nis =
        q +
        (prior_delta == nullptr ?
          0.0 : prior_delta->dot(prior_information * *prior_delta)) -
        rhs.dot(system_factor.solve(rhs));
      const double normalized_pose_correction = std::sqrt(
        std::pow(
          delta.head<3>().norm() /
          std::max(1.0e-12,
            joint_update_config.max_rotation_increment_rad), 2.0) +
        std::pow(
          delta.segment<3>(3).norm() /
          std::max(1.0e-12,
            joint_update_config.max_translation_increment_m), 2.0));
      if (
        !std::isfinite(nis) ||
        !std::isfinite(normalized_pose_correction))
      {
        return std::nullopt;
      }

      DryRunSolution result;
      result.delta = delta;
      result.state = lio_prior;
      result.state += delta;
      result.state.cov = posterior_covariance;
      result.physical_information = physical_information;
      // Numerical cancellation may produce a tiny negative NIS, but a
      // materially negative value is inconsistent with the innovation
      // covariance and must fail closed rather than being clipped into a
      // passing observation.
      if (nis < -1.0e-9)
      {
        return std::nullopt;
      }
      result.nis = std::max(0.0, nis);
      result.normalized_pose_correction =
        normalized_pose_correction;
      result.information_gain = information_gain;
      return result;
    };

  const double scout_start = omp_get_wtime();
  const int threads = std::clamp(adaptive_camera_threads, 1, 4);
  std::array<std::exception_ptr, kCameraCount> failures{};
  #pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
  for (std::size_t index = 0U; index < contexts.size(); ++index)
  {
    try
    {
    JointCameraContext & context = contexts[index];
    context.tentative_state.reset();
    context.tentative_accepted_point_indices.clear();
    context.tentative_inlier_points.clear();
    JointCameraContext scout_context;
    scout_context.camera_id = context.camera_id;
    scout_context.gray_image = context.gray_image;
    scout_context.submap = context.submap;
    const bool sealed_adaptive_runtime =
      adaptive_policy_config.enabled &&
      adaptive_policy_config.minimum_selected_cameras == 1U &&
      adaptive_policy_config.maximum_selected_cameras == 1U;
    const std::size_t scout_candidate_cap =
      sealed_adaptive_runtime ?
      adaptive_policy_config.total_candidate_budget :
      livo::adaptiveScoutCandidateBudget(
        adaptive_policy_config.total_candidate_budget,
        context.camera_id);
    limitJointCameraCandidates(
      scout_context, scout_candidate_cap);
    const std::size_t scout_patch_cap =
      sealed_adaptive_runtime ?
      adaptive_policy_config.total_patch_budget :
      std::numeric_limits<std::size_t>::max();
    const JointCameraLinearization preview =
      linearizeJointCameraAtState(scout_context, 0, scout_patch_cap, lio_prior);
    std::optional<DryRunSolution> first_solution =
      solve_dry_run(preview, nullptr);
    JointCameraLinearization final_linearization = preview;
    std::optional<DryRunSolution> final_solution;
    if (first_solution.has_value())
    {
      final_linearization = linearizeJointCameraAtState(
        scout_context, 0, scout_patch_cap, first_solution->state);
      final_solution = solve_dry_run(
        final_linearization, &first_solution->delta);
    }
    // The eventual winner consumes this same final dry-run evidence; there
    // is no post-selection relinearization.
    context.submap.errors = scout_context.submap.errors;

    livo::AdaptiveCameraObservation observation;
    observation.camera_id = context.camera_id;
    observation.frame_received = true;
    observation.source_timestamp_sec = context.timestamp;
    observation.tracking_map_initialized =
      adaptive_visual_map_lifecycle_[context.camera_id] !=
      livo::AdaptiveVisualMapLifecycle::kEmpty;
    observation.zero_map_reentry_ready =
      observation.tracking_map_initialized &&
      !adaptive_zero_map_reseed_latched_[context.camera_id] &&
      adaptive_zero_map_reentry_healthy_scouts_[context.camera_id] >=
        adaptive_policy_config.on_hysteresis_frames;
    if (
      bind_candidate_generations &&
      batch.validatedReceipt().has_value() &&
      batch.validatedReceipt()->valid())
    {
      observation.observation_generation =
        batch.validatedReceipt()->observationGeneration(
        context.camera_id);
      observation.image_generation =
        batch.validatedReceipt()->imageGeneration(
        context.camera_id);
    }
    observation.available_metrics =
      livo::adaptiveMetricMask(
        livo::AdaptiveHealthMetric::kCandidateCount) |
      livo::adaptiveMetricMask(
        livo::AdaptiveHealthMetric::kTrackedCount) |
      livo::adaptiveMetricMask(
        livo::AdaptiveHealthMetric::kSaturationFraction) |
      livo::adaptiveMetricMask(
        livo::AdaptiveHealthMetric::kDarkFraction) |
      livo::adaptiveMetricMask(
        livo::AdaptiveHealthMetric::kMeanGradient) |
      livo::adaptiveMetricMask(
        livo::AdaptiveHealthMetric::kSpatialCoverage);
    observation.candidate_points =
      scout_context.submap.voxel_points.size();
    const JointCameraLinearization & evidence =
      final_solution.has_value() ? final_linearization : preview;
    observation.tracked_points = evidence.tracked_points;
    observation.residual_count = evidence.residual_count;
    observation.inlier_count = evidence.inlier_count;
    observation.total_patch_count = evidence.total_patch_count;
    observation.inlier_ratio =
      evidence.total_patch_count == 0U ? 0.0 :
      static_cast<double>(evidence.inlier_count) /
      static_cast<double>(evidence.total_patch_count);
    observation.squared_residual_sum =
      evidence.squared_residual_sum;
    observation.saturation_fraction =
      context.saturation_fraction;
    observation.dark_fraction = context.dark_fraction;
    observation.mean_gradient = context.mean_gradient;
    observation.spatial_coverage =
      evidence.spatial_coverage;
    observation.visual_information =
      final_solution.has_value() ?
      final_solution->physical_information :
      JointVisualNormalEquation::Matrix6::Zero();
    observation.nis_dof = evidence.inlier_count;
    if (final_solution.has_value())
    {
      observation.nis = final_solution->nis;
      observation.normalized_pose_correction =
        final_solution->normalized_pose_correction;
      observation.robust_information_gain =
        final_solution->information_gain;
      context.tentative_state = final_solution->state;
      context.tentative_accepted_point_indices =
        final_linearization.accepted_point_indices;
      for (const std::size_t point_index :
           final_linearization.accepted_point_indices)
      {
        if (point_index < scout_context.submap.voxel_points.size())
        {
          context.tentative_inlier_points.push_back(
            scout_context.submap.voxel_points[point_index]);
        }
      }
    }
    if (
      evidence.residual_count > 0U &&
      std::isfinite(evidence.squared_residual_sum))
    {
      observation.available_metrics |=
        livo::adaptiveMetricMask(
          livo::AdaptiveHealthMetric::
          kPhotometricResidual);
    }
    if (
      observation.visual_information.allFinite() &&
      observation.visual_information.trace() > 0.0)
    {
      observation.available_metrics |=
        livo::adaptiveMetricMask(
          livo::AdaptiveHealthMetric::
          kVisualInformation);
    }
    if (
      observation.inlier_count > 0U &&
      observation.total_patch_count > 0U &&
      std::isfinite(observation.inlier_ratio) &&
      std::isfinite(observation.nis) &&
      std::isfinite(observation.normalized_pose_correction) &&
      std::isfinite(observation.robust_information_gain))
    {
      observation.available_metrics |=
        livo::kAdaptiveRobustAdmissionMetricMask;
    }
    observations[context.camera_id] = observation;
    }
    catch (...)
    {
      failures[index] = std::current_exception();
    }
  }
  for (const auto & failure : failures)
  {
    if (failure) std::rethrow_exception(failure);
  }
  adaptive_scout_call_count_ += contexts.size();
  livo::emitStructuredMetric(
    "[CAMERA_SCOUT_TIMING] threads=%d cameras=%zu total_s=%.9f\n",
    threads, contexts.size(), omp_get_wtime() - scout_start);
  return observations;
}

void VIOManager::advanceTrustedZeroMapReentryState(
  const livo::AdaptiveVisualCandidatePlanCapability &
    candidate_plan,
  const std::uint64_t lidar_generation)
{
  const livo::AdaptiveCameraObservations & observations =
    candidate_plan.observations();
  const livo::AdaptiveVisualPolicyPlan & plan =
    candidate_plan.decisionPlan();
  for (std::size_t camera_id = 0U;
       camera_id < kCameraCount; ++camera_id)
  {
    const livo::AdaptiveCameraObservation & observation =
      observations[camera_id];
    const livo::AdaptiveCameraDecision & decision =
      plan.cameras[camera_id];
    std::size_t & healthy_scouts =
      adaptive_zero_map_reentry_healthy_scouts_[camera_id];
    bool & reseed_latched =
      adaptive_zero_map_reseed_latched_[camera_id];
    std::uint64_t & last_trusted_generation =
      adaptive_zero_map_last_trusted_generation_[camera_id];
    const bool generation_is_consecutive =
      last_trusted_generation !=
        std::numeric_limits<std::uint64_t>::max() &&
      last_trusted_generation + 1U == lidar_generation;
    last_trusted_generation = lidar_generation;

    if (
      adaptive_visual_map_lifecycle_[camera_id] !=
      livo::AdaptiveVisualMapLifecycle::kInitialized)
    {
      healthy_scouts = 0U;
      reseed_latched = false;
      continue;
    }
    if (hasAdaptiveVisualMapSupport(
        observation, adaptive_policy_config))
    {
      healthy_scouts = 0U;
      reseed_latched = false;
      continue;
    }
    if (
      !observation.frame_received ||
      decision.image_hard_failure ||
      !decision.image_recovery_healthy)
    {
      healthy_scouts = 0U;
      continue;
    }
    if (!generation_is_consecutive)
    {
      healthy_scouts = 0U;
    }
    if (reseed_latched)
    {
      if (
        healthy_scouts <
        adaptive_policy_config.on_hysteresis_frames)
      {
        ++healthy_scouts;
      }
      if (
        healthy_scouts >=
        adaptive_policy_config.on_hysteresis_frames)
      {
        healthy_scouts = 0U;
        reseed_latched = false;
      }
      continue;
    }
    if (
      healthy_scouts <
      adaptive_policy_config.on_hysteresis_frames)
    {
      ++healthy_scouts;
    }
  }
}

VIOManager::JointUpdateResult
VIOManager::computeJointJacobianAndUpdateEKF(
  std::vector<JointCameraContext> &contexts,
  const livo::AdaptiveVisualPolicy::Matrix6 & lidar_pose_information,
  const bool lidar_pose_information_valid,
  const CameraObservationBatch & batch,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle,
  const bool runtime_observation_authority_ready,
  const livo::AdaptiveVisualCandidatePlanCapability *
    candidate_plan_capability,
  const bool commit_state)
{
  JointUpdateResult result;
  const double visual_update_start = omp_get_wtime();
  class StateWorkspace
  {
  public:
    StateWorkspace(
      VIOManager & manager, const bool isolated)
    : manager_(manager), original_(manager.state), isolated_(isolated)
    {
      if (isolated_ && original_ != nullptr)
      {
        working_.emplace(*original_);
        manager_.state = &*working_;
      }
    }

    ~StateWorkspace()
    {
      manager_.state = original_;
    }

    StateWorkspace(const StateWorkspace &) = delete;
    StateWorkspace & operator=(const StateWorkspace &) = delete;

    bool valid() const noexcept
    {
      return original_ != nullptr &&
        (!isolated_ || working_.has_value());
    }

    const std::optional<StatesGroup> & working() const noexcept
    {
      return working_;
    }

  private:
    VIOManager & manager_;
    StatesGroup * original_{nullptr};
    bool isolated_{false};
    std::optional<StatesGroup> working_;
  } state_workspace(*this, !commit_state);
  const auto finish =
    [&result, &state_workspace, visual_update_start]() -> JointUpdateResult
    {
      result.visual_update_time_sec =
        omp_get_wtime() - visual_update_start;
      if (state_workspace.working().has_value() && result.accepted)
      {
        result.tentative_state = state_workspace.working();
      }
      return result;
    };
  if (!state_workspace.valid())
  {
    result.reason = "missing_state";
    return finish();
  }
  if (
    contexts.size() != kCameraCount &&
    !(
      candidate_plan_capability != nullptr &&
      contexts.size() == 1U) &&
    !isAdaptiveSingleCameraBatch(
      adaptive_policy_config, batch, contexts.size()))
  {
    result.reason = "incomplete_camera_batch";
    return finish();
  }

  const StatesGroup batch_prior = *state;
  bool accepted_iteration = false;
  G.setZero();
  compute_jacobian_time = 0.0;
  update_ekf_time = 0.0;

  const double policy_start = omp_get_wtime();
  if (candidate_plan_capability != nullptr)
  {
    result.adaptive_observations =
      candidate_plan_capability->observations();
    result.adaptive_plan =
      candidate_plan_capability->decisionPlan();
  }
  else if (adaptive_policy_config.enabled)
  {
    result.adaptive_observations =
      buildAdaptiveObservations(
        batch, contexts, false, &lidar_pose_information,
        lidar_pose_information_valid);
  }
  if (candidate_plan_capability == nullptr)
  {
    result.adaptive_plan = adaptive_visual_policy_->plan(
      lidar_pose_information, result.adaptive_observations,
      lidar_pose_information_valid);
  }
  if (candidate_plan_capability == nullptr)
  {
    result.validated_adaptive_plan_capability =
      issueValidatedAdaptiveRuntimePlan(
      batch, runtime_bundle,
      runtime_observation_authority_ready,
      lidar_pose_information,
      lidar_pose_information_valid,
      result.adaptive_observations,
      result.adaptive_plan);
  }
  if (adaptive_policy_config.enabled)
  {
    if (result.adaptive_plan.reason !=
        livo::AdaptivePolicyReason::kReady)
    {
      result.adaptive_policy_time_sec =
        omp_get_wtime() - policy_start;
      result.reason =
        std::string("adaptive_") +
        livo::adaptivePolicyReasonName(
          result.adaptive_plan.reason);
      return finish();
    }
    std::size_t enabled_camera_count = 0U;
    for (JointCameraContext & context : contexts)
    {
      const livo::AdaptiveCameraDecision & decision =
        result.adaptive_plan.cameras[context.camera_id];
      if (decision.enabled)
      {
        ++enabled_camera_count;
        limitJointCameraCandidates(
          context, decision.candidate_budget);
      }
    }
    if (enabled_camera_count <
        joint_update_config.min_joint_cameras)
    {
      result.adaptive_policy_time_sec =
        omp_get_wtime() - policy_start;
      result.reason = "adaptive_no_usable_camera";
      return finish();
    }
  }
  result.adaptive_policy_time_sec =
    omp_get_wtime() - policy_start;

  for (int level = patch_pyrimid_level - 1; level >= 0; --level)
  {
    double last_error = std::numeric_limits<double>::infinity();
    StatesGroup best_state = *state;

    for (int iteration = 0; iteration < max_iterations; ++iteration)
    {
      const double linearization_start = omp_get_wtime();
      JointVisualNormalEquation equation(joint_update_config);
      const auto accepted_point_indices_before =
        result.accepted_point_indices;
      for (JointCameraContext &context : contexts)
      {
        const livo::AdaptiveCameraDecision & decision =
          result.adaptive_plan.cameras[context.camera_id];
        if (!decision.enabled)
        {
          continue;
        }
        const JointCameraLinearization camera_linearization =
          linearizeJointCamera(
            context, level, decision.patch_budget);
        result.accepted_point_indices[context.camera_id] =
          camera_linearization.accepted_point_indices;
        equation.addCamera(
          context.camera_id,
          camera_linearization.tracked_points,
          camera_linearization.residual_count,
          camera_linearization.squared_residual_sum,
          camera_linearization.hessian,
          camera_linearization.gradient,
          static_cast<std::size_t>(patch_size_total),
          decision.covariance_scale);
      }
      compute_jacobian_time += omp_get_wtime() - linearization_start;

      if (!equation.trackingReady())
      {
        result.accepted_point_indices =
          accepted_point_indices_before;
        break;
      }
      result.contributing_cameras = equation.contributingCameras();
      result.tracked_points = equation.trackedPoints();
      result.residual_rmse = equation.residualRmse();
      if (!equation.innovationAccepted())
      {
        *state = batch_prior;
        result.reason = "innovation_gate";
        return finish();
      }

      const double current_error =
        equation.residualRmse() * equation.residualRmse();
      if (current_error > last_error)
      {
        *state = best_state;
        break;
      }
      best_state = *state;
      last_error = current_error;

      const double update_start = omp_get_wtime();
      H_T_H.setZero();
      H_T_H.block<6, 6>(0, 0) = equation.hessian();
      const MD(DIM_STATE, DIM_STATE) prior_information =
        (state->cov / joint_img_point_cov).inverse();
      const MD(DIM_STATE, DIM_STATE) system =
        H_T_H + prior_information;
      Eigen::LDLT<MD(DIM_STATE, DIM_STATE)> solver(system);
      if (solver.info() != Eigen::Success)
      {
        *state = batch_prior;
        result.reason = "normal_equation_factorization";
        return finish();
      }
      const MD(DIM_STATE, DIM_STATE) K_1 =
        solver.solve(MD(DIM_STATE, DIM_STATE)::Identity());
      if (solver.info() != Eigen::Success || !K_1.allFinite())
      {
        *state = batch_prior;
        result.reason = "normal_equation_solve";
        return finish();
      }

      const MD(DIM_STATE, 1) vec = (*state_propagat) - (*state);
      G.setZero();
      G.block<DIM_STATE, 6>(0, 0) =
        K_1.block<DIM_STATE, 6>(0, 0) * equation.hessian();
      const MD(DIM_STATE, 1) solution =
        -K_1.block<DIM_STATE, 6>(0, 0) * equation.gradient() +
        vec - G.block<DIM_STATE, 6>(0, 0) *
        vec.block<6, 1>(0, 0);
      if (!solution.allFinite())
      {
        *state = batch_prior;
        result.reason = "non_finite_increment";
        return finish();
      }

      const double rotation_increment =
        solution.block<3, 1>(0, 0).norm();
      const double translation_increment =
        solution.block<3, 1>(3, 0).norm();
      if (!equation.incrementAccepted(
          rotation_increment, translation_increment))
      {
        *state = batch_prior;
        result.rotation_increment_rad = rotation_increment;
        result.translation_increment_m = translation_increment;
        result.reason = "pose_increment_gate";
        return finish();
      }

      (*state) += solution;
      accepted_iteration = true;
      update_ekf_time += omp_get_wtime() - update_start;
      if (rotation_increment * 57.3 < 0.001 &&
          translation_increment * 100.0 < 0.001)
      {
        break;
      }
    }
  }

  if (!accepted_iteration)
  {
    *state = batch_prior;
    result.reason = "insufficient_tracking";
    return finish();
  }

  const MD(DIM_STATE, 1) total_increment = (*state) - batch_prior;
  result.rotation_increment_rad =
    total_increment.block<3, 1>(0, 0).norm();
  result.translation_increment_m =
    total_increment.block<3, 1>(3, 0).norm();
  JointVisualNormalEquation final_gate(joint_update_config);
  if (!final_gate.incrementAccepted(
      result.rotation_increment_rad,
      result.translation_increment_m))
  {
    *state = batch_prior;
    result.reason = "total_pose_increment_gate";
    return finish();
  }

  state->cov -= G * state->cov;
  state->cov =
    0.5 * (state->cov + state->cov.transpose()).eval();
  if (!state->cov.allFinite())
  {
    *state = batch_prior;
    result.reason = "non_finite_covariance";
    return finish();
  }

  result.accepted = true;
  result.reason = "accepted";
  return finish();
}

void VIOManager::runMultiCameraLandmarkLinearizationShadow(
  const CameraObservationBatch & batch,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle,
  const livo::VIORuntimeObservationAuthorityStatus
    runtime_authority,
  const JointUpdateResult & update_result,
  const double timestamp_sec,
  const std::uint64_t lidar_feature_generation)
{
  ++multi_camera_linearization_shadow_entry_count_;
  adaptive_visual_runtime_audit_.
    multi_camera_linearization_shadow_called = true;
  livo::VIOMultiCameraLandmarkLinearizationShadowBatchAudit
    audit;
  audit.configured_enabled =
    multi_camera_landmark_linearization_shadow_config_.enabled;
  audit.runtime_authority = runtime_authority;
  audit.plan_capability_present =
    update_result.validated_adaptive_plan_capability.has_value();
  audit.plan_capability_validated =
    update_result.validated_adaptive_plan_capability.has_value();
  audit.module_active =
    multi_camera_landmark_linearization_shadow_ != nullptr &&
    multi_camera_landmark_linearization_shadow_->active();
  if (multi_camera_landmark_linearization_shadow_ != nullptr)
  {
    audit.configuration_generation =
      multi_camera_landmark_linearization_shadow_->
      configurationGeneration();
    audit.accepted_target_generation =
      multi_camera_landmark_linearization_shadow_->
      acceptedTargetGeneration();
  }

  const auto finish = [this, timestamp_sec,
      lidar_feature_generation](
      const livo::
        VIOMultiCameraLandmarkLinearizationShadowBatchAudit &
          completed) {
      multi_camera_landmark_linearization_shadow_batch_audit_ =
        completed;
      if (completed.configured_enabled)
      {
        emitMultiCameraLandmarkLinearizationShadowMetric(
          timestamp_sec, lidar_feature_generation, completed);
      }
    };

  if (!audit.configured_enabled)
  {
    audit.invocation =
      livo::VIOMultiCameraLandmarkShadowInvocation::kDisabled;
    finish(audit);
    return;
  }
  if (!audit.module_active)
  {
    audit.invocation =
      livo::VIOMultiCameraLandmarkShadowInvocation::
        kModuleUnavailable;
    finish(audit);
    return;
  }
  if (runtime_bundle == nullptr)
  {
    audit.invocation =
      livo::VIOMultiCameraLandmarkShadowInvocation::
        kMissingRuntimeBundle;
    finish(audit);
    return;
  }
  if (
    runtime_authority !=
      livo::VIORuntimeObservationAuthorityStatus::kReady)
  {
    audit.invocation =
      livo::VIOMultiCameraLandmarkShadowInvocation::
        kRuntimeAuthorityNotReady;
    finish(audit);
    return;
  }
  if (!update_result.validated_adaptive_plan_capability.has_value())
  {
    audit.invocation =
      livo::VIOMultiCameraLandmarkShadowInvocation::
        kMissingPlanCapability;
    finish(audit);
    return;
  }

  audit.invocation =
    livo::VIOMultiCameraLandmarkShadowInvocation::kProcessed;
  audit.process_called = true;
  const livo::MultiCameraLandmarkShadowBatchResult result =
    multi_camera_landmark_linearization_shadow_->processBatch(
    *runtime_bundle,
    *update_result.validated_adaptive_plan_capability);
  audit.batch_status = result.status();
  audit.configuration_generation =
    result.configurationGeneration();
  audit.accepted_target_generation =
    result.acceptedTargetGeneration();
  audit.target_camera_batch_sequence =
    result.targetCameraBatchSequence();
  audit.target_runtime_publication_sequence =
    result.targetRuntimePublicationSequence();
  audit.store_revision = result.storeRevision().value;
  audit.inspected_source_observations =
    result.inspectedSourceObservationCount();
  audit.store_authority_violations =
    result.storeAuthorityViolationCount();
  audit.skipped_reference_only_sources =
    result.skippedReferenceOnlySourceCount();
  audit.skipped_tombstoned_sources =
    result.skippedTombstonedSourceCount();
  audit.skipped_inactive_plane_sources =
    result.skippedInactivePlaneSourceCount();
  audit.skipped_sampling_geometry =
    result.skippedSamplingGeometryCount();
  audit.skipped_source_order =
    result.skippedSourceOrderCount();
  audit.skipped_source_runtime_authority =
    result.skippedSourceRuntimeAuthorityCount();
  audit.skipped_source_age =
    result.skippedSourceAgeCount();
  audit.candidate_evaluations =
    result.candidateEvaluationCount();
  audit.strict_photometric_rejected =
    result.strictPhotometricRejectedCount();
  audit.decision_count = result.decisions().size();
  for (const livo::MultiCameraLandmarkShadowDecision & decision :
       result.decisions())
  {
    const bool has_trace = decision.hasSelectionTrace();
    audit.selection_trace_count += has_trace ? 1U : 0U;
    if (
      has_trace &&
      decision.sourceCameraId() == decision.targetCameraId())
    {
      ++audit.same_camera_trace_count;
    }
    if (
      decision.status() ==
      livo::MultiCameraLandmarkShadowDecisionStatus::
        kStrictCrossCameraEvidenceUnavailable)
    {
      ++audit.cross_camera_strict_rejected_count;
    }
    if (
      decision.photometricPath() ==
      livo::MultiCameraLandmarkPhotometricPath::
        kCrossCameraRelativeAffineExperimental)
    {
      ++audit.relative_affine_experimental_count;
    }
  }
  finish(audit);
}

void VIOManager::
emitMultiCameraLandmarkLinearizationShadowMetric(
  const double timestamp_sec,
  const std::uint64_t lidar_feature_generation,
  const livo::
    VIOMultiCameraLandmarkLinearizationShadowBatchAudit &
      audit) const
{
  const char * const batch_status =
    audit.batch_status.has_value() ?
    livo::multiCameraLandmarkShadowBatchStatusName(
      *audit.batch_status) :
    "not_invoked";
  livo::emitStructuredMetric(
    "[VIO_MULTI_CAMERA_LANDMARK_LINEARIZATION_SHADOW] "
    "t=%.9f generation=%llu invocation=%s "
    "runtime_authority=%s batch_status=%s "
    "module_active=%d capability_present=%d "
    "capability_validated=%d process_called=%d "
    "configuration_generation=%llu "
    "accepted_target_generation=%llu batch_sequence=%llu "
    "publication_sequence=%llu store_revision=%llu "
    "inspected_sources=%zu store_authority_violations=%zu "
    "skipped_reference_only=%zu skipped_tombstoned=%zu "
    "skipped_inactive_plane=%zu skipped_geometry=%zu "
    "skipped_order=%zu "
    "skipped_runtime=%zu skipped_age=%zu "
    "candidate_evaluations=%zu decisions=%zu "
    "selection_traces=%zu strict_photo_rejected=%zu "
    "same_camera_traces=%zu cross_camera_strict_rejected=%zu "
    "relative_affine_experimental=%zu "
    "has_linearization=0 consumable=0 production_ready=0 "
    "solver_applied=0\n",
    timestamp_sec,
    static_cast<unsigned long long>(
      lidar_feature_generation),
    livo::vioMultiCameraLandmarkShadowInvocationName(
      audit.invocation),
    livo::vioRuntimeObservationAuthorityStatusName(
      audit.runtime_authority),
    batch_status,
    audit.module_active ? 1 : 0,
    audit.plan_capability_present ? 1 : 0,
    audit.plan_capability_validated ? 1 : 0,
    audit.process_called ? 1 : 0,
    static_cast<unsigned long long>(
      audit.configuration_generation),
    static_cast<unsigned long long>(
      audit.accepted_target_generation),
    static_cast<unsigned long long>(
      audit.target_camera_batch_sequence),
    static_cast<unsigned long long>(
      audit.target_runtime_publication_sequence),
    static_cast<unsigned long long>(audit.store_revision),
    audit.inspected_source_observations,
    audit.store_authority_violations,
    audit.skipped_reference_only_sources,
    audit.skipped_tombstoned_sources,
    audit.skipped_inactive_plane_sources,
    audit.skipped_sampling_geometry,
    audit.skipped_source_order,
    audit.skipped_source_runtime_authority,
    audit.skipped_source_age,
    audit.candidate_evaluations,
    audit.decision_count,
    audit.selection_trace_count,
    audit.strict_photometric_rejected,
    audit.same_camera_trace_count,
    audit.cross_camera_strict_rejected_count,
    audit.relative_affine_experimental_count);
}

void VIOManager::limitJointCameraCandidates(
  JointCameraContext & context,
  const std::size_t maximum_candidates)
{
  const std::size_t candidate_count =
    context.submap.voxel_points.size();
  if (maximum_candidates >= candidate_count)
  {
    return;
  }

  std::vector<std::size_t> selected_indices;
  selected_indices.reserve(maximum_candidates);
  for (std::size_t index = 0U;
       index < maximum_candidates; ++index)
  {
    const double location =
      (static_cast<double>(index) + 0.5) *
      static_cast<double>(candidate_count) /
      static_cast<double>(maximum_candidates);
    selected_indices.push_back(std::min(
      static_cast<std::size_t>(location),
      candidate_count - 1U));
  }

  context.submap.propa_errors =
    selectValues(context.submap.propa_errors, selected_indices);
  context.submap.errors =
    selectValues(context.submap.errors, selected_indices);
  context.submap.warp_patch =
    selectValues(context.submap.warp_patch, selected_indices);
  context.submap.search_levels =
    selectValues(context.submap.search_levels, selected_indices);
  context.submap.voxel_points =
    selectValues(context.submap.voxel_points, selected_indices);
  context.submap.inv_expo_list =
    selectValues(context.submap.inv_expo_list, selected_indices);
}

void VIOManager::finalizeJointCamera(
  JointCameraContext &context, vector<pointWithVar> &pg,
  const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
  const double img_time, const double camera_time_offset_sec,
  const livo::AdaptiveVisualMapMaintenanceDecision &
    map_maintenance)
{
  ++adaptive_finalize_call_count_;
  if (!activateCamera(context.camera_id))
  {
    return;
  }

  img_rgb = context.color_image;
  img_cp = context.color_image.clone();
  new_frame_ = std::move(context.frame);
  updateFrameState(*state);
  resetGrid();

  const double start = omp_get_wtime();
  const bool map_action =
    map_maintenance.seed_new_points ||
    map_maintenance.update_existing_points;
  const std::vector<float> *metric_errors =
    &context.submap.errors;
  std::size_t metric_tracked_points =
    context.submap.voxel_points.size();
  if (map_action)
  {
    retrieveFromVisualSparseMap(
      context.gray_image, pg, feat_map,
      map_maintenance.update_existing_points);
    if (
      map_maintenance.update_existing_points &&
      !context.tentative_inlier_points.empty())
    {
      std::vector<std::size_t> inlier_indices;
      inlier_indices.reserve(
        visual_submap->voxel_points.size());
      for (std::size_t index = 0U;
           index < visual_submap->voxel_points.size(); ++index)
      {
        const VisualPoint * point =
          visual_submap->voxel_points[index];
        if (
          std::find(
            context.tentative_inlier_points.begin(),
            context.tentative_inlier_points.end(), point) !=
          context.tentative_inlier_points.end())
        {
          inlier_indices.push_back(index);
        }
      }
      visual_submap->propa_errors = selectValues(
        visual_submap->propa_errors, inlier_indices);
      visual_submap->errors = selectValues(
        visual_submap->errors, inlier_indices);
      visual_submap->warp_patch = selectValues(
        visual_submap->warp_patch, inlier_indices);
      visual_submap->search_levels = selectValues(
        visual_submap->search_levels, inlier_indices);
      visual_submap->voxel_points = selectValues(
        visual_submap->voxel_points, inlier_indices);
      visual_submap->inv_expo_list = selectValues(
        visual_submap->inv_expo_list, inlier_indices);
    }
    // The inlier filter changes every visual_submap parallel array. Keep the
    // legacy update loop bound in sync with the filtered point count.
    total_points = static_cast<int>(visual_submap->voxel_points.size());
    if (map_maintenance.seed_new_points)
    {
      generateVisualMapPoints(
        context.gray_image, pg, &context.generated_points);
    }
    plotTrackedPoints();
    if (map_maintenance.update_existing_points)
    {
      if (plot_flag) projectPatchFromRefToCur(feat_map);
      updateVisualMapPoints(context.gray_image);
      updateReferencePatch(feat_map);
    }
    metric_errors = &visual_submap->errors;
    metric_tracked_points = visual_submap->voxel_points.size();
  }
  const double elapsed = omp_get_wtime() - start;

  double squared_residual_sum = 0.0;
  for (const float error : *metric_errors)
  {
    squared_residual_sum += error;
  }
  const double residual_rmse =
    metric_errors->empty() ?
    std::numeric_limits<double>::quiet_NaN() :
    std::sqrt(
      squared_residual_sum /
      (static_cast<double>(metric_errors->size()) *
       patch_size_total));
  printf(
    "[VIO_METRIC] t=%.9f camera_id=%u camera=%s tracked=%d "
    "visual_voxels=%zu residual_rmse=%.6f total_s=%.6f joint=1 "
    "camera_dt_ms=%.3f\n",
    img_time, static_cast<unsigned int>(active_camera_id_),
    cameraName(active_camera_id_).c_str(),
    static_cast<int>(metric_tracked_points),
    active_visual_feature_map_->size(), residual_rmse, elapsed,
    camera_time_offset_sec * 1000.0);
}

void VIOManager::refreshSharedLandmarkRevisionSet(
  const std::vector<JointCameraContext> & contexts,
  const std::uint64_t lidar_feature_generation,
  const livo::CameraBatchTimeTransferBundle *
    time_transfer_bundle,
  SharedLandmarkShadowMetrics & metrics)
{
  static_cast<void>(time_transfer_bundle);
  shared_landmark_shadow_session_.
    observeLioPosteriorGeneration(lidar_feature_generation);
  const livo::SharedLandmarkShadowAudit audit =
    shared_landmark_shadow_session_.audit();
  metrics.time_transfer_revision_valid =
    audit.time_transfer_configuration_revision != 0U;
  metrics.photometric_revision_valid =
    !contexts.empty() &&
    std::all_of(
      contexts.begin(), contexts.end(),
      [&audit](const JointCameraContext & context) {
        return
          context.camera_id <
            audit.photometric_capabilities.size() &&
          audit.photometric_capabilities[
            context.camera_id].status ==
          livo::ShadowPhotometricCapabilityStatus::
          kStrictCalibrated;
      });

  if (audit.visual_revision_snapshot.state.value == 0U)
  {
    metrics.revision_reason = "missing_state_revision";
  }
  else if (
    audit.visual_revision_snapshot.
      rig_calibration_revision == 0U)
  {
    metrics.revision_reason = "missing_rig_revision";
  }
  else if (
    std::any_of(
      contexts.begin(), contexts.end(),
      [&audit](const JointCameraContext & context) {
        return
          context.camera_id >= kCameraCount ||
          audit.visual_revision_snapshot.
          camera_model_revisions[context.camera_id] == 0U;
      }))
  {
    metrics.revision_reason =
      "missing_camera_model_revision";
  }
  else if (!metrics.time_transfer_revision_valid)
  {
    metrics.revision_reason =
      "missing_time_transfer_configuration";
  }
  else if (!metrics.photometric_revision_valid)
  {
    metrics.revision_reason =
      "strict_photometric_unavailable";
  }
  else
  {
    metrics.revision_reason = "ready";
  }
}

bool VIOManager::attachSharedLandmarkIdentity(
  VisualPoint & visual_point,
  const pointWithVar & lidar_point) const
{
  if (!lidar_point.plane_support_valid)
  {
    return false;
  }
  livo::LegacyPlaneLandmarkSample sample;
  sample.plane = livo::PlaneIdentity{
    lidar_point.plane_id, lidar_point.plane_epoch};
  sample.geometry_revision = lidar_point.plane_revision;
  sample.identity_origin_world_m =
    lidar_point.plane_identity_origin;
  sample.identity_u_axis_world =
    lidar_point.plane_identity_u_axis;
  sample.identity_v_axis_world =
    lidar_point.plane_identity_v_axis;
  sample.position_world_m = lidar_point.point_w;
  sample.position_covariance_m2 = lidar_point.var;
  sample.normal_world = lidar_point.normal;
  sample.plane_covariance = lidar_point.plane_covariance;
  sample.camera_id = active_camera_id_;
  sample.level = lidar_point.plane_level;

  livo::GeometrySupportKey support_key;
  livo::GeometryVersion geometry_version;
  if (!shared_landmark_shadow_session_.resolveGeometryIdentity(
      sample, support_key, geometry_version))
  {
    return false;
  }
  const std::optional<livo::LandmarkSnapshot> identity =
    shared_landmark_shadow_session_.findLandmark(support_key);
  if (
    !identity.has_value() ||
    identity->lifecycle ==
      livo::LandmarkLifecycle::kTombstoned ||
    identity->geometry_version.plane.id !=
      geometry_version.plane.id ||
    identity->geometry_version.plane.epoch !=
      geometry_version.plane.epoch ||
    identity->geometry_version.revision !=
      geometry_version.revision)
  {
    return false;
  }
  visual_point.shadow_identity_valid_ = true;
  visual_point.shadow_support_key_ = support_key;
  visual_point.shadow_geometry_version_ = geometry_version;
  visual_point.shadow_landmark_id_ =
    identity->landmark_id;
  return true;
}

livo::VIORuntimeObservationAuthorityStatus
VIOManager::validateRuntimeObservationAuthority(
  const CameraObservationBatch & batch,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle)
{
  using Status =
    livo::VIORuntimeObservationAuthorityStatus;
  if (runtime_bundle == nullptr)
  {
    return Status::kMissingRuntimeBundle;
  }
  if (
    !reference_only_migration_authorities_configured_ ||
    !runtime_bundle->valid() ||
    !runtime_bundle->issuedBy(
      reference_only_migration_publisher_authority_))
  {
    return Status::kRuntimePublisherAuthorityMismatch;
  }
  const livo::ValidatedBatchRuntimeComposition &
    composition = runtime_bundle->validatedComposition();
  if (
    !composition.issuedBy(
      reference_only_migration_batch_authority_,
      reference_only_migration_publisher_authority_) ||
    !composition.accepts(*runtime_bundle) ||
    !batch.validatedReceipt().has_value() ||
    !batch.validatedReceipt()->valid() ||
    !batch.validatedReceipt()->issuedBy(
      reference_only_migration_batch_authority_) ||
    batch.validatedReceipt()->batchSequence() !=
      composition.cameraBatchSequence())
  {
    return Status::kCameraBatchAuthorityMismatch;
  }
  if (
    runtime_bundle->activeStateSnapshotStatus() !=
      livo::ActiveStateSnapshotStatus::kValid ||
    state == nullptr ||
    stateScalars(runtime_bundle->activeStateSnapshot()) !=
      stateScalars(*state))
  {
    return Status::kActiveStateSnapshotMismatch;
  }
  const livo::CameraBatchTimeTransferBundle & transfer_bundle =
    runtime_bundle->timeTransferBundle();
  if (
    transfer_bundle.present_camera_count != kCameraCount ||
    !transfer_bundle.allPresentFullStateTransfersValid() ||
    !transfer_bundle.validatedReceipt().has_value())
  {
    return Status::kInvalidTimeTransferBundle;
  }

  const std::uint64_t batch_sequence =
    runtime_bundle->cameraBatchReceipt().batchSequence();
  const std::uint64_t publication_sequence =
    runtime_bundle->publicationSequence();
  if (
    batch_sequence == 0U ||
    publication_sequence == 0U ||
    batch_sequence <=
      last_runtime_observation_batch_sequence_ ||
    publication_sequence <=
      last_runtime_observation_publication_sequence_)
  {
    return Status::kReplayedRuntimeBundle;
  }

  last_runtime_observation_batch_sequence_ = batch_sequence;
  last_runtime_observation_publication_sequence_ =
    publication_sequence;
  return Status::kReady;
}

bool VIOManager::collectRuntimeObservationFrameProvenances(
  const CameraObservationBatch & batch,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle,
  SharedLandmarkShadowMetrics & metrics,
  std::array<std::shared_ptr<
    const livo::ReferenceOnlyMigrationFrameProvenance>,
    kCameraCount> & frame_provenances) const
{
  using Status =
    livo::VIORuntimeObservationAuthorityStatus;
  frame_provenances.fill(nullptr);
  if (
    metrics.runtime_observation_authority !=
      Status::kReady ||
    runtime_bundle == nullptr)
  {
    return false;
  }

  std::array<std::shared_ptr<
    const livo::ReferenceOnlyMigrationFrameProvenance>,
    kCameraCount> accepted;
  accepted.fill(nullptr);
  for (std::uint8_t camera_id = 0U;
       camera_id < kCameraCount; ++camera_id)
  {
    const auto frame =
      runtime_bundle->frameProvenance(camera_id);
    if (
      !frame.has_value() ||
      !frame->valid() ||
      !frame->issuedBy(
        reference_only_migration_publisher_authority_) ||
      frame->cameraBatchSequence() !=
        runtime_bundle->cameraBatchReceipt().
        batchSequence() ||
      frame->runtimePublicationSequence() !=
        runtime_bundle->publicationSequence() ||
      frame->activeStateSnapshotRevision() !=
        runtime_bundle->activeStateSnapshotRevision() ||
      !batch.present[camera_id] ||
      frame->cameraId() != camera_id ||
      frame->exposureTimestampSec() !=
        batch.observations[camera_id].timestamp)
    {
      metrics.runtime_observation_authority =
        Status::kInvalidFrameProvenance;
      metrics.runtime_observation_frames = 0U;
      return false;
    }
    accepted[camera_id] =
      std::make_shared<
      const livo::ReferenceOnlyMigrationFrameProvenance>(
      *frame);
  }
  frame_provenances = std::move(accepted);
  metrics.runtime_observation_frames = kCameraCount;
  return true;
}

std::optional<
  std::vector<VIOManager::ReferenceOnlyMigrationGeometrySeed>>
VIOManager::bootstrapSharedLandmarksShadow(
  const CameraObservationBatch & batch,
  const std::vector<JointCameraContext> & contexts,
  const std::vector<pointWithVar> & lidar_points,
  const std::uint64_t lidar_feature_generation,
  const livo::ActivePlaneIdentitySnapshot *
    active_plane_snapshot,
  const livo::CameraBatchTimeTransferBundle *
    time_transfer_bundle,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle,
  SharedLandmarkShadowMetrics & metrics)
{
  ++shared_landmark_shadow_entry_count_;
  adaptive_visual_runtime_audit_.
    shared_landmark_shadow_called = true;
  static_cast<void>(time_transfer_bundle);
  const livo::SharedLandmarkShadowAudit before =
    shared_landmark_shadow_session_.audit();
  std::optional<
    std::vector<ReferenceOnlyMigrationGeometrySeed>>
    migration_seeds;
  if (
    reference_only_migration_shadow_runtime_init_audit_.
    effective)
  {
    migration_seeds.emplace();
    migration_seeds->reserve(std::min(
      lidar_points.size(),
      before.maximum_geometry_seeds_per_batch));
  }
  if (!before.geometry_enabled)
  {
    metrics.bootstrap_status = "disabled";
    return migration_seeds;
  }

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry =
    livo::GeometrySnapshotId{lidar_feature_generation};
  request.samples.reserve(std::min(
    lidar_points.size(),
    before.maximum_geometry_seeds_per_batch));
  livo::SharedLandmarkShadowObservationRequest
    observation_request;
  if (active_plane_snapshot != nullptr)
  {
    request.active_planes = *active_plane_snapshot;
    observation_request.active_planes =
      *active_plane_snapshot;
  }
  std::array<std::shared_ptr<
    const livo::ReferenceOnlyMigrationFrameProvenance>,
    kCameraCount> frame_provenances;
  const bool runtime_observation_ready =
    before.observation_enabled &&
    collectRuntimeObservationFrameProvenances(
      batch, runtime_bundle, metrics,
      frame_provenances);
  if (runtime_observation_ready)
  {
    observation_request.time_transfer_batch =
      runtime_bundle->timeTransferBundle().
      validatedReceipt();
  }
  else if (before.observation_enabled)
  {
    metrics.observation_status =
      std::string("runtime_") +
      livo::vioRuntimeObservationAuthorityStatusName(
        metrics.runtime_observation_authority);
  }
  observation_request.samples.reserve(std::min(
    lidar_points.size(),
    before.maximum_observations_per_batch));

  for (const pointWithVar & point : lidar_points)
  {
    if (!point.plane_support_valid)
    {
      continue;
    }

    livo::LegacyPlaneLandmarkSample sample;
    sample.plane = livo::PlaneIdentity{
      point.plane_id, point.plane_epoch};
    sample.geometry_revision = point.plane_revision;
    sample.identity_origin_world_m =
      point.plane_identity_origin;
    sample.identity_u_axis_world =
      point.plane_identity_u_axis;
    sample.identity_v_axis_world =
      point.plane_identity_v_axis;
    sample.position_world_m = point.point_w;
    sample.position_covariance_m2 = point.var;
    sample.normal_world = point.normal;
    sample.plane_covariance = point.plane_covariance;
    sample.level = point.plane_level;

    const JointCameraContext * visible_context = nullptr;
    Eigen::Vector2d visible_pixel =
      Eigen::Vector2d::Zero();
    if (
      before.observation_enabled &&
      runtime_observation_ready &&
      observation_request.samples.size() <
        before.maximum_observations_per_batch)
    {
      for (const JointCameraContext & context : contexts)
      {
        if (
          context.frame == nullptr ||
          context.frame->cam_ == nullptr ||
          context.gray_image.empty() ||
          frame_provenances[context.camera_id] == nullptr)
        {
          continue;
        }
        const Eigen::Vector3d point_camera =
          context.frame->w2f(point.point_w);
        if (
          !point_camera.allFinite() ||
          point_camera.z() <= 0.0)
        {
          continue;
        }
        const Eigen::Vector2d pixel =
          context.frame->cam_->world2cam(point_camera);
        if (
          !pixel.allFinite() ||
          !context.frame->cam_->isInFrame(
            pixel.cast<int>(), border))
        {
          continue;
        }
        const float texture_score = vk::shiTomasiScore(
          context.gray_image,
          static_cast<int>(pixel.x()),
          static_cast<int>(pixel.y()));
        if (
          !std::isfinite(texture_score) ||
          texture_score <= 1.0F)
        {
          continue;
        }
        visible_context = &context;
        visible_pixel = pixel;
        break;
      }
    }

    if (visible_context != nullptr)
    {
      sample.camera_id = visible_context->camera_id;
    }
    if (
      migration_seeds.has_value() &&
      migration_seeds->size() <
        before.maximum_geometry_seeds_per_batch)
    {
      ReferenceOnlyMigrationGeometrySeed migration_seed;
      if (shared_landmark_shadow_session_.
          resolveGeometryIdentity(
            sample, migration_seed.support_key,
            migration_seed.geometry_version))
      {
        migration_seed.plane.support_key =
          migration_seed.support_key;
        migration_seed.plane.geometry_version =
          migration_seed.geometry_version;
        migration_seed.plane.point_world_m =
          sample.position_world_m;
        migration_seed.plane.point_covariance_m2 =
          sample.position_covariance_m2;
        migration_seed.plane.normal_world =
          sample.normal_world;
        migration_seed.plane.normal_covariance =
          sample.plane_covariance.block<3, 3>(0, 0);
        migration_seeds->push_back(
          std::move(migration_seed));
      }
    }
    request.samples.push_back(sample);

    if (
      visible_context != nullptr &&
      before.observation_enabled)
    {
      livo::GeometrySupportKey support_key;
      livo::GeometryVersion geometry_version;
      if (!shared_landmark_shadow_session_.
          resolveGeometryIdentity(
            sample, support_key, geometry_version))
      {
        continue;
      }
      livo::SharedLandmarkShadowObservationSample observation;
      observation.support_key = support_key;
      observation.camera_id = visible_context->camera_id;
      observation.exposure_timestamp_sec =
        frame_provenances[visible_context->camera_id]->
        exposureTimestampSec();
      observation.time_transfer =
        frame_provenances[visible_context->camera_id]->
        timeTransferEntry();
      observation.frame_provenance =
        frame_provenances[visible_context->camera_id];
      observation.pixel = visible_pixel;
      observation.level = 0;
      observation.owned_patch_pyramid.resize(1U);
      observation.owned_patch_pyramid.front().resize(
        static_cast<std::size_t>(patch_size_total));
      getImagePatch(
        visible_context->gray_image, visible_pixel,
        observation.owned_patch_pyramid.front().data(), 0);
      observation.frame_photometric_status =
        visible_context->photometric_status;
      observation.anchor_pose_covariance =
        runtime_bundle->activeStateSnapshot().
        cov.block<6, 6>(0, 0);
      observation.is_bootstrap = true;
      observation_request.samples.push_back(
        std::move(observation));
    }
  }

  const livo::SharedLandmarkShadowOperationResult result =
    shared_landmark_shadow_session_.bootstrapGeometry(request);
  metrics.bootstrap_status =
    livo::sharedLandmarkShadowOperationStatusName(
      result.status);
  metrics.geometry_writes += result.geometry_writes;
  if (
    result.status ==
      livo::SharedLandmarkShadowOperationStatus::
      kStaleStoreRevision ||
    result.status ==
      livo::SharedLandmarkShadowOperationStatus::
      kStaleRevision)
  {
    ++metrics.stale_cas;
  }

  if (
    before.observation_enabled &&
    !observation_request.samples.empty())
  {
    ++metrics.adapter_stage_calls;
    livo::SharedLandmarkShadowObservationStageResult staged =
      shared_landmark_shadow_session_.stageObservations(
      observation_request);
    metrics.observation_status =
      std::string("bootstrap_") +
      livo::sharedLandmarkShadowOperationStatusName(
        staged.status);
    metrics.rejected_observations +=
      observation_request.samples.size() -
      staged.staged_observation_count;
    if (staged.transaction.valid())
    {
      const livo::SharedLandmarkShadowOperationResult commit =
        shared_landmark_shadow_session_.commitObservations(
        std::move(staged.transaction));
      if (
        commit.status ==
        livo::SharedLandmarkShadowOperationStatus::
        kObservationCommitted)
      {
        metrics.bootstrapped_observations +=
          commit.observation_writes;
      }
      else
      {
        metrics.rejected_observations +=
          staged.staged_observation_count;
        if (
          commit.status ==
            livo::SharedLandmarkShadowOperationStatus::
            kStaleStoreRevision ||
          commit.status ==
            livo::SharedLandmarkShadowOperationStatus::
            kStaleRevision)
        {
          ++metrics.stale_cas;
        }
        metrics.observation_status =
          std::string("bootstrap_commit_") +
          livo::sharedLandmarkShadowOperationStatusName(
            commit.status);
      }
    }
  }
  return migration_seeds;
}

void VIOManager::runReferenceOnlyMigrationShadowRuntime(
  const CameraObservationBatch & batch,
  const std::vector<JointCameraContext> & contexts,
  const std::vector<ReferenceOnlyMigrationGeometrySeed> &
    geometry_seeds,
  const double batch_timestamp_sec,
  const std::uint64_t lidar_feature_generation,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle)
{
  ++reference_only_migration_shadow_entry_count_;
  adaptive_visual_runtime_audit_.
    reference_only_migration_shadow_called = true;
  const double runtime_start = omp_get_wtime();
  livo::ReferenceOnlyMigrationShadowRuntimeBatchAudit audit;
  audit.lidar_feature_generation =
    lidar_feature_generation;
  audit.batch_timestamp_sec = batch_timestamp_sec;
  audit.captured_seed_count = geometry_seeds.size();
  const auto finish =
    [this, runtime_start](
      livo::ReferenceOnlyMigrationShadowRuntimeBatchAudit
        completed)
    {
      const double measured_runtime =
        omp_get_wtime() - runtime_start;
      completed.runtime_sec =
        std::isfinite(measured_runtime) &&
        measured_runtime >= 0.0 ?
        measured_runtime : 0.0;
      const bool p50_valid =
        completed.ready_warp_condition_p50.has_value() &&
        std::isfinite(
        *completed.ready_warp_condition_p50);
      const bool p95_valid =
        completed.ready_warp_condition_p95.has_value() &&
        std::isfinite(
        *completed.ready_warp_condition_p95);
      const double p50 =
        p50_valid ?
        *completed.ready_warp_condition_p50 : 0.0;
      const double p95 =
        p95_valid ?
        *completed.ready_warp_condition_p95 : 0.0;
      const std::string pair_attempts =
        commaSeparatedCounts(
        completed.pair_attempt_counts);
      const std::string pair_ready =
        commaSeparatedCounts(
        completed.pair_ready_counts);
      const std::string status_counts =
        commaSeparatedCounts(
        completed.migration_status_counts);
      const double source_pose_covariance_trace =
        std::isfinite(
        completed.source_pose_covariance_trace_sum) ?
        completed.source_pose_covariance_trace_sum : 0.0;
      livo::emitStructuredMetric(
        "[VIO_REFERENCE_ONLY_MIGRATION_SHADOW_BATCH] "
        "t=%.9f generation=%llu sequence=%llu reason=%s "
        "publication_sequence=%llu state_revision=%llu "
        "captured=%zu descriptor=%zu canonical_descriptor=%zu "
        "materialized=%zu source_cov_evaluations=%zu "
        "seal_hashes=%llu validation_hashes=%llu "
        "candidate=%zu unique=%zu duplicate=%zu "
        "ambiguous=%zu retained=%zu pair_truncated=%zu "
        "total_truncated=%zu attempt=%zu ready=%zu reject=%zu "
        "p50_valid=%d p50=%.9g p95_valid=%d p95=%.9g "
        "source_pose_cov_trace=%.9g runtime_sec=%.9f "
        "pair_attempts=%s pair_ready=%s "
        "status_counts=%s visibility_verified=0 experimental=1 "
        "consumable=0 production_ready=0 solver_applied=0 "
        "acceptance_receipt_issued=0 "
        "independence_proof_issued=0\n",
        completed.batch_timestamp_sec,
        static_cast<unsigned long long>(
          completed.lidar_feature_generation),
        static_cast<unsigned long long>(
          completed.batch_sequence),
        livo::referenceOnlyMigrationShadowRuntimeBatchReasonName(
          completed.reason),
        static_cast<unsigned long long>(
          completed.runtime_publication_sequence),
        static_cast<unsigned long long>(
          completed.active_state_snapshot_revision),
        completed.captured_seed_count,
        completed.descriptor_count,
        completed.canonical_descriptor_count,
        completed.materialized_candidate_count,
        completed.source_covariance_evaluation_count,
        static_cast<unsigned long long>(
          completed.seal_image_hash_count),
        static_cast<unsigned long long>(
          completed.validation_image_hash_count),
        completed.candidate_count,
        completed.unique_candidate_count,
        completed.duplicate_candidate_count,
        completed.ambiguous_duplicate_count,
        completed.retained_candidate_count,
        completed.pair_budget_truncated_count,
        completed.total_budget_truncated_count,
        completed.attempted_count,
        completed.ready_count,
        completed.rejected_count,
        p50_valid ? 1 : 0, p50,
        p95_valid ? 1 : 0, p95,
        source_pose_covariance_trace,
        completed.runtime_sec,
        pair_attempts.c_str(),
        pair_ready.c_str(),
        status_counts.c_str());
      reference_only_migration_shadow_runtime_batch_audit_ =
        std::move(completed);
    };

  std::array<const JointCameraContext *, kCameraCount>
    contexts_by_camera{{nullptr, nullptr, nullptr, nullptr}};
  bool valid_context_set = contexts.size() == kCameraCount;
  for (const JointCameraContext & context : contexts)
  {
    if (
      context.camera_id >= kCameraCount ||
      contexts_by_camera[context.camera_id] != nullptr)
    {
      valid_context_set = false;
      break;
    }
    contexts_by_camera[context.camera_id] = &context;
  }
  valid_context_set =
    valid_context_set &&
    std::all_of(
      contexts_by_camera.begin(), contexts_by_camera.end(),
      [](const JointCameraContext * context) {
        return context != nullptr;
      });
  if (!valid_context_set)
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kIncompleteFourCameraContexts;
    finish(std::move(audit));
    return;
  }
  if (runtime_bundle == nullptr)
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kMissingRuntimeProvenance;
    finish(std::move(audit));
    return;
  }
  if (
    !runtime_bundle->valid() ||
    !runtime_bundle->issuedBy(
      reference_only_migration_publisher_authority_))
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kRuntimeProvenanceMismatch;
    finish(std::move(audit));
    return;
  }
  audit.runtime_publication_sequence =
    runtime_bundle->publicationSequence();
  audit.active_state_snapshot_revision =
    runtime_bundle->activeStateSnapshotRevision();
  audit.batch_sequence =
    runtime_bundle->cameraBatchReceipt().batchSequence();
  audit.seal_image_hash_count =
    runtime_bundle->cameraBatchReceipt().
    sealImageHashCount();
  const livo::ValidatedBatchRuntimeComposition &
    runtime_composition =
    runtime_bundle->validatedComposition();
  const bool batch_authority_valid =
    runtime_composition.issuedBy(
      reference_only_migration_batch_authority_,
      reference_only_migration_publisher_authority_) &&
    runtime_composition.accepts(*runtime_bundle) &&
    batch.validatedReceipt().has_value() &&
    batch.validatedReceipt()->valid() &&
    batch.validatedReceipt()->issuedBy(
      reference_only_migration_batch_authority_) &&
    batch.validatedReceipt()->batchSequence() ==
      runtime_composition.cameraBatchSequence();
  audit.validation_image_hash_count =
    runtime_bundle->cameraBatchReceipt().
    validationImageHashCount();
  if (
    !batch_authority_valid ||
    audit.batch_sequence == 0U ||
    audit.runtime_publication_sequence == 0U ||
    audit.active_state_snapshot_revision == 0U ||
    audit.active_state_snapshot_revision !=
      audit.runtime_publication_sequence ||
    (reference_only_migration_shadow_runtime_batch_audit_.
      batch_sequence != 0U &&
    audit.batch_sequence <=
      reference_only_migration_shadow_runtime_batch_audit_.
      batch_sequence) ||
    (reference_only_migration_shadow_runtime_batch_audit_.
      runtime_publication_sequence != 0U &&
    audit.runtime_publication_sequence <=
      reference_only_migration_shadow_runtime_batch_audit_.
      runtime_publication_sequence) ||
    (reference_only_migration_shadow_runtime_batch_audit_.
      active_state_snapshot_revision != 0U &&
    audit.active_state_snapshot_revision <=
      reference_only_migration_shadow_runtime_batch_audit_.
      active_state_snapshot_revision))
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kRuntimeProvenanceMismatch;
    finish(std::move(audit));
    return;
  }

  const livo::SharedLandmarkShadowAudit dependency_audit =
    shared_landmark_shadow_session_.audit();
  const auto & init_audit =
    reference_only_migration_shadow_runtime_init_audit_;
  if (
    !init_audit.effective ||
    init_audit.bound_cross_camera_owner_generation == 0U ||
    init_audit.bound_cross_camera_owner_generation !=
      cross_camera_migration_shadow_init_audit_.
      owner_generation ||
    init_audit.bound_rig_revision == 0U ||
    init_audit.bound_rig_revision !=
      cross_camera_migration_shadow_init_audit_.rig_revision ||
    !init_audit.bound_rig_fingerprint.has_value() ||
    init_audit.bound_rig_fingerprint !=
      cross_camera_migration_shadow_init_audit_.
      rig_fingerprint ||
    init_audit.bound_geometry_configuration_generation == 0U ||
    init_audit.bound_geometry_configuration_generation !=
      dependency_audit.configuration_generation ||
    init_audit.bound_time_transfer_revision == 0U ||
    init_audit.bound_time_transfer_revision !=
      dependency_audit.time_transfer_configuration_revision ||
    init_audit.bound_authority_generation == 0U ||
    init_audit.bound_authority_generation !=
      reference_only_migration_authority_generation_)
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kDependencyStale;
    finish(std::move(audit));
    return;
  }

  for (std::size_t camera_id = 0U;
       camera_id < kCameraCount; ++camera_id)
  {
    if (
      !registered_camera_parameters_[camera_id].has_value() ||
      !batch.present[camera_id] ||
      batch.observations[camera_id].image.cols !=
        registered_camera_parameters_[camera_id]->width ||
      batch.observations[camera_id].image.rows !=
        registered_camera_parameters_[camera_id]->height)
    {
      audit.reason =
        livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kImageGeometryMismatch;
      finish(std::move(audit));
      return;
    }
  }

  if (
    runtime_bundle->activeStateSnapshotStatus() ==
      livo::ActiveStateSnapshotStatus::
      kInvalidCovariance)
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidSourceCovariance;
    finish(std::move(audit));
    return;
  }
  if (
    runtime_bundle->activeStateSnapshotStatus() !=
      livo::ActiveStateSnapshotStatus::kValid)
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidActiveStateSnapshot;
    finish(std::move(audit));
    return;
  }
  if (
    state == nullptr ||
    stateScalars(runtime_bundle->activeStateSnapshot()) !=
      stateScalars(*state))
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kRuntimeProvenanceMismatch;
    finish(std::move(audit));
    return;
  }
  const livo::CameraBatchTimeTransferBundle *
    time_transfer_bundle =
    &runtime_bundle->timeTransferBundle();
  if (
    !time_transfer_bundle->
      allPresentFullStateTransfersValid() ||
    time_transfer_bundle->present_camera_count !=
      kCameraCount ||
    !time_transfer_bundle->validatedReceipt().has_value())
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidFullStateTimeTransferBundle;
    finish(std::move(audit));
    return;
  }
  const livo::CameraBatchTimeTransferReceipt & receipt =
    *time_transfer_bundle->validatedReceipt();
  const auto timestamps_match =
    [](const double left, const double right)
    {
      constexpr double kTimestampConsistencyToleranceSec =
        1.0e-9;
      return
        std::isfinite(left) &&
        std::isfinite(right) &&
        std::abs(left - right) <=
        kTimestampConsistencyToleranceSec;
    };
  bool current_batch_binding_valid = timestamps_match(
    receipt.batchTimestampSec(), batch_timestamp_sec) &&
    timestamps_match(
      receipt.anchorTimestampSec(),
      runtime_bundle->activeStateTimestampSec());
  for (std::uint8_t camera_id = 0U;
       current_batch_binding_valid &&
       camera_id < kCameraCount;
       ++camera_id)
  {
    const JointCameraContext & context =
      *contexts_by_camera[camera_id];
    current_batch_binding_valid =
      context.time_transfer_entry.has_value() &&
      context.time_transfer_entry->has_value() &&
      context.time_transfer_entry->cameraId() == camera_id &&
      receipt.accepts(*context.time_transfer_entry) &&
      timestamps_match(
      context.time_transfer_entry->targetTimestampSec(),
      context.timestamp);
  }
  if (!current_batch_binding_valid)
  {
    audit.reason =
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kTimeTransferBatchBindingMismatch;
    finish(std::move(audit));
    return;
  }

  std::array<
    std::optional<livo::MigrationEndpoint>,
    kCameraCount> endpoints;
  std::array<
    std::optional<livo::ReferenceOnlyMigrationSourcePose>,
    kCameraCount> source_poses;
  const auto & active_state_snapshot =
    runtime_bundle->activeStateSnapshot();
  for (std::uint8_t camera_id = 0U;
       camera_id < kCameraCount; ++camera_id)
  {
    const JointCameraContext & context =
      *contexts_by_camera[camera_id];
    endpoints[camera_id] =
      crossCameraMigrationShadowEndpoint(camera_id);
    if (
      context.gray_image.empty() ||
      !context.time_transfer_entry.has_value() ||
      !context.time_transfer_entry->has_value() ||
      !receipt.accepts(*context.time_transfer_entry) ||
      !endpoints[camera_id].has_value() ||
      endpoints[camera_id]->camera_model == nullptr ||
      !endpoints[camera_id]->camera_model->valid())
    {
      continue;
    }

    const livo::CameraBatchTimeTransferEntry & source_entry =
      *context.time_transfer_entry;
    const livo::CameraPoseTransferRaw & source_transfer =
      source_entry.value().raw();
    livo::ReferenceOnlyMigrationSourcePose source_pose;
    source_pose.exposure_timestamp_sec =
      source_entry.targetTimestampSec();
    source_pose.rotation_world_from_imu =
      source_transfer.rotation_world_from_imu;
    source_pose.position_world_m =
      source_transfer.position_world_m;
    source_pose.pose_covariance =
      source_transfer.target_pose_wrt_active_anchor_state *
      active_state_snapshot.cov *
      source_transfer.target_pose_wrt_active_anchor_state.transpose();
    source_pose.pose_covariance +=
      source_transfer.pose_transfer_covariance;
    source_pose.pose_covariance +=
      source_transfer.time_alignment_pose_covariance;
    source_pose.pose_covariance =
      0.5 *
      (source_pose.pose_covariance +
       source_pose.pose_covariance.transpose()).eval();
    ++audit.source_covariance_evaluation_count;
    Eigen::SelfAdjointEigenSolver<
      Eigen::Matrix<double, 6, 6>> covariance_solver(
      source_pose.pose_covariance,
      Eigen::EigenvaluesOnly);
    if (
      !source_pose.pose_covariance.allFinite() ||
      covariance_solver.info() != Eigen::Success ||
      !covariance_solver.eigenvalues().allFinite() ||
      covariance_solver.eigenvalues().minCoeff() < -1.0e-12)
    {
      audit.reason =
        livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kInvalidSourceCovariance;
      finish(std::move(audit));
      return;
    }
    audit.source_pose_covariance_trace_sum +=
      source_pose.pose_covariance.trace();
    source_poses[camera_id] = std::move(source_pose);
  }

  std::vector<
    livo::ReferenceOnlyMigrationLightweightDescriptor>
    descriptors;
  for (std::size_t seed_index = 0U;
       seed_index < geometry_seeds.size(); ++seed_index)
  {
    const ReferenceOnlyMigrationGeometrySeed & seed =
      geometry_seeds[seed_index];
    if (!seed.plane.point_world_m.allFinite()) {
      continue;
    }
    const auto evaluator_fingerprint =
      livo::referenceOnlyMigrationEvaluatorFingerprint(
      seed.support_key,
      seed.geometry_version,
      seed.plane);
    for (std::uint8_t source_camera_id = 0U;
         source_camera_id < kCameraCount;
         ++source_camera_id)
    {
      if (
        !source_poses[source_camera_id].has_value() ||
        !endpoints[source_camera_id].has_value())
      {
        continue;
      }
      const JointCameraContext & source_context =
        *contexts_by_camera[source_camera_id];
      const auto & source_endpoint =
        *endpoints[source_camera_id];
      const auto & source_pose =
        *source_poses[source_camera_id];
      const Eigen::Matrix3d rotation_world_from_camera =
        source_pose.rotation_world_from_imu *
        source_endpoint.rotation_imu_from_camera;
      const Eigen::Vector3d position_world_from_camera =
        source_pose.position_world_m +
        source_pose.rotation_world_from_imu *
        source_endpoint.position_imu_from_camera;
      const Eigen::Vector3d point_camera =
        rotation_world_from_camera.transpose() *
        (seed.plane.point_world_m -
         position_world_from_camera);
      const livo::CameraProjectionResult projection =
        source_endpoint.camera_model->project(point_camera);
      const int source_border = patch_size_half + 1;
      if (
        projection.status !=
          livo::ProjectionAdapterStatus::kValid ||
        !projection.pixel.allFinite() ||
        !source_endpoint.camera_model->contains(
          projection.pixel, source_border) ||
        projection.pixel.x() <
          static_cast<double>(source_border) ||
        projection.pixel.y() <
          static_cast<double>(source_border) ||
        projection.pixel.x() >=
          static_cast<double>(
          source_context.gray_image.cols - source_border) ||
        projection.pixel.y() >=
          static_cast<double>(
          source_context.gray_image.rows - source_border))
      {
        continue;
      }
      const float texture_score = vk::shiTomasiScore(
        source_context.gray_image,
        static_cast<int>(projection.pixel.x()),
        static_cast<int>(projection.pixel.y()));
      if (
        !std::isfinite(texture_score) ||
        texture_score <= 1.0F)
      {
        continue;
      }

      for (std::uint8_t target_camera_id = 0U;
           target_camera_id < kCameraCount;
           ++target_camera_id)
      {
        if (
          target_camera_id == source_camera_id ||
          !endpoints[target_camera_id].has_value() ||
          !contexts_by_camera[target_camera_id]->
            time_transfer_entry.has_value())
        {
          continue;
        }
        livo::ReferenceOnlyMigrationLightweightDescriptor
          descriptor;
        descriptor.payload_index = seed_index;
        descriptor.source_camera_id = source_camera_id;
        descriptor.target_camera_id = target_camera_id;
        descriptor.support_key = seed.support_key;
        descriptor.geometry_version = seed.geometry_version;
        descriptor.source_pixel = projection.pixel;
        descriptor.plane_point_world_m =
          seed.plane.point_world_m;
        descriptor.evaluator_fingerprint =
          evaluator_fingerprint;
        descriptors.push_back(std::move(descriptor));
      }
    }
  }
  audit.descriptor_count = descriptors.size();
  audit.candidate_count = audit.descriptor_count;

  const livo::ReferenceOnlyMigrationRuntimeBudgetPlan
    budget_plan =
    livo::planReferenceOnlyMigrationRuntimeBudget(
    std::move(descriptors));
  audit.canonical_descriptor_count =
    budget_plan.canonical_descriptor_count;
  audit.duplicate_candidate_count =
    budget_plan.duplicate_descriptor_count;
  audit.ambiguous_duplicate_count =
    budget_plan.ambiguous_duplicate_count;
  audit.pair_budget_truncated_count =
    budget_plan.pair_budget_truncated_count;

  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate>
    candidates;
  candidates.reserve(
    budget_plan.retained_descriptors.size());
  for (
    const livo::ReferenceOnlyMigrationLightweightDescriptor &
      descriptor : budget_plan.retained_descriptors)
  {
    const auto & seed =
      geometry_seeds[descriptor.payload_index];
    const auto & target_context =
      *contexts_by_camera[descriptor.target_camera_id];
    const auto & target_entry =
      *target_context.time_transfer_entry;
    livo::ReferenceOnlyMigrationShadowRequest request(
      livo::ReferenceOnlyMigrationTargetTime(
        receipt, target_entry));
    request.source_support_key = seed.support_key;
    request.source_geometry_version =
      seed.geometry_version;
    request.source_endpoint =
      *endpoints[descriptor.source_camera_id];
    request.source_pose =
      *source_poses[descriptor.source_camera_id];
    request.patch.source_pixel =
      descriptor.source_pixel;
    request.patch.source_level = 0U;
    request.patch.patch_width = patch_size;
    request.target_endpoint =
      *endpoints[descriptor.target_camera_id];
    request.target_image_bounds =
      livo::ReferenceOnlyMigrationTargetImageBounds{
      target_context.gray_image.cols,
      target_context.gray_image.rows};
    request.plane = seed.plane;
    candidates.emplace_back(std::move(request));
  }
  audit.materialized_candidate_count = candidates.size();
  audit.retained_candidate_count = candidates.size();
  ++audit.runner_call_count;
  const livo::ReferenceOnlyMigrationShadowBatchResult result =
    livo::runReferenceOnlyMigrationShadowBatch(
    lidar_feature_generation, std::move(candidates));
  audit.unique_candidate_count =
    audit.canonical_descriptor_count;
  audit.duplicate_candidate_count +=
    result.duplicateCandidateCount();
  audit.ambiguous_duplicate_count +=
    result.ambiguousDuplicateCount();
  audit.retained_candidate_count =
    result.retainedCandidateCount();
  audit.pair_budget_truncated_count +=
    result.pairBudgetTruncatedCount();
  audit.total_budget_truncated_count =
    result.totalBudgetTruncatedCount();
  audit.attempted_count = result.attemptedCount();
  audit.ready_count = result.readyCount();
  audit.rejected_count = result.rejectedCount();
  audit.pair_attempt_counts = result.pairAttemptCounts();
  audit.pair_ready_counts = result.pairReadyCounts();
  audit.selected_sequence = result.selectedSequence();
  for (std::size_t status_index = 0U;
       status_index <
       audit.migration_status_counts.size();
       ++status_index)
  {
    audit.migration_status_counts[status_index] =
      result.statusCount(
      static_cast<livo::MigrationStatus>(status_index));
  }
  audit.ready_warp_condition_p50 =
    result.readyWarpConditionP50();
  audit.ready_warp_condition_p95 =
    result.readyWarpConditionP95();
  switch (result.reason())
  {
    case livo::ReferenceOnlyMigrationShadowBatchReason::
      kEmptyInput:
      audit.reason =
        livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kNoCandidate;
      break;
    case livo::ReferenceOnlyMigrationShadowBatchReason::
      kNoEligibleDirectedPair:
      audit.reason =
        livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kNoEligibleDirectedPair;
      break;
    case livo::ReferenceOnlyMigrationShadowBatchReason::
      kEvaluated:
      audit.reason =
        livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kEvaluated;
      break;
  }
  finish(std::move(audit));
}

void VIOManager::mirrorSharedLandmarkObservationsShadow(
  const CameraObservationBatch & batch,
  const std::vector<JointCameraContext> & contexts,
  const JointUpdateResult & update_result,
  const std::uint64_t lidar_feature_generation,
  const livo::ActivePlaneIdentitySnapshot *
    active_plane_snapshot,
  const livo::CameraBatchTimeTransferBundle *
    time_transfer_bundle,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle,
  SharedLandmarkShadowMetrics & metrics)
{
  static_cast<void>(time_transfer_bundle);
  if (!update_result.accepted)
  {
    metrics.observation_status = "joint_rejected";
    return;
  }
  const livo::SharedLandmarkShadowAudit audit =
    shared_landmark_shadow_session_.audit();
  if (!audit.observation_enabled)
  {
    if (metrics.observation_status == "not_called")
    {
      metrics.observation_status = "disabled";
    }
    return;
  }
  if (
    audit.geometry.value != lidar_feature_generation)
  {
    metrics.observation_status =
      "stale_geometry";
    return;
  }

  std::array<std::shared_ptr<
    const livo::ReferenceOnlyMigrationFrameProvenance>,
    kCameraCount> frame_provenances;
  if (!collectRuntimeObservationFrameProvenances(
      batch, runtime_bundle, metrics, frame_provenances))
  {
    metrics.observation_status =
      std::string("runtime_") +
      livo::vioRuntimeObservationAuthorityStatusName(
        metrics.runtime_observation_authority);
    return;
  }

  livo::SharedLandmarkShadowObservationRequest request;
  if (active_plane_snapshot != nullptr)
  {
    request.active_planes = *active_plane_snapshot;
  }
  request.time_transfer_batch =
    runtime_bundle->timeTransferBundle().
    validatedReceipt();
  request.samples.reserve(std::min(
    update_result.tracked_points,
    audit.maximum_observations_per_batch));

  for (const JointCameraContext & context : contexts)
  {
    const livo::AdaptiveCameraDecision & decision =
      update_result.adaptive_plan.cameras[
        context.camera_id];
    if (decision.isolated || !decision.enabled ||
        context.frame == nullptr ||
        frame_provenances[context.camera_id] == nullptr)
    {
      continue;
    }
    const std::vector<std::size_t> & point_indices =
      update_result.accepted_point_indices[
        context.camera_id];
    for (const std::size_t point_index : point_indices)
    {
      if (
        request.samples.size() >=
          audit.maximum_observations_per_batch ||
        point_index >=
          context.submap.voxel_points.size())
      {
        break;
      }
      const VisualPoint * visual_point =
        context.submap.voxel_points[point_index];
      if (
        visual_point == nullptr ||
        !visual_point->shadow_identity_valid_)
      {
        ++metrics.rejected_observations;
        continue;
      }
      const Eigen::Vector3d point_camera =
        context.frame->w2f(visual_point->pos_);
      if (!point_camera.allFinite() ||
          point_camera.z() <= 0.0)
      {
        ++metrics.rejected_observations;
        continue;
      }
      const Eigen::Vector2d pixel =
        context.frame->cam_->world2cam(point_camera);
      if (!pixel.allFinite() ||
          !context.frame->cam_->isInFrame(
            pixel.cast<int>(), border))
      {
        ++metrics.rejected_observations;
        continue;
      }

      livo::SharedLandmarkShadowObservationSample sample;
      sample.support_key =
        visual_point->shadow_support_key_;
      sample.camera_id = context.camera_id;
      sample.exposure_timestamp_sec =
        frame_provenances[context.camera_id]->
        exposureTimestampSec();
      sample.time_transfer =
        frame_provenances[context.camera_id]->
        timeTransferEntry();
      sample.frame_provenance =
        frame_provenances[context.camera_id];
      sample.pixel = pixel;
      sample.level = 0;
      sample.owned_patch_pyramid.resize(1U);
      sample.owned_patch_pyramid.front().resize(
        static_cast<std::size_t>(patch_size_total));
      getImagePatch(
        context.gray_image, pixel,
        sample.owned_patch_pyramid.front().data(), 0);
      sample.frame_photometric_status =
        context.photometric_status;
      sample.anchor_pose_covariance =
        runtime_bundle->activeStateSnapshot().
        cov.block<6, 6>(0, 0);
      sample.visual_update_accepted = true;
      sample.isolated = false;
      request.samples.push_back(std::move(sample));
    }
  }

  if (request.samples.empty())
  {
    metrics.observation_status =
      "no_accepted_identity_points";
    return;
  }

  ++metrics.adapter_stage_calls;
  livo::SharedLandmarkShadowObservationStageResult staged =
    shared_landmark_shadow_session_.stageObservations(request);
  metrics.observation_status =
    livo::sharedLandmarkShadowOperationStatusName(
      staged.status);
  metrics.rejected_observations +=
    request.samples.size() -
    staged.staged_observation_count;
  if (!staged.transaction.valid())
  {
    return;
  }

  const livo::SharedLandmarkShadowOperationResult commit =
    shared_landmark_shadow_session_.commitObservations(
    std::move(staged.transaction));
  if (
    commit.status ==
    livo::SharedLandmarkShadowOperationStatus::
    kObservationCommitted)
  {
    metrics.accepted_observations +=
      commit.observation_writes;
    metrics.observation_status = "accepted_committed";
    return;
  }
  metrics.rejected_observations +=
    staged.staged_observation_count;
  metrics.observation_status =
    std::string("accepted_commit_") +
    livo::sharedLandmarkShadowOperationStatusName(
      commit.status);
  if (
    commit.status ==
      livo::SharedLandmarkShadowOperationStatus::
      kStaleStoreRevision ||
    commit.status ==
      livo::SharedLandmarkShadowOperationStatus::
      kStaleRevision)
  {
    ++metrics.stale_cas;
  }
}

void VIOManager::refreshSharedLandmarkLifecycleMetrics(
  SharedLandmarkShadowMetrics & metrics) const
{
  const livo::SharedLandmarkShadowAudit audit =
    shared_landmark_shadow_session_.audit();
  metrics.lifecycle_lidar_seeded =
    audit.lifecycle_lidar_seeded;
  metrics.lifecycle_same_camera =
    audit.lifecycle_same_camera;
  metrics.lifecycle_migration_shadow =
    audit.lifecycle_migration_shadow;
  metrics.lifecycle_migration_active =
    audit.lifecycle_migration_active;
  metrics.lifecycle_tombstoned =
    audit.lifecycle_tombstoned;
}

void VIOManager::emitSharedLandmarkShadowMetric(
  const double timestamp_sec,
  const std::uint64_t lidar_feature_generation,
  const SharedLandmarkShadowMetrics & metrics) const
{
  const livo::SharedLandmarkShadowAudit audit =
    shared_landmark_shadow_session_.audit();
  livo::emitStructuredMetric(
    "[VIO_LANDMARK_OBSERVATION_SHADOW] t=%.9f generation=%llu "
    "geometry=%llu bootstrap_status=%s observation_status=%s "
    "revision_reason=%s runtime_observation_authority=%s "
    "runtime_observation_frames=%zu "
    "geometry_writes=%zu bootstrapped=%zu "
    "accepted=%zu rejected=%zu stale_cas=%zu "
    "adapter_stage_calls=%zu preview_stage_calls=0 "
    "photo_revision_valid=%d time_revision_valid=%d "
    "lifecycle_lidar_seeded=%zu lifecycle_same_camera=%zu "
    "lifecycle_migration_shadow=%zu "
    "lifecycle_migration_active=%zu lifecycle_tombstoned=%zu "
    "consumable=0 solver_applied=0\n",
    timestamp_sec,
    static_cast<unsigned long long>(
      lidar_feature_generation),
    static_cast<unsigned long long>(
      audit.geometry.value),
    metrics.bootstrap_status.c_str(),
    metrics.observation_status.c_str(),
    metrics.revision_reason.c_str(),
    livo::vioRuntimeObservationAuthorityStatusName(
      metrics.runtime_observation_authority),
    metrics.runtime_observation_frames,
    metrics.geometry_writes,
    metrics.bootstrapped_observations,
    metrics.accepted_observations,
    metrics.rejected_observations,
    metrics.stale_cas,
    metrics.adapter_stage_calls,
    metrics.photometric_revision_valid ? 1 : 0,
    metrics.time_transfer_revision_valid ? 1 : 0,
    metrics.lifecycle_lidar_seeded,
    metrics.lifecycle_same_camera,
    metrics.lifecycle_migration_shadow,
    metrics.lifecycle_migration_active,
    metrics.lifecycle_tombstoned);
}

VIOManager::AdaptiveVisualRuntimePreflightResult
VIOManager::preflightAdaptiveSingleCameraRuntime(
  const CameraObservationBatch & batch,
  const livo::AdaptiveVisualPolicy::Matrix6 &
    lidar_pose_information,
  const bool lidar_pose_information_valid,
  const std::uint64_t lidar_feature_generation,
  const livo::AdaptiveVisualRuntimeAuthorityInput *
    adaptive_runtime_input)
{
  AdaptiveVisualRuntimePreflightResult result;
  result.audit.outcome =
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly;
  result.audit.candidate_mask = batch.present;
  result.audit.policy_revision_before =
    adaptive_visual_policy_->revision();
  result.audit.policy_revision_after =
    result.audit.policy_revision_before;
  if (batch.validatedReceipt().has_value())
  {
    result.audit.camera_batch_sequence =
      batch.validatedReceipt()->batchSequence();
  }

  const auto reject =
    [this, &result](
      const livo::AdaptiveVisualLioOnlyReason reason)
    {
      result.audit.lio_only_reason = reason;
      result.audit.policy_revision_after =
        adaptive_visual_policy_->revision();
      return std::move(result);
    };
  if (adaptive_policy_config.on_hysteresis_frames == 0U)
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::kInvalidRuntimeInput);
  }
  if (adaptive_runtime_input == nullptr)
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kMissingRuntimeInput);
  }
  if (!adaptive_runtime_input->valid())
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kInvalidRuntimeInput);
  }

  result.audit.lidar_status =
    livo::AdaptiveVisualSelectionStatus::kReady;
  result.audit.lidar_generation =
    adaptive_runtime_input->lidarGeneration();
  if (
    adaptive_runtime_input->lidarGeneration() !=
    lidar_feature_generation)
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kLidarGenerationMismatch);
  }
  if (
    adaptive_runtime_input->lidarInformationReady() !=
      lidar_pose_information_valid ||
    (adaptive_runtime_input->lidarPoseInformation().array() !=
      lidar_pose_information.array()).any())
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kLidarInformationMismatch);
  }
  result.audit.validated_lidar_pose_information =
    adaptive_runtime_input->lidarPoseInformation();
  if (!adaptive_runtime_input->lidarInformationReady())
  {
    result.audit.lidar_status =
      livo::AdaptiveVisualSelectionStatus::
      kLidarInformationNotReady;
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kLidarInformationNotReady);
  }

  result.audit.batch_preflight_status =
    adaptive_runtime_input->preflightBatch(batch);
  if (
    *result.audit.batch_preflight_status !=
    livo::AdaptiveVisualSelectionStatus::kReady)
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kBatchPreflightRejected);
  }
  for (std::size_t camera_id = 0U;
       camera_id < batch.observations.size(); ++camera_id)
  {
    if (
      batch.present[camera_id] &&
      (
        !camera_configured_[camera_id] ||
        batch.observations[camera_id].camera_id !=
          camera_id ||
        batch.observations[camera_id].image.empty()))
    {
      return reject(
        livo::AdaptiveVisualLioOnlyReason::
        kCameraPreflightRejected);
    }
  }

  livo::AdaptiveVisualRuntimeAttemptResult attempt =
    adaptive_runtime_input->beginRuntimeAttempt();
  result.audit.runtime_attempt_status = attempt.status;
  result.audit.runtime_attempt_count =
    attempt.runtime_attempt_count;
  result.audit.selection_consumption_count =
    attempt.selection_consumption_count;
  if (
    attempt.status ==
    livo::AdaptiveVisualRuntimeAttemptStatus::
    kAlreadyAttempted)
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kGenerationAlreadyProcessed);
  }
  if (
    attempt.status ==
    livo::AdaptiveVisualRuntimeAttemptStatus::
    kStaleGeneration)
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kStaleRuntimeGeneration);
  }
  if (
    attempt.status !=
      livo::AdaptiveVisualRuntimeAttemptStatus::kReady ||
    !attempt.permit.has_value())
  {
    return reject(
      livo::AdaptiveVisualLioOnlyReason::
      kInvalidRuntimeInput);
  }

  result.permit = std::move(attempt.permit);
  result.ready = true;
  return result;
}

livo::AdaptiveVisualRuntimeAudit
VIOManager::processAdaptiveSingleCameraRuntime(
  CameraObservationBatch & batch,
  vector<pointWithVar> & pg,
  const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &
    feat_map,
  const double img_time,
  const double process_start_time,
  const livo::AdaptiveVisualRuntimeAuthorityInput &
    adaptive_runtime_input,
  livo::AdaptiveVisualRuntimeAttemptPermit
    runtime_attempt_permit,
  livo::AdaptiveVisualRuntimeAudit audit)
{
  const auto finish_lio_only =
    [this, &audit](
      const livo::AdaptiveVisualLioOnlyReason reason)
    {
      audit.outcome =
        livo::AdaptiveVisualRuntimeOutcome::kLioOnly;
      audit.lio_only_reason = reason;
      audit.policy_revision_after =
        adaptive_visual_policy_->revision();
      return audit;
    };
  class ScopedActiveCameraRestore
  {
  public:
    explicit ScopedActiveCameraRestore(
      VIOManager & manager)
    : manager_(manager),
      original_camera_id_(manager.active_camera_id_)
    {
    }

    ~ScopedActiveCameraRestore()
    {
      if (armed_)
      {
        (void)manager_.activateCamera(original_camera_id_);
      }
    }

    ScopedActiveCameraRestore(
      const ScopedActiveCameraRestore &) = delete;
    ScopedActiveCameraRestore & operator=(
      const ScopedActiveCameraRestore &) = delete;

    void release() noexcept
    {
      armed_ = false;
    }

  private:
    VIOManager & manager_;
    std::uint8_t original_camera_id_;
    bool armed_{true};
  };
  ScopedActiveCameraRestore active_camera_restore(*this);

  const double prepare_start = omp_get_wtime();
  std::array<JointCameraContext, kCameraCount> image_contexts;
  std::array<std::exception_ptr, kCameraCount> image_failures{};
  const int image_threads = std::clamp(adaptive_camera_threads, 1, 4);
  #pragma omp parallel for num_threads(image_threads) schedule(static) if(image_threads > 1)
  for (std::size_t camera_id = 0U; camera_id < kCameraCount; ++camera_id)
  {
    if (!batch.present[camera_id]) continue;
    try
    {
      prepareJointCameraImage(batch.observations[camera_id], image_contexts[camera_id]);
    }
    catch (...)
    {
      image_failures[camera_id] = std::current_exception();
    }
  }
  for (const auto & failure : image_failures)
  {
    if (failure) std::rethrow_exception(failure);
  }
  const double image_end = omp_get_wtime();
  // Allocate frame IDs and poses on the estimator thread. Retrieval uses only
  // per-camera scratch and read-only maps until every worker has joined.
  for (std::size_t camera_id = 0U; camera_id < kCameraCount; ++camera_id)
  {
    if (!batch.present[camera_id] || image_contexts[camera_id].gray_image.empty()) continue;
    activateCamera(camera_id);
    new_frame_.reset(new Frame(cam, image_contexts[camera_id].gray_image));
    updateFrameState(*state);
    image_contexts[camera_id].frame = std::move(new_frame_);
    ++adaptive_prepare_call_count_;
    ++audit.prepare_call_count;
  }
  #pragma omp parallel for num_threads(image_threads) schedule(static) if(image_threads > 1)
  for (std::size_t camera_id = 0U; camera_id < kCameraCount; ++camera_id)
  {
    auto &context = image_contexts[camera_id];
    if (!context.frame) continue;
    try
    {
      vector<int> grid(length, TYPE_UNKNOWN);
      vector<float> distances(length, 10000.0f);
      vector<VisualPoint *> points(length, nullptr);
      retrieveCameraVisualMap(context.gray_image, pg, feat_map, false,
        context.frame.get(), camera_id, context.submap, grid, distances, points);
      context.unbounded_candidate_points = context.submap.voxel_points.size();
    }
    catch (...)
    {
      image_failures[camera_id] = std::current_exception();
    }
  }
  for (const auto &failure : image_failures)
  {
    if (failure) std::rethrow_exception(failure);
  }
  std::vector<JointCameraContext> contexts;
  contexts.reserve(kCameraCount);
  for (std::size_t camera_id = 0U;
       camera_id < batch.observations.size(); ++camera_id)
  {
    if (!batch.present[camera_id])
    {
      continue;
    }
    JointCameraContext context = std::move(image_contexts[camera_id]);
    if (context.frame)
    {
      contexts.push_back(std::move(context));
    }
  }
  if (contexts.size() != batch.camera_count)
  {
    return finish_lio_only(
      livo::AdaptiveVisualLioOnlyReason::
      kCameraPreflightRejected);
  }

  livo::emitStructuredMetric(
    "[CAMERA_PREPARE_TIMING] threads=%d image_s=%.9f map_s=%.9f total_s=%.9f\n",
    image_threads, image_end - prepare_start,
    omp_get_wtime() - image_end, omp_get_wtime() - prepare_start);

  for (JointCameraContext & context : contexts)
  {
    if (context.frame != nullptr)
    {
      context.frame->camera_id_ = context.camera_id;
      context.frame->photometric_status_ =
        context.photometric_status;
    }
  }

  // Cold-start is a maintenance-only generation. Seed every present empty
  // camera from the same LIO posterior before visual admission; no camera is
  // allowed to win merely because its map was seeded first. The next
  // generation is the first one eligible for robust visual selection.
  bool lio_seed_maintenance_performed = false;
  if (adaptive_runtime_input.lidarInformationReady())
  {
    for (JointCameraContext & context : contexts)
    {
      const std::size_t camera_id = context.camera_id;
      if (
        adaptive_visual_map_lifecycle_[camera_id] !=
        livo::AdaptiveVisualMapLifecycle::kEmpty)
      {
        continue;
      }
      const bool image_seed_eligible =
        context.saturation_fraction <=
          adaptive_policy_config.maximum_saturation_fraction &&
        context.dark_fraction <=
          adaptive_policy_config.maximum_dark_fraction &&
        context.mean_gradient >=
          0.5 * adaptive_policy_config.healthy_mean_gradient;
      const livo::AdaptiveVisualMapMaintenanceDecision seed_decision =
        livo::planAdaptiveLioAnchoredMapMaintenance(
        livo::AdaptiveVisualLioAnchoredMaintenanceRequest::kBootstrap,
        livo::AdaptiveVisualMapLifecycle::kEmpty,
        true, false, image_seed_eligible,
        adaptive_runtime_input.lidarInformationReady());
      if (!seed_decision.seed_new_points)
      {
        continue;
      }
      finalizeJointCamera(
        context, pg, feat_map, img_time,
        context.timestamp - batch.timestamp, seed_decision);
      ++audit.finalize_call_count;
      if (!camera_feature_maps_[camera_id].empty())
      {
        adaptive_visual_map_lifecycle_[camera_id] =
          livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated;
        lio_seed_maintenance_performed = true;
      }
    }
  }
  if (lio_seed_maintenance_performed)
  {
    audit.generation_update_receipt =
      livo::AdaptiveVisualGenerationUpdateReceipt{
      false, false, "not_called", 0U, 0U,
      livo::AdaptiveVisualMapAction::kColdStartSeed, true, false};
    audit.outcome = livo::AdaptiveVisualRuntimeOutcome::kLioOnly;
    audit.lio_only_reason =
      livo::AdaptiveVisualLioOnlyReason::kVisualMapBootstrap;
    audit.policy_revision_after =
      adaptive_visual_policy_->revision();
    audit.legacy_fallback_used = false;
    ++frame_count;
    const double elapsed = omp_get_wtime() - process_start_time;
    ave_total =
      ave_total * static_cast<double>(frame_count - 1) /
      static_cast<double>(frame_count) +
      elapsed / static_cast<double>(frame_count);
    return audit;
  }

  const livo::AdaptiveCameraObservations observations =
    buildAdaptiveObservations(
      batch, contexts, true,
      &adaptive_runtime_input.lidarPoseInformation(),
      adaptive_runtime_input.lidarInformationReady());
  audit.scouted_camera_count = contexts.size();

  auto candidate_plan =
    adaptive_visual_policy_->issueCandidatePlan(
    batch, adaptive_runtime_input.lidarPoseInformation(),
    observations,
    adaptive_runtime_input.lidarInformationReady());
  if (!candidate_plan.has_value())
  {
    return finish_lio_only(
      livo::AdaptiveVisualLioOnlyReason::
      kCandidatePlanRejected);
  }
  const livo::AdaptiveVisualPolicyPlan &
    authoritative_plan = candidate_plan->decisionPlan();
  for (std::size_t camera_id = 0U;
       camera_id < authoritative_plan.cameras.size();
       ++camera_id)
  {
    const livo::AdaptiveCameraDecision & decision =
      authoritative_plan.cameras[camera_id];
    audit.eligible_mask[camera_id] = decision.eligible;
    audit.reject_reasons[camera_id] =
      decision.reject_reason;
    audit.selected_mask[camera_id] = decision.enabled;
  }
  const std::size_t selected_count =
    static_cast<std::size_t>(std::count(
    audit.selected_mask.begin(), audit.selected_mask.end(),
    true));
  // Every trusted scout advances the per-camera map-recovery state. This is
  // deliberately independent of winner selection so a present side camera
  // cannot starve merely because another camera keeps winning.
  advanceTrustedZeroMapReentryState(
    *candidate_plan, audit.lidar_generation);
  const auto maintain_unvalidated_bootstrap = [&]()
  {
    bool maintenance_performed = false;
    if (!adaptive_runtime_input.lidarInformationReady())
    {
      return false;
    }
    for (JointCameraContext & context : contexts)
    {
      const std::size_t camera_id = context.camera_id;
      const livo::AdaptiveCameraDecision & side_decision =
        authoritative_plan.cameras[camera_id];
      if (
        side_decision.enabled ||
        adaptive_visual_map_lifecycle_[camera_id] !=
          livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated ||
        candidate_plan->observations()[camera_id].inlier_count >=
          adaptive_policy_config.minimum_inliers)
      {
        continue;
      }
      const bool image_seed_eligible =
        context.saturation_fraction <=
          adaptive_policy_config.maximum_saturation_fraction &&
        context.dark_fraction <=
          adaptive_policy_config.maximum_dark_fraction &&
        context.mean_gradient >=
          0.5 * adaptive_policy_config.healthy_mean_gradient;
      const livo::AdaptiveVisualMapMaintenanceDecision maintenance =
        livo::planAdaptiveLioAnchoredMapMaintenance(
        livo::AdaptiveVisualLioAnchoredMaintenanceRequest::kBootstrap,
        livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated,
        true, side_decision.isolated, image_seed_eligible,
        adaptive_runtime_input.lidarInformationReady());
      if (
        !maintenance.seed_new_points ||
        maintenance.update_existing_points)
      {
        continue;
      }
      finalizeJointCamera(
        context, pg, feat_map, img_time,
        context.timestamp - batch.timestamp, maintenance);
      ++audit.finalize_call_count;
      maintenance_performed = true;
    }
    return maintenance_performed;
  };
  const auto maintain_initialized_low_support_maps =
    [&](const std::optional<std::uint8_t> excluded_camera)
  {
    std::size_t reseeded_camera_count = 0U;
    if (!adaptive_runtime_input.lidarInformationReady())
    {
      return reseeded_camera_count;
    }
    for (JointCameraContext & context : contexts)
    {
      const std::size_t camera_id = context.camera_id;
      const livo::AdaptiveCameraObservation & observation =
        candidate_plan->observations()[camera_id];
      const livo::AdaptiveCameraDecision & decision =
        authoritative_plan.cameras[camera_id];
      const bool low_support = !hasAdaptiveVisualMapSupport(
        observation, adaptive_policy_config);
      if (
        (excluded_camera.has_value() &&
         camera_id == *excluded_camera) ||
        adaptive_visual_map_lifecycle_[camera_id] !=
          livo::AdaptiveVisualMapLifecycle::kInitialized ||
        !observation.zero_map_reentry_ready ||
        !low_support ||
        adaptive_zero_map_reseed_latched_[camera_id] ||
        adaptive_zero_map_reentry_healthy_scouts_[camera_id] <
          adaptive_policy_config.on_hysteresis_frames ||
        decision.image_hard_failure ||
        !decision.image_recovery_healthy)
      {
        continue;
      }
      const bool image_seed_eligible =
        context.saturation_fraction <=
          adaptive_policy_config.maximum_saturation_fraction &&
        context.dark_fraction <=
          adaptive_policy_config.maximum_dark_fraction &&
        context.mean_gradient >=
          0.5 * adaptive_policy_config.healthy_mean_gradient;
      const livo::AdaptiveVisualMapMaintenanceDecision maintenance =
        livo::planAdaptiveLioAnchoredMapMaintenance(
        livo::AdaptiveVisualLioAnchoredMaintenanceRequest::kReseed,
        livo::AdaptiveVisualMapLifecycle::kInitialized,
        true, false, image_seed_eligible,
        adaptive_runtime_input.lidarInformationReady());
      if (
        !maintenance.seed_new_points ||
        maintenance.update_existing_points)
      {
        continue;
      }

      context.generated_points.clear();
      finalizeJointCamera(
        context, pg, feat_map, img_time,
        context.timestamp - batch.timestamp, maintenance);
      ++audit.finalize_call_count;
      adaptive_zero_map_reentry_healthy_scouts_[camera_id] = 0U;
      adaptive_zero_map_reseed_latched_[camera_id] = true;
      const bool reseed_succeeded =
        !context.generated_points.empty();
      livo::emitStructuredMetric(
        "[ADAPTIVE_VIO_MAP_MAINTENANCE] schema_version=2 "
        "runtime_path=adaptive_single generation=%llu camera_id=%zu "
        "map_action=%s seed_new_points=1 update_existing_points=0 "
        "attempted=1 succeeded=%d generated_points=%zu\n",
        static_cast<unsigned long long>(audit.lidar_generation),
        camera_id,
        adaptiveVisualMapActionName(
          livo::AdaptiveVisualMapAction::kLidarAnchoredReseed),
        reseed_succeeded ? 1 : 0,
        context.generated_points.size());
      if (!reseed_succeeded)
      {
        continue;
      }
      ++reseeded_camera_count;
    }
    return reseeded_camera_count;
  };
  const auto emit_authoritative_camera_metrics =
    [&batch, &audit, &observations, &authoritative_plan,
      img_time, this](
      const std::array<
        bool, livo::kAdaptiveVisualCameraCount> &
        candidate_mask)
    {
      for (std::size_t camera_id = 0U;
           camera_id < kCameraCount; ++camera_id)
      {
        emitAdaptiveCameraHealthMetric(
          img_time,
          candidate_mask[camera_id] ?
          batch.observations[camera_id].timestamp :
          batch.timestamp,
          batch.timestamp,
          audit.lidar_generation, camera_id,
          cameraName(static_cast<std::uint8_t>(camera_id)),
          observations[camera_id],
          authoritative_plan.cameras[camera_id],
          candidate_mask[camera_id],
          audit.eligible_mask[camera_id],
          audit.selected_mask[camera_id],
          audit.reject_reasons[camera_id],
          adaptive_policy_config);
      }
    };
  const livo::AdaptiveVisualSelectionIssueResult
    selection_issue =
    runtime_attempt_permit.issueSelection(
    *adaptive_visual_policy_, batch, *candidate_plan);
  audit.selection_issue_status =
    selection_issue.status;
  if (
    selection_issue.status !=
      livo::AdaptiveVisualSelectionStatus::kReady ||
    !selection_issue.capability.has_value())
  {
    if (
      selection_issue.status ==
        livo::AdaptiveVisualSelectionStatus::
      kNoSelectedCamera &&
      selected_count == 0U)
    {
      const std::size_t reseeded_camera_count =
        maintain_initialized_low_support_maps(std::nullopt);
      const bool maintenance_performed =
        maintain_unvalidated_bootstrap();
      emit_authoritative_camera_metrics(
        candidate_plan->candidateMask());
      if (reseeded_camera_count > 0U)
      {
        audit.generation_update_receipt =
          livo::AdaptiveVisualGenerationUpdateReceipt{
          false, false, "not_called", 0U, 0U,
          livo::AdaptiveVisualMapAction::kLidarAnchoredReseed,
          true, false};
        audit.outcome =
          livo::AdaptiveVisualRuntimeOutcome::kLioOnly;
        audit.lio_only_reason =
          livo::AdaptiveVisualLioOnlyReason::kVisualMapReseed;
        audit.policy_revision_after =
          adaptive_visual_policy_->revision();
        audit.legacy_fallback_used = false;
        ++frame_count;
        const double elapsed = omp_get_wtime() - process_start_time;
        ave_total =
          ave_total * static_cast<double>(frame_count - 1) /
          static_cast<double>(frame_count) +
          elapsed / static_cast<double>(frame_count);
        return audit;
      }
      if (maintenance_performed)
      {
        audit.generation_update_receipt =
          livo::AdaptiveVisualGenerationUpdateReceipt{
          false, false, "not_called", 0U, 0U,
          livo::AdaptiveVisualMapAction::kColdStartSeed,
          true, false};
        audit.outcome =
          livo::AdaptiveVisualRuntimeOutcome::kLioOnly;
        audit.lio_only_reason =
          livo::AdaptiveVisualLioOnlyReason::kVisualMapBootstrap;
        audit.policy_revision_after =
          adaptive_visual_policy_->revision();
        audit.legacy_fallback_used = false;
        ++frame_count;
        const double elapsed = omp_get_wtime() - process_start_time;
        ave_total =
          ave_total * static_cast<double>(frame_count - 1) /
          static_cast<double>(frame_count) +
          elapsed / static_cast<double>(frame_count);
        return audit;
      }
      audit.policy_revision_after =
        adaptive_visual_policy_->revision();
    }
    return finish_lio_only(
      livo::AdaptiveVisualLioOnlyReason::
      kSelectionIssueRejected);
  }
  audit.capability_issued = true;

  const livo::AdaptiveVisualSelectionConsumeResult
    selection_consume =
    runtime_attempt_permit.consumeSelection(
    *adaptive_visual_policy_, batch,
    *selection_issue.capability);
  audit.selection_consume_status =
    selection_consume.status;
  audit.selection_consumption_count =
    selection_consume.selection_consumption_count;
  if (
    selection_consume.status !=
      livo::AdaptiveVisualSelectionStatus::kReady ||
    !selection_consume.receipt.has_value())
  {
    return finish_lio_only(
      livo::AdaptiveVisualLioOnlyReason::
      kSelectionConsumeRejected);
  }
  const livo::AdaptiveVisualSelectionReceipt & receipt =
    *selection_consume.receipt;
  audit.receipt_consumed = true;
  audit.selected_camera_id = receipt.selectedCameraId();
  audit.candidate_mask = receipt.candidateMask();
  audit.selection_reason = receipt.selectionReason();
  if (
    selected_count != 1U ||
    !audit.selected_camera_id.has_value() ||
    *audit.selected_camera_id >= kCameraCount ||
    receipt.selectedMask() != audit.selected_mask ||
    !audit.candidate_mask[*audit.selected_camera_id] ||
    !audit.selected_mask[*audit.selected_camera_id])
  {
    return finish_lio_only(
      livo::AdaptiveVisualLioOnlyReason::
      kInvalidSelectionReceipt);
  }

  const std::uint8_t selected_camera =
    *audit.selected_camera_id;
  // A seeded-but-unvalidated camera may continue LIO-anchored bootstrap while
  // another camera owns the visual winner. This is maintenance only: it does
  // not enter the EKF transaction, update existing points, or become visual
  // evidence of acceptance.
  (void)maintain_unvalidated_bootstrap();
  (void)maintain_initialized_low_support_maps(selected_camera);
  const auto emit_generation_update_metric =
    [&audit, &batch](
      const std::uint8_t camera_id,
      const livo::AdaptiveVisualGenerationUpdateReceipt &
        generation_receipt)
    {
      const int fusion_camera_id =
        audit.fusion_camera_id.has_value() ?
        static_cast<int>(*audit.fusion_camera_id) : -1;
      const double pose_covariance_volume_reduction =
        audit.pose_covariance_volume_reduction.value_or(
        std::numeric_limits<double>::quiet_NaN());
      livo::emitStructuredMetric(
        "[ADAPTIVE_VIO_UPDATE_METRIC] schema_version=2 "
        "runtime_path=adaptive_single generation=%llu "
        "batch_source_stamp_sec=%.9f camera_batch_sequence=%llu "
        "selected_camera_id=%u fusion_camera_id=%d "
        "selection_receipt_consumed=%d legacy_fallback=%d "
        "solver_called=%d visual_update_accepted=%d "
        "solver_reason=%s solver_tracked_points=%zu "
        "solver_contributing_cameras=%zu map_action=%s "
        "seed_new_points=%d update_existing_points=%d "
        "pose_covariance_volume_reduction=%.9g "
        "selection_reason=%s\n",
        static_cast<unsigned long long>(audit.lidar_generation),
        batch.timestamp,
        static_cast<unsigned long long>(
          audit.camera_batch_sequence),
        static_cast<unsigned int>(camera_id),
        fusion_camera_id,
        audit.receipt_consumed ? 1 : 0,
        audit.legacy_fallback_used ? 1 : 0,
        generation_receipt.solver_called ? 1 : 0,
        generation_receipt.visual_update_accepted ? 1 : 0,
        generation_receipt.solver_reason.c_str(),
        generation_receipt.solver_tracked_points,
        generation_receipt.solver_contributing_cameras,
        adaptiveVisualMapActionName(
          generation_receipt.map_action),
        generation_receipt.seed_new_points ? 1 : 0,
        generation_receipt.update_existing_points ? 1 : 0,
        pose_covariance_volume_reduction,
        livo::adaptiveSelectionReasonName(audit.selection_reason));
    };
  contexts.erase(
    std::remove_if(
      contexts.begin(), contexts.end(),
      [selected_camera](const JointCameraContext & context)
      {
        return context.camera_id != selected_camera;
      }),
    contexts.end());
  if (contexts.size() != 1U)
  {
    return finish_lio_only(
      livo::AdaptiveVisualLioOnlyReason::
      kInvalidSelectionReceipt);
  }
  JointCameraContext & selected_context = contexts.front();
  audit.selected_exposure_timestamp_sec =
    receipt.candidateExposureTimestampSec(selected_camera);
  if (
    !audit.selected_exposure_timestamp_sec.has_value() ||
    !std::isfinite(
      *audit.selected_exposure_timestamp_sec) ||
    selected_context.frame == nullptr)
  {
    return finish_lio_only(
      livo::AdaptiveVisualLioOnlyReason::
      kInvalidSelectionReceipt);
  }
  selected_context.frame->exposure_timestamp_sec_ =
    receipt.candidateExposureTimestampSec(selected_camera);

  const livo::AdaptiveCameraDecision & decision =
    candidate_plan->decisionPlan().
    cameras[selected_camera];
  const livo::AdaptiveCameraObservation &
    selected_observation =
    candidate_plan->observations()[selected_camera];
  audit.solver_called = true;
  ++audit.solver_call_count;
  JointUpdateResult update_result;
  update_result.adaptive_plan = candidate_plan->decisionPlan();
  update_result.adaptive_observations = candidate_plan->observations();
  if (selected_context.tentative_state.has_value())
  {
    update_result.tentative_state =
      std::move(selected_context.tentative_state);
    update_result.accepted_point_indices[selected_camera] =
      std::move(selected_context.tentative_accepted_point_indices);
    update_result.accepted = true;
    update_result.reason = "accepted_dry_run";
    update_result.contributing_cameras = 1U;
    update_result.tracked_points =
      selected_observation.tracked_points;
  }
  else
  {
    update_result.reason = "dry_run_no_tentative_state";
  }
  last_joint_solver_audit_.accepted =
    update_result.accepted;
  last_joint_solver_audit_.reason =
    update_result.reason;
  last_joint_solver_audit_.contributing_cameras =
    update_result.contributing_cameras;
  last_joint_solver_audit_.tracked_points =
    update_result.tracked_points;

  livo::AdaptiveVisualMapMaintenanceDecision map_maintenance;
  if (update_result.accepted)
  {
    // Preflight camera activation before policy accounting. This keeps an
    // activation failure from advancing the sealed policy revision.
    if (!activateCamera(selected_camera))
    {
      // activateCamera is a side-effect-free preflight for this generation's
      // state/map commit. finalizeJointCamera repeats the same deterministic
      // check, so activation failure cannot be reported as fusion.
      update_result.accepted = false;
      update_result.reason = "camera_activation_failed";
    }
    else
    {
      // Commit policy accounting before touching EKF state or maps. A stale
      // or foreign receipt therefore cannot turn a dry-run into a visual
      // write.
      audit.health_commit_status =
        adaptive_visual_policy_->commitHealthObservation(
        *candidate_plan, receipt, true);
      audit.policy_revision_after =
        adaptive_visual_policy_->revision();
      if (
        !audit.health_commit_status.has_value() ||
        *audit.health_commit_status !=
          livo::AdaptiveCandidateHealthCommitStatus::kCommitted)
      {
        update_result.accepted = false;
        update_result.reason = "health_commit_rejected";
      }
      else
      {
        // The selected posterior is the exact one evaluated during
        // admission. This is the sole EKF state commit for the generation.
        *state = std::move(*update_result.tentative_state);
        map_maintenance = livo::planAdaptiveVisualSolverMapCommit(
          true, decision.enabled, true);
        finalizeJointCamera(
          selected_context, pg, feat_map, img_time,
          selected_context.timestamp - batch.timestamp,
          map_maintenance);
        ++audit.finalize_call_count;
        audit.fusion_camera_id = selected_camera;
        adaptive_visual_map_lifecycle_[selected_camera] =
          livo::AdaptiveVisualMapLifecycle::kInitialized;
      }
    }
  }
  audit.generation_update_receipt =
    livo::AdaptiveVisualGenerationUpdateReceipt{
    update_result.accepted,
    true,
    update_result.reason,
    update_result.tracked_points,
    update_result.contributing_cameras,
    (
      update_result.accepted &&
      map_maintenance.seed_new_points &&
      map_maintenance.update_existing_points) ?
    livo::AdaptiveVisualMapAction::
      kAcceptedVisualUpdate :
    livo::AdaptiveVisualMapAction::kNone,
    map_maintenance.seed_new_points,
    map_maintenance.update_existing_points};
  const bool update_remains_lio_only =
    update_result.reason == "dry_run_no_tentative_state" ||
    update_result.reason == "health_commit_rejected" ||
    update_result.reason == "camera_activation_failed";
  audit.outcome = update_result.accepted ?
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted :
    update_remains_lio_only ?
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly :
    livo::AdaptiveVisualRuntimeOutcome::kVisualRejected;
  audit.lio_only_reason =
    update_result.accepted ?
    livo::AdaptiveVisualLioOnlyReason::kNone :
    update_result.reason == "health_commit_rejected" ?
    livo::AdaptiveVisualLioOnlyReason::kHealthCommitRejected :
    update_result.reason == "camera_activation_failed" ?
    livo::AdaptiveVisualLioOnlyReason::kCameraPreflightRejected :
    update_result.reason == "dry_run_no_tentative_state" ?
    livo::AdaptiveVisualLioOnlyReason::kCandidatePlanRejected :
    livo::AdaptiveVisualLioOnlyReason::kNone;
  audit.legacy_fallback_used = false;
  emit_generation_update_metric(
    selected_camera, *audit.generation_update_receipt);

  ++frame_count;
  const double elapsed =
    omp_get_wtime() - process_start_time;
  ave_total =
    ave_total * static_cast<double>(frame_count - 1) /
    static_cast<double>(frame_count) +
    elapsed / static_cast<double>(frame_count);
  emit_authoritative_camera_metrics(audit.candidate_mask);
  return audit;
}

std::optional<livo::AdaptiveVisualPolicyPlanCapability>
VIOManager::issueValidatedAdaptiveRuntimePlan(
  const CameraObservationBatch & batch,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle,
  const bool runtime_observation_authority_ready,
  const livo::AdaptiveVisualPolicy::Matrix6 &
    lidar_pose_information,
  const bool lidar_pose_information_valid,
  const livo::AdaptiveCameraObservations & observations,
  const livo::AdaptiveVisualPolicyPlan & plan)
{
  if (
    !adaptive_policy_config.enabled ||
    !runtime_observation_authority_ready ||
    runtime_bundle == nullptr ||
    !batch.validatedReceipt().has_value())
  {
    return std::nullopt;
  }

  auto capability = adaptive_visual_policy_->issuePlan(
    *runtime_bundle, lidar_pose_information, observations,
    lidar_pose_information_valid);
  if (
    !capability.has_value() ||
    !adaptive_visual_policy_->accepts(*capability) ||
    !capability->runtimeBundle().valid() ||
    !capability->runtimeBundle().issuedBy(
      reference_only_migration_publisher_authority_) ||
    !capability->runtimeBundle().validatedComposition().issuedBy(
      reference_only_migration_batch_authority_,
      reference_only_migration_publisher_authority_) ||
    capability->runtimePublicationSequence() !=
      runtime_bundle->publicationSequence() ||
    capability->cameraBatchSequence() !=
      batch.validatedReceipt()->batchSequence() ||
    capability->runtimeBundle().
        activeStateSnapshotRevision() !=
      runtime_bundle->activeStateSnapshotRevision() ||
    capability->runtimeBundle().
      cameraBatchReceipt().batchSequence() !=
      runtime_bundle->cameraBatchReceipt().batchSequence() ||
    !sameReferenceOnlyMigrationFrameBindings(
      capability->runtimeBundle(), *runtime_bundle) ||
    capability->lidarInformationReady() !=
      lidar_pose_information_valid ||
    !(capability->lidarPoseInformation().array() ==
      lidar_pose_information.array()).all() ||
    !sameAdaptiveCameraObservations(
      capability->observations(), observations) ||
    !sameAdaptiveVisualPolicyPlan(
      capability->decisionPlan(), plan))
  {
    return std::nullopt;
  }
  return capability;
}

void VIOManager::commitAdaptiveHealthObservation(
  JointUpdateResult & update_result)
{
  if (
    !adaptive_policy_config.enabled ||
    !update_result.adaptive_plan.adaptive_enabled)
  {
    return;
  }

  // Health hysteresis consumes only the read-only pre-update scout. It is
  // independent from state/map acceptance, which prevents isolation and
  // recovery counters from deadlocking on a rejected joint update.
  if (
    update_result.validated_adaptive_plan_capability.has_value())
  {
    update_result.adaptive_policy_committed =
      adaptive_visual_policy_->commitHealthObservation(
      *update_result.validated_adaptive_plan_capability,
      update_result.adaptive_plan.
      lidar_information_valid);
    return;
  }

  const bool sealed_adaptive_runtime =
    adaptive_policy_config.minimum_selected_cameras == 1U &&
    adaptive_policy_config.maximum_selected_cameras == 1U;
  if (sealed_adaptive_runtime)
  {
    // A sealed runtime may only commit the capability consumed by the
    // selection receipt. Never promote a mutable legacy plan here.
    update_result.adaptive_legacy_fallback_used = false;
    update_result.adaptive_policy_committed = false;
    return;
  }

  update_result.adaptive_legacy_fallback_used = false;
  update_result.adaptive_policy_committed =
    adaptive_visual_policy_->commitHealthObservation(
      update_result.adaptive_plan,
      update_result.adaptive_plan.
      lidar_information_valid);
}

std::optional<VIOManager::JointUpdateResult>
VIOManager::executeAuthorizedAdaptiveMapBootstrap(
  const CameraObservationBatch & batch,
  std::vector<JointCameraContext> & contexts,
  vector<pointWithVar> & pg,
  const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &
    feat_map,
  const double img_time,
  const livo::AdaptiveVisualPolicy::Matrix6 &
    lidar_pose_information,
  const bool lidar_pose_information_valid,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    runtime_bundle,
  const bool runtime_observation_authority_ready)
{
  if (
    !adaptive_policy_config.enabled ||
    std::none_of(
      contexts.begin(), contexts.end(),
      [this](const JointCameraContext & context)
      {
        return
          adaptive_visual_map_lifecycle_[context.camera_id] ==
          livo::AdaptiveVisualMapLifecycle::kEmpty;
      }))
  {
    return std::nullopt;
  }

  const double policy_start = omp_get_wtime();
  const livo::AdaptiveCameraObservations observations =
    buildAdaptiveObservations(
      batch, contexts, true, &lidar_pose_information,
      lidar_pose_information_valid);
  livo::AdaptiveVisualPolicyPlan plan =
    adaptive_visual_policy_->plan(
    lidar_pose_information, observations,
    lidar_pose_information_valid);
  auto runtime_plan_capability =
    issueValidatedAdaptiveRuntimePlan(
    batch, runtime_bundle,
    runtime_observation_authority_ready,
    lidar_pose_information,
    lidar_pose_information_valid, observations, plan);
  if (!runtime_plan_capability.has_value())
  {
    auto candidate_plan =
      adaptive_visual_policy_->issueCandidatePlan(
      batch, lidar_pose_information, observations,
      lidar_pose_information_valid);
    if (
      !candidate_plan.has_value() ||
      !adaptive_visual_policy_->accepts(*candidate_plan))
    {
      return std::nullopt;
    }
    plan = candidate_plan->decisionPlan();
  }
  if (plan.reason != livo::AdaptivePolicyReason::kReady)
  {
    return std::nullopt;
  }

  bool maintenance_performed = false;
  for (JointCameraContext & context : contexts)
  {
    const livo::AdaptiveCameraDecision & decision =
      plan.cameras[context.camera_id];
    if (
      !decision.enabled ||
      adaptive_visual_map_lifecycle_[context.camera_id] !=
        livo::AdaptiveVisualMapLifecycle::kEmpty)
    {
      continue;
    }
    const bool image_seed_eligible =
      context.saturation_fraction <=
        adaptive_policy_config.maximum_saturation_fraction &&
      context.dark_fraction <=
        adaptive_policy_config.maximum_dark_fraction &&
      context.mean_gradient >=
        0.5 * adaptive_policy_config.healthy_mean_gradient;
    const livo::AdaptiveVisualMapMaintenanceDecision
      map_maintenance =
      livo::planAdaptiveLioAnchoredMapMaintenance(
      livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
        kBootstrap,
      livo::AdaptiveVisualMapLifecycle::kEmpty,
      true, decision.isolated, image_seed_eligible,
      plan.lidar_information_valid);
    if (!map_maintenance.seed_new_points)
    {
      continue;
    }
    finalizeJointCamera(
      context, pg, feat_map, img_time,
      context.timestamp - batch.timestamp,
      map_maintenance);
    maintenance_performed = true;
    if (!camera_feature_maps_[context.camera_id].empty())
    {
      adaptive_visual_map_lifecycle_[context.camera_id] =
        livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated;
    }
  }
  if (!maintenance_performed)
  {
    return std::nullopt;
  }

  JointUpdateResult result;
  result.reason = "lio_anchored_map_maintenance";
  result.adaptive_plan = plan;
  result.adaptive_observations = observations;
  result.validated_adaptive_plan_capability =
    std::move(runtime_plan_capability);
  result.adaptive_policy_time_sec =
    omp_get_wtime() - policy_start;
  return result;
}

void VIOManager::processFrames(
  CameraObservationBatch &batch, vector<pointWithVar> &pg,
  const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
  const double img_time,
  const livo::AdaptiveVisualPolicy::Matrix6 & lidar_pose_information,
  const bool lidar_pose_information_valid,
  const std::uint64_t lidar_feature_generation,
  const livo::ActivePlaneIdentitySnapshot *
    active_plane_snapshot,
  const livo::CameraBatchTimeTransferBundle *
    time_transfer_bundle,
  const livo::ReferenceOnlyMigrationRuntimeBundle *
    migration_runtime_bundle,
  const livo::AdaptiveVisualRuntimeAuthorityInput *
    adaptive_runtime_input)
{
  const double start = omp_get_wtime();
  const bool adaptive_single_camera_runtime =
    adaptive_policy_config.enabled &&
    adaptive_policy_config.minimum_selected_cameras == 1U &&
    adaptive_policy_config.maximum_selected_cameras == 1U;
  if (adaptive_single_camera_runtime)
  {
    AdaptiveVisualRuntimePreflightResult preflight =
      preflightAdaptiveSingleCameraRuntime(
      batch, lidar_pose_information,
      lidar_pose_information_valid,
      lidar_feature_generation, adaptive_runtime_input);
    if (!preflight.ready)
    {
      adaptive_visual_runtime_audit_ = preflight.audit;
      return;
    }

    adaptive_visual_runtime_audit_ =
      processAdaptiveSingleCameraRuntime(
      batch, pg, feat_map, img_time, start,
      *adaptive_runtime_input, std::move(*preflight.permit),
      std::move(preflight.audit));
    return;
  }

  std::vector<JointCameraContext> contexts;
  contexts.reserve(kCameraCount);
  for (std::size_t camera_id = 0U;
       camera_id < batch.observations.size(); ++camera_id)
  {
    if (!batch.present[camera_id])
    {
      continue;
    }
    CameraObservation & observation =
      batch.observations[camera_id];
    JointCameraContext context;
    if (prepareJointCamera(observation, pg, feat_map, context))
    {
      contexts.push_back(std::move(context));
    }
  }

  adaptive_visual_runtime_audit_ = {};
  adaptive_visual_runtime_audit_.outcome =
    livo::AdaptiveVisualRuntimeOutcome::kNotApplicable;
  SharedLandmarkShadowMetrics shadow_metrics;
  const bool runtime_authority_required =
    shared_landmark_shadow_session_.audit().
    observation_enabled ||
    adaptive_policy_config.enabled ||
    multi_camera_landmark_linearization_shadow_config_.
      enabled ||
    (
      reference_only_migration_shadow_runtime_init_audit_.effective &&
      migration_runtime_bundle != nullptr);
  if (runtime_authority_required)
  {
    shadow_metrics.runtime_observation_authority =
      validateRuntimeObservationAuthority(
      batch, migration_runtime_bundle);
  }
  const livo::CameraBatchTimeTransferBundle *
    effective_time_transfer_bundle = time_transfer_bundle;
  if (
    shadow_metrics.runtime_observation_authority ==
      livo::VIORuntimeObservationAuthorityStatus::
      kReady &&
    migration_runtime_bundle != nullptr)
  {
    // A validated runtime publication is the sole time-transfer authority
    // for this batch. Never combine its provenance with a caller-supplied
    // context transfer from another publication.
    effective_time_transfer_bundle =
      &migration_runtime_bundle->timeTransferBundle();
  }
  for (JointCameraContext & context : contexts)
  {
    if (
      effective_time_transfer_bundle != nullptr &&
      context.camera_id < kCameraCount &&
      effective_time_transfer_bundle->
      present[context.camera_id] &&
      effective_time_transfer_bundle->transfers[
        context.camera_id].has_value())
    {
      context.time_transfer_entry =
        effective_time_transfer_bundle->transfers[
        context.camera_id];
    }
    if (context.frame != nullptr)
    {
      context.frame->camera_id_ = context.camera_id;
      context.frame->exposure_timestamp_sec_ =
        context.timestamp;
      context.frame->photometric_status_ =
        context.photometric_status;
      if (context.time_transfer_entry.has_value())
      {
        const livo::CameraPoseTransferRaw & pose_transfer =
          context.time_transfer_entry->value().raw();
        context.frame->time_status_ =
          pose_transfer.status;
        context.frame->rotation_world_from_imu_ =
          pose_transfer.rotation_world_from_imu;
        context.frame->position_world_from_imu_ =
          pose_transfer.position_world_m;
      }
    }
  }
  refreshSharedLandmarkRevisionSet(
    contexts, lidar_feature_generation,
    effective_time_transfer_bundle, shadow_metrics);
  const auto reference_only_migration_seeds =
    bootstrapSharedLandmarksShadow(
    batch, contexts, pg, lidar_feature_generation,
    active_plane_snapshot,
    effective_time_transfer_bundle,
    migration_runtime_bundle,
    shadow_metrics);
  if (reference_only_migration_seeds.has_value())
  {
    runReferenceOnlyMigrationShadowRuntime(
      batch, contexts, *reference_only_migration_seeds,
      batch.timestamp,
      lidar_feature_generation,
      migration_runtime_bundle);
  }

  JointUpdateResult update_result;
  auto lio_anchored_maintenance =
    executeAuthorizedAdaptiveMapBootstrap(
    batch, contexts, pg, feat_map, img_time,
    lidar_pose_information, lidar_pose_information_valid,
    migration_runtime_bundle,
    shadow_metrics.runtime_observation_authority ==
      livo::VIORuntimeObservationAuthorityStatus::kReady);
  const bool lio_anchored_maintenance_performed =
    lio_anchored_maintenance.has_value();
  if (lio_anchored_maintenance_performed)
  {
    update_result = std::move(*lio_anchored_maintenance);
  }

  bool solver_called = false;
  if (
    !lio_anchored_maintenance_performed &&
    (
      contexts.size() == kCameraCount ||
      isAdaptiveSingleCameraBatch(
        adaptive_policy_config, batch, contexts.size())))
  {
    solver_called = true;
    update_result = computeJointJacobianAndUpdateEKF(
      contexts, lidar_pose_information,
      lidar_pose_information_valid, batch,
      migration_runtime_bundle,
      shadow_metrics.runtime_observation_authority ==
        livo::VIORuntimeObservationAuthorityStatus::
        kReady,
      nullptr);
  }
  else if (!lio_anchored_maintenance_performed)
  {
    update_result.reason = "incomplete_camera_batch";
  }
  if (solver_called || lio_anchored_maintenance_performed)
  {
    runMultiCameraLandmarkLinearizationShadow(
      batch, migration_runtime_bundle,
      shadow_metrics.runtime_observation_authority,
      update_result, img_time, lidar_feature_generation);
  }
  if (solver_called)
  {
    last_joint_solver_audit_.accepted =
      update_result.accepted;
    last_joint_solver_audit_.reason =
      update_result.reason;
    last_joint_solver_audit_.contributing_cameras =
      update_result.contributing_cameras;
    last_joint_solver_audit_.tracked_points =
      update_result.tracked_points;
    if (update_result.accepted)
    {
      shared_landmark_shadow_session_.
        advanceVisualStateRevision();
    }
  }

  commitAdaptiveHealthObservation(update_result);

  if (solver_called)
  {
    mirrorSharedLandmarkObservationsShadow(
      batch, contexts, update_result,
      lidar_feature_generation,
      active_plane_snapshot,
      effective_time_transfer_bundle,
      migration_runtime_bundle,
      shadow_metrics);
    for (JointCameraContext &context : contexts)
    {
      const livo::AdaptiveCameraDecision & decision =
        update_result.adaptive_plan.cameras[context.camera_id];
      const bool camera_update_accepted =
        update_result.accepted &&
        (!adaptive_policy_config.enabled ||
         decision.enabled);
      const livo::AdaptiveVisualMapMaintenanceDecision map_maintenance =
        livo::planAdaptiveVisualSolverMapCommit(
          adaptive_policy_config.enabled,
          decision.enabled,
          camera_update_accepted);
      finalizeJointCamera(
        context, pg, feat_map, img_time,
        context.timestamp - batch.timestamp,
        map_maintenance);
      if (adaptive_policy_config.enabled &&
          camera_update_accepted)
      {
        adaptive_visual_map_lifecycle_[context.camera_id] =
          livo::AdaptiveVisualMapLifecycle::kInitialized;
      }
    }
  }
  refreshSharedLandmarkLifecycleMetrics(shadow_metrics);
  if (shared_landmark_shadow_session_.active())
  {
    emitSharedLandmarkShadowMetric(
      img_time, lidar_feature_generation,
      shadow_metrics);
  }

  ++frame_count;
  const double elapsed = omp_get_wtime() - start;
  ave_total =
    ave_total * static_cast<double>(frame_count - 1) /
    static_cast<double>(frame_count) +
    elapsed / static_cast<double>(frame_count);

  if (adaptive_policy_config.enabled &&
      update_result.adaptive_plan.adaptive_enabled)
  {
    const auto & plan = update_result.adaptive_plan;
    livo::emitStructuredMetric(
      "[ADAPTIVE_VIO_LIDAR_METRIC] t=%.9f generation=%llu valid=%d "
      "weak_dirs=%zu eigenvalues=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g "
      "policy_reason=%s policy_s=%.9f\n",
      img_time,
      static_cast<unsigned long long>(lidar_feature_generation),
      plan.lidar_information_valid ? 1 : 0,
      plan.weak_direction_count,
      plan.lidar_eigenvalues[0], plan.lidar_eigenvalues[1],
      plan.lidar_eigenvalues[2], plan.lidar_eigenvalues[3],
      plan.lidar_eigenvalues[4], plan.lidar_eigenvalues[5],
      livo::adaptivePolicyReasonName(plan.reason),
      update_result.adaptive_policy_time_sec);

    for (const JointCameraContext & context : contexts)
    {
      const std::size_t camera_id = context.camera_id;
      const livo::AdaptiveCameraObservation & observation =
        update_result.adaptive_observations[camera_id];
      const livo::AdaptiveCameraDecision & decision =
        plan.cameras[camera_id];
      emitAdaptiveCameraHealthMetric(
        img_time,
        batch.observations[camera_id].timestamp,
        batch.timestamp,
        lidar_feature_generation, camera_id,
        cameraName(static_cast<std::uint8_t>(camera_id)),
        observation, decision, decision.frame_received,
        decision.eligible, decision.enabled,
        decision.reject_reason, adaptive_policy_config);

      if (update_result.adaptive_policy_committed &&
          decision.transition !=
          livo::AdaptiveCameraTransition::kNone)
      {
        const bool isolated =
          decision.transition ==
          livo::AdaptiveCameraTransition::kIsolated;
        livo::emitStructuredMetric(
          "[ADAPTIVE_VIO_EVENT] t=%.9f generation=%llu camera_id=%zu "
          "from=%s to=%s reason=%s\n",
          img_time,
          static_cast<unsigned long long>(lidar_feature_generation),
          camera_id,
          isolated ? "active" : "isolated",
          isolated ? "isolated" : "active",
          isolated ? "health_below_off" : "health_above_on");
      }
    }
    const double visual_update_time = std::max(
      0.0,
      update_result.visual_update_time_sec -
      update_result.adaptive_policy_time_sec);
    livo::emitStructuredMetric(
      "[ADAPTIVE_VIO_UPDATE_METRIC] t=%.9f generation=%llu "
      "accepted=%d reason=%s tracked=%zu contributing=%zu "
      "rotation_deg=%.6f translation_m=%.6f "
      "policy_s=%.9f visual_s=%.9f total_s=%.9f "
      "plan_capability=%d plan_capability_validated=%d "
      "legacy_fallback=%d runtime_exclusive=0 "
      "health_committed=%d committed=%d\n",
      img_time,
      static_cast<unsigned long long>(lidar_feature_generation),
      update_result.accepted ? 1 : 0,
      update_result.reason.c_str(), update_result.tracked_points,
      update_result.contributing_cameras,
      update_result.rotation_increment_rad * 180.0 / M_PI,
      update_result.translation_increment_m,
      update_result.adaptive_policy_time_sec, visual_update_time,
      elapsed,
      update_result.validated_adaptive_plan_capability.has_value() ?
      1 : 0,
      update_result.validated_adaptive_plan_capability.has_value() ?
      1 : 0,
      update_result.adaptive_legacy_fallback_used ? 1 : 0,
      update_result.adaptive_policy_committed ? 1 : 0,
      update_result.adaptive_policy_committed ? 1 : 0);
  }

  livo::emitStructuredMetric(
    "[VIO_JOINT_METRIC] t=%.9f cameras=%zu contributing=%zu tracked=%zu "
    "residual_rmse=%.6f accepted=%d reason=%s rotation_deg=%.6f "
    "translation_m=%.6f skew_ms=%.3f total_s=%.6f generation=%llu\n",
    img_time, contexts.size(), update_result.contributing_cameras,
    update_result.tracked_points, update_result.residual_rmse,
    update_result.accepted ? 1 : 0, update_result.reason.c_str(),
    update_result.rotation_increment_rad * 180.0 / M_PI,
    update_result.translation_increment_m,
    batch.max_time_offset_sec * 1000.0, elapsed,
    static_cast<unsigned long long>(lidar_feature_generation));
}

void VIOManager::processFrame(
  CameraObservation &observation, vector<pointWithVar> &pg,
  const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
  const double img_time)
{
  if (!activateCamera(observation.camera_id))
  {
    printf("[ VIO ] Drop frame from unconfigured camera id %u\n",
           static_cast<unsigned int>(observation.camera_id));
    return;
  }

  cv::Mat &img = observation.image;
  if (width != img.cols || height != img.rows)
  {
    if (img.empty()) printf("[ VIO ] Empty Image!\n");
    cv::resize(img, img, cv::Size(width, height), 0, 0, CV_INTER_LINEAR);
  }
  img_rgb = img.clone();
  img_cp = img.clone();
  // img_test = img.clone();

  if (img.channels() == 3) cv::cvtColor(img, img, CV_BGR2GRAY);

  new_frame_.reset(new Frame(cam, img));
  updateFrameState(*state);
  
  resetGrid();

  double t1 = omp_get_wtime();

  retrieveFromVisualSparseMap(img, pg, feat_map);

  double t2 = omp_get_wtime();

  computeJacobianAndUpdateEKF(img);

  double t3 = omp_get_wtime();

  generateVisualMapPoints(img, pg);
  if (
    observation.camera_id < kCameraCount &&
    adaptive_visual_map_lifecycle_[observation.camera_id] ==
      livo::AdaptiveVisualMapLifecycle::kEmpty &&
    !camera_feature_maps_[observation.camera_id].empty())
  {
    adaptive_visual_map_lifecycle_[observation.camera_id] =
      livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated;
  }

  double t4 = omp_get_wtime();
  
  plotTrackedPoints();

  if (plot_flag) projectPatchFromRefToCur(feat_map);

  double t5 = omp_get_wtime();

  updateVisualMapPoints(img);

  double t6 = omp_get_wtime();

  updateReferencePatch(feat_map);

  double t7 = omp_get_wtime();
  
  if(colmap_output_en)  dumpDataForColmap();

  frame_count++;
  ave_total = ave_total * (frame_count - 1) / frame_count + (t7 - t1 - (t5 - t4)) / frame_count;

  double squared_residual_sum = 0.0;
  for (const float error : visual_submap->errors)
  {
    squared_residual_sum += error;
  }
  const double residual_rmse =
    visual_submap->errors.empty() ? std::numeric_limits<double>::quiet_NaN() :
    std::sqrt(squared_residual_sum /
              (static_cast<double>(visual_submap->errors.size()) * patch_size_total));
  printf(
    "[VIO_METRIC] t=%.9f camera_id=%u camera=%s tracked=%d visual_voxels=%zu "
    "residual_rmse=%.6f total_s=%.6f\n",
    img_time, static_cast<unsigned int>(active_camera_id_),
    cameraName(active_camera_id_).c_str(), total_points,
    active_visual_feature_map_->size(),
    residual_rmse, t7 - t1 - (t5 - t4));

  // printf("[ VIO ] feat_map.size(): %zu\n", feat_map.size());
  // printf("\033[1;32m[ VIO time ]: current frame: retrieveFromVisualSparseMap time: %.6lf secs.\033[0m\n", t2 - t1);
  // printf("\033[1;32m[ VIO time ]: current frame: computeJacobianAndUpdateEKF time: %.6lf secs, comp H: %.6lf secs, ekf: %.6lf secs.\033[0m\n", t3 - t2, computeH, ekf_time);
  // printf("\033[1;32m[ VIO time ]: current frame: generateVisualMapPoints time: %.6lf secs.\033[0m\n", t4 - t3);
  // printf("\033[1;32m[ VIO time ]: current frame: updateVisualMapPoints time: %.6lf secs.\033[0m\n", t6 - t5);
  // printf("\033[1;32m[ VIO time ]: current frame: updateReferencePatch time: %.6lf secs.\033[0m\n", t7 - t6);
  // printf("\033[1;32m[ VIO time ]: current total time: %.6lf, average total time: %.6lf secs.\033[0m\n", t7 - t1 - (t5 - t4), ave_total);

  // ave_build_residual_time = ave_build_residual_time * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;
  // ave_ekf_time = ave_ekf_time * (frame_count - 1) / frame_count + (t3 - t2) / frame_count;
 
  // cout << BLUE << "ave_build_residual_time: " << ave_build_residual_time << RESET << endl;
  // cout << BLUE << "ave_ekf_time: " << ave_ekf_time << RESET << endl;
  
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         VIO Time                            |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27zu |\033[0m\n", "Sparse Map Size", feat_map.size());
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "retrieveFromVisualSparseMap", t2 - t1);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "computeJacobianAndUpdateEKF", t3 - t2);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> computeJacobian", compute_jacobian_time);
  printf("\033[1;32m| %-27s   | %-27lf |\033[0m\n", "-> updateEKF", update_ekf_time);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "generateVisualMapPoints", t4 - t3);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateVisualMapPoints", t6 - t5);
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "updateReferencePatch", t7 - t6);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Current Total Time", t7 - t1 - (t5 - t4));
  printf("\033[1;32m| %-29s | %-27lf |\033[0m\n", "Average Total Time", ave_total);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

  // std::string text = std::to_string(int(1 / (t7 - t1 - (t5 - t4)))) + " HZ";
  // cv::Point2f origin;
  // origin.x = 20;
  // origin.y = 20;
  // cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
}
