#include "reference_only_migration_shadow_batch_runner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include <limits>
#include <utility>

namespace livo
{
namespace
{

std::optional<std::size_t> directedPairIndex(
  const std::uint8_t source_camera_id,
  const std::uint8_t target_camera_id)
{
  if (
    source_camera_id >= 4U ||
    target_camera_id >= 4U ||
    source_camera_id == target_camera_id)
  {
    return std::nullopt;
  }
  const std::size_t target_offset =
    target_camera_id < source_camera_id ?
    static_cast<std::size_t>(target_camera_id) :
    static_cast<std::size_t>(target_camera_id - 1U);
  return
    static_cast<std::size_t>(source_camera_id) * 3U +
    target_offset;
}

std::uint64_t normalizedDoubleOrderKey(const double value)
{
  if (std::isnan(value))
  {
    return std::numeric_limits<std::uint64_t>::max();
  }
  const double normalized = value == 0.0 ? 0.0 : value;
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(normalized));
  std::memcpy(&bits, &normalized, sizeof(bits));
  constexpr std::uint64_t kSign =
    std::uint64_t{1U} << 63U;
  return
    (bits & kSign) != 0U ?
    ~bits : bits | kSign;
}

bool candidateLess(
  const ReferenceOnlyMigrationShadowBatchCandidate & left,
  const ReferenceOnlyMigrationShadowBatchCandidate & right)
{
  if (left.supportKey() < right.supportKey())
  {
    return true;
  }
  if (right.supportKey() < left.supportKey())
  {
    return false;
  }
  if (left.geometryRevision() != right.geometryRevision())
  {
    return left.geometryRevision() > right.geometryRevision();
  }

  const std::array<std::uint64_t, 5U> left_values{{
    normalizedDoubleOrderKey(left.sourcePixel().y()),
    normalizedDoubleOrderKey(left.sourcePixel().x()),
    normalizedDoubleOrderKey(left.planePointWorld().x()),
    normalizedDoubleOrderKey(left.planePointWorld().y()),
    normalizedDoubleOrderKey(left.planePointWorld().z())}};
  const std::array<std::uint64_t, 5U> right_values{{
    normalizedDoubleOrderKey(right.sourcePixel().y()),
    normalizedDoubleOrderKey(right.sourcePixel().x()),
    normalizedDoubleOrderKey(right.planePointWorld().x()),
    normalizedDoubleOrderKey(right.planePointWorld().y()),
    normalizedDoubleOrderKey(right.planePointWorld().z())}};
  return left_values < right_values;
}

bool sameGeometryVersion(
  const GeometryVersion & left,
  const GeometryVersion & right)
{
  return
    left.plane == right.plane &&
    left.revision == right.revision;
}

bool sameEndpoint(
  const MigrationEndpoint & left,
  const MigrationEndpoint & right)
{
  const bool same_camera_model =
    (!left.camera_model && !right.camera_model) ||
    left.sharesCameraModelIdentityWith(right);
  return
    left.camera_id == right.camera_id &&
    same_camera_model &&
    left.camera_model_revision ==
      right.camera_model_revision &&
    left.rig_revision == right.rig_revision &&
    left.rotation_imu_from_camera.isApprox(
      right.rotation_imu_from_camera, 0.0) &&
    left.position_imu_from_camera.isApprox(
      right.position_imu_from_camera, 0.0);
}

bool sameTransferRaw(
  const CameraPoseTransferRaw & left,
  const CameraPoseTransferRaw & right)
{
  return
    left.status == right.status &&
    left.anchor_timestamp_sec == right.anchor_timestamp_sec &&
    left.target_timestamp_sec == right.target_timestamp_sec &&
    left.delta_time_sec == right.delta_time_sec &&
    left.imu_epoch == right.imu_epoch &&
    left.time_uncertainty_stddev_sec ==
      right.time_uncertainty_stddev_sec &&
    left.maximum_imu_gap_sec == right.maximum_imu_gap_sec &&
    left.rotation_world_from_imu.isApprox(
      right.rotation_world_from_imu, 0.0) &&
    left.position_world_m.isApprox(
      right.position_world_m, 0.0) &&
    left.velocity_world_mps.isApprox(
      right.velocity_world_mps, 0.0) &&
    left.error_state_transition.isApprox(
      right.error_state_transition, 0.0) &&
    left.target_pose_wrt_anchor_state.isApprox(
      right.target_pose_wrt_anchor_state, 0.0) &&
    left.target_pose_wrt_active_anchor_state.isApprox(
      right.target_pose_wrt_active_anchor_state, 0.0) &&
    left.process_noise_covariance.isApprox(
      right.process_noise_covariance, 0.0) &&
    left.pose_transfer_covariance.isApprox(
      right.pose_transfer_covariance, 0.0) &&
    left.time_alignment_pose_covariance.isApprox(
      right.time_alignment_pose_covariance, 0.0) &&
    left.conditional_pose_only.target_pose_wrt_anchor_pose.
      isApprox(
      right.conditional_pose_only.
      target_pose_wrt_anchor_pose,
      0.0);
}

bool sameTargetTime(
  const ReferenceOnlyMigrationTargetTime & left,
  const ReferenceOnlyMigrationTargetTime & right)
{
  if (
    left.receipt.valid() != right.receipt.valid() ||
    left.receipt.accepts(left.entry) !=
      right.receipt.accepts(right.entry) ||
    left.receipt.batchTimestampSec() !=
      right.receipt.batchTimestampSec() ||
    left.receipt.anchorTimestampSec() !=
      right.receipt.anchorTimestampSec() ||
    left.receipt.imuEpoch() != right.receipt.imuEpoch() ||
    left.receipt.activeStateJacobianRevision() !=
      right.receipt.activeStateJacobianRevision() ||
    left.receipt.activeStateLayoutId() !=
      right.receipt.activeStateLayoutId() ||
    left.entry.has_value() != right.entry.has_value() ||
    left.entry.cameraId() != right.entry.cameraId() ||
    left.entry.batchTimestampSec() !=
      right.entry.batchTimestampSec() ||
    left.entry.anchorTimestampSec() !=
      right.entry.anchorTimestampSec() ||
    left.entry.targetTimestampSec() !=
      right.entry.targetTimestampSec() ||
    left.entry.imuEpoch() != right.entry.imuEpoch() ||
    left.entry.activeStateJacobianRevision() !=
      right.entry.activeStateJacobianRevision() ||
    left.entry.activeStateLayoutId() !=
      right.entry.activeStateLayoutId())
  {
    return false;
  }
  return
    !left.entry.has_value() ||
    sameTransferRaw(
      left.entry.value().raw(),
      right.entry.value().raw());
}

bool sameOptionalBounds(
  const std::optional<
    ReferenceOnlyMigrationTargetImageBounds> & left,
  const std::optional<
    ReferenceOnlyMigrationTargetImageBounds> & right)
{
  return
    left.has_value() == right.has_value() &&
    (!left ||
    (left->width == right->width &&
    left->height == right->height));
}

bool evaluatorInputsEquivalent(
  const ReferenceOnlyMigrationShadowBatchCandidate & left,
  const ReferenceOnlyMigrationShadowBatchCandidate & right)
{
  const auto & left_request = left.request();
  const auto & right_request = right.request();
  return
    left_request.source_support_key ==
      right_request.source_support_key &&
    sameGeometryVersion(
      left_request.source_geometry_version,
      right_request.source_geometry_version) &&
    sameEndpoint(
      left_request.source_endpoint,
      right_request.source_endpoint) &&
    left_request.source_pose.exposure_timestamp_sec ==
      right_request.source_pose.exposure_timestamp_sec &&
    left_request.source_pose.rotation_world_from_imu.isApprox(
      right_request.source_pose.rotation_world_from_imu,
      0.0) &&
    left_request.source_pose.position_world_m.isApprox(
      right_request.source_pose.position_world_m, 0.0) &&
    left_request.source_pose.pose_covariance.isApprox(
      right_request.source_pose.pose_covariance, 0.0) &&
    left_request.patch.source_pixel.isApprox(
      right_request.patch.source_pixel, 0.0) &&
    left_request.patch.source_level ==
      right_request.patch.source_level &&
    left_request.patch.patch_width ==
      right_request.patch.patch_width &&
    sameEndpoint(
      left_request.target_endpoint,
      right_request.target_endpoint) &&
    sameTargetTime(
      left_request.target_time,
      right_request.target_time) &&
    sameOptionalBounds(
      left_request.target_image_bounds,
      right_request.target_image_bounds) &&
    left_request.plane.support_key ==
      right_request.plane.support_key &&
    sameGeometryVersion(
      left_request.plane.geometry_version,
      right_request.plane.geometry_version) &&
    left_request.plane.point_world_m.isApprox(
      right_request.plane.point_world_m, 0.0) &&
    left_request.plane.point_covariance_m2.isApprox(
      right_request.plane.point_covariance_m2, 0.0) &&
    left_request.plane.normal_world.isApprox(
      right_request.plane.normal_world, 0.0) &&
    left_request.plane.normal_covariance.isApprox(
      right_request.plane.normal_covariance, 0.0);
}

bool orderEquivalent(
  const ReferenceOnlyMigrationShadowBatchCandidate & left,
  const ReferenceOnlyMigrationShadowBatchCandidate & right)
{
  return
    !candidateLess(left, right) &&
    !candidateLess(right, left);
}

}  // namespace

ReferenceOnlyMigrationShadowBatchResult
runReferenceOnlyMigrationShadowBatch(
  const std::uint64_t lidar_feature_generation,
  std::vector<ReferenceOnlyMigrationShadowBatchCandidate> candidates)
{
  ReferenceOnlyMigrationShadowBatchResult result;
  result.input_candidate_count_ = candidates.size();
  if (!candidates.empty())
  {
    result.reason_ =
      ReferenceOnlyMigrationShadowBatchReason::
      kNoEligibleDirectedPair;
  }
  std::array<
    std::vector<ReferenceOnlyMigrationShadowBatchCandidate>,
    kReferenceOnlyMigrationShadowDirectedPairCount> buckets;
  for (auto & candidate : candidates)
  {
    const std::optional<std::size_t> pair_index =
      directedPairIndex(
      candidate.sourceCameraId(), candidate.targetCameraId());
    if (!pair_index)
    {
      continue;
    }
    buckets[*pair_index].push_back(std::move(candidate));
  }

  std::vector<double> ready_warp_conditions;
  ready_warp_conditions.reserve(
    kReferenceOnlyMigrationShadowTotalBudget);
  const auto evaluate_candidate =
    [&result, &ready_warp_conditions](
      const ReferenceOnlyMigrationShadowBatchCandidate & candidate,
      const std::size_t pair_index)
    {
      ++result.attempted_count_;
      ++result.pair_attempt_counts_[pair_index];
      const ReferenceOnlyMigrationShadowResult evaluation =
        evaluateReferenceOnlyMigrationShadow(candidate.request());
      const MigrationStatus migration_status =
        evaluation.migrationStatus();
      const std::size_t status_index =
        static_cast<std::size_t>(migration_status);
      if (status_index < result.migration_status_counts_.size())
      {
        ++result.migration_status_counts_[status_index];
      }

      ReferenceOnlyMigrationShadowBatchSelection selection;
      selection.source_camera_id = candidate.sourceCameraId();
      selection.target_camera_id = candidate.targetCameraId();
      selection.support_key = candidate.supportKey();
      selection.geometry_version =
        candidate.request().source_geometry_version;
      selection.source_pixel = candidate.sourcePixel();
      selection.plane_point_world_m =
        candidate.planePointWorld();
      selection.migration_status = migration_status;
      result.selected_sequence_.push_back(
        std::move(selection));

      if (
        evaluation.status() ==
          ReferenceOnlyMigrationShadowStatus::
          kGeometryReadyVisibilityUnverified &&
        migration_status == MigrationStatus::kReady)
      {
        ++result.ready_count_;
        ++result.pair_ready_counts_[pair_index];
        if (std::isfinite(evaluation.warpConditionNumber()))
        {
          ready_warp_conditions.push_back(
            evaluation.warpConditionNumber());
        }
      }
    };

  for (std::size_t pair_index = 0U;
       pair_index < buckets.size(); ++pair_index)
  {
    auto & bucket = buckets[pair_index];
    std::sort(bucket.begin(), bucket.end(), candidateLess);
    std::vector<ReferenceOnlyMigrationShadowBatchCandidate>
      unique_candidates;
    unique_candidates.reserve(bucket.size());
    std::size_t support_begin = 0U;
    while (support_begin < bucket.size())
    {
      std::size_t support_end = support_begin + 1U;
      while (
        support_end < bucket.size() &&
        bucket[support_begin].supportKey() ==
        bucket[support_end].supportKey())
      {
        ++support_end;
      }
      result.duplicate_candidate_count_ +=
        support_end - support_begin - 1U;

      // A public-order tie with different evaluator inputs has no canonical
      // winner, so quarantine the whole support instead of depending on the
      // unspecified order of equivalent std::sort elements.
      bool ambiguous = false;
      for (std::size_t index = support_begin + 1U;
           index < support_end; ++index)
      {
        if (
          orderEquivalent(bucket[index - 1U], bucket[index]) &&
          !evaluatorInputsEquivalent(
            bucket[index - 1U], bucket[index]))
        {
          ambiguous = true;
          break;
        }
      }
      if (ambiguous)
      {
        ++result.ambiguous_duplicate_count_;
      }
      else
      {
        unique_candidates.push_back(
          std::move(bucket[support_begin]));
      }
      support_begin = support_end;
    }
    bucket = std::move(unique_candidates);
    result.unique_candidate_count_ += bucket.size();
    const std::size_t retained = std::min(
      bucket.size(),
      kReferenceOnlyMigrationShadowPerPairBudget);
    result.retained_candidate_count_ += retained;
    result.pair_budget_truncated_count_ +=
      bucket.size() - retained;
    bucket.erase(
      bucket.begin() + static_cast<std::ptrdiff_t>(retained),
      bucket.end());
  }

  // Rotate the directed-pair origin while keeping rank-first allocation, so
  // the eight remaining third-rank slots are fair across LiDAR generations.
  const std::size_t first_pair =
    static_cast<std::size_t>(
    lidar_feature_generation %
    kReferenceOnlyMigrationShadowDirectedPairCount);
  bool total_budget_reached = false;
  for (std::size_t rank = 0U;
       rank < kReferenceOnlyMigrationShadowPerPairBudget &&
       !total_budget_reached;
       ++rank)
  {
    for (std::size_t offset = 0U;
         offset <
         kReferenceOnlyMigrationShadowDirectedPairCount;
         ++offset)
    {
      if (
        result.attempted_count_ >=
        kReferenceOnlyMigrationShadowTotalBudget)
      {
        total_budget_reached = true;
        break;
      }
      const std::size_t pair_index =
        (first_pair + offset) %
        kReferenceOnlyMigrationShadowDirectedPairCount;
      const auto & bucket = buckets[pair_index];
      if (
        rank < bucket.size() &&
        rank < kReferenceOnlyMigrationShadowPerPairBudget)
      {
        evaluate_candidate(bucket[rank], pair_index);
      }
    }
  }
  result.total_budget_truncated_count_ =
    result.retained_candidate_count_ -
    result.attempted_count_;
  if (!ready_warp_conditions.empty())
  {
    std::sort(
      ready_warp_conditions.begin(),
      ready_warp_conditions.end());
    const std::size_t p50_index =
      (ready_warp_conditions.size() - 1U) / 2U;
    const std::size_t p95_rank =
      (95U * ready_warp_conditions.size() + 99U) / 100U;
    result.ready_warp_condition_p50_ =
      ready_warp_conditions[p50_index];
    result.ready_warp_condition_p95_ =
      ready_warp_conditions[p95_rank - 1U];
  }
  if (result.attempted_count_ != 0U)
  {
    result.reason_ =
      ReferenceOnlyMigrationShadowBatchReason::kEvaluated;
  }
  return result;
}

}  // namespace livo
