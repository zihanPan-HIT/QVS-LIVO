#include "reference_only_migration_shadow_batch_runner.h"
#include "camera_batch_time_transfer.h"
#include "pinhole_camera_projection_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Geometry>

#include <gtest/gtest.h>

namespace
{

constexpr std::size_t kCameraCount = 4U;

static_assert(
  livo::ReferenceOnlyMigrationShadowBatchResult::experimental());
static_assert(
  !livo::ReferenceOnlyMigrationShadowBatchResult::consumable());
static_assert(
  !livo::ReferenceOnlyMigrationShadowBatchResult::productionReady());
static_assert(
  !livo::ReferenceOnlyMigrationShadowBatchResult::solverApplied());
static_assert(
  !livo::ReferenceOnlyMigrationShadowBatchResult::
  acceptanceReceiptIssued());
static_assert(
  !livo::ReferenceOnlyMigrationShadowBatchResult::
  independenceProofIssued());

livo::CameraTimeTransferImuSample imuSample(
  const double timestamp_sec)
{
  livo::CameraTimeTransferImuSample sample;
  sample.timestamp_sec = timestamp_sec;
  sample.angular_velocity_radps = Eigen::Vector3d::Zero();
  sample.specific_force_mps2 = Eigen::Vector3d::Zero();
  return sample;
}

livo::CameraBatchTimeTransferBundle targetTimeBundle(
  const double batch_timestamp_sec)
{
  livo::CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.05;
  config.max_imu_gap_sec = 0.01;
  livo::CameraTimeTransfer transfer(config);
  for (int index = 0; index <= 8; ++index)
  {
    EXPECT_EQ(
      transfer.appendImu(
        imuSample(
          batch_timestamp_sec - 0.02 +
          0.005 * static_cast<double>(index))),
      livo::CameraTimeTransferAppendStatus::kAccepted);
  }

  CameraObservationBatch batch;
  batch.camera_count = kCameraCount;
  batch.partial = false;
  batch.timestamp = batch_timestamp_sec;
  for (std::uint8_t camera_id = 0U;
       camera_id < kCameraCount; ++camera_id)
  {
    batch.present[camera_id] = true;
    batch.observations[camera_id].camera_id = camera_id;
    batch.observations[camera_id].timestamp =
      batch_timestamp_sec;
    batch.observations[camera_id].image =
      cv::Mat::ones(2, 2, CV_8UC1);
  }

  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch_timestamp_sec;
  livo::CameraBatchTimeTransferBundle result =
    livo::buildCameraBatchTimeTransferBundle(
    batch, anchor, 0.0, transfer);
  EXPECT_TRUE(result.allPresentFullStateTransfersValid());
  return result;
}

std::shared_ptr<const livo::PinholeCameraProjectionAdapter>
cameraModel(const std::uint64_t revision)
{
  livo::PinholeCameraProjectionConfig config;
  config.model = "Pinhole";
  config.width = 640;
  config.height = 480;
  config.fx = 400.0;
  config.fy = 400.0;
  config.cx = 320.0;
  config.cy = 240.0;
  config.model_revision = revision;
  return std::make_shared<
    const livo::PinholeCameraProjectionAdapter>(config);
}

livo::MigrationEndpoint endpoint(
  const std::uint8_t camera_id)
{
  const std::uint64_t model_revision =
    101U + static_cast<std::uint64_t>(camera_id);
  livo::MigrationEndpoint result;
  result.camera_id = camera_id;
  result.camera_model = cameraModel(model_revision);
  result.camera_model_revision = model_revision;
  result.rig_revision = 77U;
  result.rotation_imu_from_camera =
    Eigen::Matrix3d::Identity();
  result.position_imu_from_camera =
    Eigen::Vector3d::Zero();
  return result;
}

livo::ReferenceOnlyMigrationShadowRequest validRequest(
  const livo::CameraBatchTimeTransferBundle & bundle,
  const std::uint8_t source_camera_id,
  const std::uint8_t target_camera_id,
  const std::uint64_t support_id)
{
  EXPECT_TRUE(bundle.validatedReceipt().has_value());
  EXPECT_TRUE(bundle.transfers[target_camera_id].has_value());
  livo::ReferenceOnlyMigrationShadowRequest request(
    livo::ReferenceOnlyMigrationTargetTime(
      *bundle.validatedReceipt(),
      bundle.transfers[target_camera_id]));

  request.source_support_key.plane = {support_id, 1U};
  request.source_support_key.cell = {
    static_cast<std::int32_t>(support_id), 0, 0U};
  request.source_geometry_version = {
    {support_id, 1U}, 5U};
  request.source_endpoint = endpoint(source_camera_id);
  request.source_pose.exposure_timestamp_sec =
    bundle.transfers[target_camera_id].targetTimestampSec() -
    0.01;
  request.source_pose.rotation_world_from_imu =
    Eigen::Matrix3d::Identity();
  request.source_pose.position_world_m =
    Eigen::Vector3d::Zero();
  request.source_pose.pose_covariance =
    Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-4;
  request.patch.source_pixel =
    Eigen::Vector2d(320.0, 240.0);
  request.patch.source_level = 0U;
  request.patch.patch_width = 8;
  request.target_endpoint = endpoint(target_camera_id);
  request.target_image_bounds =
    livo::ReferenceOnlyMigrationTargetImageBounds{640, 480};
  request.plane.support_key = request.source_support_key;
  request.plane.geometry_version =
    request.source_geometry_version;
  request.plane.point_world_m =
    Eigen::Vector3d(0.0, 0.0, 5.0);
  request.plane.point_covariance_m2 =
    Eigen::Matrix3d::Identity() * 1.0e-4;
  request.plane.normal_world = Eigen::Vector3d::UnitZ();
  request.plane.normal_covariance =
    Eigen::Matrix3d::Identity() * 1.0e-4;
  return request;
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  EmptyInputProducesNoAttempts)
{
  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    17U,
    std::vector<
      livo::ReferenceOnlyMigrationShadowBatchCandidate>{});

  EXPECT_EQ(
    result.reason(),
    livo::ReferenceOnlyMigrationShadowBatchReason::kEmptyInput);
  EXPECT_EQ(result.inputCandidateCount(), 0U);
  EXPECT_EQ(result.attemptedCount(), 0U);
  EXPECT_EQ(result.readyCount(), 0U);
  EXPECT_TRUE(result.selectedSequence().empty());
  EXPECT_FALSE(result.readyWarpConditionP50().has_value());
  EXPECT_FALSE(result.readyWarpConditionP95().has_value());
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  OneValidCandidateProducesReadyMetrics)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> candidates;
  candidates.emplace_back(
    validRequest(bundle, 0U, 1U, 9U));

  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    17U, std::move(candidates));

  EXPECT_EQ(
    result.reason(),
    livo::ReferenceOnlyMigrationShadowBatchReason::kEvaluated);
  EXPECT_EQ(result.inputCandidateCount(), 1U);
  EXPECT_EQ(result.uniqueCandidateCount(), 1U);
  EXPECT_EQ(result.retainedCandidateCount(), 1U);
  EXPECT_EQ(result.attemptedCount(), 1U);
  EXPECT_EQ(result.readyCount(), 1U);
  EXPECT_EQ(result.rejectedCount(), 0U);
  EXPECT_EQ(result.pairAttemptCounts()[0U], 1U);
  EXPECT_EQ(result.pairReadyCounts()[0U], 1U);
  EXPECT_EQ(
    result.statusCount(livo::MigrationStatus::kReady), 1U);
  ASSERT_EQ(result.selectedSequence().size(), 1U);
  EXPECT_EQ(result.selectedSequence()[0U].source_camera_id, 0U);
  EXPECT_EQ(result.selectedSequence()[0U].target_camera_id, 1U);
  EXPECT_EQ(result.selectedSequence()[0U].support_key.plane.id, 9U);
  EXPECT_TRUE(result.selectedSequence()[0U].source_pixel.isApprox(
    Eigen::Vector2d(320.0, 240.0), 0.0));
  EXPECT_EQ(
    result.selectedSequence()[0U].migration_status,
    livo::MigrationStatus::kReady);
  ASSERT_TRUE(result.readyWarpConditionP50().has_value());
  ASSERT_TRUE(result.readyWarpConditionP95().has_value());
  EXPECT_NEAR(*result.readyWarpConditionP50(), 1.0, 1.0e-12);
  EXPECT_NEAR(*result.readyWarpConditionP95(), 1.0, 1.0e-12);
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  DirectedPairBudgetRetainsOnlyFourCanonicalCandidates)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> candidates;
  for (std::uint64_t support_id = 8U;
       support_id >= 1U; --support_id)
  {
    candidates.emplace_back(
      validRequest(bundle, 0U, 1U, support_id));
  }

  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    0U, std::move(candidates));

  EXPECT_EQ(result.uniqueCandidateCount(), 8U);
  EXPECT_EQ(result.retainedCandidateCount(), 4U);
  EXPECT_EQ(result.pairBudgetTruncatedCount(), 4U);
  EXPECT_EQ(result.totalBudgetTruncatedCount(), 0U);
  EXPECT_EQ(result.attemptedCount(), 4U);
  EXPECT_EQ(result.pairAttemptCounts()[0U], 4U);
  ASSERT_EQ(result.selectedSequence().size(), 4U);
  EXPECT_EQ(result.selectedSequence()[0U].support_key.plane.id, 1U);
  EXPECT_EQ(result.selectedSequence()[1U].support_key.plane.id, 2U);
  EXPECT_EQ(result.selectedSequence()[2U].support_key.plane.id, 3U);
  EXPECT_EQ(result.selectedSequence()[3U].support_key.plane.id, 4U);
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  DuplicateSupportUsesCanonicalRequestFields)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest larger_pixel =
    validRequest(bundle, 0U, 1U, 9U);
  larger_pixel.patch.source_pixel.x() = 330.0;
  livo::ReferenceOnlyMigrationShadowRequest smaller_pixel =
    validRequest(bundle, 0U, 1U, 9U);
  smaller_pixel.patch.source_pixel.x() = 310.0;

  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> candidates;
  candidates.emplace_back(std::move(larger_pixel));
  candidates.emplace_back(std::move(smaller_pixel));
  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    0U, std::move(candidates));

  EXPECT_EQ(result.inputCandidateCount(), 2U);
  EXPECT_EQ(result.uniqueCandidateCount(), 1U);
  EXPECT_EQ(result.duplicateCandidateCount(), 1U);
  EXPECT_EQ(result.retainedCandidateCount(), 1U);
  EXPECT_EQ(result.attemptedCount(), 1U);
  ASSERT_EQ(result.selectedSequence().size(), 1U);
  EXPECT_DOUBLE_EQ(
    result.selectedSequence()[0U].source_pixel.x(), 310.0);
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  TotalBudgetUsesRankFirstFairSelection)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> candidates;
  std::size_t pair_index = 0U;
  for (std::uint8_t source = 0U; source < kCameraCount; ++source)
  {
    for (std::uint8_t target = 0U; target < kCameraCount; ++target)
    {
      if (source == target)
      {
        continue;
      }
      for (std::uint64_t rank = 0U; rank < 4U; ++rank)
      {
        candidates.emplace_back(
          validRequest(
            bundle, source, target,
            100U +
            static_cast<std::uint64_t>(pair_index) * 10U +
            rank));
      }
      ++pair_index;
    }
  }
  ASSERT_EQ(pair_index, 12U);

  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    0U, std::move(candidates));

  EXPECT_EQ(result.inputCandidateCount(), 48U);
  EXPECT_EQ(result.uniqueCandidateCount(), 48U);
  EXPECT_EQ(result.retainedCandidateCount(), 48U);
  EXPECT_EQ(result.pairBudgetTruncatedCount(), 0U);
  EXPECT_EQ(result.totalBudgetTruncatedCount(), 16U);
  EXPECT_EQ(result.attemptedCount(), 32U);
  for (std::size_t index = 0U; index < 12U; ++index)
  {
    EXPECT_EQ(
      result.pairAttemptCounts()[index],
      index < 8U ? 3U : 2U);
  }
  ASSERT_EQ(result.selectedSequence().size(), 32U);
  for (std::size_t index = 0U; index < 12U; ++index)
  {
    EXPECT_EQ(
      result.selectedSequence()[index].support_key.plane.id,
      100U + index * 10U);
    EXPECT_EQ(
      result.selectedSequence()[12U + index].
      support_key.plane.id,
      101U + index * 10U);
  }
  for (std::size_t index = 0U; index < 8U; ++index)
  {
    EXPECT_EQ(
      result.selectedSequence()[24U + index].
      support_key.plane.id,
      102U + index * 10U);
  }
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  GenerationRotatesThirdSlotAllocation)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> candidates;
  for (std::uint8_t source = 0U; source < kCameraCount; ++source)
  {
    for (std::uint8_t target = 0U; target < kCameraCount; ++target)
    {
      if (source == target)
      {
        continue;
      }
      const std::size_t pair_index =
        static_cast<std::size_t>(source) * 3U +
        (target < source ?
        static_cast<std::size_t>(target) :
        static_cast<std::size_t>(target - 1U));
      for (std::uint64_t rank = 0U; rank < 4U; ++rank)
      {
        candidates.emplace_back(
          validRequest(
            bundle, source, target,
            100U +
            static_cast<std::uint64_t>(pair_index) * 10U +
            rank));
      }
    }
  }

  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    8U, std::move(candidates));

  EXPECT_EQ(result.attemptedCount(), 32U);
  for (std::size_t index = 0U; index < 12U; ++index)
  {
    const bool receives_third_slot =
      index >= 8U || index <= 3U;
    EXPECT_EQ(
      result.pairAttemptCounts()[index],
      receives_third_slot ? 3U : 2U);
  }
  ASSERT_EQ(result.selectedSequence().size(), 32U);
  EXPECT_EQ(
    result.selectedSequence()[0U].support_key.plane.id,
    180U);
  EXPECT_EQ(
    result.selectedSequence()[11U].support_key.plane.id,
    170U);
  EXPECT_EQ(
    result.selectedSequence()[24U].support_key.plane.id,
    182U);
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  OneHundredInputPermutationsProduceIdenticalOutput)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> base;
  for (std::uint8_t source = 0U; source < kCameraCount; ++source)
  {
    for (std::uint8_t target = 0U; target < kCameraCount; ++target)
    {
      if (source == target)
      {
        continue;
      }
      const std::size_t pair_index =
        static_cast<std::size_t>(source) * 3U +
        (target < source ?
        static_cast<std::size_t>(target) :
        static_cast<std::size_t>(target - 1U));
      for (std::uint64_t rank = 0U; rank < 4U; ++rank)
      {
        base.emplace_back(
          validRequest(
            bundle, source, target,
            100U +
            static_cast<std::uint64_t>(pair_index) * 10U +
            rank));
      }
    }
  }
  livo::ReferenceOnlyMigrationShadowRequest larger_duplicate =
    validRequest(bundle, 0U, 1U, 100U);
  larger_duplicate.patch.source_pixel.x() = 330.0;
  base.emplace_back(std::move(larger_duplicate));
  livo::ReferenceOnlyMigrationShadowRequest smaller_duplicate =
    validRequest(bundle, 0U, 1U, 100U);
  smaller_duplicate.patch.source_pixel.x() = 310.0;
  base.emplace_back(std::move(smaller_duplicate));
  livo::ReferenceOnlyMigrationShadowRequest ambiguous_ready =
    validRequest(bundle, 0U, 1U, 90U);
  livo::ReferenceOnlyMigrationShadowRequest ambiguous_rejected =
    ambiguous_ready;
  ambiguous_rejected.plane.normal_world =
    Eigen::Vector3d::UnitX();
  base.emplace_back(std::move(ambiguous_ready));
  base.emplace_back(std::move(ambiguous_rejected));

  const auto expected =
    livo::runReferenceOnlyMigrationShadowBatch(5U, base);
  ASSERT_EQ(expected.inputCandidateCount(), 52U);
  ASSERT_EQ(expected.duplicateCandidateCount(), 3U);
  ASSERT_EQ(expected.ambiguousDuplicateCount(), 1U);
  ASSERT_EQ(expected.uniqueCandidateCount(), 48U);
  ASSERT_EQ(expected.attemptedCount(), 32U);

  std::mt19937 generator(20260729U);
  for (int repetition = 0; repetition < 100; ++repetition)
  {
    auto shuffled = base;
    std::shuffle(
      shuffled.begin(), shuffled.end(), generator);
    const auto actual =
      livo::runReferenceOnlyMigrationShadowBatch(
      5U, std::move(shuffled));

    EXPECT_EQ(actual.reason(), expected.reason());
    EXPECT_EQ(
      actual.inputCandidateCount(),
      expected.inputCandidateCount());
    EXPECT_EQ(
      actual.uniqueCandidateCount(),
      expected.uniqueCandidateCount());
    EXPECT_EQ(
      actual.duplicateCandidateCount(),
      expected.duplicateCandidateCount());
    EXPECT_EQ(
      actual.ambiguousDuplicateCount(),
      expected.ambiguousDuplicateCount());
    EXPECT_EQ(
      actual.retainedCandidateCount(),
      expected.retainedCandidateCount());
    EXPECT_EQ(
      actual.pairBudgetTruncatedCount(),
      expected.pairBudgetTruncatedCount());
    EXPECT_EQ(
      actual.totalBudgetTruncatedCount(),
      expected.totalBudgetTruncatedCount());
    EXPECT_EQ(actual.attemptedCount(), expected.attemptedCount());
    EXPECT_EQ(actual.readyCount(), expected.readyCount());
    EXPECT_EQ(actual.rejectedCount(), expected.rejectedCount());
    EXPECT_EQ(
      actual.pairAttemptCounts(),
      expected.pairAttemptCounts());
    EXPECT_EQ(
      actual.pairReadyCounts(),
      expected.pairReadyCounts());
    for (std::size_t index = 0U;
         index < livo::kReferenceOnlyMigrationShadowStatusCount;
         ++index)
    {
      EXPECT_EQ(
        actual.statusCount(
          static_cast<livo::MigrationStatus>(index)),
        expected.statusCount(
          static_cast<livo::MigrationStatus>(index)));
    }
    EXPECT_EQ(
      actual.readyWarpConditionP50().has_value(),
      expected.readyWarpConditionP50().has_value());
    EXPECT_EQ(
      actual.readyWarpConditionP95().has_value(),
      expected.readyWarpConditionP95().has_value());
    if (expected.readyWarpConditionP50())
    {
      EXPECT_DOUBLE_EQ(
        *actual.readyWarpConditionP50(),
        *expected.readyWarpConditionP50());
      EXPECT_DOUBLE_EQ(
        *actual.readyWarpConditionP95(),
        *expected.readyWarpConditionP95());
    }

    ASSERT_EQ(
      actual.selectedSequence().size(),
      expected.selectedSequence().size());
    for (std::size_t index = 0U;
         index < expected.selectedSequence().size(); ++index)
    {
      const auto & actual_selection =
        actual.selectedSequence()[index];
      const auto & expected_selection =
        expected.selectedSequence()[index];
      EXPECT_EQ(
        actual_selection.source_camera_id,
        expected_selection.source_camera_id);
      EXPECT_EQ(
        actual_selection.target_camera_id,
        expected_selection.target_camera_id);
      EXPECT_TRUE(
        actual_selection.support_key ==
        expected_selection.support_key);
      EXPECT_EQ(
        actual_selection.geometry_version.plane,
        expected_selection.geometry_version.plane);
      EXPECT_EQ(
        actual_selection.geometry_version.revision,
        expected_selection.geometry_version.revision);
      EXPECT_TRUE(actual_selection.source_pixel.isApprox(
        expected_selection.source_pixel, 0.0));
      EXPECT_TRUE(
        actual_selection.plane_point_world_m.isApprox(
          expected_selection.plane_point_world_m, 0.0));
      EXPECT_EQ(
        actual_selection.migration_status,
        expected_selection.migration_status);
    }
  }
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  RejectedCandidatesPopulateStatusHistogram)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest non_finite =
    validRequest(bundle, 0U, 1U, 10U);
  non_finite.source_pose.position_world_m.x() =
    std::numeric_limits<double>::quiet_NaN();
  livo::ReferenceOnlyMigrationShadowRequest ray_degenerate =
    validRequest(bundle, 0U, 2U, 11U);
  ray_degenerate.plane.normal_world =
    Eigen::Vector3d::UnitX();
  livo::ReferenceOnlyMigrationShadowRequest out_of_frame =
    validRequest(bundle, 0U, 3U, 12U);
  out_of_frame.target_image_bounds =
    livo::ReferenceOnlyMigrationTargetImageBounds{100, 100};

  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> candidates;
  candidates.emplace_back(std::move(non_finite));
  candidates.emplace_back(std::move(ray_degenerate));
  candidates.emplace_back(std::move(out_of_frame));
  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    0U, std::move(candidates));

  EXPECT_EQ(result.attemptedCount(), 3U);
  EXPECT_EQ(result.readyCount(), 0U);
  EXPECT_EQ(result.rejectedCount(), 3U);
  EXPECT_EQ(
    result.statusCount(livo::MigrationStatus::kNonFinite), 1U);
  EXPECT_EQ(
    result.statusCount(
      livo::MigrationStatus::kRayPlaneDegenerate),
    1U);
  EXPECT_EQ(
    result.statusCount(livo::MigrationStatus::kOutOfFrame),
    1U);
  EXPECT_EQ(
    result.statusCount(livo::MigrationStatus::kReady), 0U);
  EXPECT_FALSE(result.readyWarpConditionP50().has_value());
  EXPECT_FALSE(result.readyWarpConditionP95().has_value());
}

TEST(
  ReferenceOnlyMigrationShadowBatchRunner,
  ReadyWarpMetricsUseNearestRankPercentiles)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  const std::array<double, 4U> target_angles{{
    0.12, 0.0, 0.08, 0.04}};
  std::vector<double> expected_warps;
  std::vector<
    livo::ReferenceOnlyMigrationShadowBatchCandidate> candidates;
  for (std::size_t index = 0U;
       index < target_angles.size(); ++index)
  {
    livo::ReferenceOnlyMigrationShadowRequest request =
      validRequest(
      bundle, 0U, 1U,
      20U + static_cast<std::uint64_t>(index));
    request.target_endpoint.rotation_imu_from_camera =
      Eigen::AngleAxisd(
      target_angles[index], Eigen::Vector3d::UnitY()).
      toRotationMatrix();
    const auto direct =
      livo::evaluateReferenceOnlyMigrationShadow(request);
    ASSERT_EQ(
      direct.migrationStatus(), livo::MigrationStatus::kReady);
    ASSERT_TRUE(std::isfinite(direct.warpConditionNumber()));
    expected_warps.push_back(direct.warpConditionNumber());
    candidates.emplace_back(std::move(request));
  }
  std::sort(expected_warps.begin(), expected_warps.end());

  const auto result =
    livo::runReferenceOnlyMigrationShadowBatch(
    0U, std::move(candidates));

  ASSERT_EQ(result.readyCount(), 4U);
  ASSERT_TRUE(result.readyWarpConditionP50().has_value());
  ASSERT_TRUE(result.readyWarpConditionP95().has_value());
  EXPECT_NEAR(
    *result.readyWarpConditionP50(), expected_warps[1U],
    1.0e-12);
  EXPECT_NEAR(
    *result.readyWarpConditionP95(), expected_warps[3U],
    1.0e-12);
}

}  // namespace
