#include "adaptive_visual_policy.h"
#include "reference_only_migration_runtime_provenance.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace
{

using livo::AdaptiveCameraObservation;
using livo::AdaptiveVisualPolicy;
using livo::AdaptiveVisualPolicyConfig;

AdaptiveCameraObservation healthyObservation(const std::uint8_t camera_id)
{
  AdaptiveCameraObservation observation;
  observation.camera_id = camera_id;
  observation.frame_received = true;
  observation.available_metrics = livo::kAdaptiveFirstVersionMetricMask;
  observation.candidate_points = 40U;
  observation.tracked_points = 24U;
  observation.residual_count = 24U * 64U;
  observation.squared_residual_sum =
    static_cast<double>(observation.residual_count) * 25.0;
  observation.mean_gradient = 16.0;
  observation.spatial_coverage = 0.30;
  observation.visual_information =
    AdaptiveCameraObservation::Matrix6::Identity();
  return observation;
}

AdaptiveVisualPolicy::CameraObservations healthyRig()
{
  AdaptiveVisualPolicy::CameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] =
      healthyObservation(static_cast<std::uint8_t>(camera_id));
  }
  return observations;
}

void addRobustEvidence(
  AdaptiveCameraObservation & observation,
  const double information_gain)
{
  observation.available_metrics |=
    livo::kAdaptiveRobustAdmissionMetricMask;
  observation.inlier_count = 20U;
  observation.total_patch_count = 24U;
  observation.inlier_ratio = 0.80;
  observation.nis = 1.0;
  observation.nis_dof = 6U;
  observation.normalized_pose_correction = 0.01;
  observation.robust_information_gain = information_gain;
}

AdaptiveVisualPolicyConfig sealedSingleWinnerConfig()
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  return config;
}

CameraObservationBatch ownedAdaptivePolicyBatch(
  CameraObservationQueue & queue,
  const double timestamp_sec,
  const std::uint8_t pixel)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  for (std::uint8_t camera_id = 0U;
       camera_id < livo::kAdaptiveVisualCameraCount; ++camera_id)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp = timestamp_sec;
    observation.image =
      cv::Mat(8, 12, CV_8UC1, cv::Scalar(pixel));
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }
  CameraObservationBatch batch;
  EXPECT_TRUE(
    queue.popSynchronized(
      timestamp_sec + 0.010, 0.010, batch));
  return batch;
}

std::optional<livo::ReferenceOnlyMigrationRuntimeBundle>
publishAdaptivePolicyRuntime(
  CameraObservationQueue & queue,
  livo::ReferenceOnlyMigrationRuntimePublisher & publisher,
  const double timestamp_sec,
  const std::uint8_t pixel)
{
  const CameraObservationBatch batch =
    ownedAdaptivePolicyBatch(queue, timestamp_sec, pixel);
  StatesGroup state;
  state.pos_end.x() = timestamp_sec;
  state.cov =
    2.0e-3 *
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
  livo::CameraTimeTransfer transfer;
  return publisher.publish(
    batch, state, batch.timestamp, 0.0, transfer);
}

TEST(
  AdaptiveVisualPolicyPlanCapabilityTest,
  OwnerSealedCapabilityFreezesPlanInputsAndSharesRuntimePublication)
{
  CameraObservationQueue queue;
  const CameraObservationBatch first_batch =
    ownedAdaptivePolicyBatch(queue, 20.0, 17U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  state.pos_end = Eigen::Vector3d(1.0, 2.0, 3.0);
  livo::CameraTimeTransfer transfer;
  const auto runtime = publisher.publish(
    first_batch, state, first_batch.timestamp, 0.0, transfer);
  ASSERT_TRUE(runtime.has_value());

  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  observations[0].tracking_map_initialized = false;
  AdaptiveVisualPolicy::Matrix6 lidar_information =
    AdaptiveVisualPolicy::Matrix6::Identity();
  const std::uint64_t base_revision = policy.revision();

  auto issued = policy.issuePlan(
    *runtime, lidar_information, observations, true);

  ASSERT_TRUE(issued.has_value());
  EXPECT_TRUE(issued->valid());
  EXPECT_TRUE(policy.accepts(*issued));
  EXPECT_EQ(
    issued->configRevision(),
    livo::kAdaptiveVisualPolicyConfigRevision);
  EXPECT_EQ(issued->basePolicyRevision(), base_revision);
  EXPECT_EQ(
    issued->runtimePublicationSequence(),
    runtime->publicationSequence());
  EXPECT_EQ(
    issued->cameraBatchSequence(),
    runtime->cameraBatchReceipt().batchSequence());
  EXPECT_TRUE(
    issued->lidarPoseInformation().isApprox(
      AdaptiveVisualPolicy::Matrix6::Identity(), 0.0));
  EXPECT_TRUE(issued->lidarInformationReady());
  EXPECT_EQ(issued->observations()[0].tracked_points, 24U);
  EXPECT_FALSE(
    issued->observations()[0].tracking_map_initialized);
  EXPECT_TRUE(issued->decisionPlan().lidar_information_valid);
  EXPECT_EQ(
    &issued->runtimeBundle().activeStateSnapshot(),
    &runtime->activeStateSnapshot());
  EXPECT_EQ(
    &issued->runtimeBundle().cameraBatchReceipt(),
    &runtime->cameraBatchReceipt());

  lidar_information(0, 0) = 99.0;
  observations[0].tracked_points = 0U;
  observations[0].tracking_map_initialized = true;
  EXPECT_DOUBLE_EQ(issued->lidarPoseInformation()(0, 0), 1.0);
  EXPECT_EQ(issued->observations()[0].tracked_points, 24U);
  EXPECT_FALSE(
    issued->observations()[0].tracking_map_initialized);
}

TEST(
  AdaptiveVisualPolicyPlanCapabilityTest,
  CapabilityIssuanceRejectsDuplicateAndReverseRuntimeOrder)
{
  CameraObservationQueue queue;
  const CameraObservationBatch first_batch =
    ownedAdaptivePolicyBatch(queue, 30.0, 21U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  const auto first = publisher.publish(
    first_batch, state, first_batch.timestamp, 0.0, transfer);
  const auto second = publishAdaptivePolicyRuntime(
    queue, publisher, 31.0, 22U);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_GT(
    second->publicationSequence(),
    first->publicationSequence());
  ASSERT_GT(
    second->cameraBatchReceipt().batchSequence(),
    first->cameraBatchReceipt().batchSequence());

  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  AdaptiveVisualPolicy policy(config);
  const auto lidar_information =
    AdaptiveVisualPolicy::Matrix6::Identity();
  const auto observations = healthyRig();

  EXPECT_TRUE(
    policy.issuePlan(
      *second, lidar_information, observations, true).has_value());
  EXPECT_FALSE(
    policy.issuePlan(
      *second, lidar_information, observations, true).has_value());
  EXPECT_FALSE(
    policy.issuePlan(
      *first, lidar_information, observations, true).has_value());

  const auto replayed_batch_publication = publisher.publish(
    first_batch, state, first_batch.timestamp, 0.0, transfer);
  EXPECT_FALSE(replayed_batch_publication.has_value());

  livo::ReferenceOnlyMigrationRuntimePublisher restarted_publisher(
    queue.batchAuthorityIdentity());
  const auto newer_batch_older_publication =
    publishAdaptivePolicyRuntime(
    queue, restarted_publisher, 32.0, 23U);
  ASSERT_TRUE(newer_batch_older_publication.has_value());
  ASSERT_GT(
    newer_batch_older_publication->
    cameraBatchReceipt().batchSequence(),
    second->cameraBatchReceipt().batchSequence());
  ASSERT_LT(
    newer_batch_older_publication->publicationSequence(),
    second->publicationSequence());
  EXPECT_FALSE(
    policy.issuePlan(
      *newer_batch_older_publication,
      lidar_information, observations, true).has_value());
}

TEST(
  AdaptiveVisualPolicyPlanCapabilityTest,
  CapabilityRejectsCrossOwnerResetCommitAndReplay)
{
  CameraObservationQueue queue;
  const CameraObservationBatch first_batch =
    ownedAdaptivePolicyBatch(queue, 40.0, 31U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  const auto first = publisher.publish(
    first_batch, state, first_batch.timestamp, 0.0, transfer);
  ASSERT_TRUE(first.has_value());

  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  AdaptiveVisualPolicy owner(config);
  AdaptiveVisualPolicy foreign_owner(config);
  const auto lidar_information =
    AdaptiveVisualPolicy::Matrix6::Identity();
  const auto observations = healthyRig();
  const auto issued = owner.issuePlan(
    *first, lidar_information, observations, true);
  ASSERT_TRUE(issued.has_value());

  EXPECT_TRUE(owner.accepts(*issued));
  EXPECT_FALSE(foreign_owner.accepts(*issued));
  EXPECT_FALSE(
    foreign_owner.commitHealthObservation(*issued, true));
  EXPECT_TRUE(owner.accepts(*issued));
  EXPECT_FALSE(owner.commitHealthObservation(*issued, false));
  EXPECT_TRUE(owner.accepts(*issued));

  const std::uint64_t before_commit = owner.revision();
  ASSERT_TRUE(owner.commitHealthObservation(*issued, true));
  EXPECT_GT(owner.revision(), before_commit);
  EXPECT_FALSE(owner.accepts(*issued));
  EXPECT_FALSE(owner.commitHealthObservation(*issued, true));

  const auto second = publishAdaptivePolicyRuntime(
    queue, publisher, 41.0, 32U);
  ASSERT_TRUE(second.has_value());
  const auto before_reset = owner.issuePlan(
    *second, lidar_information, observations, true);
  ASSERT_TRUE(before_reset.has_value());
  ASSERT_TRUE(owner.accepts(*before_reset));
  const std::uint64_t revision_before_reset = owner.revision();

  owner.reset();

  EXPECT_GT(owner.revision(), revision_before_reset);
  EXPECT_NE(owner.revision(), 0U);
  EXPECT_FALSE(owner.accepts(*before_reset));
  EXPECT_FALSE(owner.commitHealthObservation(*before_reset, true));
}

TEST(
  AdaptiveVisualPolicyPlanCapabilityTest,
  SealedPlanCapabilityCommitCannotAdvancePrimaryOrChallengerHistory)
{
  CameraObservationQueue queue;
  const CameraObservationBatch first_batch =
    ownedAdaptivePolicyBatch(queue, 50.0, 41U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  const auto first_runtime = publisher.publish(
    first_batch, state, first_batch.timestamp, 0.0, transfer);
  ASSERT_TRUE(first_runtime.has_value());

  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  config.complement_score_gain = 4.0;
  config.selection_challenger_frames = 2U;
  AdaptiveVisualPolicy policy(config);
  const auto lidar_information =
    AdaptiveVisualPolicy::Matrix6::Identity();
  auto observations = healthyRig();
  observations[3U].saturation_fraction = 0.95;
  const auto hysteresis_before = policy.hysteresis();

  const auto initial = policy.issuePlan(
    *first_runtime, lidar_information, observations, true);
  ASSERT_TRUE(initial.has_value());
  ASSERT_TRUE(initial->decisionPlan().cameras[0U].enabled);
  ASSERT_TRUE(
    initial->decisionPlan().
    proposed_hysteresis[3U].isolated);
  const std::uint64_t revision_before = policy.revision();
  ASSERT_TRUE(policy.commitHealthObservation(*initial, true));
  EXPECT_EQ(policy.revision(), revision_before + 1U);
  for (std::size_t camera_id = 0U;
       camera_id < hysteresis_before.size(); ++camera_id)
  {
    EXPECT_EQ(
      policy.hysteresis()[camera_id].isolated,
      hysteresis_before[camera_id].isolated);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].consecutive_unhealthy,
      hysteresis_before[camera_id].consecutive_unhealthy);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].consecutive_healthy,
      hysteresis_before[camera_id].consecutive_healthy);
  }

  observations[1U].visual_information =
    100.0 * AdaptiveCameraObservation::Matrix6::Identity();
  const auto second_runtime = publishAdaptivePolicyRuntime(
    queue, publisher, 51.0, 42U);
  ASSERT_TRUE(second_runtime.has_value());
  const auto next = policy.issuePlan(
    *second_runtime, lidar_information, observations, true);
  ASSERT_TRUE(next.has_value());
  EXPECT_FALSE(next->decisionPlan().cameras[0U].enabled);
  EXPECT_TRUE(next->decisionPlan().cameras[1U].enabled);
  EXPECT_EQ(
    next->decisionPlan().proposed_primary_camera,
    std::optional<std::uint8_t>{1U});
  EXPECT_FALSE(
    next->decisionPlan().
    proposed_selection_challenger.camera_id.has_value());
  EXPECT_EQ(
    next->decisionPlan().
    proposed_selection_challenger.consecutive_confirmations,
    0U);
}

TEST(
  AdaptiveVisualPolicyPlanCapabilityTest,
  SealedSingleWinnerFallsBackToOneBaseCandidateWhenEveryRobustGateFails)
{
  AdaptiveVisualPolicy policy(sealedSingleWinnerConfig());
  auto observations = healthyRig();
  observations[2U].visual_information =
    100.0 * AdaptiveCameraObservation::Matrix6::Identity();

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);

  EXPECT_EQ(
    std::count_if(
      plan.cameras.begin(), plan.cameras.end(),
      [](const livo::AdaptiveCameraDecision & camera)
      {
        return camera.enabled;
      }),
    1);
  EXPECT_TRUE(plan.cameras[2U].enabled);
  EXPECT_FALSE(plan.cameras[2U].robust_gate_passed);
  EXPECT_EQ(
    plan.selection_reason,
    livo::AdaptiveSelectionReason::kSelectedBestHealthy);
}

TEST(
  AdaptiveVisualPolicyPlanCapabilityTest,
  SealedSingleWinnerIsEquivariantWhenCameraQualitiesArePermuted)
{
  for (std::size_t strongest_camera = 0U;
       strongest_camera < livo::kAdaptiveVisualCameraCount;
       ++strongest_camera)
  {
    SCOPED_TRACE(strongest_camera);
    AdaptiveVisualPolicy policy(sealedSingleWinnerConfig());
    auto observations = healthyRig();
    for (std::size_t camera_id = 0U;
         camera_id < observations.size(); ++camera_id)
    {
      addRobustEvidence(
        observations[camera_id],
        camera_id == strongest_camera ? 10.0 :
        1.0 + static_cast<double>(
          (camera_id + strongest_camera) %
          livo::kAdaptiveVisualCameraCount));
    }

    const auto plan = policy.plan(
      AdaptiveVisualPolicy::Matrix6::Identity(),
      observations, true);

    EXPECT_EQ(
      std::count_if(
        plan.cameras.begin(), plan.cameras.end(),
        [](const livo::AdaptiveCameraDecision & camera)
        {
          return camera.enabled;
        }),
      1);
    EXPECT_TRUE(plan.cameras[strongest_camera].enabled);
  }
}

TEST(
  AdaptiveVisualPolicyPlanCapabilityTest,
  SealedSingleWinnerSelectsOnlyFromRobustPoolWhenOneExists)
{
  AdaptiveVisualPolicy policy(sealedSingleWinnerConfig());
  auto observations = healthyRig();
  observations[2U].visual_information =
    1000.0 * AdaptiveCameraObservation::Matrix6::Identity();
  addRobustEvidence(observations[1U], 0.01);

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);

  EXPECT_TRUE(plan.cameras[1U].enabled);
  EXPECT_TRUE(plan.cameras[1U].robust_gate_passed);
  EXPECT_FALSE(plan.cameras[2U].enabled);
  EXPECT_FALSE(plan.cameras[2U].robust_gate_passed);
  EXPECT_EQ(
    plan.selection_reason,
    livo::AdaptiveSelectionReason::kSelectedBestInformationGain);
}

}  // namespace
