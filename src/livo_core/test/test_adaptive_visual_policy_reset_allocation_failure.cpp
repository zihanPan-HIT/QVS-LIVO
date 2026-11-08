#include "adaptive_visual_failure_injection_test_support.h"
#include "adaptive_visual_selection_authority.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <optional>
#include <utility>

namespace
{

constexpr std::uint8_t kInitialPrimary = 0U;
constexpr std::uint8_t kChallenger = 2U;

livo::AdaptiveVisualPolicyConfig singleCameraConfig()
{
  livo::AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  config.selection_challenger_frames = 3U;
  config.complement_score_gain = 4.0;
  return config;
}

CameraObservationBatch ownedCandidateBatch(
  CameraObservationQueue & queue,
  const std::initializer_list<std::uint8_t> camera_ids,
  const double timestamp_sec)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  std::size_t index = 0U;
  for (const std::uint8_t camera_id : camera_ids)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      timestamp_sec + 0.004 * static_cast<double>(index);
    observation.image =
      cv::Mat(
      8, 12, CV_8UC1,
      cv::Scalar(51U + camera_id));
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
    ++index;
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      timestamp_sec + 0.100,
      0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

livo::AdaptiveCameraObservation healthyCamera(
  const std::uint8_t camera_id)
{
  livo::AdaptiveCameraObservation observation;
  observation.camera_id = camera_id;
  observation.frame_received = true;
  observation.available_metrics =
    livo::kAdaptiveFirstVersionMetricMask;
  observation.candidate_points = 40U;
  observation.tracked_points = 24U;
  observation.residual_count = 24U * 64U;
  observation.squared_residual_sum =
    static_cast<double>(observation.residual_count) * 25.0;
  observation.mean_gradient = 16.0;
  observation.spatial_coverage = 0.30;
  observation.visual_information =
    livo::AdaptiveCameraObservation::Matrix6::Identity();
  return observation;
}

livo::AdaptiveCameraObservations healthyCandidatesBoundToBatch(
  const CameraObservationBatch & batch,
  const std::uint8_t preferred_camera)
{
  livo::AdaptiveCameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size();
       ++camera_id)
  {
    observations[camera_id].camera_id =
      static_cast<std::uint8_t>(camera_id);
    if (!batch.present[camera_id])
    {
      continue;
    }
    observations[camera_id] = healthyCamera(
      static_cast<std::uint8_t>(camera_id));
    EXPECT_TRUE(batch.validatedReceipt().has_value());
    if (!batch.validatedReceipt().has_value())
    {
      continue;
    }
    observations[camera_id].observation_generation =
      batch.validatedReceipt()->observationGeneration(
      static_cast<std::uint8_t>(camera_id));
    observations[camera_id].image_generation =
      batch.validatedReceipt()->imageGeneration(
      static_cast<std::uint8_t>(camera_id));
  }
  if (batch.present[preferred_camera])
  {
    observations[preferred_camera].visual_information =
      1000.0 *
      livo::AdaptiveCameraObservation::Matrix6::Identity();
  }
  return observations;
}

std::optional<livo::AdaptiveVisualCandidatePlanCapability>
issueCandidateCapability(
  livo::AdaptiveVisualPolicy & policy,
  const CameraObservationBatch & batch,
  const std::uint8_t preferred_camera)
{
  return policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    healthyCandidatesBoundToBatch(
      batch, preferred_camera),
    true);
}

std::optional<livo::AdaptiveVisualSelectionReceipt>
consumeCandidateCapability(
  livo::AdaptiveVisualPolicy & policy,
  const CameraObservationBatch & batch,
  const livo::AdaptiveVisualCandidatePlanCapability &
    policy_capability,
  livo::AdaptiveVisualSelectionAuthority & authority,
  const std::uint64_t lidar_generation)
{
  const auto lidar_token = authority.issueLidarGeneration(
    lidar_generation,
    policy_capability.lidarPoseInformation(),
    policy_capability.lidarInformationReady());
  EXPECT_EQ(
    lidar_token.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(lidar_token.token.has_value());
  if (!lidar_token.token.has_value())
  {
    return std::nullopt;
  }
  const auto selection = authority.issue(
    policy, batch, policy_capability,
    *lidar_token.token);
  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(selection.capability.has_value());
  if (!selection.capability.has_value())
  {
    return std::nullopt;
  }
  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(consumed.receipt.has_value());
  return consumed.receipt;
}

void expectHysteresisEqual(
  const std::array<
    livo::AdaptiveCameraHysteresisState,
    livo::kAdaptiveVisualCameraCount> & actual,
  const std::array<
    livo::AdaptiveCameraHysteresisState,
    livo::kAdaptiveVisualCameraCount> & expected)
{
  for (std::size_t camera_id = 0U;
       camera_id < actual.size();
       ++camera_id)
  {
    EXPECT_EQ(
      actual[camera_id].isolated,
      expected[camera_id].isolated);
    EXPECT_EQ(
      actual[camera_id].consecutive_unhealthy,
      expected[camera_id].consecutive_unhealthy);
    EXPECT_EQ(
      actual[camera_id].consecutive_healthy,
      expected[camera_id].consecutive_healthy);
  }
}

TEST(
  AdaptiveVisualPolicyResetAllocationFailureTest,
  FailureIsAtomicAndPreservesGenerationHighWater)
{
  constexpr std::uint64_t kCommittedGeneration = 101U;
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());

  CameraObservationQueue initial_queue;
  const CameraObservationBatch initial_batch =
    ownedCandidateBatch(
    initial_queue, {kInitialPrimary}, 41.0);
  const auto initial_capability =
    issueCandidateCapability(
    policy, initial_batch, kInitialPrimary);
  ASSERT_TRUE(initial_capability.has_value());
  auto initial_authority =
    initial_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(initial_authority, nullptr);
  const auto initial_receipt = consumeCandidateCapability(
    policy, initial_batch, *initial_capability,
    *initial_authority, kCommittedGeneration - 1U);
  ASSERT_TRUE(initial_receipt.has_value());
  ASSERT_EQ(
    policy.commitHealthObservation(
      *initial_capability, *initial_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);

  CameraObservationQueue challenger_queue;
  const CameraObservationBatch challenger_batch =
    ownedCandidateBatch(
    challenger_queue,
    {kInitialPrimary, kChallenger}, 42.0);
  const auto challenger_capability =
    issueCandidateCapability(
    policy, challenger_batch, kChallenger);
  ASSERT_TRUE(challenger_capability.has_value());
  ASSERT_EQ(
    challenger_capability->decisionPlan().
    proposed_primary_camera,
    std::optional<std::uint8_t>{kInitialPrimary});
  ASSERT_EQ(
    challenger_capability->decisionPlan().
    proposed_selection_challenger.camera_id,
    std::optional<std::uint8_t>{kChallenger});
  ASSERT_EQ(
    challenger_capability->decisionPlan().
    proposed_selection_challenger.
    consecutive_confirmations,
    1U);
  auto challenger_authority =
    challenger_queue.
    issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(challenger_authority, nullptr);
  const auto challenger_receipt =
    consumeCandidateCapability(
    policy, challenger_batch, *challenger_capability,
    *challenger_authority, kCommittedGeneration);
  ASSERT_TRUE(challenger_receipt.has_value());
  ASSERT_EQ(
    policy.commitHealthObservation(
      *challenger_capability, *challenger_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);

  CameraObservationQueue outstanding_queue;
  const CameraObservationBatch outstanding_batch =
    ownedCandidateBatch(
    outstanding_queue,
    {kInitialPrimary, kChallenger}, 43.0);
  const auto outstanding_capability =
    issueCandidateCapability(
    policy, outstanding_batch, kChallenger);
  ASSERT_TRUE(outstanding_capability.has_value());
  ASSERT_TRUE(policy.accepts(*outstanding_capability));
  auto outstanding_authority =
    outstanding_queue.
    issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(outstanding_authority, nullptr);
  const auto outstanding_receipt =
    consumeCandidateCapability(
    policy, outstanding_batch, *outstanding_capability,
    *outstanding_authority, kCommittedGeneration);
  ASSERT_TRUE(outstanding_receipt.has_value());

  const std::uint64_t revision_before =
    policy.revision();
  const auto hysteresis_before = policy.hysteresis();
  const auto state_before = policy.plan(
    outstanding_capability->lidarPoseInformation(),
    outstanding_capability->observations(),
    outstanding_capability->lidarInformationReady());
  ASSERT_EQ(
    state_before.proposed_primary_camera,
    std::optional<std::uint8_t>{kInitialPrimary});
  ASSERT_EQ(
    state_before.
    proposed_selection_challenger.camera_id,
    std::optional<std::uint8_t>{kChallenger});

  livo::test::failNextPolicyOwnerAllocation();
  EXPECT_THROW(policy.reset(), std::bad_alloc);

  EXPECT_TRUE(policy.accepts(*outstanding_capability));
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);
  const auto state_after_failed_reset = policy.plan(
    outstanding_capability->lidarPoseInformation(),
    outstanding_capability->observations(),
    outstanding_capability->lidarInformationReady());
  EXPECT_EQ(
    state_after_failed_reset.proposed_primary_camera,
    state_before.proposed_primary_camera);
  EXPECT_EQ(
    state_after_failed_reset.
    proposed_selection_challenger.camera_id,
    state_before.
    proposed_selection_challenger.camera_id);
  EXPECT_EQ(
    state_after_failed_reset.
    proposed_selection_challenger.
    consecutive_confirmations,
    state_before.
    proposed_selection_challenger.
    consecutive_confirmations);
  EXPECT_EQ(
    policy.commitHealthObservation(
      *outstanding_capability,
      *outstanding_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::
    kNonMonotonicOrReplayedLidarGeneration);
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);

  policy.reset();

  EXPECT_FALSE(policy.accepts(*outstanding_capability));
  EXPECT_EQ(policy.revision(), revision_before + 1U);
  const std::array<
    livo::AdaptiveCameraHysteresisState,
    livo::kAdaptiveVisualCameraCount> reset_hysteresis{};
  expectHysteresisEqual(
    policy.hysteresis(), reset_hysteresis);
  const auto reset_state = policy.plan(
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    livo::AdaptiveCameraObservations{}, true);
  EXPECT_FALSE(
    reset_state.proposed_primary_camera.has_value());
  EXPECT_FALSE(
    reset_state.proposed_selection_challenger.
    camera_id.has_value());
  EXPECT_EQ(
    reset_state.proposed_selection_challenger.
    consecutive_confirmations,
    0U);

  CameraObservationQueue replay_queue;
  const CameraObservationBatch replay_batch =
    ownedCandidateBatch(
    replay_queue, {kInitialPrimary}, 44.0);
  const auto replay_capability =
    issueCandidateCapability(
    policy, replay_batch, kInitialPrimary);
  ASSERT_TRUE(replay_capability.has_value());
  EXPECT_TRUE(policy.accepts(*replay_capability));
  auto replay_authority =
    replay_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(replay_authority, nullptr);
  const auto replay_receipt = consumeCandidateCapability(
    policy, replay_batch, *replay_capability,
    *replay_authority, kCommittedGeneration);
  ASSERT_TRUE(replay_receipt.has_value());
  EXPECT_EQ(
    policy.commitHealthObservation(
      *replay_capability, *replay_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::
    kNonMonotonicOrReplayedLidarGeneration);

  CameraObservationQueue forward_queue;
  const CameraObservationBatch forward_batch =
    ownedCandidateBatch(
    forward_queue, {kInitialPrimary}, 45.0);
  const auto forward_capability =
    issueCandidateCapability(
    policy, forward_batch, kInitialPrimary);
  ASSERT_TRUE(forward_capability.has_value());
  auto forward_authority =
    forward_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(forward_authority, nullptr);
  const auto forward_receipt = consumeCandidateCapability(
    policy, forward_batch, *forward_capability,
    *forward_authority, kCommittedGeneration + 1U);
  ASSERT_TRUE(forward_receipt.has_value());
  EXPECT_EQ(
    policy.commitHealthObservation(
      *forward_capability, *forward_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
}

}  // namespace
