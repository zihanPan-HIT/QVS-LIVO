#include "adaptive_visual_selection_authority.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>
#include <utility>

namespace
{

livo::AdaptiveCameraObservations singleHealthyCamera(
  const std::uint8_t selected_camera)
{
  livo::AdaptiveCameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    auto & observation = observations[camera_id];
    observation.camera_id =
      static_cast<std::uint8_t>(camera_id);
  }

  auto & selected = observations[selected_camera];
  selected.frame_received = true;
  selected.available_metrics =
    livo::kAdaptiveFirstVersionMetricMask;
  selected.candidate_points = 40U;
  selected.tracked_points = 24U;
  selected.residual_count = 24U * 64U;
  selected.squared_residual_sum =
    static_cast<double>(selected.residual_count) * 25.0;
  selected.mean_gradient = 16.0;
  selected.spatial_coverage = 0.30;
  selected.visual_information =
    livo::AdaptiveCameraObservation::Matrix6::Identity();
  return observations;
}

livo::AdaptiveCameraObservations singleHealthyCameraBoundToBatch(
  const CameraObservationBatch & batch,
  const std::uint8_t selected_camera)
{
  auto observations = singleHealthyCamera(selected_camera);
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  if (!batch.validatedReceipt().has_value()) {
    return observations;
  }
  observations[selected_camera].observation_generation =
    batch.validatedReceipt()->observationGeneration(
    selected_camera);
  observations[selected_camera].image_generation =
    batch.validatedReceipt()->imageGeneration(
    selected_camera);
  return observations;
}

livo::AdaptiveCameraObservations zeroSelectionCameraBoundToBatch(
  const CameraObservationBatch & batch,
  const std::uint8_t camera_id)
{
  auto observations =
    singleHealthyCameraBoundToBatch(batch, camera_id);
  observations[camera_id].visual_information =
    livo::AdaptiveCameraObservation::Matrix6::Zero();
  return observations;
}

CameraObservationBatch ownedSingleCameraBatch(
  CameraObservationQueue & queue,
  const std::uint8_t camera_id,
  const double timestamp_sec,
  const std::uint8_t pixel)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = timestamp_sec;
  observation.image =
    cv::Mat(8, 12, CV_8UC1, cv::Scalar(pixel));
  EXPECT_EQ(
    queue.push(std::move(observation)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      timestamp_sec + 0.100, 0.010, 0.030, 1U,
      batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

CameraObservationBatch ownedTwoCameraCandidateBatch(
  CameraObservationQueue & queue,
  const bool later_camera_arrives_first,
  const double timestamp_sec = 11.000,
  const std::uint8_t earlier_pixel = 31U,
  const std::uint8_t later_pixel = 47U)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  auto make_observation =
    [](const std::uint8_t camera_id,
       const double timestamp_sec,
       const std::uint8_t pixel)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = timestamp_sec;
      observation.image =
        cv::Mat(8, 12, CV_8UC1, cv::Scalar(pixel));
      return observation;
    };
  CameraObservation earlier =
    make_observation(
    0U, timestamp_sec, earlier_pixel);
  CameraObservation later =
    make_observation(
    2U, timestamp_sec + 0.008, later_pixel);
  if (later_camera_arrives_first)
  {
    EXPECT_EQ(
      queue.push(std::move(later)),
      CameraObservationPushResult::ACCEPTED);
    EXPECT_EQ(
      queue.push(std::move(earlier)),
      CameraObservationPushResult::ACCEPTED);
  }
  else
  {
    EXPECT_EQ(
      queue.push(std::move(earlier)),
      CameraObservationPushResult::ACCEPTED);
    EXPECT_EQ(
      queue.push(std::move(later)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      timestamp_sec + 0.100,
      0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

livo::AdaptiveCameraObservations
twoCandidateQualitiesBoundToBatch(
  const CameraObservationBatch & batch)
{
  livo::AdaptiveCameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id].camera_id =
      static_cast<std::uint8_t>(camera_id);
  }

  observations[0U] = singleHealthyCamera(0U)[0U];
  observations[0U].tracked_points = 8U;
  observations[0U].squared_residual_sum =
    static_cast<double>(
      observations[0U].residual_count) * 400.0;
  observations[0U].mean_gradient = 6.0;
  observations[0U].spatial_coverage = 0.10;

  observations[2U] = singleHealthyCamera(2U)[2U];
  observations[2U].visual_information =
    100.0 *
    livo::AdaptiveCameraObservation::Matrix6::Identity();

  EXPECT_TRUE(batch.validatedReceipt().has_value());
  if (!batch.validatedReceipt().has_value()) {
    return observations;
  }
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 2U>{{0U, 2U}})
  {
    observations[camera_id].observation_generation =
      batch.validatedReceipt()->observationGeneration(
      camera_id);
    observations[camera_id].image_generation =
      batch.validatedReceipt()->imageGeneration(
      camera_id);
  }
  return observations;
}

CameraObservationBatch ownedFourCameraCandidateBatch(
  CameraObservationQueue & queue)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  const std::array<std::uint8_t, 4U>
    push_order{{3U, 1U, 2U, 0U}};
  const std::array<double, 4U>
    timestamps{{31.000, 31.004, 31.006, 31.009}};
  for (const std::uint8_t camera_id : push_order)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp = timestamps[camera_id];
    observation.image =
      cv::Mat(
      8, 12, CV_8UC1,
      cv::Scalar(60U + camera_id));
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      31.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

livo::AdaptiveCameraObservations
fourCandidateQualitiesBoundToBatch(
  const CameraObservationBatch & batch)
{
  livo::AdaptiveCameraObservations observations;
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] =
      singleHealthyCamera(
      static_cast<std::uint8_t>(camera_id))[camera_id];
    if (batch.validatedReceipt().has_value())
    {
      observations[camera_id].observation_generation =
        batch.validatedReceipt()->observationGeneration(
        static_cast<std::uint8_t>(camera_id));
      observations[camera_id].image_generation =
        batch.validatedReceipt()->imageGeneration(
        static_cast<std::uint8_t>(camera_id));
    }
  }
  observations[3U].visual_information =
    1000.0 *
    livo::AdaptiveCameraObservation::Matrix6::Identity();
  return observations;
}

livo::AdaptiveVisualPolicyConfig singleCameraConfig()
{
  livo::AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  return config;
}

livo::AuthoritativeLidarGenerationIssueResult
issueReadyIdentityLidarGeneration(
  livo::AdaptiveVisualSelectionAuthority & authority,
  const std::uint64_t generation)
{
  return authority.issueLidarGeneration(
    generation,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true);
}

template<typename Work>
void runTwoWorkersTogether(Work work)
{
  std::atomic<std::size_t> arrived{0U};
  std::atomic<bool> start{false};
  const auto guarded_work =
    [&arrived, &start, &work](const std::size_t worker_id)
    {
      arrived.fetch_add(1U, std::memory_order_acq_rel);
      while (!start.load(std::memory_order_acquire))
      {
        std::this_thread::yield();
      }
      work(worker_id);
    };

  std::thread first(guarded_work, 0U);
  std::thread second(guarded_work, 1U);
  while (arrived.load(std::memory_order_acquire) != 2U)
  {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  first.join();
  second.join();
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
       camera_id < actual.size(); ++camera_id)
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

std::optional<livo::AdaptiveVisualCandidatePlanCapability>
ownerSealedPolicyCapability(
  CameraObservationQueue & queue,
  const CameraObservationBatch & batch,
  livo::AdaptiveVisualPolicy & policy,
  const std::uint8_t selected_camera,
  const std::uint64_t observation_generation_offset = 0U,
  const std::uint64_t image_generation_offset = 0U)
{
  (void)queue;
  auto observations =
    singleHealthyCameraBoundToBatch(
    batch, selected_camera);
  observations[selected_camera].observation_generation +=
    observation_generation_offset;
  observations[selected_camera].image_generation +=
    image_generation_offset;
  return policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);
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
  }
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  if (!batch.validatedReceipt().has_value()) {
    return observations;
  }
  for (std::size_t camera_id = 0U;
       camera_id < observations.size();
       ++camera_id)
  {
    if (!batch.present[camera_id])
    {
      continue;
    }
    const auto single =
      singleHealthyCamera(
      static_cast<std::uint8_t>(camera_id));
    observations[camera_id] = single[camera_id];
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
      100.0 *
      livo::AdaptiveCameraObservation::Matrix6::Identity();
  }
  return observations;
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
  if (!lidar_token.token.has_value()) {
    return std::nullopt;
  }
  const auto selection = authority.issue(
    policy, batch, policy_capability,
    *lidar_token.token);
  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(selection.capability.has_value());
  if (!selection.capability.has_value()) {
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

void isolateCameraThroughSelectedCommit(
  CameraObservationQueue & queue,
  livo::AdaptiveVisualPolicy & policy,
  livo::AdaptiveVisualSelectionAuthority & authority,
  const std::uint64_t lidar_generation,
  const double timestamp_sec)
{
  constexpr std::uint8_t kTargetCamera = 0U;
  constexpr std::uint8_t kSurvivorCamera = 2U;
  const CameraObservationBatch batch =
    ownedTwoCameraCandidateBatch(
    queue, false, timestamp_sec);
  auto observations =
    healthyCandidatesBoundToBatch(
    batch, kSurvivorCamera);
  observations[kTargetCamera].saturation_fraction = 0.95;
  const auto capability =
    policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);
  ASSERT_TRUE(capability.has_value());
  ASSERT_TRUE(
    capability->decisionPlan().
    cameras[kTargetCamera].image_hard_failure);
  ASSERT_TRUE(
    capability->decisionPlan().
    cameras[kSurvivorCamera].enabled);
  const auto receipt = consumeCandidateCapability(
    policy, batch, *capability, authority,
    lidar_generation);
  ASSERT_TRUE(receipt.has_value());
  ASSERT_EQ(
    policy.commitHealthObservation(
      *capability, *receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  ASSERT_TRUE(policy.hysteresis()[kTargetCamera].isolated);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  CandidateSelectionChoosesBestQualityRegardlessOfTimestampOrArrivalOrder)
{
  for (const bool later_camera_arrives_first :
       std::array<bool, 2U>{{false, true}})
  {
    CameraObservationQueue queue;
    const CameraObservationBatch batch =
      ownedTwoCameraCandidateBatch(
      queue, later_camera_arrives_first);
    ASSERT_EQ(batch.camera_count, 2U);
    ASSERT_TRUE(batch.present[0U]);
    ASSERT_TRUE(batch.present[2U]);
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto observations =
      twoCandidateQualitiesBoundToBatch(batch);

    const auto policy_capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      observations, true);

    ASSERT_TRUE(policy_capability.has_value());
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar_token =
      issueReadyIdentityLidarGeneration(
      *authority, later_camera_arrives_first ? 2U : 1U);
    ASSERT_TRUE(lidar_token.token.has_value());
    const auto selection = authority->issue(
      policy, batch, *policy_capability,
      *lidar_token.token);
    ASSERT_EQ(
      selection.status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(selection.capability.has_value());
    EXPECT_EQ(
      selection.capability->selectedCameraId(), 2U);
    EXPECT_EQ(
      selection.capability->candidateMask(),
      batch.present);
    EXPECT_EQ(
      selection.capability->selectedMask(),
      (std::array<bool, 4U>{{false, false, true, false}}));
    EXPECT_EQ(
      selection.capability->candidateObservationGeneration(0U),
      batch.validatedReceipt()->observationGeneration(0U));
    EXPECT_EQ(
      selection.capability->candidateImageGeneration(2U),
      batch.validatedReceipt()->imageGeneration(2U));
    const auto consumed = authority->consume(
      policy, batch, *selection.capability,
      *lidar_token.token);
    ASSERT_EQ(
      consumed.status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(consumed.receipt.has_value());
    EXPECT_EQ(
      consumed.receipt->selectedCameraId(), 2U);
    EXPECT_EQ(
      consumed.receipt->candidateMask(), batch.present);
    EXPECT_EQ(
      consumed.receipt->selectedMask(),
      (std::array<bool, 4U>{{false, false, true, false}}));
    EXPECT_DOUBLE_EQ(
      consumed.receipt->candidateExposureTimestampSec(0U),
      batch.observations[0U].timestamp);
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  CandidatePlanRejectsEveryNonCanonicalFieldOutsideCandidateMask)
{
  enum class Mutation
  {
    kCameraId,
    kFrameReceived,
    kTrackingMapInitialized,
    kObservationGeneration,
    kImageGeneration,
    kAvailableMetrics,
    kCandidatePoints,
    kTrackedPoints,
    kResidualCount,
    kSquaredResidualSum,
    kSaturationFraction,
    kDarkFraction,
    kMeanGradient,
    kSpatialCoverage,
    kVisualInformation,
  };
  constexpr std::array<Mutation, 15U> kMutations{{
      Mutation::kCameraId,
      Mutation::kFrameReceived,
      Mutation::kTrackingMapInitialized,
      Mutation::kObservationGeneration,
      Mutation::kImageGeneration,
      Mutation::kAvailableMetrics,
      Mutation::kCandidatePoints,
      Mutation::kTrackedPoints,
      Mutation::kResidualCount,
      Mutation::kSquaredResidualSum,
      Mutation::kSaturationFraction,
      Mutation::kDarkFraction,
      Mutation::kMeanGradient,
      Mutation::kSpatialCoverage,
      Mutation::kVisualInformation}};

  for (const std::uint8_t non_candidate :
       std::array<std::uint8_t, 2U>{{1U, 3U}})
  {
    for (std::size_t mutation_index = 0U;
         mutation_index < kMutations.size();
         ++mutation_index)
    {
      SCOPED_TRACE(
        "non_candidate=" +
        std::to_string(non_candidate) +
        " mutation=" + std::to_string(mutation_index));
      CameraObservationQueue queue;
      const CameraObservationBatch batch =
        ownedTwoCameraCandidateBatch(queue, false);
      ASSERT_TRUE(batch.validatedReceipt().has_value());
      livo::AdaptiveVisualPolicy policy(
        singleCameraConfig());
      const auto canonical_observations =
        twoCandidateQualitiesBoundToBatch(batch);
      auto dirty_observations = canonical_observations;
      auto & dirty = dirty_observations[non_candidate];
      switch (kMutations[mutation_index])
      {
        case Mutation::kCameraId:
          dirty.camera_id =
            static_cast<std::uint8_t>(
            (non_candidate + 1U) %
            livo::kAdaptiveVisualCameraCount);
          break;
        case Mutation::kFrameReceived:
          dirty.frame_received = true;
          break;
        case Mutation::kTrackingMapInitialized:
          dirty.tracking_map_initialized = false;
          break;
        case Mutation::kObservationGeneration:
          dirty.observation_generation = 1U;
          break;
        case Mutation::kImageGeneration:
          dirty.image_generation = 1U;
          break;
        case Mutation::kAvailableMetrics:
          dirty.available_metrics =
            livo::kAdaptiveFirstVersionMetricMask;
          break;
        case Mutation::kCandidatePoints:
          dirty.candidate_points = 1U;
          break;
        case Mutation::kTrackedPoints:
          dirty.tracked_points = 1U;
          break;
        case Mutation::kResidualCount:
          dirty.residual_count = 1U;
          break;
        case Mutation::kSquaredResidualSum:
          dirty.squared_residual_sum = 1.0;
          break;
        case Mutation::kSaturationFraction:
          dirty.saturation_fraction = 0.1;
          break;
        case Mutation::kDarkFraction:
          dirty.dark_fraction = 0.1;
          break;
        case Mutation::kMeanGradient:
          dirty.mean_gradient = 1.0;
          break;
        case Mutation::kSpatialCoverage:
          dirty.spatial_coverage = 0.1;
          break;
        case Mutation::kVisualInformation:
          dirty.visual_information(0, 0) = 1.0;
          break;
      }
      const auto revision_before = policy.revision();
      const auto hysteresis_before = policy.hysteresis();
      const auto validation_hashes_before =
        batch.validatedReceipt()->
        validationImageHashCount();
      const auto state_before = policy.plan(
        livo::AdaptiveVisualPolicy::Matrix6::Identity(),
        canonical_observations, true);

      const auto rejected = policy.issueCandidatePlan(
        batch,
        livo::AdaptiveVisualPolicy::Matrix6::Identity(),
        dirty_observations, true);

      ASSERT_FALSE(rejected.has_value());
      EXPECT_EQ(policy.revision(), revision_before);
      expectHysteresisEqual(
        policy.hysteresis(), hysteresis_before);
      EXPECT_FALSE(
        batch.validatedReceipt()->
        adaptiveVisualSelectionClaimed());
      EXPECT_EQ(
        batch.validatedReceipt()->
        validationImageHashCount(),
        validation_hashes_before);
      const auto state_after = policy.plan(
        livo::AdaptiveVisualPolicy::Matrix6::Identity(),
        canonical_observations, true);
      EXPECT_EQ(
        state_after.proposed_primary_camera,
        state_before.proposed_primary_camera);
      EXPECT_EQ(
        state_after.proposed_selection_challenger.camera_id,
        state_before.proposed_selection_challenger.camera_id);
      EXPECT_EQ(
        state_after.proposed_selection_challenger.
        consecutive_confirmations,
        state_before.proposed_selection_challenger.
        consecutive_confirmations);

      const auto canonical_capability =
        policy.issueCandidatePlan(
        batch,
        livo::AdaptiveVisualPolicy::Matrix6::Identity(),
        canonical_observations, true);
      ASSERT_TRUE(canonical_capability.has_value());
      auto authority =
        queue.issueAdaptiveVisualSelectionAuthority();
      ASSERT_NE(authority, nullptr);
      const auto lidar_token =
        issueReadyIdentityLidarGeneration(
        *authority,
        1000U + mutation_index +
        static_cast<std::size_t>(non_candidate) *
        kMutations.size());
      ASSERT_TRUE(lidar_token.token.has_value());
      const auto selection = authority->issue(
        policy, batch, *canonical_capability,
        *lidar_token.token);
      ASSERT_TRUE(selection.capability.has_value());
      const auto consumed = authority->consume(
        policy, batch, *selection.capability,
        *lidar_token.token);
      ASSERT_TRUE(consumed.receipt.has_value());
      EXPECT_TRUE(std::isnan(
        consumed.receipt->
        candidateExposureTimestampSec(non_candidate)));
    }
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  UnselectedCandidateGenerationMismatchRejectsWithoutPolicyOrIssueWrites)
{
  for (const bool mutate_image_generation :
       std::array<bool, 2U>{{false, true}})
  {
    CameraObservationQueue queue;
    const CameraObservationBatch batch =
      ownedTwoCameraCandidateBatch(queue, false);
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto valid_observations =
      twoCandidateQualitiesBoundToBatch(batch);
    auto mismatched_observations = valid_observations;
    if (mutate_image_generation)
    {
      ++mismatched_observations[0U].image_generation;
    }
    else
    {
      ++mismatched_observations[0U].
        observation_generation;
    }
    const auto revision_before = policy.revision();
    const auto hysteresis_before = policy.hysteresis();
    const auto state_before = policy.plan(
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      valid_observations, true);

    const auto mismatched_capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      mismatched_observations, true);
    ASSERT_TRUE(mismatched_capability.has_value());
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar_token =
      issueReadyIdentityLidarGeneration(*authority, 3U);
    ASSERT_TRUE(lidar_token.token.has_value());
    const auto rejected = authority->issue(
      policy, batch, *mismatched_capability,
      *lidar_token.token);

    EXPECT_EQ(
      rejected.status,
      livo::AdaptiveVisualSelectionStatus::
      kPolicyObservationIdentityMismatch);
    EXPECT_FALSE(rejected.capability.has_value());
    EXPECT_EQ(policy.revision(), revision_before);
    expectHysteresisEqual(
      policy.hysteresis(), hysteresis_before);
    const auto state_after = policy.plan(
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      valid_observations, true);
    EXPECT_EQ(
      state_after.proposed_primary_camera,
      state_before.proposed_primary_camera);
    EXPECT_EQ(
      state_after.proposed_selection_challenger.camera_id,
      state_before.proposed_selection_challenger.camera_id);
    EXPECT_EQ(
      state_after.proposed_selection_challenger.
      consecutive_confirmations,
      state_before.proposed_selection_challenger.
      consecutive_confirmations);

    const auto valid_capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      valid_observations, true);
    ASSERT_TRUE(valid_capability.has_value());
    const auto accepted = authority->issue(
      policy, batch, *valid_capability,
      *lidar_token.token);
    EXPECT_EQ(
      accepted.status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    EXPECT_TRUE(accepted.capability.has_value());
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  UnselectedCandidateImageMutationRejectsWithoutConsumingAuthority)
{
  for (const bool mutate_shape :
       std::array<bool, 2U>{{false, true}})
  {
    CameraObservationQueue queue;
    CameraObservationBatch batch =
      ownedTwoCameraCandidateBatch(queue, false);
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto observations =
      twoCandidateQualitiesBoundToBatch(batch);
    const auto policy_capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      observations, true);
    ASSERT_TRUE(policy_capability.has_value());
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar_token =
      issueReadyIdentityLidarGeneration(*authority, 4U);
    ASSERT_TRUE(lidar_token.token.has_value());
    const auto selection = authority->issue(
      policy, batch, *policy_capability,
      *lidar_token.token);
    ASSERT_TRUE(selection.capability.has_value());
    const auto revision_before = policy.revision();
    const auto hysteresis_before = policy.hysteresis();

    cv::Mat original_image =
      batch.observations[0U].image;
    if (mutate_shape)
    {
      batch.observations[0U].image =
        original_image.rowRange(
        0, original_image.rows - 1);
    }
    else
    {
      batch.observations[0U].
        image.at<std::uint8_t>(0, 0) ^= 0x5AU;
    }
    const auto rejected = authority->consume(
      policy, batch, *selection.capability,
      *lidar_token.token);

    EXPECT_EQ(
      rejected.status,
      livo::AdaptiveVisualSelectionStatus::
      kObservationImageIdentityMismatch);
    EXPECT_FALSE(rejected.receipt.has_value());
    EXPECT_EQ(policy.revision(), revision_before);
    expectHysteresisEqual(
      policy.hysteresis(), hysteresis_before);

    if (mutate_shape)
    {
      batch.observations[0U].image = original_image;
    }
    else
    {
      batch.observations[0U].
        image.at<std::uint8_t>(0, 0) ^= 0x5AU;
    }
    const auto accepted = authority->consume(
      policy, batch, *selection.capability,
      *lidar_token.token);
    EXPECT_EQ(
      accepted.status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    EXPECT_TRUE(accepted.receipt.has_value());
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  CandidatePlanRejectsObservationOutsideCandidateMask)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedTwoCameraCandidateBatch(queue, false);
  livo::AdaptiveVisualPolicy policy(
    singleCameraConfig());
  auto observations =
    twoCandidateQualitiesBoundToBatch(batch);
  observations[1U] = singleHealthyCamera(1U)[1U];
  observations[1U].visual_information =
    10000.0 *
    livo::AdaptiveCameraObservation::Matrix6::Identity();
  const auto revision_before = policy.revision();
  const auto hysteresis_before = policy.hysteresis();
  const auto capability = policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);

  EXPECT_FALSE(capability.has_value());
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);
  EXPECT_FALSE(
    batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueRejectsMultipleSelectedCameras)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedTwoCameraCandidateBatch(queue, false);
  auto config = singleCameraConfig();
  config.maximum_selected_cameras = 2U;
  livo::AdaptiveVisualPolicy policy(config);
  const auto observations =
    twoCandidateQualitiesBoundToBatch(batch);
  const auto capability = policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);
  ASSERT_TRUE(capability.has_value());
  const auto revision_before = policy.revision();
  const auto hysteresis_before = policy.hysteresis();
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(*authority, 6U);
  ASSERT_TRUE(lidar_token.token.has_value());

  const auto rejected = authority->issue(
    policy, batch, *capability, *lidar_token.token);

  EXPECT_EQ(
    rejected.status,
    livo::AdaptiveVisualSelectionStatus::
    kMultipleSelectedCameras);
  EXPECT_FALSE(rejected.capability.has_value());
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueRejectsZeroSelectedCameras)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedTwoCameraCandidateBatch(queue, false);
  livo::AdaptiveVisualPolicy policy(
    singleCameraConfig());
  auto observations =
    twoCandidateQualitiesBoundToBatch(batch);
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 2U>{{0U, 2U}})
  {
    observations[camera_id].tracked_points = 0U;
    observations[camera_id].residual_count = 0U;
    observations[camera_id].squared_residual_sum = 0.0;
    observations[camera_id].visual_information =
      livo::AdaptiveCameraObservation::Matrix6::Zero();
  }
  const auto capability = policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);
  ASSERT_TRUE(capability.has_value());
  const auto revision_before = policy.revision();
  const auto hysteresis_before = policy.hysteresis();
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(*authority, 7U);
  ASSERT_TRUE(lidar_token.token.has_value());

  const auto rejected = authority->issue(
    policy, batch, *capability, *lidar_token.token);

  EXPECT_EQ(
    rejected.status,
    livo::AdaptiveVisualSelectionStatus::
    kNoSelectedCamera);
  EXPECT_FALSE(rejected.capability.has_value());
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  FourCameraCandidateBatchProducesExactlyOneSelectedAuthority)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedFourCameraCandidateBatch(queue);
  ASSERT_EQ(batch.camera_count, 4U);
  ASSERT_FALSE(batch.partial);
  livo::AdaptiveVisualPolicy policy(
    singleCameraConfig());
  const auto observations =
    fourCandidateQualitiesBoundToBatch(batch);

  const auto policy_capability =
    policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);

  ASSERT_TRUE(policy_capability.has_value());
  EXPECT_EQ(policy_capability->cameraCount(), 4U);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(*authority, 8U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority->issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(selection.capability.has_value());
  EXPECT_EQ(
    selection.capability->candidateMask(),
    (std::array<bool, 4U>{{true, true, true, true}}));
  EXPECT_EQ(
    selection.capability->selectedMask(),
    (std::array<bool, 4U>{{false, false, false, true}}));
  EXPECT_EQ(
    std::count(
      selection.capability->selectedMask().begin(),
      selection.capability->selectedMask().end(), true),
    1);
  EXPECT_EQ(
    selection.capability->selectedCameraId(), 3U);
  const auto consumed = authority->consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  ASSERT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(consumed.receipt.has_value());
  EXPECT_EQ(
    consumed.receipt->candidateMask(),
    (std::array<bool, 4U>{{true, true, true, true}}));
  EXPECT_EQ(
    std::count(
      consumed.receipt->selectedMask().begin(),
      consumed.receipt->selectedMask().end(), true),
    1);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  OwnerConsumesOneSealedCandidateSelectionForOneLidarGeneration)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 10.0, 17U);
  const auto config = singleCameraConfig();
  livo::AdaptiveVisualPolicy policy(config);
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  EXPECT_EQ(
    policy_capability->selectionReason(),
    livo::AdaptiveSelectionReason::kSelectedBestHealthy);

  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 41U);
  ASSERT_EQ(
    lidar_token.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(lidar_token.token.has_value());

  const auto selection = authority.issue(
    policy, batch, *policy_capability, *lidar_token.token);
  ASSERT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(selection.capability.has_value());
  EXPECT_EQ(
    selection.capability->selectedCameraId(),
    kSelectedCamera);
  EXPECT_EQ(
    selection.capability->selectionReason(),
    policy_capability->selectionReason());

  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  ASSERT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(consumed.receipt.has_value());
  EXPECT_TRUE(consumed.receipt->valid());
  EXPECT_EQ(consumed.receipt->lidarGeneration(), 41U);
  EXPECT_EQ(
    consumed.receipt->cameraBatchSequence(),
    batch.validatedReceipt()->batchSequence());
  EXPECT_EQ(
    consumed.receipt->policyRevision(),
    policy_capability->basePolicyRevision());
  EXPECT_EQ(
    consumed.receipt->selectionReason(),
    policy_capability->selectionReason());
  EXPECT_TRUE(
    consumed.receipt->lidarInformationReady());
  EXPECT_TRUE(
    (consumed.receipt->lidarPoseInformation().array() ==
     livo::AdaptiveVisualPolicy::Matrix6::Identity().
     array()).all());
  EXPECT_EQ(
    consumed.receipt->selectedCameraId(),
    kSelectedCamera);
  EXPECT_EQ(
    consumed.receipt->candidateBudget(),
    config.total_candidate_budget);
  EXPECT_EQ(
    consumed.receipt->patchBudget(),
    config.total_patch_budget);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  DefaultConstructedTypedResultsNeverReportReadyWithoutValue)
{
  const livo::AuthoritativeLidarGenerationIssueResult
    lidar_result;
  const livo::AdaptiveVisualSelectionIssueResult
    selection_result;
  const livo::AdaptiveVisualSelectionConsumeResult
    consume_result;

  EXPECT_NE(
    lidar_result.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_FALSE(lidar_result.token.has_value());
  EXPECT_NE(
    selection_result.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_FALSE(selection_result.capability.has_value());
  EXPECT_NE(
    consume_result.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_FALSE(consume_result.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AuthoritativeCandidateCommitAdvancesExactlyOnceAndRejectsReplayWithoutWrites)
{
  constexpr std::uint8_t kSelectedCamera = 1U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 14.0, 18U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 45U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  ASSERT_TRUE(consumed.receipt.has_value());
  const std::uint64_t initial_revision = policy.revision();

  const auto committed = policy.commitHealthObservation(
    *policy_capability, *consumed.receipt, true);

  EXPECT_EQ(
    committed,
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  ASSERT_EQ(policy.revision(), initial_revision + 1U);
  const auto revision_after_commit = policy.revision();
  const auto hysteresis_after_commit = policy.hysteresis();
  const auto state_after_commit = policy.plan(
    policy_capability->lidarPoseInformation(),
    policy_capability->observations(),
    policy_capability->lidarInformationReady());

  const auto replayed = policy.commitHealthObservation(
    *policy_capability, *consumed.receipt, true);

  EXPECT_EQ(
    replayed,
    livo::AdaptiveCandidateHealthCommitStatus::
    kNonMonotonicOrReplayedLidarGeneration);
  EXPECT_EQ(policy.revision(), revision_after_commit);
  for (std::size_t camera_id = 0U;
       camera_id < hysteresis_after_commit.size();
       ++camera_id)
  {
    EXPECT_EQ(
      policy.hysteresis()[camera_id].isolated,
      hysteresis_after_commit[camera_id].isolated);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].consecutive_unhealthy,
      hysteresis_after_commit[camera_id].
      consecutive_unhealthy);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].consecutive_healthy,
      hysteresis_after_commit[camera_id].
      consecutive_healthy);
  }
  const auto state_after_replay = policy.plan(
    policy_capability->lidarPoseInformation(),
    policy_capability->observations(),
    policy_capability->lidarInformationReady());
  EXPECT_EQ(
    state_after_replay.proposed_primary_camera,
    state_after_commit.proposed_primary_camera);
  EXPECT_EQ(
    state_after_replay.
    proposed_selection_challenger.camera_id,
    state_after_commit.
    proposed_selection_challenger.camera_id);
  EXPECT_EQ(
    state_after_replay.
    proposed_selection_challenger.consecutive_confirmations,
    state_after_commit.
    proposed_selection_challenger.consecutive_confirmations);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConcurrentReceiptCopiesCommitExactlyOnceAndRejectReplayWithoutWrites)
{
  constexpr std::uint8_t kSelectedCamera = 1U;
  constexpr std::size_t kConcurrentTrials = 32U;
  for (std::size_t trial = 0U;
       trial < kConcurrentTrials; ++trial)
  {
    SCOPED_TRACE(trial);
    CameraObservationQueue queue;
    const CameraObservationBatch batch =
      ownedSingleCameraBatch(
      queue, kSelectedCamera, 15.0, 19U);
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto capability =
      ownerSealedPolicyCapability(
      queue, batch, policy, kSelectedCamera);
    ASSERT_TRUE(capability.has_value());
    auto authority_owner =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority_owner, nullptr);
    const auto receipt = consumeCandidateCapability(
      policy, batch, *capability, *authority_owner, 46U);
    ASSERT_TRUE(receipt.has_value());
    const auto receipt_copy = *receipt;
    const std::uint64_t base_revision = policy.revision();
    const auto expected_hysteresis =
      capability->decisionPlan().proposed_hysteresis;
    std::array<
      livo::AdaptiveCandidateHealthCommitStatus, 2U>
      statuses{
        livo::AdaptiveCandidateHealthCommitStatus::
        kBindingMismatch,
        livo::AdaptiveCandidateHealthCommitStatus::
        kBindingMismatch};

    runTwoWorkersTogether(
      [&](const std::size_t worker_id)
      {
        const auto & worker_receipt =
          worker_id == 0U ? *receipt : receipt_copy;
        statuses[worker_id] =
          policy.commitHealthObservation(
          *capability, worker_receipt, true);
      });

    EXPECT_EQ(
      std::count(
        statuses.begin(), statuses.end(),
        livo::AdaptiveCandidateHealthCommitStatus::kCommitted),
      1);
    EXPECT_EQ(
      std::count(
        statuses.begin(), statuses.end(),
        livo::AdaptiveCandidateHealthCommitStatus::
        kNonMonotonicOrReplayedLidarGeneration),
      1);
    EXPECT_EQ(policy.revision(), base_revision + 1U);
    for (std::size_t camera_id = 0U;
         camera_id < expected_hysteresis.size();
         ++camera_id)
    {
      EXPECT_EQ(
        policy.hysteresis()[camera_id].isolated,
        expected_hysteresis[camera_id].isolated);
      EXPECT_EQ(
        policy.hysteresis()[camera_id].
        consecutive_unhealthy,
        expected_hysteresis[camera_id].
        consecutive_unhealthy);
      EXPECT_EQ(
        policy.hysteresis()[camera_id].
        consecutive_healthy,
        expected_hysteresis[camera_id].
        consecutive_healthy);
    }
    const auto state_after = policy.plan(
      capability->lidarPoseInformation(),
      capability->observations(),
      capability->lidarInformationReady());
    EXPECT_EQ(
      state_after.proposed_primary_camera,
      capability->decisionPlan().
      proposed_primary_camera);
    EXPECT_EQ(
      state_after.
      proposed_selection_challenger.camera_id,
      capability->decisionPlan().
      proposed_selection_challenger.camera_id);
    EXPECT_EQ(
      state_after.
      proposed_selection_challenger.
      consecutive_confirmations,
      capability->decisionPlan().
      proposed_selection_challenger.
      consecutive_confirmations);
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ResetPreservesCommittedLidarGenerationHighWater)
{
  constexpr std::uint8_t kInitialCamera = 0U;
  constexpr std::uint8_t kAfterResetCamera = 2U;
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());

  CameraObservationQueue initial_queue;
  const CameraObservationBatch initial_batch =
    ownedSingleCameraBatch(
    initial_queue, kInitialCamera, 32.0, 34U);
  const auto initial_capability =
    ownerSealedPolicyCapability(
    initial_queue, initial_batch, policy,
    kInitialCamera);
  ASSERT_TRUE(initial_capability.has_value());
  auto initial_authority_owner =
    initial_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(initial_authority_owner, nullptr);
  const auto initial_receipt = consumeCandidateCapability(
    policy, initial_batch, *initial_capability,
    *initial_authority_owner, 100U);
  ASSERT_TRUE(initial_receipt.has_value());
  ASSERT_EQ(
    policy.commitHealthObservation(
      *initial_capability, *initial_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);

  policy.reset();
  CameraObservationQueue after_reset_queue;
  const CameraObservationBatch after_reset_batch =
    ownedSingleCameraBatch(
    after_reset_queue, kAfterResetCamera, 33.0, 35U);
  const auto after_reset_capability =
    ownerSealedPolicyCapability(
    after_reset_queue, after_reset_batch, policy,
    kAfterResetCamera);
  ASSERT_TRUE(after_reset_capability.has_value());
  auto after_reset_authority_owner =
    after_reset_queue.
    issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(after_reset_authority_owner, nullptr);
  const auto replay_receipt = consumeCandidateCapability(
    policy, after_reset_batch, *after_reset_capability,
    *after_reset_authority_owner, 50U);
  ASSERT_TRUE(replay_receipt.has_value());
  const std::uint64_t revision_before_replay =
    policy.revision();
  const auto hysteresis_before_replay =
    policy.hysteresis();
  const auto state_before_replay = policy.plan(
    after_reset_capability->lidarPoseInformation(),
    after_reset_capability->observations(),
    after_reset_capability->lidarInformationReady());

  const auto rejected = policy.commitHealthObservation(
    *after_reset_capability, *replay_receipt, true);

  EXPECT_EQ(
    rejected,
    livo::AdaptiveCandidateHealthCommitStatus::
    kNonMonotonicOrReplayedLidarGeneration);
  EXPECT_EQ(policy.revision(), revision_before_replay);
  for (std::size_t camera_id = 0U;
       camera_id < hysteresis_before_replay.size();
       ++camera_id)
  {
    EXPECT_EQ(
      policy.hysteresis()[camera_id].isolated,
      hysteresis_before_replay[camera_id].isolated);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].
      consecutive_unhealthy,
      hysteresis_before_replay[camera_id].
      consecutive_unhealthy);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].
      consecutive_healthy,
      hysteresis_before_replay[camera_id].
      consecutive_healthy);
  }
  const auto state_after_replay = policy.plan(
    after_reset_capability->lidarPoseInformation(),
    after_reset_capability->observations(),
    after_reset_capability->lidarInformationReady());
  EXPECT_EQ(
    state_after_replay.proposed_primary_camera,
    state_before_replay.proposed_primary_camera);
  EXPECT_EQ(
    state_after_replay.
    proposed_selection_challenger.camera_id,
    state_before_replay.
    proposed_selection_challenger.camera_id);
  EXPECT_EQ(
    state_after_replay.
    proposed_selection_challenger.consecutive_confirmations,
    state_before_replay.
    proposed_selection_challenger.consecutive_confirmations);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AuthoritativeCandidateCommitRejectsForeignReceiptBindingWithoutWrites)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue expected_queue;
  const CameraObservationBatch expected_batch =
    ownedSingleCameraBatch(
    expected_queue, kSelectedCamera, 16.0, 20U);
  livo::AdaptiveVisualPolicy expected_policy(
    singleCameraConfig());
  const auto expected_capability =
    ownerSealedPolicyCapability(
    expected_queue, expected_batch, expected_policy,
    kSelectedCamera);
  ASSERT_TRUE(expected_capability.has_value());

  CameraObservationQueue foreign_queue;
  const CameraObservationBatch foreign_batch =
    ownedSingleCameraBatch(
    foreign_queue, kSelectedCamera, 16.0, 20U);
  livo::AdaptiveVisualPolicy foreign_policy(
    singleCameraConfig());
  const auto foreign_capability =
    ownerSealedPolicyCapability(
    foreign_queue, foreign_batch, foreign_policy,
    kSelectedCamera);
  ASSERT_TRUE(foreign_capability.has_value());
  auto foreign_authority_owner =
    foreign_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(foreign_authority_owner, nullptr);
  auto & foreign_authority =
    *foreign_authority_owner;
  const auto foreign_lidar_token =
    issueReadyIdentityLidarGeneration(
    foreign_authority, 47U);
  ASSERT_TRUE(foreign_lidar_token.token.has_value());
  const auto foreign_selection =
    foreign_authority.issue(
    foreign_policy, foreign_batch,
    *foreign_capability, *foreign_lidar_token.token);
  ASSERT_TRUE(foreign_selection.capability.has_value());
  const auto foreign_consumed =
    foreign_authority.consume(
    foreign_policy, foreign_batch,
    *foreign_selection.capability,
    *foreign_lidar_token.token);
  ASSERT_TRUE(foreign_consumed.receipt.has_value());
  const auto revision_before = expected_policy.revision();
  const auto hysteresis_before =
    expected_policy.hysteresis();
  const auto state_before = expected_policy.plan(
    expected_capability->lidarPoseInformation(),
    expected_capability->observations(),
    expected_capability->lidarInformationReady());

  const auto rejected =
    expected_policy.commitHealthObservation(
    *expected_capability,
    *foreign_consumed.receipt, true);

  EXPECT_EQ(
    rejected,
    livo::AdaptiveCandidateHealthCommitStatus::
    kBindingMismatch);
  EXPECT_EQ(expected_policy.revision(), revision_before);
  for (std::size_t camera_id = 0U;
       camera_id < hysteresis_before.size();
       ++camera_id)
  {
    EXPECT_EQ(
      expected_policy.hysteresis()[camera_id].isolated,
      hysteresis_before[camera_id].isolated);
    EXPECT_EQ(
      expected_policy.hysteresis()[camera_id].
      consecutive_unhealthy,
      hysteresis_before[camera_id].
      consecutive_unhealthy);
    EXPECT_EQ(
      expected_policy.hysteresis()[camera_id].
      consecutive_healthy,
      hysteresis_before[camera_id].
      consecutive_healthy);
  }
  const auto state_after = expected_policy.plan(
    expected_capability->lidarPoseInformation(),
    expected_capability->observations(),
    expected_capability->lidarInformationReady());
  EXPECT_EQ(
    state_after.proposed_primary_camera,
    state_before.proposed_primary_camera);
  EXPECT_EQ(
    state_after.proposed_selection_challenger.camera_id,
    state_before.proposed_selection_challenger.camera_id);
  EXPECT_EQ(
    state_after.proposed_selection_challenger.
    consecutive_confirmations,
    state_before.proposed_selection_challenger.
    consecutive_confirmations);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AuthoritativeCandidateCommitRejectsSameGenerationWithDifferentCapability)
{
  constexpr std::uint8_t kSelectedCamera = 0U;
  constexpr std::uint64_t kLidarGeneration = 49U;
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());

  CameraObservationQueue first_queue;
  const CameraObservationBatch first_batch =
    ownedSingleCameraBatch(
    first_queue, kSelectedCamera, 18.0, 21U);
  const auto first_capability =
    ownerSealedPolicyCapability(
    first_queue, first_batch, policy,
    kSelectedCamera);
  ASSERT_TRUE(first_capability.has_value());
  auto first_authority_owner =
    first_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(first_authority_owner, nullptr);
  const auto first_receipt = consumeCandidateCapability(
    policy, first_batch, *first_capability,
    *first_authority_owner, kLidarGeneration);
  ASSERT_TRUE(first_receipt.has_value());
  ASSERT_EQ(
    policy.commitHealthObservation(
      *first_capability, *first_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);

  CameraObservationQueue second_queue;
  const CameraObservationBatch unused_first_batch =
    ownedSingleCameraBatch(
    second_queue, kSelectedCamera, 19.0, 22U);
  ASSERT_EQ(
    unused_first_batch.validatedReceipt()->batchSequence(),
    first_capability->cameraBatchSequence());
  const CameraObservationBatch second_batch =
    ownedSingleCameraBatch(
    second_queue, kSelectedCamera, 20.0, 23U);
  ASSERT_GT(
    second_batch.validatedReceipt()->batchSequence(),
    first_capability->cameraBatchSequence());
  const auto second_capability =
    ownerSealedPolicyCapability(
    second_queue, second_batch, policy,
    kSelectedCamera);
  ASSERT_TRUE(second_capability.has_value());
  auto second_authority_owner =
    second_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(second_authority_owner, nullptr);
  const auto second_receipt = consumeCandidateCapability(
    policy, second_batch, *second_capability,
    *second_authority_owner, kLidarGeneration);
  ASSERT_TRUE(second_receipt.has_value());
  const auto revision_before_replay = policy.revision();
  const auto hysteresis_before_replay =
    policy.hysteresis();

  const auto rejected = policy.commitHealthObservation(
    *second_capability, *second_receipt, true);

  EXPECT_EQ(
    rejected,
    livo::AdaptiveCandidateHealthCommitStatus::
    kNonMonotonicOrReplayedLidarGeneration);
  EXPECT_EQ(policy.revision(), revision_before_replay);
  for (std::size_t camera_id = 0U;
       camera_id < hysteresis_before_replay.size();
       ++camera_id)
  {
    EXPECT_EQ(
      policy.hysteresis()[camera_id].isolated,
      hysteresis_before_replay[camera_id].isolated);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].
      consecutive_unhealthy,
      hysteresis_before_replay[camera_id].
      consecutive_unhealthy);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].
      consecutive_healthy,
      hysteresis_before_replay[camera_id].
      consecutive_healthy);
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AuthoritativeCandidateCommitRejectsStaleCapabilityWithoutWrites)
{
  constexpr std::uint8_t kSelectedCamera = 3U;
  CameraObservationQueue queue;
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const CameraObservationBatch stale_batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 22.0, 24U);
  const auto stale_capability =
    ownerSealedPolicyCapability(
    queue, stale_batch, policy, kSelectedCamera);
  ASSERT_TRUE(stale_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  const auto stale_receipt = consumeCandidateCapability(
    policy, stale_batch, *stale_capability,
    *authority_owner, 50U);
  ASSERT_TRUE(stale_receipt.has_value());

  const CameraObservationBatch current_batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 23.0, 25U);
  const auto current_capability =
    ownerSealedPolicyCapability(
    queue, current_batch, policy, kSelectedCamera);
  ASSERT_TRUE(current_capability.has_value());
  ASSERT_TRUE(
    policy.commitHealthObservation(
      current_capability->decisionPlan(), true));
  const auto revision_before = policy.revision();
  const auto hysteresis_before = policy.hysteresis();
  const auto state_before = policy.plan(
    current_capability->lidarPoseInformation(),
    current_capability->observations(),
    current_capability->lidarInformationReady());

  const auto rejected = policy.commitHealthObservation(
    *stale_capability, *stale_receipt, true);

  EXPECT_EQ(
    rejected,
    livo::AdaptiveCandidateHealthCommitStatus::
    kStalePolicyRevision);
  EXPECT_EQ(policy.revision(), revision_before);
  for (std::size_t camera_id = 0U;
       camera_id < hysteresis_before.size();
       ++camera_id)
  {
    EXPECT_EQ(
      policy.hysteresis()[camera_id].isolated,
      hysteresis_before[camera_id].isolated);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].
      consecutive_unhealthy,
      hysteresis_before[camera_id].
      consecutive_unhealthy);
    EXPECT_EQ(
      policy.hysteresis()[camera_id].
      consecutive_healthy,
      hysteresis_before[camera_id].
      consecutive_healthy);
  }
  const auto state_after = policy.plan(
    current_capability->lidarPoseInformation(),
    current_capability->observations(),
    current_capability->lidarInformationReady());
  EXPECT_EQ(
    state_after.proposed_primary_camera,
    state_before.proposed_primary_camera);
  EXPECT_EQ(
    state_after.proposed_selection_challenger.camera_id,
    state_before.proposed_selection_challenger.camera_id);
  EXPECT_EQ(
    state_after.
    proposed_selection_challenger.consecutive_confirmations,
    state_before.
    proposed_selection_challenger.consecutive_confirmations);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AuthoritativeCandidateCommitRequiresThreeDistinctGenerationsToSwitchPrimary)
{
  constexpr std::uint8_t kInitialPrimary = 0U;
  constexpr std::uint8_t kChallenger = 2U;
  constexpr std::uint64_t kInitialLidarGeneration = 60U;
  auto config = singleCameraConfig();
  config.selection_challenger_frames = 3U;
  config.complement_score_gain = 4.0;
  livo::AdaptiveVisualPolicy policy(config);
  CameraObservationQueue queue;

  const CameraObservationBatch initial_batch =
    ownedSingleCameraBatch(
    queue, kInitialPrimary, 24.0, 26U);
  const auto initial_observations =
    healthyCandidatesBoundToBatch(
    initial_batch, kInitialPrimary);
  const auto initial_capability =
    policy.issueCandidatePlan(
    initial_batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    initial_observations, true);
  ASSERT_TRUE(initial_capability.has_value());
  ASSERT_EQ(
    initial_capability->selectionReason(),
    livo::AdaptiveSelectionReason::kSelectedBestHealthy);
  ASSERT_EQ(
    initial_capability->decisionPlan().
    proposed_primary_camera,
    std::optional<std::uint8_t>{kInitialPrimary});
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  const auto initial_receipt = consumeCandidateCapability(
    policy, initial_batch, *initial_capability,
    *authority_owner, kInitialLidarGeneration);
  ASSERT_TRUE(initial_receipt.has_value());
  EXPECT_EQ(
    initial_receipt->selectionReason(),
    initial_capability->selectionReason());
  ASSERT_EQ(
    policy.commitHealthObservation(
      *initial_capability, *initial_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);

  for (std::size_t confirmation = 1U;
       confirmation <= config.selection_challenger_frames;
       ++confirmation)
  {
    const bool switch_confirmed =
      confirmation == config.selection_challenger_frames;
    const std::uint8_t selected_camera =
      switch_confirmed ? kChallenger : kInitialPrimary;
    const CameraObservationBatch batch =
      ownedTwoCameraCandidateBatch(
      queue, false,
      24.0 + static_cast<double>(confirmation),
      static_cast<std::uint8_t>(26U + confirmation),
      static_cast<std::uint8_t>(36U + confirmation));
    const auto observations =
      healthyCandidatesBoundToBatch(
      batch, kChallenger);
    const auto capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      observations, true);
    ASSERT_TRUE(capability.has_value());
    EXPECT_TRUE(
      capability->decisionPlan().
      cameras[selected_camera].enabled);
    EXPECT_EQ(
      capability->decisionPlan().
      proposed_primary_camera,
      std::optional<std::uint8_t>{selected_camera});
    EXPECT_EQ(
      capability->selectionReason(),
      switch_confirmed ?
      livo::AdaptiveSelectionReason::
      kSwitchedAfterChallengerConfirmation :
      livo::AdaptiveSelectionReason::
      kRetainedHealthyPrimary);
    EXPECT_EQ(
      capability->decisionPlan().
      proposed_selection_challenger.
      consecutive_confirmations,
      switch_confirmed ? 0U : confirmation);

    const auto receipt = consumeCandidateCapability(
      policy, batch, *capability, *authority_owner,
      kInitialLidarGeneration + confirmation);
    ASSERT_TRUE(receipt.has_value());
    EXPECT_EQ(
      receipt->selectedCameraId(), selected_camera);
    EXPECT_EQ(
      receipt->selectionReason(),
      capability->selectionReason());
    ASSERT_EQ(
      policy.commitHealthObservation(
        *capability, *receipt, true),
      livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
    EXPECT_EQ(
      policy.revision(),
      static_cast<std::uint64_t>(confirmation + 1U));
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AuthoritativeCandidateCommitSwitchesImmediatelyOnOneHardFaultGeneration)
{
  constexpr std::uint8_t kInitialPrimary = 0U;
  constexpr std::uint8_t kFallbackCamera = 2U;
  auto config = singleCameraConfig();
  config.selection_challenger_frames = 3U;
  livo::AdaptiveVisualPolicy policy(config);
  CameraObservationQueue queue;

  const CameraObservationBatch initial_batch =
    ownedSingleCameraBatch(
    queue, kInitialPrimary, 29.0, 31U);
  const auto initial_observations =
    healthyCandidatesBoundToBatch(
    initial_batch, kInitialPrimary);
  const auto initial_capability =
    policy.issueCandidatePlan(
    initial_batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    initial_observations, true);
  ASSERT_TRUE(initial_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  const auto initial_receipt = consumeCandidateCapability(
    policy, initial_batch, *initial_capability,
    *authority_owner, 70U);
  ASSERT_TRUE(initial_receipt.has_value());
  ASSERT_EQ(
    policy.commitHealthObservation(
      *initial_capability, *initial_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);

  const CameraObservationBatch fault_batch =
    ownedTwoCameraCandidateBatch(
    queue, false, 30.0, 32U, 42U);
  auto fault_observations =
    healthyCandidatesBoundToBatch(
    fault_batch, kFallbackCamera);
  fault_observations[kInitialPrimary].
    saturation_fraction = 0.95;
  const auto fault_capability =
    policy.issueCandidatePlan(
    fault_batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    fault_observations, true);
  ASSERT_TRUE(fault_capability.has_value());
  EXPECT_TRUE(
    fault_capability->decisionPlan().
    cameras[kInitialPrimary].image_hard_failure);
  EXPECT_EQ(
    fault_capability->selectionReason(),
    livo::AdaptiveSelectionReason::
    kSwitchedAfterHardFault);
  EXPECT_EQ(
    fault_capability->decisionPlan().
    proposed_primary_camera,
    std::optional<std::uint8_t>{kFallbackCamera});
  const auto fault_receipt = consumeCandidateCapability(
    policy, fault_batch, *fault_capability,
    *authority_owner, 71U);
  ASSERT_TRUE(fault_receipt.has_value());
  EXPECT_EQ(
    fault_receipt->selectionReason(),
    livo::AdaptiveSelectionReason::
    kSwitchedAfterHardFault);

  ASSERT_EQ(
    policy.commitHealthObservation(
      *fault_capability, *fault_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_TRUE(
    policy.hysteresis()[kInitialPrimary].isolated);
  const auto state_after_fault = policy.plan(
    fault_capability->lidarPoseInformation(),
    fault_capability->observations(),
    fault_capability->lidarInformationReady());
  EXPECT_EQ(
    state_after_fault.proposed_primary_camera,
    std::optional<std::uint8_t>{kFallbackCamera});
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AuthoritativeCandidateCommitReturnsTypedTrustAndOwnershipFailuresWithoutWrites)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 31.0, 33U);
  livo::AdaptiveVisualPolicy capability_owner(
    singleCameraConfig());
  const auto capability =
    ownerSealedPolicyCapability(
    queue, batch, capability_owner, kSelectedCamera);
  ASSERT_TRUE(capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto receipt = consumeCandidateCapability(
    capability_owner, batch, *capability,
    *authority_owner, 72U);
  ASSERT_TRUE(receipt.has_value());
  const auto owner_hysteresis_before =
    capability_owner.hysteresis();

  EXPECT_EQ(
    capability_owner.commitHealthObservation(
      *capability, *receipt, false),
    livo::AdaptiveCandidateHealthCommitStatus::
    kUntrustedScout);
  EXPECT_EQ(capability_owner.revision(), 0U);
  for (std::size_t camera_id = 0U;
       camera_id < owner_hysteresis_before.size();
       ++camera_id)
  {
    EXPECT_EQ(
      capability_owner.hysteresis()[camera_id].isolated,
      owner_hysteresis_before[camera_id].isolated);
    EXPECT_EQ(
      capability_owner.hysteresis()[camera_id].
      consecutive_unhealthy,
      owner_hysteresis_before[camera_id].
      consecutive_unhealthy);
    EXPECT_EQ(
      capability_owner.hysteresis()[camera_id].
      consecutive_healthy,
      owner_hysteresis_before[camera_id].
      consecutive_healthy);
  }

  auto valid_receipt = std::move(*receipt);
  ASSERT_FALSE(receipt->valid());
  EXPECT_EQ(
    capability_owner.commitHealthObservation(
      *capability, *receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::
    kInvalidOrForeignReceipt);
  EXPECT_EQ(capability_owner.revision(), 0U);

  livo::AdaptiveVisualPolicy foreign_policy(
    singleCameraConfig());
  const auto foreign_hysteresis_before =
    foreign_policy.hysteresis();
  EXPECT_EQ(
    foreign_policy.commitHealthObservation(
      *capability, valid_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::
    kInvalidOrForeignCapability);
  EXPECT_EQ(foreign_policy.revision(), 0U);
  for (std::size_t camera_id = 0U;
       camera_id < foreign_hysteresis_before.size();
       ++camera_id)
  {
    EXPECT_EQ(
      foreign_policy.hysteresis()[camera_id].isolated,
      foreign_hysteresis_before[camera_id].isolated);
    EXPECT_EQ(
      foreign_policy.hysteresis()[camera_id].
      consecutive_unhealthy,
      foreign_hysteresis_before[camera_id].
      consecutive_unhealthy);
    EXPECT_EQ(
      foreign_policy.hysteresis()[camera_id].
      consecutive_healthy,
      foreign_hysteresis_before[camera_id].
      consecutive_healthy);
  }

  EXPECT_EQ(
    capability_owner.commitHealthObservation(
      *capability, valid_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_EQ(capability_owner.revision(), 1U);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  OwnerSealedCandidatePlanDoesNotRequireReferenceOnlyRuntimePublisher)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 15.0, 19U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto observations =
    singleHealthyCameraBoundToBatch(
    batch, kSelectedCamera);

  const auto policy_capability = policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);

  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 46U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ReceiptPreservesAuthorityAndSelectedObservationProvenance)
{
  constexpr std::uint8_t kSelectedCamera = 3U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 17.0, 21U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  CameraObservationQueue foreign_queue;
  auto foreign_authority_owner =
    foreign_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(foreign_authority_owner, nullptr);
  auto & foreign_authority = *foreign_authority_owner;
  const auto identity = authority.identity();
  const auto foreign_identity = foreign_authority.identity();
  ASSERT_TRUE(identity.valid());
  ASSERT_TRUE(foreign_identity.valid());
  ASSERT_FALSE(identity.sameAuthority(foreign_identity));
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 48U);
  ASSERT_TRUE(lidar_token.token.has_value());
  ASSERT_TRUE(lidar_token.token->issuedBy(identity));
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  ASSERT_TRUE(selection.capability->issuedBy(identity));
  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  ASSERT_TRUE(consumed.receipt.has_value());

  const auto & receipt = *consumed.receipt;
  EXPECT_TRUE(receipt.issuedBy(identity));
  EXPECT_FALSE(receipt.issuedBy(foreign_identity));
  EXPECT_EQ(
    receipt.selectionStatus(),
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_EQ(receipt.candidateMask(), batch.present);
  EXPECT_TRUE(receipt.cameraPresent(kSelectedCamera));
  EXPECT_EQ(
    receipt.observationGeneration(),
    batch.validatedReceipt()->observationGeneration(
      kSelectedCamera));
  EXPECT_EQ(
    receipt.imageGeneration(),
    batch.validatedReceipt()->imageGeneration(
      kSelectedCamera));
  EXPECT_DOUBLE_EQ(
    receipt.exposureTimestampSec(),
    batch.observations[kSelectedCamera].timestamp);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  CameraQueueIssuesExactlyOneSelectionAuthority)
{
  CameraObservationQueue queue;

  const auto first =
    queue.issueAdaptiveVisualSelectionAuthority();
  const auto duplicate =
    queue.issueAdaptiveVisualSelectionAuthority();

  ASSERT_NE(first, nullptr);
  EXPECT_TRUE(first->identity().valid());
  EXPECT_EQ(duplicate, nullptr);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConcurrentQueueAuthorityIssueHasExactlyOneWinner)
{
  CameraObservationQueue queue;
  std::array<
    std::shared_ptr<livo::AdaptiveVisualSelectionAuthority>,
    2U> issued;

  runTwoWorkersTogether(
    [&queue, &issued](const std::size_t worker_id)
    {
      issued[worker_id] =
        queue.issueAdaptiveVisualSelectionAuthority();
    });

  const std::size_t winner_count =
    static_cast<std::size_t>(issued[0] != nullptr) +
    static_cast<std::size_t>(issued[1] != nullptr);
  EXPECT_EQ(winner_count, 1U);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConcurrentRuntimeInputCopiesGiveOnlyWinnerAConsumablePermit)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(queue, 2U, 35.0, 73U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, 2U);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto lidar_issue =
    issueReadyIdentityLidarGeneration(*authority, 75U);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  const auto direct_capability = authority->issue(
    policy, batch, *policy_capability, *lidar_issue.token);
  ASSERT_EQ(
    direct_capability.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(direct_capability.capability.has_value());
  std::array<
    livo::AdaptiveVisualRuntimeAttemptResult, 2U> results;

  runTwoWorkersTogether(
    [&runtime_input, &results](
      const std::size_t worker_id)
    {
      const auto input_copy = *runtime_input;
      results[worker_id] =
        input_copy.beginRuntimeAttempt();
    });

  const std::size_t ready_count =
    static_cast<std::size_t>(
      results[0].status ==
      livo::AdaptiveVisualRuntimeAttemptStatus::kReady) +
    static_cast<std::size_t>(
      results[1].status ==
      livo::AdaptiveVisualRuntimeAttemptStatus::kReady);
  const std::size_t replay_count =
    static_cast<std::size_t>(
      results[0].status ==
      livo::AdaptiveVisualRuntimeAttemptStatus::
      kAlreadyAttempted) +
    static_cast<std::size_t>(
      results[1].status ==
      livo::AdaptiveVisualRuntimeAttemptStatus::
      kAlreadyAttempted);
  EXPECT_EQ(ready_count, 1U);
  EXPECT_EQ(replay_count, 1U);
  EXPECT_EQ(results[0].runtime_attempt_count, 1U);
  EXPECT_EQ(results[1].runtime_attempt_count, 1U);
  EXPECT_EQ(results[0].selection_consumption_count, 0U);
  EXPECT_EQ(results[1].selection_consumption_count, 0U);

  const std::size_t winner =
    results[0].status ==
      livo::AdaptiveVisualRuntimeAttemptStatus::kReady ?
    0U : 1U;
  const std::size_t loser = 1U - winner;
  ASSERT_TRUE(results[winner].permit.has_value());
  EXPECT_FALSE(results[loser].permit.has_value());

  const auto skipped_issue =
    results[winner].permit->consumeSelection(
    policy, batch, *direct_capability.capability);
  EXPECT_EQ(
    skipped_issue.status,
    livo::AdaptiveVisualSelectionStatus::
    kRuntimeAttemptSelectionNotIssued);
  EXPECT_FALSE(skipped_issue.receipt.has_value());
  EXPECT_FALSE(skipped_issue.consumed_by_this_call);
  EXPECT_EQ(
    skipped_issue.selection_consumption_count, 0U);

  auto selection =
    results[winner].permit->issueSelection(
    policy, batch, *policy_capability);
  ASSERT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(selection.capability.has_value());
  const auto reissue =
    results[winner].permit->issueSelection(
    policy, batch, *policy_capability);
  EXPECT_EQ(
    reissue.status,
    livo::AdaptiveVisualSelectionStatus::
    kRuntimeAttemptSelectionAlreadyIssued);
  EXPECT_FALSE(reissue.capability.has_value());

  const auto consumed =
    results[winner].permit->consumeSelection(
    policy, batch, *selection.capability);
  ASSERT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(consumed.receipt.has_value());
  EXPECT_TRUE(consumed.consumed_by_this_call);
  EXPECT_EQ(consumed.selection_consumption_count, 1U);
  const auto reconsumed =
    results[winner].permit->consumeSelection(
    policy, batch, *selection.capability);
  EXPECT_EQ(
    reconsumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kRuntimeAttemptSelectionAlreadyConsumed);
  EXPECT_FALSE(reconsumed.receipt.has_value());
  EXPECT_FALSE(reconsumed.consumed_by_this_call);
  EXPECT_EQ(reconsumed.selection_consumption_count, 1U);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  CandidatePlanIssueRejectsTimestampMutatedAfterBatchWasSealed)
{
  constexpr std::uint8_t kSelectedCamera = 1U;
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 18.0, 22U);
  const auto observations =
    singleHealthyCameraBoundToBatch(
    batch, kSelectedCamera);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  batch.timestamp = 28.0;

  const auto capability = policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);

  EXPECT_FALSE(capability.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueRejectsLidarTokenFromForeignOwner)
{
  constexpr std::uint8_t kSelectedCamera = 1U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 20.0, 23U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());

  auto expected_owner_storage =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(expected_owner_storage, nullptr);
  auto & expected_owner = *expected_owner_storage;
  CameraObservationQueue foreign_queue;
  auto foreign_owner_storage =
    foreign_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(foreign_owner_storage, nullptr);
  auto & foreign_owner = *foreign_owner_storage;
  const auto foreign_token =
    issueReadyIdentityLidarGeneration(
    foreign_owner, 51U);
  ASSERT_TRUE(foreign_token.token.has_value());

  const auto selection = expected_owner.issue(
    policy, batch, *policy_capability,
    *foreign_token.token);

  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignLidarGenerationOwner);
  EXPECT_FALSE(selection.capability.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueRejectsPolicyHessianNotBoundToLidarGeneration)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 25.0, 29U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const livo::AdaptiveVisualPolicy::Matrix6
    different_lidar_information =
    2.0 * livo::AdaptiveVisualPolicy::Matrix6::Identity();
  const auto lidar_token =
    authority.issueLidarGeneration(
    56U, different_lidar_information, true);
  ASSERT_TRUE(lidar_token.token.has_value());

  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);

  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarInformationIdentityMismatch);
  EXPECT_FALSE(selection.capability.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueRejectsLidarReadinessNotBoundToGeneration)
{
  constexpr std::uint8_t kSelectedCamera = 3U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 26.0, 30U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  ASSERT_TRUE(policy_capability->lidarInformationReady());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    authority.issueLidarGeneration(
    57U,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    false);
  ASSERT_TRUE(lidar_token.token.has_value());

  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);

  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarInformationReadinessMismatch);
  EXPECT_FALSE(selection.capability.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  LidarGenerationIssueTypedRejectsNonFiniteHessianWithoutClaimingGeneration)
{
  CameraObservationQueue queue;
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  livo::AdaptiveVisualPolicy::Matrix6
    non_finite_information =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  non_finite_information(0, 0) =
    std::numeric_limits<double>::quiet_NaN();

  const auto rejected =
    authority.issueLidarGeneration(
    58U, non_finite_information, true);
  const auto recovered =
    authority.issueLidarGeneration(
    58U,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true);

  EXPECT_EQ(
    rejected.status,
    livo::AdaptiveVisualSelectionStatus::
    kInvalidLidarInformation);
  EXPECT_FALSE(rejected.token.has_value());
  EXPECT_EQ(
    recovered.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(recovered.token.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueRejectsCandidateMaskDifferentFromSealedReceipt)
{
  constexpr std::uint8_t kSelectedCamera = 0U;
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 30.0, 31U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  batch.present[1U] = true;
  batch.camera_count = 2U;

  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 61U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);

  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::
    kCameraBatchIdentityMismatch);
  EXPECT_FALSE(selection.capability.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueBindsPolicySelectionToSoleCandidate)
{
  constexpr std::uint8_t kPresentCamera = 0U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kPresentCamera, 40.0, 37U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kPresentCamera);
  ASSERT_TRUE(policy_capability.has_value());
  ASSERT_TRUE(
    policy_capability->decisionPlan().
    cameras[kPresentCamera].enabled);

  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 71U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);

  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(selection.capability.has_value());
  EXPECT_EQ(
    selection.capability->selectedCameraId(),
    kPresentCamera);
  EXPECT_EQ(
    selection.capability->candidateMask(),
    (std::array<bool, 4U>{{
      true, false, false, false}}));
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueBindsPolicyMetricsToSealedObservationAndImageGenerations)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue observation_mismatch_queue;
  const CameraObservationBatch observation_mismatch_batch =
    ownedSingleCameraBatch(
    observation_mismatch_queue, kSelectedCamera,
    45.0, 39U);
  livo::AdaptiveVisualPolicy observation_mismatch_policy(
    singleCameraConfig());
  const auto observation_mismatch_capability =
    ownerSealedPolicyCapability(
    observation_mismatch_queue,
    observation_mismatch_batch,
    observation_mismatch_policy,
    kSelectedCamera, 1U, 0U);
  ASSERT_TRUE(observation_mismatch_capability.has_value());
  auto observation_mismatch_authority_owner =
    observation_mismatch_queue.
    issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(
    observation_mismatch_authority_owner, nullptr);
  auto & observation_mismatch_authority =
    *observation_mismatch_authority_owner;
  const auto observation_mismatch_token =
    issueReadyIdentityLidarGeneration(
    observation_mismatch_authority, 76U);
  ASSERT_TRUE(observation_mismatch_token.token.has_value());

  const auto observation_mismatch =
    observation_mismatch_authority.issue(
    observation_mismatch_policy,
    observation_mismatch_batch,
    *observation_mismatch_capability,
    *observation_mismatch_token.token);

  EXPECT_EQ(
    observation_mismatch.status,
    livo::AdaptiveVisualSelectionStatus::
    kPolicyObservationIdentityMismatch);
  EXPECT_FALSE(
    observation_mismatch.capability.has_value());

  CameraObservationQueue image_mismatch_queue;
  const CameraObservationBatch image_mismatch_batch =
    ownedSingleCameraBatch(
    image_mismatch_queue, kSelectedCamera,
    46.0, 40U);
  livo::AdaptiveVisualPolicy image_mismatch_policy(
    singleCameraConfig());
  const auto image_mismatch_capability =
    ownerSealedPolicyCapability(
    image_mismatch_queue, image_mismatch_batch,
    image_mismatch_policy, kSelectedCamera, 0U, 1U);
  ASSERT_TRUE(image_mismatch_capability.has_value());
  auto image_mismatch_authority_owner =
    image_mismatch_queue.
    issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(image_mismatch_authority_owner, nullptr);
  auto & image_mismatch_authority =
    *image_mismatch_authority_owner;
  const auto image_mismatch_token =
    issueReadyIdentityLidarGeneration(
    image_mismatch_authority, 77U);
  ASSERT_TRUE(image_mismatch_token.token.has_value());

  const auto image_mismatch =
    image_mismatch_authority.issue(
    image_mismatch_policy, image_mismatch_batch,
    *image_mismatch_capability,
    *image_mismatch_token.token);

  EXPECT_EQ(
    image_mismatch.status,
    livo::AdaptiveVisualSelectionStatus::
    kPolicyObservationIdentityMismatch);
  EXPECT_FALSE(image_mismatch.capability.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConsumptionRejectsImageByteMutationAfterSelectionIssuance)
{
  constexpr std::uint8_t kSelectedCamera = 3U;
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 50.0, 41U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 81U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());

  batch.observations[kSelectedCamera].
  image.at<std::uint8_t>(0, 0) = 99U;
  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);

  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kObservationImageIdentityMismatch);
  EXPECT_FALSE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  RevalidationCountsExactlyPresentImagesAcrossBothDigestPartitions)
{
  for (const unsigned int mask : {1U, 8U, 3U, 12U, 5U, 10U, 15U})
  {
    SCOPED_TRACE(mask);
    CameraObservationQueue queue;
    ASSERT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      if ((mask & (1U << camera_id)) == 0U) continue;
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = 51.0 + 0.001 * camera_id;
      observation.image = cv::Mat(8, 12, CV_8UC1, cv::Scalar(41U + camera_id));
      ASSERT_EQ(queue.push(std::move(observation)), CameraObservationPushResult::ACCEPTED);
    }
    CameraObservationBatch batch;
    ASSERT_EQ(
      queue.popAdaptiveSynchronized(51.1, 0.010, 0.030, 1U, batch),
      AdaptiveCameraBatchResult::READY);
    ASSERT_TRUE(batch.validatedReceipt().has_value());
    livo::AdaptiveVisualPolicy policy(singleCameraConfig());
    const auto observations = healthyCandidatesBoundToBatch(batch, 0U);
    for (int attempt = 0; attempt < 2; ++attempt)
    {
      const auto before = batch.validatedReceipt()->validationImageHashCount();
      const auto candidate = policy.issueCandidatePlan(
        batch, livo::AdaptiveVisualPolicy::Matrix6::Identity(), observations, true);
      ASSERT_TRUE(candidate.has_value());
      EXPECT_EQ(
        batch.validatedReceipt()->validationImageHashCount(), before + batch.camera_count);
    }
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  AllCameraDigestsRejectLastPixelMutationAndRehashAfterRestoration)
{
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    SCOPED_TRACE(camera_id);
    CameraObservationQueue queue;
    CameraObservationBatch batch = ownedFourCameraCandidateBatch(queue);
    livo::AdaptiveVisualPolicy policy(singleCameraConfig());
    const auto candidate = policy.issueCandidatePlan(
      batch, livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      fourCandidateQualitiesBoundToBatch(batch), true);
    ASSERT_TRUE(candidate.has_value());
    auto authority = queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar_token = issueReadyIdentityLidarGeneration(*authority, 81U);
    ASSERT_TRUE(lidar_token.token.has_value());
    const auto selection = authority->issue(policy, batch, *candidate, *lidar_token.token);
    ASSERT_TRUE(selection.capability.has_value());
    cv::Mat & image = batch.observations[camera_id].image;
    image.at<std::uint8_t>(image.rows - 1, image.cols - 1) ^= 0x5AU;
    const auto before = batch.validatedReceipt()->validationImageHashCount();
    const auto rejected = authority->consume(
      policy, batch, *selection.capability, *lidar_token.token);
    EXPECT_EQ(rejected.status, livo::AdaptiveVisualSelectionStatus::kObservationImageIdentityMismatch);
    EXPECT_FALSE(rejected.receipt.has_value());
    EXPECT_FALSE(batch.validatedReceipt()->adaptiveVisualSelectionClaimed());
    // OpenMP validates every present image before combining the results.
    EXPECT_EQ(
      batch.validatedReceipt()->validationImageHashCount(), before + batch.camera_count);
    image.at<std::uint8_t>(image.rows - 1, image.cols - 1) ^= 0x5AU;
    const auto after_rejection = batch.validatedReceipt()->validationImageHashCount();
    const auto accepted = authority->consume(
      policy, batch, *selection.capability, *lidar_token.token);
    EXPECT_EQ(accepted.status, livo::AdaptiveVisualSelectionStatus::kReady);
    EXPECT_TRUE(accepted.receipt.has_value());
    EXPECT_EQ(batch.validatedReceipt()->validationImageHashCount(), after_rejection + 4U);
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConsumptionRehashesNonContinuousLogicalRowsAndRejectsLastPixelMutation)
{
  constexpr std::uint8_t kSelectedCamera = 3U;
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedSingleCameraBatch(queue, kSelectedCamera, 50.0, 41U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority = queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto lidar_token = issueReadyIdentityLidarGeneration(*authority, 81U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority->issue(
    policy, batch, *policy_capability, *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());

  const cv::Mat original = batch.observations[kSelectedCamera].image;
  cv::Mat backing(
    original.rows + 2, original.cols + 4, original.type(), cv::Scalar(197U));
  cv::Mat roi = backing(cv::Rect(2, 1, original.cols, original.rows));
  ASSERT_FALSE(roi.isContinuous());
  original.copyTo(roi);
  batch.observations[kSelectedCamera].image = roi;
  roi.at<std::uint8_t>(roi.rows - 1, roi.cols - 1) ^= 0x5AU;
  const auto hashes_before =
    batch.validatedReceipt()->validationImageHashCount();
  const auto rejected = authority->consume(
    policy, batch, *selection.capability, *lidar_token.token);
  EXPECT_EQ(
    rejected.status,
    livo::AdaptiveVisualSelectionStatus::kObservationImageIdentityMismatch);
  EXPECT_FALSE(rejected.receipt.has_value());
  EXPECT_GT(
    batch.validatedReceipt()->validationImageHashCount(), hashes_before);

  roi.at<std::uint8_t>(roi.rows - 1, roi.cols - 1) ^= 0x5AU;
  const auto hashes_after_rejection =
    batch.validatedReceipt()->validationImageHashCount();
  const auto accepted = authority->consume(
    policy, batch, *selection.capability, *lidar_token.token);
  EXPECT_EQ(accepted.status, livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(accepted.receipt.has_value());
  EXPECT_GT(
    batch.validatedReceipt()->validationImageHashCount(), hashes_after_rejection);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConsumptionRejectsImageShapeMutationAfterSelectionIssuance)
{
  constexpr std::uint8_t kSelectedCamera = 0U;
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 60.0, 43U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 91U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());

  batch.observations[kSelectedCamera].image =
    cv::Mat(9, 12, CV_8UC1, cv::Scalar(43U));
  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);

  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kObservationImageIdentityMismatch);
  EXPECT_FALSE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConsumptionRejectsCapabilityAfterPolicyRevisionAdvances)
{
  constexpr std::uint8_t kSelectedCamera = 1U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 70.0, 47U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 101U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  ASSERT_TRUE(
    policy.commitHealthObservation(
      policy_capability->decisionPlan(), true));
  ASSERT_GT(
    policy.revision(),
    policy_capability->basePolicyRevision());

  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);

  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kStalePolicyRevision);
  EXPECT_FALSE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  LidarGenerationIssueRejectsZero)
{
  CameraObservationQueue queue;
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;

  const auto issued =
    issueReadyIdentityLidarGeneration(authority, 0U);

  EXPECT_EQ(
    issued.status,
    livo::AdaptiveVisualSelectionStatus::
    kZeroLidarGeneration);
  EXPECT_FALSE(issued.token.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  LidarGenerationIssueRejectsDuplicateAndRegression)
{
  CameraObservationQueue queue;
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto first =
    issueReadyIdentityLidarGeneration(authority, 110U);
  ASSERT_EQ(
    first.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(first.token.has_value());

  const auto duplicate =
    issueReadyIdentityLidarGeneration(authority, 110U);
  const auto regression =
    issueReadyIdentityLidarGeneration(authority, 109U);

  EXPECT_EQ(
    duplicate.status,
    livo::AdaptiveVisualSelectionStatus::
    kNonMonotonicLidarGeneration);
  EXPECT_FALSE(duplicate.token.has_value());
  EXPECT_EQ(
    regression.status,
    livo::AdaptiveVisualSelectionStatus::
    kNonMonotonicLidarGeneration);
  EXPECT_FALSE(regression.token.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConcurrentDuplicateLidarGenerationIssueHasExactlyOneWinner)
{
  CameraObservationQueue queue;
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  std::array<
    livo::AuthoritativeLidarGenerationIssueResult,
    2U> results;

  runTwoWorkersTogether(
    [&authority, &results](const std::size_t worker_id)
    {
      results[worker_id] =
        issueReadyIdentityLidarGeneration(
        authority, 1101U);
    });

  std::size_t ready_count = 0U;
  std::size_t rejected_count = 0U;
  for (const auto & result : results)
  {
    if (
      result.status ==
      livo::AdaptiveVisualSelectionStatus::kReady &&
      result.token.has_value())
    {
      ++ready_count;
    }
    if (
      result.status ==
      livo::AdaptiveVisualSelectionStatus::
      kNonMonotonicLidarGeneration &&
      !result.token.has_value())
    {
      ++rejected_count;
    }
  }
  EXPECT_EQ(ready_count, 1U);
  EXPECT_EQ(rejected_count, 1U);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConsumptionRejectsLidarTokenAfterNewerGenerationIsIssued)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 80.0, 53U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto stale_token =
    issueReadyIdentityLidarGeneration(authority, 111U);
  ASSERT_TRUE(stale_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *stale_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  const auto current_token =
    issueReadyIdentityLidarGeneration(authority, 112U);
  ASSERT_TRUE(current_token.token.has_value());

  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *stale_token.token);

  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kStaleLidarGeneration);
  EXPECT_FALSE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  CapabilityCopiesShareOneLidarGenerationConsumption)
{
  constexpr std::uint8_t kSelectedCamera = 3U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 90.0, 59U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 121U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  const livo::AdaptiveVisualSelectionCapability copied =
    *selection.capability;

  const auto first = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  const auto replay = authority.consume(
    policy, batch, copied, *lidar_token.token);

  ASSERT_EQ(
    first.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(first.receipt.has_value());
  EXPECT_EQ(
    replay.status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarGenerationAlreadyConsumed);
  EXPECT_FALSE(replay.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConcurrentCandidateCapabilityCopiesProduceExactlyOneReceipt)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedTwoCameraCandidateBatch(queue, false);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto observations =
    twoCandidateQualitiesBoundToBatch(batch);
  const auto policy_capability =
    policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, true);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 122U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  const std::array<
    livo::AdaptiveVisualSelectionCapability, 2U>
    capability_copies{{
      *selection.capability, *selection.capability}};
  std::array<
    livo::AdaptiveVisualSelectionConsumeResult, 2U>
    results;

  runTwoWorkersTogether(
    [&authority, &policy, &batch, &capability_copies,
      &lidar_token, &results](const std::size_t worker_id)
    {
      results[worker_id] = authority.consume(
        policy, batch, capability_copies[worker_id],
        *lidar_token.token);
    });

  std::size_t ready_count = 0U;
  std::size_t rejected_count = 0U;
  for (const auto & result : results)
  {
    if (
      result.status ==
      livo::AdaptiveVisualSelectionStatus::kReady &&
      result.receipt.has_value())
    {
      ++ready_count;
    }
    if (
      result.status ==
      livo::AdaptiveVisualSelectionStatus::
      kLidarGenerationAlreadyConsumed &&
      !result.receipt.has_value())
    {
      ++rejected_count;
    }
  }
  EXPECT_EQ(ready_count, 1U);
  EXPECT_EQ(rejected_count, 1U);
  EXPECT_TRUE(
    batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SealedBatchAndPolicyCapabilityCannotBeConsumedAcrossLidarGenerations)
{
  constexpr std::uint8_t kSelectedCamera = 1U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 95.0, 60U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto first_token =
    issueReadyIdentityLidarGeneration(authority, 126U);
  ASSERT_TRUE(first_token.token.has_value());
  const auto first_selection = authority.issue(
    policy, batch, *policy_capability,
    *first_token.token);
  ASSERT_TRUE(first_selection.capability.has_value());
  const auto first = authority.consume(
    policy, batch, *first_selection.capability,
    *first_token.token);
  ASSERT_EQ(
    first.status,
    livo::AdaptiveVisualSelectionStatus::kReady);

  const auto second_token =
    issueReadyIdentityLidarGeneration(authority, 127U);
  ASSERT_TRUE(second_token.token.has_value());
  const auto second_selection = authority.issue(
    policy, batch, *policy_capability,
    *second_token.token);
  ASSERT_TRUE(second_selection.capability.has_value());
  const auto replay = authority.consume(
    policy, batch, *second_selection.capability,
    *second_token.token);

  EXPECT_EQ(
    replay.status,
    livo::AdaptiveVisualSelectionStatus::
    kCameraBatchAlreadyConsumed);
  EXPECT_FALSE(replay.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionConsumptionGloballyClaimsAllCopiesOfSealedBatchReceipt)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 97.0, 60U);
  ASSERT_TRUE(batch.validatedReceipt().has_value());
  const livo::CameraObservationBatchReceipt receipt_copy =
    *batch.validatedReceipt();
  ASSERT_FALSE(
    receipt_copy.adaptiveVisualSelectionClaimed());
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 128U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());

  const auto consumed = authority.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);

  ASSERT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(consumed.receipt.has_value());
  EXPECT_TRUE(
    batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());
  EXPECT_TRUE(
    receipt_copy.adaptiveVisualSelectionClaimed());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ForeignAuthorityCannotConsumeSelectionCapability)
{
  constexpr std::uint8_t kSelectedCamera = 0U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 100.0, 61U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto expected_owner_storage =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(expected_owner_storage, nullptr);
  auto & expected_owner = *expected_owner_storage;
  CameraObservationQueue foreign_queue;
  auto foreign_owner_storage =
    foreign_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(foreign_owner_storage, nullptr);
  auto & foreign_owner = *foreign_owner_storage;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(
    expected_owner, 131U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = expected_owner.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());

  const auto consumed = foreign_owner.consume(
    policy, batch, *selection.capability,
    *lidar_token.token);

  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignSelectionOwner);
  EXPECT_FALSE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssueRejectsBatchFromForeignQueueOwner)
{
  constexpr std::uint8_t kSelectedCamera = 1U;
  CameraObservationQueue expected_queue;
  CameraObservationQueue foreign_queue;
  const CameraObservationBatch foreign_batch =
    ownedSingleCameraBatch(
    foreign_queue, kSelectedCamera, 110.0, 67U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    foreign_queue, foreign_batch, policy,
    kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    expected_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 141U);
  ASSERT_TRUE(lidar_token.token.has_value());

  const auto selection = authority.issue(
    policy, foreign_batch, *policy_capability,
    *lidar_token.token);

  EXPECT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignCameraBatchOwner);
  EXPECT_FALSE(selection.capability.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConsumptionRejectsForeignPolicyOwnerAtSameRevision)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 120.0, 71U);
  livo::AdaptiveVisualPolicy expected_policy(
    singleCameraConfig());
  livo::AdaptiveVisualPolicy foreign_policy(
    singleCameraConfig());
  ASSERT_EQ(
    expected_policy.revision(),
    foreign_policy.revision());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, expected_policy,
    kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 151U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    expected_policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());

  const auto consumed = authority.consume(
    foreign_policy, batch, *selection.capability,
    *lidar_token.token);

  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignPolicyOwner);
  EXPECT_FALSE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SameLidarGenerationCannotBeConsumedAgainWithDifferentBatchAndCapability)
{
  constexpr std::uint8_t kSelectedCamera = 3U;
  CameraObservationQueue queue;
  const CameraObservationBatch first_batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 130.0, 73U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto first_observations =
    singleHealthyCameraBoundToBatch(
    first_batch, kSelectedCamera);
  const auto first_policy_capability =
    policy.issueCandidatePlan(
    first_batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    first_observations, true);
  ASSERT_TRUE(first_policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 161U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto first_selection = authority.issue(
    policy, first_batch, *first_policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(first_selection.capability.has_value());
  const auto first_consumed = authority.consume(
    policy, first_batch, *first_selection.capability,
    *lidar_token.token);
  ASSERT_EQ(
    first_consumed.status,
    livo::AdaptiveVisualSelectionStatus::kReady);

  const CameraObservationBatch second_batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 131.0, 79U);
  const auto second_observations =
    singleHealthyCameraBoundToBatch(
    second_batch, kSelectedCamera);
  const auto second_policy_capability =
    policy.issueCandidatePlan(
    second_batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    second_observations, true);
  ASSERT_TRUE(second_policy_capability.has_value());
  const auto second_selection = authority.issue(
    policy, second_batch, *second_policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(second_selection.capability.has_value());

  const auto replay = authority.consume(
    policy, second_batch, *second_selection.capability,
    *lidar_token.token);

  EXPECT_EQ(
    replay.status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarGenerationAlreadyConsumed);
  EXPECT_FALSE(replay.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ConsumptionRejectsDifferentSealedBatchBeforeGenerationClaim)
{
  constexpr std::uint8_t kSelectedCamera = 0U;
  CameraObservationQueue queue;
  const CameraObservationBatch selected_batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 140.0, 83U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, selected_batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 171U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, selected_batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());
  const CameraObservationBatch swapped_batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 141.0, 89U);

  const auto consumed = authority.consume(
    policy, swapped_batch, *selection.capability,
    *lidar_token.token);

  EXPECT_EQ(
    consumed.status,
    livo::AdaptiveVisualSelectionStatus::
    kCameraBatchIdentityMismatch);
  EXPECT_FALSE(consumed.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionIssuePreservesTypedStatusPrecedenceForSimultaneousFailures)
{
  constexpr std::uint8_t kSelectedCamera = 0U;
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 150.0, 91U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;

  CameraObservationQueue foreign_queue;
  auto foreign_authority_owner =
    foreign_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(foreign_authority_owner, nullptr);
  const auto foreign_lidar_token =
    issueReadyIdentityLidarGeneration(
    *foreign_authority_owner, 181U);
  ASSERT_TRUE(foreign_lidar_token.token.has_value());

  CameraObservationBatch invalid_present_batch = batch;
  invalid_present_batch.present[1U] = true;
  const auto foreign_lidar_result = authority.issue(
    policy, invalid_present_batch, *policy_capability,
    *foreign_lidar_token.token);
  EXPECT_EQ(
    foreign_lidar_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignLidarGenerationOwner);

  const auto readiness_mismatch_token =
    authority.issueLidarGeneration(
    182U,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    false);
  ASSERT_TRUE(readiness_mismatch_token.token.has_value());
  livo::AdaptiveVisualPolicy foreign_policy(
    singleCameraConfig());
  const auto foreign_policy_result = authority.issue(
    foreign_policy, invalid_present_batch,
    *policy_capability, *readiness_mismatch_token.token);
  EXPECT_EQ(
    foreign_policy_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignPolicyOwner);

  const auto readiness_result = authority.issue(
    policy, invalid_present_batch, *policy_capability,
    *readiness_mismatch_token.token);
  EXPECT_EQ(
    readiness_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarInformationReadinessMismatch);

  const auto ready_token =
    issueReadyIdentityLidarGeneration(authority, 183U);
  ASSERT_TRUE(ready_token.token.has_value());
  const auto invalid_present_result = authority.issue(
    policy, invalid_present_batch, *policy_capability,
    *ready_token.token);
  EXPECT_EQ(
    invalid_present_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kInvalidCandidatePresentMask);

  CameraObservationBatch identity_mismatch_batch = batch;
  identity_mismatch_batch.present[1U] = true;
  identity_mismatch_batch.camera_count = 2U;
  identity_mismatch_batch.timestamp += 1.0;
  const auto identity_mismatch_result = authority.issue(
    policy, identity_mismatch_batch, *policy_capability,
    *ready_token.token);
  EXPECT_EQ(
    identity_mismatch_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kCameraBatchIdentityMismatch);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  SelectionConsumePreservesTypedStatusPrecedenceAndRevalidatesBytes)
{
  constexpr std::uint8_t kSelectedCamera = 2U;
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kSelectedCamera, 160.0, 97U);
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    ownerSealedPolicyCapability(
    queue, batch, policy, kSelectedCamera);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority_owner =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority_owner, nullptr);
  auto & authority = *authority_owner;
  const auto lidar_token =
    issueReadyIdentityLidarGeneration(authority, 191U);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority.issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_TRUE(selection.capability.has_value());

  CameraObservationQueue foreign_queue;
  auto foreign_authority_owner =
    foreign_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(foreign_authority_owner, nullptr);
  const auto foreign_lidar_token =
    issueReadyIdentityLidarGeneration(
    *foreign_authority_owner, 191U);
  ASSERT_TRUE(foreign_lidar_token.token.has_value());
  const auto foreign_lidar_result = authority.consume(
    policy, batch, *selection.capability,
    *foreign_lidar_token.token);
  EXPECT_EQ(
    foreign_lidar_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignLidarGenerationOwner);

  const auto different_lidar_token =
    issueReadyIdentityLidarGeneration(authority, 192U);
  ASSERT_TRUE(different_lidar_token.token.has_value());
  livo::AdaptiveVisualPolicy foreign_policy(
    singleCameraConfig());
  CameraObservationBatch invalid_batch = batch;
  invalid_batch.present[0U] = true;
  const auto token_mismatch_result = authority.consume(
    foreign_policy, invalid_batch, *selection.capability,
    *different_lidar_token.token);
  EXPECT_EQ(
    token_mismatch_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarTokenMismatch);

  const auto foreign_policy_result = authority.consume(
    foreign_policy, invalid_batch, *selection.capability,
    *lidar_token.token);
  EXPECT_EQ(
    foreign_policy_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignPolicyOwner);

  ASSERT_TRUE(
    policy.commitHealthObservation(
      policy_capability->decisionPlan(), true));
  const auto stale_policy_result = authority.consume(
    policy, invalid_batch, *selection.capability,
    *lidar_token.token);
  EXPECT_EQ(
    stale_policy_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kStalePolicyRevision);

  CameraObservationQueue revalidation_queue;
  CameraObservationBatch revalidation_batch =
    ownedSingleCameraBatch(
    revalidation_queue, kSelectedCamera, 170.0, 101U);
  livo::AdaptiveVisualPolicy revalidation_policy(
    singleCameraConfig());
  const auto revalidation_policy_capability =
    ownerSealedPolicyCapability(
    revalidation_queue, revalidation_batch,
    revalidation_policy, kSelectedCamera);
  ASSERT_TRUE(revalidation_policy_capability.has_value());
  auto revalidation_authority_owner =
    revalidation_queue.
    issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(revalidation_authority_owner, nullptr);
  auto & revalidation_authority =
    *revalidation_authority_owner;
  const auto revalidation_lidar_token =
    issueReadyIdentityLidarGeneration(
    revalidation_authority, 201U);
  ASSERT_TRUE(revalidation_lidar_token.token.has_value());
  const auto revalidation_selection =
    revalidation_authority.issue(
    revalidation_policy, revalidation_batch,
    *revalidation_policy_capability,
    *revalidation_lidar_token.token);
  ASSERT_TRUE(revalidation_selection.capability.has_value());
  ASSERT_TRUE(
    revalidation_batch.validatedReceipt().has_value());
  const std::uint64_t validation_hashes_before =
    revalidation_batch.validatedReceipt()->
    validationImageHashCount();
  revalidation_batch.observations[kSelectedCamera].
  image.at<std::uint8_t>(0, 0) = 103U;
  const auto newer_lidar_token =
    issueReadyIdentityLidarGeneration(
    revalidation_authority, 202U);
  ASSERT_TRUE(newer_lidar_token.token.has_value());

  const auto revalidation_result =
    revalidation_authority.consume(
    revalidation_policy, revalidation_batch,
    *revalidation_selection.capability,
    *revalidation_lidar_token.token);

  EXPECT_EQ(
    revalidation_result.status,
    livo::AdaptiveVisualSelectionStatus::
    kObservationImageIdentityMismatch);
  EXPECT_FALSE(revalidation_result.receipt.has_value());
  EXPECT_FALSE(
    revalidation_batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());
  EXPECT_EQ(
    revalidation_batch.validatedReceipt()->
    validationImageHashCount(),
    validation_hashes_before + 1U);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ReceiptRemainsCompleteAfterAllIssuingObjectsAreDestroyed)
{
  std::optional<livo::AdaptiveVisualSelectionReceipt>
    retained_receipt;
  livo::AdaptiveVisualSelectionAuthorityIdentity
    retained_identity;
  std::array<bool, livo::kAdaptiveVisualCameraCount>
    expected_candidate_mask{};
  std::array<bool, livo::kAdaptiveVisualCameraCount>
    expected_selected_mask{};
  std::array<std::uint64_t, livo::kAdaptiveVisualCameraCount>
    expected_observation_generations{};
  std::array<std::uint64_t, livo::kAdaptiveVisualCameraCount>
    expected_image_generations{};
  std::array<double, livo::kAdaptiveVisualCameraCount>
    expected_exposure_timestamps;
  expected_exposure_timestamps.fill(
    std::numeric_limits<double>::quiet_NaN());
  std::uint64_t expected_lidar_generation = 0U;
  std::uint64_t expected_batch_sequence = 0U;
  std::uint64_t expected_policy_revision = 0U;
  std::uint8_t expected_selected_camera =
    static_cast<std::uint8_t>(
    livo::kAdaptiveVisualCameraCount);
  std::size_t expected_candidate_budget = 0U;
  std::size_t expected_patch_budget = 0U;
  livo::AdaptiveSelectionReason expected_reason =
    livo::AdaptiveSelectionReason::kNotApplicable;

  {
    CameraObservationQueue queue;
    const CameraObservationBatch batch =
      ownedTwoCameraCandidateBatch(queue, false);
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto policy_capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      twoCandidateQualitiesBoundToBatch(batch), true);
    ASSERT_TRUE(policy_capability.has_value());
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    retained_identity = authority->identity();
    const auto lidar_token =
      issueReadyIdentityLidarGeneration(
      *authority, 211U);
    ASSERT_TRUE(lidar_token.token.has_value());
    const auto selection = authority->issue(
      policy, batch, *policy_capability,
      *lidar_token.token);
    ASSERT_TRUE(selection.capability.has_value());
    const auto consumed = authority->consume(
      policy, batch, *selection.capability,
      *lidar_token.token);
    ASSERT_EQ(
      consumed.status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(consumed.receipt.has_value());

    const auto & receipt = *consumed.receipt;
    expected_candidate_mask = receipt.candidateMask();
    expected_selected_mask = receipt.selectedMask();
    expected_lidar_generation =
      receipt.lidarGeneration();
    expected_batch_sequence =
      receipt.cameraBatchSequence();
    expected_policy_revision =
      receipt.policyRevision();
    expected_selected_camera =
      receipt.selectedCameraId();
    expected_candidate_budget =
      receipt.candidateBudget();
    expected_patch_budget = receipt.patchBudget();
    expected_reason = receipt.selectionReason();
    for (std::size_t camera_id = 0U;
         camera_id < livo::kAdaptiveVisualCameraCount;
         ++camera_id)
    {
      const auto candidate_id =
        static_cast<std::uint8_t>(camera_id);
      expected_observation_generations[camera_id] =
        receipt.candidateObservationGeneration(
        candidate_id);
      expected_image_generations[camera_id] =
        receipt.candidateImageGeneration(candidate_id);
      expected_exposure_timestamps[camera_id] =
        receipt.candidateExposureTimestampSec(
        candidate_id);
    }
    retained_receipt = receipt;
  }

  ASSERT_TRUE(retained_identity.valid());
  ASSERT_TRUE(retained_receipt.has_value());
  ASSERT_TRUE(retained_receipt->valid());
  EXPECT_TRUE(
    retained_receipt->issuedBy(retained_identity));
  EXPECT_EQ(
    retained_receipt->selectionStatus(),
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_EQ(
    retained_receipt->candidateMask(),
    expected_candidate_mask);
  EXPECT_EQ(
    retained_receipt->selectedMask(),
    expected_selected_mask);
  EXPECT_EQ(
    retained_receipt->lidarGeneration(),
    expected_lidar_generation);
  EXPECT_EQ(
    retained_receipt->cameraBatchSequence(),
    expected_batch_sequence);
  EXPECT_EQ(
    retained_receipt->policyRevision(),
    expected_policy_revision);
  EXPECT_EQ(
    retained_receipt->selectedCameraId(),
    expected_selected_camera);
  EXPECT_EQ(
    retained_receipt->candidateBudget(),
    expected_candidate_budget);
  EXPECT_EQ(
    retained_receipt->patchBudget(),
    expected_patch_budget);
  EXPECT_EQ(
    retained_receipt->selectionReason(),
    expected_reason);
  for (std::size_t camera_id = 0U;
       camera_id < livo::kAdaptiveVisualCameraCount;
       ++camera_id)
  {
    const auto candidate_id =
      static_cast<std::uint8_t>(camera_id);
    EXPECT_EQ(
      retained_receipt->
      candidateObservationGeneration(candidate_id),
      expected_observation_generations[camera_id]);
    EXPECT_EQ(
      retained_receipt->
      candidateImageGeneration(candidate_id),
      expected_image_generations[camera_id]);
    if (std::isfinite(
        expected_exposure_timestamps[camera_id]))
    {
      EXPECT_DOUBLE_EQ(
        retained_receipt->
        candidateExposureTimestampSec(candidate_id),
        expected_exposure_timestamps[camera_id]);
    }
    else
    {
      EXPECT_TRUE(std::isnan(
        retained_receipt->
        candidateExposureTimestampSec(candidate_id)));
    }
  }
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ZeroSelectionPermitCommitsRecoveryStateOnceAndSharesGenerationLedger)
{
  constexpr std::uint8_t kTargetCamera = 0U;
  auto config = singleCameraConfig();
  config.on_hysteresis_frames = 2U;
  CameraObservationQueue queue;
  livo::AdaptiveVisualPolicy policy(config);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  isolateCameraThroughSelectedCommit(
    queue, policy, *authority, 301U, 201.0);

  const CameraObservationBatch recovery_batch =
    ownedSingleCameraBatch(
    queue, kTargetCamera, 202.0, 111U);
  const auto recovery_observations =
    zeroSelectionCameraBoundToBatch(
    recovery_batch, kTargetCamera);
  const auto recovery_capability =
    policy.issueCandidatePlan(
    recovery_batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    recovery_observations, true);
  ASSERT_TRUE(recovery_capability.has_value());
  EXPECT_TRUE(
    std::none_of(
      recovery_capability->decisionPlan().cameras.begin(),
      recovery_capability->decisionPlan().cameras.end(),
      [](const livo::AdaptiveCameraDecision & decision)
      {
        return decision.enabled;
      }));
  ASSERT_TRUE(
    recovery_capability->decisionPlan().
    proposed_hysteresis[kTargetCamera].isolated);
  EXPECT_EQ(
    recovery_capability->decisionPlan().
    proposed_hysteresis[kTargetCamera].
    consecutive_healthy,
    1U);
  const auto lidar =
    issueReadyIdentityLidarGeneration(*authority, 302U);
  ASSERT_TRUE(lidar.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar.token);
  ASSERT_TRUE(runtime_input.has_value());
  EXPECT_EQ(
    runtime_input->preflightBatch(recovery_batch),
    livo::AdaptiveVisualSelectionStatus::kReady);
  auto attempt = runtime_input->beginRuntimeAttempt();
  ASSERT_EQ(
    attempt.status,
    livo::AdaptiveVisualRuntimeAttemptStatus::kReady);
  ASSERT_TRUE(attempt.permit.has_value());
  const auto issue = attempt.permit->issueSelection(
    policy, recovery_batch, *recovery_capability);
  ASSERT_EQ(
    issue.status,
    livo::AdaptiveVisualSelectionStatus::kNoSelectedCamera);
  ASSERT_FALSE(issue.capability.has_value());
  ASSERT_FALSE(
    recovery_batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());
  const std::uint64_t revision_before = policy.revision();
  const auto expected_hysteresis =
    recovery_capability->decisionPlan().
    proposed_hysteresis;
  const auto expected_primary =
    recovery_capability->decisionPlan().
    proposed_primary_camera;
  const auto expected_challenger =
    recovery_capability->decisionPlan().
    proposed_selection_challenger;

  const auto committed =
    policy.commitZeroSelectionHealthObservation(
    *recovery_capability, std::move(*attempt.permit));

  EXPECT_EQ(
    committed,
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_EQ(policy.revision(), revision_before + 1U);
  expectHysteresisEqual(
    policy.hysteresis(), expected_hysteresis);
  livo::AdaptiveCameraObservations invalid_observations;
  for (std::size_t camera_id = 0U;
       camera_id < invalid_observations.size();
       ++camera_id)
  {
    invalid_observations[camera_id].camera_id =
      static_cast<std::uint8_t>(camera_id);
  }
  invalid_observations[0U].camera_id = 3U;
  const auto committed_state = policy.plan(
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    invalid_observations, true);
  EXPECT_EQ(
    committed_state.proposed_primary_camera,
    expected_primary);
  EXPECT_EQ(
    committed_state.proposed_selection_challenger.camera_id,
    expected_challenger.camera_id);
  EXPECT_EQ(
    committed_state.proposed_selection_challenger.
    consecutive_confirmations,
    expected_challenger.consecutive_confirmations);
  EXPECT_FALSE(
    recovery_batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());

  const auto revision_after_commit = policy.revision();
  const auto replay =
    policy.commitZeroSelectionHealthObservation(
    *recovery_capability, std::move(*attempt.permit));
  EXPECT_NE(
    replay,
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_EQ(policy.revision(), revision_after_commit);
  expectHysteresisEqual(
    policy.hysteresis(), expected_hysteresis);

  CameraObservationQueue selected_queue;
  const CameraObservationBatch selected_batch =
    ownedSingleCameraBatch(
    selected_queue, 2U, 203.0, 113U);
  const auto selected_capability =
    ownerSealedPolicyCapability(
    selected_queue, selected_batch, policy, 2U);
  ASSERT_TRUE(selected_capability.has_value());
  auto selected_authority =
    selected_queue.
    issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(selected_authority, nullptr);
  const auto selected_receipt =
    consumeCandidateCapability(
    policy, selected_batch, *selected_capability,
    *selected_authority, 302U);
  ASSERT_TRUE(selected_receipt.has_value());
  EXPECT_EQ(
    policy.commitHealthObservation(
      *selected_capability, *selected_receipt, true),
    livo::AdaptiveCandidateHealthCommitStatus::
    kNonMonotonicOrReplayedLidarGeneration);
  EXPECT_EQ(policy.revision(), revision_after_commit);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ZeroSelectionCommitRejectsForeignCapabilityAndBurnsPermit)
{
  constexpr std::uint8_t kTargetCamera = 0U;
  auto config = singleCameraConfig();
  config.on_hysteresis_frames = 2U;
  CameraObservationQueue queue;
  livo::AdaptiveVisualPolicy policy(config);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  isolateCameraThroughSelectedCommit(
    queue, policy, *authority, 311U, 211.0);
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kTargetCamera, 212.0, 115U);
  const auto capability =
    policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    zeroSelectionCameraBoundToBatch(
      batch, kTargetCamera),
    true);
  ASSERT_TRUE(capability.has_value());
  const auto lidar =
    issueReadyIdentityLidarGeneration(*authority, 312U);
  ASSERT_TRUE(lidar.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar.token);
  ASSERT_TRUE(runtime_input.has_value());
  auto attempt = runtime_input->beginRuntimeAttempt();
  ASSERT_TRUE(attempt.permit.has_value());
  const auto issue = attempt.permit->issueSelection(
    policy, batch, *capability);
  ASSERT_EQ(
    issue.status,
    livo::AdaptiveVisualSelectionStatus::kNoSelectedCamera);

  CameraObservationQueue foreign_queue;
  const CameraObservationBatch foreign_batch =
    ownedSingleCameraBatch(
    foreign_queue, 2U, 213.0, 117U);
  livo::AdaptiveVisualPolicy foreign_policy(config);
  const auto foreign_capability =
    ownerSealedPolicyCapability(
    foreign_queue, foreign_batch,
    foreign_policy, 2U);
  ASSERT_TRUE(foreign_capability.has_value());
  const std::uint64_t revision_before = policy.revision();
  const auto hysteresis_before = policy.hysteresis();

  EXPECT_EQ(
    policy.commitZeroSelectionHealthObservation(
      *foreign_capability, std::move(*attempt.permit)),
    livo::AdaptiveCandidateHealthCommitStatus::
    kInvalidOrForeignCapability);
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);
  EXPECT_NE(
    policy.commitZeroSelectionHealthObservation(
      *capability, std::move(*attempt.permit)),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  MovedFromZeroSelectionPermitRejectsWithoutInvalidatingMovedPermit)
{
  constexpr std::uint8_t kTargetCamera = 0U;
  auto config = singleCameraConfig();
  config.on_hysteresis_frames = 2U;
  CameraObservationQueue queue;
  livo::AdaptiveVisualPolicy policy(config);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  isolateCameraThroughSelectedCommit(
    queue, policy, *authority, 321U, 221.0);
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(
    queue, kTargetCamera, 222.0, 119U);
  const auto capability =
    policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    zeroSelectionCameraBoundToBatch(
      batch, kTargetCamera),
    true);
  ASSERT_TRUE(capability.has_value());
  const auto lidar =
    issueReadyIdentityLidarGeneration(*authority, 322U);
  ASSERT_TRUE(lidar.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar.token);
  ASSERT_TRUE(runtime_input.has_value());
  auto attempt = runtime_input->beginRuntimeAttempt();
  ASSERT_TRUE(attempt.permit.has_value());
  ASSERT_EQ(
    attempt.permit->issueSelection(
      policy, batch, *capability).status,
    livo::AdaptiveVisualSelectionStatus::kNoSelectedCamera);
  auto moved_permit = std::move(*attempt.permit);
  const std::uint64_t revision_before = policy.revision();

  EXPECT_NE(
    policy.commitZeroSelectionHealthObservation(
      *capability, std::move(*attempt.permit)),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_EQ(policy.revision(), revision_before);
  EXPECT_EQ(
    policy.commitZeroSelectionHealthObservation(
      *capability, std::move(moved_permit)),
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_EQ(policy.revision(), revision_before + 1U);
}

TEST(
  AdaptiveVisualSelectionCapabilityTest,
  ZeroSelectionHealthAuthorizationRejectsStaleSelectedAndInvalidInputs)
{
  {
    auto config = singleCameraConfig();
    config.on_hysteresis_frames = 2U;
    CameraObservationQueue queue;
    livo::AdaptiveVisualPolicy policy(config);
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    isolateCameraThroughSelectedCommit(
      queue, policy, *authority, 331U, 231.0);
    const CameraObservationBatch batch =
      ownedSingleCameraBatch(queue, 0U, 232.0, 121U);
    const auto capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      zeroSelectionCameraBoundToBatch(batch, 0U),
      true);
    ASSERT_TRUE(capability.has_value());
    const auto lidar =
      issueReadyIdentityLidarGeneration(*authority, 332U);
    ASSERT_TRUE(lidar.token.has_value());
    const auto runtime_input =
      authority->bindRuntimeInput(*lidar.token);
    ASSERT_TRUE(runtime_input.has_value());
    auto attempt = runtime_input->beginRuntimeAttempt();
    ASSERT_TRUE(attempt.permit.has_value());
    ASSERT_EQ(
      attempt.permit->issueSelection(
        policy, batch, *capability).status,
      livo::AdaptiveVisualSelectionStatus::
      kNoSelectedCamera);
    ASSERT_TRUE(
      policy.commitHealthObservation(
        capability->decisionPlan(), true));
    const std::uint64_t stale_revision = policy.revision();
    const auto stale_hysteresis = policy.hysteresis();

    EXPECT_EQ(
      policy.commitZeroSelectionHealthObservation(
        *capability, std::move(*attempt.permit)),
      livo::AdaptiveCandidateHealthCommitStatus::
      kStalePolicyRevision);
    EXPECT_EQ(policy.revision(), stale_revision);
    expectHysteresisEqual(
      policy.hysteresis(), stale_hysteresis);
  }

  {
    CameraObservationQueue queue;
    const CameraObservationBatch batch =
      ownedSingleCameraBatch(queue, 2U, 233.0, 123U);
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto capability =
      ownerSealedPolicyCapability(
      queue, batch, policy, 2U);
    ASSERT_TRUE(capability.has_value());
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar =
      issueReadyIdentityLidarGeneration(*authority, 333U);
    ASSERT_TRUE(lidar.token.has_value());
    const auto runtime_input =
      authority->bindRuntimeInput(*lidar.token);
    ASSERT_TRUE(runtime_input.has_value());
    auto attempt = runtime_input->beginRuntimeAttempt();
    ASSERT_TRUE(attempt.permit.has_value());
    const auto issue = attempt.permit->issueSelection(
      policy, batch, *capability);
    ASSERT_EQ(
      issue.status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(issue.capability.has_value());

    EXPECT_EQ(
      policy.commitZeroSelectionHealthObservation(
        *capability, std::move(*attempt.permit)),
      livo::AdaptiveCandidateHealthCommitStatus::
      kBindingMismatch);
    EXPECT_EQ(policy.revision(), 0U);
    EXPECT_EQ(
      attempt.permit->consumeSelection(
        policy, batch, *issue.capability).status,
      livo::AdaptiveVisualSelectionStatus::
      kInvalidRuntimeAttemptPermit);
    EXPECT_FALSE(
      batch.validatedReceipt()->
      adaptiveVisualSelectionClaimed());
  }

  {
    CameraObservationQueue queue;
    CameraObservationBatch batch =
      ownedSingleCameraBatch(queue, 0U, 234.0, 125U);
    auto observations =
      singleHealthyCameraBoundToBatch(batch, 0U);
    observations[0U].saturation_fraction = 0.95;
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      observations, false);
    ASSERT_TRUE(capability.has_value());
    ASSERT_FALSE(
      capability->decisionPlan().
      lidar_information_valid);
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar = authority->issueLidarGeneration(
      334U,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      false);
    ASSERT_TRUE(lidar.token.has_value());
    const auto runtime_input =
      authority->bindRuntimeInput(*lidar.token);
    ASSERT_TRUE(runtime_input.has_value());
    auto attempt = runtime_input->beginRuntimeAttempt();
    ASSERT_TRUE(attempt.permit.has_value());
    EXPECT_EQ(
      attempt.permit->issueSelection(
        policy, batch, *capability).status,
      livo::AdaptiveVisualSelectionStatus::
      kLidarInformationNotReady);
    EXPECT_NE(
      policy.commitZeroSelectionHealthObservation(
        *capability, std::move(*attempt.permit)),
      livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
    EXPECT_EQ(policy.revision(), 0U);
  }

  {
    CameraObservationQueue queue;
    CameraObservationBatch batch =
      ownedSingleCameraBatch(queue, 0U, 235.0, 127U);
    auto observations =
      singleHealthyCameraBoundToBatch(batch, 0U);
    observations[0U].saturation_fraction = 0.95;
    livo::AdaptiveVisualPolicy policy(
      singleCameraConfig());
    const auto capability =
      policy.issueCandidatePlan(
      batch,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      observations, true);
    ASSERT_TRUE(capability.has_value());
    batch.observations[0U].
    image.at<std::uint8_t>(0, 0) = 129U;
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar =
      issueReadyIdentityLidarGeneration(*authority, 335U);
    ASSERT_TRUE(lidar.token.has_value());
    const auto runtime_input =
      authority->bindRuntimeInput(*lidar.token);
    ASSERT_TRUE(runtime_input.has_value());
    auto attempt = runtime_input->beginRuntimeAttempt();
    ASSERT_TRUE(attempt.permit.has_value());
    EXPECT_EQ(
      attempt.permit->issueSelection(
        policy, batch, *capability).status,
      livo::AdaptiveVisualSelectionStatus::
      kObservationImageIdentityMismatch);
    EXPECT_NE(
      policy.commitZeroSelectionHealthObservation(
        *capability, std::move(*attempt.permit)),
      livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
    EXPECT_EQ(policy.revision(), 0U);
  }
}

}  // namespace
