#include "adaptive_visual_failure_injection_test_support.h"
#include "adaptive_visual_selection_authority.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <utility>

namespace
{

std::atomic<bool> fail_next_global_allocation{false};

bool consumeAllocationFailure() noexcept
{
  return fail_next_global_allocation.exchange(
    false, std::memory_order_acq_rel);
}

void * allocateBytes(
  const std::size_t size,
  const std::size_t alignment = alignof(std::max_align_t))
{
  if (consumeAllocationFailure())
  {
    throw std::bad_alloc();
  }
  void * allocation = nullptr;
  const std::size_t nonzero_size = size == 0U ? 1U : size;
  if (alignment <= alignof(std::max_align_t))
  {
    allocation = std::malloc(nonzero_size);
  }
  else if (
    posix_memalign(
      &allocation, alignment, nonzero_size) != 0)
  {
    allocation = nullptr;
  }
  if (allocation == nullptr)
  {
    throw std::bad_alloc();
  }
  return allocation;
}

void failNextGlobalAllocation() noexcept
{
  fail_next_global_allocation.store(
    true, std::memory_order_release);
}

}  // namespace

void * operator new(const std::size_t size)
{
  return allocateBytes(size);
}

void * operator new[](const std::size_t size)
{
  return allocateBytes(size);
}

void * operator new(
  const std::size_t size, const std::align_val_t alignment)
{
  return allocateBytes(
    size, static_cast<std::size_t>(alignment));
}

void * operator new[](
  const std::size_t size, const std::align_val_t alignment)
{
  return allocateBytes(
    size, static_cast<std::size_t>(alignment));
}

void operator delete(void * allocation) noexcept
{
  std::free(allocation);
}

void operator delete[](void * allocation) noexcept
{
  std::free(allocation);
}

void operator delete(
  void * allocation, const std::size_t) noexcept
{
  std::free(allocation);
}

void operator delete[](
  void * allocation, const std::size_t) noexcept
{
  std::free(allocation);
}

void operator delete(
  void * allocation, const std::align_val_t) noexcept
{
  std::free(allocation);
}

void operator delete[](
  void * allocation, const std::align_val_t) noexcept
{
  std::free(allocation);
}

void operator delete(
  void * allocation, const std::size_t,
  const std::align_val_t) noexcept
{
  std::free(allocation);
}

void operator delete[](
  void * allocation, const std::size_t,
  const std::align_val_t) noexcept
{
  std::free(allocation);
}

namespace
{

livo::AdaptiveVisualPolicyConfig singleCameraConfig()
{
  livo::AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  return config;
}

CameraObservationBatch ownedSingleCameraBatch(
  CameraObservationQueue & queue)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  CameraObservation observation;
  observation.camera_id = 0U;
  observation.timestamp = 11.0;
  observation.image =
    cv::Mat(8, 12, CV_8UC1, cv::Scalar(31U));
  EXPECT_EQ(
    queue.push(std::move(observation)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      11.1, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

livo::AdaptiveCameraObservations healthyObservationsBoundToBatch(
  const CameraObservationBatch & batch)
{
  livo::AdaptiveCameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size();
       ++camera_id)
  {
    observations[camera_id].camera_id =
      static_cast<std::uint8_t>(camera_id);
  }

  auto & selected = observations[0U];
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

  EXPECT_TRUE(batch.validatedReceipt().has_value());
  if (batch.validatedReceipt().has_value())
  {
    selected.observation_generation =
      batch.validatedReceipt()->observationGeneration(0U);
    selected.image_generation =
      batch.validatedReceipt()->imageGeneration(0U);
  }
  return observations;
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
  AdaptiveVisualSelectionReceiptAllocationFailureTest,
  CandidateAllocationFailurePreservesUnclaimedBatchAndRejectsTampering)
{
  for (const bool tamper : {false, true})
  {
    SCOPED_TRACE(tamper);
    CameraObservationQueue queue;
    ASSERT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = 11.0 + 0.001 * camera_id;
      observation.image = cv::Mat(8, 12, CV_8UC1, cv::Scalar(31U + camera_id));
      ASSERT_EQ(queue.push(std::move(observation)), CameraObservationPushResult::ACCEPTED);
    }
    CameraObservationBatch batch;
    ASSERT_TRUE(queue.popSynchronized(11.1, 0.010, batch));
    ASSERT_TRUE(batch.validatedReceipt().has_value());
    auto observations = healthyObservationsBoundToBatch(batch);
    for (std::uint8_t camera_id = 1U; camera_id < 4U; ++camera_id)
    {
      observations[camera_id] = observations[0U];
      observations[camera_id].camera_id = camera_id;
      observations[camera_id].observation_generation =
        batch.validatedReceipt()->observationGeneration(camera_id);
      observations[camera_id].image_generation =
        batch.validatedReceipt()->imageGeneration(camera_id);
    }
    livo::AdaptiveVisualPolicy policy(singleCameraConfig());
    cv::Mat & image = batch.observations[3U].image;
    if (tamper) image.at<std::uint8_t>(image.rows - 1, image.cols - 1) ^= 0x5AU;
    const auto before = batch.validatedReceipt()->validationImageHashCount();
    std::optional<livo::AdaptiveVisualCandidatePlanCapability> candidate;
    bool allocation_escaped = false;
    // Digest validation allocates no async shared state. A valid batch reaches
    // candidate allocation; a tampered batch returns before that allocation.
    failNextGlobalAllocation();
    try
    {
      candidate = policy.issueCandidatePlan(
        batch, livo::AdaptiveVisualPolicy::Matrix6::Identity(), observations, true);
    }
    catch (const std::bad_alloc &)
    {
      allocation_escaped = true;
    }
    const bool failure_consumed = !fail_next_global_allocation.exchange(false);
    EXPECT_EQ(failure_consumed, !tamper);
    EXPECT_EQ(allocation_escaped, !tamper);
    EXPECT_FALSE(candidate.has_value());
    EXPECT_FALSE(batch.validatedReceipt()->adaptiveVisualSelectionClaimed());
    EXPECT_EQ(policy.revision(), 0U);
    EXPECT_EQ(batch.validatedReceipt()->validationImageHashCount(), before + 4U);
    const auto retried = policy.issueCandidatePlan(
      batch, livo::AdaptiveVisualPolicy::Matrix6::Identity(), observations, true);
    EXPECT_EQ(retried.has_value(), !tamper);
  }
}

TEST(
  AdaptiveVisualSelectionReceiptAllocationFailureTest,
  LidarGenerationAllocationFailureLeavesGenerationRetryable)
{
  CameraObservationQueue queue;
  ASSERT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto information =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();

  bool allocation_failed = false;
  failNextGlobalAllocation();
  try
  {
    (void)authority->issueLidarGeneration(
      8U, information, true);
  }
  catch (const std::bad_alloc &)
  {
    allocation_failed = true;
  }
  EXPECT_TRUE(allocation_failed);

  const auto retried = authority->issueLidarGeneration(
    8U, information, true);
  EXPECT_EQ(
    retried.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(retried.token.has_value());
}

TEST(
  AdaptiveVisualSelectionReceiptAllocationFailureTest,
  AllocationFailureLeavesSelectionAuthorityRetryable)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(queue);
  ASSERT_TRUE(batch.validatedReceipt().has_value());
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    healthyObservationsBoundToBatch(batch), true);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto lidar_token = authority->issueLidarGeneration(
    9U,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true);
  ASSERT_EQ(
    lidar_token.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto selection = authority->issue(
    policy, batch, *policy_capability,
    *lidar_token.token);
  ASSERT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(selection.capability.has_value());
  const auto revision_before = policy.revision();
  const auto hysteresis_before = policy.hysteresis();

  livo::test::failNextSelectionReceiptAllocation();
  EXPECT_THROW(
    authority->consume(
      policy, batch, *selection.capability,
      *lidar_token.token),
    std::bad_alloc);

  EXPECT_FALSE(
    batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());
  EXPECT_EQ(policy.revision(), revision_before);
  expectHysteresisEqual(
    policy.hysteresis(), hysteresis_before);
  const auto retried = authority->consume(
    policy, batch, *selection.capability,
    *lidar_token.token);
  EXPECT_EQ(
    retried.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(retried.receipt.has_value());
}

TEST(
  AdaptiveVisualSelectionReceiptAllocationFailureTest,
  RuntimePermitRemainsRetryableAfterReceiptAllocationFailure)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedSingleCameraBatch(queue);
  ASSERT_TRUE(batch.validatedReceipt().has_value());
  livo::AdaptiveVisualPolicy policy(singleCameraConfig());
  const auto policy_capability =
    policy.issueCandidatePlan(
    batch,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    healthyObservationsBoundToBatch(batch), true);
  ASSERT_TRUE(policy_capability.has_value());
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto lidar_token = authority->issueLidarGeneration(
    10U,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true);
  ASSERT_EQ(
    lidar_token.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(lidar_token.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_token.token);
  ASSERT_TRUE(runtime_input.has_value());
  auto attempt = runtime_input->beginRuntimeAttempt();
  ASSERT_EQ(
    attempt.status,
    livo::AdaptiveVisualRuntimeAttemptStatus::kReady);
  ASSERT_TRUE(attempt.permit.has_value());
  auto permit = std::move(*attempt.permit);
  const auto selection = permit.issueSelection(
    policy, batch, *policy_capability);
  ASSERT_EQ(
    selection.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(selection.capability.has_value());

  livo::test::failNextSelectionReceiptAllocation();
  EXPECT_THROW(
    permit.consumeSelection(
      policy, batch, *selection.capability),
    std::bad_alloc);

  EXPECT_FALSE(
    batch.validatedReceipt()->
    adaptiveVisualSelectionClaimed());
  const auto retried = permit.consumeSelection(
    policy, batch, *selection.capability);
  EXPECT_EQ(
    retried.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(retried.receipt.has_value());
  EXPECT_EQ(retried.selection_consumption_count, 1U);
  EXPECT_TRUE(retried.consumed_by_this_call);
}

}  // namespace
