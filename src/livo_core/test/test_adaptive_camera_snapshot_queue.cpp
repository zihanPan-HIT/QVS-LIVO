#include "adaptive_camera_snapshot_queue.h"
#include "adaptive_camera_snapshot_queue_failure_injection_test_support.h"
#include "frozen_camera_image_failure_injection_test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <gtest/gtest.h>

namespace
{

using Queue = livo::AdaptiveCameraSnapshotQueue;
using CreateStatus =
  livo::AdaptiveCameraSnapshotQueueCreateStatus;
using PushStatus = livo::AdaptiveCameraSnapshotPushStatus;
using RequestStatus =
  livo::AdaptiveCameraSnapshotRequestStatus;
using AdvanceStatus =
  livo::AdaptiveCameraSnapshotAdvanceStatus;
using State = livo::AdaptiveCameraSnapshotQueueState;
using Disposition =
  livo::AdaptiveCameraSnapshotTerminalDisposition;

livo::AdaptiveCameraSnapshotQueueConfiguration configuration(
  const std::size_t capacity = 8U)
{
  livo::AdaptiveCameraSnapshotQueueConfiguration value;
  value.per_camera_capacity = capacity;
  value.reorder_window_sec = 0.010;
  value.request_deadline_offset_sec = 0.030;
  value.max_abs_exposure_offset_sec = 0.020;
  return value;
}

std::unique_ptr<Queue> makeQueue(
  const std::size_t capacity = 8U)
{
  auto result = Queue::create(configuration(capacity));
  EXPECT_EQ(result.status, CreateStatus::READY);
  EXPECT_TRUE(result.ready());
  return std::move(result.queue);
}

cv::Mat image(
  const unsigned char pixel = 17U,
  const int rows = 2,
  const int columns = 3)
{
  return cv::Mat(
    rows, columns, CV_8UC1, cv::Scalar(pixel)).clone();
}

cv::Mat legacyGoldenImage()
{
  cv::Mat source(2, 3, CV_8UC1);
  const std::array<unsigned char, 6U> pixels{{
      1U, 2U, 3U, 4U, 5U, 6U}};
  for (int row = 0; row < source.rows; ++row)
  {
    for (int column = 0; column < source.cols; ++column)
    {
      source.at<unsigned char>(row, column) =
        pixels[static_cast<std::size_t>(
          row * source.cols + column)];
    }
  }
  return source;
}

livo::AdaptiveCameraSnapshotImageDigest legacyGoldenDigest()
{
  return {{
      0xcfU, 0x1aU, 0xc5U, 0x94U, 0x0fU, 0x11U, 0x25U, 0xd0U,
      0xc1U, 0x7dU, 0xdcU, 0x30U, 0xe0U, 0x74U, 0x78U, 0xa3U,
      0x57U, 0x04U, 0xbfU, 0x50U, 0x24U, 0xa2U, 0x86U, 0x57U,
      0x55U, 0x57U, 0xd9U, 0xf9U, 0xf0U, 0x11U, 0x35U, 0x31U}};
}

class ThreadGate
{
public:
  explicit ThreadGate(const std::size_t participants)
  : participants_(participants)
  {
  }

  void arriveAndWait()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ++arrived_;
    if (arrived_ == participants_)
    {
      condition_.notify_all();
      return;
    }
    condition_.wait(
      lock, [this]() {return arrived_ == participants_;});
  }

private:
  const std::size_t participants_;
  std::size_t arrived_{0U};
  std::mutex mutex_;
  std::condition_variable condition_;
};

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FactoryCreatesFourIndependentPreallocatedCameraSlots)
{
  livo::AdaptiveCameraSnapshotQueueConfiguration configuration;
  configuration.per_camera_capacity = 2U;
  configuration.reorder_window_sec = 0.010;
  configuration.request_deadline_offset_sec = 0.030;
  configuration.max_abs_exposure_offset_sec = 0.020;

  auto result =
    livo::AdaptiveCameraSnapshotQueue::create(configuration);

  EXPECT_EQ(
    result.status,
    livo::AdaptiveCameraSnapshotQueueCreateStatus::READY);
  ASSERT_TRUE(result.ready());
  EXPECT_TRUE(result.queue);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FactoryRejectsInvalidCapacityAndTimeConfigurations)
{
  std::vector<livo::AdaptiveCameraSnapshotQueueConfiguration>
    invalid;
  invalid.push_back(configuration());
  invalid.back().per_camera_capacity = 0U;
  invalid.push_back(configuration());
  invalid.back().per_camera_capacity =
    std::numeric_limits<std::size_t>::max();
  invalid.push_back(configuration());
  invalid.back().reorder_window_sec = -0.001;
  invalid.push_back(configuration());
  invalid.back().reorder_window_sec =
    std::numeric_limits<double>::quiet_NaN();
  invalid.push_back(configuration());
  invalid.back().reorder_window_sec =
    std::numeric_limits<double>::infinity();
  invalid.push_back(configuration());
  invalid.back().request_deadline_offset_sec = 0.009;
  invalid.push_back(configuration());
  invalid.back().request_deadline_offset_sec = -0.001;
  invalid.push_back(configuration());
  invalid.back().request_deadline_offset_sec =
    std::numeric_limits<double>::quiet_NaN();
  invalid.push_back(configuration());
  invalid.back().request_deadline_offset_sec =
    std::numeric_limits<double>::infinity();
  invalid.push_back(configuration());
  invalid.back().max_abs_exposure_offset_sec = 0.0;
  invalid.push_back(configuration());
  invalid.back().max_abs_exposure_offset_sec = -0.001;
  invalid.push_back(configuration());
  invalid.back().max_abs_exposure_offset_sec =
    std::numeric_limits<double>::quiet_NaN();
  invalid.push_back(configuration());
  invalid.back().max_abs_exposure_offset_sec =
    std::numeric_limits<double>::infinity();

  for (const auto & item : invalid)
  {
    auto result = Queue::create(item);
    EXPECT_EQ(result.status, CreateStatus::INVALID_CONFIGURATION);
    EXPECT_FALSE(result.ready());
    EXPECT_FALSE(result.queue);
  }
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FactoryResourceFailureDoesNotPreventRetry)
{
  livo::test::
  failNextAdaptiveCameraSnapshotQueueFactoryAllocation();

  auto failed = Queue::create(configuration());
  EXPECT_EQ(failed.status, CreateStatus::RESOURCE_FAILURE);
  EXPECT_FALSE(failed.queue);

  auto retried = Queue::create(configuration());
  EXPECT_EQ(retried.status, CreateStatus::READY);
  EXPECT_TRUE(retried.ready());
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  IdentityIsOwnerSealedAndDifferentQueuesAreForeign)
{
  livo::AdaptiveCameraSnapshotQueueIdentity invalid;
  auto first = makeQueue();
  auto second = makeQueue();
  const auto first_identity = first->identity();
  const auto first_copy = first_identity;
  const auto second_identity = second->identity();

  EXPECT_FALSE(invalid.valid());
  EXPECT_TRUE(first_identity.valid());
  EXPECT_TRUE(first_identity.sameAuthority(first_copy));
  EXPECT_FALSE(first_identity.sameAuthority(invalid));
  EXPECT_FALSE(first_identity.sameAuthority(second_identity));
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  PushRejectsInvalidInputsWithoutChangingAnySlot)
{
  auto queue = makeQueue();
  const std::array<double, 4U> invalid_times{{
      0.0,
      -1.0,
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()}};

  EXPECT_EQ(
    queue->push(4U, 1.0, image()),
    PushStatus::INVALID_CAMERA_ID);
  for (const double timestamp : invalid_times)
  {
    EXPECT_EQ(
      queue->push(0U, timestamp, image()),
      PushStatus::INVALID_TIMESTAMP);
  }
  EXPECT_EQ(
    queue->push(0U, 1.0, cv::Mat{}),
    PushStatus::INVALID_IMAGE);

  const std::array<int, 3U> dimensions{{2, 2, 2}};
  cv::Mat three_dimensional(
    3, dimensions.data(), CV_8UC1, cv::Scalar(4U));
  EXPECT_EQ(
    queue->push(0U, 1.0, three_dimensional),
    PushStatus::INVALID_IMAGE);

  EXPECT_EQ(
    queue->diagnostics().queued_observation_count,
    (std::array<std::size_t, 4U>{{0U, 0U, 0U, 0U}}));
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  PushAcceptsReorderingAndRejectsSameCameraTimestampDuplicate)
{
  auto queue = makeQueue();
  EXPECT_EQ(
    queue->push(1U, 10.020, image(1U)),
    PushStatus::ACCEPTED);
  EXPECT_EQ(
    queue->push(1U, 10.010, image(2U)),
    PushStatus::ACCEPTED_REORDERED);
  EXPECT_EQ(
    queue->push(1U, 10.010, image(3U)),
    PushStatus::DUPLICATE_TIMESTAMP);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[1U],
    2U);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  CapacityIsIndependentPerCameraAndNeverDropsOldest)
{
  auto queue = makeQueue(2U);
  EXPECT_EQ(
    queue->push(0U, 20.000, image(1U)),
    PushStatus::ACCEPTED);
  EXPECT_EQ(
    queue->push(0U, 20.010, image(2U)),
    PushStatus::ACCEPTED);
  EXPECT_EQ(
    queue->push(0U, 20.020, image(3U)),
    PushStatus::CAPACITY_EXCEEDED);
  EXPECT_EQ(
    queue->push(1U, 20.020, image(4U)),
    PushStatus::ACCEPTED);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count,
    (std::array<std::size_t, 4U>{{2U, 1U, 0U, 0U}}));
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  ConflictingPushAndRequestFailuresHaveFrozenPrecedence)
{
  auto duplicate_at_capacity = makeQueue(1U);
  ASSERT_EQ(
    duplicate_at_capacity->push(0U, 21.000, image()),
    PushStatus::ACCEPTED);
  EXPECT_EQ(
    duplicate_at_capacity->push(0U, 21.000, image(2U)),
    PushStatus::DUPLICATE_TIMESTAMP);

  auto retired_and_closed = makeQueue();
  ASSERT_EQ(
    retired_and_closed->push(1U, 22.000, image()),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    retired_and_closed->request(1U, {22.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    retired_and_closed->
    advanceUpstreamEventTime(22.010).state,
    State::SEALED_READY);
  ASSERT_EQ(
    retired_and_closed->
    advanceUpstreamEventTime(22.030).state,
    State::SEALED_READY);
  EXPECT_EQ(
    retired_and_closed->push(1U, 22.000, image(2U)),
    PushStatus::RETIRED_OR_REPLAY);

  auto terminal_and_exhausted = makeQueue();
  ASSERT_EQ(
    terminal_and_exhausted->request(2U, {23.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    terminal_and_exhausted->
    advanceUpstreamEventTime(23.030).state,
    State::LIO_ONLY);
  livo::test::
  exhaustNextAdaptiveCameraSnapshotImageGeneration(0U);
  ASSERT_EQ(
    terminal_and_exhausted->
    push(0U, 23.050, image()),
    PushStatus::AUTHORITY_EXHAUSTED);
  EXPECT_EQ(
    terminal_and_exhausted->
    request(3U, {23.100}).status,
    RequestStatus::TERMINAL_RECEIPT_PENDING);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  CameraPushNeverAdvancesNonVisualEventTime)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(0U, 30.000, image()),
    PushStatus::ACCEPTED);
  const auto diagnostics = queue->diagnostics();
  EXPECT_TRUE(std::isinf(
    diagnostics.latest_upstream_event_time_sec));
  EXPECT_LT(diagnostics.latest_upstream_event_time_sec, 0.0);
  EXPECT_TRUE(std::isinf(
    diagnostics.arrival_closed_through_sec));
  EXPECT_LT(diagnostics.arrival_closed_through_sec, 0.0);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  AdvanceIsMonotonicAndClosureBoundaryIsClosed)
{
  auto queue = makeQueue();
  const auto advanced =
    queue->advanceUpstreamEventTime(40.000);
  ASSERT_EQ(advanced.status, AdvanceStatus::ADVANCED);
  const double closure =
    queue->diagnostics().arrival_closed_through_sec;

  EXPECT_EQ(
    queue->push(0U, closure, image()),
    PushStatus::ARRIVAL_CLOSED);
  EXPECT_EQ(
    queue->push(
      0U,
      std::nextafter(
        closure,
        -std::numeric_limits<double>::infinity()),
      image()),
    PushStatus::ARRIVAL_CLOSED);
  EXPECT_EQ(
    queue->push(
      0U,
      std::nextafter(
        closure,
        std::numeric_limits<double>::infinity()),
      image()),
    PushStatus::ACCEPTED);

  EXPECT_EQ(
    queue->advanceUpstreamEventTime(40.000).status,
    AdvanceStatus::UNCHANGED);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(39.999).status,
    AdvanceStatus::REGRESSED);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(
      std::numeric_limits<double>::quiet_NaN()).status,
    AdvanceStatus::INVALID_TIMESTAMP);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(
      std::numeric_limits<double>::infinity()).status,
    AdvanceStatus::INVALID_TIMESTAMP);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(-1.0).status,
    AdvanceStatus::INVALID_TIMESTAMP);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FreezeResourceFailureConsumesNeitherSlotNorGeneration)
{
  auto queue = makeQueue();
  livo::test::failNextFrozenCameraImageStorageAllocation();

  EXPECT_EQ(
    queue->push(0U, 50.000, image()),
    PushStatus::RESOURCE_FAILURE);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[0U],
    0U);
  EXPECT_EQ(
    queue->push(0U, 50.000, image()),
    PushStatus::ACCEPTED);
  EXPECT_EQ(
    queue->request(1U, {50.000}).status,
    RequestStatus::WAITING);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(50.010).state,
    State::SEALED_READY);
  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_EQ(receipt->observationGeneration(0U), 1U);
  EXPECT_EQ(receipt->imageGeneration(0U), 1U);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  ImageGenerationExhaustionFailsClosedWithoutPublishingFrame)
{
  auto queue = makeQueue();
  livo::test::exhaustNextAdaptiveCameraSnapshotImageGeneration(2U);

  EXPECT_EQ(
    queue->push(2U, 60.000, image()),
    PushStatus::AUTHORITY_EXHAUSTED);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[2U],
    0U);
  EXPECT_EQ(
    queue->push(1U, 60.001, image()),
    PushStatus::AUTHORITY_EXHAUSTED);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  ObservationGenerationExhaustionFailsClosedWithoutPublishingFrame)
{
  auto queue = makeQueue();
  livo::test::
  exhaustNextAdaptiveCameraSnapshotObservationGeneration(1U);

  EXPECT_EQ(
    queue->push(1U, 61.000, image()),
    PushStatus::AUTHORITY_EXHAUSTED);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[1U],
    0U);
  EXPECT_EQ(
    queue->request(1U, {61.000}).status,
    RequestStatus::AUTHORITY_EXHAUSTED);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  RequestWaitsUntilOneSafeFreshCameraThenFinalizesExactlyOnce)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(3U, 70.000, image(3U)),
    PushStatus::ACCEPTED);

  const auto requested = queue->request(10U, {70.000});
  EXPECT_EQ(requested.status, RequestStatus::WAITING);
  EXPECT_EQ(requested.state, State::WAITING);
  EXPECT_FALSE(queue->takeFinalized().has_value());

  const auto advanced =
    queue->advanceUpstreamEventTime(70.010);
  EXPECT_EQ(advanced.status, AdvanceStatus::ADVANCED);
  EXPECT_EQ(advanced.state, State::SEALED_READY);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(70.010).state,
    State::SEALED_READY);

  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_TRUE(receipt->valid());
  EXPECT_EQ(receipt->snapshotSequence(), 1U);
  EXPECT_EQ(receipt->lidarGeneration(), 10U);
  EXPECT_EQ(receipt->cameraCount(), 1U);
  EXPECT_TRUE(receipt->cameraPresent(3U));
  EXPECT_EQ(
    receipt->disposition(),
    Disposition::CANDIDATES_READY);
  EXPECT_EQ(queue->diagnostics().state, State::IDLE);
  EXPECT_FALSE(queue->takeFinalized().has_value());
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FourAsynchronousCamerasSelectTheirIndependentLatestSafeFrames)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(0U, 79.990, image(1U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(0U, 80.010, image(2U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(1U, 80.005, image(3U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(2U, 80.015, image(4U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(3U, 80.020, image(5U)),
    PushStatus::ACCEPTED);

  ASSERT_EQ(
    queue->advanceUpstreamEventTime(80.030).status,
    AdvanceStatus::ADVANCED);
  const auto requested = queue->request(11U, {80.000});
  EXPECT_EQ(requested.status, RequestStatus::SEALED_READY);

  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_EQ(receipt->cameraCount(), 4U);
  EXPECT_EQ(
    receipt->presentMask(),
    (std::array<bool, 4U>{{true, true, true, true}}));
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(0U), 80.010);
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(1U), 80.005);
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(2U), 80.015);
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(3U), 80.020);
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_EQ(receipt->cameraId(camera_id), camera_id);
    const std::uint64_t expected_generation =
      camera_id == 0U ? 2U : 1U;
    EXPECT_EQ(
      receipt->observationGeneration(camera_id),
      expected_generation);
    EXPECT_EQ(
      receipt->imageGeneration(camera_id),
      expected_generation);
  }
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FreshAndCutoffEndpointsAreInclusive)
{
  auto queue = makeQueue();
  const double reference = 90.000;
  const double fresh_lower = 89.980;
  const double deadline_cutoff = 90.020;
  ASSERT_EQ(
    queue->push(0U, fresh_lower, image(1U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(1U, deadline_cutoff, image(2U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(
      2U,
      std::nextafter(
        fresh_lower,
        -std::numeric_limits<double>::infinity()),
      image(3U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(
      3U,
      std::nextafter(
        deadline_cutoff,
        std::numeric_limits<double>::infinity()),
      image(4U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(12U, {reference}).status,
    RequestStatus::WAITING);

  ASSERT_EQ(
    queue->advanceUpstreamEventTime(90.030).state,
    State::SEALED_READY);
  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_TRUE(receipt->cameraPresent(0U));
  EXPECT_TRUE(receipt->cameraPresent(1U));
  EXPECT_FALSE(receipt->cameraPresent(2U));
  EXPECT_FALSE(receipt->cameraPresent(3U));
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(0U), fresh_lower);
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(1U), deadline_cutoff);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count,
    (std::array<std::size_t, 4U>{{0U, 0U, 0U, 1U}}));
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  DeadlineEpsilonWaitsAndEqualityFallsBackToLioOnly)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->request(13U, {100.000}).status,
    RequestStatus::WAITING);

  const double before_deadline =
    std::nextafter(
    100.030,
    -std::numeric_limits<double>::infinity());
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(before_deadline).state,
    State::WAITING);
  EXPECT_FALSE(queue->takeFinalized().has_value());

  EXPECT_EQ(
    queue->advanceUpstreamEventTime(100.030).state,
    State::LIO_ONLY);
  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_TRUE(receipt->valid());
  EXPECT_EQ(receipt->snapshotSequence(), 1U);
  EXPECT_EQ(receipt->lidarGeneration(), 13U);
  EXPECT_DOUBLE_EQ(
    receipt->lioReferenceTimestampSec(), 100.000);
  EXPECT_DOUBLE_EQ(
    receipt->deadlineTimestampSec(), 100.030);
  EXPECT_DOUBLE_EQ(
    receipt->arrivalClosedThroughSec(), 100.020);
  EXPECT_DOUBLE_EQ(receipt->cutoffTimestampSec(), 100.020);
  EXPECT_EQ(
    receipt->disposition(),
    Disposition::DEADLINE_EXPIRED_NO_FRESH_CAMERA);
  EXPECT_EQ(receipt->cameraCount(), 0U);
  EXPECT_EQ(
    receipt->presentMask(),
    (std::array<bool, 4U>{{false, false, false, false}}));
  EXPECT_EQ(receipt->observationGeneration(0U), 0U);
  EXPECT_EQ(receipt->imageGeneration(0U), 0U);
  EXPECT_TRUE(std::isnan(
    receipt->exposureTimestampSec(0U)));
  EXPECT_EQ(
    receipt->imageDigest(0U),
    livo::AdaptiveCameraSnapshotImageDigest{});
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  CandidateAtDeadlineWinsBeforeLioOnly)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(0U, 110.020, image()),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(14U, {110.000}).status,
    RequestStatus::WAITING);

  const auto advanced =
    queue->advanceUpstreamEventTime(110.030);
  EXPECT_EQ(advanced.state, State::SEALED_READY);
  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_TRUE(receipt->cameraPresent(0U));
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(0U), 110.020);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  DeadlineAftermathNeverSelectsBeyondFrozenCutoff)
{
  auto queue = makeQueue();
  const double beyond_cutoff =
    std::nextafter(
    120.020,
    std::numeric_limits<double>::infinity());
  ASSERT_EQ(
    queue->push(1U, beyond_cutoff, image()),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(15U, {120.000}).status,
    RequestStatus::WAITING);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(120.040).state,
    State::LIO_ONLY);

  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_DOUBLE_EQ(receipt->cutoffTimestampSec(), 120.020);
  EXPECT_EQ(receipt->cameraCount(), 0U);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[1U],
    1U);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FrozenRoiDigestAndMetadataSurviveSourceMutation)
{
  auto queue = makeQueue();
  cv::Mat backing(3, 5, CV_8UC1, cv::Scalar(0U));
  cv::Mat roi = backing(cv::Rect(1, 1, 3, 2));
  ASSERT_FALSE(roi.isContinuous());
  legacyGoldenImage().copyTo(roi);

  ASSERT_EQ(
    queue->push(2U, 130.000, roi),
    PushStatus::ACCEPTED);
  roi.setTo(cv::Scalar(255U));
  ASSERT_EQ(
    queue->request(16U, {130.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(130.010).state,
    State::SEALED_READY);

  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_EQ(receipt->imageRows(2U), 2);
  EXPECT_EQ(receipt->imageColumns(2U), 3);
  EXPECT_EQ(receipt->imageType(2U), CV_8UC1);
  EXPECT_EQ(receipt->imageDigest(2U), legacyGoldenDigest());
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  RequestValidatesGenerationReferenceAndActiveIdentity)
{
  auto queue = makeQueue();
  EXPECT_EQ(
    queue->request(0U, {140.000}).status,
    RequestStatus::ZERO_LIDAR_GENERATION);
  EXPECT_EQ(
    queue->request(
      1U,
      {std::numeric_limits<double>::quiet_NaN()}).status,
    RequestStatus::INVALID_LIO_REFERENCE);
  EXPECT_EQ(
    queue->request(
      1U,
      {std::numeric_limits<double>::max()}).status,
    RequestStatus::DEADLINE_OVERFLOW);

  ASSERT_EQ(
    queue->request(20U, {140.000}).status,
    RequestStatus::WAITING);
  EXPECT_EQ(
    queue->request(20U, {140.000}).status,
    RequestStatus::DUPLICATE_ACTIVE_REQUEST);
  EXPECT_EQ(
    queue->request(20U, {140.001}).status,
    RequestStatus::REQUEST_IDENTITY_MISMATCH);
  EXPECT_EQ(
    queue->request(21U, {140.000}).status,
    RequestStatus::DIFFERENT_GENERATION_WHILE_ACTIVE);

  ASSERT_EQ(
    queue->advanceUpstreamEventTime(140.030).state,
    State::LIO_ONLY);
  EXPECT_EQ(
    queue->request(21U, {140.000}).status,
    RequestStatus::TERMINAL_RECEIPT_PENDING);
  ASSERT_TRUE(queue->takeFinalized().has_value());
  EXPECT_EQ(
    queue->request(20U, {140.000}).status,
    RequestStatus::STALE_OR_REPLAYED_GENERATION);
  EXPECT_EQ(
    queue->request(21U, {139.999}).status,
    RequestStatus::NON_MONOTONIC_LIO_REFERENCE);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  RequestStorageFailureDoesNotInstallWaitingOrConsumeSequence)
{
  auto queue = makeQueue();
  livo::test::
  failNextAdaptiveCameraSnapshotRequestStorageAllocation();
  const auto failed = queue->request(30U, {150.000});
  EXPECT_EQ(failed.status, RequestStatus::RESOURCE_FAILURE);
  EXPECT_EQ(failed.state, State::IDLE);
  EXPECT_FALSE(queue->takeFinalized().has_value());

  ASSERT_EQ(
    queue->request(30U, {150.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(150.030).state,
    State::LIO_ONLY);
  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_EQ(receipt->snapshotSequence(), 1U);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  SnapshotSequenceExhaustionDoesNotInstallPartialRequest)
{
  auto queue = makeQueue();
  livo::test::exhaustNextAdaptiveCameraSnapshotSequence();
  const auto result = queue->request(31U, {160.000});
  EXPECT_EQ(result.status, RequestStatus::AUTHORITY_EXHAUSTED);
  EXPECT_EQ(result.state, State::IDLE);
  EXPECT_FALSE(queue->takeFinalized().has_value());
  EXPECT_EQ(
    queue->request(31U, {160.000}).status,
    RequestStatus::AUTHORITY_EXHAUSTED);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  SameReferenceWithNewGenerationIsLegalAndRetiredFrameIsNotReused)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(0U, 170.000, image()),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(40U, {170.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(170.010).state,
    State::SEALED_READY);
  ASSERT_TRUE(queue->takeFinalized().has_value());

  EXPECT_EQ(
    queue->request(41U, {170.000}).status,
    RequestStatus::WAITING);
  EXPECT_EQ(
    queue->advanceUpstreamEventTime(170.030).state,
    State::LIO_ONLY);
  const auto second = queue->takeFinalized();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->snapshotSequence(), 2U);
  EXPECT_EQ(second->cameraCount(), 0U);
  EXPECT_EQ(
    queue->push(0U, 170.000, image()),
    PushStatus::RETIRED_OR_REPLAY);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  FinalizePrunesStaleRetiresPresentAndPreservesFuture)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(0U, 179.970, image(1U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(0U, 180.000, image(2U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(0U, 180.015, image(3U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(0U, 180.025, image(4U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(1U, 179.970, image(5U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(1U, 180.025, image(6U)),
    PushStatus::ACCEPTED);

  ASSERT_EQ(
    queue->request(50U, {180.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(180.025).state,
    State::SEALED_READY);
  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(0U), 180.015);
  EXPECT_FALSE(receipt->cameraPresent(1U));

  const auto diagnostics = queue->diagnostics();
  EXPECT_EQ(
    diagnostics.queued_observation_count,
    (std::array<std::size_t, 4U>{{1U, 1U, 0U, 0U}}));
  EXPECT_DOUBLE_EQ(
    diagnostics.retired_through_sec[0U], 180.015);
  EXPECT_TRUE(std::isinf(
    diagnostics.retired_through_sec[1U]));
  EXPECT_LT(diagnostics.retired_through_sec[1U], 0.0);
  EXPECT_EQ(
    queue->push(0U, 180.015, image()),
    PushStatus::RETIRED_OR_REPLAY);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  LioOnlyPrunesAllStaleSlotsWithoutRetiringAbsentCameras)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(0U, 299.970, image(1U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(0U, 300.025, image(2U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(2U, 299.979, image(3U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->push(2U, 300.021, image(4U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(51U, {300.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(300.030).state,
    State::LIO_ONLY);

  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_EQ(receipt->cameraCount(), 0U);
  const auto diagnostics = queue->diagnostics();
  EXPECT_EQ(
    diagnostics.queued_observation_count,
    (std::array<std::size_t, 4U>{{1U, 0U, 1U, 0U}}));
  for (const double retired : diagnostics.retired_through_sec)
  {
    EXPECT_TRUE(std::isinf(retired));
    EXPECT_LT(retired, 0.0);
  }
  EXPECT_EQ(
    queue->push(0U, 299.970, image()),
    PushStatus::ARRIVAL_CLOSED);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  PerCameraObservationAndImageGenerationsAdvanceTogether)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(3U, 310.000, image(1U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(52U, {310.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(310.010).state,
    State::SEALED_READY);
  const auto first = queue->takeFinalized();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->observationGeneration(3U), 1U);
  EXPECT_EQ(first->imageGeneration(3U), 1U);

  ASSERT_EQ(
    queue->push(3U, 310.100, image(2U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(53U, {310.100}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(310.110).state,
    State::SEALED_READY);
  const auto second = queue->takeFinalized();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->observationGeneration(3U), 2U);
  EXPECT_EQ(second->imageGeneration(3U), 2U);
}

TEST(
  AdaptiveCameraSnapshotQueueTest,
  CandidateReceiptRetainsFrozenStorageAfterQueueDestruction)
{
  auto queue = makeQueue();
  auto foreign = makeQueue();
  const auto owner = queue->identity();
  const auto foreign_owner = foreign->identity();
  ASSERT_EQ(
    queue->push(2U, 190.000, legacyGoldenImage()),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(60U, {190.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(190.010).state,
    State::SEALED_READY);
  auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  ASSERT_TRUE(receipt->cameraPresent(2U));
  const auto digest = receipt->imageDigest(2U);
  const double timestamp =
    receipt->exposureTimestampSec(2U);
  EXPECT_TRUE(receipt->valid());

  queue.reset();

  EXPECT_TRUE(receipt->valid());
  EXPECT_TRUE(receipt->issuedBy(owner));
  EXPECT_FALSE(receipt->issuedBy(foreign_owner));
  EXPECT_EQ(receipt->snapshotSequence(), 1U);
  EXPECT_EQ(receipt->imageDigest(2U), digest);
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(2U), timestamp);
  EXPECT_TRUE(receipt->valid());
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  SimultaneousSameTimestampPushHasExactlyOneWinner)
{
  auto queue = makeQueue();
  ThreadGate gate(2U);
  std::array<PushStatus, 2U> results;
  std::array<std::thread, 2U> workers;
  for (std::size_t index = 0U; index < workers.size(); ++index)
  {
    workers[index] = std::thread(
      [&, index]()
      {
        const cv::Mat source =
          image(static_cast<unsigned char>(index + 1U));
        gate.arriveAndWait();
        results[index] =
          queue->push(0U, 200.000, source);
      });
  }
  for (auto & worker : workers) {
    worker.join();
  }

  EXPECT_EQ(
    std::count(
      results.begin(), results.end(),
      PushStatus::ACCEPTED),
    1);
  EXPECT_EQ(
    std::count(
      results.begin(), results.end(),
      PushStatus::DUPLICATE_TIMESTAMP),
    1);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[0U],
    1U);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  CapacityCompetitionAcceptsOnlyPreallocatedSlots)
{
  constexpr std::size_t kWorkerCount = 8U;
  constexpr std::size_t kCapacity = 3U;
  auto queue = makeQueue(kCapacity);
  ThreadGate gate(kWorkerCount);
  std::array<PushStatus, kWorkerCount> results;
  std::array<std::thread, kWorkerCount> workers;
  for (std::size_t index = 0U; index < workers.size(); ++index)
  {
    workers[index] = std::thread(
      [&, index]()
      {
        const cv::Mat source =
          image(static_cast<unsigned char>(index + 1U));
        gate.arriveAndWait();
        results[index] = queue->push(
          1U,
          210.000 + static_cast<double>(index) * 0.001,
          source);
      });
  }
  for (auto & worker : workers) {
    worker.join();
  }

  const std::size_t accepted =
    static_cast<std::size_t>(std::count_if(
      results.begin(), results.end(),
      [](const PushStatus status)
      {
        return
          status == PushStatus::ACCEPTED ||
          status == PushStatus::ACCEPTED_REORDERED;
      }));
  EXPECT_EQ(accepted, kCapacity);
  EXPECT_EQ(
    std::count(
      results.begin(), results.end(),
      PushStatus::CAPACITY_EXCEEDED),
    static_cast<int>(kWorkerCount - kCapacity));
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[1U],
    kCapacity);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  PushRevalidatesClosureAfterSlowFreezeWithoutHoldingQueueMutex)
{
  auto queue = makeQueue();
  livo::test::pauseNextAdaptiveCameraSnapshotPushAfterFreeze();
  PushStatus push_status = PushStatus::RESOURCE_FAILURE;
  std::thread worker(
    [&]()
    {
      push_status =
        queue->push(0U, 220.000, image());
    });

  const bool paused =
    livo::test::
    waitForAdaptiveCameraSnapshotPushPausedAfterFreeze();
  EXPECT_TRUE(paused);
  const auto advanced =
    queue->advanceUpstreamEventTime(220.010);
  EXPECT_EQ(advanced.status, AdvanceStatus::ADVANCED);
  livo::test::releaseAdaptiveCameraSnapshotPushAfterFreeze();
  worker.join();

  EXPECT_EQ(push_status, PushStatus::ARRIVAL_CLOSED);
  EXPECT_EQ(
    queue->diagnostics().queued_observation_count[0U],
    0U);

  auto earlier_commit = makeQueue();
  EXPECT_EQ(
    earlier_commit->push(0U, 221.000, image()),
    PushStatus::ACCEPTED);
  EXPECT_EQ(
    earlier_commit->
    advanceUpstreamEventTime(221.010).status,
    AdvanceStatus::ADVANCED);
  EXPECT_EQ(
    earlier_commit->diagnostics().
    queued_observation_count[0U],
    1U);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  ConcurrentDuplicateRequestInstallsOnlyOneWaitingTransaction)
{
  auto queue = makeQueue();
  ThreadGate gate(2U);
  std::array<RequestStatus, 2U> statuses;
  std::array<std::thread, 2U> workers;
  for (std::size_t index = 0U; index < workers.size(); ++index)
  {
    workers[index] = std::thread(
      [&, index]()
      {
        gate.arriveAndWait();
        statuses[index] =
          queue->request(70U, {230.000}).status;
      });
  }
  for (auto & worker : workers) {
    worker.join();
  }

  EXPECT_EQ(
    std::count(
      statuses.begin(), statuses.end(),
      RequestStatus::WAITING),
    1);
  EXPECT_EQ(
    std::count(
      statuses.begin(), statuses.end(),
      RequestStatus::DUPLICATE_ACTIVE_REQUEST),
    1);
  EXPECT_EQ(queue->diagnostics().state, State::WAITING);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  RequestRevalidationRejectsDuplicateCommittedDuringStorageAllocation)
{
  auto queue = makeQueue();
  livo::test::
  pauseNextAdaptiveCameraSnapshotRequestBeforeRevalidation();
  RequestStatus paused_status =
    RequestStatus::RESOURCE_FAILURE;
  std::thread paused_request(
    [&]()
    {
      paused_status =
        queue->request(71U, {231.000}).status;
    });

  const bool paused =
    livo::test::
    waitForAdaptiveCameraSnapshotRequestPausedBeforeRevalidation();
  EXPECT_TRUE(paused);
  const auto committed =
    queue->request(71U, {231.000});
  EXPECT_EQ(committed.status, RequestStatus::WAITING);
  livo::test::
  releaseAdaptiveCameraSnapshotRequestBeforeRevalidation();
  paused_request.join();

  EXPECT_EQ(
    paused_status,
    RequestStatus::DUPLICATE_ACTIVE_REQUEST);
  EXPECT_EQ(queue->diagnostics().state, State::WAITING);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  RequestRevalidationRejectsDifferentGenerationCommittedDuringAllocation)
{
  auto queue = makeQueue();
  livo::test::
  pauseNextAdaptiveCameraSnapshotRequestBeforeRevalidation();
  RequestStatus paused_status =
    RequestStatus::RESOURCE_FAILURE;
  std::thread paused_request(
    [&]()
    {
      paused_status =
        queue->request(80U, {241.000}).status;
    });

  const bool paused =
    livo::test::
    waitForAdaptiveCameraSnapshotRequestPausedBeforeRevalidation();
  EXPECT_TRUE(paused);
  const auto committed =
    queue->request(81U, {241.000});
  EXPECT_EQ(committed.status, RequestStatus::WAITING);
  livo::test::
  releaseAdaptiveCameraSnapshotRequestBeforeRevalidation();
  paused_request.join();

  EXPECT_EQ(
    paused_status,
    RequestStatus::DIFFERENT_GENERATION_WHILE_ACTIVE);
  EXPECT_EQ(queue->diagnostics().state, State::WAITING);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  ConcurrentDifferentGenerationsCannotReplaceActiveRequest)
{
  auto queue = makeQueue();
  ThreadGate gate(2U);
  std::array<RequestStatus, 2U> statuses;
  std::thread first(
    [&]()
    {
      gate.arriveAndWait();
      statuses[0U] =
        queue->request(80U, {240.000}).status;
    });
  std::thread second(
    [&]()
    {
      gate.arriveAndWait();
      statuses[1U] =
        queue->request(81U, {240.000}).status;
    });
  first.join();
  second.join();

  EXPECT_EQ(
    std::count(
      statuses.begin(), statuses.end(),
      RequestStatus::WAITING),
    1);
  EXPECT_EQ(
    std::count(
      statuses.begin(), statuses.end(),
      RequestStatus::DIFFERENT_GENERATION_WHILE_ACTIVE),
    1);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  RequestDeadlineRaceStillPublishesOneTerminalReceipt)
{
  auto queue = makeQueue();
  ThreadGate gate(2U);
  RequestStatus request_status =
    RequestStatus::RESOURCE_FAILURE;
  AdvanceStatus advance_status =
    AdvanceStatus::INVALID_TIMESTAMP;
  std::thread requester(
    [&]()
    {
      gate.arriveAndWait();
      request_status =
        queue->request(90U, {250.000}).status;
    });
  std::thread advancer(
    [&]()
    {
      gate.arriveAndWait();
      advance_status =
        queue->advanceUpstreamEventTime(250.030).status;
    });
  requester.join();
  advancer.join();

  EXPECT_EQ(advance_status, AdvanceStatus::ADVANCED);
  EXPECT_TRUE(
    request_status == RequestStatus::WAITING ||
    request_status == RequestStatus::LIO_ONLY);
  EXPECT_EQ(queue->diagnostics().state, State::LIO_ONLY);
  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_EQ(
    receipt->disposition(),
    Disposition::DEADLINE_EXPIRED_NO_FRESH_CAMERA);
  EXPECT_FALSE(queue->takeFinalized().has_value());
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  ConcurrentTakeHasExactlyOneReceiptWinner)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->request(100U, {260.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(260.030).state,
    State::LIO_ONLY);

  ThreadGate gate(2U);
  std::array<bool, 2U> won{{false, false}};
  std::array<std::thread, 2U> workers;
  for (std::size_t index = 0U; index < workers.size(); ++index)
  {
    workers[index] = std::thread(
      [&, index]()
      {
        gate.arriveAndWait();
        won[index] =
          queue->takeFinalized().has_value();
      });
  }
  for (auto & worker : workers) {
    worker.join();
  }

  EXPECT_EQ(
    std::count(won.begin(), won.end(), true),
    1);
  EXPECT_EQ(queue->diagnostics().state, State::IDLE);
}

TEST(
  AdaptiveCameraSnapshotQueueConcurrencyTest,
  PushAfterTerminalCannotMutateSealedReceipt)
{
  auto queue = makeQueue();
  ASSERT_EQ(
    queue->push(0U, 270.000, image(1U)),
    PushStatus::ACCEPTED);
  ASSERT_EQ(
    queue->request(110U, {270.000}).status,
    RequestStatus::WAITING);
  ASSERT_EQ(
    queue->advanceUpstreamEventTime(270.010).state,
    State::SEALED_READY);
  ASSERT_EQ(
    queue->push(1U, 270.020, image(2U)),
    PushStatus::ACCEPTED);

  const auto receipt = queue->takeFinalized();
  ASSERT_TRUE(receipt.has_value());
  EXPECT_EQ(receipt->cameraCount(), 1U);
  EXPECT_TRUE(receipt->cameraPresent(0U));
  EXPECT_FALSE(receipt->cameraPresent(1U));
  EXPECT_DOUBLE_EQ(
    receipt->exposureTimestampSec(0U), 270.000);
}

}  // namespace
