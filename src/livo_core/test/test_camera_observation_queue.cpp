#include "camera_observation.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace
{

CameraObservation observation(const std::uint8_t camera_id, const double timestamp)
{
  CameraObservation value;
  value.camera_id = camera_id;
  value.timestamp = timestamp;
  value.image = cv::Mat::ones(2, 2, CV_8UC1);
  return value;
}

TEST(CameraObservationQueueTest, OrdersAsynchronousCamerasByTimestamp)
{
  CameraObservationQueue queue;

  EXPECT_EQ(
    queue.push(observation(0U, 10.020)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(
    queue.push(observation(2U, 10.011)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(
    queue.push(observation(1U, 10.006)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservation next;
  ASSERT_NE(queue.nextReady(10.010), nullptr);
  ASSERT_TRUE(queue.pop(next));
  EXPECT_EQ(next.camera_id, 1U);
  EXPECT_DOUBLE_EQ(next.timestamp, 10.006);

  EXPECT_EQ(queue.nextReady(10.010), nullptr);
  ASSERT_NE(queue.nextReady(10.011), nullptr);
  ASSERT_TRUE(queue.pop(next));
  EXPECT_EQ(next.camera_id, 2U);
}

TEST(CameraObservationQueueTest, PreservesExactSourceStampIdentity)
{
  CameraObservationQueue queue;
  CameraObservation source = observation(2U, 10.125);
  source.source_stamp_ns = 1234567890123456789ULL;
  ASSERT_EQ(
    queue.push(std::move(source)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservation popped;
  ASSERT_TRUE(queue.pop(popped));
  EXPECT_EQ(popped.source_stamp_ns, 1234567890123456789ULL);
}

TEST(CameraObservationQueueTest, RejectsPerCameraDuplicatesAndProcessedFrames)
{
  CameraObservationQueue queue;

  EXPECT_EQ(
    queue.push(observation(0U, 20.0)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(
    queue.push(observation(0U, 20.0)),
    CameraObservationPushResult::OUT_OF_ORDER);
  EXPECT_EQ(
    queue.push(observation(0U, 19.9)),
    CameraObservationPushResult::OUT_OF_ORDER);

  CameraObservation next;
  ASSERT_TRUE(queue.pop(next));
  EXPECT_EQ(
    queue.push(observation(1U, 20.0)),
    CameraObservationPushResult::ALREADY_PROCESSED);
}

TEST(CameraObservationQueueTest, AppliesWatermarkWithoutDroppingFutureFrames)
{
  CameraObservationQueue queue;
  ASSERT_EQ(
    queue.push(observation(3U, 30.015)),
    CameraObservationPushResult::ACCEPTED);

  EXPECT_EQ(queue.nextReady(30.014), nullptr);
  EXPECT_NE(queue.nextReady(30.015), nullptr);
  EXPECT_EQ(queue.size(), 1U);
}

TEST(CameraObservationQueueTest, KeepsSimultaneousFramesInCameraIdOrder)
{
  CameraObservationQueue queue;
  ASSERT_EQ(
    queue.push(observation(2U, 35.0)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(0U, 35.0)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(1U, 35.0)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservation next;
  ASSERT_TRUE(queue.pop(next));
  EXPECT_EQ(next.camera_id, 0U);
  ASSERT_TRUE(queue.pop(next));
  EXPECT_EQ(next.camera_id, 1U);
  ASSERT_TRUE(queue.pop(next));
  EXPECT_EQ(next.camera_id, 2U);
}

TEST(CameraObservationQueueTest, FailsClosedAtCapacity)
{
  CameraObservationQueue queue(2U);
  EXPECT_EQ(
    queue.push(observation(0U, 40.0)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(
    queue.push(observation(1U, 40.1)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(
    queue.push(observation(2U, 40.2)),
    CameraObservationPushResult::CAPACITY_EXCEEDED);
  EXPECT_EQ(queue.size(), 2U);
}

TEST(CameraObservationQueueTest, WaitsForEveryActiveCameraBeforeAdvancingWatermark)
{
  CameraObservationQueue queue;
  EXPECT_EQ(queue.push(observation(0U, 1.00)), CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(queue.push(observation(1U, 0.99)), CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(queue.push(observation(2U, 0.98)), CameraObservationPushResult::ACCEPTED);

  const double incomplete_watermark =
    queue.synchronizedWatermark(1.20, 4U, 0.02);
  EXPECT_TRUE(std::isinf(incomplete_watermark));
  EXPECT_LT(incomplete_watermark, 0.0);
  EXPECT_EQ(queue.nextReady(incomplete_watermark), nullptr);

  EXPECT_EQ(queue.push(observation(3U, 0.97)), CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(queue.push(observation(0U, 1.10)), CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(queue.push(observation(1U, 1.09)), CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(queue.push(observation(2U, 1.08)), CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(queue.push(observation(3U, 1.07)), CameraObservationPushResult::ACCEPTED);

  const double synchronized_watermark =
    queue.synchronizedWatermark(1.20, 4U, 0.02);
  EXPECT_NEAR(synchronized_watermark, 1.05, 1e-12);
  ASSERT_NE(queue.nextReady(synchronized_watermark), nullptr);
  EXPECT_DOUBLE_EQ(queue.nextReady(synchronized_watermark)->timestamp, 0.97);
}

TEST(CameraObservationQueueTest, BuildsOneNearestFramePerCameraBatch)
{
  CameraObservationQueue queue;

  ASSERT_EQ(queue.push(observation(1U, 10.0000)), CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(queue.push(observation(0U, 10.0068)), CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(queue.push(observation(2U, 10.0148)), CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(queue.push(observation(3U, 10.0297)), CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(queue.push(observation(1U, 10.0334)), CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(queue.push(observation(0U, 10.0402)), CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch next;
  ASSERT_TRUE(queue.nextSynchronized(10.0600, 0.0200, next));
  EXPECT_NEAR(next.timestamp, 10.0275, 1e-9);
  EXPECT_NEAR(next.max_time_offset_sec, 0.0127, 1e-9);
  EXPECT_DOUBLE_EQ(next.observations[0U].timestamp, 10.0402);
  EXPECT_DOUBLE_EQ(next.observations[1U].timestamp, 10.0334);
  EXPECT_DOUBLE_EQ(next.observations[2U].timestamp, 10.0148);
  EXPECT_DOUBLE_EQ(next.observations[3U].timestamp, 10.0297);

  CameraObservationBatch popped;
  std::vector<CameraObservation> discarded;
  ASSERT_TRUE(queue.popSynchronized(
    10.0600, 0.0200, popped, &discarded));
  EXPECT_NEAR(popped.timestamp, next.timestamp, 1e-12);
  ASSERT_EQ(discarded.size(), 2U);
  EXPECT_DOUBLE_EQ(discarded[0U].timestamp, 10.0000);
  EXPECT_DOUBLE_EQ(discarded[1U].timestamp, 10.0068);
  EXPECT_EQ(queue.size(), 0U);
  EXPECT_NEAR(queue.processedUntil(), popped.timestamp, 1e-12);
}

TEST(CameraObservationQueueTest, AdaptiveWatermarkPreservesFourCameraFrontierBeforeDeadline)
{
  CameraObservationQueue queue;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    ASSERT_EQ(queue.push(observation(camera_id, 10.0)), CameraObservationPushResult::ACCEPTED);
    ASSERT_EQ(queue.push(observation(camera_id, 10.05)), CameraObservationPushResult::ACCEPTED);
  }
  EXPECT_DOUBLE_EQ(
    queue.adaptiveSynchronizedWatermark(10.06, 4U, 0.02, 0.02, 0.03),
    queue.synchronizedWatermark(10.06, 4U, 0.02));
}

TEST(CameraObservationQueueTest, AdaptiveDeadlineReleasesNeverSeenAndStoppedCameras)
{
  for (const bool previously_seen : {false, true})
  {
    SCOPED_TRACE(previously_seen);
    CameraObservationQueue queue;
    if (previously_seen)
    {
      for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
      {
        ASSERT_EQ(queue.push(observation(camera_id, 9.9)), CameraObservationPushResult::ACCEPTED);
      }
      CameraObservationBatch previous;
      ASSERT_TRUE(queue.popSynchronized(9.95, 0.02, previous));
    }
    ASSERT_EQ(queue.push(observation(0U, 10.0)), CameraObservationPushResult::ACCEPTED);
    EXPECT_LT(queue.adaptiveSynchronizedWatermark(10.06, 4U, 0.02, 0.02, 0.03), 10.0);
    const double watermark = queue.adaptiveSynchronizedWatermark(10.08, 4U, 0.02, 0.02, 0.03);
    CameraObservationBatch recovered;
    ASSERT_EQ(queue.popAdaptiveSynchronized(watermark, 0.02, 0.03, 1U, recovered),
      AdaptiveCameraBatchResult::READY);
    EXPECT_EQ(recovered.camera_count, 1U);
    EXPECT_TRUE(recovered.present[0U]);
    EXPECT_TRUE(queue.empty());
  }
}

TEST(CameraObservationQueueTest, AdaptiveDeadlineReleasesLastFrontFrameWithoutAnotherImage)
{
  CameraObservationQueue queue;
  ASSERT_EQ(queue.push(observation(0U, 10.0)), CameraObservationPushResult::ACCEPTED);
  const double watermark = queue.adaptiveSynchronizedWatermark(10.08, 1U, 0.02, 0.02, 0.03);
  CameraObservationBatch batch;
  ASSERT_EQ(queue.popAdaptiveSynchronized(watermark, 0.02, 0.03, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_EQ(batch.camera_count, 1U);
  EXPECT_DOUBLE_EQ(batch.timestamp, 10.0);
}

TEST(CameraObservationQueueTest, SynchronizedBatchWaitsForCompleteSafeWindow)
{
  CameraObservationQueue queue;
  ASSERT_EQ(queue.push(observation(0U, 20.000)), CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(queue.push(observation(1U, 20.005)), CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(queue.push(observation(2U, 20.010)), CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_FALSE(queue.nextSynchronized(20.050, 0.020, batch));

  ASSERT_EQ(queue.push(observation(3U, 20.015)), CameraObservationPushResult::ACCEPTED);
  EXPECT_FALSE(queue.nextSynchronized(20.009, 0.020, batch));
  EXPECT_TRUE(queue.nextSynchronized(20.050, 0.020, batch));
  EXPECT_EQ(queue.size(), 4U);
}

TEST(CameraObservationQueueTest, ContinuesAcrossCameraPhaseBoundary)
{
  CameraObservationQueue queue(256U);
  constexpr double period = 1.0 / 30.0;
  const std::array<double, 4U> phase = {{0.0068, 0.0, 0.0148, 0.0297}};

  for (int frame = 0; frame < 40; ++frame)
  {
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      const double drift =
        static_cast<double>(frame) *
        static_cast<double>(camera_id + 1U) * 2.0e-6;
      ASSERT_EQ(
        queue.push(observation(
          camera_id, 30.0 + frame * period + phase[camera_id] + drift)),
        CameraObservationPushResult::ACCEPTED);
    }
  }

  int batch_count = 0;
  CameraObservationBatch batch;
  while (queue.popSynchronized(32.0, 0.020, batch))
  {
    EXPECT_LE(batch.max_time_offset_sec, 0.020);
    ++batch_count;
  }
  EXPECT_GE(batch_count, 39);
  EXPECT_LT(queue.size(), 8U);
}

TEST(
  CameraObservationQueueTest,
  AdaptivePartialBatchContinuesWhenOneCameraDropsOut)
{
  CameraObservationQueue queue;
  ASSERT_EQ(
    queue.push(observation(0U, 100.000)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(1U, 100.005)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(2U, 100.010)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.nextAdaptiveSynchronized(
      100.040, 0.010, 0.030, 3U, batch),
    AdaptiveCameraBatchResult::WAITING);
  EXPECT_EQ(
    queue.nextAdaptiveSynchronized(
      100.060, 0.010, 0.030, 3U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_TRUE(batch.partial);
  EXPECT_EQ(batch.camera_count, 3U);
  EXPECT_TRUE(batch.present[0U]);
  EXPECT_TRUE(batch.present[1U]);
  EXPECT_TRUE(batch.present[2U]);
  EXPECT_FALSE(batch.present[3U]);
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      100.060, 0.010, 0.030, 3U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_TRUE(queue.empty());

  // A delayed frame belonging to the completed partial batch cannot be
  // reused as the missing camera in a future batch.
  EXPECT_EQ(
    queue.push(observation(3U, 100.008)),
    CameraObservationPushResult::ALREADY_PROCESSED);

  ASSERT_EQ(
    queue.push(observation(0U, 100.100)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(1U, 100.105)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(2U, 100.110)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(3U, 100.108)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      100.130, 0.010, 0.030, 3U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_FALSE(batch.partial);
  EXPECT_EQ(batch.camera_count, 4U);
  EXPECT_TRUE(std::all_of(
    batch.present.begin(), batch.present.end(),
    [](const bool present) {return present;}));
}

TEST(
  CameraObservationQueueTest,
  AdaptivePartialBatchFailsClosedBelowMinimumAndMakesProgress)
{
  CameraObservationQueue queue;
  ASSERT_EQ(
    queue.push(observation(0U, 200.000)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(1U, 200.005)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.nextAdaptiveSynchronized(
      200.100, 0.010, 0.030, 3U, batch),
    AdaptiveCameraBatchResult::INSUFFICIENT_CAMERAS);
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      200.100, 0.010, 0.030, 3U, batch),
    AdaptiveCameraBatchResult::INSUFFICIENT_CAMERAS);
  EXPECT_TRUE(queue.empty());

  ASSERT_EQ(
    queue.push(observation(0U, 200.200)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(1U, 200.205)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(observation(2U, 200.210)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      200.260, 0.010, 0.030, 3U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_EQ(batch.camera_count, 3U);
}

TEST(
  CameraObservationQueueAuthorityTest,
  FreezesAcceptedImageBeforeTheCallerCanMutateItsSource)
{
  CameraObservationQueue queue;
  ASSERT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  CameraObservation source = observation(0U, 300.000);
  source.image.at<std::uint8_t>(0, 0) = 17U;

  ASSERT_EQ(
    queue.push(source),
    CameraObservationPushResult::ACCEPTED);
  source.image.at<std::uint8_t>(0, 0) = 91U;

  for (std::uint8_t camera_id = 1U; camera_id < 4U; ++camera_id)
  {
    ASSERT_EQ(
      queue.push(observation(
        camera_id, 300.000 + 0.001 * camera_id)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  ASSERT_TRUE(queue.popSynchronized(300.010, 0.010, batch));
  EXPECT_EQ(batch.observations[0U].image.at<std::uint8_t>(0, 0), 17U);
  ASSERT_TRUE(batch.validatedReceipt().has_value());
  EXPECT_EQ(batch.validatedReceipt()->sealImageHashCount(), 4U);
  EXPECT_EQ(batch.validatedReceipt()->validationImageHashCount(), 0U);
}

TEST(
  CameraObservationQueueAuthorityTest,
  ShallowBatchCopySharesTheOwnerFrozenImageStorage)
{
  CameraObservationQueue queue;
  ASSERT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    ASSERT_EQ(
      queue.push(observation(
        camera_id, 310.000 + 0.001 * camera_id)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  ASSERT_TRUE(queue.popSynchronized(310.010, 0.010, batch));
  CameraObservationBatch shallow_copy = batch;
  shallow_copy.observations[2U].image.at<std::uint8_t>(0, 0) = 44U;

  ASSERT_TRUE(batch.validatedReceipt().has_value());
  ASSERT_TRUE(shallow_copy.validatedReceipt().has_value());
  EXPECT_EQ(
    batch.observations[2U].image.at<std::uint8_t>(0, 0),
    44U);
  EXPECT_EQ(
    shallow_copy.validatedReceipt()->
    validationImageHashCount(),
    0U);
}

TEST(
  CameraObservationQueueAuthorityTest,
  DistinguishesForeignOwnerAtTheSameTimestampsAndAdvancesGenerations)
{
  CameraObservationQueue expected_owner;
  CameraObservationQueue foreign_owner;
  ASSERT_TRUE(
    expected_owner.setBatchAuthoritySealingEnabled(true));
  ASSERT_TRUE(
    foreign_owner.setBatchAuthoritySealingEnabled(true));
  const auto expected_identity =
    expected_owner.batchAuthorityIdentity();
  const auto foreign_identity =
    foreign_owner.batchAuthorityIdentity();
  ASSERT_TRUE(expected_identity.valid());
  ASSERT_TRUE(foreign_identity.valid());

  auto push_batch = [](
      CameraObservationQueue & queue,
      const double timestamp,
      const std::uint8_t pixel)
    {
      for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
      {
        CameraObservation value =
          observation(camera_id, timestamp + 0.001 * camera_id);
        value.image.setTo(pixel);
        EXPECT_EQ(
          queue.push(std::move(value)),
          CameraObservationPushResult::ACCEPTED);
      }
    };

  push_batch(expected_owner, 320.000, 5U);
  push_batch(foreign_owner, 320.000, 9U);
  CameraObservationBatch first;
  CameraObservationBatch foreign;
  ASSERT_TRUE(expected_owner.popSynchronized(320.010, 0.010, first));
  ASSERT_TRUE(foreign_owner.popSynchronized(320.010, 0.010, foreign));

  ASSERT_TRUE(first.validatedReceipt().has_value());
  ASSERT_TRUE(foreign.validatedReceipt().has_value());
  EXPECT_TRUE(first.validatedReceipt()->issuedBy(expected_identity));
  EXPECT_FALSE(first.validatedReceipt()->issuedBy(foreign_identity));
  EXPECT_FALSE(foreign.validatedReceipt()->issuedBy(expected_identity));
  EXPECT_TRUE(foreign.validatedReceipt()->issuedBy(foreign_identity));
  EXPECT_EQ(first.validatedReceipt()->batchSequence(), 1U);

  push_batch(expected_owner, 321.000, 7U);
  CameraObservationBatch second;
  ASSERT_TRUE(expected_owner.popSynchronized(321.010, 0.010, second));
  ASSERT_TRUE(second.validatedReceipt().has_value());
  EXPECT_EQ(second.validatedReceipt()->batchSequence(), 2U);
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_GT(
      second.validatedReceipt()->observationGeneration(camera_id),
      first.validatedReceipt()->observationGeneration(camera_id));
    EXPECT_GT(
      second.validatedReceipt()->imageGeneration(camera_id),
      first.validatedReceipt()->imageGeneration(camera_id));
  }
}

TEST(
  CameraObservationQueueAuthorityTest,
  DisabledAuthorityPathPreservesTheBaselineZeroHashZeroCloneBehavior)
{
  CameraObservationQueue queue;
  CameraObservation source = observation(0U, 330.000);
  const unsigned char * source_data = source.image.data;
  ASSERT_EQ(
    queue.push(source),
    CameraObservationPushResult::ACCEPTED);
  for (std::uint8_t camera_id = 1U; camera_id < 4U; ++camera_id)
  {
    ASSERT_EQ(
      queue.push(observation(
        camera_id, 330.000 + 0.001 * camera_id)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  ASSERT_TRUE(queue.popSynchronized(330.010, 0.010, batch));
  EXPECT_EQ(batch.observations[0U].image.data, source_data);
  EXPECT_FALSE(batch.validatedReceipt().has_value());
}

}  // namespace
