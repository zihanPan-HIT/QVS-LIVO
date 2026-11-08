#include "reference_only_migration_runtime_provenance.h"

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace
{

static_assert(!std::is_default_constructible_v<
  livo::ReferenceOnlyMigrationFrameProvenance>);
static_assert(!std::is_aggregate_v<
  livo::ReferenceOnlyMigrationFrameProvenance>);
static_assert(std::is_copy_constructible_v<
  livo::ReferenceOnlyMigrationFrameProvenance>);
static_assert(!std::is_default_constructible_v<
  livo::ValidatedBatchRuntimeComposition>);
static_assert(!std::is_aggregate_v<
  livo::ValidatedBatchRuntimeComposition>);

CameraObservationBatch ownedBatch(
  CameraObservationQueue & queue,
  const double timestamp_sec,
  const std::uint8_t pixel)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
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

CameraObservationBatch ownedPartialBatch(
  CameraObservationQueue & queue,
  const double timestamp_sec,
  const std::uint8_t absent_camera_id)
{
  EXPECT_TRUE(queue.setBatchAuthoritySealingEnabled(true));
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    if (camera_id == absent_camera_id) {
      continue;
    }
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp = timestamp_sec;
    observation.image =
      cv::Mat(
      8, 12, CV_8UC1,
      cv::Scalar(static_cast<unsigned char>(40U + camera_id)));
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }
  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      timestamp_sec + 0.100, 0.005, 0.010, 3U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  OwnerPublicationBindsBatchStateCovarianceAndCheckedRevisions)
{
  CameraObservationQueue queue;
  CameraObservationBatch first_batch =
    ownedBatch(queue, 10.0, 23U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  ASSERT_TRUE(publisher.valid());
  EXPECT_TRUE(
    publisher.identity().bindsBatchAuthority(
      queue.batchAuthorityIdentity()));

  StatesGroup first_state;
  first_state.pos_end =
    Eigen::Vector3d(1.0, 2.0, 3.0);
  first_state.inv_expo_time = 0.75;
  first_state.cov =
    2.0e-3 *
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
  livo::CameraTimeTransfer transfer;
  auto first = publisher.publish(
    first_batch, first_state, first_batch.timestamp,
    0.0, transfer);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(first_batch.validatedReceipt().has_value());
  EXPECT_EQ(
    first_batch.validatedReceipt()->
    validationImageHashCount(),
    4U);
  EXPECT_EQ(first->publicationSequence(), 1U);
  EXPECT_EQ(first->activeStateSnapshotRevision(), 1U);
  EXPECT_TRUE(first->issuedBy(publisher.identity()));
  EXPECT_EQ(
    first->activeStateSnapshotStatus(),
    livo::ActiveStateSnapshotStatus::kValid);
  const auto & first_composition =
    first->validatedComposition();
  EXPECT_TRUE(first_composition.issuedBy(
      queue.batchAuthorityIdentity(), publisher.identity()));
  EXPECT_TRUE(first_composition.accepts(*first));
  EXPECT_TRUE(first->timeTransferBundle().
    allPresentFullStateTransfersValid());
  EXPECT_TRUE(
    first->activeStateSnapshot().pos_end.isApprox(
      first_state.pos_end, 0.0));
  EXPECT_DOUBLE_EQ(
    first->activeStateSnapshot().inv_expo_time,
    first_state.inv_expo_time);
  EXPECT_TRUE(
    first->activeStateSnapshot().cov.isApprox(
      first_state.cov, 0.0));
  EXPECT_EQ(
    first_batch.validatedReceipt()->
    validationImageHashCount(),
    4U);

  CameraObservationBatch second_batch =
    ownedBatch(queue, 11.0, 31U);
  auto second = publisher.publish(
    second_batch, first_state, second_batch.timestamp,
    0.0, transfer);
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->publicationSequence(), 2U);
  EXPECT_EQ(second->activeStateSnapshotRevision(), 2U);
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  PresentCameraProjectsOwnerIssuedFrameProvenanceAndAbsentDoesNot)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedPartialBatch(queue, 15.0, 1U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  const auto runtime = publisher.publish(
    batch, state, batch.timestamp, 0.0, transfer);
  ASSERT_TRUE(runtime.has_value());

  EXPECT_FALSE(runtime->frameProvenance(1U).has_value());
  EXPECT_FALSE(runtime->frameProvenance(4U).has_value());
  const auto frame = runtime->frameProvenance(2U);
  ASSERT_TRUE(frame.has_value());
  EXPECT_TRUE(frame->valid());
  EXPECT_TRUE(frame->issuedBy(publisher.identity()));
  EXPECT_TRUE(
    frame->accepts(
      runtime->validatedComposition()));
  EXPECT_EQ(frame->cameraBatchSequence(), 1U);
  EXPECT_EQ(frame->runtimePublicationSequence(), 1U);
  EXPECT_EQ(frame->activeStateSnapshotRevision(), 1U);
  EXPECT_EQ(frame->cameraId(), 2U);
  EXPECT_EQ(frame->observationGeneration(), 1U);
  EXPECT_EQ(frame->imageGeneration(), 1U);
  EXPECT_DOUBLE_EQ(frame->exposureTimestampSec(), 15.0);
  EXPECT_TRUE(frame->timeTransferEntry().has_value());
  EXPECT_EQ(frame->timeTransferEntry().cameraId(), 2U);
  EXPECT_DOUBLE_EQ(
    frame->timeTransferEntry().targetTimestampSec(), 15.0);
  EXPECT_TRUE(frame->experimentalOnly());
  EXPECT_FALSE(frame->consumable());
  EXPECT_FALSE(frame->productionReady());
  EXPECT_FALSE(frame->solverApplied());
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  SameTimestampForeignImageOwnerCannotBePublishedByPinnedOwner)
{
  CameraObservationQueue expected_queue;
  CameraObservationQueue foreign_queue;
  CameraObservationBatch expected =
    ownedBatch(expected_queue, 20.0, 7U);
  CameraObservationBatch foreign =
    ownedBatch(foreign_queue, 20.0, 99U);
  ASSERT_DOUBLE_EQ(expected.timestamp, foreign.timestamp);

  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    expected_queue.batchAuthorityIdentity());
  EXPECT_FALSE(
    publisher.identity().bindsBatchAuthority(
      foreign_queue.batchAuthorityIdentity()));
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  EXPECT_TRUE(publisher.publish(
      expected, state, expected.timestamp, 0.0,
      transfer).has_value());
  EXPECT_FALSE(publisher.publish(
      foreign, state, foreign.timestamp, 0.0,
      transfer).has_value());
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  MutatedBatchIsRejectedWithoutConsumingPublicationSequence)
{
  CameraObservationQueue queue;
  CameraObservationBatch mutated =
    ownedBatch(queue, 25.0, 17U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;

  mutated.observations[2U].image.at<std::uint8_t>(0, 0) =
    91U;
  EXPECT_FALSE(
    publisher.publish(
      mutated, state, mutated.timestamp, 0.0,
      transfer).has_value());

  CameraObservationBatch accepted =
    ownedBatch(queue, 26.0, 29U);
  const auto publication = publisher.publish(
    accepted, state, accepted.timestamp, 0.0, transfer);
  ASSERT_TRUE(publication.has_value());
  EXPECT_EQ(publication->publicationSequence(), 1U);
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  OwnerCompositionIsIndependentOfCallerBatchMutation)
{
  CameraObservationQueue image_queue;
  CameraObservationBatch image_mutated =
    ownedBatch(image_queue, 27.0, 63U);
  livo::ReferenceOnlyMigrationRuntimePublisher image_publisher(
    image_queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  const auto image_runtime = image_publisher.publish(
    image_mutated, state, image_mutated.timestamp, 0.0,
    transfer);
  ASSERT_TRUE(image_runtime.has_value());
  const auto image_frame =
    image_runtime->frameProvenance(0U);
  ASSERT_TRUE(image_frame.has_value());

  image_mutated.observations[0U].image.at<std::uint8_t>(
    0, 0) = 88U;
  EXPECT_TRUE(
    image_frame->accepts(
      image_runtime->validatedComposition()));

  CameraObservationQueue swap_queue;
  CameraObservationBatch camera_swapped =
    ownedBatch(swap_queue, 28.0, 71U);
  livo::ReferenceOnlyMigrationRuntimePublisher swap_publisher(
    swap_queue.batchAuthorityIdentity());
  const auto swap_runtime = swap_publisher.publish(
    camera_swapped, state, camera_swapped.timestamp, 0.0,
    transfer);
  ASSERT_TRUE(swap_runtime.has_value());
  const auto swap_frame =
    swap_runtime->frameProvenance(0U);
  ASSERT_TRUE(swap_frame.has_value());

  std::swap(
    camera_swapped.observations[0U],
    camera_swapped.observations[1U]);
  EXPECT_TRUE(
    swap_frame->accepts(
      swap_runtime->validatedComposition()));
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  FrameProvenanceRejectsForeignPublisherAndSealedEntry)
{
  CameraObservationQueue queue;
  const CameraObservationBatch batch =
    ownedBatch(queue, 29.0, 79U);
  const CameraObservationBatch foreign_batch =
    ownedBatch(queue, 29.1, 81U);
  livo::ReferenceOnlyMigrationRuntimePublisher owner(
    queue.batchAuthorityIdentity());
  livo::ReferenceOnlyMigrationRuntimePublisher foreign(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  const auto owner_runtime = owner.publish(
    batch, state, batch.timestamp, 0.0, transfer);
  ASSERT_TRUE(batch.validatedReceipt().has_value());
  const std::uint64_t validation_hash_count =
    batch.validatedReceipt()->validationImageHashCount();
  livo::ReferenceOnlyMigrationRuntimePublisher replay(
    queue.batchAuthorityIdentity());
  EXPECT_FALSE(replay.publish(
      batch, state, batch.timestamp, 0.0, transfer).
    has_value());
  EXPECT_EQ(
    batch.validatedReceipt()->validationImageHashCount(),
    validation_hash_count);
  const auto foreign_runtime = foreign.publish(
    foreign_batch, state, foreign_batch.timestamp, 0.0,
    transfer);
  ASSERT_TRUE(owner_runtime.has_value());
  ASSERT_TRUE(foreign_runtime.has_value());
  const auto owner_frame =
    owner_runtime->frameProvenance(3U);
  ASSERT_TRUE(owner_frame.has_value());

  EXPECT_TRUE(owner_frame->issuedBy(owner.identity()));
  EXPECT_FALSE(owner_frame->issuedBy(foreign.identity()));
  EXPECT_FALSE(
    owner_frame->accepts(
      foreign_runtime->validatedComposition()));
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  FrameProvenanceRejectsForeignQueueRuntimeBatch)
{
  CameraObservationQueue owner_queue;
  CameraObservationQueue foreign_queue;
  const CameraObservationBatch owner_batch =
    ownedBatch(owner_queue, 31.0, 83U);
  const CameraObservationBatch foreign_batch =
    ownedBatch(foreign_queue, 31.0, 83U);
  livo::ReferenceOnlyMigrationRuntimePublisher owner(
    owner_queue.batchAuthorityIdentity());
  livo::ReferenceOnlyMigrationRuntimePublisher foreign(
    foreign_queue.batchAuthorityIdentity());
  StatesGroup state;
  livo::CameraTimeTransfer transfer;
  const auto owner_runtime = owner.publish(
    owner_batch, state, owner_batch.timestamp, 0.0,
    transfer);
  const auto foreign_runtime = foreign.publish(
    foreign_batch, state, foreign_batch.timestamp, 0.0,
    transfer);
  ASSERT_TRUE(owner_runtime.has_value());
  ASSERT_TRUE(foreign_runtime.has_value());
  const auto owner_frame =
    owner_runtime->frameProvenance(1U);
  ASSERT_TRUE(owner_frame.has_value());

  EXPECT_FALSE(
    owner_frame->accepts(
      foreign_runtime->validatedComposition()));
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  ForeignStateOrCovariancePublicationHasDifferentAuthority)
{
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedBatch(queue, 30.0, 41U);
  CameraObservationBatch foreign_batch =
    ownedBatch(queue, 30.1, 43U);
  livo::ReferenceOnlyMigrationRuntimePublisher expected_publisher(
    queue.batchAuthorityIdentity());
  livo::ReferenceOnlyMigrationRuntimePublisher foreign_publisher(
    queue.batchAuthorityIdentity());
  StatesGroup expected_state;
  StatesGroup foreign_state = expected_state;
  foreign_state.pos_end.x() = 0.25;
  foreign_state.cov(3, 3) *= 9.0;
  livo::CameraTimeTransfer transfer;

  const auto expected = expected_publisher.publish(
    batch, expected_state, batch.timestamp, 0.0,
    transfer);
  const auto foreign = foreign_publisher.publish(
    foreign_batch, foreign_state, foreign_batch.timestamp,
    0.0,
    transfer);
  ASSERT_TRUE(expected.has_value());
  ASSERT_TRUE(foreign.has_value());
  EXPECT_TRUE(expected->issuedBy(expected_publisher.identity()));
  EXPECT_FALSE(foreign->issuedBy(expected_publisher.identity()));
  EXPECT_FALSE(
    expected->activeStateSnapshot().pos_end.isApprox(
      foreign->activeStateSnapshot().pos_end, 0.0));
  EXPECT_NE(
    expected->activeStateSnapshot().cov(3, 3),
    foreign->activeStateSnapshot().cov(3, 3));
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  TinyCovarianceAsymmetryWithinToleranceIsAccepted)
{
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedBatch(queue, 39.0, 49U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  state.cov(0, 1) = 5.0e-12;
  state.cov(1, 0) = 0.0;
  livo::CameraTimeTransfer transfer;

  const auto publication = publisher.publish(
    batch, state, batch.timestamp, 0.0, transfer);
  ASSERT_TRUE(publication.has_value());
  EXPECT_EQ(
    publication->activeStateSnapshotStatus(),
    livo::ActiveStateSnapshotStatus::kValid);
}

TEST(
  ReferenceOnlyMigrationRuntimeProvenance,
  InvalidCovarianceIsPreservedForExplicitFailClosedDiagnosis)
{
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    ownedBatch(queue, 40.0, 51U);
  livo::ReferenceOnlyMigrationRuntimePublisher publisher(
    queue.batchAuthorityIdentity());
  StatesGroup state;
  state.cov(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  livo::CameraTimeTransfer transfer;

  const auto publication = publisher.publish(
    batch, state, batch.timestamp, 0.0, transfer);
  ASSERT_TRUE(publication.has_value());
  EXPECT_EQ(
    publication->activeStateSnapshotStatus(),
    livo::ActiveStateSnapshotStatus::kInvalidCovariance);
  EXPECT_FALSE(
    publication->frameProvenance(0U).has_value());
}

}  // namespace
