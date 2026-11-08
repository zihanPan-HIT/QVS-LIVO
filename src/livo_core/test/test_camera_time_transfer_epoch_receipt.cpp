#include "camera_time_transfer.h"
#include "camera_time_transfer_failure_injection_test_support.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{

static_assert(
  !std::is_default_constructible_v<
    livo::CameraTimeTransferAuthorityIdentity>);
static_assert(
  !std::is_default_constructible_v<
    livo::CameraTimeTransferEpochReceipt>);
static_assert(
  !std::is_default_constructible_v<
    livo::EpochBoundCameraPoseTransfer>);
static_assert(
  !std::is_constructible_v<
    livo::EpochBoundCameraPoseTransfer,
    livo::CameraPoseTransfer>);
static_assert(
  !std::is_convertible_v<
    livo::CameraPoseTransfer,
    livo::EpochBoundCameraPoseTransfer>);
static_assert(
  std::is_copy_constructible_v<
    livo::CameraTimeTransferEpochReceipt>);
static_assert(
  std::is_copy_constructible_v<
    livo::CameraTimeTransferAuthorityIdentity>);

livo::CameraTimeTransferConfig testConfig()
{
  livo::CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.1;
  config.max_imu_gap_sec = 0.02;
  config.maximum_buffered_samples = 16U;
  return config;
}

livo::CameraTimeTransferImuSample sample(
  const double timestamp_sec,
  const double acceleration_x = 0.0)
{
  livo::CameraTimeTransferImuSample result;
  result.timestamp_sec = timestamp_sec;
  result.angular_velocity_radps.setZero();
  result.specific_force_mps2 =
    Eigen::Vector3d(acceleration_x, 0.0, 0.0);
  return result;
}

livo::CameraTimeTransferState anchor(const double timestamp_sec)
{
  livo::CameraTimeTransferState result;
  result.timestamp_sec = timestamp_sec;
  result.gravity_world_mps2.setZero();
  return result;
}

void appendCoveredWindow(livo::CameraTimeTransfer & transfer)
{
  ASSERT_EQ(
    transfer.appendImu(sample(1.00, 1.0)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  ASSERT_EQ(
    transfer.appendImu(sample(1.01, 1.0)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  ASSERT_EQ(
    transfer.appendImu(sample(1.02, 1.0)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
}

TEST(
  CameraTimeTransferEpochReceipt,
  EmptyAuthorityCannotPublishEpochReceipt)
{
  livo::CameraTimeTransfer transfer;
  const livo::CameraTimeTransferEpochCaptureResult captured =
    transfer.captureEpoch();

  EXPECT_EQ(
    captured.status,
    livo::CameraTimeTransferEpochCaptureStatus::kEmpty);
  EXPECT_FALSE(captured.receipt.has_value());
}

TEST(
  CameraTimeTransferEpochReceipt,
  SameSnapshotMintsExactOwnerBoundTransfer)
{
  livo::CameraTimeTransfer transfer(testConfig());
  appendCoveredWindow(transfer);
  const auto identity = transfer.authorityIdentity();
  ASSERT_TRUE(identity.has_value());

  const auto captured = transfer.captureEpoch();
  ASSERT_EQ(
    captured.status,
    livo::CameraTimeTransferEpochCaptureStatus::kCaptured);
  ASSERT_TRUE(captured.receipt.has_value());
  EXPECT_TRUE(captured.receipt->valid());
  EXPECT_TRUE(captured.receipt->belongsTo(*identity));
  EXPECT_EQ(captured.receipt->epoch(), 1U);
  EXPECT_EQ(captured.receipt->streamRevision(), 4U);
  EXPECT_EQ(captured.receipt->snapshotSequence(), 1U);
  EXPECT_EQ(captured.receipt->bufferedSamples(), 3U);

  const auto computed =
    captured.receipt->transfer(anchor(1.00), 1.02);
  ASSERT_EQ(
    computed.status, livo::CameraTimeTransferStatus::kValid);
  ASSERT_TRUE(computed.value.has_value());
  EXPECT_TRUE(computed.value->valid());
  EXPECT_TRUE(
    computed.value->wasProducedBy(*captured.receipt));
  EXPECT_EQ(computed.value->raw().imu_epoch, 1U);
  EXPECT_NEAR(
    computed.value->raw().position_world_m.x(),
    0.5 * 1.0 * 0.02 * 0.02, 1.0e-12);
}

TEST(
  CameraTimeTransferEpochReceipt,
  ForeignOwnerAndIndependentSnapshotCannotAuthorizeBoundValue)
{
  livo::CameraTimeTransfer first(testConfig());
  livo::CameraTimeTransfer second(testConfig());
  appendCoveredWindow(first);
  appendCoveredWindow(second);
  const auto first_identity = first.authorityIdentity();
  const auto second_identity = second.authorityIdentity();
  const auto first_capture = first.captureEpoch();
  const auto second_capture = second.captureEpoch();
  const auto repeated_capture = first.captureEpoch();
  ASSERT_TRUE(first_identity && second_identity);
  ASSERT_TRUE(first_capture.receipt);
  ASSERT_TRUE(second_capture.receipt);
  ASSERT_TRUE(repeated_capture.receipt);

  EXPECT_TRUE(
    first_capture.receipt->belongsTo(*first_identity));
  EXPECT_FALSE(
    first_capture.receipt->belongsTo(*second_identity));
  const auto computed =
    first_capture.receipt->transfer(anchor(1.00), 1.02);
  ASSERT_TRUE(computed.value);
  EXPECT_FALSE(
    computed.value->wasProducedBy(*second_capture.receipt));
  EXPECT_FALSE(
    computed.value->wasProducedBy(*repeated_capture.receipt));
  EXPECT_EQ(
    repeated_capture.receipt->snapshotSequence(), 2U);
}

TEST(
  CameraTimeTransferEpochReceipt,
  MoveTransfersAuthorityIdentityAndInvalidatesMovedFromHandle)
{
  livo::CameraTimeTransfer source(testConfig());
  appendCoveredWindow(source);
  const auto identity_before_move = source.authorityIdentity();
  ASSERT_TRUE(identity_before_move);

  livo::CameraTimeTransfer destination(std::move(source));
  const auto identity_after_move =
    destination.authorityIdentity();
  ASSERT_TRUE(identity_after_move);
  const auto capture = destination.captureEpoch();
  ASSERT_TRUE(capture.receipt);
  EXPECT_TRUE(
    capture.receipt->belongsTo(*identity_before_move));
  EXPECT_TRUE(
    capture.receipt->belongsTo(*identity_after_move));
  EXPECT_FALSE(source.authorityIdentity().has_value());
  EXPECT_EQ(
    source.captureEpoch().status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kInvalidAuthority);
}

TEST(
  CameraTimeTransferEpochReceipt,
  MoveAssignmentTransfersSourceOwnerAndPreservesOldLineageAudit)
{
  livo::CameraTimeTransfer destination(testConfig());
  appendCoveredWindow(destination);
  const auto old_target_identity =
    destination.authorityIdentity();
  const auto old_target_receipt =
    destination.captureEpoch().receipt;
  ASSERT_TRUE(old_target_identity);
  ASSERT_TRUE(old_target_receipt);

  livo::CameraTimeTransfer source(testConfig());
  appendCoveredWindow(source);
  const auto source_identity = source.authorityIdentity();
  const auto source_receipt = source.captureEpoch().receipt;
  ASSERT_TRUE(source_identity);
  ASSERT_TRUE(source_receipt);

  destination = std::move(source);
  const auto moved_identity = destination.authorityIdentity();
  ASSERT_TRUE(moved_identity);
  EXPECT_TRUE(source_receipt->belongsTo(*moved_identity));
  EXPECT_FALSE(old_target_receipt->belongsTo(*moved_identity));
  EXPECT_TRUE(old_target_identity->valid());
  EXPECT_TRUE(
    old_target_receipt->belongsTo(*old_target_identity));
  EXPECT_FALSE(source_receipt->belongsTo(*old_target_identity));
  EXPECT_TRUE(old_target_receipt->valid());
  EXPECT_TRUE(
    old_target_receipt->transfer(anchor(1.00), 1.02).value);
  EXPECT_FALSE(source.authorityIdentity());
}

TEST(
  CameraTimeTransferEpochReceipt,
  FrozenReceiptSurvivesAuthorityDestructionAppendAndClear)
{
  std::optional<livo::CameraTimeTransferEpochReceipt> receipt;
  {
    livo::CameraTimeTransfer transfer(testConfig());
    appendCoveredWindow(transfer);
    receipt = transfer.captureEpoch().receipt;
    ASSERT_TRUE(receipt);
    ASSERT_EQ(
      transfer.appendImu(sample(1.03, 50.0)),
      livo::CameraTimeTransferAppendStatus::kAccepted);
    ASSERT_EQ(
      transfer.clear(),
      livo::CameraTimeTransferClearStatus::kCleared);
    const auto still_frozen =
      receipt->transfer(anchor(1.00), 1.02);
    ASSERT_TRUE(still_frozen.value);
    EXPECT_NEAR(
      still_frozen.value->raw().position_world_m.x(),
      0.5 * 1.0 * 0.02 * 0.02, 1.0e-12);
  }

  const auto after_destruction =
    receipt->transfer(anchor(1.00), 1.02);
  ASSERT_TRUE(after_destruction.value);
  EXPECT_NEAR(
    after_destruction.value->raw().position_world_m.x(),
    0.5 * 1.0 * 0.02 * 0.02, 1.0e-12);
}

TEST(
  CameraTimeTransferEpochReceipt,
  SameEpochReceiptSupportsValidBackwardTransfer)
{
  livo::CameraTimeTransfer transfer(testConfig());
  appendCoveredWindow(transfer);
  const auto captured = transfer.captureEpoch();
  ASSERT_TRUE(captured.receipt);

  const auto backward =
    captured.receipt->transfer(anchor(1.02), 1.00);
  ASSERT_EQ(
    backward.status, livo::CameraTimeTransferStatus::kValid);
  ASSERT_TRUE(backward.value);
  EXPECT_EQ(
    backward.value->raw().direction,
    livo::CameraTimeTransferDirection::kBackward);
  EXPECT_TRUE(backward.value->wasProducedBy(*captured.receipt));
}

TEST(
  CameraTimeTransferEpochReceipt,
  WindowCaptureFreezesOnlyRequiredOrderedBrackets)
{
  livo::CameraTimeTransfer transfer(testConfig());
  ASSERT_EQ(
    transfer.appendImu(sample(0.99)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  appendCoveredWindow(transfer);
  ASSERT_EQ(
    transfer.appendImu(sample(1.03)),
    livo::CameraTimeTransferAppendStatus::kAccepted);

  const auto capture = transfer.captureWindow(1.005, 1.015);
  ASSERT_EQ(
    capture.status,
    livo::CameraTimeTransferEpochCaptureStatus::kCaptured);
  ASSERT_TRUE(capture.receipt);
  EXPECT_EQ(capture.receipt->bufferedSamples(), 3U);
  const auto computed =
    capture.receipt->transfer(anchor(1.005), 1.015);
  EXPECT_EQ(
    computed.status, livo::CameraTimeTransferStatus::kValid);

  EXPECT_EQ(
    transfer.captureWindow(0.98, 1.01).status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kWindowNotCovered);
  EXPECT_EQ(
    transfer.captureWindow(1.02, 1.01).status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kInvalidInput);
}

TEST(
  CameraTimeTransferEpochReceipt,
  WindowCaptureUsesSameEpsilonAtBothEndpoints)
{
  constexpr double kInsideEpsilonSec = 5.0e-13;
  constexpr double kOutsideEpsilonSec = 2.0e-12;
  livo::CameraTimeTransfer transfer(testConfig());
  appendCoveredWindow(transfer);

  const auto within = transfer.captureWindow(
    1.00 - kInsideEpsilonSec,
    1.02 + kInsideEpsilonSec);
  EXPECT_EQ(
    within.status,
    livo::CameraTimeTransferEpochCaptureStatus::kCaptured);
  ASSERT_TRUE(within.receipt);
  EXPECT_TRUE(within.receipt->valid());

  EXPECT_EQ(
    transfer.captureWindow(
      1.00 - kOutsideEpsilonSec, 1.02).status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kWindowNotCovered);
  EXPECT_EQ(
    transfer.captureWindow(
      1.00, 1.02 + kOutsideEpsilonSec).status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kWindowNotCovered);
}

TEST(
  CameraTimeTransferEpochReceipt,
  CapturedEpochRetainsBoundaryAndRejectsCrossEpochTransfer)
{
  livo::CameraTimeTransferConfig config = testConfig();
  config.max_abs_transfer_sec = 0.6;
  livo::CameraTimeTransfer transfer(config);
  ASSERT_EQ(
    transfer.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  ASSERT_EQ(
    transfer.appendImu(sample(1.01)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  ASSERT_EQ(
    transfer.appendImu(sample(0.50)),
    livo::CameraTimeTransferAppendStatus::kNewEpoch);
  ASSERT_EQ(
    transfer.appendImu(sample(0.51)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  const auto captured = transfer.captureEpoch();
  ASSERT_TRUE(captured.receipt);
  EXPECT_EQ(captured.receipt->epoch(), 2U);

  const auto crosses =
    captured.receipt->transfer(anchor(1.01), 0.50);
  EXPECT_EQ(
    crosses.status,
    livo::CameraTimeTransferStatus::kCrossesImuEpoch);
  EXPECT_FALSE(crosses.value);
}

TEST(
  CameraTimeTransferEpochReceipt,
  CapacityFailureDoesNotMutateRingOrRevision)
{
  livo::CameraTimeTransferConfig config = testConfig();
  config.maximum_buffered_samples = 2U;
  livo::CameraTimeTransfer transfer(config);
  ASSERT_EQ(
    transfer.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  ASSERT_EQ(
    transfer.appendImu(sample(1.01)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  const auto before = transfer.captureEpoch();
  ASSERT_TRUE(before.receipt);

  EXPECT_EQ(
    transfer.appendImu(sample(1.02)),
    livo::CameraTimeTransferAppendStatus::
    kCapacityExceeded);
  EXPECT_EQ(transfer.bufferedSamples(), 2U);
  const auto after = transfer.captureEpoch();
  ASSERT_TRUE(after.receipt);
  EXPECT_EQ(
    after.receipt->streamRevision(),
    before.receipt->streamRevision());

  std::vector<livo::CameraTimeTransferImuSample> dense{
    sample(1.00), sample(1.01), sample(1.02)};
  EXPECT_EQ(
    transfer.transferWithSamples(
      anchor(1.00), dense, 1.02).status,
    livo::CameraTimeTransferStatus::kCapacityExceeded);
}

TEST(
  CameraTimeTransferEpochReceipt,
  InvalidCapacityAndClearHaveTypedFailClosedStatuses)
{
  livo::CameraTimeTransferConfig invalid = testConfig();
  invalid.maximum_buffered_samples = 1U;
  livo::CameraTimeTransfer transfer(invalid);

  EXPECT_FALSE(transfer.configValid());
  EXPECT_EQ(
    transfer.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::
    kInvalidConfiguration);
  EXPECT_EQ(
    transfer.clear(),
    livo::CameraTimeTransferClearStatus::
    kInvalidConfiguration);
  EXPECT_EQ(
    transfer.captureEpoch().status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kInvalidConfiguration);
}

TEST(
  CameraTimeTransferEpochReceipt,
  FactoryAllocationFailureCreatesTypedInvalidAuthority)
{
  livo::test::failNextCameraTimeTransferFactoryAllocation();
  livo::CameraTimeTransfer transfer(testConfig());

  EXPECT_FALSE(transfer.configValid());
  EXPECT_FALSE(transfer.authorityIdentity().has_value());
  EXPECT_EQ(
    transfer.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::
    kAuthorityUnavailable);
  EXPECT_EQ(
    transfer.clear(),
    livo::CameraTimeTransferClearStatus::
    kAuthorityUnavailable);
  EXPECT_EQ(
    transfer.captureEpoch().status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kInvalidAuthority);
}

TEST(
  CameraTimeTransferEpochReceipt,
  AppendAndEpochJumpAllocationFailuresHaveNoPartialWrite)
{
  livo::CameraTimeTransfer transfer(testConfig());
  ASSERT_EQ(
    transfer.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  const auto initial = transfer.captureEpoch();
  ASSERT_TRUE(initial.receipt);

  livo::test::failNextCameraTimeTransferAppendAllocation();
  EXPECT_EQ(
    transfer.appendImu(sample(1.01)),
    livo::CameraTimeTransferAppendStatus::kResourceFailure);
  EXPECT_EQ(transfer.bufferedSamples(), 1U);
  EXPECT_EQ(transfer.epoch(), 1U);
  const auto after_normal_failure = transfer.captureEpoch();
  ASSERT_TRUE(after_normal_failure.receipt);
  EXPECT_EQ(
    after_normal_failure.receipt->streamRevision(),
    initial.receipt->streamRevision());

  livo::test::failNextCameraTimeTransferAppendAllocation();
  EXPECT_EQ(
    transfer.appendImu(sample(0.50)),
    livo::CameraTimeTransferAppendStatus::kResourceFailure);
  EXPECT_EQ(transfer.bufferedSamples(), 1U);
  EXPECT_EQ(transfer.epoch(), 1U);
  const auto after_jump_failure = transfer.captureEpoch();
  ASSERT_TRUE(after_jump_failure.receipt);
  EXPECT_EQ(
    after_jump_failure.receipt->streamRevision(),
    initial.receipt->streamRevision());
}

TEST(
  CameraTimeTransferEpochReceipt,
  ClearAndCaptureResourceFailuresAreStronglyAtomic)
{
  livo::CameraTimeTransfer transfer(testConfig());
  appendCoveredWindow(transfer);
  const auto first = transfer.captureEpoch();
  ASSERT_TRUE(first.receipt);

  livo::test::failNextCameraTimeTransferClearResource();
  EXPECT_EQ(
    transfer.clear(),
    livo::CameraTimeTransferClearStatus::kResourceFailure);
  EXPECT_EQ(transfer.bufferedSamples(), 3U);
  EXPECT_EQ(transfer.epoch(), 1U);

  livo::test::failNextCameraTimeTransferCaptureAllocation();
  const auto failed = transfer.captureEpoch();
  EXPECT_EQ(
    failed.status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kResourceFailure);
  EXPECT_FALSE(failed.receipt);

  const auto second = transfer.captureEpoch();
  ASSERT_TRUE(second.receipt);
  EXPECT_EQ(second.receipt->snapshotSequence(), 2U);
  EXPECT_EQ(
    second.receipt->streamRevision(),
    first.receipt->streamRevision());
}

TEST(
  CameraTimeTransferEpochReceipt,
  CounterExhaustionHasNoPartialWriteOrSequenceConsumption)
{
  livo::CameraTimeTransfer revision(testConfig());
  ASSERT_EQ(
    revision.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  const auto first = revision.captureEpoch();
  ASSERT_TRUE(first.receipt);

  livo::test::setNextCameraTimeTransferRevisionToMaximum();
  EXPECT_EQ(
    revision.appendImu(sample(1.01)),
    livo::CameraTimeTransferAppendStatus::
    kRevisionExhausted);
  EXPECT_EQ(revision.bufferedSamples(), 1U);
  const auto maximum_revision = revision.captureEpoch();
  ASSERT_TRUE(maximum_revision.receipt);
  EXPECT_EQ(
    maximum_revision.receipt->streamRevision(),
    std::numeric_limits<std::uint64_t>::max());

  livo::CameraTimeTransfer epoch(testConfig());
  ASSERT_EQ(
    epoch.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  livo::test::setNextCameraTimeTransferEpochToMaximum();
  EXPECT_EQ(
    epoch.appendImu(sample(0.50)),
    livo::CameraTimeTransferAppendStatus::kEpochExhausted);
  EXPECT_EQ(
    epoch.epoch(),
    std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(epoch.bufferedSamples(), 1U);

  livo::CameraTimeTransfer sequence(testConfig());
  ASSERT_EQ(
    sequence.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  livo::test::
  setNextCameraTimeTransferSnapshotSequenceToMaximum();
  const auto exhausted = sequence.captureEpoch();
  EXPECT_EQ(
    exhausted.status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kSequenceExhausted);
  EXPECT_FALSE(exhausted.receipt);
  EXPECT_EQ(
    sequence.captureEpoch().status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kSequenceExhausted);

  livo::CameraTimeTransfer clear_epoch(testConfig());
  ASSERT_EQ(
    clear_epoch.appendImu(sample(1.0)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  livo::test::setNextCameraTimeTransferEpochToMaximum();
  EXPECT_EQ(
    clear_epoch.clear(),
    livo::CameraTimeTransferClearStatus::kEpochExhausted);
  EXPECT_EQ(clear_epoch.bufferedSamples(), 1U);

  livo::CameraTimeTransfer clear_revision(testConfig());
  ASSERT_EQ(
    clear_revision.appendImu(sample(1.0)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  livo::test::setNextCameraTimeTransferRevisionToMaximum();
  EXPECT_EQ(
    clear_revision.clear(),
    livo::CameraTimeTransferClearStatus::
    kRevisionExhausted);
  EXPECT_EQ(clear_revision.bufferedSamples(), 1U);
}

TEST(
  CameraTimeTransferEpochReceipt,
  NearMaximumCountersReachMaximumOnceThenNeverWrap)
{
  livo::CameraTimeTransfer revision(testConfig());
  ASSERT_EQ(
    revision.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  livo::test::setNextCameraTimeTransferRevisionNearMaximum();
  EXPECT_EQ(
    revision.appendImu(sample(1.01)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  const auto max_revision = revision.captureEpoch();
  ASSERT_TRUE(max_revision.receipt);
  EXPECT_EQ(
    max_revision.receipt->streamRevision(),
    std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(
    revision.appendImu(sample(1.02)),
    livo::CameraTimeTransferAppendStatus::
    kRevisionExhausted);

  livo::CameraTimeTransfer epoch(testConfig());
  ASSERT_EQ(
    epoch.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  livo::test::setNextCameraTimeTransferEpochNearMaximum();
  EXPECT_EQ(
    epoch.appendImu(sample(0.50)),
    livo::CameraTimeTransferAppendStatus::kNewEpoch);
  EXPECT_EQ(
    epoch.epoch(),
    std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(
    epoch.clear(),
    livo::CameraTimeTransferClearStatus::kEpochExhausted);

  livo::CameraTimeTransfer sequence(testConfig());
  ASSERT_EQ(
    sequence.appendImu(sample(1.00)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  livo::test::
  setNextCameraTimeTransferSnapshotSequenceNearMaximum();
  const auto max_sequence = sequence.captureEpoch();
  ASSERT_TRUE(max_sequence.receipt);
  EXPECT_EQ(
    max_sequence.receipt->snapshotSequence(),
    std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(
    sequence.captureEpoch().status,
    livo::CameraTimeTransferEpochCaptureStatus::
    kSequenceExhausted);
}

TEST(
  CameraTimeTransferEpochReceipt,
  ReceiptComputationFactoryFailureIsTypedAndDoesNotEscapeAnOwner)
{
  livo::CameraTimeTransfer transfer(testConfig());
  appendCoveredWindow(transfer);
  const auto captured = transfer.captureEpoch();
  ASSERT_TRUE(captured.receipt);

  livo::test::failNextCameraTimeTransferFactoryAllocation();
  const auto failed =
    captured.receipt->transfer(anchor(1.00), 1.02);
  EXPECT_EQ(
    failed.status,
    livo::CameraTimeTransferStatus::kResourceFailure);
  EXPECT_FALSE(failed.value);

  const auto recovered =
    captured.receipt->transfer(anchor(1.00), 1.02);
  ASSERT_EQ(
    recovered.status,
    livo::CameraTimeTransferStatus::kValid);
  ASSERT_TRUE(recovered.value);
  EXPECT_TRUE(
    recovered.value->wasProducedBy(*captured.receipt));
}

TEST(
  CameraTimeTransferEpochReceipt,
  NumericalAllocationFailureMapsToAppendedResourceStatus)
{
  static_assert(
    static_cast<int>(
      livo::CameraTimeTransferStatus::kResourceFailure) >
    static_cast<int>(
      livo::CameraTimeTransferStatus::
      kIllConditionedTransition));

  livo::CameraTimeTransfer transfer(testConfig());
  appendCoveredWindow(transfer);
  const auto captured = transfer.captureEpoch();
  ASSERT_TRUE(captured.receipt);

  livo::test::failNextCameraTimeTransferNumericalAllocation();
  const auto failed =
    captured.receipt->transfer(anchor(1.00), 1.02);
  EXPECT_EQ(
    failed.status,
    livo::CameraTimeTransferStatus::kResourceFailure);
  EXPECT_FALSE(failed.value);

  const auto recovered =
    captured.receipt->transfer(anchor(1.00), 1.02);
  EXPECT_EQ(
    recovered.status, livo::CameraTimeTransferStatus::kValid);
  EXPECT_TRUE(recovered.value);
}

TEST(
  CameraTimeTransferEpochReceipt,
  ConcurrentAppendAndCapturePublishOnlyOrderedFrozenWindows)
{
  livo::CameraTimeTransferConfig config = testConfig();
  config.ring_buffer_duration_sec = 10.0;
  config.epoch_jump_threshold_sec = 1.0;
  config.maximum_buffered_samples = 512U;
  livo::CameraTimeTransfer transfer(config);
  ASSERT_EQ(
    transfer.appendImu(sample(1.0)),
    livo::CameraTimeTransferAppendStatus::kAccepted);

  std::atomic<bool> start{false};
  std::atomic<bool> append_done{false};
  std::atomic<bool> failed{false};
  std::thread appender([&]() {
    while (!start.load()) {
      std::this_thread::yield();
    }
    for (std::size_t index = 1U; index <= 200U; ++index)
    {
      const auto status = transfer.appendImu(
        sample(1.0 + 0.001 * static_cast<double>(index)));
      if (status !=
        livo::CameraTimeTransferAppendStatus::kAccepted)
      {
        failed.store(true);
        break;
      }
      std::this_thread::yield();
    }
    append_done.store(true);
  });
  std::thread capturer([&]() {
    start.store(true);
    std::uint64_t previous_sequence = 0U;
    do
    {
      const auto captured = transfer.captureEpoch();
      if (
        captured.status !=
          livo::CameraTimeTransferEpochCaptureStatus::
          kCaptured ||
        !captured.receipt ||
        !captured.receipt->valid() ||
        captured.receipt->snapshotSequence() <=
          previous_sequence ||
        captured.receipt->bufferedSamples() >
          config.maximum_buffered_samples)
      {
        failed.store(true);
        break;
      }
      previous_sequence =
        captured.receipt->snapshotSequence();
      std::this_thread::yield();
    } while (!append_done.load());
  });
  appender.join();
  capturer.join();

  EXPECT_FALSE(failed.load());
  EXPECT_EQ(transfer.bufferedSamples(), 201U);
}

TEST(
  CameraTimeTransferEpochReceipt,
  ConcurrentClearAndCaptureAreLinearizableWithoutTornReceipt)
{
  livo::CameraTimeTransfer transfer(testConfig());
  ASSERT_EQ(
    transfer.appendImu(sample(1.0)),
    livo::CameraTimeTransferAppendStatus::kAccepted);
  std::atomic<bool> start{false};
  std::atomic<bool> mutation_done{false};
  std::atomic<bool> failed{false};

  std::thread mutator([&]() {
    while (!start.load()) {
      std::this_thread::yield();
    }
    for (std::size_t index = 0U; index < 100U; ++index)
    {
      if (transfer.clear() !=
        livo::CameraTimeTransferClearStatus::kCleared)
      {
        failed.store(true);
        break;
      }
      const double timestamp =
        2.0 + 0.01 * static_cast<double>(index);
      if (transfer.appendImu(sample(timestamp)) !=
        livo::CameraTimeTransferAppendStatus::kAccepted)
      {
        failed.store(true);
        break;
      }
      std::this_thread::yield();
    }
    mutation_done.store(true);
  });
  std::thread capturer([&]() {
    start.store(true);
    std::uint64_t previous_sequence = 0U;
    do
    {
      const auto captured = transfer.captureEpoch();
      if (captured.status ==
        livo::CameraTimeTransferEpochCaptureStatus::kCaptured)
      {
        if (
          !captured.receipt ||
          !captured.receipt->valid() ||
          captured.receipt->snapshotSequence() <=
            previous_sequence)
        {
          failed.store(true);
          break;
        }
        previous_sequence =
          captured.receipt->snapshotSequence();
      }
      else if (
        captured.status !=
        livo::CameraTimeTransferEpochCaptureStatus::kEmpty)
      {
        failed.store(true);
        break;
      }
      std::this_thread::yield();
    } while (!mutation_done.load());
  });
  mutator.join();
  capturer.join();

  EXPECT_FALSE(failed.load());
}

}  // namespace
