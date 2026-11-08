#include "adaptive_lio_mode_ledger.h"

#include <limits>

#include <gtest/gtest.h>

namespace
{

void expectRecorded(
  const livo::AdaptiveLioModeRecordResult & result,
  const std::uint64_t generation, const double source_stamp_sec,
  const livo::AdaptiveLioMode mode,
  const livo::AdaptiveLioModeReason reason,
  const livo::AdaptiveLioModeTransition transition)
{
  ASSERT_EQ(
    result.status,
    livo::AdaptiveLioModeRecordStatus::kRecorded);
  ASSERT_TRUE(result.receipt.has_value());
  EXPECT_EQ(result.receipt->generation(), generation);
  EXPECT_DOUBLE_EQ(
    result.receipt->sourceStampSec(), source_stamp_sec);
  EXPECT_EQ(result.receipt->mode(), mode);
  EXPECT_EQ(result.receipt->reason(), reason);
  EXPECT_EQ(result.receipt->transition(), transition);
}

void expectRejected(
  const livo::AdaptiveLioModeRecordResult & result,
  const livo::AdaptiveLioModeRecordStatus status)
{
  EXPECT_EQ(result.status, status);
  EXPECT_FALSE(result.receipt.has_value());
}

}  // namespace

TEST(
  AdaptiveLioModeLedgerTest,
  AcceptsUnobservedVisualOutcomeOnlyForLioOnly)
{
  livo::AdaptiveLioModeLedger ledger;

  expectRecorded(
    ledger.record({
      1U, 10.0,
      livo::AdaptiveLioMode::kLioOnly,
      livo::AdaptiveLioModeReason::
      kVisualOutcomeNotObservedBeforeCloseout}),
    1U, 10.0,
    livo::AdaptiveLioMode::kLioOnly,
    livo::AdaptiveLioModeReason::
    kVisualOutcomeNotObservedBeforeCloseout,
    livo::AdaptiveLioModeTransition::kEnterLioOnly);

  livo::AdaptiveLioModeLedger mismatch_ledger;
  expectRejected(
    mismatch_ledger.record({
      1U, 10.0,
      livo::AdaptiveLioMode::kSelectedVisual,
      livo::AdaptiveLioModeReason::
      kVisualOutcomeNotObservedBeforeCloseout}),
    livo::AdaptiveLioModeRecordStatus::kModeReasonMismatch);
}

TEST(
  AdaptiveLioModeLedgerTest,
  SealsExactlyOneMonotonicReceiptPerLidarGeneration)
{
  livo::AdaptiveLioModeLedger ledger;

  expectRecorded(
    ledger.record({
      1U, 10.0,
      livo::AdaptiveLioMode::kSelectedVisual,
      livo::AdaptiveLioModeReason::kEligibleVisualSelected}),
    1U, 10.0,
    livo::AdaptiveLioMode::kSelectedVisual,
    livo::AdaptiveLioModeReason::kEligibleVisualSelected,
    livo::AdaptiveLioModeTransition::kNone);
  expectRecorded(
    ledger.record({
      2U, 11.0,
      livo::AdaptiveLioMode::kLioOnly,
      livo::AdaptiveLioModeReason::
      kAllCamerasMissingAfterDeadline}),
    2U, 11.0,
    livo::AdaptiveLioMode::kLioOnly,
    livo::AdaptiveLioModeReason::
    kAllCamerasMissingAfterDeadline,
    livo::AdaptiveLioModeTransition::kEnterLioOnly);
  expectRecorded(
    ledger.record({
      3U, 12.0,
      livo::AdaptiveLioMode::kLioOnly,
      livo::AdaptiveLioModeReason::kNoEligibleVisual}),
    3U, 12.0,
    livo::AdaptiveLioMode::kLioOnly,
    livo::AdaptiveLioModeReason::kNoEligibleVisual,
    livo::AdaptiveLioModeTransition::kNone);
  expectRecorded(
    ledger.record({
      4U, 13.0,
      livo::AdaptiveLioMode::kSelectedVisual,
      livo::AdaptiveLioModeReason::kEligibleVisualSelected}),
    4U, 13.0,
    livo::AdaptiveLioMode::kSelectedVisual,
    livo::AdaptiveLioModeReason::kEligibleVisualSelected,
    livo::AdaptiveLioModeTransition::kExitLioOnly);

  expectRejected(
    ledger.record({
      4U, 14.0,
      livo::AdaptiveLioMode::kSelectedVisual,
      livo::AdaptiveLioModeReason::kEligibleVisualSelected}),
    livo::AdaptiveLioModeRecordStatus::
    kNonContinuousGeneration);
  expectRejected(
    ledger.record({
      6U, 14.0,
      livo::AdaptiveLioMode::kLioOnly,
      livo::AdaptiveLioModeReason::kNoEligibleVisual}),
    livo::AdaptiveLioModeRecordStatus::
    kNonContinuousGeneration);
  expectRejected(
    ledger.record({
      5U, 12.5,
      livo::AdaptiveLioMode::kLioOnly,
      livo::AdaptiveLioModeReason::kNoEligibleVisual}),
    livo::AdaptiveLioModeRecordStatus::
    kNonMonotonicSourceStamp);
  expectRejected(
    ledger.record({
      5U, 14.0,
      livo::AdaptiveLioMode::kSelectedVisual,
      livo::AdaptiveLioModeReason::kNoEligibleVisual}),
    livo::AdaptiveLioModeRecordStatus::kModeReasonMismatch);
  expectRejected(
    ledger.record({
      5U, std::numeric_limits<double>::infinity(),
      livo::AdaptiveLioMode::kLioOnly,
      livo::AdaptiveLioModeReason::kNoEligibleVisual}),
    livo::AdaptiveLioModeRecordStatus::kInvalidSourceStamp);

  expectRecorded(
    ledger.record({
      5U, 14.0,
      livo::AdaptiveLioMode::kSelectedVisual,
      livo::AdaptiveLioModeReason::kFrontFallback}),
    5U, 14.0,
    livo::AdaptiveLioMode::kSelectedVisual,
    livo::AdaptiveLioModeReason::kFrontFallback,
    livo::AdaptiveLioModeTransition::kNone);
}
