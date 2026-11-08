#include "adaptive_lio_mode_ledger.h"

#include <cmath>
#include <limits>

namespace livo
{

namespace
{

bool modeMatchesReason(
  const AdaptiveLioMode mode,
  const AdaptiveLioModeReason reason) noexcept
{
  if (mode == AdaptiveLioMode::kSelectedVisual)
  {
    return
      reason ==
      AdaptiveLioModeReason::kEligibleVisualSelected ||
      reason == AdaptiveLioModeReason::kFrontFallback;
  }
  return
    reason ==
      AdaptiveLioModeReason::
      kAllCamerasMissingAfterDeadline ||
    reason == AdaptiveLioModeReason::kNoEligibleVisual ||
    reason ==
      AdaptiveLioModeReason::
      kVisualOutcomeNotObservedBeforeCloseout;
}

}  // namespace

AdaptiveLioModeReceipt::AdaptiveLioModeReceipt(
  const std::uint64_t generation,
  const double source_stamp_sec,
  const AdaptiveLioMode mode,
  const AdaptiveLioModeReason reason,
  const AdaptiveLioModeTransition transition) noexcept
: generation_(generation),
  source_stamp_sec_(source_stamp_sec),
  mode_(mode),
  reason_(reason),
  transition_(transition)
{
}

std::uint64_t AdaptiveLioModeReceipt::generation()
  const noexcept
{
  return generation_;
}

double AdaptiveLioModeReceipt::sourceStampSec()
  const noexcept
{
  return source_stamp_sec_;
}

AdaptiveLioMode AdaptiveLioModeReceipt::mode()
  const noexcept
{
  return mode_;
}

AdaptiveLioModeReason AdaptiveLioModeReceipt::reason()
  const noexcept
{
  return reason_;
}

AdaptiveLioModeTransition
AdaptiveLioModeReceipt::transition() const noexcept
{
  return transition_;
}

AdaptiveLioModeRecordResult AdaptiveLioModeLedger::record(
  const AdaptiveLioModeObservation & observation) noexcept
{
  if (observation.generation == 0U)
  {
    return {
      AdaptiveLioModeRecordStatus::kZeroGeneration,
      std::nullopt};
  }
  if (
    (!has_record_ && observation.generation != 1U) ||
    (has_record_ &&
     (
       last_generation_ ==
       std::numeric_limits<std::uint64_t>::max() ||
       observation.generation != last_generation_ + 1U)))
  {
    return {
      AdaptiveLioModeRecordStatus::
      kNonContinuousGeneration,
      std::nullopt};
  }
  if (!std::isfinite(observation.source_stamp_sec))
  {
    return {
      AdaptiveLioModeRecordStatus::kInvalidSourceStamp,
      std::nullopt};
  }
  if (
    has_record_ &&
    observation.source_stamp_sec <= last_source_stamp_sec_)
  {
    return {
      AdaptiveLioModeRecordStatus::
      kNonMonotonicSourceStamp,
      std::nullopt};
  }
  if (!modeMatchesReason(observation.mode, observation.reason))
  {
    return {
      AdaptiveLioModeRecordStatus::kModeReasonMismatch,
      std::nullopt};
  }

  AdaptiveLioModeTransition transition =
    AdaptiveLioModeTransition::kNone;
  if (observation.mode == AdaptiveLioMode::kLioOnly)
  {
    if (
      !has_record_ ||
      last_mode_ != AdaptiveLioMode::kLioOnly)
    {
      transition =
        AdaptiveLioModeTransition::kEnterLioOnly;
    }
  }
  else if (
    has_record_ &&
    last_mode_ == AdaptiveLioMode::kLioOnly)
  {
    transition =
      AdaptiveLioModeTransition::kExitLioOnly;
  }

  const AdaptiveLioModeReceipt receipt{
    observation.generation,
    observation.source_stamp_sec,
    observation.mode,
    observation.reason,
    transition};
  has_record_ = true;
  last_generation_ = observation.generation;
  last_source_stamp_sec_ = observation.source_stamp_sec;
  last_mode_ = observation.mode;
  return {
    AdaptiveLioModeRecordStatus::kRecorded,
    receipt};
}

}  // namespace livo
