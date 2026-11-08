#ifndef LIVO_ADAPTIVE_LIO_MODE_LEDGER_H_
#define LIVO_ADAPTIVE_LIO_MODE_LEDGER_H_

#include <cstdint>
#include <optional>

namespace livo
{

enum class AdaptiveLioMode
{
  kSelectedVisual,
  kLioOnly,
};

enum class AdaptiveLioModeReason
{
  kEligibleVisualSelected,
  kFrontFallback,
  kAllCamerasMissingAfterDeadline,
  kNoEligibleVisual,
  kVisualOutcomeNotObservedBeforeCloseout,
};

enum class AdaptiveLioModeTransition
{
  kNone,
  kEnterLioOnly,
  kExitLioOnly,
};

enum class AdaptiveLioModeRecordStatus
{
  kRecorded,
  kZeroGeneration,
  kNonContinuousGeneration,
  kInvalidSourceStamp,
  kNonMonotonicSourceStamp,
  kModeReasonMismatch,
};

struct AdaptiveLioModeObservation
{
  std::uint64_t generation{0U};
  double source_stamp_sec{0.0};
  AdaptiveLioMode mode{AdaptiveLioMode::kLioOnly};
  AdaptiveLioModeReason reason{
    AdaptiveLioModeReason::kNoEligibleVisual};
};

class AdaptiveLioModeReceipt
{
public:
  std::uint64_t generation() const noexcept;
  double sourceStampSec() const noexcept;
  AdaptiveLioMode mode() const noexcept;
  AdaptiveLioModeReason reason() const noexcept;
  AdaptiveLioModeTransition transition() const noexcept;

private:
  friend class AdaptiveLioModeLedger;

  AdaptiveLioModeReceipt(
    std::uint64_t generation, double source_stamp_sec,
    AdaptiveLioMode mode, AdaptiveLioModeReason reason,
    AdaptiveLioModeTransition transition) noexcept;

  std::uint64_t generation_{0U};
  double source_stamp_sec_{0.0};
  AdaptiveLioMode mode_{AdaptiveLioMode::kLioOnly};
  AdaptiveLioModeReason reason_{
    AdaptiveLioModeReason::kNoEligibleVisual};
  AdaptiveLioModeTransition transition_{
    AdaptiveLioModeTransition::kNone};
};

struct AdaptiveLioModeRecordResult
{
  AdaptiveLioModeRecordStatus status{
    AdaptiveLioModeRecordStatus::kZeroGeneration};
  std::optional<AdaptiveLioModeReceipt> receipt;
};

class AdaptiveLioModeLedger
{
public:
  AdaptiveLioModeRecordResult record(
    const AdaptiveLioModeObservation & observation) noexcept;

private:
  bool has_record_{false};
  std::uint64_t last_generation_{0U};
  double last_source_stamp_sec_{0.0};
  AdaptiveLioMode last_mode_{AdaptiveLioMode::kSelectedVisual};
};

}  // namespace livo

#endif  // LIVO_ADAPTIVE_LIO_MODE_LEDGER_H_
