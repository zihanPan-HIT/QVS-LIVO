#ifndef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_H_
#define LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <opencv2/core/mat.hpp>

namespace livo
{

inline constexpr std::size_t kAdaptiveCameraSnapshotCameraCount = 4U;
using AdaptiveCameraSnapshotImageDigest =
  std::array<unsigned char, 32U>;

// All times share one event-time domain. The queue closes arrivals through
// U - reorder_window_sec and waits no later than
// LIO_reference + request_deadline_offset_sec. A request is valid only when
// deadline_offset >= reorder_window and max_abs_exposure_offset > 0.
struct AdaptiveCameraSnapshotQueueConfiguration
{
  std::size_t per_camera_capacity{0U};
  double reorder_window_sec{0.0};
  double request_deadline_offset_sec{0.0};
  double max_abs_exposure_offset_sec{0.0};
};

struct LioReference
{
  double timestamp_sec{0.0};
};

enum class AdaptiveCameraSnapshotQueueState
{
  IDLE,
  WAITING,
  SEALED_READY,
  LIO_ONLY
};

enum class AdaptiveCameraSnapshotQueueCreateStatus
{
  READY,
  INVALID_CONFIGURATION,
  RESOURCE_FAILURE
};

enum class AdaptiveCameraSnapshotPushStatus
{
  ACCEPTED,
  ACCEPTED_REORDERED,
  INVALID_CAMERA_ID,
  INVALID_TIMESTAMP,
  INVALID_IMAGE,
  DUPLICATE_TIMESTAMP,
  RETIRED_OR_REPLAY,
  ARRIVAL_CLOSED,
  CAPACITY_EXCEEDED,
  RESOURCE_FAILURE,
  AUTHORITY_EXHAUSTED
};

enum class AdaptiveCameraSnapshotRequestStatus
{
  WAITING,
  SEALED_READY,
  LIO_ONLY,
  ZERO_LIDAR_GENERATION,
  INVALID_LIO_REFERENCE,
  DEADLINE_OVERFLOW,
  STALE_OR_REPLAYED_GENERATION,
  NON_MONOTONIC_LIO_REFERENCE,
  DUPLICATE_ACTIVE_REQUEST,
  REQUEST_IDENTITY_MISMATCH,
  DIFFERENT_GENERATION_WHILE_ACTIVE,
  TERMINAL_RECEIPT_PENDING,
  RESOURCE_FAILURE,
  AUTHORITY_EXHAUSTED
};

enum class AdaptiveCameraSnapshotAdvanceStatus
{
  ADVANCED,
  UNCHANGED,
  INVALID_TIMESTAMP,
  REGRESSED
};

enum class AdaptiveCameraSnapshotTerminalDisposition
{
  CANDIDATES_READY,
  DEADLINE_EXPIRED_NO_FRESH_CAMERA
};

struct AdaptiveCameraSnapshotRequestResult
{
  AdaptiveCameraSnapshotRequestStatus status{
    AdaptiveCameraSnapshotRequestStatus::RESOURCE_FAILURE};
  AdaptiveCameraSnapshotQueueState state{
    AdaptiveCameraSnapshotQueueState::IDLE};
};

struct AdaptiveCameraSnapshotAdvanceResult
{
  AdaptiveCameraSnapshotAdvanceStatus status{
    AdaptiveCameraSnapshotAdvanceStatus::INVALID_TIMESTAMP};
  AdaptiveCameraSnapshotQueueState state{
    AdaptiveCameraSnapshotQueueState::IDLE};
};

struct AdaptiveCameraSnapshotQueueDiagnostics
{
  AdaptiveCameraSnapshotQueueState state{
    AdaptiveCameraSnapshotQueueState::IDLE};
  double latest_upstream_event_time_sec{0.0};
  double arrival_closed_through_sec{0.0};
  std::array<double, kAdaptiveCameraSnapshotCameraCount>
    retired_through_sec{};
  std::array<std::size_t, kAdaptiveCameraSnapshotCameraCount>
    queued_observation_count{};
};

class AdaptiveCameraSnapshotQueue;
class AdaptiveCameraSnapshotReceipt;
struct AdaptiveCameraSnapshotQueueCreateResult;

class AdaptiveCameraSnapshotQueueIdentity
{
public:
  AdaptiveCameraSnapshotQueueIdentity() = default;

  bool valid() const noexcept;
  bool sameAuthority(
    const AdaptiveCameraSnapshotQueueIdentity & other) const noexcept;

private:
  explicit AdaptiveCameraSnapshotQueueIdentity(
    std::shared_ptr<const void> owner) noexcept;

  std::shared_ptr<const void> owner_;

  friend class AdaptiveCameraSnapshotQueue;
  friend class AdaptiveCameraSnapshotReceipt;
};

// Immutable owner-issued terminal evidence. CANDIDATES_READY receipts retain
// private ownership of every frozen image but expose only value metadata and
// digests. LIO_ONLY receipts are valid and have an empty present mask.
class AdaptiveCameraSnapshotReceipt final
{
public:
  AdaptiveCameraSnapshotReceipt(
    const AdaptiveCameraSnapshotReceipt &) noexcept = default;
  AdaptiveCameraSnapshotReceipt(
    AdaptiveCameraSnapshotReceipt &&) noexcept = default;
  AdaptiveCameraSnapshotReceipt & operator=(
    const AdaptiveCameraSnapshotReceipt &) noexcept = default;
  AdaptiveCameraSnapshotReceipt & operator=(
    AdaptiveCameraSnapshotReceipt &&) noexcept = default;

  bool valid() const noexcept;
  bool issuedBy(
    const AdaptiveCameraSnapshotQueueIdentity & identity) const noexcept;

  std::uint64_t snapshotSequence() const noexcept;
  std::uint64_t lidarGeneration() const noexcept;
  double lioReferenceTimestampSec() const noexcept;
  double deadlineTimestampSec() const noexcept;
  double arrivalClosedThroughSec() const noexcept;
  double cutoffTimestampSec() const noexcept;
  AdaptiveCameraSnapshotTerminalDisposition disposition() const noexcept;
  std::array<bool, kAdaptiveCameraSnapshotCameraCount>
  presentMask() const noexcept;
  std::size_t cameraCount() const noexcept;
  bool cameraPresent(std::uint8_t camera_id) const noexcept;
  std::uint8_t cameraId(std::uint8_t camera_id) const noexcept;
  double exposureTimestampSec(std::uint8_t camera_id) const noexcept;
  std::uint64_t observationGeneration(
    std::uint8_t camera_id) const noexcept;
  std::uint64_t imageGeneration(
    std::uint8_t camera_id) const noexcept;
  int imageRows(std::uint8_t camera_id) const noexcept;
  int imageColumns(std::uint8_t camera_id) const noexcept;
  int imageType(std::uint8_t camera_id) const noexcept;
  AdaptiveCameraSnapshotImageDigest imageDigest(
    std::uint8_t camera_id) const noexcept;

private:
  struct SealedData;

  explicit AdaptiveCameraSnapshotReceipt(
    std::shared_ptr<const SealedData> sealed_data) noexcept;

  std::shared_ptr<const SealedData> sealed_data_;

  friend class AdaptiveCameraSnapshotQueue;
};

// Four-camera event-time queue. Camera pushes never advance event time.
// advanceUpstreamEventTime() is the sole arrival-closure/deadline clock.
// Each accepted request follows IDLE -> WAITING -> one terminal state, and
// takeFinalized() is the only transition from a terminal state back to IDLE.
class AdaptiveCameraSnapshotQueue final
{
public:
  static AdaptiveCameraSnapshotQueueCreateResult create(
    const AdaptiveCameraSnapshotQueueConfiguration & configuration) noexcept;

  ~AdaptiveCameraSnapshotQueue();

  AdaptiveCameraSnapshotQueue(
    const AdaptiveCameraSnapshotQueue &) = delete;
  AdaptiveCameraSnapshotQueue & operator=(
    const AdaptiveCameraSnapshotQueue &) = delete;
  AdaptiveCameraSnapshotQueue(
    AdaptiveCameraSnapshotQueue &&) = delete;
  AdaptiveCameraSnapshotQueue & operator=(
    AdaptiveCameraSnapshotQueue &&) = delete;

  AdaptiveCameraSnapshotPushStatus push(
    std::uint8_t camera_id,
    double exposure_timestamp_sec,
    const cv::Mat & image) noexcept;
  AdaptiveCameraSnapshotRequestResult request(
    std::uint64_t lidar_generation,
    LioReference reference) noexcept;
  AdaptiveCameraSnapshotAdvanceResult advanceUpstreamEventTime(
    double timestamp_sec) noexcept;
  std::optional<AdaptiveCameraSnapshotReceipt>
  takeFinalized() noexcept;

  AdaptiveCameraSnapshotQueueIdentity identity() const noexcept;
  AdaptiveCameraSnapshotQueueDiagnostics diagnostics() const noexcept;

private:
  struct Implementation;

  explicit AdaptiveCameraSnapshotQueue(
    const AdaptiveCameraSnapshotQueueConfiguration & configuration);

  std::unique_ptr<Implementation> implementation_;
};

struct AdaptiveCameraSnapshotQueueCreateResult
{
  AdaptiveCameraSnapshotQueueCreateStatus status{
    AdaptiveCameraSnapshotQueueCreateStatus::INVALID_CONFIGURATION};
  std::unique_ptr<AdaptiveCameraSnapshotQueue> queue;

  bool ready() const noexcept
  {
    return
      status == AdaptiveCameraSnapshotQueueCreateStatus::READY &&
      static_cast<bool>(queue);
  }
};

}  // namespace livo

#endif  // LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_H_
