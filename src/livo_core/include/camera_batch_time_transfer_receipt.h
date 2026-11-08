#ifndef LIVO_CAMERA_BATCH_TIME_TRANSFER_RECEIPT_H_
#define LIVO_CAMERA_BATCH_TIME_TRANSFER_RECEIPT_H_

#include "camera_time_transfer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace livo
{

inline constexpr std::size_t kBoundCameraTimeTransferCount = 4U;
inline constexpr std::uint8_t kInvalidBoundCameraId =
  std::numeric_limits<std::uint8_t>::max();

class CameraBatchTimeTransferBuilder;
class CameraBatchTimeTransferReceipt;

// A default entry is a public diagnostic value only. A valid entry can be
// produced only by CameraBatchTimeTransferBuilder, which binds the sealed pose
// transfer to one camera and one batch.
class CameraBatchTimeTransferEntry
{
public:
  CameraBatchTimeTransferEntry() = default;
  CameraBatchTimeTransferEntry(
    const CameraBatchTimeTransferEntry &) = default;
  CameraBatchTimeTransferEntry(
    CameraBatchTimeTransferEntry &&) = default;
  CameraBatchTimeTransferEntry & operator=(
    const CameraBatchTimeTransferEntry &) = default;
  CameraBatchTimeTransferEntry & operator=(
    CameraBatchTimeTransferEntry &&) = default;

  bool has_value() const;
  const CameraPoseTransfer & value() const;
  const CameraPoseTransferRaw & data() const;

  std::uint8_t cameraId() const;
  double batchTimestampSec() const;
  double anchorTimestampSec() const;
  double targetTimestampSec() const;
  std::uint64_t imuEpoch() const;
  std::uint32_t activeStateJacobianRevision() const;
  const std::string & activeStateLayoutId() const;

private:
  CameraBatchTimeTransferEntry(
    std::uint8_t camera_id,
    double batch_timestamp_sec,
    double anchor_timestamp_sec,
    std::uint64_t imu_epoch,
    std::uint32_t active_state_jacobian_revision,
    std::string active_state_layout_id,
    const CameraPoseTransferData & transfer,
    std::shared_ptr<const void> authority);

  std::uint8_t camera_id_{kInvalidBoundCameraId};
  double batch_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  double anchor_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t imu_epoch_{0U};
  std::uint32_t active_state_jacobian_revision_{0U};
  std::string active_state_layout_id_;
  CameraPoseTransferRaw diagnostic_data_;
  std::optional<CameraPoseTransfer> validated_value_;
  std::shared_ptr<const void> authority_;

  friend class CameraBatchTimeTransferBuilder;
  friend class CameraBatchTimeTransferReceipt;
};

// Batch capability used at downstream authorization seams. Its immutable
// bindings let a consumer detect camera swaps and entries copied from another
// batch without trusting mutable bundle diagnostics.
class CameraBatchTimeTransferReceipt
{
public:
  CameraBatchTimeTransferReceipt(
    const CameraBatchTimeTransferReceipt &) = default;
  CameraBatchTimeTransferReceipt(
    CameraBatchTimeTransferReceipt &&) = default;
  CameraBatchTimeTransferReceipt & operator=(
    const CameraBatchTimeTransferReceipt &) = default;
  CameraBatchTimeTransferReceipt & operator=(
    CameraBatchTimeTransferReceipt &&) = default;

  bool valid() const;
  bool accepts(
    const CameraBatchTimeTransferEntry & entry) const;
  bool matchesBundleDiagnostics(
    double anchor_timestamp_sec,
    std::uint64_t imu_epoch,
    std::uint32_t active_state_jacobian_revision,
    const std::string & active_state_layout_id,
    const std::array<bool, kBoundCameraTimeTransferCount> &
      present,
    std::size_t present_camera_count) const;

  double batchTimestampSec() const;
  double anchorTimestampSec() const;
  std::uint64_t imuEpoch() const;
  std::uint32_t activeStateJacobianRevision() const;
  const std::string & activeStateLayoutId() const;

private:
  CameraBatchTimeTransferReceipt(
    double batch_timestamp_sec,
    double anchor_timestamp_sec,
    std::uint64_t imu_epoch,
    std::uint32_t active_state_jacobian_revision,
    std::string active_state_layout_id,
    std::array<bool, kBoundCameraTimeTransferCount> present,
    std::array<double, kBoundCameraTimeTransferCount>
      target_timestamps_sec,
    std::shared_ptr<const void> authority);

  double batch_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  double anchor_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t imu_epoch_{0U};
  std::uint32_t active_state_jacobian_revision_{0U};
  std::string active_state_layout_id_;
  std::array<bool, kBoundCameraTimeTransferCount> present_{
    {false, false, false, false}};
  std::array<double, kBoundCameraTimeTransferCount>
    target_timestamps_sec_{{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()}};
  std::shared_ptr<const void> authority_;

  friend class CameraBatchTimeTransferBuilder;
};

}  // namespace livo

#endif  // LIVO_CAMERA_BATCH_TIME_TRANSFER_RECEIPT_H_
