#ifndef LIVO_CAMERA_BATCH_TIME_TRANSFER_H_
#define LIVO_CAMERA_BATCH_TIME_TRANSFER_H_

#include "camera_batch_time_transfer_receipt.h"
#include "camera_observation.h"
#include "camera_time_transfer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace livo
{

static_assert(kVioCameraCount == kBoundCameraTimeTransferCount);

inline constexpr Eigen::Index
  kCameraBatchJointTargetPoseTransferNoiseDimension =
  static_cast<Eigen::Index>(kVioCameraCount) *
  kCameraTimeTransferPoseDimension;

using CameraBatchJointTargetPoseTransferNoiseCovariance =
  Eigen::Matrix<
  double,
  kCameraBatchJointTargetPoseTransferNoiseDimension,
  kCameraBatchJointTargetPoseTransferNoiseDimension>;

// Camera blocks remain in canonical rig order. A partial batch keeps the
// absent camera rows and columns exactly zero.
// Experimental S1a assumption: timestamp-alignment errors are independent
// between cameras, so their covariance contributes to diagonal blocks only.
enum class CameraBatchTimeAlignmentNoiseModel
{
  kIndependentPerCameraDiagonalExperimental
};

class CameraBatchJointTargetPoseTransferNoiseReceipt
{
public:
  CameraBatchJointTargetPoseTransferNoiseReceipt(
    const CameraBatchJointTargetPoseTransferNoiseReceipt &) =
    default;
  CameraBatchJointTargetPoseTransferNoiseReceipt(
    CameraBatchJointTargetPoseTransferNoiseReceipt &&) = default;
  CameraBatchJointTargetPoseTransferNoiseReceipt & operator=(
    const CameraBatchJointTargetPoseTransferNoiseReceipt &) =
    default;
  CameraBatchJointTargetPoseTransferNoiseReceipt & operator=(
    CameraBatchJointTargetPoseTransferNoiseReceipt &&) = default;

  bool valid() const;
  CameraBatchTimeAlignmentNoiseModel
  timeAlignmentNoiseModel() const;

private:
  CameraBatchJointTargetPoseTransferNoiseReceipt(
    double batch_timestamp_sec,
    double anchor_timestamp_sec,
    std::uint64_t imu_epoch,
    std::uint32_t active_state_jacobian_revision,
    std::string active_state_layout_id,
    std::array<bool, kVioCameraCount> present,
    std::array<double, kVioCameraCount> target_timestamps_sec,
    std::shared_ptr<const void> authority);

  double batch_timestamp_sec_{0.0};
  double anchor_timestamp_sec_{0.0};
  std::uint64_t imu_epoch_{0U};
  std::uint32_t active_state_jacobian_revision_{0U};
  std::string active_state_layout_id_;
  std::array<bool, kVioCameraCount> present_{
    {false, false, false, false}};
  std::array<double, kVioCameraCount> target_timestamps_sec_{{
      0.0, 0.0, 0.0, 0.0}};
  std::shared_ptr<const void> authority_;

  friend class CameraBatchTimeTransferBuilder;
  friend struct CameraBatchTimeTransferBundle;
};

struct CameraBatchTimeTransferBundle
{
  double anchor_timestamp_sec{0.0};
  std::uint64_t imu_epoch{0U};
  bool batch_metadata_valid{false};
  std::array<bool, kVioCameraCount> present{
    {false, false, false, false}};
  std::size_t present_camera_count{0U};
  std::size_t valid_camera_count{0U};
  std::size_t conditional_pose_only_shadow_count{0U};
  std::uint32_t active_state_jacobian_revision{
    kCameraTimeTransferActiveJacobianRevision};
  std::string active_state_layout_id{
    kCameraTimeTransferActiveStateLayoutId};
  std::array<CameraBatchTimeTransferEntry, kVioCameraCount>
    transfers;

  bool activeStateLayoutCompatible() const
  {
    return
      active_state_jacobian_revision ==
        kCameraTimeTransferActiveJacobianRevision &&
      active_state_layout_id ==
        kCameraTimeTransferActiveStateLayoutId;
  }

  bool allPresentFullStateTransfersValid() const
  {
    if (
      !batch_metadata_valid ||
      !activeStateLayoutCompatible() ||
      !validated_receipt_.has_value() ||
      !validated_receipt_->valid() ||
      !validated_receipt_->matchesBundleDiagnostics(
        anchor_timestamp_sec,
        imu_epoch,
        active_state_jacobian_revision,
        active_state_layout_id,
        present,
        present_camera_count) ||
      present_camera_count == 0U ||
      valid_camera_count != present_camera_count)
    {
      return false;
    }

    std::size_t checked_camera_count = 0U;
    for (std::size_t camera_id = 0U;
         camera_id < kVioCameraCount; ++camera_id)
    {
      if (!present[camera_id]) {
        continue;
      }
      if (
        !transfers[camera_id].has_value() ||
        transfers[camera_id].cameraId() != camera_id ||
        !validated_receipt_->accepts(
          transfers[camera_id]))
      {
        return false;
      }
      ++checked_camera_count;
    }
    return checked_camera_count == present_camera_count;
  }

  // Compatibility name for the current shadow caller. Its semantics are now
  // deliberately identical to the full-state gate above.
  bool allPresentTransfersValid() const
  {
    return allPresentFullStateTransfersValid();
  }

  const std::optional<CameraBatchTimeTransferReceipt> &
  validatedReceipt() const
  {
    return validated_receipt_;
  }

  const std::optional<
    CameraBatchJointTargetPoseTransferNoiseReceipt> &
  validatedJointTargetPoseTransferNoiseReceipt() const
  {
    return validated_joint_target_pose_transfer_noise_receipt_;
  }

  const CameraBatchJointTargetPoseTransferNoiseCovariance *
  validatedJointTargetPoseTransferNoiseCovariance(
    const CameraBatchJointTargetPoseTransferNoiseReceipt &
      receipt) const;

private:
  std::optional<CameraBatchTimeTransferReceipt>
    validated_receipt_;
  std::optional<
    CameraBatchJointTargetPoseTransferNoiseCovariance>
    joint_target_pose_transfer_noise_covariance_;
  std::optional<
    CameraBatchJointTargetPoseTransferNoiseReceipt>
    validated_joint_target_pose_transfer_noise_receipt_;

  friend class CameraBatchTimeTransferBuilder;
};

class CameraBatchTimeTransferBuilder
{
public:
  static CameraBatchTimeTransferBundle build(
    const CameraObservationBatch & batch,
    const CameraTimeTransferState & anchor,
    double exposure_time_sec,
    const CameraTimeTransfer & time_transfer);
};

CameraBatchTimeTransferBundle buildCameraBatchTimeTransferBundle(
  const CameraObservationBatch & batch,
  const CameraTimeTransferState & anchor,
  double exposure_time_sec,
  const CameraTimeTransfer & time_transfer);

}  // namespace livo

#endif  // LIVO_CAMERA_BATCH_TIME_TRANSFER_H_
