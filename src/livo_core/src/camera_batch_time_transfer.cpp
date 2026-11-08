#include "camera_batch_time_transfer.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace livo
{
namespace
{

constexpr double kTimestampConsistencyToleranceSec = 1.0e-9;

struct CameraBatchTimeTransferAuthority final
{
};

bool sameAuthority(
  const std::shared_ptr<const void> & left,
  const std::shared_ptr<const void> & right)
{
  return
    left != nullptr &&
    right != nullptr &&
    !left.owner_before(right) &&
    !right.owner_before(left);
}

bool batchMetadataValid(
  const CameraObservationBatch & batch,
  const CameraTimeTransferState & anchor,
  const double exposure_time_sec)
{
  if (!std::isfinite(exposure_time_sec) ||
      !std::isfinite(batch.timestamp) ||
      !std::isfinite(anchor.timestamp_sec) ||
      std::abs(
        anchor.timestamp_sec -
        (batch.timestamp + exposure_time_sec)) >
        kTimestampConsistencyToleranceSec)
  {
    return false;
  }

  const std::size_t present_camera_count =
    static_cast<std::size_t>(std::count(
      batch.present.begin(), batch.present.end(), true));
  if (present_camera_count == 0U ||
      batch.camera_count != present_camera_count ||
      batch.partial !=
        (present_camera_count < kVioCameraCount))
  {
    return false;
  }

  for (std::size_t camera_id = 0U;
       camera_id < kVioCameraCount; ++camera_id)
  {
    if (!batch.present[camera_id]) {
      continue;
    }
    const CameraObservation & observation =
      batch.observations[camera_id];
    if (observation.camera_id != camera_id ||
        !std::isfinite(observation.timestamp) ||
        observation.timestamp <= 0.0 ||
        observation.image.empty())
    {
      return false;
    }
  }
  return true;
}

template<typename MatrixType>
bool positiveSemidefinite(const MatrixType & covariance)
{
  if (
    !covariance.allFinite() ||
    !covariance.isApprox(covariance.transpose(), 1.0e-12))
  {
    return false;
  }
  const Eigen::MatrixXd dynamic_covariance = covariance;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
    dynamic_covariance);
  return
    solver.info() == Eigen::Success &&
    solver.eigenvalues().minCoeff() >= -1.0e-12;
}

bool sharedProcessPoseCrossCovariance(
  const CameraPoseTransferRaw & left,
  const CameraPoseTransferRaw & right,
  CameraTimeTransferPoseMatrix * const left_wrt_right)
{
  if (
    left_wrt_right == nullptr ||
    !left.internallyValid() ||
    !right.internallyValid() ||
    left.anchor_timestamp_sec != right.anchor_timestamp_sec ||
    left.imu_epoch != right.imu_epoch)
  {
    return false;
  }

  left_wrt_right->setZero();
  constexpr double kZeroTransferToleranceSec = 1.0e-12;
  const double left_delta = left.delta_time_sec;
  const double right_delta = right.delta_time_sec;
  if (
    std::abs(left_delta) <= kZeroTransferToleranceSec ||
    std::abs(right_delta) <= kZeroTransferToleranceSec ||
    (left_delta < 0.0 && right_delta > 0.0) ||
    (left_delta > 0.0 && right_delta < 0.0))
  {
    // White IMU process-noise intervals on opposite sides of the common
    // anchor are disjoint. A zero-duration transfer contains no process
    // noise to share.
    return true;
  }

  const bool left_is_near =
    std::abs(left_delta) <= std::abs(right_delta);
  const CameraPoseTransferRaw & near =
    left_is_near ? left : right;
  const CameraPoseTransferRaw & far =
    left_is_near ? right : left;

  // Solve Phi_near_to_far * Phi_anchor_to_near =
  // Phi_anchor_to_far without explicitly forming an inverse.
  const Eigen::FullPivLU<CameraTimeTransferErrorStateMatrix>
    transition_solver(
    near.error_state_transition.transpose());
  if (!transition_solver.isInvertible())
  {
    return false;
  }
  const CameraTimeTransferErrorStateMatrix
    near_to_far_transition =
    transition_solver.solve(
    far.error_state_transition.transpose()).transpose();
  if (
    !near_to_far_transition.allFinite() ||
    !(near_to_far_transition * near.error_state_transition).
      isApprox(far.error_state_transition, 1.0e-10))
  {
    return false;
  }

  const CameraTimeTransferErrorStateMatrix
    remaining_process_noise =
    far.process_noise_covariance -
    near_to_far_transition *
    near.process_noise_covariance *
    near_to_far_transition.transpose();
  if (!positiveSemidefinite(remaining_process_noise))
  {
    // The supplied marginals do not admit the requested nested Markov
    // factorization. Fail closed instead of projecting or guessing a
    // cross-covariance.
    return false;
  }

  const CameraTimeTransferPoseMatrix near_wrt_far =
    near.process_noise_covariance.topRows<
      kCameraTimeTransferPoseDimension>() *
    near_to_far_transition.topRows<
      kCameraTimeTransferPoseDimension>().transpose();
  *left_wrt_right =
    left_is_near ? near_wrt_far : near_wrt_far.transpose();
  return left_wrt_right->allFinite();
}

}  // namespace

CameraBatchJointTargetPoseTransferNoiseReceipt::
CameraBatchJointTargetPoseTransferNoiseReceipt(
  const double batch_timestamp_sec,
  const double anchor_timestamp_sec,
  const std::uint64_t imu_epoch,
  const std::uint32_t active_state_jacobian_revision,
  std::string active_state_layout_id,
  std::array<bool, kVioCameraCount> present,
  std::array<double, kVioCameraCount> target_timestamps_sec,
  std::shared_ptr<const void> authority)
: batch_timestamp_sec_(batch_timestamp_sec),
  anchor_timestamp_sec_(anchor_timestamp_sec),
  imu_epoch_(imu_epoch),
  active_state_jacobian_revision_(
    active_state_jacobian_revision),
  active_state_layout_id_(std::move(active_state_layout_id)),
  present_(std::move(present)),
  target_timestamps_sec_(std::move(target_timestamps_sec)),
  authority_(std::move(authority))
{
}

bool CameraBatchJointTargetPoseTransferNoiseReceipt::valid() const
{
  if (
    authority_ == nullptr ||
    !std::isfinite(batch_timestamp_sec_) ||
    !std::isfinite(anchor_timestamp_sec_) ||
    imu_epoch_ == 0U ||
    active_state_jacobian_revision_ !=
      kCameraTimeTransferActiveJacobianRevision ||
    active_state_layout_id_ !=
      kCameraTimeTransferActiveStateLayoutId)
  {
    return false;
  }
  bool any_present = false;
  for (std::size_t camera_id = 0U;
       camera_id < kVioCameraCount; ++camera_id)
  {
    if (present_[camera_id])
    {
      any_present = true;
      if (!std::isfinite(target_timestamps_sec_[camera_id]))
      {
        return false;
      }
    }
    else if (!std::isnan(target_timestamps_sec_[camera_id]))
    {
      return false;
    }
  }
  return any_present;
}

CameraBatchTimeAlignmentNoiseModel
CameraBatchJointTargetPoseTransferNoiseReceipt::
timeAlignmentNoiseModel() const
{
  return CameraBatchTimeAlignmentNoiseModel::
         kIndependentPerCameraDiagonalExperimental;
}

const CameraBatchJointTargetPoseTransferNoiseCovariance *
CameraBatchTimeTransferBundle::
validatedJointTargetPoseTransferNoiseCovariance(
  const CameraBatchJointTargetPoseTransferNoiseReceipt &
    receipt) const
{
  if (
    !allPresentFullStateTransfersValid() ||
    !joint_target_pose_transfer_noise_covariance_.has_value() ||
    !validated_joint_target_pose_transfer_noise_receipt_.
      has_value())
  {
    return nullptr;
  }

  const CameraBatchJointTargetPoseTransferNoiseReceipt &
    sealed_receipt =
    *validated_joint_target_pose_transfer_noise_receipt_;
  if (
    !receipt.valid() ||
    !sealed_receipt.valid() ||
    !sameAuthority(
      receipt.authority_, sealed_receipt.authority_) ||
    receipt.batch_timestamp_sec_ !=
      sealed_receipt.batch_timestamp_sec_ ||
    receipt.anchor_timestamp_sec_ != anchor_timestamp_sec ||
    receipt.anchor_timestamp_sec_ !=
      sealed_receipt.anchor_timestamp_sec_ ||
    receipt.imu_epoch_ != imu_epoch ||
    receipt.imu_epoch_ != sealed_receipt.imu_epoch_ ||
    receipt.active_state_jacobian_revision_ !=
      active_state_jacobian_revision ||
    receipt.active_state_jacobian_revision_ !=
      sealed_receipt.active_state_jacobian_revision_ ||
    receipt.active_state_layout_id_ != active_state_layout_id ||
    receipt.active_state_layout_id_ !=
      sealed_receipt.active_state_layout_id_ ||
    receipt.present_ != present ||
    receipt.present_ != sealed_receipt.present_ ||
    !validated_receipt_.has_value() ||
    receipt.batch_timestamp_sec_ !=
      validated_receipt_->batchTimestampSec())
  {
    return nullptr;
  }

  const CameraBatchJointTargetPoseTransferNoiseCovariance &
    covariance =
    *joint_target_pose_transfer_noise_covariance_;
  for (std::size_t camera_id = 0U;
       camera_id < kVioCameraCount; ++camera_id)
  {
    const Eigen::Index offset =
      static_cast<Eigen::Index>(camera_id) *
      kCameraTimeTransferPoseDimension;
    if (!present[camera_id])
    {
      if (
        transfers[camera_id].has_value() ||
        !std::isnan(
          receipt.target_timestamps_sec_[camera_id]) ||
        !std::isnan(
          sealed_receipt.
          target_timestamps_sec_[camera_id]) ||
        !covariance.middleRows<
          kCameraTimeTransferPoseDimension>(
          offset).isZero(0.0) ||
        !covariance.middleCols<
          kCameraTimeTransferPoseDimension>(
          offset).isZero(0.0))
      {
        return nullptr;
      }
      continue;
    }
    if (
      receipt.target_timestamps_sec_[camera_id] !=
        sealed_receipt.target_timestamps_sec_[camera_id] ||
      receipt.target_timestamps_sec_[camera_id] !=
        transfers[camera_id].targetTimestampSec())
    {
      return nullptr;
    }
    const CameraTimeTransferPoseMatrix expected_marginal =
      transfers[camera_id].data().pose_transfer_covariance +
      transfers[camera_id].data().
      time_alignment_pose_covariance;
    if (
      !covariance.block<
        kCameraTimeTransferPoseDimension,
        kCameraTimeTransferPoseDimension>(
        offset, offset).isApprox(expected_marginal, 0.0))
    {
      return nullptr;
    }
  }
  return positiveSemidefinite(covariance) ?
         &covariance : nullptr;
}

CameraBatchTimeTransferEntry::CameraBatchTimeTransferEntry(
  const std::uint8_t camera_id,
  const double batch_timestamp_sec,
  const double anchor_timestamp_sec,
  const std::uint64_t imu_epoch,
  const std::uint32_t active_state_jacobian_revision,
  std::string active_state_layout_id,
  const CameraPoseTransferData & transfer,
  std::shared_ptr<const void> authority)
: camera_id_(camera_id),
  batch_timestamp_sec_(batch_timestamp_sec),
  anchor_timestamp_sec_(anchor_timestamp_sec),
  imu_epoch_(imu_epoch),
  active_state_jacobian_revision_(
    active_state_jacobian_revision),
  active_state_layout_id_(std::move(active_state_layout_id)),
  diagnostic_data_(transfer),
  validated_value_(transfer.validatedValue()),
  authority_(std::move(authority))
{
}

bool CameraBatchTimeTransferEntry::has_value() const
{
  if (
    !validated_value_.has_value() ||
    authority_ == nullptr ||
    camera_id_ >= kBoundCameraTimeTransferCount ||
    !std::isfinite(batch_timestamp_sec_) ||
    !std::isfinite(anchor_timestamp_sec_) ||
    imu_epoch_ == 0U ||
    active_state_jacobian_revision_ !=
      kCameraTimeTransferActiveJacobianRevision ||
    active_state_layout_id_ !=
      kCameraTimeTransferActiveStateLayoutId)
  {
    return false;
  }
  const CameraPoseTransferRaw & raw =
    validated_value_->raw();
  return
    raw.internallyValid() &&
    raw.anchor_timestamp_sec == anchor_timestamp_sec_ &&
    raw.imu_epoch == imu_epoch_;
}

const CameraPoseTransfer &
CameraBatchTimeTransferEntry::value() const
{
  return validated_value_.value();
}

const CameraPoseTransferRaw &
CameraBatchTimeTransferEntry::data() const
{
  return has_value() ?
    validated_value_->raw() : diagnostic_data_;
}

std::uint8_t CameraBatchTimeTransferEntry::cameraId() const
{
  return camera_id_;
}

double CameraBatchTimeTransferEntry::batchTimestampSec() const
{
  return batch_timestamp_sec_;
}

double CameraBatchTimeTransferEntry::anchorTimestampSec() const
{
  return anchor_timestamp_sec_;
}

double CameraBatchTimeTransferEntry::targetTimestampSec() const
{
  return data().target_timestamp_sec;
}

std::uint64_t CameraBatchTimeTransferEntry::imuEpoch() const
{
  return imu_epoch_;
}

std::uint32_t
CameraBatchTimeTransferEntry::activeStateJacobianRevision() const
{
  return active_state_jacobian_revision_;
}

const std::string &
CameraBatchTimeTransferEntry::activeStateLayoutId() const
{
  return active_state_layout_id_;
}

CameraBatchTimeTransferReceipt::CameraBatchTimeTransferReceipt(
  const double batch_timestamp_sec,
  const double anchor_timestamp_sec,
  const std::uint64_t imu_epoch,
  const std::uint32_t active_state_jacobian_revision,
  std::string active_state_layout_id,
  std::array<bool, kBoundCameraTimeTransferCount> present,
  std::array<double, kBoundCameraTimeTransferCount>
    target_timestamps_sec,
  std::shared_ptr<const void> authority)
: batch_timestamp_sec_(batch_timestamp_sec),
  anchor_timestamp_sec_(anchor_timestamp_sec),
  imu_epoch_(imu_epoch),
  active_state_jacobian_revision_(
    active_state_jacobian_revision),
  active_state_layout_id_(std::move(active_state_layout_id)),
  present_(std::move(present)),
  target_timestamps_sec_(std::move(target_timestamps_sec)),
  authority_(std::move(authority))
{
}

bool CameraBatchTimeTransferReceipt::valid() const
{
  if (
    authority_ == nullptr ||
    !std::isfinite(batch_timestamp_sec_) ||
    !std::isfinite(anchor_timestamp_sec_) ||
    imu_epoch_ == 0U ||
    active_state_jacobian_revision_ !=
      kCameraTimeTransferActiveJacobianRevision ||
    active_state_layout_id_ !=
      kCameraTimeTransferActiveStateLayoutId)
  {
    return false;
  }
  bool any_present = false;
  for (std::size_t camera_id = 0U;
       camera_id < present_.size(); ++camera_id)
  {
    if (!present_[camera_id])
    {
      continue;
    }
    any_present = true;
    if (!std::isfinite(target_timestamps_sec_[camera_id]))
    {
      return false;
    }
  }
  return any_present;
}

bool CameraBatchTimeTransferReceipt::accepts(
  const CameraBatchTimeTransferEntry & entry) const
{
  if (
    !valid() ||
    !entry.has_value() ||
    !sameAuthority(authority_, entry.authority_) ||
    entry.camera_id_ >= present_.size() ||
    !present_[entry.camera_id_])
  {
    return false;
  }
  return
    entry.batch_timestamp_sec_ == batch_timestamp_sec_ &&
    entry.anchor_timestamp_sec_ == anchor_timestamp_sec_ &&
    entry.imu_epoch_ == imu_epoch_ &&
    entry.active_state_jacobian_revision_ ==
      active_state_jacobian_revision_ &&
    entry.active_state_layout_id_ ==
      active_state_layout_id_ &&
    entry.targetTimestampSec() ==
      target_timestamps_sec_[entry.camera_id_];
}

bool CameraBatchTimeTransferReceipt::matchesBundleDiagnostics(
  const double anchor_timestamp_sec,
  const std::uint64_t imu_epoch,
  const std::uint32_t active_state_jacobian_revision,
  const std::string & active_state_layout_id,
  const std::array<bool, kBoundCameraTimeTransferCount> &
    present,
  const std::size_t present_camera_count) const
{
  return
    valid() &&
    anchor_timestamp_sec == anchor_timestamp_sec_ &&
    imu_epoch == imu_epoch_ &&
    active_state_jacobian_revision ==
      active_state_jacobian_revision_ &&
    active_state_layout_id == active_state_layout_id_ &&
    present == present_ &&
    present_camera_count ==
      static_cast<std::size_t>(std::count(
        present_.begin(), present_.end(), true));
}

double CameraBatchTimeTransferReceipt::batchTimestampSec() const
{
  return batch_timestamp_sec_;
}

double CameraBatchTimeTransferReceipt::anchorTimestampSec() const
{
  return anchor_timestamp_sec_;
}

std::uint64_t CameraBatchTimeTransferReceipt::imuEpoch() const
{
  return imu_epoch_;
}

std::uint32_t
CameraBatchTimeTransferReceipt::activeStateJacobianRevision() const
{
  return active_state_jacobian_revision_;
}

const std::string &
CameraBatchTimeTransferReceipt::activeStateLayoutId() const
{
  return active_state_layout_id_;
}

CameraBatchTimeTransferBundle
CameraBatchTimeTransferBuilder::build(
  const CameraObservationBatch & batch,
  const CameraTimeTransferState & anchor,
  const double exposure_time_sec,
  const CameraTimeTransfer & time_transfer)
{
  CameraBatchTimeTransferBundle bundle;
  bundle.anchor_timestamp_sec = anchor.timestamp_sec;
  bundle.imu_epoch = time_transfer.epoch();
  bundle.present = batch.present;
  bundle.present_camera_count =
    static_cast<std::size_t>(std::count(
      batch.present.begin(), batch.present.end(), true));
  bundle.batch_metadata_valid =
    batchMetadataValid(batch, anchor, exposure_time_sec);
  if (!bundle.batch_metadata_valid) {
    return bundle;
  }

  const std::shared_ptr<const void> authority =
    std::make_shared<const CameraBatchTimeTransferAuthority>();
  std::array<double, kBoundCameraTimeTransferCount>
    target_timestamps_sec{{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()}};
  for (std::size_t camera_id = 0U;
       camera_id < kVioCameraCount; ++camera_id)
  {
    if (batch.present[camera_id])
    {
      target_timestamps_sec[camera_id] =
        batch.observations[camera_id].timestamp +
        exposure_time_sec;
    }
  }
  bundle.validated_receipt_ =
    CameraBatchTimeTransferReceipt(
    batch.timestamp,
    anchor.timestamp_sec,
    time_transfer.epoch(),
    bundle.active_state_jacobian_revision,
    bundle.active_state_layout_id,
    batch.present,
    target_timestamps_sec,
    authority);

  for (std::size_t camera_id = 0U;
       camera_id < kVioCameraCount; ++camera_id)
  {
    if (!batch.present[camera_id]) {
      continue;
    }
    bundle.transfers[camera_id] =
      CameraBatchTimeTransferEntry(
      static_cast<std::uint8_t>(camera_id),
      batch.timestamp,
      anchor.timestamp_sec,
      time_transfer.epoch(),
      bundle.active_state_jacobian_revision,
      bundle.active_state_layout_id,
      time_transfer.transfer(
        anchor, target_timestamps_sec[camera_id]),
      authority);
    if (bundle.transfers[camera_id].has_value()) {
      ++bundle.valid_camera_count;
    }
    if (bundle.transfers[camera_id].data().
        conditional_pose_only.valid())
    {
      ++bundle.conditional_pose_only_shadow_count;
    }
  }

  if (bundle.allPresentFullStateTransfersValid())
  {
    CameraBatchJointTargetPoseTransferNoiseCovariance
      covariance =
      CameraBatchJointTargetPoseTransferNoiseCovariance::Zero();
    for (std::size_t camera_id = 0U;
         camera_id < kVioCameraCount; ++camera_id)
    {
      if (!bundle.present[camera_id]) {
        continue;
      }
      const Eigen::Index offset =
        static_cast<Eigen::Index>(camera_id) *
        kCameraTimeTransferPoseDimension;
      covariance.block<
        kCameraTimeTransferPoseDimension,
        kCameraTimeTransferPoseDimension>(offset, offset) =
        bundle.transfers[camera_id].data().
        pose_transfer_covariance +
        bundle.transfers[camera_id].data().
        time_alignment_pose_covariance;
    }

    bool cross_covariances_valid = true;
    for (std::size_t left_camera_id = 0U;
         left_camera_id < kVioCameraCount &&
         cross_covariances_valid;
         ++left_camera_id)
    {
      for (std::size_t right_camera_id =
             left_camera_id + 1U;
           right_camera_id < kVioCameraCount;
           ++right_camera_id)
      {
        if (
          !bundle.present[left_camera_id] ||
          !bundle.present[right_camera_id])
        {
          continue;
        }
        CameraTimeTransferPoseMatrix cross_covariance;
        if (!sharedProcessPoseCrossCovariance(
            bundle.transfers[left_camera_id].data(),
            bundle.transfers[right_camera_id].data(),
            &cross_covariance))
        {
          cross_covariances_valid = false;
          break;
        }
        const Eigen::Index left_offset =
          static_cast<Eigen::Index>(left_camera_id) *
          kCameraTimeTransferPoseDimension;
        const Eigen::Index right_offset =
          static_cast<Eigen::Index>(right_camera_id) *
          kCameraTimeTransferPoseDimension;
        covariance.block<
          kCameraTimeTransferPoseDimension,
          kCameraTimeTransferPoseDimension>(
          left_offset, right_offset) = cross_covariance;
        covariance.block<
          kCameraTimeTransferPoseDimension,
          kCameraTimeTransferPoseDimension>(
          right_offset, left_offset) =
          cross_covariance.transpose();
      }
    }

    if (
      cross_covariances_valid &&
      positiveSemidefinite(covariance))
    {
      bundle.joint_target_pose_transfer_noise_covariance_ =
        covariance;
      bundle.validated_joint_target_pose_transfer_noise_receipt_ =
        CameraBatchJointTargetPoseTransferNoiseReceipt(
        batch.timestamp,
        anchor.timestamp_sec,
        time_transfer.epoch(),
        bundle.active_state_jacobian_revision,
        bundle.active_state_layout_id,
        batch.present,
        target_timestamps_sec,
        authority);
    }
  }
  return bundle;
}

CameraBatchTimeTransferBundle buildCameraBatchTimeTransferBundle(
  const CameraObservationBatch & batch,
  const CameraTimeTransferState & anchor,
  const double exposure_time_sec,
  const CameraTimeTransfer & time_transfer)
{
  return CameraBatchTimeTransferBuilder::build(
    batch, anchor, exposure_time_sec, time_transfer);
}

}  // namespace livo
