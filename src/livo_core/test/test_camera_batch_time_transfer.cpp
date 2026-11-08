#include "camera_batch_time_transfer.h"

#include <type_traits>
#include <utility>

#include <Eigen/Eigenvalues>
#include <gtest/gtest.h>

namespace
{

static_assert(std::is_same_v<
  decltype(std::declval<
    const livo::CameraBatchTimeTransferEntry &>().data()),
  const livo::CameraPoseTransferRaw &>);
static_assert(!std::is_default_constructible_v<
  livo::CameraBatchJointTargetPoseTransferNoiseReceipt>);
static_assert(!std::is_aggregate_v<
  livo::CameraBatchJointTargetPoseTransferNoiseReceipt>);
static_assert(!std::is_constructible_v<
  livo::CameraBatchJointTargetPoseTransferNoiseReceipt,
  double,
  double,
  std::uint64_t,
  std::uint32_t,
  std::string,
  std::array<bool, kVioCameraCount>,
  std::array<double, kVioCameraCount>,
  std::shared_ptr<const void>>);
static_assert(std::is_same_v<
  decltype(std::declval<
    const livo::CameraBatchTimeTransferBundle &>().
    validatedJointTargetPoseTransferNoiseCovariance(
      std::declval<const
        livo::CameraBatchJointTargetPoseTransferNoiseReceipt &>())),
  const livo::CameraBatchJointTargetPoseTransferNoiseCovariance *>);
using JointCovarianceGetter =
  const livo::CameraBatchJointTargetPoseTransferNoiseCovariance *
  (livo::CameraBatchTimeTransferBundle::*)(
  const livo::CameraBatchJointTargetPoseTransferNoiseReceipt &)
  const;
static_assert(!std::is_invocable_v<
  JointCovarianceGetter,
  const livo::CameraBatchTimeTransferBundle &,
  const livo::CameraBatchTimeTransferReceipt &>);

livo::CameraTimeTransferImuSample imuSample(const double timestamp_sec)
{
  livo::CameraTimeTransferImuSample sample;
  sample.timestamp_sec = timestamp_sec;
  sample.angular_velocity_radps = Eigen::Vector3d(0.0, 0.0, 0.2);
  sample.specific_force_mps2 = Eigen::Vector3d::Zero();
  return sample;
}

CameraObservation imageObservation(
  const std::uint8_t camera_id, const double timestamp_sec)
{
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = timestamp_sec;
  observation.image = cv::Mat::ones(2, 2, CV_8UC1);
  return observation;
}

livo::CameraTimeTransfer configuredTransfer()
{
  livo::CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.05;
  config.max_imu_gap_sec = 0.01;
  livo::CameraTimeTransfer transfer(config);
  for (int index = 0; index <= 8; ++index)
  {
    EXPECT_EQ(
      transfer.appendImu(imuSample(9.98 + 0.005 * index)),
      livo::CameraTimeTransferAppendStatus::kAccepted);
  }
  return transfer;
}

CameraObservationBatch fourCameraBatch(
  const std::array<double, CameraObservationQueue::kCameraCount> &
    timestamps_sec)
{
  CameraObservationBatch batch;
  batch.present.fill(true);
  batch.camera_count = CameraObservationQueue::kCameraCount;
  batch.partial = false;
  batch.timestamp = 10.0;
  for (std::size_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount; ++camera_id)
  {
    batch.observations[camera_id] = imageObservation(
      static_cast<std::uint8_t>(camera_id),
      timestamps_sec[camera_id]);
  }
  return batch;
}

livo::CameraTimeTransferState transferredState(
  const livo::CameraPoseTransferRaw & transfer,
  const livo::CameraTimeTransferState & anchor)
{
  livo::CameraTimeTransferState state;
  state.timestamp_sec = transfer.target_timestamp_sec;
  state.rotation_world_from_imu =
    transfer.rotation_world_from_imu;
  state.position_world_m = transfer.position_world_m;
  state.velocity_world_mps = transfer.velocity_world_mps;
  state.gyro_bias_radps = anchor.gyro_bias_radps;
  state.accel_bias_mps2 = anchor.accel_bias_mps2;
  state.gravity_world_mps2 = anchor.gravity_world_mps2;
  return state;
}

TEST(
  CameraBatchTimeTransfer,
  SealsJointCovarianceWithExactPerCameraMarginals)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  const CameraObservationBatch batch = fourCameraBatch(
    {{10.005, 10.015, 9.995, 10.0}});
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;
  anchor.timestamp_stddev_sec = 0.002;

  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);

  ASSERT_TRUE(bundle.allPresentFullStateTransfersValid());
  ASSERT_TRUE(
    bundle.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  const auto & receipt =
    *bundle.validatedJointTargetPoseTransferNoiseReceipt();
  ASSERT_TRUE(receipt.valid());
  EXPECT_EQ(
    receipt.timeAlignmentNoiseModel(),
    livo::CameraBatchTimeAlignmentNoiseModel::
    kIndependentPerCameraDiagonalExperimental);

  const auto * const covariance =
    bundle.validatedJointTargetPoseTransferNoiseCovariance(
      receipt);
  ASSERT_NE(covariance, nullptr);
  for (Eigen::Index camera_id = 0;
       camera_id <
       static_cast<Eigen::Index>(
         CameraObservationQueue::kCameraCount);
       ++camera_id)
  {
    const livo::CameraTimeTransferPoseMatrix expected =
      bundle.transfers[static_cast<std::size_t>(camera_id)].
      data().pose_transfer_covariance +
      bundle.transfers[static_cast<std::size_t>(camera_id)].
      data().time_alignment_pose_covariance;
    const livo::CameraTimeTransferPoseMatrix actual =
      covariance->block<
        livo::kCameraTimeTransferPoseDimension,
        livo::kCameraTimeTransferPoseDimension>(
        camera_id * livo::kCameraTimeTransferPoseDimension,
        camera_id * livo::kCameraTimeTransferPoseDimension);
    EXPECT_TRUE(actual.isApprox(expected, 0.0));
  }
}

TEST(
  CameraBatchTimeTransfer,
  ThreeCameraPartialBatchLeavesAbsentPoseBlockExactlyZero)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  CameraObservationBatch batch;
  batch.present = {{true, false, true, true}};
  batch.camera_count = 3U;
  batch.partial = true;
  batch.timestamp = 10.0;
  batch.observations[0U] = imageObservation(0U, 10.005);
  batch.observations[2U] = imageObservation(2U, 10.015);
  batch.observations[3U] = imageObservation(3U, 9.995);
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;

  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);

  ASSERT_TRUE(bundle.allPresentFullStateTransfersValid());
  ASSERT_TRUE(
    bundle.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  const auto * const covariance =
    bundle.validatedJointTargetPoseTransferNoiseCovariance(
      *bundle.validatedJointTargetPoseTransferNoiseReceipt());
  ASSERT_NE(covariance, nullptr);

  const Eigen::Index absent_offset =
    livo::kCameraTimeTransferPoseDimension;
  EXPECT_TRUE(
    covariance->middleRows<
      livo::kCameraTimeTransferPoseDimension>(
      absent_offset).isZero(0.0));
  EXPECT_TRUE(
    covariance->middleCols<
      livo::kCameraTimeTransferPoseDimension>(
      absent_offset).isZero(0.0));

  const livo::CameraTimeTransferPoseMatrix same_side_cross =
    covariance->block<
      livo::kCameraTimeTransferPoseDimension,
      livo::kCameraTimeTransferPoseDimension>(
      0, 2 * livo::kCameraTimeTransferPoseDimension);
  EXPECT_GT(same_side_cross.norm(), 0.0);
  const livo::CameraTimeTransferPoseMatrix
    opposite_direction_cross =
    covariance->block<
      livo::kCameraTimeTransferPoseDimension,
      livo::kCameraTimeTransferPoseDimension>(
      0, 3 * livo::kCameraTimeTransferPoseDimension);
  EXPECT_TRUE(opposite_direction_cross.isZero(0.0));

  livo::CameraBatchTimeTransferBundle tampered_absent = bundle;
  tampered_absent.transfers[1U] = bundle.transfers[0U];
  EXPECT_EQ(
    tampered_absent.
    validatedJointTargetPoseTransferNoiseCovariance(
      *bundle.
      validatedJointTargetPoseTransferNoiseReceipt()),
    nullptr);
}

TEST(
  CameraBatchTimeTransfer,
  SameSideNestedTransfersMatchIndependentSegmentComposition)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  const CameraObservationBatch batch = fourCameraBatch(
    {{10.005, 10.015, 9.995, 9.985}});
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;

  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);
  ASSERT_TRUE(
    bundle.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  const auto * const covariance =
    bundle.validatedJointTargetPoseTransferNoiseCovariance(
      *bundle.validatedJointTargetPoseTransferNoiseReceipt());
  ASSERT_NE(covariance, nullptr);

  const auto verify_nested_pair =
    [&](const std::size_t near_camera_id,
      const std::size_t far_camera_id)
    {
      const livo::CameraPoseTransferRaw & near =
        bundle.transfers[near_camera_id].data();
      const livo::CameraPoseTransferRaw & far =
        bundle.transfers[far_camera_id].data();
      const livo::CameraTimeTransferState near_state =
        transferredState(near, anchor);
      const livo::CameraPoseTransferData segment =
        transfer.transfer(
        near_state, far.target_timestamp_sec);
      ASSERT_TRUE(segment.valid());

      const livo::CameraTimeTransferErrorStateMatrix
        independently_composed_far_covariance =
        segment.error_state_transition *
        near.process_noise_covariance *
        segment.error_state_transition.transpose() +
        segment.process_noise_covariance;
      EXPECT_TRUE(
        independently_composed_far_covariance.isApprox(
          far.process_noise_covariance, 1.0e-12));

      const livo::CameraTimeTransferPoseMatrix
        independent_cross_covariance =
        near.process_noise_covariance.topRows<
          livo::kCameraTimeTransferPoseDimension>() *
        segment.error_state_transition.topRows<
          livo::kCameraTimeTransferPoseDimension>().transpose();
      const Eigen::Index near_offset =
        static_cast<Eigen::Index>(near_camera_id) *
        livo::kCameraTimeTransferPoseDimension;
      const Eigen::Index far_offset =
        static_cast<Eigen::Index>(far_camera_id) *
        livo::kCameraTimeTransferPoseDimension;
      const livo::CameraTimeTransferPoseMatrix actual =
        covariance->block<
          livo::kCameraTimeTransferPoseDimension,
          livo::kCameraTimeTransferPoseDimension>(
          near_offset, far_offset);
      EXPECT_GT(actual.norm(), 0.0);
      EXPECT_TRUE(
        actual.isApprox(independent_cross_covariance, 1.0e-12));
    };

  verify_nested_pair(0U, 1U);
  verify_nested_pair(2U, 3U);

  const livo::CameraTimeTransferPoseMatrix
    opposite_direction_cross =
    covariance->block<
      livo::kCameraTimeTransferPoseDimension,
      livo::kCameraTimeTransferPoseDimension>(
      0, 2 * livo::kCameraTimeTransferPoseDimension);
  EXPECT_TRUE(opposite_direction_cross.isZero(0.0));
  EXPECT_TRUE(
    covariance->isApprox(covariance->transpose(), 1.0e-12));

  const Eigen::MatrixXd dynamic_covariance = *covariance;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
    dynamic_covariance);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1.0e-12);
}

TEST(
  CameraBatchTimeTransfer,
  JointReceiptFailsClosedForForeignOrTamperedBundle)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  const CameraObservationBatch batch = fourCameraBatch(
    {{10.005, 10.015, 9.995, 9.985}});
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;
  const livo::CameraBatchTimeTransferBundle first =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);
  const livo::CameraBatchTimeTransferBundle second =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);
  ASSERT_TRUE(
    first.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  ASSERT_TRUE(
    second.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  const auto & first_receipt =
    *first.validatedJointTargetPoseTransferNoiseReceipt();
  const auto & second_receipt =
    *second.validatedJointTargetPoseTransferNoiseReceipt();

  EXPECT_EQ(
    first.validatedJointTargetPoseTransferNoiseCovariance(
      second_receipt),
    nullptr);

  const livo::CameraBatchTimeTransferBundle clean_copy = first;
  EXPECT_NE(
    clean_copy.validatedJointTargetPoseTransferNoiseCovariance(
      first_receipt),
    nullptr);

  livo::CameraBatchTimeTransferBundle swapped = first;
  std::swap(swapped.transfers[0U], swapped.transfers[1U]);
  EXPECT_EQ(
    swapped.validatedJointTargetPoseTransferNoiseCovariance(
      first_receipt),
    nullptr);

  livo::CameraBatchTimeTransferBundle cross_build_entry = first;
  cross_build_entry.transfers[0U] = second.transfers[0U];
  EXPECT_EQ(
    cross_build_entry.
    validatedJointTargetPoseTransferNoiseCovariance(
      first_receipt),
    nullptr);

  livo::CameraBatchTimeTransferBundle stale_epoch = first;
  ++stale_epoch.imu_epoch;
  EXPECT_EQ(
    stale_epoch.validatedJointTargetPoseTransferNoiseCovariance(
      first_receipt),
    nullptr);

  livo::CameraBatchTimeTransferBundle stale_layout = first;
  stale_layout.active_state_layout_id = "stale-layout";
  EXPECT_EQ(
    stale_layout.
    validatedJointTargetPoseTransferNoiseCovariance(
      first_receipt),
    nullptr);

  livo::CameraBatchTimeTransferBundle tampered_diagnostics =
    first;
  tampered_diagnostics.anchor_timestamp_sec += 0.001;
  EXPECT_EQ(
    tampered_diagnostics.
    validatedJointTargetPoseTransferNoiseCovariance(
    first_receipt),
    nullptr);
}

TEST(
  CameraBatchTimeTransfer,
  SameTimestampSharesOnlyProcessNoiseAndZeroDurationSharesNone)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  const CameraObservationBatch batch = fourCameraBatch(
    {{10.010, 10.010, 10.000, 9.990}});
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;
  anchor.timestamp_stddev_sec = 0.002;
  anchor.velocity_world_mps = Eigen::Vector3d(1.0, -0.5, 0.2);

  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);
  ASSERT_TRUE(
    bundle.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  const auto * const covariance =
    bundle.validatedJointTargetPoseTransferNoiseCovariance(
      *bundle.validatedJointTargetPoseTransferNoiseReceipt());
  ASSERT_NE(covariance, nullptr);

  const livo::CameraTimeTransferPoseMatrix same_timestamp_cross =
    covariance->block<
      livo::kCameraTimeTransferPoseDimension,
      livo::kCameraTimeTransferPoseDimension>(
      0, livo::kCameraTimeTransferPoseDimension);
  EXPECT_TRUE(
    same_timestamp_cross.isApprox(
      bundle.transfers[0U].data().pose_transfer_covariance,
      1.0e-12));
  EXPECT_GT(same_timestamp_cross.norm(), 0.0);
  EXPECT_GT(
    bundle.transfers[0U].data().
    time_alignment_pose_covariance.norm(),
    0.0);

  const Eigen::Index zero_offset =
    2 * livo::kCameraTimeTransferPoseDimension;
  for (Eigen::Index camera_id = 0; camera_id < 4; ++camera_id)
  {
    if (camera_id == 2) {
      continue;
    }
    const livo::CameraTimeTransferPoseMatrix zero_cross =
      covariance->block<
        livo::kCameraTimeTransferPoseDimension,
        livo::kCameraTimeTransferPoseDimension>(
        zero_offset,
        camera_id * livo::kCameraTimeTransferPoseDimension);
    EXPECT_TRUE(zero_cross.isZero(0.0));
  }
}

TEST(
  CameraBatchTimeTransfer,
  NonGridTimestampsAreEquivariantUnderCameraPermutation)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  const CameraObservationBatch first_batch = fourCameraBatch(
    {{10.003, 10.012, 9.997, 9.988}});
  const CameraObservationBatch permuted_batch = fourCameraBatch(
    {{9.997, 9.988, 10.012, 10.003}});
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = first_batch.timestamp;

  const livo::CameraBatchTimeTransferBundle first =
    livo::buildCameraBatchTimeTransferBundle(
      first_batch, anchor, 0.0, transfer);
  const livo::CameraBatchTimeTransferBundle permuted =
    livo::buildCameraBatchTimeTransferBundle(
      permuted_batch, anchor, 0.0, transfer);
  ASSERT_TRUE(
    first.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  ASSERT_TRUE(
    permuted.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  const auto * const first_covariance =
    first.validatedJointTargetPoseTransferNoiseCovariance(
      *first.validatedJointTargetPoseTransferNoiseReceipt());
  const auto * const permuted_covariance =
    permuted.validatedJointTargetPoseTransferNoiseCovariance(
      *permuted.
      validatedJointTargetPoseTransferNoiseReceipt());
  ASSERT_NE(first_covariance, nullptr);
  ASSERT_NE(permuted_covariance, nullptr);

  const std::array<Eigen::Index, 4U> permuted_camera_for_first{{
      3, 2, 0, 1}};
  for (Eigen::Index first_left = 0; first_left < 4; ++first_left)
  {
    for (Eigen::Index first_right = 0;
         first_right < 4; ++first_right)
    {
      const Eigen::Index permuted_left =
        permuted_camera_for_first[
        static_cast<std::size_t>(first_left)];
      const Eigen::Index permuted_right =
        permuted_camera_for_first[
        static_cast<std::size_t>(first_right)];
      const livo::CameraTimeTransferPoseMatrix first_block =
        first_covariance->block<
          livo::kCameraTimeTransferPoseDimension,
          livo::kCameraTimeTransferPoseDimension>(
          first_left *
          livo::kCameraTimeTransferPoseDimension,
          first_right *
          livo::kCameraTimeTransferPoseDimension);
      const livo::CameraTimeTransferPoseMatrix permuted_block =
        permuted_covariance->block<
          livo::kCameraTimeTransferPoseDimension,
          livo::kCameraTimeTransferPoseDimension>(
          permuted_left *
          livo::kCameraTimeTransferPoseDimension,
          permuted_right *
          livo::kCameraTimeTransferPoseDimension);
      EXPECT_TRUE(first_block.isApprox(permuted_block, 0.0));
    }
  }
}

TEST(
  CameraBatchTimeTransfer,
  JointCovarianceRemainsPositiveSemidefiniteAcrossDirectionCases)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  const std::array<std::array<double, 4U>, 6U> timestamp_cases{{
      {{10.000, 10.000, 10.000, 10.000}},
      {{10.001, 10.004, 10.009, 10.017}},
      {{9.999, 9.996, 9.991, 9.983}},
      {{10.003, 10.014, 9.997, 9.986}},
      {{10.007, 10.007, 10.013, 10.019}},
      {{9.981, 9.989, 10.011, 10.019}}}};
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = 10.0;
  anchor.timestamp_stddev_sec = 0.001;
  anchor.velocity_world_mps = Eigen::Vector3d(0.7, -0.2, 0.1);

  for (std::size_t case_index = 0U;
       case_index < timestamp_cases.size(); ++case_index)
  {
    SCOPED_TRACE(case_index);
    const CameraObservationBatch batch =
      fourCameraBatch(timestamp_cases[case_index]);
    const livo::CameraBatchTimeTransferBundle bundle =
      livo::buildCameraBatchTimeTransferBundle(
        batch, anchor, 0.0, transfer);
    ASSERT_TRUE(
      bundle.validatedJointTargetPoseTransferNoiseReceipt().
      has_value());
    const auto * const covariance =
      bundle.validatedJointTargetPoseTransferNoiseCovariance(
        *bundle.
        validatedJointTargetPoseTransferNoiseReceipt());
    ASSERT_NE(covariance, nullptr);
    EXPECT_TRUE(
      covariance->isApprox(covariance->transpose(), 1.0e-12));

    const Eigen::MatrixXd dynamic_covariance = *covariance;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      dynamic_covariance);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GE(solver.eigenvalues().minCoeff(), -1.0e-12);

    for (Eigen::Index camera_id = 0; camera_id < 4; ++camera_id)
    {
      const livo::CameraTimeTransferPoseMatrix expected =
        bundle.transfers[
        static_cast<std::size_t>(camera_id)].data().
        pose_transfer_covariance +
        bundle.transfers[
        static_cast<std::size_t>(camera_id)].data().
        time_alignment_pose_covariance;
      const livo::CameraTimeTransferPoseMatrix actual =
        covariance->block<
          livo::kCameraTimeTransferPoseDimension,
          livo::kCameraTimeTransferPoseDimension>(
          camera_id *
          livo::kCameraTimeTransferPoseDimension,
          camera_id *
          livo::kCameraTimeTransferPoseDimension);
      EXPECT_TRUE(actual.isApprox(expected, 0.0));
    }
  }
}

TEST(CameraBatchTimeTransfer, BuildsOnlyPresentCameraTransfers)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  CameraObservationBatch batch;
  batch.present = {{true, true, false, true}};
  batch.camera_count = 3U;
  batch.partial = true;
  batch.timestamp = 10.0025;
  batch.observations[0U] = imageObservation(0U, 9.995);
  batch.observations[1U] = imageObservation(1U, 10.005);
  batch.observations[3U] = imageObservation(3U, 10.010);

  livo::CameraTimeTransferState anchor;
  constexpr double kExposureSec = 0.001;
  anchor.timestamp_sec = batch.timestamp + kExposureSec;
  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, kExposureSec, transfer);

  ASSERT_TRUE(bundle.batch_metadata_valid);
  EXPECT_EQ(bundle.present_camera_count, 3U);
  EXPECT_EQ(bundle.valid_camera_count, 3U);
  EXPECT_EQ(bundle.conditional_pose_only_shadow_count, 3U);
  EXPECT_EQ(
    bundle.active_state_jacobian_revision,
    livo::kCameraTimeTransferActiveJacobianRevision);
  EXPECT_EQ(
    bundle.active_state_layout_id,
    livo::kCameraTimeTransferActiveStateLayoutId);
  EXPECT_TRUE(bundle.activeStateLayoutCompatible());
  EXPECT_TRUE(bundle.allPresentFullStateTransfersValid());
  EXPECT_TRUE(bundle.allPresentTransfersValid());
  ASSERT_TRUE(bundle.validatedReceipt().has_value());
  ASSERT_TRUE(
    bundle.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  EXPECT_NE(
    bundle.validatedJointTargetPoseTransferNoiseCovariance(
      *bundle.
      validatedJointTargetPoseTransferNoiseReceipt()),
    nullptr);
  EXPECT_TRUE(bundle.validatedReceipt()->valid());
  EXPECT_EQ(bundle.imu_epoch, transfer.epoch());
  EXPECT_TRUE(bundle.present[0U]);
  EXPECT_TRUE(bundle.present[1U]);
  EXPECT_FALSE(bundle.present[2U]);
  EXPECT_TRUE(bundle.present[3U]);
  EXPECT_TRUE(bundle.transfers[0U].has_value());
  EXPECT_TRUE(bundle.transfers[1U].has_value());
  EXPECT_FALSE(bundle.transfers[2U].has_value());
  EXPECT_TRUE(bundle.transfers[3U].has_value());
  EXPECT_EQ(bundle.transfers[0U].cameraId(), 0U);
  EXPECT_EQ(bundle.transfers[1U].cameraId(), 1U);
  EXPECT_TRUE(
    bundle.validatedReceipt()->accepts(
      bundle.transfers[0U]));
  EXPECT_TRUE(
    bundle.validatedReceipt()->accepts(
      bundle.transfers[1U]));
  EXPECT_TRUE(
    bundle.transfers[0U].data().
    target_pose_wrt_active_anchor_state.col(
      livo::kCameraTimeTransferActiveExposureOffset).isZero(0.0));
  EXPECT_DOUBLE_EQ(
    bundle.transfers[0U].data().target_timestamp_sec, 9.996);
  EXPECT_DOUBLE_EQ(
    bundle.transfers[1U].data().target_timestamp_sec, 10.006);
  EXPECT_DOUBLE_EQ(
    bundle.transfers[3U].data().target_timestamp_sec, 10.011);

  const livo::CameraPoseTransfer & sealed_transfer =
    bundle.transfers[0U].value();
  EXPECT_TRUE(sealed_transfer.raw().internallyValid());
  EXPECT_DOUBLE_EQ(
    sealed_transfer.raw().target_timestamp_sec,
    bundle.transfers[0U].data().target_timestamp_sec);

  livo::CameraBatchTimeTransferBundle wrong_revision = bundle;
  ++wrong_revision.active_state_jacobian_revision;
  EXPECT_FALSE(wrong_revision.activeStateLayoutCompatible());
  EXPECT_FALSE(
    wrong_revision.allPresentFullStateTransfersValid());

  livo::CameraBatchTimeTransferBundle wrong_layout = bundle;
  wrong_layout.active_state_layout_id = "unknown";
  EXPECT_FALSE(wrong_layout.activeStateLayoutCompatible());
  EXPECT_FALSE(wrong_layout.allPresentFullStateTransfersValid());

  livo::CameraBatchTimeTransferBundle stale_count = bundle;
  stale_count.transfers[0U] =
    livo::CameraBatchTimeTransferEntry{};
  EXPECT_EQ(
    stale_count.valid_camera_count,
    stale_count.present_camera_count);
  EXPECT_FALSE(stale_count.transfers[0U].has_value());
  EXPECT_FALSE(stale_count.allPresentFullStateTransfersValid());

  livo::CameraBatchTimeTransferBundle swapped = bundle;
  std::swap(
    swapped.transfers[0U], swapped.transfers[1U]);
  EXPECT_FALSE(swapped.allPresentFullStateTransfersValid());
  EXPECT_EQ(swapped.transfers[0U].cameraId(), 1U);
  EXPECT_TRUE(
    swapped.validatedReceipt()->accepts(
      swapped.transfers[0U]));

  livo::CameraBatchTimeTransferBundle narrowed = bundle;
  narrowed.present[1U] = false;
  narrowed.present_camera_count = 2U;
  narrowed.valid_camera_count = 2U;
  EXPECT_FALSE(narrowed.allPresentFullStateTransfersValid());
}

TEST(
  CameraBatchTimeTransfer,
  UnbracketedCameraDoesNotInvalidateHealthyBoundEntry)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  CameraObservationBatch batch;
  batch.present = {{true, true, false, false}};
  batch.camera_count = 2U;
  batch.partial = true;
  batch.timestamp = 10.0;
  batch.observations[0U] = imageObservation(0U, 10.0);
  batch.observations[1U] = imageObservation(1U, 10.03);

  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;
  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);

  ASSERT_TRUE(bundle.batch_metadata_valid);
  EXPECT_EQ(bundle.present_camera_count, 2U);
  EXPECT_EQ(bundle.valid_camera_count, 1U);
  ASSERT_TRUE(bundle.validatedReceipt().has_value());
  ASSERT_TRUE(bundle.validatedReceipt()->valid());
  ASSERT_TRUE(bundle.transfers[0U].has_value());
  EXPECT_TRUE(
    bundle.validatedReceipt()->accepts(
      bundle.transfers[0U]));
  EXPECT_FALSE(bundle.transfers[1U].has_value());
  EXPECT_EQ(
    bundle.transfers[1U].data().status,
    livo::CameraTimeTransferStatus::kMissingRightBracket);
  EXPECT_FALSE(
    bundle.validatedReceipt()->accepts(
      bundle.transfers[1U]));
  EXPECT_FALSE(bundle.allPresentFullStateTransfersValid());
  EXPECT_FALSE(
    bundle.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
}

TEST(
  CameraBatchTimeTransfer,
  IdenticalMetadataFromIndependentBuildsDoesNotShareAuthority)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  CameraObservationBatch batch;
  batch.present = {{true, false, false, false}};
  batch.camera_count = 1U;
  batch.partial = true;
  batch.timestamp = 10.0;
  batch.observations[0U] = imageObservation(0U, 10.0);

  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;
  const livo::CameraBatchTimeTransferBundle first =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);
  const livo::CameraBatchTimeTransferBundle second =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);

  ASSERT_TRUE(first.validatedReceipt().has_value());
  ASSERT_TRUE(second.validatedReceipt().has_value());
  ASSERT_TRUE(
    first.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  ASSERT_TRUE(
    second.validatedJointTargetPoseTransferNoiseReceipt().
    has_value());
  ASSERT_TRUE(first.transfers[0U].has_value());
  ASSERT_TRUE(second.transfers[0U].has_value());
  EXPECT_FALSE(
    first.validatedReceipt()->accepts(
      second.transfers[0U]));
  EXPECT_FALSE(
    second.validatedReceipt()->accepts(
      first.transfers[0U]));
  EXPECT_NE(
    first.validatedJointTargetPoseTransferNoiseCovariance(
      *first.
      validatedJointTargetPoseTransferNoiseReceipt()),
    nullptr);
  EXPECT_EQ(
    first.validatedJointTargetPoseTransferNoiseCovariance(
      *second.
      validatedJointTargetPoseTransferNoiseReceipt()),
    nullptr);

  const livo::CameraBatchTimeTransferReceipt copied_receipt =
    *first.validatedReceipt();
  const livo::CameraBatchTimeTransferEntry copied_entry =
    first.transfers[0U];
  EXPECT_TRUE(copied_receipt.accepts(copied_entry));
}

TEST(CameraBatchTimeTransfer, RejectsInconsistentBatchMetadataAtomically)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  CameraObservationBatch batch;
  batch.present = {{true, true, false, false}};
  batch.camera_count = 3U;
  batch.partial = true;
  batch.timestamp = 10.0;
  batch.observations[0U] = imageObservation(0U, 9.995);
  batch.observations[1U] = imageObservation(1U, 10.005);

  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch.timestamp;
  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);

  EXPECT_FALSE(bundle.batch_metadata_valid);
  EXPECT_FALSE(bundle.validatedReceipt().has_value());
  EXPECT_EQ(bundle.valid_camera_count, 0U);
  EXPECT_FALSE(bundle.allPresentTransfersValid());
  for (
    const livo::CameraBatchTimeTransferEntry & camera :
    bundle.transfers)
  {
    EXPECT_FALSE(camera.has_value());
  }
}

TEST(CameraBatchTimeTransfer, RejectsAnchorTimestampMismatchAtomically)
{
  livo::CameraTimeTransfer transfer = configuredTransfer();
  CameraObservationBatch batch;
  batch.present.fill(true);
  batch.camera_count = CameraObservationQueue::kCameraCount;
  batch.partial = false;
  batch.timestamp = 10.0;
  for (std::size_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount; ++camera_id)
  {
    batch.observations[camera_id] = imageObservation(
      static_cast<std::uint8_t>(camera_id),
      9.995 + 0.003 * static_cast<double>(camera_id));
  }

  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = 10.01;
  const livo::CameraBatchTimeTransferBundle bundle =
    livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, 0.0, transfer);

  EXPECT_FALSE(bundle.batch_metadata_valid);
  EXPECT_EQ(bundle.valid_camera_count, 0U);
}

}  // namespace
