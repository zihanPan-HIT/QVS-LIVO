#include "camera_time_transfer.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace
{

using livo::CameraPoseTransfer;
using livo::CameraPoseTransferData;
using livo::CameraTimeTransfer;
using livo::CameraTimeTransferAppendStatus;
using livo::CameraTimeTransferConfig;
using livo::CameraTimeTransferImuSample;
using livo::CameraTimeTransferState;
using livo::CameraTimeTransferStatus;

static_assert(
  !std::is_default_constructible_v<CameraPoseTransfer>);
static_assert(!std::is_aggregate_v<CameraPoseTransfer>);
static_assert(!std::is_copy_constructible_v<CameraTimeTransfer>);
static_assert(!std::is_copy_assignable_v<CameraTimeTransfer>);
static_assert(std::is_nothrow_move_constructible_v<CameraTimeTransfer>);
static_assert(std::is_nothrow_move_assignable_v<CameraTimeTransfer>);

CameraPoseTransferData validPoseTransferValue()
{
  CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.1;
  config.max_imu_gap_sec = 0.02;
  config.imu_timestamp_stddev_sec = 1.0e-5;
  CameraTimeTransfer transfer(config);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.0;
  anchor.gravity_world_mps2.setZero();

  std::vector<CameraTimeTransferImuSample> samples(3U);
  samples[0U].timestamp_sec = 1.0;
  samples[1U].timestamp_sec = 1.005;
  samples[2U].timestamp_sec = 1.01;
  for (CameraTimeTransferImuSample & imu : samples)
  {
    imu.angular_velocity_radps.setZero();
    imu.specific_force_mps2.setZero();
  }
  return transfer.transferWithSamples(
    anchor, samples, 1.01, 0.0);
}

TEST(
  CameraTimeTransfer,
  CallerFilledRawDtoCannotMintSuccessfulTransfer)
{
  const CameraPoseTransferData sealed = validPoseTransferValue();
  CameraPoseTransferData raw_only;
  static_cast<livo::CameraPoseTransferRaw &>(raw_only) =
    static_cast<const livo::CameraPoseTransferRaw &>(sealed);

  ASSERT_TRUE(sealed.validatedValue().has_value());
  EXPECT_TRUE(raw_only.internallyValid());
  EXPECT_FALSE(raw_only.valid());
  EXPECT_FALSE(raw_only.validatedValue().has_value());
}

TEST(
  CameraTimeTransfer,
  RawDtoRejectsNonFiniteNonSymmetricAndIndefiniteCovariance)
{
  CameraPoseTransferData nonfinite = validPoseTransferValue();
  nonfinite.process_noise_covariance(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(nonfinite.internallyValid());
  EXPECT_FALSE(nonfinite.valid());

  CameraPoseTransferData nonsymmetric = validPoseTransferValue();
  nonsymmetric.process_noise_covariance(0, 1) += 1.0e-3;
  EXPECT_FALSE(nonsymmetric.internallyValid());
  EXPECT_FALSE(nonsymmetric.valid());

  CameraPoseTransferData indefinite = validPoseTransferValue();
  indefinite.process_noise_covariance.setZero();
  indefinite.pose_transfer_covariance.setZero();
  indefinite.process_noise_covariance(0, 0) = -1.0;
  indefinite.pose_transfer_covariance(0, 0) = -1.0;
  EXPECT_FALSE(indefinite.internallyValid());
  EXPECT_FALSE(indefinite.valid());
}

enum class InvalidTransferScalarCase
{
  kAnchorNan,
  kAnchorPositiveInfinity,
  kTargetNan,
  kTargetPositiveInfinity,
  kDeltaNan,
  kDeltaPositiveInfinity,
  kDeltaInconsistent,
  kTimeUncertaintyNan,
  kTimeUncertaintyPositiveInfinity,
  kTimeUncertaintyNegative,
  kMaximumGapNan,
  kMaximumGapPositiveInfinity,
  kMaximumGapNegative,
  kTransitionConditionNan,
  kTransitionConditionPositiveInfinity,
  kTransitionConditionBelowOne,
  kZeroEpoch
};

class CameraPoseTransferScalarValidityTest :
  public ::testing::TestWithParam<InvalidTransferScalarCase>
{
};

TEST_P(
  CameraPoseTransferScalarValidityTest,
  RejectsEveryInvalidScalarBeforeUse)
{
  CameraPoseTransferData value = validPoseTransferValue();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  switch (GetParam())
  {
    case InvalidTransferScalarCase::kAnchorNan:
      value.anchor_timestamp_sec = nan;
      break;
    case InvalidTransferScalarCase::kAnchorPositiveInfinity:
      value.anchor_timestamp_sec = infinity;
      break;
    case InvalidTransferScalarCase::kTargetNan:
      value.target_timestamp_sec = nan;
      break;
    case InvalidTransferScalarCase::kTargetPositiveInfinity:
      value.target_timestamp_sec = infinity;
      break;
    case InvalidTransferScalarCase::kDeltaNan:
      value.delta_time_sec = nan;
      break;
    case InvalidTransferScalarCase::kDeltaPositiveInfinity:
      value.delta_time_sec = infinity;
      break;
    case InvalidTransferScalarCase::kDeltaInconsistent:
      value.delta_time_sec = 0.02;
      break;
    case InvalidTransferScalarCase::kTimeUncertaintyNan:
      value.time_uncertainty_stddev_sec = nan;
      break;
    case InvalidTransferScalarCase::kTimeUncertaintyPositiveInfinity:
      value.time_uncertainty_stddev_sec = infinity;
      break;
    case InvalidTransferScalarCase::kTimeUncertaintyNegative:
      value.time_uncertainty_stddev_sec = -1.0;
      break;
    case InvalidTransferScalarCase::kMaximumGapNan:
      value.maximum_imu_gap_sec = nan;
      break;
    case InvalidTransferScalarCase::kMaximumGapPositiveInfinity:
      value.maximum_imu_gap_sec = infinity;
      break;
    case InvalidTransferScalarCase::kMaximumGapNegative:
      value.maximum_imu_gap_sec = -1.0;
      break;
    case InvalidTransferScalarCase::kTransitionConditionNan:
      value.transition_condition_number = nan;
      break;
    case InvalidTransferScalarCase::
      kTransitionConditionPositiveInfinity:
      value.transition_condition_number = infinity;
      break;
    case InvalidTransferScalarCase::kTransitionConditionBelowOne:
      value.transition_condition_number = 0.999;
      break;
    case InvalidTransferScalarCase::kZeroEpoch:
      value.imu_epoch = 0U;
      break;
  }

  EXPECT_FALSE(value.valid());
}

INSTANTIATE_TEST_SUITE_P(
  TimeFieldAudit,
  CameraPoseTransferScalarValidityTest,
  ::testing::Values(
    InvalidTransferScalarCase::kAnchorNan,
    InvalidTransferScalarCase::kAnchorPositiveInfinity,
    InvalidTransferScalarCase::kTargetNan,
    InvalidTransferScalarCase::kTargetPositiveInfinity,
    InvalidTransferScalarCase::kDeltaNan,
    InvalidTransferScalarCase::kDeltaPositiveInfinity,
    InvalidTransferScalarCase::kDeltaInconsistent,
    InvalidTransferScalarCase::kTimeUncertaintyNan,
    InvalidTransferScalarCase::kTimeUncertaintyPositiveInfinity,
    InvalidTransferScalarCase::kTimeUncertaintyNegative,
    InvalidTransferScalarCase::kMaximumGapNan,
    InvalidTransferScalarCase::kMaximumGapPositiveInfinity,
    InvalidTransferScalarCase::kMaximumGapNegative,
    InvalidTransferScalarCase::kTransitionConditionNan,
    InvalidTransferScalarCase::
    kTransitionConditionPositiveInfinity,
    InvalidTransferScalarCase::kTransitionConditionBelowOne,
    InvalidTransferScalarCase::kZeroEpoch));

CameraTimeTransferImuSample sample(
  const double timestamp_sec,
  const Eigen::Vector3d & angular_velocity,
  const Eigen::Vector3d & specific_force)
{
  CameraTimeTransferImuSample result;
  result.timestamp_sec = timestamp_sec;
  result.angular_velocity_radps = angular_velocity;
  result.specific_force_mps2 = specific_force;
  return result;
}

CameraTimeTransferConfig testConfig()
{
  CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.1;
  config.max_imu_gap_sec = 0.02;
  config.imu_timestamp_stddev_sec = 1.0e-5;
  return config;
}

Eigen::Vector3d logSo3(const Eigen::Matrix3d & rotation)
{
  const Eigen::AngleAxisd angle_axis(rotation);
  if (std::abs(angle_axis.angle()) <= 1.0e-15)
  {
    return Eigen::Vector3d::Zero();
  }
  return angle_axis.axis() * angle_axis.angle();
}

Eigen::Vector3d independentOracleAngularVelocity(
  const double timestamp_sec)
{
  const double phase = timestamp_sec / 0.040;
  return Eigen::Vector3d(
    0.31 + 0.17 * phase + 0.03 * phase * phase,
    -0.24 + 0.11 * phase - 0.02 * phase * phase,
    0.19 - 0.07 * phase + 0.04 * phase * phase);
}

Eigen::Vector3d independentOracleSpecificForce(
  const double timestamp_sec)
{
  const double phase = timestamp_sec / 0.040;
  return Eigen::Vector3d(
    1.20 + 0.40 * phase - 0.10 * phase * phase,
    -0.70 + 0.20 * phase + 0.08 * phase * phase,
    9.40 - 0.30 * phase + 0.12 * phase * phase);
}

struct QuaternionRk4NominalState
{
  Eigen::Vector4d rotation_world_from_imu_wxyz{
    Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)};
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_world_mps{Eigen::Vector3d::Zero()};
};

struct QuaternionRk4NominalDerivative
{
  Eigen::Vector4d rotation_wxyz_per_sec{
    Eigen::Vector4d::Zero()};
  Eigen::Vector3d position_mps{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_mps2{Eigen::Vector3d::Zero()};
};

Eigen::Vector4d normalizedQuaternionWxyz(
  const Eigen::Vector4d & quaternion_wxyz)
{
  return quaternion_wxyz / quaternion_wxyz.norm();
}

QuaternionRk4NominalState advanceQuaternionRk4State(
  const QuaternionRk4NominalState & state,
  const QuaternionRk4NominalDerivative & derivative,
  const double duration_sec)
{
  QuaternionRk4NominalState advanced;
  advanced.rotation_world_from_imu_wxyz =
    normalizedQuaternionWxyz(
      state.rotation_world_from_imu_wxyz +
      duration_sec * derivative.rotation_wxyz_per_sec);
  advanced.position_world_m =
    state.position_world_m +
    duration_sec * derivative.position_mps;
  advanced.velocity_world_mps =
    state.velocity_world_mps +
    duration_sec * derivative.velocity_mps2;
  return advanced;
}

QuaternionRk4NominalDerivative evaluateQuaternionRk4Derivative(
  const QuaternionRk4NominalState & state,
  const CameraTimeTransferState & fixed_parameters,
  const double timestamp_sec)
{
  const Eigen::Vector4d quaternion_wxyz =
    normalizedQuaternionWxyz(
      state.rotation_world_from_imu_wxyz);
  const double scalar = quaternion_wxyz[0];
  const Eigen::Vector3d vector = quaternion_wxyz.tail<3>();
  const Eigen::Vector3d angular_velocity =
    independentOracleAngularVelocity(timestamp_sec) -
    fixed_parameters.gyro_bias_radps;
  QuaternionRk4NominalDerivative derivative;
  derivative.rotation_wxyz_per_sec[0] =
    -0.5 * vector.dot(angular_velocity);
  derivative.rotation_wxyz_per_sec.tail<3>() =
    0.5 *
    (scalar * angular_velocity +
     vector.cross(angular_velocity));
  derivative.position_mps = state.velocity_world_mps;
  const Eigen::Quaterniond rotation(
    scalar, vector.x(), vector.y(), vector.z());
  derivative.velocity_mps2 =
    rotation.toRotationMatrix() *
    (independentOracleSpecificForce(timestamp_sec) -
     fixed_parameters.accel_bias_mps2) +
    fixed_parameters.gravity_world_mps2;
  return derivative;
}

QuaternionRk4NominalState quaternionRk4Step(
  const QuaternionRk4NominalState & state,
  const CameraTimeTransferState & fixed_parameters,
  const double timestamp_sec,
  const double duration_sec)
{
  const QuaternionRk4NominalDerivative first =
    evaluateQuaternionRk4Derivative(
      state, fixed_parameters, timestamp_sec);
  const QuaternionRk4NominalDerivative second =
    evaluateQuaternionRk4Derivative(
      advanceQuaternionRk4State(
        state, first, 0.5 * duration_sec),
      fixed_parameters,
      timestamp_sec + 0.5 * duration_sec);
  const QuaternionRk4NominalDerivative third =
    evaluateQuaternionRk4Derivative(
      advanceQuaternionRk4State(
        state, second, 0.5 * duration_sec),
      fixed_parameters,
      timestamp_sec + 0.5 * duration_sec);
  const QuaternionRk4NominalDerivative fourth =
    evaluateQuaternionRk4Derivative(
      advanceQuaternionRk4State(
        state, third, duration_sec),
      fixed_parameters,
      timestamp_sec + duration_sec);

  QuaternionRk4NominalState result;
  result.rotation_world_from_imu_wxyz =
    normalizedQuaternionWxyz(
      state.rotation_world_from_imu_wxyz +
      (duration_sec / 6.0) *
      (first.rotation_wxyz_per_sec +
       2.0 * second.rotation_wxyz_per_sec +
       2.0 * third.rotation_wxyz_per_sec +
       fourth.rotation_wxyz_per_sec));
  result.position_world_m =
    state.position_world_m +
    (duration_sec / 6.0) *
    (first.position_mps +
     2.0 * second.position_mps +
     2.0 * third.position_mps +
     fourth.position_mps);
  result.velocity_world_mps =
    state.velocity_world_mps +
    (duration_sec / 6.0) *
    (first.velocity_mps2 +
     2.0 * second.velocity_mps2 +
     2.0 * third.velocity_mps2 +
     fourth.velocity_mps2);
  return result;
}

CameraTimeTransferState integrateIndependentQuaternionRk4(
  const CameraTimeTransferState & anchor,
  const double target_timestamp_sec,
  const double maximum_step_sec)
{
  const Eigen::Quaterniond anchor_rotation(
    anchor.rotation_world_from_imu);
  QuaternionRk4NominalState state;
  state.rotation_world_from_imu_wxyz =
    Eigen::Vector4d(
      anchor_rotation.w(),
      anchor_rotation.x(),
      anchor_rotation.y(),
      anchor_rotation.z());
  state.position_world_m = anchor.position_world_m;
  state.velocity_world_mps = anchor.velocity_world_mps;

  double timestamp_sec = anchor.timestamp_sec;
  while (
    std::abs(target_timestamp_sec - timestamp_sec) >
    1.0e-15)
  {
    const double remaining_sec =
      target_timestamp_sec - timestamp_sec;
    const double step_sec = std::copysign(
      std::min(maximum_step_sec, std::abs(remaining_sec)),
      remaining_sec);
    state = quaternionRk4Step(
      state, anchor, timestamp_sec, step_sec);
    timestamp_sec += step_sec;
  }

  const Eigen::Vector4d quaternion_wxyz =
    normalizedQuaternionWxyz(
      state.rotation_world_from_imu_wxyz);
  CameraTimeTransferState result = anchor;
  result.timestamp_sec = target_timestamp_sec;
  result.rotation_world_from_imu =
    Eigen::Quaterniond(
      quaternion_wxyz[0],
      quaternion_wxyz[1],
      quaternion_wxyz[2],
      quaternion_wxyz[3]).toRotationMatrix();
  result.position_world_m = state.position_world_m;
  result.velocity_world_mps = state.velocity_world_mps;
  return result;
}

CameraTimeTransferState perturbState(
  const CameraTimeTransferState & state,
  const int column,
  const double increment)
{
  CameraTimeTransferState perturbed = state;
  Eigen::Vector3d delta = Eigen::Vector3d::Zero();
  if (column < 3)
  {
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    axis[column] = 1.0;
    perturbed.rotation_world_from_imu *=
      Eigen::AngleAxisd(increment, axis)
      .toRotationMatrix();
  }
  else
  {
    delta[column % 3] = increment;
    if (column < 6)
    {
      perturbed.position_world_m += delta;
    }
    else if (column < 9)
    {
      perturbed.velocity_world_mps += delta;
    }
    else if (column < 12)
    {
      perturbed.gyro_bias_radps += delta;
    }
    else if (column < 15)
    {
      perturbed.accel_bias_mps2 += delta;
    }
    else
    {
      perturbed.gravity_world_mps2 += delta;
    }
  }
  return perturbed;
}

void expectFullPoseJacobianMatchesFiniteDifference(
  const CameraTimeTransfer & transfer,
  const CameraTimeTransferState & anchor,
  const std::vector<CameraTimeTransferImuSample> & imu,
  const double target_timestamp_sec)
{
  const CameraPoseTransferData nominal =
    transfer.transferWithSamples(
      anchor, imu, target_timestamp_sec);
  ASSERT_TRUE(nominal.valid());
  EXPECT_TRUE(
    nominal.target_pose_wrt_anchor_state.isApprox(
      nominal.error_state_transition.topRows<6>(),
      1.0e-14));
  EXPECT_TRUE(
    nominal.target_pose_wrt_active_anchor_state.isApprox(
      livo::embedCameraTimeTransferPoseJacobianInActiveState(
        nominal.target_pose_wrt_anchor_state),
      1.0e-14));
  EXPECT_TRUE(
    nominal.target_pose_wrt_active_anchor_state.col(
      livo::kCameraTimeTransferActiveExposureOffset).isZero(0.0));
  EXPECT_TRUE(
    nominal.conditional_pose_only.target_pose_wrt_anchor_pose.
    isApprox(
      nominal.target_pose_wrt_anchor_state.leftCols<6>(),
      1.0e-14));
  EXPECT_TRUE(
    livo::ConditionalPoseOnlyTimeTransfer::kShadowOnly);
  EXPECT_GT(
    (nominal.error_state_transition.block<3, 3>(
      livo::kCameraTimeTransferThetaOffset,
      livo::kCameraTimeTransferGyroBiasOffset).norm()),
    0.0);
  EXPECT_GT(
    (nominal.error_state_transition.block<3, 3>(
      livo::kCameraTimeTransferPositionOffset,
      livo::kCameraTimeTransferVelocityOffset).norm()),
    0.0);
  EXPECT_GT(
    (nominal.error_state_transition.block<3, 3>(
      livo::kCameraTimeTransferVelocityOffset,
      livo::kCameraTimeTransferThetaOffset).norm()),
    0.0);
  EXPECT_GT(
    (nominal.error_state_transition.block<3, 3>(
      livo::kCameraTimeTransferVelocityOffset,
      livo::kCameraTimeTransferAccelBiasOffset).norm()),
    0.0);
  EXPECT_GT(
    (nominal.error_state_transition.block<3, 3>(
      livo::kCameraTimeTransferVelocityOffset,
      livo::kCameraTimeTransferGravityOffset).norm()),
    0.0);
  EXPECT_GT(nominal.pose_transfer_covariance.trace(), 0.0);
  const Eigen::SelfAdjointEigenSolver<
    livo::CameraTimeTransferErrorStateMatrix> noise_solver(
      nominal.process_noise_covariance);
  ASSERT_EQ(noise_solver.info(), Eigen::Success);
  EXPECT_GE(noise_solver.eigenvalues().minCoeff(), -1.0e-12);

  constexpr double kEpsilon = 1.0e-6;
  for (int column = 0;
       column < livo::kCameraTimeTransferErrorStateDimension;
       ++column)
  {
    SCOPED_TRACE(
      "dt=" + std::to_string(
        target_timestamp_sec - anchor.timestamp_sec) +
      " column=" + std::to_string(column));
    const CameraPoseTransferData plus =
      transfer.transferWithSamples(
        perturbState(anchor, column, kEpsilon),
        imu, target_timestamp_sec);
    const CameraPoseTransferData minus =
      transfer.transferWithSamples(
        perturbState(anchor, column, -kEpsilon),
        imu, target_timestamp_sec);
    ASSERT_TRUE(plus.valid());
    ASSERT_TRUE(minus.valid());

    Eigen::Matrix<double, 6, 1> observed;
    observed.head<3>() =
      (logSo3(
        nominal.rotation_world_from_imu.transpose() *
        plus.rotation_world_from_imu) -
       logSo3(
        nominal.rotation_world_from_imu.transpose() *
        minus.rotation_world_from_imu)) /
      (2.0 * kEpsilon);
    observed.tail<3>() =
      (plus.position_world_m - minus.position_world_m) /
      (2.0 * kEpsilon);
    EXPECT_TRUE(observed.isApprox(
      nominal.target_pose_wrt_anchor_state.col(column),
      2.0e-6))
      << "observed=" << observed.transpose()
      << " expected="
      << nominal.target_pose_wrt_anchor_state.col(column).transpose();
  }
}

TEST(CameraTimeTransfer, ActiveStateEmbeddingMapsEveryColumn)
{
  livo::CameraTimeTransferPoseJacobian canonical;
  for (Eigen::Index row = 0;
       row < canonical.rows(); ++row)
  {
    for (Eigen::Index column = 0;
         column < canonical.cols(); ++column)
    {
      canonical(row, column) =
        100.0 * static_cast<double>(row + 1) +
        static_cast<double>(column + 1);
    }
  }

  const livo::CameraTimeTransferActivePoseJacobian active =
    livo::embedCameraTimeTransferPoseJacobianInActiveState(
      canonical);

  static_assert(
    livo::CameraTimeTransferActivePoseJacobian::ColsAtCompileTime == 19,
    "FAST-LIVO2 active state must remain 19-dimensional");
  EXPECT_STREQ(
    livo::kCameraTimeTransferActiveStateLayoutId,
    "fast_livo2_states_group_v1:"
    "theta[0:3],p[3:6],inv_exposure[6],"
    "v[7:10],bg[10:13],ba[13:16],g[16:19]");
  EXPECT_EQ(livo::kCameraTimeTransferActiveJacobianRevision, 1U);

  for (Eigen::Index active_column = 0;
       active_column <
       livo::kCameraTimeTransferActiveStateDimension;
       ++active_column)
  {
    SCOPED_TRACE(
      "active column=" + std::to_string(active_column));
    if (
      active_column ==
      livo::kCameraTimeTransferActiveExposureOffset)
    {
      EXPECT_TRUE(active.col(active_column).isZero(0.0));
      continue;
    }
    const Eigen::Index canonical_column =
      active_column <
        livo::kCameraTimeTransferActiveExposureOffset ?
      active_column : active_column - 1;
    EXPECT_TRUE(
      active.col(active_column).isApprox(
        canonical.col(canonical_column), 0.0));
  }
}

TEST(CameraTimeTransfer, DefaultMaximumTransferIntervalRemainsFiftyMilliseconds)
{
  const CameraTimeTransferConfig default_config;
  EXPECT_DOUBLE_EQ(
    default_config.maximum_transition_condition_number,
    1.0e12);
  CameraTimeTransfer transfer;
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(0.00, zero, zero),
    sample(0.01, zero, zero),
    sample(0.02, zero, zero),
    sample(0.03, zero, zero),
    sample(0.04, zero, zero),
    sample(0.05, zero, zero),
    sample(0.06, zero, zero)};

  EXPECT_TRUE(
    transfer.transferWithSamples(anchor, imu, 0.05).valid());
  EXPECT_EQ(
    transfer.transferWithSamples(anchor, imu, 0.050001).status,
    CameraTimeTransferStatus::kTargetOutOfRange);
}

TEST(
  CameraTimeTransfer,
  LegacyNoiseConfigurationAdaptsToExplicitContinuousDensityUnits)
{
  CameraTimeTransferConfig config = testConfig();
  config.gyro_noise_stddev_radps = 0.011;
  config.accel_noise_stddev_mps2 = 0.12;
  config.gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 1.3e-4;
  config.accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 2.4e-3;

  const auto density =
    config.validatedContinuousNoiseDensity();

  ASSERT_TRUE(density.has_value());
  EXPECT_DOUBLE_EQ(
    density->gyro_measurement_rad_per_sqrt_sec,
    0.011);
  EXPECT_DOUBLE_EQ(
    density->accel_measurement_mps_per_sqrt_sec,
    0.12);
  EXPECT_DOUBLE_EQ(
    density->gyro_bias_random_walk_radps_per_sqrt_sec,
    1.3e-4);
  EXPECT_DOUBLE_EQ(
    density->accel_bias_random_walk_mps2_per_sqrt_sec,
    2.4e-3);
}

TEST(
  CameraTimeTransfer,
  ContinuousNoiseDensityAdapterRejectsEveryInvalidLegacyAmplitude)
{
  const std::array<double, 3> invalid_values{{
      -1.0e-12,
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()}};
  for (std::size_t field = 0U; field < 4U; ++field)
  {
    for (const double invalid_value : invalid_values)
    {
      SCOPED_TRACE(
        "field=" + std::to_string(field) +
        " value=" + std::to_string(invalid_value));
      CameraTimeTransferConfig config = testConfig();
      switch (field)
      {
        case 0U:
          config.gyro_noise_stddev_radps = invalid_value;
          break;
        case 1U:
          config.accel_noise_stddev_mps2 = invalid_value;
          break;
        case 2U:
          config.
            gyro_bias_random_walk_stddev_radps_per_sqrt_sec =
            invalid_value;
          break;
        case 3U:
          config.
            accel_bias_random_walk_stddev_mps2_per_sqrt_sec =
            invalid_value;
          break;
      }

      EXPECT_FALSE(
        config.validatedContinuousNoiseDensity().has_value());
      EXPECT_FALSE(CameraTimeTransfer(config).configValid());
    }
  }
}

TEST(CameraTimeTransfer, ConstantAngularVelocitySupportsPositiveDt)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d omega(0.0, 0.0, 1.0);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(-0.01, omega, Eigen::Vector3d::Zero()),
    sample(0.00, omega, Eigen::Vector3d::Zero()),
    sample(0.01, omega, Eigen::Vector3d::Zero()),
    sample(0.02, omega, Eigen::Vector3d::Zero()),
    sample(0.03, omega, Eigen::Vector3d::Zero())};

  const CameraPoseTransferData result =
    transfer.transferWithSamples(anchor, imu, 0.025, 2.0e-5);

  ASSERT_TRUE(result.valid());
  const Eigen::AngleAxisd expected(0.025, Eigen::Vector3d::UnitZ());
  EXPECT_TRUE(result.rotation_world_from_imu.isApprox(
    expected.toRotationMatrix(), 1.0e-10));
  EXPECT_TRUE(result.position_world_m.isZero(1.0e-12));
  EXPECT_TRUE((
    result.target_pose_wrt_anchor_state.block<3, 3>(0, 0).isApprox(
      expected.toRotationMatrix().transpose(), 1.0e-10)));
}

TEST(
  CameraTimeTransfer,
  ShortStepMatchesFastLivoSiBiasGravityAndRightErrorConventions)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  anchor.position_world_m = Eigen::Vector3d(1.0, 2.0, 3.0);
  anchor.velocity_world_mps =
    Eigen::Vector3d(0.5, -0.2, 0.1);
  anchor.gyro_bias_radps =
    Eigen::Vector3d(0.01, -0.02, 0.03);
  anchor.accel_bias_mps2 =
    Eigen::Vector3d(0.1, -0.2, 0.3);
  anchor.gravity_world_mps2 =
    Eigen::Vector3d(0.0, 0.0, -9.81);
  const Eigen::Vector3d true_angular_velocity(0.0, 0.0, 0.2);
  const Eigen::Vector3d measured_angular_velocity =
    true_angular_velocity + anchor.gyro_bias_radps;
  const Eigen::Vector3d measured_specific_force =
    anchor.accel_bias_mps2 +
    Eigen::Vector3d(0.0, 0.0, 9.81);

  const CameraPoseTransferData result = transfer.transferWithSamples(
    anchor,
    {
      sample(
        0.000, measured_angular_velocity,
        measured_specific_force),
      sample(
        0.010, measured_angular_velocity,
        measured_specific_force)},
    0.010);

  ASSERT_TRUE(result.valid());
  const Eigen::Matrix3d expected_rotation =
    Eigen::AngleAxisd(0.002, Eigen::Vector3d::UnitZ())
    .toRotationMatrix();
  EXPECT_TRUE(result.rotation_world_from_imu.isApprox(
    expected_rotation, 1.0e-12));
  EXPECT_TRUE(result.velocity_world_mps.isApprox(
    anchor.velocity_world_mps, 1.0e-12));
  EXPECT_TRUE(result.position_world_m.isApprox(
    anchor.position_world_m +
    0.010 * anchor.velocity_world_mps,
    1.0e-12));
  EXPECT_NEAR(
    result.error_state_transition(
      livo::kCameraTimeTransferThetaOffset + 2,
      livo::kCameraTimeTransferGyroBiasOffset + 2),
    -0.010, 1.0e-12);
  EXPECT_NEAR(
    result.error_state_transition(
      livo::kCameraTimeTransferPositionOffset + 2,
      livo::kCameraTimeTransferVelocityOffset + 2),
    0.010, 1.0e-12);
  EXPECT_NEAR(
    result.error_state_transition(
      livo::kCameraTimeTransferVelocityOffset + 2,
      livo::kCameraTimeTransferAccelBiasOffset + 2),
    -0.010, 1.0e-12);
  EXPECT_NEAR(
    result.error_state_transition(
      livo::kCameraTimeTransferVelocityOffset + 2,
      livo::kCameraTimeTransferGravityOffset + 2),
    0.010, 1.0e-12);
}

TEST(CameraTimeTransfer, ConstantAccelerationSupportsPositiveAndNegativeDt)
{
  CameraTimeTransfer transfer(testConfig());
  const Eigen::Vector3d acceleration(1.0, -2.0, 0.5);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(-0.01, Eigen::Vector3d::Zero(), acceleration),
    sample(0.00, Eigen::Vector3d::Zero(), acceleration),
    sample(0.01, Eigen::Vector3d::Zero(), acceleration),
    sample(0.02, Eigen::Vector3d::Zero(), acceleration),
    sample(0.03, Eigen::Vector3d::Zero(), acceleration)};

  CameraTimeTransferState start;
  start.timestamp_sec = 0.0;
  const CameraPoseTransferData forward =
    transfer.transferWithSamples(start, imu, 0.02);
  ASSERT_TRUE(forward.valid());
  EXPECT_TRUE(forward.velocity_world_mps.isApprox(
    acceleration * 0.02, 1.0e-12));
  EXPECT_TRUE(forward.position_world_m.isApprox(
    0.5 * acceleration * 0.02 * 0.02, 1.0e-12));

  CameraTimeTransferState end = start;
  end.timestamp_sec = 0.02;
  end.position_world_m = forward.position_world_m;
  end.velocity_world_mps = forward.velocity_world_mps;
  const CameraPoseTransferData backward =
    transfer.transferWithSamples(end, imu, 0.0);
  ASSERT_TRUE(backward.valid());
  EXPECT_TRUE(backward.velocity_world_mps.isZero(1.0e-12));
  EXPECT_TRUE(backward.position_world_m.isZero(1.0e-12));
}

TEST(CameraTimeTransfer, ZeroDtDoesNotRequireImuSamples)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 5.0;
  anchor.position_world_m = Eigen::Vector3d(1.0, 2.0, 3.0);

  const CameraPoseTransferData result =
    transfer.transferWithSamples(anchor, {}, 5.0);

  ASSERT_TRUE(result.valid());
  EXPECT_TRUE(result.position_world_m.isApprox(anchor.position_world_m));
  EXPECT_TRUE(result.error_state_transition.isIdentity());
  EXPECT_TRUE(
    result.target_pose_wrt_anchor_state.isApprox(
      result.error_state_transition.topRows<6>()));
  EXPECT_DOUBLE_EQ(result.transition_condition_number, 1.0);
}

TEST(CameraTimeTransfer, ExpiredIntervalReportsMissingLeftBracket)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  const CameraPoseTransferData result = transfer.transferWithSamples(
    anchor,
    {
      sample(0.005, zero, zero),
      sample(0.010, zero, zero),
      sample(0.015, zero, zero)},
    0.015);

  EXPECT_EQ(
    result.status, CameraTimeTransferStatus::kMissingLeftBracket);
  EXPECT_FALSE(result.valid());
}

TEST(CameraTimeTransfer, FutureIntervalReportsMissingRightBracket)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  const CameraPoseTransferData result = transfer.transferWithSamples(
    anchor,
    {
      sample(-0.005, zero, zero),
      sample(0.000, zero, zero),
      sample(0.010, zero, zero)},
    0.015);

  EXPECT_EQ(
    result.status, CameraTimeTransferStatus::kMissingRightBracket);
  EXPECT_FALSE(result.valid());
}

TEST(CameraTimeTransfer, ExactEndpointDoesNotChargeUnusedOutsideGap)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Vector3d force(1.0, -0.5, 0.25);
  const std::vector<CameraTimeTransferImuSample> interval_samples{
    sample(0.000, zero, force),
    sample(0.010, zero, force)};
  std::vector<CameraTimeTransferImuSample> with_outside_gap{
    sample(-0.100, zero, force)};
  with_outside_gap.insert(
    with_outside_gap.end(),
    interval_samples.begin(), interval_samples.end());
  std::vector<CameraTimeTransferImuSample> with_right_outside_gap =
    interval_samples;
  with_right_outside_gap.push_back(
    sample(0.100, zero, force));

  const CameraPoseTransferData baseline =
    transfer.transferWithSamples(anchor, interval_samples, 0.010);
  const CameraPoseTransferData result =
    transfer.transferWithSamples(anchor, with_outside_gap, 0.010);
  const CameraPoseTransferData right_result =
    transfer.transferWithSamples(
      anchor, with_right_outside_gap, 0.010);

  ASSERT_TRUE(baseline.valid());
  ASSERT_TRUE(result.valid());
  ASSERT_TRUE(right_result.valid());
  EXPECT_DOUBLE_EQ(result.maximum_imu_gap_sec, 0.010);
  EXPECT_DOUBLE_EQ(right_result.maximum_imu_gap_sec, 0.010);
  EXPECT_TRUE(result.position_world_m.isApprox(
    baseline.position_world_m, 0.0));
  EXPECT_TRUE(right_result.position_world_m.isApprox(
    baseline.position_world_m, 0.0));
  EXPECT_TRUE(result.error_state_transition.isApprox(
    baseline.error_state_transition, 0.0));
  EXPECT_TRUE(right_result.error_state_transition.isApprox(
    baseline.error_state_transition, 0.0));
  EXPECT_TRUE(result.process_noise_covariance.isApprox(
    baseline.process_noise_covariance, 0.0));
  EXPECT_TRUE(right_result.process_noise_covariance.isApprox(
    baseline.process_noise_covariance, 0.0));
}

TEST(CameraTimeTransfer, MissingCoverageAndLargeGapFailClosed)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  const CameraPoseTransferData missing = transfer.transferWithSamples(
    anchor,
    {
      sample(0.005, zero, zero),
      sample(0.010, zero, zero)},
    0.015);
  EXPECT_EQ(
    missing.status, CameraTimeTransferStatus::kImuCoverageMissing);
  EXPECT_FALSE(missing.valid());
  EXPECT_FALSE(missing.position_world_m.allFinite());

  const CameraPoseTransferData gap = transfer.transferWithSamples(
    anchor,
    {
      sample(-0.001, zero, zero),
      sample(0.000, zero, zero),
      sample(0.030, zero, zero)},
    0.020);
  EXPECT_EQ(gap.status, CameraTimeTransferStatus::kImuGapTooLarge);
  EXPECT_FALSE(gap.valid());
}

TEST(CameraTimeTransfer, EpsilonCoverageAtImuEndpointsIsSafe)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d acceleration(1.0, 0.0, 0.0);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(0.0, Eigen::Vector3d::Zero(), acceleration),
    sample(0.01, Eigen::Vector3d::Zero(), acceleration)};

  constexpr double kEndpointOffsetSec = 0.5e-12;
  const CameraPoseTransferData after_last =
    transfer.transferWithSamples(
      anchor, imu, 0.01 + kEndpointOffsetSec);

  ASSERT_TRUE(after_last.valid());
  EXPECT_TRUE(after_last.position_world_m.isApprox(
    Eigen::Vector3d(
      0.5 * std::pow(0.01 + kEndpointOffsetSec, 2),
      0.0, 0.0),
    1.0e-15));

  CameraTimeTransferState reverse_anchor = anchor;
  reverse_anchor.timestamp_sec = 0.01;
  reverse_anchor.position_world_m =
    Eigen::Vector3d(0.5 * 0.01 * 0.01, 0.0, 0.0);
  reverse_anchor.velocity_world_mps =
    Eigen::Vector3d(0.01, 0.0, 0.0);
  const CameraPoseTransferData before_first =
    transfer.transferWithSamples(
      reverse_anchor, imu, -kEndpointOffsetSec);

  ASSERT_TRUE(before_first.valid());
  EXPECT_NEAR(
    before_first.position_world_m.x(),
    0.5 * kEndpointOffsetSec * kEndpointOffsetSec,
    1.0e-15);
  EXPECT_NEAR(before_first.position_world_m.y(), 0.0, 1.0e-15);
  EXPECT_NEAR(before_first.position_world_m.z(), 0.0, 1.0e-15);
}

TEST(CameraTimeTransfer, OutOfRangeAndUnorderedSamplesFailClosed)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const std::vector<CameraTimeTransferImuSample> covered{
    sample(0.99, zero, zero),
    sample(1.00, zero, zero),
    sample(1.01, zero, zero)};

  EXPECT_EQ(
    transfer.transferWithSamples(anchor, covered, 1.2).status,
    CameraTimeTransferStatus::kTargetOutOfRange);

  const std::vector<CameraTimeTransferImuSample> unordered{
    sample(0.99, zero, zero),
    sample(1.01, zero, zero),
    sample(1.00, zero, zero)};
  EXPECT_EQ(
    transfer.transferWithSamples(anchor, unordered, 1.01).status,
    CameraTimeTransferStatus::kImuNotStrictlyOrdered);
}

TEST(CameraTimeTransfer, DuplicateAndNonFiniteTimestampsFailClosed)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  const std::vector<CameraTimeTransferImuSample> duplicate{
    sample(0.99, zero, zero),
    sample(1.00, zero, zero),
    sample(1.00, zero, zero),
    sample(1.01, zero, zero)};
  EXPECT_EQ(
    transfer.transferWithSamples(anchor, duplicate, 1.01).status,
    CameraTimeTransferStatus::kImuNotStrictlyOrdered);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<CameraTimeTransferImuSample> non_finite{
    sample(0.99, zero, zero),
    sample(nan, zero, zero),
    sample(1.01, zero, zero)};
  EXPECT_EQ(
    transfer.transferWithSamples(anchor, non_finite, 1.01).status,
    CameraTimeTransferStatus::kInvalidInput);
  EXPECT_EQ(
    transfer.transferWithSamples(
      anchor,
      {
        sample(0.99, zero, zero),
        sample(1.01, zero, zero)},
      nan).status,
    CameraTimeTransferStatus::kInvalidInput);
}

TEST(
  CameraTimeTransfer,
  FiniteInputsWhosePropagationOverflowsReportInvalidNumericalResult)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const double huge =
    std::numeric_limits<double>::max() / 4.0;
  const Eigen::Vector3d angular_velocity(huge, huge, huge);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  const CameraPoseTransferData result = transfer.transferWithSamples(
    anchor,
    {
      sample(0.000, angular_velocity, zero),
      sample(0.010, angular_velocity, zero)},
    0.010);

  EXPECT_EQ(
    result.status,
    CameraTimeTransferStatus::kInvalidNumericalResult);
  EXPECT_FALSE(result.valid());
  EXPECT_FALSE(result.validatedValue().has_value());
  EXPECT_FALSE(result.rotation_world_from_imu.allFinite());
  EXPECT_FALSE(result.error_state_transition.allFinite());
  EXPECT_FALSE(result.target_pose_wrt_anchor_state.allFinite());
  EXPECT_FALSE(result.process_noise_covariance.allFinite());
}

TEST(CameraTimeTransfer, RejectsTransitionAboveConfiguredConditionLimit)
{
  CameraTimeTransferConfig config = testConfig();
  config.maximum_transition_condition_number = 1.0;
  CameraTimeTransfer transfer(config);
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

  const CameraPoseTransferData result = transfer.transferWithSamples(
    anchor,
    {
      sample(0.000, zero, zero),
      sample(0.010, zero, zero)},
    0.010);

  EXPECT_EQ(
    result.status,
    CameraTimeTransferStatus::kIllConditionedTransition);
  EXPECT_GT(result.transition_condition_number, 1.0);
  EXPECT_FALSE(result.valid());
  EXPECT_FALSE(result.validatedValue().has_value());
}

TEST(CameraTimeTransfer, RejectsInvalidTransitionConditionConfiguration)
{
  for (const double invalid_limit : {
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), 0.999})
  {
    SCOPED_TRACE(invalid_limit);
    CameraTimeTransferConfig config = testConfig();
    config.maximum_transition_condition_number = invalid_limit;
    CameraTimeTransfer transfer(config);
    CameraTimeTransferState anchor;

    EXPECT_FALSE(transfer.configValid());
    EXPECT_EQ(
      transfer.transferWithSamples(anchor, {}, 0.0).status,
      CameraTimeTransferStatus::kInvalidConfiguration);
  }
}

TEST(CameraTimeTransfer, TransitionAtConfiguredConditionLimitIsAccepted)
{
  CameraTimeTransferConfig baseline_config = testConfig();
  CameraTimeTransfer baseline_transfer(baseline_config);
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(0.000, zero, zero),
    sample(0.010, zero, zero)};
  const CameraPoseTransferData baseline =
    baseline_transfer.transferWithSamples(anchor, imu, 0.010);
  ASSERT_TRUE(baseline.valid());

  CameraTimeTransferConfig exact_config = testConfig();
  exact_config.maximum_transition_condition_number =
    baseline.transition_condition_number;
  CameraTimeTransfer exact_transfer(exact_config);
  const CameraPoseTransferData result =
    exact_transfer.transferWithSamples(anchor, imu, 0.010);

  EXPECT_DOUBLE_EQ(
    result.transition_condition_number,
    baseline.transition_condition_number);
  EXPECT_TRUE(result.valid());
}

TEST(CameraTimeTransfer, SealedTransferBindsTransitionConditionAudit)
{
  CameraPoseTransferData value = validPoseTransferValue();
  ASSERT_TRUE(value.valid());

  value.transition_condition_number += 1.0;

  EXPECT_TRUE(value.internallyValid());
  EXPECT_FALSE(value.valid());
  EXPECT_FALSE(value.validatedValue().has_value());
}

TEST(
  CameraTimeTransfer,
  TypedDirectionAndUncertaintySemanticsRejectTamperingAndBindSeal)
{
  const CameraPoseTransferData sealed = validPoseTransferValue();
  ASSERT_TRUE(sealed.valid());
  ASSERT_TRUE(sealed.validatedValue().has_value());
  ASSERT_EQ(
    sealed.direction,
    livo::CameraTimeTransferDirection::kForward);
  ASSERT_EQ(
    sealed.uncertainty_semantics,
    livo::CameraTimeTransferUncertaintySemantics::
    kForwardIndependentProcessNoise);

  CameraPoseTransferData mismatched_semantics = sealed;
  mismatched_semantics.uncertainty_semantics =
    livo::CameraTimeTransferUncertaintySemantics::
    kNoProcessNoise;
  EXPECT_FALSE(mismatched_semantics.internallyValid());
  EXPECT_FALSE(mismatched_semantics.valid());
  EXPECT_FALSE(
    mismatched_semantics.validatedValue().has_value());

  CameraPoseTransferData identity_with_process_noise = sealed;
  identity_with_process_noise.target_timestamp_sec =
    identity_with_process_noise.anchor_timestamp_sec;
  identity_with_process_noise.delta_time_sec = 0.0;
  identity_with_process_noise.direction =
    livo::CameraTimeTransferDirection::kIdentity;
  identity_with_process_noise.uncertainty_semantics =
    livo::CameraTimeTransferUncertaintySemantics::
    kBackwardCorrelatedTransportApproximation;
  EXPECT_FALSE(identity_with_process_noise.internallyValid());
  EXPECT_FALSE(identity_with_process_noise.valid());
  EXPECT_FALSE(
    identity_with_process_noise.validatedValue().has_value());

  CameraPoseTransferData coherent_raw_tamper = sealed;
  std::swap(
    coherent_raw_tamper.anchor_timestamp_sec,
    coherent_raw_tamper.target_timestamp_sec);
  coherent_raw_tamper.delta_time_sec =
    coherent_raw_tamper.target_timestamp_sec -
    coherent_raw_tamper.anchor_timestamp_sec;
  coherent_raw_tamper.direction =
    livo::CameraTimeTransferDirection::kBackward;
  coherent_raw_tamper.uncertainty_semantics =
    livo::CameraTimeTransferUncertaintySemantics::
    kBackwardCorrelatedTransportApproximation;
  EXPECT_TRUE(coherent_raw_tamper.internallyValid());
  EXPECT_FALSE(coherent_raw_tamper.valid());
  EXPECT_FALSE(
    coherent_raw_tamper.validatedValue().has_value());
}

TEST(
  CameraTimeTransfer,
  FullPoseJacobianMatchesFiniteDifferenceForPositiveDt)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  anchor.rotation_world_from_imu =
    Eigen::AngleAxisd(
      0.3, Eigen::Vector3d(0.2, -0.4, 0.8).normalized())
    .toRotationMatrix();
  anchor.position_world_m = Eigen::Vector3d(2.0, -1.0, 0.5);
  anchor.velocity_world_mps = Eigen::Vector3d(0.3, 0.1, -0.2);
  anchor.gyro_bias_radps = Eigen::Vector3d(0.01, -0.02, 0.005);
  anchor.accel_bias_mps2 = Eigen::Vector3d(0.02, -0.01, 0.03);
  anchor.gravity_world_mps2 = Eigen::Vector3d(0.0, 0.0, -9.81);
  const Eigen::Vector3d omega(0.2, -0.1, 0.4);
  const Eigen::Vector3d force(1.0, 0.5, -0.3);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(-0.01, omega, force),
    sample(0.00, omega, force),
    sample(0.01, omega, force),
    sample(0.02, omega, force),
    sample(0.03, omega, force),
    sample(0.04, omega, force),
    sample(0.05, omega, force)};

  expectFullPoseJacobianMatchesFiniteDifference(
    transfer, anchor, imu, 0.04);
}

TEST(
  CameraTimeTransfer,
  FullPoseJacobianMatchesFiniteDifferenceForNegativeDt)
{
  CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.04;
  anchor.rotation_world_from_imu =
    Eigen::AngleAxisd(
      -0.2, Eigen::Vector3d(-0.3, 0.6, 0.2).normalized())
    .toRotationMatrix();
  anchor.position_world_m = Eigen::Vector3d(-1.0, 0.5, 2.0);
  anchor.velocity_world_mps = Eigen::Vector3d(-0.2, 0.4, 0.1);
  anchor.gyro_bias_radps = Eigen::Vector3d(-0.015, 0.01, 0.02);
  anchor.accel_bias_mps2 = Eigen::Vector3d(0.01, 0.03, -0.02);
  anchor.gravity_world_mps2 = Eigen::Vector3d(0.1, -0.2, -9.7);
  const Eigen::Vector3d omega(-0.1, 0.3, 0.25);
  const Eigen::Vector3d force(0.6, -0.8, 0.4);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(-0.01, omega, force),
    sample(0.00, omega, force),
    sample(0.01, omega, force),
    sample(0.02, omega, force),
    sample(0.03, omega, force),
    sample(0.04, omega, force),
    sample(0.05, omega, force)};

  expectFullPoseJacobianMatchesFiniteDifference(
    transfer, anchor, imu, 0.0);
}

TEST(
  CameraTimeTransfer,
  BackwardProcessNoiseEqualsTransportedForwardDisturbanceAt5_10_20_40ms)
{
  CameraTimeTransferConfig config = testConfig();
  config.max_imu_gap_sec = 0.006;
  config.gyro_noise_stddev_radps = 0.013;
  config.accel_noise_stddev_mps2 = 0.12;
  config.gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 2.0e-4;
  config.accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 1.5e-3;
  const CameraTimeTransfer transfer(config);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.040;
  anchor.rotation_world_from_imu =
    Eigen::AngleAxisd(
      0.37, Eigen::Vector3d(0.4, -0.2, 0.7).normalized())
    .toRotationMatrix();
  anchor.position_world_m = Eigen::Vector3d(1.2, -0.8, 0.45);
  anchor.velocity_world_mps = Eigen::Vector3d(0.7, -0.3, 0.2);
  anchor.gyro_bias_radps = Eigen::Vector3d(0.012, -0.018, 0.007);
  anchor.accel_bias_mps2 = Eigen::Vector3d(0.08, -0.05, 0.11);
  anchor.gravity_world_mps2 = Eigen::Vector3d(0.03, -0.04, -9.79);

  std::vector<CameraTimeTransferImuSample> imu;
  imu.reserve(9U);
  for (std::size_t index = 0U; index <= 8U; ++index)
  {
    const double timestamp_sec =
      0.005 * static_cast<double>(index);
    const double phase = timestamp_sec / anchor.timestamp_sec;
    const Eigen::Vector3d measured_angular_velocity(
      0.31 + 0.17 * phase + 0.03 * phase * phase,
      -0.24 + 0.11 * phase - 0.02 * phase * phase,
      0.19 - 0.07 * phase + 0.04 * phase * phase);
    const Eigen::Vector3d measured_specific_force(
      1.20 + 0.40 * phase - 0.10 * phase * phase,
      -0.70 + 0.20 * phase + 0.08 * phase * phase,
      9.40 - 0.30 * phase + 0.12 * phase * phase);
    imu.push_back(sample(
      timestamp_sec,
      measured_angular_velocity,
      measured_specific_force));
  }

  using ErrorStateMatrix =
    livo::CameraTimeTransferErrorStateMatrix;
  const ErrorStateMatrix identity = ErrorStateMatrix::Identity();
  constexpr std::array<double, 4> kWindowsSec{{
      0.005, 0.010, 0.020, 0.040}};
  constexpr double kNominalRoundTripTolerance = 1.0e-12;
  constexpr double kTransitionTolerance = 1.0e-10;
  constexpr double kSolveResidualTolerance = 1.0e-11;

  for (const double window_sec : kWindowsSec)
  {
    const double past_timestamp_sec =
      anchor.timestamp_sec - window_sec;
    const CameraPoseTransferData backward =
      transfer.transferWithSamples(
        anchor, imu, past_timestamp_sec);
    ASSERT_TRUE(backward.valid()) << "window_sec=" << window_sec;

    CameraTimeTransferState past;
    past.timestamp_sec = past_timestamp_sec;
    past.rotation_world_from_imu =
      backward.rotation_world_from_imu;
    past.position_world_m = backward.position_world_m;
    past.velocity_world_mps = backward.velocity_world_mps;
    past.gyro_bias_radps = anchor.gyro_bias_radps;
    past.accel_bias_mps2 = anchor.accel_bias_mps2;
    past.gravity_world_mps2 = anchor.gravity_world_mps2;
    past.timestamp_stddev_sec = anchor.timestamp_stddev_sec;

    const CameraPoseTransferData forward =
      transfer.transferWithSamples(
        past, imu, anchor.timestamp_sec);
    ASSERT_TRUE(forward.valid()) << "window_sec=" << window_sec;

    const Eigen::JacobiSVD<ErrorStateMatrix> forward_svd(
      forward.error_state_transition,
      Eigen::ComputeFullU | Eigen::ComputeFullV);
    ASSERT_EQ(forward_svd.info(), Eigen::Success);
    const auto & singular_values = forward_svd.singularValues();
    ASSERT_GT(singular_values.minCoeff(), 0.0);
    const double condition_number =
      singular_values.maxCoeff() / singular_values.minCoeff();
    const ErrorStateMatrix anchor_to_past =
      forward_svd.solve(identity);
    const double forward_left_solve_residual =
      (forward.error_state_transition * anchor_to_past -
       identity).norm();
    const double forward_right_solve_residual =
      (anchor_to_past * forward.error_state_transition -
       identity).norm();

    const double backward_transition_error =
      (backward.error_state_transition - anchor_to_past).norm();
    const double forward_after_backward_error =
      (forward.error_state_transition *
       backward.error_state_transition - identity).norm();
    const double backward_after_forward_error =
      (backward.error_state_transition *
       forward.error_state_transition - identity).norm();

    const ErrorStateMatrix transported_forward_disturbance_covariance =
      anchor_to_past *
      forward.process_noise_covariance *
      anchor_to_past.transpose();
    const ErrorStateMatrix &
      backward_correlated_transport_approximation =
      backward.process_noise_covariance;
    const double disturbance_error =
      (backward_correlated_transport_approximation -
       transported_forward_disturbance_covariance).norm();
    const double backward_disturbance_norm =
      backward_correlated_transport_approximation.norm();
    const double transported_disturbance_norm =
      transported_forward_disturbance_covariance.norm();
    const double relative_disturbance_error =
      2.0 * disturbance_error /
      std::max(
        backward_disturbance_norm +
        transported_disturbance_norm,
        std::numeric_limits<double>::min());

    const double rotation_round_trip_error =
      logSo3(
        anchor.rotation_world_from_imu.transpose() *
        forward.rotation_world_from_imu).norm();
    const double position_round_trip_error =
      (forward.position_world_m - anchor.position_world_m).norm();
    const double velocity_round_trip_error =
      (forward.velocity_world_mps - anchor.velocity_world_mps).norm();

    std::ostringstream audit;
    audit << std::scientific << std::setprecision(17)
          << "window_sec=" << window_sec
          << " condition=" << condition_number
          << " solve_left=" << forward_left_solve_residual
          << " solve_right=" << forward_right_solve_residual
          << " phi_vs_solve=" << backward_transition_error
          << " forward_backward=" << forward_after_backward_error
          << " backward_forward=" << backward_after_forward_error
          << " disturbance_abs=" << disturbance_error
          << " disturbance_rel=" << relative_disturbance_error
          << " disturbance_backward_norm="
          << backward_disturbance_norm
          << " disturbance_transported_norm="
          << transported_disturbance_norm
          << " nominal_rotation=" << rotation_round_trip_error
          << " nominal_position=" << position_round_trip_error
          << " nominal_velocity=" << velocity_round_trip_error;
    SCOPED_TRACE(audit.str());
    std::cout << "[backward-transport-oracle] "
              << audit.str() << '\n';

    EXPECT_TRUE(std::isfinite(condition_number));
    EXPECT_LE(condition_number, 2.0);
    EXPECT_TRUE(anchor_to_past.allFinite());
    EXPECT_TRUE(
      backward_correlated_transport_approximation.allFinite());
    EXPECT_TRUE(
      transported_forward_disturbance_covariance.allFinite());
    EXPECT_TRUE(
      backward_correlated_transport_approximation.isApprox(
        backward_correlated_transport_approximation.transpose(),
        1.0e-12));
    EXPECT_TRUE(
      transported_forward_disturbance_covariance.isApprox(
        transported_forward_disturbance_covariance.transpose(),
        1.0e-12));
    const Eigen::SelfAdjointEigenSolver<ErrorStateMatrix>
      backward_disturbance_psd_solver(
        backward_correlated_transport_approximation);
    ASSERT_EQ(
      backward_disturbance_psd_solver.info(),
      Eigen::Success);
    EXPECT_GE(
      backward_disturbance_psd_solver.eigenvalues().minCoeff(),
      -1.0e-12);
    const Eigen::SelfAdjointEigenSolver<ErrorStateMatrix>
      disturbance_psd_solver(
        transported_forward_disturbance_covariance);
    ASSERT_EQ(disturbance_psd_solver.info(), Eigen::Success);
    EXPECT_GE(
      disturbance_psd_solver.eigenvalues().minCoeff(),
      -1.0e-12);

    EXPECT_LE(
      forward_left_solve_residual,
      kSolveResidualTolerance);
    EXPECT_LE(
      forward_right_solve_residual,
      kSolveResidualTolerance);
    EXPECT_LE(
      rotation_round_trip_error,
      kNominalRoundTripTolerance);
    EXPECT_LE(
      position_round_trip_error,
      kNominalRoundTripTolerance);
    EXPECT_LE(
      velocity_round_trip_error,
      kNominalRoundTripTolerance);
    EXPECT_LE(
      backward_transition_error,
      kTransitionTolerance);
    EXPECT_LE(
      forward_after_backward_error,
      kTransitionTolerance);
    EXPECT_LE(
      backward_after_forward_error,
      kTransitionTolerance);
    EXPECT_LE(
      relative_disturbance_error,
      1.0e-8);
  }
}

TEST(
  CameraTimeTransfer,
  BackwardNominalMatchesIndependentQuaternionRk4At5_10_20_40ms)
{
  CameraTimeTransferConfig config = testConfig();
  config.max_imu_gap_sec = 0.001;
  config.gyro_noise_stddev_radps = 0.0;
  config.accel_noise_stddev_mps2 = 0.0;
  config.gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 0.0;
  config.accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 0.0;
  const CameraTimeTransfer transfer(config);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.040;
  anchor.rotation_world_from_imu =
    Eigen::AngleAxisd(
      0.37, Eigen::Vector3d(0.4, -0.2, 0.7).normalized())
    .toRotationMatrix();
  anchor.position_world_m = Eigen::Vector3d(1.2, -0.8, 0.45);
  anchor.velocity_world_mps = Eigen::Vector3d(0.7, -0.3, 0.2);
  anchor.gyro_bias_radps = Eigen::Vector3d(0.012, -0.018, 0.007);
  anchor.accel_bias_mps2 = Eigen::Vector3d(0.08, -0.05, 0.11);
  anchor.gravity_world_mps2 = Eigen::Vector3d(0.03, -0.04, -9.79);

  std::vector<CameraTimeTransferImuSample> imu;
  imu.reserve(81U);
  for (std::size_t index = 0U; index <= 80U; ++index)
  {
    const double timestamp_sec =
      0.0005 * static_cast<double>(index);
    imu.push_back(sample(
      timestamp_sec,
      independentOracleAngularVelocity(timestamp_sec),
      independentOracleSpecificForce(timestamp_sec)));
  }

  constexpr std::array<double, 4> kWindowsSec{{
      0.005, 0.010, 0.020, 0.040}};
  constexpr double kCoarseStepSec = 2.0e-4;
  constexpr double kMediumStepSec = 1.0e-4;
  constexpr double kFineStepSec = 5.0e-5;
  for (const double window_sec : kWindowsSec)
  {
    const double target_timestamp_sec =
      anchor.timestamp_sec - window_sec;
    const CameraPoseTransferData production =
      transfer.transferWithSamples(
        anchor, imu, target_timestamp_sec);
    ASSERT_TRUE(production.valid())
      << "window_sec=" << window_sec;

    const CameraTimeTransferState coarse =
      integrateIndependentQuaternionRk4(
        anchor, target_timestamp_sec, kCoarseStepSec);
    const CameraTimeTransferState medium =
      integrateIndependentQuaternionRk4(
        anchor, target_timestamp_sec, kMediumStepSec);
    const CameraTimeTransferState fine =
      integrateIndependentQuaternionRk4(
        anchor, target_timestamp_sec, kFineStepSec);

    const double coarse_to_medium_rotation =
      logSo3(
        medium.rotation_world_from_imu.transpose() *
        coarse.rotation_world_from_imu).norm();
    const double medium_to_fine_rotation =
      logSo3(
        fine.rotation_world_from_imu.transpose() *
        medium.rotation_world_from_imu).norm();
    const double coarse_to_medium_position =
      (coarse.position_world_m -
       medium.position_world_m).norm();
    const double medium_to_fine_position =
      (medium.position_world_m -
       fine.position_world_m).norm();
    const double coarse_to_medium_velocity =
      (coarse.velocity_world_mps -
       medium.velocity_world_mps).norm();
    const double medium_to_fine_velocity =
      (medium.velocity_world_mps -
       fine.velocity_world_mps).norm();

    const double production_rotation_error =
      logSo3(
        fine.rotation_world_from_imu.transpose() *
        production.rotation_world_from_imu).norm();
    const double production_position_error =
      (fine.position_world_m -
       production.position_world_m).norm();
    const double production_velocity_error =
      (fine.velocity_world_mps -
       production.velocity_world_mps).norm();

    std::ostringstream audit;
    audit << std::scientific << std::setprecision(17)
          << "window_sec=" << window_sec
          << " rk4_coarse_medium_rotation="
          << coarse_to_medium_rotation
          << " rk4_medium_fine_rotation="
          << medium_to_fine_rotation
          << " rk4_coarse_medium_position="
          << coarse_to_medium_position
          << " rk4_medium_fine_position="
          << medium_to_fine_position
          << " rk4_coarse_medium_velocity="
          << coarse_to_medium_velocity
          << " rk4_medium_fine_velocity="
          << medium_to_fine_velocity
          << " production_rotation="
          << production_rotation_error
          << " production_position="
          << production_position_error
          << " production_velocity="
          << production_velocity_error;
    SCOPED_TRACE(audit.str());
    std::cout << "[quaternion-rk4-oracle] "
              << audit.str() << '\n';

    EXPECT_LE(
      medium_to_fine_rotation,
      coarse_to_medium_rotation + 1.0e-13);
    EXPECT_LE(
      medium_to_fine_position,
      coarse_to_medium_position + 1.0e-13);
    EXPECT_LE(
      medium_to_fine_velocity,
      coarse_to_medium_velocity + 1.0e-13);
    EXPECT_LE(medium_to_fine_rotation, 1.0e-10);
    EXPECT_LE(medium_to_fine_position, 1.0e-10);
    EXPECT_LE(medium_to_fine_velocity, 1.0e-9);
    EXPECT_LE(production_rotation_error, 3.0e-6);
    EXPECT_LE(production_position_error, 3.0e-6);
    EXPECT_LE(production_velocity_error, 8.0e-5);
  }
}

TEST(
  CameraTimeTransfer,
  ZeroNoiseProducesExactlyZeroProcessCovarianceInBothDirections)
{
  CameraTimeTransferConfig config = testConfig();
  config.gyro_noise_stddev_radps = 0.0;
  config.accel_noise_stddev_mps2 = 0.0;
  config.gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 0.0;
  config.accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 0.0;
  const auto density =
    config.validatedContinuousNoiseDensity();
  ASSERT_TRUE(density.has_value());
  EXPECT_DOUBLE_EQ(
    density->gyro_measurement_rad_per_sqrt_sec, 0.0);
  EXPECT_DOUBLE_EQ(
    density->accel_measurement_mps_per_sqrt_sec, 0.0);
  EXPECT_DOUBLE_EQ(
    density->gyro_bias_random_walk_radps_per_sqrt_sec, 0.0);
  EXPECT_DOUBLE_EQ(
    density->accel_bias_random_walk_mps2_per_sqrt_sec, 0.0);
  const CameraTimeTransfer transfer(config);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(
      0.00, Eigen::Vector3d(0.2, -0.1, 0.3),
      Eigen::Vector3d(1.0, -0.4, 9.5)),
    sample(
      0.01, Eigen::Vector3d(0.3, -0.2, 0.1),
      Eigen::Vector3d(1.2, -0.2, 9.3)),
    sample(
      0.02, Eigen::Vector3d(0.1, -0.3, 0.2),
      Eigen::Vector3d(0.8, -0.1, 9.7)),
    sample(
      0.03, Eigen::Vector3d(0.4, -0.1, 0.3),
      Eigen::Vector3d(1.1, -0.5, 9.4)),
    sample(
      0.04, Eigen::Vector3d(0.2, -0.4, 0.1),
      Eigen::Vector3d(0.9, -0.3, 9.6))};

  CameraTimeTransferState forward_anchor;
  forward_anchor.timestamp_sec = 0.0;
  const CameraPoseTransferData forward =
    transfer.transferWithSamples(
      forward_anchor, imu, 0.04);

  CameraTimeTransferState backward_anchor;
  backward_anchor.timestamp_sec = 0.04;
  backward_anchor.rotation_world_from_imu =
    Eigen::AngleAxisd(
      0.2, Eigen::Vector3d(0.2, 0.5, -0.3).normalized())
    .toRotationMatrix();
  backward_anchor.position_world_m =
    Eigen::Vector3d(0.5, -0.2, 1.0);
  backward_anchor.velocity_world_mps =
    Eigen::Vector3d(0.3, -0.1, 0.2);
  const CameraPoseTransferData backward =
    transfer.transferWithSamples(
      backward_anchor, imu, 0.0);

  ASSERT_TRUE(forward.valid());
  ASSERT_TRUE(backward.valid());
  EXPECT_TRUE(
    forward.process_noise_covariance.isZero(0.0));
  EXPECT_TRUE(
    forward.pose_transfer_covariance.isZero(0.0));
  EXPECT_TRUE(
    backward.process_noise_covariance.isZero(0.0));
  EXPECT_TRUE(
    backward.pose_transfer_covariance.isZero(0.0));
}

TEST(
  CameraTimeTransfer,
  ContinuousNoiseDensityAmplitudeScalingSquaresProcessCovariance)
{
  CameraTimeTransferConfig base_config = testConfig();
  base_config.gyro_noise_stddev_radps = 0.01;
  base_config.accel_noise_stddev_mps2 = 0.10;
  base_config.gyro_bias_random_walk_stddev_radps_per_sqrt_sec =
    1.0e-4;
  base_config.accel_bias_random_walk_stddev_mps2_per_sqrt_sec =
    1.0e-3;
  constexpr double kAmplitudeScale = 3.0;
  constexpr double kCovarianceScale =
    kAmplitudeScale * kAmplitudeScale;
  CameraTimeTransferConfig scaled_config = base_config;
  scaled_config.gyro_noise_stddev_radps *= kAmplitudeScale;
  scaled_config.accel_noise_stddev_mps2 *= kAmplitudeScale;
  scaled_config.gyro_bias_random_walk_stddev_radps_per_sqrt_sec *=
    kAmplitudeScale;
  scaled_config.accel_bias_random_walk_stddev_mps2_per_sqrt_sec *=
    kAmplitudeScale;
  const CameraTimeTransfer base_transfer(base_config);
  const CameraTimeTransfer scaled_transfer(scaled_config);

  const std::vector<CameraTimeTransferImuSample> imu{
    sample(
      0.00, Eigen::Vector3d(0.2, -0.1, 0.3),
      Eigen::Vector3d(1.0, -0.4, 9.5)),
    sample(
      0.01, Eigen::Vector3d(0.3, -0.2, 0.1),
      Eigen::Vector3d(1.2, -0.2, 9.3)),
    sample(
      0.02, Eigen::Vector3d(0.1, -0.3, 0.2),
      Eigen::Vector3d(0.8, -0.1, 9.7))};
  CameraTimeTransferState forward_anchor;
  forward_anchor.timestamp_sec = 0.0;
  CameraTimeTransferState backward_anchor;
  backward_anchor.timestamp_sec = 0.02;
  backward_anchor.rotation_world_from_imu =
    Eigen::AngleAxisd(
      0.2, Eigen::Vector3d(0.2, 0.5, -0.3).normalized())
    .toRotationMatrix();

  const CameraPoseTransferData base_forward =
    base_transfer.transferWithSamples(
      forward_anchor, imu, 0.02);
  const CameraPoseTransferData scaled_forward =
    scaled_transfer.transferWithSamples(
      forward_anchor, imu, 0.02);
  const CameraPoseTransferData base_backward =
    base_transfer.transferWithSamples(
      backward_anchor, imu, 0.0);
  const CameraPoseTransferData scaled_backward =
    scaled_transfer.transferWithSamples(
      backward_anchor, imu, 0.0);

  ASSERT_TRUE(base_forward.valid());
  ASSERT_TRUE(scaled_forward.valid());
  ASSERT_TRUE(base_backward.valid());
  ASSERT_TRUE(scaled_backward.valid());
  EXPECT_GT(base_forward.process_noise_covariance.norm(), 0.0);
  EXPECT_GT(base_backward.process_noise_covariance.norm(), 0.0);
  EXPECT_TRUE(
    scaled_forward.process_noise_covariance.isApprox(
      kCovarianceScale *
      base_forward.process_noise_covariance,
      1.0e-12));
  EXPECT_TRUE(
    scaled_backward.process_noise_covariance.isApprox(
      kCovarianceScale *
      base_backward.process_noise_covariance,
      1.0e-12));
}

TEST(
  CameraTimeTransfer,
  ReportsTypedDirectionAndUncertaintySemantics)
{
  const CameraTimeTransfer transfer(testConfig());
  CameraTimeTransferConfig zero_noise_config = testConfig();
  zero_noise_config.gyro_noise_stddev_radps = 0.0;
  zero_noise_config.accel_noise_stddev_mps2 = 0.0;
  zero_noise_config.
    gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 0.0;
  zero_noise_config.
    accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 0.0;
  const CameraTimeTransfer zero_noise_transfer(zero_noise_config);
  const Eigen::Vector3d omega(0.2, -0.1, 0.3);
  const Eigen::Vector3d force(1.0, -0.4, 9.5);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(0.00, omega, force),
    sample(0.01, omega, force),
    sample(0.02, omega, force)};
  CameraTimeTransferState forward_anchor;
  forward_anchor.timestamp_sec = 0.0;
  CameraTimeTransferState backward_anchor;
  backward_anchor.timestamp_sec = 0.02;

  const CameraPoseTransferData identity =
    transfer.transferWithSamples(
      forward_anchor, {}, 0.0);
  const CameraPoseTransferData forward =
    transfer.transferWithSamples(
      forward_anchor, imu, 0.02);
  const CameraPoseTransferData backward =
    transfer.transferWithSamples(
      backward_anchor, imu, 0.0);
  const CameraPoseTransferData zero_noise_forward =
    zero_noise_transfer.transferWithSamples(
      forward_anchor, imu, 0.02);
  const CameraPoseTransferData zero_noise_backward =
    zero_noise_transfer.transferWithSamples(
      backward_anchor, imu, 0.0);

  ASSERT_TRUE(identity.valid());
  ASSERT_TRUE(forward.valid());
  ASSERT_TRUE(backward.valid());
  ASSERT_TRUE(zero_noise_forward.valid());
  ASSERT_TRUE(zero_noise_backward.valid());
  EXPECT_EQ(
    identity.direction,
    livo::CameraTimeTransferDirection::kIdentity);
  EXPECT_EQ(
    identity.uncertainty_semantics,
    livo::CameraTimeTransferUncertaintySemantics::
    kNoProcessNoise);
  EXPECT_EQ(
    forward.direction,
    livo::CameraTimeTransferDirection::kForward);
  EXPECT_EQ(
    forward.uncertainty_semantics,
    livo::CameraTimeTransferUncertaintySemantics::
    kForwardIndependentProcessNoise);
  EXPECT_EQ(
    backward.direction,
    livo::CameraTimeTransferDirection::kBackward);
  EXPECT_EQ(
    backward.uncertainty_semantics,
    livo::CameraTimeTransferUncertaintySemantics::
    kBackwardCorrelatedTransportApproximation);
  EXPECT_EQ(
    zero_noise_forward.direction,
    livo::CameraTimeTransferDirection::kForward);
  EXPECT_EQ(
    zero_noise_forward.uncertainty_semantics,
    livo::CameraTimeTransferUncertaintySemantics::
    kNoProcessNoise);
  EXPECT_EQ(
    zero_noise_backward.direction,
    livo::CameraTimeTransferDirection::kBackward);
  EXPECT_EQ(
    zero_noise_backward.uncertainty_semantics,
    livo::CameraTimeTransferUncertaintySemantics::
    kNoProcessNoise);
}

TEST(CameraTimeTransfer, BiasRandomWalkGrowsFullAndPoseProcessNoise)
{
  CameraTimeTransferConfig zero_noise_config = testConfig();
  zero_noise_config.gyro_noise_stddev_radps = 0.0;
  zero_noise_config.accel_noise_stddev_mps2 = 0.0;
  zero_noise_config.
    gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 0.0;
  zero_noise_config.
    accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 0.0;
  CameraTimeTransferConfig random_walk_config = zero_noise_config;
  random_walk_config.
    gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 0.1;
  random_walk_config.
    accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 0.2;
  const CameraTimeTransfer zero_noise(zero_noise_config);
  const CameraTimeTransfer random_walk(random_walk_config);
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d omega(0.2, -0.1, 0.3);
  const Eigen::Vector3d force(1.0, 0.5, -0.2);
  const std::vector<CameraTimeTransferImuSample> imu{
    sample(-0.01, omega, force),
    sample(0.00, omega, force),
    sample(0.01, omega, force),
    sample(0.02, omega, force),
    sample(0.03, omega, force),
    sample(0.04, omega, force),
    sample(0.05, omega, force)};

  const CameraPoseTransferData none =
    zero_noise.transferWithSamples(anchor, imu, 0.04);
  const CameraPoseTransferData short_transfer =
    random_walk.transferWithSamples(anchor, imu, 0.02);
  const CameraPoseTransferData long_transfer =
    random_walk.transferWithSamples(anchor, imu, 0.04);
  ASSERT_TRUE(none.valid());
  ASSERT_TRUE(short_transfer.valid());
  ASSERT_TRUE(long_transfer.valid());

  EXPECT_TRUE(none.process_noise_covariance.isZero(1.0e-15));
  EXPECT_GT(
    (long_transfer.process_noise_covariance.block<3, 3>(
      livo::kCameraTimeTransferGyroBiasOffset,
      livo::kCameraTimeTransferGyroBiasOffset).trace()),
    0.0);
  EXPECT_GT(
    (long_transfer.process_noise_covariance.block<3, 3>(
      livo::kCameraTimeTransferAccelBiasOffset,
      livo::kCameraTimeTransferAccelBiasOffset).trace()),
    0.0);
  EXPECT_GT(
    (long_transfer.pose_transfer_covariance.block<3, 3>(0, 0).trace()),
    0.0);
  EXPECT_GT(
    (long_transfer.pose_transfer_covariance.block<3, 3>(3, 3).trace()),
    0.0);
  EXPECT_GT(
    long_transfer.pose_transfer_covariance.trace(),
    short_transfer.pose_transfer_covariance.trace());
  EXPECT_TRUE(
    long_transfer.pose_transfer_covariance.isApprox(
      long_transfer.process_noise_covariance.topLeftCorner<6, 6>(),
      1.0e-15));
  EXPECT_TRUE(
    long_transfer.process_noise_covariance.isApprox(
      long_transfer.process_noise_covariance.transpose(),
      1.0e-15));
  const Eigen::SelfAdjointEigenSolver<
    livo::CameraTimeTransferErrorStateMatrix> solver(
      long_transfer.process_noise_covariance);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GE(solver.eigenvalues().minCoeff(), -1.0e-12);
}

TEST(CameraTimeTransfer, AppendDuplicateTimestampStartsNewEpoch)
{
  CameraTimeTransfer transfer(testConfig());
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);
  EXPECT_EQ(transfer.epoch(), 2U);
  EXPECT_EQ(transfer.bufferedSamples(), 1U);
}

TEST(CameraTimeTransfer, AppendLargeForwardJumpStartsNewEpoch)
{
  CameraTimeTransfer transfer(testConfig());
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.200, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);
  EXPECT_EQ(transfer.epoch(), 2U);
  EXPECT_EQ(transfer.bufferedSamples(), 1U);
}

TEST(CameraTimeTransfer, OrdinaryLargeGapStaysInEpochButQueryRejectsGap)
{
  CameraTimeTransfer transfer(testConfig());
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.030, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(transfer.epoch(), 1U);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.000;
  EXPECT_EQ(
    transfer.transfer(anchor, 1.030).status,
    CameraTimeTransferStatus::kImuGapTooLarge);
}

TEST(CameraTimeTransfer, BufferedRingStartsNewEpochOnClockJump)
{
  CameraTimeTransfer transfer(testConfig());
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.005, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.010, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.005;
  const CameraPoseTransferData result =
    transfer.transfer(anchor, 1.008);
  ASSERT_TRUE(result.valid());
  EXPECT_EQ(result.imu_epoch, 1U);
  EXPECT_NEAR(result.maximum_imu_gap_sec, 0.005, 1.0e-12);
  EXPECT_TRUE(result.pose_transfer_covariance.allFinite());
  EXPECT_GE(
    result.pose_transfer_covariance.diagonal().minCoeff(), 0.0);

  EXPECT_EQ(
    transfer.appendImu(sample(0.500, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);
  EXPECT_EQ(transfer.epoch(), 2U);
  EXPECT_EQ(transfer.bufferedSamples(), 1U);
  EXPECT_EQ(
    transfer.transfer(anchor, 1.008).status,
    CameraTimeTransferStatus::kMissingRightBracket);
}

TEST(CameraTimeTransfer, QueryAcrossRecordedEpochBoundaryReportsCrossEpoch)
{
  CameraTimeTransferConfig config = testConfig();
  config.max_abs_transfer_sec = 0.5;
  CameraTimeTransfer transfer(config);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.010, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.200, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.010;
  const CameraPoseTransferData result =
    transfer.transfer(anchor, 1.200);

  EXPECT_EQ(
    result.status, CameraTimeTransferStatus::kCrossesImuEpoch);
  EXPECT_FALSE(result.valid());
  EXPECT_EQ(result.imu_epoch, transfer.epoch());
}

TEST(CameraTimeTransfer, LatestEpochBoundaryReplacesOlderBoundary)
{
  CameraTimeTransferConfig config = testConfig();
  config.max_abs_transfer_sec = 1.0;
  CameraTimeTransfer transfer(config);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.010, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.200, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);
  EXPECT_EQ(
    transfer.appendImu(sample(1.210, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.400, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);

  CameraTimeTransferState old_anchor;
  old_anchor.timestamp_sec = 1.010;
  EXPECT_EQ(
    transfer.transfer(old_anchor, 1.200).status,
    CameraTimeTransferStatus::kMissingLeftBracket);

  CameraTimeTransferState latest_anchor;
  latest_anchor.timestamp_sec = 1.210;
  EXPECT_EQ(
    transfer.transfer(latest_anchor, 1.400).status,
    CameraTimeTransferStatus::kCrossesImuEpoch);
}

TEST(CameraTimeTransfer, RejectedAppendPreservesRecordedEpochBoundary)
{
  CameraTimeTransferConfig config = testConfig();
  config.max_abs_transfer_sec = 0.5;
  CameraTimeTransfer transfer(config);
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.010, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.200, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);
  const std::uint64_t epoch_before_rejection = transfer.epoch();
  const std::size_t samples_before_rejection =
    transfer.bufferedSamples();

  EXPECT_EQ(
    transfer.appendImu(sample(
      std::numeric_limits<double>::quiet_NaN(), zero, zero)),
    CameraTimeTransferAppendStatus::kInvalidSample);
  EXPECT_EQ(transfer.epoch(), epoch_before_rejection);
  EXPECT_EQ(
    transfer.bufferedSamples(), samples_before_rejection);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.010;
  EXPECT_EQ(
    transfer.transfer(anchor, 1.200).status,
    CameraTimeTransferStatus::kCrossesImuEpoch);
}

TEST(CameraTimeTransfer, ZeroDurationAtEpochTimestampRemainsIdentity)
{
  CameraTimeTransfer transfer(testConfig());
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kAccepted);
  EXPECT_EQ(
    transfer.appendImu(sample(1.000, zero, zero)),
    CameraTimeTransferAppendStatus::kNewEpoch);

  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.000;
  anchor.position_world_m = Eigen::Vector3d(1.0, 2.0, 3.0);
  const CameraPoseTransferData result =
    transfer.transfer(anchor, 1.000);

  EXPECT_EQ(result.status, CameraTimeTransferStatus::kValid);
  ASSERT_TRUE(result.valid());
  EXPECT_TRUE(result.position_world_m.isApprox(
    anchor.position_world_m, 0.0));
  EXPECT_TRUE(result.error_state_transition.isIdentity());
  EXPECT_DOUBLE_EQ(result.transition_condition_number, 1.0);
}

TEST(
  CameraTimeTransfer,
  RejectsTransferDurationWhoseNoiseStepCountCannotFitSizeT)
{
  CameraTimeTransferConfig config = testConfig();
  config.max_abs_transfer_sec = 2.0e20;
  const CameraTimeTransfer transfer(config);

  EXPECT_FALSE(transfer.configValid());
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  EXPECT_EQ(
    transfer.transferWithSamples(anchor, {}, 0.0).status,
    CameraTimeTransferStatus::kInvalidConfiguration);
}

}  // namespace
