#include "camera_time_transfer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace livo
{
namespace
{

constexpr double kTimestampEpsilonSec = 1.0e-12;
constexpr double kMaximumNoiseIntegrationStepSec = 1.0e-3;
constexpr std::size_t kMaximumNoiseIntegrationStepCount = 1000000U;
constexpr int kInjectCounterMaximum = 1;
constexpr int kInjectCounterNearMaximum = 2;

#ifdef LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING
std::atomic<bool> fail_next_factory_allocation{false};
std::atomic<bool> fail_next_append_allocation{false};
std::atomic<bool> fail_next_clear_resource{false};
std::atomic<bool> fail_next_capture_allocation{false};
std::atomic<bool> fail_next_numerical_allocation{false};
std::atomic<int> epoch_counter_injection{0};
std::atomic<int> revision_counter_injection{0};
std::atomic<int> snapshot_sequence_counter_injection{0};

bool consume(std::atomic<bool> & flag)
{
  return flag.exchange(false);
}

int consume(std::atomic<int> & injection)
{
  return injection.exchange(0);
}
#else
bool failFactoryAllocation()
{
  return false;
}

bool failAppendAllocation()
{
  return false;
}

bool failClearResource()
{
  return false;
}

bool failCaptureAllocation()
{
  return false;
}

bool failNumericalAllocation()
{
  return false;
}

int epochCounterInjection()
{
  return 0;
}

int revisionCounterInjection()
{
  return 0;
}

int snapshotSequenceCounterInjection()
{
  return 0;
}
#endif

#ifdef LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING
bool failFactoryAllocation()
{
  return consume(fail_next_factory_allocation);
}

bool failAppendAllocation()
{
  return consume(fail_next_append_allocation);
}

bool failClearResource()
{
  return consume(fail_next_clear_resource);
}

bool failCaptureAllocation()
{
  return consume(fail_next_capture_allocation);
}

bool failNumericalAllocation()
{
  return consume(fail_next_numerical_allocation);
}

int epochCounterInjection()
{
  return consume(epoch_counter_injection);
}

int revisionCounterInjection()
{
  return consume(revision_counter_injection);
}

int snapshotSequenceCounterInjection()
{
  return consume(snapshot_sequence_counter_injection);
}
#endif

bool checkedNoiseIntegrationStepCount(
  const double duration_sec,
  std::size_t * const step_count)
{
  if (
    step_count == nullptr ||
    !std::isfinite(duration_sec) ||
    duration_sec < 0.0)
  {
    return false;
  }
  const double required_steps = std::ceil(
    duration_sec / kMaximumNoiseIntegrationStepSec);
  if (
    !std::isfinite(required_steps) ||
    required_steps >
      static_cast<double>(
        kMaximumNoiseIntegrationStepCount) ||
    required_steps >
      static_cast<double>(
        std::numeric_limits<std::size_t>::max()))
  {
    return false;
  }
  *step_count = std::max<std::size_t>(
    1U, static_cast<std::size_t>(required_steps));
  return true;
}

double transitionConditionNumber(
  const CameraTimeTransferErrorStateMatrix & transition)
{
  const Eigen::JacobiSVD<CameraTimeTransferErrorStateMatrix>
    decomposition(transition);
  if (decomposition.info() != Eigen::Success)
  {
    return std::numeric_limits<double>::infinity();
  }
  const auto & singular_values = decomposition.singularValues();
  const double maximum_singular_value =
    singular_values.maxCoeff();
  const double minimum_singular_value =
    singular_values.minCoeff();
  if (
    !std::isfinite(maximum_singular_value) ||
    !std::isfinite(minimum_singular_value) ||
    minimum_singular_value <= 0.0)
  {
    return std::numeric_limits<double>::infinity();
  }
  return maximum_singular_value / minimum_singular_value;
}

Eigen::Matrix3d skew(const Eigen::Vector3d & value)
{
  Eigen::Matrix3d result;
  result <<
    0.0, -value.z(), value.y(),
    value.z(), 0.0, -value.x(),
    -value.y(), value.x(), 0.0;
  return result;
}

Eigen::Matrix3d expSo3(const Eigen::Vector3d & tangent)
{
  const double angle = tangent.norm();
  const Eigen::Matrix3d tangent_skew = skew(tangent);
  if (angle < 1.0e-8)
  {
    return Eigen::Matrix3d::Identity() + tangent_skew +
           0.5 * tangent_skew * tangent_skew;
  }
  const double sine_scale = std::sin(angle) / angle;
  const double cosine_scale =
    (1.0 - std::cos(angle)) / (angle * angle);
  return Eigen::Matrix3d::Identity() +
         sine_scale * tangent_skew +
         cosine_scale * tangent_skew * tangent_skew;
}

Eigen::Matrix3d rightJacobianSo3(const Eigen::Vector3d & tangent)
{
  const double angle = tangent.norm();
  const Eigen::Matrix3d tangent_skew = skew(tangent);
  if (angle < 1.0e-8)
  {
    return Eigen::Matrix3d::Identity() -
           0.5 * tangent_skew +
           (1.0 / 6.0) * tangent_skew * tangent_skew;
  }
  const double angle_squared = angle * angle;
  return
    Eigen::Matrix3d::Identity() -
    (1.0 - std::cos(angle)) / angle_squared * tangent_skew +
    (angle - std::sin(angle)) /
      (angle_squared * angle) *
      tangent_skew * tangent_skew;
}

CameraTimeTransferErrorStateMatrix matrixExponentialFourthOrder(
  const CameraTimeTransferErrorStateMatrix & dynamics,
  const double duration_sec)
{
  const CameraTimeTransferErrorStateMatrix first =
    dynamics * duration_sec;
  const CameraTimeTransferErrorStateMatrix second = first * first;
  const CameraTimeTransferErrorStateMatrix third = second * first;
  const CameraTimeTransferErrorStateMatrix fourth = third * first;
  return
    CameraTimeTransferErrorStateMatrix::Identity() +
    first +
    0.5 * second +
    (1.0 / 6.0) * third +
    (1.0 / 24.0) * fourth;
}

CameraTimeTransferErrorStateMatrix discreteStateTransition(
  const Eigen::Matrix3d & midpoint_rotation_world_from_imu,
  const Eigen::Vector3d & angular_velocity_radps,
  const Eigen::Vector3d & specific_force_mps2,
  const double dt)
{
  CameraTimeTransferErrorStateMatrix transition =
    CameraTimeTransferErrorStateMatrix::Identity();
  const Eigen::Vector3d full_rotation_vector =
    angular_velocity_radps * dt;
  const Eigen::Vector3d half_rotation_vector =
    0.5 * full_rotation_vector;
  const Eigen::Matrix3d full_rotation =
    expSo3(full_rotation_vector);
  const Eigen::Matrix3d half_rotation =
    expSo3(half_rotation_vector);
  const Eigen::Matrix3d acceleration_wrt_theta =
    -midpoint_rotation_world_from_imu *
    skew(specific_force_mps2) *
    half_rotation.transpose();
  const Eigen::Matrix3d acceleration_wrt_gyro_bias =
    midpoint_rotation_world_from_imu *
    skew(specific_force_mps2) *
    rightJacobianSo3(half_rotation_vector) *
    (0.5 * dt);
  const Eigen::Matrix3d acceleration_wrt_accel_bias =
    -midpoint_rotation_world_from_imu;

  transition.block<3, 3>(
    kCameraTimeTransferThetaOffset,
    kCameraTimeTransferThetaOffset) =
    full_rotation.transpose();
  transition.block<3, 3>(
    kCameraTimeTransferThetaOffset,
    kCameraTimeTransferGyroBiasOffset) =
    -rightJacobianSo3(full_rotation_vector) * dt;

  transition.block<3, 3>(
    kCameraTimeTransferPositionOffset,
    kCameraTimeTransferVelocityOffset) =
    dt * Eigen::Matrix3d::Identity();
  transition.block<3, 3>(
    kCameraTimeTransferPositionOffset,
    kCameraTimeTransferThetaOffset) =
    0.5 * dt * dt * acceleration_wrt_theta;
  transition.block<3, 3>(
    kCameraTimeTransferPositionOffset,
    kCameraTimeTransferGyroBiasOffset) =
    0.5 * dt * dt * acceleration_wrt_gyro_bias;
  transition.block<3, 3>(
    kCameraTimeTransferPositionOffset,
    kCameraTimeTransferAccelBiasOffset) =
    0.5 * dt * dt * acceleration_wrt_accel_bias;
  transition.block<3, 3>(
    kCameraTimeTransferPositionOffset,
    kCameraTimeTransferGravityOffset) =
    0.5 * dt * dt * Eigen::Matrix3d::Identity();

  transition.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferThetaOffset) =
    dt * acceleration_wrt_theta;
  transition.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferGyroBiasOffset) =
    dt * acceleration_wrt_gyro_bias;
  transition.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferAccelBiasOffset) =
    dt * acceleration_wrt_accel_bias;
  transition.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferGravityOffset) =
    dt * Eigen::Matrix3d::Identity();
  return transition;
}

CameraTimeTransferErrorStateMatrix continuousErrorDynamics(
  const Eigen::Matrix3d & midpoint_rotation_world_from_imu,
  const Eigen::Vector3d & angular_velocity_radps,
  const Eigen::Vector3d & specific_force_mps2)
{
  CameraTimeTransferErrorStateMatrix dynamics =
    CameraTimeTransferErrorStateMatrix::Zero();
  dynamics.block<3, 3>(
    kCameraTimeTransferThetaOffset,
    kCameraTimeTransferThetaOffset) =
    -skew(angular_velocity_radps);
  dynamics.block<3, 3>(
    kCameraTimeTransferThetaOffset,
    kCameraTimeTransferGyroBiasOffset) =
    -Eigen::Matrix3d::Identity();
  dynamics.block<3, 3>(
    kCameraTimeTransferPositionOffset,
    kCameraTimeTransferVelocityOffset) =
    Eigen::Matrix3d::Identity();
  dynamics.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferThetaOffset) =
    -midpoint_rotation_world_from_imu *
    skew(specific_force_mps2);
  dynamics.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferAccelBiasOffset) =
    -midpoint_rotation_world_from_imu;
  dynamics.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferGravityOffset) =
    Eigen::Matrix3d::Identity();
  return dynamics;
}

CameraTimeTransferErrorStateMatrix processNoiseDensity(
  const Eigen::Matrix3d & midpoint_rotation_world_from_imu,
  const CameraTimeTransferContinuousNoiseDensity & density_config)
{
  CameraTimeTransferErrorStateMatrix density =
    CameraTimeTransferErrorStateMatrix::Zero();
  const double gyro_variance =
    density_config.gyro_measurement_rad_per_sqrt_sec *
    density_config.gyro_measurement_rad_per_sqrt_sec;
  const double accel_variance =
    density_config.accel_measurement_mps_per_sqrt_sec *
    density_config.accel_measurement_mps_per_sqrt_sec;
  const double gyro_bias_variance =
    density_config.gyro_bias_random_walk_radps_per_sqrt_sec *
    density_config.gyro_bias_random_walk_radps_per_sqrt_sec;
  const double accel_bias_variance =
    density_config.accel_bias_random_walk_mps2_per_sqrt_sec *
    density_config.accel_bias_random_walk_mps2_per_sqrt_sec;
  density.block<3, 3>(
    kCameraTimeTransferThetaOffset,
    kCameraTimeTransferThetaOffset) =
    gyro_variance * Eigen::Matrix3d::Identity();
  density.block<3, 3>(
    kCameraTimeTransferVelocityOffset,
    kCameraTimeTransferVelocityOffset) =
    accel_variance *
    midpoint_rotation_world_from_imu *
    midpoint_rotation_world_from_imu.transpose();
  density.block<3, 3>(
    kCameraTimeTransferGyroBiasOffset,
    kCameraTimeTransferGyroBiasOffset) =
    gyro_bias_variance * Eigen::Matrix3d::Identity();
  density.block<3, 3>(
    kCameraTimeTransferAccelBiasOffset,
    kCameraTimeTransferAccelBiasOffset) =
    accel_bias_variance * Eigen::Matrix3d::Identity();
  return density;
}

CameraTimeTransferErrorStateMatrix integrateProcessNoise(
  const CameraTimeTransferErrorStateMatrix & forward_dynamics,
  const CameraTimeTransferErrorStateMatrix & noise_density,
  const double signed_duration_sec)
{
  const double duration_sec = std::abs(signed_duration_sec);
  if (duration_sec <= kTimestampEpsilonSec)
  {
    return CameraTimeTransferErrorStateMatrix::Zero();
  }

  std::size_t step_count = 0U;
  if (!checkedNoiseIntegrationStepCount(
      duration_sec, &step_count))
  {
    return CameraTimeTransferErrorStateMatrix::Constant(
      std::numeric_limits<double>::quiet_NaN());
  }
  const double step_duration_sec =
    duration_sec / static_cast<double>(step_count);
  const double direction =
    signed_duration_sec >= 0.0 ? 1.0 : -1.0;
  const CameraTimeTransferErrorStateMatrix directed_dynamics =
    direction * forward_dynamics;
  const CameraTimeTransferErrorStateMatrix sub_transition =
    matrixExponentialFourthOrder(
      directed_dynamics, step_duration_sec);
  const CameraTimeTransferErrorStateMatrix half_transition =
    matrixExponentialFourthOrder(
      directed_dynamics, 0.5 * step_duration_sec);
  const CameraTimeTransferErrorStateMatrix end_density =
    sub_transition * noise_density * sub_transition.transpose();
  const CameraTimeTransferErrorStateMatrix midpoint_density =
    half_transition * noise_density * half_transition.transpose();
  const CameraTimeTransferErrorStateMatrix substep_noise =
    (step_duration_sec / 6.0) *
    (noise_density + 4.0 * midpoint_density + end_density);

  CameraTimeTransferErrorStateMatrix covariance =
    CameraTimeTransferErrorStateMatrix::Zero();
  for (std::size_t index = 0U; index < step_count; ++index)
  {
    covariance =
      sub_transition * covariance * sub_transition.transpose() +
      substep_noise;
  }
  return 0.5 * (covariance + covariance.transpose()).eval();
}

bool finiteVector(const Eigen::Vector3d & value)
{
  return value.allFinite();
}

bool validRotation(const Eigen::Matrix3d & rotation)
{
  if (!rotation.allFinite())
  {
    return false;
  }
  return
    std::abs(rotation.determinant() - 1.0) <= 1.0e-6 &&
    (rotation.transpose() * rotation -
     Eigen::Matrix3d::Identity()).norm() <= 1.0e-6;
}

bool finiteNonnegative(const double value)
{
  return std::isfinite(value) && value >= 0.0;
}

template <int Size>
bool positiveSemidefinite(
  const Eigen::Matrix<double, Size, Size> & covariance)
{
  if (
    !covariance.allFinite() ||
    !covariance.isApprox(covariance.transpose(), 1.0e-12))
  {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<
    Eigen::Matrix<double, Size, Size>> solver(covariance);
  return
    solver.info() == Eigen::Success &&
    solver.eigenvalues().minCoeff() >= -1.0e-12;
}

bool validImuSample(const CameraTimeTransferImuSample & sample)
{
  return
    std::isfinite(sample.timestamp_sec) &&
    finiteVector(sample.angular_velocity_radps) &&
    finiteVector(sample.specific_force_mps2);
}

CameraPoseTransferData invalidResult(
  const CameraTimeTransferStatus status,
  const double anchor_timestamp_sec,
  const double target_timestamp_sec)
{
  CameraPoseTransferData result;
  result.status = status;
  result.anchor_timestamp_sec = anchor_timestamp_sec;
  result.target_timestamp_sec = target_timestamp_sec;
  result.delta_time_sec = target_timestamp_sec - anchor_timestamp_sec;
  return result;
}

CameraTimeTransferImuSample interpolate(
  const std::vector<CameraTimeTransferImuSample> & samples,
  const double timestamp_sec)
{
  const auto upper = std::lower_bound(
    samples.begin(), samples.end(), timestamp_sec,
    [](const CameraTimeTransferImuSample & sample, const double timestamp) {
      return sample.timestamp_sec < timestamp;
    });
  if (upper != samples.end() &&
      std::abs(upper->timestamp_sec - timestamp_sec) <=
      kTimestampEpsilonSec)
  {
    return *upper;
  }
  if (upper == samples.end())
  {
    return samples.back();
  }

  const auto lower = std::prev(upper);
  const double interval = upper->timestamp_sec - lower->timestamp_sec;
  const double ratio = (timestamp_sec - lower->timestamp_sec) / interval;
  CameraTimeTransferImuSample result;
  result.timestamp_sec = timestamp_sec;
  result.angular_velocity_radps =
    (1.0 - ratio) * lower->angular_velocity_radps +
    ratio * upper->angular_velocity_radps;
  result.specific_force_mps2 =
    (1.0 - ratio) * lower->specific_force_mps2 +
    ratio * upper->specific_force_mps2;
  return result;
}

}  // namespace

CameraTimeTransferActivePoseJacobian
embedCameraTimeTransferPoseJacobianInActiveState(
  const CameraTimeTransferPoseJacobian & canonical_jacobian)
{
  CameraTimeTransferActivePoseJacobian active_jacobian =
    CameraTimeTransferActivePoseJacobian::Zero();
  active_jacobian.leftCols<6>() =
    canonical_jacobian.leftCols<6>();
  active_jacobian.middleCols<3>(7) =
    canonical_jacobian.middleCols<3>(
      kCameraTimeTransferVelocityOffset);
  active_jacobian.middleCols<3>(10) =
    canonical_jacobian.middleCols<3>(
      kCameraTimeTransferGyroBiasOffset);
  active_jacobian.middleCols<3>(13) =
    canonical_jacobian.middleCols<3>(
      kCameraTimeTransferAccelBiasOffset);
  active_jacobian.middleCols<3>(16) =
    canonical_jacobian.middleCols<3>(
      kCameraTimeTransferGravityOffset);
  return active_jacobian;
}

bool CameraPoseTransferRaw::internallyValid() const
{
  if (
    status != CameraTimeTransferStatus::kValid ||
    !std::isfinite(anchor_timestamp_sec) ||
    !std::isfinite(target_timestamp_sec) ||
    !std::isfinite(delta_time_sec) ||
    imu_epoch == 0U ||
    !finiteNonnegative(time_uncertainty_stddev_sec) ||
    !finiteNonnegative(maximum_imu_gap_sec) ||
    !std::isfinite(transition_condition_number) ||
    transition_condition_number < 1.0 ||
    !validRotation(rotation_world_from_imu) ||
    !position_world_m.allFinite() ||
    !velocity_world_mps.allFinite())
  {
    return false;
  }
  const double expected_delta =
    target_timestamp_sec - anchor_timestamp_sec;
  const double timestamp_scale = std::max(
    {1.0, std::abs(anchor_timestamp_sec),
     std::abs(target_timestamp_sec)});
  const double delta_tolerance =
    64.0 * std::numeric_limits<double>::epsilon() *
    timestamp_scale;
  if (
    !std::isfinite(expected_delta) ||
    std::abs(delta_time_sec - expected_delta) >
      delta_tolerance)
  {
    return false;
  }
  if (
    !error_state_transition.allFinite() ||
    !target_pose_wrt_anchor_state.allFinite() ||
    !target_pose_wrt_active_anchor_state.allFinite() ||
    !target_pose_wrt_anchor_state.isApprox(
      error_state_transition.topRows<
        kCameraTimeTransferPoseDimension>(),
      1.0e-12) ||
    !target_pose_wrt_active_anchor_state.isApprox(
      embedCameraTimeTransferPoseJacobianInActiveState(
        target_pose_wrt_anchor_state),
      1.0e-12) ||
    !conditional_pose_only.valid() ||
    !conditional_pose_only.target_pose_wrt_anchor_pose.isApprox(
      target_pose_wrt_anchor_state.leftCols<
        kCameraTimeTransferPoseDimension>(),
      1.0e-12) ||
    !positiveSemidefinite(process_noise_covariance) ||
    !positiveSemidefinite(pose_transfer_covariance) ||
    !positiveSemidefinite(time_alignment_pose_covariance) ||
    !pose_transfer_covariance.isApprox(
      process_noise_covariance.topLeftCorner<
        kCameraTimeTransferPoseDimension,
        kCameraTimeTransferPoseDimension>(),
      1.0e-12))
  {
    return false;
  }
  const CameraTimeTransferDirection expected_direction =
    std::abs(delta_time_sec) <= kTimestampEpsilonSec ?
    CameraTimeTransferDirection::kIdentity :
    delta_time_sec > 0.0 ?
    CameraTimeTransferDirection::kForward :
    CameraTimeTransferDirection::kBackward;
  if (direction != expected_direction)
  {
    return false;
  }
  const bool has_process_noise =
    !process_noise_covariance.isZero(0.0);
  if (
    direction == CameraTimeTransferDirection::kIdentity &&
    has_process_noise)
  {
    return false;
  }
  CameraTimeTransferUncertaintySemantics expected_semantics =
    CameraTimeTransferUncertaintySemantics::kNoProcessNoise;
  if (has_process_noise)
  {
    expected_semantics =
      direction == CameraTimeTransferDirection::kForward ?
      CameraTimeTransferUncertaintySemantics::
      kForwardIndependentProcessNoise :
      CameraTimeTransferUncertaintySemantics::
      kBackwardCorrelatedTransportApproximation;
  }
  if (uncertainty_semantics != expected_semantics)
  {
    return false;
  }
  return true;
}

CameraPoseTransfer::CameraPoseTransfer(
  const CameraPoseTransferRaw & raw)
: sealed_raw_(raw)
{
}

const CameraPoseTransferRaw & CameraPoseTransfer::raw() const
{
  return sealed_raw_;
}

bool CameraPoseTransfer::matches(
  const CameraPoseTransferRaw & raw) const
{
  return
    sealed_raw_.status == raw.status &&
    sealed_raw_.direction == raw.direction &&
    sealed_raw_.uncertainty_semantics ==
      raw.uncertainty_semantics &&
    sealed_raw_.anchor_timestamp_sec == raw.anchor_timestamp_sec &&
    sealed_raw_.target_timestamp_sec == raw.target_timestamp_sec &&
    sealed_raw_.delta_time_sec == raw.delta_time_sec &&
    sealed_raw_.imu_epoch == raw.imu_epoch &&
    sealed_raw_.time_uncertainty_stddev_sec ==
      raw.time_uncertainty_stddev_sec &&
    sealed_raw_.maximum_imu_gap_sec == raw.maximum_imu_gap_sec &&
    sealed_raw_.transition_condition_number ==
      raw.transition_condition_number &&
    sealed_raw_.rotation_world_from_imu.isApprox(
      raw.rotation_world_from_imu, 0.0) &&
    sealed_raw_.position_world_m.isApprox(
      raw.position_world_m, 0.0) &&
    sealed_raw_.velocity_world_mps.isApprox(
      raw.velocity_world_mps, 0.0) &&
    sealed_raw_.error_state_transition.isApprox(
      raw.error_state_transition, 0.0) &&
    sealed_raw_.target_pose_wrt_anchor_state.isApprox(
      raw.target_pose_wrt_anchor_state, 0.0) &&
    sealed_raw_.target_pose_wrt_active_anchor_state.isApprox(
      raw.target_pose_wrt_active_anchor_state, 0.0) &&
    sealed_raw_.process_noise_covariance.isApprox(
      raw.process_noise_covariance, 0.0) &&
    sealed_raw_.pose_transfer_covariance.isApprox(
      raw.pose_transfer_covariance, 0.0) &&
    sealed_raw_.time_alignment_pose_covariance.isApprox(
      raw.time_alignment_pose_covariance, 0.0) &&
    sealed_raw_.conditional_pose_only.
      target_pose_wrt_anchor_pose.isApprox(
        raw.conditional_pose_only.
          target_pose_wrt_anchor_pose,
        0.0);
}

void CameraPoseTransferData::sealSuccessful()
{
  if (!internallyValid())
  {
    if (status == CameraTimeTransferStatus::kValid)
    {
      status =
        CameraTimeTransferStatus::kInvalidNumericalResult;
    }
    validation_token_.reset();
    return;
  }
  validation_token_ = CameraPoseTransfer(*this);
}

bool CameraPoseTransferData::valid() const
{
  return
    internallyValid() &&
    validation_token_ &&
    validation_token_->matches(*this);
}

std::optional<CameraPoseTransfer>
CameraPoseTransferData::validatedValue() const
{
  return valid() ? validation_token_ : std::nullopt;
}

std::optional<CameraTimeTransferContinuousNoiseDensity>
CameraTimeTransferConfig::validatedContinuousNoiseDensity() const
{
  if (
    !finiteNonnegative(gyro_noise_stddev_radps) ||
    !finiteNonnegative(accel_noise_stddev_mps2) ||
    !finiteNonnegative(
      gyro_bias_random_walk_stddev_radps_per_sqrt_sec) ||
    !finiteNonnegative(
      accel_bias_random_walk_stddev_mps2_per_sqrt_sec))
  {
    return std::nullopt;
  }
  return CameraTimeTransferContinuousNoiseDensity{
    gyro_noise_stddev_radps,
    accel_noise_stddev_mps2,
    gyro_bias_random_walk_stddev_radps_per_sqrt_sec,
    accel_bias_random_walk_stddev_mps2_per_sqrt_sec};
}

struct CameraTimeTransferAuthorityIdentity::Data final
{
};

struct CameraTimeTransfer::Implementation
{
  explicit Implementation(
    const CameraTimeTransferConfig & input_config)
  : config(input_config),
    continuous_noise_density(
      config.validatedContinuousNoiseDensity()),
    owner(
      std::make_shared<
        const CameraTimeTransferAuthorityIdentity::Data>())
  {
  }

  mutable std::mutex mutex;
  CameraTimeTransferConfig config;
  std::optional<CameraTimeTransferContinuousNoiseDensity>
    continuous_noise_density;
  std::deque<CameraTimeTransferImuSample> imu_ring;
  std::optional<std::pair<double, double>>
    latest_epoch_boundary;
  std::shared_ptr<
    const CameraTimeTransferAuthorityIdentity::Data> owner;
  // Epoch zero is reserved for "unbound/invalid".
  std::uint64_t epoch{1U};
  // Revision zero is reserved for "unbound/invalid".
  std::uint64_t stream_revision{1U};
  // The first successfully published receipt receives sequence one.
  std::uint64_t published_snapshot_sequence{0U};
};

struct CameraTimeTransferEpochReceipt::Data final
{
  Data(
    std::shared_ptr<
      const CameraTimeTransferAuthorityIdentity::Data>
      input_owner,
    const CameraTimeTransferConfig & input_config,
    const CameraTimeTransferContinuousNoiseDensity &
      input_continuous_noise_density,
    std::vector<CameraTimeTransferImuSample> input_samples,
    std::optional<std::pair<double, double>>
      input_latest_epoch_boundary,
    const std::uint64_t input_epoch,
    const std::uint64_t input_stream_revision,
    const std::uint64_t input_snapshot_sequence) noexcept
  : owner(std::move(input_owner)),
    config(input_config),
    continuous_noise_density(input_continuous_noise_density),
    samples(std::move(input_samples)),
    latest_epoch_boundary(
      std::move(input_latest_epoch_boundary)),
    epoch(input_epoch),
    stream_revision(input_stream_revision),
    snapshot_sequence(input_snapshot_sequence)
  {
  }

  std::shared_ptr<
    const CameraTimeTransferAuthorityIdentity::Data> owner;
  CameraTimeTransferConfig config;
  CameraTimeTransferContinuousNoiseDensity
    continuous_noise_density;
  std::vector<CameraTimeTransferImuSample> samples;
  std::optional<std::pair<double, double>>
    latest_epoch_boundary;
  std::uint64_t epoch{0U};
  std::uint64_t stream_revision{0U};
  std::uint64_t snapshot_sequence{0U};
};

namespace
{

bool validConfiguration(
  const CameraTimeTransferConfig & config,
  const std::optional<
    CameraTimeTransferContinuousNoiseDensity> &
    continuous_noise_density)
{
  std::size_t maximum_noise_step_count = 0U;
  return
    std::isfinite(config.max_abs_transfer_sec) &&
    config.max_abs_transfer_sec > 0.0 &&
    checkedNoiseIntegrationStepCount(
      config.max_abs_transfer_sec,
      &maximum_noise_step_count) &&
    std::isfinite(config.max_imu_gap_sec) &&
    config.max_imu_gap_sec > 0.0 &&
    finiteNonnegative(config.imu_timestamp_stddev_sec) &&
    std::isfinite(config.ring_buffer_duration_sec) &&
    config.ring_buffer_duration_sec >= 0.25 &&
    std::isfinite(config.epoch_jump_threshold_sec) &&
    config.epoch_jump_threshold_sec > config.max_imu_gap_sec &&
    continuous_noise_density.has_value() &&
    std::isfinite(
      config.maximum_transition_condition_number) &&
    config.maximum_transition_condition_number >= 1.0 &&
    config.maximum_buffered_samples >= 2U;
}

template<typename Left, typename Right>
bool sameSharedOwner(
  const std::shared_ptr<Left> & left,
  const std::shared_ptr<Right> & right)
{
  return
    left != nullptr &&
    right != nullptr &&
    !left.owner_before(right) &&
    !right.owner_before(left);
}

bool sameContinuousNoiseDensity(
  const CameraTimeTransferContinuousNoiseDensity & left,
  const CameraTimeTransferContinuousNoiseDensity & right)
{
  return
    left.gyro_measurement_rad_per_sqrt_sec ==
      right.gyro_measurement_rad_per_sqrt_sec &&
    left.accel_measurement_mps_per_sqrt_sec ==
      right.accel_measurement_mps_per_sqrt_sec &&
    left.gyro_bias_random_walk_radps_per_sqrt_sec ==
      right.gyro_bias_random_walk_radps_per_sqrt_sec &&
    left.accel_bias_random_walk_mps2_per_sqrt_sec ==
      right.accel_bias_random_walk_mps2_per_sqrt_sec;
}

}  // namespace

CameraTimeTransferAuthorityIdentity::
CameraTimeTransferAuthorityIdentity(
  std::shared_ptr<const Data> data)
: data_(std::move(data))
{
}

bool CameraTimeTransferAuthorityIdentity::valid() const
{
  return data_ != nullptr;
}

CameraTimeTransferEpochReceipt::CameraTimeTransferEpochReceipt(
  std::shared_ptr<const Data> data)
: data_(std::move(data))
{
}

bool CameraTimeTransferEpochReceipt::valid() const
{
  if (
    data_ == nullptr ||
    data_->owner == nullptr ||
    data_->epoch == 0U ||
    data_->stream_revision == 0U ||
    data_->snapshot_sequence == 0U ||
    data_->samples.empty() ||
    data_->samples.size() >
      data_->config.maximum_buffered_samples)
  {
    return false;
  }
  const auto validated_density =
    data_->config.validatedContinuousNoiseDensity();
  if (
    !validConfiguration(data_->config, validated_density) ||
    !validated_density.has_value() ||
    !sameContinuousNoiseDensity(
      data_->continuous_noise_density,
      *validated_density))
  {
    return false;
  }
  for (std::size_t index = 0U;
       index < data_->samples.size(); ++index)
  {
    if (
      !validImuSample(data_->samples[index]) ||
      (index > 0U &&
      data_->samples[index].timestamp_sec <=
      data_->samples[index - 1U].timestamp_sec))
    {
      return false;
    }
  }
  if (
    data_->latest_epoch_boundary.has_value() &&
    (!std::isfinite(
      data_->latest_epoch_boundary->first) ||
    !std::isfinite(
      data_->latest_epoch_boundary->second)))
  {
    return false;
  }
  return true;
}

bool CameraTimeTransferEpochReceipt::belongsTo(
  const CameraTimeTransferAuthorityIdentity & identity) const
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(data_->owner, identity.data_);
}

std::uint64_t CameraTimeTransferEpochReceipt::epoch() const
{
  return valid() ? data_->epoch : 0U;
}

std::uint64_t
CameraTimeTransferEpochReceipt::streamRevision() const
{
  return valid() ? data_->stream_revision : 0U;
}

std::uint64_t
CameraTimeTransferEpochReceipt::snapshotSequence() const
{
  return valid() ? data_->snapshot_sequence : 0U;
}

std::size_t CameraTimeTransferEpochReceipt::bufferedSamples() const
{
  return valid() ? data_->samples.size() : 0U;
}

EpochBoundCameraPoseTransfer::EpochBoundCameraPoseTransfer(
  CameraPoseTransfer value,
  std::shared_ptr<const void> receipt_identity)
: value_(std::move(value)),
  receipt_identity_(std::move(receipt_identity))
{
}

bool EpochBoundCameraPoseTransfer::valid() const
{
  return
    receipt_identity_ != nullptr &&
    value_.raw().internallyValid();
}

bool EpochBoundCameraPoseTransfer::wasProducedBy(
  const CameraTimeTransferEpochReceipt & receipt) const
{
  return
    valid() &&
    receipt.valid() &&
    sameSharedOwner(receipt_identity_, receipt.data_);
}

const CameraPoseTransfer &
EpochBoundCameraPoseTransfer::value() const
{
  return value_;
}

const CameraPoseTransferRaw &
EpochBoundCameraPoseTransfer::raw() const
{
  return value_.raw();
}

CameraTimeTransferEpochTransferResult
CameraTimeTransferEpochReceipt::transfer(
  const CameraTimeTransferState & anchor,
  const double target_timestamp_sec,
  const double target_timestamp_stddev_sec) const
{
  CameraTimeTransferEpochTransferResult output;
  if (!valid())
  {
    output.status = CameraTimeTransferStatus::kInvalidInput;
    return output;
  }

  CameraTimeTransfer computation(data_->config);
  if (!computation.configValid())
  {
    output.status =
      CameraTimeTransferStatus::kResourceFailure;
    return output;
  }
  CameraPoseTransferData result =
    computation.transferWithSamples(
    anchor, data_->samples, target_timestamp_sec,
    target_timestamp_stddev_sec);
  if (
    data_->latest_epoch_boundary.has_value() &&
    std::abs(target_timestamp_sec - anchor.timestamp_sec) >
      kTimestampEpsilonSec &&
    result.status !=
      CameraTimeTransferStatus::kInvalidConfiguration &&
    result.status != CameraTimeTransferStatus::kInvalidInput &&
    result.status !=
      CameraTimeTransferStatus::kTargetOutOfRange &&
    result.status !=
      CameraTimeTransferStatus::kCapacityExceeded &&
    result.status !=
      CameraTimeTransferStatus::kResourceFailure)
  {
    const double interval_start = std::min(
      anchor.timestamp_sec, target_timestamp_sec);
    const double interval_end = std::max(
      anchor.timestamp_sec, target_timestamp_sec);
    const double boundary_start = std::min(
      data_->latest_epoch_boundary->first,
      data_->latest_epoch_boundary->second);
    const double boundary_end = std::max(
      data_->latest_epoch_boundary->first,
      data_->latest_epoch_boundary->second);
    if (
      interval_start <=
      boundary_start + kTimestampEpsilonSec &&
      interval_end >=
      boundary_end - kTimestampEpsilonSec)
    {
      result = invalidResult(
        CameraTimeTransferStatus::kCrossesImuEpoch,
        anchor.timestamp_sec, target_timestamp_sec);
    }
  }
  result.imu_epoch = data_->epoch;
  result.sealSuccessful();
  output.status = result.status;
  const std::optional<CameraPoseTransfer> value =
    result.validatedValue();
  if (!value.has_value())
  {
    return output;
  }
  output.value = EpochBoundCameraPoseTransfer(
    *value, std::shared_ptr<const void>(data_));
  return output;
}

CameraTimeTransfer::CameraTimeTransfer(CameraTimeTransferConfig config)
{
  try
  {
    if (failFactoryAllocation())
    {
      throw std::bad_alloc();
    }
    implementation_ =
      std::make_unique<Implementation>(config);
  }
  catch (const std::bad_alloc &)
  {
    implementation_.reset();
  }
}

CameraTimeTransfer::~CameraTimeTransfer() = default;

CameraTimeTransfer::CameraTimeTransfer(
  CameraTimeTransfer &&) noexcept = default;

CameraTimeTransfer & CameraTimeTransfer::operator=(
  CameraTimeTransfer &&) noexcept = default;

bool CameraTimeTransfer::configValid() const
{
  if (!implementation_)
  {
    return false;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  return validConfiguration(
    implementation_->config,
    implementation_->continuous_noise_density);
}

CameraTimeTransferAppendStatus CameraTimeTransfer::appendImu(
  const CameraTimeTransferImuSample & sample)
{
  if (!implementation_)
  {
    return CameraTimeTransferAppendStatus::
           kAuthorityUnavailable;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (!validConfiguration(
      implementation_->config,
      implementation_->continuous_noise_density))
  {
    return CameraTimeTransferAppendStatus::
           kInvalidConfiguration;
  }
  if (!validImuSample(sample))
  {
    return CameraTimeTransferAppendStatus::kInvalidSample;
  }
  const int revision_injection = revisionCounterInjection();
  if (revision_injection == kInjectCounterMaximum)
  {
    implementation_->stream_revision =
      std::numeric_limits<std::uint64_t>::max();
  }
  else if (
    revision_injection == kInjectCounterNearMaximum)
  {
    implementation_->stream_revision =
      std::numeric_limits<std::uint64_t>::max() - 1U;
  }
  if (
    implementation_->stream_revision ==
    std::numeric_limits<std::uint64_t>::max())
  {
    return CameraTimeTransferAppendStatus::
           kRevisionExhausted;
  }

  const std::uint64_t next_revision =
    implementation_->stream_revision + 1U;
  bool starts_new_epoch = false;
  if (!implementation_->imu_ring.empty())
  {
    const double dt =
      sample.timestamp_sec -
      implementation_->imu_ring.back().timestamp_sec;
    starts_new_epoch =
      dt <= 0.0 ||
      dt >
      implementation_->config.epoch_jump_threshold_sec;
  }

  if (starts_new_epoch)
  {
    const int epoch_injection = epochCounterInjection();
    if (epoch_injection == kInjectCounterMaximum)
    {
      implementation_->epoch =
        std::numeric_limits<std::uint64_t>::max();
    }
    else if (epoch_injection == kInjectCounterNearMaximum)
    {
      implementation_->epoch =
        std::numeric_limits<std::uint64_t>::max() - 1U;
    }
    if (
      implementation_->epoch ==
      std::numeric_limits<std::uint64_t>::max())
    {
      return CameraTimeTransferAppendStatus::
             kEpochExhausted;
    }
    try
    {
      if (failAppendAllocation())
      {
        throw std::bad_alloc();
      }
      std::deque<CameraTimeTransferImuSample>
        candidate_ring;
      candidate_ring.push_back(sample);
      const std::pair<double, double> candidate_boundary{
        implementation_->imu_ring.back().timestamp_sec,
        sample.timestamp_sec};
      implementation_->imu_ring.swap(candidate_ring);
      implementation_->latest_epoch_boundary =
        candidate_boundary;
      ++implementation_->epoch;
      implementation_->stream_revision = next_revision;
      return CameraTimeTransferAppendStatus::kNewEpoch;
    }
    catch (const std::bad_alloc &)
    {
      return CameraTimeTransferAppendStatus::
             kResourceFailure;
    }
  }

  const double oldest_timestamp =
    sample.timestamp_sec -
    implementation_->config.ring_buffer_duration_sec;
  if (
    implementation_->imu_ring.size() ==
    std::numeric_limits<std::size_t>::max())
  {
    return CameraTimeTransferAppendStatus::
           kCapacityExceeded;
  }
  const std::size_t candidate_size =
    implementation_->imu_ring.size() + 1U;
  std::size_t prune_count = 0U;
  while (
    candidate_size - prune_count > 2U &&
    prune_count + 1U <
    implementation_->imu_ring.size() &&
    implementation_->imu_ring[prune_count + 1U].
    timestamp_sec < oldest_timestamp)
  {
    ++prune_count;
  }
  if (
    candidate_size - prune_count >
    implementation_->config.maximum_buffered_samples)
  {
    return CameraTimeTransferAppendStatus::
           kCapacityExceeded;
  }
  try
  {
    if (failAppendAllocation())
    {
      throw std::bad_alloc();
    }
    implementation_->imu_ring.push_back(sample);
  }
  catch (const std::bad_alloc &)
  {
    return CameraTimeTransferAppendStatus::
           kResourceFailure;
  }
  for (std::size_t index = 0U;
       index < prune_count; ++index)
  {
    implementation_->imu_ring.pop_front();
  }
  implementation_->stream_revision = next_revision;
  return CameraTimeTransferAppendStatus::kAccepted;
}

CameraTimeTransferClearStatus CameraTimeTransfer::clear()
{
  if (!implementation_)
  {
    return CameraTimeTransferClearStatus::
           kAuthorityUnavailable;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (!validConfiguration(
      implementation_->config,
      implementation_->continuous_noise_density))
  {
    return CameraTimeTransferClearStatus::
           kInvalidConfiguration;
  }
  if (failClearResource())
  {
    return CameraTimeTransferClearStatus::kResourceFailure;
  }
  const int epoch_injection = epochCounterInjection();
  if (epoch_injection == kInjectCounterMaximum)
  {
    implementation_->epoch =
      std::numeric_limits<std::uint64_t>::max();
  }
  else if (epoch_injection == kInjectCounterNearMaximum)
  {
    implementation_->epoch =
      std::numeric_limits<std::uint64_t>::max() - 1U;
  }
  if (
    implementation_->epoch ==
    std::numeric_limits<std::uint64_t>::max())
  {
    return CameraTimeTransferClearStatus::kEpochExhausted;
  }
  const int revision_injection = revisionCounterInjection();
  if (revision_injection == kInjectCounterMaximum)
  {
    implementation_->stream_revision =
      std::numeric_limits<std::uint64_t>::max();
  }
  else if (
    revision_injection == kInjectCounterNearMaximum)
  {
    implementation_->stream_revision =
      std::numeric_limits<std::uint64_t>::max() - 1U;
  }
  if (
    implementation_->stream_revision ==
    std::numeric_limits<std::uint64_t>::max())
  {
    return CameraTimeTransferClearStatus::
           kRevisionExhausted;
  }
  implementation_->imu_ring.clear();
  implementation_->latest_epoch_boundary.reset();
  ++implementation_->epoch;
  ++implementation_->stream_revision;
  return CameraTimeTransferClearStatus::kCleared;
}

std::uint64_t CameraTimeTransfer::epoch() const
{
  if (!implementation_)
  {
    return 0U;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  return implementation_->epoch;
}

std::size_t CameraTimeTransfer::bufferedSamples() const
{
  if (!implementation_)
  {
    return 0U;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  return implementation_->imu_ring.size();
}

std::optional<CameraTimeTransferAuthorityIdentity>
CameraTimeTransfer::authorityIdentity() const
{
  if (!implementation_)
  {
    return std::nullopt;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (implementation_->owner == nullptr)
  {
    return std::nullopt;
  }
  return CameraTimeTransferAuthorityIdentity(
    implementation_->owner);
}

CameraTimeTransferEpochCaptureResult
CameraTimeTransfer::captureEpoch()
{
  CameraTimeTransferEpochCaptureResult result;
  if (!implementation_)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kInvalidAuthority;
    return result;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (implementation_->owner == nullptr)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kInvalidAuthority;
    return result;
  }
  if (!validConfiguration(
      implementation_->config,
      implementation_->continuous_noise_density))
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kInvalidConfiguration;
    return result;
  }
  if (implementation_->imu_ring.empty())
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::kEmpty;
    return result;
  }
  if (
    implementation_->imu_ring.size() >
    implementation_->config.maximum_buffered_samples)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kCapacityExceeded;
    return result;
  }
  const int sequence_injection =
    snapshotSequenceCounterInjection();
  if (sequence_injection == kInjectCounterMaximum)
  {
    implementation_->published_snapshot_sequence =
      std::numeric_limits<std::uint64_t>::max();
  }
  else if (
    sequence_injection == kInjectCounterNearMaximum)
  {
    implementation_->published_snapshot_sequence =
      std::numeric_limits<std::uint64_t>::max() - 1U;
  }
  if (
    implementation_->published_snapshot_sequence ==
    std::numeric_limits<std::uint64_t>::max())
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kSequenceExhausted;
    return result;
  }
  const std::uint64_t next_sequence =
    implementation_->published_snapshot_sequence + 1U;
  try
  {
    if (failCaptureAllocation())
    {
      throw std::bad_alloc();
    }
    std::vector<CameraTimeTransferImuSample> samples;
    samples.reserve(implementation_->imu_ring.size());
    samples.assign(
      implementation_->imu_ring.begin(),
      implementation_->imu_ring.end());
    const auto data =
      std::make_shared<
        const CameraTimeTransferEpochReceipt::Data>(
      implementation_->owner,
      implementation_->config,
      *implementation_->continuous_noise_density,
      std::move(samples),
      implementation_->latest_epoch_boundary,
      implementation_->epoch,
      implementation_->stream_revision,
      next_sequence);
    result.receipt =
      CameraTimeTransferEpochReceipt(data);
    implementation_->published_snapshot_sequence =
      next_sequence;
    result.status =
      CameraTimeTransferEpochCaptureStatus::kCaptured;
  }
  catch (const std::bad_alloc &)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kResourceFailure;
  }
  return result;
}

CameraTimeTransferEpochCaptureResult
CameraTimeTransfer::captureWindow(
  const double interval_start_timestamp_sec,
  const double interval_end_timestamp_sec)
{
  CameraTimeTransferEpochCaptureResult result;
  if (
    !std::isfinite(interval_start_timestamp_sec) ||
    !std::isfinite(interval_end_timestamp_sec) ||
    interval_start_timestamp_sec >
    interval_end_timestamp_sec)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::kInvalidInput;
    return result;
  }
  if (!implementation_)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kInvalidAuthority;
    return result;
  }
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (implementation_->owner == nullptr)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kInvalidAuthority;
    return result;
  }
  if (!validConfiguration(
      implementation_->config,
      implementation_->continuous_noise_density))
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kInvalidConfiguration;
    return result;
  }
  if (implementation_->imu_ring.empty())
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::kEmpty;
    return result;
  }
  if (
    implementation_->imu_ring.front().timestamp_sec >
      interval_start_timestamp_sec +
      kTimestampEpsilonSec ||
    implementation_->imu_ring.back().timestamp_sec <
      interval_end_timestamp_sec -
      kTimestampEpsilonSec)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kWindowNotCovered;
    return result;
  }
  const int sequence_injection =
    snapshotSequenceCounterInjection();
  if (sequence_injection == kInjectCounterMaximum)
  {
    implementation_->published_snapshot_sequence =
      std::numeric_limits<std::uint64_t>::max();
  }
  else if (
    sequence_injection == kInjectCounterNearMaximum)
  {
    implementation_->published_snapshot_sequence =
      std::numeric_limits<std::uint64_t>::max() - 1U;
  }
  if (
    implementation_->published_snapshot_sequence ==
    std::numeric_limits<std::uint64_t>::max())
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kSequenceExhausted;
    return result;
  }

  auto first = std::lower_bound(
    implementation_->imu_ring.begin(),
    implementation_->imu_ring.end(),
    interval_start_timestamp_sec,
    [](const CameraTimeTransferImuSample & sample,
      const double timestamp_sec) {
      return sample.timestamp_sec < timestamp_sec;
    });
  if (
    first == implementation_->imu_ring.end() ||
    first->timestamp_sec >
    interval_start_timestamp_sec + kTimestampEpsilonSec)
  {
    if (first == implementation_->imu_ring.begin())
    {
      result.status =
        CameraTimeTransferEpochCaptureStatus::
        kWindowNotCovered;
      return result;
    }
    --first;
  }
  auto last = std::lower_bound(
    implementation_->imu_ring.begin(),
    implementation_->imu_ring.end(),
    interval_end_timestamp_sec,
    [](const CameraTimeTransferImuSample & sample,
      const double timestamp_sec) {
      return sample.timestamp_sec < timestamp_sec;
    });
  if (last == implementation_->imu_ring.end())
  {
    if (
      std::abs(
        implementation_->imu_ring.back().timestamp_sec -
        interval_end_timestamp_sec) >
      kTimestampEpsilonSec)
    {
      result.status =
        CameraTimeTransferEpochCaptureStatus::
        kWindowNotCovered;
      return result;
    }
    last = std::prev(implementation_->imu_ring.end());
  }
  ++last;
  const std::size_t sample_count =
    static_cast<std::size_t>(std::distance(first, last));
  if (
    sample_count >
    implementation_->config.maximum_buffered_samples)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kCapacityExceeded;
    return result;
  }
  const std::uint64_t next_sequence =
    implementation_->published_snapshot_sequence + 1U;
  try
  {
    if (failCaptureAllocation())
    {
      throw std::bad_alloc();
    }
    std::vector<CameraTimeTransferImuSample> samples;
    samples.reserve(sample_count);
    samples.assign(first, last);
    const auto data =
      std::make_shared<
        const CameraTimeTransferEpochReceipt::Data>(
      implementation_->owner,
      implementation_->config,
      *implementation_->continuous_noise_density,
      std::move(samples),
      implementation_->latest_epoch_boundary,
      implementation_->epoch,
      implementation_->stream_revision,
      next_sequence);
    result.receipt =
      CameraTimeTransferEpochReceipt(data);
    implementation_->published_snapshot_sequence =
      next_sequence;
    result.status =
      CameraTimeTransferEpochCaptureStatus::kCaptured;
  }
  catch (const std::bad_alloc &)
  {
    result.status =
      CameraTimeTransferEpochCaptureStatus::
      kResourceFailure;
  }
  return result;
}

CameraPoseTransferData CameraTimeTransfer::transfer(
  const CameraTimeTransferState & anchor,
  const double target_timestamp_sec,
  const double target_timestamp_stddev_sec) const
{
  if (!implementation_)
  {
    return invalidResult(
      CameraTimeTransferStatus::kInvalidConfiguration,
      anchor.timestamp_sec, target_timestamp_sec);
  }
  std::vector<CameraTimeTransferImuSample> samples;
  std::optional<std::pair<double, double>>
    latest_epoch_boundary;
  std::uint64_t epoch = 0U;
  try
  {
    const std::lock_guard<std::mutex> lock(
      implementation_->mutex);
    samples.assign(
      implementation_->imu_ring.begin(),
      implementation_->imu_ring.end());
    latest_epoch_boundary =
      implementation_->latest_epoch_boundary;
    epoch = implementation_->epoch;
  }
  catch (const std::bad_alloc &)
  {
    return invalidResult(
      CameraTimeTransferStatus::kResourceFailure,
      anchor.timestamp_sec, target_timestamp_sec);
  }
  CameraPoseTransferData result = transferWithSamples(
    anchor, samples, target_timestamp_sec,
    target_timestamp_stddev_sec);
  if (
    latest_epoch_boundary.has_value() &&
    std::abs(target_timestamp_sec - anchor.timestamp_sec) >
      kTimestampEpsilonSec &&
    result.status !=
      CameraTimeTransferStatus::kInvalidConfiguration &&
    result.status != CameraTimeTransferStatus::kInvalidInput &&
    result.status !=
      CameraTimeTransferStatus::kTargetOutOfRange &&
    result.status !=
      CameraTimeTransferStatus::kCapacityExceeded &&
    result.status !=
      CameraTimeTransferStatus::kResourceFailure)
  {
    const double interval_start =
      std::min(anchor.timestamp_sec, target_timestamp_sec);
    const double interval_end =
      std::max(anchor.timestamp_sec, target_timestamp_sec);
    const double boundary_start = std::min(
      latest_epoch_boundary->first,
      latest_epoch_boundary->second);
    const double boundary_end = std::max(
      latest_epoch_boundary->first,
      latest_epoch_boundary->second);
    if (
      interval_start <= boundary_start + kTimestampEpsilonSec &&
      interval_end >= boundary_end - kTimestampEpsilonSec)
    {
      result = invalidResult(
        CameraTimeTransferStatus::kCrossesImuEpoch,
        anchor.timestamp_sec, target_timestamp_sec);
    }
  }
  result.imu_epoch = epoch;
  result.sealSuccessful();
  return result;
}

CameraPoseTransferData CameraTimeTransfer::transferWithSamples(
  const CameraTimeTransferState & anchor,
  const std::vector<CameraTimeTransferImuSample> & imu_samples,
  const double target_timestamp_sec,
  const double target_timestamp_stddev_sec) const
{
  if (!implementation_)
  {
    return invalidResult(
      CameraTimeTransferStatus::kInvalidConfiguration,
      anchor.timestamp_sec, target_timestamp_sec);
  }
  CameraTimeTransferConfig config;
  std::optional<CameraTimeTransferContinuousNoiseDensity>
    continuous_noise_density;
  {
    const std::lock_guard<std::mutex> lock(
      implementation_->mutex);
    config = implementation_->config;
    continuous_noise_density =
      implementation_->continuous_noise_density;
  }
  if (!validConfiguration(config, continuous_noise_density))
  {
    return invalidResult(
      CameraTimeTransferStatus::kInvalidConfiguration,
      anchor.timestamp_sec, target_timestamp_sec);
  }
  if (imu_samples.size() > config.maximum_buffered_samples)
  {
    return invalidResult(
      CameraTimeTransferStatus::kCapacityExceeded,
      anchor.timestamp_sec, target_timestamp_sec);
  }
  try
  {
  if (failNumericalAllocation())
  {
    throw std::bad_alloc();
  }
  if (
    !std::isfinite(anchor.timestamp_sec) ||
    !std::isfinite(target_timestamp_sec) ||
    !finiteNonnegative(anchor.timestamp_stddev_sec) ||
    !finiteNonnegative(target_timestamp_stddev_sec) ||
    !validRotation(anchor.rotation_world_from_imu) ||
    !finiteVector(anchor.position_world_m) ||
    !finiteVector(anchor.velocity_world_mps) ||
    !finiteVector(anchor.gyro_bias_radps) ||
    !finiteVector(anchor.accel_bias_mps2) ||
    !finiteVector(anchor.gravity_world_mps2))
  {
    return invalidResult(
      CameraTimeTransferStatus::kInvalidInput,
      anchor.timestamp_sec, target_timestamp_sec);
  }

  const double delta_time_sec =
    target_timestamp_sec - anchor.timestamp_sec;
  if (std::abs(delta_time_sec) >
      config.max_abs_transfer_sec + kTimestampEpsilonSec)
  {
    return invalidResult(
      CameraTimeTransferStatus::kTargetOutOfRange,
      anchor.timestamp_sec, target_timestamp_sec);
  }

  if (std::abs(delta_time_sec) <= kTimestampEpsilonSec)
  {
    CameraPoseTransferData result;
    result.status = CameraTimeTransferStatus::kValid;
    result.direction = CameraTimeTransferDirection::kIdentity;
    result.uncertainty_semantics =
      CameraTimeTransferUncertaintySemantics::kNoProcessNoise;
    result.anchor_timestamp_sec = anchor.timestamp_sec;
    result.target_timestamp_sec = target_timestamp_sec;
    result.delta_time_sec = delta_time_sec;
    result.imu_epoch = 1U;
    result.time_uncertainty_stddev_sec = std::hypot(
      anchor.timestamp_stddev_sec, target_timestamp_stddev_sec);
    result.maximum_imu_gap_sec = 0.0;
    result.transition_condition_number = 1.0;
    result.rotation_world_from_imu =
      anchor.rotation_world_from_imu;
    result.position_world_m = anchor.position_world_m;
    result.velocity_world_mps = anchor.velocity_world_mps;
    result.error_state_transition.setIdentity();
    result.target_pose_wrt_anchor_state =
      result.error_state_transition.topRows<
        kCameraTimeTransferPoseDimension>();
    result.target_pose_wrt_active_anchor_state =
      embedCameraTimeTransferPoseJacobianInActiveState(
        result.target_pose_wrt_anchor_state);
    result.process_noise_covariance.setZero();
    result.pose_transfer_covariance.setZero();
    Eigen::Matrix<double, 6, 1> time_derivative =
      Eigen::Matrix<double, 6, 1>::Zero();
    time_derivative.tail<3>() = anchor.velocity_world_mps;
    const double time_variance =
      result.time_uncertainty_stddev_sec *
      result.time_uncertainty_stddev_sec;
    result.time_alignment_pose_covariance =
      time_variance *
      time_derivative * time_derivative.transpose();
    result.conditional_pose_only.target_pose_wrt_anchor_pose =
      result.target_pose_wrt_anchor_state.leftCols<
        kCameraTimeTransferPoseDimension>();
    result.sealSuccessful();
    return result;
  }

  for (std::size_t index = 0U; index < imu_samples.size(); ++index)
  {
    const CameraTimeTransferImuSample & sample = imu_samples[index];
    if (!validImuSample(sample))
    {
      return invalidResult(
        CameraTimeTransferStatus::kInvalidInput,
        anchor.timestamp_sec, target_timestamp_sec);
    }
    if (
      index > 0U &&
      sample.timestamp_sec <=
      imu_samples[index - 1U].timestamp_sec)
    {
      return invalidResult(
        CameraTimeTransferStatus::kImuNotStrictlyOrdered,
        anchor.timestamp_sec, target_timestamp_sec);
    }
  }

  const double interval_start =
    std::min(anchor.timestamp_sec, target_timestamp_sec);
  const double interval_end =
    std::max(anchor.timestamp_sec, target_timestamp_sec);
  const bool missing_left_bracket =
    !imu_samples.empty() &&
    imu_samples.front().timestamp_sec >
      interval_start + kTimestampEpsilonSec;
  const bool missing_right_bracket =
    !imu_samples.empty() &&
    imu_samples.back().timestamp_sec <
      interval_end - kTimestampEpsilonSec;
  CameraTimeTransferStatus coverage_status =
    CameraTimeTransferStatus::kImuCoverageMissing;
  if (missing_left_bracket && !missing_right_bracket)
  {
    coverage_status =
      CameraTimeTransferStatus::kMissingLeftBracket;
  }
  else if (missing_right_bracket && !missing_left_bracket)
  {
    coverage_status =
      CameraTimeTransferStatus::kMissingRightBracket;
  }
  if (
    imu_samples.size() < 2U ||
    missing_left_bracket ||
    missing_right_bracket)
  {
    return invalidResult(
      coverage_status,
      anchor.timestamp_sec, target_timestamp_sec);
  }

  double maximum_bracketing_gap_sec = 0.0;
  for (std::size_t index = 1U; index < imu_samples.size(); ++index)
  {
    const double left = imu_samples[index - 1U].timestamp_sec;
    const double right = imu_samples[index].timestamp_sec;
    if (right <= interval_start + kTimestampEpsilonSec ||
        left >= interval_end - kTimestampEpsilonSec)
    {
      continue;
    }
    const double gap = right - left;
    maximum_bracketing_gap_sec =
      std::max(maximum_bracketing_gap_sec, gap);
    if (gap > config.max_imu_gap_sec + kTimestampEpsilonSec)
    {
      CameraPoseTransferData result = invalidResult(
        CameraTimeTransferStatus::kImuGapTooLarge,
        anchor.timestamp_sec, target_timestamp_sec);
      result.maximum_imu_gap_sec = maximum_bracketing_gap_sec;
      return result;
    }
  }

  std::vector<double> integration_times;
  integration_times.push_back(anchor.timestamp_sec);
  if (delta_time_sec > 0.0)
  {
    for (const CameraTimeTransferImuSample & sample : imu_samples)
    {
      if (
        sample.timestamp_sec >
          anchor.timestamp_sec + kTimestampEpsilonSec &&
        sample.timestamp_sec <
          target_timestamp_sec - kTimestampEpsilonSec)
      {
        integration_times.push_back(sample.timestamp_sec);
      }
    }
  }
  else
  {
    for (auto sample = imu_samples.rbegin();
         sample != imu_samples.rend(); ++sample)
    {
      if (
        sample->timestamp_sec <
          anchor.timestamp_sec - kTimestampEpsilonSec &&
        sample->timestamp_sec >
          target_timestamp_sec + kTimestampEpsilonSec)
      {
        integration_times.push_back(sample->timestamp_sec);
      }
    }
  }
  integration_times.push_back(target_timestamp_sec);

  Eigen::Matrix3d rotation = anchor.rotation_world_from_imu;
  Eigen::Vector3d position = anchor.position_world_m;
  Eigen::Vector3d velocity = anchor.velocity_world_mps;
  CameraTimeTransferErrorStateMatrix state_transition =
    CameraTimeTransferErrorStateMatrix::Identity();
  CameraTimeTransferErrorStateMatrix process_covariance =
    CameraTimeTransferErrorStateMatrix::Zero();

  for (std::size_t index = 1U;
       index < integration_times.size(); ++index)
  {
    const double start_time = integration_times[index - 1U];
    const double end_time = integration_times[index];
    const double dt = end_time - start_time;
    const CameraTimeTransferImuSample start =
      interpolate(imu_samples, start_time);
    const CameraTimeTransferImuSample end =
      interpolate(imu_samples, end_time);
    const Eigen::Vector3d angular_velocity =
      0.5 * (start.angular_velocity_radps +
             end.angular_velocity_radps) -
      anchor.gyro_bias_radps;
    const Eigen::Vector3d specific_force =
      0.5 * (start.specific_force_mps2 +
             end.specific_force_mps2) -
      anchor.accel_bias_mps2;

    const Eigen::Matrix3d midpoint_rotation =
      rotation * expSo3(0.5 * angular_velocity * dt);
    const Eigen::Vector3d acceleration_world =
      midpoint_rotation * specific_force +
      anchor.gravity_world_mps2;
    const CameraTimeTransferErrorStateMatrix step_transition =
      discreteStateTransition(
        midpoint_rotation, angular_velocity,
        specific_force, dt);
    const CameraTimeTransferErrorStateMatrix step_process_noise =
      integrateProcessNoise(
        continuousErrorDynamics(
          midpoint_rotation, angular_velocity,
          specific_force),
        processNoiseDensity(
          midpoint_rotation, *continuous_noise_density),
        dt);

    position += velocity * dt +
      0.5 * acceleration_world * dt * dt;
    velocity += acceleration_world * dt;
    rotation *= expSo3(angular_velocity * dt);
    process_covariance =
      step_transition * process_covariance *
      step_transition.transpose() +
      step_process_noise;
    process_covariance =
      0.5 *
      (process_covariance + process_covariance.transpose()).eval();
    state_transition = step_transition * state_transition;
  }

  CameraPoseTransferData result;
  result.status = CameraTimeTransferStatus::kValid;
  result.direction =
    delta_time_sec > 0.0 ?
    CameraTimeTransferDirection::kForward :
    CameraTimeTransferDirection::kBackward;
  result.anchor_timestamp_sec = anchor.timestamp_sec;
  result.target_timestamp_sec = target_timestamp_sec;
  result.delta_time_sec = delta_time_sec;
  result.imu_epoch = 1U;
  result.rotation_world_from_imu = rotation;
  result.position_world_m = position;
  result.velocity_world_mps = velocity;
  result.time_uncertainty_stddev_sec = std::sqrt(
    anchor.timestamp_stddev_sec * anchor.timestamp_stddev_sec +
    target_timestamp_stddev_sec * target_timestamp_stddev_sec +
    config.imu_timestamp_stddev_sec *
      config.imu_timestamp_stddev_sec +
      maximum_bracketing_gap_sec *
      maximum_bracketing_gap_sec / 12.0);
  result.maximum_imu_gap_sec = maximum_bracketing_gap_sec;
  result.error_state_transition = state_transition;
  if (state_transition.allFinite())
  {
    result.transition_condition_number =
      transitionConditionNumber(state_transition);
  }
  result.target_pose_wrt_anchor_state =
    state_transition.topRows<kCameraTimeTransferPoseDimension>();
  result.target_pose_wrt_active_anchor_state =
    embedCameraTimeTransferPoseJacobianInActiveState(
      result.target_pose_wrt_anchor_state);
  result.process_noise_covariance = process_covariance;
  result.uncertainty_semantics =
    process_covariance.isZero(0.0) ?
    CameraTimeTransferUncertaintySemantics::kNoProcessNoise :
    result.direction == CameraTimeTransferDirection::kForward ?
    CameraTimeTransferUncertaintySemantics::
    kForwardIndependentProcessNoise :
    CameraTimeTransferUncertaintySemantics::
    kBackwardCorrelatedTransportApproximation;
  result.pose_transfer_covariance =
    process_covariance.topLeftCorner<
      kCameraTimeTransferPoseDimension,
      kCameraTimeTransferPoseDimension>();
  result.conditional_pose_only.target_pose_wrt_anchor_pose =
    result.target_pose_wrt_anchor_state.leftCols<
      kCameraTimeTransferPoseDimension>();
  const CameraTimeTransferImuSample target_imu =
    interpolate(imu_samples, target_timestamp_sec);
  const Eigen::Vector3d target_angular_velocity =
    target_imu.angular_velocity_radps - anchor.gyro_bias_radps;
  const double time_variance =
    result.time_uncertainty_stddev_sec *
    result.time_uncertainty_stddev_sec;
  Eigen::Matrix<double, 6, 1> time_derivative;
  time_derivative <<
    target_angular_velocity,
    result.velocity_world_mps;
  result.time_alignment_pose_covariance =
    time_variance *
    time_derivative * time_derivative.transpose();
  if (
    state_transition.allFinite() &&
    (!std::isfinite(result.transition_condition_number) ||
      result.transition_condition_number >
        config.maximum_transition_condition_number))
  {
    result.status =
      CameraTimeTransferStatus::kIllConditionedTransition;
    return result;
  }
  result.sealSuccessful();
  return result;
  }
  catch (const std::bad_alloc &)
  {
    return invalidResult(
      CameraTimeTransferStatus::kResourceFailure,
      anchor.timestamp_sec, target_timestamp_sec);
  }
}

const char * cameraTimeTransferStatusName(
  const CameraTimeTransferStatus status)
{
  switch (status)
  {
    case CameraTimeTransferStatus::kValid:
      return "valid";
    case CameraTimeTransferStatus::kInvalidConfiguration:
      return "invalid_configuration";
    case CameraTimeTransferStatus::kInvalidInput:
      return "invalid_input";
    case CameraTimeTransferStatus::kTargetOutOfRange:
      return "target_out_of_range";
    case CameraTimeTransferStatus::kImuNotStrictlyOrdered:
      return "imu_not_strictly_ordered";
    case CameraTimeTransferStatus::kImuCoverageMissing:
      return "imu_coverage_missing";
    case CameraTimeTransferStatus::kImuGapTooLarge:
      return "imu_gap_too_large";
    case CameraTimeTransferStatus::kMissingLeftBracket:
      return "missing_left_bracket";
    case CameraTimeTransferStatus::kMissingRightBracket:
      return "missing_right_bracket";
    case CameraTimeTransferStatus::kCrossesImuEpoch:
      return "crosses_imu_epoch";
    case CameraTimeTransferStatus::kInvalidNumericalResult:
      return "invalid_numerical_result";
    case CameraTimeTransferStatus::kIllConditionedTransition:
      return "ill_conditioned_transition";
    case CameraTimeTransferStatus::kCapacityExceeded:
      return "capacity_exceeded";
    case CameraTimeTransferStatus::kResourceFailure:
      return "resource_failure";
  }
  return "unknown";
}

#ifdef LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING
namespace test
{

void failNextCameraTimeTransferFactoryAllocation()
{
  fail_next_factory_allocation.store(true);
}

void failNextCameraTimeTransferAppendAllocation()
{
  fail_next_append_allocation.store(true);
}

void failNextCameraTimeTransferClearResource()
{
  fail_next_clear_resource.store(true);
}

void failNextCameraTimeTransferCaptureAllocation()
{
  fail_next_capture_allocation.store(true);
}

void failNextCameraTimeTransferNumericalAllocation()
{
  fail_next_numerical_allocation.store(true);
}

void setNextCameraTimeTransferEpochToMaximum()
{
  epoch_counter_injection.store(kInjectCounterMaximum);
}

void setNextCameraTimeTransferEpochNearMaximum()
{
  epoch_counter_injection.store(kInjectCounterNearMaximum);
}

void setNextCameraTimeTransferRevisionToMaximum()
{
  revision_counter_injection.store(kInjectCounterMaximum);
}

void setNextCameraTimeTransferRevisionNearMaximum()
{
  revision_counter_injection.store(kInjectCounterNearMaximum);
}

void setNextCameraTimeTransferSnapshotSequenceToMaximum()
{
  snapshot_sequence_counter_injection.store(
    kInjectCounterMaximum);
}

void setNextCameraTimeTransferSnapshotSequenceNearMaximum()
{
  snapshot_sequence_counter_injection.store(
    kInjectCounterNearMaximum);
}

}  // namespace test
#endif

}  // namespace livo
