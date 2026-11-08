#ifndef LIVO_CAMERA_TIME_TRANSFER_H_
#define LIVO_CAMERA_TIME_TRANSFER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace livo
{

inline constexpr Eigen::Index kCameraTimeTransferErrorStateDimension = 18;
inline constexpr Eigen::Index kCameraTimeTransferPoseDimension = 6;
inline constexpr Eigen::Index kCameraTimeTransferThetaOffset = 0;
inline constexpr Eigen::Index kCameraTimeTransferPositionOffset = 3;
inline constexpr Eigen::Index kCameraTimeTransferVelocityOffset = 6;
inline constexpr Eigen::Index kCameraTimeTransferGyroBiasOffset = 9;
inline constexpr Eigen::Index kCameraTimeTransferAccelBiasOffset = 12;
inline constexpr Eigen::Index kCameraTimeTransferGravityOffset = 15;

// FAST-LIVO2 StatesGroup error-state layout:
// [theta(0:3), p(3:6), inv_exposure(6), v(7:10),
//  bg(10:13), ba(13:16), g(16:19)].
inline constexpr Eigen::Index kCameraTimeTransferActiveStateDimension = 19;
inline constexpr Eigen::Index kCameraTimeTransferActiveExposureOffset = 6;
inline constexpr std::uint32_t
  kCameraTimeTransferActiveJacobianRevision = 1U;
inline constexpr char kCameraTimeTransferActiveStateLayoutId[] =
  "fast_livo2_states_group_v1:"
  "theta[0:3],p[3:6],inv_exposure[6],"
  "v[7:10],bg[10:13],ba[13:16],g[16:19]";

using CameraTimeTransferErrorStateMatrix =
  Eigen::Matrix<double, kCameraTimeTransferErrorStateDimension,
    kCameraTimeTransferErrorStateDimension>;
using CameraTimeTransferPoseJacobian =
  Eigen::Matrix<double, kCameraTimeTransferPoseDimension,
    kCameraTimeTransferErrorStateDimension>;
using CameraTimeTransferActivePoseJacobian =
  Eigen::Matrix<double, kCameraTimeTransferPoseDimension,
    kCameraTimeTransferActiveStateDimension>;
using CameraTimeTransferPoseMatrix =
  Eigen::Matrix<double, kCameraTimeTransferPoseDimension,
    kCameraTimeTransferPoseDimension>;

CameraTimeTransferActivePoseJacobian
embedCameraTimeTransferPoseJacobianInActiveState(
  const CameraTimeTransferPoseJacobian & canonical_jacobian);

enum class CameraTimeTransferStatus
{
  kValid = 0,
  kInvalidConfiguration = 1,
  kInvalidInput = 2,
  kTargetOutOfRange = 3,
  kImuNotStrictlyOrdered = 4,
  kImuCoverageMissing = 5,
  kImuGapTooLarge = 6,
  kMissingLeftBracket = 7,
  kMissingRightBracket = 8,
  kCrossesImuEpoch = 9,
  kInvalidNumericalResult = 10,
  kIllConditionedTransition = 11,
  kCapacityExceeded = 12,
  kResourceFailure = 13
};

enum class CameraTimeTransferAppendStatus
{
  kAccepted = 0,
  kNewEpoch = 1,
  kInvalidSample = 2,
  kInvalidConfiguration = 3,
  kCapacityExceeded = 4,
  kResourceFailure = 5,
  kEpochExhausted = 6,
  kRevisionExhausted = 7,
  kAuthorityUnavailable = 8
};

enum class CameraTimeTransferClearStatus
{
  kCleared,
  kInvalidConfiguration,
  kResourceFailure,
  kEpochExhausted,
  kRevisionExhausted,
  kAuthorityUnavailable
};

enum class CameraTimeTransferEpochCaptureStatus
{
  kCaptured,
  kInvalidAuthority,
  kInvalidConfiguration,
  kInvalidInput,
  kEmpty,
  kWindowNotCovered,
  kCapacityExceeded,
  kResourceFailure,
  kSequenceExhausted
};

enum class CameraTimeTransferDirection
{
  kIdentity,
  kForward,
  kBackward
};

enum class CameraTimeTransferUncertaintySemantics
{
  kNoProcessNoise,
  kForwardIndependentProcessNoise,
  kBackwardCorrelatedTransportApproximation
};

struct CameraTimeTransferState
{
  double timestamp_sec{0.0};
  Eigen::Matrix3d rotation_world_from_imu{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity_world_mps{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gyro_bias_radps{Eigen::Vector3d::Zero()};
  Eigen::Vector3d accel_bias_mps2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d gravity_world_mps2{Eigen::Vector3d::Zero()};
  double timestamp_stddev_sec{0.0};
};

struct CameraTimeTransferImuSample
{
  double timestamp_sec{0.0};
  Eigen::Vector3d angular_velocity_radps{Eigen::Vector3d::Zero()};
  Eigen::Vector3d specific_force_mps2{Eigen::Vector3d::Zero()};
};

struct CameraTimeTransferContinuousNoiseDensity
{
  double gyro_measurement_rad_per_sqrt_sec{0.0};
  double accel_measurement_mps_per_sqrt_sec{0.0};
  double gyro_bias_random_walk_radps_per_sqrt_sec{0.0};
  double accel_bias_random_walk_mps2_per_sqrt_sec{0.0};
};

struct CameraTimeTransferConfig
{
  double max_abs_transfer_sec{0.05};
  double max_imu_gap_sec{0.02};
  double imu_timestamp_stddev_sec{0.0};
  double ring_buffer_duration_sec{0.25};
  double epoch_jump_threshold_sec{0.10};
  double gyro_noise_stddev_radps{0.01};
  double accel_noise_stddev_mps2{0.10};
  double gyro_bias_random_walk_stddev_radps_per_sqrt_sec{1.0e-4};
  double accel_bias_random_walk_stddev_mps2_per_sqrt_sec{1.0e-3};
  double maximum_transition_condition_number{1.0e12};
  std::size_t maximum_buffered_samples{4096U};

  std::optional<CameraTimeTransferContinuousNoiseDensity>
  validatedContinuousNoiseDensity() const;
};

struct ConditionalPoseOnlyTimeTransfer
{
  static constexpr bool kShadowOnly = true;

  // This adapter conditions on anchor v/bg/ba/g being exact and fixed. It is
  // retained only for existing shadow diagnostics and must never be used to
  // map an active normal equation.
  CameraTimeTransferPoseMatrix target_pose_wrt_anchor_pose{
    CameraTimeTransferPoseMatrix::Constant(
      std::numeric_limits<double>::quiet_NaN())};

  bool valid() const
  {
    return target_pose_wrt_anchor_pose.allFinite();
  }
};

struct CameraPoseTransferRaw
{
  CameraTimeTransferStatus status{CameraTimeTransferStatus::kInvalidInput};
  CameraTimeTransferDirection direction{
    CameraTimeTransferDirection::kIdentity};
  CameraTimeTransferUncertaintySemantics uncertainty_semantics{
    CameraTimeTransferUncertaintySemantics::kNoProcessNoise};
  double anchor_timestamp_sec{0.0};
  double target_timestamp_sec{0.0};
  double delta_time_sec{0.0};
  std::uint64_t imu_epoch{0U};
  double time_uncertainty_stddev_sec{
    std::numeric_limits<double>::infinity()};
  double maximum_imu_gap_sec{
    std::numeric_limits<double>::infinity()};
  double transition_condition_number{
    std::numeric_limits<double>::infinity()};
  Eigen::Matrix3d rotation_world_from_imu{
    Eigen::Matrix3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Vector3d position_world_m{
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Vector3d velocity_world_mps{
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};

  // Error-state order is fixed as [theta, p, v, bg, ba, g], with a right-local
  // rotation error and additive errors for all vector states.
  CameraTimeTransferErrorStateMatrix error_state_transition{
    CameraTimeTransferErrorStateMatrix::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  CameraTimeTransferPoseJacobian target_pose_wrt_anchor_state{
    CameraTimeTransferPoseJacobian::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  CameraTimeTransferActivePoseJacobian
    target_pose_wrt_active_anchor_state{
      CameraTimeTransferActivePoseJacobian::Constant(
        std::numeric_limits<double>::quiet_NaN())};

  // Discrete process noise accumulated along the transfer. The 6x6 pose
  // covariance is exactly the [theta, p] principal block of this matrix; it
  // deliberately excludes timestamp uncertainty.
  CameraTimeTransferErrorStateMatrix process_noise_covariance{
    CameraTimeTransferErrorStateMatrix::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  CameraTimeTransferPoseMatrix pose_transfer_covariance{
    CameraTimeTransferPoseMatrix::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  CameraTimeTransferPoseMatrix time_alignment_pose_covariance{
    CameraTimeTransferPoseMatrix::Constant(
      std::numeric_limits<double>::quiet_NaN())};

  ConditionalPoseOnlyTimeTransfer conditional_pose_only;

  bool internallyValid() const;
};

// Opaque validation token. Public code can inspect/copy a successful token,
// but only CameraTimeTransfer can mint one.
class CameraPoseTransfer
{
public:
  CameraPoseTransfer(const CameraPoseTransfer &) = default;
  CameraPoseTransfer(CameraPoseTransfer &&) = default;
  CameraPoseTransfer & operator=(const CameraPoseTransfer &) = default;
  CameraPoseTransfer & operator=(CameraPoseTransfer &&) = default;

  const CameraPoseTransferRaw & raw() const;

private:
  explicit CameraPoseTransfer(const CameraPoseTransferRaw & raw);
  bool matches(const CameraPoseTransferRaw & raw) const;

  CameraPoseTransferRaw sealed_raw_;

  friend class CameraTimeTransfer;
  friend struct CameraPoseTransferData;
};

// Public diagnostic DTO. Mutating any raw field after validation invalidates
// the attached token; invalid computations never carry a token.
struct CameraPoseTransferData : public CameraPoseTransferRaw
{
  bool valid() const;
  std::optional<CameraPoseTransfer> validatedValue() const;

private:
  std::optional<CameraPoseTransfer> validation_token_;

  void sealSuccessful();

  friend class CameraTimeTransfer;
  friend class CameraTimeTransferEpochReceipt;
};

class CameraTimeTransfer;
class CameraTimeTransferEpochReceipt;
class EpochBoundCameraPoseTransfer;
struct CameraTimeTransferEpochTransferResult;

// Opaque identity for one CameraTimeTransfer authority lineage. It carries no
// caller-chosen numeric identity, remains O(1) to copy, and stays usable for
// read-only receipt audit after the mutable authority handle is gone.
class CameraTimeTransferAuthorityIdentity
{
public:
  CameraTimeTransferAuthorityIdentity(
    const CameraTimeTransferAuthorityIdentity &) = default;
  CameraTimeTransferAuthorityIdentity(
    CameraTimeTransferAuthorityIdentity &&) = default;
  CameraTimeTransferAuthorityIdentity & operator=(
    const CameraTimeTransferAuthorityIdentity &) = default;
  CameraTimeTransferAuthorityIdentity & operator=(
    CameraTimeTransferAuthorityIdentity &&) = default;

  bool valid() const;

private:
  struct Data;
  explicit CameraTimeTransferAuthorityIdentity(
    std::shared_ptr<const Data> data);

  std::shared_ptr<const Data> data_;

  friend class CameraTimeTransfer;
  friend class CameraTimeTransferEpochReceipt;
};

// Immutable epoch/window snapshot. It owns all data needed for deterministic
// transfer after the live authority has moved, changed, or been destroyed.
class CameraTimeTransferEpochReceipt
{
public:
  CameraTimeTransferEpochReceipt(
    const CameraTimeTransferEpochReceipt &) = default;
  CameraTimeTransferEpochReceipt(
    CameraTimeTransferEpochReceipt &&) = default;
  CameraTimeTransferEpochReceipt & operator=(
    const CameraTimeTransferEpochReceipt &) = default;
  CameraTimeTransferEpochReceipt & operator=(
    CameraTimeTransferEpochReceipt &&) = default;

  bool valid() const;
  bool belongsTo(
    const CameraTimeTransferAuthorityIdentity & identity) const;
  std::uint64_t epoch() const;
  std::uint64_t streamRevision() const;
  std::uint64_t snapshotSequence() const;
  std::size_t bufferedSamples() const;

  CameraTimeTransferEpochTransferResult transfer(
    const CameraTimeTransferState & anchor,
    double target_timestamp_sec,
    double target_timestamp_stddev_sec = 0.0) const;

private:
  struct Data;
  explicit CameraTimeTransferEpochReceipt(
    std::shared_ptr<const Data> data);

  std::shared_ptr<const Data> data_;

  friend class CameraTimeTransfer;
  friend class EpochBoundCameraPoseTransfer;
};

// Production-bound pose transfer. Legacy raw/sealed transfer values have no
// conversion path to this type.
class EpochBoundCameraPoseTransfer
{
public:
  EpochBoundCameraPoseTransfer(
    const EpochBoundCameraPoseTransfer &) = default;
  EpochBoundCameraPoseTransfer(
    EpochBoundCameraPoseTransfer &&) = default;
  EpochBoundCameraPoseTransfer & operator=(
    const EpochBoundCameraPoseTransfer &) = default;
  EpochBoundCameraPoseTransfer & operator=(
    EpochBoundCameraPoseTransfer &&) = default;

  bool valid() const;
  bool wasProducedBy(
    const CameraTimeTransferEpochReceipt & receipt) const;
  const CameraPoseTransfer & value() const;
  const CameraPoseTransferRaw & raw() const;

private:
  EpochBoundCameraPoseTransfer(
    CameraPoseTransfer value,
    std::shared_ptr<const void> receipt_identity);

  CameraPoseTransfer value_;
  std::shared_ptr<const void> receipt_identity_;

  friend class CameraTimeTransferEpochReceipt;
};

struct CameraTimeTransferEpochTransferResult
{
  CameraTimeTransferStatus status{
    CameraTimeTransferStatus::kInvalidInput};
  std::optional<EpochBoundCameraPoseTransfer> value;
};

struct CameraTimeTransferEpochCaptureResult
{
  CameraTimeTransferEpochCaptureStatus status{
    CameraTimeTransferEpochCaptureStatus::kInvalidAuthority};
  std::optional<CameraTimeTransferEpochReceipt> receipt;
};

class CameraTimeTransfer
{
public:
  explicit CameraTimeTransfer(
    CameraTimeTransferConfig config = CameraTimeTransferConfig{});
  ~CameraTimeTransfer();

  CameraTimeTransfer(const CameraTimeTransfer &) = delete;
  CameraTimeTransfer & operator=(const CameraTimeTransfer &) = delete;
  CameraTimeTransfer(CameraTimeTransfer &&) noexcept;
  CameraTimeTransfer & operator=(CameraTimeTransfer &&) noexcept;

  bool configValid() const;

  CameraTimeTransferAppendStatus appendImu(
    const CameraTimeTransferImuSample & sample);

  CameraTimeTransferClearStatus clear();

  std::uint64_t epoch() const;
  std::size_t bufferedSamples() const;
  std::optional<CameraTimeTransferAuthorityIdentity>
  authorityIdentity() const;

  CameraTimeTransferEpochCaptureResult captureEpoch();
  CameraTimeTransferEpochCaptureResult captureWindow(
    double interval_start_timestamp_sec,
    double interval_end_timestamp_sec);

  CameraPoseTransferData transfer(
    const CameraTimeTransferState & anchor,
    double target_timestamp_sec,
    double target_timestamp_stddev_sec = 0.0) const;

  CameraPoseTransferData transferWithSamples(
    const CameraTimeTransferState & anchor,
    const std::vector<CameraTimeTransferImuSample> & imu_samples,
    double target_timestamp_sec,
    double target_timestamp_stddev_sec = 0.0) const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

const char * cameraTimeTransferStatusName(
  CameraTimeTransferStatus status);

}  // namespace livo

#endif  // LIVO_CAMERA_TIME_TRANSFER_H_
