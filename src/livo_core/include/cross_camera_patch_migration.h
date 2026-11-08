#ifndef LIVO_CROSS_CAMERA_PATCH_MIGRATION_H_
#define LIVO_CROSS_CAMERA_PATCH_MIGRATION_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "camera_time_transfer.h"
#include "multi_view_landmark.h"
#include "photometric_calibration.h"

namespace livo
{

struct CameraEndpointUncertainty;
struct MigrationEndpoint;
struct FrozenAcceptedSourcePose;
struct RawIntensityNoiseModel;
struct MigrationTarget;

enum class ProjectionAdapterStatus
{
  kValid,
  kInvalidModel,
  kBehindCamera,
  kOutOfDomain,
  kNonFinite
};

struct CameraRayResult
{
  ProjectionAdapterStatus status{
    ProjectionAdapterStatus::kInvalidModel};
  Eigen::Vector3d direction_camera{
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
};

struct CameraRayJacobianResult
{
  ProjectionAdapterStatus status{
    ProjectionAdapterStatus::kInvalidModel};
  Eigen::Matrix<double, 3, 2> jacobian{
    Eigen::Matrix<double, 3, 2>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
};

struct CameraProjectionResult
{
  ProjectionAdapterStatus status{
    ProjectionAdapterStatus::kInvalidModel};
  Eigen::Vector2d pixel{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
};

struct CameraProjectionJacobianResult
{
  ProjectionAdapterStatus status{
    ProjectionAdapterStatus::kInvalidModel};
  Eigen::Matrix<double, 2, 3> jacobian{
    Eigen::Matrix<double, 2, 3>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
};

class CameraProjectionAdapter
{
public:
  virtual ~CameraProjectionAdapter() = default;

  virtual bool valid() const = 0;
  // Read-only structural version of the concrete model. Equality with an
  // endpoint revision prevents accidental stale-model use, but is not a
  // calibration-authority receipt.
  virtual std::uint64_t modelRevision() const noexcept = 0;
  virtual CameraRayResult unproject(
    const Eigen::Vector2d & pixel) const = 0;
  virtual CameraRayJacobianResult unprojectJacobian(
    const Eigen::Vector2d & pixel) const
  {
    (void)pixel;
    return CameraRayJacobianResult{};
  }
  // A finite positive-depth point may project outside the declared image and
  // still return kValid. Image/border classification belongs to contains().
  virtual CameraProjectionResult project(
    const Eigen::Vector3d & point_camera) const = 0;
  virtual CameraProjectionJacobianResult projectJacobian(
    const Eigen::Vector3d & point_camera) const = 0;
  // Classifies the half-open image interior after applying border.
  virtual bool contains(
    const Eigen::Vector2d & pixel,
    int border) const = 0;
};

class CameraEndpointUncertaintyReceipt
{
public:
  CameraEndpointUncertaintyReceipt(
    const CameraEndpointUncertaintyReceipt &) = default;
  CameraEndpointUncertaintyReceipt(
    CameraEndpointUncertaintyReceipt &&) = default;
  CameraEndpointUncertaintyReceipt & operator=(
    const CameraEndpointUncertaintyReceipt &) = default;
  CameraEndpointUncertaintyReceipt & operator=(
    CameraEndpointUncertaintyReceipt &&) = default;

private:
  explicit CameraEndpointUncertaintyReceipt(
    const MigrationEndpoint & endpoint);
  bool matches(
    const MigrationEndpoint & endpoint) const;

  std::uint8_t camera_id_{0U};
  std::shared_ptr<const CameraProjectionAdapter> camera_model_;
  std::uint64_t endpoint_camera_model_revision_{0U};
  std::uint64_t endpoint_rig_revision_{0U};
  Eigen::Matrix3d rotation_imu_from_camera_{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_imu_from_camera_{
    Eigen::Vector3d::Zero()};
  Eigen::Matrix2d observation_pixel_covariance_px2_{
    Eigen::Matrix2d::Zero()};
  Eigen::Matrix2d camera_model_equivalent_pixel_covariance_px2_{
    Eigen::Matrix2d::Zero()};
  Eigen::Matrix<double, 6, 6> extrinsic_covariance_{
    Eigen::Matrix<double, 6, 6>::Zero()};
  std::uint64_t camera_model_revision_{0U};
  std::uint64_t rig_revision_{0U};
  std::uint64_t revision_{0U};

  friend struct CameraEndpointUncertainty;
};

// Pixel covariances are in px^2. Camera-model uncertainty is represented as
// its equivalent local pixel covariance. Extrinsic covariance uses the
// [right-rotation, IMU-frame translation] tangent of T_IC.
struct CameraEndpointUncertainty
{
  Eigen::Matrix2d observation_pixel_covariance_px2{
    Eigen::Matrix2d::Zero()};
  Eigen::Matrix2d camera_model_equivalent_pixel_covariance_px2{
    Eigen::Matrix2d::Zero()};
  Eigen::Matrix<double, 6, 6> extrinsic_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  std::uint64_t camera_model_revision{0U};
  std::uint64_t rig_revision{0U};
  std::uint64_t revision{0U};

  bool valid() const;
  bool productionReady(
    const MigrationEndpoint & endpoint) const;

private:
  std::optional<CameraEndpointUncertaintyReceipt>
    validation_receipt_;

  void sealValidationReceipt(
    const MigrationEndpoint & endpoint);
};

struct MigrationEndpoint
{
  std::uint8_t camera_id{0U};
  // Immutable shared ownership makes endpoint values self-contained across
  // queues and owner destruction. Receipts additionally bind the control
  // block, pointed-to adapter, and structural revision.
  std::shared_ptr<const CameraProjectionAdapter> camera_model;
  std::uint64_t camera_model_revision{0U};
  std::uint64_t rig_revision{0U};

  // T_IC: camera coordinates to IMU coordinates.
  Eigen::Matrix3d rotation_imu_from_camera{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_imu_from_camera{
    Eigen::Vector3d::Zero()};
  CameraEndpointUncertainty uncertainty;

  // Structural consistency only; this does not prove trusted rig authority.
  bool modelRevisionMatchesAdapter() const noexcept;
  bool sharesCameraModelIdentityWith(
    const MigrationEndpoint & other) const noexcept;
};

struct MigrationImage
{
  int width{0};
  int height{0};
  std::vector<double> pixels;
};

class FrozenSourceAcceptanceReceipt
{
public:
  FrozenSourceAcceptanceReceipt(
    const FrozenSourceAcceptanceReceipt &) = default;
  FrozenSourceAcceptanceReceipt(
    FrozenSourceAcceptanceReceipt &&) = default;
  FrozenSourceAcceptanceReceipt & operator=(
    const FrozenSourceAcceptanceReceipt &) = default;
  FrozenSourceAcceptanceReceipt & operator=(
    FrozenSourceAcceptanceReceipt &&) = default;

private:
  explicit FrozenSourceAcceptanceReceipt(
    const FrozenAcceptedSourcePose & frozen_pose);
  bool matches(
    const FrozenAcceptedSourcePose & frozen_pose) const;

  double exposure_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  Eigen::Matrix3d rotation_world_from_imu_{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_world_m_{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 6, 6> pose_covariance_{
    Eigen::Matrix<double, 6, 6>::Zero()};
  BatchId accepted_batch_id_;
  StateRevision accepted_state_revision_;
  MapRevision store_revision_;
  LandmarkId landmark_id_{0U};
  ObservationId observation_id_{0U};
  std::uint64_t observation_revision_{0U};
  std::uint64_t selector_snapshot_revision_{0U};
  std::uint64_t selector_receipt_id_{0U};
  std::uint64_t store_receipt_id_{0U};

  friend struct FrozenAcceptedSourcePose;
};

class FrozenSourceIndependenceProof
{
public:
  FrozenSourceIndependenceProof(
    const FrozenSourceIndependenceProof &) = default;
  FrozenSourceIndependenceProof(
    FrozenSourceIndependenceProof &&) = default;
  FrozenSourceIndependenceProof & operator=(
    const FrozenSourceIndependenceProof &) = default;
  FrozenSourceIndependenceProof & operator=(
    FrozenSourceIndependenceProof &&) = default;

private:
  FrozenSourceIndependenceProof(
    const FrozenAcceptedSourcePose & frozen_pose,
    std::uint64_t active_state_revision,
    std::uint64_t provider_revision);
  bool matches(
    const FrozenAcceptedSourcePose & frozen_pose,
    std::uint64_t active_state_revision) const;

  LandmarkId landmark_id_{0U};
  ObservationId observation_id_{0U};
  std::uint64_t observation_revision_{0U};
  std::uint64_t active_state_revision_{0U};
  std::uint64_t provider_revision_{0U};

  friend struct FrozenAcceptedSourcePose;
};

// Immutable pose copied from an accepted observation snapshot. It is a
// nuisance measurement with its own uncertainty, never a second path from the
// current 19D anchor state.
struct FrozenAcceptedSourcePose
{
  double exposure_timestamp_sec{
    std::numeric_limits<double>::quiet_NaN()};
  Eigen::Matrix3d rotation_world_from_imu{
    Eigen::Matrix3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Vector3d position_world_m{
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix<double, 6, 6> pose_covariance{
    Eigen::Matrix<double, 6, 6>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  BatchId accepted_batch_id;
  StateRevision accepted_state_revision;
  MapRevision store_revision;
  LandmarkId landmark_id{0U};
  ObservationId observation_id{0U};
  std::uint64_t observation_revision{0U};
  std::uint64_t selector_snapshot_revision{0U};
  std::uint64_t selector_receipt_id{0U};
  std::uint64_t store_receipt_id{0U};

  bool valid() const;
  bool acceptedSnapshotProofReady() const;
  bool independentFromActivePrior(
    std::uint64_t active_state_revision) const;
  bool validFor(
    BatchId current_batch_id,
    LandmarkId expected_landmark_id,
    ObservationId expected_observation_id) const;

private:
  std::optional<FrozenSourceAcceptanceReceipt>
    acceptance_receipt_;
  std::optional<FrozenSourceIndependenceProof>
    independence_proof_;

  void sealAcceptedSnapshot();
  void sealIndependent(
    std::uint64_t active_state_revision,
    std::uint64_t provider_revision);
};

class EndpointNoiseConditionerGate
{
public:
  EndpointNoiseConditionerGate(
    const EndpointNoiseConditionerGate &) = default;
  EndpointNoiseConditionerGate(
    EndpointNoiseConditionerGate &&) = default;
  EndpointNoiseConditionerGate & operator=(
    const EndpointNoiseConditionerGate &) = default;
  EndpointNoiseConditionerGate & operator=(
    EndpointNoiseConditionerGate &&) = default;

private:
  EndpointNoiseConditionerGate(
    std::uint64_t source_imu_epoch,
    std::uint64_t target_imu_epoch,
    std::uint64_t source_imu_history_revision,
    std::uint64_t target_imu_history_revision);

  std::uint64_t imu_epoch_{0U};
  std::uint64_t imu_history_revision_{0U};
  bool same_imu_history_{false};

  friend class EndpointJointPoseCovariance;
};

// Joint covariance of [source frozen pose, target transferred pose]. Only the
// endpoint noise-conditioning boundary may create a production token; public
// callers can inspect but cannot write the cross block.
class EndpointJointPoseCovariance
{
public:
  EndpointJointPoseCovariance(
    const EndpointJointPoseCovariance &) = default;
  EndpointJointPoseCovariance(
    EndpointJointPoseCovariance &&) = default;
  EndpointJointPoseCovariance & operator=(
    const EndpointJointPoseCovariance &) = default;
  EndpointJointPoseCovariance & operator=(
    EndpointJointPoseCovariance &&) = default;

  const Eigen::Matrix<double, 12, 12> & covariance() const;
  bool valid() const;
  bool productionReadyFor(
    BatchId batch_id,
    ObservationId source_observation_id,
    std::uint64_t source_observation_revision,
    double source_timestamp_sec,
    double target_timestamp_sec,
    std::uint64_t target_imu_epoch,
    std::uint64_t active_state_revision) const;

private:
  EndpointJointPoseCovariance(
    const Eigen::Matrix<double, 12, 12> & covariance,
    std::uint64_t revision,
    BatchId batch_id,
    ObservationId source_observation_id,
    std::uint64_t source_observation_revision,
    double source_timestamp_sec,
    double target_timestamp_sec,
    std::uint64_t active_state_revision,
    const EndpointNoiseConditionerGate & gate);

  Eigen::Matrix<double, 12, 12> covariance_{
    Eigen::Matrix<double, 12, 12>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  std::uint64_t revision_{0U};
  BatchId batch_id_;
  ObservationId source_observation_id_{0U};
  std::uint64_t source_observation_revision_{0U};
  double source_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  double target_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  std::uint64_t imu_epoch_{0U};
  std::uint64_t imu_history_revision_{0U};
  bool same_imu_history_{false};
  std::uint64_t active_state_revision_{0U};

  friend class CrossCameraPatchMigration;
};

class RawIntensityNoiseReceipt
{
public:
  RawIntensityNoiseReceipt(
    const RawIntensityNoiseReceipt &) = default;
  RawIntensityNoiseReceipt(
    RawIntensityNoiseReceipt &&) = default;
  RawIntensityNoiseReceipt & operator=(
    const RawIntensityNoiseReceipt &) = default;
  RawIntensityNoiseReceipt & operator=(
    RawIntensityNoiseReceipt &&) = default;

private:
  explicit RawIntensityNoiseReceipt(
    const RawIntensityNoiseModel & noise);
  bool matches(const RawIntensityNoiseModel & noise) const;

  double standard_deviation_raw_{0.0};
  std::uint64_t calibration_revision_{0U};

  friend struct RawIntensityNoiseModel;
};

// Noise is expressed in raw sensor-intensity units. Photometric Jacobians
// convert it to radiance variance at each residual.
struct RawIntensityNoiseModel
{
  double standard_deviation_raw{1.0};
  std::uint64_t calibration_revision{0U};

  bool usable() const;
  bool productionReady(
    std::uint64_t expected_calibration_revision) const;

private:
  std::optional<RawIntensityNoiseReceipt>
    validation_receipt_;

  void sealValidationReceipt();
};

struct MigrationSource
{
  MigrationEndpoint endpoint;
  FrozenAcceptedSourcePose frozen_pose;
  Eigen::Vector2d source_pixel{Eigen::Vector2d::Zero()};
  std::uint8_t source_level{0U};
  int patch_width{0};
  std::vector<double> source_patch;
  PhotometricFrameMetadata photo_metadata;
  RawIntensityNoiseModel raw_intensity_noise;
  GeometryVersion geometry_version;
  std::uint64_t observation_camera_model_revision{0U};
  std::uint64_t observation_rig_revision{0U};
  std::uint64_t
    observation_photometric_calibration_revision{0U};
};

class TargetDepthFrameReceipt
{
public:
  TargetDepthFrameReceipt(
    const TargetDepthFrameReceipt &) = default;
  TargetDepthFrameReceipt(
    TargetDepthFrameReceipt &&) = default;
  TargetDepthFrameReceipt & operator=(
    const TargetDepthFrameReceipt &) = default;
  TargetDepthFrameReceipt & operator=(
    TargetDepthFrameReceipt &&) = default;

private:
  explicit TargetDepthFrameReceipt(
    const MigrationTarget & target);
  bool matches(const MigrationTarget & target) const;

  double exposure_timestamp_sec_{
    std::numeric_limits<double>::quiet_NaN()};
  std::vector<double> depth_m_;
  std::uint64_t visibility_revision_{0U};
  std::uint64_t frame_camera_model_revision_{0U};
  std::uint64_t frame_rig_revision_{0U};

  friend struct MigrationTarget;
};

struct MigrationTarget
{
  MigrationEndpoint endpoint;
  double exposure_timestamp_sec{0.0};
  CameraPoseTransferData pose_transfer;
  MigrationImage image;
  std::vector<double> depth_m;
  PhotometricFrameMetadata photo_metadata;
  RawIntensityNoiseModel raw_intensity_noise;
  std::uint64_t visibility_revision{0U};
  std::uint64_t frame_camera_model_revision{0U};
  std::uint64_t frame_rig_revision{0U};
  std::uint64_t frame_photometric_calibration_revision{0U};

  bool depthFrameProofReady() const;

private:
  std::optional<TargetDepthFrameReceipt>
    depth_frame_receipt_;

  void sealDepthFrameReceipt();
};

struct MigrationPlane
{
  GeometrySupportKey support_key;
  GeometryVersion geometry_version;
  Eigen::Vector3d point_world_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d point_covariance_m2{Eigen::Matrix3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d normal_covariance{Eigen::Matrix3d::Zero()};
};

struct MigrationGateConfig
{
  double maximum_observation_age_sec{0.5};
  double minimum_ray_plane_cosine{0.05};
  double minimum_depth_m{0.2};
  double maximum_depth_m{50.0};
  double maximum_depth_disagreement_m{0.5};
  bool require_target_depth{false};
  int image_border{2};
  std::size_t minimum_valid_pixels{8U};
  double minimum_patch_standard_deviation{2.0};
  double minimum_zncc{0.7};
  double maximum_residual_rmse{0.1};
  double maximum_warp_condition_number{8.0};
  double minimum_warp_singular_value{0.125};
  double maximum_warp_singular_value{8.0};
  double minimum_positive_warp_determinant{1.0e-6};
  double minimum_gradient_direction_coverage{0.02};
  double maximum_saturated_pixel_fraction{0.25};
  double maximum_dark_pixel_fraction{0.25};
  bool allow_relative_affine_experiment{false};
  double robust_huber_delta_in_standard_deviations{3.0};
  double affine_gain_prior_standard_deviation{0.5};
  double affine_bias_prior_standard_deviation{0.25};
  double maximum_affine_schur_condition_number{1.0e8};
  double minimum_information_eigenvalue{1.0e-12};
};

struct MigrationRequest
{
  BatchId batch_id;
  std::uint64_t active_state_revision{0U};
  LandmarkId landmark_id{0U};
  ObservationId source_observation_id{0U};
  MigrationSource source;
  MigrationTarget target;
  std::optional<EndpointJointPoseCovariance>
    endpoint_joint_pose_covariance;
  MigrationPlane plane;
  MigrationGateConfig gates;
  PhotometricMode photometric_mode{
    PhotometricMode::kCalibratedDirect};
};

enum class MigrationStatus
{
  kReady,
  kStaleLandmark,
  kModelRevisionMismatch,
  kRigRevisionMismatch,
  kPhotometricRevisionMismatch,
  kInvalidSourceModel,
  kInvalidTargetModel,
  kInvalidTimeTransfer,
  kPhotometricUnavailable,
  kInvalidPlane,
  kRayPlaneDegenerate,
  kBehindCamera,
  kOutOfFrame,
  kWarpIllConditioned,
  kOccluded,
  kLowTexture,
  kNccRejected,
  kAffineIllConditioned,
  kResidualRejected,
  kNonFinite
};

struct MigrationResult
{
  MigrationStatus status{MigrationStatus::kNonFinite};
  bool consumable{false};
  bool shadow_only{true};
  LandmarkId landmark_id{0U};
  ObservationId source_observation_id{0U};
  std::uint8_t source_camera_id{0U};
  std::uint8_t target_camera_id{0U};
  Eigen::Vector2d target_pixel{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix2d warp{
    Eigen::Matrix2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  std::vector<double> warped_reference_patch;
  std::vector<double> current_patch;
  std::size_t valid_pixel_count{0U};
  double warp_condition_number{
    std::numeric_limits<double>::infinity()};
  double zncc{
    std::numeric_limits<double>::quiet_NaN()};
  double residual_rmse{
    std::numeric_limits<double>::quiet_NaN()};
  double warp_determinant{
    std::numeric_limits<double>::quiet_NaN()};
  double gradient_direction_coverage{0.0};
  PhotometricPatchResult photometric;

  bool geometry_ready() const
  {
    return status == MigrationStatus::kReady;
  }

  bool ready() const
  {
    return geometry_ready() && consumable && !shadow_only;
  }

private:
  bool target_visibility_validated_{false};
  bool photometric_quality_metadata_validated_{false};

  friend class CrossCameraPatchMigration;
};

enum class PatchContributionStatus
{
  kReady,
  kExperimentalShadowOnly,
  kDuplicateLandmark,
  kMigrationRejected,
  kProjectionJacobianUnavailable,
  kPhotometricUnavailable,
  kAffineIllConditioned,
  kDegenerateInformation,
  kNonFinite
};

using PatchActiveStateJacobian =
  Eigen::Matrix<
    double, Eigen::Dynamic,
    kCameraTimeTransferActiveStateDimension>;
using PatchActiveStateInformation =
  Eigen::Matrix<
    double,
    kCameraTimeTransferActiveStateDimension,
    kCameraTimeTransferActiveStateDimension>;
using PatchActiveStateGradient =
  Eigen::Matrix<
    double,
    kCameraTimeTransferActiveStateDimension, 1>;

class ProductionPatchCapabilities
{
public:
  ProductionPatchCapabilities(
    const ProductionPatchCapabilities &) = default;
  ProductionPatchCapabilities(
    ProductionPatchCapabilities &&) = default;
  ProductionPatchCapabilities & operator=(
    const ProductionPatchCapabilities &) = default;
  ProductionPatchCapabilities & operator=(
    ProductionPatchCapabilities &&) = default;

  bool frozenSourcePriorIndependence() const;
  bool jointEndpointPoseCovariance() const;
  bool sourceRawIntensityNoise() const;
  bool targetRawIntensityNoise() const;
  bool sourceEndpointUncertainty() const;
  bool targetEndpointUncertainty() const;
  bool sourceAdapterJacobian() const;
  bool targetAdapterJacobian() const;
  bool targetVisibility() const;
  bool photometricQualityMetadata() const;
  bool orientationPreservingWarp() const;
  bool gradientDirectionCoverage() const;
  bool all() const;

private:
  ProductionPatchCapabilities() = default;

  bool frozen_source_prior_independence_{false};
  bool joint_endpoint_pose_covariance_{false};
  bool source_raw_intensity_noise_{false};
  bool target_raw_intensity_noise_{false};
  bool source_endpoint_uncertainty_{false};
  bool target_endpoint_uncertainty_{false};
  bool source_adapter_jacobian_{false};
  bool target_adapter_jacobian_{false};
  bool target_visibility_{false};
  bool photometric_quality_metadata_{false};
  bool orientation_preserving_warp_{false};
  bool gradient_direction_coverage_{false};

  friend struct WhitenedPatchContribution;
  friend class CrossCameraPatchMigration;
};

// Canonical, value-semantic least-squares contribution. The exposed factor is
// authoritative: information == J'J, gradient == J'r, and squared_error ==
// r'r. Robust-affine mode returns the square-root factor of its Schur
// complement and remains shadow-only.
struct WhitenedPatchContribution
{
  PatchContributionStatus status{
    PatchContributionStatus::kNonFinite};
  MigrationStatus migration_status{MigrationStatus::kNonFinite};
  bool shadow_only{true};
  bool experimental{false};
  BatchId batch_id;
  LandmarkId landmark_id{0U};
  ObservationId source_observation_id{0U};
  std::uint8_t source_camera_id{0U};
  std::uint8_t target_camera_id{0U};
  std::size_t raw_valid_pixel_count{0U};
  Eigen::VectorXd whitened_residual;
  PatchActiveStateJacobian whitened_jacobian{
    PatchActiveStateJacobian::Zero(
      0, kCameraTimeTransferActiveStateDimension)};
  PatchActiveStateInformation information{
    PatchActiveStateInformation::Zero()};
  PatchActiveStateGradient gradient{
    PatchActiveStateGradient::Zero()};
  double normalized_squared_error{0.0};
  double information_units{0.0};
  double minimum_effective_variance{0.0};
  double maximum_effective_variance{0.0};

  bool productionActiveReady() const
  {
    return production_active_ready_;
  }

  const ProductionPatchCapabilities &
  productionCapabilities() const
  {
    return production_capabilities_;
  }

  bool zeroContribution() const
  {
    return
      !production_active_ready_ &&
      whitened_residual.size() == 0 &&
      whitened_jacobian.rows() == 0 &&
      information.isZero(0.0) &&
      gradient.isZero(0.0) &&
      normalized_squared_error == 0.0 &&
      information_units == 0.0;
  }

private:
  bool production_active_ready_{false};
  ProductionPatchCapabilities production_capabilities_;

  friend class CrossCameraPatchMigration;
};

class CrossCameraPatchMigration
{
public:
  explicit CrossCameraPatchMigration(
    const PhotometricCalibration & photometric_calibration);

  MigrationResult prepare(
    const MigrationRequest & request) const;

  // Returns one output per input and atomically enforces at most one nonzero
  // information unit for each (BatchId, LandmarkId).
  std::vector<WhitenedPatchContribution> linearizeBatch(
    const std::vector<MigrationRequest> & requests) const;

private:
  PhotometricCalibration photometric_calibration_;
};

const char * migrationStatusName(MigrationStatus status);
const char * patchContributionStatusName(
  PatchContributionStatus status);

}  // namespace livo

#endif  // LIVO_CROSS_CAMERA_PATCH_MIGRATION_H_
