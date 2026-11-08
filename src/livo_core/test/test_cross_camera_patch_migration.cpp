#include "cross_camera_patch_migration.h"
#include "pinhole_camera_projection_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace
{

static_assert(
  !std::is_default_constructible_v<
    livo::EndpointJointPoseCovariance>);
static_assert(
  !std::is_aggregate_v<
    livo::EndpointJointPoseCovariance>);
static_assert(
  !std::is_default_constructible_v<
    livo::EndpointNoiseConditionerGate>);
static_assert(
  !std::is_default_constructible_v<
    livo::FrozenSourceAcceptanceReceipt>);
static_assert(
  !std::is_default_constructible_v<
    livo::FrozenSourceIndependenceProof>);
static_assert(
  !std::is_default_constructible_v<
    livo::CameraEndpointUncertaintyReceipt>);
static_assert(
  !std::is_default_constructible_v<
    livo::RawIntensityNoiseReceipt>);
static_assert(
  !std::is_default_constructible_v<
    livo::TargetDepthFrameReceipt>);
static_assert(
  !std::is_default_constructible_v<
    livo::ProductionPatchCapabilities>);

class PinholeAdapter :
  public livo::CameraProjectionAdapter
{
public:
  PinholeAdapter(
    const double fx,
    const double fy,
    const double cx,
    const double cy,
    const int width,
    const int height,
    const std::uint64_t model_revision = 1U)
  : fx_(fx),
    fy_(fy),
    cx_(cx),
    cy_(cy),
    width_(width),
    height_(height),
    model_revision_(model_revision)
  {
  }

  bool valid() const override
  {
    return
      fx_ > 0.0 &&
      fy_ > 0.0 &&
      width_ > 0 &&
      height_ > 0;
  }

  std::uint64_t modelRevision() const noexcept override
  {
    return model_revision_;
  }

  livo::CameraRayResult unproject(
    const Eigen::Vector2d & pixel) const override
  {
    ++unproject_calls;
    livo::CameraRayResult result;
    if (!pixel.allFinite())
    {
      result.status = livo::ProjectionAdapterStatus::kNonFinite;
      return result;
    }
    result.direction_camera =
      Eigen::Vector3d(
        (pixel.x() - cx_) / fx_,
        (pixel.y() - cy_) / fy_,
        1.0).normalized();
    result.status = livo::ProjectionAdapterStatus::kValid;
    return result;
  }

  livo::CameraRayJacobianResult unprojectJacobian(
    const Eigen::Vector2d & pixel) const override
  {
    livo::CameraRayJacobianResult result;
    if (!pixel.allFinite())
    {
      result.status = livo::ProjectionAdapterStatus::kNonFinite;
      return result;
    }
    const Eigen::Vector3d raw_direction(
      (pixel.x() - cx_) / fx_,
      (pixel.y() - cy_) / fy_,
      1.0);
    const double raw_norm = raw_direction.norm();
    const Eigen::Vector3d direction = raw_direction / raw_norm;
    Eigen::Matrix<double, 3, 2> raw_jacobian =
      Eigen::Matrix<double, 3, 2>::Zero();
    raw_jacobian(0, 0) = 1.0 / fx_;
    raw_jacobian(1, 1) = 1.0 / fy_;
    result.jacobian =
      (Eigen::Matrix3d::Identity() -
      direction * direction.transpose()) *
      raw_jacobian / raw_norm;
    result.status = livo::ProjectionAdapterStatus::kValid;
    return result;
  }

  livo::CameraProjectionResult project(
    const Eigen::Vector3d & point_camera) const override
  {
    ++project_calls;
    livo::CameraProjectionResult result;
    if (!point_camera.allFinite())
    {
      result.status = livo::ProjectionAdapterStatus::kNonFinite;
      return result;
    }
    if (point_camera.z() <= 0.0)
    {
      result.status =
        livo::ProjectionAdapterStatus::kBehindCamera;
      return result;
    }
    result.pixel = Eigen::Vector2d(
      fx_ * point_camera.x() / point_camera.z() + cx_,
      fy_ * point_camera.y() / point_camera.z() + cy_);
    result.status = livo::ProjectionAdapterStatus::kValid;
    return result;
  }

  livo::CameraProjectionJacobianResult projectJacobian(
    const Eigen::Vector3d & point_camera) const override
  {
    livo::CameraProjectionJacobianResult result;
    if (!point_camera.allFinite())
    {
      result.status = livo::ProjectionAdapterStatus::kNonFinite;
      return result;
    }
    if (point_camera.z() <= 0.0)
    {
      result.status =
        livo::ProjectionAdapterStatus::kBehindCamera;
      return result;
    }
    const double inverse_depth = 1.0 / point_camera.z();
    const double inverse_depth_squared =
      inverse_depth * inverse_depth;
    result.jacobian <<
      fx_ * inverse_depth,
      0.0,
      -fx_ * point_camera.x() * inverse_depth_squared,
      0.0,
      fy_ * inverse_depth,
      -fy_ * point_camera.y() * inverse_depth_squared;
    result.status = livo::ProjectionAdapterStatus::kValid;
    return result;
  }

  bool contains(
    const Eigen::Vector2d & pixel,
    const int border) const override
  {
    return
      pixel.allFinite() &&
      pixel.x() >= static_cast<double>(border) &&
      pixel.y() >= static_cast<double>(border) &&
      pixel.x() < static_cast<double>(width_ - border) &&
      pixel.y() < static_cast<double>(height_ - border);
  }

  mutable std::size_t unproject_calls{0U};
  mutable std::size_t project_calls{0U};

private:
  double fx_;
  double fy_;
  double cx_;
  double cy_;
  int width_;
  int height_;
  std::uint64_t model_revision_;
};

class FiniteRayWithInfiniteNormAdapter final :
  public PinholeAdapter
{
public:
  using PinholeAdapter::PinholeAdapter;

  livo::CameraRayResult unproject(
    const Eigen::Vector2d & pixel) const override
  {
    static_cast<void>(pixel);
    livo::CameraRayResult result;
    result.status = livo::ProjectionAdapterStatus::kValid;
    result.direction_camera = Eigen::Vector3d(
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::max(),
      1.0);
    return result;
  }
};

livo::PhotometricCameraCalibration calibration(
  const std::size_t camera_id,
  const int width,
  const int height,
  const double covariance_scale = 1.0)
{
  livo::PhotometricCameraCalibration result;
  result.camera_id = camera_id;
  result.calibration_revision =
    11U + static_cast<std::uint64_t>(camera_id);
  result.image_width = width;
  result.image_height = height;
  result.black_level = 0.0;
  result.white_level = 255.0;
  result.inverse_response_lut.resize(256U);
  for (std::size_t index = 0U;
    index < result.inverse_response_lut.size(); ++index)
  {
    result.inverse_response_lut[index] =
      static_cast<double>(index) / 255.0;
  }
  result.vignetting_transmission.assign(
    static_cast<std::size_t>(width * height), 1.0);
  result.covariance_scale = covariance_scale;
  result.source_id =
    "untrusted-shadow-camera-" + std::to_string(camera_id);
  return result;
}

livo::PhotometricFrameMetadata untrustedMetadata()
{
  livo::PhotometricFrameMetadata result;
  result.exposure_sec = 0.01;
  result.sensor_gain = 1.0;
  result.saturated_pixel_fraction = 0.0;
  result.dark_pixel_fraction = 0.0;
  result.quality_statistics_revision = 1U;
  return result;
}

livo::CameraPoseTransferData transferredPose(
  const double timestamp_sec,
  const Eigen::Vector3d & position_world_m)
{
  livo::CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.1;
  config.max_imu_gap_sec = 0.02;
  config.gyro_noise_stddev_radps = 0.0;
  config.accel_noise_stddev_mps2 = 0.0;
  config.gyro_bias_random_walk_stddev_radps_per_sqrt_sec = 0.0;
  config.accel_bias_random_walk_stddev_mps2_per_sqrt_sec = 0.0;
  livo::CameraTimeTransfer transfer(config);

  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = 1.0;
  anchor.position_world_m = position_world_m;
  anchor.gravity_world_mps2.setZero();

  std::vector<livo::CameraTimeTransferImuSample> samples(3U);
  samples[0U].timestamp_sec = 1.0;
  samples[1U].timestamp_sec =
    0.5 * (1.0 + timestamp_sec);
  samples[2U].timestamp_sec = timestamp_sec;
  for (livo::CameraTimeTransferImuSample & sample : samples)
  {
    sample.angular_velocity_radps.setZero();
    sample.specific_force_mps2.setZero();
  }
  return transfer.transferWithSamples(
    anchor, samples, timestamp_sec);
}

double texture(const double x, const double y)
{
  return
    100.0 + 9.0 * x + 6.0 * y +
    3.0 * x * y + 2.0 * x * x - 1.5 * y * y;
}

livo::MigrationRequest request(
  std::shared_ptr<const livo::CameraProjectionAdapter> source_model,
  std::shared_ptr<const livo::CameraProjectionAdapter> target_model,
  const int width,
  const int height)
{
  livo::MigrationRequest result;
  result.batch_id.value = 51U;
  result.active_state_revision = 21U;
  result.landmark_id = 8U;
  result.source_observation_id = 13U;

  result.source.endpoint.camera_id = 0U;
  result.source.endpoint.camera_model = source_model;
  result.source.endpoint.camera_model_revision =
    source_model->modelRevision();
  result.source.endpoint.rig_revision = 3U;
  result.source.endpoint.rotation_imu_from_camera =
    Eigen::Matrix3d::Identity();
  result.source.endpoint.position_imu_from_camera =
    Eigen::Vector3d(0.1, 0.0, 0.0);
  result.source.endpoint.uncertainty.camera_model_revision =
    source_model->modelRevision();
  result.source.endpoint.uncertainty.rig_revision = 3U;
  result.source.endpoint.uncertainty.revision = 1U;

  result.source.frozen_pose.exposure_timestamp_sec = 1.0;
  result.source.frozen_pose.rotation_world_from_imu =
    Eigen::Matrix3d::Identity();
  result.source.frozen_pose.position_world_m =
    Eigen::Vector3d::Zero();
  result.source.frozen_pose.pose_covariance.setZero();
  result.source.frozen_pose.accepted_batch_id.value = 49U;
  result.source.frozen_pose.accepted_state_revision.value = 7U;
  result.source.frozen_pose.store_revision.value = 8U;
  result.source.frozen_pose.landmark_id = result.landmark_id;
  result.source.frozen_pose.observation_id =
    result.source_observation_id;
  result.source.frozen_pose.observation_revision = 9U;
  result.source.frozen_pose.selector_snapshot_revision = 10U;
  result.source.frozen_pose.selector_receipt_id = 101U;
  result.source.frozen_pose.store_receipt_id = 102U;
  result.source.raw_intensity_noise.standard_deviation_raw = 1.0;
  result.source.raw_intensity_noise.calibration_revision = 11U;
  result.source.source_pixel = Eigen::Vector2d(120.0, 80.0);
  result.source.source_level = 0U;
  result.source.patch_width = 5;
  for (int row = -2; row <= 2; ++row)
  {
    for (int column = -2; column <= 2; ++column)
    {
      result.source.source_patch.push_back(
        texture(
          static_cast<double>(column),
          static_cast<double>(row)));
    }
  }
  result.source.photo_metadata = untrustedMetadata();
  result.source.geometry_version.plane =
    livo::PlaneIdentity{6U, 2U};
  result.source.geometry_version.revision = 7U;
  result.source.observation_camera_model_revision =
    source_model->modelRevision();
  result.source.observation_rig_revision = 3U;
  result.source.observation_photometric_calibration_revision = 11U;

  result.target.endpoint.camera_id = 1U;
  result.target.endpoint.camera_model = target_model;
  result.target.endpoint.camera_model_revision =
    target_model->modelRevision();
  result.target.endpoint.rig_revision = 3U;
  result.target.endpoint.rotation_imu_from_camera =
    Eigen::Matrix3d::Identity();
  result.target.endpoint.position_imu_from_camera =
    Eigen::Vector3d(-0.05, 0.0, 0.0);
  result.target.endpoint.uncertainty.camera_model_revision =
    target_model->modelRevision();
  result.target.endpoint.uncertainty.rig_revision = 3U;
  result.target.endpoint.uncertainty.revision = 2U;
  result.target.exposure_timestamp_sec = 1.01;
  result.target.pose_transfer =
    transferredPose(
      result.target.exposure_timestamp_sec,
      Eigen::Vector3d(0.2, 0.0, 0.0));
  result.target.image.width = width;
  result.target.image.height = height;
  result.target.image.pixels.resize(
    static_cast<std::size_t>(width * height));
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      result.target.image.pixels[
        static_cast<std::size_t>(y * width + x)] =
        std::clamp(
          texture(
            0.5 * (static_cast<double>(x) - 132.0),
            0.5 * (static_cast<double>(y) - 80.0)),
          1.0, 254.0);
    }
  }
  result.target.depth_m.assign(
    result.target.image.pixels.size(), 5.0);
  result.target.photo_metadata = untrustedMetadata();
  result.target.raw_intensity_noise.standard_deviation_raw = 1.0;
  result.target.raw_intensity_noise.calibration_revision = 12U;
  result.target.visibility_revision = 1U;
  result.target.frame_camera_model_revision =
    target_model->modelRevision();
  result.target.frame_rig_revision = 3U;
  result.target.frame_photometric_calibration_revision = 12U;

  result.plane.support_key.plane =
    livo::PlaneIdentity{6U, 2U};
  result.plane.support_key.cell =
    livo::PlaneLocalCell{0, 0, 0U};
  result.plane.geometry_version =
    result.source.geometry_version;
  result.plane.point_world_m =
    Eigen::Vector3d(0.0, 0.0, 5.0);
  result.plane.point_covariance_m2.setZero();
  result.plane.normal_world = Eigen::Vector3d::UnitZ();
  result.plane.normal_covariance.setZero();

  result.gates.maximum_observation_age_sec = 1.0;
  result.gates.minimum_ray_plane_cosine = 0.05;
  result.gates.minimum_depth_m = 0.5;
  result.gates.maximum_depth_m = 30.0;
  result.gates.maximum_depth_disagreement_m = 0.5;
  result.gates.image_border = 2;
  result.gates.minimum_valid_pixels = 8U;
  result.gates.minimum_patch_standard_deviation = 1.0;
  result.gates.minimum_zncc = 0.99;
  result.gates.maximum_residual_rmse = 1.0e-6;
  result.gates.maximum_warp_condition_number = 5.0;
  result.gates.minimum_warp_singular_value = 0.1;
  result.gates.maximum_warp_singular_value = 10.0;
  result.gates.minimum_positive_warp_determinant = 1.0e-6;
  result.gates.minimum_gradient_direction_coverage = 0.02;
  result.gates.allow_relative_affine_experiment = true;
  result.photometric_mode =
    livo::PhotometricMode::kRobustAffineExperimental;
  return result;
}

livo::WhitenedPatchContribution linearizeOne(
  const livo::CrossCameraPatchMigration & migration,
  const livo::MigrationRequest & value)
{
  const std::vector<livo::WhitenedPatchContribution> result =
    migration.linearizeBatch({value});
  EXPECT_EQ(result.size(), 1U);
  return result.empty() ?
    livo::WhitenedPatchContribution{} : result.front();
}

void expectAtomicZero(
  const livo::WhitenedPatchContribution & contribution)
{
  EXPECT_FALSE(contribution.productionActiveReady());
  EXPECT_TRUE(contribution.shadow_only);
  EXPECT_TRUE(contribution.zeroContribution());
}

TEST(
  FrozenAcceptedSourcePose,
  RawSnapshotCannotAcquireAcceptanceOrIndependenceProof)
{
  livo::FrozenAcceptedSourcePose frozen;
  frozen.exposure_timestamp_sec = 1.0;
  frozen.rotation_world_from_imu = Eigen::Matrix3d::Identity();
  frozen.position_world_m = Eigen::Vector3d::Zero();
  frozen.pose_covariance.setZero();
  frozen.accepted_batch_id.value = 49U;
  frozen.accepted_state_revision.value = 7U;
  frozen.store_revision.value = 8U;
  frozen.landmark_id = 8U;
  frozen.observation_id = 13U;
  frozen.observation_revision = 9U;
  frozen.selector_snapshot_revision = 10U;
  frozen.selector_receipt_id = 101U;
  frozen.store_receipt_id = 102U;

  EXPECT_TRUE(frozen.valid());
  EXPECT_TRUE(frozen.validFor(livo::BatchId{51U}, 8U, 13U));
  EXPECT_FALSE(frozen.acceptedSnapshotProofReady());
  EXPECT_FALSE(frozen.independentFromActivePrior(21U));
}

TEST(
  CrossCameraPatchMigration,
  UsesDifferentModelsTimestampsAndExtrinsicsInShadow)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};

  const livo::MigrationResult result =
    migration.prepare(
      request(source_model, target_model, kWidth, kHeight));

  ASSERT_EQ(result.status, livo::MigrationStatus::kReady);
  EXPECT_TRUE(result.geometry_ready());
  EXPECT_FALSE(result.ready());
  EXPECT_TRUE(result.shadow_only);
  EXPECT_TRUE(result.target_pixel.isApprox(
    Eigen::Vector2d(132.0, 80.0), 1.0e-9));
  EXPECT_TRUE(result.warp.isApprox(
    2.0 * Eigen::Matrix2d::Identity(), 1.0e-9));
  EXPECT_GE(result.valid_pixel_count, 8U);
}

TEST(
  CrossCameraPatchMigration,
  RejectsFiniteSourceRayWhoseNormIsNotFinite)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model =
    std::make_shared<FiniteRayWithInfiniteNormAdapter>(
      400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};

  const livo::MigrationResult result =
    migration.prepare(
      request(source_model, target_model, kWidth, kHeight));

  EXPECT_EQ(
    result.status, livo::MigrationStatus::kInvalidSourceModel);
}

TEST(
  CrossCameraPatchMigration,
  ProductionPinholeOutOfFrameIsNotAnInvalidTargetModel)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  livo::PinholeCameraProjectionConfig source_config;
  source_config.model = "Pinhole";
  source_config.width = kWidth;
  source_config.height = kHeight;
  source_config.fx = 400.0;
  source_config.fy = 400.0;
  source_config.cx = 100.0;
  source_config.cy = 80.0;
  source_config.model_revision = 2U;
  const auto source_model =
    std::make_shared<const livo::PinholeCameraProjectionAdapter>(
      source_config);

  auto target_config = source_config;
  target_config.fx = 800.0;
  target_config.fy = 800.0;
  target_config.cx = 1000.0;
  target_config.model_revision = 5U;
  const auto target_model =
    std::make_shared<const livo::PinholeCameraProjectionAdapter>(
      target_config);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};

  const livo::MigrationResult result =
    migration.prepare(
      request(source_model, target_model, kWidth, kHeight));

  EXPECT_EQ(result.status, livo::MigrationStatus::kOutOfFrame);
  EXPECT_NE(
    result.status, livo::MigrationStatus::kInvalidTargetModel);
}

TEST(
  CrossCameraPatchMigration,
  RejectsEndpointRevisionThatDoesNotMatchAdapterStructure)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight, 2U);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight, 5U);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};

  livo::MigrationRequest source_mismatch =
    request(source_model, target_model, kWidth, kHeight);
  source_mismatch.source.endpoint.camera_model_revision = 9U;
  source_mismatch.source.endpoint.uncertainty.
    camera_model_revision = 9U;
  source_mismatch.source.observation_camera_model_revision = 9U;
  EXPECT_EQ(
    migration.prepare(source_mismatch).status,
    livo::MigrationStatus::kModelRevisionMismatch);

  livo::MigrationRequest target_mismatch =
    request(source_model, target_model, kWidth, kHeight);
  target_mismatch.target.endpoint.camera_model_revision = 9U;
  target_mismatch.target.endpoint.uncertainty.
    camera_model_revision = 9U;
  target_mismatch.target.frame_camera_model_revision = 9U;
  EXPECT_EQ(
    migration.prepare(target_mismatch).status,
    livo::MigrationStatus::kModelRevisionMismatch);
}

TEST(
  CrossCameraPatchMigration,
  UnavailableReceiptsKeepCanonicalFactorShadowOnly)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};
  const livo::WhitenedPatchContribution contribution =
    linearizeOne(
      migration,
      request(source_model, target_model, kWidth, kHeight));

  ASSERT_EQ(
    contribution.status,
    livo::PatchContributionStatus::kExperimentalShadowOnly);
  EXPECT_FALSE(contribution.productionActiveReady());
  EXPECT_TRUE(contribution.shadow_only);
  EXPECT_TRUE(contribution.experimental);
  EXPECT_FALSE(contribution.productionCapabilities().all());
  EXPECT_FALSE(
    contribution.productionCapabilities().
      frozenSourcePriorIndependence());
  EXPECT_FALSE(
    contribution.productionCapabilities().
      jointEndpointPoseCovariance());
  EXPECT_FALSE(
    contribution.productionCapabilities().
      sourceEndpointUncertainty());
  EXPECT_FALSE(
    contribution.productionCapabilities().
      targetEndpointUncertainty());
  EXPECT_FALSE(
    contribution.productionCapabilities().
      sourceRawIntensityNoise());
  EXPECT_FALSE(
    contribution.productionCapabilities().
      targetRawIntensityNoise());
  EXPECT_FALSE(
    contribution.productionCapabilities().targetVisibility());
  EXPECT_FALSE(
    contribution.productionCapabilities().
      photometricQualityMetadata());
  EXPECT_FALSE(contribution.zeroContribution());
  EXPECT_TRUE(
    contribution.information.isApprox(
      contribution.whitened_jacobian.transpose() *
      contribution.whitened_jacobian,
      1.0e-10));
  EXPECT_TRUE(
    contribution.gradient.isApprox(
      contribution.whitened_jacobian.transpose() *
      contribution.whitened_residual,
      1.0e-10));
  EXPECT_NEAR(
    contribution.normalized_squared_error,
    contribution.whitened_residual.squaredNorm(),
    1.0e-10);
}

TEST(
  CrossCameraPatchMigration,
  RawOnlyAndFullZeroDepthNeverBecomeProductionReady)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};

  livo::MigrationRequest raw_only =
    request(source_model, target_model, kWidth, kHeight);
  EXPECT_FALSE(raw_only.target.depthFrameProofReady());
  const livo::WhitenedPatchContribution raw_only_result =
    linearizeOne(migration, raw_only);
  ASSERT_EQ(
    raw_only_result.status,
    livo::PatchContributionStatus::kExperimentalShadowOnly);
  EXPECT_FALSE(raw_only_result.productionActiveReady());
  EXPECT_FALSE(
    raw_only_result.productionCapabilities().targetVisibility());

  livo::MigrationRequest zero_depth = raw_only;
  std::fill(
    zero_depth.target.depth_m.begin(),
    zero_depth.target.depth_m.end(), 0.0);
  const livo::WhitenedPatchContribution zero_depth_result =
    linearizeOne(migration, zero_depth);
  ASSERT_EQ(
    zero_depth_result.status,
    livo::PatchContributionStatus::kExperimentalShadowOnly);
  EXPECT_FALSE(zero_depth_result.productionActiveReady());
  EXPECT_FALSE(
    zero_depth_result.productionCapabilities().
      targetVisibility());
}

TEST(
  CrossCameraPatchMigration,
  StrictDirectWithoutTrustedProvidersFailsClosed)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};
  livo::MigrationRequest strict =
    request(source_model, target_model, kWidth, kHeight);
  strict.photometric_mode =
    livo::PhotometricMode::kCalibratedDirect;

  const livo::WhitenedPatchContribution contribution =
    linearizeOne(migration, strict);

  EXPECT_EQ(
    contribution.status,
    livo::PatchContributionStatus::kMigrationRejected);
  EXPECT_EQ(
    contribution.migration_status,
    livo::MigrationStatus::kPhotometricUnavailable);
  expectAtomicZero(contribution);
}

TEST(
  CrossCameraPatchMigration,
  PhotometricCovarianceScaleInflatesShadowVariance)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration unit_migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};
  const livo::CrossCameraPatchMigration scaled_migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight, 100.0),
        calibration(1U, kWidth, kHeight, 100.0)})};
  const livo::MigrationRequest value =
    request(source_model, target_model, kWidth, kHeight);

  const livo::WhitenedPatchContribution unit =
    linearizeOne(unit_migration, value);
  const livo::WhitenedPatchContribution scaled =
    linearizeOne(scaled_migration, value);

  ASSERT_EQ(
    unit.status,
    livo::PatchContributionStatus::kExperimentalShadowOnly);
  ASSERT_EQ(
    scaled.status,
    livo::PatchContributionStatus::kExperimentalShadowOnly);
  EXPECT_NEAR(
    scaled.minimum_effective_variance,
    100.0 * unit.minimum_effective_variance,
    1.0e-12);
  EXPECT_GT(
    unit.information.norm(),
    scaled.information.norm());
}

TEST(
  CrossCameraPatchMigration,
  PixelAndExtrinsicCovariancesIncreaseShadowVariance)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};
  const livo::MigrationRequest baseline =
    request(source_model, target_model, kWidth, kHeight);
  const livo::WhitenedPatchContribution nominal =
    linearizeOne(migration, baseline);

  livo::MigrationRequest pixel_noise = baseline;
  pixel_noise.source.endpoint.uncertainty.
    observation_pixel_covariance_px2 =
    0.25 * Eigen::Matrix2d::Identity();
  const livo::WhitenedPatchContribution pixel_result =
    linearizeOne(migration, pixel_noise);
  EXPECT_GT(
    pixel_result.maximum_effective_variance,
    nominal.maximum_effective_variance);

  livo::MigrationRequest extrinsic_noise = baseline;
  extrinsic_noise.target.endpoint.uncertainty.
    extrinsic_covariance =
    1.0e-6 * Eigen::Matrix<double, 6, 6>::Identity();
  const livo::WhitenedPatchContribution extrinsic_result =
    linearizeOne(migration, extrinsic_noise);
  EXPECT_GT(
    extrinsic_result.maximum_effective_variance,
    nominal.maximum_effective_variance);
}

TEST(
  CrossCameraPatchMigration,
  DeduplicatesOneShadowInformationUnitPerLandmarkAndBatch)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};
  const livo::MigrationRequest first =
    request(source_model, target_model, kWidth, kHeight);
  const livo::MigrationRequest duplicate = first;
  livo::MigrationRequest next_batch = first;
  ++next_batch.batch_id.value;

  const std::vector<livo::WhitenedPatchContribution> result =
    migration.linearizeBatch(
      {first, duplicate, next_batch});

  ASSERT_EQ(result.size(), 3U);
  EXPECT_EQ(
    result[0U].status,
    livo::PatchContributionStatus::kExperimentalShadowOnly);
  EXPECT_EQ(
    result[1U].status,
    livo::PatchContributionStatus::kDuplicateLandmark);
  EXPECT_EQ(
    result[2U].status,
    livo::PatchContributionStatus::kExperimentalShadowOnly);
  expectAtomicZero(result[1U]);
}

TEST(
  CrossCameraPatchMigration,
  InvalidTransferNoiseAndGeometryAreAtomicZero)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};
  const livo::MigrationRequest nominal =
    request(source_model, target_model, kWidth, kHeight);

  std::vector<livo::MigrationRequest> invalid;
  invalid.push_back(nominal);
  invalid.back().target.pose_transfer.delta_time_sec += 1.0e-3;
  invalid.push_back(nominal);
  invalid.back().source.raw_intensity_noise.standard_deviation_raw =
    std::numeric_limits<double>::quiet_NaN();
  invalid.push_back(nominal);
  invalid.back().plane.normal_world.setZero();
  invalid.push_back(nominal);
  invalid.back().source.endpoint.uncertainty.
    observation_pixel_covariance_px2(0, 0) = -1.0;
  invalid.push_back(nominal);
  invalid.back().source.geometry_version.revision++;

  for (std::size_t index = 0U; index < invalid.size(); ++index)
  {
    SCOPED_TRACE(index);
    const livo::WhitenedPatchContribution contribution =
      linearizeOne(migration, invalid[index]);
    EXPECT_EQ(
      contribution.status,
      livo::PatchContributionStatus::kMigrationRejected);
    expectAtomicZero(contribution);
  }
}

TEST(
  CrossCameraPatchMigration,
  OversizedPatchFailsBeforeProjection)
{
  constexpr int kWidth = 200;
  constexpr int kHeight = 160;
  const auto source_model = std::make_shared<PinholeAdapter>(
    400.0, 400.0, 100.0, 80.0, kWidth, kHeight);
  const auto target_model = std::make_shared<PinholeAdapter>(
    800.0, 800.0, 100.0, 80.0, kWidth, kHeight);
  const livo::CrossCameraPatchMigration migration{
    livo::PhotometricCalibration(
      {
        calibration(0U, kWidth, kHeight),
        calibration(1U, kWidth, kHeight)})};
  livo::MigrationRequest oversized =
    request(source_model, target_model, kWidth, kHeight);
  oversized.source.patch_width = std::numeric_limits<int>::max();
  oversized.source.source_patch = {100.0};

  EXPECT_EQ(
    migration.prepare(oversized).status,
    livo::MigrationStatus::kNonFinite);
  EXPECT_EQ(source_model->unproject_calls, 0U);
  EXPECT_EQ(target_model->project_calls, 0U);
}

}  // namespace
