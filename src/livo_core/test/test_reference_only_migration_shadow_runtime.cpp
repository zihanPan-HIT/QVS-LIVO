#include "vio.h"
#include "active_plane_authority_test_support.h"
#include "camera_model_factory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{

livo::CameraModelParameters pinholeParameters(
  const std::uint8_t camera_id)
{
  livo::CameraModelParameters parameters;
  parameters.model = "Pinhole";
  parameters.width = 640;
  parameters.height = 480;
  parameters.scale = 1.0;
  parameters.fx = 400.0 + static_cast<double>(camera_id);
  parameters.fy = 400.0 + static_cast<double>(camera_id);
  parameters.cx = 320.0;
  parameters.cy = 240.0;
  return parameters;
}

livo::CrossCameraMigrationShadowInitializationConfig
crossCameraInitializationConfig()
{
  constexpr std::array<const char *, 4U> kCameraNames{{
    "front", "left", "back", "right"}};
  livo::CrossCameraMigrationShadowInitializationConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    config.cameras[camera_id].camera_id = camera_id;
    config.cameras[camera_id].camera_name =
      kCameraNames[camera_id];
    config.cameras[camera_id].projection =
      pinholeParameters(camera_id);
    config.cameras[camera_id].
      projection_calibration_present = true;
    config.cameras[camera_id].
      extrinsic_calibration_present = true;
  }
  return config;
}

void initializeReadyCrossCameraOwner(VIOManager & manager)
{
  constexpr std::array<const char *, 4U> kCameraNames{{
    "front", "left", "back", "right"}};
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());

  livo::CrossCameraMigrationShadowInitializationConfig config =
    crossCameraInitializationConfig();
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const livo::CameraModelParameters parameters =
      pinholeParameters(camera_id);
    manager.registerCamera(
      camera_id, kCameraNames[camera_id],
      livo::createCameraModel(parameters),
      M3D::Identity(), V3D::Zero());
  }
  manager.initializeCrossCameraMigrationShadow(config);
  ASSERT_TRUE(
    manager.crossCameraMigrationShadowInitAudit().ready);
}

cv::Mat texturedImage()
{
  cv::Mat image(480, 640, CV_8UC1);
  for (int row = 0; row < image.rows; ++row)
  {
    auto * pixels = image.ptr<std::uint8_t>(row);
    for (int column = 0; column < image.cols; ++column)
    {
      pixels[column] =
        ((row / 8 + column / 8) % 2 == 0) ?
        32U : 224U;
    }
  }
  return image;
}

struct RuntimeAuthorityHarness
{
  CameraObservationQueue queue;
  livo::ReferenceOnlyMigrationRuntimePublisher publisher;

  RuntimeAuthorityHarness()
  : publisher(queue.batchAuthorityIdentity())
  {
    if (!queue.setBatchAuthoritySealingEnabled(true))
    {
      throw std::runtime_error(
              "failed to enable camera batch sealing");
    }
  }
};

CameraObservationBatch sealFourCameraBatch(
  RuntimeAuthorityHarness & harness,
  const std::array<double, 4U> & timestamps,
  const std::optional<std::uint8_t>
    wrong_size_camera_id = std::nullopt,
  const std::uint8_t image_variant = 0U)
{
  const cv::Mat image = texturedImage();
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp = timestamps[camera_id];
    observation.image =
      wrong_size_camera_id.has_value() &&
      *wrong_size_camera_id == camera_id ?
      cv::Mat(479, 640, CV_8UC1, cv::Scalar(127)) :
      image.clone();
    observation.image.at<std::uint8_t>(0, 0) ^=
      image_variant;
    if (
      harness.queue.push(std::move(observation)) !=
      CameraObservationPushResult::ACCEPTED)
    {
      throw std::runtime_error(
              "failed to enqueue sealed camera observation");
    }
  }

  CameraObservationBatch batch;
  const double watermark =
    *std::max_element(timestamps.begin(), timestamps.end());
  if (!harness.queue.popSynchronized(
      watermark, 0.1, batch))
  {
    throw std::runtime_error(
            "failed to pop sealed four-camera batch");
  }
  if (!batch.validatedReceipt().has_value())
  {
    throw std::runtime_error(
            "sealed batch did not carry a receipt");
  }
  return batch;
}

CameraObservationBatch sealFourCameraBatch(
  RuntimeAuthorityHarness & harness,
  const double timestamp_sec)
{
  return sealFourCameraBatch(
    harness,
    {timestamp_sec, timestamp_sec,
      timestamp_sec, timestamp_sec});
}

livo::CameraTimeTransfer coupledTimeTransfer(
  const StatesGroup & state,
  const double anchor_timestamp_sec)
{
  livo::CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.05;
  config.max_imu_gap_sec = 0.01;
  livo::CameraTimeTransfer transfer(config);
  for (int index = 0; index <= 12; ++index)
  {
    livo::CameraTimeTransferImuSample sample;
    sample.timestamp_sec =
      anchor_timestamp_sec - 0.04 +
      0.005 * static_cast<double>(index);
    sample.angular_velocity_radps =
      state.bias_g + Eigen::Vector3d(0.02, -0.01, 0.03);
    sample.specific_force_mps2 =
      state.bias_a + Eigen::Vector3d(0.1, -0.2, 9.7);
    EXPECT_EQ(
      transfer.appendImu(sample),
      livo::CameraTimeTransferAppendStatus::kAccepted);
  }
  return transfer;
}

livo::ReferenceOnlyMigrationRuntimeBundle publishRuntimeBundle(
  RuntimeAuthorityHarness & harness,
  const CameraObservationBatch & batch,
  const StatesGroup & state,
  const double anchor_timestamp_sec)
{
  livo::CameraTimeTransfer transfer =
    coupledTimeTransfer(state, anchor_timestamp_sec);
  auto publication = harness.publisher.publish(
    batch, state, anchor_timestamp_sec, 0.0, transfer);
  if (!publication.has_value())
  {
    throw std::runtime_error(
            "failed to publish migration runtime bundle");
  }
  return std::move(*publication);
}

pointWithVar geometryPoint()
{
  pointWithVar point;
  point.point_w = Eigen::Vector3d(0.0, 0.0, 5.0);
  point.var = 1.0e-4 * Eigen::Matrix3d::Identity();
  point.normal = Eigen::Vector3d::UnitZ();
  point.plane_support_valid = true;
  point.plane_id = 91U;
  point.plane_epoch = 1U;
  point.plane_revision = 3U;
  point.plane_level = 0U;
  point.plane_identity_origin = Eigen::Vector3d::Zero();
  point.plane_identity_u_axis = Eigen::Vector3d::UnitX();
  point.plane_identity_v_axis = Eigen::Vector3d::UnitY();
  point.plane_covariance =
    1.0e-4 *
    Eigen::Matrix<double, 6, 6>::Identity();
  return point;
}

void initializeProcessFramesManager(
  VIOManager & manager,
  StatesGroup & state,
  StatesGroup & state_propagat,
  RuntimeAuthorityHarness & harness,
  const bool runtime_enabled = true,
  const bool geometry_enabled = true,
  const bool time_transfer_configured = true,
  const bool observation_enabled = false,
  const bool adaptive_enabled = false)
{
  initializeReadyCrossCameraOwner(manager);
  if (geometry_enabled)
  {
    livo::LegacyLandmarkShadowConfig geometry_config;
    geometry_config.enabled = true;
    manager.configureSharedLandmarkShadow(geometry_config);
  }
  if (observation_enabled)
  {
    livo::LegacyLandmarkObservationShadowConfig
      observation_config;
    observation_config.enabled = true;
    manager.configureSharedLandmarkObservationShadow(
      observation_config);
  }
  if (time_transfer_configured)
  {
    manager.configureSharedLandmarkTimeTransfer(
      livo::CameraTimeTransferConfig{});
  }
  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    harness.queue.batchAuthorityIdentity(),
    harness.publisher.identity());
  livo::ReferenceOnlyMigrationShadowRuntimeConfig runtime_config;
  runtime_config.enabled = runtime_enabled;
  runtime_config.four_camera_mode = true;
  manager.configureReferenceOnlyMigrationShadowRuntime(
    runtime_config);

  manager.grid_size = 20;
  manager.patch_size = 8;
  manager.outlier_threshold = 400;
  manager.state = &state;
  manager.state_propagat = &state_propagat;
  manager.max_iterations = 1;
  manager.img_point_cov = 100.0;
  manager.joint_img_point_cov = 1000.0;
  manager.normal_en = true;
  manager.inverse_composition_en = false;
  manager.raycast_en = false;
  manager.grid_n_width = 32;
  manager.grid_n_height = 24;
  manager.patch_pyrimid_level = 1;
  manager.exposure_estimate_en = false;
  manager.colmap_output_en = false;
  manager.plot_flag = false;
  manager.adaptive_policy_config.enabled = adaptive_enabled;
  manager.initializeVIO();
}

void processAndExpectDependencyStale(
  VIOManager & manager,
  RuntimeAuthorityHarness & harness,
  StatesGroup & state,
  const double timestamp_sec,
  const std::uint64_t generation)
{
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        generation}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, timestamp_sec);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, generation, &active_planes, &bundle, &publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kDependencyStale);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.materialized_candidate_count, 0U);
}

}  // namespace

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  ReasonNamesAreStableSnakeCase)
{
  EXPECT_STREQ(
    livo::referenceOnlyMigrationShadowRuntimeInitReasonName(
      livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
        kMetricsOnlyReady),
    "metrics_only_ready");
  EXPECT_STREQ(
    livo::referenceOnlyMigrationShadowRuntimeBatchReasonName(
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kTimeTransferBatchBindingMismatch),
    "time_transfer_batch_binding_mismatch");
  EXPECT_STREQ(
    livo::referenceOnlyMigrationShadowRuntimeBatchReasonName(
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kEvaluated),
    "evaluated");
  EXPECT_STREQ(
    livo::vioRuntimeObservationAuthorityStatusName(
      livo::VIORuntimeObservationAuthorityStatus::kReady),
    "ready");
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  DisabledByDefaultDoesNotBecomeEffective)
{
  VIOManager manager;

  manager.configureReferenceOnlyMigrationShadowRuntime(
    livo::ReferenceOnlyMigrationShadowRuntimeConfig{});

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_FALSE(audit.enabled);
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kDisabled);
  const auto & batch_audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    batch_audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kNotRun);
  EXPECT_EQ(batch_audit.batch_sequence, 0U);
  EXPECT_EQ(batch_audit.runner_call_count, 0U);
  EXPECT_EQ(batch_audit.captured_seed_count, 0U);
  EXPECT_TRUE(audit.experimental);
  EXPECT_FALSE(audit.visibility_verified);
  EXPECT_FALSE(audit.consumable);
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
  EXPECT_FALSE(audit.acceptance_receipt_issued);
  EXPECT_FALSE(audit.independence_proof_issued);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  EnabledWithoutFourCameraModeFailsClosed)
{
  VIOManager manager;
  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = false;

  EXPECT_NO_THROW(
    manager.configureReferenceOnlyMigrationShadowRuntime(config));

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRequiresFourCameraMode);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  EnabledWithoutSharedGeometryShadowFailsClosed)
{
  VIOManager manager;
  initializeReadyCrossCameraOwner(manager);

  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  EXPECT_NO_THROW(
    manager.configureReferenceOnlyMigrationShadowRuntime(config));

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingSharedGeometryShadow);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  EnabledWithoutTimeTransferConfigurationFailsClosed)
{
  VIOManager manager;
  initializeReadyCrossCameraOwner(manager);
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);

  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  EXPECT_NO_THROW(
    manager.configureReferenceOnlyMigrationShadowRuntime(config));

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingTimeTransferConfiguration);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  AllInitializationDependenciesEnableMetricsOnlyRuntime)
{
  VIOManager manager;
  RuntimeAuthorityHarness harness;
  initializeReadyCrossCameraOwner(manager);
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);
  manager.configureSharedLandmarkTimeTransfer(
    livo::CameraTimeTransferConfig{});
  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    harness.queue.batchAuthorityIdentity(),
    harness.publisher.identity());

  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  testing::internal::CaptureStdout();
  manager.configureReferenceOnlyMigrationShadowRuntime(config);
  const std::string init_metric =
    testing::internal::GetCapturedStdout();

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_TRUE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMetricsOnlyReady);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count, 0U);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    captured_seed_count, 0U);
  EXPECT_TRUE(audit.experimental);
  EXPECT_FALSE(audit.visibility_verified);
  EXPECT_FALSE(audit.consumable);
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
  EXPECT_NE(
    init_metric.find(
      "[VIO_REFERENCE_ONLY_MIGRATION_SHADOW_INIT]"),
    std::string::npos);
  EXPECT_NE(
    init_metric.find("reason=metrics_only_ready"),
    std::string::npos);
  EXPECT_NE(
    init_metric.find("rig_revision="),
    std::string::npos);
  EXPECT_NE(
    init_metric.find(
      "visibility_verified=0 experimental=1 "
      "consumable=0 production_ready=0 solver_applied=0 "
      "acceptance_receipt_issued=0 "
      "independence_proof_issued=0"),
    std::string::npos);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  MismatchedQueueAndPublisherOwnersFailInitializationClosed)
{
  VIOManager manager;
  RuntimeAuthorityHarness batch_owner;
  RuntimeAuthorityHarness foreign_publisher_owner;
  initializeReadyCrossCameraOwner(manager);
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);
  manager.configureSharedLandmarkTimeTransfer(
    livo::CameraTimeTransferConfig{});
  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    batch_owner.queue.batchAuthorityIdentity(),
    foreign_publisher_owner.publisher.identity());

  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  manager.configureReferenceOnlyMigrationShadowRuntime(config);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRuntimeAuthorityOwnerMismatch);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  RuntimeObservationAuthorityRejectsAnyRebind)
{
  VIOManager manager;
  RuntimeAuthorityHarness harness;
  initializeReadyCrossCameraOwner(manager);
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);
  manager.configureSharedLandmarkTimeTransfer(
    livo::CameraTimeTransferConfig{});
  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    harness.queue.batchAuthorityIdentity(),
    harness.publisher.identity());

  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  manager.configureReferenceOnlyMigrationShadowRuntime(config);
  ASSERT_TRUE(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().
    effective);

  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    harness.queue.batchAuthorityIdentity(),
    harness.publisher.identity());
  manager.configureReferenceOnlyMigrationShadowRuntime(config);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kRuntimeObservationAuthorityBindingRejected);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  MissingBatchAuthorityFailsInitializationClosed)
{
  VIOManager manager;
  initializeReadyCrossCameraOwner(manager);
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);
  manager.configureSharedLandmarkTimeTransfer(
    livo::CameraTimeTransferConfig{});
  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;

  manager.configureReferenceOnlyMigrationShadowRuntime(config);

  EXPECT_FALSE(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().
    effective);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingBatchAuthority);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  MissingPublisherAuthorityFailsInitializationClosed)
{
  VIOManager manager;
  RuntimeAuthorityHarness harness;
  initializeReadyCrossCameraOwner(manager);
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);
  manager.configureSharedLandmarkTimeTransfer(
    livo::CameraTimeTransferConfig{});
  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    harness.queue.batchAuthorityIdentity(), {});
  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;

  manager.configureReferenceOnlyMigrationShadowRuntime(config);

  EXPECT_FALSE(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().
    effective);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingRuntimePublisherAuthority);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  EnabledWithoutCrossCameraOwnerFailsClosed)
{
  VIOManager manager;
  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;

  EXPECT_NO_THROW(
    manager.configureReferenceOnlyMigrationShadowRuntime(config));

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingCrossCameraShadowOwner);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  EnabledWithUnreadyCrossCameraOwnerFailsClosed)
{
  VIOManager manager;
  livo::CrossCameraMigrationShadowInitializationConfig
    owner_config;
  owner_config.enabled = true;
  owner_config.four_camera_mode = true;
  manager.initializeCrossCameraMigrationShadow(owner_config);

  livo::ReferenceOnlyMigrationShadowRuntimeConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  EXPECT_NO_THROW(
    manager.configureReferenceOnlyMigrationShadowRuntime(config));

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kCrossCameraShadowOwnerNotReady);
  EXPECT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  CrossCameraOwnerAloneNeverRunsMigrationRuntime)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness,
    true, false, false);
  ASSERT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().
    reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingSharedGeometryShadow);
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 9.0);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 40U, nullptr, &bundle, &publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kNotRun);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.captured_seed_count, 0U);
  EXPECT_TRUE(audit.selected_sequence.empty());
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  EffectiveRuntimeWithoutProvenanceFailsClosedAfterSeedCapture)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);

  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        41U}}));
  const livo::ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.0);
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  EXPECT_NO_THROW(
    manager.processFrames(
      batch, lidar_points, feature_map, 10.0,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      true, 41U, &active_planes, nullptr));

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kMissingRuntimeProvenance);
  EXPECT_EQ(audit.batch_sequence, 0U);
  EXPECT_EQ(audit.lidar_feature_generation, 41U);
  EXPECT_EQ(audit.captured_seed_count, 1U);
  EXPECT_EQ(audit.candidate_count, 0U);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.attempted_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  ValidRuntimeInjectsFourOwnerIssuedFrameProvenances)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness,
    true, true, true, true);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 64U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.005);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & transfer_bundle =
    publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  testing::internal::CaptureStdout();
  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 64U, &active_planes, &transfer_bundle,
    &publication);
  const std::string metrics =
    testing::internal::GetCapturedStdout();

  const auto structural = manager.readOnlyStructuralAudit();
  EXPECT_EQ(
    structural.shared_landmark_shadow.landmark_count, 1U);
  EXPECT_EQ(
    structural.shared_landmark_shadow.observation_count, 0U);
  EXPECT_NE(
    metrics.find(
      "runtime_observation_authority=ready "
      "runtime_observation_frames=4"),
    std::string::npos);
  EXPECT_NE(
    metrics.find("adapter_stage_calls=1"),
    std::string::npos);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  MissingRuntimeKeepsGeometryAndSkipsObservationWrite)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness,
    true, true, true, true);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 65U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.006);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & transfer_bundle =
    publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  testing::internal::CaptureStdout();
  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 65U, &active_planes, &transfer_bundle, nullptr);
  const std::string metrics =
    testing::internal::GetCapturedStdout();

  const auto structural = manager.readOnlyStructuralAudit();
  EXPECT_EQ(
    structural.shared_landmark_shadow.landmark_count, 1U);
  EXPECT_EQ(
    structural.shared_landmark_shadow.observation_count, 0U);
  EXPECT_NE(
    metrics.find(
      "runtime_observation_authority=missing_runtime_bundle "
      "runtime_observation_frames=0"),
    std::string::npos);
  EXPECT_NE(
    metrics.find("adapter_stage_calls=0"),
    std::string::npos);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  ValidRuntimeCanonicalizesForeignContextTransfer)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  RuntimeAuthorityHarness foreign_harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness,
    true, true, true, true);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 69U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.0065);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const CameraObservationBatch foreign_batch =
    sealFourCameraBatch(foreign_harness, 10.0065);
  const auto foreign_publication =
    publishRuntimeBundle(
    foreign_harness, foreign_batch, state,
    foreign_batch.timestamp);
  const auto & foreign_transfer_bundle =
    foreign_publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  testing::internal::CaptureStdout();
  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 69U, &active_planes,
    &foreign_transfer_bundle, &publication);
  const std::string metrics =
    testing::internal::GetCapturedStdout();

  const auto & runtime_audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    runtime_audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kEvaluated);
  EXPECT_EQ(
    runtime_audit.source_covariance_evaluation_count, 4U);
  EXPECT_NE(
    metrics.find(
      "runtime_observation_authority=ready "
      "runtime_observation_frames=4"),
    std::string::npos);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  ForeignRuntimeKeepsGeometryAndSkipsObservationWrite)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  RuntimeAuthorityHarness foreign_harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness,
    true, true, true, true);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 66U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.007);
  const CameraObservationBatch foreign_batch =
    sealFourCameraBatch(foreign_harness, 10.007);
  const auto foreign_publication =
    publishRuntimeBundle(
    foreign_harness, foreign_batch, state,
    foreign_batch.timestamp);
  const auto & foreign_transfer_bundle =
    foreign_publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  testing::internal::CaptureStdout();
  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 66U, &active_planes,
    &foreign_transfer_bundle, &foreign_publication);
  const std::string metrics =
    testing::internal::GetCapturedStdout();

  const auto structural = manager.readOnlyStructuralAudit();
  EXPECT_EQ(
    structural.shared_landmark_shadow.landmark_count, 1U);
  EXPECT_EQ(
    structural.shared_landmark_shadow.observation_count, 0U);
  EXPECT_NE(
    metrics.find(
      "runtime_observation_authority="
      "runtime_publisher_authority_mismatch "
      "runtime_observation_frames=0"),
    std::string::npos);
  EXPECT_NE(
    metrics.find("adapter_stage_calls=0"),
    std::string::npos);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  AdaptivePlanCapabilityUsesTheAcceptedRuntimePublication)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness,
    true, true, true, false, true);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 67U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.008);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & transfer_bundle =
    publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  testing::internal::CaptureStdout();
  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 67U, &active_planes, &transfer_bundle,
    &publication);
  const std::string metrics =
    testing::internal::GetCapturedStdout();

  EXPECT_NE(
    metrics.find(
      "plan_capability=1 plan_capability_validated=1 "
      "legacy_fallback=0 "
      "runtime_exclusive=0"),
    std::string::npos);
  EXPECT_NE(
    metrics.find("health_committed=1"),
    std::string::npos);
  EXPECT_FALSE(
    livo::AdaptiveVisualPolicy::
    ownerSealedRuntimeExclusive());
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  MissingRuntimeCommitsUnsealedHealthWithoutClaimingLegacyFallback)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness,
    true, true, true, false, true);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 68U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.009);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & transfer_bundle =
    publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  testing::internal::CaptureStdout();
  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 68U, &active_planes, &transfer_bundle, nullptr);
  const std::string metrics =
    testing::internal::GetCapturedStdout();

  EXPECT_NE(
    metrics.find(
      "plan_capability=0 plan_capability_validated=0 "
      "legacy_fallback=0 "
      "runtime_exclusive=0"),
    std::string::npos);
  EXPECT_NE(
    metrics.find("health_committed=1"),
    std::string::npos);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  InvalidFullStateTransferFailsClosedWithoutRunnerCall)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);

  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        42U}}));
  const livo::ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.0);
  const auto publication =
    publishRuntimeBundle(harness, batch, state, 10.2);
  const auto & bundle = publication.timeTransferBundle();
  ASSERT_FALSE(bundle.allPresentFullStateTransfersValid());
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  EXPECT_NO_THROW(
    manager.processFrames(
      batch, lidar_points, feature_map, 10.0,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      true, 42U, &active_planes, &bundle,
      &publication));

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidFullStateTimeTransferBundle);
  EXPECT_EQ(audit.lidar_feature_generation, 42U);
  EXPECT_EQ(audit.captured_seed_count, 1U);
  EXPECT_EQ(audit.candidate_count, 0U);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.attempted_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  IncompleteCameraContextsFailClosedWithoutRunnerCall)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);

  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        42U}}));
  const livo::ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.0);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  batch.observations[3U].image.release();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 42U, &active_planes, &bundle, &publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kIncompleteFourCameraContexts);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.candidate_count, 0U);
  EXPECT_EQ(audit.attempted_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  SameTimestampForeignImageOwnerFailsClosedWithoutRunnerCall)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat;
  RuntimeAuthorityHarness harness;
  RuntimeAuthorityHarness foreign_harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);

  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        42U}}));
  const livo::ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  CameraObservationBatch current_batch =
    sealFourCameraBatch(harness, 10.01);
  const CameraObservationBatch foreign_batch =
    sealFourCameraBatch(
    foreign_harness,
    {10.01, 10.01, 10.01, 10.01},
    std::nullopt, 0x5aU);
  const auto foreign_publication =
    publishRuntimeBundle(
    foreign_harness, foreign_batch, state,
    foreign_batch.timestamp);
  const auto & foreign_bundle =
    foreign_publication.timeTransferBundle();
  ASSERT_TRUE(
    foreign_bundle.allPresentFullStateTransfersValid());
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    current_batch, lidar_points, feature_map, 10.01,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 42U, &active_planes, &foreign_bundle,
    &foreign_publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kRuntimeProvenanceMismatch);
  EXPECT_EQ(audit.captured_seed_count, 1U);
  EXPECT_EQ(audit.candidate_count, 0U);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.attempted_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  SameTimestampForeignNominalStateFailsClosedWithoutRunnerCall)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 43U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.11);
  StatesGroup foreign_state = state;
  foreign_state.pos_end.x() += 0.25;
  const auto foreign_publication =
    publishRuntimeBundle(
    harness, batch, foreign_state, batch.timestamp);
  const auto & bundle =
    foreign_publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 43U, &active_planes, &bundle,
    &foreign_publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kRuntimeProvenanceMismatch);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.materialized_candidate_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  SameTimestampForeignCovarianceFailsClosedWithoutRunnerCall)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 44U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.12);
  StatesGroup foreign_state = state;
  foreign_state.cov(0, 0) += 0.125;
  const auto foreign_publication =
    publishRuntimeBundle(
    harness, batch, foreign_state, batch.timestamp);
  const auto & bundle =
    foreign_publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 44U, &active_planes, &bundle,
    &foreign_publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kRuntimeProvenanceMismatch);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.source_covariance_evaluation_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  InvalidPublishedCovarianceHasExplicitReasonAndNoRunnerCall)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 45U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.13);
  StatesGroup invalid_state = state;
  invalid_state.cov(0, 0) = -1.0;
  const auto invalid_publication =
    publishRuntimeBundle(
    harness, batch, invalid_state, batch.timestamp);
  const auto & bundle =
    invalid_publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 45U, &active_planes, &bundle,
    &invalid_publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidSourceCovariance);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.source_covariance_evaluation_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  NaNNominalStateHasExplicitReasonAndNoRunnerCall)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 46U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.135);
  StatesGroup invalid_state = state;
  invalid_state.pos_end.x() =
    std::numeric_limits<double>::quiet_NaN();
  const auto invalid_publication =
    publishRuntimeBundle(
    harness, batch, invalid_state, batch.timestamp);
  const auto & bundle =
    invalid_publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 46U, &active_planes, &bundle,
    &invalid_publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kInvalidActiveStateSnapshot);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.source_covariance_evaluation_count, 0U);
  EXPECT_TRUE(std::isfinite(audit.runtime_sec));
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  OwnerSealedRuntimeImageDimensionsMustMatchRegisteredRig)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, 46U}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(
    harness, {10.14, 10.14, 10.14, 10.14}, 2U);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 46U, &active_planes, &bundle, &publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kImageGeometryMismatch);
  EXPECT_EQ(audit.runner_call_count, 0U);
  EXPECT_EQ(audit.materialized_candidate_count, 0U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  RigOwnerReentryInvalidatesBoundRuntimeDependencies)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  const auto init =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();

  manager.initializeCrossCameraMigrationShadow(
    crossCameraInitializationConfig());
  ASSERT_TRUE(manager.crossCameraMigrationShadowInitAudit().ready);
  ASSERT_NE(
    manager.crossCameraMigrationShadowInitAudit().
    owner_generation,
    init.bound_cross_camera_owner_generation);

  processAndExpectDependencyStale(
    manager, harness, state, 10.21, 51U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  GeometryConfigurationReentryInvalidatesBoundRuntimeDependencies)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  const auto init =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();

  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);
  ASSERT_NE(
    manager.readOnlyStructuralAudit().shared_landmark_shadow.
    configuration_generation,
    init.bound_geometry_configuration_generation);

  processAndExpectDependencyStale(
    manager, harness, state, 10.22, 52U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  TimeTransferRevisionReentryInvalidatesBoundRuntimeDependencies)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  const auto init =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();

  livo::CameraTimeTransferConfig time_config;
  time_config.max_abs_transfer_sec = 0.075;
  manager.configureSharedLandmarkTimeTransfer(time_config);
  ASSERT_NE(
    manager.readOnlyStructuralAudit().shared_landmark_shadow.
    time_transfer_configuration_revision,
    init.bound_time_transfer_revision);

  processAndExpectDependencyStale(
    manager, harness, state, 10.23, 53U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  RuntimeAuthorityReentryInvalidatesBoundRuntimeDependencies)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  const auto init =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();

  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    harness.queue.batchAuthorityIdentity(),
    harness.publisher.identity());
  ASSERT_NE(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().
    bound_authority_generation,
    0U);
  ASSERT_EQ(
    manager.referenceOnlyMigrationShadowRuntimeInitAudit().
    bound_authority_generation,
    init.bound_authority_generation);

  processAndExpectDependencyStale(
    manager, harness, state, 10.24, 54U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  ValidFourCameraBatchEvaluatesReadyCandidatesWithFullStateCovariance)
{
  VIOManager manager;
  StatesGroup state;
  state.pos_end = Eigen::Vector3d(0.1, -0.1, 0.0);
  state.vel_end = Eigen::Vector3d(1.0, -0.5, 0.2);
  state.bias_g = Eigen::Vector3d(0.01, -0.02, 0.03);
  state.bias_a = Eigen::Vector3d(0.1, -0.1, 0.05);
  state.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  state.cov.setZero();
  state.cov.diagonal() <<
    0.01, 0.02, 0.03,
    0.04, 0.05, 0.06,
    0.001,
    4.0, 5.0, 6.0,
    0.10, 0.11, 0.12,
    0.20, 0.21, 0.22,
    0.30, 0.31, 0.32;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);

  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        43U}}));
  const livo::ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 10.01);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  testing::internal::CaptureStdout();
  manager.processFrames(
    batch, lidar_points, feature_map, 10.01,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 43U, &active_planes, &bundle, &publication);
  const std::string batch_metrics =
    testing::internal::GetCapturedStdout();

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kEvaluated);
  EXPECT_EQ(audit.captured_seed_count, 1U);
  EXPECT_EQ(audit.candidate_count, 12U);
  EXPECT_EQ(audit.descriptor_count, 12U);
  EXPECT_EQ(audit.materialized_candidate_count, 12U);
  EXPECT_EQ(audit.source_covariance_evaluation_count, 4U);
  EXPECT_EQ(audit.seal_image_hash_count, 4U);
  // The runtime publisher claims and validates each sealed image exactly
  // once. VIO and R2b consume the owner-minted composition without hashing.
  EXPECT_EQ(audit.validation_image_hash_count, 4U);
  EXPECT_EQ(audit.runner_call_count, 1U);
  EXPECT_EQ(audit.attempted_count, 12U);
  EXPECT_EQ(audit.ready_count, 12U);
  EXPECT_EQ(audit.rejected_count, 0U);
  EXPECT_TRUE(std::all_of(
    audit.pair_attempt_counts.begin(),
    audit.pair_attempt_counts.end(),
    [](const std::size_t count) {return count == 1U;}));
  EXPECT_TRUE(std::all_of(
    audit.pair_ready_counts.begin(),
    audit.pair_ready_counts.end(),
    [](const std::size_t count) {return count == 1U;}));
  ASSERT_TRUE(audit.ready_warp_condition_p50.has_value());
  ASSERT_TRUE(audit.ready_warp_condition_p95.has_value());
  EXPECT_GE(audit.runtime_sec, 0.0);

  double expected_covariance_trace_sum = 0.0;
  for (std::size_t source_camera_id = 0U;
       source_camera_id < 4U; ++source_camera_id)
  {
    const auto & raw =
      bundle.transfers[source_camera_id].value().raw();
    const auto expected =
      raw.target_pose_wrt_active_anchor_state *
      state.cov *
      raw.target_pose_wrt_active_anchor_state.transpose() +
      raw.pose_transfer_covariance +
      raw.time_alignment_pose_covariance;
    expected_covariance_trace_sum +=
      expected.trace();
  }
  EXPECT_NEAR(
    audit.source_pose_covariance_trace_sum,
    expected_covariance_trace_sum, 1.0e-10);
  EXPECT_NE(
    batch_metrics.find(
      "[VIO_REFERENCE_ONLY_MIGRATION_SHADOW_BATCH]"),
    std::string::npos);
  EXPECT_NE(
    batch_metrics.find(
      "t=10.010000000 generation=43 sequence=1 "
      "reason=evaluated"),
    std::string::npos);
  EXPECT_NE(
    batch_metrics.find("pair_attempts=1,1,1,1,1,1,1,1,1,1,1,1"),
    std::string::npos);
  EXPECT_NE(
    batch_metrics.find("pair_ready=1,1,1,1,1,1,1,1,1,1,1,1"),
    std::string::npos);
  EXPECT_NE(
    batch_metrics.find("status_counts="),
    std::string::npos);
  EXPECT_NE(
    batch_metrics.find(
      "p50_valid=1"),
    std::string::npos);
  EXPECT_NE(
    batch_metrics.find(
      "visibility_verified=0 experimental=1 "
      "consumable=0 production_ready=0 solver_applied=0 "
      "acceptance_receipt_issued=0 "
      "independence_proof_issued=0"),
    std::string::npos);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  FiveHundredTwelveSeedsRespectBothMaterializationBudgets)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);

  constexpr std::uint64_t kGeneration = 61U;
  std::vector<pointWithVar> lidar_points;
  std::vector<livo::ActivePlaneDescriptor> active_descriptors;
  lidar_points.reserve(512U);
  active_descriptors.reserve(512U);
  for (std::uint64_t index = 0U; index < 512U; ++index)
  {
    pointWithVar point = geometryPoint();
    point.plane_id = 1000U + index;
    lidar_points.push_back(point);
    active_descriptors.push_back(
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{point.plane_id, 1U}, 3U},
        kGeneration});
  }
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace(
      std::move(active_descriptors)));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 11.0);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, kGeneration, &active_planes, &bundle,
    &publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kEvaluated);
  EXPECT_EQ(audit.captured_seed_count, 512U);
  EXPECT_LE(audit.materialized_candidate_count, 48U);
  EXPECT_EQ(audit.materialized_candidate_count, 48U);
  EXPECT_LE(audit.attempted_count, 32U);
  EXPECT_EQ(audit.attempted_count, 32U);
  EXPECT_EQ(audit.source_covariance_evaluation_count, 4U);
  EXPECT_EQ(audit.runner_call_count, 1U);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  StaggeredCameraTimesUseEachSealedSourcePoseAndRejectReverseTime)
{
  VIOManager manager;
  StatesGroup state;
  state.pos_end = Eigen::Vector3d(0.2, -0.1, 0.0);
  state.vel_end = Eigen::Vector3d(1.5, -0.4, 0.1);
  state.bias_g = Eigen::Vector3d(0.01, -0.02, 0.03);
  state.bias_a = Eigen::Vector3d(0.1, -0.1, 0.05);
  state.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  state.cov.setZero();
  state.cov.diagonal() <<
    0.01, 0.02, 0.03,
    0.04, 0.05, 0.06,
    0.001,
    4.0, 5.0, 6.0,
    0.10, 0.11, 0.12,
    0.20, 0.21, 0.22,
    0.30, 0.31, 0.32;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);

  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        44U}}));
  const livo::ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(
    harness, {12.00, 12.01, 12.02, 12.03});
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  std::vector<pointWithVar> lidar_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 44U, &active_planes, &bundle, &publication);

  const auto & audit =
    manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kEvaluated);
  EXPECT_EQ(audit.attempted_count, 12U);
  EXPECT_EQ(audit.source_covariance_evaluation_count, 4U);
  EXPECT_EQ(audit.ready_count, 6U);
  EXPECT_EQ(audit.rejected_count, 6U);
  ASSERT_EQ(audit.selected_sequence.size(), 12U);

  double expected_covariance_trace_sum = 0.0;
  for (std::size_t source_camera_id = 0U;
       source_camera_id < 4U; ++source_camera_id)
  {
    const auto & raw =
      bundle.transfers[source_camera_id].value().raw();
    const auto expected =
      raw.target_pose_wrt_active_anchor_state *
      state.cov *
      raw.target_pose_wrt_active_anchor_state.transpose() +
      raw.pose_transfer_covariance +
      raw.time_alignment_pose_covariance;
    expected_covariance_trace_sum += expected.trace();
  }
  EXPECT_NEAR(
    audit.source_pose_covariance_trace_sum,
    expected_covariance_trace_sum, 1.0e-10);
  EXPECT_GT(
    std::abs(
      audit.source_pose_covariance_trace_sum -
      4.0 * state.cov.block<6, 6>(0, 0).trace()),
    1.0e-3);

  const auto & owner_audit =
    manager.crossCameraMigrationShadowInitAudit();
  for (const auto & selection : audit.selected_sequence)
  {
    const std::uint8_t source_id =
      selection.source_camera_id;
    const std::uint8_t target_id =
      selection.target_camera_id;
    ASSERT_TRUE(owner_audit.cameras[source_id].has_value());
    ASSERT_TRUE(bundle.transfers[source_id].has_value());
    ASSERT_TRUE(bundle.transfers[target_id].has_value());
    const auto & endpoint =
      *owner_audit.cameras[source_id];
    const auto & raw =
      bundle.transfers[source_id].value().raw();
    const Eigen::Matrix3d rotation_world_from_camera =
      raw.rotation_world_from_imu *
      endpoint.rotation_imu_from_camera;
    const Eigen::Vector3d position_world_from_camera =
      raw.position_world_m +
      raw.rotation_world_from_imu *
      endpoint.position_imu_from_camera;
    const Eigen::Vector3d point_camera =
      rotation_world_from_camera.transpose() *
      (selection.plane_point_world_m -
       position_world_from_camera);
    const auto parameters = pinholeParameters(source_id);
    const Eigen::Vector2d expected_pixel(
      parameters.fx * point_camera.x() / point_camera.z() +
      parameters.cx,
      parameters.fy * point_camera.y() / point_camera.z() +
      parameters.cy);
    EXPECT_TRUE(selection.source_pixel.isApprox(
      expected_pixel, 1.0e-12));

    const bool forward_in_time =
      bundle.transfers[target_id].targetTimestampSec() >=
      bundle.transfers[source_id].targetTimestampSec();
    EXPECT_EQ(
      selection.migration_status,
      forward_in_time ? livo::MigrationStatus::kReady :
      livo::MigrationStatus::kInvalidTimeTransfer);
  }

  const auto source_zero = std::find_if(
    audit.selected_sequence.begin(),
    audit.selected_sequence.end(),
    [](const auto & selection) {
      return selection.source_camera_id == 0U;
    });
  ASSERT_NE(source_zero, audit.selected_sequence.end());
  const auto & endpoint_zero =
    *owner_audit.cameras[0U];
  const Eigen::Matrix3d anchor_rotation_world_from_camera =
    state.rot_end * endpoint_zero.rotation_imu_from_camera;
  const Eigen::Vector3d anchor_position_world_from_camera =
    state.pos_end +
    state.rot_end * endpoint_zero.position_imu_from_camera;
  const Eigen::Vector3d anchor_point_camera =
    anchor_rotation_world_from_camera.transpose() *
    (source_zero->plane_point_world_m -
     anchor_position_world_from_camera);
  const auto parameters_zero = pinholeParameters(0U);
  const Eigen::Vector2d anchor_pixel(
    parameters_zero.fx * anchor_point_camera.x() /
    anchor_point_camera.z() + parameters_zero.cx,
    parameters_zero.fy * anchor_point_camera.y() /
    anchor_point_camera.z() + parameters_zero.cy);
  EXPECT_GT(
    (source_zero->source_pixel - anchor_pixel).norm(),
    1.0e-3);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  MetricsOnlyRuntimeCanonicalizesForeignTransferAndIsStructurallyIdentical)
{
  VIOManager enabled_manager;
  VIOManager disabled_manager;
  StatesGroup enabled_state;
  enabled_state.pos_end =
    Eigen::Vector3d(0.1, -0.2, 0.05);
  enabled_state.vel_end =
    Eigen::Vector3d(0.7, -0.3, 0.1);
  enabled_state.bias_g =
    Eigen::Vector3d(0.01, -0.02, 0.03);
  enabled_state.bias_a =
    Eigen::Vector3d(0.1, -0.1, 0.05);
  enabled_state.gravity =
    Eigen::Vector3d(0.0, 0.0, -9.81);
  enabled_state.cov.setIdentity();
  enabled_state.cov.diagonal() =
    Eigen::VectorXd::LinSpaced(DIM_STATE, 0.01, 0.19);
  StatesGroup disabled_state = enabled_state;
  StatesGroup enabled_propagated = enabled_state;
  StatesGroup disabled_propagated = enabled_state;
  RuntimeAuthorityHarness enabled_harness;
  RuntimeAuthorityHarness disabled_harness;
  RuntimeAuthorityHarness enabled_foreign_harness;
  RuntimeAuthorityHarness disabled_foreign_harness;
  initializeProcessFramesManager(
    enabled_manager, enabled_state, enabled_propagated,
    enabled_harness, true);
  initializeProcessFramesManager(
    disabled_manager, disabled_state, disabled_propagated,
    disabled_harness, false);

  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        livo::GeometryVersion{
          livo::PlaneIdentity{91U, 1U}, 3U},
        45U}}));
  const livo::ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  CameraObservationBatch enabled_batch =
    sealFourCameraBatch(enabled_harness, 15.01);
  CameraObservationBatch disabled_batch =
    sealFourCameraBatch(disabled_harness, 15.01);
  const CameraObservationBatch enabled_foreign_batch =
    sealFourCameraBatch(enabled_foreign_harness, 15.01);
  const CameraObservationBatch disabled_foreign_batch =
    sealFourCameraBatch(disabled_foreign_harness, 15.01);
  const auto enabled_publication =
    publishRuntimeBundle(
    enabled_harness, enabled_batch, enabled_state,
    enabled_batch.timestamp);
  const auto disabled_publication =
    publishRuntimeBundle(
    disabled_harness, disabled_batch, disabled_state,
    disabled_batch.timestamp);
  const auto enabled_foreign_publication =
    publishRuntimeBundle(
    enabled_foreign_harness, enabled_foreign_batch,
    enabled_state, enabled_foreign_batch.timestamp);
  const auto disabled_foreign_publication =
    publishRuntimeBundle(
    disabled_foreign_harness, disabled_foreign_batch,
    disabled_state, disabled_foreign_batch.timestamp);
  const auto & enabled_foreign_bundle =
    enabled_foreign_publication.timeTransferBundle();
  const auto & disabled_foreign_bundle =
    disabled_foreign_publication.timeTransferBundle();
  std::vector<pointWithVar> enabled_points{geometryPoint()};
  std::vector<pointWithVar> disabled_points{geometryPoint()};
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  enabled_manager.processFrames(
    enabled_batch, enabled_points, feature_map,
    enabled_batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 45U, &active_planes, &enabled_foreign_bundle,
    &enabled_publication);
  disabled_manager.processFrames(
    disabled_batch, disabled_points, feature_map,
    disabled_batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, 45U, &active_planes, &disabled_foreign_bundle,
    &disabled_publication);

  const auto enabled =
    enabled_manager.readOnlyStructuralAudit();
  const auto disabled =
    disabled_manager.readOnlyStructuralAudit();
  ASSERT_TRUE(enabled.active_state_available);
  ASSERT_TRUE(disabled.active_state_available);
  ASSERT_TRUE(enabled.propagated_state_available);
  ASSERT_TRUE(disabled.propagated_state_available);
  EXPECT_EQ(
    std::memcmp(
      enabled.active_state.data(),
      disabled.active_state.data(),
      sizeof(enabled.active_state)),
    0);
  EXPECT_EQ(
    std::memcmp(
      enabled.propagated_state.data(),
      disabled.propagated_state.data(),
      sizeof(enabled.propagated_state)),
    0);
  EXPECT_EQ(
    enabled.shared_landmark_shadow.geometry.value,
    disabled.shared_landmark_shadow.geometry.value);
  EXPECT_EQ(
    enabled.shared_landmark_shadow.store_revision,
    disabled.shared_landmark_shadow.store_revision);
  EXPECT_EQ(
    enabled.shared_landmark_shadow.landmark_count,
    disabled.shared_landmark_shadow.landmark_count);
  EXPECT_EQ(
    enabled.shared_landmark_shadow.observation_count,
    disabled.shared_landmark_shadow.observation_count);
  EXPECT_EQ(
    enabled.legacy_feature_voxel_counts,
    disabled.legacy_feature_voxel_counts);
  EXPECT_EQ(
    enabled.legacy_feature_point_counts,
    disabled.legacy_feature_point_counts);
  EXPECT_EQ(
    enabled.last_joint_solver.accepted,
    disabled.last_joint_solver.accepted);
  EXPECT_EQ(
    enabled.last_joint_solver.reason,
    disabled.last_joint_solver.reason);
  EXPECT_EQ(
    enabled.last_joint_solver.contributing_cameras,
    disabled.last_joint_solver.contributing_cameras);
  EXPECT_EQ(
    enabled.last_joint_solver.tracked_points,
    disabled.last_joint_solver.tracked_points);
  EXPECT_EQ(
    enabled_manager.
    referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count,
    1U);
  EXPECT_EQ(
    enabled_manager.
    referenceOnlyMigrationShadowRuntimeBatchAudit().reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kEvaluated);
  EXPECT_EQ(
    enabled_manager.
    referenceOnlyMigrationShadowRuntimeBatchAudit().
    source_covariance_evaluation_count,
    4U);
  EXPECT_EQ(
    disabled_manager.
    referenceOnlyMigrationShadowRuntimeBatchAudit().reason,
    livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
      kNotRun);
  EXPECT_EQ(
    disabled_manager.
    referenceOnlyMigrationShadowRuntimeBatchAudit().
    runner_call_count,
    0U);
  EXPECT_EQ(
    disabled_manager.
    referenceOnlyMigrationShadowRuntimeBatchAudit().
    captured_seed_count,
    0U);
  EXPECT_TRUE(
    disabled_manager.
    referenceOnlyMigrationShadowRuntimeBatchAudit().
    selected_sequence.empty());
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  MultipleBatchesAdvanceBatchAuditWithoutMutatingInitializationAudit)
{
  VIOManager manager;
  StatesGroup state;
  state.vel_end = Eigen::Vector3d(0.5, -0.1, 0.0);
  state.gravity = Eigen::Vector3d(0.0, 0.0, -9.81);
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  const auto init_before =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  livo::test_support::ActivePlaneAuthority authority;
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  for (std::uint64_t sequence = 1U; sequence <= 2U;
       ++sequence)
  {
    const std::uint64_t generation = 45U + sequence;
    ASSERT_TRUE(authority.replace({
        livo::ActivePlaneDescriptor{
          livo::GeometryVersion{
            livo::PlaneIdentity{91U, 1U}, 3U},
          generation}}));
    const livo::ActivePlaneIdentitySnapshot active_planes =
      authority.snapshot();
    CameraObservationBatch batch =
      sealFourCameraBatch(
      harness,
      16.0 + 0.01 * static_cast<double>(sequence));
    const auto publication =
      publishRuntimeBundle(
      harness, batch, state, batch.timestamp);
    const auto & bundle = publication.timeTransferBundle();
    std::vector<pointWithVar> lidar_points{geometryPoint()};

    manager.processFrames(
      batch, lidar_points, feature_map, batch.timestamp,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      true, generation, &active_planes, &bundle,
      &publication);

    const auto & batch_audit =
      manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
    EXPECT_EQ(batch_audit.batch_sequence, sequence);
    EXPECT_EQ(
      batch_audit.lidar_feature_generation, generation);
    EXPECT_EQ(batch_audit.runner_call_count, 1U);
    EXPECT_EQ(batch_audit.captured_seed_count, 1U);
    EXPECT_EQ(
      batch_audit.reason,
      livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
        kEvaluated);
  }

  const auto & init_after =
    manager.referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_EQ(init_after.enabled, init_before.enabled);
  EXPECT_EQ(init_after.effective, init_before.effective);
  EXPECT_EQ(init_after.reason, init_before.reason);
  EXPECT_EQ(
    init_after.visibility_verified,
    init_before.visibility_verified);
  EXPECT_EQ(init_after.experimental, init_before.experimental);
  EXPECT_EQ(init_after.consumable, init_before.consumable);
  EXPECT_EQ(
    init_after.production_ready,
    init_before.production_ready);
  EXPECT_EQ(
    init_after.solver_applied,
    init_before.solver_applied);
}

TEST(
  ReferenceOnlyMigrationShadowRuntime,
  ReplayedBatchAndPublicationSequencesFailClosed)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup state_propagat = state;
  RuntimeAuthorityHarness harness;
  initializeProcessFramesManager(
    manager, state, state_propagat, harness);
  livo::test_support::ActivePlaneAuthority authority;
  constexpr std::uint64_t kGeneration = 63U;
  ASSERT_TRUE(authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, kGeneration}}));
  const auto active_planes = authority.snapshot();
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, 17.0);
  const auto publication =
    publishRuntimeBundle(
    harness, batch, state, batch.timestamp);
  const auto & bundle = publication.timeTransferBundle();
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;

  for (int attempt = 0; attempt < 2; ++attempt)
  {
    std::vector<pointWithVar> lidar_points{geometryPoint()};
    manager.processFrames(
      batch, lidar_points, feature_map, batch.timestamp,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(),
      true, kGeneration, &active_planes, &bundle,
      &publication);
    const auto & audit =
      manager.referenceOnlyMigrationShadowRuntimeBatchAudit();
    if (attempt == 0)
    {
      EXPECT_EQ(
        audit.reason,
        livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
          kEvaluated);
      EXPECT_EQ(audit.runner_call_count, 1U);
    }
    else
    {
      EXPECT_EQ(
        audit.reason,
        livo::ReferenceOnlyMigrationShadowRuntimeBatchReason::
          kRuntimeProvenanceMismatch);
      EXPECT_EQ(audit.runner_call_count, 0U);
    }
  }
}
