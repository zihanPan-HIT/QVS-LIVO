#include "vio.h"
#include "active_plane_authority_test_support.h"
#include "camera_model_factory.h"
#include "trusted_photometric_evidence_test_support.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

cv::Mat texturedImage()
{
  cv::Mat image(480, 640, CV_8UC1);
  for (int row = 0; row < image.rows; ++row)
  {
    auto * pixels = image.ptr<std::uint8_t>(row);
    for (int column = 0; column < image.cols; ++column)
    {
      pixels[column] =
        ((row / 8 + column / 8) % 2 == 0) ? 32U : 224U;
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
      throw std::runtime_error("failed to enable batch sealing");
    }
  }
};

CameraObservationBatch sealFourCameraBatch(
  RuntimeAuthorityHarness & harness,
  const double timestamp_sec,
  const livo::PhotometricStatus photometric_status =
    livo::PhotometricStatus::kValid)
{
  const cv::Mat image = texturedImage();
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp = timestamp_sec;
    observation.image = image.clone();
    observation.photometric_status = photometric_status;
    if (
      harness.queue.push(std::move(observation)) !=
      CameraObservationPushResult::ACCEPTED)
    {
      throw std::runtime_error("failed to enqueue observation");
    }
  }

  CameraObservationBatch batch;
  if (!harness.queue.popSynchronized(
      timestamp_sec, 0.1, batch))
  {
    throw std::runtime_error("failed to seal four-camera batch");
  }
  return batch;
}

CameraObservationBatch sealSingleCameraPartialBatch(
  RuntimeAuthorityHarness & harness,
  const std::uint8_t camera_id,
  const double timestamp_sec)
{
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = timestamp_sec;
  observation.image = texturedImage();
  observation.photometric_status =
    livo::PhotometricStatus::kValid;
  if (
    harness.queue.push(std::move(observation)) !=
    CameraObservationPushResult::ACCEPTED)
  {
    throw std::runtime_error(
      "failed to enqueue single-camera observation");
  }

  CameraObservationBatch batch;
  const AdaptiveCameraBatchResult result =
    harness.queue.popAdaptiveSynchronized(
    timestamp_sec + 0.2, 0.01, 0.01, 1U, batch);
  if (result != AdaptiveCameraBatchResult::READY)
  {
    throw std::runtime_error(
      "failed to seal single-camera partial batch");
  }
  return batch;
}

livo::ReferenceOnlyMigrationRuntimeBundle publishRuntime(
  RuntimeAuthorityHarness & harness,
  const CameraObservationBatch & batch,
  const StatesGroup & state)
{
  livo::CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.05;
  config.max_imu_gap_sec = 0.01;
  livo::CameraTimeTransfer transfer(config);
  for (int index = 0; index <= 12; ++index)
  {
    livo::CameraTimeTransferImuSample sample;
    sample.timestamp_sec =
      batch.timestamp - 0.04 +
      0.005 * static_cast<double>(index);
    sample.angular_velocity_radps =
      Eigen::Vector3d(0.02, -0.01, 0.03);
    sample.specific_force_mps2 =
      Eigen::Vector3d(0.1, -0.2, 9.7);
    if (
      transfer.appendImu(sample) !=
      livo::CameraTimeTransferAppendStatus::kAccepted)
    {
      throw std::runtime_error("failed to append IMU sample");
    }
  }
  auto runtime = harness.publisher.publish(
    batch, state, batch.timestamp, 0.0, transfer);
  if (!runtime.has_value())
  {
    throw std::runtime_error("failed to publish runtime");
  }
  return std::move(*runtime);
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
    1.0e-4 * Eigen::Matrix<double, 6, 6>::Identity();
  return point;
}

std::vector<pointWithVar> geometryPoints()
{
  std::vector<pointWithVar> points;
  points.reserve(16U);
  for (int row = 0; row < 4; ++row)
  {
    for (int column = 0; column < 4; ++column)
    {
      pointWithVar point = geometryPoint();
      point.point_w = Eigen::Vector3d(
        0.8 * static_cast<double>(column) - 1.2,
        0.8 * static_cast<double>(row) - 1.2, 5.0);
      point.plane_identity_origin =
        Eigen::Vector3d(0.0, 0.0, 5.0);
      points.push_back(std::move(point));
    }
  }
  return points;
}

void initializeManager(
  VIOManager & manager,
  StatesGroup & state,
  StatesGroup & propagated,
  RuntimeAuthorityHarness & harness,
  const bool linearization_shadow_enabled,
  const bool single_channel_mode = false)
{
  constexpr std::array<const char *, 4U> kNames{{
    "front", "left", "back", "right"}};
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  manager.configureSharedLandmarkShadow(geometry_config);
  livo::LegacyLandmarkObservationShadowConfig
    observation_config;
  observation_config.enabled = true;
  manager.configureSharedLandmarkObservationShadow(
    observation_config);
  manager.configureSharedLandmarkTimeTransfer(
    livo::CameraTimeTransferConfig{});

  livo::CrossCameraMigrationShadowInitializationConfig
    cross_camera_config;
  cross_camera_config.enabled = true;
  cross_camera_config.four_camera_mode = true;
  livo::test_support::TrustedPhotometricEvidenceTestBundle
    photometric_bundle(640, 480);
  const auto trusted_photometric = photometric_bundle.load();
  if (
    !trusted_photometric.trustedCalibrationReady() ||
    !trusted_photometric.calibration.has_value())
  {
    throw std::runtime_error(
      "failed to load trusted photometric test calibration");
  }
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const auto parameters = pinholeParameters(camera_id);
    manager.registerCamera(
      camera_id, kNames[camera_id],
      livo::createCameraModel(parameters),
      M3D::Identity(), V3D::Zero());
    livo::ShadowPhotometricCapabilityEvidence evidence;
    evidence.mode = livo::PhotometricMode::kCalibratedDirect;
    evidence.strict_calibration =
      &*trusted_photometric.calibration;
    manager.configureSharedLandmarkPhotometric(
      camera_id, evidence);
    cross_camera_config.cameras[camera_id].camera_id =
      camera_id;
    cross_camera_config.cameras[camera_id].camera_name =
      kNames[camera_id];
    cross_camera_config.cameras[camera_id].projection =
      parameters;
    cross_camera_config.cameras[camera_id].
      projection_calibration_present = true;
    cross_camera_config.cameras[camera_id].
      extrinsic_calibration_present = true;
  }
  manager.initializeCrossCameraMigrationShadow(
    cross_camera_config);

  manager.configureReferenceOnlyMigrationRuntimeAuthorities(
    harness.queue.batchAuthorityIdentity(),
    harness.publisher.identity());
  livo::ReferenceOnlyMigrationShadowRuntimeConfig
    runtime_config;
  runtime_config.enabled = true;
  runtime_config.four_camera_mode = true;
  manager.configureReferenceOnlyMigrationShadowRuntime(
    runtime_config);
  livo::MultiCameraLandmarkLinearizationShadowConfig
    linearization_config;
  linearization_config.enabled =
    linearization_shadow_enabled;
  ASSERT_TRUE(
    manager.configureMultiCameraLandmarkLinearizationShadow(
      linearization_config));

  state.cov.setIdentity();
  propagated = state;
  manager.grid_size = 20;
  manager.patch_size = 8;
  manager.outlier_threshold = 400;
  manager.state = &state;
  manager.state_propagat = &propagated;
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
  manager.adaptive_policy_config.enabled = true;
  if (single_channel_mode)
  {
    manager.adaptive_policy_config.minimum_selected_cameras = 1U;
    manager.adaptive_policy_config.maximum_selected_cameras = 1U;
    manager.joint_update_config.min_joint_cameras = 1U;
  }
  manager.initializeVIO();
}

livo::ActivePlaneIdentitySnapshot activePlanes(
  livo::test_support::ActivePlaneAuthority & authority,
  const std::uint64_t generation)
{
  if (!authority.replace({
      livo::ActivePlaneDescriptor{
        {{91U, 1U}, 3U}, generation}}))
  {
    throw std::runtime_error("failed to replace active planes");
  }
  return authority.snapshot();
}

void processWithLidarPoints(
  VIOManager & manager,
  CameraObservationBatch & batch,
  const livo::ActivePlaneIdentitySnapshot & active_planes,
  const std::uint64_t generation,
  const livo::ReferenceOnlyMigrationRuntimeBundle * runtime,
  std::vector<pointWithVar> lidar_points)
{
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;
  manager.processFrames(
    batch, lidar_points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true, generation, &active_planes,
    runtime != nullptr ? &runtime->timeTransferBundle() : nullptr,
    runtime);
}

void process(
  VIOManager & manager,
  CameraObservationBatch & batch,
  const livo::ActivePlaneIdentitySnapshot & active_planes,
  const std::uint64_t generation,
  const livo::ReferenceOnlyMigrationRuntimeBundle * runtime)
{
  processWithLidarPoints(
    manager, batch, active_planes, generation, runtime,
    std::vector<pointWithVar>{geometryPoint()});
}

void processActiveFixture(
  VIOManager & manager,
  CameraObservationBatch & batch,
  const livo::ActivePlaneIdentitySnapshot & active_planes,
  const std::uint64_t generation,
  const livo::ReferenceOnlyMigrationRuntimeBundle * runtime)
{
  processWithLidarPoints(
    manager, batch, active_planes, generation, runtime,
    geometryPoints());
}

void processExactBatch(
  VIOManager & manager,
  RuntimeAuthorityHarness & harness,
  StatesGroup & state,
  const livo::ActivePlaneIdentitySnapshot & active_planes,
  const double timestamp_sec,
  const std::uint64_t generation)
{
  CameraObservationBatch batch =
    sealFourCameraBatch(harness, timestamp_sec);
  const auto runtime = publishRuntime(harness, batch, state);
  processActiveFixture(
    manager, batch, active_planes, generation, &runtime);
}

void expectStructuralIdentity(
  const livo::VIOReadOnlyStructuralAudit & left,
  const livo::VIOReadOnlyStructuralAudit & right)
{
  ASSERT_TRUE(left.active_state_available);
  ASSERT_TRUE(right.active_state_available);
  EXPECT_EQ(
    std::memcmp(
      left.active_state.data(), right.active_state.data(),
      sizeof(left.active_state)),
    0);
  ASSERT_TRUE(left.propagated_state_available);
  ASSERT_TRUE(right.propagated_state_available);
  EXPECT_EQ(
    std::memcmp(
      left.propagated_state.data(),
      right.propagated_state.data(),
      sizeof(left.propagated_state)),
    0);
  EXPECT_EQ(
    left.shared_landmark_shadow.geometry.value,
    right.shared_landmark_shadow.geometry.value);
  EXPECT_EQ(
    left.shared_landmark_shadow.store_revision,
    right.shared_landmark_shadow.store_revision);
  EXPECT_EQ(
    left.shared_landmark_shadow.landmark_count,
    right.shared_landmark_shadow.landmark_count);
  EXPECT_EQ(
    left.shared_landmark_shadow.observation_count,
    right.shared_landmark_shadow.observation_count);
  EXPECT_EQ(
    left.legacy_feature_voxel_counts,
    right.legacy_feature_voxel_counts);
  EXPECT_EQ(
    left.legacy_feature_point_counts,
    right.legacy_feature_point_counts);
  EXPECT_EQ(
    left.last_joint_solver.accepted,
    right.last_joint_solver.accepted);
  EXPECT_EQ(
    left.last_joint_solver.reason,
    right.last_joint_solver.reason);
  EXPECT_EQ(
    left.last_joint_solver.contributing_cameras,
    right.last_joint_solver.contributing_cameras);
  EXPECT_EQ(
    left.last_joint_solver.tracked_points,
    right.last_joint_solver.tracked_points);
}

}  // namespace

TEST(
  VIOMultiCameraLandmarkLinearizationShadow,
  DefaultConfigurationHasNoRuntimeEffect)
{
  VIOManager manager;

  const auto & audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();

  EXPECT_EQ(
    audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::kDisabled);
  EXPECT_FALSE(audit.process_called);
  EXPECT_FALSE(audit.batch_status.has_value());
  EXPECT_EQ(audit.accepted_target_generation, 0U);
  EXPECT_FALSE(audit.has_linearization);
  EXPECT_FALSE(audit.consumable);
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  VIOMultiCameraLandmarkLinearizationShadow,
  EnabledWithoutRequiredOwnersRemainsUnavailable)
{
  VIOManager manager;
  const auto parameters = pinholeParameters(0U);
  manager.registerCamera(
    0U, "front", livo::createCameraModel(parameters),
    M3D::Identity(), V3D::Zero());
  manager.grid_size = 20;
  manager.patch_size = 8;
  manager.grid_n_height = 24;
  manager.raycast_en = false;
  manager.adaptive_policy_config.enabled = true;
  livo::MultiCameraLandmarkLinearizationShadowConfig config;
  config.enabled = true;
  ASSERT_TRUE(
    manager.configureMultiCameraLandmarkLinearizationShadow(
      config));

  manager.initializeVIO();

  const auto & audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_TRUE(audit.configured_enabled);
  EXPECT_FALSE(audit.module_active);
  EXPECT_EQ(
    audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::
      kModuleUnavailable);
  EXPECT_FALSE(audit.process_called);
  EXPECT_EQ(audit.accepted_target_generation, 0U);
  EXPECT_FALSE(
    manager.configureMultiCameraLandmarkLinearizationShadow(
      config));
}

TEST(
  VIOMultiCameraLandmarkLinearizationShadow,
  ExactRuntimeAndOwnerSealedCapabilityReachMetricsOnlyModule)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  RuntimeAuthorityHarness harness;
  initializeManager(
    manager, state, propagated, harness, true);
  livo::test_support::ActivePlaneAuthority authority;
  const auto active_planes = activePlanes(authority, 41U);
  const auto entry_counts_before =
    manager.readOnlyStructuralAudit();

  testing::internal::CaptureStdout();
  processExactBatch(
    manager, harness, state, active_planes, 5.0, 41U);
  processExactBatch(
    manager, harness, state, active_planes, 5.1, 41U);
  const auto tracked_audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  const auto tracked_structure =
    manager.readOnlyStructuralAudit();
  EXPECT_GT(
    tracked_structure.shared_landmark_shadow_entry_count,
    entry_counts_before.shared_landmark_shadow_entry_count);
  EXPECT_GT(
    tracked_structure.
    reference_only_migration_shadow_entry_count,
    entry_counts_before.
    reference_only_migration_shadow_entry_count);
  EXPECT_GT(
    tracked_structure.
    multi_camera_linearization_shadow_entry_count,
    entry_counts_before.
    multi_camera_linearization_shadow_entry_count);
  const auto & runtime_audit =
    manager.adaptiveVisualRuntimeAudit();
  EXPECT_TRUE(runtime_audit.shared_landmark_shadow_called);
  EXPECT_TRUE(
    runtime_audit.reference_only_migration_shadow_called);
  EXPECT_TRUE(
    runtime_audit.multi_camera_linearization_shadow_called);
  SCOPED_TRACE(
    testing::Message()
    << "tracked_solver_accepted="
    << tracked_structure.last_joint_solver.accepted
    << " tracked_solver_reason="
    << tracked_structure.last_joint_solver.reason
    << " tracked_points="
    << tracked_structure.last_joint_solver.tracked_points
    << " store_revision="
    << tracked_structure.shared_landmark_shadow.store_revision
    << " landmarks="
    << tracked_structure.shared_landmark_shadow.landmark_count
    << " observations="
    << tracked_structure.shared_landmark_shadow.observation_count);
  ASSERT_TRUE(tracked_audit.batch_status.has_value());
  EXPECT_EQ(
    *tracked_audit.batch_status,
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_GT(tracked_audit.accepted_target_generation, 0U);
  processExactBatch(
    manager, harness, state, active_planes, 5.2, 41U);
  processExactBatch(
    manager, harness, state, active_planes, 5.3, 41U);
  const std::string output =
    testing::internal::GetCapturedStdout();

  const auto & audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_EQ(
    audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::kProcessed);
  EXPECT_TRUE(audit.module_active);
  EXPECT_TRUE(audit.plan_capability_present);
  EXPECT_TRUE(audit.plan_capability_validated);
  EXPECT_TRUE(audit.process_called);
  ASSERT_TRUE(audit.batch_status.has_value());
  EXPECT_EQ(
    *audit.batch_status,
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  SCOPED_TRACE(
    testing::Message()
    << "inspected=" << audit.inspected_source_observations
    << " reference_only=" << audit.skipped_reference_only_sources
    << " tombstoned=" << audit.skipped_tombstoned_sources
    << " inactive_plane=" << audit.skipped_inactive_plane_sources
    << " geometry=" << audit.skipped_sampling_geometry
    << " order=" << audit.skipped_source_order
    << " runtime=" << audit.skipped_source_runtime_authority
    << " age=" << audit.skipped_source_age);
  EXPECT_GT(audit.accepted_target_generation, 0U);
  EXPECT_GT(
    audit.accepted_target_generation,
    tracked_audit.accepted_target_generation);
  EXPECT_GT(audit.candidate_evaluations, 0U);
  EXPECT_GT(audit.decision_count, 0U);
  EXPECT_GT(audit.selection_trace_count, 0U);
  EXPECT_EQ(audit.skipped_inactive_plane_sources, 0U);
  EXPECT_FALSE(audit.has_linearization);
  EXPECT_FALSE(audit.consumable);
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);

  const std::string prefix =
    "[VIO_MULTI_CAMERA_LANDMARK_LINEARIZATION_SHADOW]";
  const std::size_t metric_begin = output.find(prefix);
  ASSERT_NE(metric_begin, std::string::npos);
  const std::size_t metric_end =
    output.find('\n', metric_begin);
  ASSERT_NE(metric_end, std::string::npos);
  EXPECT_LT(metric_end - metric_begin + 1U, PIPE_BUF);
  const std::string metric = output.substr(
    metric_begin, metric_end - metric_begin);
  EXPECT_NE(metric.find("invocation=processed"), std::string::npos);
  EXPECT_NE(metric.find("process_called=1"), std::string::npos);
  EXPECT_NE(
    metric.find("skipped_inactive_plane=0"),
    std::string::npos);
  EXPECT_NE(metric.find("has_linearization=0"), std::string::npos);
  EXPECT_NE(metric.find("consumable=0"), std::string::npos);
  EXPECT_NE(metric.find("production_ready=0"), std::string::npos);
  EXPECT_NE(metric.find("solver_applied=0"), std::string::npos);

  manager.initializeVIO();
  const auto & reinitialized_audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_EQ(
    reinitialized_audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::kNotEvaluated);
  EXPECT_TRUE(reinitialized_audit.module_active);
  EXPECT_FALSE(reinitialized_audit.process_called);
  EXPECT_FALSE(reinitialized_audit.batch_status.has_value());
  EXPECT_EQ(reinitialized_audit.accepted_target_generation, 0U);
}

TEST(
  VIOMultiCameraLandmarkLinearizationShadow,
  MissingForeignAndReplayedRuntimeNeverInvokeModule)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  RuntimeAuthorityHarness expected;
  RuntimeAuthorityHarness foreign;
  initializeManager(
    manager, state, propagated, expected, true);
  livo::test_support::ActivePlaneAuthority authority;

  CameraObservationBatch missing_batch =
    sealFourCameraBatch(expected, 6.0);
  auto active_planes = activePlanes(authority, 51U);
  process(
    manager, missing_batch, active_planes, 51U, nullptr);
  auto audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_EQ(
    audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::
      kMissingRuntimeBundle);
  EXPECT_FALSE(audit.process_called);
  EXPECT_FALSE(audit.plan_capability_present);
  EXPECT_FALSE(audit.plan_capability_validated);
  const std::uint64_t generation_after_missing =
    audit.accepted_target_generation;

  CameraObservationBatch foreign_batch =
    sealFourCameraBatch(foreign, 7.0);
  const auto foreign_runtime =
    publishRuntime(foreign, foreign_batch, state);
  active_planes = activePlanes(authority, 52U);
  process(
    manager, foreign_batch, active_planes, 52U,
    &foreign_runtime);
  audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_EQ(
    audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::
      kRuntimeAuthorityNotReady);
  EXPECT_EQ(
    audit.runtime_authority,
    livo::VIORuntimeObservationAuthorityStatus::
      kRuntimePublisherAuthorityMismatch);
  EXPECT_FALSE(audit.process_called);
  EXPECT_FALSE(audit.plan_capability_present);
  EXPECT_FALSE(audit.plan_capability_validated);
  EXPECT_EQ(
    audit.accepted_target_generation,
    generation_after_missing);

  CameraObservationBatch exact_batch =
    sealFourCameraBatch(expected, 8.0);
  const auto exact_runtime =
    publishRuntime(expected, exact_batch, state);
  active_planes = activePlanes(authority, 53U);
  process(
    manager, exact_batch, active_planes, 53U,
    &exact_runtime);
  audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  ASSERT_EQ(
    audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::kProcessed);
  EXPECT_TRUE(audit.plan_capability_present);
  EXPECT_TRUE(audit.plan_capability_validated);
  const std::uint64_t generation_after_exact =
    audit.accepted_target_generation;

  process(
    manager, exact_batch, active_planes, 53U,
    &exact_runtime);
  audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_EQ(
    audit.invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::
      kRuntimeAuthorityNotReady);
  EXPECT_EQ(
    audit.runtime_authority,
    livo::VIORuntimeObservationAuthorityStatus::
      kReplayedRuntimeBundle);
  EXPECT_FALSE(audit.process_called);
  EXPECT_FALSE(audit.plan_capability_present);
  EXPECT_FALSE(audit.plan_capability_validated);
  EXPECT_EQ(
    audit.accepted_target_generation,
    generation_after_exact);
}

TEST(
  VIOMultiCameraLandmarkLinearizationShadow,
  AdaptiveSingleChannelPartialBatchReachesSolverWithoutInvokingFullFourShadow)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  RuntimeAuthorityHarness harness;
  initializeManager(
    manager, state, propagated, harness, true, true);
  livo::test_support::ActivePlaneAuthority authority;
  const auto active_planes = activePlanes(authority, 54U);

  for (const double timestamp_sec :
       std::array<double, 4U>{{7.1, 7.2, 7.3, 7.4}})
  {
    processExactBatch(
      manager, harness, state, active_planes,
      timestamp_sec, 54U);
  }

  CameraObservationBatch partial_batch =
    sealSingleCameraPartialBatch(harness, 2U, 7.5);
  ASSERT_TRUE(partial_batch.partial);
  ASSERT_EQ(partial_batch.camera_count, 1U);
  ASSERT_TRUE(partial_batch.present[2U]);
  const auto entries_before =
    manager.readOnlyStructuralAudit();

  processActiveFixture(
    manager, partial_batch, active_planes, 54U, nullptr);

  const auto structure = manager.readOnlyStructuralAudit();
  EXPECT_EQ(
    structure.shared_landmark_shadow_entry_count,
    entries_before.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    structure.reference_only_migration_shadow_entry_count,
    entries_before.
    reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    structure.multi_camera_linearization_shadow_entry_count,
    entries_before.
    multi_camera_linearization_shadow_entry_count);
  const auto & runtime_audit =
    manager.adaptiveVisualRuntimeAudit();
  EXPECT_FALSE(runtime_audit.shared_landmark_shadow_called);
  EXPECT_FALSE(
    runtime_audit.reference_only_migration_shadow_called);
  EXPECT_FALSE(
    runtime_audit.multi_camera_linearization_shadow_called);
  EXPECT_NE(
    structure.last_joint_solver.reason,
    "incomplete_camera_batch");
  const auto & shadow_audit =
    manager.multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_FALSE(shadow_audit.process_called);
  EXPECT_FALSE(shadow_audit.plan_capability_present);
  EXPECT_FALSE(shadow_audit.plan_capability_validated);
  EXPECT_FALSE(shadow_audit.has_linearization);
  EXPECT_FALSE(shadow_audit.consumable);
  EXPECT_FALSE(shadow_audit.production_ready);
  EXPECT_FALSE(shadow_audit.solver_applied);
}

TEST(
  VIOMultiCameraLandmarkLinearizationShadow,
  OwnerSealedInvalidPhotometricTargetFailsClosed)
{
  for (
    const livo::PhotometricStatus invalid_status :
    std::array<livo::PhotometricStatus, 2U>{{
      livo::PhotometricStatus::kMissingExposureGain,
      livo::PhotometricStatus::kInvalidInput}})
  {
    SCOPED_TRACE(
      livo::photometricStatusName(invalid_status));
    VIOManager manager;
    StatesGroup state;
    StatesGroup propagated;
    RuntimeAuthorityHarness harness;
    initializeManager(
      manager, state, propagated, harness, true);
    livo::test_support::ActivePlaneAuthority authority;
    const auto active_planes = activePlanes(authority, 56U);

    for (const double timestamp_sec :
         std::array<double, 4U>{{8.1, 8.2, 8.3, 8.4}})
    {
      processExactBatch(
        manager, harness, state, active_planes,
        timestamp_sec, 56U);
    }
    const auto accepted_audit =
      manager.multiCameraLandmarkLinearizationShadowBatchAudit();
    ASSERT_TRUE(accepted_audit.batch_status.has_value());
    ASSERT_EQ(
      *accepted_audit.batch_status,
      livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
    ASSERT_GT(accepted_audit.accepted_target_generation, 0U);
    const auto structure_before =
      manager.readOnlyStructuralAudit();

    CameraObservationBatch invalid_batch =
      sealFourCameraBatch(
      harness, 8.5, invalid_status);
    const auto invalid_runtime =
      publishRuntime(harness, invalid_batch, state);
    for (std::uint8_t camera_id = 0U;
         camera_id < 4U; ++camera_id)
    {
      ASSERT_EQ(
        invalid_runtime.validatedComposition().
        photometricStatus(camera_id),
        invalid_status);
    }
    processActiveFixture(
      manager, invalid_batch, active_planes, 56U,
      &invalid_runtime);

    const auto & rejected_audit =
      manager.multiCameraLandmarkLinearizationShadowBatchAudit();
    ASSERT_TRUE(rejected_audit.batch_status.has_value());
    EXPECT_EQ(
      rejected_audit.invocation,
      livo::VIOMultiCameraLandmarkShadowInvocation::kProcessed);
    EXPECT_TRUE(rejected_audit.process_called);
    EXPECT_EQ(
      *rejected_audit.batch_status,
      livo::MultiCameraLandmarkShadowBatchStatus::
      kTargetCompositionRejected);
    EXPECT_EQ(
      rejected_audit.accepted_target_generation,
      accepted_audit.accepted_target_generation);
    EXPECT_EQ(rejected_audit.candidate_evaluations, 0U);
    EXPECT_EQ(rejected_audit.decision_count, 0U);
    EXPECT_EQ(rejected_audit.selection_trace_count, 0U);

    const auto structure_after =
      manager.readOnlyStructuralAudit();
    EXPECT_EQ(
      structure_after.shared_landmark_shadow.store_revision,
      structure_before.shared_landmark_shadow.store_revision);
    EXPECT_EQ(
      structure_after.shared_landmark_shadow.observation_count,
      structure_before.shared_landmark_shadow.observation_count);
  }
}

TEST(
  VIOMultiCameraLandmarkLinearizationShadow,
  EnabledAndDisabledRunsAreStructurallyIdentical)
{
  VIOManager enabled_manager;
  VIOManager disabled_manager;
  StatesGroup enabled_state;
  StatesGroup disabled_state;
  enabled_state.cov.setIdentity();
  disabled_state = enabled_state;
  StatesGroup enabled_propagated;
  StatesGroup disabled_propagated;
  RuntimeAuthorityHarness enabled_harness;
  RuntimeAuthorityHarness disabled_harness;
  initializeManager(
    enabled_manager, enabled_state, enabled_propagated,
    enabled_harness, true);
  initializeManager(
    disabled_manager, disabled_state, disabled_propagated,
    disabled_harness, false);
  livo::test_support::ActivePlaneAuthority
    enabled_plane_authority;
  livo::test_support::ActivePlaneAuthority
    disabled_plane_authority;
  const auto enabled_active_planes =
    activePlanes(enabled_plane_authority, 61U);
  const auto disabled_active_planes =
    activePlanes(disabled_plane_authority, 61U);
  for (const double timestamp_sec :
       std::array<double, 4U>{{9.0, 9.1, 9.2, 9.3}})
  {
    processExactBatch(
      enabled_manager, enabled_harness, enabled_state,
      enabled_active_planes, timestamp_sec, 61U);
    processExactBatch(
      disabled_manager, disabled_harness, disabled_state,
      disabled_active_planes, timestamp_sec, 61U);
  }

  EXPECT_EQ(
    enabled_manager.
      multiCameraLandmarkLinearizationShadowBatchAudit().
      invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::kProcessed);
  EXPECT_EQ(
    disabled_manager.
      multiCameraLandmarkLinearizationShadowBatchAudit().
      invocation,
    livo::VIOMultiCameraLandmarkShadowInvocation::kDisabled);
  const auto & enabled_audit =
    enabled_manager.
    multiCameraLandmarkLinearizationShadowBatchAudit();
  ASSERT_TRUE(enabled_audit.batch_status.has_value());
  EXPECT_EQ(
    *enabled_audit.batch_status,
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_GT(enabled_audit.accepted_target_generation, 0U);
  EXPECT_GT(enabled_audit.candidate_evaluations, 0U);
  EXPECT_GT(enabled_audit.decision_count, 0U);
  EXPECT_GT(enabled_audit.selection_trace_count, 0U);
  EXPECT_EQ(
    enabled_audit.selection_trace_count,
    enabled_audit.same_camera_trace_count);
  EXPECT_FALSE(enabled_audit.has_linearization);
  EXPECT_FALSE(enabled_audit.consumable);
  EXPECT_FALSE(enabled_audit.production_ready);
  EXPECT_FALSE(enabled_audit.solver_applied);
  const auto & disabled_audit =
    disabled_manager.
    multiCameraLandmarkLinearizationShadowBatchAudit();
  EXPECT_FALSE(disabled_audit.has_linearization);
  EXPECT_FALSE(disabled_audit.consumable);
  EXPECT_FALSE(disabled_audit.production_ready);
  EXPECT_FALSE(disabled_audit.solver_applied);
  expectStructuralIdentity(
    enabled_manager.readOnlyStructuralAudit(),
    disabled_manager.readOnlyStructuralAudit());
}
