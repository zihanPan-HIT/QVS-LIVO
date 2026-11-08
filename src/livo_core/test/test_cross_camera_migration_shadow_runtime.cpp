#include "LIVMapper.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

#if LIVO_BUILD_EXPERIMENTAL_SHADOWS
std::string fingerprintHex(
  const livo::FourCameraPinholeRigFingerprint & fingerprint)
{
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result;
  result.reserve(fingerprint.size() * 2U);
  for (const std::uint8_t value : fingerprint)
  {
    result.push_back(kHexDigits[value >> 4U]);
    result.push_back(kHexDigits[value & 0x0fU]);
  }
  return result;
}
#endif

rclcpp::NodeOptions outdoorOptions(
  const std::initializer_list<std::string> parameter_overrides)
{
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  std::vector<std::string> arguments{
    "--ros-args",
    "--params-file",
    LIVO_TEST_AIRY_VIO_CONFIG,
    "-p",
    "vio.adaptive.metrics.enable:=true"};
  for (const std::string & parameter_override : parameter_overrides)
  {
    arguments.emplace_back("-p");
    arguments.push_back(parameter_override);
  }
  options.arguments(arguments);
  return options;
}

#if LIVO_BUILD_EXPERIMENTAL_SHADOWS
rclcpp::NodeOptions minimalFourCameraOptions()
{
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  options.append_parameter_override(
    "vio.camera_mode", std::string("four"));
  options.append_parameter_override(
    "vio.multiview.cross_camera_migration_shadow_enable", true);
  options.append_parameter_override("vio.exposure_estimate_en", false);
  options.append_parameter_override("common.img_en", 0);
  options.append_parameter_override(
    "extrin_calib.extrinsic_T",
    std::vector<double>{0.0, 0.0, 0.0});
  options.append_parameter_override(
    "extrin_calib.extrinsic_R",
    std::vector<double>{
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0});
  return options;
}

void appendPinholeProjectionCalibration(
  rclcpp::NodeOptions & options, const std::string & camera_name,
  const bool include_d3 = true)
{
  const std::string prefix =
    "multi_camera.cameras." + camera_name + ".";
  options.append_parameter_override(
    prefix + "model", std::string("Pinhole"));
  options.append_parameter_override(prefix + "width", 640);
  options.append_parameter_override(prefix + "height", 480);
  options.append_parameter_override(prefix + "scale", 1.0);
  options.append_parameter_override(prefix + "fx", 600.0);
  options.append_parameter_override(prefix + "fy", 600.0);
  options.append_parameter_override(prefix + "cx", 320.0);
  options.append_parameter_override(prefix + "cy", 240.0);
  options.append_parameter_override(prefix + "d0", 0.0);
  options.append_parameter_override(prefix + "d1", 0.0);
  options.append_parameter_override(prefix + "d2", 0.0);
  if (include_d3)
  {
    options.append_parameter_override(prefix + "d3", 0.0);
  }
}

void appendIdentityVioExtrinsic(
  rclcpp::NodeOptions & options, const std::string & camera_name,
  const bool include_translation = true,
  const bool include_rotation = true)
{
  const std::string prefix =
    "multi_camera.vio_extrinsics." + camera_name + ".";
  if (include_rotation)
  {
    options.append_parameter_override(
      prefix + "R",
      std::vector<double>{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0});
  }
  if (include_translation)
  {
    options.append_parameter_override(
      prefix + "T", std::vector<double>{0.0, 0.0, 0.0});
  }
}

void expectShadowNotReadyAndVioCamerasAlive(
  const LIVMapper & mapper, const std::string & reason)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(audit.reason, reason);
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  for (std::size_t camera_id = 0U;
       camera_id < kNames.size();
       ++camera_id)
  {
    EXPECT_FALSE(audit.cameras[camera_id].has_value());
    ASSERT_NE(mapper.vio_manager->cameraModel(camera_id), nullptr);
    EXPECT_EQ(
      mapper.vio_manager->cameraName(camera_id), kNames[camera_id]);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);

  const auto & runtime_audit =
    mapper.vio_manager->
    referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_FALSE(runtime_audit.enabled);
  EXPECT_FALSE(runtime_audit.effective);
  EXPECT_EQ(
    runtime_audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kDisabled);
}

Eigen::Matrix3d rowMajorMatrix(
  const std::array<double, 9U> & values)
{
  Eigen::Matrix3d result;
  result <<
    values[0U], values[1U], values[2U],
    values[3U], values[4U], values[5U],
    values[6U], values[7U], values[8U];
  return result;
}
#endif

class CrossCameraMigrationShadowRuntimeTest :
  public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok())
    {
      int argc = 0;
      char ** argv = nullptr;
      rclcpp::init(argc, argv);
      owns_rclcpp_context_ = true;
    }
  }

  static void TearDownTestSuite()
  {
    if (owns_rclcpp_context_)
    {
      rclcpp::shutdown();
    }
  }

private:
  static bool owns_rclcpp_context_;
};

bool CrossCameraMigrationShadowRuntimeTest::
  owns_rclcpp_context_{false};

void initializeRuntimePublishers(LIVMapper & mapper)
{
  rclcpp::Node::SharedPtr node = mapper.node;
  image_transport::ImageTransport image_transport(node);
  mapper.initializeSubscribersAndPublishers(
    node, image_transport);
}

void populatePlanarLidarFeatures(LIVMapper & mapper)
{
  mapper.feats_undistort->clear();
  for (int row = 0; row < 6; ++row)
  {
    for (int column = 0; column < 8; ++column)
    {
      PointType point;
      point.x = 0.4F * static_cast<float>(column);
      point.y = 0.4F * static_cast<float>(row);
      point.z = 2.0F;
      point.curvature = 0.0F;
      mapper.feats_undistort->push_back(point);
    }
  }
}

std::size_t countOccurrences(
  const std::string & text, const std::string & needle)
{
  std::size_t count = 0U;
  std::size_t position = 0U;
  while ((position = text.find(needle, position)) !=
         std::string::npos)
  {
    ++count;
    position += needle.size();
  }
  return count;
}

}  // namespace

#if LIVO_BUILD_EXPERIMENTAL_SHADOWS
TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  OutdoorParametersKeepShadowDisabledByDefault)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_off_test",
    outdoorOptions({}));

  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  EXPECT_FALSE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(audit.reason, "disabled");
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);

  const auto & runtime_audit =
    mapper.vio_manager->
    referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_FALSE(runtime_audit.enabled);
  EXPECT_FALSE(runtime_audit.effective);
  EXPECT_EQ(
    runtime_audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kDisabled);
}
#endif

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  AdaptivePartialSchedulerEmitsSingleCameraVioBatchAfterTimeout)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "adaptive_partial_scheduler_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.partial_batch.enable:=true",
      "vio.adaptive.partial_batch.missing_camera_timeout_ms:=10.0",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  mapper.slam_mode_ = LIVO;
  mapper.joint_vio_warmup_sec_ = 0.0;
  mapper.exposure_time_init = 0.0;
  mapper.last_timestamp_imu = 2.2;

  CameraObservation observation;
  observation.camera_id = 2U;
  observation.timestamp = 2.0;
  observation.image = cv::Mat(16, 16, CV_8UC1, cv::Scalar(128));
  ASSERT_EQ(
    mapper.camera_observation_queue_.push(std::move(observation)),
    CameraObservationPushResult::ACCEPTED);

  PointCloudXYZI::Ptr lidar(new PointCloudXYZI());
  PointType lidar_point;
  lidar_point.curvature = 0.0F;
  lidar->points.push_back(lidar_point);
  mapper.lid_raw_data_buffer.push_back(lidar);
  mapper.lid_header_time_buffer.push_back(1.9);

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp.sec = 2;
  imu->header.stamp.nanosec = 200000000U;
  mapper.imu_buffer.push_back(imu);

  LidarMeasureGroup measurements;
  measurements.last_lio_update_time = 2.0;
  measurements.lio_vio_flg = WAIT;

  ASSERT_TRUE(mapper.sync_packages(measurements));
  ASSERT_EQ(measurements.lio_vio_flg, VIO);
  ASSERT_EQ(measurements.measures.size(), 1U);
  const CameraObservationBatch & batch =
    measurements.measures.front().camera_batch;
  EXPECT_TRUE(batch.partial);
  EXPECT_EQ(batch.camera_count, 1U);
  EXPECT_TRUE(batch.present[2U]);
  EXPECT_EQ(batch.observations[2U].camera_id, 2U);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  AdaptiveSwitchPolicyParametersReachRuntimeConfiguration)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "adaptive_switch_policy_parameters_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.adaptive.selection_switch_margin:=0.23",
      "vio.adaptive.selection_challenger_frames:=7",
      "vio.min_joint_cameras:=1",
    }));

  EXPECT_EQ(
    mapper.adaptive_policy_config_.maximum_selected_cameras,
    1U);
  EXPECT_DOUBLE_EQ(
    mapper.adaptive_policy_config_.selection_switch_margin,
    0.23);
  EXPECT_EQ(
    mapper.adaptive_policy_config_.selection_challenger_frames,
    7U);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  AdaptiveSwitchPolicyRejectsZeroChallengerFrames)
{
  rclcpp::Node::SharedPtr unused_node;

  EXPECT_THROW(
    {
      LIVMapper mapper(
        unused_node,
        "adaptive_switch_policy_invalid_challenger_frames_test",
        outdoorOptions({
          "vio.adaptive.selection_challenger_frames:=0",
        }));
    },
    std::runtime_error);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  AdaptiveSingleChannelModeWaitsForVisualDeadlineBeforeLioFallback)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "adaptive_visual_deadline_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.partial_batch.enable:=true",
      "vio.adaptive.partial_batch.missing_camera_timeout_ms:=30.0",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  mapper.slam_mode_ = LIVO;
  mapper.last_timestamp_imu = 1.14;

  PointCloudXYZI::Ptr lidar(new PointCloudXYZI());
  PointType first_lidar_point;
  first_lidar_point.curvature = 0.0F;
  lidar->points.push_back(first_lidar_point);
  PointType last_lidar_point;
  last_lidar_point.curvature = 100.0F;
  lidar->points.push_back(last_lidar_point);
  mapper.lid_raw_data_buffer.push_back(lidar);
  mapper.lid_header_time_buffer.push_back(1.0);

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp.sec = 1;
  imu->header.stamp.nanosec = 140000000U;
  mapper.imu_buffer.push_back(imu);

  LidarMeasureGroup measurements;
  measurements.lio_vio_flg = WAIT;

  EXPECT_FALSE(mapper.sync_packages(measurements));
  EXPECT_EQ(measurements.lio_vio_flg, WAIT);
  EXPECT_TRUE(measurements.measures.empty());
  EXPECT_EQ(mapper.lid_raw_data_buffer.size(), 1U);
  EXPECT_EQ(mapper.lid_header_time_buffer.size(), 1U);
  EXPECT_EQ(mapper.imu_buffer.size(), 1U);
  EXPECT_FALSE(mapper.lidar_pushed);

  mapper.last_timestamp_imu = 1.16;
  ASSERT_TRUE(mapper.sync_packages(measurements));
  EXPECT_EQ(measurements.lio_vio_flg, LIO);
  ASSERT_EQ(measurements.measures.size(), 1U);
  EXPECT_NEAR(measurements.measures.back().lio_time, 1.1, 1.0e-9);
  EXPECT_TRUE(mapper.lid_raw_data_buffer.empty());
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  AdaptiveLioFallbackResumesRecoveredPartialCameraWithoutLostState)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "adaptive_lio_recovery_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.partial_batch.enable:=true",
      "vio.adaptive.partial_batch.missing_camera_timeout_ms:=30.0",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  mapper.slam_mode_ = LIVO;
  mapper.joint_vio_warmup_sec_ = 0.0;
  mapper.exposure_time_init = 0.0;
  mapper.last_timestamp_imu = 1.16;

  PointCloudXYZI::Ptr first_lidar(new PointCloudXYZI());
  PointType first_scan_begin;
  first_scan_begin.curvature = 0.0F;
  first_lidar->points.push_back(first_scan_begin);
  PointType first_scan_end;
  first_scan_end.curvature = 100.0F;
  first_lidar->points.push_back(first_scan_end);
  mapper.lid_raw_data_buffer.push_back(first_lidar);
  mapper.lid_header_time_buffer.push_back(1.0);

  auto first_imu = std::make_shared<sensor_msgs::msg::Imu>();
  first_imu->header.stamp.sec = 1;
  first_imu->header.stamp.nanosec = 160000000U;
  mapper.imu_buffer.push_back(first_imu);

  LidarMeasureGroup measurements;
  measurements.lio_vio_flg = WAIT;

  ASSERT_TRUE(mapper.sync_packages(measurements));
  ASSERT_EQ(measurements.lio_vio_flg, LIO);
  ASSERT_EQ(measurements.measures.size(), 1U);
  EXPECT_FALSE(mapper.pending_vio_batch_.has_value());

  measurements.last_lio_update_time =
    measurements.measures.back().lio_time;
  measurements.measures.clear();

  CameraObservation recovered_observation;
  recovered_observation.camera_id = 1U;
  recovered_observation.timestamp =
    measurements.last_lio_update_time;
  recovered_observation.image =
    cv::Mat(16, 16, CV_8UC1, cv::Scalar(128));
  ASSERT_EQ(
    mapper.camera_observation_queue_.push(
      std::move(recovered_observation)),
    CameraObservationPushResult::ACCEPTED);

  PointCloudXYZI::Ptr second_lidar(new PointCloudXYZI());
  PointType second_scan_begin;
  second_scan_begin.curvature = 0.0F;
  second_lidar->points.push_back(second_scan_begin);
  PointType second_scan_end;
  second_scan_end.curvature = 100.0F;
  second_lidar->points.push_back(second_scan_end);
  mapper.lid_raw_data_buffer.push_back(second_lidar);
  mapper.lid_header_time_buffer.push_back(1.1);

  auto second_imu = std::make_shared<sensor_msgs::msg::Imu>();
  second_imu->header.stamp.sec = 1;
  second_imu->header.stamp.nanosec = 300000000U;
  mapper.imu_buffer.push_back(second_imu);
  mapper.last_timestamp_imu = 1.3;

  EXPECT_FALSE(mapper.pending_vio_batch_.has_value());
  ASSERT_TRUE(mapper.sync_packages(measurements));
  EXPECT_EQ(measurements.lio_vio_flg, VIO);
  ASSERT_EQ(measurements.measures.size(), 1U);
  const CameraObservationBatch & recovered_batch =
    measurements.measures.front().camera_batch;
  EXPECT_TRUE(recovered_batch.partial);
  EXPECT_EQ(recovered_batch.camera_count, 1U);
  EXPECT_TRUE(recovered_batch.present[1U]);
  EXPECT_FALSE(mapper.pending_vio_batch_.has_value());
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  AdaptiveSingleChannelModeContinuesLioWhenAllCamerasDropOut)
{
  std::string shutdown_output;
  rclcpp::Node::SharedPtr unused_node;
  auto mapper_owner = std::make_unique<LIVMapper>(
    unused_node,
    "adaptive_all_camera_dropout_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.partial_batch.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  LIVMapper & mapper = *mapper_owner;
  initializeRuntimePublishers(mapper);
  mapper.slam_mode_ = LIVO;
  mapper.last_timestamp_imu = 1.2;

  PointCloudXYZI::Ptr lidar(new PointCloudXYZI());
  PointType first_lidar_point;
  first_lidar_point.curvature = 0.0F;
  lidar->points.push_back(first_lidar_point);
  PointType last_lidar_point;
  last_lidar_point.curvature = 100.0F;
  lidar->points.push_back(last_lidar_point);
  mapper.lid_raw_data_buffer.push_back(lidar);
  mapper.lid_header_time_buffer.push_back(1.0);

  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp.sec = 1;
  imu->header.stamp.nanosec = 200000000U;
  mapper.imu_buffer.push_back(imu);

  mapper.LidarMeasures.lio_vio_flg = WAIT;

  ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
  EXPECT_EQ(mapper.LidarMeasures.lio_vio_flg, LIO);
  ASSERT_EQ(mapper.LidarMeasures.measures.size(), 1U);
  EXPECT_NEAR(
    mapper.LidarMeasures.measures.back().lio_time,
    1.1, 1.0e-9);
  EXPECT_EQ(
    mapper.LidarMeasures.measures.back().
    camera_batch.camera_count,
    0U);
  EXPECT_FALSE(mapper.pending_vio_batch_.has_value());
  EXPECT_TRUE(mapper.lid_raw_data_buffer.empty());

  mapper.LidarMeasures.last_lio_update_time = 1.1;
  mapper._state.cov.setIdentity();
  mapper.state_propagat = mapper._state;
  mapper.voxelmap_manager->state_ = mapper._state;
  populatePlanarLidarFeatures(mapper);

  testing::internal::CaptureStdout();
  mapper.handleLIO();
  const std::string first_output =
    testing::internal::GetCapturedStdout();

  const livo::AdaptiveLioModePublicationAudit pending_first =
    mapper.adaptiveLioModeAudit();
  EXPECT_FALSE(pending_first.receipt.has_value());
  EXPECT_FALSE(pending_first.metric_emitted);
  EXPECT_EQ(
    countOccurrences(
      first_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    0U);
  EXPECT_EQ(
    countOccurrences(
      first_output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);

  mapper.LidarMeasures.measures.clear();
  mapper.last_timestamp_imu = 1.3;
  PointCloudXYZI::Ptr next_lidar(new PointCloudXYZI());
  PointType next_first_point;
  next_first_point.curvature = 0.0F;
  next_lidar->points.push_back(next_first_point);
  PointType next_last_point;
  next_last_point.curvature = 100.0F;
  next_lidar->points.push_back(next_last_point);
  mapper.lid_raw_data_buffer.push_back(next_lidar);
  mapper.lid_header_time_buffer.push_back(1.1);
  auto next_imu = std::make_shared<sensor_msgs::msg::Imu>();
  next_imu->header.stamp.sec = 1;
  next_imu->header.stamp.nanosec = 300000000U;
  mapper.imu_buffer.push_back(next_imu);

  ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
  ASSERT_EQ(mapper.LidarMeasures.lio_vio_flg, LIO);
  mapper.LidarMeasures.last_lio_update_time = 1.2;
  populatePlanarLidarFeatures(mapper);

  testing::internal::CaptureStdout();
  mapper.handleLIO();
  const std::string second_output =
    testing::internal::GetCapturedStdout();

  const livo::AdaptiveLioModePublicationAudit second =
    mapper.adaptiveLioModeAudit();
  ASSERT_EQ(
    second.status,
    livo::AdaptiveLioModeRecordStatus::kRecorded);
  ASSERT_TRUE(second.receipt.has_value());
  EXPECT_EQ(second.receipt->generation(), 1U);
  EXPECT_DOUBLE_EQ(second.receipt->sourceStampSec(), 1.1);
  EXPECT_EQ(
    second.receipt->mode(),
    livo::AdaptiveLioMode::kLioOnly);
  EXPECT_EQ(
    second.receipt->reason(),
    livo::AdaptiveLioModeReason::
    kAllCamerasMissingAfterDeadline);
  EXPECT_EQ(
    second.receipt->transition(),
    livo::AdaptiveLioModeTransition::kEnterLioOnly);
  EXPECT_TRUE(second.metric_emitted);
  EXPECT_EQ(
    countOccurrences(
      second_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_EQ(
    countOccurrences(
      second_output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);

  mapper.LidarMeasures.measures.clear();
  mapper.last_timestamp_imu = 1.4;
  PointCloudXYZI::Ptr third_lidar(new PointCloudXYZI());
  PointType third_first_point;
  third_first_point.curvature = 0.0F;
  third_lidar->points.push_back(third_first_point);
  PointType third_last_point;
  third_last_point.curvature = 100.0F;
  third_lidar->points.push_back(third_last_point);
  mapper.lid_raw_data_buffer.push_back(third_lidar);
  mapper.lid_header_time_buffer.push_back(1.2);
  auto third_imu =
    std::make_shared<sensor_msgs::msg::Imu>();
  third_imu->header.stamp.sec = 1;
  third_imu->header.stamp.nanosec = 400000000U;
  mapper.imu_buffer.push_back(third_imu);

  ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
  ASSERT_EQ(mapper.LidarMeasures.lio_vio_flg, LIO);
  mapper.LidarMeasures.last_lio_update_time = 1.3;
  populatePlanarLidarFeatures(mapper);

  testing::internal::CaptureStdout();
  mapper.handleLIO();
  const std::string third_output =
    testing::internal::GetCapturedStdout();

  const livo::AdaptiveLioModePublicationAudit third =
    mapper.adaptiveLioModeAudit();
  ASSERT_EQ(
    third.status,
    livo::AdaptiveLioModeRecordStatus::kRecorded);
  ASSERT_TRUE(third.receipt.has_value());
  EXPECT_EQ(third.receipt->generation(), 2U);
  EXPECT_DOUBLE_EQ(third.receipt->sourceStampSec(), 1.2);
  EXPECT_EQ(
    third.receipt->mode(),
    livo::AdaptiveLioMode::kLioOnly);
  EXPECT_EQ(
    third.receipt->reason(),
    livo::AdaptiveLioModeReason::
    kAllCamerasMissingAfterDeadline);
  EXPECT_EQ(
    third.receipt->transition(),
    livo::AdaptiveLioModeTransition::kNone);
  EXPECT_TRUE(third.metric_emitted);
  EXPECT_EQ(
    countOccurrences(
      third_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_EQ(
    countOccurrences(
      third_output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);
  testing::internal::CaptureStdout();
  mapper_owner.reset();
  shutdown_output = testing::internal::GetCapturedStdout();
  EXPECT_EQ(
    countOccurrences(
      shutdown_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_NE(
    shutdown_output.find(
      "generation=3 source_stamp_sec=1.300000000 "
      "mode=lio_only transition=none "
      "reason=all_cameras_missing_after_deadline"),
    std::string::npos);
  EXPECT_EQ(
    countOccurrences(
      shutdown_output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  AdaptiveNormalLioWithoutVioClosesAtShutdown)
{
  rclcpp::Node::SharedPtr unused_node;
  auto mapper_owner = std::make_unique<LIVMapper>(
    unused_node,
    "adaptive_normal_lio_closeout_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  LIVMapper & mapper = *mapper_owner;
  initializeRuntimePublishers(mapper);
  mapper.LidarMeasures.last_lio_update_time = 2.1;
  mapper._state.cov.setIdentity();
  mapper.state_propagat = mapper._state;
  mapper.voxelmap_manager->state_ = mapper._state;
  populatePlanarLidarFeatures(mapper);

  testing::internal::CaptureStdout();
  mapper.handleLIO();
  const std::string lio_output =
    testing::internal::GetCapturedStdout();

  EXPECT_EQ(mapper.lio_feature_generation_, 1U);
  EXPECT_EQ(
    countOccurrences(
      lio_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    0U);

  testing::internal::CaptureStdout();
  mapper_owner.reset();
  const std::string shutdown_output =
    testing::internal::GetCapturedStdout();
  EXPECT_EQ(
    countOccurrences(
      shutdown_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_NE(
    shutdown_output.find(
      "generation=1 source_stamp_sec=2.100000000 "
      "mode=lio_only transition=enter_lio_only "
      "reason=visual_outcome_not_observed_before_closeout"),
    std::string::npos);
  EXPECT_EQ(
    countOccurrences(
      shutdown_output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);
}

#if !LIVO_BUILD_EXPERIMENTAL_SHADOWS
TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  ExperimentalShadowParameterFailsFastWhenBuildGateIsOff)
{
  rclcpp::Node::SharedPtr unused_node;
  EXPECT_THROW(
    LIVMapper(
      unused_node, "cross_camera_shadow_build_gate_test",
      outdoorOptions({
        "vio.camera_mode:=four",
        "vio.multiview.cross_camera_migration_shadow_enable:=true",
      })),
    std::runtime_error);
}
#else
TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  ReferenceOnlyMigrationRuntimeParameterFailsClosedWithoutDependencies)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "reference_only_migration_runtime_missing_deps_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.multiview.reference_only_migration_shadow.enable:=true",
    }));

  const auto & audit =
    mapper.vio_manager->
    referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMissingCrossCameraShadowOwner);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  ReferenceOnlyMigrationRuntimeParameterEnablesMetricsOnlyWithAllDependencies)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "reference_only_migration_runtime_ready_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.multiview.cross_camera_migration_shadow_enable:=true",
      "vio.multiview.time_transfer.shadow_enable:=true",
      "vio.multiview.shared_landmarks.shadow_enable:=true",
      "vio.multiview.reference_only_migration_shadow.enable:=true",
    }));

  const auto & audit =
    mapper.vio_manager->
    referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_TRUE(audit.effective);
  EXPECT_EQ(
    audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
      kMetricsOnlyReady);
  EXPECT_FALSE(audit.visibility_verified);
  EXPECT_TRUE(audit.experimental);
  EXPECT_FALSE(audit.consumable);
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  MissingAllProjectionAndExtrinsicCalibrationFailsClosed)
{
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_missing_all_calibration_test",
    options);

  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_projection_calibration");
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  RepeatedInitializationCannotChangeMissingCalibrationState)
{
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_missing_calibration_reentry_test",
    options);
  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_projection_calibration");
  const livo::CrossCameraMigrationShadowInitAudit initial_audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();

  EXPECT_THROW(
    mapper.initializeComponents(mapper.node),
    std::logic_error);

  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_projection_calibration");
  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  EXPECT_EQ(
    audit.owner_generation, initial_audit.owner_generation);
  EXPECT_EQ(audit.reason, initial_audit.reason);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  ExplicitProjectionWithoutAnyExtrinsicCalibrationFailsClosed)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  for (const char * camera_name : kNames)
  {
    appendPinholeProjectionCalibration(options, camera_name);
  }
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_missing_all_extrinsics_test",
    options);

  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_extrinsic_calibration");
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  MissingOneExtrinsicCalibrationFailsClosed)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  for (const char * camera_name : kNames)
  {
    appendPinholeProjectionCalibration(options, camera_name);
  }
  for (std::size_t camera_id = 0U;
       camera_id < kNames.size();
       ++camera_id)
  {
    appendIdentityVioExtrinsic(
      options, kNames[camera_id],
      camera_id + 1U < kNames.size());
  }
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_missing_right_extrinsic_test",
    options);

  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_extrinsic_calibration");
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  MissingOneExtrinsicRotationCalibrationFailsClosed)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  for (std::size_t camera_id = 0U;
       camera_id < kNames.size();
       ++camera_id)
  {
    appendPinholeProjectionCalibration(
      options, kNames[camera_id]);
    appendIdentityVioExtrinsic(
      options, kNames[camera_id], true,
      camera_id + 1U < kNames.size());
  }
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_missing_right_rotation_test",
    options);

  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_extrinsic_calibration");
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  ExplicitExtrinsicsWithoutAnyProjectionCalibrationFailsClosed)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  for (const char * camera_name : kNames)
  {
    appendIdentityVioExtrinsic(options, camera_name);
  }
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_missing_all_projections_test",
    options);

  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_projection_calibration");
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  MissingOneProjectionCalibrationFailsClosed)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  for (const char * camera_name : kNames)
  {
    appendIdentityVioExtrinsic(options, camera_name);
  }
  for (std::size_t camera_id = 0U;
       camera_id < kNames.size();
       ++camera_id)
  {
    appendPinholeProjectionCalibration(
      options, kNames[camera_id],
      camera_id + 1U < kNames.size());
  }
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_missing_right_projection_test",
    options);

  expectShadowNotReadyAndVioCamerasAlive(
    mapper, "missing_camera_projection_calibration");
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  ExplicitIdentityCalibrationIsValidWithoutD4)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  for (const char * camera_name : kNames)
  {
    appendPinholeProjectionCalibration(options, camera_name);
    appendIdentityVioExtrinsic(options, camera_name);
  }
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_explicit_identity_calibration_test",
    options);

  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  ASSERT_TRUE(audit.enabled);
  ASSERT_TRUE(audit.ready);
  EXPECT_EQ(audit.reason, "shadow_owner_ready");
  EXPECT_NE(audit.rig_revision, 0U);
  EXPECT_TRUE(audit.rig_fingerprint.has_value());
  for (std::size_t camera_id = 0U;
       camera_id < kNames.size();
       ++camera_id)
  {
    ASSERT_TRUE(audit.cameras[camera_id].has_value());
    ASSERT_NE(mapper.vio_manager->cameraModel(camera_id), nullptr);
    EXPECT_EQ(
      mapper.vio_manager->cameraName(camera_id), kNames[camera_id]);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  InitialOverridesRemainEvidenceWhenAutomaticDeclarationIsDisabled)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::NodeOptions options = minimalFourCameraOptions();
  for (const char * camera_name : kNames)
  {
    appendPinholeProjectionCalibration(options, camera_name);
    appendIdentityVioExtrinsic(options, camera_name);
  }
  options.automatically_declare_parameters_from_overrides(false);
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_initial_override_map_test",
    options);

  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  ASSERT_TRUE(audit.enabled);
  ASSERT_TRUE(audit.ready);
  EXPECT_EQ(audit.reason, "shadow_owner_ready");
  EXPECT_NE(audit.rig_revision, 0U);
  EXPECT_TRUE(audit.rig_fingerprint.has_value());
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  OutdoorParametersCreateIdMappedFourCameraShadowOwner)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  constexpr std::array<std::uint64_t, 4U>
    kModelRevisions{{
      1172720795579671296ULL,
      14974416105296971232ULL,
      10800054445785174959ULL,
      11386666025749495921ULL,
    }};
  const std::array<std::array<double, 9U>, 4U> expected_rotations{{
    {{
      0.0031916438524545324,
      0.0050081472031502138,
      -0.99998236578007249,
      0.99999348814234101,
      -0.0017003358397726313,
      0.0031831636693519043,
      -0.0016843641034486483,
      -0.99998601356200167,
      -0.0050135414572385585,
    }},
    {{
      -0.99982795897656507,
      -0.014464221993487875,
      0.011612008046814171,
      -0.011402097122951621,
      -0.014511318976066556,
      -0.9998296923990474,
      0.014630264179260499,
      -0.99979008191904872,
      0.014343899969920915,
    }},
    {{
      -0.0016205753677488634,
      0.0044431698369502559,
      0.99998881592609712,
      -0.99977697666663712,
      0.021048989597706597,
      -0.0017137573440959807,
      -0.021056368699190493,
      -0.99976857235998173,
      0.0044080674323445646,
    }},
    {{
      0.99969405042166892,
      -0.019778473062576601,
      -0.014853200154535728,
      0.015445004958050595,
      0.030112931589163727,
      0.99942716751794991,
      -0.019319869910627285,
      -0.99935080094750484,
      0.030409197164955909,
    }},
  }};
  const std::array<Eigen::Vector3d, 4U> expected_positions{{
    Eigen::Vector3d(
      -0.16056981004032952,
      -0.023585423545053538,
      -0.27252740986020108),
    Eigen::Vector3d(
      0.023585423545053535,
      -0.16056981004032955,
      -0.27252740986020108),
    Eigen::Vector3d(
      0.16056981004032955,
      0.023585423545053563,
      -0.27252740986020113),
    Eigen::Vector3d(
      -0.023585423545053545,
      0.16056981004032955,
      -0.27252740986020108),
  }};

  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_four_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.multiview.cross_camera_migration_shadow_enable:=true",
    }));

  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  ASSERT_TRUE(audit.enabled);
  ASSERT_TRUE(audit.ready);
  EXPECT_EQ(audit.reason, "shadow_owner_ready");
  EXPECT_EQ(audit.rig_revision, 8206638252039177439ULL);
  ASSERT_TRUE(audit.rig_fingerprint.has_value());
  EXPECT_EQ(
    fingerprintHex(*audit.rig_fingerprint),
    "71e3d60274ed84df4237393f2d79db6f"
    "1d48ceef9d0f0bbfb97fb0fa84fcd048");
  for (std::size_t camera_id = 0U;
       camera_id < kNames.size();
       ++camera_id)
  {
    ASSERT_TRUE(audit.cameras[camera_id].has_value());
    const auto & camera = *audit.cameras[camera_id];
    EXPECT_EQ(camera.camera_id, camera_id);
    EXPECT_EQ(camera.camera_name, kNames[camera_id]);
    EXPECT_EQ(camera.model_revision, kModelRevisions[camera_id]);
    EXPECT_TRUE(camera.rotation_imu_from_camera.isApprox(
      rowMajorMatrix(expected_rotations[camera_id]), 2.0e-15));
    EXPECT_TRUE(camera.position_imu_from_camera.isApprox(
      expected_positions[camera_id], 2.0e-15));
    ASSERT_NE(mapper.vio_manager->cameraModel(camera_id), nullptr);
    EXPECT_EQ(
      mapper.vio_manager->cameraName(camera_id), kNames[camera_id]);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  RepeatedInitializationCannotChangeExplicitOutdoorGolden)
{
  constexpr std::array<std::uint64_t, 4U>
    kModelRevisions{{
      1172720795579671296ULL,
      14974416105296971232ULL,
      10800054445785174959ULL,
      11386666025749495921ULL,
    }};
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_outdoor_reentry_test",
    outdoorOptions({
      "common.img_en:=0",
      "vio.camera_mode:=four",
      "vio.multiview.cross_camera_migration_shadow_enable:=true",
    }));
  const livo::CrossCameraMigrationShadowInitAudit initial_audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  ASSERT_TRUE(initial_audit.ready);

  EXPECT_THROW(
    mapper.initializeComponents(mapper.node),
    std::logic_error);

  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  ASSERT_TRUE(audit.enabled);
  ASSERT_TRUE(audit.ready);
  EXPECT_EQ(audit.reason, "shadow_owner_ready");
  EXPECT_EQ(audit.rig_revision, 8206638252039177439ULL);
  ASSERT_TRUE(audit.rig_fingerprint.has_value());
  EXPECT_EQ(
    fingerprintHex(*audit.rig_fingerprint),
    "71e3d60274ed84df4237393f2d79db6f"
    "1d48ceef9d0f0bbfb97fb0fa84fcd048");
  EXPECT_EQ(audit.rig_revision, initial_audit.rig_revision);
  EXPECT_EQ(
    audit.owner_generation, initial_audit.owner_generation);
  EXPECT_EQ(audit.rig_fingerprint, initial_audit.rig_fingerprint);
  for (std::size_t camera_id = 0U;
       camera_id < kModelRevisions.size();
       ++camera_id)
  {
    ASSERT_TRUE(audit.cameras[camera_id].has_value());
    ASSERT_TRUE(initial_audit.cameras[camera_id].has_value());
    EXPECT_EQ(
      audit.cameras[camera_id]->model_revision,
      kModelRevisions[camera_id]);
    EXPECT_EQ(
      audit.cameras[camera_id]->model_revision,
      initial_audit.cameras[camera_id]->model_revision);
    EXPECT_TRUE(
      audit.cameras[camera_id]->rotation_imu_from_camera.isApprox(
        initial_audit.cameras[camera_id]->rotation_imu_from_camera,
        1.0e-15));
    EXPECT_TRUE(
      audit.cameras[camera_id]->position_imu_from_camera.isApprox(
        initial_audit.cameras[camera_id]->position_imu_from_camera,
        1.0e-15));
    EXPECT_NE(mapper.vio_manager->cameraModel(camera_id), nullptr);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  EnabledShadowInFrontModeFailsClosedWithoutInterruptingVio)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_mono_test",
    outdoorOptions({
      "vio.multiview.cross_camera_migration_shadow_enable:=true",
    }));

  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(audit.reason, "requires_four_camera_mode");
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  ASSERT_NE(mapper.vio_manager->cameraModel(0U), nullptr);
  EXPECT_EQ(mapper.vio_manager->cameraName(0U), "front");
  EXPECT_EQ(mapper.vio_manager->cameraModel(1U), nullptr);
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST_F(
  CrossCameraMigrationShadowRuntimeTest,
  UnsupportedScaleFailsClosedAfterAllFourVioCamerasRegister)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node,
    "cross_camera_shadow_invalid_scale_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.multiview.cross_camera_migration_shadow_enable:=true",
      "multi_camera.cameras.front.scale:=0.5",
      "multi_camera.cameras.left.scale:=0.5",
      "multi_camera.cameras.back.scale:=0.5",
      "multi_camera.cameras.right.scale:=0.5",
    }));

  const auto & audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(audit.reason, "unsupported_camera_scale");
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  for (std::size_t camera_id = 0U;
       camera_id < kNames.size();
       ++camera_id)
  {
    EXPECT_FALSE(audit.cameras[camera_id].has_value());
    ASSERT_NE(mapper.vio_manager->cameraModel(camera_id), nullptr);
    EXPECT_EQ(
      mapper.vio_manager->cameraName(camera_id), kNames[camera_id]);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}
#endif
