#include "LIVMapper.h"
#include "checked_increment.h"
#include "structured_metric.h"

#include <algorithm>
#include <chrono>
#include <initializer_list>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstring>

#include <gtest/gtest.h>

namespace
{

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
  for (const std::string & parameter_override :
       parameter_overrides)
  {
    arguments.emplace_back("-p");
    arguments.push_back(parameter_override);
  }
  options.arguments(arguments);
  return options;
}

rclcpp::NodeOptions outdoorIdentityVioExtrinsicOptions(
  const std::initializer_list<std::string> parameter_overrides)
{
  rclcpp::NodeOptions options =
    outdoorOptions(parameter_overrides);
  constexpr std::array<const char *, 4U> kCameraNames{{
    "front", "left", "back", "right"}};
  for (const char * camera_name : kCameraNames)
  {
    const std::string prefix =
      "multi_camera.vio_extrinsics." +
      std::string(camera_name) + ".";
    options.append_parameter_override(
      prefix + "R",
      std::vector<double>{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0});
    options.append_parameter_override(
      prefix + "T",
      std::vector<double>{0.0, 0.0, 0.0});
  }
  return options;
}

class LIVMapperAdaptiveVisualRuntimeTest :
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

bool LIVMapperAdaptiveVisualRuntimeTest::
  owns_rclcpp_context_{false};

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, LocalFastPreservesExactNativeImuStamp)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper mapper(unused, "local_imu_stamp_test", outdoorOptions({
    "local_fast.enable:=true", "local_map.half_map_size:=20"}));
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp.sec = 1750000000;
  imu->header.stamp.nanosec = 123456789;
  imu->linear_acceleration.z = 9.81;
  mapper.imu_cbk(imu);
  ASSERT_EQ(mapper.imu_buffer.size(), 1U);
  EXPECT_EQ(mapper.imu_buffer.back()->header.stamp.sec, imu->header.stamp.sec);
  EXPECT_EQ(mapper.imu_buffer.back()->header.stamp.nanosec, imu->header.stamp.nanosec);
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, LocalFastRejectsAlignedImuStampConfiguration)
{
  rclcpp::Node::SharedPtr unused;
  EXPECT_THROW(LIVMapper(unused, "local_imu_offset_test", outdoorOptions({
    "local_fast.enable:=true", "local_map.half_map_size:=20",
    "time_offset.imu_time_offset:=0.01"})), std::runtime_error);
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, LocalFastBoundsCloudQueueWhileWaitingForImu)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper mapper(unused, "local_queue_test", outdoorOptions({
    "local_fast.enable:=true", "local_map.half_map_size:=20"}));
  for (int scan = 0; scan < 5; ++scan) {
    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.stamp.sec = 100 + scan;
    cloud->header.stamp.nanosec = 123456789;
    cloud->width = 2; cloud->height = 1; cloud->point_step = 32; cloud->row_step = 64;
    const char * names[] = {"x", "y", "z", "intensity", "timestamp", "ring"};
    const uint32_t offsets[] = {0,4,8,12,16,24};
    for (int i=0;i<6;++i) {
      sensor_msgs::msg::PointField field;
      field.name=names[i]; field.offset=offsets[i]; field.count=1;
      field.datatype=i==4 ? field.FLOAT64 : i==5 ? field.UINT16 : field.FLOAT32;
      cloud->fields.push_back(field);
    }
    cloud->data.resize(64);
    for (int i=0;i<2;++i) {
      const float x=1.0F; const double t=100.0 + i*0.1;
      std::memcpy(cloud->data.data()+i*32,&x,4);
      std::memcpy(cloud->data.data()+i*32+16,&t,8);
    }
    mapper.standard_pcl_cbk(cloud);
    if (scan==0) mapper.lidar_pushed=true;
  }
  EXPECT_EQ(mapper.lid_raw_data_buffer.size(),2U);
  EXPECT_EQ(mapper.local_scan_stamps_.size(),2U);
  EXPECT_EQ(mapper.local_scan_ends_.size(),2U);
  EXPECT_EQ(mapper.local_scan_drop_count_.load(),3U);
  EXPECT_EQ(mapper.local_scan_stamps_.front(),100123456789LL);
}

void initializeRuntimePublishers(LIVMapper & mapper)
{
  rclcpp::Node::SharedPtr node = mapper.node;
  image_transport::ImageTransport image_transport(node);
  mapper.initializeSubscribersAndPublishers(
    node, image_transport);
}

sensor_msgs::msg::Imu::SharedPtr historyImu(const std::size_t index)
{
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp.sec = 100 + static_cast<int>(index / 200U);
  imu->header.stamp.nanosec = static_cast<std::uint32_t>(index % 200U) * 5000000U;
  imu->linear_acceleration.z = 9.81;
  imu->angular_velocity.x = static_cast<double>(index);
  return imu;
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, ImuHistoryRejectsInvalidParametersEarly)
{
  for (const std::string value : {"0", "-1", "65537", "1.5", "true", "'8192'"})
  {
    rclcpp::Node::SharedPtr unused;
    EXPECT_ANY_THROW(LIVMapper(unused, "imu_history_invalid_capacity",
      outdoorOptions({"buffer.imu_algorithm_capacity:=" + value})));
  }
  rclcpp::Node::SharedPtr unused;
  EXPECT_ANY_THROW(LIVMapper(unused, "imu_history_invalid_fatal",
    outdoorOptions({"buffer.imu_overflow_fatal:=1"})));
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, ImuHistoryDefault4096DropsOldest)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper mapper(unused, "imu_history_default", outdoorOptions({}));
  EXPECT_EQ(mapper.node->get_parameter("buffer.imu_algorithm_capacity").as_int(), 4096);
  EXPECT_FALSE(mapper.node->get_parameter("buffer.imu_overflow_fatal").as_bool());
  mapper.imu_cbk(historyImu(0));
  const auto first = mapper.imu_buffer.front();
  for (std::size_t index = 1; index <= 4096U; ++index)
  {
    mapper.imu_cbk(historyImu(index));
  }
  ASSERT_EQ(mapper.imu_buffer.size(), 4096U);
  EXPECT_EQ(std::find(mapper.imu_buffer.begin(), mapper.imu_buffer.end(), first),
    mapper.imu_buffer.end());
  EXPECT_EQ(mapper.imu_buffer.front()->angular_velocity.x, 1.0);
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, ImuHistory8192RetainsFullHistoryAndFailsBeforeDeletion)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper mapper(unused, "imu_history_extended", outdoorOptions({
    "buffer.imu_algorithm_capacity:=8192", "buffer.imu_overflow_fatal:=true"}));
  for (std::size_t index = 0; index < 8192U; ++index)
  {
    ASSERT_NO_THROW(mapper.imu_cbk(historyImu(index)));
  }
  ASSERT_EQ(mapper.imu_buffer.size(), 8192U);
  EXPECT_EQ(mapper.imu_buffer.front()->angular_velocity.x, 0.0);
  const auto before = mapper.imu_buffer;
  const auto latest = mapper.last_timestamp_imu;
  EXPECT_THROW(mapper.imu_cbk(historyImu(8192)), std::runtime_error);
  EXPECT_EQ(mapper.imu_buffer, before);
  EXPECT_EQ(mapper.last_timestamp_imu, latest);
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, ImuHistoryNormalAdmissionKeepsOrderingAndValues)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper original(unused, "imu_history_original", outdoorOptions({}));
  LIVMapper guarded(unused, "imu_history_guarded", outdoorOptions({
    "buffer.imu_algorithm_capacity:=8192", "buffer.imu_overflow_fatal:=true"}));
  for (const std::size_t index : {0U, 3U, 1U, 2U, 2U, 4U})
  {
    const auto sample = historyImu(index);
    original.imu_cbk(sample);
    guarded.imu_cbk(sample);
  }
  ASSERT_EQ(original.imu_buffer.size(), guarded.imu_buffer.size());
  for (std::size_t index = 0; index < original.imu_buffer.size(); ++index)
  {
    EXPECT_EQ(*original.imu_buffer[index], *guarded.imu_buffer[index]);
  }
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, ImuHistoryAlgorithmFailureStopsRunAndJoinsIngress)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper mapper(unused, "imu_history_run_failure", outdoorOptions({
    "buffer.imu_algorithm_capacity:=2", "buffer.imu_overflow_fatal:=true"}));
  initializeRuntimePublishers(mapper);
  mapper.imu_cbk(historyImu(0));
  mapper.imu_cbk(historyImu(1));
  const auto before = mapper.imu_buffer;
  const auto position = mapper._state.pos_end;
  mapper.imu_transport_cbk(historyImu(2));
  auto node = mapper.node;
  const auto started = std::chrono::steady_clock::now();
  EXPECT_THROW(mapper.run(node), std::runtime_error);
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(2));
  EXPECT_EQ(mapper.imu_buffer, before);
  EXPECT_EQ(mapper._state.pos_end, position);
  ASSERT_TRUE(mapper.mtx_buffer.try_lock());
  mapper.mtx_buffer.unlock();
  ASSERT_TRUE(mapper.imu_transport_mutex_.try_lock());
  mapper.imu_transport_mutex_.unlock();
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, ImuHistoryTransportFailureRetainsQueueAndStopsRun)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper mapper(unused, "imu_history_transport_failure", outdoorOptions({
    "buffer.imu_algorithm_capacity:=8192", "buffer.imu_overflow_fatal:=true"}));
  initializeRuntimePublishers(mapper);
  for (std::size_t index = 0; index < 4096U; ++index)
  {
    mapper.imu_transport_cbk(historyImu(index));
  }
  EXPECT_FALSE(mapper.imu_transport_overflow_failed_.load());
  const auto before = mapper.imu_transport_buffer_;
  ASSERT_NO_THROW(mapper.imu_transport_cbk(historyImu(4096)));
  ASSERT_TRUE(mapper.imu_transport_overflow_failed_.load());
  mapper.imu_transport_cbk(historyImu(4097));
  EXPECT_EQ(mapper.imu_transport_buffer_, before);
  auto node = mapper.node;
  EXPECT_THROW(mapper.run(node), std::runtime_error);
  EXPECT_TRUE(mapper.imu_buffer.empty());
  EXPECT_EQ(mapper.imu_transport_buffer_, before);
}

TEST_F(LIVMapperAdaptiveVisualRuntimeTest, ImuHistoryDefaultTransportStillDropsOldest)
{
  rclcpp::Node::SharedPtr unused;
  LIVMapper mapper(unused, "imu_history_transport_default", outdoorOptions({}));
  for (std::size_t index = 0; index <= 4096U; ++index)
  {
    mapper.imu_transport_cbk(historyImu(index));
  }
  EXPECT_FALSE(mapper.imu_transport_overflow_failed_.load());
  ASSERT_EQ(mapper.imu_transport_buffer_.size(), 4096U);
  EXPECT_EQ(mapper.imu_transport_buffer_.front()->angular_velocity.x, 1.0);
}

bool hasPointCloudField(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const std::string & field_name)
{
  return std::any_of(
    cloud.fields.begin(), cloud.fields.end(),
    [&field_name](const sensor_msgs::msg::PointField & field)
    {
      return field.name == field_name;
    });
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

void appendVioMeasurement(
  LIVMapper & mapper,
  CameraObservationBatch batch = CameraObservationBatch{})
{
  MeasureGroup measurement;
  measurement.vio_time = 20.0;
  measurement.camera_batch = std::move(batch);
  mapper.LidarMeasures.measures.push_back(
    std::move(measurement));
  mapper.LidarMeasures.last_lio_update_time = 20.0;
  mapper.LidarMeasures.lio_vio_flg = VIO;
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

std::vector<pointWithVar> cameraVisibleLidarPoints(
  VIOManager & manager, const std::uint8_t camera_id)
{
  EXPECT_TRUE(manager.activateCamera(camera_id));
  std::vector<pointWithVar> points;
  points.reserve(35U);
  for (int row = 0; row < 5; ++row)
  {
    for (int column = 0; column < 7; ++column)
    {
      const Eigen::Vector3d point_camera(
        0.45 * static_cast<double>(column) - 1.35,
        0.45 * static_cast<double>(row) - 0.90,
        5.0);
      pointWithVar point;
      point.point_w =
        manager.Rci.transpose() *
        (point_camera - manager.Pci);
      point.var = 1.0e-4 * Eigen::Matrix3d::Identity();
      point.normal =
        manager.Rci.transpose() * Eigen::Vector3d::UnitZ();
      points.push_back(std::move(point));
    }
  }
  return points;
}

void seedCameraMap(
  LIVMapper & mapper, const std::uint8_t camera_id,
  const cv::Mat & image)
{
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = 10.0;
  observation.image = image.clone();
  observation.photometric_status =
    livo::PhotometricStatus::kValid;
  std::vector<pointWithVar> points =
    cameraVisibleLidarPoints(*mapper.vio_manager, camera_id);
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;
  mapper.vio_manager->processFrame(
    observation, points, feature_map,
    observation.timestamp);
}

CameraObservationBatch sealFourCameraBatch(
  LIVMapper & mapper, const double timestamp)
{
  for (std::uint8_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount;
       ++camera_id)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      timestamp + 0.002 * static_cast<double>(camera_id);
    observation.image =
      camera_id == 3U ?
      texturedImage() :
      cv::Mat(480, 640, CV_8UC1, cv::Scalar(96U));
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      mapper.camera_observation_queue_.push(
        std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    mapper.camera_observation_queue_.
    popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  return batch;
}

CameraObservationBatch sealTwoCameraBatch(
  LIVMapper & mapper, const double timestamp)
{
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 2U>{{0U, 2U}})
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      timestamp + 0.008 * static_cast<double>(camera_id / 2U);
    observation.image =
      camera_id == 2U ?
      texturedImage() :
      cv::Mat(480, 640, CV_8UC1, cv::Scalar(96U));
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      mapper.camera_observation_queue_.push(
        std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    mapper.camera_observation_queue_.
    popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  return batch;
}

CameraObservationBatch sealUnhealthyFourCameraBatch(
  LIVMapper & mapper, const double timestamp)
{
  for (std::uint8_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount;
       ++camera_id)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      timestamp + 0.002 * static_cast<double>(camera_id);
    observation.image =
      cv::Mat(480, 640, CV_8UC1, cv::Scalar(255U));
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      mapper.camera_observation_queue_.push(
        std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    mapper.camera_observation_queue_.
    popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_EQ(batch.camera_count, 4U);
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  return batch;
}

CameraObservationBatch sealIsolationBatch(
  LIVMapper & mapper, const double timestamp,
  const std::uint8_t target_camera,
  const std::uint8_t survivor_camera)
{
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 2U>{{
         target_camera, survivor_camera}})
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      timestamp + 0.002 * static_cast<double>(camera_id);
    observation.image =
      camera_id == target_camera ?
      cv::Mat(480, 640, CV_8UC1, cv::Scalar(255U)) :
      texturedImage();
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      mapper.camera_observation_queue_.push(
        std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    mapper.camera_observation_queue_.
    popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_EQ(batch.camera_count, 2U);
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  return batch;
}

CameraObservationBatch sealHealthySingleCameraBatch(
  LIVMapper & mapper, const double timestamp,
  const std::uint8_t camera_id)
{
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = timestamp;
  observation.image = texturedImage();
  observation.photometric_status =
    livo::PhotometricStatus::kValid;
  EXPECT_EQ(
    mapper.camera_observation_queue_.push(
      std::move(observation)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_EQ(
    mapper.camera_observation_queue_.
    popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_EQ(batch.camera_count, 1U);
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  return batch;
}

CameraObservationBatch makeSingleCameraBatch(
  const double timestamp, const std::uint8_t camera_id,
  const cv::Mat & image)
{
  CameraObservationBatch batch;
  batch.present.fill(false);
  batch.present[camera_id] = true;
  batch.camera_count = 1U;
  batch.partial = true;
  batch.timestamp = timestamp;
  batch.observations[camera_id].camera_id = camera_id;
  batch.observations[camera_id].timestamp = timestamp;
  batch.observations[camera_id].image = image.clone();
  batch.observations[camera_id].photometric_status =
    livo::PhotometricStatus::kValid;
  return batch;
}

void enqueueCameraObservation(
  LIVMapper & mapper, const double timestamp,
  const std::uint8_t camera_id, cv::Mat image)
{
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = timestamp;
  observation.image = std::move(image);
  observation.photometric_status =
    livo::PhotometricStatus::kValid;
  EXPECT_EQ(
    mapper.camera_observation_queue_.push(
      std::move(observation)),
    CameraObservationPushResult::ACCEPTED);
}

CameraObservationBatch sealMalformedSingleCameraBatch(
  LIVMapper & mapper, const double timestamp,
  const std::uint8_t camera_id = 0U)
{
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = timestamp;
  observation.image =
    cv::Mat(480, 640, CV_32FC1, cv::Scalar(1.0F));
  observation.photometric_status =
    livo::PhotometricStatus::kValid;
  EXPECT_EQ(
    mapper.camera_observation_queue_.push(
      std::move(observation)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_EQ(
    mapper.camera_observation_queue_.
    popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  EXPECT_TRUE(batch.validatedReceipt().has_value());
  return batch;
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

std::vector<std::string> linesContaining(
  const std::string & text, const std::string & needle)
{
  std::vector<std::string> lines;
  std::size_t line_begin = 0U;
  while (line_begin < text.size())
  {
    const std::size_t line_end = text.find('\n', line_begin);
    const std::size_t line_length =
      line_end == std::string::npos ?
      text.size() - line_begin :
      line_end - line_begin;
    const std::string line =
      text.substr(line_begin, line_length);
    if (line.find(needle) != std::string::npos)
    {
      lines.push_back(line);
    }
    if (line_end == std::string::npos)
    {
      break;
    }
    line_begin = line_end + 1U;
  }
  return lines;
}

std::string metricField(
  const std::string & line, const std::string & key)
{
  const std::string prefix = key + "=";
  const std::size_t value_begin = line.find(prefix);
  if (value_begin == std::string::npos)
  {
    return {};
  }
  const std::size_t first_character =
    value_begin + prefix.size();
  const std::size_t value_end =
    line.find(' ', first_character);
  return line.substr(
    first_character,
    value_end == std::string::npos ?
    std::string::npos :
    value_end - first_character);
}

}  // namespace

TEST(
  CheckedIncrementFoundationTest,
  MaximumGenerationDoesNotWrap)
{
  std::uint64_t generation =
    std::numeric_limits<std::uint64_t>::max();

  EXPECT_FALSE(livo::checkedIncrement(generation));
  EXPECT_EQ(
    generation,
    std::numeric_limits<std::uint64_t>::max());
}

TEST(
  StructuredMetricFoundationTest,
  DisabledMetricsDoNotWriteToStdout)
{
  livo::setStructuredMetricEnabled(false);
  testing::internal::CaptureStdout();
  EXPECT_FALSE(livo::emitStructuredMetric("metric_should_be_silent\n"));
  const std::string output =
    testing::internal::GetCapturedStdout();
  livo::setStructuredMetricEnabled(true);

  EXPECT_TRUE(output.empty());
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  InitializationSharesOneQueueAuthorityForAdaptiveAndR2b)
{
  struct Case
  {
    const char * node_name;
    std::initializer_list<std::string> overrides;
    bool adaptive_single_camera_runtime;
    bool r2b_runtime;
    bool authority_required;
  };
  const std::initializer_list<Case> cases{
    {
      "adaptive_authority_neither_test",
      {},
      false, false, false,
    },
    {
      "adaptive_authority_clamps_max_test",
      {
        "vio.camera_mode:=four",
        "vio.adaptive.enable:=true",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=2",
        "vio.min_joint_cameras:=1",
      },
      true, false, true,
    },
    {
      "adaptive_authority_max1_test",
      {
        "vio.camera_mode:=four",
        "vio.adaptive.enable:=true",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=1",
        "vio.min_joint_cameras:=1",
      },
      true, false, true,
    },
#if LIVO_BUILD_EXPERIMENTAL_SHADOWS
    {
      "adaptive_authority_r2b_test",
      {
        "vio.camera_mode:=four",
        "vio.multiview.reference_only_migration_shadow.enable:=true",
      },
      false, true, true,
    },
    {
      "adaptive_authority_both_test",
      {
        "vio.camera_mode:=four",
        "vio.adaptive.enable:=true",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=1",
        "vio.min_joint_cameras:=1",
        "vio.multiview.reference_only_migration_shadow.enable:=true",
      },
      true, true, true,
    },
#endif
  };

  for (const Case & test_case : cases)
  {
    SCOPED_TRACE(test_case.node_name);
    rclcpp::Node::SharedPtr unused_node;
    LIVMapper mapper(
      unused_node, test_case.node_name,
      outdoorOptions(test_case.overrides));

    const auto & audit =
      mapper.adaptiveVisualLidarRuntimeInitializationAudit();
    EXPECT_EQ(
      audit.adaptive_single_camera_runtime,
      test_case.adaptive_single_camera_runtime);
    EXPECT_EQ(audit.r2b_runtime, test_case.r2b_runtime);
    EXPECT_EQ(
      audit.authority_required,
      test_case.authority_required);
    EXPECT_EQ(
      audit.batch_sealing_request_count,
      test_case.authority_required ? 1U : 0U);
    EXPECT_EQ(
      audit.authority_issue_count,
      test_case.authority_required ? 1U : 0U);
    EXPECT_EQ(
      audit.authority_valid,
      test_case.authority_required);
    EXPECT_EQ(
      audit.status,
      test_case.authority_required ?
      livo::AdaptiveVisualLidarRuntimeInitializationStatus::
      kReady :
      livo::AdaptiveVisualLidarRuntimeInitializationStatus::
      kNotRequired);
    EXPECT_EQ(
      mapper.camera_observation_queue_.
      batchAuthoritySealingEnabled(),
      test_case.authority_required);

    const auto second_authority =
      mapper.camera_observation_queue_.
      issueAdaptiveVisualSelectionAuthority();
    EXPECT_EQ(
      static_cast<bool>(second_authority),
      !test_case.authority_required);
  }
}

#if !LIVO_BUILD_EXPERIMENTAL_SHADOWS
TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  ExperimentalShadowParametersFailClosedInProductionBuild)
{
  rclcpp::Node::SharedPtr unused_node;
  EXPECT_THROW(
    LIVMapper(
      unused_node, "experimental_shadow_build_gate_test",
      outdoorOptions({
        "vio.camera_mode:=four",
        "vio.multiview.time_transfer.shadow_enable:=true",
      })),
    std::runtime_error);
}
#endif

#if LIVO_BUILD_EXPERIMENTAL_SHADOWS
TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  NonMaxAdaptiveModeRetainsLegacyFreshGenerationGuard)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_nonmax_legacy_guard_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=2",
      "vio.min_joint_cameras:=1",
      "vio.multiview.time_transfer.shadow_enable:=true",
    }));
  appendVioMeasurement(mapper);
  mapper._pv_list.emplace_back();
  mapper.lio_feature_generation_ = 31U;
  mapper.last_joint_vio_feature_generation_ = 31U;

  mapper.handleVIO();

  const auto & audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kNotApplicable);
  EXPECT_EQ(audit.pre_vio_shadow_entry_count, 1U);
  EXPECT_EQ(audit.post_vio_shadow_entry_count, 0U);
  EXPECT_FALSE(audit.lidar_posterior.has_value());
  EXPECT_FALSE(audit.generation_issued);
  EXPECT_FALSE(audit.runtime_bound);
  EXPECT_FALSE(audit.vio_dispatch_started);
  EXPECT_FALSE(audit.vio_dispatch_completed);
  EXPECT_EQ(mapper.last_joint_vio_feature_generation_, 31U);
  EXPECT_FALSE(
    mapper.camera_observation_queue_.
    batchAuthoritySealingEnabled());
  EXPECT_EQ(
    mapper.vio_manager->adaptiveVisualRuntimeAudit().
    outcome,
    livo::AdaptiveVisualRuntimeOutcome::kNotRun);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  LegacyRuntimeReportsActualPostVioShadowEntries)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "legacy_post_vio_shadow_audit_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.multiview.time_transfer.shadow_enable:=true",
      "vio.multiview.shared_landmarks.shadow_enable:=true",
      "vio.multiview.shared_landmarks.tombstone_shadow_enable:=true",
      "vio.multiview.shared_landmarks.refit_shadow_enable:=true",
    }));
  appendVioMeasurement(mapper);
  mapper._pv_list.emplace_back();
  mapper.lio_feature_generation_ = 32U;
  mapper.last_joint_vio_feature_generation_ = 31U;

  mapper.handleVIO();

  const auto & audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kNotApplicable);
  EXPECT_FALSE(audit.lidar_posterior.has_value());
  EXPECT_FALSE(audit.generation_issued);
  EXPECT_FALSE(audit.runtime_bound);
  EXPECT_FALSE(audit.vio_dispatch_started);
  EXPECT_FALSE(audit.vio_dispatch_completed);
  EXPECT_EQ(audit.pre_vio_shadow_entry_count, 1U);
  EXPECT_EQ(audit.post_vio_shadow_entry_count, 2U);
  EXPECT_EQ(mapper.last_joint_vio_feature_generation_, 32U);
}
#endif

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  ReinitializationIsRejectedBeforeConfigurationStateChanges)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "linearization_config_reentry_test",
    outdoorOptions({"common.img_en:=0"}));
  const livo::CrossCameraMigrationShadowInitAudit
    initial_cross_camera_audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();

  mapper.multi_camera_landmark_linearization_shadow_config_.
    maximum_source_age_sec += 0.125;
  EXPECT_THROW(
    mapper.initializeComponents(mapper.node),
    std::logic_error);

  const auto & cross_camera_audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  EXPECT_EQ(
    cross_camera_audit.owner_generation,
    initial_cross_camera_audit.owner_generation);
  EXPECT_EQ(
    cross_camera_audit.reason,
    initial_cross_camera_audit.reason);
}

#if LIVO_BUILD_EXPERIMENTAL_SHADOWS
TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  FullAdaptiveR2bInitializationIsOneShotAndKeepsBindings)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_r2b_one_shot_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
      "vio.multiview.cross_camera_migration_shadow_enable:=true",
      "vio.multiview.time_transfer.shadow_enable:=true",
      "vio.multiview.shared_landmarks.shadow_enable:=true",
      "vio.multiview.reference_only_migration_shadow.enable:=true",
    }));
  const livo::AdaptiveVisualLidarRuntimeInitializationAudit
    initial_mapper_audit =
    mapper.adaptiveVisualLidarRuntimeInitializationAudit();
  const livo::CrossCameraMigrationShadowInitAudit
    initial_cross_camera_audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  const livo::ReferenceOnlyMigrationShadowRuntimeInitAudit
    initial_r2b_audit =
    mapper.vio_manager->
    referenceOnlyMigrationShadowRuntimeInitAudit();
  ASSERT_EQ(
    initial_mapper_audit.status,
    livo::AdaptiveVisualLidarRuntimeInitializationStatus::
    kReady);
  ASSERT_EQ(initial_mapper_audit.authority_issue_count, 1U);
  ASSERT_TRUE(initial_mapper_audit.authority_valid);
  ASSERT_TRUE(initial_cross_camera_audit.ready);
  ASSERT_TRUE(initial_r2b_audit.effective);
  ASSERT_EQ(
    initial_r2b_audit.reason,
    livo::ReferenceOnlyMigrationShadowRuntimeInitReason::
    kMetricsOnlyReady);

  EXPECT_THROW(
    mapper.initializeComponents(mapper.node),
    std::logic_error);

  const auto & mapper_audit =
    mapper.adaptiveVisualLidarRuntimeInitializationAudit();
  EXPECT_EQ(
    mapper_audit.status, initial_mapper_audit.status);
  EXPECT_EQ(
    mapper_audit.authority_issue_count,
    initial_mapper_audit.authority_issue_count);
  EXPECT_EQ(
    mapper_audit.batch_sealing_request_count,
    initial_mapper_audit.batch_sealing_request_count);
  EXPECT_EQ(
    mapper_audit.authority_valid,
    initial_mapper_audit.authority_valid);

  const auto & cross_camera_audit =
    mapper.vio_manager->crossCameraMigrationShadowInitAudit();
  EXPECT_EQ(
    cross_camera_audit.owner_generation,
    initial_cross_camera_audit.owner_generation);
  EXPECT_EQ(
    cross_camera_audit.rig_revision,
    initial_cross_camera_audit.rig_revision);
  EXPECT_EQ(
    cross_camera_audit.rig_fingerprint,
    initial_cross_camera_audit.rig_fingerprint);

  const auto & r2b_audit =
    mapper.vio_manager->
    referenceOnlyMigrationShadowRuntimeInitAudit();
  EXPECT_EQ(r2b_audit.enabled, initial_r2b_audit.enabled);
  EXPECT_EQ(r2b_audit.effective, initial_r2b_audit.effective);
  EXPECT_EQ(r2b_audit.reason, initial_r2b_audit.reason);
  EXPECT_EQ(
    r2b_audit.bound_cross_camera_owner_generation,
    initial_r2b_audit.bound_cross_camera_owner_generation);
  EXPECT_EQ(
    r2b_audit.bound_rig_revision,
    initial_r2b_audit.bound_rig_revision);
  EXPECT_EQ(
    r2b_audit.bound_geometry_configuration_generation,
    initial_r2b_audit.
    bound_geometry_configuration_generation);
  EXPECT_EQ(
    r2b_audit.bound_time_transfer_revision,
    initial_r2b_audit.bound_time_transfer_revision);
  EXPECT_EQ(
    r2b_audit.bound_authority_generation,
    initial_r2b_audit.bound_authority_generation);
}
#endif

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  VioManagerPlotFlagDefaultsFalse)
{
  VIOManager manager;

  EXPECT_FALSE(manager.plot_flag);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  AdaptiveMaxOneSetsPlotFlagFromLegacyFormula)
{
  struct Case
  {
    const char * node_name;
    double plot_time;
    bool initial_plot_flag;
    bool expected_plot_flag;
  };
  const std::initializer_list<Case> cases{
    {
      "adaptive_plot_true_test",
      20.0,
      false,
      true,
    },
    {
      "adaptive_plot_false_test",
      21.0,
      true,
      false,
    },
  };

  for (const Case & test_case : cases)
  {
    SCOPED_TRACE(test_case.node_name);
    rclcpp::Node::SharedPtr unused_node;
    LIVMapper mapper(
      unused_node, test_case.node_name,
      outdoorOptions({
        "vio.camera_mode:=four",
        "vio.adaptive.enable:=true",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=1",
        "vio.min_joint_cameras:=1",
      }));
    appendVioMeasurement(mapper);
    mapper._pv_list.emplace_back();
    mapper.lio_feature_generation_ = 41U;
    mapper.voxelmap_manager->pose_information_ =
      livo::AdaptiveVisualPolicy::Matrix6::Identity();
    mapper.voxelmap_manager->pose_information_valid_ = false;
    mapper.plot_time = test_case.plot_time;
    mapper.frame_cnt = 6;
    mapper.vio_manager->plot_flag =
      test_case.initial_plot_flag;

    mapper.handleVIO();

    EXPECT_EQ(
      mapper.vio_manager->plot_flag,
      test_case.expected_plot_flag);
    EXPECT_EQ(
      mapper.adaptiveVisualLidarRuntimeDispatchAudit().status,
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kVioDispatched);
  }
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  EmptyLidarCacheDoesNotAdvanceAdaptiveAuthority)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_empty_lidar_cache_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
      "vio.grid_size:=20",
      "vio.grid_n_width:=32",
      "vio.grid_n_height:=24",
      "vio.patch_pyrimid_level:=1",
      "vio.max_iterations:=1",
    }));
  mapper.lio_feature_generation_ = 41U;
  mapper.last_joint_vio_feature_generation_ = 41U;
  mapper.voxelmap_manager->pose_information_ =
    3.0 * livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper.voxelmap_manager->pose_information_valid_ = true;

  testing::internal::CaptureStdout();
  mapper.handleVIO();
  const std::string output =
    testing::internal::GetCapturedStdout();

  const auto & audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kNoCachedLidarFeatures);
  EXPECT_FALSE(audit.lidar_posterior.has_value());
  EXPECT_FALSE(audit.generation_issued);
  EXPECT_FALSE(audit.runtime_bound);
  EXPECT_FALSE(audit.vio_dispatch_started);
  EXPECT_FALSE(audit.vio_dispatch_completed);
  EXPECT_EQ(audit.pre_vio_shadow_entry_count, 0U);
  EXPECT_EQ(audit.post_vio_shadow_entry_count, 0U);
  EXPECT_EQ(mapper.last_joint_vio_feature_generation_, 41U);
  EXPECT_EQ(
    countOccurrences(
      output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  InvalidLidarSnapshotFailsClosedWithTypedAuthorityStatus)
{
  struct Case
  {
    const char * node_name;
    std::uint64_t generation;
    bool nonfinite_information;
    livo::AdaptiveVisualLidarRuntimeDispatchStatus
      expected_dispatch_status;
    livo::AdaptiveVisualSelectionStatus
      expected_selection_status;
  };
  const std::initializer_list<Case> cases{
    {
      "adaptive_zero_lidar_generation_test",
      0U,
      false,
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kZeroLidarGeneration,
      livo::AdaptiveVisualSelectionStatus::
      kZeroLidarGeneration,
    },
    {
      "adaptive_nonfinite_lidar_information_test",
      42U,
      true,
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kNonFiniteLidarInformation,
      livo::AdaptiveVisualSelectionStatus::
      kInvalidLidarInformation,
    },
  };

  for (const Case & test_case : cases)
  {
    SCOPED_TRACE(test_case.node_name);
    rclcpp::Node::SharedPtr unused_node;
    LIVMapper mapper(
      unused_node, test_case.node_name,
      outdoorOptions({
        "vio.camera_mode:=four",
        "vio.adaptive.enable:=true",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=1",
        "vio.min_joint_cameras:=1",
      }));
    mapper._pv_list.emplace_back();
    mapper.lio_feature_generation_ = test_case.generation;
    mapper.last_joint_vio_feature_generation_ = 777U;
    mapper.voxelmap_manager->pose_information_ =
      5.0 * livo::AdaptiveVisualPolicy::Matrix6::Identity();
    if (test_case.nonfinite_information)
    {
      mapper.voxelmap_manager->pose_information_(0, 0) =
        std::numeric_limits<double>::quiet_NaN();
    }
    mapper.voxelmap_manager->pose_information_valid_ = true;

    appendVioMeasurement(
      mapper,
      makeSingleCameraBatch(
        13.0, 2U, texturedImage()));

    mapper.handleVIO();

    const auto & audit =
      mapper.adaptiveVisualLidarRuntimeDispatchAudit();
    EXPECT_EQ(audit.status, test_case.expected_dispatch_status);
    ASSERT_TRUE(audit.selection_status.has_value());
    EXPECT_EQ(
      *audit.selection_status,
      test_case.expected_selection_status);
    ASSERT_TRUE(audit.lidar_posterior.has_value());
    EXPECT_EQ(
      audit.lidar_posterior->generation,
      test_case.generation);
    EXPECT_FALSE(audit.generation_issued);
    EXPECT_FALSE(audit.runtime_bound);
    EXPECT_FALSE(audit.vio_dispatch_started);
    EXPECT_FALSE(audit.vio_dispatch_completed);
    EXPECT_EQ(audit.pre_vio_shadow_entry_count, 0U);
    EXPECT_EQ(audit.post_vio_shadow_entry_count, 0U);
    EXPECT_EQ(
      mapper.last_joint_vio_feature_generation_, 777U);
  }
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  LidarNotReadyDispatchesOnePreflightWithoutShadows)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_lidar_not_ready_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  initializeRuntimePublishers(mapper);
  appendVioMeasurement(mapper);
  mapper._pv_list.emplace_back();
  mapper.lio_feature_generation_ = 51U;
  mapper.last_joint_vio_feature_generation_ = 51U;
  livo::AdaptiveVisualPolicy::Matrix6 known_information =
    livo::AdaptiveVisualPolicy::Matrix6::Zero();
  known_information.diagonal() <<
    1.0, 2.0, 3.0, 5.0, 7.0, 11.0;
  mapper.voxelmap_manager->pose_information_ =
    known_information;
  mapper.voxelmap_manager->pose_information_valid_ = false;
  const livo::VIOReadOnlyStructuralAudit before =
    mapper.vio_manager->readOnlyStructuralAudit();

  mapper.handleVIO();

  const auto & mapper_audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    mapper_audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kVioDispatched);
  ASSERT_TRUE(mapper_audit.selection_status.has_value());
  EXPECT_EQ(
    *mapper_audit.selection_status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(mapper_audit.lidar_posterior.has_value());
  EXPECT_EQ(mapper_audit.lidar_posterior->generation, 51U);
  EXPECT_FALSE(
    mapper_audit.lidar_posterior->information_ready);
  EXPECT_TRUE(
    mapper_audit.lidar_posterior->pose_information.isApprox(
      known_information, 0.0));
  EXPECT_TRUE(mapper_audit.generation_issued);
  EXPECT_TRUE(mapper_audit.runtime_bound);
  EXPECT_TRUE(mapper_audit.vio_dispatch_started);
  EXPECT_TRUE(mapper_audit.vio_dispatch_completed);
  EXPECT_EQ(mapper_audit.pre_vio_shadow_entry_count, 0U);
  EXPECT_EQ(mapper_audit.post_vio_shadow_entry_count, 0U);
  EXPECT_EQ(mapper.last_joint_vio_feature_generation_, 51U);

  const auto & vio_audit =
    mapper.vio_manager->adaptiveVisualRuntimeAudit();
  EXPECT_EQ(
    vio_audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    vio_audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kLidarInformationNotReady);
  ASSERT_TRUE(vio_audit.lidar_status.has_value());
  EXPECT_EQ(
    *vio_audit.lidar_status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarInformationNotReady);
  EXPECT_EQ(vio_audit.lidar_generation, 51U);
  ASSERT_TRUE(
    vio_audit.validated_lidar_pose_information.has_value());
  EXPECT_TRUE(
    vio_audit.validated_lidar_pose_information->isApprox(
      known_information, 0.0));
  EXPECT_EQ(vio_audit.runtime_attempt_count, 0U);
  EXPECT_EQ(vio_audit.prepare_call_count, 0U);
  EXPECT_EQ(vio_audit.scouted_camera_count, 0U);
  EXPECT_EQ(vio_audit.solver_call_count, 0U);
  EXPECT_EQ(vio_audit.finalize_call_count, 0U);
  EXPECT_FALSE(vio_audit.solver_called);

  const livo::VIOReadOnlyStructuralAudit after =
    mapper.vio_manager->readOnlyStructuralAudit();
  EXPECT_EQ(
    after.shared_landmark_shadow_entry_count,
    before.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    after.reference_only_migration_shadow_entry_count,
    before.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    after.multi_camera_linearization_shadow_entry_count,
    before.multi_camera_linearization_shadow_entry_count);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  MapperRuntimeSelectsHealthyCameraAndFinalizesOnce)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_mapper_solver_test",
    outdoorIdentityVioExtrinsicOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
      "vio.grid_size:=20",
      "vio.grid_n_width:=32",
      "vio.grid_n_height:=24",
      "vio.patch_pyrimid_level:=1",
      "vio.max_iterations:=1",
    }));
  initializeRuntimePublishers(mapper);
  mapper._state.cov.setIdentity();
  mapper.state_propagat = mapper._state;
  const cv::Mat reference = texturedImage();
  seedCameraMap(mapper, 0U, reference);
  seedCameraMap(mapper, 2U, reference);
  appendVioMeasurement(
    mapper, sealTwoCameraBatch(mapper, 11.0));
  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 2U);
  mapper.lio_feature_generation_ = 60U;
  mapper.voxelmap_manager->pose_information_ =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper.voxelmap_manager->pose_information_valid_ = true;

  mapper.handleVIO();

  EXPECT_EQ(
    mapper.adaptiveVisualLidarRuntimeDispatchAudit().status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kVioDispatched);
  const auto & audit =
    mapper.vio_manager->adaptiveVisualRuntimeAudit();
  EXPECT_NE(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  ASSERT_TRUE(audit.selected_camera_id.has_value());
  EXPECT_EQ(*audit.selected_camera_id, 2U);
  EXPECT_TRUE(audit.solver_called);
  EXPECT_EQ(audit.solver_call_count, 1U);
  EXPECT_EQ(audit.finalize_call_count, 1U);
  EXPECT_EQ(audit.prepare_call_count, 2U);
  EXPECT_EQ(audit.scouted_camera_count, 2U);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  LioOnlyPublishesCameraColoredPointCloudWhenColoringEnabled)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "lio_only_camera_coloring_test",
    outdoorOptions({
      "common.img_en:=0",
      "multi_camera.enable:=true",
    }));
  initializeRuntimePublishers(mapper);

  auto observer = std::make_shared<rclcpp::Node>(
    "lio_only_camera_coloring_observer");
  sensor_msgs::msg::PointCloud2::SharedPtr received;
  auto subscription = observer->create_subscription<
    sensor_msgs::msg::PointCloud2>(
    "/cloud_registered", 10,
    [&received](sensor_msgs::msg::PointCloud2::SharedPtr message)
    {
      received = std::move(message);
    });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(observer);
  for (int attempt = 0;
       attempt < 100 &&
       mapper.pubLaserCloudFullRes->get_subscription_count() == 0U;
       ++attempt)
  {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  ASSERT_GT(
    mapper.pubLaserCloudFullRes->get_subscription_count(), 0U);

  ASSERT_NE(mapper.cam_front_, nullptr);
  mapper._state.rot_end.setIdentity();
  mapper._state.pos_end.setZero();
  mapper.img_rgb_front = cv::Mat(
    480, 640, CV_8UC3, cv::Scalar(17, 83, 191));
  const Eigen::Vector3d point_world =
    mapper.T_base_cam_front_ * Eigen::Vector3d(0.0, 0.0, 5.0);
  PointType point;
  point.x = static_cast<float>(point_world.x());
  point.y = static_cast<float>(point_world.y());
  point.z = static_cast<float>(point_world.z());
  mapper.pcl_w_wait_pub->push_back(point);

  mapper.publish_frame_world(
    mapper.pubLaserCloudFullRes, mapper.vio_manager);
  for (int attempt = 0; attempt < 100 && !received; ++attempt)
  {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  ASSERT_NE(received, nullptr);
  EXPECT_TRUE(hasPointCloudField(*received, "rgb"));
  EXPECT_GT(received->width * received->height, 0U);
  PointCloudXYZRGB colored_cloud;
  pcl::fromROSMsg(*received, colored_cloud);
  ASSERT_EQ(colored_cloud.size(), 1U);
  EXPECT_EQ(colored_cloud.front().r, 191U);
  EXPECT_EQ(colored_cloud.front().g, 83U);
  EXPECT_EQ(colored_cloud.front().b, 17U);
  EXPECT_EQ(mapper.img_en, 0);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  RuntimeReadyReachesLateSubscriberAfterColoringSubscriptionsExist)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive3_ready_test",
    outdoorOptions({
      "common.img_en:=0", "vio.camera_mode:=four",
      "multi_camera.enable:=true",
    }));
  initializeRuntimePublishers(mapper);
  ASSERT_NE(mapper.sub_img_front, nullptr);
  ASSERT_NE(mapper.sub_img_left, nullptr);
  ASSERT_NE(mapper.sub_img_back, nullptr);
  ASSERT_NE(mapper.sub_img_right, nullptr);
  ASSERT_NE(mapper.pubRuntimeReady, nullptr);
  EXPECT_STREQ(mapper.pubRuntimeReady->get_topic_name(), "/livo/runtime_ready");
  const auto qos = mapper.pubRuntimeReady->get_actual_qos().get_rmw_qos_profile();
  EXPECT_EQ(qos.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(qos.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
  EXPECT_EQ(qos.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(qos.depth, 1U);

  auto observer = std::make_shared<rclcpp::Node>("adaptive3_late_ready_observer");
  bool ready = false;
  auto subscription = observer->create_subscription<std_msgs::msg::Bool>(
    "/livo/runtime_ready", rclcpp::QoS(1).reliable().transient_local(),
    [&ready](std_msgs::msg::Bool::ConstSharedPtr message) { ready = message->data; });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(observer);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!ready && std::chrono::steady_clock::now() < deadline)
  {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(ready);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  FrontColoringUsesOneSubscriptionAndOnlyFrontEntersVisualQueue)
{
  for (const bool adaptive : {false, true})
  {
    SCOPED_TRACE(adaptive);
    rclcpp::Node::SharedPtr unused_node;
    LIVMapper mapper(
      unused_node, "adaptive3_front_coloring_test",
      outdoorOptions({
        "common.img_en:=1", "vio.camera_mode:=front",
        "multi_camera.enable:=true",
        adaptive ? "vio.adaptive.enable:=true" : "vio.adaptive.enable:=false",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=1",
        "vio.min_joint_cameras:=1",
      }));
    initializeRuntimePublishers(mapper);
    EXPECT_EQ(mapper.sub_img, nullptr);
    ASSERT_NE(mapper.sub_img_front, nullptr);
    ASSERT_NE(mapper.cam_front_, nullptr);
    ASSERT_NE(mapper.cam_left_, nullptr);
    auto message = cv_bridge::CvImage(
      std_msgs::msg::Header{}, "bgr8",
      cv::Mat(480, 640, CV_8UC3, cv::Scalar(17, 83, 191))).toImageMsg();
    message->header.stamp = sec2Stamp(10.1);
    mapper.img_cbk_front(message);
    mapper.img_cbk_left(message);
    mapper.drainCameraTransportBuffer();
    ASSERT_EQ(mapper.camera_observation_queue_.size(), 1U);
    EXPECT_EQ(mapper.camera_observation_queue_.nextReady(11.0)->camera_id, 0U);
    EXPECT_FALSE(mapper.img_rgb_front.empty());
    EXPECT_EQ(mapper.coloring_image_buffers_[1U].size(), 1U);
  }
}

// Exercise the same scheduler seam as the runtime, with one recoverable frame.
void prepareAdaptiveCameraRecovery(LIVMapper & mapper)
{
  mapper.slam_mode_ = LIVO;
  mapper.LidarMeasures.lio_vio_flg = WAIT;
  mapper.LidarMeasures.last_lio_update_time = 10.0;
  mapper.last_timestamp_imu = 10.5;
  auto imu = std::make_shared<sensor_msgs::msg::Imu>();
  imu->header.stamp = sec2Stamp(10.5);
  mapper.imu_buffer.push_back(imu);
  auto scan = std::make_shared<PointCloudXYZI>();
  PointType point{};
  point.curvature = 0.0F;
  scan->push_back(point);
  point.curvature = 200.0F;
  scan->push_back(point);
  mapper.lid_raw_data_buffer.push_back(scan);
  mapper.lid_header_time_buffer.push_back(10.0);
  CameraObservation observation;
  observation.camera_id = 0U;
  observation.timestamp = 10.1;
  observation.image = cv::Mat::ones(480, 640, CV_8UC1);
  ASSERT_EQ(mapper.camera_observation_queue_.push(std::move(observation)),
    CameraObservationPushResult::ACCEPTED);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  AdaptiveDeadlineWaitsForReceivedFramesInBothTransportStages)
{
  for (const bool staged : {false, true})
  {
    SCOPED_TRACE(staged);
    rclcpp::Node::SharedPtr unused_node;
    LIVMapper mapper(
      unused_node, "adaptive3_pending_camera_test",
      outdoorOptions({
        "common.img_en:=1", "vio.camera_mode:=four",
        "vio.adaptive.enable:=true", "vio.adaptive.partial_batch.enable:=true",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=1", "vio.min_joint_cameras:=1",
      }));
    initializeRuntimePublishers(mapper);
    prepareAdaptiveCameraRecovery(mapper);
    auto message = cv_bridge::CvImage(
      std_msgs::msg::Header{}, "bgr8",
      cv::Mat(480, 640, CV_8UC3, cv::Scalar(17, 83, 191))).toImageMsg();
    message->header.stamp = sec2Stamp(10.11);
    mapper.img_cbk_left(message);
    if (staged)
    {
      mapper.camera_transport_processing_buffer_.swap(mapper.camera_transport_buffer_);
    }
    EXPECT_FALSE(mapper.sync_packages(mapper.LidarMeasures));
    mapper.drainCameraTransportBuffer();
    ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
    ASSERT_TRUE(mapper.pending_vio_batch_.has_value());
    EXPECT_EQ(mapper.pending_vio_batch_->camera_count, 2U);
    EXPECT_TRUE(mapper.pending_vio_batch_->present[0U]);
    EXPECT_TRUE(mapper.pending_vio_batch_->present[1U]);
  }
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  FullCameraQueueReleasesExpiredBatchBeforeDrainingPendingImages)
{
  for (const bool staged : {false, true})
  {
    for (const double pending_stamp : {10.09, 10.1, 10.11})
    {
      SCOPED_TRACE(staged);
      SCOPED_TRACE(pending_stamp);
      rclcpp::Node::SharedPtr unused_node;
      LIVMapper mapper(
        unused_node, "adaptive3_full_camera_queue_test",
        outdoorOptions({
          "common.img_en:=1", "vio.camera_mode:=four",
          "vio.camera_queue_capacity:=4",
          "vio.adaptive.enable:=true", "vio.adaptive.partial_batch.enable:=true",
          "vio.adaptive.minimum_selected_cameras:=1",
          "vio.adaptive.maximum_selected_cameras:=1", "vio.min_joint_cameras:=1",
        }));
      initializeRuntimePublishers(mapper);
      prepareAdaptiveCameraRecovery(mapper);
      // Direct queue seeding must also mirror the ingress timestamp ledger.
      mapper.last_camera_timestamps_[0U] = 10.1;
      for (std::uint8_t camera_id = 1U; camera_id < 4U; ++camera_id)
      {
        CameraObservation observation;
        observation.camera_id = camera_id;
        observation.timestamp = 10.1;
        observation.image = cv::Mat::ones(480, 640, CV_8UC1);
        ASSERT_EQ(mapper.camera_observation_queue_.push(std::move(observation)),
          CameraObservationPushResult::ACCEPTED);
      }
      ASSERT_EQ(mapper.camera_observation_queue_.size(), mapper.camera_observation_queue_.capacity());
      auto message = cv_bridge::CvImage(
        std_msgs::msg::Header{}, "bgr8",
        cv::Mat(480, 640, CV_8UC3, cv::Scalar(17, 83, 191))).toImageMsg();
      message->header.stamp = sec2Stamp(pending_stamp);
      mapper.img_cbk_front(message);
      if (staged)
      {
        mapper.camera_transport_processing_buffer_.swap(mapper.camera_transport_buffer_);
      }
      mapper.last_timestamp_imu = 10.15;
      EXPECT_FALSE(mapper.sync_packages(mapper.LidarMeasures));
      mapper.last_timestamp_imu = 10.5;
      ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
      ASSERT_TRUE(mapper.pending_vio_batch_.has_value());
      EXPECT_EQ(mapper.pending_vio_batch_->camera_count, 4U);
      mapper.drainCameraTransportBuffer();
      EXPECT_TRUE(mapper.camera_transport_processing_buffer_.empty());
      EXPECT_TRUE(mapper.camera_transport_buffer_.empty());
      EXPECT_EQ(mapper.dropped_camera_frames_[0U], 1U);
    }
  }
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  AdaptiveConsumesLastCameraFrameThenContinuesLioAfterDropout)
{
  for (const char * camera_mode : {"vio.camera_mode:=front", "vio.camera_mode:=four"})
  {
    SCOPED_TRACE(camera_mode);
    rclcpp::Node::SharedPtr unused_node;
    LIVMapper mapper(
      unused_node, "adaptive3_last_camera_frame_test",
      outdoorOptions({
        "common.img_en:=1", camera_mode, "multi_camera.enable:=false",
        "vio.adaptive.enable:=true", "vio.adaptive.partial_batch.enable:=true",
        "vio.adaptive.minimum_selected_cameras:=1",
        "vio.adaptive.maximum_selected_cameras:=1", "vio.min_joint_cameras:=1",
      }));
    initializeRuntimePublishers(mapper);
    prepareAdaptiveCameraRecovery(mapper);
    ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
    ASSERT_EQ(mapper.LidarMeasures.lio_vio_flg, LIO);
    mapper.LidarMeasures.last_lio_update_time = 10.1;
    auto scan = std::make_shared<PointCloudXYZI>();
    PointType point{};
    point.curvature = 0.0F;
    scan->push_back(point);
    point.curvature = 100.0F;
    scan->push_back(point);
    mapper.lid_raw_data_buffer.push_back(scan);
    mapper.lid_header_time_buffer.push_back(10.2);
    ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
    EXPECT_EQ(mapper.LidarMeasures.lio_vio_flg, VIO);
    ASSERT_TRUE(mapper.camera_observation_queue_.empty());
    ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
    EXPECT_EQ(mapper.LidarMeasures.lio_vio_flg, LIO);
    EXPECT_NEAR(mapper.LidarMeasures.measures.back().lio_time, 10.3, 1e-9);
  }
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  LioOnlyBoundsUnprojectableColoringBatchAcrossPublishInterval)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "lio_only_unprojectable_coloring_batch_test",
    outdoorOptions({
      "common.img_en:=0",
      "multi_camera.enable:=true",
      "publish.pub_scan_num:=2",
    }));

  ASSERT_TRUE(mapper.img_rgb_front.empty());
  ASSERT_TRUE(mapper.img_rgb_left.empty());
  ASSERT_TRUE(mapper.img_rgb_back.empty());
  ASSERT_TRUE(mapper.img_rgb_right.empty());

  PointType first_point;
  first_point.x = 1.0F;
  first_point.y = 0.0F;
  first_point.z = 5.0F;
  mapper.pcl_w_wait_pub->push_back(first_point);
  mapper.publish_frame_world(
    mapper.pubLaserCloudFullRes, mapper.vio_manager);
  EXPECT_TRUE(mapper.pcl_w_wait_pub->empty());
  ASSERT_EQ(mapper.pcl_wait_pub->size(), 1U);

  PointType second_point;
  second_point.x = 2.0F;
  second_point.y = 0.0F;
  second_point.z = 5.0F;
  mapper.pcl_w_wait_pub->push_back(second_point);
  mapper.publish_frame_world(
    mapper.pubLaserCloudFullRes, mapper.vio_manager);
  EXPECT_TRUE(mapper.pcl_w_wait_pub->empty());
  EXPECT_TRUE(mapper.pcl_wait_pub->empty());
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  FourHealthPublishSkipsColoringUntilVisualFrameReady)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "four_health_null_visual_frame_publish_test",
    outdoorOptions({
      "common.img_en:=1",
      "vio.camera_mode:=four",
      "multi_camera.enable:=false",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  initializeRuntimePublishers(mapper);
  appendVioMeasurement(mapper);
  mapper._pv_list.emplace_back();
  mapper.lio_feature_generation_ = 62U;
  mapper.last_joint_vio_feature_generation_ = 62U;
  mapper.voxelmap_manager->pose_information_ =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper.voxelmap_manager->pose_information_valid_ = false;

  mapper.handleVIO();

  ASSERT_TRUE(
    mapper.adaptiveVisualLidarRuntimeDispatchAudit().
    vio_dispatch_completed);
  ASSERT_EQ(mapper.img_en, 1);
  ASSERT_EQ(mapper.vio_camera_mode_, "four");
  ASSERT_FALSE(mapper.multi_camera_enable_);
  ASSERT_NE(mapper.vio_manager, nullptr);
  ASSERT_EQ(mapper.vio_manager->new_frame_, nullptr);
  ASSERT_TRUE(mapper.pcl_w_wait_pub->empty());
  ASSERT_TRUE(mapper.pcl_wait_pub->empty());
  PointType pending_point;
  pending_point.x = 1.0F;
  pending_point.y = 0.0F;
  pending_point.z = 5.0F;
  mapper.pcl_w_wait_pub->push_back(pending_point);
  const livo::VIOReadOnlyStructuralAudit before =
    mapper.vio_manager->readOnlyStructuralAudit();

  mapper.publish_frame_world(
    mapper.pubLaserCloudFullRes, mapper.vio_manager);

  EXPECT_EQ(mapper.vio_manager->new_frame_, nullptr);
  EXPECT_EQ(mapper.pcl_w_wait_pub->size(), 1U);
  EXPECT_TRUE(mapper.pcl_wait_pub->empty());
  const livo::VIOReadOnlyStructuralAudit after =
    mapper.vio_manager->readOnlyStructuralAudit();
  EXPECT_EQ(after.active_state, before.active_state);
  EXPECT_EQ(after.frame_count, before.frame_count);
  EXPECT_EQ(
    after.adaptive_policy_revision,
    before.adaptive_policy_revision);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  UnexpectedProcessFailureRethrowsAndConsumesGeneration)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_unexpected_process_failure_test",
    outdoorOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
    }));
  appendVioMeasurement(
    mapper, sealMalformedSingleCameraBatch(mapper, 13.0, 2U));
  mapper._pv_list.emplace_back();
  mapper.lio_feature_generation_ = 63U;
  mapper.voxelmap_manager->pose_information_ =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper.voxelmap_manager->pose_information_valid_ = true;

  testing::internal::CaptureStdout();
  EXPECT_THROW(mapper.handleVIO(), std::runtime_error);
  const std::string failed_output =
    testing::internal::GetCapturedStdout();

  const auto & failed =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    failed.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kUnexpectedFailure);
  ASSERT_TRUE(failed.lidar_posterior.has_value());
  EXPECT_EQ(failed.lidar_posterior->generation, 63U);
  ASSERT_TRUE(failed.selection_status.has_value());
  EXPECT_EQ(
    *failed.selection_status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(failed.generation_issued);
  EXPECT_TRUE(failed.runtime_bound);
  EXPECT_TRUE(failed.vio_dispatch_started);
  EXPECT_FALSE(failed.vio_dispatch_completed);
  EXPECT_EQ(
    countOccurrences(
      failed_output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);

  EXPECT_NO_THROW(mapper.handleVIO());

  const auto & retry =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    retry.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kIssueRejected);
  ASSERT_TRUE(retry.selection_status.has_value());
  EXPECT_EQ(
    *retry.selection_status,
    livo::AdaptiveVisualSelectionStatus::
    kNonMonotonicLidarGeneration);
  EXPECT_FALSE(retry.generation_issued);
  EXPECT_FALSE(retry.runtime_bound);
  EXPECT_FALSE(retry.vio_dispatch_started);
  EXPECT_FALSE(retry.vio_dispatch_completed);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  NoEligibleAuthoritativePlanPublishesFactualZeroSelectionBatch)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_no_eligible_metric_test",
    outdoorIdentityVioExtrinsicOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
      "vio.grid_size:=20",
      "vio.grid_n_width:=32",
      "vio.grid_n_height:=24",
      "vio.patch_pyrimid_level:=1",
      "vio.max_iterations:=1",
    }));
  initializeRuntimePublishers(mapper);
  mapper._state.cov.setIdentity();
  mapper.state_propagat = mapper._state;
  const cv::Mat reference = texturedImage();
  for (std::uint8_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount;
       ++camera_id)
  {
    seedCameraMap(mapper, camera_id, reference);
  }
  CameraObservationBatch unhealthy_batch =
    sealUnhealthyFourCameraBatch(mapper, 13.0);
  const double batch_source_stamp = unhealthy_batch.timestamp;
  appendVioMeasurement(mapper, std::move(unhealthy_batch));
  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 0U);
  mapper.lio_feature_generation_ = 1U;
  mapper.voxelmap_manager->pose_information_ =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper.voxelmap_manager->pose_information_valid_ = true;
  const livo::VIOReadOnlyStructuralAudit before =
    mapper.vio_manager->readOnlyStructuralAudit();

  testing::internal::CaptureStdout();
  mapper.handleVIO();
  const std::string output =
    testing::internal::GetCapturedStdout();

  const livo::AdaptiveVisualRuntimeAudit & audit =
    mapper.vio_manager->adaptiveVisualRuntimeAudit();
  EXPECT_EQ(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kSelectionIssueRejected);
  ASSERT_TRUE(audit.selection_issue_status.has_value());
  EXPECT_EQ(
    *audit.selection_issue_status,
    livo::AdaptiveVisualSelectionStatus::
    kNoSelectedCamera);
  EXPECT_EQ(audit.prepare_call_count, 4U);
  EXPECT_EQ(audit.scouted_camera_count, 4U);
  EXPECT_TRUE(
    std::none_of(
      audit.eligible_mask.begin(), audit.eligible_mask.end(),
      [](const bool value) { return value; }));
  EXPECT_TRUE(
    std::none_of(
      audit.selected_mask.begin(), audit.selected_mask.end(),
      [](const bool value) { return value; }));
  EXPECT_TRUE(
    std::all_of(
      audit.reject_reasons.begin(), audit.reject_reasons.end(),
      [](const livo::AdaptiveCameraRejectReason reason)
      {
        return
          reason ==
          livo::AdaptiveCameraRejectReason::
          kImageHardFailure;
      }));
  EXPECT_FALSE(audit.capability_issued);
  EXPECT_FALSE(audit.receipt_consumed);
  EXPECT_FALSE(audit.health_commit_status.has_value());
  EXPECT_EQ(audit.selection_consumption_count, 0U);
  EXPECT_FALSE(audit.solver_called);
  EXPECT_EQ(audit.solver_call_count, 0U);
  EXPECT_EQ(audit.finalize_call_count, 0U);

  const std::vector<std::string> camera_metrics =
    linesContaining(output, "[ADAPTIVE_VIO_CAMERA_METRIC]");
  ASSERT_EQ(camera_metrics.size(), 4U);
  for (std::size_t camera_id = 0U;
       camera_id < camera_metrics.size(); ++camera_id)
  {
    const std::string & metric = camera_metrics[camera_id];
    EXPECT_EQ(metricField(metric, "generation"), "1");
    EXPECT_EQ(
      metricField(metric, "camera_id"),
      std::to_string(camera_id));
    EXPECT_EQ(metricField(metric, "frame_received"), "1");
    EXPECT_EQ(metricField(metric, "eligible"), "0");
    EXPECT_EQ(metricField(metric, "selected"), "0");
    EXPECT_EQ(
      metricField(metric, "reject_reason"),
      "image_hard_failure");
    ASSERT_FALSE(
      metricField(metric, "batch_source_stamp_sec").empty());
    EXPECT_NEAR(
      std::stod(
        metricField(metric, "batch_source_stamp_sec")),
      batch_source_stamp, 1.0e-9);
  }
  const livo::AdaptiveLioModePublicationAudit mode_audit =
    mapper.adaptiveLioModeAudit();
  ASSERT_TRUE(mode_audit.receipt.has_value());
  EXPECT_EQ(
    mode_audit.receipt->reason(),
    livo::AdaptiveLioModeReason::kNoEligibleVisual);
  EXPECT_EQ(
    countOccurrences(
      output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_NE(
    output.find(
      "generation=1 source_stamp_sec=20.000000000 "
      "mode=lio_only transition=enter_lio_only "
      "reason=no_eligible_visual"),
    std::string::npos);

  const livo::VIOReadOnlyStructuralAudit after =
    mapper.vio_manager->readOnlyStructuralAudit();
  EXPECT_EQ(after.active_state, before.active_state);
  EXPECT_EQ(after.propagated_state, before.propagated_state);
  EXPECT_EQ(
    after.legacy_feature_voxel_counts,
    before.legacy_feature_voxel_counts);
  EXPECT_EQ(
    after.legacy_feature_point_counts,
    before.legacy_feature_point_counts);
  EXPECT_EQ(
    after.last_joint_solver.accepted,
    before.last_joint_solver.accepted);
  EXPECT_EQ(
    after.last_joint_solver.reason,
    before.last_joint_solver.reason);
  EXPECT_EQ(
    after.last_joint_solver.contributing_cameras,
    before.last_joint_solver.contributing_cameras);
  EXPECT_EQ(
    after.last_joint_solver.tracked_points,
    before.last_joint_solver.tracked_points);
  EXPECT_EQ(
    after.active_camera_id, before.active_camera_id);
  EXPECT_EQ(
    after.adaptive_camera_initialized,
    before.adaptive_camera_initialized);
  EXPECT_EQ(after.frame_count, before.frame_count);
  EXPECT_EQ(
    after.adaptive_policy_revision,
    before.adaptive_policy_revision);
  EXPECT_EQ(
    after.adaptive_finalize_call_count,
    before.adaptive_finalize_call_count);
  EXPECT_EQ(
    after.shared_landmark_shadow_entry_count,
    before.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    after.reference_only_migration_shadow_entry_count,
    before.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    after.multi_camera_linearization_shadow_entry_count,
    before.multi_camera_linearization_shadow_entry_count);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  SealedFourCameraBatchUsesOneImmutableLidarSnapshot)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_sealed_snapshot_test",
    outdoorIdentityVioExtrinsicOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
      "vio.grid_size:=20",
      "vio.grid_n_width:=32",
      "vio.grid_n_height:=24",
      "vio.patch_pyrimid_level:=1",
      "vio.max_iterations:=1",
    }));
  initializeRuntimePublishers(mapper);
  const cv::Mat reference = texturedImage();
  for (std::uint8_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount;
       ++camera_id)
  {
    seedCameraMap(mapper, camera_id, reference);
  }
  appendVioMeasurement(
    mapper, sealFourCameraBatch(mapper, 11.0));
  mapper.lio_feature_generation_ = 61U;
  mapper.last_joint_vio_feature_generation_ = 61U;
  livo::AdaptiveVisualPolicy::Matrix6 known_information =
    livo::AdaptiveVisualPolicy::Matrix6::Zero();
  known_information.diagonal() <<
    13.0, 17.0, 19.0, 23.0, 29.0, 31.0;
  mapper.voxelmap_manager->pose_information_ =
    known_information;
  mapper.voxelmap_manager->pose_information_valid_ = true;
  const livo::VIOReadOnlyStructuralAudit before =
    mapper.vio_manager->readOnlyStructuralAudit();

  mapper._pv_list.clear();
  mapper.handleVIO();

  const auto & empty_audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    empty_audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kNoCachedLidarFeatures);
  EXPECT_FALSE(empty_audit.lidar_posterior.has_value());
  EXPECT_FALSE(empty_audit.generation_issued);
  EXPECT_FALSE(empty_audit.runtime_bound);
  EXPECT_FALSE(empty_audit.vio_dispatch_started);
  EXPECT_FALSE(empty_audit.vio_dispatch_completed);

  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 3U);
  mapper.handleVIO();

  const auto & mapper_audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    mapper_audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kVioDispatched);
  ASSERT_TRUE(mapper_audit.lidar_posterior.has_value());
  EXPECT_TRUE(
    mapper_audit.lidar_posterior->pose_information.isApprox(
      known_information, 0.0));
  EXPECT_EQ(mapper_audit.lidar_posterior->generation, 61U);
  EXPECT_TRUE(
    mapper_audit.lidar_posterior->information_ready);
  EXPECT_TRUE(mapper_audit.generation_issued);
  EXPECT_TRUE(mapper_audit.runtime_bound);
  EXPECT_TRUE(mapper_audit.vio_dispatch_started);
  EXPECT_TRUE(mapper_audit.vio_dispatch_completed);
  EXPECT_EQ(mapper.last_joint_vio_feature_generation_, 61U);

  const auto & vio_audit =
    mapper.vio_manager->adaptiveVisualRuntimeAudit();
  EXPECT_EQ(vio_audit.lidar_generation, 61U);
  ASSERT_TRUE(
    vio_audit.validated_lidar_pose_information.has_value());
  EXPECT_TRUE(
    vio_audit.validated_lidar_pose_information->isApprox(
      known_information, 0.0));
  ASSERT_TRUE(vio_audit.batch_preflight_status.has_value());
  EXPECT_EQ(
    *vio_audit.batch_preflight_status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(vio_audit.runtime_attempt_status.has_value());
  EXPECT_EQ(
    *vio_audit.runtime_attempt_status,
    livo::AdaptiveVisualRuntimeAttemptStatus::kReady);
  EXPECT_EQ(vio_audit.runtime_attempt_count, 1U);
  EXPECT_EQ(vio_audit.prepare_call_count, 4U);
  EXPECT_EQ(vio_audit.scouted_camera_count, 4U);
  ASSERT_TRUE(vio_audit.selection_issue_status.has_value());
  EXPECT_EQ(
    *vio_audit.selection_issue_status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(vio_audit.selected_camera_id.has_value());
  EXPECT_EQ(*vio_audit.selected_camera_id, 3U);
  EXPECT_TRUE(vio_audit.capability_issued);
  EXPECT_TRUE(vio_audit.receipt_consumed);
  EXPECT_EQ(vio_audit.selection_consumption_count, 1U);
  EXPECT_TRUE(vio_audit.solver_called);
  EXPECT_EQ(vio_audit.solver_call_count, 1U);
  EXPECT_EQ(vio_audit.finalize_call_count, 1U);
  EXPECT_NE(
    vio_audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    vio_audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kNone);
  EXPECT_FALSE(vio_audit.legacy_fallback_used);

  const livo::VIOReadOnlyStructuralAudit after =
    mapper.vio_manager->readOnlyStructuralAudit();
  EXPECT_EQ(
    after.shared_landmark_shadow_entry_count,
    before.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    after.reference_only_migration_shadow_entry_count,
    before.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    after.multi_camera_linearization_shadow_entry_count,
    before.multi_camera_linearization_shadow_entry_count);

  mapper.voxelmap_manager->pose_information_.setConstant(999.0);
  ASSERT_TRUE(
    mapper.adaptiveVisualLidarRuntimeDispatchAudit().
    lidar_posterior.has_value());
  EXPECT_TRUE(
    mapper.adaptiveVisualLidarRuntimeDispatchAudit().
    lidar_posterior->pose_information.isApprox(
      known_information, 0.0));
  EXPECT_TRUE(
    mapper.vio_manager->adaptiveVisualRuntimeAudit().
    validated_lidar_pose_information->isApprox(
      known_information, 0.0));

  livo::AdaptiveVisualPolicy::Matrix6
    duplicate_information =
    2.0 * livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 3U);
  mapper.voxelmap_manager->pose_information_ =
    duplicate_information;
  mapper.handleVIO();

  const auto & duplicate_audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    duplicate_audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kIssueRejected);
  ASSERT_TRUE(duplicate_audit.selection_status.has_value());
  EXPECT_EQ(
    *duplicate_audit.selection_status,
    livo::AdaptiveVisualSelectionStatus::
    kNonMonotonicLidarGeneration);
  ASSERT_TRUE(duplicate_audit.lidar_posterior.has_value());
  EXPECT_EQ(duplicate_audit.lidar_posterior->generation, 61U);
  EXPECT_TRUE(
    duplicate_audit.lidar_posterior->pose_information.isApprox(
      duplicate_information, 0.0));
  EXPECT_FALSE(duplicate_audit.generation_issued);
  EXPECT_FALSE(duplicate_audit.runtime_bound);
  EXPECT_FALSE(duplicate_audit.vio_dispatch_started);
  EXPECT_FALSE(duplicate_audit.vio_dispatch_completed);
  EXPECT_EQ(mapper.last_joint_vio_feature_generation_, 61U);
  EXPECT_EQ(
    mapper.vio_manager->adaptiveVisualRuntimeAudit().
    lidar_generation,
    61U);

  appendVioMeasurement(
    mapper, sealFourCameraBatch(mapper, 12.0));
  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 3U);
  mapper.lio_feature_generation_ = 62U;
  livo::AdaptiveVisualPolicy::Matrix6 next_information =
    livo::AdaptiveVisualPolicy::Matrix6::Zero();
  next_information.diagonal() <<
    37.0, 41.0, 43.0, 47.0, 53.0, 59.0;
  mapper.voxelmap_manager->pose_information_ =
    next_information;
  mapper.handleVIO();

  const auto & recovered_mapper_audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    recovered_mapper_audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kVioDispatched);
  ASSERT_TRUE(
    recovered_mapper_audit.lidar_posterior.has_value());
  EXPECT_EQ(
    recovered_mapper_audit.lidar_posterior->generation, 62U);
  EXPECT_TRUE(
    recovered_mapper_audit.lidar_posterior->
    pose_information.isApprox(
      next_information, 0.0));
  EXPECT_TRUE(recovered_mapper_audit.generation_issued);
  EXPECT_TRUE(recovered_mapper_audit.runtime_bound);
  EXPECT_TRUE(recovered_mapper_audit.vio_dispatch_started);
  EXPECT_TRUE(recovered_mapper_audit.vio_dispatch_completed);
  EXPECT_EQ(mapper.last_joint_vio_feature_generation_, 61U);

  const auto & recovered_vio_audit =
    mapper.vio_manager->adaptiveVisualRuntimeAudit();
  EXPECT_EQ(recovered_vio_audit.lidar_generation, 62U);
  ASSERT_TRUE(
    recovered_vio_audit.
    validated_lidar_pose_information.has_value());
  EXPECT_TRUE(
    recovered_vio_audit.
    validated_lidar_pose_information->isApprox(
      next_information, 0.0));
  ASSERT_TRUE(
    recovered_vio_audit.batch_preflight_status.has_value());
  EXPECT_EQ(
    *recovered_vio_audit.batch_preflight_status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(
    recovered_vio_audit.runtime_attempt_status.has_value());
  EXPECT_EQ(
    *recovered_vio_audit.runtime_attempt_status,
    livo::AdaptiveVisualRuntimeAttemptStatus::kReady);
  EXPECT_EQ(recovered_vio_audit.runtime_attempt_count, 1U);
  EXPECT_EQ(recovered_vio_audit.prepare_call_count, 4U);
  EXPECT_EQ(recovered_vio_audit.scouted_camera_count, 4U);
  EXPECT_FALSE(recovered_vio_audit.legacy_fallback_used);

  const livo::VIOReadOnlyStructuralAudit recovered =
    mapper.vio_manager->readOnlyStructuralAudit();
  EXPECT_EQ(
    recovered.shared_landmark_shadow_entry_count,
    before.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    recovered.reference_only_migration_shadow_entry_count,
    before.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    recovered.multi_camera_linearization_shadow_entry_count,
    before.multi_camera_linearization_shadow_entry_count);

  mapper.voxelmap_manager->pose_information_.setConstant(-999.0);
  ASSERT_TRUE(
    mapper.adaptiveVisualLidarRuntimeDispatchAudit().
    lidar_posterior.has_value());
  EXPECT_TRUE(
    mapper.adaptiveVisualLidarRuntimeDispatchAudit().
    lidar_posterior->pose_information.isApprox(
      next_information, 0.0));
  EXPECT_TRUE(
    mapper.vio_manager->adaptiveVisualRuntimeAudit().
    validated_lidar_pose_information->isApprox(
      next_information, 0.0));
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  VisualRecoveryUsesAdaptiveGenerationAfterLioFallback)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_same_generation_recovery_test",
    outdoorIdentityVioExtrinsicOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.partial_batch.enable:=true",
      "vio.adaptive.partial_batch.missing_camera_timeout_ms:=30.0",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
      "vio.grid_size:=20",
      "vio.grid_n_width:=32",
      "vio.grid_n_height:=24",
      "vio.patch_pyrimid_level:=1",
      "vio.max_iterations:=1",
    }));
  initializeRuntimePublishers(mapper);
  mapper.joint_vio_warmup_sec_ = 0.0;
  mapper.exposure_time_init = 0.0;
  mapper._state.cov.setIdentity();
  mapper.state_propagat = mapper._state;
  mapper.voxelmap_manager->state_ = mapper._state;
  const cv::Mat reference = texturedImage();
  seedCameraMap(mapper, 0U, reference);
  seedCameraMap(mapper, 2U, reference);

  mapper.slam_mode_ = LIVO;
  mapper.last_timestamp_imu = 20.2;
  PointCloudXYZI::Ptr first_lidar(new PointCloudXYZI());
  PointType first_scan_begin;
  first_scan_begin.curvature = 0.0F;
  first_lidar->points.push_back(first_scan_begin);
  PointType first_scan_end;
  first_scan_end.curvature = 100.0F;
  first_lidar->points.push_back(first_scan_end);
  mapper.lid_raw_data_buffer.push_back(first_lidar);
  mapper.lid_header_time_buffer.push_back(20.0);
  auto first_imu =
    std::make_shared<sensor_msgs::msg::Imu>();
  first_imu->header.stamp.sec = 20;
  first_imu->header.stamp.nanosec = 200000000U;
  mapper.imu_buffer.push_back(first_imu);
  mapper.LidarMeasures.lio_vio_flg = WAIT;
  ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
  ASSERT_EQ(mapper.LidarMeasures.lio_vio_flg, LIO);
  ASSERT_EQ(mapper.LidarMeasures.measures.size(), 1U);
  mapper.LidarMeasures.last_lio_update_time =
    mapper.LidarMeasures.measures.back().lio_time;
  populatePlanarLidarFeatures(mapper);

  testing::internal::CaptureStdout();
  mapper.handleLIO();
  const std::string fallback_output =
    testing::internal::GetCapturedStdout();

  EXPECT_EQ(mapper.lio_feature_generation_, 1U);
  EXPECT_EQ(
    countOccurrences(
      fallback_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    0U);
  EXPECT_FALSE(
    mapper.adaptiveLioModeAudit().receipt.has_value());

  mapper.LidarMeasures.measures.clear();
  CameraObservation recovered_observation;
  recovered_observation.camera_id = 0U;
  recovered_observation.timestamp =
    mapper.LidarMeasures.last_lio_update_time;
  recovered_observation.image = reference.clone();
  recovered_observation.photometric_status =
    livo::PhotometricStatus::kValid;
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
  mapper.lid_header_time_buffer.push_back(20.1);
  auto second_imu =
    std::make_shared<sensor_msgs::msg::Imu>();
  second_imu->header.stamp.sec = 20;
  second_imu->header.stamp.nanosec = 300000000U;
  mapper.imu_buffer.push_back(second_imu);
  mapper.last_timestamp_imu = 20.3;
  ASSERT_TRUE(mapper.sync_packages(mapper.LidarMeasures));
  ASSERT_EQ(mapper.LidarMeasures.lio_vio_flg, VIO);
  ASSERT_EQ(mapper.LidarMeasures.measures.size(), 1U);
  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 0U);
  mapper.voxelmap_manager->pose_information_ =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper.voxelmap_manager->pose_information_valid_ = true;

  testing::internal::CaptureStdout();
  mapper.handleVIO();
  const std::string recovery_output =
    testing::internal::GetCapturedStdout();

  const auto & dispatch_audit =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    dispatch_audit.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::kVioDispatched);
  EXPECT_TRUE(dispatch_audit.generation_issued);
  EXPECT_TRUE(dispatch_audit.vio_dispatch_completed);
  EXPECT_EQ(
    countOccurrences(
      fallback_output + recovery_output,
      "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_EQ(
    countOccurrences(
      fallback_output + recovery_output,
      "mode=lio_only"),
    0U);
  EXPECT_NE(
    recovery_output.find(
      "[VIO_METRIC] t=20.100000000 camera_id=0 camera=front"),
    std::string::npos);
}

TEST_F(
  LIVMapperAdaptiveVisualRuntimeTest,
  LidarNotReadyPublishesUnobservedThenSelectedVisualRecoveryModeReceipts)
{
  rclcpp::Node::SharedPtr unused_node;
  LIVMapper mapper(
    unused_node, "adaptive_mode_receipt_test",
    outdoorIdentityVioExtrinsicOptions({
      "vio.camera_mode:=four",
      "vio.adaptive.enable:=true",
      "vio.adaptive.minimum_selected_cameras:=1",
      "vio.adaptive.maximum_selected_cameras:=1",
      "vio.min_joint_cameras:=1",
      "vio.grid_size:=20",
      "vio.grid_n_width:=32",
      "vio.grid_n_height:=24",
      "vio.patch_pyrimid_level:=1",
      "vio.max_iterations:=1",
    }));
  initializeRuntimePublishers(mapper);
  mapper._state.cov.setIdentity();
  mapper.state_propagat = mapper._state;
  const cv::Mat reference = texturedImage();
  seedCameraMap(mapper, 0U, reference);
  seedCameraMap(mapper, 2U, reference);

  const livo::VIOReadOnlyStructuralAudit before =
    mapper.vio_manager->readOnlyStructuralAudit();
  appendVioMeasurement(
    mapper, sealTwoCameraBatch(mapper, 11.0));
  mapper.LidarMeasures.last_lio_update_time = 20.0;
  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 2U);
  mapper.lio_feature_generation_ = 1U;
  mapper.voxelmap_manager->pose_information_ =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  mapper.voxelmap_manager->pose_information_valid_ = false;

  testing::internal::CaptureStdout();
  mapper.handleVIO();
  const livo::AdaptiveLioModePublicationAudit lio_only =
    mapper.adaptiveLioModeAudit();
  const std::string first_output =
    testing::internal::GetCapturedStdout();

  ASSERT_EQ(
    lio_only.status,
    livo::AdaptiveLioModeRecordStatus::kRecorded);
  ASSERT_TRUE(lio_only.receipt.has_value());
  EXPECT_EQ(lio_only.receipt->generation(), 1U);
  EXPECT_DOUBLE_EQ(
    lio_only.receipt->sourceStampSec(), 20.0);
  EXPECT_EQ(
    lio_only.receipt->mode(),
    livo::AdaptiveLioMode::kLioOnly);
  EXPECT_EQ(
    lio_only.receipt->reason(),
    livo::AdaptiveLioModeReason::
    kVisualOutcomeNotObservedBeforeCloseout);
  EXPECT_EQ(
    lio_only.receipt->transition(),
    livo::AdaptiveLioModeTransition::kEnterLioOnly);
  EXPECT_TRUE(lio_only.metric_emitted);
  EXPECT_EQ(
    countOccurrences(
      first_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_NE(
    first_output.find(
      "generation=1 source_stamp_sec=20.000000000 "
      "mode=lio_only transition=enter_lio_only "
      "reason=visual_outcome_not_observed_before_closeout"),
    std::string::npos);
  EXPECT_EQ(
    countOccurrences(
      first_output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    0U);
  const livo::AdaptiveVisualLidarRuntimeDispatchAudit &
    first_dispatch =
    mapper.adaptiveVisualLidarRuntimeDispatchAudit();
  EXPECT_EQ(
    first_dispatch.status,
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kVioDispatched);
  ASSERT_TRUE(first_dispatch.selection_status.has_value());
  EXPECT_EQ(
    *first_dispatch.selection_status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  const livo::AdaptiveVisualRuntimeAudit & first_visual =
    mapper.vio_manager->adaptiveVisualRuntimeAudit();
  EXPECT_EQ(
    first_visual.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    first_visual.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kLidarInformationNotReady);
  EXPECT_EQ(first_visual.runtime_attempt_count, 0U);
  EXPECT_EQ(first_visual.prepare_call_count, 0U);
  EXPECT_EQ(first_visual.scouted_camera_count, 0U);
  EXPECT_EQ(first_visual.solver_call_count, 0U);
  EXPECT_EQ(first_visual.finalize_call_count, 0U);
  EXPECT_FALSE(first_visual.solver_called);
  const livo::VIOReadOnlyStructuralAudit after_lio_only =
    mapper.vio_manager->readOnlyStructuralAudit();
  EXPECT_EQ(after_lio_only.active_state, before.active_state);
  EXPECT_EQ(
    after_lio_only.propagated_state,
    before.propagated_state);
  EXPECT_EQ(
    after_lio_only.shared_landmark_shadow_entry_count,
    before.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    after_lio_only.reference_only_migration_shadow_entry_count,
    before.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    after_lio_only.multi_camera_linearization_shadow_entry_count,
    before.multi_camera_linearization_shadow_entry_count);
  EXPECT_EQ(
    after_lio_only.legacy_feature_voxel_counts,
    before.legacy_feature_voxel_counts);
  EXPECT_EQ(
    after_lio_only.legacy_feature_point_counts,
    before.legacy_feature_point_counts);
  EXPECT_EQ(
    after_lio_only.last_joint_solver.accepted,
    before.last_joint_solver.accepted);
  EXPECT_EQ(
    after_lio_only.last_joint_solver.reason,
    before.last_joint_solver.reason);
  EXPECT_EQ(
    after_lio_only.last_joint_solver.contributing_cameras,
    before.last_joint_solver.contributing_cameras);
  EXPECT_EQ(
    after_lio_only.last_joint_solver.tracked_points,
    before.last_joint_solver.tracked_points);
  EXPECT_EQ(
    after_lio_only.active_camera_id,
    before.active_camera_id);
  EXPECT_EQ(
    after_lio_only.adaptive_camera_initialized,
    before.adaptive_camera_initialized);
  EXPECT_EQ(
    after_lio_only.frame_count,
    before.frame_count);
  EXPECT_EQ(
    after_lio_only.adaptive_policy_revision,
    before.adaptive_policy_revision);
  EXPECT_EQ(
    after_lio_only.adaptive_finalize_call_count,
    before.adaptive_finalize_call_count);

  appendVioMeasurement(
    mapper, sealTwoCameraBatch(mapper, 12.0));
  mapper.LidarMeasures.last_lio_update_time = 21.0;
  mapper._pv_list =
    cameraVisibleLidarPoints(*mapper.vio_manager, 2U);
  mapper.lio_feature_generation_ = 2U;
  mapper.voxelmap_manager->pose_information_valid_ = true;

  testing::internal::CaptureStdout();
  mapper.handleVIO();
  const livo::AdaptiveLioModePublicationAudit recovered =
    mapper.adaptiveLioModeAudit();
  const std::string second_output =
    testing::internal::GetCapturedStdout();

  ASSERT_EQ(
    recovered.status,
    livo::AdaptiveLioModeRecordStatus::kRecorded);
  ASSERT_TRUE(recovered.receipt.has_value());
  EXPECT_EQ(recovered.receipt->generation(), 2U);
  EXPECT_DOUBLE_EQ(
    recovered.receipt->sourceStampSec(), 21.0);
  EXPECT_EQ(
    recovered.receipt->mode(),
    livo::AdaptiveLioMode::kSelectedVisual);
  EXPECT_EQ(
    recovered.receipt->reason(),
    livo::AdaptiveLioModeReason::kEligibleVisualSelected);
  EXPECT_EQ(
    recovered.receipt->transition(),
    livo::AdaptiveLioModeTransition::kExitLioOnly);
  EXPECT_TRUE(recovered.metric_emitted);
  EXPECT_EQ(
    countOccurrences(
      second_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    1U);
  EXPECT_NE(
    second_output.find(
      "generation=2 source_stamp_sec=21.000000000 "
      "mode=selected_visual transition=exit_lio_only "
      "reason=eligible_visual_selected"),
    std::string::npos);

  testing::internal::CaptureStdout();
  mapper.handleVIO();
  const std::string duplicate_output =
    testing::internal::GetCapturedStdout();
  EXPECT_EQ(
    mapper.adaptiveLioModeAudit().status,
    livo::AdaptiveLioModeRecordStatus::
    kNonContinuousGeneration);
  EXPECT_FALSE(
    mapper.adaptiveLioModeAudit().receipt.has_value());
  EXPECT_FALSE(mapper.adaptiveLioModeAudit().metric_emitted);
  EXPECT_EQ(
    countOccurrences(
      duplicate_output, "[ADAPTIVE_VIO_MODE_METRIC]"),
    0U);
}
