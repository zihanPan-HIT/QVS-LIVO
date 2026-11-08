/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "LIVMapper.h"
#include "checked_increment.h"
#include "shared_landmark_lifecycle_routing.h"
#include "structured_metric.h"
#include "local_scan_contract.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iterator>
#include <new>
#include <omp.h>
#include <stdexcept>
#include <thread>
#include <type_traits>

using namespace Sophus;

namespace
{
constexpr std::uint64_t kFnv1aOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1aPrime = 1099511628211ULL;
constexpr std::size_t kColoringImageBufferCapacity = 8U;
constexpr std::size_t kLidarSubscriptionDepth = 128U;
constexpr std::size_t kImuSubscriptionDepth = 4096U;
constexpr std::size_t kCameraSubscriptionDepth = 128U;
constexpr std::size_t kImuTransportBufferCapacity = 4096U;
constexpr std::size_t kImuPropagationBufferCapacity = 4096U;
constexpr std::size_t kCameraTransportBufferCapacity = 512U;
constexpr std::size_t kCameraTransportDrainBudget =
  CameraObservationQueue::kCameraCount;

void hashBytes(
  std::uint64_t & hash, const void * data, const std::size_t size) noexcept
{
  const auto * bytes = static_cast<const unsigned char *>(data);
  for (std::size_t index = 0U; index < size; ++index)
  {
    hash ^= static_cast<std::uint64_t>(bytes[index]);
    hash *= kFnv1aPrime;
  }
}

template<typename ValueT>
void hashValue(std::uint64_t & hash, const ValueT & value) noexcept
{
  static_assert(std::is_trivially_copyable<ValueT>::value);
  hashBytes(hash, &value, sizeof(ValueT));
}

std::uint64_t hashPointCloud(const PointCloudXYZI & cloud) noexcept
{
  std::uint64_t hash = kFnv1aOffsetBasis;
  const std::uint64_t point_count = cloud.points.size();
  hashValue(hash, point_count);
  for (const PointType & point : cloud.points)
  {
    hashValue(hash, point.x);
    hashValue(hash, point.y);
    hashValue(hash, point.z);
    hashValue(hash, point.intensity);
    hashValue(hash, point.normal_x);
    hashValue(hash, point.normal_y);
    hashValue(hash, point.normal_z);
    hashValue(hash, point.curvature);
  }
  return hash;
}

std::uint64_t doubleBits(const double value) noexcept
{
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

using ImuPayloadKey = std::array<std::uint64_t, 37U>;

ImuPayloadKey imuPayloadKey(const sensor_msgs::msg::Imu & imu) noexcept
{
  ImuPayloadKey key{};
  std::size_t index = 0U;
  const auto append = [&key, &index](const double value)
  {
    key[index++] = doubleBits(value);
  };
  append(imu.orientation.x);
  append(imu.orientation.y);
  append(imu.orientation.z);
  append(imu.orientation.w);
  for (const double value : imu.orientation_covariance) append(value);
  append(imu.angular_velocity.x);
  append(imu.angular_velocity.y);
  append(imu.angular_velocity.z);
  for (const double value : imu.angular_velocity_covariance) append(value);
  append(imu.linear_acceleration.x);
  append(imu.linear_acceleration.y);
  append(imu.linear_acceleration.z);
  for (const double value : imu.linear_acceleration_covariance) append(value);
  return key;
}

bool imuMessageLess(
  const sensor_msgs::msg::Imu & left,
  const sensor_msgs::msg::Imu & right) noexcept
{
  if (left.header.stamp.sec != right.header.stamp.sec)
  {
    return left.header.stamp.sec < right.header.stamp.sec;
  }
  if (left.header.stamp.nanosec != right.header.stamp.nanosec)
  {
    return left.header.stamp.nanosec < right.header.stamp.nanosec;
  }
  if (left.header.frame_id != right.header.frame_id)
  {
    return left.header.frame_id < right.header.frame_id;
  }
  return imuPayloadKey(left) < imuPayloadKey(right);
}

void hashImuPayload(
  std::uint64_t & hash, const sensor_msgs::msg::Imu & imu) noexcept
{
  const std::uint64_t frame_id_size = imu.header.frame_id.size();
  hashValue(hash, frame_id_size);
  hashBytes(hash, imu.header.frame_id.data(), imu.header.frame_id.size());
  const ImuPayloadKey key = imuPayloadKey(imu);
  for (const std::uint64_t value : key) hashValue(hash, value);
}

std::uint64_t hashImuSamples(
  const LidarMeasureGroup & measurements,
  std::uint64_t & stamp_hash,
  std::size_t & sample_count) noexcept
{
  stamp_hash = kFnv1aOffsetBasis;
  std::uint64_t sample_hash = kFnv1aOffsetBasis;
  sample_count = 0U;
  if (!measurements.measures.empty())
  {
    const MeasureGroup & measurement = measurements.measures.back();
    for (const sensor_msgs::msg::Imu::ConstSharedPtr & imu : measurement.imu)
    {
      hashValue(stamp_hash, imu->header.stamp.sec);
      hashValue(stamp_hash, imu->header.stamp.nanosec);
      hashValue(sample_hash, imu->header.stamp.sec);
      hashValue(sample_hash, imu->header.stamp.nanosec);
      hashImuPayload(sample_hash, *imu);
      ++sample_count;
    }
  }
  const std::uint64_t count = sample_count;
  hashValue(stamp_hash, count);
  hashValue(sample_hash, count);
  return sample_hash;
}

template<typename MatrixT>
void hashMatrix(std::uint64_t & hash, const MatrixT & matrix) noexcept
{
  for (Eigen::Index row = 0; row < matrix.rows(); ++row)
  {
    for (Eigen::Index column = 0; column < matrix.cols(); ++column)
    {
      hashValue(hash, matrix(row, column));
    }
  }
}

std::uint64_t hashNominalState(const StatesGroup & state) noexcept
{
  std::uint64_t hash = kFnv1aOffsetBasis;
  hashMatrix(hash, state.rot_end);
  hashMatrix(hash, state.pos_end);
  hashMatrix(hash, state.vel_end);
  hashMatrix(hash, state.bias_g);
  hashMatrix(hash, state.bias_a);
  hashMatrix(hash, state.gravity);
  hashValue(hash, state.inv_expo_time);
  return hash;
}

std::uint64_t hashState(const StatesGroup & state) noexcept
{
  std::uint64_t hash = hashNominalState(state);
  hashMatrix(hash, state.cov);
  return hash;
}

std::uint64_t hashStateCovariance(const StatesGroup & state) noexcept
{
  std::uint64_t hash = kFnv1aOffsetBasis;
  hashMatrix(hash, state.cov);
  return hash;
}

const char * updateTypeName(const EKF_STATE update_type) noexcept
{
  switch (update_type)
  {
    case VIO:
      return "vio";
    case LIO:
      return "lio";
    case LO:
      return "lo";
    case WAIT:
      return "wait";
  }
  return "unknown";
}

class AdaptiveVisualShadowEntryAuditScope
{
public:
  AdaptiveVisualShadowEntryAuditScope(
    livo::AdaptiveVisualLidarRuntimeDispatchAudit & audit,
    const std::size_t & pre_entry_count,
    const std::size_t & post_entry_count) noexcept
  : audit_(audit),
    pre_entry_count_(pre_entry_count),
    post_entry_count_(post_entry_count),
    pre_entry_count_before_(pre_entry_count),
    post_entry_count_before_(post_entry_count)
  {
  }

  ~AdaptiveVisualShadowEntryAuditScope() noexcept
  {
    audit_.pre_vio_shadow_entry_count =
      pre_entry_count_ - pre_entry_count_before_;
    audit_.post_vio_shadow_entry_count =
      post_entry_count_ - post_entry_count_before_;
  }

  AdaptiveVisualShadowEntryAuditScope(
    const AdaptiveVisualShadowEntryAuditScope &) = delete;
  AdaptiveVisualShadowEntryAuditScope & operator=(
    const AdaptiveVisualShadowEntryAuditScope &) = delete;

private:
  livo::AdaptiveVisualLidarRuntimeDispatchAudit & audit_;
  const std::size_t & pre_entry_count_;
  const std::size_t & post_entry_count_;
  std::size_t pre_entry_count_before_{0U};
  std::size_t post_entry_count_before_{0U};
};

bool shouldPlotVioFrame(
  const double relative_time,
  const double configured_plot_time,
  const int configured_frame_count) noexcept
{
  return
    std::fabs(relative_time - configured_plot_time) <
    (configured_frame_count / 2 * 0.1);
}

const char *adaptiveLioModeName(
  const livo::AdaptiveLioMode mode) noexcept
{
  switch (mode)
  {
    case livo::AdaptiveLioMode::kSelectedVisual:
      return "selected_visual";
    case livo::AdaptiveLioMode::kLioOnly:
      return "lio_only";
  }
  return "unknown";
}

const char *adaptiveLioModeReasonName(
  const livo::AdaptiveLioModeReason reason) noexcept
{
  switch (reason)
  {
    case livo::AdaptiveLioModeReason::kEligibleVisualSelected:
      return "eligible_visual_selected";
    case livo::AdaptiveLioModeReason::kFrontFallback:
      return "front_fallback";
    case livo::AdaptiveLioModeReason::
      kAllCamerasMissingAfterDeadline:
      return "all_cameras_missing_after_deadline";
    case livo::AdaptiveLioModeReason::kNoEligibleVisual:
      return "no_eligible_visual";
    case livo::AdaptiveLioModeReason::
      kVisualOutcomeNotObservedBeforeCloseout:
      return "visual_outcome_not_observed_before_closeout";
  }
  return "unknown";
}

const char *adaptiveLioModeTransitionName(
  const livo::AdaptiveLioModeTransition transition) noexcept
{
  switch (transition)
  {
    case livo::AdaptiveLioModeTransition::kNone:
      return "none";
    case livo::AdaptiveLioModeTransition::kEnterLioOnly:
      return "enter_lio_only";
    case livo::AdaptiveLioModeTransition::kExitLioOnly:
      return "exit_lio_only";
  }
  return "unknown";
}

const char *cameraPushResultName(const CameraObservationPushResult result)
{
  switch (result)
  {
    case CameraObservationPushResult::ACCEPTED:
      return "accepted";
    case CameraObservationPushResult::INVALID_CAMERA_ID:
      return "invalid_camera_id";
    case CameraObservationPushResult::INVALID_TIMESTAMP:
      return "invalid_timestamp";
    case CameraObservationPushResult::EMPTY_IMAGE:
      return "empty_image";
    case CameraObservationPushResult::ALREADY_PROCESSED:
      return "already_processed";
    case CameraObservationPushResult::OUT_OF_ORDER:
      return "out_of_order";
    case CameraObservationPushResult::RATE_LIMITED:
      return "rate_limited";
    case CameraObservationPushResult::CAPACITY_EXCEEDED:
      return "capacity_exceeded";
    case CameraObservationPushResult::AUTHORITY_EXHAUSTED:
      return "authority_exhausted";
  }
  return "unknown";
}

std::uint64_t sourceStampNanoseconds(
  const builtin_interfaces::msg::Time & stamp) noexcept
{
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000U)
  {
    return 0U;
  }
  return static_cast<std::uint64_t>(stamp.sec) * 1000000000ULL +
         static_cast<std::uint64_t>(stamp.nanosec);
}

std::uint64_t monotonicNanoseconds() noexcept
{
  timespec now{};
  if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0)
  {
    return 0U;
  }
  return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ULL +
         static_cast<std::uint64_t>(now.tv_nsec);
}

void emitCameraSupplyReceipt(
  const bool enabled,
  const char * stage,
  const std::uint8_t camera_id,
  const std::uint64_t source_stamp_ns,
  const double queue_stamp_sec,
  const char * result)
{
  if (!enabled)
  {
    return;
  }
  static std::mutex receipt_mutex;
  static std::uint64_t sequence = 0U;
  std::lock_guard<std::mutex> lock(receipt_mutex);
  const std::uint64_t receipt_sequence = ++sequence;
  livo::emitStructuredMetric(
    "[CAMERA_SUPPLY_RECEIPT] schema_version=1 "
    "sequence=%llu stage=%s camera_id=%u source_stamp_ns=%llu "
    "queue_stamp_sec=%.9f observed_monotonic_ns=%llu result=%s\n",
    static_cast<unsigned long long>(receipt_sequence), stage,
    static_cast<unsigned int>(camera_id),
    static_cast<unsigned long long>(source_stamp_ns),
    queue_stamp_sec,
    static_cast<unsigned long long>(monotonicNanoseconds()),
    result);
}

void emitCameraBatchSupplyOutcome(
  const bool enabled,
  const CameraObservationBatch & batch,
  const char * result)
{
  if (!enabled)
  {
    return;
  }
  for (std::size_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount; ++camera_id)
  {
    if (!batch.present[camera_id])
    {
      continue;
    }
    const CameraObservation & observation = batch.observations[camera_id];
    emitCameraSupplyReceipt(
      true, "batch_outcome", observation.camera_id,
      observation.source_stamp_ns, observation.timestamp, result);
  }
}

void emitDiscardedCameraSupplyOutcomes(
  const bool enabled,
  const std::vector<CameraObservation> & observations)
{
  if (!enabled)
  {
    return;
  }
  for (const CameraObservation & observation : observations)
  {
    emitCameraSupplyReceipt(
      true, "batch_outcome", observation.camera_id,
      observation.source_stamp_ns, observation.timestamp,
      "stale_rejected");
  }
}

void emitCameraTimeTransferMetric(
  const livo::CameraBatchTimeTransferBundle & bundle,
  const std::uint64_t lidar_feature_generation,
  const double anchor_timestamp_sec)
{
  std::array<const char *, CameraObservationQueue::kCameraCount>
    statuses{{"missing_frame", "missing_frame",
      "missing_frame", "missing_frame"}};
  std::array<double, CameraObservationQueue::kCameraCount>
    delta_time_ms{{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()}};
  std::array<double, CameraObservationQueue::kCameraCount>
    maximum_gap_ms = delta_time_ms;
  std::array<double, CameraObservationQueue::kCameraCount>
    transfer_covariance_trace = delta_time_ms;
  unsigned int present_mask = 0U;
  for (std::size_t camera_id = 0U;
       camera_id < CameraObservationQueue::kCameraCount; ++camera_id)
  {
    if (!bundle.present[camera_id]) {
      continue;
    }
    present_mask |= 1U << camera_id;
    const livo::CameraPoseTransferRaw & transfer =
      bundle.transfers[camera_id].data();
    statuses[camera_id] =
      livo::cameraTimeTransferStatusName(transfer.status);
    delta_time_ms[camera_id] =
      1000.0 * transfer.delta_time_sec;
    maximum_gap_ms[camera_id] =
      1000.0 * transfer.maximum_imu_gap_sec;
    if (transfer.pose_transfer_covariance.allFinite())
    {
      transfer_covariance_trace[camera_id] =
        transfer.pose_transfer_covariance.trace();
    }
  }

  livo::emitStructuredMetric(
    "[VIO_TIME_TRANSFER_SHADOW] generation=%llu "
    "anchor_ts=%.9f imu_epoch=%llu batch_present_mask=0x%02x "
    "metadata_valid=%d present=%zu valid=%zu "
    "camera0_id=0 camera0_dt_ms=%.6f camera0_status=%s "
    "camera0_gap_ms=%.6f camera0_q_trace=%.9g "
    "camera1_id=1 camera1_dt_ms=%.6f camera1_status=%s "
    "camera1_gap_ms=%.6f camera1_q_trace=%.9g "
    "camera2_id=2 camera2_dt_ms=%.6f camera2_status=%s "
    "camera2_gap_ms=%.6f camera2_q_trace=%.9g "
    "camera3_id=3 camera3_dt_ms=%.6f camera3_status=%s "
    "camera3_gap_ms=%.6f camera3_q_trace=%.9g applied=0\n",
    static_cast<unsigned long long>(lidar_feature_generation),
    anchor_timestamp_sec,
    static_cast<unsigned long long>(bundle.imu_epoch),
    present_mask,
    bundle.batch_metadata_valid ? 1 : 0,
    bundle.present_camera_count,
    bundle.valid_camera_count,
    delta_time_ms[0U], statuses[0U],
    maximum_gap_ms[0U], transfer_covariance_trace[0U],
    delta_time_ms[1U], statuses[1U],
    maximum_gap_ms[1U], transfer_covariance_trace[1U],
    delta_time_ms[2U], statuses[2U],
    maximum_gap_ms[2U], transfer_covariance_trace[2U],
    delta_time_ms[3U], statuses[3U],
    maximum_gap_ms[3U], transfer_covariance_trace[3U]);
}

template<typename ParameterT>
ParameterT requiredCameraParameter(
  const rclcpp::Node::SharedPtr &node, const std::string &parameter_namespace,
  const std::string &name)
{
  ParameterT value{};
  const std::string full_name = parameter_namespace + "." + name;
  if (!node->get_parameter(full_name, value))
  {
    throw std::runtime_error("Missing camera parameter: " + full_name);
  }
  return value;
}

template<typename ParameterT>
ParameterT optionalCameraParameter(
  const rclcpp::Node::SharedPtr &node, const std::string &parameter_namespace,
  const std::string &name, const ParameterT &default_value)
{
  ParameterT value{};
  if (node->get_parameter(parameter_namespace + "." + name, value))
  {
    return value;
  }
  return default_value;
}

livo::CameraModelParameters readCameraModelParameters(
  const rclcpp::Node::SharedPtr &node, const std::string &parameter_namespace)
{
  livo::CameraModelParameters p;
  p.model = requiredCameraParameter<std::string>(
    node, parameter_namespace, "model");
  if (p.model == "Ocam")
  {
    p.calib_file = optionalCameraParameter<std::string>(
      node, parameter_namespace, "calib_file", "");
    return p;
  }

  p.width = requiredCameraParameter<int>(node, parameter_namespace, "width");
  p.height = requiredCameraParameter<int>(node, parameter_namespace, "height");
  p.fx = requiredCameraParameter<double>(node, parameter_namespace, "fx");
  p.fy = requiredCameraParameter<double>(node, parameter_namespace, "fy");
  p.cx = requiredCameraParameter<double>(node, parameter_namespace, "cx");
  p.cy = requiredCameraParameter<double>(node, parameter_namespace, "cy");

  if (p.model == "Pinhole")
  {
    p.scale = optionalCameraParameter<double>(
      node, parameter_namespace, "scale", 1.0);
    p.d0 = optionalCameraParameter<double>(
      node, parameter_namespace, "d0", 0.0);
    p.d1 = optionalCameraParameter<double>(
      node, parameter_namespace, "d1", 0.0);
    p.d2 = optionalCameraParameter<double>(
      node, parameter_namespace, "d2", 0.0);
    p.d3 = optionalCameraParameter<double>(
      node, parameter_namespace, "d3", 0.0);
    p.d4 = optionalCameraParameter<double>(
      node, parameter_namespace, "d4", 0.0);
  }
  else if (p.model == "EquidistantCamera")
  {
    p.scale = optionalCameraParameter<double>(
      node, parameter_namespace, "scale", 1.0);
    p.k1 = optionalCameraParameter<double>(
      node, parameter_namespace, "k1", 0.0);
    p.k2 = optionalCameraParameter<double>(
      node, parameter_namespace, "k2", 0.0);
    p.k3 = optionalCameraParameter<double>(
      node, parameter_namespace, "k3", 0.0);
    p.k4 = optionalCameraParameter<double>(
      node, parameter_namespace, "k4", 0.0);
  }
  else if (p.model == "PolynomialCamera")
  {
    p.skew = requiredCameraParameter<double>(
      node, parameter_namespace, "skew");
    p.k2 = optionalCameraParameter<double>(
      node, parameter_namespace, "k2", 0.0);
    p.k3 = optionalCameraParameter<double>(
      node, parameter_namespace, "k3", 0.0);
    p.k4 = optionalCameraParameter<double>(
      node, parameter_namespace, "k4", 0.0);
    p.k5 = optionalCameraParameter<double>(
      node, parameter_namespace, "k5", 0.0);
    p.k6 = optionalCameraParameter<double>(
      node, parameter_namespace, "k6", 0.0);
    p.k7 = optionalCameraParameter<double>(
      node, parameter_namespace, "k7", 0.0);
  }
  else if (p.model == "ATAN")
  {
    p.d0 = requiredCameraParameter<double>(
      node, parameter_namespace, "d0");
  }
  return p;
}
}  // namespace

LIVMapper::LIVMapper(rclcpp::Node::SharedPtr &node, std::string node_name, const rclcpp::NodeOptions & options)
    : node(std::make_shared<rclcpp::Node>(node_name, options)),
      extT(0, 0, 0),
      extR(M3D::Identity())
{
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);
  cameraextrinT.assign(3, 0.0);
  cameraextrinR.assign(9, 0.0);
  
  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(this->node);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(this->node, voxel_config);
  if (local_fast_mode_)
  {
    if (!voxel_config.map_sliding_en || voxel_config.half_map_size > 30)
      throw std::runtime_error("local_fast requires a bounded sliding map (half_map_size <= 30)");
    voxel_config.is_pub_plane_map_ = false;
  }
#if !LIVO_BUILD_EXPERIMENTAL_SHADOWS
  const bool experimental_shadow_requested =
    cross_camera_migration_shadow_enable_ ||
    reference_only_migration_shadow_enable_ ||
    camera_time_transfer_shadow_enable_ ||
    shared_landmark_shadow_config_.enabled ||
    observation_shadow_config_.enabled ||
    shared_landmark_tombstone_shadow_enable_ ||
    shared_landmark_refit_shadow_enable_ ||
    multi_camera_landmark_linearization_shadow_config_.enabled ||
    multi_camera_landmark_linearization_shadow_config_.
      allow_relative_affine_experiment ||
    voxel_config.plane_lifecycle_shadow_enable_;
  if (experimental_shadow_requested)
  {
    throw std::runtime_error(
      "P3-P5 shadow/reference-only parameters require a build with "
      "-DLIVO_BUILD_EXPERIMENTAL_SHADOWS=ON");
  }
#endif
  const livo::SharedLandmarkLifecycleRouting
    shared_landmark_lifecycle_routing{
      shared_landmark_shadow_config_.enabled,
      shared_landmark_tombstone_shadow_enable_,
      shared_landmark_refit_shadow_enable_};
  voxel_config.plane_lifecycle_shadow_enable_ =
    voxel_config.plane_lifecycle_shadow_enable_ ||
    shared_landmark_lifecycle_routing.
    activePlaneLifecycleRequired();

  visual_sub_map.reset(new PointCloudXYZI());
  feats_undistort.reset(new PointCloudXYZI());
  feats_down_body.reset(new PointCloudXYZI());
  feats_down_world.reset(new PointCloudXYZI());
  pcl_w_wait_pub.reset(new PointCloudXYZI());
  pcl_wait_pub.reset(new PointCloudXYZI());
  persistent_lio_map_.reset(new PointCloudXYZI());
  persistent_rgb_map_.reset(new PointCloudXYZRGB());
  pcl_wait_save.reset(new PointCloudXYZRGB());
  pcl_wait_save_intensity.reset(new PointCloudXYZI());
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
  vio_manager.reset(new VIOManager());
  root_dir = ROOT_DIR;
  initializeFiles();
  initializeComponents(this->node);          // initialize components errors
  path.header.stamp = this->node->now();
  path.header.frame_id = "camera_init";
}

LIVMapper::~LIVMapper()
{
  finalizePendingAdaptiveLioMode();
  if (!img_en)
  {
    return;
  }
  std::size_t ingress_pending = 0U;
  {
    std::lock_guard<std::mutex> lock(camera_transport_mutex_);
    ingress_pending = camera_transport_buffer_.size();
  }
  std::printf(
    "[CAMERA_TRANSPORT_SUMMARY] processing_peak=%zu "
    "processing_pending=%zu ingress_pending=%zu\n",
    camera_transport_processing_peak_,
    camera_transport_processing_buffer_.size(),
    ingress_pending);
  for (std::size_t camera_id = 0;
       camera_id < CameraObservationQueue::kCameraCount; ++camera_id)
  {
    if (vio_camera_mode_ == "front" && camera_id != 0U)
    {
      continue;
    }
    std::printf(
      "[CAMERA_QUEUE_METRIC] camera_id=%zu accepted=%llu dropped=%llu\n",
      camera_id,
      static_cast<unsigned long long>(accepted_camera_frames_[camera_id]),
      static_cast<unsigned long long>(dropped_camera_frames_[camera_id]));
    std::printf(
      "[CAMERA_TRANSPORT_METRIC] camera_id=%zu received=%llu dropped=%llu\n",
      camera_id,
      static_cast<unsigned long long>(
        camera_transport_received_[camera_id]),
      static_cast<unsigned long long>(
        camera_transport_dropped_[camera_id]));
  }
}

const livo::AdaptiveVisualLidarRuntimeInitializationAudit &
LIVMapper::adaptiveVisualLidarRuntimeInitializationAudit()
  const noexcept
{
  return adaptive_visual_lidar_runtime_initialization_audit_;
}

const livo::AdaptiveVisualLidarRuntimeDispatchAudit &
LIVMapper::adaptiveVisualLidarRuntimeDispatchAudit()
  const noexcept
{
  return adaptive_visual_lidar_runtime_dispatch_audit_;
}

const livo::AdaptiveLioModePublicationAudit &
LIVMapper::adaptiveLioModeAudit() const noexcept
{
  return adaptive_lio_mode_publication_audit_;
}

void LIVMapper::recordAdaptiveLioMode(
  const livo::AdaptiveLioModeObservation &
  observation) noexcept
{
  const livo::AdaptiveLioModeRecordResult result =
    adaptive_lio_mode_ledger_.record(observation);
  adaptive_lio_mode_publication_audit_ = {
    result.status,
    result.receipt,
    false};
  if (!result.receipt.has_value())
  {
    return;
  }

  const livo::AdaptiveLioModeReceipt & receipt =
    *result.receipt;
  try
  {
    adaptive_lio_mode_publication_audit_.metric_emitted =
      livo::emitStructuredMetric(
      "[ADAPTIVE_VIO_MODE_METRIC] schema_version=1 "
      "generation=%llu source_stamp_sec=%.9f "
      "mode=%s transition=%s reason=%s\n",
      static_cast<unsigned long long>(receipt.generation()),
      receipt.sourceStampSec(),
      adaptiveLioModeName(receipt.mode()),
      adaptiveLioModeTransitionName(receipt.transition()),
      adaptiveLioModeReasonName(receipt.reason()));
  }
  catch (...)
  {
    adaptive_lio_mode_publication_audit_.metric_emitted = false;
  }
}

void LIVMapper::finalizePendingAdaptiveLioMode() noexcept
{
  if (!pending_adaptive_lio_mode_observation_.has_value())
  {
    return;
  }
  const livo::AdaptiveLioModeObservation observation =
    *pending_adaptive_lio_mode_observation_;
  pending_adaptive_lio_mode_observation_.reset();
  recordAdaptiveLioMode(observation);
}

void LIVMapper::finalizeAdaptiveVisualOutcome(
  const livo::AdaptiveLioMode mode,
  const livo::AdaptiveLioModeReason reason) noexcept
{
  livo::AdaptiveLioModeObservation observation{
    lio_feature_generation_,
    LidarMeasures.last_lio_update_time,
    mode,
    reason};
  if (
    pending_adaptive_lio_mode_observation_.has_value() &&
    pending_adaptive_lio_mode_observation_->generation ==
      lio_feature_generation_)
  {
    observation.generation =
      pending_adaptive_lio_mode_observation_->generation;
    observation.source_stamp_sec =
      pending_adaptive_lio_mode_observation_->source_stamp_sec;
    pending_adaptive_lio_mode_observation_.reset();
  }
  recordAdaptiveLioMode(observation);
}

bool LIVMapper::adaptiveSingleCameraRuntimeEnabled()
  const noexcept
{
  return
    (vio_camera_mode_ == "front" || vio_camera_mode_ == "four") &&
    adaptive_policy_config_.enabled &&
    adaptive_policy_config_.minimum_selected_cameras == 1U &&
    adaptive_policy_config_.maximum_selected_cameras == 1U;
}

bool LIVMapper::dispatchAdaptiveSingleCameraVio()
{
  adaptive_visual_lidar_runtime_dispatch_audit_ = {};
  if (_pv_list.empty())
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kNoCachedLidarFeatures;
    std::cout << "[ VIO ] No cached LiDAR features!!!"
              << std::endl;
    return false;
  }
  if (LidarMeasures.measures.empty())
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kRuntimeInvariantViolation;
    return false;
  }

  CameraObservationBatch & scheduled_batch =
    LidarMeasures.measures.back().camera_batch;
  vio_manager->plot_flag = shouldPlotVioFrame(
    LidarMeasures.last_lio_update_time - _first_lidar_time,
    plot_time, frame_cnt);

  const std::uint64_t lidar_generation =
    lio_feature_generation_;
  const livo::AdaptiveVisualPolicy::Matrix6
    lidar_pose_information =
    voxelmap_manager->pose_information_;
  const bool lidar_information_ready =
    voxelmap_manager->pose_information_valid_;
  adaptive_visual_lidar_runtime_dispatch_audit_.
    lidar_posterior =
    livo::AdaptiveVisualLidarPosteriorSnapshot{
      lidar_generation,
      lidar_pose_information,
      lidar_information_ready};

  if (!adaptive_visual_selection_authority_)
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kAuthorityUnavailable;
    return false;
  }

  livo::AuthoritativeLidarGenerationIssueResult
    lidar_issue;
  try
  {
    lidar_issue =
      adaptive_visual_selection_authority_->
      issueLidarGeneration(
      lidar_generation, lidar_pose_information,
      lidar_information_ready);
  }
  catch (const std::bad_alloc &)
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kResourceFailure;
    return false;
  }
  adaptive_visual_lidar_runtime_dispatch_audit_.
    selection_status = lidar_issue.status;
  if (lidar_issue.status !=
      livo::AdaptiveVisualSelectionStatus::kReady)
  {
    if (
      lidar_issue.status ==
      livo::AdaptiveVisualSelectionStatus::
      kZeroLidarGeneration)
    {
      adaptive_visual_lidar_runtime_dispatch_audit_.status =
        livo::AdaptiveVisualLidarRuntimeDispatchStatus::
        kZeroLidarGeneration;
    }
    else if (
      lidar_issue.status ==
      livo::AdaptiveVisualSelectionStatus::
      kInvalidLidarInformation)
    {
      adaptive_visual_lidar_runtime_dispatch_audit_.status =
        livo::AdaptiveVisualLidarRuntimeDispatchStatus::
        kNonFiniteLidarInformation;
    }
    else
    {
      adaptive_visual_lidar_runtime_dispatch_audit_.status =
        livo::AdaptiveVisualLidarRuntimeDispatchStatus::
        kIssueRejected;
    }
    return false;
  }
  if (!lidar_issue.token.has_value())
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kRuntimeInvariantViolation;
    return false;
  }
  adaptive_visual_lidar_runtime_dispatch_audit_.
    generation_issued = true;

  const auto runtime_input =
    adaptive_visual_selection_authority_->bindRuntimeInput(
    *lidar_issue.token);
  if (!runtime_input.has_value())
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kRuntimeInvariantViolation;
    return false;
  }
  adaptive_visual_lidar_runtime_dispatch_audit_.
    runtime_bound = true;
  const livo::AdaptiveVisualRuntimeAuthorityInput &
    bound_runtime_input = *runtime_input;
  adaptive_visual_lidar_runtime_dispatch_audit_.
    vio_dispatch_started = true;
  try
  {
    vio_manager->processFrames(
      scheduled_batch, _pv_list,
      voxelmap_manager->voxel_map_,
      LidarMeasures.last_lio_update_time - _first_lidar_time,
      bound_runtime_input.lidarPoseInformation(),
      bound_runtime_input.lidarInformationReady(),
      bound_runtime_input.lidarGeneration(),
      nullptr, nullptr, nullptr, &bound_runtime_input);
  }
  catch (const std::bad_alloc &)
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kResourceFailure;
    throw;
  }
  catch (...)
  {
    adaptive_visual_lidar_runtime_dispatch_audit_.status =
      livo::AdaptiveVisualLidarRuntimeDispatchStatus::
      kUnexpectedFailure;
    throw;
  }
  adaptive_visual_lidar_runtime_dispatch_audit_.
    vio_dispatch_completed = true;
  adaptive_visual_lidar_runtime_dispatch_audit_.status =
    livo::AdaptiveVisualLidarRuntimeDispatchStatus::
    kVioDispatched;
  return adaptive_visual_lidar_runtime_dispatch_audit_.
    vio_dispatch_completed;
}

void LIVMapper::readParameters(rclcpp::Node::SharedPtr &node)
{
  if (!camera_calibration_provenance_captured_)
  {
    const auto & initial_parameter_overrides =
      node->get_node_parameters_interface()->get_parameter_overrides();
    auto has_explicit_parameter_source =
      [&node, &initial_parameter_overrides](const std::string & name)
      {
        return
          node->has_parameter(name) ||
          initial_parameter_overrides.find(name) !=
          initial_parameter_overrides.end();
      };
    constexpr std::array<const char *, 12U>
      kRequiredProjectionCalibrationFields{{
        "model", "width", "height", "scale",
        "fx", "fy", "cx", "cy",
        "d0", "d1", "d2", "d3",
      }};
    constexpr std::array<const char *, 4U> kCameraNames{
      {"front", "left", "back", "right"}};
    for (std::size_t camera_id = 0U;
         camera_id < kCameraNames.size();
         ++camera_id)
    {
      const std::string camera_name = kCameraNames[camera_id];
      const std::string projection_prefix =
        "multi_camera.cameras." + camera_name + ".";
      bool projection_calibration_present = true;
      for (const char * field : kRequiredProjectionCalibrationFields)
      {
        projection_calibration_present =
          projection_calibration_present &&
          has_explicit_parameter_source(projection_prefix + field);
      }
      multi_camera_projection_calibration_present_[camera_id] =
        projection_calibration_present;

      const std::string extrinsic_prefix =
        "multi_camera.vio_extrinsics." + camera_name + ".";
      multi_camera_extrinsic_calibration_present_[camera_id] =
        has_explicit_parameter_source(extrinsic_prefix + "R") &&
        has_explicit_parameter_source(extrinsic_prefix + "T");
    }
    camera_calibration_provenance_captured_ = true;
  }

  // declare parameters
  auto try_declare = [node]<typename ParameterT>(const std::string & name,
    const ParameterT & default_value)
  {
    if (!node->has_parameter(name))
    {
      return node->declare_parameter<ParameterT>(name, default_value);
    }
    else
    {
      return node->get_parameter(name).get_value<ParameterT>();
    }
  };

  // declare parameter
  const std::int64_t imu_algorithm_capacity =
    try_declare.template operator()<std::int64_t>(
      "buffer.imu_algorithm_capacity", 4096);
  if (imu_algorithm_capacity < 1 || imu_algorithm_capacity > 65536)
  {
    throw std::invalid_argument(
      "buffer.imu_algorithm_capacity must be an integer in [1, 65536]");
  }
  imu_algorithm_capacity_ = static_cast<std::size_t>(imu_algorithm_capacity);
  imu_overflow_fatal_ = try_declare.template operator()<bool>(
    "buffer.imu_overflow_fatal", false);
  try_declare.template operator()<std::string>("common.lid_topic", "/livox/lidar");
  try_declare.template operator()<std::string>("common.imu_topic", "/livox/imu");
  try_declare.template operator()<bool>("common.ros_driver_bug_fix", false);
  try_declare.template operator()<int>("common.img_en", 1);
  try_declare.template operator()<int>("common.lidar_en", 1);
  try_declare.template operator()<std::string>("common.img_topic", "/left_camera/image");
  try_declare.template operator()<std::string>("vio.camera_mode", "front");
  try_declare.template operator()<int>("vio.adaptive.camera_threads", 1);
  try_declare.template operator()<double>("vio.camera_reorder_window_ms", 20.0);
  try_declare.template operator()<double>("vio.camera_sync_tolerance_ms", 20.0);
  try_declare.template operator()<bool>(
    "vio.camera_supply_receipt_enable", false);
  try_declare.template operator()<bool>(
    "vio.adaptive.partial_batch.enable", false);
  try_declare.template operator()<double>(
    "vio.adaptive.partial_batch.missing_camera_timeout_ms", 30.0);
  try_declare.template operator()<bool>(
    "vio.multiview.cross_camera_migration_shadow_enable", false);
  try_declare.template operator()<bool>(
    "vio.multiview.reference_only_migration_shadow.enable", false);
  try_declare.template operator()<bool>(
    "vio.multiview.time_transfer.shadow_enable", false);
  try_declare.template operator()<double>(
    "vio.multiview.time_transfer.max_abs_transfer_ms", 50.0);
  try_declare.template operator()<double>(
    "vio.multiview.time_transfer.max_imu_gap_ms", 20.0);
  try_declare.template operator()<double>(
    "vio.multiview.time_transfer.imu_timestamp_stddev_ms", 0.0);
  try_declare.template operator()<double>(
    "vio.multiview.time_transfer.ring_buffer_ms", 250.0);
  try_declare.template operator()<double>(
    "vio.multiview.time_transfer.epoch_jump_ms", 100.0);
  try_declare.template operator()<double>(
    "vio.multiview.time_transfer.gyro_noise_stddev_radps", 0.01);
  try_declare.template operator()<double>(
    "vio.multiview.time_transfer.accel_noise_stddev_mps2", 0.10);
  try_declare.template operator()<bool>(
    "vio.multiview.shared_landmarks.shadow_enable", false);
  try_declare.template operator()<double>(
    "vio.multiview.shared_landmarks.plane_cell_size_m", 0.25);
  try_declare.template operator()<int>(
    "vio.multiview.shared_landmarks.maximum_seeds_per_batch", 512);
  try_declare.template operator()<bool>(
    "vio.multiview.shared_landmarks.observation_shadow_enable", false);
  try_declare.template operator()<bool>(
    "vio.multiview.shared_landmarks.tombstone_shadow_enable", false);
  try_declare.template operator()<bool>(
    "vio.multiview.shared_landmarks.refit_shadow_enable", false);
  try_declare.template operator()<int>(
    "vio.multiview.shared_landmarks.maximum_observations_per_batch", 512);
  try_declare.template operator()<int>(
    "vio.multiview.shared_landmarks.maximum_patch_levels", 4);
  try_declare.template operator()<int>(
    "vio.multiview.shared_landmarks.maximum_patch_values_per_observation",
    1024);
  try_declare.template operator()<bool>(
    "vio.multiview.shared_landmarks.linearization_shadow.enable", false);
  try_declare.template operator()<bool>(
    "vio.multiview.shared_landmarks.linearization_shadow.allow_relative_affine_experiment", false);
  try_declare.template operator()<double>(
    "vio.multiview.shared_landmarks.linearization_shadow.maximum_source_age_ms", 500.0);
  try_declare.template operator()<int>(
    "vio.multiview.shared_landmarks.linearization_shadow.maximum_candidate_evaluations_per_batch", 16384);
  try_declare.template operator()<double>("vio.joint_warmup_sec", 1.0);
  try_declare.template operator()<int>("vio.camera_queue_capacity", 768);
  try_declare.template operator()<int>("vio.min_tracked_points_per_camera", 4);
  try_declare.template operator()<int>("vio.min_joint_tracked_points", 12);
  try_declare.template operator()<int>("vio.min_joint_cameras", 2);
  try_declare.template operator()<double>("vio.max_residual_rmse", 20.0);
  try_declare.template operator()<double>("vio.max_rotation_increment_deg", 0.1);
  try_declare.template operator()<double>("vio.max_translation_increment_m", 0.01);
  try_declare.template operator()<bool>("vio.adaptive.enable", false);
  try_declare.template operator()<bool>(
    "vio.adaptive.metrics.enable", false);
  try_declare.template operator()<double>(
    "vio.adaptive.rotation_scale_rad", 0.017453292519943295);
  try_declare.template operator()<double>(
    "vio.adaptive.translation_scale_m", 0.10);
  try_declare.template operator()<double>(
    "vio.adaptive.weak_relative_eigenvalue", 0.05);
  try_declare.template operator()<double>(
    "vio.adaptive.weak_absolute_eigenvalue", 1.0e-9);
  try_declare.template operator()<double>(
    "vio.adaptive.health_off_threshold", 0.25);
  try_declare.template operator()<double>(
    "vio.adaptive.health_on_threshold", 0.45);
  try_declare.template operator()<double>(
    "vio.adaptive.hard_reject_health", 0.08);
  try_declare.template operator()<int>(
    "vio.adaptive.off_hysteresis_frames", 3);
  try_declare.template operator()<int>(
    "vio.adaptive.on_hysteresis_frames", 5);
  try_declare.template operator()<double>(
    "vio.adaptive.minimum_probation_duration_sec", 0.5);
  try_declare.template operator()<double>(
    "vio.adaptive.covariance_health_floor", 0.10);
  try_declare.template operator()<double>(
    "vio.adaptive.maximum_covariance_scale", 10.0);
  try_declare.template operator()<double>(
    "vio.adaptive.complement_score_gain", 2.0);
  try_declare.template operator()<double>(
    "vio.adaptive.selection_switch_margin", 0.10);
  try_declare.template operator()<int>(
    "vio.adaptive.selection_challenger_frames", 3);
  try_declare.template operator()<int>(
    "vio.adaptive.minimum_selection_tracked_points", 1);
  try_declare.template operator()<int>(
    "vio.adaptive.healthy_tracked_points", 24);
  try_declare.template operator()<double>(
    "vio.adaptive.healthy_tracking_ratio", 0.50);
  try_declare.template operator()<double>(
    "vio.adaptive.healthy_residual_rmse", 10.0);
  try_declare.template operator()<double>(
    "vio.adaptive.maximum_saturation_fraction", 0.25);
  try_declare.template operator()<double>(
    "vio.adaptive.maximum_dark_fraction", 0.35);
  try_declare.template operator()<double>(
    "vio.adaptive.healthy_mean_gradient", 12.0);
  try_declare.template operator()<double>(
    "vio.adaptive.healthy_spatial_coverage", 0.20);
  try_declare.template operator()<int>(
    "vio.adaptive.total_candidate_budget", 240);
  try_declare.template operator()<int>(
    "vio.adaptive.minimum_candidate_budget", 16);
  try_declare.template operator()<int>(
    "vio.adaptive.total_patch_budget", 120);
  try_declare.template operator()<int>(
    "vio.adaptive.minimum_patch_budget", 8);
  try_declare.template operator()<int>(
    "vio.adaptive.minimum_selected_cameras", 2);
  try_declare.template operator()<int>(
    "vio.adaptive.maximum_selected_cameras", 4);
  try_declare.template operator()<bool>("multi_camera.enable", false);
  try_declare.template operator()<std::string>("multi_camera.topics.front", "/camera/color/image_raw");
  try_declare.template operator()<std::string>("multi_camera.topics.left", "/camera_left/color/image_raw");
  try_declare.template operator()<std::string>("multi_camera.topics.back", "/camera_back/color/image_raw");
  try_declare.template operator()<std::string>("multi_camera.topics.right", "/camera_right/color/image_raw");
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.front.R", vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.front.T", vector<double>{0, 0, 0});
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.left.R", vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.left.T", vector<double>{0, 0, 0});
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.back.R", vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.back.T", vector<double>{0, 0, 0});
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.right.R", vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
  try_declare.template operator()<vector<double>>("multi_camera.extrinsics.right.T", vector<double>{0, 0, 0});

  const std::array<std::string, CameraObservationQueue::kCameraCount> camera_names =
    {{"front", "left", "back", "right"}};
  for (const std::string &camera_name : camera_names)
  {
    const std::string camera_ns = "multi_camera.cameras." + camera_name + ".";
    try_declare.template operator()<std::string>(camera_ns + "model", "Pinhole");
    try_declare.template operator()<int>(camera_ns + "width", 640);
    try_declare.template operator()<int>(camera_ns + "height", 480);
    try_declare.template operator()<double>(camera_ns + "scale", 1.0);
    try_declare.template operator()<double>(camera_ns + "fx", 600.0);
    try_declare.template operator()<double>(camera_ns + "fy", 600.0);
    try_declare.template operator()<double>(camera_ns + "cx", 320.0);
    try_declare.template operator()<double>(camera_ns + "cy", 240.0);
    try_declare.template operator()<double>(camera_ns + "d0", 0.0);
    try_declare.template operator()<double>(camera_ns + "d1", 0.0);
    try_declare.template operator()<double>(camera_ns + "d2", 0.0);
    try_declare.template operator()<double>(camera_ns + "d3", 0.0);
    try_declare.template operator()<vector<double>>(
      "multi_camera.vio_extrinsics." + camera_name + ".R",
      vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
    try_declare.template operator()<vector<double>>(
      "multi_camera.vio_extrinsics." + camera_name + ".T",
      vector<double>{0, 0, 0});
    try_declare.template operator()<double>(
      "multi_camera.time_offsets." + camera_name, 0.0);
  }

  try_declare.template operator()<bool>("vio.normal_en", true);
  try_declare.template operator()<bool>("vio.inverse_composition_en", false);
  try_declare.template operator()<int>("vio.max_iterations", 5);
  try_declare.template operator()<int>("vio.img_point_cov", 100);
  try_declare.template operator()<int>("vio.joint_img_point_cov", 1000);
  try_declare.template operator()<bool>("vio.raycast_en", false);
  try_declare.template operator()<bool>("vio.exposure_estimate_en", true);
  try_declare.template operator()<double>("vio.inv_expo_cov", 0.1);
  try_declare.template operator()<int>("vio.grid_size", 5);
  try_declare.template operator()<int>("vio.grid_n_height", 17);
  try_declare.template operator()<int>("vio.patch_pyrimid_level", 4);
  try_declare.template operator()<int>("vio.patch_size", 8);
  try_declare.template operator()<int>("vio.outlier_threshold", 100);
  try_declare.template operator()<double>("time_offset.exposure_time_init", 0.0);
  try_declare.template operator()<double>("time_offset.img_time_offset", 0.0);
  try_declare.template operator()<bool>("uav.imu_rate_odom", false);
  try_declare.template operator()<bool>("uav.gravity_align_en", false);

  try_declare.template operator()<std::string>("evo.seq_name", "01");
  try_declare.template operator()<bool>("evo.pose_output_en", false);
  try_declare.template operator()<double>("imu.gyr_cov", 1.0);
  try_declare.template operator()<double>("imu.acc_cov", 1.0);
  try_declare.template operator()<int>("imu.imu_int_frame", 30);
  try_declare.template operator()<bool>("imu.imu_en", true);
  try_declare.template operator()<bool>("imu.gravity_est_en", true);
  try_declare.template operator()<bool>("imu.ba_bg_est_en", true);

  try_declare.template operator()<double>("preprocess.blind", 0.01);
  try_declare.template operator()<double>("preprocess.filter_size_surf", 0.5);
  try_declare.template operator()<int>("preprocess.lidar_type", AVIA);
  try_declare.template operator()<int>("preprocess.scan_line",6);
  try_declare.template operator()<int>("preprocess.point_filter_num", 3);
  try_declare.template operator()<int>("preprocess.scan_rate", 10);
  try_declare.template operator()<bool>("preprocess.feature_extract_enabled", false);

  try_declare.template operator()<int>("pcd_save.interval", -1);
  try_declare.template operator()<bool>("pcd_save.pcd_save_en", false);
  try_declare.template operator()<bool>("pcd_save.colmap_output_en", false);
  try_declare.template operator()<double>("pcd_save.filter_size_pcd", 0.5);
  try_declare.template operator()<vector<double>>("extrin_calib.extrinsic_T", vector<double>{});
  try_declare.template operator()<vector<double>>("extrin_calib.extrinsic_R", vector<double>{});
  try_declare.template operator()<vector<double>>("extrin_calib.Pcl", vector<double>{});
  try_declare.template operator()<vector<double>>("extrin_calib.Rcl", vector<double>{});
  try_declare.template operator()<double>("debug.plot_time", -10);
  try_declare.template operator()<int>("debug.frame_cnt", 6);
  try_declare.template operator()<bool>(
    "debug.determinism_audit_enable", false);

  try_declare.template operator()<double>("publish.blind_rgb_points", 0.01);
  try_declare.template operator()<int>("publish.pub_scan_num", 1);
  try_declare.template operator()<std::string>("publish.rgb_topic", "/cloud_registered_rgb");
  try_declare.template operator()<bool>("publish.persistent_map_en", false);
  try_declare.template operator()<std::string>("publish.map_topic", "/cloud_registered_map");
  try_declare.template operator()<std::string>("publish.rgb_map_topic", "/cloud_registered_rgb_map");
  try_declare.template operator()<int>("publish.map_publish_interval", 10);
  try_declare.template operator()<double>("publish.map_voxel_size", 0.1);
  try_declare.template operator()<bool>("publish.pub_effect_point_en", false);
  try_declare.template operator()<bool>("publish.dense_map_en", false);

  // get parameter
  this->node->get_parameter("common.lid_topic", lid_topic);
  this->node->get_parameter("common.imu_topic", imu_topic);
  this->node->get_parameter("common.ros_driver_bug_fix", ros_driver_fix_en);
  this->node->get_parameter("common.img_en", img_en);
  this->node->get_parameter("common.lidar_en", lidar_en);
  this->node->get_parameter("common.img_topic", img_topic);
  this->node->get_parameter("vio.camera_mode", vio_camera_mode_);
  this->node->get_parameter(
    "vio.camera_supply_receipt_enable",
    camera_supply_receipt_enable_);
  this->node->get_parameter(
    "debug.determinism_audit_enable",
    determinism_audit_enable_);
  double camera_reorder_window_ms = 20.0;
  this->node->get_parameter("vio.camera_reorder_window_ms", camera_reorder_window_ms);
  camera_reorder_window_sec_ = std::max(0.0, camera_reorder_window_ms / 1000.0);
  double camera_sync_tolerance_ms = 20.0;
  this->node->get_parameter(
    "vio.camera_sync_tolerance_ms", camera_sync_tolerance_ms);
  camera_sync_tolerance_sec_ =
    std::max(0.0, camera_sync_tolerance_ms / 1000.0);
  this->node->get_parameter(
    "vio.adaptive.partial_batch.enable",
    adaptive_partial_batch_enable_);
  double missing_camera_timeout_ms = 30.0;
  this->node->get_parameter(
    "vio.adaptive.partial_batch.missing_camera_timeout_ms",
    missing_camera_timeout_ms);
  adaptive_missing_camera_timeout_sec_ =
    missing_camera_timeout_ms / 1000.0;
  this->node->get_parameter(
    "vio.multiview.cross_camera_migration_shadow_enable",
    cross_camera_migration_shadow_enable_);
  this->node->get_parameter(
    "vio.multiview.reference_only_migration_shadow.enable",
    reference_only_migration_shadow_enable_);
  this->node->get_parameter(
    "vio.multiview.time_transfer.shadow_enable",
    camera_time_transfer_shadow_enable_);
  double max_abs_transfer_ms = 50.0;
  double max_imu_gap_ms = 20.0;
  double imu_timestamp_stddev_ms = 0.0;
  double ring_buffer_ms = 250.0;
  double epoch_jump_ms = 100.0;
  this->node->get_parameter(
    "vio.multiview.time_transfer.max_abs_transfer_ms",
    max_abs_transfer_ms);
  this->node->get_parameter(
    "vio.multiview.time_transfer.max_imu_gap_ms",
    max_imu_gap_ms);
  this->node->get_parameter(
    "vio.multiview.time_transfer.imu_timestamp_stddev_ms",
    imu_timestamp_stddev_ms);
  this->node->get_parameter(
    "vio.multiview.time_transfer.ring_buffer_ms",
    ring_buffer_ms);
  this->node->get_parameter(
    "vio.multiview.time_transfer.epoch_jump_ms",
    epoch_jump_ms);
  this->node->get_parameter(
    "vio.multiview.time_transfer.gyro_noise_stddev_radps",
    camera_time_transfer_config_.gyro_noise_stddev_radps);
  this->node->get_parameter(
    "vio.multiview.time_transfer.accel_noise_stddev_mps2",
    camera_time_transfer_config_.accel_noise_stddev_mps2);
  camera_time_transfer_config_.max_abs_transfer_sec =
    max_abs_transfer_ms / 1000.0;
  camera_time_transfer_config_.max_imu_gap_sec =
    max_imu_gap_ms / 1000.0;
  camera_time_transfer_config_.imu_timestamp_stddev_sec =
    imu_timestamp_stddev_ms / 1000.0;
  camera_time_transfer_config_.ring_buffer_duration_sec =
    ring_buffer_ms / 1000.0;
  camera_time_transfer_config_.epoch_jump_threshold_sec =
    epoch_jump_ms / 1000.0;
  camera_time_transfer_ =
    livo::CameraTimeTransfer(camera_time_transfer_config_);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.shadow_enable",
    shared_landmark_shadow_config_.enabled);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.plane_cell_size_m",
    shared_landmark_shadow_config_.plane_cell_size_m);
  int maximum_shadow_seeds_per_batch = 512;
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.maximum_seeds_per_batch",
    maximum_shadow_seeds_per_batch);
  shared_landmark_shadow_config_.maximum_seeds_per_batch =
    maximum_shadow_seeds_per_batch > 0 ?
    static_cast<std::size_t>(maximum_shadow_seeds_per_batch) :
    0U;
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.observation_shadow_enable",
    observation_shadow_config_.enabled);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.tombstone_shadow_enable",
    shared_landmark_tombstone_shadow_enable_);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.refit_shadow_enable",
    shared_landmark_refit_shadow_enable_);
  int maximum_shadow_observations_per_batch = 512;
  int maximum_shadow_patch_levels = 4;
  int maximum_shadow_patch_values = 1024;
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.maximum_observations_per_batch",
    maximum_shadow_observations_per_batch);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.maximum_patch_levels",
    maximum_shadow_patch_levels);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.maximum_patch_values_per_observation",
    maximum_shadow_patch_values);
  observation_shadow_config_.maximum_observations_per_batch =
    maximum_shadow_observations_per_batch > 0 ?
    static_cast<std::size_t>(
      maximum_shadow_observations_per_batch) : 0U;
  observation_shadow_config_.maximum_patch_levels =
    maximum_shadow_patch_levels > 0 ?
    static_cast<std::size_t>(maximum_shadow_patch_levels) : 0U;
  observation_shadow_config_.
    maximum_patch_values_per_observation =
    maximum_shadow_patch_values > 0 ?
    static_cast<std::size_t>(maximum_shadow_patch_values) : 0U;
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.linearization_shadow.enable",
    multi_camera_landmark_linearization_shadow_config_.enabled);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.linearization_shadow.allow_relative_affine_experiment",
    multi_camera_landmark_linearization_shadow_config_.
      allow_relative_affine_experiment);
  double maximum_linearization_source_age_ms = 500.0;
  int maximum_linearization_candidate_evaluations = 16384;
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.linearization_shadow.maximum_source_age_ms",
    maximum_linearization_source_age_ms);
  this->node->get_parameter(
    "vio.multiview.shared_landmarks.linearization_shadow.maximum_candidate_evaluations_per_batch",
    maximum_linearization_candidate_evaluations);
  multi_camera_landmark_linearization_shadow_config_.
    maximum_source_age_sec =
    maximum_linearization_source_age_ms / 1000.0;
  multi_camera_landmark_linearization_shadow_config_.
    maximum_candidate_evaluations_per_batch =
    maximum_linearization_candidate_evaluations > 0 ?
    static_cast<std::size_t>(
      maximum_linearization_candidate_evaluations) : 0U;
  this->node->get_parameter(
    "vio.joint_warmup_sec", joint_vio_warmup_sec_);
  this->node->get_parameter("vio.camera_queue_capacity", camera_queue_capacity_);
  camera_queue_capacity_ = std::max(1, camera_queue_capacity_);
  this->node->get_parameter(
    "vio.min_tracked_points_per_camera",
    min_tracked_points_per_camera_);
  this->node->get_parameter(
    "vio.min_joint_tracked_points", min_joint_tracked_points_);
  this->node->get_parameter("vio.min_joint_cameras", min_joint_cameras_);
  this->node->get_parameter(
    "vio.max_residual_rmse", max_visual_residual_rmse_);
  this->node->get_parameter(
    "vio.max_rotation_increment_deg",
    max_visual_rotation_increment_deg_);
  this->node->get_parameter(
    "vio.max_translation_increment_m",
    max_visual_translation_increment_m_);
  this->node->get_parameter(
    "vio.adaptive.enable", adaptive_policy_config_.enabled);
  bool structured_metrics_enabled = false;
  this->node->get_parameter(
    "vio.adaptive.metrics.enable", structured_metrics_enabled);
  livo::setStructuredMetricEnabled(structured_metrics_enabled);
  this->node->get_parameter(
    "vio.adaptive.rotation_scale_rad",
    adaptive_policy_config_.rotation_scale_rad);
  this->node->get_parameter(
    "vio.adaptive.translation_scale_m",
    adaptive_policy_config_.translation_scale_m);
  this->node->get_parameter(
    "vio.adaptive.weak_relative_eigenvalue",
    adaptive_policy_config_.weak_relative_eigenvalue);
  this->node->get_parameter(
    "vio.adaptive.weak_absolute_eigenvalue",
    adaptive_policy_config_.weak_absolute_eigenvalue);
  this->node->get_parameter(
    "vio.adaptive.health_off_threshold",
    adaptive_policy_config_.health_off_threshold);
  this->node->get_parameter(
    "vio.adaptive.health_on_threshold",
    adaptive_policy_config_.health_on_threshold);
  this->node->get_parameter(
    "vio.adaptive.hard_reject_health",
    adaptive_policy_config_.hard_reject_health);
  this->node->get_parameter(
    "vio.adaptive.minimum_probation_duration_sec",
    adaptive_policy_config_.minimum_probation_duration_sec);
  this->node->get_parameter(
    "vio.adaptive.covariance_health_floor",
    adaptive_policy_config_.covariance_health_floor);
  this->node->get_parameter(
    "vio.adaptive.maximum_covariance_scale",
    adaptive_policy_config_.maximum_covariance_scale);
  this->node->get_parameter(
    "vio.adaptive.complement_score_gain",
    adaptive_policy_config_.complement_score_gain);
  this->node->get_parameter(
    "vio.adaptive.selection_switch_margin",
    adaptive_policy_config_.selection_switch_margin);
  this->node->get_parameter(
    "vio.adaptive.healthy_tracking_ratio",
    adaptive_policy_config_.healthy_tracking_ratio);
  this->node->get_parameter(
    "vio.adaptive.healthy_residual_rmse",
    adaptive_policy_config_.healthy_residual_rmse);
  this->node->get_parameter(
    "vio.adaptive.maximum_saturation_fraction",
    adaptive_policy_config_.maximum_saturation_fraction);
  this->node->get_parameter(
    "vio.adaptive.maximum_dark_fraction",
    adaptive_policy_config_.maximum_dark_fraction);
  this->node->get_parameter(
    "vio.adaptive.healthy_mean_gradient",
    adaptive_policy_config_.healthy_mean_gradient);
  this->node->get_parameter(
    "vio.adaptive.healthy_spatial_coverage",
    adaptive_policy_config_.healthy_spatial_coverage);
  const auto read_positive_size =
    [this](const std::string & parameter_name) -> std::size_t
    {
      int value = 0;
      this->node->get_parameter(parameter_name, value);
      if (value <= 0)
      {
        throw std::runtime_error(
          parameter_name + " must be a positive integer");
      }
      return static_cast<std::size_t>(value);
    };
  adaptive_policy_config_.off_hysteresis_frames =
    read_positive_size("vio.adaptive.off_hysteresis_frames");
  adaptive_policy_config_.on_hysteresis_frames =
    read_positive_size("vio.adaptive.on_hysteresis_frames");
  adaptive_policy_config_.selection_challenger_frames =
    read_positive_size("vio.adaptive.selection_challenger_frames");
  adaptive_policy_config_.minimum_selection_tracked_points =
    read_positive_size("vio.adaptive.minimum_selection_tracked_points");
  adaptive_policy_config_.healthy_tracked_points =
    read_positive_size("vio.adaptive.healthy_tracked_points");
  adaptive_policy_config_.total_candidate_budget =
    read_positive_size("vio.adaptive.total_candidate_budget");
  adaptive_policy_config_.minimum_candidate_budget =
    read_positive_size("vio.adaptive.minimum_candidate_budget");
  adaptive_policy_config_.total_patch_budget =
    read_positive_size("vio.adaptive.total_patch_budget");
  adaptive_policy_config_.minimum_patch_budget =
    read_positive_size("vio.adaptive.minimum_patch_budget");
  adaptive_policy_config_.minimum_selected_cameras =
    read_positive_size("vio.adaptive.minimum_selected_cameras");
  adaptive_policy_config_.maximum_selected_cameras =
    read_positive_size("vio.adaptive.maximum_selected_cameras");
  if (adaptive_policy_config_.enabled)
  {
    // Adaptive production has one sealed winner per partial batch. The legacy
    // YAML top-k values remain readable for compatibility, but cannot restore
    // the removed primary/challenger or Front-fallback semantics.
    adaptive_policy_config_.minimum_selected_cameras = 1U;
    adaptive_policy_config_.maximum_selected_cameras = 1U;
  }
  camera_observation_queue_.setCapacity(
    static_cast<std::size_t>(camera_queue_capacity_));
  this->node->get_parameter("multi_camera.enable", multi_camera_enable_);
  this->node->get_parameter("multi_camera.topics.front", img_topic_front_);
  this->node->get_parameter("multi_camera.topics.left", img_topic_left_);
  this->node->get_parameter("multi_camera.topics.back", img_topic_back_);
  this->node->get_parameter("multi_camera.topics.right", img_topic_right_);
  this->node->get_parameter("multi_camera.extrinsics.front.R", multi_front_R_);
  this->node->get_parameter("multi_camera.extrinsics.front.T", multi_front_T_);
  this->node->get_parameter("multi_camera.extrinsics.left.R", multi_left_R_);
  this->node->get_parameter("multi_camera.extrinsics.left.T", multi_left_T_);
  this->node->get_parameter("multi_camera.extrinsics.back.R", multi_back_R_);
  this->node->get_parameter("multi_camera.extrinsics.back.T", multi_back_T_);
  this->node->get_parameter("multi_camera.extrinsics.right.R", multi_right_R_);
  this->node->get_parameter("multi_camera.extrinsics.right.T", multi_right_T_);
  for (std::size_t camera_id = 0; camera_id < camera_names.size(); ++camera_id)
  {
    const std::string &camera_name = camera_names[camera_id];
    this->node->get_parameter(
      "multi_camera.vio_extrinsics." + camera_name + ".R",
      multi_vio_R_[camera_id]);
    this->node->get_parameter(
      "multi_camera.vio_extrinsics." + camera_name + ".T",
      multi_vio_T_[camera_id]);
    this->node->get_parameter(
      "multi_camera.time_offsets." + camera_name,
      camera_time_offsets_[camera_id]);
  }

  this->node->get_parameter("vio.normal_en", normal_en);
  this->node->get_parameter("vio.inverse_composition_en", inverse_composition_en);
  this->node->get_parameter("vio.max_iterations", max_iterations);
  this->node->get_parameter("vio.img_point_cov", IMG_POINT_COV);
  this->node->get_parameter(
    "vio.joint_img_point_cov", JOINT_IMG_POINT_COV);
  this->node->get_parameter("vio.raycast_en", raycast_en);
  this->node->get_parameter("vio.exposure_estimate_en", exposure_estimate_en);
  this->node->get_parameter("vio.inv_expo_cov", inv_expo_cov);
  this->node->get_parameter("vio.grid_size", grid_size);
  this->node->get_parameter("vio.grid_n_height", grid_n_height);
  this->node->get_parameter("vio.patch_pyrimid_level", patch_pyrimid_level);
  this->node->get_parameter("vio.patch_size", patch_size);
  this->node->get_parameter("vio.outlier_threshold", outlier_threshold);
  this->node->get_parameter("time_offset.exposure_time_init", exposure_time_init);
  this->node->get_parameter("time_offset.img_time_offset", img_time_offset);
  this->node->get_parameter("uav.imu_rate_odom", imu_prop_enable);
  this->node->get_parameter("uav.gravity_align_en", gravity_align_en);

  this->node->get_parameter("evo.seq_name", seq_name);
  this->node->get_parameter("evo.pose_output_en", pose_output_en);
  this->node->get_parameter("imu.gyr_cov", gyr_cov);
  this->node->get_parameter("imu.acc_cov", acc_cov);
  this->node->get_parameter("imu.imu_int_frame", imu_int_frame);
  this->node->get_parameter("imu.imu_en", imu_en);
  this->node->get_parameter("imu.gravity_est_en", gravity_est_en);
  this->node->get_parameter("imu.ba_bg_est_en", ba_bg_est_en);

  this->node->get_parameter("preprocess.blind", p_pre->blind);
  this->node->get_parameter("preprocess.filter_size_surf", filter_size_surf_min);
  this->node->get_parameter("preprocess.lidar_type", p_pre->lidar_type);
  this->node->get_parameter("preprocess.scan_line", p_pre->N_SCANS);
  this->node->get_parameter("preprocess.scan_rate", p_pre->SCAN_RATE);
  this->node->get_parameter("preprocess.point_filter_num", p_pre->point_filter_num);
  this->node->get_parameter("preprocess.feature_extract_enabled", p_pre->feature_enabled);

  this->node->get_parameter("pcd_save.interval", pcd_save_interval);
  this->node->get_parameter("pcd_save.pcd_save_en", pcd_save_en);
  this->node->get_parameter("pcd_save.colmap_output_en", colmap_output_en);
  this->node->get_parameter("pcd_save.filter_size_pcd", filter_size_pcd);
  this->node->get_parameter("extrin_calib.extrinsic_T", extrinT);
  this->node->get_parameter("extrin_calib.extrinsic_R", extrinR);
  this->node->get_parameter("extrin_calib.Pcl", cameraextrinT);
  this->node->get_parameter("extrin_calib.Rcl", cameraextrinR);
  this->node->get_parameter("debug.plot_time", plot_time);
  this->node->get_parameter("debug.frame_cnt", frame_cnt);

  this->node->get_parameter("publish.blind_rgb_points", blind_rgb_points);
  this->node->get_parameter("publish.pub_scan_num", pub_scan_num);
  this->node->get_parameter("publish.rgb_topic", rgb_cloud_topic_);
  this->node->get_parameter("publish.persistent_map_en", persistent_map_en_);
  this->node->get_parameter("publish.map_topic", persistent_lio_map_topic_);
  this->node->get_parameter("publish.rgb_map_topic", persistent_rgb_map_topic_);
  this->node->get_parameter("publish.map_publish_interval", persistent_map_publish_interval_);
  this->node->get_parameter("publish.map_voxel_size", persistent_map_voxel_size_);
  this->node->get_parameter("publish.pub_effect_point_en", pub_effect_point_en);
  this->node->get_parameter("publish.dense_map_en", dense_map_en);

  try_declare.template operator()<bool>("local_fast.enable", false);
  try_declare.template operator()<bool>("local_fast.allow_instantaneous_scan", false);
  this->node->get_parameter("local_fast.allow_instantaneous_scan", local_allow_instantaneous_scan_);
  try_declare.template operator()<bool>("local_fast.body_dense", true);
  this->node->get_parameter("local_fast.enable", local_fast_mode_);
  this->node->get_parameter("local_fast.body_dense", local_body_dense_);
  if (local_fast_mode_)
  {
    double configured_imu_offset = 0.0;
    this->node->get_parameter_or("time_offset.imu_time_offset", configured_imu_offset, 0.0);
    if (configured_imu_offset != 0.0 || imu_time_offset != 0.0 || ros_driver_fix_en)
      throw std::runtime_error("local_fast requires raw IMU stamps: offset=0 and driver fix disabled");
    img_en = 0;
    multi_camera_enable_ = false;
    adaptive_policy_config_.enabled = false;
    vio_camera_mode_ = "front";
    imu_prop_enable = false;
    gravity_align_en = true;
    pose_output_en = false;
    pcd_save_en = false;
    colmap_output_en = false;
    pcd_save_interval = -1;
    persistent_map_en_ = false;
    pub_effect_point_en = false;
  }

  persistent_map_publish_interval_ = std::max(1, persistent_map_publish_interval_);
  persistent_map_voxel_size_ = std::max(0.01, persistent_map_voxel_size_);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}

void LIVMapper::initializeComponents(rclcpp::Node::SharedPtr &node) 
{
  if (component_initialization_started_)
  {
    throw std::logic_error("LIVMapper component initialization is one-shot");
  }
  component_initialization_started_ = true;

  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  
  // extrinT.assign({0.04165, 0.02326, -0.0284});
  // extrinR.assign({1, 0, 0, 0, 1, 0, 0, 0, 1});
  // cameraextrinT.assign({0.0194384, 0.104689,-0.0251952});
  // cameraextrinR.assign({0.00610193,-0.999863,-0.0154172,-0.00615449,0.0153796,-0.999863,0.999962,0.00619598,-0.0060598});

  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);

  if (vio_camera_mode_ != "front" && vio_camera_mode_ != "four")
  {
    throw std::runtime_error(
      "vio.camera_mode must be either 'front' or 'four'");
  }
  if (adaptive_policy_config_.enabled &&
      vio_camera_mode_ != "front" &&
      vio_camera_mode_ != "four")
  {
    throw std::runtime_error(
      "vio.adaptive.enable requires vio.camera_mode='front' or 'four'");
  }
  if (adaptive_partial_batch_enable_ &&
      (!adaptive_policy_config_.enabled ||
       (vio_camera_mode_ != "front" &&
        vio_camera_mode_ != "four") ||
       adaptive_policy_config_.minimum_selected_cameras != 1U ||
       adaptive_policy_config_.maximum_selected_cameras != 1U))
  {
    throw std::runtime_error(
      "vio.adaptive.partial_batch.enable requires "
      "vio.adaptive.enable=true, vio.camera_mode='front' or 'four', and "
      "a single selected camera");
  }
  if (adaptive_partial_batch_enable_ &&
      (!std::isfinite(adaptive_missing_camera_timeout_sec_) ||
       adaptive_missing_camera_timeout_sec_ < 0.0))
  {
    throw std::runtime_error(
      "Invalid adaptive partial camera batch timeout");
  }
  if (camera_time_transfer_shadow_enable_ &&
      vio_camera_mode_ != "four")
  {
    throw std::runtime_error(
      "vio.multiview.time_transfer.shadow_enable requires "
      "vio.camera_mode='four'");
  }
  if (camera_time_transfer_shadow_enable_ &&
      !camera_time_transfer_.configValid())
  {
    throw std::runtime_error(
      "Invalid multiview camera time-transfer configuration");
  }
  if (shared_landmark_shadow_config_.enabled &&
      (vio_camera_mode_ != "four" ||
       !camera_time_transfer_shadow_enable_))
  {
    throw std::runtime_error(
      "vio.multiview.shared_landmarks.shadow_enable requires "
      "four-camera mode and time-transfer shadow");
  }
  if (shared_landmark_shadow_config_.enabled &&
      (!std::isfinite(
         shared_landmark_shadow_config_.plane_cell_size_m) ||
       shared_landmark_shadow_config_.plane_cell_size_m <= 0.0 ||
       shared_landmark_shadow_config_.
         maximum_seeds_per_batch == 0U))
  {
    throw std::runtime_error(
      "Invalid shared landmark shadow configuration");
  }
  if (observation_shadow_config_.enabled &&
      (!shared_landmark_shadow_config_.enabled ||
       observation_shadow_config_.
         maximum_observations_per_batch == 0U ||
       observation_shadow_config_.maximum_patch_levels == 0U ||
       observation_shadow_config_.
         maximum_patch_values_per_observation == 0U))
  {
    throw std::runtime_error(
      "vio.multiview.shared_landmarks.observation_shadow_enable "
      "requires a valid shared-landmark geometry shadow");
  }
  if (
    multi_camera_landmark_linearization_shadow_config_.enabled &&
    (
      vio_camera_mode_ != "four" ||
      !adaptive_policy_config_.enabled ||
      !reference_only_migration_shadow_enable_ ||
      !cross_camera_migration_shadow_enable_ ||
      !camera_time_transfer_shadow_enable_ ||
      !shared_landmark_shadow_config_.enabled ||
      !observation_shadow_config_.enabled))
  {
    throw std::runtime_error(
      "vio.multiview.shared_landmarks.linearization_shadow.enable "
      "requires four-camera adaptive VIO, the R2b runtime, "
      "cross-camera owner, time transfer, and shared-landmark "
      "geometry/observation shadows");
  }
  if (
    multi_camera_landmark_linearization_shadow_config_.enabled &&
    !multi_camera_landmark_linearization_shadow_config_.valid())
  {
    throw std::runtime_error(
      "Invalid multi-camera landmark linearization shadow "
      "configuration");
  }
  if (
    shared_landmark_tombstone_shadow_enable_ &&
    !shared_landmark_shadow_config_.enabled)
  {
    throw std::runtime_error(
      "vio.multiview.shared_landmarks.tombstone_shadow_enable "
      "requires the shared-landmark geometry shadow");
  }
  if (
    shared_landmark_refit_shadow_enable_ &&
    !shared_landmark_shadow_config_.enabled)
  {
    throw std::runtime_error(
      "vio.multiview.shared_landmarks.refit_shadow_enable "
      "requires the shared-landmark geometry shadow");
  }
  if (vio_camera_mode_ == "four" && raycast_en)
  {
    throw std::runtime_error(
      "vio.raycast_en is not supported with four-camera VIO");
  }
  if (vio_camera_mode_ == "four" && colmap_output_en)
  {
    throw std::runtime_error(
      "COLMAP export is not supported with four-camera VIO");
  }
  if (vio_camera_mode_ == "four" && inverse_composition_en)
  {
    throw std::runtime_error(
      "vio.inverse_composition_en is not supported with joint four-camera VIO");
  }
  if (vio_camera_mode_ == "four" && exposure_estimate_en)
  {
    throw std::runtime_error(
      "joint four-camera VIO requires vio.exposure_estimate_en=false "
      "because the cameras have independent exposure");
  }
  if (vio_camera_mode_ == "four" &&
      (camera_sync_tolerance_sec_ <= 0.0 ||
       !std::isfinite(joint_vio_warmup_sec_) ||
       joint_vio_warmup_sec_ < 0.0 ||
       min_tracked_points_per_camera_ <= 0 ||
       min_joint_tracked_points_ <= 0 ||
       min_joint_cameras_ <= 0 ||
       min_joint_cameras_ >
         static_cast<int>(CameraObservationQueue::kCameraCount) ||
       JOINT_IMG_POINT_COV <= 0 ||
       max_visual_residual_rmse_ <= 0.0 ||
       max_visual_rotation_increment_deg_ <= 0.0 ||
       max_visual_translation_increment_m_ <= 0.0))
  {
    throw std::runtime_error("Invalid joint four-camera VIO gate configuration");
  }
  if (adaptive_policy_config_.enabled &&
      adaptive_policy_config_.minimum_selected_cameras !=
        static_cast<std::size_t>(min_joint_cameras_))
  {
    throw std::runtime_error(
      "vio.adaptive.minimum_selected_cameras must equal "
      "vio.min_joint_cameras");
  }

  auto parse_vio_extrinsic =
    [](const vector<double> &rotation, const vector<double> &translation,
       const std::string &camera_name) -> std::pair<M3D, V3D>
    {
      if (rotation.size() != 9U || translation.size() != 3U)
      {
        throw std::runtime_error(
          "Invalid LiDAR-to-camera VIO extrinsic for camera " + camera_name);
      }
      M3D R;
      V3D T;
      R << MAT_FROM_ARRAY(rotation);
      T << VEC_FROM_ARRAY(translation);
      return {R, T};
    };

  // Configure the shadow session before camera registration so the session,
  // rather than VIOManager, owns the first rig/model revision events.
  vio_manager->configureSharedLandmarkShadow(
    shared_landmark_shadow_config_);
  vio_manager->configureSharedLandmarkObservationShadow(
    observation_shadow_config_);
  if (camera_time_transfer_shadow_enable_)
  {
    vio_manager->configureSharedLandmarkTimeTransfer(
      camera_time_transfer_config_);
  }
  livo::SharedLandmarkTombstoneShadowConfig
    tombstone_shadow_config;
  tombstone_shadow_config.enabled =
    shared_landmark_tombstone_shadow_enable_;
  vio_manager->configureSharedLandmarkTombstoneShadow(
    tombstone_shadow_config);
  livo::SharedLandmarkGeometryRefitShadowConfig
    refit_shadow_config;
  refit_shadow_config.enabled =
    shared_landmark_refit_shadow_enable_;
  vio_manager->configureSharedLandmarkRefitShadow(
    refit_shadow_config);

  vio_manager->setImuToLidarExtrinsic(extT, extR);
  livo::CrossCameraMigrationShadowInitializationConfig
    cross_camera_migration_shadow_config;
  cross_camera_migration_shadow_config.enabled =
    cross_camera_migration_shadow_enable_;
  cross_camera_migration_shadow_config.four_camera_mode =
    vio_camera_mode_ == "four";
  const bool configure_four_camera_rig =
    vio_camera_mode_ == "four" || multi_camera_enable_;
  if (configure_four_camera_rig)
  {
    const std::array<std::string, CameraObservationQueue::kCameraCount> camera_names =
      {{"front", "left", "back", "right"}};
    for (std::size_t camera_id = 0; camera_id < camera_names.size(); ++camera_id)
    {
      const std::string camera_ns =
        "multi_camera.cameras." + camera_names[camera_id];
      const livo::CameraModelParameters camera_parameters =
        readCameraModelParameters(this->node, camera_ns);
      vk::AbstractCamera *camera = livo::createCameraModel(
        camera_parameters);
      if (camera == nullptr)
      {
        throw std::runtime_error(
          "Camera model not correctly specified: " + camera_ns);
      }
      livo::CameraModelParameters shadow_projection =
        camera_parameters;
      if (camera_parameters.model == "Pinhole")
      {
        shadow_projection.width = camera->width();
        shadow_projection.height = camera->height();
        shadow_projection.scale = camera->scale();
        shadow_projection.fx = camera->fx();
        shadow_projection.fy = camera->fy();
        shadow_projection.cx = camera->cx();
        shadow_projection.cy = camera->cy();
      }
      const auto [Rcl_camera, Pcl_camera] = parse_vio_extrinsic(
        multi_vio_R_[camera_id], multi_vio_T_[camera_id],
        camera_names[camera_id]);
      vio_manager->registerCamera(
        static_cast<std::uint8_t>(camera_id), camera_names[camera_id],
        camera, Rcl_camera, Pcl_camera);
      if (cross_camera_migration_shadow_enable_)
      {
        auto & shadow_camera =
          cross_camera_migration_shadow_config.cameras[camera_id];
        shadow_camera.camera_id =
          static_cast<std::uint8_t>(camera_id);
        shadow_camera.camera_name = camera_names[camera_id];
        shadow_camera.projection = shadow_projection;
        shadow_camera.projection_calibration_present =
          multi_camera_projection_calibration_present_[camera_id];
        shadow_camera.extrinsic_calibration_present =
          multi_camera_extrinsic_calibration_present_[camera_id];
      }
    }
  }
  else
  {
    vk::AbstractCamera *front_camera = livo::createCameraModel(
      readCameraModelParameters(this->node, "camera"));
    if (front_camera == nullptr)
    {
      throw std::runtime_error("Camera model not correctly specified.");
    }
    const auto [Rcl_front, Pcl_front] = parse_vio_extrinsic(
      cameraextrinR, cameraextrinT, "front");
    vio_manager->registerCamera(
      0U, "front", front_camera, Rcl_front, Pcl_front);
  }
  vio_manager->initializeCrossCameraMigrationShadow(
    cross_camera_migration_shadow_config);
  const bool adaptive_single_camera_runtime =
    adaptiveSingleCameraRuntimeEnabled();
  const bool authority_required =
    adaptive_single_camera_runtime ||
    reference_only_migration_shadow_enable_;
  adaptive_visual_lidar_runtime_initialization_audit_ = {};
  adaptive_visual_lidar_runtime_initialization_audit_.
    adaptive_single_camera_runtime =
    adaptive_single_camera_runtime;
  adaptive_visual_lidar_runtime_initialization_audit_.
    r2b_runtime = reference_only_migration_shadow_enable_;
  adaptive_visual_lidar_runtime_initialization_audit_.
    authority_required = authority_required;
  if (authority_required)
  {
    ++adaptive_visual_lidar_runtime_initialization_audit_.
      batch_sealing_request_count;
    if (!camera_observation_queue_.
        setBatchAuthoritySealingEnabled(true))
    {
      adaptive_visual_lidar_runtime_initialization_audit_.
        status =
        livo::AdaptiveVisualLidarRuntimeInitializationStatus::
        kBatchSealingRejected;
      throw std::runtime_error(
        "Cannot enable adaptive visual batch authority after "
        "camera queue use");
    }
    if (!adaptive_visual_selection_authority_)
    {
      ++adaptive_visual_lidar_runtime_initialization_audit_.
        authority_issue_count;
      try
      {
        adaptive_visual_selection_authority_ =
          camera_observation_queue_.
          issueAdaptiveVisualSelectionAuthority();
      }
      catch (const std::bad_alloc &)
      {
        adaptive_visual_lidar_runtime_initialization_audit_.
          status =
          livo::AdaptiveVisualLidarRuntimeInitializationStatus::
          kResourceFailure;
        throw std::runtime_error(
          "Cannot allocate adaptive visual selection authority");
      }
    }
    adaptive_visual_lidar_runtime_initialization_audit_.
      authority_valid =
      adaptive_visual_selection_authority_ &&
      adaptive_visual_selection_authority_->identity().valid();
    if (!adaptive_visual_lidar_runtime_initialization_audit_.
        authority_valid)
    {
      adaptive_visual_lidar_runtime_initialization_audit_.
        status =
        livo::AdaptiveVisualLidarRuntimeInitializationStatus::
        kAuthorityUnavailable;
      throw std::runtime_error(
        "Cannot initialize adaptive visual selection authority");
    }
    adaptive_visual_lidar_runtime_initialization_audit_.
      status =
      livo::AdaptiveVisualLidarRuntimeInitializationStatus::
      kReady;
  }
  if (reference_only_migration_shadow_enable_)
  {
    reference_only_migration_runtime_publisher_ =
      std::make_unique<
      livo::ReferenceOnlyMigrationRuntimePublisher>(
      camera_observation_queue_.batchAuthorityIdentity());
    if (!reference_only_migration_runtime_publisher_->valid())
    {
      throw std::runtime_error(
        "Cannot initialize R2b runtime publication authority");
    }
    vio_manager->
      configureReferenceOnlyMigrationRuntimeAuthorities(
      camera_observation_queue_.batchAuthorityIdentity(),
      reference_only_migration_runtime_publisher_->identity());
  }
  livo::ReferenceOnlyMigrationShadowRuntimeConfig
    reference_only_migration_shadow_config;
  reference_only_migration_shadow_config.enabled =
    reference_only_migration_shadow_enable_;
  reference_only_migration_shadow_config.four_camera_mode =
    vio_camera_mode_ == "four";
  vio_manager->configureReferenceOnlyMigrationShadowRuntime(
    reference_only_migration_shadow_config);
  if (!vio_manager->configureMultiCameraLandmarkLinearizationShadow(
      multi_camera_landmark_linearization_shadow_config_))
  {
    throw std::runtime_error(
      "Cannot configure multi-camera landmark linearization "
      "shadow");
  }

  vio_manager->grid_size = grid_size;
  vio_manager->patch_size = patch_size;
  vio_manager->outlier_threshold = outlier_threshold;
  vio_manager->state = &_state;
  vio_manager->state_propagat = &state_propagat;
  vio_manager->max_iterations = max_iterations;
  vio_manager->img_point_cov = IMG_POINT_COV;
  vio_manager->joint_img_point_cov = JOINT_IMG_POINT_COV;
  vio_manager->normal_en = normal_en;
  vio_manager->inverse_composition_en = inverse_composition_en;
  vio_manager->raycast_en = raycast_en;
  vio_manager->grid_n_width = grid_n_width;
  vio_manager->grid_n_height = grid_n_height;
  vio_manager->patch_pyrimid_level = patch_pyrimid_level;
  vio_manager->exposure_estimate_en = exposure_estimate_en;
  vio_manager->colmap_output_en = colmap_output_en;
  vio_manager->joint_update_config.min_tracked_points_per_camera =
    static_cast<std::size_t>(min_tracked_points_per_camera_);
  vio_manager->joint_update_config.min_joint_tracked_points =
    static_cast<std::size_t>(min_joint_tracked_points_);
  vio_manager->joint_update_config.min_joint_cameras =
    static_cast<std::size_t>(min_joint_cameras_);
  vio_manager->joint_update_config.max_residual_rmse =
    max_visual_residual_rmse_;
  vio_manager->joint_update_config.max_rotation_increment_rad =
    max_visual_rotation_increment_deg_ * M_PI / 180.0;
  vio_manager->joint_update_config.max_translation_increment_m =
    max_visual_translation_increment_m_;
  vio_manager->adaptive_policy_config = adaptive_policy_config_;
  int camera_threads = 1;
  this->node->get_parameter("vio.adaptive.camera_threads", camera_threads);
  if (camera_threads < 1 || camera_threads > 4)
    throw std::invalid_argument("vio.adaptive.camera_threads must be in [1, 4]");
  vio_manager->adaptive_camera_threads = camera_threads;
  vio_manager->initializeVIO();

  RCLCPP_INFO(
    this->node->get_logger(),
    "[fast_livo_bev] VIO camera mode: %s, reorder window: %.1f ms, "
    "sync tolerance: %.1f ms, queue: %d, adaptive: %s",
    vio_camera_mode_.c_str(), camera_reorder_window_sec_ * 1000.0,
    camera_sync_tolerance_sec_ * 1000.0, camera_queue_capacity_,
    adaptive_policy_config_.enabled ? "enabled" : "disabled");

  p_imu->set_extrinsic(extT, extR);
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_inv_expo_cov(inv_expo_cov);
  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));
  p_imu->set_imu_init_frame_num(imu_int_frame);

  if (!imu_en) p_imu->disable_imu();
  if (!gravity_est_en) p_imu->disable_gravity_est();
  if (!ba_bg_est_en) p_imu->disable_bias_est();
  if (!exposure_estimate_en) p_imu->disable_exposure_est();

  slam_mode_ = (img_en && lidar_en) ? LIVO : imu_en ? ONLY_LIO : ONLY_LO;
}

void LIVMapper::initializeFiles() 
{
  if (local_fast_mode_) return;
  if (pose_output_en)
  {
    std::error_code error;
    const std::filesystem::path result_directory =
      std::filesystem::path(ROOT_DIR) / "Log" / "result";
    std::filesystem::create_directories(result_directory, error);
    if (error)
    {
      throw std::runtime_error(
        "Failed to create pose result directory: " +
        result_directory.string() + ": " + error.message());
    }
  }
  if (pcd_save_en && colmap_output_en)
  {
      const std::string folderPath = std::string(ROOT_DIR) + "/scripts/colmap_output.sh";
      
      std::string chmodCommand = "chmod +x " + folderPath;
      
      int chmodRet = system(chmodCommand.c_str());  
      if (chmodRet != 0) {
          std::cerr << "Failed to set execute permissions for the script." << std::endl;
          return;
      }

      int executionRet = system(folderPath.c_str());
      if (executionRet != 0) {
          std::cerr << "Failed to execute the script." << std::endl;
          return;
      }
  }
  if(colmap_output_en) fout_points.open(std::string(ROOT_DIR) + "Log/Colmap/sparse/0/points3D.txt", std::ios::out);
  if(pcd_save_interval > 0) fout_pcd_pos.open(std::string(ROOT_DIR) + "Log/PCD/scans_pos.json", std::ios::out);
}

void LIVMapper::initializeSubscribersAndPublishers(rclcpp::Node::SharedPtr &node, image_transport::ImageTransport &it_)
{
  image_transport::ImageTransport it(this->node);
  imu_ingress_callback_group_ = this->node->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive, false);
  for (rclcpp::CallbackGroup::SharedPtr & callback_group :
       camera_ingress_callback_groups_)
  {
    callback_group = this->node->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive, false);
  }
  // [FIX 2026-06-09] 上游 ROS2 移植用 #if 0 屏蔽了 Livox CustomMsg 订阅分支, 导致无论
  // lidar_type 设何值都只订阅 PointCloud2; 本数据集 /livox/lidar 为 CustomMsg, 恢复条件分支
  if (p_pre->lidar_type == AVIA) {
    sub_pcl = this->node->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, local_fast_mode_ ? 2U : kLidarSubscriptionDepth, std::bind(&LIVMapper::livox_pcl_cbk, this, std::placeholders::_1));
  } else
  {
    sub_pcl = this->node->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, local_fast_mode_ ? 2U : kLidarSubscriptionDepth, std::bind(&LIVMapper::standard_pcl_cbk, this, std::placeholders::_1));
  }
  rclcpp::SubscriptionOptions imu_subscription_options;
  imu_subscription_options.callback_group = imu_ingress_callback_group_;
  std::array<
    rclcpp::SubscriptionOptions,
    CameraObservationQueue::kCameraCount> camera_subscription_options;
  for (std::size_t camera_id = 0U;
       camera_id < camera_subscription_options.size(); ++camera_id)
  {
    camera_subscription_options[camera_id].callback_group =
      camera_ingress_callback_groups_[camera_id];
  }
  sub_imu = this->node->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, kImuSubscriptionDepth,
    std::bind(
      &LIVMapper::imu_transport_cbk, this, std::placeholders::_1),
    imu_subscription_options);
  if (img_en && vio_camera_mode_ == "front" && !multi_camera_enable_)
  {
    sub_img = this->node->create_subscription<sensor_msgs::msg::Image>(
      img_topic, kCameraSubscriptionDepth,
      std::bind(&LIVMapper::img_cbk, this, std::placeholders::_1),
      camera_subscription_options[0U]);
  }

  const bool four_camera_vio = img_en && vio_camera_mode_ == "four";
  if (multi_camera_enable_ || four_camera_vio)
  {
    cam_front_ = vio_manager->cameraModel(0U);
    cam_left_ = vio_manager->cameraModel(1U);
    cam_back_ = vio_manager->cameraModel(2U);
    cam_right_ = vio_manager->cameraModel(3U);
    if (cam_left_ == nullptr) cam_left_ = cam_front_;
    if (cam_back_ == nullptr) cam_back_ = cam_front_;
    if (cam_right_ == nullptr) cam_right_ = cam_front_;

    Eigen::Matrix3d R_camLink_camOptical;
    R_camLink_camOptical << 0, 0, 1,
                            -1, 0, 0,
                             0, -1, 0;
    Sophus::SE3<double> T_camLink_camOptical(R_camLink_camOptical, Eigen::Vector3d::Zero());

    auto make_base_cam = [this](const std::string &name, const vector<double> &R_v, const vector<double> &T_v) {
      if (R_v.size() != 9 || T_v.size() != 3)
      {
        RCLCPP_WARN(this->node->get_logger(), "[fast_livo_bev] invalid extrinsic for %s, using identity", name.c_str());
        return Sophus::SE3<double>();
      }
      Eigen::Matrix3d R;
      R << MAT_FROM_ARRAY(R_v);
      Eigen::Vector3d T(T_v[0], T_v[1], T_v[2]);
      return Sophus::SE3<double>(R, T);
    };

    T_base_cam_front_ = make_base_cam("front", multi_front_R_, multi_front_T_) * T_camLink_camOptical;
    T_base_cam_left_ = make_base_cam("left", multi_left_R_, multi_left_T_) * T_camLink_camOptical;
    T_base_cam_back_ = make_base_cam("back", multi_back_R_, multi_back_T_) * T_camLink_camOptical;
    T_base_cam_right_ = make_base_cam("right", multi_right_R_, multi_right_T_) * T_camLink_camOptical;

    std::array<Sophus::SE3<double> *, CameraObservationQueue::kCameraCount>
      camera_poses{{
        &T_base_cam_front_, &T_base_cam_left_,
        &T_base_cam_back_, &T_base_cam_right_}};
    for (std::size_t camera_id = 0U;
         camera_id < camera_poses.size(); ++camera_id)
    {
      if (!multi_camera_extrinsic_calibration_present_[camera_id])
      {
        continue;
      }
      M3D rotation_camera_from_imu;
      V3D translation_camera_from_imu;
      if (!vio_manager->cameraFromImuExtrinsic(
          static_cast<std::uint8_t>(camera_id),
          rotation_camera_from_imu,
          translation_camera_from_imu))
      {
        throw std::runtime_error(
          "Configured camera extrinsic is unavailable for RGB coloring");
      }
      *camera_poses[camera_id] = Sophus::SE3<double>(
        rotation_camera_from_imu.transpose(),
        -rotation_camera_from_imu.transpose() *
        translation_camera_from_imu);
    }

    sub_img_front = this->node->create_subscription<sensor_msgs::msg::Image>(
      img_topic_front_, kCameraSubscriptionDepth,
      std::bind(&LIVMapper::img_cbk_front, this, std::placeholders::_1),
      camera_subscription_options[0U]);
    sub_img_left = this->node->create_subscription<sensor_msgs::msg::Image>(
      img_topic_left_, kCameraSubscriptionDepth,
      std::bind(&LIVMapper::img_cbk_left, this, std::placeholders::_1),
      camera_subscription_options[1U]);
    sub_img_back = this->node->create_subscription<sensor_msgs::msg::Image>(
      img_topic_back_, kCameraSubscriptionDepth,
      std::bind(&LIVMapper::img_cbk_back, this, std::placeholders::_1),
      camera_subscription_options[2U]);
    sub_img_right = this->node->create_subscription<sensor_msgs::msg::Image>(
      img_topic_right_, kCameraSubscriptionDepth,
      std::bind(&LIVMapper::img_cbk_right, this, std::placeholders::_1),
      camera_subscription_options[3U]);

    RCLCPP_INFO(
      this->node->get_logger(),
      "[fast_livo_bev] four-camera subscriptions enabled (vio=%s, coloring=%s): %s / %s / %s / %s",
      four_camera_vio ? "true" : "false",
      multi_camera_enable_ ? "true" : "false",
      img_topic_front_.c_str(), img_topic_left_.c_str(),
      img_topic_back_.c_str(), img_topic_right_.c_str());
  }
  
  if (local_fast_mode_)
  {
    pubLaserCloudBody = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/local/fast_lio/cloud_body", rclcpp::QoS(2).reliable());
    pubOdomAftMapped = this->node->create_publisher<nav_msgs::msg::Odometry>(
      "/local/fast_lio/odometry", rclcpp::QoS(20).reliable());
    local_scan_status_pub_ = this->node->create_publisher<std_msgs::msg::String>(
      "/local/fast_lio/scan_status", rclcpp::QoS(20).reliable());
    return;
  }
  pubLaserCloudFullRes = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 100);
  pubLaserCloudRGB = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(rgb_cloud_topic_, 100);
  if (persistent_map_en_)
  {
    const auto map_qos = rclcpp::QoS(1).reliable().transient_local();
    pubPersistentLioMap = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      persistent_lio_map_topic_, map_qos);
    pubPersistentRGBMap = this->node->create_publisher<sensor_msgs::msg::PointCloud2>(
      persistent_rgb_map_topic_, map_qos);
    RCLCPP_INFO(
      this->node->get_logger(),
      "[fast_livo_bev] persistent map snapshots enabled: %s / %s (%.3fm, every %d scans)",
      persistent_lio_map_topic_.c_str(), persistent_rgb_map_topic_.c_str(),
      persistent_map_voxel_size_, persistent_map_publish_interval_);
  }
  pubLaserCloudBody = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_undistorted_body", 100);
  pubNormal = this->node->create_publisher<visualization_msgs::msg::MarkerArray>("/visualization_marker", 100);
  pubSubVisualMap = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_visual_sub_map_before", 100);
  pubLaserCloudEffect = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 100);
  pubLaserCloudMap = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 100);
  pubOdomAftMapped = this->node->create_publisher<nav_msgs::msg::Odometry>("/aft_mapped_to_init", 10);
  odom_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this->node);
  pubPath = this->node->create_publisher<nav_msgs::msg::Path>("/path", 10);
  plane_pub = this->node->create_publisher<visualization_msgs::msg::Marker>("/planner_normal", 1);
  voxel_pub = this->node->create_publisher<visualization_msgs::msg::MarkerArray>("/voxels", 1);
  pubLaserCloudDyn = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/dyn_obj", 100);
  pubLaserCloudDynRmed = this->node->create_publisher<sensor_msgs::msg::PointCloud2>("/dyn_obj_removed", 100);
  mavros_pose_publisher = this->node->create_publisher<geometry_msgs::msg::PoseStamped>("/mavros/vision_pose/pose", 10);
  pubImage = it.advertise("/rgb_img", 1);
  pubImuPropOdom = this->node->create_publisher<nav_msgs::msg::Odometry>("/LIVO2/imu_propagate", 10000);
  imu_prop_timer = this->node->create_wall_timer(0.004s, std::bind(&LIVMapper::imu_prop_callback, this));
  voxelmap_manager->voxel_map_pub_= this->node->create_publisher<visualization_msgs::msg::MarkerArray>("/planes", 10000);

  pubRuntimeReady = this->node->create_publisher<std_msgs::msg::Bool>(
    "/livo/runtime_ready",
    rclcpp::QoS(1).reliable().transient_local());
  std_msgs::msg::Bool ready;
  ready.data = true;
  pubRuntimeReady->publish(ready);
  RCLCPP_INFO(
    this->node->get_logger(),
    "[RUNTIME_READY] subscriptions_created=true topic=/livo/runtime_ready");
}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Eigen::Quaterniond G_q_I0 = Eigen::Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  const MeasureGroup & measurement = LidarMeasures.measures.back();
  const bool zero_duration_vio =
    LidarMeasures.lio_vio_flg == VIO &&
    measurement.imu.empty() &&
    measurement.vio_time == LidarMeasures.last_lio_update_time;
  if (zero_duration_vio)
  {
    feats_undistort->clear();
  }
  else
  {
    p_imu->Process2(LidarMeasures, _state, feats_undistort);
  }

  if (determinism_audit_enable_)
  {
    current_imu_sample_hash_ = hashImuSamples(
      LidarMeasures, current_imu_stamp_hash_, current_imu_stamp_count_);
    current_undistorted_point_count_ = feats_undistort->points.size();
    current_undistorted_cloud_hash_ = hashPointCloud(*feats_undistort);
    current_voxel_grid_hash_ = 0U;
    current_voxel_grid_point_count_ = 0U;
    current_voxel_grid_hash_applicable_ = false;
  }

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  const EKF_STATE update_type = LidarMeasures.lio_vio_flg;
  switch (LidarMeasures.lio_vio_flg) 
  {
    case VIO:
      handleVIO();
      break;
    case LIO:
    case LO:
      handleLIO();
      break;
  }
  if (determinism_audit_enable_)
  {
    ++frame_hash_sequence_;
    const std::uint64_t state_hash = hashState(_state);
    const std::uint64_t propagated_state_hash = hashState(state_propagat);
    const std::uint64_t nominal_state_hash = hashNominalState(_state);
    const std::uint64_t covariance_hash = hashStateCovariance(_state);
    const double update_timestamp =
      LidarMeasures.measures.empty() ?
      LidarMeasures.last_lio_update_time :
      update_type == VIO ?
      LidarMeasures.measures.back().vio_time :
      LidarMeasures.measures.back().lio_time;
    livo::emitStructuredMetric(
      "[FRAME_HASH] schema_version=3 sequence=%llu update=%s "
      "timestamp=%.9f imu_stamp_hash=%016llx imu_sample_hash=%016llx "
      "imu_count=%zu "
      "undistorted_cloud_hash=%016llx undistorted_count=%zu "
      "voxel_grid_hash=%016llx voxel_grid_count=%zu "
      "voxel_grid_applicable=%u propagated_state_hash=%016llx "
      "nominal_state_hash=%016llx covariance_hash=%016llx "
      "state_hash=%016llx\n",
      static_cast<unsigned long long>(frame_hash_sequence_),
      updateTypeName(update_type), update_timestamp,
      static_cast<unsigned long long>(current_imu_stamp_hash_),
      static_cast<unsigned long long>(current_imu_sample_hash_),
      current_imu_stamp_count_,
      static_cast<unsigned long long>(current_undistorted_cloud_hash_),
      current_undistorted_point_count_,
      static_cast<unsigned long long>(current_voxel_grid_hash_),
      current_voxel_grid_point_count_,
      current_voxel_grid_hash_applicable_ ? 1U : 0U,
      static_cast<unsigned long long>(propagated_state_hash),
      static_cast<unsigned long long>(nominal_state_hash),
      static_cast<unsigned long long>(covariance_hash),
      static_cast<unsigned long long>(state_hash));
  }
}

void LIVMapper::handleVIO() 
{
  adaptive_visual_lidar_runtime_dispatch_audit_ = {};
  const AdaptiveVisualShadowEntryAuditScope
    shadow_entry_audit_scope{
      adaptive_visual_lidar_runtime_dispatch_audit_,
      pre_vio_shadow_entry_count_,
      post_vio_shadow_entry_count_};
  if (adaptiveSingleCameraRuntimeEnabled())
  {
    bool dispatched = false;
    try
    {
      dispatched = dispatchAdaptiveSingleCameraVio();
    }
    catch (...)
    {
      pending_adaptive_lio_mode_observation_.reset();
      throw;
    }
    const livo::AdaptiveVisualRuntimeAudit & visual_audit =
      vio_manager->adaptiveVisualRuntimeAudit();
    if (LidarMeasures.measures.empty())
    {
      return;
    }
    const bool qualified_visual_fused =
      dispatched &&
      adaptive_visual_lidar_runtime_dispatch_audit_.generation_issued &&
      visual_audit.outcome ==
        livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted &&
      visual_audit.fusion_camera_id.has_value();
    const bool authoritative_zero_selection =
      dispatched &&
      visual_audit.outcome ==
        livo::AdaptiveVisualRuntimeOutcome::kLioOnly &&
      visual_audit.lio_only_reason ==
        livo::AdaptiveVisualLioOnlyReason::
        kSelectionIssueRejected &&
      visual_audit.selection_issue_status.has_value() &&
      *visual_audit.selection_issue_status ==
        livo::AdaptiveVisualSelectionStatus::
        kNoSelectedCamera &&
      std::none_of(
        visual_audit.selected_mask.begin(),
        visual_audit.selected_mask.end(),
        [](const bool selected) { return selected; });
    finalizeAdaptiveVisualOutcome(
      qualified_visual_fused ?
      livo::AdaptiveLioMode::kSelectedVisual :
      livo::AdaptiveLioMode::kLioOnly,
      qualified_visual_fused ?
      livo::AdaptiveLioModeReason::
      kEligibleVisualSelected :
      authoritative_zero_selection ?
      livo::AdaptiveLioModeReason::kNoEligibleVisual :
      livo::AdaptiveLioModeReason::
      kVisualOutcomeNotObservedBeforeCloseout);
    if (!dispatched)
    {
      if (multi_camera_enable_ && !pcl_w_wait_pub->empty())
      {
        publish_frame_world(pubLaserCloudFullRes, vio_manager);
      }
      return;
    }
  }
  else
  {
    std::optional<livo::CameraBatchTimeTransferBundle>
      time_transfer_bundle;
    std::optional<livo::ReferenceOnlyMigrationRuntimeBundle>
      migration_runtime_bundle;
    if (vio_camera_mode_ == "four" &&
        camera_time_transfer_shadow_enable_)
    {
      if (reference_only_migration_shadow_enable_)
      {
        migration_runtime_bundle =
          emitReferenceOnlyMigrationRuntimePublication(
          LidarMeasures.measures.back().camera_batch,
          LidarMeasures.measures.back().vio_time);
      }
      if (!migration_runtime_bundle.has_value())
      {
        time_transfer_bundle = emitCameraTimeTransferShadow(
        LidarMeasures.measures.back().camera_batch,
        LidarMeasures.measures.back().vio_time);
      }
    }

    if (_pv_list.empty())
    {
      std::cout << "[ VIO ] No cached LiDAR features!!!" << std::endl;
      return;
    }

    if (vio_camera_mode_ == "four" &&
        lio_feature_generation_ == last_joint_vio_feature_generation_)
    {
      printf(
        "[VIO_JOINT_SKIP_METRIC] t=%.9f reason=no_fresh_lidar "
        "generation=%llu\n",
        LidarMeasures.last_lio_update_time - _first_lidar_time,
        static_cast<unsigned long long>(lio_feature_generation_));
      return;
    }

    std::cout << "[ VIO ] Raw feature num: " << _pv_list.size()
              << std::endl;

    if (shouldPlotVioFrame(
        LidarMeasures.last_lio_update_time - _first_lidar_time,
        plot_time, frame_cnt))
    {
      vio_manager->plot_flag = true;
    }
    else
    {
      vio_manager->plot_flag = false;
    }

    if (vio_camera_mode_ == "four")
    {
      last_joint_vio_feature_generation_ = lio_feature_generation_;
      std::optional<livo::ActivePlaneIdentitySnapshot>
        active_plane_snapshot;
      const livo::SharedLandmarkLifecycleRouting
        shared_landmark_lifecycle_routing{
          shared_landmark_shadow_config_.enabled,
          shared_landmark_tombstone_shadow_enable_,
          shared_landmark_refit_shadow_enable_};
      if (shared_landmark_lifecycle_routing.
          activePlaneSnapshotRequired())
      {
        active_plane_snapshot =
          voxelmap_manager->snapshotActivePlaneIdentities();
      }
      vio_manager->processFrames(
        LidarMeasures.measures.back().camera_batch, _pv_list,
        voxelmap_manager->voxel_map_,
        LidarMeasures.last_lio_update_time - _first_lidar_time,
        voxelmap_manager->pose_information_,
        voxelmap_manager->pose_information_valid_,
        lio_feature_generation_,
        active_plane_snapshot ?
        &*active_plane_snapshot : nullptr,
        migration_runtime_bundle ?
        &migration_runtime_bundle->timeTransferBundle() :
        (time_transfer_bundle ?
        &*time_transfer_bundle : nullptr),
        migration_runtime_bundle ?
        &*migration_runtime_bundle : nullptr);
      if (shared_landmark_refit_shadow_enable_)
      {
        ++post_vio_shadow_entry_count_;
        const livo::PlaneRefitEventBatch events =
          voxelmap_manager->snapshotPlaneRefitEvents();
        if (!events.events.empty())
        {
          const livo::SharedLandmarkShadowOperationResult result =
            vio_manager->applySharedLandmarkRefitsShadow(
            events.events);
          voxelmap_manager->acknowledgePlaneRefitEvents(
            events, result.acknowledge_events);
        }
      }
      if (shared_landmark_tombstone_shadow_enable_)
      {
        ++post_vio_shadow_entry_count_;
        const std::vector<livo::PlaneTombstoneEvent> events =
          voxelmap_manager->snapshotPlaneTombstoneEvents();
        if (!events.empty())
        {
          const livo::SharedLandmarkShadowOperationResult result =
            vio_manager->applySharedLandmarkTombstonesShadow(events);
          if (result.acknowledge_events)
          {
            voxelmap_manager->consumePlaneTombstoneEvents(
              shared_landmark_tombstone_shadow_enable_);
          }
        }
      }
    }
    else
    {
      vio_manager->processFrame(
        LidarMeasures.measures.back().camera, _pv_list,
        voxelmap_manager->voxel_map_,
        LidarMeasures.last_lio_update_time - _first_lidar_time);
    }
  }

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  // int size_sub_map = vio_manager->visual_sub_map_cur.size();
  // visual_sub_map->reserve(size_sub_map);
  // for (int i = 0; i < size_sub_map; i++) 
  // {
  //   PointType temp_map;
  //   temp_map.x = vio_manager->visual_sub_map_cur[i]->pos_[0];
  //   temp_map.y = vio_manager->visual_sub_map_cur[i]->pos_[1];
  //   temp_map.z = vio_manager->visual_sub_map_cur[i]->pos_[2];
  //   temp_map.intensity = 0.;
  //   visual_sub_map->push_back(temp_map);
  // }

  if (!pcl_w_wait_pub->empty())
  {
    // handleLIO already published this unchanged body-frame observation.
    // Repeating it here would fuse the same sample twice in elevation mapping.
    publish_frame_world(pubLaserCloudFullRes, vio_manager);
  }
  publish_img_rgb(pubImage, vio_manager);
}

livo::CameraBatchTimeTransferBundle
LIVMapper::emitCameraTimeTransferShadow(
  const CameraObservationBatch & batch,
  const double anchor_timestamp_sec)
{
  ++pre_vio_shadow_entry_count_;
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = anchor_timestamp_sec;
  anchor.rotation_world_from_imu = _state.rot_end;
  anchor.position_world_m = _state.pos_end;
  anchor.velocity_world_mps = _state.vel_end;
  anchor.gyro_bias_radps = _state.bias_g;
  anchor.accel_bias_mps2 = _state.bias_a;
  anchor.gravity_world_mps2 = _state.gravity;

  livo::CameraBatchTimeTransferBundle bundle;
  {
    std::lock_guard<std::mutex> lock(camera_time_transfer_mutex_);
    bundle = livo::buildCameraBatchTimeTransferBundle(
      batch, anchor, exposure_time_init, camera_time_transfer_);
  }

  emitCameraTimeTransferMetric(
    bundle, lio_feature_generation_,
    anchor_timestamp_sec);
  return bundle;
}

std::optional<livo::ReferenceOnlyMigrationRuntimeBundle>
LIVMapper::emitReferenceOnlyMigrationRuntimePublication(
  const CameraObservationBatch & batch,
  const double anchor_timestamp_sec)
{
  ++pre_vio_shadow_entry_count_;
  if (
    reference_only_migration_runtime_publisher_ == nullptr ||
    !reference_only_migration_runtime_publisher_->valid())
  {
    return std::nullopt;
  }

  std::optional<livo::ReferenceOnlyMigrationRuntimeBundle>
    publication;
  {
    std::lock_guard<std::mutex> lock(
      camera_time_transfer_mutex_);
    publication =
      reference_only_migration_runtime_publisher_->publish(
      batch, _state, anchor_timestamp_sec,
      exposure_time_init, camera_time_transfer_);
  }
  if (publication.has_value())
  {
    emitCameraTimeTransferMetric(
      publication->timeTransferBundle(),
      lio_feature_generation_, anchor_timestamp_sec);
  }
  return publication;
}

void LIVMapper::handleLIO() 
{
  finalizePendingAdaptiveLioMode();
  if (feats_undistort->empty() || (feats_undistort == nullptr)) 
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);
  if (determinism_audit_enable_)
  {
    current_voxel_grid_point_count_ = feats_down_body->points.size();
    current_voxel_grid_hash_ = hashPointCloud(*feats_down_body);
    current_voxel_grid_hash_applicable_ = true;
  }
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size();
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);
  voxelmap_manager->feats_down_world_ = feats_down_world;
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  _pv_list = voxelmap_manager->pv_list_;

  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) RCLCPP_ERROR(this->node->get_logger(), "open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) RCLCPP_ERROR(this->node->get_logger(), "open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));
  publish_odometry(pubOdomAftMapped);

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);
  for (size_t i = 0; i < world_lidar->points.size(); i++) 
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];
    M3D var = voxelmap_manager->body_cov_list_[i];
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);
    voxelmap_manager->pv_list_[i].var = var;
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  std::cout << "[ LIO ] Update Voxel Map" << std::endl;
  _pv_list = voxelmap_manager->pv_list_;
  if (!livo::checkedIncrement(lio_feature_generation_))
  {
    throw std::overflow_error(
      "LIO feature generation exhausted");
  }
  if (adaptiveSingleCameraRuntimeEnabled())
  {
    pending_adaptive_lio_mode_observation_ = {
      lio_feature_generation_,
      LidarMeasures.last_lio_update_time,
      livo::AdaptiveLioMode::kLioOnly,
      adaptive_lio_only_fallback_active_ ?
      livo::AdaptiveLioModeReason::
      kAllCamerasMissingAfterDeadline :
      livo::AdaptiveLioModeReason::
      kVisualOutcomeNotObservedBeforeCloseout};
  }
  
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)
  {
    voxelmap_manager->mapSliding();
  }
  if (local_fast_mode_)
  {
    publish_frame_body(pubLaserCloudBody);
    return;  // 不构造世界发布云、路径、持久化地图或控制输出。
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_w_wait_pub = *laserCloudWorld;

  publish_frame_body(pubLaserCloudBody);
  if (
    !img_en ||
    (multi_camera_enable_ && adaptive_lio_only_fallback_active_))
  {
    publish_frame_world(pubLaserCloudFullRes, vio_manager);
  }
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);
  printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);
  printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");

}

void LIVMapper::savePCD() 
{
  if (pcd_save_en && (pcl_wait_save->points.size() > 0 || pcl_wait_save_intensity->points.size() > 0) && pcd_save_interval < 0) 
  {
    std::string raw_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_raw_points.pcd";
    std::string downsampled_points_dir = std::string(ROOT_DIR) + "Log/PCD/all_downsampled_points.pcd";
    pcl::PCDWriter pcd_writer;

    if (img_en)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
      voxel_filter.setInputCloud(pcl_wait_save);
      voxel_filter.setLeafSize(filter_size_pcd, filter_size_pcd, filter_size_pcd);
      voxel_filter.filter(*downsampled_cloud);
  
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save); // Save the raw point cloud data
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save->points.size() << RESET << std::endl;
      
      pcd_writer.writeBinary(downsampled_points_dir, *downsampled_cloud); // Save the downsampled point cloud data
      std::cout << GREEN << "Downsampled point cloud data saved to: " << downsampled_points_dir 
                << " with point count after filtering: " << downsampled_cloud->points.size() << RESET << std::endl;

      if(colmap_output_en)
      {
        fout_points << "# 3D point list with one line of data per point\n";
        fout_points << "#  POINT_ID, X, Y, Z, R, G, B, ERROR\n";
        for (size_t i = 0; i < downsampled_cloud->size(); ++i) 
        {
            const auto& point = downsampled_cloud->points[i];
            fout_points << i << " "
                        << std::fixed << std::setprecision(6)
                        << point.x << " " << point.y << " " << point.z << " "
                        << static_cast<int>(point.r) << " "
                        << static_cast<int>(point.g) << " "
                        << static_cast<int>(point.b) << " "
                        << 0 << std::endl;
        }
      }
    }
    else
    {      
      pcd_writer.writeBinary(raw_points_dir, *pcl_wait_save_intensity);
      std::cout << GREEN << "Raw point cloud data saved to: " << raw_points_dir 
                << " with point count: " << pcl_wait_save_intensity->points.size() << RESET << std::endl;
    }
  }
}

void LIVMapper::run(rclcpp::Node::SharedPtr &node) 
{
  const auto check_imu_transport_history = [&]()
  {
    if (imu_transport_overflow_failed_.load(std::memory_order_acquire))
    {
      throw std::runtime_error("IMU transport history overflow: restart required");
    }
  };
  // 预先失败时不启动 ingress，避免 cancel 先于工作线程 spin 的启动竞态。
  check_imu_transport_history();
  rclcpp::Rate rate(5000);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(this->node);
  rclcpp::executors::MultiThreadedExecutor sensor_ingress_executor(
    rclcpp::ExecutorOptions(),
    1U + CameraObservationQueue::kCameraCount);
  sensor_ingress_executor.add_callback_group(
    imu_ingress_callback_group_, this->node->get_node_base_interface());
  for (const rclcpp::CallbackGroup::SharedPtr & callback_group :
       camera_ingress_callback_groups_)
  {
    sensor_ingress_executor.add_callback_group(
      callback_group, this->node->get_node_base_interface());
  }
  std::exception_ptr sensor_ingress_exception;
  std::atomic<bool> sensor_ingress_failed{false};
  std::atomic<bool> sensor_ingress_finished{false};
  std::thread sensor_ingress_thread(
    [&sensor_ingress_executor, &sensor_ingress_exception,
     &sensor_ingress_failed, &sensor_ingress_finished]()
    {
      try
      {
        sensor_ingress_executor.spin();
      }
      catch (...)
      {
        sensor_ingress_exception = std::current_exception();
        sensor_ingress_failed.store(true, std::memory_order_release);
      }
      sensor_ingress_finished.store(true, std::memory_order_release);
    });
  const auto stop_sensor_ingress = [&]()
  {
    // cancel 不能先于 spin 设置 spinning=true，否则该取消会被启动覆盖。
    // finished 同时覆盖 spin 提前返回和异常路径，避免等待已结束的线程。
    while (!sensor_ingress_finished.load(std::memory_order_acquire) &&
           !sensor_ingress_executor.is_spinning())
    {
      std::this_thread::yield();
    }
    if (!sensor_ingress_finished.load(std::memory_order_acquire))
    {
      sensor_ingress_executor.cancel();
    }
    if (sensor_ingress_thread.joinable())
    {
      sensor_ingress_thread.join();
    }
  };
  double last_resource_time = omp_get_wtime();
  try
  {
    while (rclcpp::ok())
    {
      check_imu_transport_history();
      if (sensor_ingress_failed.load(std::memory_order_acquire))
      {
        break;
      }
      try
      {
        executor.spin_some();
      }
      catch (const rclcpp::exceptions::RCLError&)
      {
        if (!rclcpp::ok())
        {
          break;
        }
        throw;
      }
      const double phase_ingress_begin = omp_get_wtime();
      drainImuTransportBuffer();
      check_imu_transport_history();
      drainCameraTransportBuffer();
      check_imu_transport_history();
      const double phase_sync_begin = omp_get_wtime();
      if (!sync_packages(LidarMeasures))
      {
        rate.sleep();
        continue;
      }
      // 已获取完整 IMU 的当前 measurement 可完成；失败后不开始下一次状态处理。
      check_imu_transport_history();
      handleFirstFrame();

      const auto local_started = std::chrono::steady_clock::now();
      local_body_published_ = false;
      const double phase_imu_begin = omp_get_wtime();
      processImu();
      const double phase_map_begin = omp_get_wtime();

      // if (!p_imu->imu_time_init) continue;

      stateEstimationAndMapping();
      if (!local_fast_mode_)
      {
        livo::emitStructuredMetric("[PIPELINE_PHASE] update=%s ingress_ms=%.3f sync_ms=%.3f imu_ms=%.3f mapping_ms=%.3f total_ms=%.3f\n",
          updateTypeName(LidarMeasures.lio_vio_flg),
          (phase_sync_begin-phase_ingress_begin)*1000.0,
          (phase_imu_begin-phase_sync_begin)*1000.0,
          (phase_map_begin-phase_imu_begin)*1000.0,
          (omp_get_wtime()-phase_map_begin)*1000.0,
          (omp_get_wtime()-phase_ingress_begin)*1000.0);
      }
      if (!local_fast_mode_ && omp_get_wtime() - last_resource_time >= 1.0)
      {
        const double current = this->node->now().seconds();
        livo::emitStructuredMetric(
          "[PIPELINE_RESOURCES] lidar_q=%zu imu_q=%zu camera_stage=%zu camera_q=%zu output_age_s=%.6f\n",
          lid_raw_data_buffer.size(), imu_buffer.size(), camera_transport_processing_buffer_.size(),
          camera_observation_queue_.size(), current - LidarMeasures.last_lio_update_time);
        last_resource_time = omp_get_wtime();
      }
      if (local_fast_mode_)
      {
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - local_started).count();
        const bool valid = local_body_published_ && !p_imu->imu_need_init &&
          local_imu_coverage_valid_ && _state.pos_end.allFinite() && _state.rot_end.allFinite();
        std::ostringstream status;
        status << "{\"schema_version\":1,\"cloud_stamp_ns\":\""
               << current_lio_stamp().nanoseconds() << "\",\"scan_stamp_ns\":\""
               << local_scan_stamp_ns_ << "\",\"imu_end_stamp_ns\":\""
               << local_imu_end_stamp_ns_ << "\",\"valid\":" << (valid ? "true" : "false")
               << ",\"imu_available_end_stamp_ns\":\"" << local_imu_available_end_ns_
               << "\",\"imu_integration_end_stamp_ns\":\"" << current_lio_stamp().nanoseconds()
               << "\",\"imu_max_gap_ms\":" << double(local_imu_max_gap_ns_) * 1e-6
               << ",\"processing_ms\":" << elapsed_ms
               << ",\"imu_coverage_valid\":" << (local_imu_coverage_valid_ ? "true" : "false")
               << ",\"dropped_scans\":" << local_scan_drop_count_.load()
               << ",\"map_voxels\":" << voxelmap_manager->voxel_map_.size() << "}";
        std_msgs::msg::String message;
        message.data = status.str();
        local_scan_status_pub_->publish(message);
      }
    }
  }
  catch (...)
  {
    stop_sensor_ingress();
    executor.remove_node(this->node);
    throw;
  }
  stop_sensor_ingress();
  executor.remove_node(this->node);
  if (sensor_ingress_exception != nullptr)
  {
    std::rethrow_exception(sensor_ingress_exception);
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback()
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制 propagate 频率和 IMU 频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && stamp2Sec(prop_imu_buffer.front().header.stamp) < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = stamp2Sec(prop_imu_buffer[i].header.stamp) - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = stamp2Sec(newest_imu.header.stamp) - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom->publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::transformLidarToBody(const PointCloudXYZI &input_cloud, PointCloudXYZI &body_cloud) const
{
  body_cloud = input_cloud;
  for (PointType &point : body_cloud.points)
  {
    const V3D point_lidar(point.x, point.y, point.z);
    const V3D point_body = extR * point_lidar + extT;
    point.x = point_body.x();
    point.y = point_body.y();
    point.z = point_body.z();
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Eigen::Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
  if (!lidar_en) return;
  std::optional<int64_t> native_end;
  if (local_fast_mode_)
  {
    native_end = livo::nativeScanEnd(*msg, local_allow_instantaneous_scan_);
    if (!native_end ||
        stamp2Sec(msg->header.stamp) <= last_timestamp_lidar)
    {
      ++local_scan_drop_count_;
      return;
    }
  }
  {
    std::lock_guard<std::mutex> lock(mtx_buffer);
    // cout<<"got feature"<<endl;
    if (stamp2Sec(msg->header.stamp) < last_timestamp_lidar)
    {
      RCLCPP_WARN(this->node->get_logger(),"lidar loop back, skip");
      return;
    }
    // ROS_INFO("get point cloud at time: %.6f", stamp2Sec(msg->header.stamp));
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    if (local_fast_mode_)
    {
      if (lidar_pushed && lid_raw_data_buffer.size() >= 2)
      {
        ++local_scan_drop_count_;
        return;
      }
      if (ptr->size() < 2 || !std::isfinite(ptr->back().curvature) ||
          ptr->back().curvature < 0 ||
          (ptr->back().curvature == 0 && !local_allow_instantaneous_scan_))
      {
        ++local_scan_drop_count_;
        return;
      }
      // 当前局部只保留两个待估计完整扫描；淘汰显式计数，不积压追赶。
      while (lid_raw_data_buffer.size() >= 2 && !lidar_pushed)
      {
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
        local_scan_stamps_.pop_front();
        local_scan_ends_.pop_front();
        ++local_scan_drop_count_;
      }
      local_scan_stamps_.push_back(rclcpp::Time(msg->header.stamp).nanoseconds());
      local_scan_ends_.push_back(*native_end);
    }
    lid_raw_data_buffer.push_back(ptr);
    lid_header_time_buffer.push_back(stamp2Sec(msg->header.stamp));
    last_timestamp_lidar = stamp2Sec(msg->header.stamp);
  }
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg_in)
{
  if (!lidar_en) return;
  {
    std::lock_guard<std::mutex> lock(mtx_buffer);
    livox_ros_driver2::msg::CustomMsg::SharedPtr msg(
      new livox_ros_driver2::msg::CustomMsg(*msg_in));
    // if ((abs(stamp2Sec(msg->header.stamp) - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
    // {
    //   ROS_WARN("lidar jumps %.3f\n", stamp2Sec(msg->header.stamp) - last_timestamp_lidar);
    //   sync_jump_flag = true;
    //   msg->header.stamp = rclcpp::Time().fromSec(last_timestamp_lidar + 0.1);
    // }
    if (abs(last_timestamp_imu - stamp2Sec(msg->header.stamp)) > 1.0 && !imu_buffer.empty())
    {
      double timediff_imu_wrt_lidar = last_timestamp_imu - stamp2Sec(msg->header.stamp);
      RCLCPP_INFO(this->node->get_logger(), "\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
      // imu_time_offset = timediff_imu_wrt_lidar;
    }

    double cur_head_time = stamp2Sec(msg->header.stamp);
    RCLCPP_INFO(this->node->get_logger(), "Get LiDAR, its header time: %.6f", cur_head_time);
    if (cur_head_time < last_timestamp_lidar)
    {
      RCLCPP_WARN(this->node->get_logger(), "lidar loop back, skip frame");
      return;
    }
    RCLCPP_INFO(this->node->get_logger(), "get point cloud at time: %.6f", stamp2Sec(msg->header.stamp));
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);

    if (!ptr || ptr->empty()) {
      RCLCPP_ERROR(this->node->get_logger(), "Received an empty point cloud");
      return;
    }

    lid_raw_data_buffer.push_back(ptr);
    lid_header_time_buffer.push_back(cur_head_time);
    last_timestamp_lidar = cur_head_time;
  }
  sig_buffer.notify_all();
}

void LIVMapper::imu_transport_cbk(
  const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in)
{
  bool overflow = false;
  {
    std::lock_guard<std::mutex> lock(imu_transport_mutex_);
    if (imu_transport_overflow_failed_.load(std::memory_order_acquire))
    {
      return;
    }
    if (imu_overflow_fatal_ && imu_transport_buffer_.size() >= kImuTransportBufferCapacity)
    {
      // executor 工作线程不抛异常：主线程接收持久失败位后统一 cancel/join。
      imu_transport_overflow_failed_.store(true, std::memory_order_release);
      RCLCPP_FATAL(
        this->node->get_logger(),
        "IMU transport history overflow: capacity=%zu first_received_stamp_ns=%lld "
        "last_received_stamp_ns=%lld incoming_source_stamp_ns=%lld; restart the complete launch",
        kImuTransportBufferCapacity,
        static_cast<long long>(rclcpp::Time(imu_transport_buffer_.front()->header.stamp).nanoseconds()),
        static_cast<long long>(rclcpp::Time(imu_transport_buffer_.back()->header.stamp).nanoseconds()),
        static_cast<long long>(rclcpp::Time(msg_in->header.stamp).nanoseconds()));
      return;
    }
    imu_transport_buffer_.push_back(msg_in);
    if (imu_transport_buffer_.size() > kImuTransportBufferCapacity)
    {
      imu_transport_buffer_.pop_front();
      overflow = true;
    }
  }
  if (overflow)
  {
    RCLCPP_WARN_THROTTLE(
      this->node->get_logger(), *this->node->get_clock(), 5000,
      "[fast_livo_bev] IMU transport overflow: capacity=%zu",
      kImuTransportBufferCapacity);
  }
}

void LIVMapper::drainImuTransportBuffer()
{
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> pending;
  {
    std::lock_guard<std::mutex> lock(imu_transport_mutex_);
    pending.swap(imu_transport_buffer_);
  }
  std::sort(
    pending.begin(), pending.end(),
    [](const sensor_msgs::msg::Imu::ConstSharedPtr & left,
       const sensor_msgs::msg::Imu::ConstSharedPtr & right)
    {
      return imuMessageLess(*left, *right);
    });
  for (const sensor_msgs::msg::Imu::ConstSharedPtr & message : pending)
  {
    imu_cbk(message);
  }
}

void LIVMapper::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in)
{
  if (!imu_en) return;

  sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
  if (!local_fast_mode_)
    msg->header.stamp = sec2Stamp(stamp2Sec(msg->header.stamp) - imu_time_offset);
  double timestamp = stamp2Sec(msg->header.stamp);

  if (last_timestamp_lidar >= 0.0 &&
      fabs(last_timestamp_lidar - timestamp) > 0.5 &&
      !ros_driver_fix_en)
  {
    RCLCPP_WARN_THROTTLE(
      this->node->get_logger(), *this->node->get_clock(), 1000,
      "IMU and LiDAR not synced! delta time: %lf .",
      last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en && last_timestamp_lidar >= 0.0)
  {
    timestamp += std::round(last_timestamp_lidar - timestamp);
  }
  if (!local_fast_mode_) msg->header.stamp = sec2Stamp(timestamp);

  if (camera_time_transfer_shadow_enable_ &&
      p_imu != nullptr &&
      !p_imu->imu_need_init &&
      std::isfinite(p_imu->IMU_mean_acc_norm) &&
      p_imu->IMU_mean_acc_norm > 1.0e-9)
  {
    livo::CameraTimeTransferImuSample sample;
    sample.timestamp_sec = timestamp;
    sample.angular_velocity_radps = Eigen::Vector3d(
      msg->angular_velocity.x,
      msg->angular_velocity.y,
      msg->angular_velocity.z);
    const double acceleration_scale =
      G_m_s2 / p_imu->IMU_mean_acc_norm;
    sample.specific_force_mps2 = acceleration_scale * Eigen::Vector3d(
      msg->linear_acceleration.x,
      msg->linear_acceleration.y,
      msg->linear_acceleration.z);

    livo::CameraTimeTransferAppendStatus append_status;
    {
      std::lock_guard<std::mutex> lock(camera_time_transfer_mutex_);
      append_status = camera_time_transfer_.appendImu(sample);
    }
    if (append_status ==
        livo::CameraTimeTransferAppendStatus::kNewEpoch)
    {
      RCLCPP_WARN(
        this->node->get_logger(),
        "[VIO_TIME_TRANSFER_SHADOW] new IMU epoch at %.9f",
        timestamp);
    }
    else if (
      append_status ==
      livo::CameraTimeTransferAppendStatus::kInvalidSample)
    {
      RCLCPP_WARN(
        this->node->get_logger(),
        "[VIO_TIME_TRANSFER_SHADOW] rejected IMU sample at %.9f",
        timestamp);
    }
  }

  bool algorithm_buffer_overflow = false;
  {
    std::lock_guard<std::mutex> lock(mtx_buffer);

    // 主线程 drain 在积分/发布前调用；失败时保留全部未消费历史并展开 run 的退出路径。
    if (imu_overflow_fatal_ && imu_buffer.size() >= imu_algorithm_capacity_)
    {
      RCLCPP_FATAL(
        this->node->get_logger(),
        "IMU algorithm history overflow: capacity=%zu oldest_stamp_ns=%lld "
        "newest_stamp_ns=%lld incoming_source_stamp_ns=%lld; restart the complete launch",
        imu_algorithm_capacity_,
        static_cast<long long>(rclcpp::Time(imu_buffer.front()->header.stamp).nanoseconds()),
        static_cast<long long>(rclcpp::Time(imu_buffer.back()->header.stamp).nanoseconds()),
        static_cast<long long>(rclcpp::Time(msg_in->header.stamp).nanoseconds()));
      throw std::runtime_error("IMU algorithm history overflow: restart required");
    }

    const bool out_of_order =
      last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu;
    if (out_of_order)
    {
      RCLCPP_WARN(
        this->node->get_logger(),
        "imu out of order, buffer by stamp. offset: %lf",
        last_timestamp_imu - timestamp);
    }

    // 2026-07-09 fronthall1: 修 ROS2 port 回归 bug —— 原代码 gap>0.2s 时丢弃消息且不更新
    // last_timestamp_imu, 导致录制掉帧(真实 bag 常见)后所有后续 IMU 永久被丢、EKF 饿死。
    // 上游 ROS1 版同一回调中的检查本就是注释掉的；历史实现可从 Git 基线 900c951 查阅。
    // 这里保留告警但接受消息。
    if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
    {
      RCLCPP_WARN(this->node->get_logger(), "imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
    }

    last_timestamp_imu = std::max(last_timestamp_imu, timestamp);
    const auto imu_position = std::lower_bound(
      imu_buffer.begin(), imu_buffer.end(), *msg,
      [](const sensor_msgs::msg::Imu::ConstSharedPtr & buffered,
         const sensor_msgs::msg::Imu & incoming)
      {
        return imuMessageLess(*buffered, incoming);
      });
    imu_buffer.insert(imu_position, msg);
    if (imu_buffer.size() > imu_algorithm_capacity_)
    {
      imu_buffer.pop_front();
      algorithm_buffer_overflow = true;
    }
    RCLCPP_INFO_THROTTLE(
      this->node->get_logger(), *this->node->get_clock(), 5000,
      "IMU stream: timestamp=%.6f buffer_size=%zu",
      timestamp, imu_buffer.size());
  }
  if (algorithm_buffer_overflow)
  {
    RCLCPP_WARN_THROTTLE(
      this->node->get_logger(), *this->node->get_clock(), 5000,
      "[fast_livo_bev] IMU algorithm buffer overflow: capacity=%zu",
      imu_algorithm_capacity_);
  }
  bool propagation_buffer_overflow = false;
  if (imu_prop_enable)
  {
    std::lock_guard<std::mutex> lock(mtx_buffer_imu_prop);
    if (imu_prop_enable && !p_imu->imu_need_init)
    {
      const auto prop_position = std::lower_bound(
        prop_imu_buffer.begin(), prop_imu_buffer.end(), *msg,
        [](const sensor_msgs::msg::Imu & buffered,
           const sensor_msgs::msg::Imu & incoming)
        {
          return imuMessageLess(buffered, incoming);
        });
      prop_imu_buffer.insert(prop_position, *msg);
      if (prop_imu_buffer.size() > kImuPropagationBufferCapacity)
      {
        prop_imu_buffer.pop_front();
        propagation_buffer_overflow = true;
      }
    }
    if (!new_imu || timestamp >= stamp2Sec(newest_imu.header.stamp))
    {
      newest_imu = *msg;
      new_imu = true;
    }
  }
  if (propagation_buffer_overflow)
  {
    RCLCPP_WARN_THROTTLE(
      this->node->get_logger(), *this->node->get_clock(), 5000,
      "[fast_livo_bev] IMU propagation buffer overflow: capacity=%zu",
      kImuPropagationBufferCapacity);
  }
  sig_buffer.notify_all();
}

cv::Mat LIVMapper::getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
{
  cv::Mat img;
  img = cv_bridge::toCvCopy(img_msg, "bgr8")->image;
  return img;
}

// static int i = 0;
void LIVMapper::img_cbk(const sensor_msgs::msg::Image::ConstSharedPtr &msg_in)
{
  camera_transport_cbk(msg_in, 0U);
}

void LIVMapper::handleFrontCameraImage(
  const sensor_msgs::msg::Image::ConstSharedPtr &msg_in)
{
  if (!img_en || vio_camera_mode_ != "front")
  {
    return;
  }

  const std::uint64_t source_stamp_ns =
    sourceStampNanoseconds(msg_in->header.stamp);
  const double timestamp =
    stamp2Sec(msg_in->header.stamp) + img_time_offset +
    camera_time_offsets_[0U];
  cv::Mat image;
  try
  {
    image = getImageFromMsg(msg_in);
  }
  catch (const cv_bridge::Exception &error)
  {
    RCLCPP_WARN_THROTTLE(
      this->node->get_logger(), *this->node->get_clock(), 5000,
      "[fast_livo_bev] cv_bridge front VIO: %s", error.what());
    return;
  }

  vk::AbstractCamera *front_camera = vio_manager->cameraModel(0U);
  if (front_camera != nullptr &&
      (image.cols != front_camera->width() ||
       image.rows != front_camera->height()))
  {
    cv::resize(
      image, image,
      cv::Size(front_camera->width(), front_camera->height()));
  }

  enqueueCameraObservation(
    0U, source_stamp_ns, timestamp, std::move(image));
}

CameraObservationPushResult LIVMapper::enqueueCameraObservation(
  const std::uint8_t camera_id,
  const std::uint64_t source_stamp_ns,
  const double timestamp,
  cv::Mat image)
{
  if (camera_id >= CameraObservationQueue::kCameraCount)
  {
    emitCameraSupplyReceipt(
      camera_supply_receipt_enable_, "queue_result", camera_id,
      source_stamp_ns, timestamp, "invalid_camera_id");
    return CameraObservationPushResult::INVALID_CAMERA_ID;
  }

  std::lock_guard<std::mutex> lock(mtx_buffer);
  const double last_timestamp = last_camera_timestamps_[camera_id];
  if (last_timestamp > 0.0 && timestamp >= last_timestamp &&
      timestamp - last_timestamp < 0.02)
  {
    ++dropped_camera_frames_[camera_id];
    emitCameraSupplyReceipt(
      camera_supply_receipt_enable_, "queue_result", camera_id,
      source_stamp_ns, timestamp, "rate_limited");
    return CameraObservationPushResult::RATE_LIMITED;
  }

  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.source_stamp_ns = source_stamp_ns;
  observation.timestamp = timestamp;
  observation.image = std::move(image);
  const CameraObservationPushResult result =
    camera_observation_queue_.push(std::move(observation));
  emitCameraSupplyReceipt(
    camera_supply_receipt_enable_, "queue_result", camera_id,
    source_stamp_ns, timestamp, cameraPushResultName(result));

  if (result == CameraObservationPushResult::ACCEPTED)
  {
    last_camera_timestamps_[camera_id] =
      std::max(last_timestamp, timestamp);
    last_timestamp_img = std::max(last_timestamp_img, timestamp);
    ++accepted_camera_frames_[camera_id];
    sig_buffer.notify_all();
    return result;
  }

  ++dropped_camera_frames_[camera_id];
  RCLCPP_WARN_THROTTLE(
    this->node->get_logger(), *this->node->get_clock(), 5000,
    "[fast_livo_bev] drop camera frame: id=%u reason=%s t=%.9f queue=%zu/%zu",
    static_cast<unsigned int>(camera_id), cameraPushResultName(result),
    timestamp, camera_observation_queue_.size(),
    camera_observation_queue_.capacity());
  return result;
}

void LIVMapper::handleMultiCameraImage(
  const sensor_msgs::msg::Image::ConstSharedPtr &msg,
  const std::uint8_t camera_id, vk::AbstractCamera *cam, cv::Mat &dst,
  const char *name)
{
  const bool supply_active =
    img_en &&
    (vio_camera_mode_ == "four" ||
     (vio_camera_mode_ == "front" && camera_id == 0U));
  const std::uint64_t source_stamp_ns =
    sourceStampNanoseconds(msg->header.stamp);
  const double timestamp =
    stamp2Sec(msg->header.stamp) + img_time_offset +
    camera_time_offsets_[camera_id];
  if (cam == nullptr) return;

  if (!supply_active)
  {
    std::lock_guard<std::mutex> lock(img_mtx_);
    auto & buffer = coloring_image_buffers_[camera_id];
    const auto insert_position = std::upper_bound(
      buffer.begin(), buffer.end(), timestamp,
      [this, camera_id](const double stamp,
         const sensor_msgs::msg::Image::ConstSharedPtr & buffered)
      {
        const double buffered_timestamp =
          stamp2Sec(buffered->header.stamp) + img_time_offset +
          camera_time_offsets_[camera_id];
        return stamp < buffered_timestamp;
      });
    buffer.insert(insert_position, msg);
    if (buffer.size() > kColoringImageBufferCapacity)
    {
      buffer.pop_front();
    }
    return;
  }

  cv::Mat img;
  try
  {
    img = cv_bridge::toCvCopy(msg, "bgr8")->image;
  }
  catch (const cv_bridge::Exception &e)
  {
    RCLCPP_WARN_THROTTLE(this->node->get_logger(), *this->node->get_clock(), 5000,
                         "[fast_livo_bev] cv_bridge %s: %s", name, e.what());
    return;
  }

  cv::Mat img_scaled;
  if (img.cols == cam->width() && img.rows == cam->height())
  {
    img_scaled = img;
  }
  else
  {
    cv::resize(img, img_scaled, cv::Size(cam->width(), cam->height()));
  }

  if (multi_camera_enable_)
  {
    std::lock_guard<std::mutex> lock(img_mtx_);
    dst = img_scaled;
  }

  if (supply_active)
  {
    enqueueCameraObservation(
      camera_id, source_stamp_ns, timestamp, std::move(img_scaled));
  }
}

void LIVMapper::camera_transport_cbk(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg,
  const std::uint8_t camera_id)
{
  const std::uint64_t source_stamp_ns =
    sourceStampNanoseconds(msg->header.stamp);
  const double timestamp =
    stamp2Sec(msg->header.stamp) + img_time_offset +
    camera_time_offsets_[camera_id];
  BufferedCameraMessage buffered;
  buffered.message = msg;
  buffered.source_stamp_ns = source_stamp_ns;
  buffered.timestamp = timestamp;
  buffered.camera_id = camera_id;
  bool overflow = false;
  {
    std::lock_guard<std::mutex> lock(camera_transport_mutex_);
    ++camera_transport_received_[camera_id];
    camera_transport_buffer_.push_back(std::move(buffered));
    if (camera_transport_buffer_.size() > kCameraTransportBufferCapacity)
    {
      const std::uint8_t dropped_camera_id =
        camera_transport_buffer_.front().camera_id;
      camera_transport_buffer_.pop_front();
      ++camera_transport_dropped_[dropped_camera_id];
      overflow = true;
    }
  }
  if (overflow)
  {
    RCLCPP_WARN_THROTTLE(
      this->node->get_logger(), *this->node->get_clock(), 5000,
      "[fast_livo_bev] camera transport overflow: id=%u capacity=%zu",
      static_cast<unsigned int>(camera_id),
      kCameraTransportBufferCapacity);
  }
}

void LIVMapper::drainCameraTransportBuffer()
{
  deque<BufferedCameraMessage> incoming;
  {
    std::lock_guard<std::mutex> lock(camera_transport_mutex_);
    incoming.swap(camera_transport_buffer_);
  }
  if (!incoming.empty())
  {
    camera_transport_processing_buffer_.insert(
      camera_transport_processing_buffer_.end(),
      std::make_move_iterator(incoming.begin()),
      std::make_move_iterator(incoming.end()));
    camera_transport_processing_peak_ = std::max(
      camera_transport_processing_peak_,
      camera_transport_processing_buffer_.size());
    std::sort(
      camera_transport_processing_buffer_.begin(),
      camera_transport_processing_buffer_.end(),
      [](const BufferedCameraMessage & left,
         const BufferedCameraMessage & right)
      {
        if (left.timestamp != right.timestamp)
        {
          return left.timestamp < right.timestamp;
        }
        if (left.camera_id != right.camera_id)
        {
          return left.camera_id < right.camera_id;
        }
        return left.source_stamp_ns < right.source_stamp_ns;
      });
  }
  deque<BufferedCameraMessage> ready;
  const std::size_t available_queue_slots =
    camera_observation_queue_.capacity() - camera_observation_queue_.size();
  const std::size_t ready_count = std::min(
    kCameraTransportDrainBudget,
    std::min(
      camera_transport_processing_buffer_.size(), available_queue_slots));
  for (std::size_t index = 0U; index < ready_count; ++index)
  {
    ready.push_back(
      std::move(camera_transport_processing_buffer_.front()));
    camera_transport_processing_buffer_.pop_front();
  }
  for (const BufferedCameraMessage & buffered : ready)
  {
    switch (buffered.camera_id)
    {
      case 0U:
        if (!multi_camera_enable_ && vio_camera_mode_ == "front")
        {
          handleFrontCameraImage(buffered.message);
        }
        else
        {
          handleMultiCameraImage(
            buffered.message, 0U, cam_front_, img_rgb_front, "front");
        }
        break;
      case 1U:
        handleMultiCameraImage(
          buffered.message, 1U, cam_left_, img_rgb_left, "left");
        break;
      case 2U:
        handleMultiCameraImage(
          buffered.message, 2U, cam_back_, img_rgb_back, "back");
        break;
      case 3U:
        handleMultiCameraImage(
          buffered.message, 3U, cam_right_, img_rgb_right, "right");
        break;
    }
  }
}

void LIVMapper::img_cbk_front(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  camera_transport_cbk(msg, 0U);
}

void LIVMapper::img_cbk_left(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  camera_transport_cbk(msg, 1U);
}

void LIVMapper::img_cbk_back(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  camera_transport_cbk(msg, 2U);
}

void LIVMapper::img_cbk_right(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
  camera_transport_cbk(msg, 3U);
}

bool LIVMapper::syncLioOnlyMeasurement(LidarMeasureGroup & meas)
{
  std::unique_lock<std::mutex> local_guard(mtx_buffer, std::defer_lock);
  if (local_fast_mode_) local_guard.lock();
  if (lid_raw_data_buffer.empty() ||
      lid_header_time_buffer.empty())
  {
    return false;
  }
  if (meas.last_lio_update_time < 0.0)
  {
    meas.last_lio_update_time =
      lid_header_time_buffer.front();
  }
  if (!lidar_pushed)
  {
    meas.lidar = lid_raw_data_buffer.front();
    if (meas.lidar->points.size() <= 1U)
    {
      return false;
    }
    meas.lidar_frame_beg_time =
      lid_header_time_buffer.front();
    meas.lidar_frame_end_time =
      meas.lidar_frame_beg_time +
      meas.lidar->points.back().curvature / 1000.0;
    if (local_fast_mode_)
      meas.lidar_frame_end_time = static_cast<double>(local_scan_ends_.front()) * 1e-9;
    meas.pcl_proc_cur = meas.lidar;
    lidar_pushed = true;
  }

  if (imu_en &&
      last_timestamp_imu < meas.lidar_frame_end_time)
  {
    return false;
  }

  MeasureGroup measurement;
  measurement.imu.clear();
  measurement.lio_time = meas.lidar_frame_end_time;
  if (local_fast_mode_)
  {
    local_scan_stamp_ns_ = local_scan_stamps_.front();
    local_scan_stamps_.pop_front();
    local_scan_ends_.pop_front();
    local_imu_end_stamp_ns_ = 0;
    local_imu_available_end_ns_ = imu_buffer.empty() ? 0 :
      rclcpp::Time(imu_buffer.back()->header.stamp).nanoseconds();
  }
  if (!local_fast_mode_) mtx_buffer.lock();
  int64_t maximum_imu_gap_ns = 0;
  int64_t previous_imu_ns = local_previous_imu_ns_;
  while (!imu_buffer.empty())
  {
    if (
      stamp2Sec(imu_buffer.front()->header.stamp) >
      meas.lidar_frame_end_time)
    {
      break;
    }
    measurement.imu.push_back(imu_buffer.front());
    if (local_fast_mode_)
    {
      local_imu_end_stamp_ns_ = rclcpp::Time(imu_buffer.front()->header.stamp).nanoseconds();
      if (previous_imu_ns > 0)
        maximum_imu_gap_ns = std::max(maximum_imu_gap_ns, local_imu_end_stamp_ns_ - previous_imu_ns);
      previous_imu_ns = local_imu_end_stamp_ns_;
    }
    imu_buffer.pop_front();
  }
  lid_raw_data_buffer.pop_front();
  lid_header_time_buffer.pop_front();
  if (local_fast_mode_)
  {
    const int64_t end_ns = int64_t(std::llround(meas.lidar_frame_end_time * 1e9));
    local_imu_coverage_valid_ = !measurement.imu.empty() &&
      maximum_imu_gap_ns <= 20000000 && local_imu_end_stamp_ns_ > 0 &&
      end_ns - local_imu_end_stamp_ns_ <= 10000000 && local_imu_available_end_ns_ >= end_ns;
    local_imu_max_gap_ns_ = maximum_imu_gap_ns;
    local_previous_imu_ns_ = previous_imu_ns;
  }
  else mtx_buffer.unlock();
  sig_buffer.notify_all();

  meas.lio_vio_flg = LIO;
  meas.measures.push_back(std::move(measurement));
  lidar_pushed = false;
  return true;
}

double LIVMapper::cameraIngressWatermark()
{
  double watermark = last_timestamp_imu - camera_reorder_window_sec_;
  if (camera_observation_queue_.size() == camera_observation_queue_.capacity())
  {
    // Pending images cannot be admitted until a queued batch retires. Let the
    // existing deadline close an expired batch before applying backpressure;
    // subsequent admission still accounts for stale/out-of-order images.
    return watermark;
  }
  const auto account_for_pending = [&](const BufferedCameraMessage & buffered)
  {
    if (vio_camera_mode_ == "four" || buffered.camera_id == 0U)
    {
      watermark = std::min(
        watermark, buffered.timestamp - camera_reorder_window_sec_);
    }
  };
  // A fast IMU stream cannot declare a camera missing while its image is
  // already received but waiting for the bounded decoder/admission budget.
  for (const auto & buffered : camera_transport_processing_buffer_)
  {
    if (vio_camera_mode_ == "four" || buffered.camera_id == 0U)
    {
      account_for_pending(buffered);
      break;  // Main-thread staging is sorted by timestamp.
    }
  }
  std::lock_guard<std::mutex> lock(camera_transport_mutex_);
  for (const auto & buffered : camera_transport_buffer_)
  {
    account_for_pending(buffered);
  }
  return watermark;
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  // A split scan or its scheduled visual update remains ready after the
  // original LiDAR message was removed from the ingress queue.
  if (lid_raw_data_buffer.empty() && lidar_en &&
      !(slam_mode_ == LIVO &&
        (!meas.pcl_proc_next->empty() || pending_vio_batch_ || pending_vio_observation_)))
    return false;
  const bool adaptive_single_camera_runtime =
    slam_mode_ == LIVO &&
    adaptiveSingleCameraRuntimeEnabled();
  const bool no_visual_transaction =
    camera_observation_queue_.empty() &&
    !pending_vio_observation_ &&
    !pending_vio_batch_;
  const bool adaptive_single_channel_lio_only =
    no_visual_transaction &&
    adaptive_single_camera_runtime;
  if (no_visual_transaction && img_en &&
      !adaptive_single_channel_lio_only)
    return false;
  if (imu_buffer.empty() && imu_en) return false;
  if (adaptive_single_channel_lio_only)
  {
    if (!imu_en ||
        lid_raw_data_buffer.empty() ||
        lid_header_time_buffer.empty() ||
        !lid_raw_data_buffer.front() ||
        lid_raw_data_buffer.front()->points.size() <= 1U ||
        !std::isfinite(last_timestamp_imu) ||
        !std::isfinite(camera_reorder_window_sec_) ||
        !std::isfinite(adaptive_missing_camera_timeout_sec_))
    {
      return false;
    }

    const double lidar_scan_end =
      lid_header_time_buffer.front() +
      lid_raw_data_buffer.front()->points.back().curvature / 1000.0;
    const double visual_watermark = cameraIngressWatermark();
    const double visual_fallback_deadline =
      lidar_scan_end + adaptive_missing_camera_timeout_sec_;
    if (!std::isfinite(lidar_scan_end) ||
        visual_watermark < visual_fallback_deadline)
    {
      return false;
    }
    const bool synchronized = syncLioOnlyMeasurement(meas);
    if (synchronized)
    {
      adaptive_lio_only_fallback_active_ = true;
    }
    return synchronized;
  }

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    return syncLioOnlyMeasurement(meas);
  }

  case LIVO:
  {
    /*** For LIVO mode, the time of LIO update is set to be the same as VIO, LIO
     * first than VIO imediatly ***/
    const bool visual_recovered_after_lio_only_fallback =
      adaptive_lio_only_fallback_active_ &&
      !camera_observation_queue_.empty() &&
      !pending_vio_observation_ &&
      !pending_vio_batch_;
    if (visual_recovered_after_lio_only_fallback)
    {
      adaptive_lio_only_fallback_active_ = false;
      if (meas.lio_vio_flg == LIO)
      {
        meas.lio_vio_flg = WAIT;
        RCLCPP_INFO(
          this->node->get_logger(),
          "[ Data Cut ] Visual input recovered after LIO-only fallback; "
          "resume camera scheduling");
      }
    }
    EKF_STATE last_lio_vio_flg = meas.lio_vio_flg;
    // double t0 = omp_get_wtime();
    switch (last_lio_vio_flg)
    {
    // double img_capture_time = meas.lidar_frame_beg_time + exposure_time_init;
    case WAIT:
    case VIO:
    {
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      const bool uses_camera_batch =
        vio_camera_mode_ == "four" || adaptiveSingleCameraRuntimeEnabled();
      const bool adaptive_partial_camera =
        uses_camera_batch &&
        (adaptive_partial_batch_enable_ ||
         adaptive_policy_config_.enabled);
      const std::size_t active_camera_count =
        vio_camera_mode_ == "four" ?
        CameraObservationQueue::kCameraCount : 1U;
      const double camera_watermark =
        adaptive_partial_camera ?
        camera_observation_queue_.adaptiveSynchronizedWatermark(
          cameraIngressWatermark(), active_camera_count,
          camera_reorder_window_sec_, camera_sync_tolerance_sec_,
          adaptive_missing_camera_timeout_sec_) :
        uses_camera_batch ?
        camera_observation_queue_.synchronizedWatermark(
          last_timestamp_imu - camera_reorder_window_sec_,
          active_camera_count,
          camera_reorder_window_sec_) :
        std::numeric_limits<double>::infinity();

      CameraObservationBatch next_batch;
      const CameraObservation *next_observation = nullptr;
      double img_capture_time = 0.0;
      auto peek_next_camera_measurement = [&]() -> bool
      {
        if (uses_camera_batch)
        {
          if (adaptive_partial_camera)
          {
            const AdaptiveCameraBatchResult result =
              camera_observation_queue_.nextAdaptiveSynchronized(
              camera_watermark, camera_sync_tolerance_sec_,
              adaptive_missing_camera_timeout_sec_,
              adaptive_policy_config_.minimum_selected_cameras,
              next_batch);
            if (result != AdaptiveCameraBatchResult::READY)
            {
              return false;
            }
          }
          else if (!camera_observation_queue_.nextSynchronized(
              camera_watermark, camera_sync_tolerance_sec_, next_batch))
          {
            return false;
          }
          img_capture_time = next_batch.timestamp + exposure_time_init;
          return true;
        }

        next_observation =
          camera_observation_queue_.nextReady(camera_watermark);
        if (next_observation == nullptr)
        {
          return false;
        }
        img_capture_time =
          next_observation->timestamp + exposure_time_init;
        return true;
      };
      const auto pop_next_camera_batch =
        [&](CameraObservationBatch & batch,
            std::vector<CameraObservation> & discarded) -> bool
        {
          std::vector<CameraObservation> * discarded_output =
            camera_supply_receipt_enable_ ? &discarded : nullptr;
          if (adaptive_partial_camera)
          {
            return
              camera_observation_queue_.popAdaptiveSynchronized(
              camera_watermark, camera_sync_tolerance_sec_,
              adaptive_missing_camera_timeout_sec_,
              adaptive_policy_config_.minimum_selected_cameras,
              batch, discarded_output) ==
              AdaptiveCameraBatchResult::READY;
          }
          return camera_observation_queue_.popSynchronized(
            camera_watermark, camera_sync_tolerance_sec_, batch,
            discarded_output);
        };
      if (!peek_next_camera_measurement()) return false;

      /*** has img topic, but img topic timestamp larger than lidar end time,
       * process lidar topic. After LIO update, the meas.lidar_frame_end_time
       * will be refresh. ***/
      if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
      const double joint_warmup_reference_time =
        is_first_frame ? _first_lidar_time : meas.last_lio_update_time;
      const double joint_warmup_end_time =
        joint_warmup_reference_time + joint_vio_warmup_sec_;
      const auto before_joint_warmup = [&]() -> bool
      {
        return uses_camera_batch &&
               !adaptive_policy_config_.enabled &&
               img_capture_time < joint_warmup_end_time - 0.00001;
      };
      // printf("[ Data Cut ] wait \n");
      // printf("[ Data Cut ] last_lio_update_time: %lf \n",
      // meas.last_lio_update_time);

      double lid_newest_time = meas.last_lio_update_time;
      if (!meas.pcl_proc_next->empty())
        lid_newest_time += meas.pcl_proc_next->points.back().curvature / 1000.0;
      if (!lid_raw_data_buffer.empty())
        lid_newest_time = std::max(lid_newest_time,
          lid_header_time_buffer.back() + lid_raw_data_buffer.back()->points.back().curvature / 1000.0);
      double imu_newest_time = stamp2Sec(imu_buffer.back()->header.stamp);

      while (img_capture_time < meas.last_lio_update_time - 0.00001 ||
             before_joint_warmup())
      {
        if (uses_camera_batch)
        {
          CameraObservationBatch stale_batch;
          std::vector<CameraObservation> discarded;
          if (!pop_next_camera_batch(stale_batch, discarded))
          {
            return false;
          }
          emitDiscardedCameraSupplyOutcomes(
            camera_supply_receipt_enable_, discarded);
          emitCameraBatchSupplyOutcome(
            camera_supply_receipt_enable_, stale_batch,
            "stale_rejected");
          for (std::size_t camera_id = 0U;
               camera_id < CameraObservationQueue::kCameraCount; ++camera_id)
          {
            if (stale_batch.present[camera_id])
            {
              ++dropped_camera_frames_[camera_id];
            }
          }
          RCLCPP_WARN(
            this->node->get_logger(),
            "[ Data Cut ] Drop stale joint camera batch: image=%.9f "
            "state=%.9f skew=%.3fms",
            img_capture_time, meas.last_lio_update_time,
            stale_batch.max_time_offset_sec * 1000.0);
        }
        else
        {
          CameraObservation stale_observation;
          if (!camera_observation_queue_.pop(stale_observation))
          {
            return false;
          }
          const unsigned int stale_camera_id = static_cast<unsigned int>(
            stale_observation.camera_id);
          emitCameraSupplyReceipt(
            camera_supply_receipt_enable_, "batch_outcome",
            stale_observation.camera_id,
            stale_observation.source_stamp_ns,
            stale_observation.timestamp, "stale_rejected");
          ++dropped_camera_frames_[stale_camera_id];
          RCLCPP_WARN(
            this->node->get_logger(),
            "[ Data Cut ] Drop stale camera frame: id=%u image=%.9f state=%.9f",
            stale_camera_id, img_capture_time, meas.last_lio_update_time);
        }
        if (!peek_next_camera_measurement()) return false;
      }

      if (std::abs(img_capture_time - meas.last_lio_update_time) <= 0.00001)
      {
        struct MeasureGroup m;
        m.vio_time = img_capture_time;
        m.lio_time = meas.last_lio_update_time;
        if (uses_camera_batch)
        {
          std::vector<CameraObservation> discarded;
          if (!pop_next_camera_batch(m.camera_batch, discarded))
          {
            return false;
          }
          emitDiscardedCameraSupplyOutcomes(
            camera_supply_receipt_enable_, discarded);
          emitCameraBatchSupplyOutcome(
            camera_supply_receipt_enable_, m.camera_batch,
            "batch_admitted");
        }
        else
        {
          if (!camera_observation_queue_.pop(m.camera))
          {
            return false;
          }
          emitCameraSupplyReceipt(
            camera_supply_receipt_enable_, "batch_outcome",
            m.camera.camera_id, m.camera.source_stamp_ns,
            m.camera.timestamp, "batch_admitted");
        }
        meas.measures.clear();
        meas.measures.push_back(std::move(m));
        meas.lio_vio_flg = VIO;
        return true;
      }

      if (img_capture_time > lid_newest_time || img_capture_time > imu_newest_time)
      {
        // RCLCPP_ERROR(this->node->get_logger(), "lost first camera frame");
        // printf("img_capture_time, lid_newest_time, imu_newest_time: %lf , %lf
        // , %lf \n", img_capture_time, lid_newest_time, imu_newest_time);
        return false;
      }

      if (uses_camera_batch)
      {
        CameraObservationBatch scheduled_batch;
        std::vector<CameraObservation> discarded;
        if (!pop_next_camera_batch(scheduled_batch, discarded))
        {
          return false;
        }
        emitDiscardedCameraSupplyOutcomes(
          camera_supply_receipt_enable_, discarded);
        emitCameraBatchSupplyOutcome(
          camera_supply_receipt_enable_, scheduled_batch,
          "batch_admitted");
        pending_vio_batch_ = std::move(scheduled_batch);
      }
      else
      {
        CameraObservation scheduled_observation;
        if (!camera_observation_queue_.pop(scheduled_observation))
        {
          return false;
        }
        emitCameraSupplyReceipt(
          camera_supply_receipt_enable_, "batch_outcome",
          scheduled_observation.camera_id,
          scheduled_observation.source_stamp_ns,
          scheduled_observation.timestamp, "batch_admitted");
        pending_vio_observation_ = std::move(scheduled_observation);
      }

      struct MeasureGroup m;

      // printf("[ Data Cut ] LIO \n");
      // printf("[ Data Cut ] img_capture_time: %lf \n", img_capture_time);
      m.imu.clear();
      m.lio_time = img_capture_time;
      mtx_buffer.lock();
      while (!imu_buffer.empty())
      {
        if (stamp2Sec(imu_buffer.front()->header.stamp) > m.lio_time) break;

        if (stamp2Sec(imu_buffer.front()->header.stamp) > meas.last_lio_update_time) m.imu.push_back(imu_buffer.front());

        imu_buffer.pop_front();
        // printf("[ Data Cut ] imu time: %lf \n",
        // stamp2Sec(imu_buffer.front()->header.stamp));
      }
      mtx_buffer.unlock();
      sig_buffer.notify_all();

      // Retained points are relative to the previous LIO cutoff. Split them
      // again at every camera exposure; otherwise future points are consumed
      // early and the third 30 Hz update in a 10 Hz scan has no points left.
      PointCloudXYZI retained_scan;
      retained_scan.swap(*meas.pcl_proc_next);
      meas.pcl_proc_cur->clear();
      std::size_t slice_capacity = retained_scan.size();
      for (std::size_t i = 0; i < lid_raw_data_buffer.size() &&
           lid_header_time_buffer[i] <= m.lio_time; ++i)
        slice_capacity += lid_raw_data_buffer[i]->size();
      meas.pcl_proc_cur->reserve(slice_capacity);
      meas.pcl_proc_next->reserve(slice_capacity);
      const float retained_cut_ms = (m.lio_time - meas.last_lio_update_time) * 1000.0;
      for (auto point : retained_scan.points)
      {
        if (point.curvature < retained_cut_ms)
          meas.pcl_proc_cur->push_back(point);
        else
        {
          point.curvature -= retained_cut_ms;
          meas.pcl_proc_next->push_back(point);
        }
      }

      // deque<PointCloudXYZI::Ptr> lidar_buffer_tmp;

      while (!lid_raw_data_buffer.empty())
      {
        if (lid_header_time_buffer.front() > img_capture_time) break;
        auto pcl(lid_raw_data_buffer.front()->points);
        double frame_header_time(lid_header_time_buffer.front());
        float max_offs_time_ms = (m.lio_time - frame_header_time) * 1000.0f;

        for (int i = 0; i < pcl.size(); i++)
        {
          auto pt = pcl[i];
          if (pcl[i].curvature < max_offs_time_ms)
          {
            pt.curvature += (frame_header_time - meas.last_lio_update_time) * 1000.0f;
            meas.pcl_proc_cur->points.push_back(pt);
          }
          else
          {
            pt.curvature += (frame_header_time - m.lio_time) * 1000.0f;
            meas.pcl_proc_next->points.push_back(pt);
          }
        }
        lid_raw_data_buffer.pop_front();
        lid_header_time_buffer.pop_front();
      }

      const auto latest_point = std::max_element(
        meas.pcl_proc_cur->begin(), meas.pcl_proc_cur->end(),
        [](const PointType &a, const PointType &b) { return a.curvature < b.curvature; });
      livo::emitStructuredMetric(
        "[LIDAR_SLICE] begin=%.9f end=%.9f cut_ms=%.6f current_points=%zu retained_points=%zu max_current_offset_ms=%.6f\n",
        meas.last_lio_update_time, m.lio_time, retained_cut_ms,
        meas.pcl_proc_cur->size(), meas.pcl_proc_next->size(),
        latest_point == meas.pcl_proc_cur->end() ? 0.0 : latest_point->curvature);
      meas.measures.push_back(m);
      meas.lio_vio_flg = LIO;
      // meas.last_lio_update_time = m.lio_time;
      // printf("!!! meas.lio_vio_flg: %d \n", meas.lio_vio_flg);
      // printf("[ Data Cut ] pcl_proc_cur number: %d \n", meas.pcl_proc_cur
      // ->points.size()); printf("[ Data Cut ] LIO process time: %lf \n",
      // omp_get_wtime() - t0);
      return true;
    }

    case LIO:
    {
      const bool uses_camera_batch =
        vio_camera_mode_ == "four" || adaptiveSingleCameraRuntimeEnabled();
      if ((uses_camera_batch && !pending_vio_batch_) ||
          (!uses_camera_batch && !pending_vio_observation_))
      {
        RCLCPP_ERROR(
          this->node->get_logger(),
          "[ Data Cut ] LIO/VIO state lost its scheduled camera measurement");
        meas.lio_vio_flg = VIO;
        return false;
      }
      const double img_capture_time =
        (uses_camera_batch ? pending_vio_batch_->timestamp :
         pending_vio_observation_->timestamp) + exposure_time_init;
      meas.lio_vio_flg = VIO;
      // printf("[ Data Cut ] VIO \n");
      meas.measures.clear();

      struct MeasureGroup m;
      m.vio_time = img_capture_time;
      m.lio_time = meas.last_lio_update_time;
      if (uses_camera_batch)
      {
        m.camera_batch = std::move(*pending_vio_batch_);
        pending_vio_batch_.reset();
      }
      else
      {
        m.camera = std::move(*pending_vio_observation_);
        pending_vio_observation_.reset();
      }
      sig_buffer.notify_all();
      meas.measures.push_back(std::move(m));
      lidar_pushed = false; // after VIO update, the _lidar_frame_end_time will be refresh.
      // printf("[ Data Cut ] VIO process time: %lf \n", omp_get_wtime() - t0);
      return true;
    }

    default:
    {
      // printf("!! WRONG EKF STATE !!");
      return false;
    }
      // return false;
    }
    break;
  }

  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  RCLCPP_ERROR(this->node->get_logger(), "out sync");
}

void LIVMapper::publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager)
{
  cv::Mat img_rgb = vio_manager->img_cp;
  cv_bridge::CvImage out_msg;
  out_msg.header.stamp = this->node->get_clock()->now();
  // out_msg.header.frame_id = "camera_init";
  out_msg.encoding = sensor_msgs::image_encodings::BGR8;
  out_msg.image = img_rgb;
  pubImage.publish(out_msg.toImageMsg());
}

void LIVMapper::publish_persistent_maps(
  const PointCloudXYZI &raw_cloud, const PointCloudXYZRGB &rgb_cloud,
  const bool rgb_cloud_ready, const rclcpp::Time &stamp)
{
  if (!persistent_map_en_ || persistent_lio_map_ == nullptr ||
      persistent_rgb_map_ == nullptr || pubPersistentLioMap == nullptr ||
      pubPersistentRGBMap == nullptr)
  {
    return;
  }

  *persistent_lio_map_ += raw_cloud;
  if (rgb_cloud_ready && !rgb_cloud.empty())
  {
    *persistent_rgb_map_ += rgb_cloud;
  }

  ++persistent_map_scan_count_;
  if (persistent_map_published_ &&
      persistent_map_scan_count_ < persistent_map_publish_interval_)
  {
    return;
  }
  persistent_map_scan_count_ = 0;
  persistent_map_published_ = true;

  pcl::VoxelGrid<PointType> raw_voxel_filter;
  raw_voxel_filter.setInputCloud(persistent_lio_map_);
  raw_voxel_filter.setLeafSize(
    persistent_map_voxel_size_, persistent_map_voxel_size_, persistent_map_voxel_size_);
  raw_voxel_filter.setDownsampleAllData(true);
  PointCloudXYZI compact_raw_map;
  raw_voxel_filter.filter(compact_raw_map);
  persistent_lio_map_->swap(compact_raw_map);

  sensor_msgs::msg::PointCloud2 raw_map_msg;
  pcl::toROSMsg(*persistent_lio_map_, raw_map_msg);
  raw_map_msg.header.stamp = stamp;
  raw_map_msg.header.frame_id = "camera_init";
  pubPersistentLioMap->publish(raw_map_msg);

  if (!persistent_rgb_map_->empty())
  {
    pcl::VoxelGrid<PointTypeRGB> rgb_voxel_filter;
    rgb_voxel_filter.setInputCloud(persistent_rgb_map_);
    rgb_voxel_filter.setLeafSize(
      persistent_map_voxel_size_, persistent_map_voxel_size_, persistent_map_voxel_size_);
    rgb_voxel_filter.setDownsampleAllData(true);
    PointCloudXYZRGB compact_rgb_map;
    rgb_voxel_filter.filter(compact_rgb_map);
    persistent_rgb_map_->swap(compact_rgb_map);

    sensor_msgs::msg::PointCloud2 rgb_map_msg;
    pcl::toROSMsg(*persistent_rgb_map_, rgb_map_msg);
    rgb_map_msg.header = raw_map_msg.header;
    pubPersistentRGBMap->publish(rgb_map_msg);
  }

  RCLCPP_INFO_THROTTLE(
    this->node->get_logger(), *this->node->get_clock(), 5000,
    "[fast_livo_bev] persistent maps: %zu LIO points, %zu RGB points",
    persistent_lio_map_->size(), persistent_rgb_map_->size());
}

void LIVMapper::publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes, VIOManagerPtr vio_manager)
{
  if (pcl_w_wait_pub->empty()) return;
  const bool coloring_enabled = img_en || multi_camera_enable_;
  if (
    coloring_enabled && !multi_camera_enable_ &&
    (!vio_manager || vio_manager->new_frame_ == nullptr))
  {
    return;
  }
  PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB());
  bool coloring_batch_ready = false;
  if (coloring_enabled)
  {
    static int pub_num = 1;
    *pcl_wait_pub += *pcl_w_wait_pub;
    if(pub_num == pub_scan_num)
    {
      coloring_batch_ready = true;
      pub_num = 1;
      size_t size = pcl_wait_pub->points.size();
      laserCloudWorldRGB->reserve(size);
      if (multi_camera_enable_)
      {
        cv::Mat imgs[4];
        std::array<
          sensor_msgs::msg::Image::ConstSharedPtr,
          CameraObservationQueue::kCameraCount> coloring_messages;
        vk::AbstractCamera *cams[4] = {cam_front_, cam_left_, cam_back_, cam_right_};
        Sophus::SE3<double> T_base_cams[4] = {
          T_base_cam_front_, T_base_cam_left_, T_base_cam_back_, T_base_cam_right_
        };
        {
          std::lock_guard<std::mutex> lock(img_mtx_);
          if (!img_rgb_front.empty()) imgs[0] = img_rgb_front.clone();
          if (!img_rgb_left.empty()) imgs[1] = img_rgb_left.clone();
          if (!img_rgb_back.empty()) imgs[2] = img_rgb_back.clone();
          if (!img_rgb_right.empty()) imgs[3] = img_rgb_right.clone();
          const double coloring_time = LidarMeasures.last_lio_update_time;
          for (std::size_t camera_id = 0U;
               camera_id < coloring_image_buffers_.size(); ++camera_id)
          {
            auto & buffer = coloring_image_buffers_[camera_id];
            while (buffer.size() > 1U &&
                   stamp2Sec(buffer[1U]->header.stamp) + img_time_offset +
                   camera_time_offsets_[camera_id] <= coloring_time)
            {
              buffer.pop_front();
            }
            if (!buffer.empty() &&
                stamp2Sec(buffer.front()->header.stamp) + img_time_offset +
                camera_time_offsets_[camera_id] <= coloring_time)
            {
              coloring_messages[camera_id] = buffer.front();
            }
          }
        }
        for (std::size_t camera_id = 0U;
             camera_id < coloring_messages.size(); ++camera_id)
        {
          if (!coloring_messages[camera_id] || cams[camera_id] == nullptr)
          {
            continue;
          }
          try
          {
            imgs[camera_id] = getImageFromMsg(coloring_messages[camera_id]);
          }
          catch (const cv_bridge::Exception & error)
          {
            RCLCPP_WARN_THROTTLE(
              this->node->get_logger(), *this->node->get_clock(), 5000,
              "[fast_livo_bev] cv_bridge coloring camera %zu: %s",
              camera_id, error.what());
            continue;
          }
          if (imgs[camera_id].cols != cams[camera_id]->width() ||
              imgs[camera_id].rows != cams[camera_id]->height())
          {
            cv::resize(
              imgs[camera_id], imgs[camera_id],
              cv::Size(
                cams[camera_id]->width(), cams[camera_id]->height()));
          }
        }

        Sophus::SE3<double> T_world_base(_state.rot_end, _state.pos_end);
        std::array<Sophus::SE3<double>, 4> T_cam_world;
        for (int c = 0; c < 4; ++c)
          T_cam_world[c] = (T_world_base * T_base_cams[c]).inverse();
        laserCloudWorldRGB->resize(size);
        std::vector<std::uint8_t> colored(size, 0U);
        const int coloring_threads = std::clamp(vio_manager->adaptive_camera_threads, 1, 4);
        #pragma omp parallel for num_threads(coloring_threads) schedule(static) if(coloring_threads > 1)
        for (size_t i = 0; i < size; i++)
        {
          V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
          double best_depth = std::numeric_limits<double>::max();
          int best_cam_idx = -1;
          V2D best_pc;

          for (int c = 0; c < 4; c++)
          {
            if (cams[c] == nullptr || imgs[c].empty()) continue;
            Eigen::Vector3d p_cam = T_cam_world[c] * p_w;
            if (p_cam.z() <= 0.0) continue;
            V2D pc = cams[c]->world2cam(p_cam);
            if (!cams[c]->isInFrame(pc.cast<int>(), 3)) continue;
            if (p_cam.z() < best_depth)
            {
              best_depth = p_cam.z();
              best_cam_idx = c;
              best_pc = pc;
            }
          }

          if (best_cam_idx >= 0)
          {
            V3F pixel = vio_manager->getInterpolatedPixel(imgs[best_cam_idx], best_pc);
            PointTypeRGB pointRGB;
            pointRGB.x = pcl_wait_pub->points[i].x;
            pointRGB.y = pcl_wait_pub->points[i].y;
            pointRGB.z = pcl_wait_pub->points[i].z;
            pointRGB.r = pixel[2];
            pointRGB.g = pixel[1];
            pointRGB.b = pixel[0];
            if (p_w.norm() > blind_rgb_points)
            {
              laserCloudWorldRGB->points[i] = pointRGB;
              colored[i] = 1U;
            }
          }
        }
        std::size_t colored_count = 0;
        for (std::size_t i = 0; i < size; ++i)
          if (colored[i]) laserCloudWorldRGB->points[colored_count++] = laserCloudWorldRGB->points[i];
        laserCloudWorldRGB->resize(colored_count);
      }
      else
      {
        cv::Mat img_rgb = vio_manager->img_rgb;
        for (size_t i = 0; i < size; i++)
        {
          PointTypeRGB pointRGB;
          pointRGB.x = pcl_wait_pub->points[i].x;
          pointRGB.y = pcl_wait_pub->points[i].y;
          pointRGB.z = pcl_wait_pub->points[i].z;

          V3D p_w(pcl_wait_pub->points[i].x, pcl_wait_pub->points[i].y, pcl_wait_pub->points[i].z);
          V3D pf(vio_manager->new_frame_->w2f(p_w)); if (pf[2] < 0) continue;
          V2D pc(vio_manager->new_frame_->w2c(p_w));

          if (vio_manager->new_frame_->cam_->isInFrame(pc.cast<int>(), 3)) // 100
          {
            V3F pixel = vio_manager->getInterpolatedPixel(img_rgb, pc);
            pointRGB.r = pixel[2];
            pointRGB.g = pixel[1];
            pointRGB.b = pixel[0];
            if (pf.norm() > blind_rgb_points) laserCloudWorldRGB->push_back(pointRGB);
          }
        }
      }
    }
    else
    {
      pub_num++;
    }
  }

  /*** Publish Frame ***/
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  if (coloring_enabled)
  {
    if (laserCloudWorldRGB->empty())
    {
      publish_persistent_maps(
        *pcl_w_wait_pub, *laserCloudWorldRGB, coloring_batch_ready,
        current_lio_stamp());
      if (coloring_batch_ready)
      {
        PointCloudXYZI().swap(*pcl_wait_pub);
      }
      PointCloudXYZI().swap(*pcl_w_wait_pub);
      return;
    }
    // cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
    pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
  }
  else 
  { 
    pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg); 
  }
  laserCloudmsg.header.stamp = current_lio_stamp();
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes->publish(laserCloudmsg);

  if (coloring_batch_ready && !laserCloudWorldRGB->empty())
  {
    sensor_msgs::msg::PointCloud2 rgb_cloud_msg;
    pcl::toROSMsg(*laserCloudWorldRGB, rgb_cloud_msg);
    rgb_cloud_msg.header = laserCloudmsg.header;
    pubLaserCloudRGB->publish(rgb_cloud_msg);
  }
  publish_persistent_maps(
    *pcl_w_wait_pub, *laserCloudWorldRGB, coloring_batch_ready,
    laserCloudmsg.header.stamp);

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en)
  {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    static int scan_wait_num = 0;

    if (coloring_enabled)
    {
      *pcl_wait_save += *laserCloudWorldRGB;
    }
    else
    {
      *pcl_wait_save_intensity += *pcl_w_wait_pub;
    }
    scan_wait_num++;

    if ((pcl_wait_save->size() > 0 || pcl_wait_save_intensity->size() > 0) && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      string all_points_dir(string(string(ROOT_DIR) + "Log/PCD/") + to_string(pcd_index) + string(".pcd"));
      pcl::PCDWriter pcd_writer;
      if (pcd_save_en)
      {
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        if (coloring_enabled)
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // pcl::io::savePCDFileASCII(all_points_dir, *pcl_wait_save);
          PointCloudXYZRGB().swap(*pcl_wait_save);
        }
        else
        {
          pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_intensity);
          PointCloudXYZI().swap(*pcl_wait_save_intensity);
        }        
        Eigen::Quaterniond q(_state.rot_end);
        fout_pcd_pos << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " " << q.w() << " " << q.x() << " " << q.y()
                     << " " << q.z() << " " << endl;
        scan_wait_num = 0;
      }
    }
  }
  if(laserCloudWorldRGB->size() > 0)  PointCloudXYZI().swap(*pcl_wait_pub); 
  PointCloudXYZI().swap(*pcl_w_wait_pub);
}

void LIVMapper::publish_frame_body(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudBody)
{
  const PointCloudXYZI::Ptr &published_cloud =
    (local_fast_mode_ ? local_body_dense_ : dense_map_en) ? feats_undistort : feats_down_body;
  if (!pubLaserCloudBody || published_cloud == nullptr || published_cloud->empty()) return;

  PointCloudXYZI cloud_body;
  transformLidarToBody(*published_cloud, cloud_body);

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(cloud_body, laserCloudmsg);
  laserCloudmsg.header.stamp = current_lio_stamp();
  laserCloudmsg.header.frame_id = local_fast_mode_ ? "local_body" : "aft_mapped";
  pubLaserCloudBody->publish(laserCloudmsg);
  local_body_published_ = true;
}

void LIVMapper::publish_visual_sub_map(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubSubVisualMap)
{
  PointCloudXYZI::Ptr laserCloudFullRes(visual_sub_map);
  int size = laserCloudFullRes->points.size(); if (size == 0) return;
  PointCloudXYZI::Ptr sub_pcl_visual_map_pub(new PointCloudXYZI());
  *sub_pcl_visual_map_pub = *laserCloudFullRes;
  if (1)
  {
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*sub_pcl_visual_map_pub, laserCloudmsg);
    laserCloudmsg.header.stamp = current_lio_stamp();
    laserCloudmsg.header.frame_id = "camera_init";
    pubSubVisualMap->publish(laserCloudmsg);
  }
}

void LIVMapper::publish_effect_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = current_lio_stamp();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect->publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

rclcpp::Time LIVMapper::current_lio_stamp() const
{
  const double seconds = LidarMeasures.last_lio_update_time;
  if (!std::isfinite(seconds) || seconds <= 0.0)
  {
    if (local_fast_mode_) return rclcpp::Time(0, 0, RCL_ROS_TIME);
    return this->node->get_clock()->now();
  }
  return rclcpp::Time(
    static_cast<int64_t>(std::llround(seconds * 1e9)), RCL_ROS_TIME);
}

void LIVMapper::publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = local_fast_mode_ ? "local_odom" : "camera_init";
  odomAftMapped.child_frame_id = local_fast_mode_ ? "local_body" : "aft_mapped";
  odomAftMapped.header.stamp = current_lio_stamp();
  set_posestamp(odomAftMapped.pose.pose);
  if (local_fast_mode_)
  {
    pubOdomAftMapped->publish(odomAftMapped);
    return;
  }

  // Keep one publisher per mapper so TF discovery/history survives pose updates.
  if (!odom_tf_broadcaster_)
  {
    odom_tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this->node);
  }
  tf2::Transform transform;
  tf2::Quaternion q;
  transform.setOrigin(tf2::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  odom_tf_broadcaster_->sendTransform(geometry_msgs::msg::TransformStamped(createTransformStamped(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped")));
  pubOdomAftMapped->publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = current_lio_stamp();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher->publish(msg_body_pose);
}

void LIVMapper::publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = current_lio_stamp();
  msg_body_pose.header.frame_id = "camera_init";
  path.header.stamp = msg_body_pose.header.stamp;
  path.poses.push_back(msg_body_pose);
  // Keep every pose, but serialize the growing history only for viewers at
  // 1 Hz. Current odometry and clouds retain their full estimator rate.
  static double last_path_publish_sec = 0.0;
  const double now = omp_get_wtime();
  if (pubPath->get_subscription_count() == 0 || now - last_path_publish_sec < 1.0)
    return;
  pubPath->publish(path);
  last_path_publish_sec = now;
}
