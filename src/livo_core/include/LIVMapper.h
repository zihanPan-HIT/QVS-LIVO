/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "adaptive_lio_mode_ledger.h"
#include "camera_batch_time_transfer.h"
#include "camera_model_factory.h"
#include "vio.h"
#include "preprocess.h"
#ifdef PRE_ROS_IRON
#include <cv_bridge/cv_bridge.h>
#else
#include <cv_bridge/cv_bridge.hpp>
#endif
#include <image_transport/image_transport.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <optional>

namespace livo
{

enum class AdaptiveVisualLidarRuntimeInitializationStatus
{
  kNotRequired,
  kReady,
  kBatchSealingRejected,
  kAuthorityUnavailable,
  kResourceFailure,
};

struct AdaptiveVisualLidarRuntimeInitializationAudit
{
  AdaptiveVisualLidarRuntimeInitializationStatus status{
    AdaptiveVisualLidarRuntimeInitializationStatus::
    kNotRequired};
  bool adaptive_single_camera_runtime{false};
  bool r2b_runtime{false};
  bool authority_required{false};
  std::size_t batch_sealing_request_count{0U};
  std::size_t authority_issue_count{0U};
  bool authority_valid{false};
};

enum class AdaptiveVisualLidarRuntimeDispatchStatus
{
  kNotApplicable,
  kNoCachedLidarFeatures,
  kAuthorityUnavailable,
  kZeroLidarGeneration,
  kNonFiniteLidarInformation,
  kIssueRejected,
  kRuntimeInvariantViolation,
  kVioDispatched,
  kResourceFailure,
  kUnexpectedFailure,
};

struct AdaptiveVisualLidarPosteriorSnapshot
{
  std::uint64_t generation{0U};
  AdaptiveVisualPolicy::Matrix6 pose_information{
    AdaptiveVisualPolicy::Matrix6::Zero()};
  bool information_ready{false};
};

struct AdaptiveVisualLidarRuntimeDispatchAudit
{
  AdaptiveVisualLidarRuntimeDispatchStatus status{
    AdaptiveVisualLidarRuntimeDispatchStatus::
    kNotApplicable};
  std::optional<AdaptiveVisualSelectionStatus>
    selection_status;
  std::optional<AdaptiveVisualLidarPosteriorSnapshot>
    lidar_posterior;
  bool generation_issued{false};
  bool runtime_bound{false};
  bool vio_dispatch_started{false};
  bool vio_dispatch_completed{false};
  std::size_t pre_vio_shadow_entry_count{0U};
  std::size_t post_vio_shadow_entry_count{0U};
};

struct AdaptiveLioModePublicationAudit
{
  AdaptiveLioModeRecordStatus status{
    AdaptiveLioModeRecordStatus::kZeroGeneration};
  std::optional<AdaptiveLioModeReceipt> receipt;
  bool metric_emitted{false};
};

}  // namespace livo

class LIVMapper
{
public:
  LIVMapper(rclcpp::Node::SharedPtr &node, std::string node_name, const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~LIVMapper();
  void initializeSubscribersAndPublishers(rclcpp::Node::SharedPtr &nh, image_transport::ImageTransport &it_);
  void initializeComponents(rclcpp::Node::SharedPtr &node);
  void initializeFiles();
  void run(rclcpp::Node::SharedPtr &node);
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleVIO();
  void handleLIO();
  void savePCD();
  void processImu();
  // Both audits have one mapper-thread writer. Callers may inspect the
  // returned reference only while mapper execution is quiescent.
  const livo::AdaptiveVisualLidarRuntimeInitializationAudit &
  adaptiveVisualLidarRuntimeInitializationAudit() const noexcept;
  const livo::AdaptiveVisualLidarRuntimeDispatchAudit &
  adaptiveVisualLidarRuntimeDispatchAudit() const noexcept;
  const livo::AdaptiveLioModePublicationAudit &
  adaptiveLioModeAudit() const noexcept;
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback();
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void transformLidarToBody(const PointCloudXYZI &input_cloud, PointCloudXYZI &body_cloud) const;
  void pointBodyToWorld(const PointType &pi, PointType &po);
 
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg_in);
  void imu_transport_cbk(
    const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in);
  void drainImuTransportBuffer();
  void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in);
  void img_cbk(const sensor_msgs::msg::Image::ConstSharedPtr &msg_in);
  void handleFrontCameraImage(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg_in);
  void camera_transport_cbk(
    const sensor_msgs::msg::Image::ConstSharedPtr &msg,
    std::uint8_t camera_id);
  void drainCameraTransportBuffer();
  void img_cbk_front(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void img_cbk_left(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void img_cbk_back(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void img_cbk_right(const sensor_msgs::msg::Image::ConstSharedPtr &msg);
  void handleMultiCameraImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg,
                              std::uint8_t camera_id, vk::AbstractCamera *cam,
                              cv::Mat &dst, const char *name);
  CameraObservationPushResult enqueueCameraObservation(
    std::uint8_t camera_id, std::uint64_t source_stamp_ns,
    double timestamp, cv::Mat image);
  livo::CameraBatchTimeTransferBundle emitCameraTimeTransferShadow(
    const CameraObservationBatch & batch, double anchor_timestamp_sec);
  std::optional<livo::ReferenceOnlyMigrationRuntimeBundle>
  emitReferenceOnlyMigrationRuntimePublication(
    const CameraObservationBatch & batch,
    double anchor_timestamp_sec);
  void publish_img_rgb(const image_transport::Publisher &pubImage, VIOManagerPtr vio_manager);
  void publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes, VIOManagerPtr vio_manager);
  void publish_persistent_maps(const PointCloudXYZI &raw_cloud,
                               const PointCloudXYZRGB &rgb_cloud,
                               bool rgb_cloud_ready,
                               const rclcpp::Time &stamp);
  void publish_frame_body(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudBody);
  void publish_visual_sub_map(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubSubVisualMap);
  void publish_effect_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pmavros_pose_publisherubOdomAftMapped);
  void publish_mavros(const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &mavros_pose_publisher);
  void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath);
  void readParameters(rclcpp::Node::SharedPtr &node);
  rclcpp::Time current_lio_stamp() const;
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);
  cv::Mat getImageFromMsg(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name, img_topic;
  string rgb_cloud_topic_ = "/cloud_registered_rgb";
  string persistent_lio_map_topic_ = "/cloud_registered_map";
  string persistent_rgb_map_topic_ = "/cloud_registered_rgb_map";
  string img_topic_front_, img_topic_left_, img_topic_back_, img_topic_right_;
  string vio_camera_mode_ = "front";
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double persistent_map_voxel_size_ = 0.1;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false;
  bool multi_camera_enable_ = false;
  bool camera_supply_receipt_enable_ = false;
  bool determinism_audit_enable_ = false;
  double camera_reorder_window_sec_ = 0.02;
  double camera_sync_tolerance_sec_ = 0.02;
  bool adaptive_partial_batch_enable_ = false;
  double adaptive_missing_camera_timeout_sec_ = 0.03;
  bool cross_camera_migration_shadow_enable_ = false;
  bool reference_only_migration_shadow_enable_ = false;
  bool camera_time_transfer_shadow_enable_ = false;
  livo::CameraTimeTransferConfig camera_time_transfer_config_;
  livo::LegacyLandmarkShadowConfig
    shared_landmark_shadow_config_;
  livo::LegacyLandmarkObservationShadowConfig
    observation_shadow_config_;
  livo::MultiCameraLandmarkLinearizationShadowConfig
    multi_camera_landmark_linearization_shadow_config_;
  bool shared_landmark_tombstone_shadow_enable_{false};
  bool shared_landmark_refit_shadow_enable_{false};
  double joint_vio_warmup_sec_ = 1.0;
  int camera_queue_capacity_ = 768;
  int min_tracked_points_per_camera_ = 4;
  int min_joint_tracked_points_ = 12;
  int min_joint_cameras_ = 2;
  double max_visual_residual_rmse_ = 20.0;
  double max_visual_rotation_increment_deg_ = 0.1;
  double max_visual_translation_increment_m_ = 0.01;
  livo::AdaptiveVisualPolicyConfig adaptive_policy_config_;
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;
  int persistent_map_publish_interval_ = 10;
  int persistent_map_scan_count_ = 0;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::msg::Imu> prop_imu_buffer;
  sensor_msgs::msg::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::msg::Odometry imu_prop_odom;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuPropOdom;
  double imu_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  // 独立局部实例只允许三个私有输出；默认关闭，保持主实例行为。
  bool local_fast_mode_ = false;
  bool local_allow_instantaneous_scan_ = false;
  bool local_body_dense_ = true;
  std::deque<int64_t> local_scan_stamps_;
  std::deque<int64_t> local_scan_ends_;
  int64_t local_scan_stamp_ns_ = 0;
  int64_t local_imu_end_stamp_ns_ = 0;
  bool local_body_published_ = false;
  std::atomic<uint64_t> local_scan_drop_count_{0};
  bool local_imu_coverage_valid_ = false;
  int64_t local_previous_imu_ns_ = 0;
  int64_t local_imu_available_end_ns_ = 0;
  int64_t local_imu_max_gap_ns_ = 0;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr local_scan_status_pub_;
  bool persistent_map_en_ = false;
  bool persistent_map_published_ = false;
  int img_en = 1, imu_int_frame = 3;
  bool normal_en = true;
  bool exposure_estimate_en = false;
  double exposure_time_init = 0.0;
  bool inverse_composition_en = false;
  bool raycast_en = false;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, patch_size, grid_n_width, grid_n_height, patch_pyrimid_level;
  int outlier_threshold;
  double plot_time;
  int frame_cnt;
  double img_time_offset = 0.0;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
  std::size_t imu_algorithm_capacity_ = 4096U;
  bool imu_overflow_fatal_ = false;
  std::atomic<bool> imu_transport_overflow_failed_{false};
  std::mutex imu_transport_mutex_;
  deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_transport_buffer_;
  struct BufferedCameraMessage
  {
    sensor_msgs::msg::Image::ConstSharedPtr message;
    std::uint64_t source_stamp_ns = 0U;
    double timestamp = 0.0;
    std::uint8_t camera_id = 0U;
  };
  std::mutex camera_transport_mutex_;
  deque<BufferedCameraMessage> camera_transport_buffer_;
  // Main-thread staging buffer. A small per-loop drain budget interleaves
  // camera admission with sync/pop instead of bursting into the image queue.
  deque<BufferedCameraMessage> camera_transport_processing_buffer_;
  std::size_t camera_transport_processing_peak_{0U};
  std::mutex camera_time_transfer_mutex_;
  livo::CameraTimeTransfer camera_time_transfer_;
  CameraObservationQueue camera_observation_queue_;
private:
  bool component_initialization_started_{false};
  // Keep the queue alive longer than its issued authority, and keep that
  // authority alive longer than its dependent runtime publisher.
  std::shared_ptr<livo::AdaptiveVisualSelectionAuthority>
    adaptive_visual_selection_authority_;
  std::unique_ptr<
    livo::ReferenceOnlyMigrationRuntimePublisher>
    reference_only_migration_runtime_publisher_;
  std::size_t pre_vio_shadow_entry_count_{0U};
  std::size_t post_vio_shadow_entry_count_{0U};
public:
  std::optional<CameraObservation> pending_vio_observation_;
  std::optional<CameraObservationBatch> pending_vio_batch_;
  std::array<double, CameraObservationQueue::kCameraCount> last_camera_timestamps_
    {{-1.0, -1.0, -1.0, -1.0}};
  std::array<double, CameraObservationQueue::kCameraCount> camera_time_offsets_
    {{0.0, 0.0, 0.0, 0.0}};
  std::array<std::uint64_t, CameraObservationQueue::kCameraCount> accepted_camera_frames_
    {{0U, 0U, 0U, 0U}};
  std::array<std::uint64_t, CameraObservationQueue::kCameraCount> dropped_camera_frames_
    {{0U, 0U, 0U, 0U}};
  std::array<std::uint64_t, CameraObservationQueue::kCameraCount>
    camera_transport_received_{{0U, 0U, 0U, 0U}};
  std::array<std::uint64_t, CameraObservationQueue::kCameraCount>
    camera_transport_dropped_{{0U, 0U, 0U, 0U}};
  std::uint64_t lio_feature_generation_{0U};
  std::uint64_t last_joint_vio_feature_generation_{0U};
  // Populated only while the opt-in determinism audit is enabled.
  std::uint64_t frame_hash_sequence_{0U};
  std::uint64_t current_imu_stamp_hash_{0U};
  std::uint64_t current_imu_sample_hash_{0U};
  std::uint64_t current_undistorted_cloud_hash_{0U};
  std::uint64_t current_voxel_grid_hash_{0U};
  std::size_t current_imu_stamp_count_{0U};
  std::size_t current_undistorted_point_count_{0U};
  std::size_t current_voxel_grid_point_count_{0U};
  bool current_voxel_grid_hash_applicable_{false};
  vector<pointWithVar> _pv_list;
  vector<double> extrinT;
  vector<double> extrinR;
  vector<double> cameraextrinT;
  vector<double> cameraextrinR;
  vector<double> multi_front_R_, multi_front_T_;
  vector<double> multi_left_R_, multi_left_T_;
  vector<double> multi_back_R_, multi_back_T_;
  vector<double> multi_right_R_, multi_right_T_;
  std::array<vector<double>, CameraObservationQueue::kCameraCount> multi_vio_R_;
  std::array<vector<double>, CameraObservationQueue::kCameraCount> multi_vio_T_;
  std::array<bool, CameraObservationQueue::kCameraCount>
    multi_camera_projection_calibration_present_
      {{false, false, false, false}};
  std::array<bool, CameraObservationQueue::kCameraCount>
    multi_camera_extrinsic_calibration_present_
      {{false, false, false, false}};
  bool camera_calibration_provenance_captured_{false};
  int IMG_POINT_COV;
  int JOINT_IMG_POINT_COV;

  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZI::Ptr persistent_lio_map_;
  PointCloudXYZRGB::Ptr persistent_rgb_map_;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pcd_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::msg::Path path;
  nav_msgs::msg::Odometry odomAftMapped;
  geometry_msgs::msg::Quaternion geoQuat;
  geometry_msgs::msg::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  VIOManagerPtr vio_manager;

  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr plane_pub;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_pub;
  std::shared_ptr<rclcpp::SubscriptionBase> sub_pcl;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_front;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_left;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_back;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_right;
  rclcpp::CallbackGroup::SharedPtr imu_ingress_callback_group_;
  std::array<
    rclcpp::CallbackGroup::SharedPtr,
    CameraObservationQueue::kCameraCount> camera_ingress_callback_groups_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudRGB;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPersistentLioMap;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPersistentRGBMap;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudBody;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubNormal;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSubVisualMap;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
  std::unique_ptr<tf2_ros::TransformBroadcaster> odom_tf_broadcaster_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pubRuntimeReady;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudDyn;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudDynRmed;
  image_transport::Publisher pubImage;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr mavros_pose_publisher;
  rclcpp::TimerBase::SharedPtr imu_prop_timer;
  rclcpp::Node::SharedPtr node;

  std::mutex img_mtx_;
  cv::Mat img_rgb_front, img_rgb_left, img_rgb_back, img_rgb_right;
  std::array<
    std::deque<sensor_msgs::msg::Image::ConstSharedPtr>,
    CameraObservationQueue::kCameraCount> coloring_image_buffers_;
  vk::AbstractCamera *cam_front_ = nullptr;
  vk::AbstractCamera *cam_left_ = nullptr;
  vk::AbstractCamera *cam_back_ = nullptr;
  vk::AbstractCamera *cam_right_ = nullptr;
  Sophus::SE3<double> T_base_cam_front_;
  Sophus::SE3<double> T_base_cam_left_;
  Sophus::SE3<double> T_base_cam_back_;
  Sophus::SE3<double> T_base_cam_right_;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  bool colmap_output_en = false;

private:
  bool adaptiveSingleCameraRuntimeEnabled() const noexcept;
  bool dispatchAdaptiveSingleCameraVio();
  void recordAdaptiveLioMode(
    const livo::AdaptiveLioModeObservation &
    observation) noexcept;
  void finalizePendingAdaptiveLioMode() noexcept;
  void finalizeAdaptiveVisualOutcome(
    livo::AdaptiveLioMode mode,
    livo::AdaptiveLioModeReason reason) noexcept;
  bool adaptive_lio_only_fallback_active_{false};
  std::optional<livo::AdaptiveLioModeObservation>
    pending_adaptive_lio_mode_observation_;
  livo::AdaptiveLioModeLedger adaptive_lio_mode_ledger_;
  livo::AdaptiveLioModePublicationAudit
    adaptive_lio_mode_publication_audit_;
  livo::AdaptiveVisualLidarRuntimeInitializationAudit
    adaptive_visual_lidar_runtime_initialization_audit_;
  livo::AdaptiveVisualLidarRuntimeDispatchAudit
    adaptive_visual_lidar_runtime_dispatch_audit_;
  bool syncLioOnlyMeasurement(LidarMeasureGroup & meas);
  double cameraIngressWatermark();
};
#endif
