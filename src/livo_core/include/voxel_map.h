/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include "active_plane_identity_registry.h"
#include "plane_lifecycle.h"
#include <algorithm>
#include <Eigen/Dense>
#include <cstdint>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int64_t> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  bool is_pub_plane_map_;
  bool plane_lifecycle_shadow_enable_{false};
  livo::PlaneReplacementGateConfig plane_replacement_gate_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;

  double data_retention_time_;
} VoxelMapConfig;

typedef struct PointToPlane
{
  Eigen::Vector3d point_b_;
  Eigen::Vector3d point_w_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  M3D body_cov_;
  int layer_;
  double d_;
  double eigen_value_;
  bool is_valid_;
  float dis_to_plane_;
} PointToPlane;

typedef struct VoxelPlane
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  float d_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
  bool is_init_ = false;
  livo::GeometryVersion geometry_version_;
  std::uint64_t geometry_generation_{0U};
  livo::PlaneLocalBasis identity_basis_;
  bool is_update_ = false;
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

inline void copyPlaneSupportToPoint(
  const VoxelPlane & plane, const int layer,
  pointWithVar & point)
{
  point.plane_support_valid =
    plane.is_plane_ &&
    plane.geometry_version_.valid();
  point.plane_id =
    point.plane_support_valid ?
    plane.geometry_version_.plane.id : 0U;
  point.plane_epoch = plane.geometry_version_.plane.epoch;
  point.plane_revision = plane.geometry_version_.revision;
  point.plane_level = static_cast<std::uint16_t>(
    std::max(0, layer));
  point.plane_identity_origin =
    plane.identity_basis_.origin_world_m;
  point.plane_identity_u_axis =
    plane.identity_basis_.u_axis_world;
  point.plane_identity_v_axis =
    plane.identity_basis_.v_axis_world;
  point.plane_covariance = plane.plane_var_;
}

class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

struct DS_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_;
  VoxelPlane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_init_num_;
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int max_points_num_;
  int max_layer_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;
  double last_update_time_;
  livo::PlaneReplacementGateConfig plane_replacement_gate_;
  livo::PlaneTombstoneEventQueue * plane_tombstone_events_{nullptr};
  const std::uint64_t * plane_geometry_generation_{nullptr};
  livo::PlaneRefitEventQueue * plane_refit_events_{nullptr};

  VoxelOctoTree(
    int max_layer, int layer, int points_size_threshold,
    int max_points_num, float planer_threshold,
    const livo::PlaneReplacementGateConfig & plane_replacement_gate =
      livo::PlaneReplacementGateConfig{},
    livo::PlaneTombstoneEventQueue * plane_tombstone_events = nullptr,
    const std::uint64_t * plane_geometry_generation = nullptr,
    livo::PlaneRefitEventQueue * plane_refit_events = nullptr)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold),
        plane_replacement_gate_(plane_replacement_gate),
        plane_tombstone_events_(plane_tombstone_events),
        plane_geometry_generation_(plane_geometry_generation),
        plane_refit_events_(plane_refit_events)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    last_update_time_ = 0.0;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);
  void init_octo_tree();
  void cut_octo_tree();
  void UpdateOctoTree(const pointWithVar &pv);
  void emitTombstonesForRemoval(
    livo::PlaneTombstoneEventQueue & events,
    std::uint64_t generation,
    livo::PlaneTombstoneReason reason) const;

  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);
  VoxelOctoTree *Insert(const pointWithVar &pv);
};

void loadVoxelConfig(rclcpp::Node::SharedPtr &node, VoxelMapConfig &voxel_config);

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;
  int current_frame_id_ = 0;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_map_pub_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;

  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  M3D extR_;
  V3D extT_;
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
  StatesGroup state_;
  V3D position_last_;

  V3D last_slide_position = {0,0,0};

  geometry_msgs::msg::Quaternion geoQuat_;

  int feats_down_size_;
  int effct_feat_num_;
  Eigen::Matrix<double, 6, 6> pose_information_{
    Eigen::Matrix<double, 6, 6>::Zero()};
  bool pose_information_valid_{false};
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
    if (
      config_setting_.plane_lifecycle_shadow_enable_ &&
      !config_setting_.plane_replacement_gate_.valid())
    {
      throw std::invalid_argument(
        "invalid enabled plane lifecycle replacement thresholds");
    }
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  void StateEstimation(StatesGroup &state_propagat);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  void BuildVoxelMap();
  V3F RGBFromVoxel(const V3D &input_point);

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);

  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);

  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl);

  void pubVoxelMap();

  void mapSliding();
  void cleanOldVoxels(double current_time);
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );
  std::vector<livo::PlaneTombstoneEvent>
  snapshotPlaneTombstoneEvents() const;
  std::vector<livo::PlaneTombstoneEvent>
  consumePlaneTombstoneEvents(bool consumer_enabled);
  livo::PlaneRefitEventBatch
  snapshotPlaneRefitEvents() const;
  void acknowledgePlaneRefitEvents(
    const livo::PlaneRefitEventBatch & events,
    bool consumer_enabled);
  livo::ActivePlaneIdentitySnapshot
  snapshotActivePlaneIdentities() const;

private:
  livo::PlaneTombstoneEventQueue plane_tombstone_events_;
  livo::PlaneRefitEventQueue plane_refit_events_;
  livo::ActivePlaneIdentityRegistry
    active_plane_identity_registry_;
  std::uint64_t plane_geometry_generation_{1U};

  bool publishActivePlaneIdentities();
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);

  void pubSinglePlane(visualization_msgs::msg::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::msg::Quaternion &q);

  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_
