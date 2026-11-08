#ifndef LIVO_TEST_ACTIVE_PLANE_AUTHORITY_TEST_SUPPORT_H_
#define LIVO_TEST_ACTIVE_PLANE_AUTHORITY_TEST_SUPPORT_H_

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "legacy_landmark_shadow_adapter.h"
#include "voxel_map.h"

namespace livo::test_support
{

class ActivePlaneAuthority
{
public:
  ActivePlaneAuthority()
  : config_(makeConfig()),
    manager_(config_, initial_map_)
  {
  }

  ~ActivePlaneAuthority()
  {
    clear();
  }

  ActivePlaneAuthority(
    const ActivePlaneAuthority &) = delete;
  ActivePlaneAuthority & operator=(
    const ActivePlaneAuthority &) = delete;

  bool replace(
    const std::vector<ActivePlaneDescriptor> & descriptors)
  {
    clear();
    std::int64_t voxel_x = 0;
    for (const ActivePlaneDescriptor & descriptor : descriptors)
    {
      auto * octree = new VoxelOctoTree();
      for (VoxelOctoTree *& child : octree->leaves_)
      {
        child = nullptr;
      }
      octree->plane_ptr_ = new VoxelPlane();
      octree->plane_ptr_->is_plane_ = true;
      octree->plane_ptr_->is_init_ = true;
      octree->plane_ptr_->geometry_version_ =
        descriptor.geometry_version;
      octree->plane_ptr_->geometry_generation_ =
        descriptor.generation;
      manager_.voxel_map_.emplace(
        VOXEL_LOCATION{voxel_x++, 0, 0}, octree);
    }
    manager_.UpdateVoxelMap({});
    const ActivePlaneIdentitySnapshot current =
      manager_.snapshotActivePlaneIdentities();
    return
      current.valid() &&
      current.size() == descriptors.size() &&
      std::all_of(
        descriptors.begin(), descriptors.end(),
        [&current](const ActivePlaneDescriptor & descriptor) {
          return current.leaseFor(
            descriptor.geometry_version).has_value();
        });
  }

  ActivePlaneIdentitySnapshot snapshot() const
  {
    return manager_.snapshotActivePlaneIdentities();
  }

private:
  static VoxelMapConfig makeConfig()
  {
    VoxelMapConfig config{};
    config.plane_lifecycle_shadow_enable_ = true;
    return config;
  }

  void clear()
  {
    for (auto & entry : manager_.voxel_map_)
    {
      delete entry.second;
    }
    manager_.voxel_map_.clear();
  }

  VoxelMapConfig config_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *>
    initial_map_;
  VoxelMapManager manager_;
};

inline LidarAnchoredCommitToken makeLidarAnchoredToken(
  const GeometrySnapshotId geometry,
  const CommitRevisionSet & expected_revisions)
{
  const GeometryVersion geometry_version{
    PlaneIdentity{9000001U, 1U}, 1U};
  ActivePlaneAuthority authority;
  if (!authority.replace({
      ActivePlaneDescriptor{geometry_version, 1U}}))
  {
    throw std::runtime_error(
      "failed to publish test LiDAR geometry");
  }
  const ActivePlaneIdentitySnapshot active =
    authority.snapshot();
  const auto lease = active.leaseFor(geometry_version);
  if (!lease.has_value())
  {
    throw std::runtime_error(
      "failed to obtain test LiDAR lease");
  }

  LegacyPlaneLandmarkSample sample;
  sample.plane = geometry_version.plane;
  sample.geometry_revision =
    geometry_version.revision;
  sample.active_plane_lease = *lease;
  sample.position_world_m =
    Eigen::Vector3d(0.1, 0.1, 0.0);
  sample.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  sample.normal_world = Eigen::Vector3d::UnitZ();
  sample.plane_covariance =
    0.001 *
    Eigen::Matrix<double, 6, 6>::Identity();

  LandmarkShadowBootstrapRequest request;
  request.batch = BatchId{1U};
  request.geometry = geometry;
  request.active_planes = active;
  request.expected_revisions = expected_revisions;
  request.lidar_plane_stable = true;
  request.samples.push_back(sample);

  MultiViewLandmarkStore authority_store;
  LegacyLandmarkShadowConfig config;
  config.enabled = true;
  const auto result =
    LegacyLandmarkShadowAdapter(config).bootstrap(
    authority_store, request);
  if (!result.commit_token.has_value())
  {
    throw std::runtime_error(
      "real LiDAR bootstrap gate rejected test evidence");
  }
  return *result.commit_token;
}

}  // namespace livo::test_support

#endif  // LIVO_TEST_ACTIVE_PLANE_AUTHORITY_TEST_SUPPORT_H_
