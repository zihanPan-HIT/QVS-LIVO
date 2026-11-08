#ifndef LIVO_LEGACY_LANDMARK_SHADOW_ADAPTER_H_
#define LIVO_LEGACY_LANDMARK_SHADOW_ADAPTER_H_

#include "multi_view_landmark.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace livo
{

struct LegacyPlaneLandmarkSample
{
  PlaneIdentity plane;
  std::uint64_t geometry_revision{0U};
  ActivePlaneIdentityLease active_plane_lease;
  Eigen::Vector3d identity_origin_world_m{
    Eigen::Vector3d::Zero()};
  Eigen::Vector3d identity_u_axis_world{
    Eigen::Vector3d::UnitX()};
  Eigen::Vector3d identity_v_axis_world{
    Eigen::Vector3d::UnitY()};
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d position_covariance_m2{
    Eigen::Matrix3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 6, 6> plane_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  std::uint8_t camera_id{0U};
  std::uint16_t level{0U};
};

struct LegacyLandmarkShadowConfig
{
  bool enabled{false};
  double plane_cell_size_m{0.25};
  std::size_t maximum_seeds_per_batch{512U};
};

struct LandmarkShadowBootstrapRequest
{
  BatchId batch;
  GeometrySnapshotId geometry;
  ActivePlaneIdentitySnapshot active_planes;
  CommitRevisionSet expected_revisions;
  bool lidar_plane_stable{false};
  bool camera_model_valid{false};
  bool time_transfer_valid{false};
  bool in_field_of_view{false};
  bool texture_valid{false};
  std::vector<LegacyPlaneLandmarkSample> samples;
};

enum class LandmarkShadowBootstrapStatus
{
  kDisabled,
  kInvalidConfiguration,
  kNoValidSeed,
  kNoGeometryChange,
  kPrepareRejected,
  kProofRejected,
  kCommitted,
  kCommitRejected
};

struct LandmarkShadowBootstrapResult
{
  LandmarkShadowBootstrapStatus status{
    LandmarkShadowBootstrapStatus::kDisabled};
  std::size_t input_sample_count{0U};
  std::size_t unique_seed_count{0U};
  std::size_t staged_seed_count{0U};
  CommitResult commit;
  std::optional<LidarAnchoredCommitToken> commit_token;
};

bool resolveLegacyPlaneLandmarkIdentity(
  const LegacyPlaneLandmarkSample & sample,
  double plane_cell_size_m,
  GeometrySupportKey & support_key,
  GeometryVersion & geometry_version);

class LegacyLandmarkShadowAdapter
{
public:
  explicit LegacyLandmarkShadowAdapter(
    LegacyLandmarkShadowConfig config =
      LegacyLandmarkShadowConfig{});

  bool configValid() const;

  LandmarkShadowBootstrapResult bootstrap(
    MultiViewLandmarkStore & store,
    const LandmarkShadowBootstrapRequest & request) const;

private:
  LegacyLandmarkShadowConfig config_;
};

const char * landmarkShadowBootstrapStatusName(
  LandmarkShadowBootstrapStatus status);

}  // namespace livo

#endif  // LIVO_LEGACY_LANDMARK_SHADOW_ADAPTER_H_
