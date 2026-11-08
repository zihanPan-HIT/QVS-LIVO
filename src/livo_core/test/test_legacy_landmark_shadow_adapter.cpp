#include "legacy_landmark_shadow_adapter.h"
#include "active_plane_authority_test_support.h"

#include <map>

#include <gtest/gtest.h>

namespace
{

livo::LegacyPlaneLandmarkSample sample(
  const Eigen::Vector3d & position_world_m,
  const std::uint8_t camera_id = 0U,
  const std::uint64_t geometry_revision = 1U)
{
  livo::LegacyPlaneLandmarkSample value;
  value.plane = livo::PlaneIdentity{7U, 1U};
  value.geometry_revision = geometry_revision;
  value.identity_origin_world_m = Eigen::Vector3d::Zero();
  value.identity_u_axis_world = Eigen::Vector3d::UnitX();
  value.identity_v_axis_world = Eigen::Vector3d::UnitY();
  value.position_world_m = position_world_m;
  value.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  value.normal_world = Eigen::Vector3d::UnitZ();
  value.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  value.camera_id = camera_id;
  value.level = 1U;
  return value;
}

livo::LandmarkShadowBootstrapRequest request()
{
  livo::LandmarkShadowBootstrapRequest value;
  value.batch = livo::BatchId{1U};
  value.geometry = livo::GeometrySnapshotId{10U};
  value.lidar_plane_stable = true;
  return value;
}

livo::LandmarkShadowBootstrapResult bootstrapWithActivePlanes(
  const livo::LegacyLandmarkShadowAdapter & adapter,
  livo::MultiViewLandmarkStore & store,
  livo::LandmarkShadowBootstrapRequest & request,
  livo::test_support::ActivePlaneAuthority & authority)
{
  std::map<
    std::pair<std::uint64_t, std::uint32_t>,
    livo::ActivePlaneDescriptor> descriptors;
  for (const auto & value : request.samples)
  {
    descriptors[{value.plane.id, value.plane.epoch}] =
      livo::ActivePlaneDescriptor{
      livo::GeometryVersion{
        value.plane, value.geometry_revision},
      std::max<std::uint64_t>(
        1U, value.geometry_revision)};
  }
  std::vector<livo::ActivePlaneDescriptor> active;
  for (const auto & entry : descriptors)
  {
    active.push_back(entry.second);
  }
  if (!authority.replace(active))
  {
    return {};
  }
  request.active_planes = authority.snapshot();
  for (auto & value : request.samples)
  {
    const auto lease = request.active_planes.leaseFor(
      livo::GeometryVersion{
        value.plane, value.geometry_revision});
    if (!lease.has_value())
    {
      return {};
    }
    value.active_plane_lease = *lease;
  }
  return adapter.bootstrap(store, request);
}

TEST(
  LegacyLandmarkShadowAdapter,
  DeduplicatesPlaneCellsAndKeepsIdentityAcrossCameras)
{
  livo::LegacyLandmarkShadowConfig config;
  config.enabled = true;
  config.plane_cell_size_m = 0.5;
  livo::LegacyLandmarkShadowAdapter adapter(config);
  livo::MultiViewLandmarkStore store;
  livo::test_support::ActivePlaneAuthority authority;
  auto bootstrap = request();
  bootstrap.samples = {
    sample(Eigen::Vector3d(0.10, 0.20, 0.0), 0U),
    sample(Eigen::Vector3d(0.20, 0.10, 0.0), 1U),
    sample(Eigen::Vector3d(0.60, 0.10, 0.0), 2U)};

  const livo::LandmarkShadowBootstrapResult result =
    bootstrapWithActivePlanes(
      adapter, store, bootstrap, authority);

  EXPECT_EQ(
    result.status,
    livo::LandmarkShadowBootstrapStatus::kCommitted);
  EXPECT_EQ(result.input_sample_count, 3U);
  EXPECT_EQ(result.unique_seed_count, 2U);
  EXPECT_EQ(result.commit.geometry_writes, 2U);
  const livo::ReadSnapshot snapshot =
    store.snapshot(
      bootstrap.geometry, bootstrap.expected_revisions);
  ASSERT_EQ(snapshot.landmarks.size(), 2U);
  EXPECT_EQ(snapshot.landmarks[0U].support_key.plane.id, 7U);
  EXPECT_EQ(snapshot.landmarks[1U].support_key.plane.id, 7U);
  EXPECT_NE(
    snapshot.landmarks[0U].support_key.cell.u,
    snapshot.landmarks[1U].support_key.cell.u);
}

TEST(
  LegacyLandmarkShadowAdapter,
  BootstrapNeverRefreshesExistingGeometryRevision)
{
  livo::LegacyLandmarkShadowConfig config;
  config.enabled = true;
  config.plane_cell_size_m = 0.5;
  livo::LegacyLandmarkShadowAdapter adapter(config);
  livo::MultiViewLandmarkStore store;
  livo::test_support::ActivePlaneAuthority authority;
  auto bootstrap = request();
  bootstrap.samples = {
    sample(Eigen::Vector3d(0.10, 0.20, 0.0))};
  ASSERT_EQ(
    bootstrapWithActivePlanes(
      adapter, store, bootstrap, authority).status,
    livo::LandmarkShadowBootstrapStatus::kCommitted);
  const livo::ReadSnapshot initial =
    store.snapshot(
      bootstrap.geometry, bootstrap.expected_revisions);
  ASSERT_EQ(initial.landmarks.size(), 1U);
  const livo::LandmarkId landmark_id =
    initial.landmarks.front().landmark_id;

  bootstrap.batch.value = 2U;
  EXPECT_EQ(
    bootstrapWithActivePlanes(
      adapter, store, bootstrap, authority).status,
    livo::LandmarkShadowBootstrapStatus::kNoGeometryChange);

  bootstrap.batch.value = 3U;
  bootstrap.geometry.value = 11U;
  bootstrap.samples.front().geometry_revision = 2U;
  const auto update = bootstrapWithActivePlanes(
    adapter, store, bootstrap, authority);
  EXPECT_EQ(
    update.status,
    livo::LandmarkShadowBootstrapStatus::kNoGeometryChange);
  const livo::ReadSnapshot updated =
    store.snapshot(
      bootstrap.geometry, bootstrap.expected_revisions);
  ASSERT_EQ(updated.landmarks.size(), 1U);
  EXPECT_EQ(updated.landmarks.front().landmark_id, landmark_id);
  EXPECT_EQ(
    updated.landmarks.front().geometry_version.revision, 1U);
}

TEST(
  LegacyLandmarkShadowAdapter,
  MissingVisualEvidenceDoesNotBlockLidarGeometryBootstrap)
{
  livo::LegacyLandmarkShadowConfig config;
  config.enabled = true;
  livo::LegacyLandmarkShadowAdapter adapter(config);
  livo::MultiViewLandmarkStore store;
  livo::test_support::ActivePlaneAuthority authority;
  auto bootstrap = request();
  bootstrap.time_transfer_valid = false;
  bootstrap.camera_model_valid = false;
  bootstrap.in_field_of_view = false;
  bootstrap.texture_valid = false;
  bootstrap.samples = {
    sample(Eigen::Vector3d(0.10, 0.20, 0.0))};

  const auto result = bootstrapWithActivePlanes(
    adapter, store, bootstrap, authority);

  EXPECT_EQ(
    result.status,
    livo::LandmarkShadowBootstrapStatus::kCommitted);
  EXPECT_EQ(
    store.snapshot(
      bootstrap.geometry,
      bootstrap.expected_revisions).landmarks.size(),
    1U);
}

TEST(LegacyLandmarkShadowAdapter, InvalidLidarEvidenceFailsClosed)
{
  livo::LegacyLandmarkShadowConfig config;
  config.enabled = true;
  livo::LegacyLandmarkShadowAdapter adapter(config);
  livo::MultiViewLandmarkStore store;
  auto bootstrap = request();
  bootstrap.lidar_plane_stable = false;
  bootstrap.samples = {
    sample(Eigen::Vector3d(0.10, 0.20, 0.0))};

  const auto result = adapter.bootstrap(store, bootstrap);

  EXPECT_EQ(
    result.status,
    livo::LandmarkShadowBootstrapStatus::kProofRejected);
  EXPECT_TRUE(
    store.snapshot(
      bootstrap.geometry,
      bootstrap.expected_revisions).landmarks.empty());
}

}  // namespace
