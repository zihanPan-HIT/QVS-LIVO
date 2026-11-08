#include "shared_landmark_tombstone_shadow.h"
#include "active_plane_authority_test_support.h"

#include <gtest/gtest.h>

namespace
{

livo::CommitRevisionSet revisions()
{
  livo::CommitRevisionSet value;
  value.state.value = 17U;
  value.rig_calibration_revision = 3U;
  value.camera_model_revisions.fill(2U);
  value.photometric_calibration_revisions.fill(11U);
  value.time_transfer_revision = 13U;
  return value;
}

livo::GeometrySeed seed(
  const livo::PlaneIdentity plane,
  const livo::ActivePlaneIdentitySnapshot & active_planes)
{
  livo::GeometrySeed value;
  value.support_key.plane = plane;
  value.support_key.cell.u = 1;
  value.geometry_version.plane = plane;
  value.geometry_version.revision = 4U;
  value.active_plane_lease =
    *active_planes.leaseFor(value.geometry_version);
  value.position_world_m = Eigen::Vector3d(1.0, 2.0, 3.0);
  value.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  value.normal_world = Eigen::Vector3d::UnitZ();
  value.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  return value;
}

void bootstrap(
  livo::MultiViewLandmarkStore & store,
  const livo::GeometrySnapshotId geometry,
  const livo::PlaneIdentity plane)
{
  const livo::CommitRevisionSet expected = revisions();
  livo::test_support::ActivePlaneAuthority authority;
  ASSERT_TRUE(authority.replace(
    {livo::ActivePlaneDescriptor{
      livo::GeometryVersion{plane, 4U}, 1U}}));
  const auto active_planes = authority.snapshot();
  const livo::ReadSnapshot snapshot =
    store.snapshot(geometry, expected, active_planes);
  livo::LandmarkQuery query;
  query.batch.value = 1U;
  query.geometry = geometry;
  query.active_planes = active_planes;
  query.expected_revisions = expected;
  query.geometry_seeds.push_back(
    seed(plane, active_planes));
  auto prepared = store.prepare(snapshot, query);
  ASSERT_TRUE(prepared.valid());
  const livo::MapCommitProof proof = livo::BootstrapGate::mint(
    livo::test_support::makeLidarAnchoredToken(
      geometry, expected),
    prepared);
  ASSERT_EQ(
    store.commit(prepared.take_delta(), proof).status,
    livo::CommitStatus::kCommitted);
}

livo::PlaneTombstoneEvent event(const livo::PlaneIdentity plane)
{
  livo::PlaneTombstoneEvent value;
  value.old_identity = plane;
  value.last_revision = 4U;
  value.generation = 12U;
  value.reason = livo::PlaneTombstoneReason::kMajorReplacement;
  return value;
}

TEST(SharedLandmarkTombstoneShadow, DefaultOffAndMissingTokenAreZeroWrite)
{
  const livo::GeometrySnapshotId geometry{21U};
  const livo::PlaneIdentity plane{90U, 1U};
  livo::MultiViewLandmarkStore store;
  bootstrap(store, geometry, plane);
  const livo::MapRevision before =
    store.snapshot(geometry, revisions()).store_revision;
  const std::vector<livo::PlaneTombstoneEvent> events{
    event(plane)};

  livo::SharedLandmarkTombstoneShadowConsumer disabled;
  const auto disabled_result =
    disabled.apply(store, events, nullptr, livo::BatchId{2U});
  EXPECT_EQ(
    disabled_result.status,
    livo::SharedLandmarkTombstoneShadowStatus::kDisabled);
  EXPECT_FALSE(disabled_result.acknowledge_events);
  EXPECT_FALSE(disabled_result.solver_applied);
  EXPECT_EQ(
    store.snapshot(geometry, revisions()).store_revision.value,
    before.value);

  livo::SharedLandmarkTombstoneShadowConfig enabled_config;
  enabled_config.enabled = true;
  livo::SharedLandmarkTombstoneShadowConsumer enabled(
    enabled_config);
  const auto missing_token =
    enabled.apply(store, events, nullptr, livo::BatchId{3U});
  EXPECT_EQ(
    missing_token.status,
    livo::SharedLandmarkTombstoneShadowStatus::kMissingLidarToken);
  EXPECT_FALSE(missing_token.acknowledge_events);
  EXPECT_FALSE(missing_token.solver_applied);
  EXPECT_EQ(
    store.snapshot(geometry, revisions()).store_revision.value,
    before.value);
}

TEST(SharedLandmarkTombstoneShadow, EnabledCommitsTypedTombstoneOnly)
{
  const livo::GeometrySnapshotId geometry{22U};
  const livo::PlaneIdentity plane{91U, 1U};
  const livo::CommitRevisionSet expected = revisions();
  livo::MultiViewLandmarkStore store;
  bootstrap(store, geometry, plane);
  const livo::LidarAnchoredCommitToken token =
    livo::test_support::makeLidarAnchoredToken(
    geometry, expected);

  livo::SharedLandmarkTombstoneShadowConfig config;
  config.enabled = true;
  livo::SharedLandmarkTombstoneShadowConsumer consumer(config);
  const auto result = consumer.apply(
    store, {event(plane)}, &token, livo::BatchId{2U});

  ASSERT_EQ(
    result.status,
    livo::SharedLandmarkTombstoneShadowStatus::kCommitted);
  EXPECT_TRUE(result.acknowledge_events);
  EXPECT_FALSE(result.solver_applied);
  EXPECT_EQ(result.tombstoned_landmarks, 1U);
  const livo::ReadSnapshot after =
    store.snapshot(geometry, expected);
  ASSERT_EQ(after.landmarks.size(), 1U);
  EXPECT_EQ(
    after.landmarks.front().lifecycle,
    livo::LandmarkLifecycle::kTombstoned);
}

}  // namespace
