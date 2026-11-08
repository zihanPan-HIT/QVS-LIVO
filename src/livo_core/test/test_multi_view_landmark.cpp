#include "multi_view_landmark.h"
#include "active_plane_authority_test_support.h"
#include "checked_increment.h"
#include "runtime_frame_provenance_test_support.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <type_traits>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

namespace
{

template<typename Store, typename = void>
struct CanTakeUnboundSnapshot : std::false_type
{
};

template<typename Store>
struct CanTakeUnboundSnapshot<
  Store,
  std::void_t<decltype(
    std::declval<const Store &>().snapshot(
      std::declval<const livo::GeometrySnapshotId &>()))>>
: std::true_type
{
};

static_assert(
  !std::is_default_constructible_v<
    livo::LidarAnchoredCommitToken>);
static_assert(
  !std::is_default_constructible_v<
    livo::AcceptedStateCommitToken>);
static_assert(
  !std::is_constructible_v<
    livo::LidarAnchoredCommitToken,
    livo::GeometrySnapshotId,
    livo::CommitRevisionSet>);
static_assert(
  !std::is_constructible_v<
    livo::AcceptedStateCommitToken,
    livo::GeometrySnapshotId,
    livo::CommitRevisionSet>);
static_assert(
  !std::is_constructible_v<
    livo::ActivePlaneIdentityLease,
    livo::GeometryVersion,
    std::uint64_t,
    std::uint64_t>);
static_assert(
  !std::is_aggregate_v<
    livo::ActivePlaneIdentityLease>);
static_assert(
  !std::is_default_constructible_v<
    livo::AcceptedObservationStoreReceipt>);
static_assert(
  !std::is_aggregate_v<
    livo::AcceptedObservationStoreReceipt>);
static_assert(
  std::is_copy_constructible_v<
    livo::AcceptedObservationStoreReceipt>);
static_assert(
  !CanTakeUnboundSnapshot<
    livo::MultiViewLandmarkStore>::value);

using livo::BootstrapGate;
using livo::ActivePlaneDescriptor;
using livo::ActivePlaneIdentityRegistry;
using livo::ActivePlaneIdentitySnapshot;
using livo::CommitStatus;
using livo::CommitRevisionSet;
using livo::GeometrySeed;
using livo::GeometryVersion;
using livo::GeometrySnapshotId;
using livo::GeometrySupportKey;
using livo::LandmarkLifecycle;
using livo::LandmarkLifecycleTransition;
using livo::LandmarkQuery;
using livo::LandmarkReferenceRequest;
using livo::LidarTombstoneGate;
using livo::MapRevision;
using livo::MultiViewLandmarkStore;
using livo::MultiViewLandmarkStoreConfig;
using livo::ObservationCandidate;
using livo::ObservationPhotometricMode;
using livo::ObservationWriteKind;
using livo::PhotometricStatus;
using livo::PlaneIdentity;
using livo::PlaneLocalCell;
using livo::PlaneTombstoneMutation;
using livo::ReadSnapshot;
using ActivePlaneTestAuthority =
  livo::test_support::ActivePlaneAuthority;

GeometrySupportKey supportKey(
  const std::uint64_t plane_id,
  const std::uint32_t plane_epoch,
  const std::int32_t cell_u,
  const std::int32_t cell_v,
  const std::uint16_t level = 0U)
{
  GeometrySupportKey key;
  key.plane = PlaneIdentity{plane_id, plane_epoch};
  key.cell = PlaneLocalCell{cell_u, cell_v, level};
  return key;
}

GeometrySeed seed(
  const GeometrySupportKey & key,
  const std::uint64_t plane_revision)
{
  GeometrySeed result;
  result.support_key = key;
  result.geometry_version.plane = key.plane;
  result.geometry_version.revision = plane_revision;
  result.position_world_m =
    Eigen::Vector3d(1.0, -2.0, 3.0);
  result.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  result.normal_world =
    Eigen::Vector3d::UnitZ();
  result.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  return result;
}

GeometrySeed activeSeed(
  const ActivePlaneIdentitySnapshot & active_planes,
  const GeometrySupportKey & key,
  const std::uint64_t plane_revision)
{
  GeometrySeed result = seed(key, plane_revision);
  const auto lease =
    active_planes.leaseFor(result.geometry_version);
  EXPECT_TRUE(lease.has_value());
  if (lease.has_value())
  {
    result.active_plane_lease = *lease;
  }
  return result;
}

ActivePlaneTestAuthority & activePlaneAuthorityForTest(
  const MultiViewLandmarkStore & store)
{
  static std::map<
    const MultiViewLandmarkStore *,
    std::unique_ptr<ActivePlaneTestAuthority>>
    authorities;
  auto & authority = authorities[&store];
  if (!authority)
  {
    authority =
      std::make_unique<ActivePlaneTestAuthority>();
  }
  return *authority;
}

livo::PreparedLandmarkBatch prepareBootstrapForTest(
  MultiViewLandmarkStore & store,
  ReadSnapshot snapshot,
  LandmarkQuery & query)
{
  ActivePlaneTestAuthority & authority =
    activePlaneAuthorityForTest(store);
  std::map<
    std::pair<std::uint64_t, std::uint32_t>,
    ActivePlaneDescriptor> descriptors;
  for (const GeometrySeed & geometry_seed :
       query.geometry_seeds)
  {
    descriptors[{
      geometry_seed.geometry_version.plane.id,
      geometry_seed.geometry_version.plane.epoch}] =
      ActivePlaneDescriptor{
      geometry_seed.geometry_version,
      std::max<std::uint64_t>(
        1U, geometry_seed.geometry_version.revision)};
  }
  for (ObservationCandidate & candidate :
       query.observations)
  {
    if (
      candidate.kind !=
      ObservationWriteKind::kBootstrapReference)
    {
      continue;
    }
    GeometryVersion geometry_version =
      candidate.geometry_version;
    if (geometry_version.revision == 0U)
    {
      const auto staged_seed = std::find_if(
        query.geometry_seeds.begin(),
        query.geometry_seeds.end(),
        [&candidate](const GeometrySeed & geometry_seed) {
          return
            geometry_seed.support_key ==
            candidate.support_key;
        });
      if (staged_seed != query.geometry_seeds.end())
      {
        geometry_version =
          staged_seed->geometry_version;
      }
      else
      {
        const auto existing = std::find_if(
          snapshot.landmarks.begin(),
          snapshot.landmarks.end(),
          [&candidate](
            const livo::LandmarkSnapshot & landmark) {
            return
              landmark.support_key ==
              candidate.support_key;
          });
        if (existing != snapshot.landmarks.end())
        {
          geometry_version =
            existing->geometry_version;
        }
      }
    }
    if (geometry_version.valid())
    {
      descriptors[{
        geometry_version.plane.id,
        geometry_version.plane.epoch}] =
        ActivePlaneDescriptor{
        geometry_version,
        std::max<std::uint64_t>(
          1U, geometry_version.revision)};
    }
  }

  std::vector<ActivePlaneDescriptor> active_descriptors;
  active_descriptors.reserve(descriptors.size());
  for (const auto & entry : descriptors)
  {
    active_descriptors.push_back(entry.second);
  }
  if (!authority.replace(active_descriptors))
  {
    return {};
  }
  const ActivePlaneIdentitySnapshot active =
    authority.snapshot();
  snapshot.active_planes = active;
  query.active_planes = active;
  for (GeometrySeed & geometry_seed : query.geometry_seeds)
  {
    const auto lease =
      active.leaseFor(geometry_seed.geometry_version);
    if (!lease.has_value())
    {
      return {};
    }
    geometry_seed.active_plane_lease = *lease;
  }
  for (ObservationCandidate & candidate :
       query.observations)
  {
    if (
      candidate.kind !=
      ObservationWriteKind::kBootstrapReference)
    {
      continue;
    }
    GeometryVersion geometry_version =
      candidate.geometry_version;
    if (geometry_version.revision == 0U)
    {
      const auto descriptor = descriptors.find({
        candidate.support_key.plane.id,
        candidate.support_key.plane.epoch});
      if (descriptor != descriptors.end())
      {
        geometry_version =
          descriptor->second.geometry_version;
      }
    }
    const auto lease = active.leaseFor(geometry_version);
    if (!lease.has_value())
    {
      return {};
    }
    candidate.active_plane_lease = *lease;
  }
  return store.prepare(snapshot, query);
}

bool bindSeedQueriesForTest(
  MultiViewLandmarkStore & store,
  ReadSnapshot & snapshot,
  const std::vector<LandmarkQuery *> & queries)
{
  std::map<
    std::pair<std::uint64_t, std::uint32_t>,
    ActivePlaneDescriptor> descriptors;
  for (const LandmarkQuery * query : queries)
  {
    if (query == nullptr)
    {
      return false;
    }
    for (const GeometrySeed & geometry_seed :
         query->geometry_seeds)
    {
      descriptors[{
        geometry_seed.geometry_version.plane.id,
        geometry_seed.geometry_version.plane.epoch}] =
        ActivePlaneDescriptor{
        geometry_seed.geometry_version,
        std::max<std::uint64_t>(
          1U, geometry_seed.geometry_version.revision)};
    }
  }
  std::vector<ActivePlaneDescriptor> active_descriptors;
  for (const auto & entry : descriptors)
  {
    active_descriptors.push_back(entry.second);
  }
  ActivePlaneTestAuthority & authority =
    activePlaneAuthorityForTest(store);
  if (!authority.replace(active_descriptors))
  {
    return false;
  }
  const ActivePlaneIdentitySnapshot active =
    authority.snapshot();
  snapshot.active_planes = active;
  for (LandmarkQuery * query : queries)
  {
    query->active_planes = active;
    for (GeometrySeed & geometry_seed :
         query->geometry_seeds)
    {
      const auto lease =
        active.leaseFor(geometry_seed.geometry_version);
      if (!lease.has_value())
      {
        return false;
      }
      geometry_seed.active_plane_lease = *lease;
    }
  }
  return true;
}

CommitRevisionSet revisions()
{
  CommitRevisionSet value;
  value.state.value = 17U;
  value.rig_calibration_revision = 3U;
  value.camera_model_revisions.fill(2U);
  value.photometric_calibration_revisions.fill(11U);
  value.time_transfer_revision = 13U;
  return value;
}

livo::LidarAnchoredCommitToken lidarTokenForTest(
  const GeometrySnapshotId geometry,
  const CommitRevisionSet & expected_revisions)
{
  return livo::test_support::makeLidarAnchoredToken(
    geometry, expected_revisions);
}

ReadSnapshot readSnapshot(
  const MultiViewLandmarkStore & store,
  const GeometrySnapshotId geometry)
{
  return store.snapshot(geometry, revisions());
}

void bindRevisions(LandmarkQuery & query)
{
  query.expected_revisions = revisions();
}

void bindTargetRevisions(
  LandmarkReferenceRequest & request)
{
  const CommitRevisionSet value = revisions();
  request.target_rig_calibration_revision =
    value.rig_calibration_revision;
  request.target_camera_model_revision =
    value.camera_model_revisions[request.target_camera_id];
  request.target_photometric_calibration_revision =
    value.photometric_calibration_revisions[
      request.target_camera_id];
  request.target_time_transfer_revision =
    value.time_transfer_revision;
}

livo::MapCommitProof bootstrapProof(
  const std::uint64_t batch,
  const MapRevision expected_store_revision,
  const GeometrySnapshotId geometry,
  const livo::PreparedLandmarkBatch & prepared)
{
  static_cast<void>(batch);
  static_cast<void>(expected_store_revision);
  return BootstrapGate::mint(
    lidarTokenForTest(geometry, revisions()),
    prepared);
}

ObservationCandidate observation(
  const GeometrySupportKey & support_key,
  const std::uint8_t camera_id,
  const double timestamp_sec,
  const ObservationWriteKind kind,
  const PhotometricStatus photometric_status)
{
  ObservationCandidate result;
  result.support_key = support_key;
  result.camera_id = camera_id;
  result.exposure_timestamp_sec = timestamp_sec;
  result.kind = kind;
  result.rotation_world_from_imu =
    Eigen::Matrix3d::Identity();
  result.position_world_m = Eigen::Vector3d::Zero();
  result.rig_calibration_revision = 3U;
  result.camera_model_revision = 2U;
  result.pixel = Eigen::Vector2d(320.0, 240.0);
  result.level = 0;
  result.owned_patch_pyramid = {{1.0F, 2.0F, 3.0F, 4.0F}};
  result.photometric_mode =
    kind == ObservationWriteKind::kAcceptedCrossCamera ?
    ObservationPhotometricMode::kCrossCameraCalibrated :
    ObservationPhotometricMode::kSameCameraOnly;
  result.photometric_status = photometric_status;
  result.photometric_cross_camera_comparable =
    kind == ObservationWriteKind::kAcceptedCrossCamera;
  result.photometric_calibration_revision = 11U;
  result.time_transfer_revision = 13U;
  using RuntimeSupport =
    livo::test_support::RuntimeFrameProvenanceTestSupport;
  static std::map<
    double, std::unique_ptr<RuntimeSupport>> runtimes;
  auto & runtime = runtimes[timestamp_sec];
  if (!runtime)
  {
    runtime = std::make_unique<RuntimeSupport>(
      timestamp_sec);
  }
  EXPECT_TRUE(runtime->valid());
  result.frame_provenance =
    runtime->frame(camera_id);
  result.pose_covariance =
    1.0e-6 * Eigen::Matrix<double, 6, 6>::Identity();
  return result;
}

livo::MapCommitProof tombstoneProof(
  const std::uint64_t batch,
  const MapRevision expected_store_revision,
  const GeometrySnapshotId geometry,
  const livo::PreparedLandmarkBatch & prepared)
{
  static_cast<void>(batch);
  static_cast<void>(expected_store_revision);
  return LidarTombstoneGate::mint(
    lidarTokenForTest(geometry, revisions()),
    prepared);
}

ReadSnapshot readSnapshotForRevisions(
  const MultiViewLandmarkStore & store,
  const GeometrySnapshotId geometry,
  const CommitRevisionSet & expected_revisions)
{
  return store.snapshot(geometry, expected_revisions);
}

ObservationCandidate observationForRevisions(
  const GeometrySupportKey & support_key,
  const std::uint8_t camera_id,
  const double timestamp_sec,
  const ObservationWriteKind kind,
  const CommitRevisionSet & expected_revisions)
{
  ObservationCandidate result = observation(
    support_key, camera_id, timestamp_sec, kind,
    PhotometricStatus::kValid);
  result.rig_calibration_revision =
    expected_revisions.rig_calibration_revision;
  result.camera_model_revision =
    expected_revisions.camera_model_revisions[camera_id];
  result.photometric_calibration_revision =
    expected_revisions.
    photometric_calibration_revisions[camera_id];
  result.time_transfer_revision =
    expected_revisions.time_transfer_revision;
  return result;
}

livo::MapCommitProof bootstrapProofForRevisions(
  const std::uint64_t batch,
  const MapRevision expected_store_revision,
  const GeometrySnapshotId geometry,
  const CommitRevisionSet & expected_revisions,
  const livo::PreparedLandmarkBatch & prepared)
{
  static_cast<void>(batch);
  static_cast<void>(expected_store_revision);
  return BootstrapGate::mint(
    lidarTokenForTest(geometry, expected_revisions),
    prepared);
}

TEST(MultiViewLandmarkStore, StableIdentityUsesPlaneEpochAndFrozenLocalCell)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{11U};
  const ReadSnapshot empty = readSnapshot(store, geometry);
  const GeometrySupportKey shared = supportKey(7U, 2U, 4, -3);

  LandmarkQuery bootstrap;
  bootstrap.batch.value = 101U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    GeometrySeed camera_seed = seed(shared, 5U);
    camera_seed.camera_id = camera_id;
    bootstrap.geometry_seeds.push_back(camera_seed);
  }
  bootstrap.geometry_seeds.push_back(
    seed(supportKey(7U, 2U, 5, -3), 5U));
  bootstrap.geometry_seeds.push_back(
    seed(supportKey(7U, 3U, 4, -3), 1U));

  auto prepared =
    prepareBootstrapForTest(store, empty, bootstrap);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(prepared.resolved_landmarks.size(), 6U);
  const auto shared_id = prepared.resolved_landmarks.front().landmark_id;
  ASSERT_NE(shared_id, 0U);
  for (std::size_t index = 0U; index < 4U; ++index)
  {
    EXPECT_EQ(
      prepared.resolved_landmarks[index].landmark_id,
      shared_id);
  }
  EXPECT_NE(prepared.resolved_landmarks[4U].landmark_id, shared_id);
  EXPECT_NE(prepared.resolved_landmarks[5U].landmark_id, shared_id);

  const auto proof = bootstrapProof(
    bootstrap.batch.value, empty.store_revision, geometry,
    prepared);
  ASSERT_TRUE(proof.valid());
  EXPECT_EQ(
    store.commit(prepared.take_delta(), proof).status,
    CommitStatus::kCommitted);

  const ReadSnapshot after_seed = readSnapshot(store, geometry);
  LandmarkQuery plane_update;
  plane_update.batch.value = 102U;
  plane_update.geometry = geometry;
  bindRevisions(plane_update);
  plane_update.geometry_seeds.push_back(seed(shared, 6U));
  const auto updated = store.prepare(after_seed, plane_update);
  EXPECT_FALSE(updated.valid());

  const ReadSnapshot after_update = readSnapshot(store, geometry);
  const auto *landmark = after_update.find(shared_id);
  ASSERT_NE(landmark, nullptr);
  EXPECT_EQ(landmark->geometry_version.revision, 5U);
  EXPECT_EQ(landmark->support_key, shared);
}

TEST(MultiViewLandmarkStore, ZeroPlaneEpochNeverStagesBootstrap)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{12U};
  const ReadSnapshot empty = readSnapshot(store, geometry);
  LandmarkQuery bootstrap;
  bootstrap.batch.value = 121U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  bootstrap.geometry_seeds.push_back(
    seed(supportKey(7U, 0U, 0, 0), 1U));

  EXPECT_FALSE(
    prepareBootstrapForTest(
      store, empty, bootstrap).valid());
  EXPECT_TRUE(readSnapshot(store, geometry).landmarks.empty());
}

TEST(
  MultiViewLandmarkStore,
  SnapshotFromForeignStoreCannotPrepareMutation)
{
  MultiViewLandmarkStore owner;
  MultiViewLandmarkStore foreign;
  const GeometrySnapshotId geometry{121U};
  const GeometrySupportKey key =
    supportKey(72U, 1U, 0, 0);
  LandmarkQuery bootstrap;
  bootstrap.batch.value = 122U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  bootstrap.geometry_seeds.push_back(seed(key, 1U));

  const ReadSnapshot initial = readSnapshot(owner, geometry);
  ASSERT_TRUE(
    prepareBootstrapForTest(
      owner, initial, bootstrap).valid());
  const ReadSnapshot owner_snapshot = owner.snapshot(
    geometry, revisions(), bootstrap.active_planes);
  ASSERT_TRUE(owner.prepare(owner_snapshot, bootstrap).valid());
  EXPECT_FALSE(
    foreign.prepare(owner_snapshot, bootstrap).valid());
  EXPECT_TRUE(readSnapshot(foreign, geometry).landmarks.empty());
}

TEST(
  MultiViewLandmarkStore,
  GeometryOnlyBootstrapDoesNotRequireVisualRevisions)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{13U};
  const CommitRevisionSet no_visual_revisions;
  const ReadSnapshot empty = readSnapshotForRevisions(
    store, geometry, no_visual_revisions);

  LandmarkQuery bootstrap;
  bootstrap.batch.value = 131U;
  bootstrap.geometry = geometry;
  bootstrap.expected_revisions = no_visual_revisions;
  bootstrap.geometry_seeds.push_back(
    seed(supportKey(8U, 1U, 0, 0), 1U));

  auto prepared =
    prepareBootstrapForTest(store, empty, bootstrap);
  ASSERT_TRUE(prepared.valid());
  EXPECT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProofForRevisions(
        bootstrap.batch.value,
        empty.store_revision,
        geometry,
        no_visual_revisions,
        prepared)).status,
    CommitStatus::kCommitted);
  EXPECT_EQ(
    readSnapshotForRevisions(
      store, geometry, no_visual_revisions).landmarks.size(),
    1U);
}

TEST(
  MultiViewLandmarkStore,
  NonPsdGeometryAndPoseCovariancesNeverStage)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{14U};
  const ReadSnapshot empty = readSnapshot(store, geometry);
  const GeometrySupportKey key = supportKey(8U, 1U, 1, 0);

  LandmarkQuery bad_position;
  bad_position.batch.value = 141U;
  bad_position.geometry = geometry;
  bindRevisions(bad_position);
  GeometrySeed bad_position_seed = seed(key, 1U);
  bad_position_seed.position_covariance_m2(0, 0) = -0.01;
  bad_position.geometry_seeds.push_back(bad_position_seed);
  EXPECT_FALSE(
    prepareBootstrapForTest(
      store, empty, bad_position).valid());

  LandmarkQuery bad_plane = bad_position;
  bad_plane.batch.value = 142U;
  bad_plane.geometry_seeds.clear();
  GeometrySeed bad_plane_seed = seed(key, 1U);
  bad_plane_seed.plane_covariance(5, 5) = -0.001;
  bad_plane.geometry_seeds.push_back(bad_plane_seed);
  EXPECT_FALSE(
    prepareBootstrapForTest(
      store, empty, bad_plane).valid());

  LandmarkQuery bad_pose;
  bad_pose.batch.value = 143U;
  bad_pose.geometry = geometry;
  bindRevisions(bad_pose);
  bad_pose.geometry_seeds.push_back(seed(key, 1U));
  ObservationCandidate candidate = observation(
    key, 0U, 1.0,
    ObservationWriteKind::kBootstrapReference,
    PhotometricStatus::kValid);
  candidate.pose_covariance(0, 0) = -1.0e-6;
  bad_pose.observations.push_back(candidate);
  EXPECT_FALSE(
    prepareBootstrapForTest(
      store, empty, bad_pose).valid());
  EXPECT_TRUE(readSnapshot(store, geometry).landmarks.empty());
}

TEST(
  MultiViewLandmarkStore,
  ObservationWithoutRuntimeFrameProvenanceNeverStages)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{141U};
  const ReadSnapshot empty = readSnapshot(store, geometry);
  const GeometrySupportKey key =
    supportKey(81U, 1U, 1, 0);

  LandmarkQuery bootstrap;
  bootstrap.batch.value = 144U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  bootstrap.geometry_seeds.push_back(seed(key, 1U));
  ObservationCandidate candidate = observation(
    key, 0U, 1.0,
    ObservationWriteKind::kBootstrapReference,
    PhotometricStatus::kValid);
  candidate.frame_provenance.reset();
  bootstrap.observations.push_back(std::move(candidate));

  EXPECT_FALSE(
    prepareBootstrapForTest(
      store, empty, bootstrap).valid());
  const ReadSnapshot after = readSnapshot(store, geometry);
  EXPECT_EQ(after.store_revision.value, 0U);
  EXPECT_TRUE(after.landmarks.empty());
}

TEST(
  MultiViewLandmarkStore,
  CameraScopedRevisionsBindMixedTransactionsAtomically)
{
  CommitRevisionSet scoped;
  scoped.state.value = 23U;
  scoped.rig_calibration_revision = 5U;
  scoped.time_transfer_revision = 17U;
  scoped.camera_model_revisions[0U] = 2U;
  scoped.photometric_calibration_revisions[0U] = 11U;
  scoped.camera_model_revisions[2U] = 7U;
  scoped.photometric_calibration_revisions[2U] = 13U;

  const GeometrySnapshotId geometry{15U};
  const GeometrySupportKey camera_zero_key =
    supportKey(80U, 1U, 0, 0);
  const GeometrySupportKey camera_two_key =
    supportKey(81U, 1U, 0, 0);

  auto make_query = [&]() {
      LandmarkQuery query;
      query.batch.value = 151U;
      query.geometry = geometry;
      query.expected_revisions = scoped;
      query.geometry_seeds.push_back(seed(camera_zero_key, 1U));
      query.geometry_seeds.push_back(seed(camera_two_key, 1U));
      query.observations.push_back(observationForRevisions(
        camera_zero_key, 0U, 1.0,
        ObservationWriteKind::kBootstrapReference, scoped));
      query.observations.push_back(observationForRevisions(
        camera_two_key, 2U, 1.0,
        ObservationWriteKind::kBootstrapReference, scoped));
      return query;
    };

  MultiViewLandmarkStore accepts_uninvolved_change;
  ReadSnapshot base = readSnapshotForRevisions(
    accepts_uninvolved_change, geometry, scoped);
  LandmarkQuery query = make_query();
  auto prepared =
    prepareBootstrapForTest(
      accepts_uninvolved_change, base, query);
  ASSERT_TRUE(prepared.valid());
  CommitRevisionSet uninvolved_change = scoped;
  uninvolved_change.camera_model_revisions[1U] = 99U;
  uninvolved_change.photometric_calibration_revisions[1U] = 101U;
  EXPECT_EQ(
    accepts_uninvolved_change.commit(
      prepared.take_delta(),
      bootstrapProofForRevisions(
        query.batch.value, base.store_revision, geometry,
        uninvolved_change, prepared)).status,
    CommitStatus::kCommitted);
  EXPECT_EQ(
    readSnapshotForRevisions(
      accepts_uninvolved_change, geometry, scoped).
      landmarks.size(),
    2U);

  MultiViewLandmarkStore rejects_involved_change;
  base = readSnapshotForRevisions(
    rejects_involved_change, geometry, scoped);
  query = make_query();
  prepared = prepareBootstrapForTest(
    rejects_involved_change, base, query);
  ASSERT_TRUE(prepared.valid());
  CommitRevisionSet involved_change = scoped;
  ++involved_change.camera_model_revisions[2U];
  EXPECT_EQ(
    rejects_involved_change.commit(
      prepared.take_delta(),
      bootstrapProofForRevisions(
        query.batch.value, base.store_revision, geometry,
        involved_change, prepared)).status,
    CommitStatus::kProofMismatch);
  const ReadSnapshot rejected = readSnapshotForRevisions(
    rejects_involved_change, geometry, scoped);
  EXPECT_EQ(rejected.store_revision.value, base.store_revision.value);
  EXPECT_TRUE(rejected.landmarks.empty());
}

TEST(
  MultiViewLandmarkStore,
  TombstoneGarbageCollectionIsBoundedAndCannotResurrect)
{
  MultiViewLandmarkStoreConfig config;
  config.maximum_tombstoned_landmarks = 32U;
  MultiViewLandmarkStore store(config);
  const GeometrySnapshotId geometry{19U};
  const PlaneIdentity plane{86U, 1U};
  ReadSnapshot current = readSnapshot(store, geometry);

  LandmarkQuery bootstrap;
  bootstrap.batch.value = 190U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  for (std::int32_t cell = 0; cell < 2000; ++cell)
  {
    bootstrap.geometry_seeds.push_back(seed(
      supportKey(plane.id, plane.epoch, cell, 0), 1U));
  }
  auto prepared =
    prepareBootstrapForTest(store, current, bootstrap);
  ASSERT_TRUE(prepared.valid());
  const livo::LandmarkId first_id =
    prepared.resolved_landmarks.front().landmark_id;
  const livo::LandmarkId last_id =
    prepared.resolved_landmarks.back().landmark_id;
  const auto bootstrap_proof = bootstrapProof(
    bootstrap.batch.value, current.store_revision,
    geometry, prepared);
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(), bootstrap_proof).status,
    CommitStatus::kCommitted);

  current = readSnapshot(store, geometry);
  ASSERT_EQ(current.landmarks.size(), 2000U);
  LandmarkQuery stale_bootstrap;
  stale_bootstrap.batch.value = 191U;
  stale_bootstrap.geometry = geometry;
  bindRevisions(stale_bootstrap);
  stale_bootstrap.geometry_seeds.push_back(
    seed(supportKey(87U, 1U, 0, 0), 1U));
  auto stale_prepared =
    prepareBootstrapForTest(
      store, current, stale_bootstrap);
  ASSERT_TRUE(stale_prepared.valid());
  const auto stale_proof = bootstrapProof(
    stale_bootstrap.batch.value, current.store_revision,
    geometry, stale_prepared);

  LandmarkQuery tombstone;
  tombstone.batch.value = 192U;
  tombstone.geometry = geometry;
  bindRevisions(tombstone);
  PlaneTombstoneMutation mutation;
  mutation.old_identity = plane;
  mutation.last_revision = 1U;
  mutation.generation = 1U;
  tombstone.plane_tombstones.push_back(mutation);
  prepared = store.prepare(current, tombstone);
  ASSERT_TRUE(prepared.valid());
  const auto tombstone_proof = tombstoneProof(
    tombstone.batch.value, current.store_revision,
    geometry, prepared);
  const auto committed = store.commit(
    prepared.take_delta(), tombstone_proof);
  ASSERT_EQ(committed.status, CommitStatus::kCommitted);
  EXPECT_EQ(committed.tombstoned_landmarks, 2000U);
  EXPECT_EQ(committed.purged_tombstones, 1968U);

  const ReadSnapshot collected = readSnapshot(store, geometry);
  ASSERT_EQ(collected.landmarks.size(), 32U);
  EXPECT_EQ(collected.find(first_id), nullptr);
  ASSERT_NE(collected.find(last_id), nullptr);
  EXPECT_EQ(
    collected.find(last_id)->lifecycle,
    LandmarkLifecycle::kTombstoned);
  EXPECT_NE(
    collected.find(last_id)->
      tombstoned_at_store_revision.value,
    0U);

  EXPECT_EQ(
    store.commit(
      stale_prepared.take_delta(), stale_proof).status,
    CommitStatus::kStaleStoreRevision);
  EXPECT_EQ(
    readSnapshot(store, geometry).landmarks.size(),
    32U);

  LandmarkQuery purged_reference;
  purged_reference.batch.value = 193U;
  purged_reference.geometry = geometry;
  bindRevisions(purged_reference);
  LandmarkReferenceRequest request;
  request.landmark_id = first_id;
  request.target_camera_id = 0U;
  bindTargetRevisions(request);
  request.target_timestamp_sec = 1.0;
  request.maximum_reference_age_sec = 1.0;
  request.selection_score = 1.0;
  request.target_camera_healthy = true;
  purged_reference.reference_requests.push_back(request);
  const auto selected =
    store.prepare(collected, purged_reference);
  ASSERT_TRUE(selected.valid());
  EXPECT_TRUE(selected.selected_patches.empty());
}

TEST(MultiViewLandmarkStore, ProofAndCasFailuresAreAtomic)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{21U};
  const GeometrySupportKey established =
    supportKey(9U, 1U, 0, 0);

  const ReadSnapshot empty = readSnapshot(store, geometry);
  LandmarkQuery initial;
  initial.batch.value = 201U;
  initial.geometry = geometry;
  bindRevisions(initial);
  initial.geometry_seeds.push_back(seed(established, 4U));
  auto initial_prepared =
    prepareBootstrapForTest(store, empty, initial);
  ASSERT_TRUE(initial_prepared.valid());
  const auto initial_proof = bootstrapProof(
    initial.batch.value, empty.store_revision, geometry,
    initial_prepared);
  ASSERT_EQ(
    store.commit(
      initial_prepared.take_delta(), initial_proof).status,
    CommitStatus::kCommitted);

  const ReadSnapshot base = readSnapshot(store, geometry);
  const GeometrySupportKey wrong_proof_key =
    supportKey(10U, 1U, 1, 0);
  LandmarkQuery wrong_proof_query;
  wrong_proof_query.batch.value = 202U;
  wrong_proof_query.geometry = geometry;
  bindRevisions(wrong_proof_query);
  wrong_proof_query.geometry_seeds.push_back(
    seed(wrong_proof_key, 1U));
  ReadSnapshot bound_base = base;
  LandmarkQuery wrong_batch_challenge_query =
    wrong_proof_query;
  ++wrong_batch_challenge_query.batch.value;
  ASSERT_TRUE(bindSeedQueriesForTest(
    store, bound_base,
    {&wrong_proof_query, &wrong_batch_challenge_query}));
  auto wrong_proof_prepared =
    store.prepare(bound_base, wrong_proof_query);
  ASSERT_TRUE(wrong_proof_prepared.valid());
  const auto wrong_batch_challenge =
    store.prepare(
      bound_base, wrong_batch_challenge_query);
  ASSERT_TRUE(wrong_batch_challenge.valid());
  const auto wrong_batch_proof = bootstrapProof(
    wrong_proof_query.batch.value + 1U,
    base.store_revision,
    geometry,
    wrong_batch_challenge);
  EXPECT_EQ(
    store.commit(
      wrong_proof_prepared.take_delta(),
      wrong_batch_proof).status,
    CommitStatus::kProofMismatch);
  EXPECT_EQ(
    readSnapshot(store, geometry).store_revision.value,
    base.store_revision.value);

  LandmarkQuery first;
  first.batch.value = 203U;
  first.geometry = geometry;
  bindRevisions(first);
  first.geometry_seeds.push_back(
    seed(supportKey(11U, 1U, 0, 0), 1U));
  LandmarkQuery second;
  second.batch.value = 204U;
  second.geometry = geometry;
  bindRevisions(second);
  const GeometrySupportKey stale_cas_key =
    supportKey(12U, 1U, 0, 0);
  second.geometry_seeds.push_back(seed(stale_cas_key, 1U));
  bound_base = base;
  ASSERT_TRUE(bindSeedQueriesForTest(
    store, bound_base, {&first, &second}));
  auto first_prepared =
    store.prepare(bound_base, first);
  auto second_prepared =
    store.prepare(bound_base, second);
  ASSERT_TRUE(first_prepared.valid());
  ASSERT_TRUE(second_prepared.valid());
  const auto first_proof = bootstrapProof(
    first.batch.value, base.store_revision, geometry,
    first_prepared);
  const auto second_proof = bootstrapProof(
    second.batch.value, base.store_revision, geometry,
    second_prepared);
  ASSERT_EQ(
    store.commit(first_prepared.take_delta(), first_proof).status,
    CommitStatus::kCommitted);
  const ReadSnapshot after_first = readSnapshot(store, geometry);
  EXPECT_EQ(
    store.commit(second_prepared.take_delta(), second_proof).status,
    CommitStatus::kStaleStoreRevision);
  EXPECT_EQ(
    after_first.landmarks.size(),
    readSnapshot(store, geometry).landmarks.size());

  LandmarkQuery atomic_stale;
  atomic_stale.batch.value = 205U;
  atomic_stale.geometry = geometry;
  bindRevisions(atomic_stale);
  const GeometrySupportKey would_be_new =
    supportKey(13U, 1U, 0, 0);
  atomic_stale.geometry_seeds.push_back(seed(would_be_new, 1U));
  atomic_stale.geometry_seeds.push_back(seed(established, 3U));
  auto atomic_prepared =
    prepareBootstrapForTest(
      store, after_first, atomic_stale);
  EXPECT_FALSE(atomic_prepared.valid());

  const ReadSnapshot after_rejection =
    readSnapshot(store, geometry);
  EXPECT_EQ(
    after_rejection.store_revision.value,
    after_first.store_revision.value);
  EXPECT_EQ(
    std::count_if(
      after_rejection.landmarks.begin(),
      after_rejection.landmarks.end(),
      [&would_be_new](const auto & landmark) {
        return landmark.support_key == would_be_new;
      }),
    0);
  EXPECT_EQ(
    std::count_if(
      after_rejection.landmarks.begin(),
      after_rejection.landmarks.end(),
      [&stale_cas_key](const auto & landmark) {
        return landmark.support_key == stale_cas_key;
      }),
    0);
}

TEST(
  MultiViewLandmarkStore,
  LidarTombstoneCasHidesOldEpochAndNewEpochGetsDistinctIds)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{25U};
  const PlaneIdentity old_plane{90U, 1U};
  const GeometrySupportKey old_a =
    supportKey(old_plane.id, old_plane.epoch, 0, 0);
  const GeometrySupportKey old_b =
    supportKey(old_plane.id, old_plane.epoch, 1, 0);

  ReadSnapshot current = readSnapshot(store, geometry);
  LandmarkQuery bootstrap;
  bootstrap.batch.value = 251U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  bootstrap.geometry_seeds.push_back(seed(old_a, 4U));
  bootstrap.geometry_seeds.push_back(seed(old_b, 4U));
  bootstrap.observations.push_back(observation(
    old_a, 0U, 10.0,
    ObservationWriteKind::kBootstrapReference,
    PhotometricStatus::kValid));
  auto prepared =
    prepareBootstrapForTest(store, current, bootstrap);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(prepared.resolved_landmarks.size(), 2U);
  const livo::LandmarkId old_a_id =
    prepared.resolved_landmarks[0U].landmark_id;
  const livo::LandmarkId old_b_id =
    prepared.resolved_landmarks[1U].landmark_id;
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(
        bootstrap.batch.value, current.store_revision,
        geometry, prepared)).status,
    CommitStatus::kCommitted);

  current = readSnapshot(store, geometry);
  LandmarkQuery selectable;
  selectable.batch.value = 252U;
  selectable.geometry = geometry;
  bindRevisions(selectable);
  LandmarkReferenceRequest request;
  request.landmark_id = old_a_id;
  request.target_camera_id = 0U;
  bindTargetRevisions(request);
  request.target_timestamp_sec = 10.1;
  request.maximum_reference_age_sec = 1.0;
  request.selection_score = 1.0;
  request.target_camera_healthy = true;
  selectable.reference_requests.push_back(request);
  ASSERT_EQ(
    store.prepare(current, selectable).selected_patches.size(), 1U);

  LandmarkQuery tombstone;
  tombstone.batch.value = 253U;
  tombstone.geometry = geometry;
  bindRevisions(tombstone);
  PlaneTombstoneMutation mutation;
  mutation.old_identity = old_plane;
  mutation.last_revision = 4U;
  mutation.generation = 12U;
  tombstone.plane_tombstones.push_back(mutation);

  prepared = store.prepare(current, tombstone);
  ASSERT_TRUE(prepared.valid());
  const MapRevision before_wrong_proof = current.store_revision;
  EXPECT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(
        tombstone.batch.value, current.store_revision,
        geometry, prepared)).status,
    CommitStatus::kProofMismatch);
  current = readSnapshot(store, geometry);
  EXPECT_EQ(current.store_revision.value, before_wrong_proof.value);
  EXPECT_NE(
    current.find(old_a_id)->lifecycle,
    LandmarkLifecycle::kTombstoned);
  EXPECT_NE(
    current.find(old_b_id)->lifecycle,
    LandmarkLifecycle::kTombstoned);

  prepared = store.prepare(current, tombstone);
  ASSERT_TRUE(prepared.valid());
  LandmarkQuery concurrent;
  concurrent.batch.value = 254U;
  concurrent.geometry = geometry;
  bindRevisions(concurrent);
  concurrent.geometry_seeds.push_back(
    seed(supportKey(91U, 1U, 0, 0), 1U));
  auto concurrent_prepared =
    prepareBootstrapForTest(
      store, current, concurrent);
  ASSERT_TRUE(concurrent_prepared.valid());
  ASSERT_EQ(
    store.commit(
      concurrent_prepared.take_delta(),
      bootstrapProof(
        concurrent.batch.value, current.store_revision,
        geometry, concurrent_prepared)).status,
    CommitStatus::kCommitted);
  EXPECT_EQ(
    store.commit(
      prepared.take_delta(),
      tombstoneProof(
        tombstone.batch.value, current.store_revision,
        geometry, prepared)).status,
    CommitStatus::kStaleStoreRevision);
  current = readSnapshot(store, geometry);
  EXPECT_NE(
    current.find(old_a_id)->lifecycle,
    LandmarkLifecycle::kTombstoned);
  EXPECT_NE(
    current.find(old_b_id)->lifecycle,
    LandmarkLifecycle::kTombstoned);

  tombstone.batch.value = 255U;
  prepared = store.prepare(current, tombstone);
  ASSERT_TRUE(prepared.valid());
  const livo::CommitResult committed = store.commit(
    prepared.take_delta(),
    tombstoneProof(
      tombstone.batch.value, current.store_revision,
      geometry, prepared));
  ASSERT_EQ(committed.status, CommitStatus::kCommitted);
  EXPECT_EQ(committed.tombstoned_landmarks, 2U);

  current = readSnapshot(store, geometry);
  ASSERT_EQ(
    current.find(old_a_id)->lifecycle,
    LandmarkLifecycle::kTombstoned);
  ASSERT_EQ(
    current.find(old_b_id)->lifecycle,
    LandmarkLifecycle::kTombstoned);
  selectable.batch.value = 256U;
  EXPECT_TRUE(
    store.prepare(current, selectable).selected_patches.empty());

  const GeometrySupportKey new_a =
    supportKey(old_plane.id, old_plane.epoch + 1U, 0, 0);
  const GeometrySupportKey new_b =
    supportKey(old_plane.id, old_plane.epoch + 1U, 1, 0);
  LandmarkQuery next_epoch;
  next_epoch.batch.value = 257U;
  next_epoch.geometry = geometry;
  bindRevisions(next_epoch);
  next_epoch.geometry_seeds.push_back(seed(new_a, 1U));
  next_epoch.geometry_seeds.push_back(seed(new_b, 1U));
  prepared =
    prepareBootstrapForTest(store, current, next_epoch);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(prepared.resolved_landmarks.size(), 2U);
  EXPECT_NE(
    prepared.resolved_landmarks[0U].landmark_id, old_a_id);
  EXPECT_NE(
    prepared.resolved_landmarks[1U].landmark_id, old_b_id);
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(
        next_epoch.batch.value, current.store_revision,
        geometry, prepared)).status,
    CommitStatus::kCommitted);
}

TEST(MultiViewLandmarkStore, StaleTombstoneRevisionIsRejectedBeforeStaging)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{26U};
  const GeometrySupportKey key = supportKey(92U, 1U, 0, 0);
  ReadSnapshot current = readSnapshot(store, geometry);
  LandmarkQuery bootstrap;
  bootstrap.batch.value = 261U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  bootstrap.geometry_seeds.push_back(seed(key, 5U));
  auto prepared =
    prepareBootstrapForTest(store, current, bootstrap);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(
        bootstrap.batch.value, current.store_revision,
        geometry, prepared)).status,
    CommitStatus::kCommitted);

  current = readSnapshot(store, geometry);
  LandmarkQuery stale;
  stale.batch.value = 262U;
  stale.geometry = geometry;
  bindRevisions(stale);
  PlaneTombstoneMutation mutation;
  mutation.old_identity = key.plane;
  mutation.last_revision = 4U;
  mutation.generation = 20U;
  stale.plane_tombstones.push_back(mutation);
  EXPECT_FALSE(store.prepare(current, stale).valid());
  EXPECT_EQ(
    readSnapshot(store, geometry).store_revision.value,
    current.store_revision.value);
  EXPECT_NE(
    readSnapshot(store, geometry).landmarks.front().lifecycle,
    LandmarkLifecycle::kTombstoned);
}

TEST(
  MultiViewLandmarkStore,
  RevisionAndGeometryCasRejectEveryStaleTypedProofAtomically)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{27U};
  const CommitRevisionSet current_revisions = revisions();
  std::array<CommitRevisionSet, 5U> stale_revisions;
  stale_revisions.fill(current_revisions);
  ++stale_revisions[0U].state.value;
  ++stale_revisions[1U].rig_calibration_revision;
  ++stale_revisions[2U].camera_model_revisions[2U];
  ++stale_revisions[3U].
    photometric_calibration_revisions[3U];
  ++stale_revisions[4U].time_transfer_revision;

  for (std::size_t index = 0U;
       index < stale_revisions.size(); ++index)
  {
    const ReadSnapshot snapshot =
      store.snapshot(geometry, current_revisions);
    LandmarkQuery query;
    query.batch.value = 270U + index;
    query.geometry = geometry;
    query.expected_revisions = current_revisions;
    const GeometrySupportKey camera_two_key =
      supportKey(70U + 2U * index, 1U, 0, 0);
    const GeometrySupportKey camera_three_key =
      supportKey(71U + 2U * index, 1U, 0, 0);
    query.geometry_seeds.push_back(seed(camera_two_key, 1U));
    query.geometry_seeds.push_back(seed(camera_three_key, 1U));
    query.observations.push_back(observationForRevisions(
      camera_two_key, 2U, 1.0,
      ObservationWriteKind::kBootstrapReference,
      current_revisions));
    query.observations.push_back(observationForRevisions(
      camera_three_key, 3U, 1.0,
      ObservationWriteKind::kBootstrapReference,
      current_revisions));
    auto prepared =
      prepareBootstrapForTest(store, snapshot, query);
    ASSERT_TRUE(prepared.valid());

    const auto stale_proof = BootstrapGate::mint(
      lidarTokenForTest(
        geometry, stale_revisions[index]),
      prepared);
    EXPECT_EQ(
      store.commit(
        prepared.take_delta(), stale_proof).status,
      CommitStatus::kProofMismatch);
    const ReadSnapshot after =
      store.snapshot(geometry, current_revisions);
    EXPECT_EQ(
      after.store_revision.value,
      snapshot.store_revision.value);
    EXPECT_TRUE(after.landmarks.empty());
  }

  const ReadSnapshot snapshot =
    store.snapshot(geometry, current_revisions);
  LandmarkQuery query;
  query.batch.value = 280U;
  query.geometry = geometry;
  query.expected_revisions = current_revisions;
  query.geometry_seeds.push_back(
    seed(supportKey(80U, 1U, 0, 0), 1U));
  auto prepared =
    prepareBootstrapForTest(store, snapshot, query);
  ASSERT_TRUE(prepared.valid());
  const auto wrong_geometry_proof = BootstrapGate::mint(
    lidarTokenForTest(
      GeometrySnapshotId{geometry.value + 1U},
      current_revisions),
    prepared);
  EXPECT_EQ(
    store.commit(
      prepared.take_delta(), wrong_geometry_proof).status,
    CommitStatus::kProofMismatch);
  EXPECT_TRUE(
    store.snapshot(
      geometry, current_revisions).landmarks.empty());
}

TEST(
  MultiViewLandmarkStore,
  VisualObservationCommitRejectsLidarAuthorityWithoutWrites)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{31U};
  const GeometrySupportKey key =
    supportKey(20U, 1U, 2, 4);

  const ReadSnapshot empty = readSnapshot(store, geometry);
  LandmarkQuery bootstrap;
  bootstrap.batch.value = 301U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  bootstrap.geometry_seeds.push_back(seed(key, 1U));
  bootstrap.observations.push_back(observation(
    key, 0U, 10.0,
    ObservationWriteKind::kBootstrapReference,
    PhotometricStatus::kValid));
  auto bootstrap_prepared =
    prepareBootstrapForTest(store, empty, bootstrap);
  ASSERT_TRUE(bootstrap_prepared.valid());
  ASSERT_EQ(
    store.commit(
      bootstrap_prepared.take_delta(),
      bootstrapProof(
        bootstrap.batch.value,
        empty.store_revision,
        geometry,
        bootstrap_prepared)).status,
    CommitStatus::kCommitted);

  ReadSnapshot current = readSnapshot(store, geometry);
  ASSERT_EQ(current.landmarks.size(), 1U);
  EXPECT_EQ(
    current.landmarks.front().lifecycle,
    LandmarkLifecycle::kLidarSeeded);
  ASSERT_EQ(current.landmarks.front().observations.size(), 1U);
  EXPECT_TRUE(
    current.landmarks.front().observations.front().reference_only);

  LandmarkQuery rejected_same_camera;
  rejected_same_camera.batch.value = 302U;
  rejected_same_camera.geometry = geometry;
  bindRevisions(rejected_same_camera);
  rejected_same_camera.observations.push_back(observation(
    key, 0U, 10.1,
    ObservationWriteKind::kAcceptedSameCamera,
    PhotometricStatus::kValid));
  auto rejected_prepared =
    store.prepare(current, rejected_same_camera);
  ASSERT_TRUE(rejected_prepared.valid());
  const auto rejected_proof = bootstrapProof(
    rejected_same_camera.batch.value,
    current.store_revision,
    geometry,
    rejected_prepared);
  EXPECT_EQ(
    store.commit(
      rejected_prepared.take_delta(),
      rejected_proof).status,
    CommitStatus::kProofMismatch);
  EXPECT_EQ(
    readSnapshot(store, geometry).
    landmarks.front().observations.size(),
    1U);
}

TEST(
  MultiViewLandmarkStore,
  ExistingLidarSeedAcceptsExactlyOneBootstrapObservation)
{
  MultiViewLandmarkStore store;
  const GeometrySnapshotId geometry{35U};
  const GeometrySupportKey key =
    supportKey(25U, 1U, 3, -2);

  ReadSnapshot current = readSnapshot(store, geometry);
  LandmarkQuery lidar_seed;
  lidar_seed.batch.value = 351U;
  lidar_seed.geometry = geometry;
  bindRevisions(lidar_seed);
  lidar_seed.geometry_seeds.push_back(seed(key, 4U));
  auto prepared =
    prepareBootstrapForTest(store, current, lidar_seed);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(
        lidar_seed.batch.value,
        current.store_revision,
        geometry,
        prepared)).status,
    CommitStatus::kCommitted);

  current = readSnapshot(store, geometry);
  const MapRevision before_bootstrap =
    current.store_revision;
  LandmarkQuery bootstrap;
  bootstrap.batch.value = 352U;
  bootstrap.geometry = geometry;
  bindRevisions(bootstrap);
  bootstrap.observations.push_back(observation(
    key, 0U, 12.0,
    ObservationWriteKind::kBootstrapReference,
    PhotometricStatus::kValid));
  prepared =
    prepareBootstrapForTest(store, current, bootstrap);
  ASSERT_TRUE(prepared.valid());

  const auto wrong_proof = LidarTombstoneGate::mint(
    lidarTokenForTest(geometry, revisions()),
    prepared);
  EXPECT_EQ(
    store.commit(
      prepared.take_delta(), wrong_proof).status,
    CommitStatus::kProofMismatch);
  EXPECT_EQ(
    readSnapshot(store, geometry).store_revision.value,
    before_bootstrap.value);
  EXPECT_TRUE(
    readSnapshot(store, geometry).landmarks.front().
    observations.empty());

  current = readSnapshot(store, geometry);
  prepared =
    prepareBootstrapForTest(store, current, bootstrap);
  ASSERT_TRUE(prepared.valid());
  LandmarkQuery concurrent;
  concurrent.batch.value = 353U;
  concurrent.geometry = geometry;
  bindRevisions(concurrent);
  concurrent.geometry_seeds.push_back(
    seed(supportKey(26U, 1U, 0, 0), 1U));
  auto concurrent_prepared =
    prepareBootstrapForTest(
      store, current, concurrent);
  ASSERT_TRUE(concurrent_prepared.valid());
  ASSERT_EQ(
    store.commit(
      concurrent_prepared.take_delta(),
      bootstrapProof(
        concurrent.batch.value,
        current.store_revision,
        geometry,
        concurrent_prepared)).status,
    CommitStatus::kCommitted);
  EXPECT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(
        bootstrap.batch.value,
        current.store_revision,
        geometry,
        prepared)).status,
    CommitStatus::kStaleStoreRevision);
  EXPECT_TRUE(
    readSnapshot(store, geometry).
    find(1U)->observations.empty());

  current = readSnapshot(store, geometry);
  bootstrap.batch.value = 354U;
  prepared =
    prepareBootstrapForTest(store, current, bootstrap);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(
        bootstrap.batch.value,
        current.store_revision,
        geometry,
        prepared)).status,
    CommitStatus::kCommitted);
  const ReadSnapshot bootstrapped =
    readSnapshot(store, geometry);
  ASSERT_EQ(
    bootstrapped.find(1U)->observations.size(), 1U);
  EXPECT_EQ(
    bootstrapped.find(1U)->lifecycle,
    LandmarkLifecycle::kLidarSeeded);

  bootstrap.batch.value = 355U;
  const auto duplicate =
    prepareBootstrapForTest(
      store, bootstrapped, bootstrap);
  EXPECT_FALSE(duplicate.valid());
  EXPECT_EQ(
    readSnapshot(store, geometry).store_revision.value,
    bootstrapped.store_revision.value);
  EXPECT_EQ(
    readSnapshot(store, geometry).
    find(1U)->observations.size(),
    1U);
}

TEST(
  MultiViewLandmarkStore,
  PurgedTombstoneCannotBeResurrectedByFreshGenericLidarToken)
{
  MultiViewLandmarkStoreConfig config;
  config.maximum_tombstoned_landmarks = 1U;
  MultiViewLandmarkStore store(config);
  ActivePlaneTestAuthority authority;
  const GeometrySnapshotId geometry{77U};
  const PlaneIdentity old_plane{77U, 1U};
  const GeometrySupportKey old_cell_zero =
    supportKey(old_plane.id, old_plane.epoch, 0, 0);
  const GeometrySupportKey old_cell_one =
    supportKey(old_plane.id, old_plane.epoch, 1, 0);
  ASSERT_TRUE(authority.replace(
    {ActivePlaneDescriptor{
      livo::GeometryVersion{old_plane, 1U}, 10U}}));

  ActivePlaneIdentitySnapshot active_planes =
    authority.snapshot();
  ReadSnapshot current = store.snapshot(
    geometry, revisions(), active_planes);
  LandmarkQuery bootstrap;
  bootstrap.batch.value = 7701U;
  bootstrap.geometry = geometry;
  bootstrap.expected_revisions = revisions();
  bootstrap.active_planes = active_planes;
  bootstrap.geometry_seeds.push_back(
    activeSeed(active_planes, old_cell_zero, 1U));
  bootstrap.geometry_seeds.push_back(
    activeSeed(active_planes, old_cell_one, 1U));
  auto prepared = store.prepare(current, bootstrap);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      BootstrapGate::mint(
        lidarTokenForTest(
          geometry, revisions()),
        prepared)).status,
    CommitStatus::kCommitted);

  current = readSnapshot(store, geometry);
  LandmarkQuery tombstone;
  tombstone.batch.value = 7702U;
  tombstone.geometry = geometry;
  tombstone.expected_revisions = revisions();
  tombstone.plane_tombstones.push_back(
    PlaneTombstoneMutation{old_plane, 1U, 11U});
  prepared = store.prepare(current, tombstone);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      LidarTombstoneGate::mint(
        lidarTokenForTest(
          geometry, revisions()),
        prepared)).status,
    CommitStatus::kCommitted);
  ASSERT_EQ(readSnapshot(store, geometry).landmarks.size(), 1U);

  ASSERT_TRUE(authority.replace({}));
  const ActivePlaneIdentitySnapshot after_delete =
    authority.snapshot();
  current = store.snapshot(
    geometry, revisions(), after_delete);
  LandmarkQuery resurrection;
  resurrection.batch.value = 7703U;
  resurrection.geometry = geometry;
  resurrection.expected_revisions = revisions();
  resurrection.active_planes = after_delete;
  resurrection.geometry_seeds.push_back(
    seed(old_cell_zero, 1U));

  const auto rejected =
    store.prepare(current, resurrection);
  EXPECT_FALSE(rejected.valid());
  const ReadSnapshot unchanged =
    readSnapshot(store, geometry);
  EXPECT_EQ(unchanged.store_revision.value, 2U);
  EXPECT_EQ(unchanged.landmarks.size(), 1U);
  EXPECT_EQ(unchanged.next_landmark_id, 3U);
}

TEST(
  ActivePlaneIdentityRegistry,
  MajorReplacementInvalidatesOldEpochAndLeasesNewEpoch)
{
  ActivePlaneTestAuthority authority;
  const PlaneIdentity old_plane{88U, 1U};
  const PlaneIdentity new_plane{88U, 2U};
  ASSERT_TRUE(authority.replace(
    {ActivePlaneDescriptor{
      livo::GeometryVersion{old_plane, 4U}, 21U}}));
  const ActivePlaneIdentitySnapshot old_snapshot =
    authority.snapshot();
  const auto old_lease = old_snapshot.leaseFor(
    livo::GeometryVersion{old_plane, 4U});
  ASSERT_TRUE(old_lease.has_value());
  ASSERT_TRUE(old_lease->valid());

  ASSERT_TRUE(authority.replace(
    {ActivePlaneDescriptor{
      livo::GeometryVersion{new_plane, 1U}, 22U}}));
  EXPECT_FALSE(old_snapshot.valid());
  EXPECT_FALSE(old_lease->valid());
  const ActivePlaneIdentitySnapshot current =
    authority.snapshot();
  EXPECT_FALSE(current.leaseFor(
    livo::GeometryVersion{old_plane, 4U}).has_value());
  const auto new_lease = current.leaseFor(
    livo::GeometryVersion{new_plane, 1U});
  ASSERT_TRUE(new_lease.has_value());
  EXPECT_TRUE(new_lease->valid());
  EXPECT_EQ(new_lease->generation(), 22U);
}

TEST(
  ActivePlaneIdentityRegistry,
  RegistryAffinitySurvivesRevisionChangesButRejectsForeignTwin)
{
  ActivePlaneTestAuthority authority;
  ActivePlaneTestAuthority foreign_authority;
  const GeometryVersion version{
    PlaneIdentity{89U, 1U}, 3U};
  const std::vector<ActivePlaneDescriptor> descriptors{
    ActivePlaneDescriptor{version, 31U}};
  ASSERT_TRUE(authority.replace(descriptors));
  ASSERT_TRUE(foreign_authority.replace(descriptors));

  const ActivePlaneIdentitySnapshot before_revision =
    authority.snapshot();
  const ActivePlaneIdentitySnapshot foreign_twin =
    foreign_authority.snapshot();
  ASSERT_TRUE(authority.replace(
    {ActivePlaneDescriptor{version, 32U}}));
  const ActivePlaneIdentitySnapshot after_revision =
    authority.snapshot();

  EXPECT_TRUE(
    before_revision.sameRegistryAs(after_revision));
  EXPECT_TRUE(
    after_revision.sameRegistryAs(before_revision));
  EXPECT_FALSE(
    after_revision.sameRegistryAs(foreign_twin));
  EXPECT_FALSE(
    ActivePlaneIdentitySnapshot{}.
    sameRegistryAs(after_revision));
}

TEST(
  MultiViewLandmarkStore,
  OlderStillActivePlaneCanBootstrapAfterNewerPlaneIdsAppear)
{
  MultiViewLandmarkStore store;
  ActivePlaneTestAuthority authority;
  const GeometrySnapshotId geometry{89U};
  const GeometrySupportKey older =
    supportKey(4U, 1U, 0, 0);
  const GeometrySupportKey newer =
    supportKey(900U, 1U, 0, 0);
  ASSERT_TRUE(authority.replace(
    {
      ActivePlaneDescriptor{
        livo::GeometryVersion{older.plane, 7U}, 31U},
      ActivePlaneDescriptor{
        livo::GeometryVersion{newer.plane, 2U}, 32U}
    }));
  const ActivePlaneIdentitySnapshot active =
    authority.snapshot();
  const ReadSnapshot current = store.snapshot(
    geometry, revisions(), active);
  LandmarkQuery query;
  query.batch.value = 8901U;
  query.geometry = geometry;
  query.expected_revisions = revisions();
  query.active_planes = active;
  query.geometry_seeds.push_back(
    activeSeed(active, older, 7U));

  auto prepared = store.prepare(current, query);
  ASSERT_TRUE(prepared.valid());
  const auto proof = BootstrapGate::mint(
    lidarTokenForTest(geometry, revisions()),
    prepared);
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(), proof).status,
    CommitStatus::kCommitted);
  const ReadSnapshot committed =
    readSnapshot(store, geometry);
  ASSERT_EQ(committed.landmarks.size(), 1U);
  EXPECT_EQ(
    committed.landmarks.front().support_key,
    older);
}

TEST(
  MultiViewLandmarkStore,
  RegistryUpdateMakesStagedBootstrapStaleWithZeroWrites)
{
  MultiViewLandmarkStore store;
  ActivePlaneTestAuthority authority;
  const GeometrySnapshotId geometry{90U};
  const GeometrySupportKey key =
    supportKey(90U, 1U, 0, 0);
  const ActivePlaneDescriptor active_descriptor{
    livo::GeometryVersion{key.plane, 1U}, 41U};
  ASSERT_TRUE(authority.replace({active_descriptor}));
  const ActivePlaneIdentitySnapshot active =
    authority.snapshot();
  const ReadSnapshot current = store.snapshot(
    geometry, revisions(), active);
  LandmarkQuery query;
  query.batch.value = 9001U;
  query.geometry = geometry;
  query.expected_revisions = revisions();
  query.active_planes = active;
  query.geometry_seeds.push_back(
    activeSeed(active, key, 1U));
  auto prepared = store.prepare(current, query);
  ASSERT_TRUE(prepared.valid());
  const auto proof = BootstrapGate::mint(
    lidarTokenForTest(geometry, revisions()),
    prepared);

  ASSERT_TRUE(authority.replace({active_descriptor}));
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(), proof).status,
    CommitStatus::kStaleGeometry);
  const ReadSnapshot unchanged =
    readSnapshot(store, geometry);
  EXPECT_EQ(unchanged.store_revision.value, 0U);
  EXPECT_EQ(unchanged.landmarks.size(), 0U);
  EXPECT_EQ(unchanged.next_landmark_id, 1U);
}

TEST(
  MultiViewLandmarkStore,
  MixedBootstrapBatchRollsBackWhenAnyPlaneLeaseBecomesStale)
{
  MultiViewLandmarkStore store;
  ActivePlaneTestAuthority authority;
  const GeometrySnapshotId geometry{91U};
  const GeometrySupportKey retained =
    supportKey(91U, 1U, 0, 0);
  const GeometrySupportKey deleted =
    supportKey(92U, 1U, 0, 0);
  const ActivePlaneDescriptor retained_descriptor{
    livo::GeometryVersion{retained.plane, 1U}, 51U};
  const ActivePlaneDescriptor deleted_descriptor{
    livo::GeometryVersion{deleted.plane, 1U}, 51U};
  ASSERT_TRUE(authority.replace(
    {retained_descriptor, deleted_descriptor}));
  const ActivePlaneIdentitySnapshot active =
    authority.snapshot();
  const ReadSnapshot current = store.snapshot(
    geometry, revisions(), active);
  LandmarkQuery query;
  query.batch.value = 9101U;
  query.geometry = geometry;
  query.expected_revisions = revisions();
  query.active_planes = active;
  query.geometry_seeds.push_back(
    activeSeed(active, retained, 1U));
  query.geometry_seeds.push_back(
    activeSeed(active, deleted, 1U));
  auto prepared = store.prepare(current, query);
  ASSERT_TRUE(prepared.valid());
  const auto proof = BootstrapGate::mint(
    lidarTokenForTest(geometry, revisions()),
    prepared);

  ASSERT_TRUE(authority.replace({retained_descriptor}));
  EXPECT_EQ(
    store.commit(
      prepared.take_delta(), proof).status,
    CommitStatus::kStaleGeometry);
  const ReadSnapshot unchanged =
    readSnapshot(store, geometry);
  EXPECT_EQ(unchanged.store_revision.value, 0U);
  EXPECT_TRUE(unchanged.landmarks.empty());
  EXPECT_EQ(unchanged.next_landmark_id, 1U);
}

TEST(
  ActivePlaneIdentityRegistry,
  CreateDeleteChurnRetainsOnlyTheActivePlaneSet)
{
  ActivePlaneTestAuthority authority;
  for (std::uint64_t index = 1U; index <= 5000U; ++index)
  {
    const ActivePlaneDescriptor active{
      livo::GeometryVersion{
        PlaneIdentity{index, 1U}, 1U},
      index};
    ASSERT_TRUE(authority.replace({active}));
    const auto snapshot = authority.snapshot();
    ASSERT_EQ(snapshot.size(), 1U);
    ASSERT_TRUE(snapshot.leaseFor(
      active.geometry_version).has_value());

    ASSERT_TRUE(authority.replace({}));
    ASSERT_EQ(authority.snapshot().size(), 0U);
  }
  EXPECT_EQ(authority.snapshot().size(), 0U);
}

TEST(
  MultiViewLandmarkStore,
  ConcurrentSnapshotsAndSelectionsRemainValueConsistentDuringCommits)
{
  constexpr std::uint64_t kCommitCount = 500U;
  MultiViewLandmarkStore store;
  ActivePlaneTestAuthority authority;
  std::vector<ActivePlaneDescriptor> descriptors;
  descriptors.reserve(kCommitCount);
  for (std::uint64_t index = 1U;
       index <= kCommitCount; ++index)
  {
    descriptors.push_back(ActivePlaneDescriptor{
      livo::GeometryVersion{
        PlaneIdentity{10000U + index, 1U}, 1U},
      1U});
  }
  ASSERT_TRUE(authority.replace(descriptors));
  const ActivePlaneIdentitySnapshot active =
    authority.snapshot();
  const GeometrySnapshotId geometry{93U};
  const auto lidar_token =
    lidarTokenForTest(geometry, revisions());
  std::atomic<bool> start{false};
  std::atomic<bool> writer_done{false};
  std::atomic<bool> failed{false};

  std::thread writer([&]() {
    while (!start.load(std::memory_order_acquire))
    {
      std::this_thread::yield();
    }
    for (std::uint64_t index = 1U;
         index <= kCommitCount; ++index)
    {
      const ReadSnapshot current = store.snapshot(
        geometry, revisions(), active);
      LandmarkQuery query;
      query.batch.value = 9300U + index;
      query.geometry = geometry;
      query.expected_revisions = revisions();
      query.active_planes = active;
      const GeometrySupportKey key =
        supportKey(10000U + index, 1U, 0, 0);
      query.geometry_seeds.push_back(
        activeSeed(active, key, 1U));
      auto prepared = store.prepare(current, query);
      if (!prepared.valid())
      {
        failed.store(true, std::memory_order_release);
        break;
      }
      const auto proof = BootstrapGate::mint(
        lidar_token,
        prepared);
      if (
        store.commit(
          prepared.take_delta(), proof).status !=
        CommitStatus::kCommitted)
      {
        failed.store(true, std::memory_order_release);
        break;
      }
    }
    writer_done.store(true, std::memory_order_release);
  });

  auto reader = [&]() {
    while (!start.load(std::memory_order_acquire))
    {
      std::this_thread::yield();
    }
    while (!writer_done.load(std::memory_order_acquire))
    {
      const ReadSnapshot snapshot =
        readSnapshot(store, geometry);
      if (
        snapshot.store_revision.value !=
          snapshot.landmarks.size() ||
        snapshot.next_landmark_id !=
          snapshot.landmarks.size() + 1U)
      {
        failed.store(true, std::memory_order_release);
        return;
      }
      if (!snapshot.landmarks.empty())
      {
        LandmarkQuery selection;
        selection.batch.value =
          100000U + snapshot.store_revision.value;
        selection.geometry = geometry;
        selection.expected_revisions = revisions();
        LandmarkReferenceRequest request;
        request.landmark_id =
          snapshot.landmarks.front().landmark_id;
        request.target_camera_id = 0U;
        bindTargetRevisions(request);
        request.target_timestamp_sec = 1.0;
        request.maximum_reference_age_sec = 1.0;
        request.selection_score = 1.0;
        request.target_camera_healthy = true;
        selection.reference_requests.push_back(request);
        const auto selected =
          store.prepare(snapshot, selection);
        if (!selected.valid() &&
            snapshot.store_revision.value ==
              readSnapshot(store, geometry).
              store_revision.value)
        {
          failed.store(true, std::memory_order_release);
          return;
        }
      }
    }
  };

  std::thread first_reader(reader);
  std::thread second_reader(reader);
  start.store(true, std::memory_order_release);
  writer.join();
  first_reader.join();
  second_reader.join();

  EXPECT_FALSE(failed.load(std::memory_order_acquire));
  const ReadSnapshot final = readSnapshot(store, geometry);
  EXPECT_EQ(final.store_revision.value, kCommitCount);
  EXPECT_EQ(final.landmarks.size(), kCommitCount);
  EXPECT_EQ(final.next_landmark_id, kCommitCount + 1U);
}

TEST(
  CheckedIncrement,
  NearMaximumStoreAndIdentityCountersFailClosed)
{
  const auto maximum =
    std::numeric_limits<std::uint64_t>::max();
  std::uint64_t store_revision = maximum;
  std::uint64_t successor = 0U;
  EXPECT_FALSE(livo::checkedSuccessor(
    store_revision, successor));
  EXPECT_EQ(store_revision, maximum);
  EXPECT_EQ(successor, maximum);

  std::uint64_t next_landmark_id = maximum;
  std::uint64_t reserved_landmark_id = 0U;
  EXPECT_FALSE(livo::checkedReserve(
    next_landmark_id, reserved_landmark_id));
  EXPECT_EQ(next_landmark_id, maximum);
  EXPECT_EQ(reserved_landmark_id, 0U);

  std::uint64_t next_observation_id = maximum;
  std::uint64_t reserved_observation_id = 0U;
  EXPECT_FALSE(livo::checkedReserve(
    next_observation_id, reserved_observation_id));
  EXPECT_EQ(next_observation_id, maximum);
  EXPECT_EQ(reserved_observation_id, 0U);
}

}  // namespace
