#include "legacy_landmark_observation_shadow_adapter.h"
#include "active_plane_authority_test_support.h"
#include "runtime_frame_provenance_test_support.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{

using livo::BatchId;
using livo::BootstrapGate;
using livo::CameraTimeTransferStatus;
using livo::CommitRevisionSet;
using livo::CommitStatus;
using livo::GeometrySeed;
using livo::GeometrySnapshotId;
using livo::GeometrySupportKey;
using livo::LandmarkLifecycle;
using livo::LandmarkQuery;
using livo::LegacyLandmarkObservationShadowAdapter;
using livo::LegacyLandmarkObservationShadowConfig;
using livo::LegacyLandmarkObservationShadowRequest;
using livo::LegacyLandmarkObservationShadowSample;
using livo::LegacyLandmarkObservationShadowStatus;
using livo::MultiViewLandmarkStore;
using livo::ObservationPhotometricMode;
using livo::PhotometricStatus;
using livo::PlaneIdentity;
using livo::PlaneLocalCell;
using livo::ReadSnapshot;
using livo::StateRevision;

constexpr std::uint64_t kPlaneId = 17U;
constexpr std::uint64_t kPlaneRevision = 3U;
constexpr std::uint64_t kGeometryId = 41U;

CommitRevisionSet revisions()
{
  CommitRevisionSet value;
  value.state = StateRevision{17U};
  value.rig_calibration_revision = 2U;
  value.camera_model_revisions.fill(5U);
  value.photometric_calibration_revisions.fill(7U);
  value.time_transfer_revision = 9U;
  return value;
}

GeometrySupportKey supportKey(const std::int32_t cell_u = 2)
{
  GeometrySupportKey key;
  key.plane = PlaneIdentity{kPlaneId, 1U};
  key.cell = PlaneLocalCell{cell_u, -4, 0U};
  return key;
}

GeometrySeed geometrySeed(const GeometrySupportKey & key)
{
  GeometrySeed seed;
  seed.support_key = key;
  seed.geometry_version.plane = key.plane;
  seed.geometry_version.revision = kPlaneRevision;
  seed.position_world_m = Eigen::Vector3d(1.0, 2.0, 3.0);
  seed.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  seed.normal_world = Eigen::Vector3d::UnitZ();
  seed.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  return seed;
}

livo::test_support::ActivePlaneAuthority & activeAuthority(
  const MultiViewLandmarkStore & store)
{
  static std::map<
    const MultiViewLandmarkStore *,
    std::unique_ptr<
      livo::test_support::ActivePlaneAuthority>>
    authorities;
  auto & authority = authorities[&store];
  if (!authority)
  {
    authority = std::make_unique<
      livo::test_support::ActivePlaneAuthority>();
  }
  return *authority;
}

livo::ActivePlaneIdentitySnapshot activate(
  const MultiViewLandmarkStore & store,
  const std::vector<livo::GeometryVersion> & versions)
{
  std::vector<livo::ActivePlaneDescriptor> descriptors;
  for (const auto & version : versions)
  {
    descriptors.push_back(
      livo::ActivePlaneDescriptor{
        version,
        std::max<std::uint64_t>(1U, version.revision)});
  }
  auto & authority = activeAuthority(store);
  EXPECT_TRUE(authority.replace(descriptors));
  return authority.snapshot();
}

livo::MapCommitProof bootstrapProof(
  const BatchId batch,
  const ReadSnapshot & snapshot,
  const livo::PreparedLandmarkBatch & prepared)
{
  static_cast<void>(batch);
  return BootstrapGate::mint(
    livo::test_support::makeLidarAnchoredToken(
      snapshot.geometry, revisions()),
    prepared);
}

void seedExistingLidarLandmark(
  MultiViewLandmarkStore & store,
  const GeometrySupportKey & key,
  const std::uint64_t batch_value = 1U)
{
  const GeometrySnapshotId geometry{kGeometryId};
  const auto active = activate(
    store,
    {livo::GeometryVersion{
      key.plane, kPlaneRevision}});
  const ReadSnapshot snapshot =
    store.snapshot(geometry, revisions(), active);
  LandmarkQuery query;
  query.batch = BatchId{batch_value};
  query.geometry = geometry;
  query.expected_revisions = revisions();
  query.active_planes = active;
  GeometrySeed lidar_seed = geometrySeed(key);
  lidar_seed.active_plane_lease =
    *active.leaseFor(lidar_seed.geometry_version);
  query.geometry_seeds.push_back(lidar_seed);

  auto prepared = store.prepare(snapshot, query);
  ASSERT_TRUE(prepared.valid());
  ASSERT_EQ(
    store.commit(
      prepared.take_delta(),
      bootstrapProof(query.batch, snapshot, prepared)).status,
    CommitStatus::kCommitted);
}

LegacyLandmarkObservationShadowSample sample(
  const GeometrySupportKey & key,
  const std::uint8_t camera_id,
  const bool is_bootstrap)
{
  LegacyLandmarkObservationShadowSample value;
  value.support_key = key;
  value.camera_id = camera_id;
  value.exposure_timestamp_sec = 10.0;
  value.rotation_world_from_imu = Eigen::Matrix3d::Identity();
  value.position_world_m = Eigen::Vector3d(0.5, -0.5, 1.0);
  value.rig_calibration_revision = 2U;
  value.camera_model_revision = 5U;
  value.pixel = Eigen::Vector2d(320.0, 240.0);
  value.level = 0;
  value.owned_patch_pyramid = {
    {11.0F, 12.0F, 13.0F, 14.0F},
    {21.0F, 22.0F}};
  value.photometric_mode =
    ObservationPhotometricMode::kSameCameraOnly;
  value.photometric_status = PhotometricStatus::kValid;
  value.photometric_calibration_revision = 7U;
  value.photometric_cross_camera_comparable = false;
  value.photometric_experimental = false;
  value.time_status = CameraTimeTransferStatus::kValid;
  value.time_transfer_revision = 9U;
  value.pose_covariance =
    1.0e-5 * Eigen::Matrix<double, 6, 6>::Identity();
  value.visual_update_accepted = !is_bootstrap;
  value.is_bootstrap = is_bootstrap;
  return value;
}

LegacyLandmarkObservationShadowRequest request(
  const std::uint64_t batch_value,
  LegacyLandmarkObservationShadowSample observation)
{
  LegacyLandmarkObservationShadowRequest value;
  value.batch = BatchId{batch_value};
  value.geometry = GeometrySnapshotId{kGeometryId};
  if (observation.is_bootstrap)
  {
    value.bootstrap_token =
      livo::test_support::makeLidarAnchoredToken(
        value.geometry, revisions());
  }
  value.samples.push_back(std::move(observation));
  return value;
}

void bindBootstrapAuthority(
  const MultiViewLandmarkStore & store,
  LegacyLandmarkObservationShadowRequest & request)
{
  std::vector<livo::GeometryVersion> versions;
  for (const auto & observation : request.samples)
  {
    if (observation.is_bootstrap)
    {
      versions.push_back(livo::GeometryVersion{
        observation.support_key.plane,
        kPlaneRevision});
    }
  }
  if (versions.empty())
  {
    return;
  }
  request.active_planes = activate(store, versions);
  for (auto & observation : request.samples)
  {
    if (!observation.is_bootstrap)
    {
      continue;
    }
    observation.active_plane_lease =
      *request.active_planes.leaseFor(
      livo::GeometryVersion{
        observation.support_key.plane,
        kPlaneRevision});
  }
}

LegacyLandmarkObservationShadowAdapter enabledAdapter()
{
  LegacyLandmarkObservationShadowConfig config;
  config.enabled = true;
  return LegacyLandmarkObservationShadowAdapter(config);
}

TEST(
  LegacyLandmarkObservationShadowAdapter,
  BootstrapAppendsReferenceToExistingLidarSeedAndKeepsLifecycle)
{
  MultiViewLandmarkStore store;
  livo::test_support::RuntimeFrameProvenanceTestSupport
    runtime;
  ASSERT_TRUE(runtime.valid());
  const GeometrySupportKey key = supportKey();
  seedExistingLidarLandmark(store, key);

  auto bootstrap = request(2U, sample(key, 0U, true));
  bootstrap.samples.front().frame_provenance =
    runtime.frame(0U);
  bindBootstrapAuthority(store, bootstrap);
  auto staged = enabledAdapter().stage(store, bootstrap);

  ASSERT_EQ(
    staged.status,
    LegacyLandmarkObservationShadowStatus::kStaged);
  EXPECT_EQ(staged.staged_observation_count, 1U);
  ASSERT_EQ(
    staged.staged.commit(store).status,
    CommitStatus::kCommitted);
  const ReadSnapshot snapshot =
    store.snapshot(bootstrap.geometry, revisions());
  ASSERT_EQ(snapshot.landmarks.size(), 1U);
  EXPECT_EQ(
    snapshot.landmarks.front().lifecycle,
    LandmarkLifecycle::kLidarSeeded);
  ASSERT_EQ(
    snapshot.landmarks.front().observations.size(), 1U);
  const auto & committed_observation =
    snapshot.landmarks.front().observations.front();
  EXPECT_EQ(
    committed_observation.accepted_camera_batch_sequence,
    runtime.runtime().cameraBatchReceipt().batchSequence());
  EXPECT_EQ(
    committed_observation.sampling_geometry_version.revision,
    kPlaneRevision);
  EXPECT_EQ(
    committed_observation.observation_revision,
    snapshot.store_revision.value);
  EXPECT_EQ(
    committed_observation.pose_runtime_publication_sequence,
    runtime.runtime().publicationSequence());
  EXPECT_EQ(
    committed_observation.pose_active_state_snapshot_revision,
    runtime.runtime().activeStateSnapshotRevision());
  ASSERT_TRUE(
    committed_observation.store_receipt.has_value());
  EXPECT_TRUE(
    committed_observation.store_receipt->valid());
  EXPECT_TRUE(
    committed_observation.store_receipt->issuedBy(
      snapshot.store_identity));
  EXPECT_TRUE(
    committed_observation.store_receipt->
    matchesObservationPayload(
      committed_observation));
  EXPECT_TRUE(
    committed_observation.store_receipt->accepts(
      snapshot.landmarks.front().landmark_id,
      committed_observation));
  EXPECT_FALSE(
    committed_observation.store_receipt->accepts(
      snapshot.landmarks.front().landmark_id + 1U,
      committed_observation));
  EXPECT_EQ(
    committed_observation.store_receipt->
    storeRevision().value,
    snapshot.store_revision.value);
  EXPECT_EQ(
    committed_observation.store_receipt->landmarkId(),
    snapshot.landmarks.front().landmark_id);
  EXPECT_EQ(
    committed_observation.store_receipt->observationId(),
    committed_observation.observation_id);
  EXPECT_EQ(
    committed_observation.store_receipt->
    observationRevision(),
    committed_observation.observation_revision);
  EXPECT_EQ(
    committed_observation.store_receipt->
    acceptedCameraBatchSequence(),
    committed_observation.accepted_camera_batch_sequence);
  EXPECT_EQ(
    committed_observation.store_receipt->
    samplingGeometryVersion().revision,
    committed_observation.
    sampling_geometry_version.revision);

  livo::ObservationSnapshot tampered =
    committed_observation;
  tampered.owned_patch_pyramid.front().front() += 1.0F;
  EXPECT_FALSE(
    committed_observation.store_receipt->
    matchesObservationPayload(tampered));
  tampered = committed_observation;
  ++tampered.sampling_geometry_version.revision;
  EXPECT_FALSE(
    committed_observation.store_receipt->
    matchesObservationPayload(tampered));

  MultiViewLandmarkStore foreign_store;
  const ReadSnapshot foreign_snapshot =
    foreign_store.snapshot(bootstrap.geometry, revisions());
  EXPECT_FALSE(
    snapshot.store_identity.sameAuthority(
      foreign_snapshot.store_identity));
  EXPECT_FALSE(
    committed_observation.store_receipt->issuedBy(
      foreign_snapshot.store_identity));
  EXPECT_TRUE(
    committed_observation.reference_only);
  EXPECT_EQ(
    snapshot.landmarks.front().observations.front().
    photometric_mode,
    ObservationPhotometricMode::kSameCameraOnly);
  EXPECT_FALSE(
    snapshot.landmarks.front().observations.front().
    photometric_cross_camera_comparable);
  EXPECT_EQ(
    snapshot.landmarks.front().observations.front().
    photometric_calibration_revision,
    7U);
  EXPECT_EQ(
    snapshot.landmarks.front().observations.front().
    time_transfer_revision,
    9U);
  EXPECT_GT(
    snapshot.landmarks.front().observations.front().
    pose_covariance.trace(),
    0.0);
}

TEST(
  LegacyLandmarkObservationShadowAdapter,
  MissingRuntimeFrameProvenanceRejectsObservationFailClosed)
{
  MultiViewLandmarkStore store;
  const GeometrySupportKey key = supportKey();
  seedExistingLidarLandmark(store, key);

  auto bootstrap = request(2U, sample(key, 0U, true));
  bindBootstrapAuthority(store, bootstrap);
  const auto staged = enabledAdapter().stage(store, bootstrap);

  EXPECT_EQ(
    staged.status,
    LegacyLandmarkObservationShadowStatus::kNoEligibleObservation);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.staged.valid());
  const ReadSnapshot snapshot =
    store.snapshot(bootstrap.geometry, revisions());
  ASSERT_EQ(snapshot.landmarks.size(), 1U);
  EXPECT_TRUE(snapshot.landmarks.front().observations.empty());
}

TEST(
  LegacyLandmarkObservationShadowAdapter,
  MixedBootstrapWithOneMissingActiveLeaseRejectsWholeBatch)
{
  MultiViewLandmarkStore store;
  livo::test_support::RuntimeFrameProvenanceTestSupport
    runtime;
  ASSERT_TRUE(runtime.valid());
  const GeometrySupportKey first_key = supportKey(2);
  GeometrySupportKey second_key = supportKey(3);
  second_key.plane = PlaneIdentity{kPlaneId + 1U, 1U};
  seedExistingLidarLandmark(store, first_key, 1U);
  seedExistingLidarLandmark(store, second_key, 2U);

  auto mixed = request(3U, sample(first_key, 0U, true));
  mixed.samples.push_back(sample(second_key, 0U, true));
  for (auto & observation : mixed.samples)
  {
    observation.frame_provenance = runtime.frame(0U);
  }
  const livo::GeometryVersion first_version{
    first_key.plane, kPlaneRevision};
  mixed.active_planes = activate(store, {first_version});
  mixed.samples.front().active_plane_lease =
    *mixed.active_planes.leaseFor(first_version);

  const auto staged = enabledAdapter().stage(store, mixed);

  EXPECT_EQ(
    staged.status,
    LegacyLandmarkObservationShadowStatus::kPrepareRejected);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.staged.valid());
  const ReadSnapshot snapshot =
    store.snapshot(mixed.geometry, revisions());
  EXPECT_EQ(snapshot.store_revision.value, 2U);
  ASSERT_EQ(snapshot.landmarks.size(), 2U);
  for (const auto & landmark : snapshot.landmarks)
  {
    EXPECT_TRUE(landmark.observations.empty());
  }
}

TEST(
  LegacyLandmarkObservationShadowAdapter,
  ActiveRegistryChangeAfterStageRollsBackBootstrapAtomically)
{
  MultiViewLandmarkStore store;
  livo::test_support::RuntimeFrameProvenanceTestSupport
    runtime;
  ASSERT_TRUE(runtime.valid());
  const GeometrySupportKey key = supportKey();
  seedExistingLidarLandmark(store, key);
  auto bootstrap = request(2U, sample(key, 0U, true));
  bootstrap.samples.front().frame_provenance =
    runtime.frame(0U);
  bindBootstrapAuthority(store, bootstrap);
  auto staged = enabledAdapter().stage(store, bootstrap);
  ASSERT_EQ(
    staged.status,
    LegacyLandmarkObservationShadowStatus::kStaged);
  ASSERT_TRUE(staged.staged.valid());

  const livo::GeometryVersion version{
    key.plane, kPlaneRevision};
  const livo::ActivePlaneIdentitySnapshot replacement =
    activate(store, {version});
  ASSERT_TRUE(replacement.valid());
  const auto committed = staged.staged.commit(store);

  EXPECT_EQ(committed.status, CommitStatus::kStaleGeometry);
  const ReadSnapshot snapshot =
    store.snapshot(bootstrap.geometry, revisions());
  EXPECT_EQ(snapshot.store_revision.value, 1U);
  ASSERT_EQ(snapshot.landmarks.size(), 1U);
  EXPECT_TRUE(snapshot.landmarks.front().observations.empty());
}

}  // namespace
