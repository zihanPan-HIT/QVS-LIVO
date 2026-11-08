#include "camera_batch_time_transfer.h"
#include "runtime_frame_provenance_test_support.h"
#include "shared_landmark_lifecycle_routing.h"
#include "shared_landmark_shadow_session.h"
#include "voxel_map.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <unordered_map>

#include <gtest/gtest.h>

namespace
{

class ActivePlaneTestMap
{
public:
  explicit ActivePlaneTestMap(
    const bool plane_lifecycle_shadow_enable = true)
  : config_(enabledConfig(plane_lifecycle_shadow_enable)),
    manager_(config_, initial_map_)
  {
  }

  ~ActivePlaneTestMap()
  {
    for (auto & entry : manager_.voxel_map_)
    {
      delete entry.second;
    }
    manager_.voxel_map_.clear();
  }

  bool publish(
    const std::vector<livo::ActivePlaneDescriptor> & descriptors)
  {
    if (!manager_.voxel_map_.empty())
    {
      manager_.clearMemOutOfMap(-1, 1, -1, 1, -1, 1);
    }
    std::int64_t voxel_x = 0;
    for (const livo::ActivePlaneDescriptor & descriptor :
         descriptors)
    {
      auto * octree = new VoxelOctoTree(
        0, 0, 1, 8, 0.01F);
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

    const livo::ActivePlaneIdentitySnapshot current =
      snapshot();
    if (!current.valid() ||
        current.size() != descriptors.size())
    {
      return false;
    }
    return std::all_of(
      descriptors.begin(), descriptors.end(),
      [&current](
        const livo::ActivePlaneDescriptor & descriptor) {
        const auto lease =
          current.leaseFor(descriptor.geometry_version);
        return
          lease.has_value() &&
          lease->generation() == descriptor.generation;
      });
  }

  livo::ActivePlaneIdentitySnapshot snapshot() const
  {
    return manager_.snapshotActivePlaneIdentities();
  }

private:
  static VoxelMapConfig enabledConfig(
    const bool plane_lifecycle_shadow_enable)
  {
    VoxelMapConfig config{};
    config.plane_lifecycle_shadow_enable_ =
      plane_lifecycle_shadow_enable;
    return config;
  }

  VoxelMapConfig config_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *>
    initial_map_;
  VoxelMapManager manager_;
};

std::unique_ptr<ActivePlaneTestMap>
activeRegistry(
  const std::vector<livo::GeometryVersion> & versions,
  const std::uint64_t generation = 77U)
{
  auto registry =
    std::make_unique<ActivePlaneTestMap>();
  std::vector<livo::ActivePlaneDescriptor> descriptors;
  descriptors.reserve(versions.size());
  for (const livo::GeometryVersion & version : versions)
  {
    descriptors.push_back(
      livo::ActivePlaneDescriptor{version, generation});
  }
  if (!registry->publish(descriptors))
  {
    return nullptr;
  }
  return registry;
}

std::unique_ptr<ActivePlaneTestMap>
activateGeometryRequest(
  livo::SharedLandmarkGeometryBootstrapRequest & request,
  const std::uint64_t generation = 77U)
{
  std::vector<livo::GeometryVersion> versions;
  for (const livo::LegacyPlaneLandmarkSample & sample :
       request.samples)
  {
    const livo::GeometryVersion version{
      sample.plane, sample.geometry_revision};
    const bool already_present = std::any_of(
      versions.begin(), versions.end(),
      [&version](const livo::GeometryVersion & candidate) {
        return
          candidate.plane == version.plane &&
          candidate.revision == version.revision;
      });
    if (!already_present)
    {
      versions.push_back(version);
    }
  }
  auto registry = activeRegistry(versions, generation);
  if (registry)
  {
    request.active_planes = registry->snapshot();
  }
  return registry;
}

livo::LegacyPlaneLandmarkSample geometrySample()
{
  livo::LegacyPlaneLandmarkSample value;
  value.plane = livo::PlaneIdentity{7U, 1U};
  value.geometry_revision = 1U;
  value.identity_origin_world_m = Eigen::Vector3d::Zero();
  value.identity_u_axis_world = Eigen::Vector3d::UnitX();
  value.identity_v_axis_world = Eigen::Vector3d::UnitY();
  value.position_world_m = Eigen::Vector3d(0.1, 0.2, 0.0);
  value.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  value.normal_world = Eigen::Vector3d::UnitZ();
  value.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  return value;
}

livo::CameraBatchTimeTransferBundle makeTimeTransferBundle(
  const double batch_timestamp_sec,
  const std::array<bool, kVioCameraCount> & present)
{
  CameraObservationBatch batch;
  batch.present = present;
  batch.camera_count = static_cast<std::size_t>(
    std::count(present.begin(), present.end(), true));
  batch.partial = batch.camera_count < kVioCameraCount;
  batch.timestamp = batch_timestamp_sec;
  for (std::size_t camera_id = 0U;
       camera_id < present.size(); ++camera_id)
  {
    if (!present[camera_id])
    {
      continue;
    }
    batch.observations[camera_id].camera_id =
      static_cast<std::uint8_t>(camera_id);
    batch.observations[camera_id].timestamp =
      batch_timestamp_sec;
    batch.observations[camera_id].image =
      cv::Mat::ones(2, 2, CV_8UC1);
  }
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch_timestamp_sec;
  livo::CameraTimeTransfer transfer;
  return livo::buildCameraBatchTimeTransferBundle(
    batch, anchor, 0.0, transfer);
}

const livo::test_support::
RuntimeFrameProvenanceTestSupport &
validObservationRuntime()
{
  static const livo::test_support::
    RuntimeFrameProvenanceTestSupport runtime;
  return runtime;
}

const livo::CameraBatchTimeTransferBundle &
validObservationTimeBundle()
{
  return
    validObservationRuntime().runtime().
    timeTransferBundle();
}

bool bindObservationRuntime(
  livo::SharedLandmarkShadowSession & session)
{
  const auto & runtime = validObservationRuntime();
  return
    runtime.valid() &&
    session.bindRuntimeObservationAuthority(
      runtime.batchAuthority(),
      runtime.runtimeAuthority());
}

livo::SharedLandmarkShadowObservationSample observationSample(
  const bool is_bootstrap = true)
{
  livo::SharedLandmarkShadowObservationSample value;
  value.support_key.plane = livo::PlaneIdentity{7U, 1U};
  value.support_key.cell = livo::PlaneLocalCell{0, 0, 0U};
  value.camera_id = 0U;
  value.time_transfer =
    validObservationTimeBundle().transfers[0U];
  value.frame_provenance =
    validObservationRuntime().frame(0U);
  value.exposure_timestamp_sec =
    value.time_transfer->targetTimestampSec();
  value.pixel = Eigen::Vector2d(320.0, 240.0);
  value.level = 0;
  value.owned_patch_pyramid = {
    {11.0F, 12.0F, 13.0F, 14.0F}};
  value.frame_photometric_status =
    livo::PhotometricStatus::kValid;
  value.anchor_pose_covariance =
    1.0e-5 * Eigen::Matrix<double, 6, 6>::Identity();
  value.visual_update_accepted = !is_bootstrap;
  value.is_bootstrap = is_bootstrap;
  return value;
}

livo::SharedLandmarkGeometryRefitSample refitSample()
{
  livo::SharedLandmarkGeometryRefitSample value;
  value.support_key.plane = livo::PlaneIdentity{7U, 1U};
  value.support_key.cell = livo::PlaneLocalCell{0, 0, 0U};
  value.position_world_m = Eigen::Vector3d(0.2, 0.3, 0.01);
  value.position_covariance_m2 =
    0.005 * Eigen::Matrix3d::Identity();
  value.normal_world =
    Eigen::Vector3d(0.0, 0.01, 1.0).normalized();
  value.plane_covariance =
    0.0005 * Eigen::Matrix<double, 6, 6>::Identity();
  value.local_basis.origin_world_m =
    Eigen::Vector3d::Zero();
  value.local_basis.u_axis_world = Eigen::Vector3d::UnitX();
  value.local_basis.v_axis_world = Eigen::Vector3d::UnitY();
  return value;
}

livo::PhotometricCalibration rawCalibration(
  const std::uint8_t camera_id,
  const std::uint64_t revision)
{
  livo::PhotometricCameraCalibration camera;
  camera.camera_id = camera_id;
  camera.calibration_revision = revision;
  camera.image_width = 4;
  camera.image_height = 4;
  camera.black_level = 0.0;
  camera.white_level = 255.0;
  camera.inverse_response_lut.resize(256U);
  for (std::size_t index = 0U;
       index < camera.inverse_response_lut.size(); ++index)
  {
    camera.inverse_response_lut[index] =
      static_cast<double>(index) / 255.0;
  }
  camera.vignetting_transmission.assign(16U, 1.0);
  camera.source_id = "shared-shadow-session-test";
  std::vector<livo::PhotometricCameraCalibration> cameras;
  cameras.push_back(std::move(camera));
  return livo::PhotometricCalibration(std::move(cameras));
}

livo::ShadowPhotometricCapabilityEvidence strictEvidence(
  const livo::PhotometricCalibration & calibration)
{
  livo::ShadowPhotometricCapabilityEvidence evidence;
  evidence.mode = livo::PhotometricMode::kCalibratedDirect;
  evidence.strict_calibration = &calibration;
  return evidence;
}

TEST(
  SharedLandmarkShadowSession,
  DefaultOffDoesNotActivateConsumeBatchWriteStoreOrTouchSolver)
{
  livo::SharedLandmarkShadowSession session;
  EXPECT_FALSE(session.active());
  const livo::SharedLandmarkShadowAudit before =
    session.audit();
  EXPECT_EQ(before.consumed_batches, 0U);
  EXPECT_EQ(before.store_revision, 0U);
  EXPECT_EQ(before.landmark_count, 0U);
  EXPECT_EQ(before.observation_count, 0U);
  EXPECT_FALSE(before.solver_applied);

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry.value = 77U;
  const livo::SharedLandmarkShadowOperationResult result =
    session.bootstrapGeometry(request);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::kDisabled);
  EXPECT_FALSE(result.consumed);
  EXPECT_FALSE(result.solver_applied);
  EXPECT_EQ(result.batch.value, 0U);
  const livo::SharedLandmarkShadowAudit after =
    session.audit();
  EXPECT_EQ(after.consumed_batches, before.consumed_batches);
  EXPECT_EQ(after.store_revision, before.store_revision);
  EXPECT_EQ(after.landmark_count, before.landmark_count);
  EXPECT_EQ(after.observation_count, before.observation_count);
  EXPECT_FALSE(after.solver_applied);

  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.samples.push_back(observationSample());
  const auto observation_result =
    session.stageObservations(observation);
  EXPECT_EQ(
    observation_result.status,
    livo::SharedLandmarkShadowOperationStatus::kDisabled);
  EXPECT_FALSE(observation_result.consumed);
  EXPECT_EQ(observation_result.batch.value, 0U);
  EXPECT_FALSE(observation_result.transaction.valid());
  EXPECT_EQ(session.audit().consumed_batches, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  ValidChangedTimeConfigurationAdvancesAuditableRevision)
{
  livo::SharedLandmarkShadowSessionConfig session_config;
  session_config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(session_config);
  ASSERT_TRUE(session.active());

  livo::CameraTimeTransferConfig first;
  const auto initial = session.configureTimeTransfer(first);
  EXPECT_EQ(
    initial.status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);
  EXPECT_TRUE(initial.changed);
  EXPECT_EQ(initial.revision, 1U);

  const auto unchanged = session.configureTimeTransfer(first);
  EXPECT_EQ(
    unchanged.status,
    livo::TimeTransferConfigurationUpdateStatus::kUnchanged);
  EXPECT_FALSE(unchanged.changed);
  EXPECT_EQ(unchanged.revision, 1U);

  livo::CameraTimeTransferConfig second = first;
  second.max_imu_gap_sec = 0.01;
  const auto changed = session.configureTimeTransfer(second);
  EXPECT_EQ(
    changed.status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);
  EXPECT_TRUE(changed.changed);
  EXPECT_EQ(changed.revision, 2U);

  livo::CameraTimeTransferConfig invalid = second;
  invalid.max_imu_gap_sec = 0.0;
  const auto rejected =
    session.configureTimeTransfer(invalid);
  EXPECT_EQ(
    rejected.status,
    livo::TimeTransferConfigurationUpdateStatus::
    kInvalidConfiguration);
  EXPECT_FALSE(rejected.changed);
  EXPECT_EQ(rejected.revision, 2U);
  EXPECT_EQ(
    session.audit().time_transfer_configuration_revision,
    2U);
}

TEST(
  SharedLandmarkShadowSession,
  RuntimeObservationAuthorityMustBeBoundExactlyOnce)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  const auto & owner = validObservationRuntime();
  livo::test_support::RuntimeFrameProvenanceTestSupport
    foreign(10.0, 91U);
  ASSERT_TRUE(owner.valid());
  ASSERT_TRUE(foreign.valid());

  EXPECT_TRUE(session.bindRuntimeObservationAuthority(
      owner.batchAuthority(), owner.runtimeAuthority()));
  EXPECT_FALSE(session.bindRuntimeObservationAuthority(
      owner.batchAuthority(), owner.runtimeAuthority()));
  EXPECT_FALSE(session.bindRuntimeObservationAuthority(
      foreign.batchAuthority(), foreign.runtimeAuthority()));

  livo::SharedLandmarkShadowSession mismatched(config);
  EXPECT_FALSE(mismatched.bindRuntimeObservationAuthority(
      owner.batchAuthority(), foreign.runtimeAuthority()));
  EXPECT_TRUE(mismatched.bindRuntimeObservationAuthority(
      owner.batchAuthority(), owner.runtimeAuthority()));
}

TEST(
  SharedLandmarkShadowSession,
  MissingRuntimeObservationAuthorityRejectsBeforeStaging)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  livo::SharedLandmarkShadowObservationRequest request;
  request.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  request.samples.push_back(observationSample());

  const auto staged = session.stageObservations(request);

  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInvalidRuntimeObservationProvenance);
  EXPECT_TRUE(staged.consumed);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_EQ(session.audit().store_revision, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  ForeignRuntimePublisherFrameProvenanceRejectsFailClosed)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  ASSERT_EQ(
    session.configureTimeTransfer(
      livo::CameraTimeTransferConfig{}).status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::test_support::RuntimeFrameProvenanceTestSupport
    foreign_runtime;
  ASSERT_TRUE(foreign_runtime.valid());
  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.active_planes = active_planes->snapshot();
  observation.samples.push_back(observationSample());
  observation.samples.front().frame_provenance =
    foreign_runtime.frame(0U);

  const auto staged = session.stageObservations(observation);

  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInvalidRuntimeObservationProvenance);
  EXPECT_TRUE(staged.consumed);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.transaction.valid());
  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  RawStrictProviderIsUnavailableAndCannotAliasExperimental)
{
  livo::SharedLandmarkShadowSessionConfig session_config;
  session_config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(session_config);

  const livo::PhotometricCalibration raw_calibration =
    rawCalibration(0U, 17U);
  const auto unavailable =
    session.configurePhotometric(
    0U, strictEvidence(raw_calibration));
  EXPECT_EQ(
    unavailable.status,
    livo::ShadowPhotometricCapabilityStatus::
    kStrictProviderUnavailable);
  EXPECT_EQ(unavailable.strict_calibration_revision, 0U);
  EXPECT_EQ(unavailable.experimental_shadow_revision, 0U);
  EXPECT_FALSE(unavailable.changed);

  livo::ShadowPhotometricCapabilityEvidence robust;
  robust.mode =
    livo::PhotometricMode::kRobustAffineExperimental;
  const auto experimental =
    session.configurePhotometric(0U, robust);
  EXPECT_EQ(
    experimental.status,
    livo::ShadowPhotometricCapabilityStatus::
    kRobustAffineExperimental);
  EXPECT_EQ(experimental.strict_calibration_revision, 0U);
  EXPECT_EQ(experimental.experimental_shadow_revision, 1U);
  EXPECT_EQ(
    session.audit().photometric_capabilities[0U].status,
    livo::ShadowPhotometricCapabilityStatus::
    kRobustAffineExperimental);
}

TEST(
  SharedLandmarkShadowSession,
  VisualRevisionRegistryIsMonotonicAndCameraScoped)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  const auto state_one = session.advanceVisualStateRevision();
  const auto state_two = session.advanceVisualStateRevision();
  EXPECT_EQ(state_one.value, 1U);
  EXPECT_EQ(state_two.value, 2U);

  const auto camera_zero =
    session.recordCameraConfiguration(0U);
  EXPECT_TRUE(camera_zero.changed);
  EXPECT_EQ(camera_zero.rig_revision, 1U);
  EXPECT_EQ(camera_zero.camera_model_revision, 1U);
  const auto camera_one =
    session.recordCameraConfiguration(1U);
  EXPECT_TRUE(camera_one.changed);
  EXPECT_EQ(camera_one.rig_revision, 2U);
  EXPECT_EQ(camera_one.camera_model_revision, 1U);

  const auto invalid =
    session.recordCameraConfiguration(4U);
  EXPECT_FALSE(invalid.changed);
  EXPECT_EQ(invalid.rig_revision, 2U);
  EXPECT_EQ(invalid.camera_model_revision, 0U);

  const auto revisions =
    session.audit().visual_revision_snapshot;
  EXPECT_EQ(revisions.state.value, 2U);
  EXPECT_EQ(revisions.rig_calibration_revision, 2U);
  EXPECT_EQ(revisions.camera_model_revisions[0U], 1U);
  EXPECT_EQ(revisions.camera_model_revisions[1U], 1U);
  EXPECT_EQ(revisions.camera_model_revisions[2U], 0U);
  EXPECT_EQ(revisions.camera_model_revisions[3U], 0U);
}

TEST(
  SharedLandmarkShadowSession,
  LioPosteriorRevisionIsTypedMonotonicAndVisualIndependent)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.refit.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  const auto first = session.advanceLioPosteriorRevision();
  const auto second = session.advanceLioPosteriorRevision();

  EXPECT_EQ(first.value, 1U);
  EXPECT_EQ(second.value, 2U);
  const auto audit = session.audit();
  EXPECT_EQ(audit.lio_posterior_revision.value, 2U);
  EXPECT_EQ(audit.visual_revision_snapshot.state.value, 0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.rig_calibration_revision,
    0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.time_transfer_revision,
    0U);
}

TEST(
  SharedLandmarkShadowSession,
  ReusedLidarGenerationDoesNotAdvancePosteriorRevision)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.refit.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  EXPECT_EQ(
    session.observeLioPosteriorGeneration(41U).value, 1U);
  EXPECT_EQ(
    session.observeLioPosteriorGeneration(41U).value, 1U);
  EXPECT_EQ(
    session.observeLioPosteriorGeneration(0U).value, 1U);
  EXPECT_EQ(
    session.observeLioPosteriorGeneration(42U).value, 2U);
}

TEST(
  SharedLandmarkShadowSession,
  ValidActiveSnapshotCommitsGeometryWithoutTimeOrPhotometricCapability)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry = livo::GeometrySnapshotId{77U};
  const livo::LegacyPlaneLandmarkSample sample = geometrySample();
  request.samples.push_back(sample);
  const auto active_planes = activeRegistry(
    {livo::GeometryVersion{
      sample.plane, sample.geometry_revision}});
  ASSERT_NE(active_planes, nullptr);
  request.active_planes = active_planes->snapshot();
  const auto result = session.bootstrapGeometry(request);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  EXPECT_TRUE(result.consumed);
  EXPECT_EQ(result.batch.value, 1U);
  EXPECT_EQ(result.geometry_writes, 1U);
  EXPECT_EQ(result.observation_writes, 0U);
  EXPECT_FALSE(result.solver_applied);

  const auto audit = session.audit();
  EXPECT_EQ(audit.consumed_batches, 1U);
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(audit.time_transfer_configuration_revision, 0U);
  EXPECT_EQ(
    audit.photometric_capabilities[0U].status,
    livo::ShadowPhotometricCapabilityStatus::
    kStrictProviderUnavailable);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  GeometryOnlyLivMapperRoutingPublishesVoxelLeaseIntoSession)
{
  const livo::SharedLandmarkLifecycleRouting routing{
    true, false, false};
  ASSERT_TRUE(routing.activePlaneLifecycleRequired());
  ASSERT_TRUE(routing.activePlaneSnapshotRequired());
  const livo::SharedLandmarkLifecycleRouting all_off;
  EXPECT_FALSE(all_off.activePlaneLifecycleRequired());
  EXPECT_FALSE(all_off.activePlaneSnapshotRequired());

  const livo::LegacyPlaneLandmarkSample sample =
    geometrySample();
  const livo::GeometryVersion version{
    sample.plane, sample.geometry_revision};
  ActivePlaneTestMap voxel_map(
    routing.activePlaneLifecycleRequired());
  ASSERT_TRUE(voxel_map.publish(
    {livo::ActivePlaneDescriptor{version, 77U}}));

  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  ASSERT_FALSE(config.observation.enabled);
  ASSERT_FALSE(config.tombstone.enabled);
  ASSERT_FALSE(config.refit.enabled);
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry = livo::GeometrySnapshotId{77U};
  request.active_planes = voxel_map.snapshot();
  request.samples.push_back(sample);
  const auto result = session.bootstrapGeometry(request);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  EXPECT_EQ(result.geometry_writes, 1U);
  const auto audit = session.audit();
  EXPECT_TRUE(audit.geometry_enabled);
  EXPECT_FALSE(audit.observation_enabled);
  EXPECT_FALSE(audit.tombstone_enabled);
  EXPECT_FALSE(audit.refit_enabled);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.store_revision, 1U);
}

TEST(
  SharedLandmarkShadowSession,
  MissingActiveSnapshotRejectsWholeGeometryBatchWithoutWrite)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry = livo::GeometrySnapshotId{77U};
  request.samples.push_back(geometrySample());
  const auto result = session.bootstrapGeometry(request);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kMissingActivePlaneSnapshot);
  EXPECT_TRUE(result.consumed);
  EXPECT_EQ(result.geometry_writes, 0U);
  EXPECT_EQ(session.audit().store_revision, 0U);
  EXPECT_EQ(session.audit().landmark_count, 0U);
  EXPECT_FALSE(result.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  StaleActiveSnapshotRejectsWholeGeometryBatchWithoutWrite)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  const livo::LegacyPlaneLandmarkSample sample = geometrySample();
  const livo::GeometryVersion version{
    sample.plane, sample.geometry_revision};
  const auto active_planes = activeRegistry({version}, 77U);
  ASSERT_NE(active_planes, nullptr);
  const livo::ActivePlaneIdentitySnapshot stale =
    active_planes->snapshot();
  ASSERT_TRUE(active_planes->publish(
    {livo::ActivePlaneDescriptor{version, 78U}}));
  ASSERT_FALSE(stale.valid());

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry = livo::GeometrySnapshotId{77U};
  request.active_planes = stale;
  request.samples.push_back(sample);
  const auto result = session.bootstrapGeometry(request);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kStaleActivePlaneSnapshot);
  EXPECT_EQ(result.geometry_writes, 0U);
  EXPECT_EQ(session.audit().store_revision, 0U);
  EXPECT_EQ(session.audit().landmark_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  MixedGeometryBatchWithOneInactiveVersionRejectsAllWrites)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  const livo::LegacyPlaneLandmarkSample active_sample =
    geometrySample();
  livo::LegacyPlaneLandmarkSample inactive_sample =
    geometrySample();
  inactive_sample.plane = livo::PlaneIdentity{8U, 1U};
  inactive_sample.position_world_m =
    Eigen::Vector3d(0.6, 0.2, 0.0);
  const livo::GeometryVersion active_version{
    active_sample.plane, active_sample.geometry_revision};
  const auto active_planes =
    activeRegistry({active_version}, 77U);
  ASSERT_NE(active_planes, nullptr);

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry = livo::GeometrySnapshotId{77U};
  request.active_planes = active_planes->snapshot();
  request.samples = {active_sample, inactive_sample};
  const auto result = session.bootstrapGeometry(request);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInactiveGeometryVersion);
  EXPECT_EQ(result.geometry_writes, 0U);
  EXPECT_EQ(session.audit().store_revision, 0U);
  EXPECT_EQ(session.audit().landmark_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  GeometryLeaseFromOtherRegistryRejectsWholeBatch)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::LegacyPlaneLandmarkSample sample = geometrySample();
  const livo::GeometryVersion version{
    sample.plane, sample.geometry_revision};
  const auto expected_registry =
    activeRegistry({version}, 77U);
  const auto other_registry =
    activeRegistry({version}, 77U);
  ASSERT_NE(expected_registry, nullptr);
  ASSERT_NE(other_registry, nullptr);
  const livo::ActivePlaneIdentitySnapshot other_snapshot =
    other_registry->snapshot();
  const auto foreign_lease = other_snapshot.leaseFor(version);
  ASSERT_TRUE(foreign_lease.has_value());
  sample.active_plane_lease = *foreign_lease;

  livo::SharedLandmarkGeometryBootstrapRequest request;
  request.geometry = livo::GeometrySnapshotId{77U};
  request.active_planes = expected_registry->snapshot();
  request.samples.push_back(sample);
  const auto result = session.bootstrapGeometry(request);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kActivePlaneRegistryMismatch);
  EXPECT_EQ(result.geometry_writes, 0U);
  EXPECT_EQ(session.audit().store_revision, 0U);
  EXPECT_EQ(session.audit().landmark_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  DestroyedRegistryCannotBeRevivedByAnotherRegistryWithSameGeometry)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  const livo::LegacyPlaneLandmarkSample first = geometrySample();
  const livo::GeometryVersion version{
    first.plane, first.geometry_revision};
  auto original_registry = activeRegistry({version}, 77U);
  ASSERT_NE(original_registry, nullptr);
  livo::SharedLandmarkGeometryBootstrapRequest initial;
  initial.geometry = livo::GeometrySnapshotId{77U};
  initial.active_planes = original_registry->snapshot();
  initial.samples.push_back(first);
  ASSERT_EQ(
    session.bootstrapGeometry(initial).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  ASSERT_EQ(session.audit().landmark_count, 1U);

  original_registry.reset();
  session.configureGeometry(config.geometry);
  const auto replacement_registry =
    activeRegistry({version}, 78U);
  ASSERT_NE(replacement_registry, nullptr);
  livo::LegacyPlaneLandmarkSample second = first;
  second.position_world_m =
    Eigen::Vector3d(0.6, 0.2, 0.0);
  livo::SharedLandmarkGeometryBootstrapRequest revival;
  revival.geometry = livo::GeometrySnapshotId{78U};
  revival.active_planes =
    replacement_registry->snapshot();
  revival.samples.push_back(second);
  const auto result = session.bootstrapGeometry(revival);

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kActivePlaneRegistryMismatch);
  EXPECT_EQ(result.geometry_writes, 0U);
  EXPECT_EQ(session.audit().store_revision, 1U);
  EXPECT_EQ(session.audit().landmark_count, 1U);
}

TEST(
  SharedLandmarkShadowSession,
  ExplicitReconfigureAllowsRegistrySwitchOnlyWhileStoreIsEmpty)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  const livo::LegacyPlaneLandmarkSample sample =
    geometrySample();
  const livo::GeometryVersion version{
    sample.plane, sample.geometry_revision};
  const auto first_registry =
    activeRegistry({version}, 77U);
  ASSERT_NE(first_registry, nullptr);
  livo::SharedLandmarkGeometryBootstrapRequest empty;
  empty.geometry = livo::GeometrySnapshotId{77U};
  empty.active_planes = first_registry->snapshot();
  const auto first = session.bootstrapGeometry(empty);
  EXPECT_EQ(first.geometry_writes, 0U);
  ASSERT_EQ(session.audit().store_revision, 0U);
  ASSERT_EQ(session.audit().landmark_count, 0U);

  session.configureGeometry(config.geometry);
  const auto replacement_registry =
    activeRegistry({version}, 78U);
  ASSERT_NE(replacement_registry, nullptr);
  livo::SharedLandmarkGeometryBootstrapRequest replacement;
  replacement.geometry = livo::GeometrySnapshotId{78U};
  replacement.active_planes =
    replacement_registry->snapshot();
  replacement.samples.push_back(sample);
  const auto accepted =
    session.bootstrapGeometry(replacement);

  EXPECT_EQ(
    accepted.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  EXPECT_EQ(accepted.geometry_writes, 1U);
  EXPECT_EQ(session.audit().store_revision, 1U);
  EXPECT_EQ(session.audit().landmark_count, 1U);
}

TEST(
  SharedLandmarkShadowSession,
  MajorReplacementRejectsOldEpochAndAcceptsNewEpochFromSameRegistry)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  const livo::LegacyPlaneLandmarkSample old_sample =
    geometrySample();
  const livo::GeometryVersion old_version{
    old_sample.plane, old_sample.geometry_revision};
  const auto registry =
    activeRegistry({old_version}, 77U);
  ASSERT_NE(registry, nullptr);
  livo::SharedLandmarkGeometryBootstrapRequest initial;
  initial.geometry = livo::GeometrySnapshotId{77U};
  initial.active_planes = registry->snapshot();
  initial.samples.push_back(old_sample);
  ASSERT_EQ(
    session.bootstrapGeometry(initial).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  const livo::GeometryVersion new_version{
    livo::PlaneIdentity{old_sample.plane.id, 2U}, 1U};
  ASSERT_TRUE(registry->publish(
    {livo::ActivePlaneDescriptor{new_version, 78U}}));
  const livo::ActivePlaneIdentitySnapshot current =
    registry->snapshot();

  livo::SharedLandmarkGeometryBootstrapRequest stale;
  stale.geometry = livo::GeometrySnapshotId{78U};
  stale.active_planes = current;
  stale.samples.push_back(old_sample);
  const auto rejected = session.bootstrapGeometry(stale);
  EXPECT_EQ(
    rejected.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInactiveGeometryVersion);
  EXPECT_EQ(rejected.geometry_writes, 0U);
  EXPECT_EQ(session.audit().landmark_count, 1U);

  livo::LegacyPlaneLandmarkSample new_sample =
    old_sample;
  new_sample.plane = new_version.plane;
  new_sample.geometry_revision = new_version.revision;
  livo::SharedLandmarkGeometryBootstrapRequest replacement;
  replacement.geometry = livo::GeometrySnapshotId{78U};
  replacement.active_planes = current;
  replacement.samples.push_back(new_sample);
  const auto accepted =
    session.bootstrapGeometry(replacement);
  EXPECT_EQ(
    accepted.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  EXPECT_EQ(accepted.geometry_writes, 1U);
  EXPECT_EQ(session.audit().store_revision, 2U);
  EXPECT_EQ(session.audit().landmark_count, 2U);
}

TEST(
  SharedLandmarkShadowSession,
  FeatureConfigurationForwardingPreservesExistingShadowStore)
{
  livo::SharedLandmarkShadowSession session;
  livo::LegacyLandmarkShadowConfig geometry_config;
  geometry_config.enabled = true;
  session.configureGeometry(geometry_config);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::LegacyLandmarkObservationShadowConfig
    observation_config;
  observation_config.enabled = true;
  session.configureObservation(observation_config);
  livo::SharedLandmarkTombstoneShadowConfig
    tombstone_config;
  tombstone_config.enabled = true;
  session.configureTombstone(tombstone_config);

  const auto audit = session.audit();
  EXPECT_TRUE(session.active());
  EXPECT_EQ(audit.geometry.value, 77U);
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.lifecycle_lidar_seeded, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  MissingOutdoorTimeCapabilityKeepsGeometryAndRejectsObservation)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  session.recordCameraConfiguration(0U);
  session.advanceVisualStateRevision();

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.active_planes =
    active_planes->snapshot();
  observation.samples.push_back(observationSample());
  auto staged = session.stageObservations(observation);

  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kMissingTimeTransferConfiguration);
  EXPECT_TRUE(staged.consumed);
  EXPECT_EQ(staged.batch.value, 2U);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_FALSE(staged.solver_applied);
  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(audit.consumed_batches, 2U);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  FrameProvenanceRejectsCrossCameraSwapBeforePhotometricGate)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  ASSERT_EQ(
    session.configureTimeTransfer(
      livo::CameraTimeTransferConfig{}).status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  const auto bundle = makeTimeTransferBundle(
    10.0, {{true, true, false, false}});
  ASSERT_TRUE(bundle.validatedReceipt().has_value());
  ASSERT_TRUE(bundle.transfers[0U].has_value());
  livo::SharedLandmarkShadowObservationRequest observation;
  observation.active_planes = active_planes->snapshot();
  observation.time_transfer_batch =
    bundle.validatedReceipt();
  auto sample = observationSample();
  sample.camera_id = 1U;
  sample.time_transfer = bundle.transfers[0U];
  sample.exposure_timestamp_sec =
    sample.time_transfer->targetTimestampSec();
  observation.samples.push_back(std::move(sample));

  const auto staged = session.stageObservations(observation);
  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInvalidRuntimeObservationProvenance);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_EQ(session.audit().observation_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  BoundTimeEntryAndFrameProvenanceRejectReplay)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  ASSERT_EQ(
    session.configureTimeTransfer(
      livo::CameraTimeTransferConfig{}).status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  const auto stale_bundle = makeTimeTransferBundle(
    10.0, {{true, false, false, false}});
  const auto current_bundle = makeTimeTransferBundle(
    10.01, {{true, false, false, false}});
  ASSERT_TRUE(current_bundle.validatedReceipt().has_value());
  livo::SharedLandmarkShadowObservationRequest observation;
  observation.active_planes = active_planes->snapshot();
  observation.time_transfer_batch =
    current_bundle.validatedReceipt();
  auto sample = observationSample();
  sample.time_transfer = stale_bundle.transfers[0U];
  sample.exposure_timestamp_sec =
    sample.time_transfer->targetTimestampSec();
  observation.samples.push_back(sample);

  auto staged = session.stageObservations(observation);
  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInvalidTimeTransferBracket);
  EXPECT_FALSE(staged.transaction.valid());

  observation.samples.front().time_transfer =
    current_bundle.transfers[0U];
  observation.samples.front().exposure_timestamp_sec +=
    1.0e-3;
  staged = session.stageObservations(observation);
  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInvalidRuntimeObservationProvenance);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_EQ(session.audit().observation_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  RawStrictProviderCannotReachShadowObservationStore)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  session.recordCameraConfiguration(0U);
  session.advanceVisualStateRevision();
  ASSERT_EQ(
    session.configureTimeTransfer(
      livo::CameraTimeTransferConfig{}).status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);
  const livo::PhotometricCalibration calibration =
    rawCalibration(0U, 17U);
  ASSERT_EQ(
    session.configurePhotometric(
      0U, strictEvidence(calibration)).status,
    livo::ShadowPhotometricCapabilityStatus::
    kStrictProviderUnavailable);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.active_planes =
    active_planes->snapshot();
  observation.samples.push_back(observationSample());
  const auto staged = session.stageObservations(observation);
  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kStrictPhotometricUnavailable);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_TRUE(staged.consumed);
  EXPECT_EQ(staged.batch.value, 2U);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.solver_applied);

  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(audit.consumed_batches, 2U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.state.value, 1U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.
    camera_model_revisions[0U],
    1U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.
    photometric_calibration_revisions[0U],
    0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.time_transfer_revision,
    1U);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  MixedObservationBootstrapWithInactiveGeometryRejectsAllWrites)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  session.recordCameraConfiguration(0U);
  session.advanceVisualStateRevision();
  ASSERT_EQ(
    session.configureTimeTransfer(
      livo::CameraTimeTransferConfig{}).status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);

  const livo::LegacyPlaneLandmarkSample active_sample =
    geometrySample();
  livo::LegacyPlaneLandmarkSample inactive_sample =
    geometrySample();
  inactive_sample.plane = livo::PlaneIdentity{8U, 1U};
  const livo::GeometryVersion active_version{
    active_sample.plane, active_sample.geometry_revision};
  const livo::GeometryVersion inactive_version{
    inactive_sample.plane,
    inactive_sample.geometry_revision};
  const auto registry = activeRegistry(
    {active_version, inactive_version}, 77U);
  ASSERT_NE(registry, nullptr);
  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.active_planes = registry->snapshot();
  geometry.samples = {active_sample, inactive_sample};
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  ASSERT_TRUE(registry->publish(
    {livo::ActivePlaneDescriptor{
      active_version, 78U}}));
  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.active_planes = registry->snapshot();
  observation.samples.push_back(observationSample());
  livo::SharedLandmarkShadowObservationSample inactive_observation =
    observationSample();
  inactive_observation.support_key.plane =
    inactive_sample.plane;
  observation.samples.push_back(
    std::move(inactive_observation));
  const auto staged = session.stageObservations(observation);

  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kInactiveGeometryVersion);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_EQ(session.audit().store_revision, 1U);
  EXPECT_EQ(session.audit().landmark_count, 2U);
  EXPECT_EQ(session.audit().observation_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  StaleObservationSnapshotRejectsWithoutAnyWrite)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  session.recordCameraConfiguration(0U);
  session.advanceVisualStateRevision();
  ASSERT_EQ(
    session.configureTimeTransfer(
      livo::CameraTimeTransferConfig{}).status,
    livo::TimeTransferConfigurationUpdateStatus::kUpdated);

  const livo::LegacyPlaneLandmarkSample sample =
    geometrySample();
  const livo::GeometryVersion version{
    sample.plane, sample.geometry_revision};
  const auto registry = activeRegistry({version}, 77U);
  ASSERT_NE(registry, nullptr);
  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.active_planes = registry->snapshot();
  geometry.samples.push_back(sample);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.active_planes = registry->snapshot();
  observation.samples.push_back(observationSample());

  ASSERT_TRUE(registry->publish(
    {livo::ActivePlaneDescriptor{version, 78U}}));
  const auto staged = session.stageObservations(observation);

  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kStaleActivePlaneSnapshot);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_EQ(session.audit().store_revision, 1U);
  EXPECT_EQ(session.audit().observation_count, 0U);
}

TEST(
  SharedLandmarkShadowSession,
  RobustAffinePhotometryAlwaysRemainsExperimentalShadow)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  session.recordCameraConfiguration(0U);
  session.advanceVisualStateRevision();
  session.configureTimeTransfer(
    livo::CameraTimeTransferConfig{});
  livo::ShadowPhotometricCapabilityEvidence robust;
  robust.mode =
    livo::PhotometricMode::kRobustAffineExperimental;
  ASSERT_EQ(
    session.configurePhotometric(0U, robust).status,
    livo::ShadowPhotometricCapabilityStatus::
    kRobustAffineExperimental);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.active_planes =
    active_planes->snapshot();
  observation.samples.push_back(observationSample());
  const auto staged = session.stageObservations(observation);

  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kExperimentalShadowOnly);
  EXPECT_TRUE(staged.consumed);
  EXPECT_EQ(staged.batch.value, 2U);
  EXPECT_EQ(staged.staged_observation_count, 0U);
  EXPECT_FALSE(staged.transaction.valid());
  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.
    photometric_calibration_revisions[0U],
    0U);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  RawPhotometricReconfigurationCannotMintRevisionOrTransaction)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.observation.enabled = true;
  livo::SharedLandmarkShadowSession session(config);
  ASSERT_TRUE(bindObservationRuntime(session));
  session.recordCameraConfiguration(0U);
  session.advanceVisualStateRevision();
  session.configureTimeTransfer(
    livo::CameraTimeTransferConfig{});
  const livo::PhotometricCalibration first_calibration =
    rawCalibration(0U, 17U);
  const auto first = session.configurePhotometric(
    0U, strictEvidence(first_calibration));
  ASSERT_EQ(
    first.status,
    livo::ShadowPhotometricCapabilityStatus::
    kStrictProviderUnavailable);
  ASSERT_FALSE(first.changed);
  ASSERT_EQ(first.strict_calibration_revision, 0U);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::SharedLandmarkShadowObservationRequest observation;
  observation.time_transfer_batch =
    validObservationTimeBundle().validatedReceipt();
  observation.active_planes =
    active_planes->snapshot();
  observation.samples.push_back(observationSample());
  const auto staged = session.stageObservations(observation);
  EXPECT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kStrictPhotometricUnavailable);
  EXPECT_FALSE(staged.transaction.valid());
  EXPECT_EQ(staged.staged_observation_count, 0U);

  const livo::PhotometricCalibration second_calibration =
    rawCalibration(0U, 18U);
  const auto changed = session.configurePhotometric(
    0U, strictEvidence(second_calibration));
  EXPECT_EQ(
    changed.status,
    livo::ShadowPhotometricCapabilityStatus::
    kStrictProviderUnavailable);
  EXPECT_FALSE(changed.changed);
  EXPECT_EQ(changed.strict_calibration_revision, 0U);

  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(audit.consumed_batches, 2U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.
    photometric_calibration_revisions[0U],
    0U);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  LioPosteriorRefitCommitsWithoutAnyVisualCapability)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.refit.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  ASSERT_EQ(
    session.advanceLioPosteriorRevision().value, 1U);

  livo::SharedLandmarkGeometryRefitRequest refit;
  refit.samples.push_back(refitSample());
  auto staged = session.stageGeometryRefits(refit);
  ASSERT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryRefitStaged);
  ASSERT_TRUE(staged.transaction.valid());
  EXPECT_TRUE(staged.consumed);
  EXPECT_EQ(staged.batch.value, 2U);
  EXPECT_EQ(staged.staged_refit_count, 1U);
  EXPECT_FALSE(staged.solver_applied);

  const auto committed =
    session.commitGeometryRefits(
    std::move(staged.transaction));
  EXPECT_EQ(
    committed.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryRefitCommitted);
  EXPECT_EQ(committed.batch.value, 2U);
  EXPECT_TRUE(committed.consumed);
  EXPECT_EQ(committed.geometry_refits, 1U);
  EXPECT_EQ(committed.observation_writes, 0U);
  EXPECT_FALSE(committed.solver_applied);

  const auto landmark =
    session.findLandmark(refitSample().support_key);
  ASSERT_TRUE(landmark.has_value());
  EXPECT_EQ(landmark->geometry_version.revision, 2U);
  EXPECT_TRUE(
    landmark->position_world_m.isApprox(
      refitSample().position_world_m));
  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 2U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.state.value, 0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.
    time_transfer_revision,
    0U);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  StaleLioPosteriorRefitRollsBackWholeBatchAtomically)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.refit.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  session.advanceLioPosteriorRevision();

  livo::SharedLandmarkGeometryRefitRequest refit;
  refit.samples.push_back(refitSample());
  auto staged = session.stageGeometryRefits(refit);
  ASSERT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryRefitStaged);
  ASSERT_TRUE(staged.transaction.valid());

  ASSERT_EQ(
    session.advanceLioPosteriorRevision().value, 2U);
  const auto committed =
    session.commitGeometryRefits(
    std::move(staged.transaction));

  EXPECT_EQ(
    committed.status,
    livo::SharedLandmarkShadowOperationStatus::
    kStaleLioPosteriorRevision);
  EXPECT_EQ(committed.batch.value, 2U);
  EXPECT_TRUE(committed.consumed);
  EXPECT_EQ(committed.geometry_refits, 0U);
  EXPECT_FALSE(committed.solver_applied);
  const auto landmark =
    session.findLandmark(refitSample().support_key);
  ASSERT_TRUE(landmark.has_value());
  EXPECT_EQ(landmark->geometry_version.revision, 1U);
  EXPECT_TRUE(
    landmark->position_world_m.isApprox(
      geometrySample().position_world_m));
  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 1U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(audit.consumed_batches, 2U);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  SharedLandmarkShadowSession,
  RefitConfigurationChangeInvalidatesStagedBatchAtomically)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.refit.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  session.advanceLioPosteriorRevision();

  livo::SharedLandmarkGeometryRefitRequest refit;
  refit.samples.push_back(refitSample());
  auto staged = session.stageGeometryRefits(refit);
  ASSERT_EQ(
    staged.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryRefitStaged);
  ASSERT_TRUE(staged.transaction.valid());

  livo::SharedLandmarkGeometryRefitShadowConfig disabled;
  session.configureRefit(disabled);
  const auto committed =
    session.commitGeometryRefits(
    std::move(staged.transaction));

  EXPECT_EQ(
    committed.status,
    livo::SharedLandmarkShadowOperationStatus::
    kStaleConfigurationGeneration);
  EXPECT_EQ(committed.geometry_refits, 0U);
  const auto landmark =
    session.findLandmark(refitSample().support_key);
  ASSERT_TRUE(landmark.has_value());
  EXPECT_EQ(landmark->geometry_version.revision, 1U);
  EXPECT_TRUE(
    landmark->position_world_m.isApprox(
      geometrySample().position_world_m));
  EXPECT_EQ(session.audit().store_revision, 1U);
}

TEST(
  CheckedConfigurationGeneration,
  ObservationStageReceiptBecomesStaleAfterReconfigure)
{
  static_assert(
    !std::is_copy_constructible_v<
      livo::CheckedConfigurationGeneration>);
  static_assert(
    !std::is_copy_assignable_v<
      livo::CheckedConfigurationGeneration>);

  livo::CheckedConfigurationGeneration generation;
  ASSERT_TRUE(generation.advance());
  const auto staged_observation = generation.receipt();
  ASSERT_TRUE(staged_observation.has_value());
  EXPECT_TRUE(generation.accepts(*staged_observation));
  const livo::ConfigurationGenerationReceipt copied_receipt =
    *staged_observation;
  EXPECT_TRUE(generation.accepts(copied_receipt));

  ASSERT_TRUE(generation.advance());
  EXPECT_FALSE(generation.accepts(*staged_observation));
  EXPECT_FALSE(generation.accepts(copied_receipt));

  livo::CheckedConfigurationGeneration foreign_generation;
  ASSERT_TRUE(foreign_generation.advance());
  EXPECT_FALSE(
    foreign_generation.accepts(*staged_observation));
}

TEST(
  CheckedConfigurationGeneration,
  ReusedOwnerAddressCannotReviveDestroyedAuthorityReceipt)
{
  using Generation = livo::CheckedConfigurationGeneration;
  alignas(Generation)
    std::byte storage[sizeof(Generation)];
  std::optional<livo::ConfigurationGenerationReceipt>
    destroyed_owner_receipt;

  auto * first = new (storage) Generation();
  ASSERT_TRUE(first->advance());
  destroyed_owner_receipt = first->receipt();
  ASSERT_TRUE(destroyed_owner_receipt.has_value());
  first->~Generation();

  auto * replacement = new (storage) Generation();
  ASSERT_TRUE(replacement->advance());
  EXPECT_FALSE(
    replacement->accepts(*destroyed_owner_receipt));
  replacement->~Generation();
}

TEST(
  CheckedConfigurationGeneration,
  OwnerDestructionExpiresReceiptAuthority)
{
  std::optional<livo::ConfigurationGenerationReceipt>
    destroyed_owner_receipt;
  {
    livo::CheckedConfigurationGeneration generation;
    ASSERT_TRUE(generation.advance());
    destroyed_owner_receipt = generation.receipt();
    ASSERT_TRUE(destroyed_owner_receipt.has_value());
    EXPECT_TRUE(generation.accepts(*destroyed_owner_receipt));
  }

  livo::CheckedConfigurationGeneration foreign_generation;
  ASSERT_TRUE(foreign_generation.advance());
  EXPECT_FALSE(
    foreign_generation.accepts(*destroyed_owner_receipt));
}

TEST(
  CheckedConfigurationGeneration,
  MoveConstructionFailsClosedAndExpiresExistingReceipt)
{
  livo::CheckedConfigurationGeneration source;
  ASSERT_TRUE(source.advance());
  const auto source_receipt = source.receipt();
  ASSERT_TRUE(source_receipt.has_value());

  livo::CheckedConfigurationGeneration destination(
    std::move(source));

  EXPECT_TRUE(source.exhausted());
  EXPECT_TRUE(destination.exhausted());
  EXPECT_FALSE(source.accepts(*source_receipt));
  EXPECT_FALSE(destination.accepts(*source_receipt));
  EXPECT_FALSE(destination.receipt().has_value());
  EXPECT_FALSE(destination.advance());
}

TEST(
  CheckedConfigurationGeneration,
  MoveAssignmentFailsClosedAndExpiresBothOwnersReceipts)
{
  livo::CheckedConfigurationGeneration source;
  ASSERT_TRUE(source.advance());
  const auto source_receipt = source.receipt();
  ASSERT_TRUE(source_receipt.has_value());

  livo::CheckedConfigurationGeneration destination;
  ASSERT_TRUE(destination.advance());
  const auto destination_receipt = destination.receipt();
  ASSERT_TRUE(destination_receipt.has_value());

  destination = std::move(source);

  EXPECT_TRUE(source.exhausted());
  EXPECT_TRUE(destination.exhausted());
  EXPECT_FALSE(source.accepts(*source_receipt));
  EXPECT_FALSE(destination.accepts(*source_receipt));
  EXPECT_FALSE(destination.accepts(*destination_receipt));
  EXPECT_FALSE(destination.receipt().has_value());
}

TEST(
  CheckedConfigurationGeneration,
  SelfMoveAssignmentFailsClosed)
{
  livo::CheckedConfigurationGeneration generation;
  ASSERT_TRUE(generation.advance());
  const auto receipt = generation.receipt();
  ASSERT_TRUE(receipt.has_value());

  generation = std::move(generation);

  EXPECT_TRUE(generation.exhausted());
  EXPECT_FALSE(generation.accepts(*receipt));
  EXPECT_FALSE(generation.receipt().has_value());
}

TEST(
  SharedLandmarkShadowSession,
  PlaneRefitEventConsumerUsesSessionOwnedLioToken)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.refit.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  ASSERT_EQ(
    session.observeLioPosteriorGeneration(77U).value, 1U);

  livo::PlaneRefitEvent event;
  event.expected_geometry_version =
    livo::GeometryVersion{
    livo::PlaneIdentity{7U, 1U}, 1U};
  event.replacement_geometry_version =
    livo::GeometryVersion{
    livo::PlaneIdentity{7U, 1U}, 2U};
  event.generation = 77U;
  event.center_world_m = Eigen::Vector3d(0.1, 0.2, 0.01);
  event.position_covariance_m2 =
    0.005 * Eigen::Matrix3d::Identity();
  event.normal_world = Eigen::Vector3d::UnitZ();
  event.plane_covariance =
    0.0005 * Eigen::Matrix<double, 6, 6>::Identity();
  event.local_basis.origin_world_m = Eigen::Vector3d::Zero();
  event.local_basis.u_axis_world = Eigen::Vector3d::UnitX();
  event.local_basis.v_axis_world = Eigen::Vector3d::UnitY();

  const auto result = session.applyRefitEvents({event});

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryRefitCommitted);
  EXPECT_TRUE(result.consumed);
  EXPECT_EQ(result.batch.value, 2U);
  EXPECT_EQ(result.input_count, 1U);
  EXPECT_EQ(result.matched_count, 1U);
  EXPECT_EQ(result.geometry_refits, 1U);
  EXPECT_TRUE(result.acknowledge_events);
  EXPECT_FALSE(result.solver_applied);
  const auto landmark =
    session.findLandmark(observationSample().support_key);
  ASSERT_TRUE(landmark.has_value());
  EXPECT_EQ(landmark->geometry_version.revision, 2U);
  EXPECT_EQ(session.audit().store_revision, 2U);
}

TEST(
  SharedLandmarkShadowSession,
  LidarTombstoneDoesNotRequireVisualRevisionCapabilities)
{
  livo::SharedLandmarkShadowSessionConfig config;
  config.geometry.enabled = true;
  config.tombstone.enabled = true;
  livo::SharedLandmarkShadowSession session(config);

  livo::SharedLandmarkGeometryBootstrapRequest geometry;
  geometry.geometry = livo::GeometrySnapshotId{77U};
  geometry.samples.push_back(geometrySample());
  const auto active_planes =
    activateGeometryRequest(geometry);
  ASSERT_NE(active_planes, nullptr);
  ASSERT_EQ(
    session.bootstrapGeometry(geometry).status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);

  livo::PlaneTombstoneEvent event;
  event.old_identity = livo::PlaneIdentity{7U, 1U};
  event.last_revision = 1U;
  event.generation = 2U;
  event.reason =
    livo::PlaneTombstoneReason::kMajorReplacement;
  const auto result = session.applyTombstones({event});

  EXPECT_EQ(
    result.status,
    livo::SharedLandmarkShadowOperationStatus::
    kTombstoneCommitted);
  EXPECT_TRUE(result.consumed);
  EXPECT_EQ(result.batch.value, 2U);
  EXPECT_EQ(result.tombstoned_landmarks, 1U);
  EXPECT_TRUE(result.acknowledge_events);
  EXPECT_FALSE(result.solver_applied);
  const auto landmark =
    session.findLandmark(
    observationSample().support_key);
  ASSERT_TRUE(landmark.has_value());
  EXPECT_EQ(
    landmark->lifecycle,
    livo::LandmarkLifecycle::kTombstoned);
  const auto audit = session.audit();
  EXPECT_EQ(audit.store_revision, 2U);
  EXPECT_EQ(audit.landmark_count, 1U);
  EXPECT_EQ(audit.observation_count, 0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.state.value, 0U);
  EXPECT_EQ(
    audit.visual_revision_snapshot.
    time_transfer_revision,
    0U);
  EXPECT_FALSE(audit.solver_applied);
}

}  // namespace
