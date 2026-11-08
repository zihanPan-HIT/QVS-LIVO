#include "shared_landmark_shadow_session.h"

#include "checked_increment.h"
#include "reference_only_migration_runtime_provenance.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <utility>

namespace livo
{
namespace
{

bool sameTimeTransferConfiguration(
  const CameraTimeTransferConfig & lhs,
  const CameraTimeTransferConfig & rhs)
{
  return
    lhs.max_abs_transfer_sec == rhs.max_abs_transfer_sec &&
    lhs.max_imu_gap_sec == rhs.max_imu_gap_sec &&
    lhs.imu_timestamp_stddev_sec ==
      rhs.imu_timestamp_stddev_sec &&
    lhs.ring_buffer_duration_sec ==
      rhs.ring_buffer_duration_sec &&
    lhs.epoch_jump_threshold_sec ==
      rhs.epoch_jump_threshold_sec &&
    lhs.gyro_noise_stddev_radps ==
      rhs.gyro_noise_stddev_radps &&
    lhs.accel_noise_stddev_mps2 ==
      rhs.accel_noise_stddev_mps2 &&
    lhs.gyro_bias_random_walk_stddev_radps_per_sqrt_sec ==
      rhs.gyro_bias_random_walk_stddev_radps_per_sqrt_sec &&
    lhs.accel_bias_random_walk_stddev_mps2_per_sqrt_sec ==
      rhs.accel_bias_random_walk_stddev_mps2_per_sqrt_sec;
}

bool transferCompatibleWithConfiguration(
  const CameraPoseTransferRaw & transfer,
  const CameraTimeTransferConfig & config)
{
  constexpr double kToleranceSec = 1.0e-12;
  return
    transfer.internallyValid() &&
    std::abs(transfer.delta_time_sec) <=
      config.max_abs_transfer_sec + kToleranceSec &&
    transfer.maximum_imu_gap_sec <=
      config.max_imu_gap_sec + kToleranceSec;
}

SharedLandmarkShadowOperationStatus geometryStatus(
  const LandmarkShadowBootstrapStatus status)
{
  switch (status)
  {
    case LandmarkShadowBootstrapStatus::kCommitted:
      return
        SharedLandmarkShadowOperationStatus::
        kGeometryCommitted;
    case LandmarkShadowBootstrapStatus::kNoValidSeed:
      return
        SharedLandmarkShadowOperationStatus::
        kNoValidGeometrySeed;
    case LandmarkShadowBootstrapStatus::kNoGeometryChange:
      return
        SharedLandmarkShadowOperationStatus::
        kNoGeometryChange;
    case LandmarkShadowBootstrapStatus::kDisabled:
      return SharedLandmarkShadowOperationStatus::kDisabled;
    case LandmarkShadowBootstrapStatus::kInvalidConfiguration:
    case LandmarkShadowBootstrapStatus::kPrepareRejected:
    case LandmarkShadowBootstrapStatus::kProofRejected:
    case LandmarkShadowBootstrapStatus::kCommitRejected:
      return SharedLandmarkShadowOperationStatus::kRejected;
  }
  return SharedLandmarkShadowOperationStatus::kRejected;
}

bool sameVisualRevisionsForCameras(
  const CommitRevisionSet & lhs,
  const CommitRevisionSet & rhs,
  const std::uint8_t camera_mask)
{
  if (
    lhs.state.value != rhs.state.value ||
    lhs.rig_calibration_revision !=
      rhs.rig_calibration_revision ||
    lhs.time_transfer_revision !=
      rhs.time_transfer_revision)
  {
    return false;
  }
  for (std::uint8_t camera_id = 0U;
       camera_id < 4U; ++camera_id)
  {
    const bool involved =
      (camera_mask &
       static_cast<std::uint8_t>(1U << camera_id)) != 0U;
    if (
      involved &&
      (lhs.camera_model_revisions[camera_id] !=
         rhs.camera_model_revisions[camera_id] ||
       lhs.photometric_calibration_revisions[camera_id] !=
         rhs.photometric_calibration_revisions[camera_id]))
    {
      return false;
    }
  }
  return true;
}

}  // namespace

struct SharedLandmarkShadowSession::Implementation
{
  explicit Implementation(
    const SharedLandmarkShadowSessionConfig & value)
  {
    configure(value);
  }

  void configure(
    const SharedLandmarkShadowSessionConfig & value)
  {
    config = value;
    if (config.geometry.enabled)
    {
      geometry_adapter =
        std::make_unique<LegacyLandmarkShadowAdapter>(
        config.geometry);
    }
    else
    {
      geometry_adapter.reset();
    }
    if (config.observation.enabled)
    {
      observation_adapter =
        std::make_unique<
        LegacyLandmarkObservationShadowAdapter>(
        config.observation);
    }
    else
    {
      observation_adapter.reset();
    }
    if (config.tombstone.enabled)
    {
      tombstone_consumer =
        std::make_unique<
        SharedLandmarkTombstoneShadowConsumer>(
        config.tombstone);
    }
    else
    {
      tombstone_consumer.reset();
    }
    if (config.refit.enabled)
    {
      SharedLandmarkRefitShadowConfig refit_config;
      refit_config.enabled = true;
      refit_consumer =
        std::make_unique<SharedLandmarkRefitShadow>(
        refit_config);
    }
    else
    {
      refit_consumer.reset();
    }
  }

  bool storeEmpty() const
  {
    return store.snapshot(
      geometry, revisions).landmarks.empty();
  }

  SharedLandmarkShadowSessionConfig config;
  MultiViewLandmarkStore store;
  std::unique_ptr<LegacyLandmarkShadowAdapter>
    geometry_adapter;
  std::unique_ptr<LegacyLandmarkObservationShadowAdapter>
    observation_adapter;
  std::unique_ptr<SharedLandmarkTombstoneShadowConsumer>
    tombstone_consumer;
  std::unique_ptr<SharedLandmarkRefitShadow>
    refit_consumer;
  CommitRevisionSet revisions;
  GeometrySnapshotId geometry;
  std::optional<LidarAnchoredCommitToken>
    lidar_commit_token;
  LioPosteriorRevision lio_posterior_revision;
  std::uint64_t last_lio_generation{0U};
  std::uint64_t batch_counter{0U};
  std::optional<CameraTimeTransferConfig> time_transfer_config;
  std::uint64_t time_transfer_configuration_revision{0U};
  std::array<ShadowPhotometricCapability, 4U>
    photometric_capabilities;
  std::uint64_t experimental_photometric_revision{0U};
  std::optional<CameraObservationBatchAuthorityIdentity>
    runtime_observation_batch_authority;
  std::optional<
    ReferenceOnlyMigrationRuntimePublisherIdentity>
    runtime_observation_publisher_authority;
  // Pins only the registry instance. Per-plane leases remain operation scoped
  // and are always reissued from the current request snapshot.
  std::optional<ActivePlaneIdentitySnapshot>
    active_plane_registry_affinity;
  // Exact snapshot last confirmed by a successful authority operation.
  // Unlike registry affinity, this becomes stale when the registry advances.
  std::optional<ActivePlaneIdentitySnapshot>
    current_active_planes;
};

struct SharedLandmarkShadowObservationTransaction::Implementation
{
  const void * owner{nullptr};
  std::optional<ConfigurationGenerationReceipt>
    configuration_receipt;
  BatchId batch;
  GeometrySnapshotId geometry;
  CommitRevisionSet revisions;
  ActivePlaneIdentitySnapshot active_planes;
  std::uint8_t camera_mask{0U};
  std::size_t staged_observation_count{0U};
  StagedLegacyLandmarkObservationShadowBatch staged;
};

struct SharedLandmarkGeometryRefitTransaction::Implementation
{
  const void * owner{nullptr};
  std::optional<ConfigurationGenerationReceipt>
    configuration_receipt;
  BatchId batch;
  GeometrySnapshotId geometry;
  LioPosteriorRevision lio_posterior_revision;
  std::size_t staged_refit_count{0U};
  StagedLandmarkDelta delta;
  std::optional<MapCommitProof> proof;
  bool valid{false};
};

SharedLandmarkShadowObservationTransaction::
SharedLandmarkShadowObservationTransaction() = default;

SharedLandmarkShadowObservationTransaction::
~SharedLandmarkShadowObservationTransaction() = default;

SharedLandmarkShadowObservationTransaction::
SharedLandmarkShadowObservationTransaction(
  SharedLandmarkShadowObservationTransaction &&) noexcept = default;

SharedLandmarkShadowObservationTransaction &
SharedLandmarkShadowObservationTransaction::operator=(
  SharedLandmarkShadowObservationTransaction &&) noexcept = default;

bool SharedLandmarkShadowObservationTransaction::valid() const
{
  return
    implementation_ != nullptr &&
    implementation_->staged.valid();
}

SharedLandmarkGeometryRefitTransaction::
SharedLandmarkGeometryRefitTransaction() = default;

SharedLandmarkGeometryRefitTransaction::
~SharedLandmarkGeometryRefitTransaction() = default;

SharedLandmarkGeometryRefitTransaction::
SharedLandmarkGeometryRefitTransaction(
  SharedLandmarkGeometryRefitTransaction &&) noexcept = default;

SharedLandmarkGeometryRefitTransaction &
SharedLandmarkGeometryRefitTransaction::operator=(
  SharedLandmarkGeometryRefitTransaction &&) noexcept = default;

bool SharedLandmarkGeometryRefitTransaction::valid() const
{
  return
    implementation_ != nullptr &&
    implementation_->valid &&
    implementation_->proof.has_value() &&
    implementation_->proof->valid();
}

SharedLandmarkShadowSession::SharedLandmarkShadowSession() = default;

SharedLandmarkShadowSession::SharedLandmarkShadowSession(
  const SharedLandmarkShadowSessionConfig & config)
{
  configure(config);
}

SharedLandmarkShadowSession::~SharedLandmarkShadowSession() = default;

SharedLandmarkShadowSession::SharedLandmarkShadowSession(
  SharedLandmarkShadowSession &&) noexcept = default;

SharedLandmarkShadowSession &
SharedLandmarkShadowSession::operator=(
  SharedLandmarkShadowSession &&) noexcept = default;

bool SharedLandmarkShadowSession::active() const
{
  return
    !configuration_generation_.exhausted() &&
    implementation_ != nullptr;
}

bool SharedLandmarkShadowSession::
advanceConfigurationGeneration()
{
  if (
    !configuration_generation_.advance())
  {
    implementation_.reset();
    return false;
  }
  return true;
}

void SharedLandmarkShadowSession::configure(
  const SharedLandmarkShadowSessionConfig & config)
{
  if (configuration_generation_.exhausted())
  {
    return;
  }
  // An explicit reconfigure is a semantic boundary even when scalar values
  // are equal: while the store is empty it intentionally releases registry
  // affinity, and every staged transaction must therefore become stale.
  const bool changed = active() || config.enabled();
  if (!changed)
  {
    return;
  }
  if (!advanceConfigurationGeneration())
  {
    return;
  }
  if (!config.enabled())
  {
    implementation_.reset();
    return;
  }
  if (implementation_ == nullptr)
  {
    implementation_ =
      std::make_unique<Implementation>(config);
  }
  else
  {
    if (implementation_->storeEmpty())
    {
      implementation_->
        active_plane_registry_affinity.reset();
      implementation_->current_active_planes.reset();
    }
    implementation_->configure(config);
  }
}

void SharedLandmarkShadowSession::configureGeometry(
  const LegacyLandmarkShadowConfig & config)
{
  SharedLandmarkShadowSessionConfig next;
  if (active())
  {
    next = implementation_->config;
  }
  next.geometry = config;
  configure(next);
}

void SharedLandmarkShadowSession::configureObservation(
  const LegacyLandmarkObservationShadowConfig & config)
{
  SharedLandmarkShadowSessionConfig next;
  if (active())
  {
    next = implementation_->config;
  }
  next.observation = config;
  configure(next);
}

void SharedLandmarkShadowSession::configureTombstone(
  const SharedLandmarkTombstoneShadowConfig & config)
{
  SharedLandmarkShadowSessionConfig next;
  if (active())
  {
    next = implementation_->config;
  }
  next.tombstone = config;
  configure(next);
}

void SharedLandmarkShadowSession::configureRefit(
  const SharedLandmarkGeometryRefitShadowConfig & config)
{
  SharedLandmarkShadowSessionConfig next;
  if (active())
  {
    next = implementation_->config;
  }
  next.refit = config;
  configure(next);
}

TimeTransferConfigurationUpdateResult
SharedLandmarkShadowSession::configureTimeTransfer(
  const CameraTimeTransferConfig & config)
{
  TimeTransferConfigurationUpdateResult result;
  if (!active())
  {
    return result;
  }
  result.revision =
    implementation_->
    time_transfer_configuration_revision;
  if (!CameraTimeTransfer(config).configValid())
  {
    result.status =
      TimeTransferConfigurationUpdateStatus::
      kInvalidConfiguration;
    return result;
  }
  if (
    implementation_->time_transfer_config.has_value() &&
    sameTimeTransferConfiguration(
      *implementation_->time_transfer_config, config))
  {
    result.status =
      TimeTransferConfigurationUpdateStatus::kUnchanged;
    return result;
  }
  if (!advanceConfigurationGeneration())
  {
    result.status =
      TimeTransferConfigurationUpdateStatus::
      kConfigurationGenerationExhausted;
    result.revision = 0U;
    return result;
  }
  implementation_->time_transfer_config = config;
  if (!checkedIncrement(
      implementation_->
      time_transfer_configuration_revision))
  {
    configuration_generation_.failClosed();
    implementation_.reset();
    result.status =
      TimeTransferConfigurationUpdateStatus::
      kConfigurationGenerationExhausted;
    result.revision = 0U;
    return result;
  }
  implementation_->revisions.time_transfer_revision =
    implementation_->
    time_transfer_configuration_revision;
  result.status =
    TimeTransferConfigurationUpdateStatus::kUpdated;
  result.revision =
    implementation_->
    time_transfer_configuration_revision;
  result.changed = true;
  return result;
}

bool SharedLandmarkShadowSession::
bindRuntimeObservationAuthority(
  const CameraObservationBatchAuthorityIdentity &
    batch_authority,
  const ReferenceOnlyMigrationRuntimePublisherIdentity &
    runtime_authority)
{
  if (
    !active() ||
    !batch_authority.valid() ||
    !runtime_authority.valid() ||
    !runtime_authority.bindsBatchAuthority(
      batch_authority) ||
    implementation_->
      runtime_observation_batch_authority.has_value() ||
    implementation_->
      runtime_observation_publisher_authority.has_value())
  {
    return false;
  }
  implementation_->runtime_observation_batch_authority =
    batch_authority;
  implementation_->
    runtime_observation_publisher_authority =
    runtime_authority;
  return true;
}

CameraConfigurationRevision
SharedLandmarkShadowSession::recordCameraConfiguration(
  const std::uint8_t camera_id)
{
  CameraConfigurationRevision result;
  if (!active())
  {
    return result;
  }
  result.rig_revision =
    implementation_->revisions.rig_calibration_revision;
  if (
    camera_id >=
    implementation_->revisions.
      camera_model_revisions.size())
  {
    return result;
  }
  result.rig_revision =
    ++implementation_->revisions.
      rig_calibration_revision;
  result.camera_model_revision =
    ++implementation_->revisions.
      camera_model_revisions[camera_id];
  result.changed = true;
  return result;
}

StateRevision
SharedLandmarkShadowSession::advanceVisualStateRevision()
{
  if (!active())
  {
    return StateRevision{};
  }
  implementation_->revisions.state.value++;
  return implementation_->revisions.state;
}

LioPosteriorRevision
SharedLandmarkShadowSession::advanceLioPosteriorRevision()
{
  if (!active())
  {
    return LioPosteriorRevision{};
  }
  ++implementation_->lio_posterior_revision.value;
  return implementation_->lio_posterior_revision;
}

LioPosteriorRevision
SharedLandmarkShadowSession::observeLioPosteriorGeneration(
  const std::uint64_t generation)
{
  if (!active())
  {
    return LioPosteriorRevision{};
  }
  if (
    generation != 0U &&
    generation != implementation_->last_lio_generation)
  {
    implementation_->last_lio_generation = generation;
    ++implementation_->lio_posterior_revision.value;
  }
  return implementation_->lio_posterior_revision;
}

ShadowPhotometricCapability
SharedLandmarkShadowSession::configurePhotometric(
  const std::uint8_t camera_id,
  const ShadowPhotometricCapabilityEvidence & evidence)
{
  ShadowPhotometricCapability result;
  if (!active() ||
      camera_id >=
        implementation_->
        photometric_capabilities.size())
  {
    return result;
  }

  const ShadowPhotometricCapability previous =
    implementation_->photometric_capabilities[camera_id];
  const bool strict_valid =
    evidence.mode == PhotometricMode::kCalibratedDirect &&
    evidence.strict_calibration != nullptr &&
    evidence.strict_calibration->cameraCalibrationStatus(
      camera_id) == PhotometricStatus::kValid &&
    evidence.strict_calibration->cameraCalibrationRevision(
      camera_id) != 0U;
  if (strict_valid)
  {
    result.status =
      ShadowPhotometricCapabilityStatus::kStrictCalibrated;
    result.strict_calibration_revision =
      evidence.strict_calibration->
      cameraCalibrationRevision(camera_id);
  }
  else if (
    evidence.mode ==
      PhotometricMode::kRobustAffineExperimental)
  {
    result.status =
      ShadowPhotometricCapabilityStatus::
      kRobustAffineExperimental;
    if (
      previous.status ==
      ShadowPhotometricCapabilityStatus::
      kRobustAffineExperimental)
    {
      result.experimental_shadow_revision =
        previous.experimental_shadow_revision;
    }
    else
    {
      result.experimental_shadow_revision =
        ++implementation_->
        experimental_photometric_revision;
    }
  }
  result.changed =
    result.status != previous.status ||
    result.strict_calibration_revision !=
      previous.strict_calibration_revision ||
    result.experimental_shadow_revision !=
      previous.experimental_shadow_revision;
  if (
    result.changed &&
    !advanceConfigurationGeneration())
  {
    return ShadowPhotometricCapability{};
  }
  implementation_->photometric_capabilities[camera_id] =
    result;
  implementation_->revisions.
    photometric_calibration_revisions[camera_id] =
    result.status ==
      ShadowPhotometricCapabilityStatus::kStrictCalibrated ?
    result.strict_calibration_revision : 0U;
  return result;
}

SharedLandmarkShadowOperationResult
SharedLandmarkShadowSession::bootstrapGeometry(
  const SharedLandmarkGeometryBootstrapRequest & request)
{
  SharedLandmarkShadowOperationResult result;
  if (
    !active() ||
    implementation_->geometry_adapter == nullptr)
  {
    return result;
  }

  result.consumed = true;
  result.batch =
    BatchId{++implementation_->batch_counter};
  if (request.active_planes.revision() == 0U)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingActivePlaneSnapshot;
    return result;
  }
  if (!request.active_planes.valid())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleActivePlaneSnapshot;
    return result;
  }
  if (
    implementation_->active_plane_registry_affinity.has_value() &&
    !implementation_->active_plane_registry_affinity->
      sameRegistryAs(request.active_planes))
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kActivePlaneRegistryMismatch;
    return result;
  }
  if (!implementation_->
      active_plane_registry_affinity.has_value())
  {
    implementation_->active_plane_registry_affinity =
      request.active_planes;
  }
  LandmarkShadowBootstrapRequest adapter_request;
  adapter_request.batch = result.batch;
  adapter_request.geometry = request.geometry;
  adapter_request.active_planes = request.active_planes;
  adapter_request.expected_revisions =
    implementation_->revisions;
  adapter_request.lidar_plane_stable =
    !request.samples.empty();
  adapter_request.samples.reserve(request.samples.size());
  for (const LegacyPlaneLandmarkSample & sample :
       request.samples)
  {
    const GeometryVersion geometry_version{
      sample.plane, sample.geometry_revision};
    const GeometryVersion supplied_version =
      sample.active_plane_lease.geometryVersion();
    if (
      supplied_version.valid() &&
      (!request.active_planes.contains(
         sample.active_plane_lease) ||
       !(supplied_version.plane ==
         geometry_version.plane) ||
       supplied_version.revision !=
         geometry_version.revision))
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kActivePlaneRegistryMismatch;
      return result;
    }
    const std::optional<ActivePlaneIdentityLease> lease =
      request.active_planes.leaseFor(geometry_version);
    if (!lease.has_value())
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kInactiveGeometryVersion;
      return result;
    }
    LegacyPlaneLandmarkSample stamped = sample;
    stamped.active_plane_lease = *lease;
    adapter_request.samples.push_back(std::move(stamped));
  }
  const LandmarkShadowBootstrapResult adapter_result =
    implementation_->geometry_adapter->bootstrap(
    implementation_->store, adapter_request);

  result.status = geometryStatus(adapter_result.status);
  result.geometry_writes =
    adapter_result.commit.geometry_writes;
  if (adapter_result.commit_token.has_value())
  {
    implementation_->lidar_commit_token =
      adapter_result.commit_token;
    implementation_->geometry = request.geometry;
    implementation_->current_active_planes =
      request.active_planes;
  }
  return result;
}

SharedLandmarkShadowObservationStageResult
SharedLandmarkShadowSession::stageObservations(
  const SharedLandmarkShadowObservationRequest & request)
{
  SharedLandmarkShadowObservationStageResult result;
  if (
    !active() ||
    implementation_->observation_adapter == nullptr)
  {
    return result;
  }
  result.consumed = true;
  result.batch =
    BatchId{++implementation_->batch_counter};
  if (
    !implementation_->
      runtime_observation_batch_authority.has_value() ||
    !implementation_->
      runtime_observation_publisher_authority.has_value())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kInvalidRuntimeObservationProvenance;
    return result;
  }
  if (
    implementation_->geometry.value == 0U ||
    !implementation_->lidar_commit_token.has_value())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingGeometryAnchor;
    return result;
  }
  if (request.active_planes.revision() == 0U)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingActivePlaneSnapshot;
    return result;
  }
  if (!request.active_planes.valid())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleActivePlaneSnapshot;
    return result;
  }
  if (
    !implementation_->
      active_plane_registry_affinity.has_value() ||
    !implementation_->active_plane_registry_affinity->
      sameRegistryAs(request.active_planes))
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kActivePlaneRegistryMismatch;
    return result;
  }
  if (request.samples.empty())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kNoEligibleObservation;
    return result;
  }
  if (
    implementation_->revisions.
      time_transfer_revision == 0U)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingTimeTransferConfiguration;
    return result;
  }
  if (
    !request.time_transfer_batch.has_value() ||
    !request.time_transfer_batch->valid())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kInvalidTimeTransferBracket;
    return result;
  }

  std::uint8_t camera_mask = 0U;
  bool has_bootstrap = false;
  bool has_visual = false;
  for (const SharedLandmarkShadowObservationSample & sample :
       request.samples)
  {
    if (sample.camera_id >= 4U)
    {
      result.status =
        SharedLandmarkShadowOperationStatus::kRejected;
      return result;
    }
    camera_mask |= static_cast<std::uint8_t>(
      1U << sample.camera_id);
    has_bootstrap = has_bootstrap || sample.is_bootstrap;
    has_visual = has_visual || !sample.is_bootstrap;
    if (
      sample.frame_provenance == nullptr ||
      !sample.frame_provenance->valid() ||
      !sample.frame_provenance->issuedBy(
        *implementation_->
        runtime_observation_publisher_authority) ||
      sample.frame_provenance->cameraId() !=
        sample.camera_id ||
      sample.frame_provenance->exposureTimestampSec() !=
        sample.exposure_timestamp_sec)
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kInvalidRuntimeObservationProvenance;
      return result;
    }
    if (
      !sample.time_transfer.has_value() ||
      !sample.time_transfer->has_value() ||
      !request.time_transfer_batch->accepts(
        sample.frame_provenance->timeTransferEntry()) ||
      !request.time_transfer_batch->accepts(
        *sample.time_transfer) ||
      sample.time_transfer->cameraId() !=
        sample.camera_id ||
      sample.time_transfer->targetTimestampSec() !=
        sample.exposure_timestamp_sec ||
      !implementation_->time_transfer_config.has_value() ||
      !transferCompatibleWithConfiguration(
        sample.time_transfer->value().raw(),
        *implementation_->time_transfer_config))
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kInvalidTimeTransferBracket;
      return result;
    }
  }
  if (has_bootstrap && has_visual)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }
  if (implementation_->revisions.state.value == 0U)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingVisualStateRevision;
    return result;
  }
  if (
    implementation_->revisions.
      rig_calibration_revision == 0U)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingRigRevision;
    return result;
  }

  const ReadSnapshot landmark_snapshot =
    implementation_->store.snapshot(
    implementation_->geometry,
    implementation_->revisions);
  std::vector<ActivePlaneIdentityLease> active_plane_leases;
  active_plane_leases.reserve(request.samples.size());
  for (const SharedLandmarkShadowObservationSample & sample :
       request.samples)
  {
    const auto landmark = std::find_if(
      landmark_snapshot.landmarks.begin(),
      landmark_snapshot.landmarks.end(),
      [&sample](const LandmarkSnapshot & candidate) {
        return candidate.support_key == sample.support_key;
      });
    if (landmark == landmark_snapshot.landmarks.end())
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kInactiveGeometryVersion;
      return result;
    }
    const std::optional<ActivePlaneIdentityLease> lease =
      request.active_planes.leaseFor(
      landmark->geometry_version);
    if (!lease.has_value())
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kInactiveGeometryVersion;
      return result;
    }
    active_plane_leases.push_back(*lease);
  }

  for (std::uint8_t camera_id = 0U;
       camera_id < 4U; ++camera_id)
  {
    const bool involved =
      (camera_mask &
       static_cast<std::uint8_t>(1U << camera_id)) != 0U;
    if (!involved)
    {
      continue;
    }
    if (
      implementation_->revisions.
      camera_model_revisions[camera_id] == 0U)
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kMissingCameraModelRevision;
      return result;
    }
    const ShadowPhotometricCapability & capability =
      implementation_->photometric_capabilities[camera_id];
    if (
      capability.status ==
      ShadowPhotometricCapabilityStatus::
      kRobustAffineExperimental)
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kExperimentalShadowOnly;
      return result;
    }
    if (
      capability.status !=
      ShadowPhotometricCapabilityStatus::kStrictCalibrated)
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kStrictPhotometricUnavailable;
      return result;
    }
  }

  LegacyLandmarkObservationShadowRequest adapter_request;
  adapter_request.batch = result.batch;
  adapter_request.geometry = implementation_->geometry;
  adapter_request.active_planes = request.active_planes;
  adapter_request.samples.reserve(request.samples.size());
  for (std::size_t sample_index = 0U;
       sample_index < request.samples.size(); ++sample_index)
  {
    const SharedLandmarkShadowObservationSample & sample =
      request.samples[sample_index];
    if (
      sample.frame_photometric_status !=
      PhotometricStatus::kValid)
    {
      result.status =
        SharedLandmarkShadowOperationStatus::
        kInvalidPhotometricEvidence;
      return result;
    }
    LegacyLandmarkObservationShadowSample stamped;
    const CameraPoseTransferRaw & transfer =
      sample.time_transfer->value().raw();
    stamped.support_key = sample.support_key;
    stamped.active_plane_lease =
      active_plane_leases[sample_index];
    stamped.camera_id = sample.camera_id;
    stamped.exposure_timestamp_sec =
      transfer.target_timestamp_sec;
    stamped.rotation_world_from_imu =
      transfer.rotation_world_from_imu;
    stamped.position_world_m =
      transfer.position_world_m;
    stamped.rig_calibration_revision =
      implementation_->revisions.
      rig_calibration_revision;
    stamped.camera_model_revision =
      implementation_->revisions.
      camera_model_revisions[sample.camera_id];
    stamped.pixel = sample.pixel;
    stamped.level = sample.level;
    stamped.owned_patch_pyramid =
      sample.owned_patch_pyramid;
    stamped.photometric_mode =
      ObservationPhotometricMode::kSameCameraOnly;
    stamped.photometric_status =
      sample.frame_photometric_status;
    stamped.photometric_cross_camera_comparable = false;
    stamped.photometric_experimental = false;
    stamped.photometric_calibration_revision =
      implementation_->revisions.
      photometric_calibration_revisions[sample.camera_id];
    stamped.time_status = transfer.status;
    stamped.time_transfer_revision =
      transfer.imu_epoch;
    stamped.frame_provenance =
      sample.frame_provenance;
    stamped.pose_covariance =
      sample.anchor_pose_covariance +
      transfer.pose_transfer_covariance +
      transfer.time_alignment_pose_covariance;
    stamped.visual_update_accepted =
      sample.visual_update_accepted;
    stamped.is_bootstrap = sample.is_bootstrap;
    stamped.isolated = sample.isolated;
    adapter_request.samples.push_back(std::move(stamped));
  }
  if (has_bootstrap)
  {
    adapter_request.bootstrap_token =
      LidarAnchoredCommitToken(
      implementation_->geometry,
      implementation_->revisions);
  }
  else
  {
    adapter_request.accepted_state_token =
      AcceptedStateCommitToken(
      implementation_->geometry,
      implementation_->revisions);
  }

  LegacyLandmarkObservationShadowStageResult staged =
    implementation_->observation_adapter->stage(
    implementation_->store, adapter_request);
  if (
    staged.status ==
    LegacyLandmarkObservationShadowStatus::
    kNoEligibleObservation)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kNoEligibleObservation;
    return result;
  }
  if (
    staged.status !=
    LegacyLandmarkObservationShadowStatus::kStaged ||
    !staged.staged.valid())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }

  auto transaction =
    std::make_unique<
    SharedLandmarkShadowObservationTransaction::
    Implementation>();
  transaction->owner = implementation_.get();
  transaction->configuration_receipt =
    configuration_generation_.receipt();
  transaction->batch = result.batch;
  transaction->geometry = implementation_->geometry;
  transaction->revisions = implementation_->revisions;
  transaction->active_planes = request.active_planes;
  transaction->camera_mask = camera_mask;
  transaction->staged_observation_count =
    staged.staged_observation_count;
  transaction->staged = std::move(staged.staged);
  result.staged_observation_count =
    transaction->staged_observation_count;
  result.transaction.implementation_ =
    std::move(transaction);
  result.status =
    SharedLandmarkShadowOperationStatus::kObservationStaged;
  return result;
}

SharedLandmarkShadowOperationResult
SharedLandmarkShadowSession::commitObservations(
  SharedLandmarkShadowObservationTransaction && transaction)
{
  SharedLandmarkShadowOperationResult result;
  std::unique_ptr<
    SharedLandmarkShadowObservationTransaction::
    Implementation> staged =
    std::move(transaction.implementation_);
  if (staged == nullptr)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }
  result.batch = staged->batch;
  result.consumed = true;
  if (configuration_generation_.exhausted())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kConfigurationGenerationExhausted;
    return result;
  }
  if (
    !staged->configuration_receipt.has_value() ||
    !configuration_generation_.accepts(
      *staged->configuration_receipt))
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleConfigurationGeneration;
    return result;
  }
  if (!staged->active_planes.valid())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleActivePlaneSnapshot;
    return result;
  }
  if (
    !active() ||
    staged->owner != implementation_.get() ||
    !implementation_->
      active_plane_registry_affinity.has_value() ||
    !implementation_->active_plane_registry_affinity->
      sameRegistryAs(staged->active_planes) ||
    staged->geometry.value !=
      implementation_->geometry.value ||
    !sameVisualRevisionsForCameras(
      staged->revisions,
      implementation_->revisions,
      staged->camera_mask))
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kStaleRevision;
    return result;
  }

  const CommitResult commit =
    staged->staged.commit(implementation_->store);
  if (commit.status == CommitStatus::kCommitted)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kObservationCommitted;
    result.observation_writes =
      staged->staged_observation_count;
    implementation_->current_active_planes =
      staged->active_planes;
  }
  else if (
    commit.status == CommitStatus::kStaleStoreRevision)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleStoreRevision;
  }
  else
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
  }
  return result;
}

SharedLandmarkGeometryRefitStageResult
SharedLandmarkShadowSession::stageGeometryRefits(
  const SharedLandmarkGeometryRefitRequest & request)
{
  SharedLandmarkGeometryRefitStageResult result;
  if (
    !active() ||
    !implementation_->config.refit.enabled)
  {
    return result;
  }
  result.consumed = true;
  result.batch =
    BatchId{++implementation_->batch_counter};
  if (
    implementation_->geometry.value == 0U ||
    !implementation_->lidar_commit_token.has_value())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingGeometryAnchor;
    return result;
  }
  if (
    implementation_->lio_posterior_revision.value == 0U)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kMissingLioPosteriorRevision;
    return result;
  }
  if (
    request.samples.empty() ||
    implementation_->config.refit.
      maximum_refits_per_batch == 0U ||
    request.samples.size() >
      implementation_->config.refit.
      maximum_refits_per_batch)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }

  const ReadSnapshot snapshot =
    implementation_->store.snapshot(
    implementation_->geometry,
    implementation_->revisions);
  LandmarkQuery query;
  query.batch = result.batch;
  query.geometry = implementation_->geometry;
  query.expected_lio_posterior_revision =
    implementation_->lio_posterior_revision;
  query.expected_revisions = implementation_->revisions;
  query.geometry_refits.reserve(request.samples.size());
  for (const SharedLandmarkGeometryRefitSample & sample :
       request.samples)
  {
    const auto landmark = std::find_if(
      snapshot.landmarks.begin(),
      snapshot.landmarks.end(),
      [&sample](const LandmarkSnapshot & candidate) {
        return candidate.support_key == sample.support_key;
      });
    if (landmark == snapshot.landmarks.end())
    {
      continue;
    }
    GeometryRefitMutation mutation;
    mutation.support_key = sample.support_key;
    mutation.expected_geometry_version =
      landmark->geometry_version;
    mutation.replacement_geometry_version =
      landmark->geometry_version;
    ++mutation.replacement_geometry_version.revision;
    mutation.position_world_m = sample.position_world_m;
    mutation.position_covariance_m2 =
      sample.position_covariance_m2;
    mutation.normal_world = sample.normal_world;
    mutation.plane_covariance = sample.plane_covariance;
    mutation.local_basis = sample.local_basis;
    query.geometry_refits.push_back(std::move(mutation));
  }
  if (query.geometry_refits.empty())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }

  PreparedLandmarkBatch prepared =
    implementation_->store.prepare(snapshot, query);
  if (!prepared.valid())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }
  const LidarGeometryRefitCommitToken token(
    implementation_->geometry,
    implementation_->lio_posterior_revision);
  MapCommitProof proof =
    GeometryRefitGate::mint(token, prepared);
  if (!proof.valid())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }

  auto transaction =
    std::make_unique<
    SharedLandmarkGeometryRefitTransaction::
    Implementation>();
  transaction->owner = implementation_.get();
  transaction->configuration_receipt =
    configuration_generation_.receipt();
  transaction->batch = result.batch;
  transaction->geometry = implementation_->geometry;
  transaction->lio_posterior_revision =
    implementation_->lio_posterior_revision;
  transaction->staged_refit_count =
    query.geometry_refits.size();
  transaction->delta = prepared.take_delta();
  transaction->proof = std::move(proof);
  transaction->valid = true;
  result.staged_refit_count =
    transaction->staged_refit_count;
  result.transaction.implementation_ =
    std::move(transaction);
  result.status =
    SharedLandmarkShadowOperationStatus::
    kGeometryRefitStaged;
  return result;
}

SharedLandmarkShadowOperationResult
SharedLandmarkShadowSession::commitGeometryRefits(
  SharedLandmarkGeometryRefitTransaction && transaction)
{
  SharedLandmarkShadowOperationResult result;
  std::unique_ptr<
    SharedLandmarkGeometryRefitTransaction::
    Implementation> staged =
    std::move(transaction.implementation_);
  if (
    staged == nullptr ||
    !staged->valid ||
    !staged->proof.has_value())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
    return result;
  }
  result.batch = staged->batch;
  result.consumed = true;
  if (configuration_generation_.exhausted())
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kConfigurationGenerationExhausted;
    return result;
  }
  if (
    !staged->configuration_receipt.has_value() ||
    !configuration_generation_.accepts(
      *staged->configuration_receipt))
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleConfigurationGeneration;
    return result;
  }
  if (
    !active() ||
    staged->owner != implementation_.get() ||
    staged->geometry.value !=
      implementation_->geometry.value ||
    staged->lio_posterior_revision.value !=
      implementation_->lio_posterior_revision.value)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleLioPosteriorRevision;
    return result;
  }

  staged->valid = false;
  const CommitResult commit =
    implementation_->store.commit(
    std::move(staged->delta), *staged->proof);
  if (commit.status == CommitStatus::kCommitted)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kGeometryRefitCommitted;
    result.geometry_refits = commit.geometry_refits;
  }
  else if (
    commit.status == CommitStatus::kStaleStoreRevision)
  {
    result.status =
      SharedLandmarkShadowOperationStatus::
      kStaleStoreRevision;
  }
  else
  {
    result.status =
      SharedLandmarkShadowOperationStatus::kRejected;
  }
  return result;
}

SharedLandmarkShadowOperationResult
SharedLandmarkShadowSession::applyTombstones(
  const std::vector<PlaneTombstoneEvent> & events)
{
  SharedLandmarkShadowOperationResult result;
  if (
    !active() ||
    implementation_->tombstone_consumer == nullptr)
  {
    return result;
  }
  result.consumed = true;
  result.batch =
    BatchId{++implementation_->batch_counter};
  const LidarAnchoredCommitToken * token =
    implementation_->lidar_commit_token.has_value() ?
    &*implementation_->lidar_commit_token : nullptr;
  const SharedLandmarkTombstoneShadowResult applied =
    implementation_->tombstone_consumer->apply(
    implementation_->store, events, token, result.batch);
  result.tombstoned_landmarks =
    applied.tombstoned_landmarks;
  result.input_count = applied.input_event_count;
  result.matched_count = applied.matched_event_count;
  result.ignored_count = applied.ignored_event_count;
  result.acknowledge_events = applied.acknowledge_events;
  switch (applied.status)
  {
    case SharedLandmarkTombstoneShadowStatus::kCommitted:
      result.status =
        SharedLandmarkShadowOperationStatus::
        kTombstoneCommitted;
      break;
    case SharedLandmarkTombstoneShadowStatus::kNoEvents:
    case SharedLandmarkTombstoneShadowStatus::
      kNoMatchingLandmarks:
      result.status =
        SharedLandmarkShadowOperationStatus::
        kNoMatchingTombstone;
      break;
    case SharedLandmarkTombstoneShadowStatus::kDisabled:
      result.status =
        SharedLandmarkShadowOperationStatus::kDisabled;
      break;
    case SharedLandmarkTombstoneShadowStatus::
      kMissingLidarToken:
      result.status =
        SharedLandmarkShadowOperationStatus::
        kMissingGeometryAnchor;
      break;
    case SharedLandmarkTombstoneShadowStatus::kInvalidInput:
    case SharedLandmarkTombstoneShadowStatus::kStaleEvent:
    case SharedLandmarkTombstoneShadowStatus::kPrepareRejected:
    case SharedLandmarkTombstoneShadowStatus::kCommitRejected:
      result.status =
        SharedLandmarkShadowOperationStatus::kRejected;
      break;
  }
  return result;
}

SharedLandmarkShadowOperationResult
SharedLandmarkShadowSession::applyRefitEvents(
  const std::vector<PlaneRefitEvent> & events)
{
  SharedLandmarkShadowOperationResult result;
  if (
    !active() ||
    implementation_->refit_consumer == nullptr)
  {
    return result;
  }
  result.consumed = true;
  result.batch =
    BatchId{++implementation_->batch_counter};
  std::optional<LidarGeometryRefitCommitToken> token;
  if (
    implementation_->geometry.value != 0U &&
    implementation_->lio_posterior_revision.value != 0U)
  {
    token =
      LidarGeometryRefitCommitToken(
      implementation_->geometry,
      implementation_->lio_posterior_revision);
  }
  const SharedLandmarkRefitShadowResult applied =
    implementation_->refit_consumer->apply(
    implementation_->store, events,
    token.has_value() ? &*token : nullptr,
    result.batch);
  result.input_count = applied.input_event_count;
  result.matched_count = applied.matched_landmark_count;
  result.ignored_count = applied.ignored_event_count;
  result.geometry_refits = applied.geometry_refits;
  result.acknowledge_events = applied.acknowledge_events;
  switch (applied.status)
  {
    case SharedLandmarkRefitShadowStatus::kCommitted:
      result.status =
        SharedLandmarkShadowOperationStatus::
        kGeometryRefitCommitted;
      break;
    case SharedLandmarkRefitShadowStatus::kNoEvents:
    case SharedLandmarkRefitShadowStatus::
      kNoMatchingLandmarks:
      result.status =
        SharedLandmarkShadowOperationStatus::
        kNoGeometryChange;
      break;
    case SharedLandmarkRefitShadowStatus::kDisabled:
      result.status =
        SharedLandmarkShadowOperationStatus::kDisabled;
      break;
    case SharedLandmarkRefitShadowStatus::kMissingLidarToken:
      result.status =
        implementation_->geometry.value == 0U ?
        SharedLandmarkShadowOperationStatus::
        kMissingGeometryAnchor :
        SharedLandmarkShadowOperationStatus::
        kMissingLioPosteriorRevision;
      break;
    case SharedLandmarkRefitShadowStatus::kCommitRejected:
      result.status =
        applied.commit_status ==
        CommitStatus::kStaleStoreRevision ?
        SharedLandmarkShadowOperationStatus::
        kStaleStoreRevision :
        SharedLandmarkShadowOperationStatus::kRejected;
      break;
    case SharedLandmarkRefitShadowStatus::kInvalidInput:
    case SharedLandmarkRefitShadowStatus::kStaleEvent:
    case SharedLandmarkRefitShadowStatus::kPrepareRejected:
      result.status =
        SharedLandmarkShadowOperationStatus::kRejected;
      break;
  }
  return result;
}

std::optional<LandmarkSnapshot>
SharedLandmarkShadowSession::findLandmark(
  const GeometrySupportKey & support_key) const
{
  if (!active())
  {
    return std::nullopt;
  }
  const ReadSnapshot snapshot =
    implementation_->store.snapshot(
    implementation_->geometry,
    implementation_->revisions);
  const auto landmark = std::find_if(
    snapshot.landmarks.begin(), snapshot.landmarks.end(),
    [&support_key](const LandmarkSnapshot & candidate) {
      return candidate.support_key == support_key;
    });
  if (landmark != snapshot.landmarks.end())
  {
    return *landmark;
  }
  return std::nullopt;
}

bool SharedLandmarkShadowSession::resolveGeometryIdentity(
  const LegacyPlaneLandmarkSample & sample,
  GeometrySupportKey & support_key,
  GeometryVersion & geometry_version) const
{
  return
    active() &&
    implementation_->config.geometry.enabled &&
    resolveLegacyPlaneLandmarkIdentity(
      sample,
      implementation_->config.geometry.plane_cell_size_m,
      support_key, geometry_version);
}

SharedLandmarkShadowAudit SharedLandmarkShadowSession::audit() const
{
  SharedLandmarkShadowAudit result;
  result.configuration_generation =
    configuration_generation_.value();
  result.configuration_generation_exhausted =
    configuration_generation_.exhausted();
  if (active())
  {
    result.geometry = implementation_->geometry;
    result.geometry_enabled =
      implementation_->config.geometry.enabled;
    result.observation_enabled =
      implementation_->config.observation.enabled;
    result.tombstone_enabled =
      implementation_->config.tombstone.enabled;
    result.refit_enabled =
      implementation_->config.refit.enabled;
    result.maximum_geometry_seeds_per_batch =
      implementation_->config.geometry.
      maximum_seeds_per_batch;
    result.maximum_observations_per_batch =
      implementation_->config.observation.
      maximum_observations_per_batch;
    result.consumed_batches =
      implementation_->batch_counter;
    const ReadSnapshot snapshot =
      implementation_->store.snapshot(
      implementation_->geometry,
      implementation_->revisions);
    result.store_revision = snapshot.store_revision.value;
    result.landmark_count = snapshot.landmarks.size();
    result.observation_count =
      std::accumulate(
      snapshot.landmarks.begin(), snapshot.landmarks.end(),
      std::size_t{0U},
      [](const std::size_t count,
         const LandmarkSnapshot & landmark) {
        return count + landmark.observations.size();
      });
    for (const LandmarkSnapshot & landmark :
         snapshot.landmarks)
    {
      switch (landmark.lifecycle)
      {
        case LandmarkLifecycle::kLidarSeeded:
          ++result.lifecycle_lidar_seeded;
          break;
        case LandmarkLifecycle::kSameCameraTrackable:
          ++result.lifecycle_same_camera;
          break;
        case LandmarkLifecycle::kMigrationShadowEligible:
          ++result.lifecycle_migration_shadow;
          break;
        case LandmarkLifecycle::kMigrationActiveEligible:
          ++result.lifecycle_migration_active;
          break;
        case LandmarkLifecycle::kTombstoned:
          ++result.lifecycle_tombstoned;
          break;
      }
    }
    result.time_transfer_configuration_revision =
      implementation_->
      time_transfer_configuration_revision;
    result.lio_posterior_revision =
      implementation_->lio_posterior_revision;
    result.visual_revision_snapshot =
      implementation_->revisions;
    result.photometric_capabilities =
      implementation_->photometric_capabilities;
  }
  return result;
}

std::optional<ReadSnapshot>
SharedLandmarkShadowSession::
snapshotForMultiCameraLandmarkLinearizationShadow() const
{
  if (!active())
  {
    return std::nullopt;
  }
  if (!implementation_->current_active_planes.has_value())
  {
    return std::nullopt;
  }
  return implementation_->store.snapshot(
    implementation_->geometry,
    implementation_->revisions,
    *implementation_->current_active_planes);
}

const char * sharedLandmarkShadowOperationStatusName(
  const SharedLandmarkShadowOperationStatus status)
{
  switch (status)
  {
    case SharedLandmarkShadowOperationStatus::kDisabled:
      return "disabled";
    case SharedLandmarkShadowOperationStatus::kGeometryCommitted:
      return "geometry_committed";
    case SharedLandmarkShadowOperationStatus::kNoValidGeometrySeed:
      return "no_valid_geometry_seed";
    case SharedLandmarkShadowOperationStatus::kNoGeometryChange:
      return "no_geometry_change";
    case SharedLandmarkShadowOperationStatus::kMissingGeometryAnchor:
      return "missing_geometry_anchor";
    case SharedLandmarkShadowOperationStatus::
      kMissingActivePlaneSnapshot:
      return "missing_active_plane_snapshot";
    case SharedLandmarkShadowOperationStatus::
      kStaleActivePlaneSnapshot:
      return "stale_active_plane_snapshot";
    case SharedLandmarkShadowOperationStatus::
      kInactiveGeometryVersion:
      return "inactive_geometry_version";
    case SharedLandmarkShadowOperationStatus::
      kActivePlaneRegistryMismatch:
      return "active_plane_registry_mismatch";
    case SharedLandmarkShadowOperationStatus::
      kMissingVisualStateRevision:
      return "missing_visual_state_revision";
    case SharedLandmarkShadowOperationStatus::kMissingRigRevision:
      return "missing_rig_revision";
    case SharedLandmarkShadowOperationStatus::
      kMissingCameraModelRevision:
      return "missing_camera_model_revision";
    case SharedLandmarkShadowOperationStatus::
      kMissingTimeTransferConfiguration:
      return "missing_time_transfer_configuration";
    case SharedLandmarkShadowOperationStatus::
      kInvalidTimeTransferBracket:
      return "invalid_time_transfer_bracket";
    case SharedLandmarkShadowOperationStatus::
      kInvalidRuntimeObservationProvenance:
      return "invalid_runtime_observation_provenance";
    case SharedLandmarkShadowOperationStatus::
      kStaleConfigurationGeneration:
      return "stale_configuration_generation";
    case SharedLandmarkShadowOperationStatus::
      kConfigurationGenerationExhausted:
      return "configuration_generation_exhausted";
    case SharedLandmarkShadowOperationStatus::
      kStrictPhotometricUnavailable:
      return "strict_photometric_unavailable";
    case SharedLandmarkShadowOperationStatus::
      kInvalidPhotometricEvidence:
      return "invalid_photometric_evidence";
    case SharedLandmarkShadowOperationStatus::
      kExperimentalShadowOnly:
      return "experimental_shadow_only";
    case SharedLandmarkShadowOperationStatus::
      kNoEligibleObservation:
      return "no_eligible_observation";
    case SharedLandmarkShadowOperationStatus::kObservationStaged:
      return "observation_staged";
    case SharedLandmarkShadowOperationStatus::kObservationCommitted:
      return "observation_committed";
    case SharedLandmarkShadowOperationStatus::kGeometryRefitStaged:
      return "geometry_refit_staged";
    case SharedLandmarkShadowOperationStatus::kGeometryRefitCommitted:
      return "geometry_refit_committed";
    case SharedLandmarkShadowOperationStatus::kTombstoneCommitted:
      return "tombstone_committed";
    case SharedLandmarkShadowOperationStatus::kNoMatchingTombstone:
      return "no_matching_tombstone";
    case SharedLandmarkShadowOperationStatus::
      kMissingLioPosteriorRevision:
      return "missing_lio_posterior_revision";
    case SharedLandmarkShadowOperationStatus::
      kStaleLioPosteriorRevision:
      return "stale_lio_posterior_revision";
    case SharedLandmarkShadowOperationStatus::kStaleRevision:
      return "stale_revision";
    case SharedLandmarkShadowOperationStatus::kStaleStoreRevision:
      return "stale_store_revision";
    case SharedLandmarkShadowOperationStatus::kRejected:
      return "rejected";
  }
  return "unknown";
}

}  // namespace livo
