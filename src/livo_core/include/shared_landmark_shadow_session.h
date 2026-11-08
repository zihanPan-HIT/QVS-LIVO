#ifndef LIVO_SHARED_LANDMARK_SHADOW_SESSION_H_
#define LIVO_SHARED_LANDMARK_SHADOW_SESSION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "camera_batch_time_transfer_receipt.h"
#include "configuration_generation.h"
#include "legacy_landmark_observation_shadow_adapter.h"
#include "legacy_landmark_shadow_adapter.h"
#include "shared_landmark_refit_shadow.h"
#include "shared_landmark_tombstone_shadow.h"

namespace livo
{

class CameraObservationBatchAuthorityIdentity;
class MultiCameraLandmarkLinearizationShadow;
class ReferenceOnlyMigrationRuntimePublisherIdentity;

struct SharedLandmarkGeometryBootstrapRequest
{
  GeometrySnapshotId geometry;
  ActivePlaneIdentitySnapshot active_planes;
  std::vector<LegacyPlaneLandmarkSample> samples;
};

// Value-owned boundary object. Revisions and typed commit tokens are injected
// by SharedLandmarkShadowSession and cannot be supplied by callers.
struct SharedLandmarkShadowObservationSample
{
  GeometrySupportKey support_key;
  std::uint8_t camera_id{0U};
  double exposure_timestamp_sec{0.0};
  std::optional<CameraBatchTimeTransferEntry>
    time_transfer;
  std::shared_ptr<const ReferenceOnlyMigrationFrameProvenance>
    frame_provenance;
  Eigen::Vector2d pixel{Eigen::Vector2d::Zero()};
  int level{0};
  std::vector<std::vector<float>> owned_patch_pyramid;
  PhotometricStatus frame_photometric_status{
    PhotometricStatus::kInvalidInput};
  // Anchor-state uncertainty is diagnostic input. Transfer and alignment
  // uncertainty are always derived from the sealed batch entry.
  Eigen::Matrix<double, 6, 6> anchor_pose_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  bool visual_update_accepted{false};
  bool is_bootstrap{false};
  bool isolated{false};
};

struct SharedLandmarkShadowObservationRequest
{
  ActivePlaneIdentitySnapshot active_planes;
  std::optional<CameraBatchTimeTransferReceipt>
    time_transfer_batch;
  std::vector<SharedLandmarkShadowObservationSample> samples;
};

struct SharedLandmarkGeometryRefitShadowConfig
{
  bool enabled{false};
  std::size_t maximum_refits_per_batch{512U};
};

struct SharedLandmarkGeometryRefitSample
{
  GeometrySupportKey support_key;
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d position_covariance_m2{
    Eigen::Matrix3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 6, 6> plane_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  PlaneLocalBasis local_basis;
};

struct SharedLandmarkGeometryRefitRequest
{
  std::vector<SharedLandmarkGeometryRefitSample> samples;
};

struct SharedLandmarkShadowSessionConfig
{
  LegacyLandmarkShadowConfig geometry;
  LegacyLandmarkObservationShadowConfig observation;
  SharedLandmarkTombstoneShadowConfig tombstone;
  SharedLandmarkGeometryRefitShadowConfig refit;

  bool enabled() const
  {
    return
      geometry.enabled ||
      observation.enabled ||
      tombstone.enabled ||
      refit.enabled;
  }
};

enum class TimeTransferConfigurationUpdateStatus
{
  kDisabled,
  kInvalidConfiguration,
  kConfigurationGenerationExhausted,
  kUnchanged,
  kUpdated
};

struct TimeTransferConfigurationUpdateResult
{
  TimeTransferConfigurationUpdateStatus status{
    TimeTransferConfigurationUpdateStatus::kDisabled};
  std::uint64_t revision{0U};
  bool changed{false};
};

enum class ShadowPhotometricCapabilityStatus
{
  kStrictProviderUnavailable,
  kStrictCalibrated,
  kRobustAffineExperimental
};

struct ShadowPhotometricCapabilityEvidence
{
  PhotometricMode mode{PhotometricMode::kCalibratedDirect};
  // Borrowed only for the duration of configurePhotometric(). Strict
  // capability is accepted exclusively through the calibration module's
  // sealed receipt validation; the session never retains this pointer.
  const PhotometricCalibration * strict_calibration{nullptr};
};

struct ShadowPhotometricCapability
{
  ShadowPhotometricCapabilityStatus status{
    ShadowPhotometricCapabilityStatus::
    kStrictProviderUnavailable};
  std::uint64_t strict_calibration_revision{0U};
  std::uint64_t experimental_shadow_revision{0U};
  bool changed{false};
};

struct CameraConfigurationRevision
{
  std::uint64_t rig_revision{0U};
  std::uint64_t camera_model_revision{0U};
  bool changed{false};
};

enum class SharedLandmarkShadowOperationStatus
{
  kDisabled,
  kGeometryCommitted,
  kNoValidGeometrySeed,
  kNoGeometryChange,
  kMissingGeometryAnchor,
  kMissingActivePlaneSnapshot,
  kStaleActivePlaneSnapshot,
  kInactiveGeometryVersion,
  kActivePlaneRegistryMismatch,
  kMissingVisualStateRevision,
  kMissingRigRevision,
  kMissingCameraModelRevision,
  kMissingTimeTransferConfiguration,
  kInvalidTimeTransferBracket,
  kInvalidRuntimeObservationProvenance,
  kStaleConfigurationGeneration,
  kConfigurationGenerationExhausted,
  kStrictPhotometricUnavailable,
  kInvalidPhotometricEvidence,
  kExperimentalShadowOnly,
  kNoEligibleObservation,
  kObservationStaged,
  kObservationCommitted,
  kGeometryRefitStaged,
  kGeometryRefitCommitted,
  kTombstoneCommitted,
  kNoMatchingTombstone,
  kMissingLioPosteriorRevision,
  kStaleLioPosteriorRevision,
  kStaleRevision,
  kStaleStoreRevision,
  kRejected
};

struct SharedLandmarkShadowOperationResult
{
  SharedLandmarkShadowOperationStatus status{
    SharedLandmarkShadowOperationStatus::kDisabled};
  BatchId batch;
  bool consumed{false};
  std::size_t geometry_writes{0U};
  std::size_t geometry_refits{0U};
  std::size_t tombstoned_landmarks{0U};
  std::size_t observation_writes{0U};
  std::size_t input_count{0U};
  std::size_t matched_count{0U};
  std::size_t ignored_count{0U};
  bool acknowledge_events{false};
  bool solver_applied{false};
};

class SharedLandmarkShadowObservationTransaction
{
public:
  SharedLandmarkShadowObservationTransaction();
  ~SharedLandmarkShadowObservationTransaction();
  SharedLandmarkShadowObservationTransaction(
    SharedLandmarkShadowObservationTransaction &&) noexcept;
  SharedLandmarkShadowObservationTransaction & operator=(
    SharedLandmarkShadowObservationTransaction &&) noexcept;

  SharedLandmarkShadowObservationTransaction(
    const SharedLandmarkShadowObservationTransaction &) = delete;
  SharedLandmarkShadowObservationTransaction & operator=(
    const SharedLandmarkShadowObservationTransaction &) = delete;

  bool valid() const;

private:
  friend class SharedLandmarkShadowSession;
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

class SharedLandmarkGeometryRefitTransaction
{
public:
  SharedLandmarkGeometryRefitTransaction();
  ~SharedLandmarkGeometryRefitTransaction();
  SharedLandmarkGeometryRefitTransaction(
    SharedLandmarkGeometryRefitTransaction &&) noexcept;
  SharedLandmarkGeometryRefitTransaction & operator=(
    SharedLandmarkGeometryRefitTransaction &&) noexcept;

  SharedLandmarkGeometryRefitTransaction(
    const SharedLandmarkGeometryRefitTransaction &) = delete;
  SharedLandmarkGeometryRefitTransaction & operator=(
    const SharedLandmarkGeometryRefitTransaction &) = delete;

  bool valid() const;

private:
  friend class SharedLandmarkShadowSession;
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

struct SharedLandmarkShadowObservationStageResult
{
  SharedLandmarkShadowOperationStatus status{
    SharedLandmarkShadowOperationStatus::kDisabled};
  BatchId batch;
  bool consumed{false};
  std::size_t staged_observation_count{0U};
  bool solver_applied{false};
  SharedLandmarkShadowObservationTransaction transaction;
};

struct SharedLandmarkGeometryRefitStageResult
{
  SharedLandmarkShadowOperationStatus status{
    SharedLandmarkShadowOperationStatus::kDisabled};
  BatchId batch;
  bool consumed{false};
  std::size_t staged_refit_count{0U};
  bool solver_applied{false};
  SharedLandmarkGeometryRefitTransaction transaction;
};

struct SharedLandmarkShadowAudit
{
  GeometrySnapshotId geometry;
  bool geometry_enabled{false};
  bool observation_enabled{false};
  bool tombstone_enabled{false};
  bool refit_enabled{false};
  std::size_t maximum_geometry_seeds_per_batch{0U};
  std::size_t maximum_observations_per_batch{0U};
  std::uint64_t consumed_batches{0U};
  std::uint64_t store_revision{0U};
  std::size_t landmark_count{0U};
  std::size_t observation_count{0U};
  std::size_t lifecycle_lidar_seeded{0U};
  std::size_t lifecycle_same_camera{0U};
  std::size_t lifecycle_migration_shadow{0U};
  std::size_t lifecycle_migration_active{0U};
  std::size_t lifecycle_tombstoned{0U};
  std::uint64_t time_transfer_configuration_revision{0U};
  std::uint64_t configuration_generation{0U};
  bool configuration_generation_exhausted{false};
  LioPosteriorRevision lio_posterior_revision;
  CommitRevisionSet visual_revision_snapshot;
  std::array<ShadowPhotometricCapability, 4U>
    photometric_capabilities;
  bool solver_applied{false};
};

class SharedLandmarkShadowSession
{
public:
  SharedLandmarkShadowSession();
  explicit SharedLandmarkShadowSession(
    const SharedLandmarkShadowSessionConfig & config);
  ~SharedLandmarkShadowSession();

  SharedLandmarkShadowSession(
    SharedLandmarkShadowSession &&) noexcept;
  SharedLandmarkShadowSession & operator=(
    SharedLandmarkShadowSession &&) noexcept;

  SharedLandmarkShadowSession(
    const SharedLandmarkShadowSession &) = delete;
  SharedLandmarkShadowSession & operator=(
    const SharedLandmarkShadowSession &) = delete;

  bool active() const;
  void configure(
    const SharedLandmarkShadowSessionConfig & config);
  void configureGeometry(
    const LegacyLandmarkShadowConfig & config);
  void configureObservation(
    const LegacyLandmarkObservationShadowConfig & config);
  void configureTombstone(
    const SharedLandmarkTombstoneShadowConfig & config);
  void configureRefit(
    const SharedLandmarkGeometryRefitShadowConfig & config);
  TimeTransferConfigurationUpdateResult
  configureTimeTransfer(
    const CameraTimeTransferConfig & config);
  bool bindRuntimeObservationAuthority(
    const CameraObservationBatchAuthorityIdentity &
      batch_authority,
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      runtime_authority);
  CameraConfigurationRevision
  recordCameraConfiguration(std::uint8_t camera_id);
  StateRevision advanceVisualStateRevision();
  LioPosteriorRevision advanceLioPosteriorRevision();
  LioPosteriorRevision observeLioPosteriorGeneration(
    std::uint64_t generation);
  ShadowPhotometricCapability configurePhotometric(
    std::uint8_t camera_id,
    const ShadowPhotometricCapabilityEvidence & evidence);
  SharedLandmarkShadowOperationResult bootstrapGeometry(
    const SharedLandmarkGeometryBootstrapRequest & request);
  SharedLandmarkShadowObservationStageResult
  stageObservations(
    const SharedLandmarkShadowObservationRequest & request);
  SharedLandmarkShadowOperationResult commitObservations(
    SharedLandmarkShadowObservationTransaction && transaction);
  SharedLandmarkGeometryRefitStageResult
  stageGeometryRefits(
    const SharedLandmarkGeometryRefitRequest & request);
  SharedLandmarkShadowOperationResult commitGeometryRefits(
    SharedLandmarkGeometryRefitTransaction && transaction);
  SharedLandmarkShadowOperationResult applyTombstones(
    const std::vector<PlaneTombstoneEvent> & events);
  SharedLandmarkShadowOperationResult applyRefitEvents(
    const std::vector<PlaneRefitEvent> & events);
  std::optional<LandmarkSnapshot> findLandmark(
    const GeometrySupportKey & support_key) const;
  bool resolveGeometryIdentity(
    const LegacyPlaneLandmarkSample & sample,
    GeometrySupportKey & support_key,
    GeometryVersion & geometry_version) const;
  SharedLandmarkShadowAudit audit() const;

private:
  friend class MultiCameraLandmarkLinearizationShadow;
  std::optional<ReadSnapshot>
  snapshotForMultiCameraLandmarkLinearizationShadow() const;

  bool advanceConfigurationGeneration();
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
  CheckedConfigurationGeneration configuration_generation_;
};

const char * sharedLandmarkShadowOperationStatusName(
  SharedLandmarkShadowOperationStatus status);

}  // namespace livo

#endif  // LIVO_SHARED_LANDMARK_SHADOW_SESSION_H_
