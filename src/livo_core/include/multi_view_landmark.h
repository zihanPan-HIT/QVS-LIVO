#ifndef LIVO_MULTI_VIEW_LANDMARK_H_
#define LIVO_MULTI_VIEW_LANDMARK_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "active_plane_identity_registry.h"
#include "geometry_identity.h"
#include "photometric_calibration.h"

class VIOManager;

namespace livo
{

class LegacyLandmarkShadowAdapter;
class PreparedLandmarkBatch;
class ReferenceOnlyMigrationFrameProvenance;
class SharedLandmarkRefitShadow;
class SharedLandmarkShadowSession;

using LandmarkId = std::uint64_t;
using ObservationId = std::uint64_t;

struct ObservationSnapshot;
struct MapRevision;

class MultiViewLandmarkStoreIdentity
{
public:
  MultiViewLandmarkStoreIdentity() = default;

  bool valid() const noexcept;
  bool sameAuthority(
    const MultiViewLandmarkStoreIdentity & other)
    const noexcept;

private:
  explicit MultiViewLandmarkStoreIdentity(
    std::shared_ptr<const void> authority);

  static MultiViewLandmarkStoreIdentity issue();

  std::shared_ptr<const void> authority_;

  friend class MultiViewLandmarkStore;
  friend class AcceptedObservationStoreReceipt;
};

// Immutable historical proof created only by a successful atomic store
// commit. It owns a complete ObservationSnapshot value copy: the patch
// pyramid is deep-copied, and every snapshot value remains bound to the
// original runtime-frame provenance ownership. This is neither a digest nor
// a live lease over the landmark's current state.
class AcceptedObservationStoreReceipt
{
public:
  AcceptedObservationStoreReceipt(
    const AcceptedObservationStoreReceipt &) = default;
  AcceptedObservationStoreReceipt(
    AcceptedObservationStoreReceipt &&) = default;
  AcceptedObservationStoreReceipt & operator=(
    const AcceptedObservationStoreReceipt &) = default;
  AcceptedObservationStoreReceipt & operator=(
    AcceptedObservationStoreReceipt &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const MultiViewLandmarkStoreIdentity & identity)
    const noexcept;
  // Diagnostic-only payload comparison. It does not establish which
  // landmark a caller expected to authorize.
  bool matchesObservationPayload(
    const ObservationSnapshot & observation) const noexcept;
  bool accepts(
    LandmarkId expected_landmark_id,
    const ObservationSnapshot & observation) const noexcept;

  MapRevision storeRevision() const noexcept;
  LandmarkId landmarkId() const noexcept;
  ObservationId observationId() const noexcept;
  std::uint64_t observationRevision() const noexcept;
  std::uint64_t acceptedCameraBatchSequence() const noexcept;
  GeometryVersion samplingGeometryVersion() const noexcept;

private:
  struct Payload;
  explicit AcceptedObservationStoreReceipt(
    std::shared_ptr<const Payload> payload);

  std::shared_ptr<const Payload> payload_;

  friend class MultiViewLandmarkStore;
};

struct BatchId
{
  std::uint64_t value{0U};
};

struct MapRevision
{
  std::uint64_t value{0U};
};

struct GeometrySnapshotId
{
  std::uint64_t value{0U};
};

struct StateRevision
{
  std::uint64_t value{0U};
};

struct LioPosteriorRevision
{
  std::uint64_t value{0U};
};

struct CommitRevisionSet
{
  StateRevision state;
  std::uint64_t rig_calibration_revision{0U};
  std::array<std::uint64_t, 4U> camera_model_revisions{
    {0U, 0U, 0U, 0U}};
  std::array<std::uint64_t, 4U>
    photometric_calibration_revisions{{0U, 0U, 0U, 0U}};
  std::uint64_t time_transfer_revision{0U};

  bool valid() const;
};

bool operator==(
  const CommitRevisionSet & lhs,
  const CommitRevisionSet & rhs);

enum class LandmarkLifecycle
{
  kLidarSeeded,
  kSameCameraTrackable,
  kMigrationShadowEligible,
  kMigrationActiveEligible,
  kTombstoned
};

enum class MapCommitKind
{
  kLidarAnchoredBootstrapAccepted,
  kLidarPlaneTombstoneAccepted,
  kLidarGeometryRefitAccepted,
  kVisualUpdateAccepted
};

enum class ObservationWriteKind
{
  kBootstrapReference,
  kAcceptedSameCamera,
  kAcceptedCrossCamera
};

enum class ObservationPhotometricMode
{
  kInvalid,
  kSameCameraOnly,
  kCrossCameraCalibrated
};

enum class CommitStatus
{
  kCommitted,
  kStaleStoreRevision,
  kStaleGeometry,
  kProofMismatch,
  kInvariantViolation,
  kNonFinite
};

struct CommitRevisionScope
{
  bool visual_revisions_required{false};
  std::uint8_t camera_mask{0U};

  bool valid() const;
  bool containsCamera(std::uint8_t camera_id) const;
};

struct GeometrySeed
{
  GeometrySupportKey support_key;
  GeometryVersion geometry_version;
  ActivePlaneIdentityLease active_plane_lease;
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d position_covariance_m2{Eigen::Matrix3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 6, 6> plane_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  PlaneLocalBasis local_basis;

  // The camera is provenance only. Identity never contains camera_id.
  std::uint8_t camera_id{0U};
};

struct PlaneTombstoneMutation
{
  PlaneIdentity old_identity;
  std::uint64_t last_revision{0U};
  std::uint64_t generation{0U};
};

struct GeometryRefitMutation
{
  GeometrySupportKey support_key;
  GeometryVersion expected_geometry_version;
  GeometryVersion replacement_geometry_version;
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d position_covariance_m2{
    Eigen::Matrix3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 6, 6> plane_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  PlaneLocalBasis local_basis;
};

struct ObservationSnapshot
{
  ObservationId observation_id{0U};
  std::uint64_t observation_revision{0U};
  std::uint8_t camera_id{0U};
  double exposure_timestamp_sec{0.0};
  Eigen::Matrix3d rotation_world_from_imu{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  std::uint64_t rig_calibration_revision{0U};
  std::uint64_t camera_model_revision{0U};
  Eigen::Vector2d pixel{Eigen::Vector2d::Zero()};
  int level{0};
  std::vector<std::vector<float>> owned_patch_pyramid;
  ObservationPhotometricMode photometric_mode{
    ObservationPhotometricMode::kInvalid};
  PhotometricStatus photometric_status{
    PhotometricStatus::kInvalidInput};
  bool photometric_cross_camera_comparable{false};
  bool photometric_experimental{false};
  std::uint64_t photometric_calibration_revision{0U};
  std::uint64_t time_transfer_revision{0U};
  BatchId accepted_batch_id;
  std::uint64_t accepted_camera_batch_sequence{0U};
  std::uint64_t pose_runtime_publication_sequence{0U};
  std::uint64_t pose_active_state_snapshot_revision{0U};
  GeometryVersion sampling_geometry_version;
  std::shared_ptr<const ReferenceOnlyMigrationFrameProvenance>
    frame_provenance;
  Eigen::Matrix<double, 6, 6> pose_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  bool reference_only{true};
  std::optional<AcceptedObservationStoreReceipt>
    store_receipt;
};

struct LandmarkSnapshot
{
  LandmarkId landmark_id{0U};
  GeometrySupportKey support_key;
  GeometryVersion geometry_version;
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d position_covariance_m2{Eigen::Matrix3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 6, 6> plane_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  PlaneLocalBasis local_basis;
  LandmarkLifecycle lifecycle{LandmarkLifecycle::kLidarSeeded};
  MapRevision tombstoned_at_store_revision;
  std::vector<ObservationSnapshot> observations;
};

struct ReadSnapshot
{
  MultiViewLandmarkStoreIdentity store_identity;
  MapRevision store_revision;
  GeometrySnapshotId geometry;
  ActivePlaneIdentitySnapshot active_planes;
  CommitRevisionSet expected_revisions;
  LandmarkId next_landmark_id{1U};
  ObservationId next_observation_id{1U};
  std::vector<LandmarkSnapshot> landmarks;

  const LandmarkSnapshot * find(LandmarkId landmark_id) const;
};

struct ObservationCandidate
{
  GeometrySupportKey support_key;
  GeometryVersion geometry_version;
  ActivePlaneIdentityLease active_plane_lease;
  std::uint8_t camera_id{0U};
  double exposure_timestamp_sec{0.0};
  Eigen::Matrix3d rotation_world_from_imu{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  std::uint64_t rig_calibration_revision{0U};
  std::uint64_t camera_model_revision{0U};
  Eigen::Vector2d pixel{Eigen::Vector2d::Zero()};
  int level{0};
  std::vector<std::vector<float>> owned_patch_pyramid;
  ObservationPhotometricMode photometric_mode{
    ObservationPhotometricMode::kInvalid};
  PhotometricStatus photometric_status{
    PhotometricStatus::kInvalidInput};
  bool photometric_cross_camera_comparable{false};
  bool photometric_experimental{false};
  std::uint64_t photometric_calibration_revision{0U};
  std::uint64_t time_transfer_revision{0U};
  std::shared_ptr<const ReferenceOnlyMigrationFrameProvenance>
    frame_provenance;
  Eigen::Matrix<double, 6, 6> pose_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  ObservationWriteKind kind{
    ObservationWriteKind::kBootstrapReference};
};

struct LandmarkLifecycleTransition
{
  LandmarkId landmark_id{0U};
  GeometryVersion expected_geometry_version;
  LandmarkLifecycle target{
    LandmarkLifecycle::kLidarSeeded};
  bool capability_gate_passed{false};
};

enum class ReferenceKind
{
  kSameCameraHistory,
  kCrossCameraShadow,
  kCrossCameraActive
};

struct LandmarkReferenceRequest
{
  LandmarkId landmark_id{0U};
  std::uint8_t target_camera_id{0U};
  std::uint64_t target_rig_calibration_revision{0U};
  std::uint64_t target_camera_model_revision{0U};
  std::uint64_t target_photometric_calibration_revision{0U};
  std::uint64_t target_time_transfer_revision{0U};
  double target_timestamp_sec{0.0};
  double maximum_reference_age_sec{0.0};
  double selection_score{0.0};
  bool target_camera_healthy{false};
  bool allow_migration_shadow{false};
  bool allow_migration_active{false};
};

struct SelectedPatch
{
  LandmarkId landmark_id{0U};
  ObservationId source_observation_id{0U};
  std::uint8_t source_camera_id{0U};
  std::uint8_t target_camera_id{0U};
  ReferenceKind kind{ReferenceKind::kSameCameraHistory};
  double score{0.0};
};

struct LandmarkQuery
{
  BatchId batch;
  GeometrySnapshotId geometry;
  ActivePlaneIdentitySnapshot active_planes;
  LioPosteriorRevision expected_lio_posterior_revision;
  CommitRevisionSet expected_revisions;
  std::vector<GeometrySeed> geometry_seeds;
  std::vector<GeometryRefitMutation> geometry_refits;
  std::vector<PlaneTombstoneMutation> plane_tombstones;
  std::vector<ObservationCandidate> observations;
  std::vector<LandmarkLifecycleTransition> lifecycle_transitions;
  std::vector<LandmarkReferenceRequest> reference_requests;
};

struct ResolvedLandmark
{
  std::size_t query_index{0U};
  LandmarkId landmark_id{0U};
  GeometrySupportKey support_key;
};

class MapCommitProof
{
public:
  bool valid() const;

private:
  MapCommitProof() = default;
  friend class BootstrapGate;
  friend class LidarTombstoneGate;
  friend class GeometryRefitGate;
  friend class JointVisualTransaction;
  friend class MultiViewLandmarkStore;

  BatchId batch_;
  MapRevision expected_store_revision_;
  GeometrySnapshotId geometry_;
  StateRevision anchor_state_;
  LioPosteriorRevision lio_posterior_revision_;
  CommitRevisionSet expected_revisions_;
  CommitRevisionScope revision_scope_;
  std::uint64_t active_plane_registry_instance_{0U};
  std::uint64_t active_plane_registry_revision_{0U};
  std::uint64_t active_plane_lease_digest_{0U};
  std::size_t active_plane_lease_count_{0U};
  MapCommitKind kind_{
    MapCommitKind::kLidarAnchoredBootstrapAccepted};
  bool valid_{false};
};

struct BootstrapGateEvidence
{
  BatchId batch;
  GeometrySnapshotId geometry;
  CommitRevisionSet expected_revisions;
  bool lidar_plane_stable{false};
  bool camera_model_valid{false};
  bool time_transfer_valid{false};
  bool in_field_of_view{false};
  bool texture_valid{false};
};

class LidarAnchoredCommitToken
{
public:
  LidarAnchoredCommitToken(
    const LidarAnchoredCommitToken &) = default;
  LidarAnchoredCommitToken & operator=(
    const LidarAnchoredCommitToken &) = default;

  GeometrySnapshotId geometry() const;
  const CommitRevisionSet & expectedRevisions() const;

private:
  friend class LegacyLandmarkShadowAdapter;
  friend class SharedLandmarkShadowSession;
  LidarAnchoredCommitToken(
    GeometrySnapshotId geometry,
    CommitRevisionSet expected_revisions);

  GeometrySnapshotId geometry_;
  CommitRevisionSet expected_revisions_;
};

class BootstrapGate
{
public:
  static MapCommitProof mint(
    const LidarAnchoredCommitToken & token,
    const PreparedLandmarkBatch & prepared);
};

class LidarTombstoneGate
{
public:
  static MapCommitProof mint(
    const LidarAnchoredCommitToken & token,
    const PreparedLandmarkBatch & prepared);
};

class LidarGeometryRefitCommitToken
{
public:
  LidarGeometryRefitCommitToken(
    const LidarGeometryRefitCommitToken &) = default;
  LidarGeometryRefitCommitToken & operator=(
    const LidarGeometryRefitCommitToken &) = default;

  GeometrySnapshotId geometry() const;
  LioPosteriorRevision lioPosteriorRevision() const;

private:
  friend class SharedLandmarkRefitShadow;
  friend class SharedLandmarkShadowSession;
  LidarGeometryRefitCommitToken(
    GeometrySnapshotId geometry,
    LioPosteriorRevision lio_posterior_revision);

  GeometrySnapshotId geometry_;
  LioPosteriorRevision lio_posterior_revision_;
};

class GeometryRefitGate
{
public:
  static MapCommitProof mint(
    const LidarGeometryRefitCommitToken & token,
    const PreparedLandmarkBatch & prepared);
};

struct JointVisualCommitEvidence
{
  BatchId batch;
  GeometrySnapshotId geometry;
  CommitRevisionSet expected_revisions;
  bool update_accepted{false};
  bool state_committed{false};
  bool posterior_finite{false};
  bool covariance_psd{false};
};

class AcceptedStateCommitToken
{
public:
  AcceptedStateCommitToken(
    const AcceptedStateCommitToken &) = default;
  AcceptedStateCommitToken & operator=(
    const AcceptedStateCommitToken &) = default;

  GeometrySnapshotId geometry() const;
  const CommitRevisionSet & expectedRevisions() const;

private:
  friend class ::VIOManager;
  friend class SharedLandmarkShadowSession;
  AcceptedStateCommitToken(
    GeometrySnapshotId geometry,
    CommitRevisionSet expected_revisions);

  GeometrySnapshotId geometry_;
  CommitRevisionSet expected_revisions_;
};

class JointVisualTransaction
{
public:
  static MapCommitProof mint(
    const AcceptedStateCommitToken & token,
    const PreparedLandmarkBatch & prepared);
};

class StagedLandmarkDelta
{
public:
  StagedLandmarkDelta() = default;

private:
  friend class MultiViewLandmarkStore;
  friend class PreparedLandmarkBatch;
  friend class BootstrapGate;
  friend class LidarTombstoneGate;
  friend class GeometryRefitGate;
  friend class JointVisualTransaction;

  struct GeometryWrite
  {
    LandmarkId landmark_id{0U};
    GeometrySeed seed;
  };

  struct ObservationWrite
  {
    LandmarkId landmark_id{0U};
    ObservationId observation_id{0U};
    ObservationCandidate observation;
  };

  struct LifecycleWrite
  {
    LandmarkLifecycleTransition transition;
  };

  struct PlaneTombstoneWrite
  {
    PlaneTombstoneMutation mutation;
    std::vector<LandmarkId> landmark_ids;
  };

  struct GeometryRefitWrite
  {
    LandmarkId landmark_id{0U};
    GeometryRefitMutation mutation;
  };

  BatchId batch_;
  MapRevision base_store_revision_;
  GeometrySnapshotId geometry_;
  LioPosteriorRevision expected_lio_posterior_revision_;
  CommitRevisionSet expected_revisions_;
  CommitRevisionScope revision_scope_;
  std::uint64_t active_plane_registry_instance_{0U};
  std::uint64_t active_plane_registry_revision_{0U};
  std::uint64_t active_plane_lease_digest_{0U};
  std::size_t active_plane_lease_count_{0U};
  MapCommitKind required_kind_{
    MapCommitKind::kLidarAnchoredBootstrapAccepted};
  std::vector<GeometryWrite> geometry_writes_;
  std::vector<GeometryRefitWrite> geometry_refit_writes_;
  std::vector<PlaneTombstoneWrite> plane_tombstone_writes_;
  std::vector<ObservationWrite> observation_writes_;
  std::vector<LifecycleWrite> lifecycle_writes_;
  bool valid_{false};
};

class PreparedLandmarkBatch
{
public:
  bool valid() const;
  CommitRevisionScope revisionScope() const;
  StagedLandmarkDelta take_delta();

  std::vector<ResolvedLandmark> resolved_landmarks;
  std::vector<SelectedPatch> selected_patches;

private:
  friend class MultiViewLandmarkStore;
  friend class BootstrapGate;
  friend class LidarTombstoneGate;
  friend class GeometryRefitGate;
  friend class JointVisualTransaction;

  StagedLandmarkDelta delta_;
  bool valid_{false};
};

struct CommitResult
{
  CommitStatus status{CommitStatus::kInvariantViolation};
  MapRevision store_revision;
  std::size_t geometry_writes{0U};
  std::size_t geometry_refits{0U};
  std::size_t tombstoned_landmarks{0U};
  std::size_t purged_tombstones{0U};
};

struct MultiViewLandmarkStoreConfig
{
  std::size_t maximum_observations_per_landmark_per_camera{
    64U};
  std::size_t maximum_tombstoned_landmarks{4096U};

  bool valid() const;
};

class MultiViewLandmarkStore
{
public:
  explicit MultiViewLandmarkStore(
    MultiViewLandmarkStoreConfig config = {});

  ReadSnapshot snapshot(
    const GeometrySnapshotId & geometry,
    const CommitRevisionSet & expected_revisions) const;
  ReadSnapshot snapshot(
    const GeometrySnapshotId & geometry,
    const CommitRevisionSet & expected_revisions,
    const ActivePlaneIdentitySnapshot & active_planes) const;
  PreparedLandmarkBatch prepare(
    const ReadSnapshot & snapshot,
    const LandmarkQuery & query) const;
  CommitResult commit(
    StagedLandmarkDelta && delta,
    const MapCommitProof & proof);

private:
  mutable std::shared_mutex mutex_;
  MultiViewLandmarkStoreConfig config_;
  MultiViewLandmarkStoreIdentity identity_;
  std::map<GeometrySupportKey, LandmarkSnapshot> landmarks_;
  MapRevision store_revision_;
  LandmarkId next_landmark_id_{1U};
  ObservationId next_observation_id_{1U};
};

const char * commitStatusName(CommitStatus status);

}  // namespace livo

#endif  // LIVO_MULTI_VIEW_LANDMARK_H_
