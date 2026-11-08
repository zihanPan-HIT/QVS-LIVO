#include "multi_view_landmark.h"

#include "checked_increment.h"
#include "reference_only_migration_runtime_provenance.h"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <stdexcept>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace livo
{
namespace
{

bool sameGeometrySnapshot(
  const GeometrySnapshotId lhs,
  const GeometrySnapshotId rhs)
{
  return lhs.value == rhs.value;
}

bool sameGeometryVersion(
  const GeometryVersion & lhs,
  const GeometryVersion & rhs)
{
  return
    lhs.plane == rhs.plane &&
    lhs.revision == rhs.revision;
}

void hashLeaseValue(
  std::uint64_t & digest,
  const std::uint64_t value)
{
  constexpr std::uint64_t kFnvPrime =
    1099511628211ULL;
  for (std::size_t byte = 0U;
       byte < sizeof(value); ++byte)
  {
    digest ^=
      (value >> (byte * 8U)) & 0xFFU;
    digest *= kFnvPrime;
  }
}

void hashLease(
  std::uint64_t & digest,
  const GeometrySupportKey & support_key,
  const ActivePlaneIdentityLease & lease)
{
  const GeometryVersion geometry =
    lease.geometryVersion();
  hashLeaseValue(digest, support_key.plane.id);
  hashLeaseValue(digest, support_key.plane.epoch);
  hashLeaseValue(
    digest,
    static_cast<std::uint32_t>(support_key.cell.u));
  hashLeaseValue(
    digest,
    static_cast<std::uint32_t>(support_key.cell.v));
  hashLeaseValue(digest, support_key.cell.level);
  hashLeaseValue(digest, geometry.plane.id);
  hashLeaseValue(digest, geometry.plane.epoch);
  hashLeaseValue(digest, geometry.revision);
  hashLeaseValue(digest, lease.generation());
  hashLeaseValue(digest, lease.registryRevision());
}

CommitRevisionScope revisionScopeFor(
  const LandmarkQuery & query)
{
  CommitRevisionScope scope;
  scope.visual_revisions_required =
    !query.observations.empty() ||
    !query.lifecycle_transitions.empty() ||
    !query.reference_requests.empty();
  for (const ObservationCandidate & observation :
       query.observations)
  {
    if (observation.camera_id < 4U)
    {
      scope.camera_mask |= static_cast<std::uint8_t>(
        1U << observation.camera_id);
    }
  }
  for (const LandmarkReferenceRequest & request :
       query.reference_requests)
  {
    if (request.target_camera_id < 4U)
    {
      scope.camera_mask |= static_cast<std::uint8_t>(
        1U << request.target_camera_id);
    }
  }
  return scope;
}

bool revisionsValidForScope(
  const CommitRevisionSet & revisions,
  const CommitRevisionScope & scope)
{
  if (!scope.valid())
  {
    return false;
  }
  if (!scope.visual_revisions_required)
  {
    return true;
  }
  if (
    revisions.state.value == 0U ||
    revisions.rig_calibration_revision == 0U ||
    revisions.time_transfer_revision == 0U)
  {
    return false;
  }
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    if (
      scope.containsCamera(camera_id) &&
      (revisions.camera_model_revisions[camera_id] == 0U ||
       revisions.photometric_calibration_revisions[camera_id] == 0U))
    {
      return false;
    }
  }
  return true;
}

bool sameRevisionsForScope(
  const CommitRevisionSet & lhs,
  const CommitRevisionSet & rhs,
  const CommitRevisionScope & scope)
{
  if (!scope.visual_revisions_required)
  {
    return true;
  }
  if (
    lhs.state.value != rhs.state.value ||
    lhs.rig_calibration_revision !=
      rhs.rig_calibration_revision ||
    lhs.time_transfer_revision !=
      rhs.time_transfer_revision)
  {
    return false;
  }
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    if (
      scope.containsCamera(camera_id) &&
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

template<typename MatrixType>
bool finitePsdCovariance(const MatrixType & covariance)
{
  if (!covariance.allFinite())
  {
    return false;
  }
  const double scale = std::max(1.0, covariance.norm());
  if (
    (covariance - covariance.transpose()).norm() >
    1.0e-9 * scale)
  {
    return false;
  }
  const MatrixType symmetric =
    0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<MatrixType> solver(
    symmetric, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success)
  {
    return false;
  }
  const double eigenvalue_scale = std::max(
    1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
  return
    solver.eigenvalues().minCoeff() >=
    -1.0e-10 * eigenvalue_scale;
}

bool finiteGeometry(const GeometrySeed & seed)
{
  const double normal_norm = seed.normal_world.norm();
  return
    seed.support_key.valid() &&
    seed.geometry_version.plane == seed.support_key.plane &&
    seed.geometry_version.valid() &&
    seed.position_world_m.allFinite() &&
    finitePsdCovariance(seed.position_covariance_m2) &&
    seed.normal_world.allFinite() &&
    std::isfinite(normal_norm) &&
    normal_norm > 1.0e-9 &&
    finitePsdCovariance(seed.plane_covariance) &&
    seed.local_basis.valid();
}

bool validGeometryRefit(
  const GeometryRefitMutation & mutation)
{
  if (
    !mutation.support_key.valid() ||
    !mutation.expected_geometry_version.valid() ||
    !mutation.replacement_geometry_version.valid() ||
    !(mutation.expected_geometry_version.plane ==
      mutation.support_key.plane) ||
    !(mutation.replacement_geometry_version.plane ==
      mutation.support_key.plane) ||
    mutation.replacement_geometry_version.revision <=
      mutation.expected_geometry_version.revision)
  {
    return false;
  }
  GeometrySeed replacement;
  replacement.support_key = mutation.support_key;
  replacement.geometry_version =
    mutation.replacement_geometry_version;
  replacement.position_world_m = mutation.position_world_m;
  replacement.position_covariance_m2 =
    mutation.position_covariance_m2;
  replacement.normal_world = mutation.normal_world;
  replacement.plane_covariance = mutation.plane_covariance;
  replacement.local_basis = mutation.local_basis;
  return finiteGeometry(replacement);
}

bool validPlaneTombstone(
  const PlaneTombstoneMutation & mutation)
{
  return
    mutation.old_identity.valid() &&
    mutation.last_revision != 0U &&
    mutation.generation != 0U;
}

bool validRotation(const Eigen::Matrix3d & rotation)
{
  return
    rotation.allFinite() &&
    std::abs(rotation.determinant() - 1.0) <= 1.0e-6 &&
    (rotation.transpose() * rotation -
     Eigen::Matrix3d::Identity()).norm() <= 1.0e-6;
}

bool finitePatchPyramid(
  const std::vector<std::vector<float>> & pyramid)
{
  if (pyramid.empty())
  {
    return false;
  }
  for (const std::vector<float> & level : pyramid)
  {
    if (level.empty())
    {
      return false;
    }
    if (!std::all_of(
        level.begin(), level.end(),
        [](const float value) {
          return std::isfinite(value);
        }))
    {
      return false;
    }
  }
  return true;
}

bool finiteObservation(const ObservationCandidate & observation)
{
  const bool pose_covariance_valid =
    finitePsdCovariance(observation.pose_covariance) &&
    observation.pose_covariance.trace() > 0.0;
  const bool photometry_usable =
    observation.photometric_status == PhotometricStatus::kValid &&
    observation.photometric_calibration_revision != 0U &&
    ((observation.photometric_mode ==
        ObservationPhotometricMode::kSameCameraOnly &&
      !observation.photometric_cross_camera_comparable) ||
     (observation.photometric_mode ==
        ObservationPhotometricMode::kCrossCameraCalibrated &&
      observation.photometric_cross_camera_comparable &&
      !observation.photometric_experimental));
  const bool runtime_frame_valid =
    observation.frame_provenance != nullptr &&
    observation.frame_provenance->valid() &&
    observation.frame_provenance->cameraId() ==
      observation.camera_id &&
    observation.frame_provenance->exposureTimestampSec() ==
      observation.exposure_timestamp_sec &&
    observation.frame_provenance->cameraBatchSequence() !=
      0U &&
    observation.frame_provenance->
      runtimePublicationSequence() != 0U &&
    observation.frame_provenance->
      activeStateSnapshotRevision() != 0U &&
    observation.frame_provenance->
      observationGeneration() != 0U &&
    observation.frame_provenance->imageGeneration() != 0U &&
    observation.frame_provenance->
      timeTransferEntry().has_value() &&
    observation.frame_provenance->
      timeTransferEntry().cameraId() ==
      observation.camera_id &&
    observation.frame_provenance->
      timeTransferEntry().targetTimestampSec() ==
      observation.exposure_timestamp_sec;
  return
    observation.support_key.plane.id != 0U &&
    observation.camera_id < 4U &&
    std::isfinite(observation.exposure_timestamp_sec) &&
    observation.exposure_timestamp_sec > 0.0 &&
    validRotation(observation.rotation_world_from_imu) &&
    observation.position_world_m.allFinite() &&
    observation.rig_calibration_revision != 0U &&
    observation.camera_model_revision != 0U &&
    observation.pixel.allFinite() &&
    observation.level >= 0 &&
    finitePatchPyramid(observation.owned_patch_pyramid) &&
    photometry_usable &&
    observation.time_transfer_revision != 0U &&
    runtime_frame_valid &&
    pose_covariance_valid;
}

bool crossCameraPhotometryReady(
  const ObservationCandidate & observation)
{
  return
    observation.photometric_status ==
      PhotometricStatus::kValid &&
    observation.photometric_mode ==
      ObservationPhotometricMode::kCrossCameraCalibrated &&
    observation.photometric_calibration_revision != 0U &&
    observation.photometric_cross_camera_comparable &&
    !observation.photometric_experimental;
}

bool observationAccepted(
  const ObservationSnapshot & observation)
{
  return
    !observation.reference_only &&
    observation.accepted_batch_id.value != 0U;
}

bool observationCapable(
  const ObservationSnapshot & observation)
{
  if (
    observation.photometric_status !=
      PhotometricStatus::kValid ||
    observation.photometric_calibration_revision == 0U ||
    observation.time_transfer_revision == 0U ||
    observation.photometric_experimental)
  {
    return false;
  }
  if (
    observation.photometric_mode ==
      ObservationPhotometricMode::kSameCameraOnly)
  {
    return
      !observation.photometric_cross_camera_comparable;
  }
  return
    observation.photometric_mode ==
      ObservationPhotometricMode::kCrossCameraCalibrated &&
    observation.photometric_cross_camera_comparable;
}

bool retentionPreferred(
  const ObservationSnapshot * lhs,
  const ObservationSnapshot * rhs)
{
  if (observationAccepted(*lhs) != observationAccepted(*rhs))
  {
    return observationAccepted(*lhs);
  }
  if (observationCapable(*lhs) != observationCapable(*rhs))
  {
    return observationCapable(*lhs);
  }
  if (
    lhs->exposure_timestamp_sec !=
    rhs->exposure_timestamp_sec)
  {
    return
      lhs->exposure_timestamp_sec >
      rhs->exposure_timestamp_sec;
  }
  const double lhs_uncertainty =
    lhs->pose_covariance.trace();
  const double rhs_uncertainty =
    rhs->pose_covariance.trace();
  if (lhs_uncertainty != rhs_uncertainty)
  {
    return lhs_uncertainty < rhs_uncertainty;
  }
  return lhs->observation_id > rhs->observation_id;
}

void pruneObservationHistory(
  LandmarkSnapshot & landmark,
  const std::size_t maximum_per_camera)
{
  std::set<ObservationId> retained;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    std::vector<const ObservationSnapshot *> candidates;
    for (const ObservationSnapshot & observation :
         landmark.observations)
    {
      if (observation.camera_id == camera_id)
      {
        candidates.push_back(&observation);
      }
    }
    std::sort(
      candidates.begin(), candidates.end(),
      retentionPreferred);
    const std::size_t retained_count =
      std::min(maximum_per_camera, candidates.size());
    for (std::size_t index = 0U;
         index < retained_count; ++index)
    {
      retained.insert(candidates[index]->observation_id);
    }
  }
  landmark.observations.erase(
    std::remove_if(
      landmark.observations.begin(),
      landmark.observations.end(),
      [&retained](const ObservationSnapshot & observation) {
        return
          retained.find(observation.observation_id) ==
          retained.end();
      }),
    landmark.observations.end());
  std::sort(
    landmark.observations.begin(),
    landmark.observations.end(),
    [](const ObservationSnapshot & lhs,
       const ObservationSnapshot & rhs) {
      return lhs.observation_id < rhs.observation_id;
    });
}

std::size_t purgeExcessTombstones(
  std::map<GeometrySupportKey, LandmarkSnapshot> & landmarks,
  const std::size_t maximum_tombstones)
{
  struct Candidate
  {
    GeometrySupportKey support_key;
    MapRevision tombstoned_at;
    LandmarkId landmark_id{0U};
  };
  std::vector<Candidate> candidates;
  for (const auto & entry : landmarks)
  {
    if (
      entry.second.lifecycle ==
      LandmarkLifecycle::kTombstoned)
    {
      candidates.push_back(Candidate{
        entry.first,
        entry.second.tombstoned_at_store_revision,
        entry.second.landmark_id});
    }
  }
  if (candidates.size() <= maximum_tombstones)
  {
    return 0U;
  }
  std::sort(
    candidates.begin(), candidates.end(),
    [](const Candidate & lhs, const Candidate & rhs) {
      if (
        lhs.tombstoned_at.value !=
        rhs.tombstoned_at.value)
      {
        return
          lhs.tombstoned_at.value <
          rhs.tombstoned_at.value;
      }
      return lhs.landmark_id < rhs.landmark_id;
    });
  const std::size_t purge_count =
    candidates.size() - maximum_tombstones;
  for (std::size_t index = 0U;
       index < purge_count; ++index)
  {
    landmarks.erase(candidates[index].support_key);
  }
  return purge_count;
}

bool lifecycleAtLeast(
  const LandmarkLifecycle lifecycle,
  const LandmarkLifecycle required)
{
  if (lifecycle == LandmarkLifecycle::kTombstoned)
  {
    return false;
  }
  return
    static_cast<int>(lifecycle) >=
    static_cast<int>(required);
}

LandmarkSnapshot snapshotFromSeed(
  const LandmarkId landmark_id,
  const GeometrySeed & seed)
{
  LandmarkSnapshot result;
  result.landmark_id = landmark_id;
  result.support_key = seed.support_key;
  result.geometry_version = seed.geometry_version;
  result.position_world_m = seed.position_world_m;
  result.position_covariance_m2 =
    seed.position_covariance_m2;
  result.normal_world = seed.normal_world.normalized();
  result.plane_covariance = seed.plane_covariance;
  result.local_basis = seed.local_basis;
  result.lifecycle = LandmarkLifecycle::kLidarSeeded;
  return result;
}

ObservationSnapshot snapshotFromObservation(
  const ObservationId observation_id,
  const ObservationCandidate & observation,
  const BatchId accepted_batch,
  const std::uint64_t observation_revision)
{
  ObservationSnapshot result;
  result.observation_id = observation_id;
  result.observation_revision = observation_revision;
  result.camera_id = observation.camera_id;
  result.exposure_timestamp_sec =
    observation.exposure_timestamp_sec;
  result.rotation_world_from_imu =
    observation.rotation_world_from_imu;
  result.position_world_m =
    observation.position_world_m;
  result.rig_calibration_revision =
    observation.rig_calibration_revision;
  result.camera_model_revision =
    observation.camera_model_revision;
  result.pixel = observation.pixel;
  result.level = observation.level;
  result.owned_patch_pyramid =
    observation.owned_patch_pyramid;
  result.photometric_mode =
    observation.photometric_mode;
  result.photometric_status =
    observation.photometric_status;
  result.photometric_cross_camera_comparable =
    observation.photometric_cross_camera_comparable;
  result.photometric_experimental =
    observation.photometric_experimental;
  result.photometric_calibration_revision =
    observation.photometric_calibration_revision;
  result.time_transfer_revision =
    observation.time_transfer_revision;
  result.frame_provenance =
    observation.frame_provenance;
  result.accepted_camera_batch_sequence =
    observation.frame_provenance->cameraBatchSequence();
  result.pose_runtime_publication_sequence =
    observation.frame_provenance->
    runtimePublicationSequence();
  result.pose_active_state_snapshot_revision =
    observation.frame_provenance->
    activeStateSnapshotRevision();
  result.sampling_geometry_version =
    observation.geometry_version;
  result.pose_covariance =
    observation.pose_covariance;
  result.reference_only =
    observation.kind ==
    ObservationWriteKind::kBootstrapReference;
  if (!result.reference_only)
  {
    result.accepted_batch_id = accepted_batch;
  }
  return result;
}

bool referenceRequestValid(
  const LandmarkReferenceRequest & request)
{
  return
    request.landmark_id != 0U &&
    request.target_camera_id < 4U &&
    request.target_rig_calibration_revision != 0U &&
    request.target_camera_model_revision != 0U &&
    request.target_photometric_calibration_revision != 0U &&
    request.target_time_transfer_revision != 0U &&
    std::isfinite(request.target_timestamp_sec) &&
    request.target_timestamp_sec > 0.0 &&
    std::isfinite(request.maximum_reference_age_sec) &&
    request.maximum_reference_age_sec >= 0.0 &&
    std::isfinite(request.selection_score);
}

const ObservationSnapshot * newestEligibleObservation(
  const LandmarkSnapshot & landmark,
  const LandmarkReferenceRequest & request,
  const BatchId current_batch,
  const CommitRevisionSet & expected_revisions,
  const bool same_camera)
{
  const ObservationSnapshot * best = nullptr;
  for (const ObservationSnapshot & observation :
       landmark.observations)
  {
    if (observation.camera_id >=
        expected_revisions.camera_model_revisions.size())
    {
      continue;
    }
    if (
      (observation.camera_id == request.target_camera_id) !=
        same_camera)
    {
      continue;
    }
    const std::size_t source_camera_id =
      static_cast<std::size_t>(observation.camera_id);
    if (
      observation.rig_calibration_revision !=
        expected_revisions.rig_calibration_revision ||
      observation.camera_model_revision !=
        expected_revisions.
          camera_model_revisions[source_camera_id] ||
      observation.photometric_calibration_revision !=
        expected_revisions.
          photometric_calibration_revisions[source_camera_id] ||
      observation.time_transfer_revision !=
        expected_revisions.time_transfer_revision ||
      (!observation.reference_only &&
       (observation.accepted_batch_id.value == 0U ||
        observation.accepted_batch_id.value >=
          current_batch.value)))
    {
      continue;
    }
    const double age =
      request.target_timestamp_sec -
      observation.exposure_timestamp_sec;
    if (
      !std::isfinite(age) ||
      age < 0.0 ||
      age > request.maximum_reference_age_sec)
    {
      continue;
    }
    if (!same_camera)
    {
      if (
        observation.reference_only ||
        observation.accepted_batch_id.value == 0U ||
        observation.photometric_status !=
          PhotometricStatus::kValid ||
        observation.photometric_mode !=
          ObservationPhotometricMode::kCrossCameraCalibrated ||
        observation.photometric_calibration_revision == 0U ||
        !observation.photometric_cross_camera_comparable ||
        observation.photometric_experimental)
      {
        continue;
      }
    }
    if (
      best == nullptr ||
      observation.exposure_timestamp_sec >
        best->exposure_timestamp_sec)
    {
      best = &observation;
    }
  }
  return best;
}

std::vector<SelectedPatch> selectLandmarkPatches(
  const ReadSnapshot & snapshot,
  const BatchId current_batch,
  const CommitRevisionSet & expected_revisions,
  const std::vector<LandmarkReferenceRequest> & requests,
  bool & valid)
{
  valid = true;
  struct Candidate
  {
    SelectedPatch selected;
    double source_timestamp_sec{0.0};
  };
  std::map<LandmarkId, Candidate> best_same_camera;
  std::map<LandmarkId, Candidate> best_cross_camera;

  for (const LandmarkReferenceRequest & request : requests)
  {
    if (!referenceRequestValid(request))
    {
      valid = false;
      return {};
    }
    const std::size_t target_camera_id =
      static_cast<std::size_t>(request.target_camera_id);
    if (
      request.target_rig_calibration_revision !=
        expected_revisions.rig_calibration_revision ||
      request.target_camera_model_revision !=
        expected_revisions.
          camera_model_revisions[target_camera_id] ||
      request.target_photometric_calibration_revision !=
        expected_revisions.
          photometric_calibration_revisions[target_camera_id] ||
      request.target_time_transfer_revision !=
        expected_revisions.time_transfer_revision)
    {
      valid = false;
      return {};
    }
    if (!request.target_camera_healthy)
    {
      continue;
    }
    const LandmarkSnapshot * landmark =
      snapshot.find(request.landmark_id);
    if (
      landmark == nullptr ||
      landmark->lifecycle == LandmarkLifecycle::kTombstoned)
    {
      continue;
    }

    const ObservationSnapshot * source =
      newestEligibleObservation(
        *landmark, request, current_batch,
        expected_revisions, true);
    ReferenceKind kind = ReferenceKind::kSameCameraHistory;
    bool same_camera = source != nullptr;
    if (!same_camera)
    {
      const bool shadow_allowed =
        landmark->lifecycle ==
          LandmarkLifecycle::kMigrationShadowEligible &&
        request.allow_migration_shadow;
      const bool active_allowed =
        landmark->lifecycle ==
          LandmarkLifecycle::kMigrationActiveEligible &&
        request.allow_migration_active;
      if (!shadow_allowed && !active_allowed)
      {
        continue;
      }
      source =
        newestEligibleObservation(
          *landmark, request, current_batch,
          expected_revisions, false);
      if (source == nullptr)
      {
        continue;
      }
      kind =
        active_allowed ?
        ReferenceKind::kCrossCameraActive :
        ReferenceKind::kCrossCameraShadow;
    }

    Candidate candidate;
    candidate.selected.landmark_id = landmark->landmark_id;
    candidate.selected.source_observation_id =
      source->observation_id;
    candidate.selected.source_camera_id = source->camera_id;
    candidate.selected.target_camera_id =
      request.target_camera_id;
    candidate.selected.kind = kind;
    candidate.selected.score = request.selection_score;
    candidate.source_timestamp_sec =
      source->exposure_timestamp_sec;

    auto & candidates =
      same_camera ? best_same_camera : best_cross_camera;
    const auto previous = candidates.find(landmark->landmark_id);
    if (
      previous == candidates.end() ||
      candidate.selected.score > previous->second.selected.score ||
      (candidate.selected.score ==
         previous->second.selected.score &&
       candidate.source_timestamp_sec >
         previous->second.source_timestamp_sec))
    {
      candidates[landmark->landmark_id] = candidate;
    }
  }

  std::vector<SelectedPatch> result;
  result.reserve(
    best_same_camera.size() + best_cross_camera.size());
  std::map<LandmarkId, bool> emitted;
  for (const auto & entry : best_same_camera)
  {
    result.push_back(entry.second.selected);
    emitted.emplace(entry.first, true);
  }
  for (const auto & entry : best_cross_camera)
  {
    if (emitted.find(entry.first) == emitted.end())
    {
      result.push_back(entry.second.selected);
    }
  }
  return result;
}

}  // namespace

namespace
{

struct MultiViewLandmarkStoreAuthorityTag final
{
};

template<typename Left, typename Right>
bool sameSharedOwner(
  const std::shared_ptr<Left> & lhs,
  const std::shared_ptr<Right> & rhs) noexcept
{
  return
    lhs && rhs &&
    !lhs.owner_before(rhs) &&
    !rhs.owner_before(lhs);
}

bool sameObservationPayload(
  const ObservationSnapshot & lhs,
  const ObservationSnapshot & rhs) noexcept
{
  return
    lhs.observation_id == rhs.observation_id &&
    lhs.observation_revision == rhs.observation_revision &&
    lhs.camera_id == rhs.camera_id &&
    lhs.exposure_timestamp_sec ==
      rhs.exposure_timestamp_sec &&
    lhs.rotation_world_from_imu.isApprox(
      rhs.rotation_world_from_imu, 0.0) &&
    lhs.position_world_m.isApprox(
      rhs.position_world_m, 0.0) &&
    lhs.rig_calibration_revision ==
      rhs.rig_calibration_revision &&
    lhs.camera_model_revision ==
      rhs.camera_model_revision &&
    lhs.pixel.isApprox(rhs.pixel, 0.0) &&
    lhs.level == rhs.level &&
    lhs.owned_patch_pyramid == rhs.owned_patch_pyramid &&
    lhs.photometric_mode == rhs.photometric_mode &&
    lhs.photometric_status == rhs.photometric_status &&
    lhs.photometric_cross_camera_comparable ==
      rhs.photometric_cross_camera_comparable &&
    lhs.photometric_experimental ==
      rhs.photometric_experimental &&
    lhs.photometric_calibration_revision ==
      rhs.photometric_calibration_revision &&
    lhs.time_transfer_revision ==
      rhs.time_transfer_revision &&
    lhs.accepted_batch_id.value ==
      rhs.accepted_batch_id.value &&
    lhs.accepted_camera_batch_sequence ==
      rhs.accepted_camera_batch_sequence &&
    lhs.pose_runtime_publication_sequence ==
      rhs.pose_runtime_publication_sequence &&
    lhs.pose_active_state_snapshot_revision ==
      rhs.pose_active_state_snapshot_revision &&
    sameGeometryVersion(
      lhs.sampling_geometry_version,
      rhs.sampling_geometry_version) &&
    sameSharedOwner(
      lhs.frame_provenance,
      rhs.frame_provenance) &&
    lhs.pose_covariance.isApprox(
      rhs.pose_covariance, 0.0) &&
    lhs.reference_only == rhs.reference_only;
}

}  // namespace

struct AcceptedObservationStoreReceipt::Payload
{
  std::shared_ptr<const void> owner;
  MapRevision store_revision;
  LandmarkId landmark_id{0U};
  ObservationSnapshot sealed_observation;
};

MultiViewLandmarkStoreIdentity::
MultiViewLandmarkStoreIdentity(
  std::shared_ptr<const void> authority)
: authority_(std::move(authority))
{
}

MultiViewLandmarkStoreIdentity
MultiViewLandmarkStoreIdentity::issue()
{
  return MultiViewLandmarkStoreIdentity{
    std::make_shared<
    const MultiViewLandmarkStoreAuthorityTag>()};
}

bool MultiViewLandmarkStoreIdentity::valid() const noexcept
{
  return authority_ != nullptr;
}

bool MultiViewLandmarkStoreIdentity::sameAuthority(
  const MultiViewLandmarkStoreIdentity & other)
  const noexcept
{
  return sameSharedOwner(authority_, other.authority_);
}

AcceptedObservationStoreReceipt::
AcceptedObservationStoreReceipt(
  std::shared_ptr<const Payload> payload)
: payload_(std::move(payload))
{
}

bool AcceptedObservationStoreReceipt::valid() const noexcept
{
  return
    payload_ != nullptr &&
    payload_->owner != nullptr &&
    payload_->store_revision.value != 0U &&
    payload_->landmark_id != 0U &&
    payload_->sealed_observation.observation_id != 0U &&
    payload_->sealed_observation.observation_revision ==
      payload_->store_revision.value &&
    payload_->sealed_observation.
      accepted_camera_batch_sequence != 0U &&
    payload_->sealed_observation.
      pose_runtime_publication_sequence != 0U &&
    payload_->sealed_observation.
      pose_active_state_snapshot_revision != 0U &&
    payload_->sealed_observation.
      sampling_geometry_version.valid() &&
    payload_->sealed_observation.frame_provenance !=
      nullptr &&
    payload_->sealed_observation.frame_provenance->valid() &&
    !payload_->sealed_observation.store_receipt.has_value();
}

bool AcceptedObservationStoreReceipt::issuedBy(
  const MultiViewLandmarkStoreIdentity & identity)
  const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(payload_->owner, identity.authority_);
}

bool AcceptedObservationStoreReceipt::matchesObservationPayload(
  const ObservationSnapshot & observation) const noexcept
{
  return
    valid() &&
    sameObservationPayload(
      payload_->sealed_observation, observation);
}

bool AcceptedObservationStoreReceipt::accepts(
  const LandmarkId expected_landmark_id,
  const ObservationSnapshot & observation) const noexcept
{
  return
    valid() &&
    expected_landmark_id != 0U &&
    payload_->landmark_id == expected_landmark_id &&
    sameObservationPayload(
      payload_->sealed_observation, observation);
}

MapRevision AcceptedObservationStoreReceipt::
storeRevision() const noexcept
{
  return valid() ? payload_->store_revision : MapRevision{};
}

LandmarkId AcceptedObservationStoreReceipt::landmarkId()
  const noexcept
{
  return valid() ? payload_->landmark_id : 0U;
}

ObservationId
AcceptedObservationStoreReceipt::observationId()
  const noexcept
{
  return
    valid() ?
    payload_->sealed_observation.observation_id : 0U;
}

std::uint64_t
AcceptedObservationStoreReceipt::observationRevision()
  const noexcept
{
  return
    valid() ?
    payload_->sealed_observation.observation_revision : 0U;
}

std::uint64_t
AcceptedObservationStoreReceipt::
acceptedCameraBatchSequence() const noexcept
{
  return
    valid() ?
    payload_->sealed_observation.
    accepted_camera_batch_sequence : 0U;
}

GeometryVersion
AcceptedObservationStoreReceipt::samplingGeometryVersion()
  const noexcept
{
  return
    valid() ?
    payload_->sealed_observation.
    sampling_geometry_version : GeometryVersion{};
}

bool CommitRevisionSet::valid() const
{
  return
    state.value != 0U &&
    rig_calibration_revision != 0U &&
    time_transfer_revision != 0U &&
    std::all_of(
      camera_model_revisions.begin(),
      camera_model_revisions.end(),
      [](const std::uint64_t revision) {
        return revision != 0U;
      }) &&
    std::all_of(
      photometric_calibration_revisions.begin(),
      photometric_calibration_revisions.end(),
      [](const std::uint64_t revision) {
        return revision != 0U;
      });
}

bool CommitRevisionScope::valid() const
{
  constexpr std::uint8_t kAllCameraBits = 0x0FU;
  return
    (camera_mask & ~kAllCameraBits) == 0U &&
    (visual_revisions_required || camera_mask == 0U);
}

bool CommitRevisionScope::containsCamera(
  const std::uint8_t camera_id) const
{
  return
    camera_id < 4U &&
    (camera_mask &
     static_cast<std::uint8_t>(1U << camera_id)) != 0U;
}

bool operator==(
  const CommitRevisionSet & lhs,
  const CommitRevisionSet & rhs)
{
  return
    lhs.state.value == rhs.state.value &&
    lhs.rig_calibration_revision ==
      rhs.rig_calibration_revision &&
    lhs.camera_model_revisions ==
      rhs.camera_model_revisions &&
    lhs.photometric_calibration_revisions ==
      rhs.photometric_calibration_revisions &&
    lhs.time_transfer_revision ==
      rhs.time_transfer_revision;
}

const LandmarkSnapshot * ReadSnapshot::find(
  const LandmarkId landmark_id) const
{
  const auto found = std::find_if(
    landmarks.begin(), landmarks.end(),
    [landmark_id](const LandmarkSnapshot & landmark) {
      return landmark.landmark_id == landmark_id;
    });
  return found == landmarks.end() ? nullptr : &*found;
}

bool MapCommitProof::valid() const
{
  return valid_;
}

LidarAnchoredCommitToken::LidarAnchoredCommitToken(
  const GeometrySnapshotId geometry,
  CommitRevisionSet expected_revisions)
: geometry_(geometry),
  expected_revisions_(std::move(expected_revisions))
{
}

GeometrySnapshotId LidarAnchoredCommitToken::geometry() const
{
  return geometry_;
}

const CommitRevisionSet &
LidarAnchoredCommitToken::expectedRevisions() const
{
  return expected_revisions_;
}

MapCommitProof BootstrapGate::mint(
  const LidarAnchoredCommitToken & token,
  const PreparedLandmarkBatch & prepared)
{
  MapCommitProof proof;
  if (
    !prepared.delta_.valid_ ||
    prepared.delta_.required_kind_ !=
      MapCommitKind::kLidarAnchoredBootstrapAccepted)
  {
    return proof;
  }
  proof.batch_ = prepared.delta_.batch_;
  proof.expected_store_revision_ =
    prepared.delta_.base_store_revision_;
  proof.geometry_ = token.geometry();
  proof.anchor_state_ = token.expectedRevisions().state;
  proof.expected_revisions_ = token.expectedRevisions();
  proof.revision_scope_ =
    prepared.delta_.revision_scope_;
  proof.active_plane_registry_instance_ =
    prepared.delta_.active_plane_registry_instance_;
  proof.active_plane_registry_revision_ =
    prepared.delta_.active_plane_registry_revision_;
  proof.active_plane_lease_digest_ =
    prepared.delta_.active_plane_lease_digest_;
  proof.active_plane_lease_count_ =
    prepared.delta_.active_plane_lease_count_;
  proof.kind_ =
    MapCommitKind::kLidarAnchoredBootstrapAccepted;
  proof.valid_ =
    proof.active_plane_registry_instance_ != 0U &&
    proof.active_plane_registry_revision_ != 0U &&
    proof.active_plane_lease_digest_ != 0U &&
    proof.active_plane_lease_count_ != 0U;
  return proof;
}

MapCommitProof LidarTombstoneGate::mint(
  const LidarAnchoredCommitToken & token,
  const PreparedLandmarkBatch & prepared)
{
  MapCommitProof proof;
  if (
    !prepared.delta_.valid_ ||
    prepared.delta_.required_kind_ !=
      MapCommitKind::kLidarPlaneTombstoneAccepted)
  {
    return proof;
  }
  proof.batch_ = prepared.delta_.batch_;
  proof.expected_store_revision_ =
    prepared.delta_.base_store_revision_;
  proof.geometry_ = token.geometry();
  proof.anchor_state_ = token.expectedRevisions().state;
  proof.expected_revisions_ = token.expectedRevisions();
  proof.revision_scope_ =
    prepared.delta_.revision_scope_;
  proof.kind_ = MapCommitKind::kLidarPlaneTombstoneAccepted;
  proof.valid_ = true;
  return proof;
}

LidarGeometryRefitCommitToken::
LidarGeometryRefitCommitToken(
  const GeometrySnapshotId geometry,
  const LioPosteriorRevision lio_posterior_revision)
: geometry_(geometry),
  lio_posterior_revision_(lio_posterior_revision)
{
}

GeometrySnapshotId
LidarGeometryRefitCommitToken::geometry() const
{
  return geometry_;
}

LioPosteriorRevision
LidarGeometryRefitCommitToken::lioPosteriorRevision() const
{
  return lio_posterior_revision_;
}

MapCommitProof GeometryRefitGate::mint(
  const LidarGeometryRefitCommitToken & token,
  const PreparedLandmarkBatch & prepared)
{
  MapCommitProof proof;
  if (
    !prepared.delta_.valid_ ||
    prepared.delta_.required_kind_ !=
      MapCommitKind::kLidarGeometryRefitAccepted)
  {
    return proof;
  }
  proof.batch_ = prepared.delta_.batch_;
  proof.expected_store_revision_ =
    prepared.delta_.base_store_revision_;
  proof.geometry_ = token.geometry();
  proof.lio_posterior_revision_ =
    token.lioPosteriorRevision();
  proof.revision_scope_ =
    prepared.delta_.revision_scope_;
  proof.kind_ = MapCommitKind::kLidarGeometryRefitAccepted;
  proof.valid_ = true;
  return proof;
}

AcceptedStateCommitToken::AcceptedStateCommitToken(
  const GeometrySnapshotId geometry,
  CommitRevisionSet expected_revisions)
: geometry_(geometry),
  expected_revisions_(std::move(expected_revisions))
{
}

GeometrySnapshotId AcceptedStateCommitToken::geometry() const
{
  return geometry_;
}

const CommitRevisionSet &
AcceptedStateCommitToken::expectedRevisions() const
{
  return expected_revisions_;
}

MapCommitProof JointVisualTransaction::mint(
  const AcceptedStateCommitToken & token,
  const PreparedLandmarkBatch & prepared)
{
  MapCommitProof proof;
  if (
    !prepared.delta_.valid_ ||
    prepared.delta_.required_kind_ !=
      MapCommitKind::kVisualUpdateAccepted)
  {
    return proof;
  }
  proof.batch_ = prepared.delta_.batch_;
  proof.expected_store_revision_ =
    prepared.delta_.base_store_revision_;
  proof.geometry_ = token.geometry();
  proof.anchor_state_ = token.expectedRevisions().state;
  proof.expected_revisions_ = token.expectedRevisions();
  proof.revision_scope_ =
    prepared.delta_.revision_scope_;
  proof.kind_ = MapCommitKind::kVisualUpdateAccepted;
  proof.valid_ = true;
  return proof;
}

bool PreparedLandmarkBatch::valid() const
{
  return valid_;
}

CommitRevisionScope
PreparedLandmarkBatch::revisionScope() const
{
  return delta_.revision_scope_;
}

StagedLandmarkDelta PreparedLandmarkBatch::take_delta()
{
  valid_ = false;
  return std::move(delta_);
}

bool MultiViewLandmarkStoreConfig::valid() const
{
  return
    maximum_observations_per_landmark_per_camera > 0U &&
    maximum_tombstoned_landmarks > 0U;
}

MultiViewLandmarkStore::MultiViewLandmarkStore(
  MultiViewLandmarkStoreConfig config)
: config_(std::move(config)),
  identity_(MultiViewLandmarkStoreIdentity::issue())
{
  if (!config_.valid())
  {
    throw std::invalid_argument(
      "invalid multi-view landmark store retention config");
  }
}

ReadSnapshot MultiViewLandmarkStore::snapshot(
  const GeometrySnapshotId & geometry,
  const CommitRevisionSet & expected_revisions) const
{
  return snapshot(
    geometry, expected_revisions,
    ActivePlaneIdentitySnapshot{});
}

ReadSnapshot MultiViewLandmarkStore::snapshot(
  const GeometrySnapshotId & geometry,
  const CommitRevisionSet & expected_revisions,
  const ActivePlaneIdentitySnapshot & active_planes) const
{
  const std::shared_lock<std::shared_mutex> lock(mutex_);
  ReadSnapshot result;
  result.store_identity = identity_;
  result.store_revision = store_revision_;
  result.geometry = geometry;
  result.active_planes = active_planes;
  result.expected_revisions = expected_revisions;
  result.next_landmark_id = next_landmark_id_;
  result.next_observation_id = next_observation_id_;
  result.landmarks.reserve(landmarks_.size());
  for (const auto & entry : landmarks_)
  {
    result.landmarks.push_back(entry.second);
  }
  return result;
}

PreparedLandmarkBatch MultiViewLandmarkStore::prepare(
  const ReadSnapshot & snapshot,
  const LandmarkQuery & query) const
{
  const std::shared_lock<std::shared_mutex> lock(mutex_);
  PreparedLandmarkBatch result;
  const bool has_mutation =
    !query.geometry_seeds.empty() ||
    !query.geometry_refits.empty() ||
    !query.plane_tombstones.empty() ||
    !query.observations.empty() ||
    !query.lifecycle_transitions.empty();
  const bool has_query =
    has_mutation || !query.reference_requests.empty();
  const bool has_bootstrap_observation =
    std::any_of(
      query.observations.begin(), query.observations.end(),
      [](const ObservationCandidate & observation) {
        return
          observation.kind ==
          ObservationWriteKind::kBootstrapReference;
      });
  const bool requires_active_plane_authority =
    !query.geometry_seeds.empty() ||
    has_bootstrap_observation;
  const bool active_plane_binding_valid =
    !requires_active_plane_authority ||
    (snapshot.active_planes.valid() &&
     query.active_planes.valid() &&
     snapshot.active_planes.registry_instance_ ==
       query.active_planes.registry_instance_ &&
     snapshot.active_planes.registry_revision_ ==
       query.active_planes.registry_revision_);
  const CommitRevisionScope revision_scope =
    revisionScopeFor(query);
  if (
    !snapshot.store_identity.sameAuthority(identity_) ||
    snapshot.store_revision.value != store_revision_.value ||
    !sameGeometrySnapshot(snapshot.geometry, query.geometry) ||
    query.batch.value == 0U ||
    query.geometry.value == 0U ||
    (query.geometry_refits.empty() &&
     query.expected_lio_posterior_revision.value != 0U) ||
    !revisionsValidForScope(
      snapshot.expected_revisions, revision_scope) ||
    !revisionsValidForScope(
      query.expected_revisions, revision_scope) ||
    !sameRevisionsForScope(
      snapshot.expected_revisions,
      query.expected_revisions,
      revision_scope) ||
    !active_plane_binding_valid ||
    !has_query)
  {
    return result;
  }

  bool selection_valid = false;
  result.selected_patches = selectLandmarkPatches(
    snapshot, query.batch, query.expected_revisions,
    query.reference_requests,
    selection_valid);
  if (!selection_valid)
  {
    return PreparedLandmarkBatch{};
  }
  if (!has_mutation)
  {
    result.valid_ = true;
    return result;
  }

  if (!query.geometry_refits.empty())
  {
    if (
      query.expected_lio_posterior_revision.value == 0U ||
      !query.geometry_seeds.empty() ||
      !query.plane_tombstones.empty() ||
      !query.observations.empty() ||
      !query.lifecycle_transitions.empty() ||
      !query.reference_requests.empty())
    {
      return PreparedLandmarkBatch{};
    }

    std::set<GeometrySupportKey> staged_support_keys;
    result.delta_.geometry_refit_writes_.reserve(
      query.geometry_refits.size());
    for (const GeometryRefitMutation & mutation :
         query.geometry_refits)
    {
      if (
        !validGeometryRefit(mutation) ||
        !staged_support_keys.insert(
          mutation.support_key).second)
      {
        return PreparedLandmarkBatch{};
      }
      const auto landmark = std::find_if(
        snapshot.landmarks.begin(),
        snapshot.landmarks.end(),
        [&mutation](const LandmarkSnapshot & candidate) {
          return
            candidate.support_key ==
            mutation.support_key;
        });
      if (
        landmark == snapshot.landmarks.end() ||
        landmark->lifecycle ==
          LandmarkLifecycle::kTombstoned ||
        !(landmark->geometry_version.plane ==
          mutation.expected_geometry_version.plane) ||
        landmark->geometry_version.revision !=
          mutation.expected_geometry_version.revision)
      {
        return PreparedLandmarkBatch{};
      }

      StagedLandmarkDelta::GeometryRefitWrite write;
      write.landmark_id = landmark->landmark_id;
      write.mutation = mutation;
      result.delta_.geometry_refit_writes_.push_back(
        std::move(write));
    }

    result.delta_.batch_ = query.batch;
    result.delta_.base_store_revision_ =
      snapshot.store_revision;
    result.delta_.geometry_ = query.geometry;
    result.delta_.expected_lio_posterior_revision_ =
      query.expected_lio_posterior_revision;
    result.delta_.expected_revisions_ =
      query.expected_revisions;
    result.delta_.revision_scope_ = revision_scope;
    result.delta_.required_kind_ =
      MapCommitKind::kLidarGeometryRefitAccepted;
    result.delta_.valid_ = true;
    result.valid_ = true;
    return result;
  }

  if (!query.plane_tombstones.empty())
  {
    if (
      !query.geometry_seeds.empty() ||
      !query.geometry_refits.empty() ||
      !query.observations.empty() ||
      !query.lifecycle_transitions.empty() ||
      !query.reference_requests.empty())
    {
      return PreparedLandmarkBatch{};
    }

    std::set<std::pair<std::uint64_t, std::uint32_t>>
      staged_planes;
    result.delta_.plane_tombstone_writes_.reserve(
      query.plane_tombstones.size());
    for (const PlaneTombstoneMutation & mutation :
         query.plane_tombstones)
    {
      const auto identity = std::make_pair(
        mutation.old_identity.id,
        mutation.old_identity.epoch);
      if (
        !validPlaneTombstone(mutation) ||
        !staged_planes.insert(identity).second)
      {
        return PreparedLandmarkBatch{};
      }

      StagedLandmarkDelta::PlaneTombstoneWrite write;
      write.mutation = mutation;
      for (const LandmarkSnapshot & landmark :
           snapshot.landmarks)
      {
        if (!(
            landmark.support_key.plane ==
            mutation.old_identity))
        {
          continue;
        }
        if (landmark.lifecycle == LandmarkLifecycle::kTombstoned)
        {
          continue;
        }
        if (
          landmark.geometry_version.revision == 0U ||
          landmark.geometry_version.revision >
            mutation.last_revision)
        {
          return PreparedLandmarkBatch{};
        }
        write.landmark_ids.push_back(landmark.landmark_id);
      }
      if (write.landmark_ids.empty())
      {
        return PreparedLandmarkBatch{};
      }
      result.delta_.plane_tombstone_writes_.push_back(
        std::move(write));
    }

    result.delta_.batch_ = query.batch;
    result.delta_.base_store_revision_ =
      snapshot.store_revision;
    result.delta_.geometry_ = query.geometry;
    result.delta_.expected_revisions_ =
      query.expected_revisions;
    result.delta_.revision_scope_ = revision_scope;
    result.delta_.required_kind_ =
      MapCommitKind::kLidarPlaneTombstoneAccepted;
    result.delta_.valid_ = true;
    result.valid_ = true;
    return result;
  }

  std::map<GeometrySupportKey, LandmarkId> resolved_ids;
  std::map<GeometrySupportKey, const LandmarkSnapshot *>
    snapshot_by_key;
  for (const LandmarkSnapshot & landmark : snapshot.landmarks)
  {
    resolved_ids.emplace(
      landmark.support_key, landmark.landmark_id);
    snapshot_by_key.emplace(
      landmark.support_key, &landmark);
  }

  LandmarkId next_landmark_id = snapshot.next_landmark_id;
  ObservationId next_observation_id =
    snapshot.next_observation_id;
  result.resolved_landmarks.reserve(
    query.geometry_seeds.size());
  result.delta_.geometry_writes_.reserve(
    query.geometry_seeds.size());
  std::map<GeometrySupportKey, std::size_t> write_by_key;

  for (std::size_t index = 0U;
       index < query.geometry_seeds.size(); ++index)
  {
    const GeometrySeed & seed = query.geometry_seeds[index];
    if (
      !finiteGeometry(seed) ||
      !query.active_planes.contains(
        seed.active_plane_lease) ||
      !sameGeometryVersion(
        seed.active_plane_lease.geometryVersion(),
        seed.geometry_version))
    {
      return PreparedLandmarkBatch{};
    }
    // Bootstrap authority may create a LiDAR-anchored identity, but it may
    // never refresh an existing geometry revision. Geometry refresh requires
    // a separate typed transaction that is intentionally not implemented.
    if (snapshot_by_key.find(seed.support_key) !=
        snapshot_by_key.end())
    {
      return PreparedLandmarkBatch{};
    }

    auto resolved = resolved_ids.find(seed.support_key);
    if (resolved == resolved_ids.end())
    {
      LandmarkId reserved_landmark_id = 0U;
      if (!checkedReserve(
          next_landmark_id, reserved_landmark_id))
      {
        return PreparedLandmarkBatch{};
      }
      resolved = resolved_ids.emplace(
        seed.support_key, reserved_landmark_id).first;
    }

    ResolvedLandmark item;
    item.query_index = index;
    item.landmark_id = resolved->second;
    item.support_key = seed.support_key;
    result.resolved_landmarks.push_back(item);

    const auto existing_write =
      write_by_key.find(seed.support_key);
    if (existing_write == write_by_key.end())
    {
      StagedLandmarkDelta::GeometryWrite write;
      write.landmark_id = resolved->second;
      write.seed = seed;
      write_by_key.emplace(
        seed.support_key,
        result.delta_.geometry_writes_.size());
      result.delta_.geometry_writes_.push_back(
        std::move(write));
    }
    else
    {
      StagedLandmarkDelta::GeometryWrite & write =
        result.delta_.geometry_writes_[existing_write->second];
      if (
        write.seed.geometry_version.revision !=
        seed.geometry_version.revision)
      {
        return PreparedLandmarkBatch{};
      }
    }
  }

  bool requires_visual_proof =
    !query.lifecycle_transitions.empty();
  result.delta_.observation_writes_.reserve(
    query.observations.size());
  for (const ObservationCandidate & candidate :
       query.observations)
  {
    if (!finiteObservation(candidate))
    {
      return PreparedLandmarkBatch{};
    }
    const std::size_t camera_id =
      static_cast<std::size_t>(candidate.camera_id);
    if (
      candidate.rig_calibration_revision !=
        query.expected_revisions.rig_calibration_revision ||
      candidate.camera_model_revision !=
        query.expected_revisions.
          camera_model_revisions[camera_id] ||
      candidate.photometric_calibration_revision !=
        query.expected_revisions.
          photometric_calibration_revisions[camera_id] ||
      candidate.time_transfer_revision !=
        query.expected_revisions.time_transfer_revision)
    {
      return PreparedLandmarkBatch{};
    }

    const auto resolved =
      resolved_ids.find(candidate.support_key);
    if (resolved == resolved_ids.end())
    {
      return PreparedLandmarkBatch{};
    }

    const auto existing =
      snapshot_by_key.find(candidate.support_key);
    const bool is_bootstrap =
      candidate.kind ==
      ObservationWriteKind::kBootstrapReference;
    if (is_bootstrap)
    {
      const bool creates_lidar_seed =
        existing == snapshot_by_key.end() &&
        write_by_key.find(candidate.support_key) !=
          write_by_key.end();
      const bool appends_to_lidar_seed =
        existing != snapshot_by_key.end() &&
        existing->second->lifecycle ==
          LandmarkLifecycle::kLidarSeeded &&
        existing->second->observations.empty() &&
        write_by_key.find(candidate.support_key) ==
          write_by_key.end();
      if (!creates_lidar_seed && !appends_to_lidar_seed)
      {
        return PreparedLandmarkBatch{};
      }
    }
    else
    {
      requires_visual_proof = true;
      if (existing == snapshot_by_key.end())
      {
        return PreparedLandmarkBatch{};
      }
      const LandmarkSnapshot & landmark = *existing->second;
      if (
        candidate.kind ==
        ObservationWriteKind::kAcceptedSameCamera)
      {
        const bool has_same_camera_reference =
          std::any_of(
            landmark.observations.begin(),
            landmark.observations.end(),
            [&candidate](
              const ObservationSnapshot & observation) {
              return
                observation.camera_id ==
                  candidate.camera_id &&
                observation.rig_calibration_revision ==
                  candidate.rig_calibration_revision &&
                observation.camera_model_revision ==
                  candidate.camera_model_revision &&
                observation.photometric_calibration_revision ==
                  candidate.photometric_calibration_revision &&
                observation.time_transfer_revision ==
                  candidate.time_transfer_revision;
            });
        if (!has_same_camera_reference ||
            landmark.lifecycle ==
              LandmarkLifecycle::kTombstoned)
        {
          return PreparedLandmarkBatch{};
        }
      }
      else if (
        candidate.kind ==
        ObservationWriteKind::kAcceptedCrossCamera)
      {
        if (
          !lifecycleAtLeast(
            landmark.lifecycle,
            LandmarkLifecycle::kMigrationShadowEligible) ||
          !crossCameraPhotometryReady(candidate))
        {
          return PreparedLandmarkBatch{};
        }
      }
    }

    ObservationCandidate staged_candidate = candidate;
    if (staged_candidate.geometry_version.revision == 0U)
    {
      if (existing != snapshot_by_key.end())
      {
        staged_candidate.geometry_version =
          existing->second->geometry_version;
      }
      else
      {
        const auto geometry_write =
          write_by_key.find(candidate.support_key);
        staged_candidate.geometry_version =
          result.delta_.geometry_writes_[
            geometry_write->second].seed.geometry_version;
      }
    }
    if (!(
        staged_candidate.geometry_version.plane ==
        candidate.support_key.plane))
    {
      return PreparedLandmarkBatch{};
    }
    if (
      is_bootstrap &&
      (!query.active_planes.contains(
         staged_candidate.active_plane_lease) ||
       !sameGeometryVersion(
         staged_candidate.active_plane_lease.geometryVersion(),
         staged_candidate.geometry_version)))
    {
      return PreparedLandmarkBatch{};
    }

    StagedLandmarkDelta::ObservationWrite write;
    write.landmark_id = resolved->second;
    if (!checkedReserve(
        next_observation_id, write.observation_id))
    {
      return PreparedLandmarkBatch{};
    }
    write.observation = std::move(staged_candidate);
    result.delta_.observation_writes_.push_back(
      std::move(write));
  }

  if (requires_visual_proof &&
      !query.geometry_seeds.empty())
  {
    return PreparedLandmarkBatch{};
  }

  result.delta_.lifecycle_writes_.reserve(
    query.lifecycle_transitions.size());
  for (const LandmarkLifecycleTransition & transition :
       query.lifecycle_transitions)
  {
    const LandmarkSnapshot * landmark =
      snapshot.find(transition.landmark_id);
    if (
      landmark == nullptr ||
      !transition.capability_gate_passed ||
      !(transition.expected_geometry_version.plane ==
        landmark->geometry_version.plane) ||
      transition.expected_geometry_version.revision !=
        landmark->geometry_version.revision)
    {
      return PreparedLandmarkBatch{};
    }
    const bool valid_transition =
      (landmark->lifecycle ==
         LandmarkLifecycle::kSameCameraTrackable &&
       transition.target ==
         LandmarkLifecycle::kMigrationShadowEligible) ||
      (landmark->lifecycle ==
         LandmarkLifecycle::kMigrationShadowEligible &&
       transition.target ==
         LandmarkLifecycle::kMigrationActiveEligible);
    if (!valid_transition)
    {
      return PreparedLandmarkBatch{};
    }
    StagedLandmarkDelta::LifecycleWrite write;
    write.transition = transition;
    result.delta_.lifecycle_writes_.push_back(write);
  }

  result.delta_.batch_ = query.batch;
  result.delta_.base_store_revision_ =
    snapshot.store_revision;
  result.delta_.geometry_ = query.geometry;
  result.delta_.expected_revisions_ =
    query.expected_revisions;
  result.delta_.revision_scope_ = revision_scope;
  result.delta_.required_kind_ =
    requires_visual_proof ?
    MapCommitKind::kVisualUpdateAccepted :
    MapCommitKind::kLidarAnchoredBootstrapAccepted;
  if (
    result.delta_.required_kind_ ==
      MapCommitKind::kLidarAnchoredBootstrapAccepted)
  {
    if (!query.active_planes.valid())
    {
      return PreparedLandmarkBatch{};
    }
    std::uint64_t lease_digest =
      1469598103934665603ULL;
    std::size_t lease_count = 0U;
    for (
      const StagedLandmarkDelta::GeometryWrite & write :
      result.delta_.geometry_writes_)
    {
      hashLease(
        lease_digest, write.seed.support_key,
        write.seed.active_plane_lease);
      ++lease_count;
    }
    for (
      const StagedLandmarkDelta::ObservationWrite & write :
      result.delta_.observation_writes_)
    {
      if (
        write.observation.kind !=
        ObservationWriteKind::kBootstrapReference)
      {
        continue;
      }
      hashLease(
        lease_digest, write.observation.support_key,
        write.observation.active_plane_lease);
      ++lease_count;
    }
    if (lease_count == 0U || lease_digest == 0U)
    {
      return PreparedLandmarkBatch{};
    }
    result.delta_.active_plane_registry_instance_ =
      query.active_planes.registry_instance_;
    result.delta_.active_plane_registry_revision_ =
      query.active_planes.registry_revision_;
    result.delta_.active_plane_lease_digest_ =
      lease_digest;
    result.delta_.active_plane_lease_count_ =
      lease_count;
  }
  result.delta_.valid_ = true;
  result.valid_ = true;
  return result;
}

CommitResult MultiViewLandmarkStore::commit(
  StagedLandmarkDelta && delta,
  const MapCommitProof & proof)
{
  const std::unique_lock<std::shared_mutex> lock(mutex_);
  CommitResult result;
  result.store_revision = store_revision_;
  if (
    !delta.valid_ ||
    !proof.valid_ ||
    proof.kind_ != delta.required_kind_ ||
    proof.batch_.value != delta.batch_.value ||
    proof.expected_store_revision_.value !=
      delta.base_store_revision_.value ||
    !sameGeometrySnapshot(proof.geometry_, delta.geometry_) ||
    proof.revision_scope_.visual_revisions_required !=
      delta.revision_scope_.visual_revisions_required ||
    proof.revision_scope_.camera_mask !=
      delta.revision_scope_.camera_mask ||
    proof.active_plane_registry_instance_ !=
      delta.active_plane_registry_instance_ ||
    proof.active_plane_registry_revision_ !=
      delta.active_plane_registry_revision_ ||
    proof.active_plane_lease_digest_ !=
      delta.active_plane_lease_digest_ ||
    proof.active_plane_lease_count_ !=
      delta.active_plane_lease_count_ ||
    proof.lio_posterior_revision_.value !=
      delta.expected_lio_posterior_revision_.value ||
    !revisionsValidForScope(
      proof.expected_revisions_, delta.revision_scope_) ||
    !sameRevisionsForScope(
      proof.expected_revisions_,
      delta.expected_revisions_,
      delta.revision_scope_))
  {
    result.status = CommitStatus::kProofMismatch;
    return result;
  }
  if (
    delta.base_store_revision_.value !=
    store_revision_.value)
  {
    result.status = CommitStatus::kStaleStoreRevision;
    return result;
  }
  if (
    delta.required_kind_ ==
      MapCommitKind::kLidarAnchoredBootstrapAccepted)
  {
    std::uint64_t lease_digest =
      1469598103934665603ULL;
    std::size_t lease_count = 0U;
    for (
      const StagedLandmarkDelta::GeometryWrite & write :
      delta.geometry_writes_)
    {
      const ActivePlaneIdentityLease & lease =
        write.seed.active_plane_lease;
      if (
        !lease.valid() ||
        lease.registry_instance_ !=
          delta.active_plane_registry_instance_ ||
        lease.registry_revision_ !=
          delta.active_plane_registry_revision_ ||
        !sameGeometryVersion(
          lease.geometryVersion(),
          write.seed.geometry_version))
      {
        result.status = CommitStatus::kStaleGeometry;
        return result;
      }
      hashLease(
        lease_digest, write.seed.support_key, lease);
      ++lease_count;
    }
    for (
      const StagedLandmarkDelta::ObservationWrite & write :
      delta.observation_writes_)
    {
      if (
        write.observation.kind !=
        ObservationWriteKind::kBootstrapReference)
      {
        continue;
      }
      const ActivePlaneIdentityLease & lease =
        write.observation.active_plane_lease;
      if (
        !lease.valid() ||
        lease.registry_instance_ !=
          delta.active_plane_registry_instance_ ||
        lease.registry_revision_ !=
          delta.active_plane_registry_revision_ ||
        !sameGeometryVersion(
          lease.geometryVersion(),
          write.observation.geometry_version))
      {
        result.status = CommitStatus::kStaleGeometry;
        return result;
      }
      hashLease(
        lease_digest, write.observation.support_key, lease);
      ++lease_count;
    }
    if (
      lease_count == 0U ||
      lease_count != delta.active_plane_lease_count_ ||
      lease_digest != delta.active_plane_lease_digest_)
    {
      result.status = CommitStatus::kStaleGeometry;
      return result;
    }
  }
  if (
    delta.geometry_writes_.empty() &&
    delta.geometry_refit_writes_.empty() &&
    delta.plane_tombstone_writes_.empty() &&
    delta.observation_writes_.empty() &&
    delta.lifecycle_writes_.empty())
  {
    result.status = CommitStatus::kInvariantViolation;
    return result;
  }
  MapRevision next_store_revision = store_revision_;
  if (!checkedIncrement(next_store_revision.value))
  {
    result.status = CommitStatus::kInvariantViolation;
    return result;
  }

  auto next_landmarks = landmarks_;
  LandmarkId next_landmark_id = next_landmark_id_;
  ObservationId next_observation_id =
    next_observation_id_;
  std::size_t tombstoned_landmarks = 0U;
  for (
    const StagedLandmarkDelta::GeometryRefitWrite & write :
    delta.geometry_refit_writes_)
  {
    if (
      write.landmark_id == 0U ||
      !validGeometryRefit(write.mutation))
    {
      result.status = CommitStatus::kInvariantViolation;
      return result;
    }
    const auto landmark =
      next_landmarks.find(write.mutation.support_key);
    if (
      landmark == next_landmarks.end() ||
      landmark->second.landmark_id != write.landmark_id ||
      landmark->second.lifecycle ==
        LandmarkLifecycle::kTombstoned ||
      !(landmark->second.geometry_version.plane ==
        write.mutation.expected_geometry_version.plane) ||
      landmark->second.geometry_version.revision !=
        write.mutation.expected_geometry_version.revision)
    {
      result.status = CommitStatus::kStaleGeometry;
      return result;
    }
    landmark->second.geometry_version =
      write.mutation.replacement_geometry_version;
    landmark->second.position_world_m =
      write.mutation.position_world_m;
    landmark->second.position_covariance_m2 =
      write.mutation.position_covariance_m2;
    landmark->second.normal_world =
      write.mutation.normal_world.normalized();
    landmark->second.plane_covariance =
      write.mutation.plane_covariance;
    landmark->second.local_basis =
      write.mutation.local_basis;
  }
  for (
    const StagedLandmarkDelta::PlaneTombstoneWrite & write :
    delta.plane_tombstone_writes_)
  {
    if (
      !validPlaneTombstone(write.mutation) ||
      write.landmark_ids.empty())
    {
      result.status = CommitStatus::kInvariantViolation;
      return result;
    }
    for (const LandmarkId landmark_id : write.landmark_ids)
    {
      auto landmark = std::find_if(
        next_landmarks.begin(), next_landmarks.end(),
        [landmark_id](const auto & entry) {
          return entry.second.landmark_id == landmark_id;
        });
      if (
        landmark == next_landmarks.end() ||
        !(landmark->second.support_key.plane ==
          write.mutation.old_identity) ||
        !(landmark->second.geometry_version.plane ==
          write.mutation.old_identity) ||
        landmark->second.geometry_version.revision == 0U ||
        landmark->second.geometry_version.revision >
          write.mutation.last_revision ||
        landmark->second.lifecycle ==
          LandmarkLifecycle::kTombstoned)
      {
        result.status = CommitStatus::kStaleGeometry;
        return result;
      }
      landmark->second.lifecycle =
        LandmarkLifecycle::kTombstoned;
      landmark->second.tombstoned_at_store_revision =
        next_store_revision;
      ++tombstoned_landmarks;
    }
  }

  for (const StagedLandmarkDelta::GeometryWrite & write :
       delta.geometry_writes_)
  {
    if (!finiteGeometry(write.seed) ||
        write.landmark_id == 0U)
    {
      result.status = CommitStatus::kNonFinite;
      return result;
    }

    const auto existing =
      next_landmarks.find(write.seed.support_key);
    if (existing != next_landmarks.end())
    {
      if (
        existing->second.landmark_id != write.landmark_id ||
        !(existing->second.geometry_version.plane ==
          write.seed.geometry_version.plane) ||
        write.seed.geometry_version.revision <
          existing->second.geometry_version.revision)
      {
        result.status = CommitStatus::kStaleGeometry;
        return result;
      }

      LandmarkSnapshot updated =
        snapshotFromSeed(write.landmark_id, write.seed);
      updated.lifecycle = existing->second.lifecycle;
      updated.tombstoned_at_store_revision =
        existing->second.tombstoned_at_store_revision;
      updated.observations = existing->second.observations;
      existing->second = std::move(updated);
      continue;
    }

    if (write.landmark_id < next_landmark_id)
    {
      result.status = CommitStatus::kInvariantViolation;
      return result;
    }
    LandmarkId landmark_successor = 0U;
    if (!checkedSuccessor(
        write.landmark_id, landmark_successor))
    {
      result.status = CommitStatus::kInvariantViolation;
      return result;
    }
    next_landmarks.emplace(
      write.seed.support_key,
      snapshotFromSeed(write.landmark_id, write.seed));
    next_landmark_id =
      std::max(next_landmark_id, landmark_successor);
  }

  for (const StagedLandmarkDelta::ObservationWrite & write :
       delta.observation_writes_)
  {
    if (
      write.landmark_id == 0U ||
      write.observation_id == 0U ||
      !finiteObservation(write.observation))
    {
      result.status = CommitStatus::kNonFinite;
      return result;
    }

    const auto landmark =
      next_landmarks.find(write.observation.support_key);
    if (
      landmark == next_landmarks.end() ||
      landmark->second.landmark_id != write.landmark_id)
    {
      result.status = CommitStatus::kStaleGeometry;
      return result;
    }
    if (
      !(landmark->second.geometry_version.plane ==
        write.observation.geometry_version.plane) ||
      landmark->second.geometry_version.revision !=
        write.observation.geometry_version.revision)
    {
      result.status = CommitStatus::kStaleGeometry;
      return result;
    }
    if (write.observation_id < next_observation_id)
    {
      result.status = CommitStatus::kInvariantViolation;
      return result;
    }

    switch (write.observation.kind)
    {
      case ObservationWriteKind::kBootstrapReference:
      {
        const auto committed_landmark =
          landmarks_.find(write.observation.support_key);
        const bool creates_lidar_seed =
          committed_landmark == landmarks_.end();
        const bool appends_to_lidar_seed =
          committed_landmark != landmarks_.end() &&
          committed_landmark->second.lifecycle ==
            LandmarkLifecycle::kLidarSeeded &&
          committed_landmark->second.observations.empty();
        if (
          delta.required_kind_ !=
            MapCommitKind::kLidarAnchoredBootstrapAccepted ||
          (!creates_lidar_seed && !appends_to_lidar_seed) ||
          !landmark->second.observations.empty())
        {
          result.status = CommitStatus::kInvariantViolation;
          return result;
        }
        break;
      }
      case ObservationWriteKind::kAcceptedSameCamera:
      {
        if (
          delta.required_kind_ !=
          MapCommitKind::kVisualUpdateAccepted)
        {
          result.status = CommitStatus::kProofMismatch;
          return result;
        }
        const bool has_same_camera_reference =
          std::any_of(
            landmark->second.observations.begin(),
            landmark->second.observations.end(),
            [&write](const ObservationSnapshot & observation) {
              return
                observation.camera_id ==
                  write.observation.camera_id &&
                observation.rig_calibration_revision ==
                  write.observation.rig_calibration_revision &&
                observation.camera_model_revision ==
                  write.observation.camera_model_revision &&
                observation.photometric_calibration_revision ==
                  write.observation.
                    photometric_calibration_revision &&
                observation.time_transfer_revision ==
                  write.observation.time_transfer_revision;
            });
        if (!has_same_camera_reference)
        {
          result.status = CommitStatus::kInvariantViolation;
          return result;
        }
        landmark->second.lifecycle =
          LandmarkLifecycle::kSameCameraTrackable;
        break;
      }
      case ObservationWriteKind::kAcceptedCrossCamera:
      {
        if (
          delta.required_kind_ !=
            MapCommitKind::kVisualUpdateAccepted ||
          !lifecycleAtLeast(
            landmark->second.lifecycle,
            LandmarkLifecycle::kMigrationShadowEligible) ||
          !crossCameraPhotometryReady(write.observation))
        {
          result.status = CommitStatus::kInvariantViolation;
          return result;
        }
        break;
      }
    }

    ObservationSnapshot committed_observation =
      snapshotFromObservation(
      write.observation_id,
      write.observation,
      delta.batch_,
      next_store_revision.value);
    auto receipt_payload =
      std::make_shared<
      AcceptedObservationStoreReceipt::Payload>();
    receipt_payload->owner = identity_.authority_;
    receipt_payload->store_revision =
      next_store_revision;
    receipt_payload->landmark_id =
      write.landmark_id;
    receipt_payload->sealed_observation =
      committed_observation;
    committed_observation.store_receipt =
      AcceptedObservationStoreReceipt{
      std::move(receipt_payload)};
    landmark->second.observations.push_back(
      std::move(committed_observation));
    ObservationId observation_successor = 0U;
    if (!checkedSuccessor(
        write.observation_id, observation_successor))
    {
      result.status = CommitStatus::kInvariantViolation;
      return result;
    }
    next_observation_id =
      std::max(
        next_observation_id,
        observation_successor);
  }

  for (const StagedLandmarkDelta::LifecycleWrite & write :
       delta.lifecycle_writes_)
  {
    auto landmark = std::find_if(
      next_landmarks.begin(), next_landmarks.end(),
      [&write](const auto & entry) {
        return
          entry.second.landmark_id ==
          write.transition.landmark_id;
      });
    if (landmark == next_landmarks.end())
    {
      result.status = CommitStatus::kStaleGeometry;
      return result;
    }
    if (
      !(landmark->second.geometry_version.plane ==
        write.transition.expected_geometry_version.plane) ||
      landmark->second.geometry_version.revision !=
        write.transition.expected_geometry_version.revision ||
      !write.transition.capability_gate_passed)
    {
      result.status = CommitStatus::kStaleGeometry;
      return result;
    }
    const bool valid_transition =
      (landmark->second.lifecycle ==
         LandmarkLifecycle::kSameCameraTrackable &&
       write.transition.target ==
         LandmarkLifecycle::kMigrationShadowEligible) ||
      (landmark->second.lifecycle ==
         LandmarkLifecycle::kMigrationShadowEligible &&
       write.transition.target ==
         LandmarkLifecycle::kMigrationActiveEligible);
    if (!valid_transition)
    {
      result.status = CommitStatus::kInvariantViolation;
      return result;
    }
    landmark->second.lifecycle = write.transition.target;
  }

  for (auto & entry : next_landmarks)
  {
    pruneObservationHistory(
      entry.second,
      config_.
      maximum_observations_per_landmark_per_camera);
  }
  const std::size_t purged_tombstones =
    purgeExcessTombstones(
      next_landmarks,
      config_.maximum_tombstoned_landmarks);

  landmarks_.swap(next_landmarks);
  next_landmark_id_ = next_landmark_id;
  next_observation_id_ = next_observation_id;
  store_revision_ = next_store_revision;
  result.status = CommitStatus::kCommitted;
  result.store_revision = store_revision_;
  result.geometry_writes = delta.geometry_writes_.size();
  result.geometry_refits =
    delta.geometry_refit_writes_.size();
  result.tombstoned_landmarks = tombstoned_landmarks;
  result.purged_tombstones = purged_tombstones;
  delta.valid_ = false;
  return result;
}

const char * commitStatusName(const CommitStatus status)
{
  switch (status)
  {
    case CommitStatus::kCommitted:
      return "committed";
    case CommitStatus::kStaleStoreRevision:
      return "stale_store_revision";
    case CommitStatus::kStaleGeometry:
      return "stale_geometry";
    case CommitStatus::kProofMismatch:
      return "proof_mismatch";
    case CommitStatus::kInvariantViolation:
      return "invariant_violation";
    case CommitStatus::kNonFinite:
      return "non_finite";
  }
  return "unknown";
}

}  // namespace livo
