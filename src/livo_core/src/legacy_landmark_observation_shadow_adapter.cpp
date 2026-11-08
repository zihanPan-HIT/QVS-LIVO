#include "legacy_landmark_observation_shadow_adapter.h"

#include "reference_only_migration_runtime_provenance.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace livo
{
namespace
{

bool validRotation(const Eigen::Matrix3d & rotation)
{
  return
    rotation.allFinite() &&
    std::abs(rotation.determinant() - 1.0) <= 1.0e-6 &&
    (rotation.transpose() * rotation -
     Eigen::Matrix3d::Identity()).norm() <= 1.0e-6;
}

bool validOwnedPatch(
  const std::vector<std::vector<float>> & pyramid,
  const LegacyLandmarkObservationShadowConfig & config)
{
  if (
    pyramid.empty() ||
    pyramid.size() > config.maximum_patch_levels)
  {
    return false;
  }
  std::size_t value_count = 0U;
  for (const std::vector<float> & patch_level : pyramid)
  {
    if (patch_level.empty())
    {
      return false;
    }
    if (
      patch_level.size() >
      config.maximum_patch_values_per_observation - value_count)
    {
      return false;
    }
    value_count += patch_level.size();
    if (!std::all_of(
        patch_level.begin(), patch_level.end(),
        [](const float value) {
          return std::isfinite(value);
        }))
    {
      return false;
    }
  }
  return value_count > 0U;
}

bool validSample(
  const LegacyLandmarkObservationShadowSample & sample,
  const LegacyLandmarkObservationShadowConfig & config)
{
  const Eigen::Matrix<double, 6, 6> symmetric_covariance =
    0.5 * (
      sample.pose_covariance +
      sample.pose_covariance.transpose());
  Eigen::SelfAdjointEigenSolver<
    Eigen::Matrix<double, 6, 6>> covariance_solver(
      symmetric_covariance, Eigen::EigenvaluesOnly);
  const bool pose_covariance_valid =
    sample.pose_covariance.allFinite() &&
    (sample.pose_covariance -
      sample.pose_covariance.transpose()).norm() <= 1.0e-9 &&
    covariance_solver.info() == Eigen::Success &&
    covariance_solver.eigenvalues().minCoeff() >= -1.0e-12 &&
    sample.pose_covariance.trace() > 0.0;
  const bool photometry_usable =
    sample.photometric_status == PhotometricStatus::kValid &&
    sample.photometric_calibration_revision != 0U &&
    ((sample.photometric_mode ==
        ObservationPhotometricMode::kSameCameraOnly &&
      !sample.photometric_cross_camera_comparable) ||
     (sample.photometric_mode ==
        ObservationPhotometricMode::kCrossCameraCalibrated &&
      sample.photometric_cross_camera_comparable &&
      !sample.photometric_experimental));
  const bool runtime_frame_valid =
    sample.frame_provenance != nullptr &&
    sample.frame_provenance->valid() &&
    sample.frame_provenance->cameraId() ==
      sample.camera_id &&
    sample.frame_provenance->exposureTimestampSec() ==
      sample.exposure_timestamp_sec &&
    sample.frame_provenance->cameraBatchSequence() != 0U &&
    sample.frame_provenance->runtimePublicationSequence() !=
      0U &&
    sample.frame_provenance->activeStateSnapshotRevision() !=
      0U &&
    sample.frame_provenance->timeTransferEntry().has_value() &&
    sample.frame_provenance->timeTransferEntry().cameraId() ==
      sample.camera_id &&
    sample.frame_provenance->timeTransferEntry().
      targetTimestampSec() ==
      sample.exposure_timestamp_sec;
  return
    sample.support_key.plane.id != 0U &&
    sample.camera_id < 4U &&
    std::isfinite(sample.exposure_timestamp_sec) &&
    sample.exposure_timestamp_sec > 0.0 &&
    validRotation(sample.rotation_world_from_imu) &&
    sample.position_world_m.allFinite() &&
    sample.rig_calibration_revision != 0U &&
    sample.camera_model_revision != 0U &&
    sample.pixel.allFinite() &&
    sample.level >= 0 &&
    validOwnedPatch(sample.owned_patch_pyramid, config) &&
    photometry_usable &&
    sample.time_status == CameraTimeTransferStatus::kValid &&
    sample.time_transfer_revision != 0U &&
    runtime_frame_valid &&
    pose_covariance_valid &&
    !sample.isolated;
}

bool hasCameraObservation(
  const LandmarkSnapshot & landmark,
  const std::uint8_t camera_id)
{
  return std::any_of(
    landmark.observations.begin(),
    landmark.observations.end(),
    [camera_id](const ObservationSnapshot & observation) {
      return observation.camera_id == camera_id;
    });
}

bool crossCameraShadowWritable(
  const LandmarkSnapshot & landmark)
{
  return
    landmark.lifecycle ==
      LandmarkLifecycle::kMigrationShadowEligible ||
    landmark.lifecycle ==
      LandmarkLifecycle::kMigrationActiveEligible;
}

ObservationCandidate candidateFromSample(
  const LegacyLandmarkObservationShadowSample & sample,
  const GeometryVersion & geometry_version,
  const ObservationWriteKind kind)
{
  ObservationCandidate candidate;
  candidate.support_key = sample.support_key;
  candidate.geometry_version = geometry_version;
  candidate.active_plane_lease =
    sample.active_plane_lease;
  candidate.camera_id = sample.camera_id;
  candidate.exposure_timestamp_sec =
    sample.exposure_timestamp_sec;
  candidate.rotation_world_from_imu =
    sample.rotation_world_from_imu;
  candidate.position_world_m = sample.position_world_m;
  candidate.rig_calibration_revision =
    sample.rig_calibration_revision;
  candidate.camera_model_revision =
    sample.camera_model_revision;
  candidate.pixel = sample.pixel;
  candidate.level = sample.level;
  candidate.owned_patch_pyramid =
    sample.owned_patch_pyramid;
  candidate.photometric_mode =
    sample.photometric_mode;
  candidate.photometric_status =
    sample.photometric_status;
  candidate.photometric_cross_camera_comparable =
    sample.photometric_cross_camera_comparable;
  candidate.photometric_experimental =
    sample.photometric_experimental;
  candidate.photometric_calibration_revision =
    sample.photometric_calibration_revision;
  candidate.time_transfer_revision =
    sample.time_transfer_revision;
  candidate.frame_provenance =
    sample.frame_provenance;
  candidate.pose_covariance =
    sample.pose_covariance;
  candidate.kind = kind;
  return candidate;
}

}  // namespace

bool StagedLegacyLandmarkObservationShadowBatch::valid() const
{
  return valid_;
}

CommitResult StagedLegacyLandmarkObservationShadowBatch::commit(
  MultiViewLandmarkStore & store)
{
  if (!valid_)
  {
    return CommitResult{};
  }
  if (!proof_.has_value())
  {
    valid_ = false;
    return CommitResult{};
  }
  valid_ = false;
  return store.commit(std::move(delta_), *proof_);
}

LegacyLandmarkObservationShadowAdapter::
LegacyLandmarkObservationShadowAdapter(
  LegacyLandmarkObservationShadowConfig config)
: config_(config)
{
}

bool LegacyLandmarkObservationShadowAdapter::configValid() const
{
  return
    config_.maximum_observations_per_batch > 0U &&
    config_.maximum_patch_levels > 0U &&
    config_.maximum_patch_values_per_observation > 0U;
}

LegacyLandmarkObservationShadowStageResult
LegacyLandmarkObservationShadowAdapter::stage(
  const MultiViewLandmarkStore & store,
  const LegacyLandmarkObservationShadowRequest & request) const
{
  LegacyLandmarkObservationShadowStageResult result;
  result.input_sample_count = request.samples.size();
  if (!config_.enabled)
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::kDisabled;
    return result;
  }
  if (!configValid())
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::
      kInvalidConfiguration;
    return result;
  }
  if (
    request.batch.value == 0U ||
    request.geometry.value == 0U)
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::kInvalidRequest;
    return result;
  }
  if (
    request.samples.size() >
    config_.maximum_observations_per_batch)
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::
      kBatchLimitExceeded;
    return result;
  }

  const bool requests_bootstrap = std::any_of(
    request.samples.begin(), request.samples.end(),
    [](const LegacyLandmarkObservationShadowSample & sample) {
      return sample.is_bootstrap;
    });
  const bool requests_visual = std::any_of(
    request.samples.begin(), request.samples.end(),
    [](const LegacyLandmarkObservationShadowSample & sample) {
      return !sample.is_bootstrap;
    });
  if (requests_bootstrap && requests_visual)
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::
      kMixedCommitKinds;
    return result;
  }

  const CommitRevisionSet * expected_revisions = nullptr;
  if (requests_bootstrap)
  {
    if (
      !request.bootstrap_token.has_value() ||
      !request.active_planes.valid() ||
      request.bootstrap_token->geometry().value !=
        request.geometry.value)
    {
      result.status =
        LegacyLandmarkObservationShadowStatus::
        kProofRejected;
      return result;
    }
    expected_revisions =
      &request.bootstrap_token->expectedRevisions();
  }
  else
  {
    if (
      !request.accepted_state_token.has_value() ||
      request.accepted_state_token->geometry().value !=
        request.geometry.value)
    {
      result.status =
        LegacyLandmarkObservationShadowStatus::
        kProofRejected;
      return result;
    }
    expected_revisions =
      &request.accepted_state_token->expectedRevisions();
  }

  const ReadSnapshot snapshot =
    requests_bootstrap ?
    store.snapshot(
      request.geometry, *expected_revisions,
      request.active_planes) :
    store.snapshot(request.geometry, *expected_revisions);
  LandmarkQuery query;
  query.batch = request.batch;
  query.geometry = request.geometry;
  if (requests_bootstrap)
  {
    query.active_planes = request.active_planes;
  }
  query.expected_revisions = *expected_revisions;
  query.observations.reserve(request.samples.size());

  bool stages_bootstrap = false;
  bool stages_visual = false;
  std::map<GeometrySupportKey, bool> staged_bootstrap_keys;
  for (const LegacyLandmarkObservationShadowSample & sample :
       request.samples)
  {
    if (!validSample(sample, config_))
    {
      continue;
    }
    const auto landmark = std::find_if(
      snapshot.landmarks.begin(), snapshot.landmarks.end(),
      [&sample](const LandmarkSnapshot & candidate) {
        return candidate.support_key == sample.support_key;
      });
    if (landmark == snapshot.landmarks.end())
    {
      continue;
    }

    ObservationWriteKind kind;
    if (sample.is_bootstrap)
    {
      if (
        landmark->lifecycle != LandmarkLifecycle::kLidarSeeded ||
        !landmark->observations.empty() ||
        staged_bootstrap_keys.find(sample.support_key) !=
          staged_bootstrap_keys.end())
      {
        continue;
      }
      kind = ObservationWriteKind::kBootstrapReference;
      stages_bootstrap = true;
      staged_bootstrap_keys.emplace(sample.support_key, true);
    }
    else
    {
      if (
        !sample.visual_update_accepted ||
        landmark->lifecycle == LandmarkLifecycle::kTombstoned)
      {
        continue;
      }
      if (hasCameraObservation(*landmark, sample.camera_id))
      {
        kind = ObservationWriteKind::kAcceptedSameCamera;
      }
      else
      {
        if (!crossCameraShadowWritable(*landmark))
        {
          continue;
        }
        if (
          sample.photometric_mode !=
            ObservationPhotometricMode::
            kCrossCameraCalibrated ||
          !sample.photometric_cross_camera_comparable ||
          sample.photometric_experimental)
        {
          continue;
        }
        kind = ObservationWriteKind::kAcceptedCrossCamera;
      }
      stages_visual = true;
    }

    query.observations.push_back(
      candidateFromSample(
        sample, landmark->geometry_version, kind));
  }

  result.eligible_sample_count = query.observations.size();
  if (query.observations.empty())
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::
      kNoEligibleObservation;
    return result;
  }
  if (stages_bootstrap && stages_visual)
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::
      kMixedCommitKinds;
    return result;
  }

  PreparedLandmarkBatch prepared =
    store.prepare(snapshot, query);
  if (!prepared.valid())
  {
    result.status =
      LegacyLandmarkObservationShadowStatus::
      kPrepareRejected;
    return result;
  }

  std::optional<MapCommitProof> proof;
  if (stages_bootstrap)
  {
    proof = BootstrapGate::mint(
      *request.bootstrap_token,
      prepared);
  }
  else
  {
    proof = JointVisualTransaction::mint(
      *request.accepted_state_token,
      prepared);
  }

  result.staged.delta_ = prepared.take_delta();
  result.staged.proof_ = std::move(proof);
  result.staged.valid_ = true;
  result.staged_observation_count =
    query.observations.size();
  result.status =
    LegacyLandmarkObservationShadowStatus::kStaged;
  return result;
}

const char * legacyLandmarkObservationShadowStatusName(
  const LegacyLandmarkObservationShadowStatus status)
{
  switch (status)
  {
    case LegacyLandmarkObservationShadowStatus::kDisabled:
      return "disabled";
    case LegacyLandmarkObservationShadowStatus::
      kInvalidConfiguration:
      return "invalid_configuration";
    case LegacyLandmarkObservationShadowStatus::kInvalidRequest:
      return "invalid_request";
    case LegacyLandmarkObservationShadowStatus::
      kBatchLimitExceeded:
      return "batch_limit_exceeded";
    case LegacyLandmarkObservationShadowStatus::
      kMixedCommitKinds:
      return "mixed_commit_kinds";
    case LegacyLandmarkObservationShadowStatus::
      kNoEligibleObservation:
      return "no_eligible_observation";
    case LegacyLandmarkObservationShadowStatus::
      kPrepareRejected:
      return "prepare_rejected";
    case LegacyLandmarkObservationShadowStatus::kProofRejected:
      return "proof_rejected";
    case LegacyLandmarkObservationShadowStatus::kStaged:
      return "staged";
  }
  return "unknown";
}

}  // namespace livo
