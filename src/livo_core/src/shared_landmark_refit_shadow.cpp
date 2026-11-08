#include "shared_landmark_refit_shadow.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include <Eigen/Eigenvalues>

namespace livo
{
namespace
{

template<typename MatrixType>
bool finiteSymmetricPsdCovariance(
  const MatrixType & covariance)
{
  if (!covariance.allFinite())
  {
    return false;
  }
  const double scale =
    std::max(1.0, covariance.norm());
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

bool validEvent(const PlaneRefitEvent & event)
{
  const double normal_norm = event.normal_world.norm();
  return
    event.expected_geometry_version.valid() &&
    event.replacement_geometry_version.valid() &&
    event.expected_geometry_version.plane ==
      event.replacement_geometry_version.plane &&
    event.replacement_geometry_version.revision >
      event.expected_geometry_version.revision &&
    event.generation != 0U &&
    event.center_world_m.allFinite() &&
    finiteSymmetricPsdCovariance(
      event.position_covariance_m2) &&
    event.normal_world.allFinite() &&
    std::isfinite(normal_norm) &&
    normal_norm > 1.0e-9 &&
    finiteSymmetricPsdCovariance(
      event.plane_covariance) &&
    event.local_basis.valid();
}

GeometryRefitMutation mutationFor(
  const LandmarkSnapshot & landmark,
  const PlaneRefitEvent & event)
{
  const Eigen::Vector3d normal =
    event.normal_world.normalized();
  GeometryRefitMutation mutation;
  mutation.support_key = landmark.support_key;
  mutation.expected_geometry_version =
    landmark.geometry_version;
  mutation.replacement_geometry_version =
    event.replacement_geometry_version;
  mutation.position_world_m =
    landmark.position_world_m -
    normal * normal.dot(
      landmark.position_world_m -
      event.center_world_m);
  mutation.position_covariance_m2 =
    landmark.position_covariance_m2 +
    event.position_covariance_m2;
  mutation.normal_world = normal;
  mutation.plane_covariance = event.plane_covariance;
  mutation.local_basis = event.local_basis;
  return mutation;
}

}  // namespace

SharedLandmarkRefitShadow::SharedLandmarkRefitShadow(
  SharedLandmarkRefitShadowConfig config)
: config_(std::move(config))
{
}

SharedLandmarkRefitShadowResult
SharedLandmarkRefitShadow::apply(
  MultiViewLandmarkStore & store,
  const std::vector<PlaneRefitEvent> & events,
  const LidarGeometryRefitCommitToken * token,
  const BatchId batch) const
{
  SharedLandmarkRefitShadowResult result;
  result.input_event_count = events.size();
  if (!config_.enabled)
  {
    result.status = SharedLandmarkRefitShadowStatus::kDisabled;
    return result;
  }
  if (events.empty())
  {
    result.status = SharedLandmarkRefitShadowStatus::kNoEvents;
    result.acknowledge_events = true;
    return result;
  }
  if (token == nullptr)
  {
    result.status =
      SharedLandmarkRefitShadowStatus::kMissingLidarToken;
    return result;
  }
  if (
    batch.value == 0U ||
    token->geometry().value == 0U ||
    token->lioPosteriorRevision().value == 0U)
  {
    result.status =
      SharedLandmarkRefitShadowStatus::kInvalidInput;
    return result;
  }

  const CommitRevisionSet no_visual_revisions;
  const ReadSnapshot snapshot = store.snapshot(
    token->geometry(), no_visual_revisions);
  LandmarkQuery query;
  query.batch = batch;
  query.geometry = token->geometry();
  query.expected_lio_posterior_revision =
    token->lioPosteriorRevision();
  query.expected_revisions = no_visual_revisions;
  std::set<std::pair<std::uint64_t, std::uint32_t>>
    staged_identities;
  for (const PlaneRefitEvent & event : events)
  {
    if (!validEvent(event))
    {
      result.status =
        SharedLandmarkRefitShadowStatus::kInvalidInput;
      return result;
    }
    const auto identity = std::make_pair(
      event.expected_geometry_version.plane.id,
      event.expected_geometry_version.plane.epoch);
    if (!staged_identities.insert(identity).second)
    {
      ++result.ignored_event_count;
      continue;
    }

    bool matched = false;
    for (const LandmarkSnapshot & landmark :
         snapshot.landmarks)
    {
      if (
        landmark.lifecycle != LandmarkLifecycle::kTombstoned &&
        landmark.support_key.plane.id ==
          event.expected_geometry_version.plane.id &&
        landmark.support_key.plane.epoch !=
          event.expected_geometry_version.plane.epoch)
      {
        result.status =
          SharedLandmarkRefitShadowStatus::kStaleEvent;
        return result;
      }
      if (
        !(landmark.support_key.plane ==
          event.expected_geometry_version.plane) ||
        landmark.lifecycle == LandmarkLifecycle::kTombstoned)
      {
        continue;
      }
      if (
        landmark.geometry_version.revision >
        event.replacement_geometry_version.revision)
      {
        result.status =
          SharedLandmarkRefitShadowStatus::kStaleEvent;
        return result;
      }
      if (
        landmark.geometry_version.revision ==
        event.replacement_geometry_version.revision)
      {
        continue;
      }
      query.geometry_refits.push_back(
        mutationFor(landmark, event));
      ++result.matched_landmark_count;
      matched = true;
    }
    if (!matched)
    {
      ++result.ignored_event_count;
    }
  }

  if (query.geometry_refits.empty())
  {
    result.status =
      SharedLandmarkRefitShadowStatus::kNoMatchingLandmarks;
    result.acknowledge_events = true;
    return result;
  }

  PreparedLandmarkBatch prepared =
    store.prepare(snapshot, query);
  if (!prepared.valid())
  {
    result.status =
      SharedLandmarkRefitShadowStatus::kPrepareRejected;
    return result;
  }
  const MapCommitProof proof =
    GeometryRefitGate::mint(*token, prepared);
  const CommitResult committed =
    store.commit(prepared.take_delta(), proof);
  result.commit_status = committed.status;
  result.geometry_refits = committed.geometry_refits;
  if (committed.status != CommitStatus::kCommitted)
  {
    result.status =
      SharedLandmarkRefitShadowStatus::kCommitRejected;
    return result;
  }

  result.status = SharedLandmarkRefitShadowStatus::kCommitted;
  result.acknowledge_events = true;
  return result;
}

const char * sharedLandmarkRefitShadowStatusName(
  const SharedLandmarkRefitShadowStatus status)
{
  switch (status)
  {
    case SharedLandmarkRefitShadowStatus::kDisabled:
      return "disabled";
    case SharedLandmarkRefitShadowStatus::kNoEvents:
      return "no_events";
    case SharedLandmarkRefitShadowStatus::kMissingLidarToken:
      return "missing_lidar_token";
    case SharedLandmarkRefitShadowStatus::kInvalidInput:
      return "invalid_input";
    case SharedLandmarkRefitShadowStatus::kStaleEvent:
      return "stale_event";
    case SharedLandmarkRefitShadowStatus::kNoMatchingLandmarks:
      return "no_matching_landmarks";
    case SharedLandmarkRefitShadowStatus::kPrepareRejected:
      return "prepare_rejected";
    case SharedLandmarkRefitShadowStatus::kCommitted:
      return "committed";
    case SharedLandmarkRefitShadowStatus::kCommitRejected:
      return "commit_rejected";
  }
  return "unknown";
}

}  // namespace livo
