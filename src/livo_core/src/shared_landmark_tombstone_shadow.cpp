#include "shared_landmark_tombstone_shadow.h"

#include <set>
#include <utility>

namespace livo
{

SharedLandmarkTombstoneShadowConsumer::
SharedLandmarkTombstoneShadowConsumer(
  SharedLandmarkTombstoneShadowConfig config)
: config_(std::move(config))
{
}

SharedLandmarkTombstoneShadowResult
SharedLandmarkTombstoneShadowConsumer::apply(
  MultiViewLandmarkStore & store,
  const std::vector<PlaneTombstoneEvent> & events,
  const LidarAnchoredCommitToken * token,
  const BatchId batch) const
{
  SharedLandmarkTombstoneShadowResult result;
  result.input_event_count = events.size();
  if (!config_.enabled)
  {
    result.status =
      SharedLandmarkTombstoneShadowStatus::kDisabled;
    return result;
  }
  if (events.empty())
  {
    result.status =
      SharedLandmarkTombstoneShadowStatus::kNoEvents;
    result.acknowledge_events = true;
    return result;
  }
  if (token == nullptr)
  {
    result.status =
      SharedLandmarkTombstoneShadowStatus::kMissingLidarToken;
    return result;
  }
  if (
    batch.value == 0U ||
    token->geometry().value == 0U)
  {
    result.status =
      SharedLandmarkTombstoneShadowStatus::kInvalidInput;
    return result;
  }

  const ReadSnapshot snapshot = store.snapshot(
    token->geometry(), token->expectedRevisions());
  LandmarkQuery query;
  query.batch = batch;
  query.geometry = token->geometry();
  query.expected_revisions = token->expectedRevisions();
  std::set<std::pair<std::uint64_t, std::uint32_t>>
    staged_identities;
  for (const PlaneTombstoneEvent & event : events)
  {
    if (
      event.old_identity.id == 0U ||
      event.old_identity.epoch == 0U ||
      event.last_revision == 0U ||
      event.generation == 0U)
    {
      result.status =
        SharedLandmarkTombstoneShadowStatus::kInvalidInput;
      return result;
    }

    const auto identity = std::make_pair(
      event.old_identity.id, event.old_identity.epoch);
    if (!staged_identities.insert(identity).second)
    {
      ++result.ignored_event_count;
      continue;
    }

    bool matched = false;
    for (const LandmarkSnapshot & landmark : snapshot.landmarks)
    {
      if (
        !(landmark.support_key.plane == event.old_identity) ||
        landmark.lifecycle == LandmarkLifecycle::kTombstoned)
      {
        continue;
      }
      if (
        landmark.geometry_version.revision == 0U ||
        landmark.geometry_version.revision > event.last_revision)
      {
        result.status =
          SharedLandmarkTombstoneShadowStatus::kStaleEvent;
        return result;
      }
      matched = true;
    }
    if (!matched)
    {
      ++result.ignored_event_count;
      continue;
    }

    PlaneTombstoneMutation mutation;
    mutation.old_identity = event.old_identity;
    mutation.last_revision = event.last_revision;
    mutation.generation = event.generation;
    query.plane_tombstones.push_back(mutation);
    ++result.matched_event_count;
  }

  if (query.plane_tombstones.empty())
  {
    result.status =
      SharedLandmarkTombstoneShadowStatus::kNoMatchingLandmarks;
    result.acknowledge_events = true;
    return result;
  }

  PreparedLandmarkBatch prepared = store.prepare(snapshot, query);
  if (!prepared.valid())
  {
    result.status =
      SharedLandmarkTombstoneShadowStatus::kPrepareRejected;
    return result;
  }
  const MapCommitProof proof = LidarTombstoneGate::mint(
    *token, prepared);
  const CommitResult committed =
    store.commit(prepared.take_delta(), proof);
  result.commit_status = committed.status;
  result.tombstoned_landmarks =
    committed.tombstoned_landmarks;
  if (committed.status != CommitStatus::kCommitted)
  {
    result.status =
      SharedLandmarkTombstoneShadowStatus::kCommitRejected;
    return result;
  }

  result.status =
    SharedLandmarkTombstoneShadowStatus::kCommitted;
  result.acknowledge_events = true;
  return result;
}

const char * sharedLandmarkTombstoneShadowStatusName(
  const SharedLandmarkTombstoneShadowStatus status)
{
  switch (status)
  {
    case SharedLandmarkTombstoneShadowStatus::kDisabled:
      return "disabled";
    case SharedLandmarkTombstoneShadowStatus::kNoEvents:
      return "no_events";
    case SharedLandmarkTombstoneShadowStatus::kMissingLidarToken:
      return "missing_lidar_token";
    case SharedLandmarkTombstoneShadowStatus::kInvalidInput:
      return "invalid_input";
    case SharedLandmarkTombstoneShadowStatus::kStaleEvent:
      return "stale_event";
    case SharedLandmarkTombstoneShadowStatus::kNoMatchingLandmarks:
      return "no_matching_landmarks";
    case SharedLandmarkTombstoneShadowStatus::kPrepareRejected:
      return "prepare_rejected";
    case SharedLandmarkTombstoneShadowStatus::kCommitted:
      return "committed";
    case SharedLandmarkTombstoneShadowStatus::kCommitRejected:
      return "commit_rejected";
  }
  return "unknown";
}

}  // namespace livo
