#ifndef LIVO_SHARED_LANDMARK_TOMBSTONE_SHADOW_H_
#define LIVO_SHARED_LANDMARK_TOMBSTONE_SHADOW_H_

#include <cstddef>
#include <vector>

#include "multi_view_landmark.h"
#include "plane_lifecycle.h"

namespace livo
{

struct SharedLandmarkTombstoneShadowConfig
{
  bool enabled{false};
};

enum class SharedLandmarkTombstoneShadowStatus
{
  kDisabled,
  kNoEvents,
  kMissingLidarToken,
  kInvalidInput,
  kStaleEvent,
  kNoMatchingLandmarks,
  kPrepareRejected,
  kCommitted,
  kCommitRejected
};

struct SharedLandmarkTombstoneShadowResult
{
  SharedLandmarkTombstoneShadowStatus status{
    SharedLandmarkTombstoneShadowStatus::kDisabled};
  CommitStatus commit_status{CommitStatus::kInvariantViolation};
  std::size_t input_event_count{0U};
  std::size_t matched_event_count{0U};
  std::size_t ignored_event_count{0U};
  std::size_t tombstoned_landmarks{0U};
  bool acknowledge_events{false};
  bool solver_applied{false};
};

class SharedLandmarkTombstoneShadowConsumer
{
public:
  SharedLandmarkTombstoneShadowConsumer() = default;
  explicit SharedLandmarkTombstoneShadowConsumer(
    SharedLandmarkTombstoneShadowConfig config);

  SharedLandmarkTombstoneShadowResult apply(
    MultiViewLandmarkStore & store,
    const std::vector<PlaneTombstoneEvent> & events,
    const LidarAnchoredCommitToken * token,
    BatchId batch) const;

private:
  SharedLandmarkTombstoneShadowConfig config_;
};

const char * sharedLandmarkTombstoneShadowStatusName(
  SharedLandmarkTombstoneShadowStatus status);

}  // namespace livo

#endif  // LIVO_SHARED_LANDMARK_TOMBSTONE_SHADOW_H_
