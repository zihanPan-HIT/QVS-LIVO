#ifndef LIVO_SHARED_LANDMARK_REFIT_SHADOW_H_
#define LIVO_SHARED_LANDMARK_REFIT_SHADOW_H_

#include <cstddef>
#include <vector>

#include "multi_view_landmark.h"
#include "plane_lifecycle.h"

namespace livo
{

struct SharedLandmarkRefitShadowConfig
{
  bool enabled{false};
};

enum class SharedLandmarkRefitShadowStatus
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

struct SharedLandmarkRefitShadowResult
{
  SharedLandmarkRefitShadowStatus status{
    SharedLandmarkRefitShadowStatus::kDisabled};
  CommitStatus commit_status{CommitStatus::kInvariantViolation};
  std::size_t input_event_count{0U};
  std::size_t matched_landmark_count{0U};
  std::size_t ignored_event_count{0U};
  std::size_t geometry_refits{0U};
  bool acknowledge_events{false};
  bool solver_applied{false};
};

class SharedLandmarkRefitShadow
{
public:
  SharedLandmarkRefitShadow() = default;
  explicit SharedLandmarkRefitShadow(
    SharedLandmarkRefitShadowConfig config);

  SharedLandmarkRefitShadowResult apply(
    MultiViewLandmarkStore & store,
    const std::vector<PlaneRefitEvent> & events,
    const LidarGeometryRefitCommitToken * token,
    BatchId batch) const;

private:
  SharedLandmarkRefitShadowConfig config_;
};

const char * sharedLandmarkRefitShadowStatusName(
  SharedLandmarkRefitShadowStatus status);

}  // namespace livo

#endif  // LIVO_SHARED_LANDMARK_REFIT_SHADOW_H_
