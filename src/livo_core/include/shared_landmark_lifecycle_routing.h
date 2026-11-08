#ifndef LIVO_SHARED_LANDMARK_LIFECYCLE_ROUTING_H_
#define LIVO_SHARED_LANDMARK_LIFECYCLE_ROUTING_H_

namespace livo
{

struct SharedLandmarkLifecycleRouting
{
  bool geometry_shadow_enabled{false};
  bool tombstone_shadow_enabled{false};
  bool refit_shadow_enabled{false};

  bool activePlaneLifecycleRequired() const
  {
    return
      geometry_shadow_enabled ||
      tombstone_shadow_enabled ||
      refit_shadow_enabled;
  }

  bool activePlaneSnapshotRequired() const
  {
    return geometry_shadow_enabled;
  }
};

}  // namespace livo

#endif  // LIVO_SHARED_LANDMARK_LIFECYCLE_ROUTING_H_
