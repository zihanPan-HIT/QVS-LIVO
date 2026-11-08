#ifndef LIVO_PLANE_LIFECYCLE_H_
#define LIVO_PLANE_LIFECYCLE_H_

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "geometry_identity.h"

namespace livo
{

struct PlaneReplacementGateConfig
{
  double maximum_normal_change_rad{0.35};
  double maximum_center_shift_m{0.50};
  double maximum_planarity_change{0.05};

  bool valid() const;
};

struct PlaneGeometryEstimate
{
  bool stable{false};
  Eigen::Vector3d center_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Vector3d principal_axis_world{Eigen::Vector3d::Zero()};
  double planarity{0.0};
};

enum class PlaneTombstoneReason
{
  kMajorReplacement,
  kPlaneInvalidated,
  kVoxelDeleted,
  kRetentionExpired
};

struct PlaneTombstoneEvent
{
  PlaneIdentity old_identity;
  std::uint64_t last_revision{0U};
  std::uint64_t generation{0U};
  PlaneTombstoneReason reason{PlaneTombstoneReason::kPlaneInvalidated};
};

class PlaneTombstoneEventQueue
{
public:
  bool record(const PlaneTombstoneEvent & event);
  std::vector<PlaneTombstoneEvent> snapshot() const;
  std::vector<PlaneTombstoneEvent> consume();
  void retirePlane(std::uint64_t plane_id);
  std::size_t deduplicationWatermarkCount() const;

private:
  using IdentityKey = std::pair<std::uint64_t, std::uint32_t>;

  mutable std::mutex mutex_;
  std::map<std::uint64_t, std::uint32_t>
    highest_tombstoned_epoch_by_plane_;
  std::set<IdentityKey> pending_identities_;
  std::vector<PlaneTombstoneEvent> pending_;
};

struct PlaneRefitEvent
{
  GeometryVersion expected_geometry_version;
  GeometryVersion replacement_geometry_version;
  std::uint64_t generation{0U};
  Eigen::Vector3d center_world_m{Eigen::Vector3d::Zero()};
  // Additive world-frame landmark-position uncertainty introduced by
  // projecting a landmark onto this refitted plane, in square metres.
  Eigen::Matrix3d position_covariance_m2{
    Eigen::Matrix3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  Eigen::Matrix<double, 6, 6> plane_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  PlaneLocalBasis local_basis;
};

struct PlaneRefitEventBatch
{
  std::uint64_t delivery_token{0U};
  std::uint64_t queue_generation{0U};
  std::uint64_t event_count{0U};
  std::array<std::uint64_t, 4U> payload_digest{};
  std::vector<PlaneRefitEvent> events;

  bool valid() const;
};

class PlaneRefitEventQueue
{
public:
  bool record(const PlaneRefitEvent & event);
  PlaneRefitEventBatch snapshot() const;
  bool acknowledge(const PlaneRefitEventBatch & applied_batch);
  void retirePlane(PlaneIdentity identity);
  std::size_t pendingEventCount() const;

private:
  using IdentityKey = std::pair<std::uint64_t, std::uint32_t>;

  mutable std::mutex mutex_;
  std::map<IdentityKey, PlaneRefitEvent> pending_;
  mutable std::optional<PlaneRefitEventBatch> active_delivery_;
  mutable std::uint64_t next_delivery_token_{1U};
  std::uint64_t queue_generation_{1U};
};

struct PlaneLifecycleSnapshot
{
  bool stable{false};
  PlaneIdentity identity;
  std::uint64_t revision{0U};
  Eigen::Vector3d identity_origin_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d identity_u_axis_world{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d identity_v_axis_world{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d center_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal_world{Eigen::Vector3d::Zero()};
  double planarity{0.0};
};

enum class PlaneLifecycleTransition
{
  kNoChange,
  kInitialized,
  kMinorRefit,
  kMajorReplacement,
  kInvalidated
};

struct PlaneLifecycleUpdate
{
  PlaneLifecycleTransition transition{PlaneLifecycleTransition::kNoChange};
  PlaneLifecycleSnapshot snapshot;
  std::optional<PlaneTombstoneEvent> tombstone;
};

PlaneLifecycleUpdate advancePlaneLifecycle(
  const PlaneLifecycleSnapshot & previous,
  const PlaneGeometryEstimate & candidate,
  std::uint64_t new_plane_id,
  std::uint64_t generation,
  const PlaneReplacementGateConfig & config);

}  // namespace livo

#endif  // LIVO_PLANE_LIFECYCLE_H_
