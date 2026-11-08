#include "plane_lifecycle.h"

#include "checked_increment.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace livo
{
namespace
{

Eigen::Vector3d normalizedOr(
  const Eigen::Vector3d & value,
  const Eigen::Vector3d & fallback)
{
  if (!value.allFinite() || value.squaredNorm() <= 1e-12) {
    return fallback;
  }
  return value.normalized();
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

bool validRefitEvent(const PlaneRefitEvent & event)
{
  return
    event.expected_geometry_version.valid() &&
    event.replacement_geometry_version.valid() &&
    event.expected_geometry_version.plane ==
      event.replacement_geometry_version.plane &&
    event.replacement_geometry_version.revision >
      event.expected_geometry_version.revision &&
    event.generation != 0U &&
    event.center_world_m.allFinite() &&
    finitePsdCovariance(event.position_covariance_m2) &&
    event.normal_world.allFinite() &&
    event.normal_world.squaredNorm() > 1.0e-12 &&
    finitePsdCovariance(event.plane_covariance) &&
    event.local_basis.valid();
}

using RefitPayloadDigest = std::array<std::uint64_t, 4U>;

void mixDigestWord(
  RefitPayloadDigest & digest,
  const std::uint64_t value)
{
  static constexpr std::array<std::uint64_t, 4U> salts{
    0x9e3779b97f4a7c15ULL,
    0xbf58476d1ce4e5b9ULL,
    0x94d049bb133111ebULL,
    0xd6e8feb86659fd93ULL};
  static constexpr std::array<std::uint64_t, 4U> primes{
    1099511628211ULL,
    0x9e3779b185ebca87ULL,
    0xc2b2ae3d27d4eb4fULL,
    0x165667b19e3779f9ULL};
  for (std::size_t lane = 0U; lane < digest.size(); ++lane)
  {
    std::uint64_t mixed = value + salts[lane];
    mixed ^= mixed >> 30U;
    mixed *= 0xbf58476d1ce4e5b9ULL;
    mixed ^= mixed >> 27U;
    digest[lane] ^= mixed;
    digest[lane] *= primes[lane];
    digest[lane] ^= digest[lane] >> 31U;
  }
}

void mixDigestDouble(
  RefitPayloadDigest & digest,
  const double value)
{
  const double canonical_value = value == 0.0 ? 0.0 : value;
  std::uint64_t bits = 0U;
  static_assert(
    sizeof(bits) == sizeof(canonical_value),
    "Plane refit receipt requires 64-bit doubles");
  std::memcpy(&bits, &canonical_value, sizeof(bits));
  mixDigestWord(digest, bits);
}

template<typename Derived>
void mixDigestMatrix(
  RefitPayloadDigest & digest,
  const Eigen::MatrixBase<Derived> & value)
{
  for (Eigen::Index row = 0; row < value.rows(); ++row)
  {
    for (Eigen::Index column = 0; column < value.cols(); ++column)
    {
      mixDigestDouble(digest, value(row, column));
    }
  }
}

RefitPayloadDigest refitPayloadDigest(
  const std::vector<PlaneRefitEvent> & events)
{
  RefitPayloadDigest digest{
    0xcbf29ce484222325ULL,
    0x84222325cbf29ce4ULL,
    0x6eed0e9da4d94a4fULL,
    0xa4093822299f31d0ULL};
  mixDigestWord(
    digest, static_cast<std::uint64_t>(events.size()));
  for (const PlaneRefitEvent & event : events)
  {
    mixDigestWord(
      digest, event.expected_geometry_version.plane.id);
    mixDigestWord(
      digest, event.expected_geometry_version.plane.epoch);
    mixDigestWord(
      digest, event.expected_geometry_version.revision);
    mixDigestWord(
      digest, event.replacement_geometry_version.plane.id);
    mixDigestWord(
      digest, event.replacement_geometry_version.plane.epoch);
    mixDigestWord(
      digest, event.replacement_geometry_version.revision);
    mixDigestWord(digest, event.generation);
    mixDigestMatrix(digest, event.center_world_m);
    mixDigestMatrix(digest, event.position_covariance_m2);
    mixDigestMatrix(digest, event.normal_world);
    mixDigestMatrix(digest, event.plane_covariance);
    mixDigestMatrix(digest, event.local_basis.origin_world_m);
    mixDigestMatrix(digest, event.local_basis.u_axis_world);
    mixDigestMatrix(digest, event.local_basis.v_axis_world);
  }
  return digest;
}

bool constantTimeDigestEqual(
  const RefitPayloadDigest & lhs,
  const RefitPayloadDigest & rhs)
{
  std::uint64_t difference = 0U;
  for (std::size_t lane = 0U; lane < lhs.size(); ++lane)
  {
    difference |= lhs[lane] ^ rhs[lane];
  }
  return difference == 0U;
}

Eigen::Vector3d projectedBasisAxis(
  const Eigen::Vector3d & normal,
  const Eigen::Vector3d & principal_axis)
{
  Eigen::Vector3d axis =
    principal_axis - normal * normal.dot(principal_axis);
  if (!axis.allFinite() || axis.squaredNorm() <= 1e-12) {
    const Eigen::Vector3d fallback =
      std::abs(normal.x()) < std::abs(normal.y()) ?
      Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    axis = fallback - normal * normal.dot(fallback);
  }
  return normalizedOr(axis, Eigen::Vector3d::UnitX());
}

PlaneLifecycleSnapshot initializeSnapshot(
  const PlaneGeometryEstimate & candidate,
  const std::uint64_t new_plane_id)
{
  PlaneLifecycleSnapshot snapshot;
  snapshot.stable = true;
  snapshot.identity.id = new_plane_id;
  snapshot.identity.epoch = 1U;
  snapshot.revision = 1U;
  snapshot.center_world_m = candidate.center_world_m;
  snapshot.normal_world =
    normalizedOr(candidate.normal_world, Eigen::Vector3d::UnitZ());
  snapshot.planarity = candidate.planarity;
  snapshot.identity_origin_world_m = candidate.center_world_m;
  snapshot.identity_u_axis_world =
    projectedBasisAxis(snapshot.normal_world, candidate.principal_axis_world);
  snapshot.identity_v_axis_world =
    normalizedOr(
      snapshot.normal_world.cross(snapshot.identity_u_axis_world),
      Eigen::Vector3d::UnitY());
  return snapshot;
}

bool validStableEstimate(const PlaneGeometryEstimate & candidate)
{
  return
    candidate.stable &&
    candidate.center_world_m.allFinite() &&
    candidate.normal_world.allFinite() &&
    candidate.normal_world.squaredNorm() > 1e-12 &&
    candidate.principal_axis_world.allFinite() &&
    std::isfinite(candidate.planarity);
}

bool majorReplacement(
  const PlaneLifecycleSnapshot & previous,
  const PlaneGeometryEstimate & candidate,
  const PlaneReplacementGateConfig & config)
{
  const Eigen::Vector3d previous_normal =
    normalizedOr(previous.normal_world, Eigen::Vector3d::UnitZ());
  const Eigen::Vector3d candidate_normal =
    normalizedOr(candidate.normal_world, previous_normal);
  const double normal_dot = std::clamp(
    std::abs(previous_normal.dot(candidate_normal)), 0.0, 1.0);
  const double normal_change_rad = std::acos(normal_dot);
  const double center_shift_m =
    (candidate.center_world_m - previous.center_world_m).norm();
  const double planarity_change =
    std::abs(candidate.planarity - previous.planarity);
  return
    normal_change_rad > config.maximum_normal_change_rad ||
    center_shift_m > config.maximum_center_shift_m ||
    planarity_change > config.maximum_planarity_change;
}

PlaneTombstoneEvent makeTombstone(
  const PlaneLifecycleSnapshot & previous,
  const std::uint64_t generation,
  const PlaneTombstoneReason reason)
{
  PlaneTombstoneEvent event;
  event.old_identity = previous.identity;
  event.last_revision = previous.revision;
  event.generation = generation;
  event.reason = reason;
  return event;
}

}  // namespace

bool PlaneReplacementGateConfig::valid() const
{
  const double pi = std::acos(-1.0);
  return
    std::isfinite(maximum_normal_change_rad) &&
    maximum_normal_change_rad >= 0.0 &&
    maximum_normal_change_rad <= pi &&
    std::isfinite(maximum_center_shift_m) &&
    maximum_center_shift_m >= 0.0 &&
    std::isfinite(maximum_planarity_change) &&
    maximum_planarity_change >= 0.0;
}

PlaneLifecycleUpdate advancePlaneLifecycle(
  const PlaneLifecycleSnapshot & previous,
  const PlaneGeometryEstimate & candidate,
  const std::uint64_t new_plane_id,
  const std::uint64_t generation,
  const PlaneReplacementGateConfig & config)
{
  PlaneLifecycleUpdate update;
  update.snapshot = previous;
  if (!config.valid() || generation == 0U)
  {
    return update;
  }
  if (
    previous.stable &&
    (!previous.identity.valid() || previous.revision == 0U))
  {
    update.snapshot = PlaneLifecycleSnapshot{};
    return update;
  }
  if (!validStableEstimate(candidate)) {
    if (previous.stable) {
      update.transition = PlaneLifecycleTransition::kInvalidated;
      update.snapshot.stable = false;
      update.tombstone = makeTombstone(
        previous, generation, PlaneTombstoneReason::kPlaneInvalidated);
    }
    return update;
  }

  if (!previous.stable) {
    const bool has_previous_identity =
      previous.identity.valid();
    if (!has_previous_identity && new_plane_id == 0U)
    {
      return update;
    }
    std::uint32_t replacement_epoch = 1U;
    if (
      has_previous_identity &&
      !checkedSuccessor(
        previous.identity.epoch,
        replacement_epoch))
    {
      return update;
    }
    update.transition = PlaneLifecycleTransition::kInitialized;
    update.snapshot = initializeSnapshot(
      candidate,
      has_previous_identity ? previous.identity.id : new_plane_id);
    if (has_previous_identity) {
      update.transition = PlaneLifecycleTransition::kMajorReplacement;
      update.snapshot.identity.epoch = replacement_epoch;
    }
    return update;
  }

  if (majorReplacement(previous, candidate, config)) {
    std::uint32_t replacement_epoch = 0U;
    if (!checkedSuccessor(
        previous.identity.epoch, replacement_epoch))
    {
      return update;
    }
    update.transition = PlaneLifecycleTransition::kMajorReplacement;
    update.snapshot = initializeSnapshot(candidate, previous.identity.id);
    update.snapshot.identity.epoch = replacement_epoch;
    update.tombstone = makeTombstone(
      previous, generation, PlaneTombstoneReason::kMajorReplacement);
    return update;
  }

  std::uint64_t replacement_revision = 0U;
  if (!checkedSuccessor(
      previous.revision, replacement_revision))
  {
    return update;
  }
  update.transition = PlaneLifecycleTransition::kMinorRefit;
  update.snapshot.stable = true;
  update.snapshot.revision = replacement_revision;
  update.snapshot.center_world_m = candidate.center_world_m;
  update.snapshot.normal_world = normalizedOr(
    candidate.normal_world, previous.normal_world);
  if (update.snapshot.normal_world.dot(previous.normal_world) < 0.0) {
    update.snapshot.normal_world = -update.snapshot.normal_world;
  }
  update.snapshot.planarity = candidate.planarity;
  return update;
}

bool PlaneTombstoneEventQueue::record(
  const PlaneTombstoneEvent & event)
{
  if (
    !event.old_identity.valid() ||
    event.last_revision == 0U ||
    event.generation == 0U)
  {
    return false;
  }

  const IdentityKey identity{
    event.old_identity.id, event.old_identity.epoch};
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending_identities_.count(identity) != 0U) {
    return false;
  }
  const auto watermark =
    highest_tombstoned_epoch_by_plane_.find(event.old_identity.id);
  if (
    watermark != highest_tombstoned_epoch_by_plane_.end() &&
    event.old_identity.epoch <= watermark->second)
  {
    return false;
  }
  highest_tombstoned_epoch_by_plane_[event.old_identity.id] =
    event.old_identity.epoch;
  pending_identities_.insert(identity);
  pending_.push_back(event);
  return true;
}

std::vector<PlaneTombstoneEvent>
PlaneTombstoneEventQueue::snapshot() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_;
}

std::vector<PlaneTombstoneEvent>
PlaneTombstoneEventQueue::consume()
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PlaneTombstoneEvent> consumed;
  consumed.swap(pending_);
  pending_identities_.clear();
  return consumed;
}

void PlaneTombstoneEventQueue::retirePlane(
  const std::uint64_t plane_id)
{
  std::lock_guard<std::mutex> lock(mutex_);
  highest_tombstoned_epoch_by_plane_.erase(plane_id);
}

std::size_t
PlaneTombstoneEventQueue::deduplicationWatermarkCount() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return highest_tombstoned_epoch_by_plane_.size();
}

bool PlaneRefitEventQueue::record(
  const PlaneRefitEvent & event)
{
  if (!validRefitEvent(event))
  {
    return false;
  }
  const IdentityKey identity{
    event.expected_geometry_version.plane.id,
    event.expected_geometry_version.plane.epoch};
  std::lock_guard<std::mutex> lock(mutex_);
  if (
    queue_generation_ == 0U ||
    queue_generation_ ==
      std::numeric_limits<std::uint64_t>::max())
  {
    return false;
  }
  const auto found = pending_.find(identity);
  if (found == pending_.end())
  {
    pending_.emplace(identity, event);
    ++queue_generation_;
    return true;
  }
  PlaneRefitEvent & pending = found->second;
  if (
    event.replacement_geometry_version.revision <=
      pending.replacement_geometry_version.revision ||
    event.expected_geometry_version.revision >
      pending.replacement_geometry_version.revision)
  {
    return false;
  }
  const GeometryVersion earliest_expected =
    pending.expected_geometry_version;
  pending = event;
  pending.expected_geometry_version = earliest_expected;
  ++queue_generation_;
  return true;
}

bool PlaneRefitEventBatch::valid() const
{
  return
    delivery_token != 0U &&
    queue_generation != 0U &&
    event_count != 0U &&
    event_count == events.size();
}

PlaneRefitEventBatch
PlaneRefitEventQueue::snapshot() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (active_delivery_.has_value())
  {
    return *active_delivery_;
  }
  PlaneRefitEventBatch result;
  if (pending_.empty())
  {
    return result;
  }
  if (
    next_delivery_token_ == 0U ||
    queue_generation_ == 0U)
  {
    return result;
  }
  result.delivery_token = next_delivery_token_;
  if (
    next_delivery_token_ ==
      std::numeric_limits<std::uint64_t>::max())
  {
    next_delivery_token_ = 0U;
  }
  else
  {
    ++next_delivery_token_;
  }
  result.queue_generation = queue_generation_;
  result.events.reserve(pending_.size());
  for (const auto & entry : pending_)
  {
    result.events.push_back(entry.second);
  }
  result.event_count =
    static_cast<std::uint64_t>(result.events.size());
  result.payload_digest =
    refitPayloadDigest(result.events);
  active_delivery_ = result;
  return *active_delivery_;
}

bool PlaneRefitEventQueue::acknowledge(
  const PlaneRefitEventBatch & applied_batch)
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (
    !active_delivery_.has_value() ||
    !applied_batch.valid() ||
    applied_batch.delivery_token !=
      active_delivery_->delivery_token ||
    applied_batch.queue_generation !=
      active_delivery_->queue_generation ||
    applied_batch.event_count !=
      active_delivery_->event_count ||
    queue_generation_ == 0U ||
    queue_generation_ ==
      std::numeric_limits<std::uint64_t>::max())
  {
    return false;
  }
  const RefitPayloadDigest applied_digest =
    refitPayloadDigest(applied_batch.events);
  const bool payload_matches_receipt =
    constantTimeDigestEqual(
    applied_digest, applied_batch.payload_digest);
  const bool receipt_matches_delivery =
    constantTimeDigestEqual(
    applied_batch.payload_digest,
    active_delivery_->payload_digest);
  if (!payload_matches_receipt || !receipt_matches_delivery)
  {
    return false;
  }
  for (const PlaneRefitEvent & applied :
       active_delivery_->events)
  {
    const IdentityKey identity{
      applied.expected_geometry_version.plane.id,
      applied.expected_geometry_version.plane.epoch};
    const auto pending = pending_.find(identity);
    if (pending == pending_.end())
    {
      continue;
    }
    if (
      pending->second.replacement_geometry_version.revision <=
      applied.replacement_geometry_version.revision)
    {
      pending_.erase(pending);
      continue;
    }
    if (
      applied.replacement_geometry_version.revision >=
        pending->second.expected_geometry_version.revision &&
      applied.replacement_geometry_version.revision <
        pending->second.replacement_geometry_version.revision)
    {
      pending->second.expected_geometry_version =
        applied.replacement_geometry_version;
    }
  }
  active_delivery_.reset();
  ++queue_generation_;
  return true;
}

void PlaneRefitEventQueue::retirePlane(
  const PlaneIdentity identity)
{
  if (!identity.valid())
  {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const IdentityKey retired_key{identity.id, identity.epoch};
  const bool removed_pending =
    pending_.erase(retired_key) != 0U;
  const bool invalidates_active =
    active_delivery_.has_value() &&
    std::any_of(
    active_delivery_->events.begin(),
    active_delivery_->events.end(),
    [&identity](const PlaneRefitEvent & event) {
      return
        event.expected_geometry_version.plane == identity;
    });
  if (!removed_pending && !invalidates_active)
  {
    return;
  }
  if (invalidates_active)
  {
    active_delivery_.reset();
  }
  if (
    queue_generation_ ==
      std::numeric_limits<std::uint64_t>::max())
  {
    queue_generation_ = 0U;
    active_delivery_.reset();
    return;
  }
  ++queue_generation_;
}

std::size_t PlaneRefitEventQueue::pendingEventCount() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return pending_.size();
}

}  // namespace livo
