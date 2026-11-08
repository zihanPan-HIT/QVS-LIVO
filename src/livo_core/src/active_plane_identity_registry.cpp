#include "active_plane_identity_registry.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

#include "checked_increment.h"

namespace livo
{
namespace
{

using IdentityKey = std::pair<std::uint64_t, std::uint32_t>;

IdentityKey identityKey(const PlaneIdentity identity)
{
  return IdentityKey{identity.id, identity.epoch};
}

bool sameGeometryVersion(
  const GeometryVersion & lhs,
  const GeometryVersion & rhs)
{
  return
    lhs.plane == rhs.plane &&
    lhs.revision == rhs.revision;
}

bool sameDescriptor(
  const ActivePlaneDescriptor & lhs,
  const ActivePlaneDescriptor & rhs)
{
  return
    sameGeometryVersion(
      lhs.geometry_version, rhs.geometry_version) &&
    lhs.generation == rhs.generation;
}

std::uint64_t nextRegistryInstance()
{
  static std::atomic<std::uint64_t> next{1U};
  std::uint64_t value{0U};
  (void)checkedReserve(next, value);
  return value;
}

}  // namespace

class ActivePlaneIdentityRegistryState
{
public:
  mutable std::mutex mutex;
  const std::uint64_t instance{nextRegistryInstance()};
  std::uint64_t revision{1U};
  std::map<IdentityKey, ActivePlaneDescriptor> active_planes;
};

bool ActivePlaneDescriptor::valid() const
{
  return
    geometry_version.valid() &&
    generation != 0U;
}

ActivePlaneIdentityLease::ActivePlaneIdentityLease(
  std::weak_ptr<ActivePlaneIdentityRegistryState> state,
  const std::uint64_t registry_instance,
  const std::uint64_t registry_revision,
  ActivePlaneDescriptor descriptor)
: state_(std::move(state)),
  registry_instance_(registry_instance),
  registry_revision_(registry_revision),
  descriptor_(std::move(descriptor)),
  issued_(true)
{
}

bool ActivePlaneIdentityLease::valid() const
{
  if (
    !issued_ ||
    registry_instance_ == 0U ||
    registry_revision_ == 0U ||
    !descriptor_.valid())
  {
    return false;
  }
  const std::shared_ptr<ActivePlaneIdentityRegistryState> state =
    state_.lock();
  if (!state)
  {
    return false;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  if (
    state->instance != registry_instance_ ||
    state->revision != registry_revision_)
  {
    return false;
  }
  const auto found = state->active_planes.find(
    identityKey(descriptor_.geometry_version.plane));
  return
    found != state->active_planes.end() &&
    sameDescriptor(found->second, descriptor_);
}

GeometryVersion
ActivePlaneIdentityLease::geometryVersion() const
{
  return descriptor_.geometry_version;
}

std::uint64_t ActivePlaneIdentityLease::generation() const
{
  return descriptor_.generation;
}

std::uint64_t
ActivePlaneIdentityLease::registryRevision() const
{
  return registry_revision_;
}

bool ActivePlaneIdentitySnapshot::valid() const
{
  if (
    !issued_ ||
    registry_instance_ == 0U ||
    registry_revision_ == 0U)
  {
    return false;
  }
  const std::shared_ptr<ActivePlaneIdentityRegistryState> state =
    state_.lock();
  if (!state)
  {
    return false;
  }
  std::lock_guard<std::mutex> lock(state->mutex);
  return
    state->instance == registry_instance_ &&
    state->revision == registry_revision_;
}

std::uint64_t ActivePlaneIdentitySnapshot::revision() const
{
  return registry_revision_;
}

std::size_t ActivePlaneIdentitySnapshot::size() const
{
  return active_planes_.size();
}

std::optional<ActivePlaneIdentityLease>
ActivePlaneIdentitySnapshot::leaseFor(
  const GeometryVersion & geometry_version) const
{
  if (!valid() || !geometry_version.valid())
  {
    return std::nullopt;
  }
  const auto found = std::find_if(
    active_planes_.begin(), active_planes_.end(),
    [&geometry_version](
      const ActivePlaneDescriptor & descriptor) {
      return sameGeometryVersion(
        descriptor.geometry_version, geometry_version);
    });
  if (found == active_planes_.end())
  {
    return std::nullopt;
  }
  return ActivePlaneIdentityLease(
    state_, registry_instance_, registry_revision_, *found);
}

bool ActivePlaneIdentitySnapshot::sameRegistryAs(
  const ActivePlaneIdentitySnapshot & other) const
{
  return
    issued_ &&
    other.issued_ &&
    registry_instance_ != 0U &&
    registry_instance_ == other.registry_instance_;
}

bool ActivePlaneIdentitySnapshot::contains(
  const ActivePlaneIdentityLease & lease) const
{
  if (
    !valid() ||
    !lease.valid() ||
    lease.registry_instance_ != registry_instance_ ||
    lease.registry_revision_ != registry_revision_)
  {
    return false;
  }
  return std::any_of(
    active_planes_.begin(), active_planes_.end(),
    [&lease](const ActivePlaneDescriptor & descriptor) {
      return sameDescriptor(descriptor, lease.descriptor_);
    });
}

ActivePlaneIdentityRegistry::ActivePlaneIdentityRegistry()
: state_(std::make_shared<ActivePlaneIdentityRegistryState>())
{
}

ActivePlaneIdentityRegistry::~ActivePlaneIdentityRegistry() = default;

ActivePlaneIdentitySnapshot
ActivePlaneIdentityRegistry::snapshot() const
{
  ActivePlaneIdentitySnapshot result;
  std::lock_guard<std::mutex> lock(state_->mutex);
  result.state_ = state_;
  result.registry_instance_ = state_->instance;
  result.registry_revision_ = state_->revision;
  result.active_planes_.reserve(state_->active_planes.size());
  for (const auto & entry : state_->active_planes)
  {
    result.active_planes_.push_back(entry.second);
  }
  result.issued_ = true;
  return result;
}

std::size_t
ActivePlaneIdentityRegistry::activePlaneCount() const
{
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->active_planes.size();
}

std::uint64_t ActivePlaneIdentityRegistry::revision() const
{
  std::lock_guard<std::mutex> lock(state_->mutex);
  return state_->revision;
}

bool ActivePlaneIdentityRegistry::replaceActivePlanes(
  const std::vector<ActivePlaneDescriptor> & active_planes)
{
  std::map<IdentityKey, ActivePlaneDescriptor> replacement;
  for (const ActivePlaneDescriptor & descriptor : active_planes)
  {
    if (
      !descriptor.valid() ||
      !replacement.emplace(
        identityKey(descriptor.geometry_version.plane),
        descriptor).second)
    {
      return false;
    }
  }

  std::lock_guard<std::mutex> lock(state_->mutex);
  if (
    state_->revision ==
    std::numeric_limits<std::uint64_t>::max())
  {
    return false;
  }
  state_->active_planes.swap(replacement);
  ++state_->revision;
  return true;
}

}  // namespace livo
