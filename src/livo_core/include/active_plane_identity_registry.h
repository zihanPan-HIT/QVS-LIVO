#ifndef LIVO_ACTIVE_PLANE_IDENTITY_REGISTRY_H_
#define LIVO_ACTIVE_PLANE_IDENTITY_REGISTRY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "geometry_identity.h"

class VoxelMapManager;

namespace livo
{

class ActivePlaneIdentityRegistryState;
class MultiViewLandmarkStore;

struct ActivePlaneDescriptor
{
  GeometryVersion geometry_version;
  std::uint64_t generation{0U};

  bool valid() const;
};

class ActivePlaneIdentityLease
{
public:
  ActivePlaneIdentityLease() = default;

  bool valid() const;
  GeometryVersion geometryVersion() const;
  std::uint64_t generation() const;
  std::uint64_t registryRevision() const;

private:
  friend class ActivePlaneIdentityRegistry;
  friend class ActivePlaneIdentitySnapshot;
  friend class MultiViewLandmarkStore;

  ActivePlaneIdentityLease(
    std::weak_ptr<ActivePlaneIdentityRegistryState> state,
    std::uint64_t registry_instance,
    std::uint64_t registry_revision,
    ActivePlaneDescriptor descriptor);

  std::weak_ptr<ActivePlaneIdentityRegistryState> state_;
  std::uint64_t registry_instance_{0U};
  std::uint64_t registry_revision_{0U};
  ActivePlaneDescriptor descriptor_;
  bool issued_{false};
};

class ActivePlaneIdentitySnapshot
{
public:
  ActivePlaneIdentitySnapshot() = default;

  bool valid() const;
  std::uint64_t revision() const;
  std::size_t size() const;
  std::optional<ActivePlaneIdentityLease> leaseFor(
    const GeometryVersion & geometry_version) const;
  // Compares registry affinity only; unlike valid(), this remains meaningful
  // across registry revisions and after the registry owner is destroyed.
  bool sameRegistryAs(
    const ActivePlaneIdentitySnapshot & other) const;
  bool contains(
    const ActivePlaneIdentityLease & lease) const;

private:
  friend class ActivePlaneIdentityRegistry;
  friend class MultiViewLandmarkStore;

  std::weak_ptr<ActivePlaneIdentityRegistryState> state_;
  std::uint64_t registry_instance_{0U};
  std::uint64_t registry_revision_{0U};
  std::vector<ActivePlaneDescriptor> active_planes_;
  bool issued_{false};
};

class ActivePlaneIdentityRegistry
{
public:
  ActivePlaneIdentityRegistry();
  ~ActivePlaneIdentityRegistry();

  ActivePlaneIdentityRegistry(
    const ActivePlaneIdentityRegistry &) = delete;
  ActivePlaneIdentityRegistry & operator=(
    const ActivePlaneIdentityRegistry &) = delete;
  ActivePlaneIdentityRegistry(
    ActivePlaneIdentityRegistry &&) = delete;
  ActivePlaneIdentityRegistry & operator=(
    ActivePlaneIdentityRegistry &&) = delete;

  ActivePlaneIdentitySnapshot snapshot() const;
  std::size_t activePlaneCount() const;
  std::uint64_t revision() const;

private:
  friend class ::VoxelMapManager;

  bool replaceActivePlanes(
    const std::vector<ActivePlaneDescriptor> & active_planes);

  std::shared_ptr<ActivePlaneIdentityRegistryState> state_;
};

}  // namespace livo

#endif  // LIVO_ACTIVE_PLANE_IDENTITY_REGISTRY_H_
