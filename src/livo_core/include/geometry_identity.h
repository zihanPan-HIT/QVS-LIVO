#ifndef LIVO_GEOMETRY_IDENTITY_H_
#define LIVO_GEOMETRY_IDENTITY_H_

#include <cstdint>

#include <Eigen/Core>

namespace livo
{

struct PlaneIdentity
{
  std::uint64_t id{0U};
  std::uint32_t epoch{0U};

  bool valid() const;
};

struct PlaneLocalCell
{
  std::int32_t u{0};
  std::int32_t v{0};
  std::uint16_t level{0U};
};

struct GeometrySupportKey
{
  PlaneIdentity plane;
  PlaneLocalCell cell;

  bool valid() const;
};

struct GeometryVersion
{
  PlaneIdentity plane;
  std::uint64_t revision{0U};

  bool valid() const;
};

struct PlaneLocalBasis
{
  Eigen::Vector3d origin_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d u_axis_world{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d v_axis_world{Eigen::Vector3d::UnitY()};

  bool valid() const;
};

bool operator==(const PlaneIdentity & lhs, const PlaneIdentity & rhs);
bool operator==(const PlaneLocalCell & lhs, const PlaneLocalCell & rhs);
bool operator==(
  const GeometrySupportKey & lhs,
  const GeometrySupportKey & rhs);
bool operator<(
  const GeometrySupportKey & lhs,
  const GeometrySupportKey & rhs);

}  // namespace livo

#endif  // LIVO_GEOMETRY_IDENTITY_H_
