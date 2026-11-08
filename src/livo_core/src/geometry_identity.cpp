#include "geometry_identity.h"

#include <cmath>
#include <tuple>

namespace livo
{

bool PlaneIdentity::valid() const
{
  return id != 0U && epoch >= 1U;
}

bool GeometrySupportKey::valid() const
{
  return plane.valid();
}

bool GeometryVersion::valid() const
{
  return plane.valid() && revision != 0U;
}

bool PlaneLocalBasis::valid() const
{
  constexpr double kTolerance = 1.0e-6;
  if (
    !origin_world_m.allFinite() ||
    !u_axis_world.allFinite() ||
    !v_axis_world.allFinite())
  {
    return false;
  }
  const double u_norm = u_axis_world.norm();
  const double v_norm = v_axis_world.norm();
  return
    std::isfinite(u_norm) &&
    std::isfinite(v_norm) &&
    std::abs(u_norm - 1.0) <= kTolerance &&
    std::abs(v_norm - 1.0) <= kTolerance &&
    std::abs(u_axis_world.dot(v_axis_world)) <= kTolerance;
}

bool operator==(const PlaneIdentity & lhs, const PlaneIdentity & rhs)
{
  return lhs.id == rhs.id && lhs.epoch == rhs.epoch;
}

bool operator==(const PlaneLocalCell & lhs, const PlaneLocalCell & rhs)
{
  return
    lhs.u == rhs.u &&
    lhs.v == rhs.v &&
    lhs.level == rhs.level;
}

bool operator==(
  const GeometrySupportKey & lhs,
  const GeometrySupportKey & rhs)
{
  return lhs.plane == rhs.plane && lhs.cell == rhs.cell;
}

bool operator<(
  const GeometrySupportKey & lhs,
  const GeometrySupportKey & rhs)
{
  return
    std::tie(
      lhs.plane.id, lhs.plane.epoch,
      lhs.cell.level, lhs.cell.u, lhs.cell.v) <
    std::tie(
      rhs.plane.id, rhs.plane.epoch,
      rhs.cell.level, rhs.cell.u, rhs.cell.v);
}

}  // namespace livo
