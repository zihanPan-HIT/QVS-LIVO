#include "pinhole_camera_projection_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace livo
{
namespace
{

constexpr std::size_t kMaximumSupportedImagePixels =
  64U * 1024U * 1024U;

bool validImageDimensions(const int width, const int height)
{
  if (width <= 1 || height <= 1)
  {
    return false;
  }
  const std::size_t unsigned_width =
    static_cast<std::size_t>(width);
  const std::size_t unsigned_height =
    static_cast<std::size_t>(height);
  return
    unsigned_width <= kMaximumSupportedImagePixels &&
    unsigned_height <=
      kMaximumSupportedImagePixels / unsigned_width;
}

bool zeroDistortion(
  const std::array<double, 4U> & distortion_coefficients)
{
  return std::all_of(
    distortion_coefficients.begin(),
    distortion_coefficients.end(),
    [](const double coefficient) {
      return std::isfinite(coefficient) && coefficient == 0.0;
    });
}

bool safeDeclaredImageRays(
  const PinholeCameraProjectionConfig & config)
{
  const double inverse_fx = 1.0 / config.fx;
  const double inverse_fy = 1.0 / config.fy;
  if (
    !std::isfinite(inverse_fx) ||
    inverse_fx <= 0.0 ||
    !std::isfinite(inverse_fy) ||
    inverse_fy <= 0.0)
  {
    return false;
  }

  const std::array<double, 2U> horizontal_bounds{{
    0.0,
    std::nextafter(
      static_cast<double>(config.width), 0.0),
  }};
  const std::array<double, 2U> vertical_bounds{{
    0.0,
    std::nextafter(
      static_cast<double>(config.height), 0.0),
  }};
  for (const double u : horizontal_bounds)
  {
    for (const double v : vertical_bounds)
    {
      const Eigen::Vector3d ray(
        (u - config.cx) / config.fx,
        (v - config.cy) / config.fy,
        1.0);
      if (
        !ray.allFinite() ||
        !std::isfinite(ray.squaredNorm()) ||
        !std::isfinite(ray.norm()))
      {
        return false;
      }
    }
  }
  return true;
}

bool validConfig(const PinholeCameraProjectionConfig & config)
{
  return
    config.model == "Pinhole" &&
    validImageDimensions(config.width, config.height) &&
    std::isfinite(config.fx) &&
    config.fx > 0.0 &&
    std::isnormal(config.fx) &&
    std::isfinite(config.fy) &&
    config.fy > 0.0 &&
    std::isnormal(config.fy) &&
    std::isfinite(config.cx) &&
    std::isfinite(config.cy) &&
    zeroDistortion(config.distortion_coefficients) &&
    config.model_revision != 0U &&
    safeDeclaredImageRays(config);
}

}  // namespace

PinholeCameraProjectionAdapter::PinholeCameraProjectionAdapter(
  const PinholeCameraProjectionConfig & config)
: width_(config.width),
  height_(config.height),
  fx_(config.fx),
  fy_(config.fy),
  cx_(config.cx),
  cy_(config.cy),
  distortion_coefficients_(config.distortion_coefficients),
  model_revision_(config.model_revision)
{
  if (!validConfig(config))
  {
    throw std::invalid_argument(
      "unsupported or invalid zero-distortion Pinhole camera");
  }
}

bool PinholeCameraProjectionAdapter::valid() const
{
  return true;
}

CameraRayResult PinholeCameraProjectionAdapter::unproject(
  const Eigen::Vector2d & pixel) const
{
  CameraRayResult result;
  result.status = pixelStatus(pixel);
  if (result.status != ProjectionAdapterStatus::kValid)
  {
    return result;
  }
  result.direction_camera = Eigen::Vector3d(
    (pixel.x() - cx_) / fx_,
    (pixel.y() - cy_) / fy_,
    1.0);
  if (!result.direction_camera.allFinite())
  {
    result.direction_camera =
      Eigen::Vector3d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    result.status = ProjectionAdapterStatus::kNonFinite;
    return result;
  }
  result.status = ProjectionAdapterStatus::kValid;
  return result;
}

CameraRayJacobianResult
PinholeCameraProjectionAdapter::unprojectJacobian(
  const Eigen::Vector2d & pixel) const
{
  CameraRayJacobianResult result;
  result.status = pixelStatus(pixel);
  if (result.status != ProjectionAdapterStatus::kValid)
  {
    return result;
  }
  result.jacobian.setZero();
  result.jacobian(0, 0) = 1.0 / fx_;
  result.jacobian(1, 1) = 1.0 / fy_;
  if (!result.jacobian.allFinite())
  {
    result.jacobian =
      Eigen::Matrix<double, 3, 2>::Constant(
        std::numeric_limits<double>::quiet_NaN());
    result.status = ProjectionAdapterStatus::kNonFinite;
    return result;
  }
  result.status = ProjectionAdapterStatus::kValid;
  return result;
}

CameraProjectionResult PinholeCameraProjectionAdapter::project(
  const Eigen::Vector3d & point_camera) const
{
  CameraProjectionResult result;
  if (!point_camera.allFinite())
  {
    result.status = ProjectionAdapterStatus::kNonFinite;
    return result;
  }
  if (point_camera.z() <= 0.0)
  {
    result.status = ProjectionAdapterStatus::kBehindCamera;
    return result;
  }
  if (point_camera.z() <
      std::numeric_limits<double>::min())
  {
    // Numerical protection for positive IEEE subnormal depth only. This is
    // not a metric near-plane threshold, so ordinary positive depth remains
    // scale invariant.
    result.status = ProjectionAdapterStatus::kOutOfDomain;
    return result;
  }

  const double normalized_x =
    point_camera.x() / point_camera.z();
  const double normalized_y =
    point_camera.y() / point_camera.z();
  result.pixel = Eigen::Vector2d(
    fx_ * normalized_x + cx_,
    fy_ * normalized_y + cy_);
  if (!result.pixel.allFinite())
  {
    result.pixel =
      Eigen::Vector2d::Constant(
        std::numeric_limits<double>::quiet_NaN());
    result.status = ProjectionAdapterStatus::kNonFinite;
    return result;
  }
  result.status = ProjectionAdapterStatus::kValid;
  return result;
}

CameraProjectionJacobianResult
PinholeCameraProjectionAdapter::projectJacobian(
  const Eigen::Vector3d & point_camera) const
{
  CameraProjectionJacobianResult result;
  const CameraProjectionResult projection =
    project(point_camera);
  result.status = projection.status;
  if (result.status != ProjectionAdapterStatus::kValid)
  {
    return result;
  }

  const double inverse_depth = 1.0 / point_camera.z();
  const double normalized_x =
    point_camera.x() * inverse_depth;
  const double normalized_y =
    point_camera.y() * inverse_depth;
  result.jacobian <<
    fx_ * inverse_depth,
    0.0,
    -fx_ * normalized_x * inverse_depth,
    0.0,
    fy_ * inverse_depth,
    -fy_ * normalized_y * inverse_depth;
  if (!result.jacobian.allFinite())
  {
    result.jacobian =
      Eigen::Matrix<double, 2, 3>::Constant(
        std::numeric_limits<double>::quiet_NaN());
    result.status = ProjectionAdapterStatus::kNonFinite;
    return result;
  }
  result.status = ProjectionAdapterStatus::kValid;
  return result;
}

bool PinholeCameraProjectionAdapter::contains(
  const Eigen::Vector2d & pixel,
  const int border) const
{
  if (border < 0 || !pixel.allFinite())
  {
    return false;
  }
  const double minimum = static_cast<double>(border);
  const double maximum_x =
    static_cast<double>(width_) - minimum;
  const double maximum_y =
    static_cast<double>(height_) - minimum;
  return
    minimum < maximum_x &&
    minimum < maximum_y &&
    pixel.x() >= minimum &&
    pixel.x() < maximum_x &&
    pixel.y() >= minimum &&
    pixel.y() < maximum_y;
}

ProjectionAdapterStatus
PinholeCameraProjectionAdapter::pixelStatus(
  const Eigen::Vector2d & pixel) const
{
  if (!pixel.allFinite())
  {
    return ProjectionAdapterStatus::kNonFinite;
  }
  if (!contains(pixel, 0))
  {
    return ProjectionAdapterStatus::kOutOfDomain;
  }
  return ProjectionAdapterStatus::kValid;
}

int PinholeCameraProjectionAdapter::width() const noexcept
{
  return width_;
}

int PinholeCameraProjectionAdapter::height() const noexcept
{
  return height_;
}

double PinholeCameraProjectionAdapter::fx() const noexcept
{
  return fx_;
}

double PinholeCameraProjectionAdapter::fy() const noexcept
{
  return fy_;
}

double PinholeCameraProjectionAdapter::cx() const noexcept
{
  return cx_;
}

double PinholeCameraProjectionAdapter::cy() const noexcept
{
  return cy_;
}

std::array<double, 4U>
PinholeCameraProjectionAdapter::distortionCoefficients() const noexcept
{
  return distortion_coefficients_;
}

std::uint64_t
PinholeCameraProjectionAdapter::modelRevision() const noexcept
{
  return model_revision_;
}

}  // namespace livo
