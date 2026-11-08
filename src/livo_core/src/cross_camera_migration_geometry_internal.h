#ifndef LIVO_CROSS_CAMERA_MIGRATION_GEOMETRY_INTERNAL_H_
#define LIVO_CROSS_CAMERA_MIGRATION_GEOMETRY_INTERNAL_H_

#include "cross_camera_patch_migration.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace livo
{
namespace migration_geometry_internal
{

struct CameraPose
{
  Eigen::Matrix3d rotation_world_from_camera{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_world_m{
    Eigen::Vector3d::Zero()};
};

struct ImageBounds
{
  int width{0};
  int height{0};
};

struct GeometryGateConfig
{
  double minimum_ray_plane_cosine{0.05};
  double minimum_depth_m{0.2};
  double maximum_depth_m{50.0};
  int image_border{2};
  double maximum_warp_condition_number{8.0};
  double minimum_warp_singular_value{0.125};
  double maximum_warp_singular_value{8.0};
  double minimum_positive_warp_determinant{1.0e-6};
};

struct GeometryProjectionRequest
{
  const MigrationEndpoint * source_endpoint{nullptr};
  const MigrationEndpoint * target_endpoint{nullptr};
  CameraPose source_pose;
  CameraPose target_pose;
  Eigen::Vector3d plane_point_world_m{
    Eigen::Vector3d::Zero()};
  Eigen::Vector3d plane_normal_world{
    Eigen::Vector3d::Zero()};
  Eigen::Vector2d source_center_pixel{
    Eigen::Vector2d::Zero()};
  std::uint8_t source_level{0U};
  int patch_width{0};
  std::optional<ImageBounds> target_image_bounds;
  GeometryGateConfig gates;
};

struct ProjectedPoint
{
  MigrationStatus status{MigrationStatus::kNonFinite};
  Eigen::Vector2d target_pixel{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  double source_depth_m{
    std::numeric_limits<double>::quiet_NaN()};
  double target_depth_m{
    std::numeric_limits<double>::quiet_NaN()};
};

struct ProjectedPatch
{
  MigrationStatus status{MigrationStatus::kNonFinite};
  Eigen::Vector2d target_center_pixel{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix2d warp{
    Eigen::Matrix2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  double warp_condition_number{
    std::numeric_limits<double>::infinity()};
  double warp_determinant{
    std::numeric_limits<double>::quiet_NaN()};
  std::size_t valid_projected_pixel_count{0U};
  std::vector<Eigen::Vector2d> target_pixels;
  std::vector<double> target_depths_m;
};

inline CameraPose cameraPose(
  const Eigen::Matrix3d & rotation_world_from_imu,
  const Eigen::Vector3d & position_world_m,
  const MigrationEndpoint & endpoint)
{
  CameraPose result;
  result.rotation_world_from_camera =
    rotation_world_from_imu *
    endpoint.rotation_imu_from_camera;
  result.position_world_m =
    position_world_m +
    rotation_world_from_imu *
    endpoint.position_imu_from_camera;
  return result;
}

inline bool insideOptionalBounds(
  const Eigen::Vector2d & pixel,
  const int border,
  const std::optional<ImageBounds> & bounds)
{
  if (!bounds)
  {
    return true;
  }
  return
    pixel.allFinite() &&
    pixel.x() >= static_cast<double>(border) &&
    pixel.y() >= static_cast<double>(border) &&
    pixel.x() <
      static_cast<double>(bounds->width - border) &&
    pixel.y() <
      static_cast<double>(bounds->height - border);
}

inline ProjectedPoint projectPixel(
  const GeometryProjectionRequest & request,
  const Eigen::Vector2d & source_pixel)
{
  ProjectedPoint result;
  if (
    request.source_endpoint == nullptr ||
    request.target_endpoint == nullptr ||
    request.source_endpoint->camera_model == nullptr ||
    request.target_endpoint->camera_model == nullptr)
  {
    return result;
  }

  const CameraRayResult ray =
    request.source_endpoint->camera_model->unproject(
    source_pixel);
  const double ray_norm = ray.direction_camera.norm();
  if (
    ray.status != ProjectionAdapterStatus::kValid ||
    !ray.direction_camera.allFinite() ||
    !std::isfinite(ray_norm) ||
    ray_norm <= 1.0e-12)
  {
    result.status = MigrationStatus::kInvalidSourceModel;
    return result;
  }

  const Eigen::Vector3d direction_world =
    request.source_pose.rotation_world_from_camera *
    (ray.direction_camera / ray_norm);
  const double denominator =
    request.plane_normal_world.dot(direction_world);
  if (
    !std::isfinite(denominator) ||
    std::abs(denominator) <
      request.gates.minimum_ray_plane_cosine)
  {
    result.status = MigrationStatus::kRayPlaneDegenerate;
    return result;
  }

  const double depth =
    request.plane_normal_world.dot(
    request.plane_point_world_m -
    request.source_pose.position_world_m) /
    denominator;
  if (
    !std::isfinite(depth) ||
    depth <= 0.0 ||
    depth < request.gates.minimum_depth_m ||
    depth > request.gates.maximum_depth_m)
  {
    result.status = MigrationStatus::kBehindCamera;
    return result;
  }

  const Eigen::Vector3d point_world =
    request.source_pose.position_world_m +
    depth * direction_world;
  const Eigen::Vector3d point_target =
    request.target_pose.rotation_world_from_camera.transpose() *
    (point_world - request.target_pose.position_world_m);
  if (!point_target.allFinite() || point_target.z() <= 0.0)
  {
    result.status = MigrationStatus::kBehindCamera;
    return result;
  }

  const CameraProjectionResult projection =
    request.target_endpoint->camera_model->project(
    point_target);
  if (
    projection.status ==
    ProjectionAdapterStatus::kBehindCamera)
  {
    result.status = MigrationStatus::kBehindCamera;
    return result;
  }
  if (
    projection.status != ProjectionAdapterStatus::kValid ||
    !projection.pixel.allFinite())
  {
    result.status = MigrationStatus::kInvalidTargetModel;
    return result;
  }
  if (
    !request.target_endpoint->camera_model->contains(
      projection.pixel, request.gates.image_border) ||
    !insideOptionalBounds(
      projection.pixel, request.gates.image_border,
      request.target_image_bounds))
  {
    result.status = MigrationStatus::kOutOfFrame;
    return result;
  }

  result.status = MigrationStatus::kReady;
  result.target_pixel = projection.pixel;
  result.source_depth_m = depth;
  result.target_depth_m = point_target.z();
  return result;
}

inline ProjectedPatch projectPatch(
  const GeometryProjectionRequest & request)
{
  ProjectedPatch result;
  const ProjectedPoint center = projectPixel(
    request, request.source_center_pixel);
  result.status = center.status;
  if (center.status != MigrationStatus::kReady)
  {
    return result;
  }
  result.target_center_pixel = center.target_pixel;

  const double scale = std::ldexp(
    1.0, static_cast<int>(request.source_level));
  const double half =
    0.5 * static_cast<double>(request.patch_width - 1);
  Eigen::Matrix2d source_second_moment =
    Eigen::Matrix2d::Zero();
  Eigen::Matrix2d target_source_cross =
    Eigen::Matrix2d::Zero();
  result.target_pixels.reserve(
    static_cast<std::size_t>(
      request.patch_width * request.patch_width));
  result.target_depths_m.reserve(
    static_cast<std::size_t>(
      request.patch_width * request.patch_width));

  for (int row = 0; row < request.patch_width; ++row)
  {
    for (int column = 0;
         column < request.patch_width; ++column)
    {
      const Eigen::Vector2d source_offset(
        (static_cast<double>(column) - half) * scale,
        (static_cast<double>(row) - half) * scale);
      const ProjectedPoint projected = projectPixel(
        request,
        request.source_center_pixel + source_offset);
      if (projected.status != MigrationStatus::kReady)
      {
        result.status = projected.status;
        return result;
      }
      ++result.valid_projected_pixel_count;
      result.target_pixels.push_back(projected.target_pixel);
      result.target_depths_m.push_back(
        projected.target_depth_m);
      source_second_moment.noalias() +=
        source_offset * source_offset.transpose();
      target_source_cross.noalias() +=
        (projected.target_pixel -
         result.target_center_pixel) *
        source_offset.transpose();
    }
  }

  if (
    std::abs(source_second_moment.determinant()) <=
    1.0e-12)
  {
    result.status = MigrationStatus::kWarpIllConditioned;
    return result;
  }
  result.warp =
    target_source_cross * source_second_moment.inverse();
  result.warp_determinant = result.warp.determinant();
  const Eigen::JacobiSVD<Eigen::Matrix2d> svd(result.warp);
  const Eigen::Vector2d singular_values =
    svd.singularValues();
  const double minimum_singular =
    singular_values.minCoeff();
  const double maximum_singular =
    singular_values.maxCoeff();
  result.warp_condition_number =
    maximum_singular / minimum_singular;
  result.status = MigrationStatus::kReady;
  return result;
}

inline MigrationStatus validateWarp(
  const ProjectedPatch & result,
  const GeometryGateConfig & gates)
{
  if (result.status != MigrationStatus::kReady)
  {
    return result.status;
  }
  const Eigen::JacobiSVD<Eigen::Matrix2d> svd(result.warp);
  const Eigen::Vector2d singular_values =
    svd.singularValues();
  if (
    !result.warp.allFinite() ||
    !std::isfinite(result.warp_determinant) ||
    result.warp_determinant <=
      gates.minimum_positive_warp_determinant ||
    !std::isfinite(result.warp_condition_number) ||
    singular_values.minCoeff() <
      gates.minimum_warp_singular_value ||
    singular_values.maxCoeff() >
      gates.maximum_warp_singular_value ||
    result.warp_condition_number >
      gates.maximum_warp_condition_number)
  {
    return MigrationStatus::kWarpIllConditioned;
  }
  return MigrationStatus::kReady;
}

}  // namespace migration_geometry_internal
}  // namespace livo

#endif
