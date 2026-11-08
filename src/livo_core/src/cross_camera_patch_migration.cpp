#include "cross_camera_patch_migration.h"

#include "cross_camera_migration_geometry_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

namespace livo
{
namespace
{

constexpr double kTimestampEpsilonSec = 1.0e-9;
constexpr std::size_t kMaximumMigrationImagePixels =
  64U * 1024U * 1024U;
constexpr std::size_t kMaximumMigrationPatchPixels = 64U * 64U;
constexpr std::uint8_t kMaximumMigrationSourceLevelExclusive =
  static_cast<std::uint8_t>(std::numeric_limits<int>::digits);

bool sameCameraModelIdentity(
  const std::shared_ptr<const CameraProjectionAdapter> & left_model,
  const std::uint64_t left_revision,
  const std::shared_ptr<const CameraProjectionAdapter> & right_model,
  const std::uint64_t right_revision) noexcept
{
  return
    static_cast<bool>(left_model) &&
    static_cast<bool>(right_model) &&
    left_model.get() == right_model.get() &&
    !left_model.owner_before(right_model) &&
    !right_model.owner_before(left_model) &&
    left_revision != 0U &&
    left_revision == right_revision;
}

bool checkedArea(
  const int width,
  const int height,
  const std::size_t maximum_area,
  std::size_t * const area)
{
  if (
    area == nullptr ||
    width <= 0 ||
    height <= 0 ||
    maximum_area == 0U)
  {
    return false;
  }
  const std::size_t unsigned_width =
    static_cast<std::size_t>(width);
  const std::size_t unsigned_height =
    static_cast<std::size_t>(height);
  if (
    unsigned_width > maximum_area ||
    unsigned_height > maximum_area / unsigned_width)
  {
    return false;
  }
  *area = unsigned_width * unsigned_height;
  return *area <= maximum_area;
}

bool validRotation(const Eigen::Matrix3d & rotation)
{
  return
    rotation.allFinite() &&
    std::abs(rotation.determinant() - 1.0) <= 1.0e-6 &&
    (rotation.transpose() * rotation -
     Eigen::Matrix3d::Identity()).norm() <= 1.0e-6;
}

template <int Size>
bool positiveSemidefinite(
  const Eigen::Matrix<double, Size, Size> & covariance)
{
  if (!covariance.allFinite() ||
      !covariance.isApprox(covariance.transpose(), 1.0e-9))
  {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<
    Eigen::Matrix<double, Size, Size>> solver(covariance);
  return
    solver.info() == Eigen::Success &&
    solver.eigenvalues().minCoeff() >= -1.0e-10;
}

bool finiteGateConfig(const MigrationGateConfig & gates)
{
  return
    std::isfinite(gates.maximum_observation_age_sec) &&
    gates.maximum_observation_age_sec >= 0.0 &&
    std::isfinite(gates.minimum_ray_plane_cosine) &&
    gates.minimum_ray_plane_cosine > 0.0 &&
    gates.minimum_ray_plane_cosine < 1.0 &&
    std::isfinite(gates.minimum_depth_m) &&
    gates.minimum_depth_m > 0.0 &&
    std::isfinite(gates.maximum_depth_m) &&
    gates.maximum_depth_m > gates.minimum_depth_m &&
    std::isfinite(gates.maximum_depth_disagreement_m) &&
    gates.maximum_depth_disagreement_m > 0.0 &&
    gates.image_border >= 0 &&
    gates.minimum_valid_pixels > 0U &&
    std::isfinite(gates.minimum_patch_standard_deviation) &&
    gates.minimum_patch_standard_deviation >= 0.0 &&
    std::isfinite(gates.minimum_zncc) &&
    gates.minimum_zncc >= -1.0 &&
    gates.minimum_zncc <= 1.0 &&
    std::isfinite(gates.maximum_residual_rmse) &&
    gates.maximum_residual_rmse >= 0.0 &&
    std::isfinite(gates.maximum_warp_condition_number) &&
    gates.maximum_warp_condition_number >= 1.0 &&
    std::isfinite(gates.minimum_warp_singular_value) &&
    gates.minimum_warp_singular_value > 0.0 &&
    std::isfinite(gates.maximum_warp_singular_value) &&
    gates.maximum_warp_singular_value >
      gates.minimum_warp_singular_value &&
    std::isfinite(gates.minimum_positive_warp_determinant) &&
    gates.minimum_positive_warp_determinant > 0.0 &&
    std::isfinite(
      gates.minimum_gradient_direction_coverage) &&
    gates.minimum_gradient_direction_coverage > 0.0 &&
    gates.minimum_gradient_direction_coverage <= 1.0 &&
    std::isfinite(
      gates.maximum_saturated_pixel_fraction) &&
    gates.maximum_saturated_pixel_fraction >= 0.0 &&
    gates.maximum_saturated_pixel_fraction <= 1.0 &&
    std::isfinite(gates.maximum_dark_pixel_fraction) &&
    gates.maximum_dark_pixel_fraction >= 0.0 &&
    gates.maximum_dark_pixel_fraction <= 1.0 &&
    std::isfinite(
      gates.robust_huber_delta_in_standard_deviations) &&
    gates.robust_huber_delta_in_standard_deviations > 0.0 &&
    std::isfinite(
      gates.affine_gain_prior_standard_deviation) &&
    gates.affine_gain_prior_standard_deviation > 0.0 &&
    std::isfinite(
      gates.affine_bias_prior_standard_deviation) &&
    gates.affine_bias_prior_standard_deviation > 0.0 &&
    std::isfinite(
      gates.maximum_affine_schur_condition_number) &&
    gates.maximum_affine_schur_condition_number > 1.0 &&
    std::isfinite(gates.minimum_information_eigenvalue) &&
    gates.minimum_information_eigenvalue >= 0.0;
}

bool qualityFractionNumericallyValid(
  const std::optional<double> & value)
{
  return
    !value ||
    (std::isfinite(*value) && *value >= 0.0 && *value <= 1.0);
}

bool qualityMetadataNumericallyValid(
  const PhotometricFrameMetadata & metadata)
{
  return
    qualityFractionNumericallyValid(
      metadata.saturated_pixel_fraction) &&
    qualityFractionNumericallyValid(
      metadata.dark_pixel_fraction);
}

bool trustedQualityMetadata(
  const PhotometricFrameMetadata & metadata,
  const MigrationGateConfig & gates)
{
  return
    metadata.hardwareQualityStatisticsReady() &&
    *metadata.saturated_pixel_fraction <=
      gates.maximum_saturated_pixel_fraction &&
    *metadata.dark_pixel_fraction <=
      gates.maximum_dark_pixel_fraction;
}

bool validVisibilityConfiguration(
  const MigrationTarget & target)
{
  if (target.depth_m.empty())
  {
    return target.visibility_revision == 0U;
  }
  return target.visibility_revision != 0U;
}

bool targetVisibilityProductionReady(
  const MigrationTarget & target)
{
  return
    validVisibilityConfiguration(target) &&
    target.depthFrameProofReady() &&
    std::any_of(
      target.depth_m.begin(), target.depth_m.end(),
      [](const double depth_m) {
        return std::isfinite(depth_m) && depth_m > 0.0;
      });
}

bool validImage(const MigrationImage & image)
{
  std::size_t expected_pixel_count = 0U;
  return
    image.width > 1 &&
    image.height > 1 &&
    checkedArea(
      image.width, image.height,
      kMaximumMigrationImagePixels,
      &expected_pixel_count) &&
    image.pixels.size() == expected_pixel_count &&
    std::all_of(
      image.pixels.begin(), image.pixels.end(),
      [](const double value) {
        return std::isfinite(value);
      });
}

double sampleBilinear(
  const int width,
  const int height,
  const std::vector<double> & values,
  const Eigen::Vector2d & pixel)
{
  const int left = static_cast<int>(std::floor(pixel.x()));
  const int top = static_cast<int>(std::floor(pixel.y()));
  const int right = left + 1;
  const int bottom = top + 1;
  if (
    left < 0 || top < 0 ||
    right >= width || bottom >= height)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double horizontal = pixel.x() - left;
  const double vertical = pixel.y() - top;
  const auto at = [&values, width](const int x, const int y) {
      return values[
        static_cast<std::size_t>(y) *
          static_cast<std::size_t>(width) +
        static_cast<std::size_t>(x)];
    };
  return
    (1.0 - vertical) *
      ((1.0 - horizontal) * at(left, top) +
       horizontal * at(right, top)) +
    vertical *
      ((1.0 - horizontal) * at(left, bottom) +
       horizontal * at(right, bottom));
}

double standardDeviation(const std::vector<double> & values)
{
  if (values.empty())
  {
    return 0.0;
  }
  double mean = 0.0;
  for (const double value : values)
  {
    mean += value;
  }
  mean /= static_cast<double>(values.size());
  double squared_sum = 0.0;
  for (const double value : values)
  {
    const double centered = value - mean;
    squared_sum += centered * centered;
  }
  return std::sqrt(
    squared_sum / static_cast<double>(values.size()));
}

Eigen::Matrix3d skew(const Eigen::Vector3d & value)
{
  Eigen::Matrix3d result;
  result <<
    0.0, -value.z(), value.y(),
    value.z(), 0.0, -value.x(),
    -value.y(), value.x(), 0.0;
  return result;
}

struct BilinearSample
{
  bool valid{false};
  double value{std::numeric_limits<double>::quiet_NaN()};
  Eigen::Vector2d gradient{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
};

BilinearSample sampleBilinearWithGradient(
  const int width,
  const int height,
  const std::vector<double> & values,
  const Eigen::Vector2d & pixel)
{
  BilinearSample result;
  if (!pixel.allFinite())
  {
    return result;
  }
  const int left = static_cast<int>(std::floor(pixel.x()));
  const int top = static_cast<int>(std::floor(pixel.y()));
  const int right = left + 1;
  const int bottom = top + 1;
  if (
    left < 0 || top < 0 ||
    right >= width || bottom >= height)
  {
    return result;
  }
  const auto at = [&values, width](const int x, const int y) {
      return values[
        static_cast<std::size_t>(y) *
          static_cast<std::size_t>(width) +
        static_cast<std::size_t>(x)];
    };
  const double top_left = at(left, top);
  const double top_right = at(right, top);
  const double bottom_left = at(left, bottom);
  const double bottom_right = at(right, bottom);
  const double horizontal =
    pixel.x() - static_cast<double>(left);
  const double vertical =
    pixel.y() - static_cast<double>(top);
  result.value =
    (1.0 - vertical) *
      ((1.0 - horizontal) * top_left +
       horizontal * top_right) +
    vertical *
      ((1.0 - horizontal) * bottom_left +
       horizontal * bottom_right);
  result.gradient.x() =
    (1.0 - vertical) * (top_right - top_left) +
    vertical * (bottom_right - bottom_left);
  result.gradient.y() =
    (1.0 - horizontal) * (bottom_left - top_left) +
    horizontal * (bottom_right - top_right);
  result.valid =
    std::isfinite(result.value) &&
    result.gradient.allFinite();
  return result;
}

struct LinearizedProjectedPoint
{
  MigrationStatus status{MigrationStatus::kNonFinite};
  bool projection_jacobian_unavailable{false};
  bool source_adapter_jacobian_available{false};
  bool target_adapter_jacobian_available{false};
  Eigen::Vector2d target_pixel{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix<double, 2, 6> target_pixel_wrt_source_pose{
    Eigen::Matrix<double, 2, 6>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix<double, 2, 6> target_pixel_wrt_target_pose{
    Eigen::Matrix<double, 2, 6>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix<double, 2, 3> target_pixel_wrt_plane_point{
    Eigen::Matrix<double, 2, 3>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix<double, 2, 3> target_pixel_wrt_plane_normal{
    Eigen::Matrix<double, 2, 3>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix2d target_pixel_wrt_source_pixel{
    Eigen::Matrix2d::Zero()};
  Eigen::Matrix<double, 2, 6>
    target_pixel_wrt_source_extrinsic{
      Eigen::Matrix<double, 2, 6>::Zero()};
  Eigen::Matrix<double, 2, 6>
    target_pixel_wrt_target_extrinsic{
      Eigen::Matrix<double, 2, 6>::Zero()};
};

LinearizedProjectedPoint migratePixelLinearized(
  const MigrationRequest & request,
  const Eigen::Vector2d & source_pixel)
{
  LinearizedProjectedPoint result;
  const CameraRayResult ray =
    request.source.endpoint.camera_model->unproject(source_pixel);
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

  const Eigen::Matrix3d & source_rotation_world_from_imu =
    request.source.frozen_pose.rotation_world_from_imu;
  const Eigen::Matrix3d & source_rotation_imu_from_camera =
    request.source.endpoint.rotation_imu_from_camera;
  const Eigen::Vector3d & source_position_imu_from_camera =
    request.source.endpoint.position_imu_from_camera;
  const Eigen::Vector3d source_direction_camera =
    ray.direction_camera / ray_norm;
  const CameraRayJacobianResult ray_jacobian =
    request.source.endpoint.camera_model->unprojectJacobian(
      source_pixel);
  Eigen::Matrix<double, 3, 2>
    source_direction_camera_wrt_pixel =
    Eigen::Matrix<double, 3, 2>::Zero();
  if (
    ray_jacobian.status == ProjectionAdapterStatus::kValid &&
    ray_jacobian.jacobian.allFinite())
  {
    source_direction_camera_wrt_pixel =
      (Eigen::Matrix3d::Identity() -
       source_direction_camera *
       source_direction_camera.transpose()) *
      ray_jacobian.jacobian / ray_norm;
    result.source_adapter_jacobian_available =
      source_direction_camera_wrt_pixel.allFinite();
  }
  const Eigen::Vector3d source_direction_imu =
    source_rotation_imu_from_camera *
    source_direction_camera;
  const Eigen::Vector3d direction_world =
    source_rotation_world_from_imu *
    source_direction_imu;
  const Eigen::Vector3d source_camera_center =
    request.source.frozen_pose.position_world_m +
    source_rotation_world_from_imu *
    source_position_imu_from_camera;

  const double normal_norm =
    request.plane.normal_world.norm();
  if (!std::isfinite(normal_norm) || normal_norm <= 1.0e-12)
  {
    result.status = MigrationStatus::kInvalidPlane;
    return result;
  }
  const Eigen::Vector3d normal_world =
    request.plane.normal_world / normal_norm;
  const double numerator =
    normal_world.dot(
      request.plane.point_world_m - source_camera_center);
  const double denominator =
    normal_world.dot(direction_world);
  if (
    !std::isfinite(numerator) ||
    !std::isfinite(denominator) ||
    std::abs(denominator) <
      request.gates.minimum_ray_plane_cosine)
  {
    result.status = MigrationStatus::kRayPlaneDegenerate;
    return result;
  }
  const double depth = numerator / denominator;
  if (
    !std::isfinite(depth) ||
    depth <= 0.0 ||
    depth < request.gates.minimum_depth_m ||
    depth > request.gates.maximum_depth_m)
  {
    result.status = MigrationStatus::kBehindCamera;
    return result;
  }

  Eigen::Matrix<double, 3, 6> source_center_wrt_pose =
    Eigen::Matrix<double, 3, 6>::Zero();
  source_center_wrt_pose.leftCols<3>() =
    -source_rotation_world_from_imu *
    skew(source_position_imu_from_camera);
  source_center_wrt_pose.rightCols<3>().setIdentity();
  Eigen::Matrix<double, 3, 6> direction_wrt_source_pose =
    Eigen::Matrix<double, 3, 6>::Zero();
  direction_wrt_source_pose.leftCols<3>() =
    -source_rotation_world_from_imu *
    skew(source_direction_imu);
  const Eigen::Matrix<double, 1, 6> numerator_wrt_source_pose =
    -normal_world.transpose() * source_center_wrt_pose;
  const Eigen::Matrix<double, 1, 6> denominator_wrt_source_pose =
    normal_world.transpose() * direction_wrt_source_pose;
  const Eigen::Matrix<double, 1, 6> depth_wrt_source_pose =
    (denominator * numerator_wrt_source_pose -
     numerator * denominator_wrt_source_pose) /
    (denominator * denominator);
  const Eigen::Matrix<double, 3, 6> point_wrt_source_pose =
    source_center_wrt_pose +
    direction_world * depth_wrt_source_pose +
    depth * direction_wrt_source_pose;

  const Eigen::Matrix<double, 3, 2>
    direction_world_wrt_source_pixel =
    source_rotation_world_from_imu *
    source_rotation_imu_from_camera *
    source_direction_camera_wrt_pixel;
  const Eigen::Matrix<double, 1, 2>
    denominator_wrt_source_pixel =
    normal_world.transpose() *
    direction_world_wrt_source_pixel;
  const Eigen::Matrix<double, 1, 2>
    depth_wrt_source_pixel =
    -numerator * denominator_wrt_source_pixel /
    (denominator * denominator);
  const Eigen::Matrix<double, 3, 2>
    point_wrt_source_pixel =
    depth * direction_world_wrt_source_pixel +
    direction_world * depth_wrt_source_pixel;

  Eigen::Matrix<double, 3, 6>
    source_center_wrt_extrinsic =
    Eigen::Matrix<double, 3, 6>::Zero();
  source_center_wrt_extrinsic.rightCols<3>() =
    source_rotation_world_from_imu;
  Eigen::Matrix<double, 3, 6>
    direction_wrt_source_extrinsic =
    Eigen::Matrix<double, 3, 6>::Zero();
  direction_wrt_source_extrinsic.leftCols<3>() =
    -source_rotation_world_from_imu *
    source_rotation_imu_from_camera *
    skew(source_direction_camera);
  const Eigen::Matrix<double, 1, 6>
    numerator_wrt_source_extrinsic =
    -normal_world.transpose() *
    source_center_wrt_extrinsic;
  const Eigen::Matrix<double, 1, 6>
    denominator_wrt_source_extrinsic =
    normal_world.transpose() *
    direction_wrt_source_extrinsic;
  const Eigen::Matrix<double, 1, 6>
    depth_wrt_source_extrinsic =
    (denominator * numerator_wrt_source_extrinsic -
     numerator * denominator_wrt_source_extrinsic) /
    (denominator * denominator);
  const Eigen::Matrix<double, 3, 6>
    point_wrt_source_extrinsic =
    source_center_wrt_extrinsic +
    direction_world * depth_wrt_source_extrinsic +
    depth * direction_wrt_source_extrinsic;

  const Eigen::Matrix3d point_wrt_plane_point =
    direction_world * normal_world.transpose() / denominator;
  const Eigen::Matrix<double, 1, 3> depth_wrt_unit_normal =
    (denominator *
       (request.plane.point_world_m -
        source_camera_center).transpose() -
     numerator * direction_world.transpose()) /
    (denominator * denominator);
  const Eigen::Matrix3d unit_normal_wrt_raw_normal =
    (Eigen::Matrix3d::Identity() -
     normal_world * normal_world.transpose()) /
    normal_norm;
  const Eigen::Matrix3d point_wrt_plane_normal =
    direction_world * depth_wrt_unit_normal *
    unit_normal_wrt_raw_normal;

  const Eigen::Vector3d point_world =
    source_camera_center + depth * direction_world;
  const Eigen::Matrix3d & target_rotation_world_from_imu =
    request.target.pose_transfer.rotation_world_from_imu;
  const Eigen::Matrix3d & target_rotation_imu_from_camera =
    request.target.endpoint.rotation_imu_from_camera;
  const Eigen::Vector3d & target_position_imu_from_camera =
    request.target.endpoint.position_imu_from_camera;
  const Eigen::Matrix3d rotation_camera_from_world =
    target_rotation_imu_from_camera.transpose() *
    target_rotation_world_from_imu.transpose();
  const Eigen::Vector3d target_point_imu =
    target_rotation_world_from_imu.transpose() *
    (point_world -
     request.target.pose_transfer.position_world_m);
  const Eigen::Vector3d point_target =
    target_rotation_imu_from_camera.transpose() *
    (target_point_imu - target_position_imu_from_camera);
  if (!point_target.allFinite() || point_target.z() <= 0.0)
  {
    result.status = MigrationStatus::kBehindCamera;
    return result;
  }

  const CameraProjectionResult projection =
    request.target.endpoint.camera_model->project(point_target);
  if (
    projection.status != ProjectionAdapterStatus::kValid ||
    !projection.pixel.allFinite())
  {
    result.status =
      projection.status == ProjectionAdapterStatus::kBehindCamera ?
      MigrationStatus::kBehindCamera :
      MigrationStatus::kInvalidTargetModel;
    return result;
  }
  if (!request.target.endpoint.camera_model->contains(
      projection.pixel, request.gates.image_border))
  {
    result.status = MigrationStatus::kOutOfFrame;
    return result;
  }
  const CameraProjectionJacobianResult projection_jacobian =
    request.target.endpoint.camera_model->projectJacobian(
      point_target);
  if (
    projection_jacobian.status !=
      ProjectionAdapterStatus::kValid ||
    !projection_jacobian.jacobian.allFinite())
  {
    result.status = MigrationStatus::kInvalidTargetModel;
    result.projection_jacobian_unavailable = true;
    return result;
  }

  Eigen::Matrix<double, 3, 6> point_target_wrt_target_pose =
    Eigen::Matrix<double, 3, 6>::Zero();
  point_target_wrt_target_pose.leftCols<3>() =
    target_rotation_imu_from_camera.transpose() *
    skew(target_point_imu);
  point_target_wrt_target_pose.rightCols<3>() =
    -rotation_camera_from_world;
  Eigen::Matrix<double, 3, 6>
    point_target_wrt_target_extrinsic =
    Eigen::Matrix<double, 3, 6>::Zero();
  point_target_wrt_target_extrinsic.leftCols<3>() =
    skew(point_target);
  point_target_wrt_target_extrinsic.rightCols<3>() =
    -target_rotation_imu_from_camera.transpose();

  result.target_pixel = projection.pixel;
  result.target_adapter_jacobian_available = true;
  result.target_pixel_wrt_source_pose =
    projection_jacobian.jacobian *
    rotation_camera_from_world *
    point_wrt_source_pose;
  result.target_pixel_wrt_target_pose =
    projection_jacobian.jacobian *
    point_target_wrt_target_pose;
  result.target_pixel_wrt_plane_point =
    projection_jacobian.jacobian *
    rotation_camera_from_world *
    point_wrt_plane_point;
  result.target_pixel_wrt_plane_normal =
    projection_jacobian.jacobian *
    rotation_camera_from_world *
    point_wrt_plane_normal;
  result.target_pixel_wrt_source_pixel =
    projection_jacobian.jacobian *
    rotation_camera_from_world *
    point_wrt_source_pixel;
  result.target_pixel_wrt_source_extrinsic =
    projection_jacobian.jacobian *
    rotation_camera_from_world *
    point_wrt_source_extrinsic;
  result.target_pixel_wrt_target_extrinsic =
    projection_jacobian.jacobian *
    point_target_wrt_target_extrinsic;
  if (
    !result.target_pixel_wrt_source_pose.allFinite() ||
    !result.target_pixel_wrt_target_pose.allFinite() ||
    !result.target_pixel_wrt_plane_point.allFinite() ||
    !result.target_pixel_wrt_plane_normal.allFinite() ||
    !result.target_pixel_wrt_source_pixel.allFinite() ||
    !result.target_pixel_wrt_source_extrinsic.allFinite() ||
    !result.target_pixel_wrt_target_extrinsic.allFinite())
  {
    result.status = MigrationStatus::kNonFinite;
    return result;
  }
  result.status = MigrationStatus::kReady;
  return result;
}

MigrationResult initializedResult(const MigrationRequest & request)
{
  MigrationResult result;
  result.landmark_id = request.landmark_id;
  result.source_observation_id =
    request.source_observation_id;
  result.source_camera_id =
    request.source.endpoint.camera_id;
  result.target_camera_id =
    request.target.endpoint.camera_id;
  return result;
}

WhitenedPatchContribution initializedContribution(
  const MigrationRequest & request)
{
  WhitenedPatchContribution result;
  result.batch_id = request.batch_id;
  result.landmark_id = request.landmark_id;
  result.source_observation_id =
    request.source_observation_id;
  result.source_camera_id =
    request.source.endpoint.camera_id;
  result.target_camera_id =
    request.target.endpoint.camera_id;
  return result;
}

bool canonicalFactorFromNormalEquation(
  const PatchActiveStateInformation & information,
  const PatchActiveStateGradient & gradient,
  const double squared_error,
  const double minimum_information_eigenvalue,
  WhitenedPatchContribution * const contribution)
{
  if (
    contribution == nullptr ||
    !information.allFinite() ||
    !gradient.allFinite() ||
    !std::isfinite(squared_error) ||
    squared_error < -1.0e-10)
  {
    return false;
  }
  const PatchActiveStateInformation symmetric_information =
    0.5 * (information + information.transpose()).eval();
  Eigen::SelfAdjointEigenSolver<PatchActiveStateInformation> solver(
    symmetric_information);
  if (solver.info() != Eigen::Success)
  {
    return false;
  }
  const double maximum_eigenvalue =
    solver.eigenvalues().maxCoeff();
  if (
    !std::isfinite(maximum_eigenvalue) ||
    maximum_eigenvalue <= minimum_information_eigenvalue)
  {
    return false;
  }
  const double negative_tolerance =
    std::max(1.0e-10, 1.0e-10 * maximum_eigenvalue);
  if (solver.eigenvalues().minCoeff() < -negative_tolerance)
  {
    return false;
  }
  const double rank_threshold = std::max(
    minimum_information_eigenvalue,
    1.0e-12 * maximum_eigenvalue);
  std::vector<Eigen::Index> active_eigenvectors;
  for (Eigen::Index index = 0;
       index < solver.eigenvalues().size(); ++index)
  {
    if (solver.eigenvalues()(index) > rank_threshold)
    {
      active_eigenvectors.push_back(index);
    }
  }
  if (active_eigenvectors.empty())
  {
    return false;
  }

  PatchActiveStateJacobian factor =
    PatchActiveStateJacobian::Zero(
      static_cast<Eigen::Index>(
        active_eigenvectors.size()),
      kCameraTimeTransferActiveStateDimension);
  Eigen::VectorXd residual =
    Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(
        active_eigenvectors.size()));
  PatchActiveStateGradient represented_gradient =
    PatchActiveStateGradient::Zero();
  for (std::size_t row = 0U;
       row < active_eigenvectors.size(); ++row)
  {
    const Eigen::Index eigenvector_index =
      active_eigenvectors[row];
    const double eigenvalue =
      solver.eigenvalues()(eigenvector_index);
    const Eigen::Matrix<
      double,
      kCameraTimeTransferActiveStateDimension, 1> eigenvector =
      solver.eigenvectors().col(eigenvector_index);
    const double square_root = std::sqrt(eigenvalue);
    factor.row(static_cast<Eigen::Index>(row)) =
      square_root * eigenvector.transpose();
    residual(static_cast<Eigen::Index>(row)) =
      eigenvector.dot(gradient) / square_root;
    represented_gradient +=
      eigenvector * eigenvector.dot(gradient);
  }
  if (
    (gradient - represented_gradient).norm() >
    1.0e-8 * (1.0 + gradient.norm()))
  {
    return false;
  }
  const double represented_squared_error =
    residual.squaredNorm();
  const double error_tolerance =
    1.0e-9 *
    (1.0 + std::abs(squared_error) +
     represented_squared_error);
  if (
    represented_squared_error >
    squared_error + error_tolerance)
  {
    return false;
  }
  const double orthogonal_squared_error = std::max(
    0.0, squared_error - represented_squared_error);
  if (orthogonal_squared_error > error_tolerance)
  {
    const Eigen::Index previous_rows = factor.rows();
    factor.conservativeResize(
      previous_rows + 1,
      kCameraTimeTransferActiveStateDimension);
    factor.row(previous_rows).setZero();
    residual.conservativeResize(previous_rows + 1);
    residual(previous_rows) =
      std::sqrt(orthogonal_squared_error);
  }

  contribution->whitened_jacobian = std::move(factor);
  contribution->whitened_residual = std::move(residual);
  contribution->information =
    contribution->whitened_jacobian.transpose() *
    contribution->whitened_jacobian;
  contribution->gradient =
    contribution->whitened_jacobian.transpose() *
    contribution->whitened_residual;
  contribution->normalized_squared_error =
    contribution->whitened_residual.squaredNorm();
  return
    contribution->information.allFinite() &&
    contribution->gradient.allFinite() &&
    std::isfinite(
      contribution->normalized_squared_error);
}

double robustHuberWeight(
  const double normalized_residual,
  const double huber_delta)
{
  const double absolute_residual =
    std::abs(normalized_residual);
  return
    absolute_residual <= huber_delta ?
    1.0 : huber_delta / absolute_residual;
}

}  // namespace

bool FrozenAcceptedSourcePose::valid() const
{
  return
    std::isfinite(exposure_timestamp_sec) &&
    validRotation(rotation_world_from_imu) &&
    position_world_m.allFinite() &&
    positiveSemidefinite(pose_covariance) &&
    accepted_batch_id.value != 0U &&
    accepted_state_revision.value != 0U &&
    store_revision.value != 0U &&
    landmark_id != 0U &&
    observation_id != 0U &&
    observation_revision != 0U &&
    selector_snapshot_revision != 0U &&
    selector_receipt_id != 0U &&
    store_receipt_id != 0U;
}

CameraEndpointUncertaintyReceipt::
CameraEndpointUncertaintyReceipt(
  const MigrationEndpoint & endpoint)
: camera_id_(endpoint.camera_id),
  camera_model_(endpoint.camera_model),
  endpoint_camera_model_revision_(
    endpoint.camera_model_revision),
  endpoint_rig_revision_(endpoint.rig_revision),
  rotation_imu_from_camera_(
    endpoint.rotation_imu_from_camera),
  position_imu_from_camera_(
    endpoint.position_imu_from_camera),
  observation_pixel_covariance_px2_(
    endpoint.uncertainty.observation_pixel_covariance_px2),
  camera_model_equivalent_pixel_covariance_px2_(
    endpoint.uncertainty.
      camera_model_equivalent_pixel_covariance_px2),
  extrinsic_covariance_(
    endpoint.uncertainty.extrinsic_covariance),
  camera_model_revision_(
    endpoint.uncertainty.camera_model_revision),
  rig_revision_(endpoint.uncertainty.rig_revision),
  revision_(endpoint.uncertainty.revision)
{
}

bool CameraEndpointUncertaintyReceipt::matches(
  const MigrationEndpoint & endpoint) const
{
  return
    camera_id_ == endpoint.camera_id &&
    sameCameraModelIdentity(
      camera_model_,
      endpoint_camera_model_revision_,
      endpoint.camera_model,
      endpoint.camera_model_revision) &&
    endpoint_rig_revision_ == endpoint.rig_revision &&
    rotation_imu_from_camera_.isApprox(
      endpoint.rotation_imu_from_camera, 0.0) &&
    position_imu_from_camera_.isApprox(
      endpoint.position_imu_from_camera, 0.0) &&
    observation_pixel_covariance_px2_.isApprox(
      endpoint.uncertainty.
        observation_pixel_covariance_px2, 0.0) &&
    camera_model_equivalent_pixel_covariance_px2_.isApprox(
      endpoint.uncertainty.
        camera_model_equivalent_pixel_covariance_px2,
      0.0) &&
    extrinsic_covariance_.isApprox(
      endpoint.uncertainty.extrinsic_covariance, 0.0) &&
    camera_model_revision_ ==
      endpoint.uncertainty.camera_model_revision &&
    rig_revision_ == endpoint.uncertainty.rig_revision &&
    revision_ == endpoint.uncertainty.revision;
}

void CameraEndpointUncertainty::sealValidationReceipt(
  const MigrationEndpoint & endpoint)
{
  validation_receipt_ =
    CameraEndpointUncertaintyReceipt(endpoint);
}

bool CameraEndpointUncertainty::valid() const
{
  const bool coherent_revisions =
    (camera_model_revision == 0U &&
     rig_revision == 0U &&
     revision == 0U) ||
    (camera_model_revision != 0U &&
     rig_revision != 0U &&
     revision != 0U);
  return
    positiveSemidefinite(observation_pixel_covariance_px2) &&
    positiveSemidefinite(
      camera_model_equivalent_pixel_covariance_px2) &&
    positiveSemidefinite(extrinsic_covariance) &&
    coherent_revisions;
}

bool CameraEndpointUncertainty::productionReady(
  const MigrationEndpoint & endpoint) const
{
  return
    valid() &&
    endpoint.modelRevisionMatchesAdapter() &&
    validation_receipt_ &&
    validation_receipt_->matches(endpoint) &&
    camera_model_revision == endpoint.camera_model_revision &&
    rig_revision == endpoint.rig_revision;
}

bool MigrationEndpoint::modelRevisionMatchesAdapter() const noexcept
{
  return
    static_cast<bool>(camera_model) &&
    camera_model_revision != 0U &&
    camera_model->modelRevision() != 0U &&
    camera_model_revision == camera_model->modelRevision();
}

bool MigrationEndpoint::sharesCameraModelIdentityWith(
  const MigrationEndpoint & other) const noexcept
{
  return
    sameCameraModelIdentity(
      camera_model,
      camera_model_revision,
      other.camera_model,
      other.camera_model_revision);
}

FrozenSourceAcceptanceReceipt::FrozenSourceAcceptanceReceipt(
  const FrozenAcceptedSourcePose & frozen_pose)
: exposure_timestamp_sec_(frozen_pose.exposure_timestamp_sec),
  rotation_world_from_imu_(frozen_pose.rotation_world_from_imu),
  position_world_m_(frozen_pose.position_world_m),
  pose_covariance_(frozen_pose.pose_covariance),
  accepted_batch_id_(frozen_pose.accepted_batch_id),
  accepted_state_revision_(frozen_pose.accepted_state_revision),
  store_revision_(frozen_pose.store_revision),
  landmark_id_(frozen_pose.landmark_id),
  observation_id_(frozen_pose.observation_id),
  observation_revision_(frozen_pose.observation_revision),
  selector_snapshot_revision_(
    frozen_pose.selector_snapshot_revision),
  selector_receipt_id_(frozen_pose.selector_receipt_id),
  store_receipt_id_(frozen_pose.store_receipt_id)
{
}

bool FrozenSourceAcceptanceReceipt::matches(
  const FrozenAcceptedSourcePose & frozen_pose) const
{
  return
    exposure_timestamp_sec_ == frozen_pose.exposure_timestamp_sec &&
    rotation_world_from_imu_.isApprox(
      frozen_pose.rotation_world_from_imu, 0.0) &&
    position_world_m_.isApprox(
      frozen_pose.position_world_m, 0.0) &&
    pose_covariance_.isApprox(
      frozen_pose.pose_covariance, 0.0) &&
    accepted_batch_id_.value ==
      frozen_pose.accepted_batch_id.value &&
    accepted_state_revision_.value ==
      frozen_pose.accepted_state_revision.value &&
    store_revision_.value == frozen_pose.store_revision.value &&
    landmark_id_ == frozen_pose.landmark_id &&
    observation_id_ == frozen_pose.observation_id &&
    observation_revision_ == frozen_pose.observation_revision &&
    selector_snapshot_revision_ ==
      frozen_pose.selector_snapshot_revision &&
    selector_receipt_id_ == frozen_pose.selector_receipt_id &&
    store_receipt_id_ == frozen_pose.store_receipt_id;
}

FrozenSourceIndependenceProof::FrozenSourceIndependenceProof(
  const FrozenAcceptedSourcePose & frozen_pose,
  const std::uint64_t active_state_revision,
  const std::uint64_t provider_revision)
: landmark_id_(frozen_pose.landmark_id),
  observation_id_(frozen_pose.observation_id),
  observation_revision_(frozen_pose.observation_revision),
  active_state_revision_(active_state_revision),
  provider_revision_(provider_revision)
{
}

bool FrozenSourceIndependenceProof::matches(
  const FrozenAcceptedSourcePose & frozen_pose,
  const std::uint64_t active_state_revision) const
{
  return
    landmark_id_ == frozen_pose.landmark_id &&
    observation_id_ == frozen_pose.observation_id &&
    observation_revision_ == frozen_pose.observation_revision &&
    active_state_revision_ == active_state_revision &&
    active_state_revision_ != 0U &&
    provider_revision_ != 0U;
}

void FrozenAcceptedSourcePose::sealAcceptedSnapshot()
{
  acceptance_receipt_ = FrozenSourceAcceptanceReceipt(*this);
}

void FrozenAcceptedSourcePose::sealIndependent(
  const std::uint64_t active_state_revision,
  const std::uint64_t provider_revision)
{
  independence_proof_ = FrozenSourceIndependenceProof(
    *this, active_state_revision, provider_revision);
}

bool FrozenAcceptedSourcePose::acceptedSnapshotProofReady() const
{
  return
    valid() &&
    acceptance_receipt_ &&
    acceptance_receipt_->matches(*this);
}

bool FrozenAcceptedSourcePose::independentFromActivePrior(
  const std::uint64_t active_state_revision) const
{
  return
    acceptedSnapshotProofReady() &&
    independence_proof_ &&
    independence_proof_->matches(
      *this, active_state_revision);
}

bool FrozenAcceptedSourcePose::validFor(
  const BatchId current_batch_id,
  const LandmarkId expected_landmark_id,
  const ObservationId expected_observation_id) const
{
  return
    valid() &&
    current_batch_id.value != 0U &&
    accepted_batch_id.value < current_batch_id.value &&
    landmark_id == expected_landmark_id &&
    observation_id == expected_observation_id;
}

EndpointNoiseConditionerGate::EndpointNoiseConditionerGate(
  const std::uint64_t source_imu_epoch,
  const std::uint64_t target_imu_epoch,
  const std::uint64_t source_imu_history_revision,
  const std::uint64_t target_imu_history_revision)
: imu_epoch_(target_imu_epoch),
  imu_history_revision_(target_imu_history_revision),
  same_imu_history_(
    source_imu_epoch != 0U &&
    source_imu_epoch == target_imu_epoch &&
    source_imu_history_revision != 0U &&
    source_imu_history_revision ==
      target_imu_history_revision)
{
}

EndpointJointPoseCovariance::EndpointJointPoseCovariance(
  const Eigen::Matrix<double, 12, 12> & covariance,
  const std::uint64_t revision,
  const BatchId batch_id,
  const ObservationId source_observation_id,
  const std::uint64_t source_observation_revision,
  const double source_timestamp_sec,
  const double target_timestamp_sec,
  const std::uint64_t active_state_revision,
  const EndpointNoiseConditionerGate & gate)
: covariance_(covariance),
  revision_(revision),
  batch_id_(batch_id),
  source_observation_id_(source_observation_id),
  source_observation_revision_(source_observation_revision),
  source_timestamp_sec_(source_timestamp_sec),
  target_timestamp_sec_(target_timestamp_sec),
  imu_epoch_(gate.imu_epoch_),
  imu_history_revision_(gate.imu_history_revision_),
  same_imu_history_(gate.same_imu_history_),
  active_state_revision_(active_state_revision)
{
}

const Eigen::Matrix<double, 12, 12> &
EndpointJointPoseCovariance::covariance() const
{
  return covariance_;
}

bool EndpointJointPoseCovariance::valid() const
{
  return
    revision_ != 0U &&
    batch_id_.value != 0U &&
    source_observation_id_ != 0U &&
    source_observation_revision_ != 0U &&
    std::isfinite(source_timestamp_sec_) &&
    std::isfinite(target_timestamp_sec_) &&
    target_timestamp_sec_ >= source_timestamp_sec_ &&
    imu_epoch_ != 0U &&
    imu_history_revision_ != 0U &&
    same_imu_history_ &&
    active_state_revision_ != 0U &&
    positiveSemidefinite(covariance_);
}

bool EndpointJointPoseCovariance::productionReadyFor(
  const BatchId batch_id,
  const ObservationId source_observation_id,
  const std::uint64_t source_observation_revision,
  const double source_timestamp_sec,
  const double target_timestamp_sec,
  const std::uint64_t target_imu_epoch,
  const std::uint64_t active_state_revision) const
{
  return
    valid() &&
    batch_id_.value == batch_id.value &&
    source_observation_id_ == source_observation_id &&
    source_observation_revision_ ==
      source_observation_revision &&
    source_timestamp_sec_ == source_timestamp_sec &&
    target_timestamp_sec_ == target_timestamp_sec &&
    imu_epoch_ == target_imu_epoch &&
    active_state_revision_ == active_state_revision;
}

RawIntensityNoiseReceipt::RawIntensityNoiseReceipt(
  const RawIntensityNoiseModel & noise)
: standard_deviation_raw_(noise.standard_deviation_raw),
  calibration_revision_(noise.calibration_revision)
{
}

bool RawIntensityNoiseReceipt::matches(
  const RawIntensityNoiseModel & noise) const
{
  return
    standard_deviation_raw_ == noise.standard_deviation_raw &&
    calibration_revision_ == noise.calibration_revision;
}

void RawIntensityNoiseModel::sealValidationReceipt()
{
  validation_receipt_ = RawIntensityNoiseReceipt(*this);
}

bool RawIntensityNoiseModel::usable() const
{
  return
    std::isfinite(standard_deviation_raw) &&
    standard_deviation_raw > 0.0;
}

bool RawIntensityNoiseModel::productionReady(
  const std::uint64_t expected_calibration_revision) const
{
  return
    usable() &&
    validation_receipt_ &&
    validation_receipt_->matches(*this) &&
    expected_calibration_revision != 0U &&
    calibration_revision == expected_calibration_revision;
}

TargetDepthFrameReceipt::TargetDepthFrameReceipt(
  const MigrationTarget & target)
: exposure_timestamp_sec_(target.exposure_timestamp_sec),
  depth_m_(target.depth_m),
  visibility_revision_(target.visibility_revision),
  frame_camera_model_revision_(
    target.frame_camera_model_revision),
  frame_rig_revision_(target.frame_rig_revision)
{
}

bool TargetDepthFrameReceipt::matches(
  const MigrationTarget & target) const
{
  return
    exposure_timestamp_sec_ == target.exposure_timestamp_sec &&
    depth_m_ == target.depth_m &&
    visibility_revision_ == target.visibility_revision &&
    frame_camera_model_revision_ ==
      target.frame_camera_model_revision &&
    frame_rig_revision_ == target.frame_rig_revision;
}

void MigrationTarget::sealDepthFrameReceipt()
{
  depth_frame_receipt_ = TargetDepthFrameReceipt(*this);
}

bool MigrationTarget::depthFrameProofReady() const
{
  return
    depth_frame_receipt_ &&
    depth_frame_receipt_->matches(*this) &&
    visibility_revision != 0U &&
    !depth_m.empty();
}

bool ProductionPatchCapabilities::
frozenSourcePriorIndependence() const
{
  return frozen_source_prior_independence_;
}

bool ProductionPatchCapabilities::
jointEndpointPoseCovariance() const
{
  return joint_endpoint_pose_covariance_;
}

bool ProductionPatchCapabilities::sourceRawIntensityNoise() const
{
  return source_raw_intensity_noise_;
}

bool ProductionPatchCapabilities::targetRawIntensityNoise() const
{
  return target_raw_intensity_noise_;
}

bool ProductionPatchCapabilities::sourceEndpointUncertainty() const
{
  return source_endpoint_uncertainty_;
}

bool ProductionPatchCapabilities::targetEndpointUncertainty() const
{
  return target_endpoint_uncertainty_;
}

bool ProductionPatchCapabilities::sourceAdapterJacobian() const
{
  return source_adapter_jacobian_;
}

bool ProductionPatchCapabilities::targetAdapterJacobian() const
{
  return target_adapter_jacobian_;
}

bool ProductionPatchCapabilities::targetVisibility() const
{
  return target_visibility_;
}

bool ProductionPatchCapabilities::
photometricQualityMetadata() const
{
  return photometric_quality_metadata_;
}

bool ProductionPatchCapabilities::orientationPreservingWarp() const
{
  return orientation_preserving_warp_;
}

bool ProductionPatchCapabilities::gradientDirectionCoverage() const
{
  return gradient_direction_coverage_;
}

bool ProductionPatchCapabilities::all() const
{
  return
    frozen_source_prior_independence_ &&
    joint_endpoint_pose_covariance_ &&
    source_raw_intensity_noise_ &&
    target_raw_intensity_noise_ &&
    source_endpoint_uncertainty_ &&
    target_endpoint_uncertainty_ &&
    source_adapter_jacobian_ &&
    target_adapter_jacobian_ &&
    target_visibility_ &&
    photometric_quality_metadata_ &&
    orientation_preserving_warp_ &&
    gradient_direction_coverage_;
}

CrossCameraPatchMigration::CrossCameraPatchMigration(
  const PhotometricCalibration & photometric_calibration)
: photometric_calibration_(photometric_calibration)
{
}

MigrationResult CrossCameraPatchMigration::prepare(
  const MigrationRequest & request) const
{
  MigrationResult result = initializedResult(request);
  if (
    request.batch_id.value == 0U ||
    request.active_state_revision == 0U ||
    request.landmark_id == 0U ||
    request.source_observation_id == 0U ||
    !finiteGateConfig(request.gates))
  {
    result.status = MigrationStatus::kStaleLandmark;
    return result;
  }
  if (
    request.source.endpoint.camera_id >= 4U ||
    request.target.endpoint.camera_id >= 4U ||
    request.source.endpoint.camera_id ==
      request.target.endpoint.camera_id ||
    request.source.endpoint.camera_model == nullptr ||
    !request.source.endpoint.camera_model->valid())
  {
    result.status = MigrationStatus::kInvalidSourceModel;
    return result;
  }
  if (
    request.target.endpoint.camera_model == nullptr ||
    !request.target.endpoint.camera_model->valid())
  {
    result.status = MigrationStatus::kInvalidTargetModel;
    return result;
  }
  if (
    !request.source.endpoint.modelRevisionMatchesAdapter() ||
    !request.target.endpoint.modelRevisionMatchesAdapter() ||
    request.source.observation_camera_model_revision !=
      request.source.endpoint.camera_model_revision ||
    request.target.frame_camera_model_revision !=
      request.target.endpoint.camera_model_revision)
  {
    result.status = MigrationStatus::kModelRevisionMismatch;
    return result;
  }
  if (
    request.source.endpoint.rig_revision == 0U ||
    request.source.observation_rig_revision !=
      request.source.endpoint.rig_revision ||
    request.target.frame_rig_revision !=
      request.target.endpoint.rig_revision ||
    request.source.endpoint.rig_revision !=
      request.target.endpoint.rig_revision)
  {
    result.status = MigrationStatus::kRigRevisionMismatch;
    return result;
  }
  if (
    !request.source.endpoint.uncertainty.valid() ||
    !request.target.endpoint.uncertainty.valid())
  {
    result.status = MigrationStatus::kNonFinite;
    return result;
  }
  const std::uint64_t source_photo_revision =
    photometric_calibration_.cameraCalibrationRevision(
      request.source.endpoint.camera_id);
  const std::uint64_t target_photo_revision =
    photometric_calibration_.cameraCalibrationRevision(
      request.target.endpoint.camera_id);
  if (
    request.source.
      observation_photometric_calibration_revision == 0U ||
    request.target.
      frame_photometric_calibration_revision == 0U ||
    (source_photo_revision != 0U &&
     request.source.
       observation_photometric_calibration_revision !=
       source_photo_revision) ||
    (target_photo_revision != 0U &&
     request.target.
       frame_photometric_calibration_revision !=
       target_photo_revision))
  {
    result.status =
      MigrationStatus::kPhotometricRevisionMismatch;
    return result;
  }
  if (
    !(request.source.geometry_version.plane ==
      request.plane.geometry_version.plane) ||
    request.source.geometry_version.revision !=
      request.plane.geometry_version.revision ||
    !(request.plane.support_key.plane ==
      request.plane.geometry_version.plane))
  {
    result.status = MigrationStatus::kStaleLandmark;
    return result;
  }
  if (
    !request.source.frozen_pose.validFor(
      request.batch_id,
      request.landmark_id,
      request.source_observation_id))
  {
    result.status = MigrationStatus::kStaleLandmark;
    return result;
  }
  const Eigen::Matrix<double, 6, 6>
    expected_target_endpoint_covariance =
    request.target.pose_transfer.pose_transfer_covariance +
    request.target.pose_transfer.time_alignment_pose_covariance;
  if (
    (request.endpoint_joint_pose_covariance &&
     (!request.endpoint_joint_pose_covariance->
        productionReadyFor(
          request.batch_id,
          request.source_observation_id,
          request.source.frozen_pose.observation_revision,
          request.source.frozen_pose.exposure_timestamp_sec,
          request.target.exposure_timestamp_sec,
          request.target.pose_transfer.imu_epoch,
          request.active_state_revision) ||
      !request.endpoint_joint_pose_covariance->covariance().
        topLeftCorner<6, 6>().isApprox(
          request.source.frozen_pose.pose_covariance, 1.0e-9) ||
      !request.endpoint_joint_pose_covariance->covariance().
        bottomRightCorner<6, 6>().isApprox(
          expected_target_endpoint_covariance, 1.0e-9))) ||
    !request.source.raw_intensity_noise.usable() ||
    !request.target.raw_intensity_noise.usable())
  {
    result.status = MigrationStatus::kPhotometricUnavailable;
    return result;
  }
  const double source_exposure_timestamp_sec =
    request.source.frozen_pose.exposure_timestamp_sec;
  if (
    !std::isfinite(source_exposure_timestamp_sec) ||
    !std::isfinite(request.target.exposure_timestamp_sec) ||
    !request.target.pose_transfer.valid() ||
    std::abs(
      request.target.pose_transfer.target_timestamp_sec -
      request.target.exposure_timestamp_sec) >
        kTimestampEpsilonSec ||
    request.target.exposure_timestamp_sec <
      source_exposure_timestamp_sec ||
    request.target.exposure_timestamp_sec -
      source_exposure_timestamp_sec >
        request.gates.maximum_observation_age_sec)
  {
    result.status = MigrationStatus::kInvalidTimeTransfer;
    return result;
  }
  if (
    !validRotation(
      request.source.endpoint.rotation_imu_from_camera) ||
    !validRotation(
      request.target.endpoint.rotation_imu_from_camera) ||
    !request.source.endpoint.position_imu_from_camera.allFinite() ||
    !request.target.endpoint.position_imu_from_camera.allFinite() ||
    !validRotation(
      request.target.pose_transfer.rotation_world_from_imu) ||
    !request.target.pose_transfer.position_world_m.allFinite())
  {
    result.status = MigrationStatus::kInvalidTimeTransfer;
    return result;
  }
  if (
    !request.plane.point_world_m.allFinite() ||
    !request.plane.normal_world.allFinite() ||
    request.plane.normal_world.norm() <= 1.0e-12 ||
    !positiveSemidefinite(
      request.plane.point_covariance_m2) ||
    !positiveSemidefinite(
      request.plane.normal_covariance))
  {
    result.status = MigrationStatus::kInvalidPlane;
    return result;
  }
  std::size_t expected_patch_size = 0U;
  if (
    !checkedArea(
      request.source.patch_width,
      request.source.patch_width,
      kMaximumMigrationPatchPixels,
      &expected_patch_size) ||
    request.source.source_patch.size() != expected_patch_size ||
    request.source.source_level >=
      kMaximumMigrationSourceLevelExclusive ||
    !request.source.source_pixel.allFinite() ||
    !qualityMetadataNumericallyValid(
      request.source.photo_metadata) ||
    !qualityMetadataNumericallyValid(
      request.target.photo_metadata) ||
    !validVisibilityConfiguration(request.target) ||
    !std::all_of(
      request.source.source_patch.begin(),
      request.source.source_patch.end(),
      [](const double value) {
        return std::isfinite(value);
      }) ||
    !validImage(request.target.image) ||
    (!request.target.depth_m.empty() &&
     (request.target.depth_m.size() !=
        request.target.image.pixels.size() ||
      !std::all_of(
        request.target.depth_m.begin(),
        request.target.depth_m.end(),
        [](const double value) {
          return
            std::isfinite(value) && value >= 0.0;
        }))) ||
    (request.gates.require_target_depth &&
     request.target.depth_m.empty()))
  {
    result.status = MigrationStatus::kNonFinite;
    return result;
  }
  result.target_visibility_validated_ =
    targetVisibilityProductionReady(request.target);
  result.photometric_quality_metadata_validated_ =
    trustedQualityMetadata(
      request.source.photo_metadata, request.gates) &&
    trustedQualityMetadata(
      request.target.photo_metadata, request.gates);
  if (
    request.photometric_mode ==
      PhotometricMode::kRobustAffineExperimental &&
    !request.gates.allow_relative_affine_experiment)
  {
    result.status = MigrationStatus::kPhotometricUnavailable;
    return result;
  }

  migration_geometry_internal::GeometryProjectionRequest
    geometry_request;
  geometry_request.source_endpoint =
    &request.source.endpoint;
  geometry_request.target_endpoint =
    &request.target.endpoint;
  geometry_request.source_pose =
    migration_geometry_internal::cameraPose(
    request.source.frozen_pose.rotation_world_from_imu,
    request.source.frozen_pose.position_world_m,
    request.source.endpoint);
  geometry_request.target_pose =
    migration_geometry_internal::cameraPose(
    request.target.pose_transfer.rotation_world_from_imu,
    request.target.pose_transfer.position_world_m,
    request.target.endpoint);
  geometry_request.plane_point_world_m =
    request.plane.point_world_m;
  geometry_request.plane_normal_world =
    request.plane.normal_world.normalized();
  geometry_request.source_center_pixel =
    request.source.source_pixel;
  geometry_request.source_level =
    request.source.source_level;
  geometry_request.patch_width =
    request.source.patch_width;
  geometry_request.gates.minimum_ray_plane_cosine =
    request.gates.minimum_ray_plane_cosine;
  geometry_request.gates.minimum_depth_m =
    request.gates.minimum_depth_m;
  geometry_request.gates.maximum_depth_m =
    request.gates.maximum_depth_m;
  geometry_request.gates.image_border =
    request.gates.image_border;
  geometry_request.gates.maximum_warp_condition_number =
    request.gates.maximum_warp_condition_number;
  geometry_request.gates.minimum_warp_singular_value =
    request.gates.minimum_warp_singular_value;
  geometry_request.gates.maximum_warp_singular_value =
    request.gates.maximum_warp_singular_value;
  geometry_request.gates.minimum_positive_warp_determinant =
    request.gates.minimum_positive_warp_determinant;

  const migration_geometry_internal::ProjectedPatch projected =
    migration_geometry_internal::projectPatch(
    geometry_request);
  if (projected.status != MigrationStatus::kReady)
  {
    result.status = projected.status;
    return result;
  }
  result.target_pixel = projected.target_center_pixel;

  const double scale =
    std::ldexp(
      1.0, static_cast<int>(request.source.source_level));
  const double half =
    0.5 * static_cast<double>(
      request.source.patch_width - 1);
  Eigen::Matrix2d gradient_second_moment =
    Eigen::Matrix2d::Zero();
  std::vector<PhotometricSample> reference_samples;
  std::vector<PhotometricSample> current_samples;
  reference_samples.reserve(
    request.source.source_patch.size());
  current_samples.reserve(
    request.source.source_patch.size());
  result.warped_reference_patch.reserve(
    request.source.source_patch.size());
  result.current_patch.reserve(
    request.source.source_patch.size());
  if (
    projected.target_pixels.size() !=
      request.source.source_patch.size() ||
    projected.target_depths_m.size() !=
      request.source.source_patch.size())
  {
    result.status = MigrationStatus::kNonFinite;
    return result;
  }

  std::size_t patch_index = 0U;
  for (int row = 0; row < request.source.patch_width; ++row)
  {
    for (int column = 0;
         column < request.source.patch_width;
         ++column, ++patch_index)
    {
      const Eigen::Vector2d source_offset(
        (static_cast<double>(column) - half) * scale,
        (static_cast<double>(row) - half) * scale);
      const Eigen::Vector2d source_pixel =
        request.source.source_pixel + source_offset;
      const Eigen::Vector2d & target_pixel =
        projected.target_pixels[patch_index];
      const BilinearSample current_sample_with_gradient =
        sampleBilinearWithGradient(
          request.target.image.width,
          request.target.image.height,
          request.target.image.pixels,
          target_pixel);
      const double current_value =
        current_sample_with_gradient.value;
      if (!current_sample_with_gradient.valid)
      {
        result.status = MigrationStatus::kOutOfFrame;
        return result;
      }
      if (!request.target.depth_m.empty())
      {
        const double measured_depth_m =
          sampleBilinear(
            request.target.image.width,
            request.target.image.height,
            request.target.depth_m,
            target_pixel);
        if (
          !std::isfinite(measured_depth_m) ||
          measured_depth_m <= 0.0)
        {
          result.target_visibility_validated_ = false;
          if (request.gates.require_target_depth)
          {
            result.status = MigrationStatus::kOccluded;
            return result;
          }
        }
        else if (
          std::abs(
            measured_depth_m -
            projected.target_depths_m[patch_index]) >
          request.gates.maximum_depth_disagreement_m)
        {
          result.status = MigrationStatus::kOccluded;
          return result;
        }
      }

      gradient_second_moment.noalias() +=
        current_sample_with_gradient.gradient *
        current_sample_with_gradient.gradient.transpose();
      result.warped_reference_patch.push_back(
        request.source.source_patch[patch_index]);
      result.current_patch.push_back(current_value);

      PhotometricSample reference_sample;
      reference_sample.raw_value =
        request.source.source_patch[patch_index];
      reference_sample.pixel_x = source_pixel.x();
      reference_sample.pixel_y = source_pixel.y();
      reference_samples.push_back(reference_sample);
      PhotometricSample current_sample;
      current_sample.raw_value = current_value;
      current_sample.pixel_x = target_pixel.x();
      current_sample.pixel_y = target_pixel.y();
      current_samples.push_back(current_sample);
    }
  }

  result.warp = projected.warp;
  result.warp_determinant =
    projected.warp_determinant;
  result.warp_condition_number =
    projected.warp_condition_number;
  const MigrationStatus warp_status =
    migration_geometry_internal::validateWarp(
    projected, geometry_request.gates);
  if (warp_status != MigrationStatus::kReady)
  {
    result.status = warp_status;
    return result;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>
    gradient_solver(
      0.5 *
      (gradient_second_moment +
       gradient_second_moment.transpose()).eval());
  if (
    gradient_solver.info() == Eigen::Success &&
    gradient_solver.eigenvalues().allFinite() &&
    gradient_solver.eigenvalues().maxCoeff() > 0.0)
  {
    result.gradient_direction_coverage = std::clamp(
      gradient_solver.eigenvalues().minCoeff() /
      gradient_solver.eigenvalues().maxCoeff(),
      0.0, 1.0);
  }

  if (
    standardDeviation(result.warped_reference_patch) <
      request.gates.minimum_patch_standard_deviation ||
    standardDeviation(result.current_patch) <
      request.gates.minimum_patch_standard_deviation)
  {
    result.status = MigrationStatus::kLowTexture;
    return result;
  }

  PhotometricPatchRequest photo_request;
  photo_request.mode = request.photometric_mode;
  photo_request.reference.camera_id =
    request.source.endpoint.camera_id;
  photo_request.reference.metadata =
    request.source.photo_metadata;
  photo_request.reference.samples =
    std::move(reference_samples);
  photo_request.current.camera_id =
    request.target.endpoint.camera_id;
  photo_request.current.metadata =
    request.target.photo_metadata;
  photo_request.current.samples =
    std::move(current_samples);
  result.photometric =
    photometric_calibration_.compare(photo_request);
  if (!result.photometric.valid())
  {
    result.status =
      result.photometric.status ==
        PhotometricStatus::kDegeneratePatch ?
      MigrationStatus::kLowTexture :
      MigrationStatus::kPhotometricUnavailable;
    return result;
  }
  if (
    request.photometric_mode ==
      PhotometricMode::kCalibratedDirect &&
    (!result.photometric.cross_camera_comparable ||
     result.photometric.experimental))
  {
    result.status = MigrationStatus::kPhotometricUnavailable;
    return result;
  }
  result.valid_pixel_count =
    result.photometric.valid_sample_count;
  result.zncc = result.photometric.zncc;
  result.residual_rmse =
    result.photometric.residual_rmse;
  if (
    result.valid_pixel_count <
      request.gates.minimum_valid_pixels)
  {
    result.status = MigrationStatus::kPhotometricUnavailable;
    return result;
  }
  if (
    !std::isfinite(result.zncc) ||
    result.zncc < request.gates.minimum_zncc)
  {
    result.status = MigrationStatus::kNccRejected;
    return result;
  }
  if (
    !std::isfinite(result.residual_rmse) ||
    result.residual_rmse >
      request.gates.maximum_residual_rmse)
  {
    result.status = MigrationStatus::kResidualRejected;
    return result;
  }

  result.consumable = false;
  result.shadow_only = true;
  result.status = MigrationStatus::kReady;
  return result;
}

std::vector<WhitenedPatchContribution>
CrossCameraPatchMigration::linearizeBatch(
  const std::vector<MigrationRequest> & requests) const
{
  struct ScalarPatchLinearization
  {
    double residual{0.0};
    double variance{0.0};
    Eigen::Matrix<
      double, 1,
      kCameraTimeTransferActiveStateDimension> state_jacobian{
      Eigen::Matrix<
        double, 1,
        kCameraTimeTransferActiveStateDimension>::Zero()};
    Eigen::RowVector2d affine_jacobian{
      Eigen::RowVector2d::Zero()};
  };

  std::vector<WhitenedPatchContribution> contributions;
  contributions.reserve(requests.size());
  std::set<std::pair<std::uint64_t, LandmarkId>>
    consumed_information_units;

  for (const MigrationRequest & request : requests)
  {
    WhitenedPatchContribution contribution =
      initializedContribution(request);
    contribution.experimental =
      request.photometric_mode ==
      PhotometricMode::kRobustAffineExperimental;
    const std::pair<std::uint64_t, LandmarkId> information_key{
      request.batch_id.value, request.landmark_id};
    if (
      consumed_information_units.find(information_key) !=
      consumed_information_units.end())
    {
      contribution.status =
        PatchContributionStatus::kDuplicateLandmark;
      contributions.push_back(std::move(contribution));
      continue;
    }

    const MigrationResult migration = prepare(request);
    contribution.migration_status = migration.status;
    if (!migration.geometry_ready())
    {
      contribution.status =
        PatchContributionStatus::kMigrationRejected;
      contributions.push_back(std::move(contribution));
      continue;
    }

    const std::size_t patch_size =
      request.source.source_patch.size();
    if (
      migration.current_patch.size() != patch_size ||
      migration.photometric.valid_mask.size() != patch_size ||
      migration.photometric.reference_values.size() != patch_size ||
      migration.photometric.current_values.size() != patch_size)
    {
      contribution.status = PatchContributionStatus::kNonFinite;
      contributions.push_back(std::move(contribution));
      continue;
    }

    std::vector<ScalarPatchLinearization> scalar_rows;
    scalar_rows.reserve(migration.valid_pixel_count);
    const double scale = std::ldexp(
      1.0, static_cast<int>(request.source.source_level));
    const double half =
      0.5 * static_cast<double>(
        request.source.patch_width - 1);
    const double affine_gain =
      request.photometric_mode ==
        PhotometricMode::kRobustAffineExperimental ?
      migration.photometric.estimated_gain : 1.0;
    const double affine_bias =
      request.photometric_mode ==
        PhotometricMode::kRobustAffineExperimental ?
      migration.photometric.estimated_bias : 0.0;
    if (
      !std::isfinite(affine_gain) ||
      !std::isfinite(affine_bias))
    {
      contribution.status =
        PatchContributionStatus::kPhotometricUnavailable;
      contributions.push_back(std::move(contribution));
      continue;
    }

    Eigen::Matrix<double, 12, 12>
      endpoint_joint_pose_covariance =
      Eigen::Matrix<double, 12, 12>::Zero();
    if (request.endpoint_joint_pose_covariance)
    {
      endpoint_joint_pose_covariance =
        request.endpoint_joint_pose_covariance->covariance();
    }
    else
    {
      endpoint_joint_pose_covariance.topLeftCorner<6, 6>() =
        request.source.frozen_pose.pose_covariance;
      endpoint_joint_pose_covariance.bottomRightCorner<6, 6>() =
        request.target.pose_transfer.pose_transfer_covariance +
        request.target.pose_transfer.
          time_alignment_pose_covariance;
    }

    bool row_failure = false;
    bool all_source_adapter_jacobians_available = true;
    bool all_target_adapter_jacobians_available = true;
    PatchContributionStatus row_failure_status =
      PatchContributionStatus::kNonFinite;
    std::size_t patch_index = 0U;
    for (int row = 0;
         row < request.source.patch_width && !row_failure;
         ++row)
    {
      for (int column = 0;
           column < request.source.patch_width;
           ++column, ++patch_index)
      {
        if (
          migration.photometric.valid_mask[patch_index] ==
          std::uint8_t{0U})
        {
          continue;
        }
        const Eigen::Vector2d source_pixel =
          request.source.source_pixel +
          scale * Eigen::Vector2d(
            static_cast<double>(column) - half,
            static_cast<double>(row) - half);
        const LinearizedProjectedPoint geometry =
          migratePixelLinearized(request, source_pixel);
        if (geometry.status != MigrationStatus::kReady)
        {
          row_failure = true;
          row_failure_status =
            geometry.projection_jacobian_unavailable ?
            PatchContributionStatus::
              kProjectionJacobianUnavailable :
            PatchContributionStatus::kNonFinite;
          break;
        }
        all_source_adapter_jacobians_available =
          all_source_adapter_jacobians_available &&
          geometry.source_adapter_jacobian_available;
        all_target_adapter_jacobians_available =
          all_target_adapter_jacobians_available &&
          geometry.target_adapter_jacobian_available;
        const BilinearSample target_sample =
          sampleBilinearWithGradient(
            request.target.image.width,
            request.target.image.height,
            request.target.image.pixels,
            geometry.target_pixel);
        if (
          !target_sample.valid ||
          std::abs(
            target_sample.value -
            migration.current_patch[patch_index]) >
            1.0e-8 *
            (1.0 + std::abs(target_sample.value)))
        {
          row_failure = true;
          row_failure_status =
            PatchContributionStatus::kNonFinite;
          break;
        }

        PhotometricSample reference_sample;
        reference_sample.raw_value =
          request.source.source_patch[patch_index];
        reference_sample.pixel_x = source_pixel.x();
        reference_sample.pixel_y = source_pixel.y();
        PhotometricSample current_sample;
        current_sample.raw_value = target_sample.value;
        current_sample.pixel_x = geometry.target_pixel.x();
        current_sample.pixel_y = geometry.target_pixel.y();
        const PhotometricSampleLinearization reference_photo =
          photometric_calibration_.linearizeSample(
            request.source.endpoint.camera_id,
            request.source.photo_metadata,
            reference_sample,
            request.photometric_mode);
        const PhotometricSampleLinearization current_photo =
          photometric_calibration_.linearizeSample(
            request.target.endpoint.camera_id,
            request.target.photo_metadata,
            current_sample,
            request.photometric_mode);
        if (
          !reference_photo.valid() ||
          !current_photo.valid() ||
          (request.photometric_mode ==
             PhotometricMode::kCalibratedDirect &&
           (!reference_photo.cross_camera_comparable ||
            !current_photo.cross_camera_comparable ||
            reference_photo.experimental ||
            current_photo.experimental)))
        {
          row_failure = true;
          row_failure_status =
            PatchContributionStatus::kPhotometricUnavailable;
          break;
        }

        const double expected_current_value =
          affine_gain * current_photo.value + affine_bias;
        if (
          std::abs(
            reference_photo.value -
            migration.photometric.
              reference_values[patch_index]) >
            1.0e-8 *
            (1.0 + std::abs(reference_photo.value)) ||
          std::abs(
            expected_current_value -
            migration.photometric.current_values[patch_index]) >
            1.0e-8 *
            (1.0 + std::abs(expected_current_value)))
        {
          row_failure = true;
          row_failure_status =
            PatchContributionStatus::kPhotometricUnavailable;
          break;
        }

        const Eigen::RowVector2d radiance_wrt_target_pixel(
          current_photo.derivative_wrt_raw *
            target_sample.gradient.x() +
            current_photo.derivative_wrt_pixel_x,
          current_photo.derivative_wrt_raw *
            target_sample.gradient.y() +
            current_photo.derivative_wrt_pixel_y);
        const Eigen::Matrix<double, 1, 6>
          residual_wrt_source_pose =
          affine_gain *
          radiance_wrt_target_pixel *
          geometry.target_pixel_wrt_source_pose;
        const Eigen::Matrix<double, 1, 6>
          residual_wrt_target_pose =
          affine_gain *
          radiance_wrt_target_pixel *
          geometry.target_pixel_wrt_target_pose;
        const Eigen::RowVector3d residual_wrt_plane_point =
          affine_gain *
          radiance_wrt_target_pixel *
          geometry.target_pixel_wrt_plane_point;
        const Eigen::RowVector3d residual_wrt_plane_normal =
          affine_gain *
          radiance_wrt_target_pixel *
          geometry.target_pixel_wrt_plane_normal;
        const Eigen::RowVector2d
          residual_wrt_source_pixel =
          affine_gain *
          radiance_wrt_target_pixel *
          geometry.target_pixel_wrt_source_pixel -
          Eigen::RowVector2d(
            reference_photo.derivative_wrt_pixel_x,
            reference_photo.derivative_wrt_pixel_y);
        const Eigen::RowVector2d
          residual_wrt_target_pixel =
          affine_gain * radiance_wrt_target_pixel;
        const Eigen::Matrix<double, 1, 6>
          residual_wrt_source_extrinsic =
          affine_gain *
          radiance_wrt_target_pixel *
          geometry.target_pixel_wrt_source_extrinsic;
        const Eigen::Matrix<double, 1, 6>
          residual_wrt_target_extrinsic =
          affine_gain *
          radiance_wrt_target_pixel *
          geometry.target_pixel_wrt_target_extrinsic;

        ScalarPatchLinearization scalar;
        scalar.residual =
          expected_current_value - reference_photo.value;
        scalar.state_jacobian =
          residual_wrt_target_pose *
            request.target.pose_transfer.
              target_pose_wrt_active_anchor_state;
        scalar.affine_jacobian <<
          current_photo.value, 1.0;
        Eigen::Matrix<double, 1, 12>
          residual_wrt_endpoint_poses;
        residual_wrt_endpoint_poses <<
          residual_wrt_source_pose,
          residual_wrt_target_pose;
        const double source_raw_jacobian =
          -reference_photo.derivative_wrt_raw;
        const double target_raw_jacobian =
          affine_gain * current_photo.derivative_wrt_raw;
        const double source_raw_standard_deviation =
          request.source.raw_intensity_noise.
            standard_deviation_raw;
        const double target_raw_standard_deviation =
          request.target.raw_intensity_noise.
            standard_deviation_raw;
        const double measurement_variance =
          reference_photo.covariance_scale *
            source_raw_jacobian * source_raw_jacobian *
            source_raw_standard_deviation *
            source_raw_standard_deviation +
          current_photo.covariance_scale *
            target_raw_jacobian * target_raw_jacobian *
            target_raw_standard_deviation *
            target_raw_standard_deviation;
        scalar.variance =
          measurement_variance +
          (residual_wrt_endpoint_poses *
           endpoint_joint_pose_covariance *
           residual_wrt_endpoint_poses.transpose())(0, 0) +
          (residual_wrt_plane_point *
           request.plane.point_covariance_m2 *
           residual_wrt_plane_point.transpose())(0, 0) +
          (residual_wrt_plane_normal *
           request.plane.normal_covariance *
           residual_wrt_plane_normal.transpose())(0, 0) +
          (residual_wrt_source_pixel *
           (request.source.endpoint.uncertainty.
              observation_pixel_covariance_px2 +
            request.source.endpoint.uncertainty.
              camera_model_equivalent_pixel_covariance_px2) *
           residual_wrt_source_pixel.transpose())(0, 0) +
          (residual_wrt_target_pixel *
           (request.target.endpoint.uncertainty.
              observation_pixel_covariance_px2 +
            request.target.endpoint.uncertainty.
              camera_model_equivalent_pixel_covariance_px2) *
           residual_wrt_target_pixel.transpose())(0, 0) +
          (residual_wrt_source_extrinsic *
           request.source.endpoint.uncertainty.
             extrinsic_covariance *
           residual_wrt_source_extrinsic.transpose())(0, 0) +
          (residual_wrt_target_extrinsic *
           request.target.endpoint.uncertainty.
             extrinsic_covariance *
           residual_wrt_target_extrinsic.transpose())(0, 0);
        if (
          !std::isfinite(scalar.residual) ||
          !scalar.state_jacobian.allFinite() ||
          !scalar.affine_jacobian.allFinite() ||
          !std::isfinite(scalar.variance) ||
          scalar.variance <= 0.0)
        {
          row_failure = true;
          row_failure_status =
            PatchContributionStatus::kNonFinite;
          break;
        }
        scalar_rows.push_back(std::move(scalar));
      }
    }
    if (row_failure)
    {
      contribution.status = row_failure_status;
      contributions.push_back(std::move(contribution));
      continue;
    }
    if (
      scalar_rows.size() <
        request.gates.minimum_valid_pixels ||
      scalar_rows.size() != migration.valid_pixel_count)
    {
      contribution.status =
        PatchContributionStatus::kPhotometricUnavailable;
      contributions.push_back(std::move(contribution));
      continue;
    }

    contribution.raw_valid_pixel_count = scalar_rows.size();
    const Eigen::Index residual_count =
      static_cast<Eigen::Index>(scalar_rows.size());
    Eigen::VectorXd whitened_residual =
      Eigen::VectorXd::Zero(residual_count);
    PatchActiveStateJacobian whitened_jacobian =
      PatchActiveStateJacobian::Zero(
        residual_count,
        kCameraTimeTransferActiveStateDimension);
    Eigen::Matrix<double, Eigen::Dynamic, 2>
      whitened_affine_jacobian =
      Eigen::Matrix<double, Eigen::Dynamic, 2>::Zero(
        residual_count, 2);
    double robust_weight_sum = 0.0;
    double minimum_variance =
      std::numeric_limits<double>::infinity();
    double maximum_variance = 0.0;
    const double patch_normalization =
      1.0 / std::sqrt(
        static_cast<double>(scalar_rows.size()));
    for (Eigen::Index index = 0;
         index < residual_count; ++index)
    {
      const ScalarPatchLinearization & scalar =
        scalar_rows[static_cast<std::size_t>(index)];
      const double standard_deviation =
        std::sqrt(scalar.variance);
      const double normalized_residual =
        scalar.residual / standard_deviation;
      const double robust_weight = robustHuberWeight(
        normalized_residual,
        request.gates.
          robust_huber_delta_in_standard_deviations);
      const double square_root_weight =
        patch_normalization *
        std::sqrt(robust_weight) /
        standard_deviation;
      whitened_residual(index) =
        square_root_weight * scalar.residual;
      whitened_jacobian.row(index) =
        square_root_weight * scalar.state_jacobian;
      whitened_affine_jacobian.row(index) =
        square_root_weight * scalar.affine_jacobian;
      robust_weight_sum += robust_weight;
      minimum_variance =
        std::min(minimum_variance, scalar.variance);
      maximum_variance =
        std::max(maximum_variance, scalar.variance);
    }
    contribution.information_units =
      robust_weight_sum /
      static_cast<double>(scalar_rows.size());
    contribution.minimum_effective_variance =
      minimum_variance;
    contribution.maximum_effective_variance =
      maximum_variance;

    if (
      request.photometric_mode ==
      PhotometricMode::kCalibratedDirect)
    {
      contribution.whitened_residual =
        std::move(whitened_residual);
      contribution.whitened_jacobian =
        std::move(whitened_jacobian);
      contribution.information =
        contribution.whitened_jacobian.transpose() *
        contribution.whitened_jacobian;
      contribution.gradient =
        contribution.whitened_jacobian.transpose() *
        contribution.whitened_residual;
      contribution.normalized_squared_error =
        contribution.whitened_residual.squaredNorm();
      Eigen::SelfAdjointEigenSolver<PatchActiveStateInformation>
        information_solver(contribution.information);
      if (
        !contribution.information.allFinite() ||
        !contribution.gradient.allFinite() ||
        !std::isfinite(
          contribution.normalized_squared_error) ||
        information_solver.info() != Eigen::Success)
      {
        contribution = initializedContribution(request);
        contribution.migration_status = migration.status;
        contribution.status = PatchContributionStatus::kNonFinite;
        contributions.push_back(std::move(contribution));
        continue;
      }
      if (
        information_solver.eigenvalues().maxCoeff() <=
        request.gates.minimum_information_eigenvalue)
      {
        contribution = initializedContribution(request);
        contribution.migration_status = migration.status;
        contribution.status =
          PatchContributionStatus::kDegenerateInformation;
        contributions.push_back(std::move(contribution));
        continue;
      }
      contribution.status = PatchContributionStatus::kReady;
      contribution.production_capabilities_.
        frozen_source_prior_independence_ =
        request.source.frozen_pose.
          independentFromActivePrior(
            request.active_state_revision);
      contribution.production_capabilities_.
        joint_endpoint_pose_covariance_ =
        request.endpoint_joint_pose_covariance &&
        request.endpoint_joint_pose_covariance->
          productionReadyFor(
            request.batch_id,
            request.source_observation_id,
            request.source.frozen_pose.observation_revision,
            request.source.frozen_pose.exposure_timestamp_sec,
            request.target.exposure_timestamp_sec,
            request.target.pose_transfer.imu_epoch,
            request.active_state_revision);
      contribution.production_capabilities_.
        source_raw_intensity_noise_ =
        request.source.raw_intensity_noise.productionReady(
          request.source.
            observation_photometric_calibration_revision);
      contribution.production_capabilities_.
        target_raw_intensity_noise_ =
        request.target.raw_intensity_noise.productionReady(
          request.target.
            frame_photometric_calibration_revision);
      contribution.production_capabilities_.
        source_endpoint_uncertainty_ =
        request.source.endpoint.uncertainty.productionReady(
          request.source.endpoint);
      contribution.production_capabilities_.
        target_endpoint_uncertainty_ =
        request.target.endpoint.uncertainty.productionReady(
          request.target.endpoint);
      contribution.production_capabilities_.
        source_adapter_jacobian_ =
        all_source_adapter_jacobians_available;
      contribution.production_capabilities_.
        target_adapter_jacobian_ =
        all_target_adapter_jacobians_available;
      contribution.production_capabilities_.
        target_visibility_ =
        migration.target_visibility_validated_;
      contribution.production_capabilities_.
        photometric_quality_metadata_ =
        migration.photometric_quality_metadata_validated_;
      contribution.production_capabilities_.
        orientation_preserving_warp_ =
        std::isfinite(migration.warp_determinant) &&
        migration.warp_determinant >
          request.gates.minimum_positive_warp_determinant;
      contribution.production_capabilities_.
        gradient_direction_coverage_ =
        std::isfinite(
          migration.gradient_direction_coverage) &&
        migration.gradient_direction_coverage >=
          request.gates.minimum_gradient_direction_coverage;
      contribution.production_active_ready_ =
        contribution.production_capabilities_.all();
      contribution.shadow_only =
        !contribution.production_active_ready_;
      contribution.experimental = false;
    }
    else
    {
      const double gain_prior_information =
        1.0 /
        (request.gates.affine_gain_prior_standard_deviation *
         request.gates.affine_gain_prior_standard_deviation);
      const double bias_prior_information =
        1.0 /
        (request.gates.affine_bias_prior_standard_deviation *
         request.gates.affine_bias_prior_standard_deviation);
      const PatchActiveStateInformation hessian_state =
        whitened_jacobian.transpose() * whitened_jacobian;
      const Eigen::Matrix<
        double,
        kCameraTimeTransferActiveStateDimension, 2>
        hessian_cross =
        whitened_jacobian.transpose() *
        whitened_affine_jacobian;
      Eigen::Matrix2d hessian_affine =
        whitened_affine_jacobian.transpose() *
        whitened_affine_jacobian;
      hessian_affine(0, 0) += gain_prior_information;
      hessian_affine(1, 1) += bias_prior_information;
      const PatchActiveStateGradient gradient_state =
        whitened_jacobian.transpose() * whitened_residual;
      Eigen::Vector2d gradient_affine =
        whitened_affine_jacobian.transpose() *
        whitened_residual;
      gradient_affine.x() +=
        gain_prior_information * (affine_gain - 1.0);
      gradient_affine.y() +=
        bias_prior_information * affine_bias;
      const double squared_error =
        whitened_residual.squaredNorm() +
        gain_prior_information *
          (affine_gain - 1.0) *
          (affine_gain - 1.0) +
        bias_prior_information *
          affine_bias * affine_bias;

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>
        affine_solver(hessian_affine);
      if (
        affine_solver.info() != Eigen::Success ||
        !affine_solver.eigenvalues().allFinite() ||
        affine_solver.eigenvalues().minCoeff() <= 0.0 ||
        affine_solver.eigenvalues().maxCoeff() /
          affine_solver.eigenvalues().minCoeff() >
          request.gates.
            maximum_affine_schur_condition_number)
      {
        contribution = initializedContribution(request);
        contribution.migration_status = migration.status;
        contribution.experimental = true;
        contribution.status =
          PatchContributionStatus::kAffineIllConditioned;
        contributions.push_back(std::move(contribution));
        continue;
      }
      const Eigen::Matrix2d inverse_affine =
        hessian_affine.inverse();
      const PatchActiveStateInformation schur_information =
        hessian_state -
        hessian_cross * inverse_affine *
        hessian_cross.transpose();
      const PatchActiveStateInformation marginalized_information =
        0.5 *
        (schur_information +
         schur_information.transpose()).eval();
      const PatchActiveStateGradient marginalized_gradient =
        gradient_state -
        hessian_cross * inverse_affine * gradient_affine;
      const double marginalized_squared_error =
        squared_error -
        (gradient_affine.transpose() *
         inverse_affine * gradient_affine)(0, 0);
      if (!canonicalFactorFromNormalEquation(
          marginalized_information,
          marginalized_gradient,
          marginalized_squared_error,
          request.gates.minimum_information_eigenvalue,
          &contribution))
      {
        contribution = initializedContribution(request);
        contribution.migration_status = migration.status;
        contribution.experimental = true;
        contribution.status =
          PatchContributionStatus::kDegenerateInformation;
        contributions.push_back(std::move(contribution));
        continue;
      }
      contribution.status =
        PatchContributionStatus::kExperimentalShadowOnly;
      contribution.production_active_ready_ = false;
      contribution.shadow_only = true;
      contribution.experimental = true;
    }

    consumed_information_units.insert(information_key);
    contributions.push_back(std::move(contribution));
  }
  return contributions;
}

const char * migrationStatusName(const MigrationStatus status)
{
  switch (status)
  {
    case MigrationStatus::kReady:
      return "ready";
    case MigrationStatus::kStaleLandmark:
      return "stale_landmark";
    case MigrationStatus::kModelRevisionMismatch:
      return "model_revision_mismatch";
    case MigrationStatus::kRigRevisionMismatch:
      return "rig_revision_mismatch";
    case MigrationStatus::kPhotometricRevisionMismatch:
      return "photometric_revision_mismatch";
    case MigrationStatus::kInvalidSourceModel:
      return "invalid_source_model";
    case MigrationStatus::kInvalidTargetModel:
      return "invalid_target_model";
    case MigrationStatus::kInvalidTimeTransfer:
      return "invalid_time_transfer";
    case MigrationStatus::kPhotometricUnavailable:
      return "photometric_unavailable";
    case MigrationStatus::kInvalidPlane:
      return "invalid_plane";
    case MigrationStatus::kRayPlaneDegenerate:
      return "ray_plane_degenerate";
    case MigrationStatus::kBehindCamera:
      return "behind_camera";
    case MigrationStatus::kOutOfFrame:
      return "out_of_frame";
    case MigrationStatus::kWarpIllConditioned:
      return "warp_ill_conditioned";
    case MigrationStatus::kOccluded:
      return "occluded";
    case MigrationStatus::kLowTexture:
      return "low_texture";
    case MigrationStatus::kNccRejected:
      return "ncc_rejected";
    case MigrationStatus::kAffineIllConditioned:
      return "affine_ill_conditioned";
    case MigrationStatus::kResidualRejected:
      return "residual_rejected";
    case MigrationStatus::kNonFinite:
      return "non_finite";
  }
  return "unknown";
}

const char * patchContributionStatusName(
  const PatchContributionStatus status)
{
  switch (status)
  {
    case PatchContributionStatus::kReady:
      return "ready";
    case PatchContributionStatus::kExperimentalShadowOnly:
      return "experimental_shadow_only";
    case PatchContributionStatus::kDuplicateLandmark:
      return "duplicate_landmark";
    case PatchContributionStatus::kMigrationRejected:
      return "migration_rejected";
    case PatchContributionStatus::kProjectionJacobianUnavailable:
      return "projection_jacobian_unavailable";
    case PatchContributionStatus::kPhotometricUnavailable:
      return "photometric_unavailable";
    case PatchContributionStatus::kAffineIllConditioned:
      return "affine_ill_conditioned";
    case PatchContributionStatus::kDegenerateInformation:
      return "degenerate_information";
    case PatchContributionStatus::kNonFinite:
      return "non_finite";
  }
  return "unknown";
}

}  // namespace livo
