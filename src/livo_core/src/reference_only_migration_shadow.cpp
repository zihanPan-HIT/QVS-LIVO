#include "reference_only_migration_shadow.h"

#include "cross_camera_migration_geometry_internal.h"

#include <cmath>
#include <cstddef>
#include <limits>

#include <Eigen/Eigenvalues>

namespace livo
{
namespace
{

constexpr double kMaximumObservationAgeSec = 0.5;
constexpr int kMaximumPatchWidth = 64;
constexpr std::uint8_t kMaximumSourceLevelExclusive =
  static_cast<std::uint8_t>(std::numeric_limits<int>::digits);
constexpr std::size_t kMaximumImagePixels =
  64U * 1024U * 1024U;

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
  if (
    !covariance.allFinite() ||
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

bool sameSupport(
  const GeometrySupportKey & left,
  const GeometrySupportKey & right)
{
  return
    left.plane.id == right.plane.id &&
    left.plane.epoch == right.plane.epoch &&
    left.cell.u == right.cell.u &&
    left.cell.v == right.cell.v &&
    left.cell.level == right.cell.level;
}

bool sameVersion(
  const GeometryVersion & left,
  const GeometryVersion & right)
{
  return
    left.plane.id == right.plane.id &&
    left.plane.epoch == right.plane.epoch &&
    left.revision == right.revision;
}

bool validImageBounds(
  const ReferenceOnlyMigrationTargetImageBounds & bounds)
{
  if (bounds.width <= 1 || bounds.height <= 1)
  {
    return false;
  }
  const std::size_t width =
    static_cast<std::size_t>(bounds.width);
  const std::size_t height =
    static_cast<std::size_t>(bounds.height);
  return
    width <= kMaximumImagePixels &&
    height <= kMaximumImagePixels / width;
}

}  // namespace

ReferenceOnlyMigrationShadowResult
evaluateReferenceOnlyMigrationShadow(
  const ReferenceOnlyMigrationShadowRequest & request)
{
  ReferenceOnlyMigrationShadowResult result;
  const auto reject =
    [&result](const MigrationStatus status)
    {
      result.status_ =
        ReferenceOnlyMigrationShadowStatus::kRejected;
      result.migration_status_ = status;
      return result;
    };

  if (
    !request.source_support_key.valid() ||
    !request.source_geometry_version.valid() ||
    !request.plane.support_key.valid() ||
    !request.plane.geometry_version.valid() ||
    !sameSupport(
      request.source_support_key,
      request.plane.support_key) ||
    !sameVersion(
      request.source_geometry_version,
      request.plane.geometry_version) ||
    request.source_support_key.plane.id !=
      request.source_geometry_version.plane.id ||
    request.source_support_key.plane.epoch !=
      request.source_geometry_version.plane.epoch)
  {
    return reject(MigrationStatus::kStaleLandmark);
  }

  if (
    request.source_endpoint.camera_id >= 4U ||
    request.target_endpoint.camera_id >= 4U ||
    request.source_endpoint.camera_id ==
      request.target_endpoint.camera_id ||
    request.source_endpoint.camera_model == nullptr ||
    !request.source_endpoint.camera_model->valid())
  {
    return reject(MigrationStatus::kInvalidSourceModel);
  }
  if (
    request.target_endpoint.camera_model == nullptr ||
    !request.target_endpoint.camera_model->valid())
  {
    return reject(MigrationStatus::kInvalidTargetModel);
  }
  if (
    !request.source_endpoint.modelRevisionMatchesAdapter() ||
    !request.target_endpoint.modelRevisionMatchesAdapter())
  {
    return reject(MigrationStatus::kModelRevisionMismatch);
  }
  if (
    request.source_endpoint.rig_revision == 0U ||
    request.source_endpoint.rig_revision !=
      request.target_endpoint.rig_revision)
  {
    return reject(MigrationStatus::kRigRevisionMismatch);
  }

  if (
    !request.target_time.receipt.valid() ||
    !request.target_time.entry.has_value() ||
    !request.target_time.receipt.accepts(
      request.target_time.entry) ||
    request.target_time.entry.cameraId() !=
      request.target_endpoint.camera_id)
  {
    return reject(MigrationStatus::kInvalidTimeTransfer);
  }
  const CameraPoseTransferRaw & target_transfer =
    request.target_time.entry.value().raw();
  if (
    !target_transfer.internallyValid() ||
    target_transfer.target_timestamp_sec !=
      request.target_time.entry.targetTimestampSec() ||
    !std::isfinite(
      request.source_pose.exposure_timestamp_sec) ||
    request.source_pose.exposure_timestamp_sec <= 0.0 ||
    target_transfer.target_timestamp_sec <
      request.source_pose.exposure_timestamp_sec ||
    target_transfer.target_timestamp_sec -
      request.source_pose.exposure_timestamp_sec >
      kMaximumObservationAgeSec)
  {
    return reject(MigrationStatus::kInvalidTimeTransfer);
  }

  if (
    !validRotation(
      request.source_pose.rotation_world_from_imu) ||
    !request.source_pose.position_world_m.allFinite() ||
    !positiveSemidefinite(
      request.source_pose.pose_covariance) ||
    !validRotation(
      target_transfer.rotation_world_from_imu) ||
    !target_transfer.position_world_m.allFinite() ||
    !validRotation(
      request.source_endpoint.rotation_imu_from_camera) ||
    !request.source_endpoint.position_imu_from_camera.allFinite() ||
    !validRotation(
      request.target_endpoint.rotation_imu_from_camera) ||
    !request.target_endpoint.position_imu_from_camera.allFinite())
  {
    return reject(MigrationStatus::kNonFinite);
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
    return reject(MigrationStatus::kInvalidPlane);
  }

  if (
    !request.patch.source_pixel.allFinite() ||
    request.patch.patch_width <= 1 ||
    request.patch.patch_width > kMaximumPatchWidth ||
    request.patch.source_level >=
      kMaximumSourceLevelExclusive ||
    (request.target_image_bounds &&
     !validImageBounds(*request.target_image_bounds)))
  {
    return reject(MigrationStatus::kNonFinite);
  }

  migration_geometry_internal::GeometryProjectionRequest
    geometry_request;
  geometry_request.source_endpoint =
    &request.source_endpoint;
  geometry_request.target_endpoint =
    &request.target_endpoint;
  geometry_request.source_pose =
    migration_geometry_internal::cameraPose(
    request.source_pose.rotation_world_from_imu,
    request.source_pose.position_world_m,
    request.source_endpoint);
  geometry_request.target_pose =
    migration_geometry_internal::cameraPose(
    target_transfer.rotation_world_from_imu,
    target_transfer.position_world_m,
    request.target_endpoint);
  geometry_request.plane_point_world_m =
    request.plane.point_world_m;
  geometry_request.plane_normal_world =
    request.plane.normal_world.normalized();
  geometry_request.source_center_pixel =
    request.patch.source_pixel;
  geometry_request.source_level =
    request.patch.source_level;
  geometry_request.patch_width =
    request.patch.patch_width;
  if (request.target_image_bounds)
  {
    geometry_request.target_image_bounds =
      migration_geometry_internal::ImageBounds{
      request.target_image_bounds->width,
      request.target_image_bounds->height};
  }

  const migration_geometry_internal::ProjectedPatch projected =
    migration_geometry_internal::projectPatch(
    geometry_request);
  result.migration_status_ =
    migration_geometry_internal::validateWarp(
    projected, geometry_request.gates);
  result.target_pixel_ = projected.target_center_pixel;
  result.warp_condition_number_ =
    projected.warp_condition_number;
  result.valid_projected_pixel_count_ =
    projected.valid_projected_pixel_count;
  if (result.migration_status_ != MigrationStatus::kReady)
  {
    return result;
  }

  result.status_ =
    ReferenceOnlyMigrationShadowStatus::
    kGeometryReadyVisibilityUnverified;
  return result;
}

}  // namespace livo
