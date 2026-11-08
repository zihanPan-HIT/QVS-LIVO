#ifndef LIVO_REFERENCE_ONLY_MIGRATION_SHADOW_H_
#define LIVO_REFERENCE_ONLY_MIGRATION_SHADOW_H_

#include "camera_batch_time_transfer_receipt.h"
#include "cross_camera_patch_migration.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include <Eigen/Core>

namespace livo
{

enum class ReferenceOnlyMigrationSourceKind
{
  kReferenceOnlyBootstrap
};

enum class ReferenceOnlyMigrationShadowStatus
{
  kRejected,
  kGeometryReadyVisibilityUnverified
};

enum class ReferenceOnlyMigrationVisibility
{
  kUnverified
};

struct ReferenceOnlyMigrationSourcePose
{
  double exposure_timestamp_sec{
    std::numeric_limits<double>::quiet_NaN()};
  Eigen::Matrix3d rotation_world_from_imu{
    Eigen::Matrix3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Vector3d position_world_m{
    Eigen::Vector3d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  Eigen::Matrix<double, 6, 6> pose_covariance{
    Eigen::Matrix<double, 6, 6>::Constant(
      std::numeric_limits<double>::quiet_NaN())};
};

struct ReferenceOnlyMigrationPatchGeometry
{
  Eigen::Vector2d source_pixel{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  std::uint8_t source_level{0U};
  int patch_width{0};
};

struct ReferenceOnlyMigrationTargetImageBounds
{
  int width{0};
  int height{0};
};

struct ReferenceOnlyMigrationTargetTime
{
  ReferenceOnlyMigrationTargetTime(
    const CameraBatchTimeTransferReceipt & receipt_value,
    const CameraBatchTimeTransferEntry & entry_value)
  : receipt(receipt_value), entry(entry_value)
  {
  }

  CameraBatchTimeTransferReceipt receipt;
  CameraBatchTimeTransferEntry entry;
};

struct ReferenceOnlyMigrationShadowRequest
{
  explicit ReferenceOnlyMigrationShadowRequest(
    ReferenceOnlyMigrationTargetTime target_time_value)
  : target_time(std::move(target_time_value))
  {
  }

  static constexpr ReferenceOnlyMigrationSourceKind sourceKind() noexcept
  {
    return
      ReferenceOnlyMigrationSourceKind::
      kReferenceOnlyBootstrap;
  }

  GeometrySupportKey source_support_key;
  GeometryVersion source_geometry_version;
  MigrationEndpoint source_endpoint;
  ReferenceOnlyMigrationSourcePose source_pose;
  ReferenceOnlyMigrationPatchGeometry patch;
  MigrationEndpoint target_endpoint;
  ReferenceOnlyMigrationTargetTime target_time;
  std::optional<ReferenceOnlyMigrationTargetImageBounds>
    target_image_bounds;
  MigrationPlane plane;
};

class ReferenceOnlyMigrationShadowResult;

ReferenceOnlyMigrationShadowResult
evaluateReferenceOnlyMigrationShadow(
  const ReferenceOnlyMigrationShadowRequest & request);

class ReferenceOnlyMigrationShadowResult
{
public:
  ReferenceOnlyMigrationShadowResult(
    const ReferenceOnlyMigrationShadowResult &) = default;
  ReferenceOnlyMigrationShadowResult(
    ReferenceOnlyMigrationShadowResult &&) = default;
  ReferenceOnlyMigrationShadowResult & operator=(
    const ReferenceOnlyMigrationShadowResult &) = default;
  ReferenceOnlyMigrationShadowResult & operator=(
    ReferenceOnlyMigrationShadowResult &&) = default;
  ~ReferenceOnlyMigrationShadowResult() = default;

  ReferenceOnlyMigrationShadowStatus status() const noexcept
  {
    return status_;
  }

  MigrationStatus migrationStatus() const noexcept
  {
    return migration_status_;
  }

  const Eigen::Vector2d & targetPixel() const noexcept
  {
    return target_pixel_;
  }

  double warpConditionNumber() const noexcept
  {
    return warp_condition_number_;
  }

  std::size_t validProjectedPixelCount() const noexcept
  {
    return valid_projected_pixel_count_;
  }

  static constexpr ReferenceOnlyMigrationVisibility
  visibility() noexcept
  {
    return ReferenceOnlyMigrationVisibility::kUnverified;
  }

  static constexpr bool experimental() noexcept
  {
    return true;
  }

  static constexpr bool consumable() noexcept
  {
    return false;
  }

  static constexpr bool productionReady() noexcept
  {
    return false;
  }

  static constexpr bool solverApplied() noexcept
  {
    return false;
  }

  static constexpr bool acceptanceReceiptIssued() noexcept
  {
    return false;
  }

  static constexpr bool independenceProofIssued() noexcept
  {
    return false;
  }

private:
  ReferenceOnlyMigrationShadowResult() = default;

  ReferenceOnlyMigrationShadowStatus status_{
    ReferenceOnlyMigrationShadowStatus::kRejected};
  MigrationStatus migration_status_{
    MigrationStatus::kNonFinite};
  Eigen::Vector2d target_pixel_{
    Eigen::Vector2d::Constant(
      std::numeric_limits<double>::quiet_NaN())};
  double warp_condition_number_{
    std::numeric_limits<double>::infinity()};
  std::size_t valid_projected_pixel_count_{0U};

  friend ReferenceOnlyMigrationShadowResult
  evaluateReferenceOnlyMigrationShadow(
    const ReferenceOnlyMigrationShadowRequest & request);
};

}  // namespace livo

#endif  // LIVO_REFERENCE_ONLY_MIGRATION_SHADOW_H_
