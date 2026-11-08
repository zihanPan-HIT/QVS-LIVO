#ifndef LIVO_FOUR_CAMERA_PINHOLE_RIG_H_
#define LIVO_FOUR_CAMERA_PINHOLE_RIG_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "pinhole_camera_projection_adapter.h"

namespace livo
{

inline constexpr std::size_t kFourCameraPinholeRigCameraCount = 4U;
inline constexpr std::size_t kFourCameraPinholeRigFingerprintSize = 32U;

using FourCameraPinholeRigFingerprint =
  std::array<
    std::uint8_t,
    kFourCameraPinholeRigFingerprintSize>;

// Raw validated input deliberately has no revision field. The owner derives
// its structural revision from a canonical SHA-256 digest.
struct FourCameraPinholeProjectionConfig
{
  std::string model;
  int width{0};
  int height{0};
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  std::array<double, 4U> distortion_coefficients{
    {0.0, 0.0, 0.0, 0.0}};
};

struct FourCameraPinholeCameraConfig
{
  std::uint8_t camera_id{
    std::numeric_limits<std::uint8_t>::max()};
  FourCameraPinholeProjectionConfig projection;

  // Matches VIOManager's p_C = R_CI * p_I + p_CI convention.
  Eigen::Matrix3d rotation_camera_from_imu{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_camera_from_imu{
    Eigen::Vector3d::Zero()};
};

struct FourCameraPinholeRigConfig
{
  std::array<
    FourCameraPinholeCameraConfig,
    kFourCameraPinholeRigCameraCount> cameras;
};

struct FourCameraPinholeRigRevisionExpectation
{
  std::uint64_t camera_model_revision{0U};
  std::uint64_t rig_revision{0U};
};

enum class FourCameraPinholeRigEndpointStatus
{
  kShadowOnlyMissingUncertaintyReceipt,
  kInvalidCameraId,
  kCameraModelRevisionMismatch,
  kRigRevisionMismatch
};

struct FourCameraPinholeRigEndpointResult
{
  FourCameraPinholeRigEndpointStatus status{
    FourCameraPinholeRigEndpointStatus::kInvalidCameraId};
  std::string reason{"invalid_camera_id"};
  std::optional<MigrationEndpoint> endpoint;

  bool shadowReady() const noexcept;
  bool productionReady() const;
};

// Read-only owner for exactly four Pinhole models and their structural
// T_CI extrinsics. Endpoint values share immutable model ownership, so queued
// copies remain valid after this rig owner is destroyed. Revisions are
// structural consistency values, not authority receipts.
class FourCameraPinholeRig
{
public:
  explicit FourCameraPinholeRig(
    const FourCameraPinholeRigConfig & config);
  ~FourCameraPinholeRig() = default;

  FourCameraPinholeRig(
    const FourCameraPinholeRig &) = delete;
  FourCameraPinholeRig(
    FourCameraPinholeRig &&) = delete;
  FourCameraPinholeRig & operator=(
    const FourCameraPinholeRig &) = delete;
  FourCameraPinholeRig & operator=(
    FourCameraPinholeRig &&) = delete;

  std::uint64_t rigRevision() const noexcept;
  // Canonical SHA-256 structural fingerprint. It detects stale content but
  // is not an authority receipt.
  FourCameraPinholeRigFingerprint
  configurationFingerprint() const noexcept;
  std::shared_ptr<const PinholeCameraProjectionAdapter> cameraModel(
    std::uint8_t camera_id) const & noexcept;
  std::shared_ptr<const PinholeCameraProjectionAdapter> cameraModel(
    std::uint8_t camera_id) const && = delete;

  FourCameraPinholeRigEndpointResult shadowEndpoint(
    std::uint8_t camera_id,
    FourCameraPinholeRigRevisionExpectation
      expected_revision) const &;
  FourCameraPinholeRigEndpointResult shadowEndpoint(
    std::uint8_t camera_id,
    FourCameraPinholeRigRevisionExpectation
      expected_revision) const && = delete;

private:
  std::array<
    std::shared_ptr<const PinholeCameraProjectionAdapter>,
    kFourCameraPinholeRigCameraCount> camera_models_;
  std::array<
    Eigen::Matrix3d,
    kFourCameraPinholeRigCameraCount>
    rotations_camera_from_imu_;
  std::array<
    Eigen::Vector3d,
    kFourCameraPinholeRigCameraCount>
    positions_camera_from_imu_;
  std::uint64_t rig_revision_{0U};
  FourCameraPinholeRigFingerprint
    configuration_fingerprint_{};
};

const char * fourCameraPinholeRigEndpointStatusName(
  FourCameraPinholeRigEndpointStatus status) noexcept;

}  // namespace livo

#endif  // LIVO_FOUR_CAMERA_PINHOLE_RIG_H_
