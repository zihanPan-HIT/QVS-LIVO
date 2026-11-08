#ifndef LIVO_PINHOLE_CAMERA_PROJECTION_ADAPTER_H_
#define LIVO_PINHOLE_CAMERA_PROJECTION_ADAPTER_H_

#include <array>
#include <cstdint>
#include <string>

#include "cross_camera_patch_migration.h"

namespace livo
{

struct PinholeCameraProjectionConfig
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
  std::uint64_t model_revision{0U};
};

// Immutable zero-distortion Pinhole value object. Construction proves that
// every ray in the declared image has finite components and norm. unproject()
// returns the unnormalized z=1 ray required by CrossCameraPatchMigration's
// normalization chain. project() does not classify image containment;
// contains() owns that half-open-frame contract. model_revision is a version
// value, not an authority receipt.
class PinholeCameraProjectionAdapter final :
  public CameraProjectionAdapter
{
public:
  explicit PinholeCameraProjectionAdapter(
    const PinholeCameraProjectionConfig & config);

  PinholeCameraProjectionAdapter(
    const PinholeCameraProjectionAdapter &) = default;
  PinholeCameraProjectionAdapter(
    PinholeCameraProjectionAdapter &&) = default;
  PinholeCameraProjectionAdapter & operator=(
    const PinholeCameraProjectionAdapter &) = delete;
  PinholeCameraProjectionAdapter & operator=(
    PinholeCameraProjectionAdapter &&) = delete;

  bool valid() const override;
  CameraRayResult unproject(
    const Eigen::Vector2d & pixel) const override;
  CameraRayJacobianResult unprojectJacobian(
    const Eigen::Vector2d & pixel) const override;
  CameraProjectionResult project(
    const Eigen::Vector3d & point_camera) const override;
  CameraProjectionJacobianResult projectJacobian(
    const Eigen::Vector3d & point_camera) const override;
  bool contains(
    const Eigen::Vector2d & pixel,
    int border) const override;

  int width() const noexcept;
  int height() const noexcept;
  double fx() const noexcept;
  double fy() const noexcept;
  double cx() const noexcept;
  double cy() const noexcept;
  std::array<double, 4U>
  distortionCoefficients() const noexcept;
  std::uint64_t modelRevision() const noexcept override;

private:
  ProjectionAdapterStatus pixelStatus(
    const Eigen::Vector2d & pixel) const;

  const int width_;
  const int height_;
  const double fx_;
  const double fy_;
  const double cx_;
  const double cy_;
  const std::array<double, 4U> distortion_coefficients_;
  const std::uint64_t model_revision_;
};

}  // namespace livo

#endif  // LIVO_PINHOLE_CAMERA_PROJECTION_ADAPTER_H_
