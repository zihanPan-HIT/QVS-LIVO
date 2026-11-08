#ifndef LIVO_CAMERA_MODEL_FACTORY_H_
#define LIVO_CAMERA_MODEL_FACTORY_H_

#include <memory>
#include <string>

#include <vikit/abstract_camera.h>

namespace livo
{

struct CameraModelParameters
{
  std::string model;
  std::string calib_file;
  int width = 0;
  int height = 0;
  double scale = 1.0;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double skew = 0.0;
  double d0 = 0.0;
  double d1 = 0.0;
  double d2 = 0.0;
  double d3 = 0.0;
  double d4 = 0.0;
  double k1 = 0.0;
  double k2 = 0.0;
  double k3 = 0.0;
  double k4 = 0.0;
  double k5 = 0.0;
  double k6 = 0.0;
  double k7 = 0.0;
};

// vikit_common is compiled as C++14 and its Eigen-aligned camera allocation ABI
// differs from C++17. Keep construction and destruction in this C++14 adapter.
vk::AbstractCamera *createCameraModel(const CameraModelParameters &parameters);
void destroyCameraModel(vk::AbstractCamera *camera) noexcept;

struct CameraModelDeleter
{
  void operator()(vk::AbstractCamera *camera) const noexcept;
};

using CameraModelPtr =
  std::unique_ptr<vk::AbstractCamera, CameraModelDeleter>;

}  // namespace livo

#endif  // LIVO_CAMERA_MODEL_FACTORY_H_
