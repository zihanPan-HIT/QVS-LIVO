#include "camera_model_factory.h"

#include <vikit/atan_camera.h>
#include <vikit/equidistant_camera.h>
#include <vikit/omni_camera.h>
#include <vikit/pinhole_camera.h>
#include <vikit/polynomial_camera.h>

#include <algorithm>
#include <cstdlib>
#include <new>
#include <utility>

namespace
{

template<typename CameraT, typename... ArgumentTs>
CameraT *constructAlignedCamera(ArgumentTs &&...arguments)
{
  void *storage = nullptr;
  const std::size_t alignment =
    std::max<std::size_t>(alignof(CameraT), sizeof(void *));
  if (posix_memalign(&storage, alignment, sizeof(CameraT)) != 0)
  {
    throw std::bad_alloc();
  }

  try
  {
    return ::new (storage) CameraT(std::forward<ArgumentTs>(arguments)...);
  }
  catch (...)
  {
    std::free(storage);
    throw;
  }
}

template<typename CameraT>
bool destroyAlignedCamera(vk::AbstractCamera *camera) noexcept
{
  CameraT *typed_camera = dynamic_cast<CameraT *>(camera);
  if (typed_camera == nullptr)
  {
    return false;
  }
  typed_camera->~CameraT();
  std::free(typed_camera);
  return true;
}

}  // namespace

namespace livo
{

vk::AbstractCamera *createCameraModel(const CameraModelParameters &p)
{
  if (p.model == "Pinhole")
  {
    return constructAlignedCamera<vk::PinholeCamera>(
      p.width, p.height, p.scale, p.fx, p.fy, p.cx, p.cy,
      p.d0, p.d1, p.d2, p.d3, p.d4);
  }
  if (p.model == "EquidistantCamera")
  {
    return constructAlignedCamera<vk::EquidistantCamera>(
      p.width, p.height, p.scale, p.fx, p.fy, p.cx, p.cy,
      p.k1, p.k2, p.k3, p.k4);
  }
  if (p.model == "PolynomialCamera")
  {
    return constructAlignedCamera<vk::PolynomialCamera>(
      p.width, p.height, p.fx, p.fy, p.cx, p.cy, p.skew,
      p.k2, p.k3, p.k4, p.k5, p.k6, p.k7);
  }
  if (p.model == "ATAN")
  {
    return constructAlignedCamera<vk::ATANCamera>(
      p.width, p.height, p.fx, p.fy, p.cx, p.cy, p.d0);
  }
  if (p.model == "Ocam")
  {
    return constructAlignedCamera<vk::OmniCamera>(p.calib_file);
  }
  return nullptr;
}

void destroyCameraModel(vk::AbstractCamera *camera) noexcept
{
  if (camera == nullptr)
  {
    return;
  }
  if (destroyAlignedCamera<vk::PinholeCamera>(camera) ||
      destroyAlignedCamera<vk::EquidistantCamera>(camera) ||
      destroyAlignedCamera<vk::PolynomialCamera>(camera) ||
      destroyAlignedCamera<vk::ATANCamera>(camera) ||
      destroyAlignedCamera<vk::OmniCamera>(camera))
  {
    return;
  }
  delete camera;
}

void CameraModelDeleter::operator()(vk::AbstractCamera *camera) const noexcept
{
  destroyCameraModel(camera);
}

}  // namespace livo
