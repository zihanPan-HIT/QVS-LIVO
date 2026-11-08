#ifndef LIVO_FROZEN_CAMERA_IMAGE_H_
#define LIVO_FROZEN_CAMERA_IMAGE_H_

#include <array>
#include <memory>
#include <optional>

#include <opencv2/core/mat.hpp>

namespace livo
{

class AdaptiveCameraSnapshotQueue;
class CameraObservationBatchReceipt;
class FrozenCameraImage;

enum class FrozenCameraImageFreezeStatus
{
  READY,
  INVALID_INPUT,
  RESOURCE_FAILURE
};

struct FrozenCameraImageFreezeResult;

// Publicly opaque, value-owned frozen image metadata. Authority-internal
// consumers may obtain a read-only cv::Mat view of the single cloned storage.
class FrozenCameraImage final
{
public:
  using Digest = std::array<unsigned char, 32U>;

  static FrozenCameraImageFreezeResult freeze(
    const cv::Mat & source) noexcept;

  FrozenCameraImage(const FrozenCameraImage &) noexcept = default;
  FrozenCameraImage(FrozenCameraImage &&) noexcept = default;
  FrozenCameraImage & operator=(
    const FrozenCameraImage &) noexcept = default;
  FrozenCameraImage & operator=(
    FrozenCameraImage &&) noexcept = default;

  bool valid() const noexcept;
  int rows() const noexcept;
  int columns() const noexcept;
  int type() const noexcept;
  Digest digest() const noexcept;

private:
  struct Storage;

  explicit FrozenCameraImage(
    std::shared_ptr<const Storage> storage) noexcept;

  static bool digestForAuthority(
    const cv::Mat & image,
    Digest & digest) noexcept;
  const cv::Mat & imageForAuthority() const noexcept;

  std::shared_ptr<const Storage> storage_;

  friend class AdaptiveCameraSnapshotQueue;
  friend class CameraObservationBatchReceipt;
};

struct FrozenCameraImageFreezeResult
{
  FrozenCameraImageFreezeStatus status{
    FrozenCameraImageFreezeStatus::INVALID_INPUT};
  std::optional<FrozenCameraImage> image;

  bool ready() const noexcept
  {
    return
      status == FrozenCameraImageFreezeStatus::READY &&
      image.has_value() &&
      image->valid();
  }
};

}  // namespace livo

#endif  // LIVO_FROZEN_CAMERA_IMAGE_H_
