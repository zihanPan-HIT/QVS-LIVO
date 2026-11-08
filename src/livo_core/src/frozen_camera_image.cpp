#include "frozen_camera_image.h"

#include <openssl/evp.h>

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

#ifdef LIVO_FROZEN_CAMERA_IMAGE_FAILURE_INJECTION_TESTING
#include <atomic>
#endif

namespace livo
{
namespace
{

#ifdef LIVO_FROZEN_CAMERA_IMAGE_FAILURE_INJECTION_TESTING
std::atomic<bool> fail_next_storage_allocation{false};
#endif

bool validImageShape(const cv::Mat & image) noexcept
{
  return
    !image.empty() &&
    image.dims == 2 &&
    image.rows > 0 &&
    image.cols > 0;
}

bool digestImage(
  const cv::Mat & image,
  FrozenCameraImage::Digest & digest) noexcept
{
  if (!validImageShape(image)) {
    return false;
  }

  EVP_MD_CTX * context = EVP_MD_CTX_new();
  if (context == nullptr) {
    return false;
  }

  bool valid =
    EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
  const std::array<std::int32_t, 3U> metadata{{
      image.rows, image.cols, image.type()}};
  valid =
    valid &&
    EVP_DigestUpdate(
      context, metadata.data(), sizeof(metadata)) == 1;

  const std::size_t row_bytes =
    static_cast<std::size_t>(image.cols) * image.elemSize();
  if (image.isContinuous())
  {
    valid =
      valid && image.data != nullptr &&
      EVP_DigestUpdate(
        context, image.data,
        row_bytes * static_cast<std::size_t>(image.rows)) == 1;
  }
  else
  {
    for (int row = 0; valid && row < image.rows; ++row)
    {
      const unsigned char * row_data = image.ptr<unsigned char>(row);
      valid =
        row_data != nullptr &&
        EVP_DigestUpdate(context, row_data, row_bytes) == 1;
    }
  }

  unsigned int digest_length = 0U;
  valid =
    valid &&
    EVP_DigestFinal_ex(
      context, digest.data(), &digest_length) == 1 &&
    digest_length == digest.size();
  EVP_MD_CTX_free(context);
  return valid;
}

}  // namespace

struct FrozenCameraImage::Storage
{
  Storage(cv::Mat frozen_image, const Digest & frozen_digest)
  : image(std::move(frozen_image)),
    digest(frozen_digest)
  {
  }

  const cv::Mat image;
  const Digest digest;
};

FrozenCameraImage::FrozenCameraImage(
  std::shared_ptr<const Storage> storage) noexcept
: storage_(std::move(storage))
{
}

FrozenCameraImageFreezeResult FrozenCameraImage::freeze(
  const cv::Mat & source) noexcept
{
  if (!validImageShape(source))
  {
    return {
      FrozenCameraImageFreezeStatus::INVALID_INPUT,
      std::nullopt};
  }

  try
  {
    cv::Mat frozen = source.clone();
    Digest digest{};
    if (!digestImage(frozen, digest))
    {
      return {
        FrozenCameraImageFreezeStatus::RESOURCE_FAILURE,
        std::nullopt};
    }

#ifdef LIVO_FROZEN_CAMERA_IMAGE_FAILURE_INJECTION_TESTING
    if (fail_next_storage_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
#endif
    auto storage =
      std::make_shared<const Storage>(
      std::move(frozen), digest);
    return {
      FrozenCameraImageFreezeStatus::READY,
      FrozenCameraImage{std::move(storage)}};
  }
  catch (const cv::Exception & exception)
  {
    const FrozenCameraImageFreezeStatus status =
      exception.code == cv::Error::StsNoMem ?
      FrozenCameraImageFreezeStatus::RESOURCE_FAILURE :
      FrozenCameraImageFreezeStatus::INVALID_INPUT;
    return {status, std::nullopt};
  }
  catch (const std::bad_alloc &)
  {
    return {
      FrozenCameraImageFreezeStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  catch (...)
  {
    return {
      FrozenCameraImageFreezeStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
}

#ifdef LIVO_FROZEN_CAMERA_IMAGE_FAILURE_INJECTION_TESTING
namespace test
{

void failNextFrozenCameraImageStorageAllocation()
{
  fail_next_storage_allocation.store(
    true, std::memory_order_release);
}

}  // namespace test
#endif

bool FrozenCameraImage::valid() const noexcept
{
  return static_cast<bool>(storage_);
}

int FrozenCameraImage::rows() const noexcept
{
  return valid() ? storage_->image.rows : 0;
}

int FrozenCameraImage::columns() const noexcept
{
  return valid() ? storage_->image.cols : 0;
}

int FrozenCameraImage::type() const noexcept
{
  return valid() ? storage_->image.type() : 0;
}

FrozenCameraImage::Digest
FrozenCameraImage::digest() const noexcept
{
  return valid() ? storage_->digest : Digest{};
}

bool FrozenCameraImage::digestForAuthority(
  const cv::Mat & image,
  Digest & digest) noexcept
{
  return digestImage(image, digest);
}

const cv::Mat &
FrozenCameraImage::imageForAuthority() const noexcept
{
  static const cv::Mat empty;
  return valid() ? storage_->image : empty;
}

}  // namespace livo
