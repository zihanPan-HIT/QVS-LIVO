#include "camera_observation_batch_authority.h"

#include "camera_observation.h"
#include "frozen_camera_image.h"

#include <array>
#include <algorithm>
#include <omp.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace livo
{
namespace
{

struct CameraObservationAuthorityTag final
{
};

using ImageDigest = std::array<unsigned char, 32U>;

template<typename Left, typename Right>
bool sameSharedOwner(
  const std::shared_ptr<Left> & lhs,
  const std::shared_ptr<Right> & rhs) noexcept
{
  return
    lhs && rhs &&
    !lhs.owner_before(rhs) &&
    !rhs.owner_before(lhs);
}

template<typename T>
bool exactFiniteEqual(const T lhs, const T rhs) noexcept
{
  static_assert(std::is_floating_point_v<T>);
  return std::isfinite(lhs) && std::isfinite(rhs) && lhs == rhs;
}

}  // namespace

struct CameraObservationBatchReceipt::SealedData
{
  struct ObservationBinding
  {
    bool present{false};
    std::uint8_t camera_id{0U};
    double timestamp_sec{0.0};
    std::uint64_t observation_generation{0U};
    std::uint64_t image_generation{0U};
    int rows{0};
    int columns{0};
    int type{0};
    PhotometricStatus photometric_status{
      PhotometricStatus::kInvalidInput};
    ImageDigest image_digest{};
  };

  std::shared_ptr<const void> authority;
  std::uint64_t batch_sequence{0U};
  std::array<bool, kCameraObservationAuthorityCameraCount> present{{
      false, false, false, false}};
  std::size_t camera_count{0U};
  bool partial{false};
  double timestamp_sec{0.0};
  double max_time_offset_sec{0.0};
  std::array<
    ObservationBinding,
    kCameraObservationAuthorityCameraCount> observations;
  std::uint64_t seal_image_hash_count{0U};
  mutable std::atomic<bool>
    runtime_validation_claimed{false};
  mutable std::atomic<bool>
    adaptive_visual_selection_claimed{false};
  mutable std::atomic<std::uint64_t>
    validation_image_hash_count{0U};
};

CameraObservationBatchAuthorityIdentity::
CameraObservationBatchAuthorityIdentity(
  std::shared_ptr<const void> authority)
: authority_(std::move(authority))
{
}

CameraObservationBatchAuthorityIdentity
CameraObservationBatchAuthorityIdentity::issue()
{
  return CameraObservationBatchAuthorityIdentity{
    std::make_shared<const CameraObservationAuthorityTag>()};
}

bool CameraObservationBatchAuthorityIdentity::valid() const noexcept
{
  return static_cast<bool>(authority_);
}

bool CameraObservationBatchAuthorityIdentity::sameAuthority(
  const CameraObservationBatchAuthorityIdentity & other) const noexcept
{
  return sameSharedOwner(authority_, other.authority_);
}

CameraObservationBatchReceipt::CameraObservationBatchReceipt(
  std::shared_ptr<const SealedData> sealed_data)
: sealed_data_(std::move(sealed_data))
{
}

bool CameraObservationBatchReceipt::freezeAndDigestImage(
  const cv::Mat & source,
  cv::Mat & frozen,
  std::array<unsigned char, 32U> & digest) noexcept
{
  const FrozenCameraImageFreezeResult result =
    FrozenCameraImage::freeze(source);
  if (!result.ready())
  {
    frozen.release();
    return false;
  }
  frozen = result.image->imageForAuthority();
  digest = result.image->digest();
  return true;
}

std::optional<CameraObservationBatchReceipt>
CameraObservationBatchReceipt::seal(
  const CameraObservationBatch & batch,
  const std::uint64_t batch_sequence,
  const CameraObservationBatchAuthorityIdentity & identity)
{
  if (
    batch_sequence == 0U ||
    !identity.valid() ||
    !std::isfinite(batch.timestamp) ||
    !std::isfinite(batch.max_time_offset_sec) ||
    batch.max_time_offset_sec < 0.0 ||
    batch.camera_count == 0U ||
    batch.camera_count > kCameraObservationAuthorityCameraCount)
  {
    return std::nullopt;
  }

  auto sealed = std::make_shared<SealedData>();
  sealed->authority = identity.authority_;
  sealed->batch_sequence = batch_sequence;
  sealed->present = batch.present;
  sealed->camera_count = batch.camera_count;
  sealed->partial = batch.partial;
  sealed->timestamp_sec = batch.timestamp;
  sealed->max_time_offset_sec = batch.max_time_offset_sec;

  std::size_t present_count = 0U;
  for (std::size_t camera_id = 0U;
       camera_id < kCameraObservationAuthorityCameraCount;
       ++camera_id)
  {
    auto & binding = sealed->observations[camera_id];
    binding.present = batch.present[camera_id];
    if (!binding.present) {
      continue;
    }
    ++present_count;

    const CameraObservation & observation =
      batch.observations[camera_id];
    if (
      observation.camera_id != camera_id ||
      observation.observation_generation_ == 0U ||
      observation.image_generation_ == 0U ||
      !std::isfinite(observation.timestamp) ||
      observation.image.empty() ||
      !observation.sealed_image_digest_valid_)
    {
      return std::nullopt;
    }
    binding.camera_id = observation.camera_id;
    binding.timestamp_sec = observation.timestamp;
    binding.observation_generation =
      observation.observation_generation_;
    binding.image_generation = observation.image_generation_;
    binding.rows = observation.image.rows;
    binding.columns = observation.image.cols;
    binding.type = observation.image.type();
    binding.photometric_status =
      observation.photometric_status;
    binding.image_digest =
      observation.sealed_image_digest_;
    ++sealed->seal_image_hash_count;
  }

  if (
    present_count != batch.camera_count ||
    batch.partial !=
      (batch.camera_count <
       kCameraObservationAuthorityCameraCount))
  {
    return std::nullopt;
  }
  return CameraObservationBatchReceipt{std::move(sealed)};
}

bool CameraObservationBatchReceipt::valid() const noexcept
{
  return
    sealed_data_ &&
    sealed_data_->authority &&
    sealed_data_->batch_sequence != 0U;
}

bool CameraObservationBatchReceipt::issuedBy(
  const CameraObservationBatchAuthorityIdentity & identity) const noexcept
{
  return
    valid() &&
    sameSharedOwner(
      sealed_data_->authority,
      identity.authority_);
}

bool CameraObservationBatchReceipt::claimAndValidate(
  const CameraObservationBatch & batch) const noexcept
{
  if (!valid()) {
    return false;
  }
  bool expected = false;
  if (
    !sealed_data_->runtime_validation_claimed.
    compare_exchange_strong(
      expected, true, std::memory_order_acq_rel))
  {
    return false;
  }
  return matchesAndValidates(batch);
}

bool CameraObservationBatchReceipt::
claimAdaptiveVisualSelection() const noexcept
{
  if (!valid()) {
    return false;
  }
  bool expected = false;
  return
    sealed_data_->adaptive_visual_selection_claimed.
    compare_exchange_strong(
      expected, true, std::memory_order_acq_rel);
}

bool CameraObservationBatchReceipt::sameSealedBatch(
  const CameraObservationBatchReceipt & other) const noexcept
{
  return
    valid() &&
    other.valid() &&
    sameSharedOwner(sealed_data_, other.sealed_data_);
}

const std::array<bool, kCameraObservationAuthorityCameraCount> &
CameraObservationBatchReceipt::sealedPresentMask() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid camera observation batch receipt");
  }
  return sealed_data_->present;
}

std::size_t
CameraObservationBatchReceipt::sealedCameraCount() const noexcept
{
  return valid() ? sealed_data_->camera_count : 0U;
}

bool CameraObservationBatchReceipt::sealedPartial() const noexcept
{
  return valid() && sealed_data_->partial;
}

double
CameraObservationBatchReceipt::sealedBatchTimestampSec() const noexcept
{
  return
    valid() ?
    sealed_data_->timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

double
CameraObservationBatchReceipt::sealedMaxTimeOffsetSec() const noexcept
{
  return
    valid() ?
    sealed_data_->max_time_offset_sec :
    std::numeric_limits<double>::quiet_NaN();
}

double
CameraObservationBatchReceipt::sealedExposureTimestampSec(
  const std::uint8_t camera_id) const noexcept
{
  if (
    !valid() ||
    camera_id >= sealed_data_->observations.size() ||
    !sealed_data_->observations[camera_id].present)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return
    sealed_data_->observations[camera_id].timestamp_sec;
}

bool CameraObservationBatchReceipt::matchesAndValidates(
  const CameraObservationBatch & batch) const noexcept
{
  if (!valid()) {
    return false;
  }
  if (
    !batch.validated_receipt_.has_value() ||
    !sameSharedOwner(
      sealed_data_,
      batch.validated_receipt_->sealed_data_) ||
    batch.present != sealed_data_->present ||
    batch.camera_count != sealed_data_->camera_count ||
    batch.partial != sealed_data_->partial ||
    !exactFiniteEqual(
      batch.timestamp, sealed_data_->timestamp_sec) ||
    !exactFiniteEqual(
      batch.max_time_offset_sec,
      sealed_data_->max_time_offset_sec))
  {
    return false;
  }

  std::array<bool, kCameraObservationAuthorityCameraCount> valid_images;
  valid_images.fill(true);
  const int threads = std::clamp(omp_get_max_threads(), 1, 4);
  #pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1 && batch.camera_count > 1)
  for (std::size_t camera_id = 0U;
       camera_id < kCameraObservationAuthorityCameraCount;
       ++camera_id)
  {
    const SealedData::ObservationBinding & binding =
      sealed_data_->observations[camera_id];
    if (!binding.present) {
      continue;
    }

    const CameraObservation & observation =
      batch.observations[camera_id];
    ImageDigest current_digest{};
    if (
      observation.camera_id != binding.camera_id ||
      !exactFiniteEqual(
        observation.timestamp, binding.timestamp_sec) ||
      observation.observation_generation_ !=
        binding.observation_generation ||
      observation.image_generation_ !=
        binding.image_generation ||
      observation.image.rows != binding.rows ||
      observation.image.cols != binding.columns ||
      observation.image.type() != binding.type ||
      observation.photometric_status !=
        binding.photometric_status ||
      !FrozenCameraImage::digestForAuthority(
        observation.image, current_digest))
    {
      valid_images[camera_id] = false;
      continue;
    }
    sealed_data_->validation_image_hash_count.fetch_add(
      1U, std::memory_order_relaxed);
    if (current_digest != binding.image_digest) {
      valid_images[camera_id] = false;
      continue;
    }
  }
  return std::all_of(valid_images.begin(), valid_images.end(), [](bool valid) { return valid; });
}

std::uint64_t
CameraObservationBatchReceipt::batchSequence() const noexcept
{
  return valid() ? sealed_data_->batch_sequence : 0U;
}

std::uint64_t
CameraObservationBatchReceipt::observationGeneration(
  const std::uint8_t camera_id) const noexcept
{
  if (
    !valid() ||
    camera_id >= kCameraObservationAuthorityCameraCount ||
    !sealed_data_->observations[camera_id].present)
  {
    return 0U;
  }
  return
    sealed_data_->observations[camera_id].
    observation_generation;
}

std::uint64_t
CameraObservationBatchReceipt::imageGeneration(
  const std::uint8_t camera_id) const noexcept
{
  if (
    !valid() ||
    camera_id >= kCameraObservationAuthorityCameraCount ||
    !sealed_data_->observations[camera_id].present)
  {
    return 0U;
  }
  return
    sealed_data_->observations[camera_id].image_generation;
}

double
CameraObservationBatchReceipt::cameraExposureTimestampSec(
  const std::uint8_t camera_id) const noexcept
{
  return sealedExposureTimestampSec(camera_id);
}

std::uint64_t
CameraObservationBatchReceipt::sealImageHashCount() const noexcept
{
  return valid() ? sealed_data_->seal_image_hash_count : 0U;
}

bool CameraObservationBatchReceipt::
adaptiveVisualSelectionClaimed() const noexcept
{
  return
    valid() &&
    sealed_data_->adaptive_visual_selection_claimed.load(
      std::memory_order_acquire);
}

std::uint64_t
CameraObservationBatchReceipt::validationImageHashCount() const noexcept
{
  return
    valid() ?
    sealed_data_->validation_image_hash_count.load(
      std::memory_order_relaxed) :
    0U;
}

}  // namespace livo
