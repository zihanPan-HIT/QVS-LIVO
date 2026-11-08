#ifndef LIVO_CAMERA_OBSERVATION_H_
#define LIVO_CAMERA_OBSERVATION_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "camera_observation_batch_authority.h"
#include "checked_increment.h"
#include "photometric_calibration.h"

namespace livo
{
class AdaptiveVisualSelectionAuthority;
}

inline constexpr std::size_t kVioCameraCount = 4U;

struct CameraObservation
{
  std::uint8_t camera_id{0U};
  std::uint64_t source_stamp_ns{0U};
  double timestamp{0.0};
  cv::Mat image;
  livo::PhotometricStatus photometric_status{
    livo::PhotometricStatus::kMissingExposureGain};

private:
  std::uint64_t observation_generation_{0U};
  std::uint64_t image_generation_{0U};
  std::array<unsigned char, 32U> sealed_image_digest_{};
  bool sealed_image_digest_valid_{false};

  friend class CameraObservationQueue;
  friend class livo::CameraObservationBatchReceipt;
};

struct CameraObservationBatch
{
  std::array<CameraObservation, kVioCameraCount> observations;
  std::array<bool, kVioCameraCount> present{{false, false, false, false}};
  std::size_t camera_count{0U};
  bool partial{false};
  double timestamp{0.0};
  double max_time_offset_sec{0.0};

  const std::optional<livo::CameraObservationBatchReceipt> &
  validatedReceipt() const noexcept
  {
    return validated_receipt_;
  }

private:
  std::optional<livo::CameraObservationBatchReceipt>
    validated_receipt_;

  friend class CameraObservationQueue;
  friend class livo::CameraObservationBatchReceipt;
};

enum class AdaptiveCameraBatchResult
{
  READY,
  WAITING,
  INSUFFICIENT_CAMERAS,
  INVALID_CONFIGURATION,
  AUTHORITY_EXHAUSTED
};

enum class CameraObservationPushResult
{
  ACCEPTED,
  INVALID_CAMERA_ID,
  INVALID_TIMESTAMP,
  EMPTY_IMAGE,
  ALREADY_PROCESSED,
  OUT_OF_ORDER,
  RATE_LIMITED,
  CAPACITY_EXCEEDED,
  AUTHORITY_EXHAUSTED
};

class CameraObservationQueue
{
public:
  static constexpr std::size_t kCameraCount = kVioCameraCount;

  explicit CameraObservationQueue(const std::size_t capacity = 768U)
  : capacity_(std::max<std::size_t>(1U, capacity)),
    batch_authority_(
      livo::CameraObservationBatchAuthorityIdentity::issue())
  {
    last_received_timestamp_.fill(-std::numeric_limits<double>::infinity());
    adaptive_processed_until_.fill(
      -std::numeric_limits<double>::infinity());
    next_observation_generation_.fill(1U);
    next_image_generation_.fill(1U);
  }

  CameraObservationQueue(const CameraObservationQueue &) = delete;
  CameraObservationQueue & operator=(
    const CameraObservationQueue &) = delete;
  CameraObservationQueue(CameraObservationQueue &&) = delete;
  CameraObservationQueue & operator=(
    CameraObservationQueue &&) = delete;

  void setCapacity(const std::size_t capacity)
  {
    if (!observations_.empty()) {
      return;
    }
    capacity_ = std::max<std::size_t>(1U, capacity);
  }

  bool setBatchAuthoritySealingEnabled(const bool enabled)
  {
    if (!observations_.empty()) {
      return false;
    }
    batch_authority_sealing_enabled_ = enabled;
    return true;
  }

  bool batchAuthoritySealingEnabled() const noexcept
  {
    return batch_authority_sealing_enabled_;
  }

  CameraObservationPushResult push(CameraObservation observation)
  {
    if (authority_exhausted_) {
      return CameraObservationPushResult::AUTHORITY_EXHAUSTED;
    }
    if (observation.camera_id >= kCameraCount) {
      return CameraObservationPushResult::INVALID_CAMERA_ID;
    }
    if (!std::isfinite(observation.timestamp) || observation.timestamp <= 0.0) {
      return CameraObservationPushResult::INVALID_TIMESTAMP;
    }
    if (observation.image.empty()) {
      return CameraObservationPushResult::EMPTY_IMAGE;
    }
    if (observation.timestamp <= processed_until_) {
      return CameraObservationPushResult::ALREADY_PROCESSED;
    }

    const std::size_t camera_id = observation.camera_id;
    if (
      camera_id == 0U &&
      observation.timestamp <= front_processed_until_)
    {
      return CameraObservationPushResult::ALREADY_PROCESSED;
    }
    if (observation.timestamp <= adaptive_processed_until_[camera_id]) {
      return CameraObservationPushResult::ALREADY_PROCESSED;
    }
    if (observation.timestamp <= last_received_timestamp_[camera_id]) {
      return CameraObservationPushResult::OUT_OF_ORDER;
    }
    if (observations_.size() >= capacity_) {
      return CameraObservationPushResult::CAPACITY_EXCEEDED;
    }

    if (batch_authority_sealing_enabled_)
    {
      cv::Mat frozen_image;
      std::array<unsigned char, 32U> image_digest{};
      if (
        !livo::CameraObservationBatchReceipt::
        freezeAndDigestImage(
          observation.image, frozen_image, image_digest))
      {
        authority_exhausted_ = true;
        return CameraObservationPushResult::AUTHORITY_EXHAUSTED;
      }
      std::uint64_t observation_generation = 0U;
      std::uint64_t image_generation = 0U;
      if (
        !livo::checkedReserve(
          next_observation_generation_[camera_id],
          observation_generation) ||
        !livo::checkedReserve(
          next_image_generation_[camera_id],
          image_generation))
      {
        authority_exhausted_ = true;
        return CameraObservationPushResult::AUTHORITY_EXHAUSTED;
      }
      observation.image = std::move(frozen_image);
      observation.observation_generation_ =
        observation_generation;
      observation.image_generation_ = image_generation;
      observation.sealed_image_digest_ = image_digest;
      observation.sealed_image_digest_valid_ = true;
    }

    const auto insert_position = std::upper_bound(
      observations_.begin(), observations_.end(), observation,
      [](const CameraObservation & lhs, const CameraObservation & rhs) {
        if (lhs.timestamp == rhs.timestamp) {
          return lhs.camera_id < rhs.camera_id;
        }
        return lhs.timestamp < rhs.timestamp;
      });
    last_received_timestamp_[camera_id] = observation.timestamp;
    observations_.insert(insert_position, std::move(observation));
    return CameraObservationPushResult::ACCEPTED;
  }

  const CameraObservation * nextReady(const double watermark) const
  {
    if (observations_.empty() || observations_.front().timestamp > watermark) {
      return nullptr;
    }
    return &observations_.front();
  }

  bool hasCameraObservation(const std::uint8_t camera_id) const noexcept
  {
    if (camera_id >= kCameraCount)
    {
      return false;
    }
    return std::any_of(
      observations_.begin(), observations_.end(),
      [camera_id](const CameraObservation & observation)
      {
        return observation.camera_id == camera_id;
      });
  }

  const CameraObservation * nextReadyForCamera(
    const std::uint8_t camera_id, const double watermark) const
  {
    if (camera_id >= kCameraCount || !std::isfinite(watermark))
    {
      return nullptr;
    }
    for (const CameraObservation & observation : observations_)
    {
      if (observation.timestamp > watermark)
      {
        break;
      }
      if (observation.camera_id == camera_id)
      {
        return &observation;
      }
    }
    return nullptr;
  }

  bool popReadyForCamera(
    const std::uint8_t camera_id, const double watermark,
    CameraObservation & observation)
  {
    if (camera_id >= kCameraCount || !std::isfinite(watermark))
    {
      return false;
    }
    for (auto it = observations_.begin(); it != observations_.end(); ++it)
    {
      if (it->timestamp > watermark)
      {
        break;
      }
      if (it->camera_id != camera_id)
      {
        continue;
      }
      observation = std::move(*it);
      observations_.erase(it);
      if (camera_id == 0U)
      {
        front_processed_until_ = std::max(
          front_processed_until_, observation.timestamp);
      }
      else
      {
        adaptive_processed_until_[camera_id] = std::max(
          adaptive_processed_until_[camera_id], observation.timestamp);
      }
      return true;
    }
    return false;
  }

  bool nextSynchronized(
    const double watermark, const double max_time_offset_sec,
    CameraObservationBatch & batch) const
  {
    SynchronizedSelection selection;
    if (!findSynchronizedSelection(
        watermark, max_time_offset_sec, selection))
    {
      return false;
    }

    batch = CameraObservationBatch{};
    batch.present.fill(true);
    batch.camera_count = kCameraCount;
    batch.partial = false;
    batch.timestamp = selection.timestamp;
    batch.max_time_offset_sec = selection.max_time_offset_sec;
    for (std::size_t camera_id = 0U; camera_id < kCameraCount; ++camera_id)
    {
      batch.observations[camera_id] =
        observations_[selection.indices[camera_id]];
    }
    return true;
  }

  bool popSynchronized(
    const double watermark, const double max_time_offset_sec,
    CameraObservationBatch & batch,
    std::vector<CameraObservation> * discarded_observations = nullptr)
  {
    SynchronizedSelection selection;
    if (!findSynchronizedSelection(
        watermark, max_time_offset_sec, selection))
    {
      return false;
    }

    CameraObservationBatch sealed_batch;
    sealed_batch.present.fill(true);
    sealed_batch.camera_count = kCameraCount;
    sealed_batch.partial = false;
    sealed_batch.timestamp = selection.timestamp;
    sealed_batch.max_time_offset_sec =
      selection.max_time_offset_sec;

    std::array<double, kCameraCount> selected_timestamps;
    for (std::size_t camera_id = 0U; camera_id < kCameraCount; ++camera_id)
    {
      selected_timestamps[camera_id] =
        observations_[selection.indices[camera_id]].timestamp;
      sealed_batch.observations[camera_id] =
        observations_[selection.indices[camera_id]];
    }
    if (batch_authority_sealing_enabled_)
    {
      std::uint64_t batch_sequence = 0U;
      if (
        authority_exhausted_ ||
        !livo::checkedReserve(
          next_batch_sequence_, batch_sequence))
      {
        authority_exhausted_ = true;
        return false;
      }
      auto receipt =
        livo::CameraObservationBatchReceipt::seal(
        sealed_batch, batch_sequence, batch_authority_);
      if (!receipt.has_value())
      {
        authority_exhausted_ = true;
        return false;
      }
      sealed_batch.validated_receipt_ = std::move(receipt);
    }

    std::deque<CameraObservation> remaining;
    for (std::size_t index = 0U; index < observations_.size(); ++index)
    {
      CameraObservation & observation = observations_[index];
      const std::size_t camera_id = observation.camera_id;
      if (index == selection.indices[camera_id])
      {
        if (!batch_authority_sealing_enabled_)
        {
          sealed_batch.observations[camera_id] =
            std::move(observation);
        }
        continue;
      }

      // A camera frame older than the selected frame can no longer belong to
      // a future joint state. Discard it together with this completed batch.
      if (observation.timestamp <= selected_timestamps[camera_id])
      {
        if (discarded_observations != nullptr)
        {
          discarded_observations->push_back(std::move(observation));
        }
        continue;
      }
      remaining.push_back(std::move(observation));
    }

    observations_ = std::move(remaining);
    batch = std::move(sealed_batch);
    processed_until_ = std::max(processed_until_, batch.timestamp);
    synchronization_initialized_ = true;
    return true;
  }

  AdaptiveCameraBatchResult nextAdaptiveSynchronized(
    const double upstream_watermark, const double max_time_offset_sec,
    const double missing_camera_timeout_sec,
    const std::size_t min_camera_count,
    CameraObservationBatch & batch) const
  {
    AdaptiveSelection selection;
    const AdaptiveCameraBatchResult result = findAdaptiveSelection(
      upstream_watermark, max_time_offset_sec,
      missing_camera_timeout_sec, min_camera_count, selection);
    populateAdaptiveBatch(selection, batch);
    return result;
  }

  AdaptiveCameraBatchResult popAdaptiveSynchronized(
    const double upstream_watermark, const double max_time_offset_sec,
    const double missing_camera_timeout_sec,
    const std::size_t min_camera_count,
    CameraObservationBatch & batch,
    std::vector<CameraObservation> * discarded_observations = nullptr)
  {
    AdaptiveSelection selection;
    const AdaptiveCameraBatchResult result = findAdaptiveSelection(
      upstream_watermark, max_time_offset_sec,
      missing_camera_timeout_sec, min_camera_count, selection);
    populateAdaptiveBatch(selection, batch);
    if (result == AdaptiveCameraBatchResult::WAITING ||
        result == AdaptiveCameraBatchResult::INVALID_CONFIGURATION)
    {
      return result;
    }

    std::optional<livo::CameraObservationBatchReceipt> receipt;
    if (
      result == AdaptiveCameraBatchResult::READY &&
      batch_authority_sealing_enabled_)
    {
      std::uint64_t batch_sequence = 0U;
      if (
        authority_exhausted_ ||
        !livo::checkedReserve(
          next_batch_sequence_, batch_sequence))
      {
        authority_exhausted_ = true;
        return AdaptiveCameraBatchResult::AUTHORITY_EXHAUSTED;
      }
      receipt = livo::CameraObservationBatchReceipt::seal(
        batch, batch_sequence, batch_authority_);
      if (!receipt.has_value())
      {
        authority_exhausted_ = true;
        return AdaptiveCameraBatchResult::AUTHORITY_EXHAUSTED;
      }
    }

    std::deque<CameraObservation> remaining;
    for (std::size_t index = 0U; index < observations_.size(); ++index)
    {
      CameraObservation & observation = observations_[index];
      if (observation.timestamp <= selection.window_end)
      {
        if (result == AdaptiveCameraBatchResult::READY &&
            selection.present[observation.camera_id] &&
            index == selection.indices[observation.camera_id])
        {
          batch.observations[observation.camera_id] =
            std::move(observation);
          continue;
        }
        if (discarded_observations != nullptr)
        {
          discarded_observations->push_back(std::move(observation));
        }
        continue;
      }
      remaining.push_back(std::move(observation));
    }

    observations_ = std::move(remaining);
    if (result == AdaptiveCameraBatchResult::READY)
    {
      batch.validated_receipt_ = std::move(receipt);
    }
    for (std::size_t camera_id = 0U;
         camera_id < kCameraCount; ++camera_id)
    {
      adaptive_processed_until_[camera_id] = std::max(
        adaptive_processed_until_[camera_id], selection.window_end);
    }
    processed_until_ = std::max(processed_until_, selection.timestamp);
    return result;
  }

  AdaptiveCameraBatchResult popAdaptiveNonFrontSynchronized(
    const double upstream_watermark, const double max_time_offset_sec,
    const double missing_camera_timeout_sec,
    const std::size_t min_camera_count,
    CameraObservationBatch & batch,
    std::vector<CameraObservation> * discarded_observations = nullptr)
  {
    AdaptiveSelection selection;
    const AdaptiveCameraBatchResult result = findAdaptiveSelection(
      upstream_watermark, max_time_offset_sec,
      missing_camera_timeout_sec, min_camera_count, 0U, selection);
    populateAdaptiveBatch(selection, batch);
    if (result == AdaptiveCameraBatchResult::WAITING ||
        result == AdaptiveCameraBatchResult::INVALID_CONFIGURATION)
    {
      return result;
    }

    std::optional<livo::CameraObservationBatchReceipt> receipt;
    if (
      result == AdaptiveCameraBatchResult::READY &&
      batch_authority_sealing_enabled_)
    {
      std::uint64_t batch_sequence = 0U;
      if (
        authority_exhausted_ ||
        !livo::checkedReserve(
          next_batch_sequence_, batch_sequence))
      {
        authority_exhausted_ = true;
        return AdaptiveCameraBatchResult::AUTHORITY_EXHAUSTED;
      }
      receipt = livo::CameraObservationBatchReceipt::seal(
        batch, batch_sequence, batch_authority_);
      if (!receipt.has_value())
      {
        authority_exhausted_ = true;
        return AdaptiveCameraBatchResult::AUTHORITY_EXHAUSTED;
      }
    }

    std::deque<CameraObservation> remaining;
    for (std::size_t index = 0U; index < observations_.size(); ++index)
    {
      CameraObservation & observation = observations_[index];
      if (
        observation.camera_id == 0U ||
        observation.timestamp > selection.window_end)
      {
        remaining.push_back(std::move(observation));
        continue;
      }
      if (
        result == AdaptiveCameraBatchResult::READY &&
        selection.present[observation.camera_id] &&
        index == selection.indices[observation.camera_id])
      {
        batch.observations[observation.camera_id] =
          std::move(observation);
        continue;
      }
      if (discarded_observations != nullptr)
      {
        discarded_observations->push_back(std::move(observation));
      }
    }

    observations_ = std::move(remaining);
    if (result == AdaptiveCameraBatchResult::READY)
    {
      batch.validated_receipt_ = std::move(receipt);
    }
    for (std::size_t camera_id = 1U;
         camera_id < kCameraCount; ++camera_id)
    {
      adaptive_processed_until_[camera_id] = std::max(
        adaptive_processed_until_[camera_id], selection.window_end);
    }
    return result;
  }

  double synchronizedWatermark(
    const double upstream_watermark, const std::size_t active_camera_count,
    const double reorder_window_sec) const
  {
    if (active_camera_count == 0U || active_camera_count > kCameraCount) {
      return -std::numeric_limits<double>::infinity();
    }

    double camera_watermark = std::numeric_limits<double>::infinity();
    for (std::size_t camera_id = 0U;
         camera_id < active_camera_count; ++camera_id)
    {
      if (!std::isfinite(last_received_timestamp_[camera_id])) {
        return -std::numeric_limits<double>::infinity();
      }
      camera_watermark =
        std::min(camera_watermark, last_received_timestamp_[camera_id]);
    }
    return std::min(
      upstream_watermark,
      camera_watermark - std::max(0.0, reorder_window_sec));
  }

  // Keep the camera frontier while a batch can still arrive. Once the oldest
  // window expires, missing cameras must not prevent partial batches or LIO.
  // The caller caps upstream_watermark before already-received images waiting
  // outside this queue, unless a full queue must expire a batch to admit them.
  double adaptiveSynchronizedWatermark(
    const double upstream_watermark, const std::size_t active_camera_count,
    const double reorder_window_sec, const double max_time_offset_sec,
    const double missing_camera_timeout_sec) const
  {
    if (!std::isfinite(upstream_watermark) ||
        !std::isfinite(reorder_window_sec) || reorder_window_sec < 0.0 ||
        !std::isfinite(max_time_offset_sec) || max_time_offset_sec < 0.0 ||
        !std::isfinite(missing_camera_timeout_sec) || missing_camera_timeout_sec < 0.0 ||
        active_camera_count == 0U || active_camera_count > kCameraCount)
    {
      return -std::numeric_limits<double>::infinity();
    }
    if (!observations_.empty())
    {
      const double expiration = observations_.front().timestamp +
        2.0 * max_time_offset_sec + missing_camera_timeout_sec;
      if (upstream_watermark >= expiration)
      {
        return upstream_watermark;
      }
    }
    return synchronizedWatermark(
      upstream_watermark, active_camera_count, reorder_window_sec);
  }

  bool pop(CameraObservation & observation)
  {
    if (observations_.empty()) {
      return false;
    }
    observation = std::move(observations_.front());
    observations_.pop_front();
    processed_until_ = std::max(processed_until_, observation.timestamp);
    return true;
  }

  bool discardFront()
  {
    CameraObservation discarded;
    return pop(discarded);
  }

  bool empty() const { return observations_.empty(); }
  std::size_t size() const { return observations_.size(); }
  std::size_t capacity() const { return capacity_; }
  double processedUntil() const { return processed_until_; }
  bool authorityExhausted() const noexcept
  {
    return authority_exhausted_;
  }
  livo::CameraObservationBatchAuthorityIdentity
  batchAuthorityIdentity() const noexcept
  {
    return batch_authority_;
  }
  std::shared_ptr<livo::AdaptiveVisualSelectionAuthority>
  issueAdaptiveVisualSelectionAuthority();

private:
  struct SynchronizedSelection
  {
    std::array<std::size_t, kCameraCount> indices;
    double timestamp{0.0};
    double max_time_offset_sec{0.0};
  };

  struct AdaptiveSelection
  {
    std::array<std::size_t, kCameraCount> indices;
    std::array<bool, kCameraCount> present{{false, false, false, false}};
    std::size_t camera_count{0U};
    double timestamp{0.0};
    double max_time_offset_sec{0.0};
    double window_end{-std::numeric_limits<double>::infinity()};

    AdaptiveSelection()
    {
      indices.fill(std::numeric_limits<std::size_t>::max());
    }
  };

  AdaptiveCameraBatchResult findAdaptiveSelection(
    const double upstream_watermark, const double max_time_offset_sec,
    const double missing_camera_timeout_sec,
    const std::size_t min_camera_count,
    AdaptiveSelection & selection) const
  {
    return findAdaptiveSelection(
      upstream_watermark, max_time_offset_sec,
      missing_camera_timeout_sec, min_camera_count,
      std::nullopt, selection);
  }

  AdaptiveCameraBatchResult findAdaptiveSelection(
    const double upstream_watermark, const double max_time_offset_sec,
    const double missing_camera_timeout_sec,
    const std::size_t min_camera_count,
    const std::optional<std::uint8_t> excluded_camera_id,
    AdaptiveSelection & selection) const
  {
    selection = AdaptiveSelection{};
    if (!std::isfinite(upstream_watermark) ||
        !std::isfinite(max_time_offset_sec) ||
        !std::isfinite(missing_camera_timeout_sec) ||
        max_time_offset_sec < 0.0 ||
        missing_camera_timeout_sec < 0.0 ||
        min_camera_count == 0U ||
        min_camera_count > kCameraCount)
    {
      return AdaptiveCameraBatchResult::INVALID_CONFIGURATION;
    }
    const auto first_candidate = std::find_if(
      observations_.begin(), observations_.end(),
      [upstream_watermark, excluded_camera_id](
        const CameraObservation & observation)
      {
        return
          observation.timestamp <= upstream_watermark &&
          (!excluded_camera_id.has_value() ||
          observation.camera_id != *excluded_camera_id);
      });
    if (first_candidate == observations_.end())
    {
      return AdaptiveCameraBatchResult::WAITING;
    }

    const double window_start = first_candidate->timestamp;
    selection.window_end = window_start + 2.0 * max_time_offset_sec;
    double selected_min = std::numeric_limits<double>::infinity();
    double selected_max = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0U; index < observations_.size(); ++index)
    {
      const CameraObservation & observation = observations_[index];
      if (observation.timestamp > selection.window_end ||
          observation.timestamp > upstream_watermark)
      {
        break;
      }

      const std::size_t camera_id = observation.camera_id;
      if (
        excluded_camera_id.has_value() &&
        camera_id == *excluded_camera_id)
      {
        continue;
      }
      if (selection.present[camera_id]) {
        continue;
      }
      selection.indices[camera_id] = index;
      selection.present[camera_id] = true;
      ++selection.camera_count;
      selected_min = std::min(selected_min, observation.timestamp);
      selected_max = std::max(selected_max, observation.timestamp);
    }

    if (selection.camera_count > 0U)
    {
      selection.timestamp = 0.5 * (selected_min + selected_max);
      selection.max_time_offset_sec =
        0.5 * (selected_max - selected_min);
    }

    if (selection.camera_count == kCameraCount) {
      // A complete batch must not expire the unused tail of its tolerance
      // window: at 30 Hz that tail can already contain the next frame.
      selection.window_end = selected_max;
      return AdaptiveCameraBatchResult::READY;
    }

    const double expiration =
      selection.window_end + missing_camera_timeout_sec;
    if (upstream_watermark < expiration) {
      return AdaptiveCameraBatchResult::WAITING;
    }
    if (selection.camera_count >= min_camera_count) {
      return AdaptiveCameraBatchResult::READY;
    }
    return AdaptiveCameraBatchResult::INSUFFICIENT_CAMERAS;
  }

  void populateAdaptiveBatch(
    const AdaptiveSelection & selection,
    CameraObservationBatch & batch) const
  {
    batch = CameraObservationBatch{};
    batch.present = selection.present;
    batch.camera_count = selection.camera_count;
    batch.partial =
      selection.camera_count > 0U &&
      selection.camera_count < kCameraCount;
    batch.timestamp = selection.timestamp;
    batch.max_time_offset_sec = selection.max_time_offset_sec;
    for (std::size_t camera_id = 0U;
         camera_id < kCameraCount; ++camera_id)
    {
      if (selection.present[camera_id])
      {
        batch.observations[camera_id] =
          observations_[selection.indices[camera_id]];
      }
    }
  }

  bool findSynchronizedSelection(
    const double watermark, const double max_time_offset_sec,
    SynchronizedSelection & selection) const
  {
    if (!std::isfinite(watermark) ||
        !std::isfinite(max_time_offset_sec) ||
        max_time_offset_sec < 0.0)
    {
      return false;
    }

    const std::size_t missing = std::numeric_limits<std::size_t>::max();
    const double full_window_sec = 2.0 * max_time_offset_sec;

    if (synchronization_initialized_)
    {
      std::array<std::size_t, kCameraCount> head_indices;
      head_indices.fill(missing);
      for (std::size_t index = 0U; index < observations_.size(); ++index)
      {
        const std::size_t camera_id = observations_[index].camera_id;
        if (head_indices[camera_id] == missing)
        {
          head_indices[camera_id] = index;
        }
      }
      if (std::find(
          head_indices.begin(), head_indices.end(), missing) ==
          head_indices.end())
      {
        double selected_min = std::numeric_limits<double>::infinity();
        double selected_max = -std::numeric_limits<double>::infinity();
        for (std::size_t camera_id = 0U;
             camera_id < kCameraCount; ++camera_id)
        {
          const double timestamp =
            observations_[head_indices[camera_id]].timestamp;
          selected_min = std::min(selected_min, timestamp);
          selected_max = std::max(selected_max, timestamp);
        }
        if (selected_max <= watermark &&
            selected_max - selected_min <= full_window_sec)
        {
          selection.indices = head_indices;
          selection.timestamp = 0.5 * (selected_min + selected_max);
          selection.max_time_offset_sec =
            0.5 * (selected_max - selected_min);
          return true;
        }
      }
    }

    // At startup (or after a missing frame), search for the minimum-skew
    // association. Once selected, popSynchronized keeps the same per-camera
    // frame cadence instead of changing cycle association on every batch.
    bool found = false;
    double best_time_offset = std::numeric_limits<double>::infinity();
    double best_timestamp = std::numeric_limits<double>::infinity();
    double search_deadline = std::numeric_limits<double>::infinity();
    for (std::size_t left = 0U; left < observations_.size(); ++left)
    {
      const double window_start = observations_[left].timestamp;
      if (window_start > watermark || window_start > search_deadline)
      {
        break;
      }
      const double window_end = window_start + full_window_sec;

      std::array<std::size_t, kCameraCount> candidate_indices;
      candidate_indices.fill(missing);

      for (std::size_t index = left;
           index < observations_.size(); ++index)
      {
        const CameraObservation & candidate = observations_[index];
        if (candidate.timestamp > window_end ||
            candidate.timestamp > watermark)
        {
          break;
        }

        const std::size_t camera_id = candidate.camera_id;
        if (candidate_indices[camera_id] == missing)
        {
          candidate_indices[camera_id] = index;
        }
      }

      if (std::find(
          candidate_indices.begin(), candidate_indices.end(), missing) !=
          candidate_indices.end())
      {
        continue;
      }

      double selected_min = std::numeric_limits<double>::infinity();
      double selected_max = -std::numeric_limits<double>::infinity();
      for (std::size_t camera_id = 0U;
           camera_id < kCameraCount; ++camera_id)
      {
        const double timestamp =
          observations_[candidate_indices[camera_id]].timestamp;
        selected_min = std::min(selected_min, timestamp);
        selected_max = std::max(selected_max, timestamp);
      }

      const double candidate_timestamp =
        0.5 * (selected_min + selected_max);
      const double candidate_time_offset =
        0.5 * (selected_max - selected_min);
      if (!found)
      {
        // Only optimize the association around the oldest complete batch.
        // Searching an arbitrary backlog globally can otherwise select a
        // slightly lower-skew batch near the queue tail and discard history.
        search_deadline = window_start + max_time_offset_sec;
      }
      constexpr double tie_epsilon_sec = 1.0e-12;
      if (candidate_time_offset <= max_time_offset_sec &&
          (candidate_time_offset + tie_epsilon_sec < best_time_offset ||
           (std::abs(candidate_time_offset - best_time_offset) <=
              tie_epsilon_sec &&
            candidate_timestamp < best_timestamp)))
      {
        found = true;
        best_time_offset = candidate_time_offset;
        best_timestamp = candidate_timestamp;
        selection.indices = candidate_indices;
        selection.timestamp = candidate_timestamp;
        selection.max_time_offset_sec = candidate_time_offset;
      }
    }
    return found;
  }

  std::size_t capacity_;
  std::deque<CameraObservation> observations_;
  std::array<double, kCameraCount> last_received_timestamp_;
  std::array<double, kCameraCount> adaptive_processed_until_;
  std::array<std::uint64_t, kCameraCount>
    next_observation_generation_;
  std::array<std::uint64_t, kCameraCount>
    next_image_generation_;
  std::uint64_t next_batch_sequence_{1U};
  livo::CameraObservationBatchAuthorityIdentity batch_authority_;
  double processed_until_{-std::numeric_limits<double>::infinity()};
  double front_processed_until_{-std::numeric_limits<double>::infinity()};
  bool synchronization_initialized_{false};
  bool authority_exhausted_{false};
  bool batch_authority_sealing_enabled_{false};
  std::atomic<bool>
    adaptive_visual_selection_authority_issued_{false};
};

#endif  // LIVO_CAMERA_OBSERVATION_H_
