#include "adaptive_camera_snapshot_queue.h"

#include "checked_increment.h"
#include "frozen_camera_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING
#include <atomic>
#include <chrono>
#include <condition_variable>
#endif

namespace livo
{
namespace
{

struct AdaptiveCameraSnapshotQueueOwner final
{
};

#ifdef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING
std::atomic<bool> fail_next_factory_allocation{false};
std::atomic<bool> fail_next_request_storage_allocation{false};
std::atomic<bool> exhaust_next_snapshot_sequence{false};
std::array<
  std::atomic<bool>,
  kAdaptiveCameraSnapshotCameraCount>
exhaust_next_observation_generation{};
std::array<
  std::atomic<bool>,
  kAdaptiveCameraSnapshotCameraCount>
exhaust_next_image_generation{};

struct DeterministicPauseState
{
  std::mutex mutex;
  std::condition_variable condition;
  bool armed{false};
  bool paused{false};
  bool released{false};
};

DeterministicPauseState freeze_pause;
DeterministicPauseState request_revalidation_pause;

void pauseIfRequested(DeterministicPauseState & state)
{
  std::unique_lock<std::mutex> lock(state.mutex);
  if (!state.armed) {
    return;
  }
  state.armed = false;
  state.paused = true;
  state.condition.notify_all();
  state.condition.wait(
    lock, [&state]() {return state.released;});
  state.paused = false;
  state.released = false;
}

void armPause(DeterministicPauseState & state)
{
  const std::lock_guard<std::mutex> lock(state.mutex);
  state.armed = true;
  state.paused = false;
  state.released = false;
}

bool waitForPause(DeterministicPauseState & state)
{
  std::unique_lock<std::mutex> lock(state.mutex);
  return state.condition.wait_for(
    lock, std::chrono::seconds(5),
    [&state]() {return state.paused;});
}

void releasePause(DeterministicPauseState & state)
{
  const std::lock_guard<std::mutex> lock(state.mutex);
  state.released = true;
  state.condition.notify_all();
}

void pauseAfterFreezeIfRequested()
{
  pauseIfRequested(freeze_pause);
}

void pauseBeforeRequestRevalidationIfRequested()
{
  pauseIfRequested(request_revalidation_pause);
}
#else
void pauseAfterFreezeIfRequested() noexcept
{
}

void pauseBeforeRequestRevalidationIfRequested() noexcept
{
}
#endif

bool validConfiguration(
  const AdaptiveCameraSnapshotQueueConfiguration & configuration) noexcept
{
  return
    configuration.per_camera_capacity > 0U &&
    configuration.per_camera_capacity <=
    std::numeric_limits<std::size_t>::max() /
    kAdaptiveCameraSnapshotCameraCount &&
    std::isfinite(configuration.reorder_window_sec) &&
    configuration.reorder_window_sec >= 0.0 &&
    std::isfinite(configuration.request_deadline_offset_sec) &&
    configuration.request_deadline_offset_sec >=
    configuration.reorder_window_sec &&
    std::isfinite(configuration.max_abs_exposure_offset_sec) &&
    configuration.max_abs_exposure_offset_sec > 0.0;
}

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

}  // namespace

struct AdaptiveCameraSnapshotReceipt::SealedData
{
  struct CameraBinding
  {
    bool present{false};
    std::uint8_t camera_id{0U};
    double exposure_timestamp_sec{
      std::numeric_limits<double>::quiet_NaN()};
    std::uint64_t observation_generation{0U};
    std::uint64_t image_generation{0U};
    int rows{0};
    int columns{0};
    int type{0};
    AdaptiveCameraSnapshotImageDigest digest{};
    std::optional<FrozenCameraImage> frozen_image;
  };

  std::shared_ptr<const void> owner;
  std::uint64_t snapshot_sequence{0U};
  std::uint64_t lidar_generation{0U};
  double lio_reference_timestamp_sec{
    std::numeric_limits<double>::quiet_NaN()};
  double deadline_timestamp_sec{
    std::numeric_limits<double>::quiet_NaN()};
  double arrival_closed_through_sec{
    std::numeric_limits<double>::quiet_NaN()};
  double cutoff_timestamp_sec{
    std::numeric_limits<double>::quiet_NaN()};
  AdaptiveCameraSnapshotTerminalDisposition disposition{
    AdaptiveCameraSnapshotTerminalDisposition::
    DEADLINE_EXPIRED_NO_FRESH_CAMERA};
  std::array<bool, kAdaptiveCameraSnapshotCameraCount> present{{
      false, false, false, false}};
  std::size_t camera_count{0U};
  std::array<
    CameraBinding,
    kAdaptiveCameraSnapshotCameraCount> cameras;
};

struct AdaptiveCameraSnapshotQueue::Implementation
{
  struct Entry
  {
    Entry(
      const double entry_timestamp_sec,
      const std::uint64_t entry_observation_generation,
      const std::uint64_t entry_image_generation,
      FrozenCameraImage entry_image) noexcept
    : timestamp_sec(entry_timestamp_sec),
      observation_generation(entry_observation_generation),
      image_generation(entry_image_generation),
      image(std::move(entry_image))
    {
    }

    Entry(const Entry &) noexcept = default;
    Entry(Entry &&) noexcept = default;
    Entry & operator=(const Entry &) noexcept = default;
    Entry & operator=(Entry &&) noexcept = default;

    double timestamp_sec;
    std::uint64_t observation_generation;
    std::uint64_t image_generation;
    FrozenCameraImage image;
  };

  static_assert(
    std::is_nothrow_move_constructible_v<Entry>);
  static_assert(
    std::is_nothrow_move_assignable_v<Entry>);

  explicit Implementation(
    const AdaptiveCameraSnapshotQueueConfiguration & queue_configuration)
  : configuration(queue_configuration),
    owner(std::make_shared<const AdaptiveCameraSnapshotQueueOwner>())
  {
    for (auto & slot : slots)
    {
      slot.reserve(configuration.per_camera_capacity);
    }
    retired_through.fill(
      -std::numeric_limits<double>::infinity());
    maximum_accepted_timestamp.fill(
      -std::numeric_limits<double>::infinity());
    next_observation_generation.fill(1U);
    next_image_generation.fill(1U);
  }

  bool terminal() const noexcept
  {
    return
      state == AdaptiveCameraSnapshotQueueState::SEALED_READY ||
      state == AdaptiveCameraSnapshotQueueState::LIO_ONLY;
  }

  std::optional<AdaptiveCameraSnapshotPushStatus>
  validatePushLocked(
    const std::uint8_t camera_id,
    const double exposure_timestamp_sec) const noexcept
  {
    if (authority_exhausted)
    {
      return
        AdaptiveCameraSnapshotPushStatus::AUTHORITY_EXHAUSTED;
    }
    if (exposure_timestamp_sec <= retired_through[camera_id])
    {
      return
        AdaptiveCameraSnapshotPushStatus::RETIRED_OR_REPLAY;
    }
    if (exposure_timestamp_sec <= arrival_closed_through)
    {
      return
        AdaptiveCameraSnapshotPushStatus::ARRIVAL_CLOSED;
    }
    const auto & slot = slots[camera_id];
    const auto duplicate = std::lower_bound(
      slot.begin(), slot.end(), exposure_timestamp_sec,
      [](const Entry & entry, const double timestamp_sec)
      {
        return entry.timestamp_sec < timestamp_sec;
      });
    if (
      duplicate != slot.end() &&
      duplicate->timestamp_sec == exposure_timestamp_sec)
    {
      return
        AdaptiveCameraSnapshotPushStatus::DUPLICATE_TIMESTAMP;
    }
    if (slot.size() >= configuration.per_camera_capacity)
    {
      return
        AdaptiveCameraSnapshotPushStatus::CAPACITY_EXCEEDED;
    }
    return std::nullopt;
  }

  double deadlineCutoff() const noexcept
  {
    return
      pending_sealed->deadline_timestamp_sec -
      configuration.reorder_window_sec;
  }

  double currentCutoff() const noexcept
  {
    return std::min(
      arrival_closed_through, deadlineCutoff());
  }

  std::array<std::optional<std::size_t>,
    kAdaptiveCameraSnapshotCameraCount>
  selectLatestCandidates() const noexcept
  {
    std::array<std::optional<std::size_t>,
      kAdaptiveCameraSnapshotCameraCount> selected;
    const double fresh_lower =
      pending_sealed->lio_reference_timestamp_sec -
      configuration.max_abs_exposure_offset_sec;
    const double fresh_upper = std::min(
      pending_sealed->lio_reference_timestamp_sec +
      configuration.max_abs_exposure_offset_sec,
      currentCutoff());
    if (fresh_upper < fresh_lower) {
      return selected;
    }

    for (std::size_t camera_id = 0U;
         camera_id < slots.size();
         ++camera_id)
    {
      const auto & slot = slots[camera_id];
      const auto after_upper = std::upper_bound(
        slot.begin(), slot.end(), fresh_upper,
        [](const double timestamp_sec, const Entry & entry) {
          return timestamp_sec < entry.timestamp_sec;
        });
      if (after_upper == slot.begin()) {
        continue;
      }
      const auto candidate = std::prev(after_upper);
      if (candidate->timestamp_sec >= fresh_lower)
      {
        selected[camera_id] =
          static_cast<std::size_t>(
          std::distance(slot.begin(), candidate));
      }
    }
    return selected;
  }

  void pruneAndRetire(
    const std::array<std::optional<std::size_t>,
      kAdaptiveCameraSnapshotCameraCount> & selected) noexcept
  {
    const double fresh_lower =
      pending_sealed->lio_reference_timestamp_sec -
      configuration.max_abs_exposure_offset_sec;
    for (std::size_t camera_id = 0U;
         camera_id < slots.size();
         ++camera_id)
    {
      auto & slot = slots[camera_id];
      double remove_through =
        -std::numeric_limits<double>::infinity();
      if (selected[camera_id].has_value())
      {
        const double selected_timestamp =
          slot[*selected[camera_id]].timestamp_sec;
        retired_through[camera_id] = std::max(
          retired_through[camera_id], selected_timestamp);
        remove_through = retired_through[camera_id];
      }

      slot.erase(
        std::remove_if(
          slot.begin(), slot.end(),
          [fresh_lower, remove_through](const Entry & entry) {
            return
              entry.timestamp_sec < fresh_lower ||
              entry.timestamp_sec <= remove_through;
          }),
        slot.end());
    }
  }

  void finalizeLocked(
    const AdaptiveCameraSnapshotTerminalDisposition disposition,
    const std::array<std::optional<std::size_t>,
      kAdaptiveCameraSnapshotCameraCount> & selected) noexcept
  {
    if (
      state != AdaptiveCameraSnapshotQueueState::WAITING ||
      !pending_sealed)
    {
      return;
    }

    pending_sealed->arrival_closed_through_sec =
      arrival_closed_through;
    pending_sealed->cutoff_timestamp_sec = currentCutoff();
    pending_sealed->disposition = disposition;

    if (
      disposition ==
      AdaptiveCameraSnapshotTerminalDisposition::CANDIDATES_READY)
    {
      for (std::size_t camera_id = 0U;
           camera_id < slots.size();
           ++camera_id)
      {
        if (!selected[camera_id].has_value()) {
          continue;
        }
        const Entry & entry =
          slots[camera_id][*selected[camera_id]];
        auto & binding =
          pending_sealed->cameras[camera_id];
        binding.present = true;
        binding.camera_id =
          static_cast<std::uint8_t>(camera_id);
        binding.exposure_timestamp_sec =
          entry.timestamp_sec;
        binding.observation_generation =
          entry.observation_generation;
        binding.image_generation =
          entry.image_generation;
        binding.rows = entry.image.rows();
        binding.columns = entry.image.columns();
        binding.type = entry.image.type();
        binding.digest = entry.image.digest();
        binding.frozen_image.emplace(entry.image);
        pending_sealed->present[camera_id] = true;
        ++pending_sealed->camera_count;
      }
      state = AdaptiveCameraSnapshotQueueState::SEALED_READY;
    }
    else
    {
      state = AdaptiveCameraSnapshotQueueState::LIO_ONLY;
    }
    pruneAndRetire(selected);
  }

  void evaluatePendingLocked() noexcept
  {
    if (
      state != AdaptiveCameraSnapshotQueueState::WAITING ||
      !pending_sealed ||
      !std::isfinite(latest_upstream_event_time))
    {
      return;
    }

    const auto selected = selectLatestCandidates();
    const bool any_candidate = std::any_of(
      selected.begin(), selected.end(),
      [](const std::optional<std::size_t> & index) {
        return index.has_value();
      });
    if (any_candidate)
    {
      finalizeLocked(
        AdaptiveCameraSnapshotTerminalDisposition::CANDIDATES_READY,
        selected);
      return;
    }

    if (
      latest_upstream_event_time >=
      pending_sealed->deadline_timestamp_sec)
    {
      finalizeLocked(
        AdaptiveCameraSnapshotTerminalDisposition::
        DEADLINE_EXPIRED_NO_FRESH_CAMERA,
        selected);
    }
  }

  AdaptiveCameraSnapshotQueueConfiguration configuration;
  std::shared_ptr<const void> owner;
  mutable std::mutex mutex;
  std::array<
    std::vector<Entry>,
    kAdaptiveCameraSnapshotCameraCount> slots;
  std::array<double, kAdaptiveCameraSnapshotCameraCount>
    retired_through;
  std::array<double, kAdaptiveCameraSnapshotCameraCount>
    maximum_accepted_timestamp;
  std::array<std::uint64_t, kAdaptiveCameraSnapshotCameraCount>
    next_observation_generation;
  std::array<std::uint64_t, kAdaptiveCameraSnapshotCameraCount>
    next_image_generation;
  AdaptiveCameraSnapshotQueueState state{
    AdaptiveCameraSnapshotQueueState::IDLE};
  double latest_upstream_event_time{
    -std::numeric_limits<double>::infinity()};
  double arrival_closed_through{
    -std::numeric_limits<double>::infinity()};
  std::uint64_t next_snapshot_sequence{1U};
  std::uint64_t last_lidar_generation{0U};
  double last_lio_reference_timestamp{
    -std::numeric_limits<double>::infinity()};
  std::shared_ptr<AdaptiveCameraSnapshotReceipt::SealedData>
    pending_sealed;
  bool authority_exhausted{false};
};

AdaptiveCameraSnapshotQueueIdentity::
AdaptiveCameraSnapshotQueueIdentity(
  std::shared_ptr<const void> owner) noexcept
: owner_(std::move(owner))
{
}

bool AdaptiveCameraSnapshotQueueIdentity::valid() const noexcept
{
  return static_cast<bool>(owner_);
}

bool AdaptiveCameraSnapshotQueueIdentity::sameAuthority(
  const AdaptiveCameraSnapshotQueueIdentity & other) const noexcept
{
  return sameSharedOwner(owner_, other.owner_);
}

AdaptiveCameraSnapshotReceipt::AdaptiveCameraSnapshotReceipt(
  std::shared_ptr<const SealedData> sealed_data) noexcept
: sealed_data_(std::move(sealed_data))
{
}

bool AdaptiveCameraSnapshotReceipt::valid() const noexcept
{
  if (
    !sealed_data_ ||
    !sealed_data_->owner ||
    sealed_data_->snapshot_sequence == 0U ||
    sealed_data_->lidar_generation == 0U ||
    !std::isfinite(
      sealed_data_->lio_reference_timestamp_sec) ||
    !std::isfinite(
      sealed_data_->deadline_timestamp_sec) ||
    !std::isfinite(
      sealed_data_->arrival_closed_through_sec) ||
    !std::isfinite(
      sealed_data_->cutoff_timestamp_sec))
  {
    return false;
  }

  std::size_t present_count = 0U;
  for (std::size_t camera_id = 0U;
       camera_id < sealed_data_->cameras.size();
       ++camera_id)
  {
    const auto & binding = sealed_data_->cameras[camera_id];
    if (binding.present != sealed_data_->present[camera_id])
    {
      return false;
    }
    if (!binding.present)
    {
      if (binding.frozen_image.has_value()) {
        return false;
      }
      continue;
    }

    ++present_count;
    if (
      binding.camera_id !=
      static_cast<std::uint8_t>(camera_id) ||
      !std::isfinite(binding.exposure_timestamp_sec) ||
      binding.observation_generation == 0U ||
      binding.image_generation == 0U ||
      binding.rows <= 0 ||
      binding.columns <= 0 ||
      !binding.frozen_image.has_value() ||
      !binding.frozen_image->valid())
    {
      return false;
    }
  }
  if (present_count != sealed_data_->camera_count)
  {
    return false;
  }

  switch (sealed_data_->disposition)
  {
    case AdaptiveCameraSnapshotTerminalDisposition::CANDIDATES_READY:
      return present_count > 0U;
    case AdaptiveCameraSnapshotTerminalDisposition::
      DEADLINE_EXPIRED_NO_FRESH_CAMERA:
      return present_count == 0U;
  }
  return false;
}

bool AdaptiveCameraSnapshotReceipt::issuedBy(
  const AdaptiveCameraSnapshotQueueIdentity & identity) const noexcept
{
  return
    valid() &&
    sameSharedOwner(sealed_data_->owner, identity.owner_);
}

std::uint64_t
AdaptiveCameraSnapshotReceipt::snapshotSequence() const noexcept
{
  return valid() ? sealed_data_->snapshot_sequence : 0U;
}

std::uint64_t
AdaptiveCameraSnapshotReceipt::lidarGeneration() const noexcept
{
  return valid() ? sealed_data_->lidar_generation : 0U;
}

double
AdaptiveCameraSnapshotReceipt::lioReferenceTimestampSec() const noexcept
{
  return
    valid() ?
    sealed_data_->lio_reference_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

double
AdaptiveCameraSnapshotReceipt::deadlineTimestampSec() const noexcept
{
  return
    valid() ?
    sealed_data_->deadline_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

double
AdaptiveCameraSnapshotReceipt::arrivalClosedThroughSec() const noexcept
{
  return
    valid() ?
    sealed_data_->arrival_closed_through_sec :
    std::numeric_limits<double>::quiet_NaN();
}

double
AdaptiveCameraSnapshotReceipt::cutoffTimestampSec() const noexcept
{
  return
    valid() ?
    sealed_data_->cutoff_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

AdaptiveCameraSnapshotTerminalDisposition
AdaptiveCameraSnapshotReceipt::disposition() const noexcept
{
  return
    valid() ?
    sealed_data_->disposition :
    AdaptiveCameraSnapshotTerminalDisposition::
    DEADLINE_EXPIRED_NO_FRESH_CAMERA;
}

std::array<bool, kAdaptiveCameraSnapshotCameraCount>
AdaptiveCameraSnapshotReceipt::presentMask() const noexcept
{
  return
    valid() ?
    sealed_data_->present :
    std::array<bool, kAdaptiveCameraSnapshotCameraCount>{{
      false, false, false, false}};
}

std::size_t
AdaptiveCameraSnapshotReceipt::cameraCount() const noexcept
{
  return valid() ? sealed_data_->camera_count : 0U;
}

bool AdaptiveCameraSnapshotReceipt::cameraPresent(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() &&
    camera_id < sealed_data_->cameras.size() &&
    sealed_data_->cameras[camera_id].present;
}

std::uint8_t AdaptiveCameraSnapshotReceipt::cameraId(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].camera_id :
    0U;
}

double AdaptiveCameraSnapshotReceipt::exposureTimestampSec(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].
    exposure_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

std::uint64_t
AdaptiveCameraSnapshotReceipt::observationGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].
    observation_generation :
    0U;
}

std::uint64_t
AdaptiveCameraSnapshotReceipt::imageGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].image_generation :
    0U;
}

int AdaptiveCameraSnapshotReceipt::imageRows(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].rows :
    0;
}

int AdaptiveCameraSnapshotReceipt::imageColumns(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].columns :
    0;
}

int AdaptiveCameraSnapshotReceipt::imageType(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].type :
    0;
}

AdaptiveCameraSnapshotImageDigest
AdaptiveCameraSnapshotReceipt::imageDigest(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    sealed_data_->cameras[camera_id].digest :
    AdaptiveCameraSnapshotImageDigest{};
}

AdaptiveCameraSnapshotQueue::AdaptiveCameraSnapshotQueue(
  const AdaptiveCameraSnapshotQueueConfiguration & configuration)
: implementation_(std::make_unique<Implementation>(configuration))
{
}

AdaptiveCameraSnapshotQueue::~AdaptiveCameraSnapshotQueue() = default;

AdaptiveCameraSnapshotQueueCreateResult
AdaptiveCameraSnapshotQueue::create(
  const AdaptiveCameraSnapshotQueueConfiguration & configuration) noexcept
{
  if (!validConfiguration(configuration))
  {
    return {
      AdaptiveCameraSnapshotQueueCreateStatus::INVALID_CONFIGURATION,
      nullptr};
  }

  try
  {
#ifdef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING
    if (fail_next_factory_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
#endif
    auto queue = std::unique_ptr<AdaptiveCameraSnapshotQueue>(
      new AdaptiveCameraSnapshotQueue(configuration));
    return {
      AdaptiveCameraSnapshotQueueCreateStatus::READY,
      std::move(queue)};
  }
  catch (const std::length_error &)
  {
    return {
      AdaptiveCameraSnapshotQueueCreateStatus::INVALID_CONFIGURATION,
      nullptr};
  }
  catch (const std::bad_alloc &)
  {
    return {
      AdaptiveCameraSnapshotQueueCreateStatus::RESOURCE_FAILURE,
      nullptr};
  }
}

AdaptiveCameraSnapshotPushStatus
AdaptiveCameraSnapshotQueue::push(
  const std::uint8_t camera_id,
  const double exposure_timestamp_sec,
  const cv::Mat & image) noexcept
{
  if (camera_id >= kAdaptiveCameraSnapshotCameraCount)
  {
    return
      AdaptiveCameraSnapshotPushStatus::INVALID_CAMERA_ID;
  }
  if (
    !std::isfinite(exposure_timestamp_sec) ||
    exposure_timestamp_sec <= 0.0)
  {
    return AdaptiveCameraSnapshotPushStatus::INVALID_TIMESTAMP;
  }
  if (
    image.empty() ||
    image.dims != 2 ||
    image.rows <= 0 ||
    image.cols <= 0)
  {
    return AdaptiveCameraSnapshotPushStatus::INVALID_IMAGE;
  }

  const auto preflight =
    [this, camera_id, exposure_timestamp_sec]() noexcept
    -> std::optional<AdaptiveCameraSnapshotPushStatus>
    {
      const std::lock_guard<std::mutex> lock(
        implementation_->mutex);
      return implementation_->validatePushLocked(
        camera_id, exposure_timestamp_sec);
    };

  if (const auto rejected = preflight();
      rejected.has_value())
  {
    return *rejected;
  }

  FrozenCameraImageFreezeResult frozen =
    FrozenCameraImage::freeze(image);
  if (!frozen.ready())
  {
    return
      frozen.status ==
      FrozenCameraImageFreezeStatus::INVALID_INPUT ?
      AdaptiveCameraSnapshotPushStatus::INVALID_IMAGE :
      AdaptiveCameraSnapshotPushStatus::RESOURCE_FAILURE;
  }
  pauseAfterFreezeIfRequested();

  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (
    const auto rejected =
      implementation_->validatePushLocked(
      camera_id, exposure_timestamp_sec);
    rejected.has_value())
  {
    return *rejected;
  }

  auto & slot = implementation_->slots[camera_id];
  const auto insert_position = std::lower_bound(
    slot.begin(), slot.end(), exposure_timestamp_sec,
    [](const Implementation::Entry & entry,
       const double timestamp_sec)
    {
      return entry.timestamp_sec < timestamp_sec;
    });
#ifdef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING
  if (exhaust_next_observation_generation[camera_id].exchange(
      false, std::memory_order_acq_rel))
  {
    implementation_->next_observation_generation[camera_id] =
      std::numeric_limits<std::uint64_t>::max();
  }
  if (exhaust_next_image_generation[camera_id].exchange(
      false, std::memory_order_acq_rel))
  {
    implementation_->next_image_generation[camera_id] =
      std::numeric_limits<std::uint64_t>::max();
  }
#endif

  std::uint64_t observation_generation = 0U;
  std::uint64_t image_generation = 0U;
  std::uint64_t next_observation =
    implementation_->next_observation_generation[camera_id];
  std::uint64_t next_image =
    implementation_->next_image_generation[camera_id];
  if (
    !checkedReserve(
      next_observation, observation_generation) ||
    !checkedReserve(next_image, image_generation))
  {
    implementation_->authority_exhausted = true;
    return
      AdaptiveCameraSnapshotPushStatus::AUTHORITY_EXHAUSTED;
  }

  const bool reordered =
    exposure_timestamp_sec <
    implementation_->maximum_accepted_timestamp[camera_id];
  Implementation::Entry entry{
    exposure_timestamp_sec,
    observation_generation,
    image_generation,
    std::move(*frozen.image)};
  slot.insert(insert_position, std::move(entry));
  implementation_->next_observation_generation[camera_id] =
    next_observation;
  implementation_->next_image_generation[camera_id] =
    next_image;
  implementation_->maximum_accepted_timestamp[camera_id] =
    std::max(
    implementation_->maximum_accepted_timestamp[camera_id],
    exposure_timestamp_sec);
  implementation_->evaluatePendingLocked();
  return
    reordered ?
    AdaptiveCameraSnapshotPushStatus::ACCEPTED_REORDERED :
    AdaptiveCameraSnapshotPushStatus::ACCEPTED;
}

AdaptiveCameraSnapshotRequestResult
AdaptiveCameraSnapshotQueue::request(
  const std::uint64_t lidar_generation,
  const LioReference reference) noexcept
{
  const auto current_state =
    [this]() noexcept
    {
      const std::lock_guard<std::mutex> lock(
        implementation_->mutex);
      return implementation_->state;
    };
  const auto result =
    [&current_state](
      const AdaptiveCameraSnapshotRequestStatus status)
    {
      return AdaptiveCameraSnapshotRequestResult{
        status, current_state()};
    };

  if (lidar_generation == 0U)
  {
    return result(
      AdaptiveCameraSnapshotRequestStatus::
      ZERO_LIDAR_GENERATION);
  }
  if (
    !std::isfinite(reference.timestamp_sec) ||
    reference.timestamp_sec <= 0.0)
  {
    return result(
      AdaptiveCameraSnapshotRequestStatus::
      INVALID_LIO_REFERENCE);
  }

  const double deadline =
    reference.timestamp_sec +
    implementation_->configuration.
    request_deadline_offset_sec;
  const double deadline_cutoff =
    deadline -
    implementation_->configuration.reorder_window_sec;
  const double fresh_lower =
    reference.timestamp_sec -
    implementation_->configuration.
    max_abs_exposure_offset_sec;
  const double fresh_upper =
    reference.timestamp_sec +
    implementation_->configuration.
    max_abs_exposure_offset_sec;
  if (
    !std::isfinite(deadline) ||
    !std::isfinite(deadline_cutoff) ||
    !std::isfinite(fresh_lower) ||
    !std::isfinite(fresh_upper) ||
    (
      implementation_->configuration.
      request_deadline_offset_sec > 0.0 &&
      deadline <= reference.timestamp_sec
    ))
  {
    return result(
      AdaptiveCameraSnapshotRequestStatus::DEADLINE_OVERFLOW);
  }

  const auto validate_locked =
    [this, lidar_generation, reference]()
    -> std::optional<AdaptiveCameraSnapshotRequestStatus>
    {
      if (implementation_->terminal())
      {
        return
          AdaptiveCameraSnapshotRequestStatus::
          TERMINAL_RECEIPT_PENDING;
      }
      if (
        implementation_->state ==
        AdaptiveCameraSnapshotQueueState::WAITING)
      {
        if (
          lidar_generation !=
          implementation_->pending_sealed->lidar_generation)
        {
          return
            AdaptiveCameraSnapshotRequestStatus::
            DIFFERENT_GENERATION_WHILE_ACTIVE;
        }
        if (
          reference.timestamp_sec !=
          implementation_->pending_sealed->
          lio_reference_timestamp_sec)
        {
          return
            AdaptiveCameraSnapshotRequestStatus::
            REQUEST_IDENTITY_MISMATCH;
        }
        return
          AdaptiveCameraSnapshotRequestStatus::
          DUPLICATE_ACTIVE_REQUEST;
      }
      if (implementation_->authority_exhausted)
      {
        return
          AdaptiveCameraSnapshotRequestStatus::
          AUTHORITY_EXHAUSTED;
      }
      if (
        lidar_generation <=
        implementation_->last_lidar_generation)
      {
        return
          AdaptiveCameraSnapshotRequestStatus::
          STALE_OR_REPLAYED_GENERATION;
      }
      if (
        reference.timestamp_sec <
        implementation_->last_lio_reference_timestamp)
      {
        return
          AdaptiveCameraSnapshotRequestStatus::
          NON_MONOTONIC_LIO_REFERENCE;
      }
      return std::nullopt;
    };

  {
    const std::lock_guard<std::mutex> lock(
      implementation_->mutex);
    if (const auto rejected = validate_locked();
        rejected.has_value())
    {
      return {*rejected, implementation_->state};
    }
  }

  std::shared_ptr<AdaptiveCameraSnapshotReceipt::SealedData>
    sealed;
  try
  {
#ifdef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING
    if (fail_next_request_storage_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
#endif
    sealed =
      std::make_shared<
      AdaptiveCameraSnapshotReceipt::SealedData>();
  }
  catch (const std::bad_alloc &)
  {
    return result(
      AdaptiveCameraSnapshotRequestStatus::RESOURCE_FAILURE);
  }

  pauseBeforeRequestRevalidationIfRequested();
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (const auto rejected = validate_locked();
      rejected.has_value())
  {
    return {*rejected, implementation_->state};
  }

#ifdef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING
  if (exhaust_next_snapshot_sequence.exchange(
      false, std::memory_order_acq_rel))
  {
    implementation_->next_snapshot_sequence =
      std::numeric_limits<std::uint64_t>::max();
  }
#endif
  std::uint64_t snapshot_sequence = 0U;
  std::uint64_t next_snapshot_sequence =
    implementation_->next_snapshot_sequence;
  if (
    !checkedReserve(
      next_snapshot_sequence, snapshot_sequence))
  {
    implementation_->authority_exhausted = true;
    return {
      AdaptiveCameraSnapshotRequestStatus::
      AUTHORITY_EXHAUSTED,
      implementation_->state};
  }

  sealed->owner = implementation_->owner;
  sealed->snapshot_sequence = snapshot_sequence;
  sealed->lidar_generation = lidar_generation;
  sealed->lio_reference_timestamp_sec =
    reference.timestamp_sec;
  sealed->deadline_timestamp_sec = deadline;
  implementation_->pending_sealed = std::move(sealed);
  implementation_->next_snapshot_sequence =
    next_snapshot_sequence;
  implementation_->last_lidar_generation =
    lidar_generation;
  implementation_->last_lio_reference_timestamp =
    reference.timestamp_sec;
  implementation_->state =
    AdaptiveCameraSnapshotQueueState::WAITING;
  implementation_->evaluatePendingLocked();

  AdaptiveCameraSnapshotRequestStatus status =
    AdaptiveCameraSnapshotRequestStatus::WAITING;
  if (
    implementation_->state ==
    AdaptiveCameraSnapshotQueueState::SEALED_READY)
  {
    status =
      AdaptiveCameraSnapshotRequestStatus::SEALED_READY;
  }
  else if (
    implementation_->state ==
    AdaptiveCameraSnapshotQueueState::LIO_ONLY)
  {
    status = AdaptiveCameraSnapshotRequestStatus::LIO_ONLY;
  }
  return {status, implementation_->state};
}

AdaptiveCameraSnapshotAdvanceResult
AdaptiveCameraSnapshotQueue::advanceUpstreamEventTime(
  const double timestamp_sec) noexcept
{
  if (
    !std::isfinite(timestamp_sec) ||
    timestamp_sec <= 0.0)
  {
    return {
      AdaptiveCameraSnapshotAdvanceStatus::INVALID_TIMESTAMP,
      diagnostics().state};
  }
  const double closure =
    timestamp_sec -
    implementation_->configuration.reorder_window_sec;
  if (!std::isfinite(closure))
  {
    return {
      AdaptiveCameraSnapshotAdvanceStatus::INVALID_TIMESTAMP,
      diagnostics().state};
  }

  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (
    timestamp_sec <
    implementation_->latest_upstream_event_time)
  {
    return {
      AdaptiveCameraSnapshotAdvanceStatus::REGRESSED,
      implementation_->state};
  }
  if (
    timestamp_sec ==
    implementation_->latest_upstream_event_time)
  {
    return {
      AdaptiveCameraSnapshotAdvanceStatus::UNCHANGED,
      implementation_->state};
  }
  implementation_->latest_upstream_event_time =
    timestamp_sec;
  implementation_->arrival_closed_through = closure;
  implementation_->evaluatePendingLocked();
  return {
    AdaptiveCameraSnapshotAdvanceStatus::ADVANCED,
    implementation_->state};
}

std::optional<AdaptiveCameraSnapshotReceipt>
AdaptiveCameraSnapshotQueue::takeFinalized() noexcept
{
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  if (
    !implementation_->terminal() ||
    !implementation_->pending_sealed)
  {
    return std::nullopt;
  }
  std::shared_ptr<const AdaptiveCameraSnapshotReceipt::SealedData>
    sealed = std::move(implementation_->pending_sealed);
  implementation_->state =
    AdaptiveCameraSnapshotQueueState::IDLE;
  AdaptiveCameraSnapshotReceipt receipt{std::move(sealed)};
  return std::optional<AdaptiveCameraSnapshotReceipt>{
    std::move(receipt)};
}

AdaptiveCameraSnapshotQueueIdentity
AdaptiveCameraSnapshotQueue::identity() const noexcept
{
  return AdaptiveCameraSnapshotQueueIdentity{
    implementation_->owner};
}

AdaptiveCameraSnapshotQueueDiagnostics
AdaptiveCameraSnapshotQueue::diagnostics() const noexcept
{
  const std::lock_guard<std::mutex> lock(
    implementation_->mutex);
  AdaptiveCameraSnapshotQueueDiagnostics result;
  result.state = implementation_->state;
  result.latest_upstream_event_time_sec =
    implementation_->latest_upstream_event_time;
  result.arrival_closed_through_sec =
    implementation_->arrival_closed_through;
  result.retired_through_sec =
    implementation_->retired_through;
  for (std::size_t camera_id = 0U;
       camera_id < implementation_->slots.size();
       ++camera_id)
  {
    result.queued_observation_count[camera_id] =
      implementation_->slots[camera_id].size();
  }
  return result;
}

#ifdef LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING
namespace test
{

void failNextAdaptiveCameraSnapshotQueueFactoryAllocation()
{
  fail_next_factory_allocation.store(
    true, std::memory_order_release);
}

void failNextAdaptiveCameraSnapshotRequestStorageAllocation()
{
  fail_next_request_storage_allocation.store(
    true, std::memory_order_release);
}

void exhaustNextAdaptiveCameraSnapshotSequence()
{
  exhaust_next_snapshot_sequence.store(
    true, std::memory_order_release);
}

void exhaustNextAdaptiveCameraSnapshotObservationGeneration(
  const std::uint8_t camera_id)
{
  if (camera_id < exhaust_next_observation_generation.size())
  {
    exhaust_next_observation_generation[camera_id].store(
      true, std::memory_order_release);
  }
}

void exhaustNextAdaptiveCameraSnapshotImageGeneration(
  const std::uint8_t camera_id)
{
  if (camera_id < exhaust_next_image_generation.size())
  {
    exhaust_next_image_generation[camera_id].store(
      true, std::memory_order_release);
  }
}

void pauseNextAdaptiveCameraSnapshotPushAfterFreeze()
{
  armPause(freeze_pause);
}

bool waitForAdaptiveCameraSnapshotPushPausedAfterFreeze()
{
  return waitForPause(freeze_pause);
}

void releaseAdaptiveCameraSnapshotPushAfterFreeze()
{
  releasePause(freeze_pause);
}

void pauseNextAdaptiveCameraSnapshotRequestBeforeRevalidation()
{
  armPause(request_revalidation_pause);
}

bool waitForAdaptiveCameraSnapshotRequestPausedBeforeRevalidation()
{
  return waitForPause(request_revalidation_pause);
}

void releaseAdaptiveCameraSnapshotRequestBeforeRevalidation()
{
  releasePause(request_revalidation_pause);
}

}  // namespace test
#endif

}  // namespace livo
