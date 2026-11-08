#include "exposure_timing_attestation.h"

#include "checked_increment.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace livo
{
namespace
{

constexpr std::size_t kCanonicalPayloadMaxLength = 4096U;
constexpr std::size_t kArtifactIdMaxLength = 128U;
constexpr std::size_t kSourceIdMaxLength = 128U;
constexpr std::size_t kTopicMaxLength = 256U;
constexpr std::size_t kSensorSerialMaxLength = 128U;
constexpr std::size_t kFrameIdMaxLength = 128U;
constexpr std::size_t kDurationSourceMaxLength = 128U;

// Each compiled archive owns a distinct address identity. It is
// zero-initialized, allocation-free, and has internal linkage.
std::uint8_t trusted_calibration_issuer_seal{0U};
std::uint8_t timing_source_authority_issuer_seal{0U};

#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
std::atomic<bool> fail_next_calibration_allocation{false};
std::atomic<bool> fail_next_source_factory_allocation{false};
std::atomic<bool> fail_next_frame_receipt_allocation{false};
std::atomic<bool> fail_next_timing_factory_allocation{false};
std::atomic<bool> fail_next_attestation_allocation{false};
std::array<std::atomic<bool>, kExposureTimingCameraCount>
exhaust_next_source_sequence{};
std::atomic<bool> exhaust_next_attestation_sequence{false};

struct DeterministicPauseState final
{
  std::mutex mutex;
  std::condition_variable condition;
  bool armed{false};
  bool paused{false};
  bool released{false};
};

DeterministicPauseState source_revalidation_pause;
DeterministicPauseState attestation_revalidation_pause;

void pauseIfRequested(DeterministicPauseState & state)
{
  std::unique_lock<std::mutex> lock(state.mutex);
  if (!state.armed)
  {
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

void pauseSourceBeforeRevalidationIfRequested()
{
  pauseIfRequested(source_revalidation_pause);
}

void pauseAttestationBeforeRevalidationIfRequested()
{
  pauseIfRequested(attestation_revalidation_pause);
}
#else
void pauseSourceBeforeRevalidationIfRequested() noexcept
{
}

void pauseAttestationBeforeRevalidationIfRequested() noexcept
{
}
#endif

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

bool isAsciiAlphanumeric(const unsigned char character) noexcept
{
  return
    (character >= static_cast<unsigned char>('a') &&
    character <= static_cast<unsigned char>('z')) ||
    (character >= static_cast<unsigned char>('A') &&
    character <= static_cast<unsigned char>('Z')) ||
    (character >= static_cast<unsigned char>('0') &&
    character <= static_cast<unsigned char>('9'));
}

bool isCompactVisibleAscii(
  const std::string & value,
  const std::size_t maximum_length) noexcept
{
  if (value.empty() || value.size() > maximum_length)
  {
    return false;
  }
  return std::all_of(
    value.begin(), value.end(),
    [](const unsigned char character)
    {
      return
        character >= static_cast<unsigned char>('!') &&
        character <= static_cast<unsigned char>('~');
    });
}

bool isIdentifier(
  const std::string & value,
  const std::size_t maximum_length) noexcept
{
  if (!isCompactVisibleAscii(value, maximum_length))
  {
    return false;
  }
  return std::all_of(
    value.begin(), value.end(),
    [](const unsigned char character)
    {
      return
        isAsciiAlphanumeric(character) ||
        character == static_cast<unsigned char>('.') ||
        character == static_cast<unsigned char>('_') ||
        character == static_cast<unsigned char>(':') ||
        character == static_cast<unsigned char>('-');
    });
}

bool isTopic(const std::string & value) noexcept
{
  if (
    !isCompactVisibleAscii(value, kTopicMaxLength) ||
    value.front() != '/')
  {
    return false;
  }
  return std::all_of(
    value.begin(), value.end(),
    [](const unsigned char character)
    {
      return
        isAsciiAlphanumeric(character) ||
        character == static_cast<unsigned char>('/') ||
        character == static_cast<unsigned char>('.') ||
        character == static_cast<unsigned char>('_') ||
        character == static_cast<unsigned char>('-');
    });
}

bool isFrameId(const std::string & value) noexcept
{
  if (!isCompactVisibleAscii(value, kFrameIdMaxLength))
  {
    return false;
  }
  return std::all_of(
    value.begin(), value.end(),
    [](const unsigned char character)
    {
      return
        isAsciiAlphanumeric(character) ||
        character == static_cast<unsigned char>('/') ||
        character == static_cast<unsigned char>('.') ||
        character == static_cast<unsigned char>('_') ||
        character == static_cast<unsigned char>('-');
    });
}

bool validSourceBinding(
  const ExposureTimingSourceBinding & binding) noexcept
{
  return
    isIdentifier(binding.source_id, kSourceIdMaxLength) &&
    isTopic(binding.topic) &&
    isIdentifier(
      binding.sensor_serial, kSensorSerialMaxLength) &&
    isFrameId(binding.frame_id);
}

bool sameSourceBinding(
  const ExposureTimingSourceBinding & left,
  const ExposureTimingSourceBinding & right) noexcept
{
  return
    left.camera_id == right.camera_id &&
    left.source_id == right.source_id &&
    left.topic == right.topic &&
    left.sensor_serial == right.sensor_serial &&
    left.frame_id == right.frame_id;
}

const ExposureTimingSourceBinding & sourceBindingOf(
  const ExposureTimingSourceBinding & binding) noexcept
{
  return binding;
}

const ExposureTimingSourceBinding & sourceBindingOf(
  const ExposureTimingCameraCalibration & camera) noexcept
{
  return camera.source_binding;
}

template<typename Entry>
bool duplicateSourceBindings(
  const std::array<
    Entry,
    kExposureTimingCameraCount> & entries) noexcept
{
  for (std::size_t left = 0U;
    left < entries.size();
    ++left)
  {
    for (std::size_t right = left + 1U;
      right < entries.size();
      ++right)
    {
      const auto & lhs = sourceBindingOf(entries[left]);
      const auto & rhs = sourceBindingOf(entries[right]);
      if (
        lhs.source_id == rhs.source_id ||
        lhs.topic == rhs.topic ||
        lhs.sensor_serial == rhs.sensor_serial ||
        lhs.frame_id == rhs.frame_id)
      {
        return true;
      }
    }
  }
  return false;
}

ExposureTimingSourceAuthorityCreateStatus
validateSourceAuthorityConfiguration(
  const ExposureTimingSourceAuthorityConfiguration &
    configuration) noexcept
{
  std::array<bool, kExposureTimingCameraCount> seen{};
  for (const auto & binding : configuration.cameras)
  {
    if (
      binding.camera_id >= kExposureTimingCameraCount ||
      seen[binding.camera_id])
    {
      return
        ExposureTimingSourceAuthorityCreateStatus::
        INVALID_CAMERA_ID;
    }
    if (!validSourceBinding(binding))
    {
      return
        ExposureTimingSourceAuthorityCreateStatus::
        INVALID_SOURCE_BINDING;
    }
    seen[binding.camera_id] = true;
  }
  if (duplicateSourceBindings(configuration.cameras))
  {
    return
      ExposureTimingSourceAuthorityCreateStatus::
      INVALID_SOURCE_BINDING;
  }
  return ExposureTimingSourceAuthorityCreateStatus::READY;
}

bool allZero(
  const ExposureTimingArtifactDigest & digest) noexcept
{
  return std::all_of(
    digest.begin(), digest.end(),
    [](const unsigned char byte) {return byte == 0U;});
}

[[maybe_unused]] ExposureTimingCalibrationStatus
validateCalibrationDocument(
  const ExposureTimingCalibrationDocument & document) noexcept
{
  if (
    document.mode != ExposureTimingAuthorityMode::PRODUCTION &&
    document.mode != ExposureTimingAuthorityMode::SIMULATION_TEST)
  {
    return
      ExposureTimingCalibrationStatus::INVALID_AUTHORITY_MODE;
  }
  if (document.provenance == ExposureTimingProvenanceKind::UNKNOWN)
  {
    return
      ExposureTimingCalibrationStatus::
      MISSING_CALIBRATION_PROVENANCE;
  }
  if (
    (document.mode == ExposureTimingAuthorityMode::PRODUCTION &&
    document.provenance !=
      ExposureTimingProvenanceKind::HARDWARE_CALIBRATION) ||
    (document.mode ==
      ExposureTimingAuthorityMode::SIMULATION_TEST &&
    document.provenance !=
      ExposureTimingProvenanceKind::SIMULATION &&
    document.provenance !=
      ExposureTimingProvenanceKind::TEST_FIXTURE))
  {
    return
      ExposureTimingCalibrationStatus::
      PROVENANCE_MODE_MISMATCH;
  }
  if (document.calibration_revision == 0U)
  {
    return
      ExposureTimingCalibrationStatus::
      INVALID_CALIBRATION_REVISION;
  }
  if (
    !isCompactVisibleAscii(
      document.canonical_payload,
      kCanonicalPayloadMaxLength))
  {
    return
      ExposureTimingCalibrationStatus::
      INVALID_CANONICAL_PAYLOAD;
  }
  if (allZero(document.canonical_payload_sha256))
  {
    return
      ExposureTimingCalibrationStatus::
      INVALID_CANONICAL_PAYLOAD_HASH;
  }
  if (!isIdentifier(document.artifact_id, kArtifactIdMaxLength))
  {
    return
      ExposureTimingCalibrationStatus::INVALID_ARTIFACT_ID;
  }
  if (!std::isfinite(document.global_offset_sec))
  {
    return
      ExposureTimingCalibrationStatus::INVALID_STATIC_OFFSET;
  }
  if (
    !std::isfinite(
      document.legacy_exposure_time_init_sec) ||
    document.legacy_exposure_time_init_sec != 0.0)
  {
    return
      ExposureTimingCalibrationStatus::
      NONZERO_OR_INVALID_LEGACY_EXPOSURE_OFFSET;
  }

  std::array<bool, kExposureTimingCameraCount> seen{};
  for (const auto & camera : document.cameras)
  {
    const auto camera_id = camera.source_binding.camera_id;
    if (
      camera_id >= kExposureTimingCameraCount ||
      seen[camera_id])
    {
      return ExposureTimingCalibrationStatus::INVALID_CAMERA_ID;
    }
    if (
      !validSourceBinding(camera.source_binding) ||
      !isIdentifier(
        camera.duration_source,
        kDurationSourceMaxLength))
    {
      return
        ExposureTimingCalibrationStatus::INVALID_SOURCE_BINDING;
    }
    if (
      camera.timestamp_convention !=
      ExposureTimestampConvention::START &&
      camera.timestamp_convention !=
      ExposureTimestampConvention::MIDPOINT &&
      camera.timestamp_convention !=
      ExposureTimestampConvention::END)
    {
      return
        ExposureTimingCalibrationStatus::
        UNKNOWN_TIMESTAMP_CONVENTION;
    }
    const double total_offset_sec =
      document.global_offset_sec +
      camera.per_camera_offset_sec;
    if (
      !std::isfinite(camera.per_camera_offset_sec) ||
      !std::isfinite(total_offset_sec))
    {
      return
        ExposureTimingCalibrationStatus::INVALID_STATIC_OFFSET;
    }
    seen[camera_id] = true;
  }
  if (duplicateSourceBindings(document.cameras))
  {
    return
      ExposureTimingCalibrationStatus::INVALID_SOURCE_BINDING;
  }
  return ExposureTimingCalibrationStatus::READY;
}

const std::string & emptyString() noexcept
{
  static const std::string value;
  return value;
}

double invalidTimestamp() noexcept
{
  return std::numeric_limits<double>::quiet_NaN();
}

}  // namespace

struct TrustedExposureTimingCalibration::Data final
{
  Data(
    const ExposureTimingCalibrationDocument & source,
    const void * source_issuer_identity)
  : issuer_identity(source_issuer_identity),
    document(source)
  {
    for (const auto & camera : source.cameras)
    {
      const auto camera_id = camera.source_binding.camera_id;
      document.cameras[camera_id] = camera;
      total_offsets_sec[camera_id] =
        source.global_offset_sec +
        camera.per_camera_offset_sec;
    }
  }

  const void * issuer_identity;
  ExposureTimingCalibrationDocument document;
  std::array<double, kExposureTimingCameraCount>
    total_offsets_sec{};
};

TrustedExposureTimingCalibration::
TrustedExposureTimingCalibration(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool TrustedExposureTimingCalibration::valid() const noexcept
{
  return
    data_ &&
    data_->issuer_identity ==
    static_cast<const void *>(
      &trusted_calibration_issuer_seal);
}

struct ExposureTimingSourceAuthorityIdentity::Data final
{
  explicit Data(
    const ExposureTimingSourceAuthorityConfiguration &
      configuration,
    const void * source_issuer_identity)
  : issuer_identity(source_issuer_identity)
  {
    for (const auto & binding : configuration.cameras)
    {
      bindings[binding.camera_id] = binding;
    }
  }

  std::array<
    ExposureTimingSourceBinding,
    kExposureTimingCameraCount> bindings;
  const void * issuer_identity;
};

struct ExposureFrameTimingReceipt::Data final
{
  Data(
    std::shared_ptr<const
      ExposureTimingSourceAuthorityIdentity::Data>
      source_owner,
    const std::uint8_t source_camera_id,
    const double source_raw_header_timestamp_sec,
    const std::optional<double>
      source_exposure_duration_sec) noexcept
  : source(std::move(source_owner)),
    camera_id(source_camera_id),
    raw_header_timestamp_sec(source_raw_header_timestamp_sec),
    exposure_duration_sec(source_exposure_duration_sec)
  {
  }

  std::shared_ptr<const
    ExposureTimingSourceAuthorityIdentity::Data> source;
  std::uint8_t camera_id{0U};
  std::uint64_t source_sequence{0U};
  double raw_header_timestamp_sec{0.0};
  std::optional<double> exposure_duration_sec;
};

struct ExposureTimingSourceAuthority::Implementation final
{
  explicit Implementation(
    std::shared_ptr<const
      ExposureTimingSourceAuthorityIdentity::Data>
      source_identity)
  : identity(std::move(source_identity))
  {
    next_source_sequence.fill(1U);
  }

  std::shared_ptr<const
    ExposureTimingSourceAuthorityIdentity::Data> identity;
  std::mutex mutex;
  std::array<std::uint64_t, kExposureTimingCameraCount>
    next_source_sequence{};
};

ExposureTimingSourceAuthorityIdentity::
ExposureTimingSourceAuthorityIdentity(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool ExposureTimingSourceAuthorityIdentity::valid() const noexcept
{
  return
    data_ &&
    data_->issuer_identity ==
    static_cast<const void *>(
      &timing_source_authority_issuer_seal);
}

bool ExposureTimingSourceAuthorityIdentity::sameAuthority(
  const ExposureTimingSourceAuthorityIdentity & other) const noexcept
{
  return
    valid() &&
    other.valid() &&
    sameSharedOwner(data_, other.data_);
}

ExposureFrameTimingIdentity::ExposureFrameTimingIdentity(
  std::shared_ptr<const void> payload) noexcept
: payload_(std::move(payload))
{
}

bool ExposureFrameTimingIdentity::valid() const noexcept
{
  return static_cast<bool>(payload_);
}

bool ExposureFrameTimingIdentity::sameFrame(
  const ExposureFrameTimingIdentity & other) const noexcept
{
  return sameSharedOwner(payload_, other.payload_);
}

ExposureFrameTimingReceipt::ExposureFrameTimingReceipt(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool ExposureFrameTimingReceipt::valid() const noexcept
{
  return
    data_ &&
    data_->source &&
    data_->source->issuer_identity ==
    static_cast<const void *>(
      &timing_source_authority_issuer_seal) &&
    data_->camera_id < kExposureTimingCameraCount &&
    data_->source_sequence > 0U &&
    std::isfinite(data_->raw_header_timestamp_sec) &&
    data_->raw_header_timestamp_sec > 0.0 &&
    (!data_->exposure_duration_sec.has_value() ||
    (std::isfinite(*data_->exposure_duration_sec) &&
    *data_->exposure_duration_sec > 0.0));
}

bool ExposureFrameTimingReceipt::issuedBy(
  const ExposureTimingSourceAuthorityIdentity & identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(data_->source, identity.data_);
}

bool ExposureFrameTimingReceipt::sameFrame(
  const ExposureFrameTimingReceipt & other) const noexcept
{
  return sameSharedOwner(data_, other.data_);
}

ExposureFrameTimingIdentity
ExposureFrameTimingReceipt::frameIdentity() const noexcept
{
  return ExposureFrameTimingIdentity{
    std::static_pointer_cast<const void>(data_)};
}

std::uint8_t ExposureFrameTimingReceipt::cameraId() const noexcept
{
  return valid() ? data_->camera_id : 0U;
}

std::uint64_t
ExposureFrameTimingReceipt::sourceSequence() const noexcept
{
  return valid() ? data_->source_sequence : 0U;
}

ExposureTimingSourceAuthority::
ExposureTimingSourceAuthority(
  std::shared_ptr<Implementation> implementation) noexcept
: implementation_(std::move(implementation))
{
}

ExposureTimingSourceAuthority::~ExposureTimingSourceAuthority() =
  default;

ExposureTimingSourceAuthorityCreateResult
ExposureTimingSourceAuthority::create(
  const ExposureTimingSourceAuthorityConfiguration &
    configuration) noexcept
{
  const auto validation =
    validateSourceAuthorityConfiguration(configuration);
  if (
    validation !=
    ExposureTimingSourceAuthorityCreateStatus::READY)
  {
    return {validation, nullptr};
  }
  try
  {
#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
    if (fail_next_source_factory_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
#endif
    auto identity = std::make_shared<const
      ExposureTimingSourceAuthorityIdentity::Data>(
      configuration,
      &timing_source_authority_issuer_seal);
    auto implementation =
      std::make_shared<Implementation>(std::move(identity));
    auto authority =
      std::unique_ptr<ExposureTimingSourceAuthority>(
      new ExposureTimingSourceAuthority(
        std::move(implementation)));
    return {
      ExposureTimingSourceAuthorityCreateStatus::READY,
      std::move(authority)};
  }
  catch (const std::length_error &)
  {
    return {
      ExposureTimingSourceAuthorityCreateStatus::
      RESOURCE_FAILURE,
      nullptr};
  }
  catch (const std::bad_alloc &)
  {
    return {
      ExposureTimingSourceAuthorityCreateStatus::
      RESOURCE_FAILURE,
      nullptr};
  }
  catch (const std::system_error &)
  {
    return {
      ExposureTimingSourceAuthorityCreateStatus::
      RESOURCE_FAILURE,
      nullptr};
  }
}

ExposureTimingSourceAuthorityIdentity
ExposureTimingSourceAuthority::identity() const noexcept
{
  return ExposureTimingSourceAuthorityIdentity{
    implementation_->identity};
}

bool ExposureTimingSourceAuthority::configuredFor(
  const ExposureTimingSourceBinding & binding) const noexcept
{
  return
    implementation_ &&
    implementation_->identity &&
    binding.camera_id < kExposureTimingCameraCount &&
    sameSourceBinding(
      implementation_->identity->bindings[binding.camera_id],
      binding);
}

ExposureFrameTimingIssueResult
ExposureTimingSourceAuthority::issue(
  const std::uint8_t camera_id,
  const double raw_header_timestamp_sec,
  const std::optional<double> exposure_duration_sec)
{
  auto implementation = implementation_;
  if (camera_id >= kExposureTimingCameraCount)
  {
    return {
      ExposureFrameTimingIssueStatus::INVALID_CAMERA_ID,
      std::nullopt};
  }
  if (
    !std::isfinite(raw_header_timestamp_sec) ||
    raw_header_timestamp_sec <= 0.0)
  {
    return {
      ExposureFrameTimingIssueStatus::
      INVALID_RAW_HEADER_TIMESTAMP,
      std::nullopt};
  }
  if (
    exposure_duration_sec.has_value() &&
    (!std::isfinite(*exposure_duration_sec) ||
    *exposure_duration_sec <= 0.0))
  {
    return {
      ExposureFrameTimingIssueStatus::
      INVALID_EXPOSURE_DURATION,
      std::nullopt};
  }
  {
    const std::lock_guard<std::mutex> lock(
      implementation->mutex);
    if (
      implementation->next_source_sequence[camera_id] == 0U ||
      implementation->next_source_sequence[camera_id] ==
      std::numeric_limits<std::uint64_t>::max())
    {
      return {
        ExposureFrameTimingIssueStatus::AUTHORITY_EXHAUSTED,
        std::nullopt};
    }
  }

  std::shared_ptr<ExposureFrameTimingReceipt::Data> data;
  try
  {
#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
    if (fail_next_frame_receipt_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
#endif
    data = std::make_shared<ExposureFrameTimingReceipt::Data>(
      implementation->identity,
      camera_id,
      raw_header_timestamp_sec,
      exposure_duration_sec);
  }
  catch (const std::length_error &)
  {
    return {
      ExposureFrameTimingIssueStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  catch (const std::bad_alloc &)
  {
    return {
      ExposureFrameTimingIssueStatus::RESOURCE_FAILURE,
      std::nullopt};
  }

  pauseSourceBeforeRevalidationIfRequested();
  const std::lock_guard<std::mutex> lock(
    implementation->mutex);
#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
  if (exhaust_next_source_sequence[camera_id].exchange(
      false, std::memory_order_acq_rel))
  {
    implementation->next_source_sequence[camera_id] =
      std::numeric_limits<std::uint64_t>::max();
  }
#endif
  std::uint64_t reserved_sequence = 0U;
  if (
    !checkedReserve(
      implementation->next_source_sequence[camera_id],
      reserved_sequence))
  {
    return {
      ExposureFrameTimingIssueStatus::AUTHORITY_EXHAUSTED,
      std::nullopt};
  }
  data->source_sequence = reserved_sequence;
  return {
    ExposureFrameTimingIssueStatus::READY,
    ExposureFrameTimingReceipt{std::move(data)}};
}

struct ExposureTimingAuditView::Data final
{
  Data(
    TrustedExposureTimingCalibration
      trusted_calibration,
    std::shared_ptr<const ExposureFrameTimingReceipt::Data>
      source_frame,
    std::shared_ptr<const void> timing_authority_owner,
    const double calculated_midpoint_timestamp_sec,
    const double calculated_effective_timestamp_sec) noexcept
  : calibration(std::move(trusted_calibration)),
    frame(std::move(source_frame)),
    authority_owner(std::move(timing_authority_owner)),
    midpoint_timestamp_sec(calculated_midpoint_timestamp_sec),
    effective_timestamp_sec(calculated_effective_timestamp_sec)
  {
  }

  TrustedExposureTimingCalibration calibration;
  std::shared_ptr<const ExposureFrameTimingReceipt::Data> frame;
  std::shared_ptr<const void> authority_owner;
  std::uint64_t attestation_sequence{0U};
  double midpoint_timestamp_sec{0.0};
  double effective_timestamp_sec{0.0};
};

struct ExposureTimingAuthority::Implementation final
{
  Implementation(
    TrustedExposureTimingCalibration
      trusted_calibration,
    std::shared_ptr<const
      ExposureTimingSourceAuthorityIdentity::Data>
      source_identity,
    std::shared_ptr<const void> timing_owner)
  : calibration(std::move(trusted_calibration)),
    source(std::move(source_identity)),
    owner(std::move(timing_owner))
  {
  }

  TrustedExposureTimingCalibration calibration;
  std::shared_ptr<const
    ExposureTimingSourceAuthorityIdentity::Data> source;
  std::shared_ptr<const void> owner;
  std::mutex mutex;
  std::array<std::uint64_t, kExposureTimingCameraCount>
    source_sequence_high_water{};
  std::uint64_t next_attestation_sequence{1U};
};

EffectiveExposureTime::EffectiveExposureTime(
  const double seconds) noexcept
: seconds_(seconds)
{
}

double EffectiveExposureTime::seconds() const noexcept
{
  return seconds_;
}

ExposureTimingAuditView::ExposureTimingAuditView(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool ExposureTimingAuditView::valid() const noexcept
{
  return
    data_ &&
    data_->calibration.valid() &&
    data_->frame &&
    data_->frame->source &&
    data_->frame->source->issuer_identity ==
    static_cast<const void *>(
      &timing_source_authority_issuer_seal) &&
    data_->authority_owner &&
    data_->attestation_sequence > 0U &&
    data_->frame->camera_id < kExposureTimingCameraCount &&
    data_->frame->source_sequence > 0U &&
    std::isfinite(data_->midpoint_timestamp_sec) &&
    data_->midpoint_timestamp_sec > 0.0 &&
    std::isfinite(data_->effective_timestamp_sec) &&
    data_->effective_timestamp_sec > 0.0;
}

std::uint64_t
ExposureTimingAuditView::attestationSequence() const noexcept
{
  return valid() ? data_->attestation_sequence : 0U;
}

std::uint8_t ExposureTimingAuditView::cameraId() const noexcept
{
  return valid() ? data_->frame->camera_id : 0U;
}

std::uint64_t
ExposureTimingAuditView::sourceSequence() const noexcept
{
  return valid() ? data_->frame->source_sequence : 0U;
}

ExposureTimingSourceAuthorityIdentity
ExposureTimingAuditView::sourceAuthorityIdentity() const noexcept
{
  return ExposureTimingSourceAuthorityIdentity{
    valid() ? data_->frame->source : nullptr};
}

ExposureFrameTimingIdentity
ExposureTimingAuditView::sourceFrameIdentity() const noexcept
{
  return ExposureFrameTimingIdentity{
    valid() ?
    std::static_pointer_cast<const void>(data_->frame) :
    nullptr};
}

const std::string &
ExposureTimingAuditView::sourceId() const noexcept
{
  return
    valid() ?
    data_->frame->source->bindings[
      data_->frame->camera_id].source_id :
    emptyString();
}

const std::string & ExposureTimingAuditView::topic() const noexcept
{
  return
    valid() ?
    data_->frame->source->bindings[
      data_->frame->camera_id].topic :
    emptyString();
}

const std::string &
ExposureTimingAuditView::sensorSerial() const noexcept
{
  return
    valid() ?
    data_->frame->source->bindings[
      data_->frame->camera_id].sensor_serial :
    emptyString();
}

const std::string &
ExposureTimingAuditView::frameId() const noexcept
{
  return
    valid() ?
    data_->frame->source->bindings[
      data_->frame->camera_id].frame_id :
    emptyString();
}

ExposureTimestampConvention
ExposureTimingAuditView::timestampConvention() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.cameras[
      data_->frame->camera_id].timestamp_convention :
    ExposureTimestampConvention::UNKNOWN;
}

double
ExposureTimingAuditView::rawHeaderTimestampSec() const noexcept
{
  return
    valid() ?
    data_->frame->raw_header_timestamp_sec :
    invalidTimestamp();
}

std::optional<double>
ExposureTimingAuditView::exposureDurationSec() const noexcept
{
  return
    valid() ?
    data_->frame->exposure_duration_sec :
    std::nullopt;
}

const std::string &
ExposureTimingAuditView::durationSource() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.cameras[
      data_->frame->camera_id].duration_source :
    emptyString();
}

double
ExposureTimingAuditView::midpointTimestampSec() const noexcept
{
  return valid() ? data_->midpoint_timestamp_sec : invalidTimestamp();
}

double
ExposureTimingAuditView::globalOffsetSec() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.global_offset_sec :
    invalidTimestamp();
}

double
ExposureTimingAuditView::perCameraOffsetSec() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.cameras[
      data_->frame->camera_id].per_camera_offset_sec :
    invalidTimestamp();
}

double
ExposureTimingAuditView::totalOffsetSec() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->total_offsets_sec[
      data_->frame->camera_id] :
    invalidTimestamp();
}

double
ExposureTimingAuditView::effectiveTimestampSec() const noexcept
{
  return valid() ? data_->effective_timestamp_sec : invalidTimestamp();
}

ExposureTimingAuthorityMode
ExposureTimingAuditView::authorityMode() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.mode :
    ExposureTimingAuthorityMode::UNKNOWN;
}

ExposureTimingProvenanceKind
ExposureTimingAuditView::provenanceKind() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.provenance :
    ExposureTimingProvenanceKind::UNKNOWN;
}

std::uint64_t
ExposureTimingAuditView::calibrationRevision() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.calibration_revision :
    0U;
}

const std::string &
ExposureTimingAuditView::artifactId() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.artifact_id :
    emptyString();
}

const std::string &
ExposureTimingAuditView::canonicalPayload() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.canonical_payload :
    emptyString();
}

ExposureTimingArtifactDigest
ExposureTimingAuditView::canonicalPayloadSha256() const noexcept
{
  return
    valid() ?
    data_->calibration.data_->document.canonical_payload_sha256 :
    ExposureTimingArtifactDigest{};
}

ExposureTimingAuthorityIdentity::
ExposureTimingAuthorityIdentity(
  std::shared_ptr<const void> owner) noexcept
: owner_(std::move(owner))
{
}

bool ExposureTimingAuthorityIdentity::valid() const noexcept
{
  return static_cast<bool>(owner_);
}

bool ExposureTimingAuthorityIdentity::sameAuthority(
  const ExposureTimingAuthorityIdentity & other) const noexcept
{
  return sameSharedOwner(owner_, other.owner_);
}

ExposureTimingAttestationIdentity::
ExposureTimingAttestationIdentity(
  std::shared_ptr<const void> payload) noexcept
: payload_(std::move(payload))
{
}

bool ExposureTimingAttestationIdentity::valid() const noexcept
{
  return static_cast<bool>(payload_);
}

bool ExposureTimingAttestationIdentity::samePayload(
  const ExposureTimingAttestationIdentity & other) const noexcept
{
  return sameSharedOwner(payload_, other.payload_);
}

ExposureTimingAttestation::ExposureTimingAttestation(
  std::shared_ptr<const ExposureTimingAuditView::Data>
    data) noexcept
: data_(std::move(data))
{
}

bool ExposureTimingAttestation::valid() const noexcept
{
  return ExposureTimingAuditView{data_}.valid();
}

bool ExposureTimingAttestation::issuedBy(
  const ExposureTimingAuthorityIdentity & identity) const noexcept
{
  return
    valid() &&
    sameSharedOwner(data_->authority_owner, identity.owner_);
}

bool ExposureTimingAttestation::sameAttestation(
  const ExposureTimingAttestation & other) const noexcept
{
  return sameSharedOwner(data_, other.data_);
}

ExposureTimingAttestationIdentity
ExposureTimingAttestation::payloadIdentity() const noexcept
{
  return ExposureTimingAttestationIdentity{
    std::static_pointer_cast<const void>(data_)};
}

std::uint8_t ExposureTimingAttestation::cameraId() const noexcept
{
  return valid() ? data_->frame->camera_id : 0U;
}

std::uint64_t
ExposureTimingAttestation::sourceSequence() const noexcept
{
  return valid() ? data_->frame->source_sequence : 0U;
}

ExposureTimingSourceAuthorityIdentity
ExposureTimingAttestation::sourceAuthorityIdentity() const noexcept
{
  return ExposureTimingSourceAuthorityIdentity{
    valid() ? data_->frame->source : nullptr};
}

ExposureFrameTimingIdentity
ExposureTimingAttestation::sourceFrameIdentity() const noexcept
{
  return ExposureFrameTimingIdentity{
    valid() ?
    std::static_pointer_cast<const void>(data_->frame) :
    nullptr};
}

std::optional<EffectiveExposureTime>
ExposureTimingAttestation::effectiveExposureTime() const noexcept
{
  if (!valid())
  {
    return std::nullopt;
  }
  return EffectiveExposureTime{data_->effective_timestamp_sec};
}

std::optional<ExposureTimingAuditView>
ExposureTimingAttestation::audit() const noexcept
{
  if (!valid())
  {
    return std::nullopt;
  }
  return ExposureTimingAuditView{data_};
}

ExposureTimingAuthority::ExposureTimingAuthority(
  std::shared_ptr<Implementation> implementation) noexcept
: implementation_(std::move(implementation))
{
}

ExposureTimingAuthority::~ExposureTimingAuthority() = default;

ExposureTimingAuthorityCreateResult
ExposureTimingAuthority::create(
  const TrustedExposureTimingCalibration & calibration,
  const ExposureTimingSourceAuthorityIdentity &
    source_authority) noexcept
{
  if (!calibration.valid())
  {
    return {
      ExposureTimingAuthorityCreateStatus::
      INVALID_TRUSTED_CALIBRATION,
      nullptr};
  }
  if (!source_authority.valid())
  {
    return {
      ExposureTimingAuthorityCreateStatus::
      INVALID_SOURCE_AUTHORITY_IDENTITY,
      nullptr};
  }
  for (std::size_t camera_id = 0U;
    camera_id < kExposureTimingCameraCount;
    ++camera_id)
  {
    if (
      !sameSourceBinding(
        calibration.data_->document.cameras[camera_id]
        .source_binding,
        source_authority.data_->bindings[camera_id]))
    {
      return {
        ExposureTimingAuthorityCreateStatus::
        CALIBRATION_SOURCE_BINDING_MISMATCH,
        nullptr};
    }
  }
  try
  {
#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
    if (fail_next_timing_factory_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
#endif
    auto owner = std::make_shared<const std::uint8_t>(0U);
    auto implementation =
      std::make_shared<Implementation>(
      calibration,
      source_authority.data_,
      owner);
    auto authority =
      std::unique_ptr<ExposureTimingAuthority>(
      new ExposureTimingAuthority(
        std::move(implementation)));
    return {
      ExposureTimingAuthorityCreateStatus::READY,
      std::move(authority)};
  }
  catch (const std::length_error &)
  {
    return {
      ExposureTimingAuthorityCreateStatus::RESOURCE_FAILURE,
      nullptr};
  }
  catch (const std::bad_alloc &)
  {
    return {
      ExposureTimingAuthorityCreateStatus::RESOURCE_FAILURE,
      nullptr};
  }
  catch (const std::system_error &)
  {
    return {
      ExposureTimingAuthorityCreateStatus::RESOURCE_FAILURE,
      nullptr};
  }
}

ExposureTimingAuthorityIdentity
ExposureTimingAuthority::identity() const noexcept
{
  return ExposureTimingAuthorityIdentity{
    implementation_->owner};
}

bool ExposureTimingAuthority::boundTo(
  const ExposureTimingSourceAuthorityIdentity &
    source_authority) const noexcept
{
  return
    implementation_ &&
    source_authority.valid() &&
    sameSharedOwner(
      implementation_->source,
      source_authority.data_);
}

ExposureTimingAttestResult ExposureTimingAuthority::attest(
  const ExposureFrameTimingReceipt & receipt)
{
  auto implementation = implementation_;
  if (!receipt.valid())
  {
    return {
      ExposureTimingAttestStatus::INVALID_FRAME_RECEIPT,
      std::nullopt};
  }
  if (
    !sameSharedOwner(
      receipt.data_->source,
      implementation->source))
  {
    return {
      ExposureTimingAttestStatus::FOREIGN_SOURCE_RECEIPT,
      std::nullopt};
  }

  const auto camera_id = receipt.data_->camera_id;
  const auto & camera =
    implementation->calibration.data_->document
    .cameras[camera_id];
  if (
    camera.timestamp_convention !=
      ExposureTimestampConvention::START &&
    camera.timestamp_convention !=
      ExposureTimestampConvention::MIDPOINT &&
    camera.timestamp_convention !=
      ExposureTimestampConvention::END)
  {
    return {
      ExposureTimingAttestStatus::
      UNKNOWN_TIMESTAMP_CONVENTION,
      std::nullopt};
  }
  if (
    receipt.data_->exposure_duration_sec.has_value() &&
    (!std::isfinite(
      *receipt.data_->exposure_duration_sec) ||
    *receipt.data_->exposure_duration_sec <= 0.0))
  {
    return {
      ExposureTimingAttestStatus::
      INVALID_EXPOSURE_DURATION,
      std::nullopt};
  }
  if (
    (camera.timestamp_convention ==
      ExposureTimestampConvention::START ||
    camera.timestamp_convention ==
      ExposureTimestampConvention::END) &&
    !receipt.data_->exposure_duration_sec.has_value())
  {
    return {
      ExposureTimingAttestStatus::
      EXPOSURE_DURATION_REQUIRED,
      std::nullopt};
  }

  const double raw_header_timestamp_sec =
    receipt.data_->raw_header_timestamp_sec;
  double midpoint_timestamp_sec = raw_header_timestamp_sec;
  if (
    camera.timestamp_convention ==
      ExposureTimestampConvention::START ||
    camera.timestamp_convention ==
      ExposureTimestampConvention::END)
  {
    const double half_duration_sec =
      *receipt.data_->exposure_duration_sec / 2.0;
    if (!std::isfinite(half_duration_sec))
    {
      return {
        ExposureTimingAttestStatus::TIMESTAMP_OVERFLOW,
        std::nullopt};
    }
    if (half_duration_sec <= 0.0)
    {
      return {
        ExposureTimingAttestStatus::
        TIMESTAMP_ARITHMETIC_FAILURE,
        std::nullopt};
    }
    if (
      camera.timestamp_convention ==
      ExposureTimestampConvention::START)
    {
      midpoint_timestamp_sec =
        raw_header_timestamp_sec + half_duration_sec;
      if (!std::isfinite(midpoint_timestamp_sec))
      {
        return {
          ExposureTimingAttestStatus::TIMESTAMP_OVERFLOW,
          std::nullopt};
      }
      if (
        midpoint_timestamp_sec <=
        raw_header_timestamp_sec)
      {
        return {
          ExposureTimingAttestStatus::
          TIMESTAMP_ARITHMETIC_FAILURE,
          std::nullopt};
      }
    }
    else
    {
      midpoint_timestamp_sec =
        raw_header_timestamp_sec - half_duration_sec;
      if (!std::isfinite(midpoint_timestamp_sec))
      {
        return {
          ExposureTimingAttestStatus::TIMESTAMP_OVERFLOW,
          std::nullopt};
      }
      if (
        midpoint_timestamp_sec >=
        raw_header_timestamp_sec ||
        midpoint_timestamp_sec <= 0.0)
      {
        return {
          ExposureTimingAttestStatus::
          TIMESTAMP_ARITHMETIC_FAILURE,
          std::nullopt};
      }
    }
  }

  const double total_offset_sec =
    implementation->calibration.data_
    ->total_offsets_sec[camera_id];
  const double effective_timestamp_sec =
    midpoint_timestamp_sec + total_offset_sec;
  if (!std::isfinite(effective_timestamp_sec))
  {
    return {
      ExposureTimingAttestStatus::TIMESTAMP_OVERFLOW,
      std::nullopt};
  }
  if (
    effective_timestamp_sec <= 0.0 ||
    (total_offset_sec > 0.0 &&
    effective_timestamp_sec <= midpoint_timestamp_sec) ||
    (total_offset_sec < 0.0 &&
    effective_timestamp_sec >= midpoint_timestamp_sec))
  {
    return {
      ExposureTimingAttestStatus::
      TIMESTAMP_ARITHMETIC_FAILURE,
      std::nullopt};
  }

  {
    const std::lock_guard<std::mutex> lock(
      implementation->mutex);
    if (
      implementation->next_attestation_sequence == 0U ||
      implementation->next_attestation_sequence ==
      std::numeric_limits<std::uint64_t>::max())
    {
      return {
        ExposureTimingAttestStatus::AUTHORITY_EXHAUSTED,
        std::nullopt};
    }
    if (
      receipt.data_->source_sequence <=
      implementation->source_sequence_high_water[camera_id])
    {
      return {
        ExposureTimingAttestStatus::
        STALE_OR_REPLAYED_SOURCE_SEQUENCE,
        std::nullopt};
    }
  }

  std::shared_ptr<ExposureTimingAuditView::Data> data;
  try
  {
#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
    if (fail_next_attestation_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
#endif
    data = std::make_shared<ExposureTimingAuditView::Data>(
      implementation->calibration,
      receipt.data_,
      implementation->owner,
      midpoint_timestamp_sec,
      effective_timestamp_sec);
  }
  catch (const std::length_error &)
  {
    return {
      ExposureTimingAttestStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  catch (const std::bad_alloc &)
  {
    return {
      ExposureTimingAttestStatus::RESOURCE_FAILURE,
      std::nullopt};
  }

  pauseAttestationBeforeRevalidationIfRequested();
  const std::lock_guard<std::mutex> lock(
    implementation->mutex);
  if (
    receipt.data_->source_sequence <=
    implementation->source_sequence_high_water[camera_id])
  {
    return {
      ExposureTimingAttestStatus::
      STALE_OR_REPLAYED_SOURCE_SEQUENCE,
      std::nullopt};
  }
#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
  if (exhaust_next_attestation_sequence.exchange(
      false, std::memory_order_acq_rel))
  {
    implementation->next_attestation_sequence =
      std::numeric_limits<std::uint64_t>::max();
  }
#endif
  std::uint64_t reserved_sequence = 0U;
  if (
    !checkedReserve(
      implementation->next_attestation_sequence,
      reserved_sequence))
  {
    return {
      ExposureTimingAttestStatus::AUTHORITY_EXHAUSTED,
      std::nullopt};
  }
  data->attestation_sequence = reserved_sequence;
  implementation->source_sequence_high_water[camera_id] =
    receipt.data_->source_sequence;
  return {
    ExposureTimingAttestStatus::READY,
    ExposureTimingAttestation{std::move(data)}};
}

TrustedExposureTimingCalibrationResult
ExposureTimingCalibrationLoader::mintFixture(
  const ExposureTimingCalibrationDocument & document) noexcept
{
#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
  const auto validation = validateCalibrationDocument(document);
  if (validation != ExposureTimingCalibrationStatus::READY)
  {
    return {validation, std::nullopt};
  }
  if (
    document.mode !=
      ExposureTimingAuthorityMode::SIMULATION_TEST ||
    document.provenance !=
      ExposureTimingProvenanceKind::TEST_FIXTURE)
  {
    return {
      ExposureTimingCalibrationStatus::
      PROVENANCE_MODE_MISMATCH,
      std::nullopt};
  }
  try
  {
    if (fail_next_calibration_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
    auto data =
      std::make_shared<const
      TrustedExposureTimingCalibration::Data>(
      document, &trusted_calibration_issuer_seal);
    return {
      ExposureTimingCalibrationStatus::READY,
      TrustedExposureTimingCalibration{std::move(data)}};
  }
  catch (const std::length_error &)
  {
    return {
      ExposureTimingCalibrationStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  catch (const std::bad_alloc &)
  {
    return {
      ExposureTimingCalibrationStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  catch (...)
  {
    return {
      ExposureTimingCalibrationStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
#else
  (void)document;
  return {
    ExposureTimingCalibrationStatus::
      FIXTURE_MINT_UNAVAILABLE,
    std::nullopt};
#endif
}

#ifdef LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING
namespace test
{

class ExposureTimingCalibrationTestAccess
{
public:
  static TrustedExposureTimingCalibrationResult mintFixture(
    const ExposureTimingCalibrationDocument & document) noexcept;
};

TrustedExposureTimingCalibrationResult
ExposureTimingCalibrationTestAccess::mintFixture(
  const ExposureTimingCalibrationDocument & document) noexcept
{
  return ExposureTimingCalibrationLoader::mintFixture(document);
}

void failNextExposureTimingCalibrationAllocation()
{
  fail_next_calibration_allocation.store(
    true, std::memory_order_release);
}

void failNextExposureTimingSourceAuthorityFactoryAllocation()
{
  fail_next_source_factory_allocation.store(
    true, std::memory_order_release);
}

void failNextExposureFrameTimingReceiptAllocation()
{
  fail_next_frame_receipt_allocation.store(
    true, std::memory_order_release);
}

void failNextExposureTimingAuthorityFactoryAllocation()
{
  fail_next_timing_factory_allocation.store(
    true, std::memory_order_release);
}

void failNextExposureTimingAttestationAllocation()
{
  fail_next_attestation_allocation.store(
    true, std::memory_order_release);
}

void exhaustNextExposureTimingSourceSequence(
  const std::uint8_t camera_id)
{
  if (camera_id < kExposureTimingCameraCount)
  {
    exhaust_next_source_sequence[camera_id].store(
      true, std::memory_order_release);
  }
}

void exhaustNextExposureTimingAttestationSequence()
{
  exhaust_next_attestation_sequence.store(
    true, std::memory_order_release);
}

namespace
{

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

}  // namespace

void pauseNextExposureFrameTimingReceiptBeforeRevalidation()
{
  armPause(source_revalidation_pause);
}

bool waitForExposureFrameTimingReceiptPausedBeforeRevalidation()
{
  return waitForPause(source_revalidation_pause);
}

void releaseExposureFrameTimingReceiptBeforeRevalidation()
{
  releasePause(source_revalidation_pause);
}

void pauseNextExposureTimingAttestationBeforeRevalidation()
{
  armPause(attestation_revalidation_pause);
}

bool waitForExposureTimingAttestationPausedBeforeRevalidation()
{
  return waitForPause(attestation_revalidation_pause);
}

void releaseExposureTimingAttestationBeforeRevalidation()
{
  releasePause(attestation_revalidation_pause);
}

}  // namespace test
#endif

}  // namespace livo
