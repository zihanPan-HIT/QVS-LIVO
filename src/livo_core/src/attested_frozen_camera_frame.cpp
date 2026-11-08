#include "attested_frozen_camera_frame.h"

#include "checked_increment.h"

#include <openssl/evp.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

namespace livo
{
namespace
{

std::uint8_t envelope_issuer_seal{0U};
std::uint8_t attested_frame_issuer_seal{0U};

#ifdef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING
std::atomic<bool> fail_next_envelope_identity_allocation{false};
std::atomic<bool> fail_next_envelope_implementation_allocation{
  false};
std::atomic<bool> fail_next_envelope_payload_allocation{false};
std::atomic<bool> fail_next_ingress_identity_allocation{false};
std::atomic<bool> fail_next_ingress_implementation_allocation{
  false};
std::atomic<bool> fail_next_attested_payload_allocation{false};
std::atomic<bool> fail_next_envelope_digest_computation{false};
std::array<
  std::atomic<bool>,
  kAttestedCameraFrameCameraCount>
exhaust_next_envelope_sequence{};
std::array<
  std::atomic<bool>,
  kAttestedCameraFrameCameraCount>
exhaust_next_ingress_sequence{};
std::atomic<bool> corrupt_next_envelope_digest{false};
std::atomic<bool> corrupt_next_envelope_metadata{false};
std::atomic<bool> corrupt_next_envelope_pixel{false};
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

bool isVisibleAscii(
  const std::string & value,
  const std::size_t maximum_length) noexcept
{
  return
    !value.empty() &&
    value.size() <= maximum_length &&
    std::all_of(
      value.begin(), value.end(),
      [](const unsigned char character)
      {
        return
          character >= static_cast<unsigned char>('!') &&
          character <= static_cast<unsigned char>('~');
      });
}

bool validSourceBinding(
  const ExposureTimingSourceBinding & binding) noexcept
{
  return
    binding.camera_id < kAttestedCameraFrameCameraCount &&
    isVisibleAscii(
      binding.source_id,
      kAttestedCameraFrameIdentifierMaxLength) &&
    isVisibleAscii(
      binding.topic,
      kAttestedCameraFrameTopicMaxLength) &&
    binding.topic.front() == '/' &&
    isVisibleAscii(
      binding.sensor_serial,
      kAttestedCameraFrameIdentifierMaxLength) &&
    isVisibleAscii(
      binding.frame_id,
      kAttestedCameraFrameIdentifierMaxLength);
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

[[maybe_unused]] bool sameRouteContract(
  const CameraDriverFrameRouteContract & left,
  const CameraDriverFrameRouteContract & right) noexcept
{
  return
    sameSourceBinding(
      left.source_binding, right.source_binding) &&
    left.driver_epoch == right.driver_epoch &&
    left.clock_domain == right.clock_domain &&
    left.hardware_frame_counter_policy ==
    right.hardware_frame_counter_policy &&
    left.width == right.width &&
    left.height == right.height &&
    left.encoding == right.encoding &&
    left.is_bigendian == right.is_bigendian &&
    left.step == right.step &&
    left.metadata_revision == right.metadata_revision &&
    left.transform_revision == right.transform_revision;
}

[[maybe_unused]] bool validRouteContract(
  const CameraDriverFrameRouteContract & route) noexcept
{
  return
    validSourceBinding(route.source_binding) &&
    route.driver_epoch > 0U &&
    route.driver_epoch <
    std::numeric_limits<std::uint64_t>::max() &&
    route.clock_domain ==
    CameraDriverClockDomain::SENSOR_HARDWARE &&
    (route.hardware_frame_counter_policy ==
    CameraDriverHardwareFrameCounterPolicy::
    REQUIRED_MONOTONIC ||
    route.hardware_frame_counter_policy ==
    CameraDriverHardwareFrameCounterPolicy::UNAVAILABLE) &&
    route.width > 0U &&
    route.width <= kAttestedCameraFrameMaximumImageDimension &&
    route.height > 0U &&
    route.height <= kAttestedCameraFrameMaximumImageDimension &&
    route.encoding == "mono8" &&
    !route.is_bigendian &&
    route.step == route.width &&
    static_cast<std::uint64_t>(route.step) *
    static_cast<std::uint64_t>(route.height) <=
    kAttestedCameraFrameMaximumPayloadBytes &&
    isVisibleAscii(
      route.metadata_revision,
      kAttestedCameraFrameIdentifierMaxLength) &&
    isVisibleAscii(
      route.transform_revision,
      kAttestedCameraFrameIdentifierMaxLength);
}

template<typename Configuration>
bool duplicateRouteSourceBindings(
  const Configuration & configuration) noexcept
{
  for (std::size_t left = 0U;
    left < configuration.routes.size();
    ++left)
  {
    for (std::size_t right = left + 1U;
      right < configuration.routes.size();
      ++right)
    {
      const auto & lhs =
        configuration.routes[left].source_binding;
      const auto & rhs =
        configuration.routes[right].source_binding;
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

template<typename Configuration>
[[maybe_unused]] bool validConfiguration(
  const Configuration & configuration) noexcept
{
  std::array<bool, kAttestedCameraFrameCameraCount> seen{};
  for (const auto & route : configuration.routes)
  {
    if (
      !validRouteContract(route) ||
      seen[route.source_binding.camera_id])
    {
      return false;
    }
    seen[route.source_binding.camera_id] = true;
  }
  return !duplicateRouteSourceBindings(configuration);
}

template<typename Configuration>
Configuration normalizedConfiguration(
  const Configuration & configuration)
{
  Configuration normalized;
  for (const auto & route : configuration.routes)
  {
    normalized.routes[route.source_binding.camera_id] = route;
  }
  return normalized;
}

bool updateDigest(
  EVP_MD_CTX * context,
  const void * data,
  const std::size_t size) noexcept
{
  return
    size == 0U ||
    EVP_DigestUpdate(context, data, size) == 1;
}

bool updateU64(
  EVP_MD_CTX * context,
  const std::uint64_t value) noexcept
{
  std::array<unsigned char, 8U> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index)
  {
    bytes[index] = static_cast<unsigned char>(
      value >> (56U - 8U * index));
  }
  return updateDigest(context, bytes.data(), bytes.size());
}

bool updateU32(
  EVP_MD_CTX * context,
  const std::uint32_t value) noexcept
{
  std::array<unsigned char, 4U> bytes{};
  for (std::size_t index = 0U; index < bytes.size(); ++index)
  {
    bytes[index] = static_cast<unsigned char>(
      value >> (24U - 8U * index));
  }
  return updateDigest(context, bytes.data(), bytes.size());
}

bool updateU8(
  EVP_MD_CTX * context,
  const std::uint8_t value) noexcept
{
  return updateDigest(context, &value, sizeof(value));
}

bool updateDouble(
  EVP_MD_CTX * context,
  const double value) noexcept
{
  static_assert(
    sizeof(double) == sizeof(std::uint64_t),
    "canonical frame digest requires binary64 doubles");
  std::uint64_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return updateU64(context, bits);
}

bool updateString(
  EVP_MD_CTX * context,
  const std::string & value) noexcept
{
  return
    updateU64(context, value.size()) &&
    updateDigest(context, value.data(), value.size());
}

[[maybe_unused]] bool canonicalDigestFor(
  const CameraDriverFrameDocument & document,
  CameraDriverFrameEnvelopeDigest & digest) noexcept
{
#ifdef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING
  if (fail_next_envelope_digest_computation.exchange(
      false, std::memory_order_acq_rel))
  {
    return false;
  }
#endif
  EVP_MD_CTX * context = EVP_MD_CTX_new();
  if (context == nullptr)
  {
    return false;
  }
  bool valid =
    EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
  valid =
    valid &&
    updateString(
      context, "livo.camera-driver-frame-envelope.v2") &&
    updateU8(context, document.source_binding.camera_id) &&
    updateString(context, document.source_binding.source_id) &&
    updateString(context, document.source_binding.topic) &&
    updateString(
      context, document.source_binding.sensor_serial) &&
    updateString(context, document.source_binding.frame_id) &&
    updateU64(context, document.driver_epoch) &&
    updateU64(context, document.driver_frame_number) &&
    updateU8(
      context,
      static_cast<std::uint8_t>(
        document.hardware_frame_counter_policy)) &&
    updateU8(
      context,
      document.hardware_frame_counter.has_value() ? 1U : 0U) &&
    (!document.hardware_frame_counter.has_value() ||
    updateU64(context, *document.hardware_frame_counter)) &&
    updateU8(
      context,
      static_cast<std::uint8_t>(document.clock_domain)) &&
    updateDouble(
      context, document.raw_header_timestamp_sec) &&
    updateU8(
      context,
      document.sensor_timestamp_sec.has_value() ? 1U : 0U) &&
    (!document.sensor_timestamp_sec.has_value() ||
    updateDouble(context, *document.sensor_timestamp_sec)) &&
    updateU8(
      context,
      document.exposure_duration_sec.has_value() ? 1U : 0U) &&
    (!document.exposure_duration_sec.has_value() ||
    updateDouble(context, *document.exposure_duration_sec)) &&
    updateU32(context, document.width) &&
    updateU32(context, document.height) &&
    updateString(context, document.encoding) &&
    updateU8(context, document.is_bigendian ? 1U : 0U) &&
    updateU32(context, document.step) &&
    updateU64(context, document.data.size()) &&
    updateDigest(
      context, document.data.data(), document.data.size()) &&
    updateString(context, document.metadata_revision) &&
    updateString(context, document.transform_revision);

  unsigned int digest_size = 0U;
  valid =
    valid &&
    EVP_DigestFinal_ex(
      context, digest.data(), &digest_size) == 1 &&
    digest_size == digest.size();
  EVP_MD_CTX_free(context);
  return valid;
}

bool allZero(
  const CameraDriverFrameEnvelopeDigest & digest) noexcept
{
  return std::all_of(
    digest.begin(), digest.end(),
    [](const unsigned char byte) {return byte == 0U;});
}

const std::string & emptyString() noexcept
{
  static const std::string value;
  return value;
}

[[maybe_unused]] CameraDriverFrameEnvelopeIssueStatus
validateDocument(
  const CameraDriverFrameDocument & document,
  const CameraDriverFrameRouteContract & route) noexcept
{
  if (!sameSourceBinding(document.source_binding, route.source_binding))
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      FOREIGN_SOURCE_BINDING;
  }
  if (document.driver_epoch != route.driver_epoch)
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      FOREIGN_DRIVER_EPOCH;
  }
  if (
    document.driver_frame_number == 0U ||
    document.driver_frame_number ==
    std::numeric_limits<std::uint64_t>::max())
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      INVALID_DRIVER_FRAME_NUMBER;
  }
  if (
    document.hardware_frame_counter_policy !=
    route.hardware_frame_counter_policy)
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      FOREIGN_HARDWARE_FRAME_COUNTER_POLICY;
  }
  if (
    document.hardware_frame_counter_policy ==
    CameraDriverHardwareFrameCounterPolicy::
    REQUIRED_MONOTONIC &&
    !document.hardware_frame_counter.has_value())
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      MISSING_REQUIRED_HARDWARE_FRAME_COUNTER;
  }
  if (
    document.hardware_frame_counter_policy ==
    CameraDriverHardwareFrameCounterPolicy::UNAVAILABLE &&
    document.hardware_frame_counter.has_value())
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      UNEXPECTED_HARDWARE_FRAME_COUNTER;
  }
  if (
    document.hardware_frame_counter.has_value() &&
    (*document.hardware_frame_counter == 0U ||
    *document.hardware_frame_counter ==
    std::numeric_limits<std::uint64_t>::max()))
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      INVALID_HARDWARE_FRAME_COUNTER;
  }
  if (
    document.clock_domain != route.clock_domain ||
    document.clock_domain !=
    CameraDriverClockDomain::SENSOR_HARDWARE)
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      INVALID_CLOCK_DOMAIN;
  }
  if (
    !std::isfinite(document.raw_header_timestamp_sec) ||
    document.raw_header_timestamp_sec <= 0.0 ||
    (document.sensor_timestamp_sec.has_value() &&
    (!std::isfinite(*document.sensor_timestamp_sec) ||
    *document.sensor_timestamp_sec <= 0.0)))
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::INVALID_TIMESTAMP;
  }
  if (
    document.exposure_duration_sec.has_value() &&
    (!std::isfinite(*document.exposure_duration_sec) ||
    *document.exposure_duration_sec <= 0.0))
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      INVALID_EXPOSURE_DURATION;
  }
  if (
    document.width != route.width ||
    document.height != route.height ||
    document.encoding != route.encoding ||
    document.is_bigendian != route.is_bigendian ||
    document.step != route.step)
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      INVALID_IMAGE_CONTRACT;
  }
  const std::uint64_t expected_size =
    static_cast<std::uint64_t>(document.step) *
    static_cast<std::uint64_t>(document.height);
  if (
    expected_size == 0U ||
    expected_size > kAttestedCameraFrameMaximumPayloadBytes ||
    document.data.size() != expected_size)
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::INVALID_FRAME_DATA;
  }
  if (
    document.metadata_revision != route.metadata_revision)
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      INVALID_METADATA_REVISION;
  }
  if (
    document.transform_revision !=
    route.transform_revision)
  {
    return
      CameraDriverFrameEnvelopeIssueStatus::
      INVALID_TRANSFORM_REVISION;
  }
  return CameraDriverFrameEnvelopeIssueStatus::READY;
}

static_assert(
  noexcept(
    checkedReserve(
      std::declval<std::uint64_t &>(),
      std::declval<std::uint64_t &>())));
static_assert(
  noexcept(
    std::declval<std::optional<ExposureTimingAttestation> &>() =
    std::declval<ExposureTimingAttestation &&>()));
static_assert(
  std::is_nothrow_move_constructible_v<
    AttestedFrozenCameraFrame>);
static_assert(
  std::is_nothrow_constructible_v<
    std::optional<AttestedFrozenCameraFrame>,
    AttestedFrozenCameraFrame &&>);
static_assert(
  std::is_nothrow_move_constructible_v<
    AttestedCameraFrameIngestResult>);

}  // namespace

struct CameraDriverFrameEnvelopeAuthorityIdentity::Data final
{
  Data(
    const CameraDriverFrameEnvelopeAuthorityConfiguration &
      source_configuration,
    const void * source_issuer_identity)
  : configuration(source_configuration),
    issuer_identity(source_issuer_identity)
  {
  }

  CameraDriverFrameEnvelopeAuthorityConfiguration configuration;
  const void * issuer_identity;
};

struct CameraDriverFrameEnvelopeAuditView::Data final
{
  Data(
    std::shared_ptr<const
      CameraDriverFrameEnvelopeAuthorityIdentity::Data>
      source_authority,
    const CameraDriverFrameDocument & source_document,
    const CameraDriverFrameEnvelopeDigest &
      source_canonical_digest,
    const void * source_issuer_identity)
  : authority(std::move(source_authority)),
    document(source_document),
    canonical_digest(source_canonical_digest),
    issuer_identity(source_issuer_identity)
  {
  }

  std::shared_ptr<const
    CameraDriverFrameEnvelopeAuthorityIdentity::Data> authority;
  CameraDriverFrameDocument document;
  CameraDriverFrameEnvelopeDigest canonical_digest{};
  const void * issuer_identity;
  std::uint64_t envelope_sequence{0U};
};

struct CameraDriverFrameEnvelopeAuthority::Implementation final
{
  struct RouteState final
  {
    std::mutex mutex;
    std::uint64_t next_envelope_sequence{1U};
    std::uint64_t driver_frame_high_water{0U};
    std::optional<std::uint64_t>
      hardware_frame_counter_high_water;
  };

  explicit Implementation(
    std::shared_ptr<const
      CameraDriverFrameEnvelopeAuthorityIdentity::Data>
      source_identity) noexcept
  : identity(std::move(source_identity))
  {
  }

  std::shared_ptr<const
    CameraDriverFrameEnvelopeAuthorityIdentity::Data> identity;
  std::array<RouteState, kAttestedCameraFrameCameraCount> routes;
};

struct AttestedCameraFrameIngressIdentity::Data final
{
  explicit Data(const void * source_issuer_identity) noexcept
  : issuer_identity(source_issuer_identity)
  {
  }

  const void * issuer_identity;
};

struct AttestedFrozenCameraFrameAuditView::Data final
{
  Data(
    std::shared_ptr<const
      CameraDriverFrameEnvelopeAuditView::Data>
      source_envelope,
    std::shared_ptr<const
      AttestedCameraFrameIngressIdentity::Data>
      source_ingress,
    const CameraDriverFrameEnvelopeDigest &
      source_envelope_digest,
    const void * source_issuer_identity) noexcept
  : envelope(std::move(source_envelope)),
    ingress(std::move(source_ingress)),
    envelope_digest(source_envelope_digest),
    issuer_identity(source_issuer_identity)
  {
  }

  std::shared_ptr<const
    CameraDriverFrameEnvelopeAuditView::Data> envelope;
  std::shared_ptr<const
    AttestedCameraFrameIngressIdentity::Data> ingress;
  std::optional<FrozenCameraImage> image;
  std::optional<ExposureTimingAttestation> timing;
  CameraDriverFrameEnvelopeDigest envelope_digest{};
  const void * issuer_identity;
  std::uint64_t ingress_sequence{0U};
};

struct AttestedCameraFrameIngress::Implementation final
{
  struct RouteState final
  {
    std::mutex mutex;
    std::uint64_t next_ingress_sequence{1U};
    std::uint64_t envelope_sequence_high_water{0U};
  };

  Implementation(
    AttestedCameraFrameIngressConfiguration
      source_configuration,
    CameraDriverFrameEnvelopeAuthorityIdentity
      source_expected_envelope,
    std::unique_ptr<ExposureTimingSourceAuthority>
      source_timing_source,
    std::unique_ptr<ExposureTimingAuthority>
      source_timing,
    ExposureTimingSourceAuthorityIdentity
      source_timing_source_identity,
    ExposureTimingAuthorityIdentity
      source_timing_identity,
    std::shared_ptr<const
      AttestedCameraFrameIngressIdentity::Data>
      source_identity) noexcept
  : configuration(std::move(source_configuration)),
    expected_envelope(std::move(source_expected_envelope)),
    timing_source(std::move(source_timing_source)),
    timing(std::move(source_timing)),
    timing_source_identity(
      std::move(source_timing_source_identity)),
    timing_identity(std::move(source_timing_identity)),
    identity(std::move(source_identity))
  {
  }

  AttestedCameraFrameIngressConfiguration configuration;
  CameraDriverFrameEnvelopeAuthorityIdentity expected_envelope;
  std::unique_ptr<ExposureTimingSourceAuthority> timing_source;
  std::unique_ptr<ExposureTimingAuthority> timing;
  ExposureTimingSourceAuthorityIdentity timing_source_identity;
  ExposureTimingAuthorityIdentity timing_identity;
  std::shared_ptr<const
    AttestedCameraFrameIngressIdentity::Data> identity;
  std::array<RouteState, kAttestedCameraFrameCameraCount> routes;
};

CameraDriverFrameEnvelopeAuthorityIdentity::
CameraDriverFrameEnvelopeAuthorityIdentity(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool
CameraDriverFrameEnvelopeAuthorityIdentity::valid() const noexcept
{
  return
    data_ &&
    data_->issuer_identity ==
    static_cast<const void *>(&envelope_issuer_seal);
}

bool
CameraDriverFrameEnvelopeAuthorityIdentity::sameAuthority(
  const CameraDriverFrameEnvelopeAuthorityIdentity &
    other) const noexcept
{
  return
    valid() &&
    other.valid() &&
    sameSharedOwner(data_, other.data_);
}

CameraDriverFrameEnvelopeAuditView::
CameraDriverFrameEnvelopeAuditView(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool CameraDriverFrameEnvelopeAuditView::valid() const noexcept
{
  return
    data_ &&
    data_->issuer_identity ==
    static_cast<const void *>(&envelope_issuer_seal) &&
    data_->authority &&
    data_->authority->issuer_identity ==
    static_cast<const void *>(&envelope_issuer_seal) &&
    data_->envelope_sequence > 0U &&
    !allZero(data_->canonical_digest);
}

std::uint8_t
CameraDriverFrameEnvelopeAuditView::cameraId() const noexcept
{
  return valid() ? data_->document.source_binding.camera_id : 0U;
}

std::uint64_t
CameraDriverFrameEnvelopeAuditView::
envelopeSequence() const noexcept
{
  return valid() ? data_->envelope_sequence : 0U;
}

std::uint64_t
CameraDriverFrameEnvelopeAuditView::driverEpoch() const noexcept
{
  return valid() ? data_->document.driver_epoch : 0U;
}

std::uint64_t
CameraDriverFrameEnvelopeAuditView::
driverFrameNumber() const noexcept
{
  return valid() ? data_->document.driver_frame_number : 0U;
}

CameraDriverHardwareFrameCounterPolicy
CameraDriverFrameEnvelopeAuditView::
hardwareFrameCounterPolicy() const noexcept
{
  return
    valid() ?
    data_->document.hardware_frame_counter_policy :
    CameraDriverHardwareFrameCounterPolicy::UNKNOWN;
}

std::optional<std::uint64_t>
CameraDriverFrameEnvelopeAuditView::
hardwareFrameCounter() const noexcept
{
  return
    valid() ?
    data_->document.hardware_frame_counter :
    std::nullopt;
}

CameraDriverClockDomain
CameraDriverFrameEnvelopeAuditView::clockDomain() const noexcept
{
  return
    valid() ?
    data_->document.clock_domain :
    CameraDriverClockDomain::UNKNOWN;
}

double
CameraDriverFrameEnvelopeAuditView::
rawHeaderTimestampSec() const noexcept
{
  return
    valid() ?
    data_->document.raw_header_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

std::optional<double>
CameraDriverFrameEnvelopeAuditView::
sensorTimestampSec() const noexcept
{
  return
    valid() ?
    data_->document.sensor_timestamp_sec :
    std::nullopt;
}

std::optional<double>
CameraDriverFrameEnvelopeAuditView::
exposureDurationSec() const noexcept
{
  return
    valid() ?
    data_->document.exposure_duration_sec :
    std::nullopt;
}

std::uint32_t
CameraDriverFrameEnvelopeAuditView::width() const noexcept
{
  return valid() ? data_->document.width : 0U;
}

std::uint32_t
CameraDriverFrameEnvelopeAuditView::height() const noexcept
{
  return valid() ? data_->document.height : 0U;
}

const std::string &
CameraDriverFrameEnvelopeAuditView::encoding() const noexcept
{
  return valid() ? data_->document.encoding : emptyString();
}

bool
CameraDriverFrameEnvelopeAuditView::isBigendian() const noexcept
{
  return valid() && data_->document.is_bigendian;
}

std::uint32_t
CameraDriverFrameEnvelopeAuditView::step() const noexcept
{
  return valid() ? data_->document.step : 0U;
}

std::size_t
CameraDriverFrameEnvelopeAuditView::dataSize() const noexcept
{
  return valid() ? data_->document.data.size() : 0U;
}

const std::string &
CameraDriverFrameEnvelopeAuditView::sourceId() const noexcept
{
  return
    valid() ?
    data_->document.source_binding.source_id :
    emptyString();
}

const std::string &
CameraDriverFrameEnvelopeAuditView::topic() const noexcept
{
  return
    valid() ?
    data_->document.source_binding.topic :
    emptyString();
}

const std::string &
CameraDriverFrameEnvelopeAuditView::sensorSerial() const noexcept
{
  return
    valid() ?
    data_->document.source_binding.sensor_serial :
    emptyString();
}

const std::string &
CameraDriverFrameEnvelopeAuditView::frameId() const noexcept
{
  return
    valid() ?
    data_->document.source_binding.frame_id :
    emptyString();
}

const std::string &
CameraDriverFrameEnvelopeAuditView::
metadataRevision() const noexcept
{
  return
    valid() ?
    data_->document.metadata_revision :
    emptyString();
}

const std::string &
CameraDriverFrameEnvelopeAuditView::
transformRevision() const noexcept
{
  return
    valid() ?
    data_->document.transform_revision :
    emptyString();
}

CameraDriverFrameEnvelopeDigest
CameraDriverFrameEnvelopeAuditView::
canonicalDigest() const noexcept
{
  return valid() ?
    data_->canonical_digest :
    CameraDriverFrameEnvelopeDigest{};
}

TrustedCameraDriverFrameEnvelope::
TrustedCameraDriverFrameEnvelope(
  std::shared_ptr<const
    CameraDriverFrameEnvelopeAuditView::Data> data) noexcept
: data_(std::move(data))
{
}

bool TrustedCameraDriverFrameEnvelope::valid() const noexcept
{
  return CameraDriverFrameEnvelopeAuditView{data_}.valid();
}

bool TrustedCameraDriverFrameEnvelope::issuedBy(
  const CameraDriverFrameEnvelopeAuthorityIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(data_->authority, identity.data_);
}

bool TrustedCameraDriverFrameEnvelope::sameFrame(
  const TrustedCameraDriverFrameEnvelope & other) const noexcept
{
  return
    valid() &&
    other.valid() &&
    sameSharedOwner(data_, other.data_);
}

std::optional<CameraDriverFrameEnvelopeAuditView>
TrustedCameraDriverFrameEnvelope::audit() const noexcept
{
  if (!valid())
  {
    return std::nullopt;
  }
  return CameraDriverFrameEnvelopeAuditView{data_};
}

AttestedCameraFrameIngressIdentity::
AttestedCameraFrameIngressIdentity(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool AttestedCameraFrameIngressIdentity::valid() const noexcept
{
  return
    data_ &&
    data_->issuer_identity ==
    static_cast<const void *>(&attested_frame_issuer_seal);
}

bool AttestedCameraFrameIngressIdentity::sameIngress(
  const AttestedCameraFrameIngressIdentity &
    other) const noexcept
{
  return
    valid() &&
    other.valid() &&
    sameSharedOwner(data_, other.data_);
}

AttestedFrozenCameraFrameAuditView::
AttestedFrozenCameraFrameAuditView(
  std::shared_ptr<const Data> data) noexcept
: data_(std::move(data))
{
}

bool AttestedFrozenCameraFrameAuditView::valid() const noexcept
{
  if (
    !data_ ||
    data_->issuer_identity !=
    static_cast<const void *>(&attested_frame_issuer_seal) ||
    !data_->ingress ||
    data_->ingress->issuer_identity !=
    static_cast<const void *>(&attested_frame_issuer_seal) ||
    !data_->envelope ||
    data_->envelope->issuer_identity !=
    static_cast<const void *>(&envelope_issuer_seal) ||
    data_->ingress_sequence == 0U ||
    !data_->image.has_value() ||
    !data_->image->valid() ||
    !data_->timing.has_value() ||
    !data_->timing->valid() ||
    data_->envelope_digest !=
    data_->envelope->canonical_digest)
  {
    return false;
  }
  return
    data_->timing->cameraId() ==
    data_->envelope->document.source_binding.camera_id;
}

std::uint8_t
AttestedFrozenCameraFrameAuditView::cameraId() const noexcept
{
  return
    valid() ?
    data_->envelope->document.source_binding.camera_id :
    0U;
}

std::uint64_t
AttestedFrozenCameraFrameAuditView::
ingressSequence() const noexcept
{
  return valid() ? data_->ingress_sequence : 0U;
}

int
AttestedFrozenCameraFrameAuditView::imageRows() const noexcept
{
  return valid() ? data_->image->rows() : 0;
}

int
AttestedFrozenCameraFrameAuditView::imageColumns() const noexcept
{
  return valid() ? data_->image->columns() : 0;
}

int
AttestedFrozenCameraFrameAuditView::imageType() const noexcept
{
  return valid() ? data_->image->type() : 0;
}

FrozenCameraImage::Digest
AttestedFrozenCameraFrameAuditView::imageDigest() const noexcept
{
  return
    valid() ?
    data_->image->digest() :
    FrozenCameraImage::Digest{};
}

std::optional<EffectiveExposureTime>
AttestedFrozenCameraFrameAuditView::
effectiveExposureTime() const noexcept
{
  return
    valid() ?
    data_->timing->effectiveExposureTime() :
    std::nullopt;
}

std::optional<CameraDriverFrameEnvelopeAuditView>
AttestedFrozenCameraFrameAuditView::envelope() const noexcept
{
  if (!valid())
  {
    return std::nullopt;
  }
  return CameraDriverFrameEnvelopeAuditView{data_->envelope};
}

std::optional<ExposureTimingAuditView>
AttestedFrozenCameraFrameAuditView::timing() const noexcept
{
  return valid() ? data_->timing->audit() : std::nullopt;
}

AttestedFrozenCameraFrame::AttestedFrozenCameraFrame(
  std::shared_ptr<const
    AttestedFrozenCameraFrameAuditView::Data> data) noexcept
: data_(std::move(data))
{
}

bool AttestedFrozenCameraFrame::valid() const noexcept
{
  return AttestedFrozenCameraFrameAuditView{data_}.valid();
}

bool AttestedFrozenCameraFrame::issuedBy(
  const AttestedCameraFrameIngressIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(data_->ingress, identity.data_);
}

bool AttestedFrozenCameraFrame::sameFrame(
  const TrustedCameraDriverFrameEnvelope &
    envelope) const noexcept
{
  return
    valid() &&
    envelope.valid() &&
    sameSharedOwner(data_->envelope, envelope.data_);
}

std::optional<AttestedFrozenCameraFrameAuditView>
AttestedFrozenCameraFrame::audit() const noexcept
{
  if (!valid())
  {
    return std::nullopt;
  }
  return AttestedFrozenCameraFrameAuditView{data_};
}

CameraDriverFrameEnvelopeAuthority::
CameraDriverFrameEnvelopeAuthority(
  std::shared_ptr<Implementation> implementation) noexcept
: implementation_(std::move(implementation))
{
}

CameraDriverFrameEnvelopeAuthority::
~CameraDriverFrameEnvelopeAuthority() = default;

CameraDriverFrameEnvelopeAuthorityCreateResult
CameraDriverFrameEnvelopeAuthority::create(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &
    configuration) noexcept
{
#ifdef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING
  if (!validConfiguration(configuration))
  {
    return {
      CameraDriverFrameEnvelopeAuthorityCreateStatus::
      INVALID_CONFIGURATION,
      nullptr};
  }
  try
  {
    const auto normalized =
      normalizedConfiguration(configuration);
    if (fail_next_envelope_identity_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
    auto identity = std::make_shared<const
      CameraDriverFrameEnvelopeAuthorityIdentity::Data>(
      normalized, &envelope_issuer_seal);
    if (fail_next_envelope_implementation_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
    auto implementation =
      std::make_shared<Implementation>(std::move(identity));
    auto authority =
      std::unique_ptr<CameraDriverFrameEnvelopeAuthority>(
      new CameraDriverFrameEnvelopeAuthority(
        std::move(implementation)));
    return {
      CameraDriverFrameEnvelopeAuthorityCreateStatus::READY,
      std::move(authority)};
  }
  catch (const std::bad_alloc &)
  {
    return {
      CameraDriverFrameEnvelopeAuthorityCreateStatus::
      RESOURCE_FAILURE,
      nullptr};
  }
#else
  (void)configuration;
  return {
    CameraDriverFrameEnvelopeAuthorityCreateStatus::
    DRIVER_CONTRACT_UNAVAILABLE,
    nullptr};
#endif
}

CameraDriverFrameEnvelopeAuthorityIdentity
CameraDriverFrameEnvelopeAuthority::identity() const noexcept
{
  return CameraDriverFrameEnvelopeAuthorityIdentity{
    implementation_ ? implementation_->identity : nullptr};
}

CameraDriverFrameEnvelopeIssueResult
CameraDriverFrameEnvelopeAuthority::issue(
  const CameraDriverFrameDocument & document)
{
#ifdef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING
  auto implementation = implementation_;
  if (
    !implementation ||
    document.source_binding.camera_id >=
    kAttestedCameraFrameCameraCount)
  {
    return {
      CameraDriverFrameEnvelopeIssueStatus::INVALID_CAMERA_ID,
      std::nullopt};
  }
  const auto camera_id = document.source_binding.camera_id;
  const auto & route =
    implementation->identity->configuration.routes[camera_id];
  const auto validation = validateDocument(document, route);
  if (validation != CameraDriverFrameEnvelopeIssueStatus::READY)
  {
    return {validation, std::nullopt};
  }

  auto & state = implementation->routes[camera_id];
  try
  {
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (exhaust_next_envelope_sequence[camera_id].exchange(
        false, std::memory_order_acq_rel))
    {
      state.next_envelope_sequence =
        std::numeric_limits<std::uint64_t>::max();
    }
    if (
      state.next_envelope_sequence == 0U ||
      state.next_envelope_sequence ==
      std::numeric_limits<std::uint64_t>::max())
    {
      return {
        CameraDriverFrameEnvelopeIssueStatus::
        AUTHORITY_EXHAUSTED,
        std::nullopt};
    }
    if (
      document.driver_frame_number <=
      state.driver_frame_high_water)
    {
      return {
        CameraDriverFrameEnvelopeIssueStatus::
        STALE_OR_REPLAYED_DRIVER_FRAME,
        std::nullopt};
    }
    if (
      document.hardware_frame_counter.has_value() &&
      state.hardware_frame_counter_high_water.has_value() &&
      *document.hardware_frame_counter <=
      *state.hardware_frame_counter_high_water)
    {
      return {
        CameraDriverFrameEnvelopeIssueStatus::
        STALE_OR_REPLAYED_HARDWARE_COUNTER,
        std::nullopt};
    }
  }
  catch (const std::system_error &)
  {
    return {
      CameraDriverFrameEnvelopeIssueStatus::RESOURCE_FAILURE,
      std::nullopt};
  }

  CameraDriverFrameEnvelopeDigest digest{};
  if (!canonicalDigestFor(document, digest))
  {
    return {
      CameraDriverFrameEnvelopeIssueStatus::RESOURCE_FAILURE,
      std::nullopt};
  }

  try
  {
    if (fail_next_envelope_payload_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
    auto data = std::make_shared<
      CameraDriverFrameEnvelopeAuditView::Data>(
      implementation->identity,
      document,
      digest,
      &envelope_issuer_seal);
    if (corrupt_next_envelope_digest.exchange(
        false, std::memory_order_acq_rel))
    {
      data->canonical_digest[0U] ^=
        static_cast<unsigned char>(0xFFU);
    }
    if (corrupt_next_envelope_metadata.exchange(
        false, std::memory_order_acq_rel))
    {
      data->document.metadata_revision += "-corrupt";
    }
    if (
      corrupt_next_envelope_pixel.exchange(
        false, std::memory_order_acq_rel) &&
      !data->document.data.empty())
    {
      data->document.data[0U] ^=
        static_cast<unsigned char>(0xFFU);
    }

    const std::lock_guard<std::mutex> lock(state.mutex);
    if (
      document.driver_frame_number <=
      state.driver_frame_high_water)
    {
      return {
        CameraDriverFrameEnvelopeIssueStatus::
        STALE_OR_REPLAYED_DRIVER_FRAME,
        std::nullopt};
    }
    if (
      document.hardware_frame_counter.has_value() &&
      state.hardware_frame_counter_high_water.has_value() &&
      *document.hardware_frame_counter <=
      *state.hardware_frame_counter_high_water)
    {
      return {
        CameraDriverFrameEnvelopeIssueStatus::
        STALE_OR_REPLAYED_HARDWARE_COUNTER,
        std::nullopt};
    }
    std::uint64_t reserved_sequence = 0U;
    if (
      !checkedReserve(
        state.next_envelope_sequence, reserved_sequence))
    {
      return {
        CameraDriverFrameEnvelopeIssueStatus::
        AUTHORITY_EXHAUSTED,
        std::nullopt};
    }
    data->envelope_sequence = reserved_sequence;
    state.driver_frame_high_water =
      document.driver_frame_number;
    if (document.hardware_frame_counter.has_value())
    {
      state.hardware_frame_counter_high_water =
        document.hardware_frame_counter;
    }
    return {
      CameraDriverFrameEnvelopeIssueStatus::READY,
      TrustedCameraDriverFrameEnvelope{std::move(data)}};
  }
  catch (const std::length_error &)
  {
    return {
      CameraDriverFrameEnvelopeIssueStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  catch (const std::bad_alloc &)
  {
    return {
      CameraDriverFrameEnvelopeIssueStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
  catch (const std::system_error &)
  {
    return {
      CameraDriverFrameEnvelopeIssueStatus::RESOURCE_FAILURE,
      std::nullopt};
  }
#else
  (void)document;
  return {
    CameraDriverFrameEnvelopeIssueStatus::
    DRIVER_CONTRACT_UNAVAILABLE,
    std::nullopt};
#endif
}

AttestedCameraFrameIngress::AttestedCameraFrameIngress(
  std::shared_ptr<Implementation> implementation) noexcept
: implementation_(std::move(implementation))
{
}

AttestedCameraFrameIngress::~AttestedCameraFrameIngress() =
  default;

AttestedCameraFrameIngressCreateResult
AttestedCameraFrameIngress::create(
  const AttestedCameraFrameIngressConfiguration &
    configuration,
  const CameraDriverFrameEnvelopeAuthorityIdentity &
    expected_envelope_authority,
  std::unique_ptr<ExposureTimingSourceAuthority>
    timing_source_authority,
  std::unique_ptr<ExposureTimingAuthority>
    timing_authority) noexcept
{
#ifdef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING
  if (!validConfiguration(configuration))
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      INVALID_CONFIGURATION,
      nullptr};
  }
  if (!expected_envelope_authority.valid())
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      INVALID_ENVELOPE_AUTHORITY_IDENTITY,
      nullptr};
  }
  for (const auto & route : configuration.routes)
  {
    const auto camera_id = route.source_binding.camera_id;
    if (
      !sameRouteContract(
        route,
        expected_envelope_authority.data_->
        configuration.routes[camera_id]))
    {
      return {
        AttestedCameraFrameIngressCreateStatus::
        ENVELOPE_CONTRACT_MISMATCH,
        nullptr};
    }
  }
  if (!timing_source_authority)
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      INVALID_TIMING_SOURCE_AUTHORITY,
      nullptr};
  }
  if (!timing_authority)
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      INVALID_TIMING_AUTHORITY,
      nullptr};
  }
  const auto source_identity =
    timing_source_authority->identity();
  if (!source_identity.valid())
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      INVALID_TIMING_SOURCE_AUTHORITY,
      nullptr};
  }
  const auto timing_identity = timing_authority->identity();
  if (!timing_identity.valid())
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      INVALID_TIMING_AUTHORITY,
      nullptr};
  }
  for (const auto & route : configuration.routes)
  {
    if (
      !timing_source_authority->configuredFor(
        route.source_binding))
    {
      return {
        AttestedCameraFrameIngressCreateStatus::
        TIMING_SOURCE_CONTRACT_MISMATCH,
        nullptr};
    }
  }
  if (!timing_authority->boundTo(source_identity))
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      TIMING_SOURCE_AUTHORITY_MISMATCH,
      nullptr};
  }
  try
  {
    const auto normalized =
      normalizedConfiguration(configuration);
    if (fail_next_ingress_identity_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
    auto identity = std::make_shared<const
      AttestedCameraFrameIngressIdentity::Data>(
      &attested_frame_issuer_seal);
    if (fail_next_ingress_implementation_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
    auto implementation =
      std::make_shared<Implementation>(
      normalized,
      expected_envelope_authority,
      std::move(timing_source_authority),
      std::move(timing_authority),
      source_identity,
      timing_identity,
      identity);
    auto ingress =
      std::unique_ptr<AttestedCameraFrameIngress>(
      new AttestedCameraFrameIngress(
        std::move(implementation)));
    return {
      AttestedCameraFrameIngressCreateStatus::READY,
      std::move(ingress)};
  }
  catch (const std::length_error &)
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      RESOURCE_FAILURE,
      nullptr};
  }
  catch (const std::bad_alloc &)
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      RESOURCE_FAILURE,
      nullptr};
  }
  catch (const std::system_error &)
  {
    return {
      AttestedCameraFrameIngressCreateStatus::
      RESOURCE_FAILURE,
      nullptr};
  }
#else
  (void)configuration;
  (void)expected_envelope_authority;
  (void)timing_source_authority;
  (void)timing_authority;
  return {
    AttestedCameraFrameIngressCreateStatus::
    DRIVER_CONTRACT_UNAVAILABLE,
    nullptr};
#endif
}

AttestedCameraFrameIngressIdentity
AttestedCameraFrameIngress::identity() const noexcept
{
  return AttestedCameraFrameIngressIdentity{
    implementation_ ? implementation_->identity : nullptr};
}

AttestedCameraFrameIngestResult
AttestedCameraFrameIngress::ingest(
  const TrustedCameraDriverFrameEnvelope & envelope)
{
#ifdef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING
  auto implementation = implementation_;
  if (!implementation || !envelope.valid())
  {
    return {
      AttestedCameraFrameIngestStatus::INVALID_ENVELOPE,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
  if (!envelope.issuedBy(implementation->expected_envelope))
  {
    return {
      AttestedCameraFrameIngestStatus::
      FOREIGN_ENVELOPE_AUTHORITY,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
  const auto camera_id =
    envelope.data_->document.source_binding.camera_id;
  if (camera_id >= kAttestedCameraFrameCameraCount)
  {
    return {
      AttestedCameraFrameIngestStatus::INVALID_ENVELOPE,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
  const auto & document = envelope.data_->document;
  const auto & route =
    implementation->configuration.routes[camera_id];
  if (
    validateDocument(document, route) !=
    CameraDriverFrameEnvelopeIssueStatus::READY)
  {
    return {
      AttestedCameraFrameIngestStatus::
      FOREIGN_ROUTE_CONTRACT,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }

  auto & state = implementation->routes[camera_id];
  try
  {
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (exhaust_next_ingress_sequence[camera_id].exchange(
        false, std::memory_order_acq_rel))
    {
      state.next_ingress_sequence =
        std::numeric_limits<std::uint64_t>::max();
    }
    if (
      envelope.data_->envelope_sequence <=
      state.envelope_sequence_high_water)
    {
      return {
        AttestedCameraFrameIngestStatus::
        STALE_OR_REPLAYED_ENVELOPE,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    }
    if (
      state.next_ingress_sequence == 0U ||
      state.next_ingress_sequence ==
      std::numeric_limits<std::uint64_t>::max())
    {
      return {
        AttestedCameraFrameIngestStatus::
        AUTHORITY_EXHAUSTED,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    }
  }
  catch (const std::system_error &)
  {
    return {
      AttestedCameraFrameIngestStatus::RESOURCE_FAILURE,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }

  CameraDriverFrameEnvelopeDigest recomputed_digest{};
  if (!canonicalDigestFor(document, recomputed_digest))
  {
    return {
      AttestedCameraFrameIngestStatus::RESOURCE_FAILURE,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
  if (recomputed_digest != envelope.data_->canonical_digest)
  {
    return {
      AttestedCameraFrameIngestStatus::
      INVALID_ENVELOPE_DIGEST,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }

  try
  {
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (
      envelope.data_->envelope_sequence <=
      state.envelope_sequence_high_water)
    {
      return {
        AttestedCameraFrameIngestStatus::
        STALE_OR_REPLAYED_ENVELOPE,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    }
    if (
      state.next_ingress_sequence == 0U ||
      state.next_ingress_sequence ==
      std::numeric_limits<std::uint64_t>::max())
    {
      return {
        AttestedCameraFrameIngestStatus::
        AUTHORITY_EXHAUSTED,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    }

    if (fail_next_attested_payload_allocation.exchange(
        false, std::memory_order_acq_rel))
    {
      throw std::bad_alloc();
    }
    auto data = std::make_shared<
      AttestedFrozenCameraFrameAuditView::Data>(
      envelope.data_,
      implementation->identity,
      recomputed_digest,
      &attested_frame_issuer_seal);

    cv::Mat image_view(
      static_cast<int>(document.height),
      static_cast<int>(document.width),
      CV_8UC1,
      const_cast<unsigned char *>(document.data.data()),
      static_cast<std::size_t>(document.step));
    auto frozen = FrozenCameraImage::freeze(image_view);
    if (!frozen.ready())
    {
      return {
        AttestedCameraFrameIngestStatus::
        IMAGE_FREEZE_FAILURE,
        frozen.status,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    }
    data->image = *frozen.image;

    auto timing_receipt =
      implementation->timing_source->issue(
      camera_id,
      document.raw_header_timestamp_sec,
      document.exposure_duration_sec);
    if (!timing_receipt.ready())
    {
      return {
        AttestedCameraFrameIngestStatus::
        TIMING_ISSUE_FAILURE,
        std::nullopt,
        timing_receipt.status,
        std::nullopt,
        std::nullopt};
    }

    auto timing_attestation =
      implementation->timing->attest(
      *timing_receipt.receipt);
    if (!timing_attestation.ready())
    {
      return {
        AttestedCameraFrameIngestStatus::
        TIMING_ATTEST_FAILURE,
        std::nullopt,
        std::nullopt,
        timing_attestation.status,
        std::nullopt};
    }
    const auto timing_audit =
      timing_attestation.attestation->audit();
    if (
      !timing_audit.has_value() ||
      !timing_attestation.attestation->issuedBy(
        implementation->timing_identity) ||
      !timing_audit->sourceAuthorityIdentity()
      .sameAuthority(
        implementation->timing_source_identity) ||
      timing_audit->cameraId() != camera_id ||
      timing_audit->sourceId() !=
      document.source_binding.source_id ||
      timing_audit->topic() !=
      document.source_binding.topic ||
      timing_audit->sensorSerial() !=
      document.source_binding.sensor_serial ||
      timing_audit->frameId() !=
      document.source_binding.frame_id ||
      timing_audit->rawHeaderTimestampSec() !=
      document.raw_header_timestamp_sec ||
      timing_audit->exposureDurationSec() !=
      document.exposure_duration_sec)
    {
      return {
        AttestedCameraFrameIngestStatus::
        SOURCE_BINDING_MISMATCH,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    }

    data->timing =
      std::move(*timing_attestation.attestation);
    std::uint64_t reserved_sequence = 0U;
    if (
      !checkedReserve(
        state.next_ingress_sequence, reserved_sequence))
    {
      return {
        AttestedCameraFrameIngestStatus::
        AUTHORITY_EXHAUSTED,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt};
    }
    data->ingress_sequence = reserved_sequence;
    state.envelope_sequence_high_water =
      envelope.data_->envelope_sequence;
    return {
      AttestedCameraFrameIngestStatus::READY,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      AttestedFrozenCameraFrame{std::move(data)}};
  }
  catch (const cv::Exception &)
  {
    return {
      AttestedCameraFrameIngestStatus::IMAGE_FREEZE_FAILURE,
      FrozenCameraImageFreezeStatus::INVALID_INPUT,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
  catch (const std::length_error &)
  {
    return {
      AttestedCameraFrameIngestStatus::RESOURCE_FAILURE,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
  catch (const std::bad_alloc &)
  {
    return {
      AttestedCameraFrameIngestStatus::RESOURCE_FAILURE,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
  catch (const std::system_error &)
  {
    return {
      AttestedCameraFrameIngestStatus::RESOURCE_FAILURE,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt};
  }
#else
  (void)envelope;
  return {
    AttestedCameraFrameIngestStatus::
    DRIVER_CONTRACT_UNAVAILABLE,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt};
#endif
}

#ifdef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING
namespace test
{

void failNextCameraDriverEnvelopeIdentityAllocation()
{
  fail_next_envelope_identity_allocation.store(
    true, std::memory_order_release);
}

void failNextCameraDriverEnvelopeImplementationAllocation()
{
  fail_next_envelope_implementation_allocation.store(
    true, std::memory_order_release);
}

void failNextCameraDriverEnvelopePayloadAllocation()
{
  fail_next_envelope_payload_allocation.store(
    true, std::memory_order_release);
}

void failNextAttestedFrameIngressIdentityAllocation()
{
  fail_next_ingress_identity_allocation.store(
    true, std::memory_order_release);
}

void failNextAttestedFrameIngressImplementationAllocation()
{
  fail_next_ingress_implementation_allocation.store(
    true, std::memory_order_release);
}

void failNextAttestedFramePayloadAllocation()
{
  fail_next_attested_payload_allocation.store(
    true, std::memory_order_release);
}

void failNextCameraDriverEnvelopeDigestComputation()
{
  fail_next_envelope_digest_computation.store(
    true, std::memory_order_release);
}

void exhaustNextCameraDriverEnvelopeSequence(
  const std::uint8_t camera_id)
{
  if (camera_id < kAttestedCameraFrameCameraCount)
  {
    exhaust_next_envelope_sequence[camera_id].store(
      true, std::memory_order_release);
  }
}

void exhaustNextAttestedFrameIngressSequence(
  const std::uint8_t camera_id)
{
  if (camera_id < kAttestedCameraFrameCameraCount)
  {
    exhaust_next_ingress_sequence[camera_id].store(
      true, std::memory_order_release);
  }
}

void corruptNextCameraDriverEnvelopeDigest()
{
  corrupt_next_envelope_digest.store(
    true, std::memory_order_release);
}

void corruptNextCameraDriverEnvelopeMetadata()
{
  corrupt_next_envelope_metadata.store(
    true, std::memory_order_release);
}

void corruptNextCameraDriverEnvelopePixel()
{
  corrupt_next_envelope_pixel.store(
    true, std::memory_order_release);
}

}  // namespace test
#endif

}  // namespace livo
