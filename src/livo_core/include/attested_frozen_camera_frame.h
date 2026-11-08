#ifndef LIVO_ATTESTED_FROZEN_CAMERA_FRAME_H_
#define LIVO_ATTESTED_FROZEN_CAMERA_FRAME_H_

#include "exposure_timing_attestation.h"
#include "frozen_camera_image.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace livo
{

inline constexpr std::size_t kAttestedCameraFrameCameraCount =
  kExposureTimingCameraCount;
inline constexpr std::size_t
kAttestedCameraFrameIdentifierMaxLength = 128U;
inline constexpr std::size_t
kAttestedCameraFrameTopicMaxLength = 256U;
inline constexpr std::uint32_t
kAttestedCameraFrameMaximumImageDimension = 16384U;
inline constexpr std::uint64_t
kAttestedCameraFrameMaximumPayloadBytes =
  64U * 1024U * 1024U;

enum class CameraDriverClockDomain
{
  UNKNOWN,
  SENSOR_HARDWARE
};

enum class CameraDriverHardwareFrameCounterPolicy
{
  UNKNOWN,
  REQUIRED_MONOTONIC,
  UNAVAILABLE
};

enum class CameraDriverFrameEnvelopeAuthorityCreateStatus
{
  READY,
  DRIVER_CONTRACT_UNAVAILABLE,
  INVALID_CONFIGURATION,
  RESOURCE_FAILURE
};

enum class CameraDriverFrameEnvelopeIssueStatus
{
  READY,
  DRIVER_CONTRACT_UNAVAILABLE,
  INVALID_CAMERA_ID,
  FOREIGN_SOURCE_BINDING,
  FOREIGN_DRIVER_EPOCH,
  INVALID_DRIVER_FRAME_NUMBER,
  FOREIGN_HARDWARE_FRAME_COUNTER_POLICY,
  MISSING_REQUIRED_HARDWARE_FRAME_COUNTER,
  UNEXPECTED_HARDWARE_FRAME_COUNTER,
  INVALID_HARDWARE_FRAME_COUNTER,
  INVALID_CLOCK_DOMAIN,
  INVALID_TIMESTAMP,
  INVALID_EXPOSURE_DURATION,
  INVALID_IMAGE_CONTRACT,
  INVALID_FRAME_DATA,
  INVALID_METADATA_REVISION,
  INVALID_TRANSFORM_REVISION,
  STALE_OR_REPLAYED_DRIVER_FRAME,
  STALE_OR_REPLAYED_HARDWARE_COUNTER,
  RESOURCE_FAILURE,
  AUTHORITY_EXHAUSTED
};

using CameraDriverFrameEnvelopeDigest =
  std::array<unsigned char, 32U>;

// A configuration contains exactly four routes. Routes may be supplied in any
// order, but camera IDs must be the permutation [0, 3], and source IDs, topics,
// sensor serials, and frame IDs must each be unique across the four routes.
// Identifiers and revisions contain visible ASCII only; topics additionally
// start with '/'. Public maximum-length and payload constants above are hard
// contract limits. E0 accepts SENSOR_HARDWARE, packed mono8 only; step equals
// width and both image dimensions obey the public maximum.
struct CameraDriverFrameRouteContract
{
  ExposureTimingSourceBinding source_binding;
  std::uint64_t driver_epoch{0U};
  CameraDriverClockDomain clock_domain{
    CameraDriverClockDomain::UNKNOWN};
  CameraDriverHardwareFrameCounterPolicy
    hardware_frame_counter_policy{
    CameraDriverHardwareFrameCounterPolicy::UNKNOWN};
  std::uint32_t width{0U};
  std::uint32_t height{0U};
  std::string encoding;
  bool is_bigendian{false};
  std::uint32_t step{0U};
  std::string metadata_revision;
  std::string transform_revision;
};

struct CameraDriverFrameEnvelopeAuthorityConfiguration
{
  std::array<
    CameraDriverFrameRouteContract,
    kAttestedCameraFrameCameraCount> routes;
};

// One owned driver document. data contains exactly step * height bytes. The
// hardware-counter policy must equal its route policy: REQUIRED_MONOTONIC
// requires a counter on every frame; UNAVAILABLE requires no counter.
struct CameraDriverFrameDocument
{
  ExposureTimingSourceBinding source_binding;
  std::uint64_t driver_epoch{0U};
  std::uint64_t driver_frame_number{0U};
  CameraDriverHardwareFrameCounterPolicy
    hardware_frame_counter_policy{
    CameraDriverHardwareFrameCounterPolicy::UNKNOWN};
  std::optional<std::uint64_t> hardware_frame_counter;
  CameraDriverClockDomain clock_domain{
    CameraDriverClockDomain::UNKNOWN};
  double raw_header_timestamp_sec{0.0};
  std::optional<double> sensor_timestamp_sec;
  std::optional<double> exposure_duration_sec;
  std::uint32_t width{0U};
  std::uint32_t height{0U};
  std::string encoding;
  bool is_bigendian{false};
  std::uint32_t step{0U};
  std::vector<unsigned char> data;
  std::string metadata_revision;
  std::string transform_revision;
};

class CameraDriverFrameEnvelopeAuthority;
class CameraDriverFrameEnvelopeAuthorityIdentity;
class CameraDriverFrameEnvelopeAuditView;
class TrustedCameraDriverFrameEnvelope;
class AttestedCameraFrameIngress;
class AttestedCameraFrameIngressIdentity;
class AttestedFrozenCameraFrame;
class AttestedFrozenCameraFrameAuditView;


// Copies retain immutable issuer identity after authority destruction.
class CameraDriverFrameEnvelopeAuthorityIdentity final
{
public:
  CameraDriverFrameEnvelopeAuthorityIdentity(
    const CameraDriverFrameEnvelopeAuthorityIdentity &) noexcept =
    default;
  CameraDriverFrameEnvelopeAuthorityIdentity(
    CameraDriverFrameEnvelopeAuthorityIdentity &&) noexcept =
    default;
  CameraDriverFrameEnvelopeAuthorityIdentity & operator=(
    const CameraDriverFrameEnvelopeAuthorityIdentity &) noexcept =
    default;
  CameraDriverFrameEnvelopeAuthorityIdentity & operator=(
    CameraDriverFrameEnvelopeAuthorityIdentity &&) noexcept =
    default;

  bool valid() const noexcept;
  bool sameAuthority(
    const CameraDriverFrameEnvelopeAuthorityIdentity &
      other) const noexcept;

private:
  struct Data;

  explicit CameraDriverFrameEnvelopeAuthorityIdentity(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class CameraDriverFrameEnvelopeAuthority;
  friend class TrustedCameraDriverFrameEnvelope;
  friend class CameraDriverFrameEnvelopeAuditView;
  friend class AttestedCameraFrameIngress;
};

// Copies retain immutable envelope audit state after issuer destruction.
class CameraDriverFrameEnvelopeAuditView final
{
public:
  CameraDriverFrameEnvelopeAuditView(
    const CameraDriverFrameEnvelopeAuditView &) noexcept =
    default;
  CameraDriverFrameEnvelopeAuditView(
    CameraDriverFrameEnvelopeAuditView &&) noexcept = default;
  CameraDriverFrameEnvelopeAuditView & operator=(
    const CameraDriverFrameEnvelopeAuditView &) noexcept =
    default;
  CameraDriverFrameEnvelopeAuditView & operator=(
    CameraDriverFrameEnvelopeAuditView &&) noexcept = default;

  bool valid() const noexcept;
  std::uint8_t cameraId() const noexcept;
  std::uint64_t envelopeSequence() const noexcept;
  std::uint64_t driverEpoch() const noexcept;
  std::uint64_t driverFrameNumber() const noexcept;
  CameraDriverHardwareFrameCounterPolicy
  hardwareFrameCounterPolicy() const noexcept;
  std::optional<std::uint64_t>
  hardwareFrameCounter() const noexcept;
  CameraDriverClockDomain clockDomain() const noexcept;
  double rawHeaderTimestampSec() const noexcept;
  std::optional<double> sensorTimestampSec() const noexcept;
  std::optional<double> exposureDurationSec() const noexcept;
  std::uint32_t width() const noexcept;
  std::uint32_t height() const noexcept;
  const std::string & encoding() const noexcept;
  bool isBigendian() const noexcept;
  std::uint32_t step() const noexcept;
  std::size_t dataSize() const noexcept;
  const std::string & sourceId() const noexcept;
  const std::string & topic() const noexcept;
  const std::string & sensorSerial() const noexcept;
  const std::string & frameId() const noexcept;
  const std::string & metadataRevision() const noexcept;
  const std::string & transformRevision() const noexcept;
  CameraDriverFrameEnvelopeDigest
  canonicalDigest() const noexcept;

private:
  struct Data;

  explicit CameraDriverFrameEnvelopeAuditView(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class TrustedCameraDriverFrameEnvelope;
  friend class CameraDriverFrameEnvelopeAuthority;
  friend class AttestedCameraFrameIngress;
  friend class AttestedFrozenCameraFrame;
  friend class AttestedFrozenCameraFrameAuditView;
};

// Copies retain immutable envelope audit state after issuer destruction.
class TrustedCameraDriverFrameEnvelope final
{
public:
  TrustedCameraDriverFrameEnvelope(
    const TrustedCameraDriverFrameEnvelope &) noexcept = default;
  TrustedCameraDriverFrameEnvelope(
    TrustedCameraDriverFrameEnvelope &&) noexcept = default;
  TrustedCameraDriverFrameEnvelope & operator=(
    const TrustedCameraDriverFrameEnvelope &) noexcept = default;
  TrustedCameraDriverFrameEnvelope & operator=(
    TrustedCameraDriverFrameEnvelope &&) noexcept = default;

  bool valid() const noexcept;
  bool issuedBy(
    const CameraDriverFrameEnvelopeAuthorityIdentity &
      identity) const noexcept;
  bool sameFrame(
    const TrustedCameraDriverFrameEnvelope & other) const noexcept;
  std::optional<CameraDriverFrameEnvelopeAuditView>
  audit() const noexcept;

private:
  explicit TrustedCameraDriverFrameEnvelope(
    std::shared_ptr<const CameraDriverFrameEnvelopeAuditView::Data>
      data) noexcept;

  std::shared_ptr<const
    CameraDriverFrameEnvelopeAuditView::Data> data_;

  friend class CameraDriverFrameEnvelopeAuthority;
  friend class AttestedCameraFrameIngress;
  friend class AttestedFrozenCameraFrame;
};

struct CameraDriverFrameEnvelopeIssueResult
{
  CameraDriverFrameEnvelopeIssueStatus status{
    CameraDriverFrameEnvelopeIssueStatus::
    DRIVER_CONTRACT_UNAVAILABLE};
  std::optional<TrustedCameraDriverFrameEnvelope> envelope;

  bool ready() const noexcept
  {
    return
      status == CameraDriverFrameEnvelopeIssueStatus::READY &&
      envelope.has_value() &&
      envelope->valid();
  }
};

struct CameraDriverFrameEnvelopeAuthorityCreateResult
{
  CameraDriverFrameEnvelopeAuthorityCreateStatus status{
    CameraDriverFrameEnvelopeAuthorityCreateStatus::
    DRIVER_CONTRACT_UNAVAILABLE};
  std::unique_ptr<CameraDriverFrameEnvelopeAuthority> authority;

  bool ready() const noexcept
  {
    return
      status ==
      CameraDriverFrameEnvelopeAuthorityCreateStatus::READY &&
      static_cast<bool>(authority);
  }
};

// Uniquely owned per-route envelope issuer. Public member calls may run
// concurrently; destruction must not overlap a member call. Driver-frame and
// required hardware-counter high-water marks are strict per route. Failed
// attempts may leave sequence holes, and no cross-camera timing gate exists.
// Same-route concurrent calls linearize at commit; invocation order does not
// determine which frame commits first.
// The production archive is a fixed DRIVER_CONTRACT_UNAVAILABLE stub; the
// functional issuer exists only in the failure-injection test fixture.
class CameraDriverFrameEnvelopeAuthority final
{
public:
  static CameraDriverFrameEnvelopeAuthorityCreateResult create(
    const CameraDriverFrameEnvelopeAuthorityConfiguration &
      configuration) noexcept;

  ~CameraDriverFrameEnvelopeAuthority();

  CameraDriverFrameEnvelopeAuthority(
    const CameraDriverFrameEnvelopeAuthority &) = delete;
  CameraDriverFrameEnvelopeAuthority & operator=(
    const CameraDriverFrameEnvelopeAuthority &) = delete;
  CameraDriverFrameEnvelopeAuthority(
    CameraDriverFrameEnvelopeAuthority &&) = delete;
  CameraDriverFrameEnvelopeAuthority & operator=(
    CameraDriverFrameEnvelopeAuthority &&) = delete;

  CameraDriverFrameEnvelopeAuthorityIdentity
  identity() const noexcept;
  CameraDriverFrameEnvelopeIssueResult issue(
    const CameraDriverFrameDocument & document);

private:
  struct Implementation;

  explicit CameraDriverFrameEnvelopeAuthority(
    std::shared_ptr<Implementation> implementation) noexcept;

  std::shared_ptr<Implementation> implementation_;
};

enum class AttestedCameraFrameIngressCreateStatus
{
  READY,
  DRIVER_CONTRACT_UNAVAILABLE,
  INVALID_CONFIGURATION,
  INVALID_ENVELOPE_AUTHORITY_IDENTITY,
  ENVELOPE_CONTRACT_MISMATCH,
  INVALID_TIMING_SOURCE_AUTHORITY,
  INVALID_TIMING_AUTHORITY,
  TIMING_SOURCE_CONTRACT_MISMATCH,
  TIMING_SOURCE_AUTHORITY_MISMATCH,
  RESOURCE_FAILURE
};

enum class AttestedCameraFrameIngestStatus
{
  READY,
  DRIVER_CONTRACT_UNAVAILABLE,
  INVALID_ENVELOPE,
  FOREIGN_ENVELOPE_AUTHORITY,
  FOREIGN_ROUTE_CONTRACT,
  INVALID_ENVELOPE_DIGEST,
  STALE_OR_REPLAYED_ENVELOPE,
  IMAGE_FREEZE_FAILURE,
  TIMING_ISSUE_FAILURE,
  TIMING_ATTEST_FAILURE,
  SOURCE_BINDING_MISMATCH,
  RESOURCE_FAILURE,
  AUTHORITY_EXHAUSTED
};

struct AttestedCameraFrameIngressConfiguration
{
  std::array<
    CameraDriverFrameRouteContract,
    kAttestedCameraFrameCameraCount> routes;
};

// Copies retain immutable issuer identity after ingress destruction.
class AttestedCameraFrameIngressIdentity final
{
public:
  AttestedCameraFrameIngressIdentity(
    const AttestedCameraFrameIngressIdentity &) noexcept = default;
  AttestedCameraFrameIngressIdentity(
    AttestedCameraFrameIngressIdentity &&) noexcept = default;
  AttestedCameraFrameIngressIdentity & operator=(
    const AttestedCameraFrameIngressIdentity &) noexcept = default;
  AttestedCameraFrameIngressIdentity & operator=(
    AttestedCameraFrameIngressIdentity &&) noexcept = default;

  bool valid() const noexcept;
  bool sameIngress(
    const AttestedCameraFrameIngressIdentity &
      other) const noexcept;

private:
  struct Data;

  explicit AttestedCameraFrameIngressIdentity(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class AttestedCameraFrameIngress;
  friend class AttestedFrozenCameraFrame;
  friend class AttestedFrozenCameraFrameAuditView;
};

// Copies retain immutable grouped audit state after issuer destruction.
class AttestedFrozenCameraFrameAuditView final
{
public:
  AttestedFrozenCameraFrameAuditView(
    const AttestedFrozenCameraFrameAuditView &) noexcept = default;
  AttestedFrozenCameraFrameAuditView(
    AttestedFrozenCameraFrameAuditView &&) noexcept = default;
  AttestedFrozenCameraFrameAuditView & operator=(
    const AttestedFrozenCameraFrameAuditView &) noexcept = default;
  AttestedFrozenCameraFrameAuditView & operator=(
    AttestedFrozenCameraFrameAuditView &&) noexcept = default;

  bool valid() const noexcept;
  std::uint8_t cameraId() const noexcept;
  std::uint64_t ingressSequence() const noexcept;
  int imageRows() const noexcept;
  int imageColumns() const noexcept;
  int imageType() const noexcept;
  FrozenCameraImage::Digest imageDigest() const noexcept;
  std::optional<EffectiveExposureTime>
  effectiveExposureTime() const noexcept;
  std::optional<CameraDriverFrameEnvelopeAuditView>
  envelope() const noexcept;
  std::optional<ExposureTimingAuditView>
  timing() const noexcept;

private:
  struct Data;

  explicit AttestedFrozenCameraFrameAuditView(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class AttestedFrozenCameraFrame;
  friend class AttestedCameraFrameIngress;
};

// Copies retain immutable grouped audit state after issuer destruction.
class AttestedFrozenCameraFrame final
{
public:
  AttestedFrozenCameraFrame(
    const AttestedFrozenCameraFrame &) noexcept = default;
  AttestedFrozenCameraFrame(
    AttestedFrozenCameraFrame &&) noexcept = default;
  AttestedFrozenCameraFrame & operator=(
    const AttestedFrozenCameraFrame &) noexcept = default;
  AttestedFrozenCameraFrame & operator=(
    AttestedFrozenCameraFrame &&) noexcept = default;

  bool valid() const noexcept;
  bool issuedBy(
    const AttestedCameraFrameIngressIdentity &
      identity) const noexcept;
  bool sameFrame(
    const TrustedCameraDriverFrameEnvelope &
      envelope) const noexcept;
  std::optional<AttestedFrozenCameraFrameAuditView>
  audit() const noexcept;

private:
  explicit AttestedFrozenCameraFrame(
    std::shared_ptr<const
      AttestedFrozenCameraFrameAuditView::Data> data) noexcept;

  std::shared_ptr<const
    AttestedFrozenCameraFrameAuditView::Data> data_;

  friend class AttestedCameraFrameIngress;
};

struct AttestedCameraFrameIngestResult
{
  AttestedCameraFrameIngestStatus status{
    AttestedCameraFrameIngestStatus::
    DRIVER_CONTRACT_UNAVAILABLE};
  std::optional<FrozenCameraImageFreezeStatus>
    image_freeze_status;
  std::optional<ExposureFrameTimingIssueStatus>
    timing_issue_status;
  std::optional<ExposureTimingAttestStatus>
    timing_attest_status;
  std::optional<AttestedFrozenCameraFrame> frame;

  bool ready() const noexcept
  {
    return
      status == AttestedCameraFrameIngestStatus::READY &&
      frame.has_value() &&
      frame->valid();
  }
};

class AttestedCameraFrameIngress;

struct AttestedCameraFrameIngressCreateResult
{
  AttestedCameraFrameIngressCreateStatus status{
    AttestedCameraFrameIngressCreateStatus::
    DRIVER_CONTRACT_UNAVAILABLE};
  std::unique_ptr<AttestedCameraFrameIngress> ingress;

  bool ready() const noexcept
  {
    return
      status == AttestedCameraFrameIngressCreateStatus::READY &&
      static_cast<bool>(ingress);
  }
};

// Uniquely owned image/timing ingress. Public member calls may run concurrently;
// destruction must not overlap a member call. Each route advances independently
// and no cross-camera timestamp comparison is performed. create() consumes both
// unique timing authorities even when validation or allocation fails.
// Same-route concurrent calls linearize at commit; invocation order does not
// determine which envelope commits first.
// The production archive is a fixed DRIVER_CONTRACT_UNAVAILABLE stub; the
// functional ingress exists only in the failure-injection test fixture.
class AttestedCameraFrameIngress final
{
public:
  static AttestedCameraFrameIngressCreateResult create(
    const AttestedCameraFrameIngressConfiguration &
      configuration,
    const CameraDriverFrameEnvelopeAuthorityIdentity &
      expected_envelope_authority,
    std::unique_ptr<ExposureTimingSourceAuthority>
      timing_source_authority,
    std::unique_ptr<ExposureTimingAuthority>
      timing_authority) noexcept;

  ~AttestedCameraFrameIngress();

  AttestedCameraFrameIngress(
    const AttestedCameraFrameIngress &) = delete;
  AttestedCameraFrameIngress & operator=(
    const AttestedCameraFrameIngress &) = delete;
  AttestedCameraFrameIngress(
    AttestedCameraFrameIngress &&) = delete;
  AttestedCameraFrameIngress & operator=(
    AttestedCameraFrameIngress &&) = delete;

  AttestedCameraFrameIngressIdentity identity() const noexcept;
  AttestedCameraFrameIngestResult ingest(
    const TrustedCameraDriverFrameEnvelope & envelope);

private:
  struct Implementation;

  explicit AttestedCameraFrameIngress(
    std::shared_ptr<Implementation> implementation) noexcept;

  std::shared_ptr<Implementation> implementation_;
};

}  // namespace livo

#endif  // LIVO_ATTESTED_FROZEN_CAMERA_FRAME_H_
