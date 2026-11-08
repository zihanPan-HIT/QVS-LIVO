#ifndef LIVO_EXPOSURE_TIMING_ATTESTATION_H_
#define LIVO_EXPOSURE_TIMING_ATTESTATION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace livo
{

inline constexpr std::size_t kExposureTimingCameraCount = 4U;
using ExposureTimingArtifactDigest =
  std::array<unsigned char, 32U>;

enum class ExposureTimingAuthorityMode
{
  UNKNOWN,
  PRODUCTION,
  SIMULATION_TEST
};

enum class ExposureTimingProvenanceKind
{
  UNKNOWN,
  HARDWARE_CALIBRATION,
  SIMULATION,
  TEST_FIXTURE
};

enum class ExposureTimestampConvention
{
  UNKNOWN,
  START,
  MIDPOINT,
  END
};

enum class ExposureTimingCalibrationStatus
{
  READY,
  INVALID_AUTHORITY_MODE,
  MISSING_CALIBRATION_PROVENANCE,
  PROVENANCE_MODE_MISMATCH,
  INVALID_CALIBRATION_REVISION,
  INVALID_CANONICAL_PAYLOAD,
  INVALID_CANONICAL_PAYLOAD_HASH,
  INVALID_ARTIFACT_ID,
  INVALID_CAMERA_ID,
  INVALID_SOURCE_BINDING,
  UNKNOWN_TIMESTAMP_CONVENTION,
  INVALID_STATIC_OFFSET,
  NONZERO_OR_INVALID_LEGACY_EXPOSURE_OFFSET,
  RESOURCE_FAILURE,
  FIXTURE_MINT_UNAVAILABLE
};

enum class ExposureTimingSourceAuthorityCreateStatus
{
  READY,
  INVALID_CAMERA_ID,
  INVALID_SOURCE_BINDING,
  RESOURCE_FAILURE
};

enum class ExposureFrameTimingIssueStatus
{
  READY,
  INVALID_CAMERA_ID,
  INVALID_RAW_HEADER_TIMESTAMP,
  INVALID_EXPOSURE_DURATION,
  RESOURCE_FAILURE,
  AUTHORITY_EXHAUSTED
};

enum class ExposureTimingAuthorityCreateStatus
{
  READY,
  INVALID_TRUSTED_CALIBRATION,
  INVALID_SOURCE_AUTHORITY_IDENTITY,
  CALIBRATION_SOURCE_BINDING_MISMATCH,
  RESOURCE_FAILURE
};

enum class ExposureTimingAttestStatus
{
  READY,
  INVALID_FRAME_RECEIPT,
  FOREIGN_SOURCE_RECEIPT,
  UNKNOWN_TIMESTAMP_CONVENTION,
  EXPOSURE_DURATION_REQUIRED,
  INVALID_EXPOSURE_DURATION,
  TIMESTAMP_OVERFLOW,
  TIMESTAMP_ARITHMETIC_FAILURE,
  STALE_OR_REPLAYED_SOURCE_SEQUENCE,
  RESOURCE_FAILURE,
  AUTHORITY_EXHAUSTED
};

// All timestamps, exposure durations, and timing offsets in this
// interface are expressed in seconds.
struct ExposureTimingSourceBinding
{
  std::uint8_t camera_id{0U};
  std::string source_id;
  std::string topic;
  std::string sensor_serial;
  std::string frame_id;
};

struct ExposureTimingCameraCalibration
{
  ExposureTimingSourceBinding source_binding;
  ExposureTimestampConvention timestamp_convention{
    ExposureTimestampConvention::UNKNOWN};
  double per_camera_offset_sec{0.0};
  std::string duration_source;
};

struct ExposureTimingCalibrationDocument
{
  ExposureTimingAuthorityMode mode{
    ExposureTimingAuthorityMode::UNKNOWN};
  ExposureTimingProvenanceKind provenance{
    ExposureTimingProvenanceKind::UNKNOWN};
  std::string canonical_payload;
  ExposureTimingArtifactDigest canonical_payload_sha256{};
  std::string artifact_id;
  std::uint64_t calibration_revision{0U};
  double global_offset_sec{0.0};
  double legacy_exposure_time_init_sec{0.0};
  std::array<
    ExposureTimingCameraCalibration,
    kExposureTimingCameraCount> cameras;
};

struct ExposureTimingSourceAuthorityConfiguration
{
  std::array<
    ExposureTimingSourceBinding,
    kExposureTimingCameraCount> cameras;
};

class ExposureTimingCalibrationLoader;
class ExposureTimingAuthority;
class ExposureTimingAuthorityIdentity;
class ExposureTimingAttestation;
class ExposureTimingAttestationIdentity;
class ExposureTimingAuditView;
class ExposureTimingSourceAuthority;
class ExposureTimingSourceAuthorityIdentity;
class ExposureFrameTimingReceipt;
class ExposureFrameTimingIdentity;
struct ExposureTimingAuthorityCreateResult;
struct ExposureTimingSourceAuthorityCreateResult;
struct ExposureFrameTimingIssueResult;

// Immutable shared handle. Copies are constant-time and remain valid
// independently of the loader or authority that received them.
class TrustedExposureTimingCalibration final
{
public:
  TrustedExposureTimingCalibration(
    const TrustedExposureTimingCalibration &) noexcept = default;
  TrustedExposureTimingCalibration(
    TrustedExposureTimingCalibration &&) noexcept = default;
  TrustedExposureTimingCalibration & operator=(
    const TrustedExposureTimingCalibration &) noexcept = default;
  TrustedExposureTimingCalibration & operator=(
    TrustedExposureTimingCalibration &&) noexcept = default;

  bool valid() const noexcept;

private:
  struct Data;

  explicit TrustedExposureTimingCalibration(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class ExposureTimingCalibrationLoader;
  friend class ExposureTimingAuthority;
  friend class ExposureTimingAuditView;
};

struct TrustedExposureTimingCalibrationResult
{
  ExposureTimingCalibrationStatus status{
    ExposureTimingCalibrationStatus::RESOURCE_FAILURE};
  std::optional<TrustedExposureTimingCalibration> calibration;

  bool ready() const noexcept
  {
    return
      status == ExposureTimingCalibrationStatus::READY &&
      calibration.has_value() &&
      calibration->valid();
  }
};

class ExposureTimingCalibrationLoader final
{
public:
  // Archive-defined linkage seam. Production returns
  // FIXTURE_MINT_UNAVAILABLE without inspecting the document;
  // the failure-injection archive validates and mints fixtures.
  static TrustedExposureTimingCalibrationResult mintFixture(
    const ExposureTimingCalibrationDocument & document) noexcept;

  ExposureTimingCalibrationLoader() = delete;
};

// Immutable shared identity that remains valid after its source
// authority is destroyed.
class ExposureTimingSourceAuthorityIdentity final
{
public:
  ExposureTimingSourceAuthorityIdentity(
    const ExposureTimingSourceAuthorityIdentity &) noexcept = default;
  ExposureTimingSourceAuthorityIdentity(
    ExposureTimingSourceAuthorityIdentity &&) noexcept = default;
  ExposureTimingSourceAuthorityIdentity & operator=(
    const ExposureTimingSourceAuthorityIdentity &) noexcept = default;
  ExposureTimingSourceAuthorityIdentity & operator=(
    ExposureTimingSourceAuthorityIdentity &&) noexcept = default;

  bool valid() const noexcept;
  bool sameAuthority(
    const ExposureTimingSourceAuthorityIdentity & other) const noexcept;

private:
  struct Data;

  explicit ExposureTimingSourceAuthorityIdentity(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class ExposureTimingSourceAuthority;
  friend class ExposureFrameTimingReceipt;
  friend class ExposureTimingAuthority;
  friend class ExposureTimingAttestation;
  friend class ExposureTimingAuditView;
};

// Immutable shared frame identity that follows the receipt lifetime.
class ExposureFrameTimingIdentity final
{
public:
  ExposureFrameTimingIdentity(
    const ExposureFrameTimingIdentity &) noexcept = default;
  ExposureFrameTimingIdentity(
    ExposureFrameTimingIdentity &&) noexcept = default;
  ExposureFrameTimingIdentity & operator=(
    const ExposureFrameTimingIdentity &) noexcept = default;
  ExposureFrameTimingIdentity & operator=(
    ExposureFrameTimingIdentity &&) noexcept = default;

  bool valid() const noexcept;
  bool sameFrame(
    const ExposureFrameTimingIdentity & other) const noexcept;

private:
  explicit ExposureFrameTimingIdentity(
    std::shared_ptr<const void> payload) noexcept;

  std::shared_ptr<const void> payload_;

  friend class ExposureFrameTimingReceipt;
  friend class ExposureTimingAttestation;
  friend class ExposureTimingAuditView;
};

// Immutable shared frame-timing receipt. Copies and derived identities
// remain valid after the issuing source authority is destroyed.
class ExposureFrameTimingReceipt final
{
public:
  ExposureFrameTimingReceipt(
    const ExposureFrameTimingReceipt &) noexcept = default;
  ExposureFrameTimingReceipt(
    ExposureFrameTimingReceipt &&) noexcept = default;
  ExposureFrameTimingReceipt & operator=(
    const ExposureFrameTimingReceipt &) noexcept = default;
  ExposureFrameTimingReceipt & operator=(
    ExposureFrameTimingReceipt &&) noexcept = default;

  bool valid() const noexcept;
  bool issuedBy(
    const ExposureTimingSourceAuthorityIdentity & identity) const noexcept;
  bool sameFrame(
    const ExposureFrameTimingReceipt & other) const noexcept;
  ExposureFrameTimingIdentity frameIdentity() const noexcept;
  std::uint8_t cameraId() const noexcept;
  std::uint64_t sourceSequence() const noexcept;

private:
  struct Data;

  explicit ExposureFrameTimingReceipt(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class ExposureTimingSourceAuthority;
  friend class ExposureTimingAuthority;
  friend class ExposureTimingAttestation;
  friend class ExposureTimingAuditView;
};

struct ExposureFrameTimingIssueResult
{
  ExposureFrameTimingIssueStatus status{
    ExposureFrameTimingIssueStatus::RESOURCE_FAILURE};
  std::optional<ExposureFrameTimingReceipt> receipt;

  bool ready() const noexcept
  {
    return
      status == ExposureFrameTimingIssueStatus::READY &&
      receipt.has_value() &&
      receipt->valid();
  }
};

// Uniquely owned four-camera sequence authority. Public member calls
// may run concurrently. Destruction must not overlap any member call.
class ExposureTimingSourceAuthority final
{
public:
  static ExposureTimingSourceAuthorityCreateResult create(
    const ExposureTimingSourceAuthorityConfiguration &
      configuration) noexcept;

  ~ExposureTimingSourceAuthority();

  ExposureTimingSourceAuthority(
    const ExposureTimingSourceAuthority &) = delete;
  ExposureTimingSourceAuthority & operator=(
    const ExposureTimingSourceAuthority &) = delete;
  ExposureTimingSourceAuthority(
    ExposureTimingSourceAuthority &&) = delete;
  ExposureTimingSourceAuthority & operator=(
    ExposureTimingSourceAuthority &&) = delete;

  ExposureTimingSourceAuthorityIdentity identity() const noexcept;
  // Returns true only for a valid authority and a complete binding value
  // match (camera ID, source ID, topic, sensor serial, and frame ID).
  // Invalid authorities and invalid or unmatched bindings return false.
  // This predicate does not expose the stored configuration.
  bool configuredFor(
    const ExposureTimingSourceBinding & binding) const noexcept;
  // raw_header_timestamp_sec and exposure_duration_sec use seconds.
  ExposureFrameTimingIssueResult issue(
    std::uint8_t camera_id,
    double raw_header_timestamp_sec,
    std::optional<double> exposure_duration_sec);

private:
  struct Implementation;

  explicit ExposureTimingSourceAuthority(
    std::shared_ptr<Implementation> implementation) noexcept;

  std::shared_ptr<Implementation> implementation_;
};

struct ExposureTimingSourceAuthorityCreateResult
{
  ExposureTimingSourceAuthorityCreateStatus status{
    ExposureTimingSourceAuthorityCreateStatus::RESOURCE_FAILURE};
  std::unique_ptr<ExposureTimingSourceAuthority> authority;

  bool ready() const noexcept
  {
    return
      status == ExposureTimingSourceAuthorityCreateStatus::READY &&
      static_cast<bool>(authority);
  }
};

// Strong solver-facing timestamp value expressed in seconds.
class EffectiveExposureTime final
{
public:
  EffectiveExposureTime(
    const EffectiveExposureTime &) noexcept = default;
  EffectiveExposureTime(
    EffectiveExposureTime &&) noexcept = default;
  EffectiveExposureTime & operator=(
    const EffectiveExposureTime &) noexcept = default;
  EffectiveExposureTime & operator=(
    EffectiveExposureTime &&) noexcept = default;

  double seconds() const noexcept;

private:
  explicit EffectiveExposureTime(double seconds) noexcept;

  double seconds_;

  friend class ExposureTimingAttestation;
};

// Immutable shared audit view. Timestamp, duration, and offset accessors
// ending in Sec return seconds.
class ExposureTimingAuditView final
{
public:
  ExposureTimingAuditView(
    const ExposureTimingAuditView &) noexcept = default;
  ExposureTimingAuditView(
    ExposureTimingAuditView &&) noexcept = default;
  ExposureTimingAuditView & operator=(
    const ExposureTimingAuditView &) noexcept = default;
  ExposureTimingAuditView & operator=(
    ExposureTimingAuditView &&) noexcept = default;

  bool valid() const noexcept;
  std::uint64_t attestationSequence() const noexcept;
  std::uint8_t cameraId() const noexcept;
  std::uint64_t sourceSequence() const noexcept;
  ExposureTimingSourceAuthorityIdentity
  sourceAuthorityIdentity() const noexcept;
  ExposureFrameTimingIdentity sourceFrameIdentity() const noexcept;
  const std::string & sourceId() const noexcept;
  const std::string & topic() const noexcept;
  const std::string & sensorSerial() const noexcept;
  const std::string & frameId() const noexcept;
  ExposureTimestampConvention timestampConvention() const noexcept;
  double rawHeaderTimestampSec() const noexcept;
  std::optional<double> exposureDurationSec() const noexcept;
  const std::string & durationSource() const noexcept;
  double midpointTimestampSec() const noexcept;
  double globalOffsetSec() const noexcept;
  double perCameraOffsetSec() const noexcept;
  double totalOffsetSec() const noexcept;
  double effectiveTimestampSec() const noexcept;
  ExposureTimingAuthorityMode authorityMode() const noexcept;
  ExposureTimingProvenanceKind provenanceKind() const noexcept;
  std::uint64_t calibrationRevision() const noexcept;
  const std::string & artifactId() const noexcept;
  const std::string & canonicalPayload() const noexcept;
  ExposureTimingArtifactDigest canonicalPayloadSha256() const noexcept;

private:
  struct Data;

  explicit ExposureTimingAuditView(
    std::shared_ptr<const Data> data) noexcept;

  std::shared_ptr<const Data> data_;

  friend class ExposureTimingAttestation;
  friend class ExposureTimingAuthority;
};

// Immutable shared identity that remains valid after its timing
// authority is destroyed.
class ExposureTimingAuthorityIdentity final
{
public:
  ExposureTimingAuthorityIdentity(
    const ExposureTimingAuthorityIdentity &) noexcept = default;
  ExposureTimingAuthorityIdentity(
    ExposureTimingAuthorityIdentity &&) noexcept = default;
  ExposureTimingAuthorityIdentity & operator=(
    const ExposureTimingAuthorityIdentity &) noexcept = default;
  ExposureTimingAuthorityIdentity & operator=(
    ExposureTimingAuthorityIdentity &&) noexcept = default;

  bool valid() const noexcept;
  bool sameAuthority(
    const ExposureTimingAuthorityIdentity & other) const noexcept;

private:
  explicit ExposureTimingAuthorityIdentity(
    std::shared_ptr<const void> owner) noexcept;

  std::shared_ptr<const void> owner_;

  friend class ExposureTimingAuthority;
  friend class ExposureTimingAttestation;
};

// Immutable shared identity that follows the attestation lifetime.
class ExposureTimingAttestationIdentity final
{
public:
  ExposureTimingAttestationIdentity(
    const ExposureTimingAttestationIdentity &) noexcept = default;
  ExposureTimingAttestationIdentity(
    ExposureTimingAttestationIdentity &&) noexcept = default;
  ExposureTimingAttestationIdentity & operator=(
    const ExposureTimingAttestationIdentity &) noexcept = default;
  ExposureTimingAttestationIdentity & operator=(
    ExposureTimingAttestationIdentity &&) noexcept = default;

  bool valid() const noexcept;
  bool samePayload(
    const ExposureTimingAttestationIdentity & other) const noexcept;

private:
  explicit ExposureTimingAttestationIdentity(
    std::shared_ptr<const void> payload) noexcept;

  std::shared_ptr<const void> payload_;

  friend class ExposureTimingAttestation;
};

// Immutable shared result retaining the exact calibration and source
// frame receipt. It remains valid after either authority is destroyed.
class ExposureTimingAttestation final
{
public:
  ExposureTimingAttestation(
    const ExposureTimingAttestation &) noexcept = default;
  ExposureTimingAttestation(
    ExposureTimingAttestation &&) noexcept = default;
  ExposureTimingAttestation & operator=(
    const ExposureTimingAttestation &) noexcept = default;
  ExposureTimingAttestation & operator=(
    ExposureTimingAttestation &&) noexcept = default;

  bool valid() const noexcept;
  bool issuedBy(
    const ExposureTimingAuthorityIdentity & identity) const noexcept;
  bool sameAttestation(
    const ExposureTimingAttestation & other) const noexcept;
  ExposureTimingAttestationIdentity payloadIdentity() const noexcept;
  std::uint8_t cameraId() const noexcept;
  std::uint64_t sourceSequence() const noexcept;
  ExposureTimingSourceAuthorityIdentity
  sourceAuthorityIdentity() const noexcept;
  ExposureFrameTimingIdentity sourceFrameIdentity() const noexcept;
  std::optional<EffectiveExposureTime>
  effectiveExposureTime() const noexcept;
  std::optional<ExposureTimingAuditView> audit() const noexcept;

private:
  explicit ExposureTimingAttestation(
    std::shared_ptr<const ExposureTimingAuditView::Data>
      data) noexcept;

  std::shared_ptr<const ExposureTimingAuditView::Data> data_;

  friend class ExposureTimingAuthority;
};

struct ExposureTimingAttestResult
{
  ExposureTimingAttestStatus status{
    ExposureTimingAttestStatus::RESOURCE_FAILURE};
  std::optional<ExposureTimingAttestation> attestation;

  bool ready() const noexcept
  {
    return
      status == ExposureTimingAttestStatus::READY &&
      attestation.has_value() &&
      attestation->valid();
  }
};

// Uniquely owned timing authority. Public member calls may run
// concurrently. Destruction must not overlap any member call.
class ExposureTimingAuthority final
{
public:
  static ExposureTimingAuthorityCreateResult create(
    const TrustedExposureTimingCalibration & calibration,
    const ExposureTimingSourceAuthorityIdentity &
      source_authority) noexcept;

  ~ExposureTimingAuthority();

  ExposureTimingAuthority(
    const ExposureTimingAuthority &) = delete;
  ExposureTimingAuthority & operator=(
    const ExposureTimingAuthority &) = delete;
  ExposureTimingAuthority(
    ExposureTimingAuthority &&) = delete;
  ExposureTimingAuthority & operator=(
    ExposureTimingAuthority &&) = delete;

  ExposureTimingAuthorityIdentity identity() const noexcept;
  // Returns true only for a valid timing authority bound to the exact
  // shared-owner source-authority identity. Invalid authorities or source
  // identities return false. This predicate exposes no configuration.
  bool boundTo(
    const ExposureTimingSourceAuthorityIdentity &
      source_authority) const noexcept;
  ExposureTimingAttestResult attest(
    const ExposureFrameTimingReceipt & receipt);

private:
  struct Implementation;

  explicit ExposureTimingAuthority(
    std::shared_ptr<Implementation> implementation) noexcept;

  std::shared_ptr<Implementation> implementation_;
};

struct ExposureTimingAuthorityCreateResult
{
  ExposureTimingAuthorityCreateStatus status{
    ExposureTimingAuthorityCreateStatus::
    INVALID_TRUSTED_CALIBRATION};
  std::unique_ptr<ExposureTimingAuthority> authority;

  bool ready() const noexcept
  {
    return
      status == ExposureTimingAuthorityCreateStatus::READY &&
      static_cast<bool>(authority);
  }
};

}  // namespace livo

#endif  // LIVO_EXPOSURE_TIMING_ATTESTATION_H_
