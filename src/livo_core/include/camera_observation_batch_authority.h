#ifndef LIVO_CAMERA_OBSERVATION_BATCH_AUTHORITY_H_
#define LIVO_CAMERA_OBSERVATION_BATCH_AUTHORITY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <opencv2/core/mat.hpp>

struct CameraObservationBatch;
class CameraObservationQueue;

namespace livo
{

inline constexpr std::size_t kCameraObservationAuthorityCameraCount = 4U;

class CameraObservationBatchReceipt;
class AdaptiveVisualSelectionAuthority;
class AdaptiveVisualPolicy;
class ReferenceOnlyMigrationRuntimePublisher;

// Opaque identity of the queue which owns observation and batch sequencing.
// The identity is intentionally copyable so consumers can pin one authority,
// but only CameraObservationQueue can issue a valid identity.
class CameraObservationBatchAuthorityIdentity
{
public:
  CameraObservationBatchAuthorityIdentity() = default;

  bool valid() const noexcept;
  bool sameAuthority(
    const CameraObservationBatchAuthorityIdentity & other) const noexcept;

private:
  explicit CameraObservationBatchAuthorityIdentity(
    std::shared_ptr<const void> authority);

  static CameraObservationBatchAuthorityIdentity issue();

  std::shared_ptr<const void> authority_;

  friend class ::CameraObservationQueue;
  friend class CameraObservationBatchReceipt;
};

// Immutable capability sealing an owner-issued batch. The receipt binds the
// queue authority, checked batch sequence, per-camera observation/image
// generations, batch metadata, and a SHA-256 digest of every present image.
class CameraObservationBatchReceipt
{
public:
  CameraObservationBatchReceipt(
    const CameraObservationBatchReceipt &) = default;
  CameraObservationBatchReceipt(
    CameraObservationBatchReceipt &&) = default;
  CameraObservationBatchReceipt & operator=(
    const CameraObservationBatchReceipt &) = default;
  CameraObservationBatchReceipt & operator=(
    CameraObservationBatchReceipt &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const CameraObservationBatchAuthorityIdentity & identity) const noexcept;

  std::uint64_t batchSequence() const noexcept;
  std::uint64_t observationGeneration(
    std::uint8_t camera_id) const noexcept;
  std::uint64_t imageGeneration(
    std::uint8_t camera_id) const noexcept;
  double cameraExposureTimestampSec(
    std::uint8_t camera_id) const noexcept;
  std::uint64_t sealImageHashCount() const noexcept;
  // Counts validation attempts that hash sealed images. A rejected attempt
  // may increment this diagnostic; it is not policy or selection-claim state.
  std::uint64_t validationImageHashCount() const noexcept;
  bool adaptiveVisualSelectionClaimed() const noexcept;

private:
  struct SealedData;

  explicit CameraObservationBatchReceipt(
    std::shared_ptr<const SealedData> sealed_data);

  static std::optional<CameraObservationBatchReceipt> seal(
    const ::CameraObservationBatch & batch,
    std::uint64_t batch_sequence,
    const CameraObservationBatchAuthorityIdentity & identity);
  static bool freezeAndDigestImage(
    const cv::Mat & source,
    cv::Mat & frozen,
    std::array<unsigned char, 32U> & digest) noexcept;
  bool sameSealedBatch(
    const CameraObservationBatchReceipt & other) const noexcept;
  const std::array<bool, kCameraObservationAuthorityCameraCount> &
  sealedPresentMask() const;
  std::size_t sealedCameraCount() const noexcept;
  bool sealedPartial() const noexcept;
  double sealedBatchTimestampSec() const noexcept;
  double sealedMaxTimeOffsetSec() const noexcept;
  double sealedExposureTimestampSec(
    std::uint8_t camera_id) const noexcept;
  bool matchesAndValidates(
    const ::CameraObservationBatch & batch) const noexcept;
  bool claimAndValidate(
    const ::CameraObservationBatch & batch) const noexcept;
  bool claimAdaptiveVisualSelection() const noexcept;

  std::shared_ptr<const SealedData> sealed_data_;

  friend class ::CameraObservationQueue;
  friend class AdaptiveVisualSelectionAuthority;
  friend class AdaptiveVisualPolicy;
  friend class ReferenceOnlyMigrationRuntimePublisher;
};

}  // namespace livo

#endif  // LIVO_CAMERA_OBSERVATION_BATCH_AUTHORITY_H_
