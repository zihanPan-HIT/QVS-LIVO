#ifndef LIVO_REFERENCE_ONLY_MIGRATION_RUNTIME_PROVENANCE_H_
#define LIVO_REFERENCE_ONLY_MIGRATION_RUNTIME_PROVENANCE_H_

#include "camera_batch_time_transfer.h"
#include "camera_observation_batch_authority.h"
#include "camera_time_transfer.h"
#include "common_lib.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace livo
{

enum class ActiveStateSnapshotStatus
{
  kValid,
  kInvalidNominalState,
  kInvalidCovariance
};

const char * activeStateSnapshotStatusName(
  ActiveStateSnapshotStatus status) noexcept;

class ReferenceOnlyMigrationRuntimeBundle;
class ReferenceOnlyMigrationFrameProvenance;
class ValidatedBatchRuntimeComposition;

class ReferenceOnlyMigrationRuntimePublisherIdentity
{
public:
  ReferenceOnlyMigrationRuntimePublisherIdentity() = default;

  bool valid() const noexcept;
  bool sameAuthority(
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      other) const noexcept;
  bool bindsBatchAuthority(
    const CameraObservationBatchAuthorityIdentity &
      batch_authority) const noexcept;

private:
  explicit ReferenceOnlyMigrationRuntimePublisherIdentity(
    std::shared_ptr<const void> authority,
    CameraObservationBatchAuthorityIdentity batch_authority);

  std::shared_ptr<const void> authority_;
  CameraObservationBatchAuthorityIdentity batch_authority_;

  friend class ReferenceOnlyMigrationRuntimePublisher;
  friend class ReferenceOnlyMigrationRuntimeBundle;
  friend class ReferenceOnlyMigrationFrameProvenance;
  friend class ValidatedBatchRuntimeComposition;
};

// Immutable authority capability proving that one sealed camera batch, one
// runtime publication, and their per-camera frame bindings were composed at
// the publisher boundary. The batch image digests are checked exactly once
// before publication; downstream consumers only compare opaque ownership and
// frozen scalar bindings through this capability.
class ValidatedBatchRuntimeComposition
{
public:
  ValidatedBatchRuntimeComposition(
    const ValidatedBatchRuntimeComposition &) = default;
  ValidatedBatchRuntimeComposition(
    ValidatedBatchRuntimeComposition &&) = default;
  ValidatedBatchRuntimeComposition & operator=(
    const ValidatedBatchRuntimeComposition &) = default;
  ValidatedBatchRuntimeComposition & operator=(
    ValidatedBatchRuntimeComposition &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const CameraObservationBatchAuthorityIdentity &
      batch_identity,
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      runtime_identity) const noexcept;
  bool issuedByBatch(
    const CameraObservationBatchAuthorityIdentity &
      identity) const noexcept;
  bool issuedByRuntime(
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      identity) const noexcept;
  bool accepts(
    const ReferenceOnlyMigrationRuntimeBundle &
      runtime) const noexcept;
  bool accepts(
    const ReferenceOnlyMigrationFrameProvenance &
      frame) const noexcept;

  std::uint64_t cameraBatchSequence() const noexcept;
  std::uint64_t runtimePublicationSequence() const noexcept;
  std::uint64_t activeStateSnapshotRevision() const noexcept;
  bool cameraPresent(std::uint8_t camera_id) const noexcept;
  std::uint64_t observationGeneration(
    std::uint8_t camera_id) const noexcept;
  std::uint64_t imageGeneration(
    std::uint8_t camera_id) const noexcept;
  double exposureTimestampSec(
    std::uint8_t camera_id) const noexcept;
  int imageRows(std::uint8_t camera_id) const noexcept;
  int imageColumns(std::uint8_t camera_id) const noexcept;
  int imageType(std::uint8_t camera_id) const noexcept;
  PhotometricStatus photometricStatus(
    std::uint8_t camera_id) const noexcept;

private:
  struct Binding;

  explicit ValidatedBatchRuntimeComposition(
    std::shared_ptr<const Binding> binding);

  std::shared_ptr<const Binding> binding_;

  friend class ReferenceOnlyMigrationRuntimePublisher;
  friend class ReferenceOnlyMigrationRuntimeBundle;
};

// Per-camera projection of one immutable runtime publication. The projection
// shares the original publication ownership and carries no image or duplicate
// 19D state. It remains an experimental, read-only trace and cannot authorize
// a solver or map write.
class ReferenceOnlyMigrationFrameProvenance
{
public:
  ReferenceOnlyMigrationFrameProvenance(
    const ReferenceOnlyMigrationFrameProvenance &) = default;
  ReferenceOnlyMigrationFrameProvenance(
    ReferenceOnlyMigrationFrameProvenance &&) = default;
  ReferenceOnlyMigrationFrameProvenance & operator=(
    const ReferenceOnlyMigrationFrameProvenance &) = default;
  ReferenceOnlyMigrationFrameProvenance & operator=(
    ReferenceOnlyMigrationFrameProvenance &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      identity) const noexcept;
  bool accepts(
    const ValidatedBatchRuntimeComposition &
      composition) const noexcept;

  std::uint64_t cameraBatchSequence() const noexcept;
  std::uint64_t runtimePublicationSequence() const noexcept;
  std::uint64_t activeStateSnapshotRevision() const noexcept;
  std::uint8_t cameraId() const noexcept;
  std::uint64_t observationGeneration() const noexcept;
  std::uint64_t imageGeneration() const noexcept;
  double exposureTimestampSec() const noexcept;
  const CameraBatchTimeTransferEntry &
  timeTransferEntry() const;

  static constexpr bool experimentalOnly() noexcept
  {
    return true;
  }

  static constexpr bool consumable() noexcept
  {
    return false;
  }

  static constexpr bool productionReady() noexcept
  {
    return false;
  }

  static constexpr bool solverApplied() noexcept
  {
    return false;
  }

private:
  struct Binding;

  explicit ReferenceOnlyMigrationFrameProvenance(
    std::shared_ptr<const Binding> binding);

  std::shared_ptr<const Binding> binding_;

  friend class ValidatedBatchRuntimeComposition;
  friend class ReferenceOnlyMigrationRuntimeBundle;
};

// Owns the only state-publication boundary used by the R2b metrics shadow.
// It constructs the frozen time-transfer bundle itself from the exact state
// snapshot, so callers cannot pair an independently built transfer with a
// different covariance snapshot.
class ReferenceOnlyMigrationRuntimePublisher
{
public:
  explicit ReferenceOnlyMigrationRuntimePublisher(
    CameraObservationBatchAuthorityIdentity batch_authority);

  ReferenceOnlyMigrationRuntimePublisher(
    const ReferenceOnlyMigrationRuntimePublisher &) = delete;
  ReferenceOnlyMigrationRuntimePublisher & operator=(
    const ReferenceOnlyMigrationRuntimePublisher &) = delete;
  ReferenceOnlyMigrationRuntimePublisher(
    ReferenceOnlyMigrationRuntimePublisher &&) = delete;
  ReferenceOnlyMigrationRuntimePublisher & operator=(
    ReferenceOnlyMigrationRuntimePublisher &&) = delete;

  bool valid() const noexcept;
  bool exhausted() const noexcept;
  ReferenceOnlyMigrationRuntimePublisherIdentity
  identity() const noexcept;

  std::optional<ReferenceOnlyMigrationRuntimeBundle> publish(
    const CameraObservationBatch & batch,
    const StatesGroup & active_state,
    double anchor_timestamp_sec,
    double exposure_time_sec,
    const CameraTimeTransfer & time_transfer);

private:
  CameraObservationBatchAuthorityIdentity batch_authority_;
  ReferenceOnlyMigrationRuntimePublisherIdentity identity_;
  std::uint64_t next_publication_sequence_{1U};
  std::uint64_t last_published_batch_sequence_{0U};
  bool exhausted_{false};
};

// Read-only capability carrying one immutable time-transfer bundle and the
// exact 19D active-state/covariance snapshot used at its publication boundary.
class ReferenceOnlyMigrationRuntimeBundle
{
public:
  ReferenceOnlyMigrationRuntimeBundle(
    const ReferenceOnlyMigrationRuntimeBundle &) = default;
  ReferenceOnlyMigrationRuntimeBundle(
    ReferenceOnlyMigrationRuntimeBundle &&) = default;
  ReferenceOnlyMigrationRuntimeBundle & operator=(
    const ReferenceOnlyMigrationRuntimeBundle &) = default;
  ReferenceOnlyMigrationRuntimeBundle & operator=(
    ReferenceOnlyMigrationRuntimeBundle &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const ReferenceOnlyMigrationRuntimePublisherIdentity &
      identity) const noexcept;
  const ValidatedBatchRuntimeComposition &
  validatedComposition() const;

  std::uint64_t publicationSequence() const noexcept;
  std::uint64_t activeStateSnapshotRevision() const noexcept;
  double activeStateTimestampSec() const noexcept;
  ActiveStateSnapshotStatus
  activeStateSnapshotStatus() const noexcept;
  const StatesGroup & activeStateSnapshot() const;
  const CameraBatchTimeTransferBundle &
  timeTransferBundle() const;
  const CameraObservationBatchReceipt &
  cameraBatchReceipt() const;
  std::optional<ReferenceOnlyMigrationFrameProvenance>
  frameProvenance(std::uint8_t camera_id) const;

private:
  struct Publication;

  explicit ReferenceOnlyMigrationRuntimeBundle(
    std::shared_ptr<const Publication> publication);

  std::shared_ptr<const Publication> publication_;

  friend class ValidatedBatchRuntimeComposition;
  friend class ReferenceOnlyMigrationRuntimePublisher;
  friend class ReferenceOnlyMigrationFrameProvenance;
};

}  // namespace livo

#endif  // LIVO_REFERENCE_ONLY_MIGRATION_RUNTIME_PROVENANCE_H_
