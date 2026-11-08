#include "reference_only_migration_runtime_provenance.h"

#include "checked_increment.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace livo
{
namespace
{

struct ReferenceOnlyMigrationRuntimeAuthorityTag final
{
};

struct ValidatedBatchRuntimeCompositionTag final
{
};

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

bool nominalStateValid(const StatesGroup & state) noexcept
{
  if (
    !state.rot_end.allFinite() ||
    !state.pos_end.allFinite() ||
    !state.vel_end.allFinite() ||
    !state.bias_g.allFinite() ||
    !state.bias_a.allFinite() ||
    !state.gravity.allFinite() ||
    !std::isfinite(state.inv_expo_time))
  {
    return false;
  }
  constexpr double kRotationTolerance = 1.0e-8;
  return
    (state.rot_end.transpose() * state.rot_end).
      isApprox(Eigen::Matrix3d::Identity(), kRotationTolerance) &&
    std::abs(state.rot_end.determinant() - 1.0) <=
      kRotationTolerance;
}

bool covarianceValid(const StatesGroup & state) noexcept
{
  if (!state.cov.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, state.cov.cwiseAbs().maxCoeff());
  if (
    !((state.cov - state.cov.transpose()).cwiseAbs().
      maxCoeff() <= 1.0e-10 * scale))
  {
    return false;
  }
  Eigen::SelfAdjointEigenSolver<
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>>
    solver(
    0.5 * (state.cov + state.cov.transpose()),
    Eigen::EigenvaluesOnly);
  return
    solver.info() == Eigen::Success &&
    solver.eigenvalues().allFinite() &&
    solver.eigenvalues().minCoeff() >=
      -1.0e-12 * scale;
}

ActiveStateSnapshotStatus snapshotStatus(
  const StatesGroup & state) noexcept
{
  if (!nominalStateValid(state)) {
    return ActiveStateSnapshotStatus::kInvalidNominalState;
  }
  if (!covarianceValid(state)) {
    return ActiveStateSnapshotStatus::kInvalidCovariance;
  }
  return ActiveStateSnapshotStatus::kValid;
}

CameraTimeTransferState transferAnchor(
  const StatesGroup & state,
  const double anchor_timestamp_sec)
{
  CameraTimeTransferState anchor;
  anchor.timestamp_sec = anchor_timestamp_sec;
  anchor.rotation_world_from_imu = state.rot_end;
  anchor.position_world_m = state.pos_end;
  anchor.velocity_world_mps = state.vel_end;
  anchor.gyro_bias_radps = state.bias_g;
  anchor.accel_bias_mps2 = state.bias_a;
  anchor.gravity_world_mps2 = state.gravity;
  return anchor;
}

}  // namespace

struct ReferenceOnlyMigrationRuntimeBundle::Publication
{
  std::shared_ptr<const void> publication_identity;
  std::shared_ptr<const void> authority;
  std::uint64_t publication_sequence{0U};
  std::uint64_t active_state_snapshot_revision{0U};
  double active_state_timestamp_sec{0.0};
  ActiveStateSnapshotStatus active_state_status{
    ActiveStateSnapshotStatus::kInvalidNominalState};
  StatesGroup active_state_snapshot;
  CameraObservationBatchReceipt camera_batch_receipt;
  CameraBatchTimeTransferBundle time_transfer_bundle;
  ValidatedBatchRuntimeComposition validated_composition;

  Publication(
    std::shared_ptr<const void> input_publication_identity,
    std::shared_ptr<const void> input_authority,
    const std::uint64_t input_publication_sequence,
    const double input_active_state_timestamp_sec,
    const ActiveStateSnapshotStatus input_active_state_status,
    const StatesGroup & input_active_state_snapshot,
    const CameraObservationBatchReceipt &
      input_camera_batch_receipt,
    CameraBatchTimeTransferBundle input_time_transfer_bundle,
    ValidatedBatchRuntimeComposition
      input_validated_composition)
  : publication_identity(
      std::move(input_publication_identity)),
    authority(std::move(input_authority)),
    publication_sequence(input_publication_sequence),
    active_state_snapshot_revision(input_publication_sequence),
    active_state_timestamp_sec(input_active_state_timestamp_sec),
    active_state_status(input_active_state_status),
    active_state_snapshot(input_active_state_snapshot),
    camera_batch_receipt(input_camera_batch_receipt),
    time_transfer_bundle(std::move(input_time_transfer_bundle)),
    validated_composition(
      std::move(input_validated_composition))
  {
  }
};

struct ReferenceOnlyMigrationFrameProvenance::Binding
{
  // Shares the owner-minted publication identity. Binding is projection
  // payload only and is never treated as an independent authority.
  std::shared_ptr<const void> publication_owner;
  std::shared_ptr<const void> publisher_authority;
  CameraObservationBatchReceipt camera_batch_receipt;
  CameraBatchTimeTransferReceipt time_transfer_receipt;
  CameraBatchTimeTransferEntry time_transfer_entry;
  std::uint64_t camera_batch_sequence{0U};
  std::uint64_t runtime_publication_sequence{0U};
  std::uint64_t active_state_snapshot_revision{0U};
  std::uint8_t camera_id{kInvalidBoundCameraId};
  std::uint64_t observation_generation{0U};
  std::uint64_t image_generation{0U};
  double exposure_timestamp_sec{
    std::numeric_limits<double>::quiet_NaN()};

  Binding(
    std::shared_ptr<const void> input_publication_owner,
    std::shared_ptr<const void> input_publisher_authority,
    CameraObservationBatchReceipt input_camera_batch_receipt,
    CameraBatchTimeTransferReceipt input_time_transfer_receipt,
    CameraBatchTimeTransferEntry input_time_transfer_entry,
    const std::uint64_t input_runtime_publication_sequence,
    const std::uint64_t input_active_state_snapshot_revision,
    const std::uint8_t input_camera_id)
  : publication_owner(std::move(input_publication_owner)),
    publisher_authority(std::move(input_publisher_authority)),
    camera_batch_receipt(
      std::move(input_camera_batch_receipt)),
    time_transfer_receipt(
      std::move(input_time_transfer_receipt)),
    time_transfer_entry(std::move(input_time_transfer_entry)),
    camera_batch_sequence(
      camera_batch_receipt.batchSequence()),
    runtime_publication_sequence(
      input_runtime_publication_sequence),
    active_state_snapshot_revision(
      input_active_state_snapshot_revision),
    camera_id(input_camera_id),
    observation_generation(
      camera_batch_receipt.observationGeneration(camera_id)),
    image_generation(
      camera_batch_receipt.imageGeneration(camera_id)),
    exposure_timestamp_sec(
      time_transfer_entry.targetTimestampSec())
  {
  }
};

struct ValidatedBatchRuntimeComposition::Binding
{
  struct ObservationBinding
  {
    bool present{false};
    std::uint8_t camera_id{kInvalidBoundCameraId};
    std::uint64_t observation_generation{0U};
    std::uint64_t image_generation{0U};
    double exposure_timestamp_sec{
      std::numeric_limits<double>::quiet_NaN()};
    int image_rows{0};
    int image_columns{0};
    int image_type{0};
    PhotometricStatus photometric_status{
      PhotometricStatus::kInvalidInput};
  };

  std::shared_ptr<const void> publication_owner;
  std::shared_ptr<const void> publisher_authority;
  CameraObservationBatchReceipt camera_batch_receipt;
  std::uint64_t runtime_publication_sequence{0U};
  std::uint64_t active_state_snapshot_revision{0U};
  std::array<ObservationBinding, kVioCameraCount> observations;

  Binding(
    std::shared_ptr<const void> input_publication_owner,
    std::shared_ptr<const void> input_publisher_authority,
    const CameraObservationBatchReceipt &
      input_camera_batch_receipt,
    const std::uint64_t input_runtime_publication_sequence,
    const std::uint64_t input_active_state_snapshot_revision)
  : publication_owner(std::move(input_publication_owner)),
    publisher_authority(std::move(input_publisher_authority)),
    camera_batch_receipt(input_camera_batch_receipt),
    runtime_publication_sequence(
      input_runtime_publication_sequence),
    active_state_snapshot_revision(
      input_active_state_snapshot_revision)
  {
  }
};

const char * activeStateSnapshotStatusName(
  const ActiveStateSnapshotStatus status) noexcept
{
  switch (status)
  {
    case ActiveStateSnapshotStatus::kValid:
      return "valid";
    case ActiveStateSnapshotStatus::kInvalidNominalState:
      return "invalid_nominal_state";
    case ActiveStateSnapshotStatus::kInvalidCovariance:
      return "invalid_covariance";
  }
  return "unknown";
}

ValidatedBatchRuntimeComposition::
ValidatedBatchRuntimeComposition(
  std::shared_ptr<const Binding> binding)
: binding_(std::move(binding))
{
}

bool ValidatedBatchRuntimeComposition::valid() const noexcept
{
  if (
    !binding_ ||
    !binding_->publication_owner ||
    !binding_->publisher_authority ||
    !binding_->camera_batch_receipt.valid() ||
    binding_->runtime_publication_sequence == 0U ||
    binding_->active_state_snapshot_revision == 0U)
  {
    return false;
  }

  for (std::size_t camera_id = 0U;
       camera_id < binding_->observations.size();
       ++camera_id)
  {
    const Binding::ObservationBinding & observation =
      binding_->observations[camera_id];
    if (!observation.present) {
      continue;
    }
    if (
      observation.camera_id != camera_id ||
      observation.observation_generation == 0U ||
      observation.image_generation == 0U ||
      !std::isfinite(observation.exposure_timestamp_sec) ||
      observation.image_rows <= 0 ||
      observation.image_columns <= 0 ||
      binding_->camera_batch_receipt.observationGeneration(
        static_cast<std::uint8_t>(camera_id)) !=
        observation.observation_generation ||
      binding_->camera_batch_receipt.imageGeneration(
        static_cast<std::uint8_t>(camera_id)) !=
        observation.image_generation)
    {
      return false;
    }
  }
  return true;
}

bool ValidatedBatchRuntimeComposition::issuedBy(
  const CameraObservationBatchAuthorityIdentity &
    batch_identity,
  const ReferenceOnlyMigrationRuntimePublisherIdentity &
    runtime_identity) const noexcept
{
  return
    valid() &&
    batch_identity.valid() &&
    runtime_identity.valid() &&
    runtime_identity.bindsBatchAuthority(batch_identity) &&
    binding_->camera_batch_receipt.issuedBy(batch_identity) &&
    sameSharedOwner(
      binding_->publisher_authority,
      runtime_identity.authority_);
}

bool ValidatedBatchRuntimeComposition::issuedByBatch(
  const CameraObservationBatchAuthorityIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    binding_->camera_batch_receipt.issuedBy(identity);
}

bool ValidatedBatchRuntimeComposition::issuedByRuntime(
  const ReferenceOnlyMigrationRuntimePublisherIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(
      binding_->publisher_authority,
      identity.authority_) &&
    binding_->camera_batch_receipt.issuedBy(
      identity.batch_authority_);
}

bool ValidatedBatchRuntimeComposition::accepts(
  const ReferenceOnlyMigrationRuntimeBundle &
    runtime) const noexcept
{
  return
    valid() &&
    runtime.valid() &&
    sameSharedOwner(
      binding_->publication_owner,
      runtime.publication_->publication_identity) &&
    binding_->runtime_publication_sequence ==
      runtime.publication_->publication_sequence &&
    binding_->active_state_snapshot_revision ==
      runtime.publication_->active_state_snapshot_revision &&
    binding_->camera_batch_receipt.batchSequence() ==
      runtime.publication_->camera_batch_receipt.batchSequence();
}

bool ValidatedBatchRuntimeComposition::accepts(
  const ReferenceOnlyMigrationFrameProvenance &
    frame) const noexcept
{
  if (
    !valid() ||
    !frame.valid() ||
    !sameSharedOwner(
      binding_->publication_owner,
      frame.binding_->publication_owner) ||
    frame.runtimePublicationSequence() !=
      binding_->runtime_publication_sequence ||
    frame.activeStateSnapshotRevision() !=
      binding_->active_state_snapshot_revision ||
    frame.cameraBatchSequence() !=
      binding_->camera_batch_receipt.batchSequence())
  {
    return false;
  }

  const std::uint8_t camera_id = frame.cameraId();
  return
    cameraPresent(camera_id) &&
    frame.observationGeneration() ==
      observationGeneration(camera_id) &&
    frame.imageGeneration() == imageGeneration(camera_id) &&
    frame.exposureTimestampSec() ==
      exposureTimestampSec(camera_id);
}

std::uint64_t
ValidatedBatchRuntimeComposition::cameraBatchSequence()
  const noexcept
{
  return
    valid() ?
    binding_->camera_batch_receipt.batchSequence() : 0U;
}

std::uint64_t
ValidatedBatchRuntimeComposition::runtimePublicationSequence()
  const noexcept
{
  return
    valid() ? binding_->runtime_publication_sequence : 0U;
}

std::uint64_t
ValidatedBatchRuntimeComposition::activeStateSnapshotRevision()
  const noexcept
{
  return
    valid() ? binding_->active_state_snapshot_revision : 0U;
}

bool ValidatedBatchRuntimeComposition::cameraPresent(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() &&
    camera_id < binding_->observations.size() &&
    binding_->observations[camera_id].present;
}

std::uint64_t
ValidatedBatchRuntimeComposition::observationGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    binding_->observations[camera_id].
    observation_generation : 0U;
}

std::uint64_t
ValidatedBatchRuntimeComposition::imageGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    binding_->observations[camera_id].image_generation : 0U;
}

double ValidatedBatchRuntimeComposition::exposureTimestampSec(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    binding_->observations[camera_id].
    exposure_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

int ValidatedBatchRuntimeComposition::imageRows(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    binding_->observations[camera_id].image_rows : 0;
}

int ValidatedBatchRuntimeComposition::imageColumns(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    binding_->observations[camera_id].image_columns : 0;
}

int ValidatedBatchRuntimeComposition::imageType(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    binding_->observations[camera_id].image_type : 0;
}

PhotometricStatus
ValidatedBatchRuntimeComposition::photometricStatus(
  const std::uint8_t camera_id) const noexcept
{
  return
    cameraPresent(camera_id) ?
    binding_->observations[camera_id].photometric_status :
    PhotometricStatus::kInvalidInput;
}

ReferenceOnlyMigrationFrameProvenance::
ReferenceOnlyMigrationFrameProvenance(
  std::shared_ptr<const Binding> binding)
: binding_(std::move(binding))
{
}

bool ReferenceOnlyMigrationFrameProvenance::valid()
  const noexcept
{
  return
    binding_ &&
    binding_->publication_owner &&
    binding_->publisher_authority &&
    binding_->camera_batch_receipt.valid() &&
    binding_->time_transfer_receipt.valid() &&
    binding_->camera_batch_sequence != 0U &&
    binding_->camera_batch_sequence ==
      binding_->camera_batch_receipt.batchSequence() &&
    binding_->runtime_publication_sequence != 0U &&
    binding_->active_state_snapshot_revision != 0U &&
    binding_->camera_id < kVioCameraCount &&
    binding_->observation_generation != 0U &&
    binding_->observation_generation ==
      binding_->camera_batch_receipt.observationGeneration(
      binding_->camera_id) &&
    binding_->image_generation != 0U &&
    binding_->image_generation ==
      binding_->camera_batch_receipt.imageGeneration(
      binding_->camera_id) &&
    std::isfinite(binding_->exposure_timestamp_sec) &&
    binding_->time_transfer_entry.has_value() &&
    binding_->time_transfer_entry.cameraId() ==
      binding_->camera_id &&
    binding_->time_transfer_entry.targetTimestampSec() ==
      binding_->exposure_timestamp_sec &&
    binding_->time_transfer_receipt.accepts(
      binding_->time_transfer_entry);
}

bool ReferenceOnlyMigrationFrameProvenance::issuedBy(
  const ReferenceOnlyMigrationRuntimePublisherIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(
      binding_->publisher_authority,
      identity.authority_) &&
    binding_->camera_batch_receipt.issuedBy(
      identity.batch_authority_);
}

bool ReferenceOnlyMigrationFrameProvenance::accepts(
  const ValidatedBatchRuntimeComposition &
    composition) const noexcept
{
  return composition.accepts(*this);
}

std::uint64_t
ReferenceOnlyMigrationFrameProvenance::
cameraBatchSequence() const noexcept
{
  return valid() ? binding_->camera_batch_sequence : 0U;
}

std::uint64_t
ReferenceOnlyMigrationFrameProvenance::
runtimePublicationSequence() const noexcept
{
  return
    valid() ? binding_->runtime_publication_sequence : 0U;
}

std::uint64_t
ReferenceOnlyMigrationFrameProvenance::
activeStateSnapshotRevision() const noexcept
{
  return
    valid() ? binding_->active_state_snapshot_revision : 0U;
}

std::uint8_t
ReferenceOnlyMigrationFrameProvenance::cameraId()
  const noexcept
{
  return valid() ? binding_->camera_id : kInvalidBoundCameraId;
}

std::uint64_t
ReferenceOnlyMigrationFrameProvenance::
observationGeneration() const noexcept
{
  return valid() ? binding_->observation_generation : 0U;
}

std::uint64_t
ReferenceOnlyMigrationFrameProvenance::
imageGeneration() const noexcept
{
  return valid() ? binding_->image_generation : 0U;
}

double
ReferenceOnlyMigrationFrameProvenance::
exposureTimestampSec() const noexcept
{
  return
    valid() ?
    binding_->exposure_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

const CameraBatchTimeTransferEntry &
ReferenceOnlyMigrationFrameProvenance::
timeTransferEntry() const
{
  if (!valid()) {
    throw std::logic_error(
            "invalid R2b frame provenance");
  }
  return binding_->time_transfer_entry;
}

ReferenceOnlyMigrationRuntimePublisherIdentity::
ReferenceOnlyMigrationRuntimePublisherIdentity(
  std::shared_ptr<const void> authority,
  CameraObservationBatchAuthorityIdentity batch_authority)
: authority_(std::move(authority)),
  batch_authority_(std::move(batch_authority))
{
}

bool ReferenceOnlyMigrationRuntimePublisherIdentity::valid()
  const noexcept
{
  return static_cast<bool>(authority_);
}

bool ReferenceOnlyMigrationRuntimePublisherIdentity::
sameAuthority(
  const ReferenceOnlyMigrationRuntimePublisherIdentity &
    other) const noexcept
{
  return sameSharedOwner(authority_, other.authority_);
}

bool ReferenceOnlyMigrationRuntimePublisherIdentity::
bindsBatchAuthority(
  const CameraObservationBatchAuthorityIdentity &
    batch_authority) const noexcept
{
  return
    valid() &&
    batch_authority_.valid() &&
    batch_authority_.sameAuthority(batch_authority);
}

ReferenceOnlyMigrationRuntimePublisher::
ReferenceOnlyMigrationRuntimePublisher(
  CameraObservationBatchAuthorityIdentity batch_authority)
: batch_authority_(std::move(batch_authority)),
  identity_(
    batch_authority_.valid() ?
    ReferenceOnlyMigrationRuntimePublisherIdentity{
      std::make_shared<
        const ReferenceOnlyMigrationRuntimeAuthorityTag>(),
      batch_authority_} :
    ReferenceOnlyMigrationRuntimePublisherIdentity{})
{
}

bool ReferenceOnlyMigrationRuntimePublisher::valid() const noexcept
{
  return
    batch_authority_.valid() &&
    identity_.valid() &&
    !exhausted_;
}

bool ReferenceOnlyMigrationRuntimePublisher::exhausted()
  const noexcept
{
  return exhausted_;
}

ReferenceOnlyMigrationRuntimePublisherIdentity
ReferenceOnlyMigrationRuntimePublisher::identity() const noexcept
{
  return identity_;
}

std::optional<ReferenceOnlyMigrationRuntimeBundle>
ReferenceOnlyMigrationRuntimePublisher::publish(
  const CameraObservationBatch & batch,
  const StatesGroup & active_state,
  const double anchor_timestamp_sec,
  const double exposure_time_sec,
  const CameraTimeTransfer & time_transfer)
{
  const auto & receipt = batch.validatedReceipt();
  const std::uint64_t batch_sequence =
    receipt.has_value() ? receipt->batchSequence() : 0U;
  if (
    !valid() ||
    !std::isfinite(anchor_timestamp_sec) ||
    !std::isfinite(exposure_time_sec) ||
    !receipt.has_value() ||
    !receipt->valid() ||
    !receipt->issuedBy(batch_authority_) ||
    batch_sequence == 0U ||
    batch_sequence <= last_published_batch_sequence_)
  {
    return std::nullopt;
  }
  last_published_batch_sequence_ = batch_sequence;
  if (!receipt->claimAndValidate(batch)) {
    return std::nullopt;
  }

  std::uint64_t publication_sequence = 0U;
  if (
    !checkedReserve(
      next_publication_sequence_,
      publication_sequence))
  {
    exhausted_ = true;
    return std::nullopt;
  }

  CameraBatchTimeTransferBundle time_bundle =
    buildCameraBatchTimeTransferBundle(
    batch,
    transferAnchor(active_state, anchor_timestamp_sec),
    exposure_time_sec,
    time_transfer);
  const auto publication_identity =
    std::make_shared<
    const ValidatedBatchRuntimeCompositionTag>();
  auto composition_binding =
    std::make_shared<
    ValidatedBatchRuntimeComposition::Binding>(
    publication_identity,
    identity_.authority_,
    *receipt,
    publication_sequence,
    publication_sequence);
  for (std::size_t camera_id = 0U;
       camera_id < composition_binding->observations.size();
       ++camera_id)
  {
    auto & target =
      composition_binding->observations[camera_id];
    target.present = batch.present[camera_id];
    if (!target.present) {
      continue;
    }
    const CameraObservation & source =
      batch.observations[camera_id];
    target.camera_id = source.camera_id;
    target.observation_generation =
      receipt->observationGeneration(
      static_cast<std::uint8_t>(camera_id));
    target.image_generation =
      receipt->imageGeneration(
      static_cast<std::uint8_t>(camera_id));
    target.exposure_timestamp_sec = source.timestamp;
    target.image_rows = source.image.rows;
    target.image_columns = source.image.cols;
    target.image_type = source.image.type();
    target.photometric_status = source.photometric_status;
  }
  ValidatedBatchRuntimeComposition composition{
    std::move(composition_binding)};
  if (!composition.valid()) {
    exhausted_ = true;
    return std::nullopt;
  }
  auto publication =
    std::make_shared<
    ReferenceOnlyMigrationRuntimeBundle::Publication>(
    publication_identity,
    identity_.authority_,
    publication_sequence,
    anchor_timestamp_sec,
    snapshotStatus(active_state),
    active_state,
    *receipt,
    std::move(time_bundle),
    std::move(composition));
  return ReferenceOnlyMigrationRuntimeBundle{
    std::move(publication)};
}

ReferenceOnlyMigrationRuntimeBundle::
ReferenceOnlyMigrationRuntimeBundle(
  std::shared_ptr<const Publication> publication)
: publication_(std::move(publication))
{
}

bool ReferenceOnlyMigrationRuntimeBundle::valid() const noexcept
{
  return
    publication_ &&
    publication_->authority &&
    publication_->publication_identity &&
    publication_->publication_sequence != 0U &&
    publication_->active_state_snapshot_revision != 0U &&
    std::isfinite(publication_->active_state_timestamp_sec) &&
    publication_->camera_batch_receipt.valid() &&
    publication_->validated_composition.valid() &&
    publication_->validated_composition.
      runtimePublicationSequence() ==
      publication_->publication_sequence &&
    publication_->validated_composition.
      activeStateSnapshotRevision() ==
      publication_->active_state_snapshot_revision;
}

bool ReferenceOnlyMigrationRuntimeBundle::issuedBy(
  const ReferenceOnlyMigrationRuntimePublisherIdentity &
    identity) const noexcept
{
  return
    valid() &&
    sameSharedOwner(
      publication_->authority,
      identity.authority_);
}

const ValidatedBatchRuntimeComposition &
ReferenceOnlyMigrationRuntimeBundle::validatedComposition() const
{
  if (!valid()) {
    throw std::logic_error(
            "invalid validated batch/runtime composition");
  }
  return publication_->validated_composition;
}

std::uint64_t
ReferenceOnlyMigrationRuntimeBundle::publicationSequence()
  const noexcept
{
  return valid() ? publication_->publication_sequence : 0U;
}

std::uint64_t
ReferenceOnlyMigrationRuntimeBundle::
activeStateSnapshotRevision() const noexcept
{
  return
    valid() ?
    publication_->active_state_snapshot_revision : 0U;
}

double
ReferenceOnlyMigrationRuntimeBundle::activeStateTimestampSec()
  const noexcept
{
  return
    valid() ?
    publication_->active_state_timestamp_sec :
    std::numeric_limits<double>::quiet_NaN();
}

ActiveStateSnapshotStatus
ReferenceOnlyMigrationRuntimeBundle::
activeStateSnapshotStatus() const noexcept
{
  return
    valid() ?
    publication_->active_state_status :
    ActiveStateSnapshotStatus::kInvalidNominalState;
}

const StatesGroup &
ReferenceOnlyMigrationRuntimeBundle::activeStateSnapshot() const
{
  if (!valid()) {
    throw std::logic_error("invalid R2b runtime publication");
  }
  return publication_->active_state_snapshot;
}

const CameraBatchTimeTransferBundle &
ReferenceOnlyMigrationRuntimeBundle::timeTransferBundle() const
{
  if (!valid()) {
    throw std::logic_error("invalid R2b runtime publication");
  }
  return publication_->time_transfer_bundle;
}

const CameraObservationBatchReceipt &
ReferenceOnlyMigrationRuntimeBundle::cameraBatchReceipt() const
{
  if (!valid()) {
    throw std::logic_error("invalid R2b runtime publication");
  }
  return publication_->camera_batch_receipt;
}

std::optional<ReferenceOnlyMigrationFrameProvenance>
ReferenceOnlyMigrationRuntimeBundle::frameProvenance(
  const std::uint8_t camera_id) const
{
  if (
    !valid() ||
    publication_->active_state_status !=
      ActiveStateSnapshotStatus::kValid ||
    camera_id >= kVioCameraCount ||
    !publication_->time_transfer_bundle.
      allPresentFullStateTransfersValid() ||
    !publication_->time_transfer_bundle.present[camera_id] ||
    !publication_->time_transfer_bundle.
      validatedReceipt().has_value())
  {
    return std::nullopt;
  }

  const std::uint64_t observation_generation =
    publication_->camera_batch_receipt.
    observationGeneration(camera_id);
  const std::uint64_t image_generation =
    publication_->camera_batch_receipt.
    imageGeneration(camera_id);
  const CameraBatchTimeTransferEntry & entry =
    publication_->time_transfer_bundle.transfers[camera_id];
  if (
    observation_generation == 0U ||
    image_generation == 0U ||
    !entry.has_value() ||
    entry.cameraId() != camera_id)
  {
    return std::nullopt;
  }

  auto binding =
    std::make_shared<
    const ReferenceOnlyMigrationFrameProvenance::Binding>(
    publication_->publication_identity,
    publication_->authority,
    publication_->camera_batch_receipt,
    *publication_->time_transfer_bundle.validatedReceipt(),
    entry,
    publication_->publication_sequence,
    publication_->active_state_snapshot_revision,
    camera_id);
  ReferenceOnlyMigrationFrameProvenance provenance{
    std::move(binding)};
  return provenance.valid() ?
         std::optional<
    ReferenceOnlyMigrationFrameProvenance>{
    std::move(provenance)} :
         std::nullopt;
}

}  // namespace livo
