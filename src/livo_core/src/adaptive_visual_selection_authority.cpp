#include "adaptive_visual_selection_authority.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

std::shared_ptr<livo::AdaptiveVisualSelectionAuthority>
CameraObservationQueue::issueAdaptiveVisualSelectionAuthority()
{
  auto authority =
    std::shared_ptr<
    livo::AdaptiveVisualSelectionAuthority>(
    new livo::AdaptiveVisualSelectionAuthority(
      batch_authority_));
  authority->self_ = authority;
  bool expected = false;
  if (
    !adaptive_visual_selection_authority_issued_.
    compare_exchange_strong(
      expected, true, std::memory_order_acq_rel))
  {
    return nullptr;
  }
  return authority;
}

namespace livo
{
namespace
{

#ifdef LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING
std::atomic<bool> fail_next_receipt_allocation{false};
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

}  // namespace

#ifdef LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING
namespace test
{

void failNextSelectionReceiptAllocation()
{
  fail_next_receipt_allocation.store(
    true, std::memory_order_release);
}

}  // namespace test
#endif

AdaptiveVisualSelectionAuthorityIdentity::
AdaptiveVisualSelectionAuthorityIdentity(
  std::shared_ptr<const std::uint8_t> owner)
: owner_(std::move(owner))
{
}

bool AdaptiveVisualSelectionAuthorityIdentity::valid() const noexcept
{
  return static_cast<bool>(owner_);
}

bool AdaptiveVisualSelectionAuthorityIdentity::sameAuthority(
  const AdaptiveVisualSelectionAuthorityIdentity & other)
  const noexcept
{
  return sameSharedOwner(owner_, other.owner_);
}

struct AuthoritativeLidarGenerationToken::Data
{
  Data(
    std::shared_ptr<const std::uint8_t> input_owner,
    const std::uint64_t input_generation,
    const AdaptiveVisualPolicy::Matrix6 &
      input_lidar_pose_information,
    const bool input_lidar_information_ready)
  : owner(std::move(input_owner)),
    generation(input_generation),
    lidar_pose_information(input_lidar_pose_information),
    lidar_information_ready(input_lidar_information_ready)
  {
  }

  const std::shared_ptr<const std::uint8_t> owner;
  const std::uint64_t generation;
  const AdaptiveVisualPolicy::Matrix6 lidar_pose_information;
  const bool lidar_information_ready;
  std::atomic<bool> runtime_attempted{false};
  std::atomic<std::uint64_t> runtime_attempt_nonce{0U};
  std::atomic<bool> consumed{false};
};

namespace
{

class BoundCandidateEvidence final
{
public:
  using CandidateMask =
    std::array<bool, kAdaptiveVisualCameraCount>;
  using GenerationArray =
    std::array<std::uint64_t, kAdaptiveVisualCameraCount>;
  using ExposureArray =
    std::array<double, kAdaptiveVisualCameraCount>;

  BoundCandidateEvidence(
    const BoundCandidateEvidence &) = default;
  BoundCandidateEvidence(
    BoundCandidateEvidence &&) = default;

  static std::optional<BoundCandidateEvidence> bind(
    AdaptiveVisualCandidatePlanCapability input_policy_capability)
  {
    if (!input_policy_capability.valid())
    {
      return std::nullopt;
    }
    BoundCandidateEvidence evidence{
      std::move(input_policy_capability)};
    if (!evidence.valid())
    {
      return std::nullopt;
    }
    return evidence;
  }

  bool valid() const noexcept
  {
    if (!policy_capability_.valid())
    {
      return false;
    }
    const auto & camera_batch_receipt =
      policy_capability_.cameraBatchReceipt();
    const auto & plan =
      policy_capability_.decisionPlan();
    if (
      !camera_batch_receipt.valid() ||
      owner_sequence_ == 0U ||
      owner_sequence_ !=
        camera_batch_receipt.batchSequence() ||
      owner_sequence_ !=
        policy_capability_.cameraBatchSequence() ||
      policy_revision_ != plan.base_revision ||
      policy_revision_ !=
        policy_capability_.basePolicyRevision() ||
      policy_capability_.selectionReason() ==
        AdaptiveSelectionReason::kNotApplicable ||
      selected_camera_id_ >= kAdaptiveVisualCameraCount ||
      policy_capability_.cameraCount() == 0U ||
      policy_capability_.cameraCount() >
        kAdaptiveVisualCameraCount ||
      candidate_mask_ !=
        policy_capability_.candidateMask() ||
      !candidate_mask_[selected_camera_id_] ||
      !selected_mask_[selected_camera_id_] ||
      std::count(
        candidate_mask_.begin(), candidate_mask_.end(),
        true) != static_cast<std::ptrdiff_t>(
        policy_capability_.cameraCount()) ||
      std::count(
        selected_mask_.begin(), selected_mask_.end(),
        true) != 1 ||
      candidate_budget_ == 0U ||
      patch_budget_ == 0U)
    {
      return false;
    }

    std::size_t selected_count = 0U;
    std::size_t conserved_candidate_budget = 0U;
    std::size_t conserved_patch_budget = 0U;
    for (std::size_t camera_id = 0U;
         camera_id < kAdaptiveVisualCameraCount;
         ++camera_id)
    {
      const auto candidate_id =
        static_cast<std::uint8_t>(camera_id);
      const std::uint64_t sealed_observation_generation =
        camera_batch_receipt.observationGeneration(
        candidate_id);
      const std::uint64_t sealed_image_generation =
        camera_batch_receipt.imageGeneration(
        candidate_id);
      const double sealed_exposure_timestamp_sec =
        camera_batch_receipt.cameraExposureTimestampSec(
        candidate_id);
      if (
        observation_generations_[camera_id] !=
          sealed_observation_generation ||
        image_generations_[camera_id] !=
          sealed_image_generation ||
        !sameExposureTimestamp(
          exposure_timestamps_sec_[camera_id],
          sealed_exposure_timestamp_sec))
      {
        return false;
      }

      const bool sealed_candidate =
        sealed_observation_generation != 0U &&
        sealed_image_generation != 0U &&
        std::isfinite(sealed_exposure_timestamp_sec);
      if (
        candidate_mask_[camera_id] !=
          sealed_candidate ||
        (!sealed_candidate &&
        (sealed_observation_generation != 0U ||
        sealed_image_generation != 0U ||
        !std::isnan(sealed_exposure_timestamp_sec))))
      {
        return false;
      }

      if (plan.cameras[camera_id].enabled)
      {
        ++selected_count;
        if (
          camera_id != selected_camera_id_ ||
          !selected_mask_[camera_id])
        {
          return false;
        }
      }
      else if (selected_mask_[camera_id])
      {
        return false;
      }

      const auto & policy_observation =
        policy_capability_.observations()[camera_id];
      if (candidate_mask_[camera_id])
      {
        if (
          policy_observation.camera_id != candidate_id ||
          !policy_observation.frame_received ||
          policy_observation.observation_generation !=
            sealed_observation_generation ||
          policy_observation.image_generation !=
            sealed_image_generation)
        {
          return false;
        }
      }

      const auto & decision = plan.cameras[camera_id];
      if (camera_id == selected_camera_id_)
      {
        if (
          decision.candidate_budget !=
            candidate_budget_ ||
          decision.patch_budget != patch_budget_)
        {
          return false;
        }
      }
      else if (
        decision.candidate_budget != 0U ||
        decision.patch_budget != 0U)
      {
        return false;
      }
      if (
        decision.candidate_budget >
          std::numeric_limits<std::size_t>::max() -
          conserved_candidate_budget ||
        decision.patch_budget >
          std::numeric_limits<std::size_t>::max() -
          conserved_patch_budget)
      {
        return false;
      }
      conserved_candidate_budget +=
        decision.candidate_budget;
      conserved_patch_budget += decision.patch_budget;
    }
    return
      selected_count == 1U &&
      conserved_candidate_budget == candidate_budget_ &&
      conserved_patch_budget == patch_budget_;
  }

  const AdaptiveVisualCandidatePlanCapability &
  policyCapability() const noexcept
  {
    return policy_capability_;
  }

  const CameraObservationBatchReceipt &
  cameraBatchReceipt() const
  {
    return policy_capability_.cameraBatchReceipt();
  }

  const CandidateMask & candidateMask() const noexcept
  {
    return candidate_mask_;
  }

  const CandidateMask & selectedMask() const noexcept
  {
    return selected_mask_;
  }

  std::uint64_t observationGeneration(
    const std::size_t camera_id) const noexcept
  {
    return observation_generations_[camera_id];
  }

  std::uint64_t imageGeneration(
    const std::size_t camera_id) const noexcept
  {
    return image_generations_[camera_id];
  }

  double exposureTimestampSec(
    const std::size_t camera_id) const noexcept
  {
    return exposure_timestamps_sec_[camera_id];
  }

  std::uint64_t ownerSequence() const noexcept
  {
    return owner_sequence_;
  }

  std::uint64_t policyRevision() const noexcept
  {
    return policy_revision_;
  }

  std::uint8_t selectedCameraId() const noexcept
  {
    return selected_camera_id_;
  }

  std::size_t candidateBudget() const noexcept
  {
    return candidate_budget_;
  }

  std::size_t patchBudget() const noexcept
  {
    return patch_budget_;
  }

private:
  static bool sameExposureTimestamp(
    const double lhs, const double rhs) noexcept
  {
    return
      (std::isfinite(lhs) &&
      std::isfinite(rhs) &&
      lhs == rhs) ||
      (std::isnan(lhs) && std::isnan(rhs));
  }

  static CandidateMask deriveSelectedMask(
    const AdaptiveVisualPolicyPlan & plan) noexcept
  {
    CandidateMask selected_mask{{
        false, false, false, false}};
    for (std::size_t camera_id = 0U;
         camera_id < kAdaptiveVisualCameraCount;
         ++camera_id)
    {
      selected_mask[camera_id] =
        plan.cameras[camera_id].enabled;
    }
    return selected_mask;
  }

  static GenerationArray deriveObservationGenerations(
    const AdaptiveVisualCandidatePlanCapability &
      policy_capability) noexcept
  {
    GenerationArray generations{{0U, 0U, 0U, 0U}};
    for (std::size_t camera_id = 0U;
         camera_id < kAdaptiveVisualCameraCount;
         ++camera_id)
    {
      generations[camera_id] =
        policy_capability.cameraBatchReceipt().
        observationGeneration(
        static_cast<std::uint8_t>(camera_id));
    }
    return generations;
  }

  static GenerationArray deriveImageGenerations(
    const AdaptiveVisualCandidatePlanCapability &
      policy_capability) noexcept
  {
    GenerationArray generations{{0U, 0U, 0U, 0U}};
    for (std::size_t camera_id = 0U;
         camera_id < kAdaptiveVisualCameraCount;
         ++camera_id)
    {
      generations[camera_id] =
        policy_capability.cameraBatchReceipt().
        imageGeneration(
        static_cast<std::uint8_t>(camera_id));
    }
    return generations;
  }

  static ExposureArray deriveExposureTimestamps(
    const AdaptiveVisualCandidatePlanCapability &
      policy_capability) noexcept
  {
    ExposureArray timestamps;
    timestamps.fill(
      std::numeric_limits<double>::quiet_NaN());
    for (std::size_t camera_id = 0U;
         camera_id < kAdaptiveVisualCameraCount;
         ++camera_id)
    {
      timestamps[camera_id] =
        policy_capability.cameraBatchReceipt().
        cameraExposureTimestampSec(
        static_cast<std::uint8_t>(camera_id));
    }
    return timestamps;
  }

  static std::uint8_t deriveSelectedCameraId(
    const CandidateMask & selected_mask) noexcept
  {
    for (std::size_t camera_id = 0U;
         camera_id < kAdaptiveVisualCameraCount;
         ++camera_id)
    {
      if (selected_mask[camera_id])
      {
        return static_cast<std::uint8_t>(camera_id);
      }
    }
    return static_cast<std::uint8_t>(
      kAdaptiveVisualCameraCount);
  }

  static std::size_t deriveBudget(
    const AdaptiveVisualCandidatePlanCapability &
      policy_capability,
    const std::uint8_t selected_camera_id,
    const bool candidate_budget) noexcept
  {
    if (selected_camera_id >= kAdaptiveVisualCameraCount)
    {
      return 0U;
    }
    const auto & decision =
      policy_capability.decisionPlan().
      cameras[selected_camera_id];
    return
      candidate_budget ?
      decision.candidate_budget :
      decision.patch_budget;
  }

  explicit BoundCandidateEvidence(
    AdaptiveVisualCandidatePlanCapability input_policy_capability)
  : policy_capability_(
      std::move(input_policy_capability)),
    candidate_mask_(policy_capability_.candidateMask()),
    selected_mask_(
      deriveSelectedMask(
        policy_capability_.decisionPlan())),
    observation_generations_(
      deriveObservationGenerations(policy_capability_)),
    image_generations_(
      deriveImageGenerations(policy_capability_)),
    exposure_timestamps_sec_(
      deriveExposureTimestamps(policy_capability_)),
    owner_sequence_(
      policy_capability_.cameraBatchReceipt().
      batchSequence()),
    policy_revision_(
      policy_capability_.basePolicyRevision()),
    selected_camera_id_(
      deriveSelectedCameraId(selected_mask_)),
    candidate_budget_(
      deriveBudget(
        policy_capability_, selected_camera_id_, true)),
    patch_budget_(
      deriveBudget(
        policy_capability_, selected_camera_id_, false))
  {
  }

  AdaptiveVisualCandidatePlanCapability
    policy_capability_;
  CandidateMask candidate_mask_;
  CandidateMask selected_mask_;
  GenerationArray observation_generations_;
  GenerationArray image_generations_;
  ExposureArray exposure_timestamps_sec_;
  std::uint64_t owner_sequence_;
  std::uint64_t policy_revision_;
  std::uint8_t selected_camera_id_;
  std::size_t candidate_budget_;
  std::size_t patch_budget_;
};

}  // namespace

struct AdaptiveVisualSelectionCapability::Data
{
  Data(
    std::shared_ptr<const std::uint8_t> input_owner,
    AuthoritativeLidarGenerationToken input_lidar_token,
    BoundCandidateEvidence input_evidence)
  : owner(std::move(input_owner)),
    lidar_token(std::move(input_lidar_token)),
    evidence(std::move(input_evidence))
  {
  }

  const std::shared_ptr<const std::uint8_t> owner;
  const AuthoritativeLidarGenerationToken lidar_token;
  const BoundCandidateEvidence evidence;
};

struct AdaptiveVisualSelectionReceipt::Data
{
  Data(
    std::shared_ptr<const std::uint8_t> input_owner,
    AuthoritativeLidarGenerationToken input_lidar_token,
    std::shared_ptr<const BoundCandidateEvidence>
      input_evidence,
    const AdaptiveVisualSelectionStatus
      input_selection_status)
  : owner(std::move(input_owner)),
    lidar_token(std::move(input_lidar_token)),
    evidence(std::move(input_evidence)),
    selection_status(input_selection_status)
  {
  }

  const std::shared_ptr<const std::uint8_t> owner;
  const AuthoritativeLidarGenerationToken lidar_token;
  const std::shared_ptr<const BoundCandidateEvidence>
    evidence;
  const AdaptiveVisualSelectionStatus selection_status;
};

AuthoritativeLidarGenerationToken::
AuthoritativeLidarGenerationToken(
  std::shared_ptr<Data> data)
: data_(std::move(data))
{
}

bool AuthoritativeLidarGenerationToken::valid() const noexcept
{
  return
    data_ &&
    data_->owner &&
    data_->generation != 0U &&
    data_->lidar_pose_information.allFinite();
}

bool AuthoritativeLidarGenerationToken::issuedBy(
  const AdaptiveVisualSelectionAuthorityIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(data_->owner, identity.owner_);
}

std::uint64_t
AuthoritativeLidarGenerationToken::generation() const noexcept
{
  return valid() ? data_->generation : 0U;
}

bool
AuthoritativeLidarGenerationToken::lidarInformationReady() const noexcept
{
  return valid() && data_->lidar_information_ready;
}

const AdaptiveVisualPolicy::Matrix6 &
AuthoritativeLidarGenerationToken::lidarPoseInformation() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid authoritative lidar generation token");
  }
  return data_->lidar_pose_information;
}

bool AuthoritativeLidarGenerationToken::matchesLidarInformation(
  const AdaptiveVisualPolicy::Matrix6 &
    lidar_pose_information) const noexcept
{
  return
    valid() &&
    lidar_pose_information.allFinite() &&
    (data_->lidar_pose_information.array() ==
     lidar_pose_information.array()).all();
}

AdaptiveVisualSelectionCapability::
AdaptiveVisualSelectionCapability(
  std::shared_ptr<const Data> data)
: data_(std::move(data))
{
}

bool AdaptiveVisualSelectionCapability::valid() const noexcept
{
  return
    data_ &&
    data_->owner &&
    data_->lidar_token.valid() &&
    data_->lidar_token.issuedBy(
      AdaptiveVisualSelectionAuthorityIdentity{
        data_->owner}) &&
    data_->evidence.valid();
}

bool AdaptiveVisualSelectionCapability::issuedBy(
  const AdaptiveVisualSelectionAuthorityIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(data_->owner, identity.owner_);
}

std::uint64_t
AdaptiveVisualSelectionCapability::lidarGeneration() const noexcept
{
  return valid() ? data_->lidar_token.generation() : 0U;
}

std::uint64_t
AdaptiveVisualSelectionCapability::cameraBatchSequence() const noexcept
{
  return valid() ? data_->evidence.ownerSequence() : 0U;
}

std::uint64_t
AdaptiveVisualSelectionCapability::policyRevision() const noexcept
{
  return valid() ? data_->evidence.policyRevision() : 0U;
}

AdaptiveSelectionReason
AdaptiveVisualSelectionCapability::selectionReason() const noexcept
{
  return
    valid() ?
    data_->evidence.policyCapability().selectionReason() :
    AdaptiveSelectionReason::kNotApplicable;
}

bool
AdaptiveVisualSelectionCapability::lidarInformationReady() const noexcept
{
  return
    valid() &&
    data_->lidar_token.lidarInformationReady();
}

const AdaptiveVisualPolicy::Matrix6 &
AdaptiveVisualSelectionCapability::lidarPoseInformation() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual selection capability");
  }
  return data_->lidar_token.lidarPoseInformation();
}

std::uint8_t
AdaptiveVisualSelectionCapability::selectedCameraId() const noexcept
{
  return valid() ? data_->evidence.selectedCameraId() :
         static_cast<std::uint8_t>(kAdaptiveVisualCameraCount);
}

const std::array<bool, kAdaptiveVisualCameraCount> &
AdaptiveVisualSelectionCapability::candidateMask() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual selection capability");
  }
  return data_->evidence.candidateMask();
}

const std::array<bool, kAdaptiveVisualCameraCount> &
AdaptiveVisualSelectionCapability::selectedMask() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual selection capability");
  }
  return data_->evidence.selectedMask();
}

std::uint64_t
AdaptiveVisualSelectionCapability::
candidateObservationGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() && camera_id < kAdaptiveVisualCameraCount ?
    data_->evidence.observationGeneration(camera_id) : 0U;
}

std::uint64_t
AdaptiveVisualSelectionCapability::candidateImageGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() && camera_id < kAdaptiveVisualCameraCount ?
    data_->evidence.imageGeneration(camera_id) :
    0U;
}

double
AdaptiveVisualSelectionCapability::
candidateExposureTimestampSec(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() && camera_id < kAdaptiveVisualCameraCount ?
    data_->evidence.exposureTimestampSec(camera_id) :
    std::numeric_limits<double>::quiet_NaN();
}

std::size_t
AdaptiveVisualSelectionCapability::candidateBudget() const noexcept
{
  return valid() ? data_->evidence.candidateBudget() : 0U;
}

std::size_t
AdaptiveVisualSelectionCapability::patchBudget() const noexcept
{
  return valid() ? data_->evidence.patchBudget() : 0U;
}

AdaptiveVisualSelectionReceipt::AdaptiveVisualSelectionReceipt(
  std::shared_ptr<const Data> data)
: data_(std::move(data))
{
}

bool AdaptiveVisualSelectionReceipt::valid() const noexcept
{
  return
    data_ &&
    data_->owner &&
    data_->lidar_token.valid() &&
    sameSharedOwner(
      data_->owner, data_->lidar_token.data_->owner) &&
    data_->evidence &&
    data_->evidence->valid() &&
    data_->evidence->cameraBatchReceipt().
      adaptiveVisualSelectionClaimed() &&
    data_->selection_status ==
      AdaptiveVisualSelectionStatus::kReady;
}

bool AdaptiveVisualSelectionReceipt::issuedBy(
  const AdaptiveVisualSelectionAuthorityIdentity &
    identity) const noexcept
{
  return
    valid() &&
    identity.valid() &&
    sameSharedOwner(data_->owner, identity.owner_);
}

std::uint64_t
AdaptiveVisualSelectionReceipt::lidarGeneration() const noexcept
{
  return valid() ? data_->lidar_token.generation() : 0U;
}

std::uint64_t
AdaptiveVisualSelectionReceipt::cameraBatchSequence() const noexcept
{
  return valid() ? data_->evidence->ownerSequence() : 0U;
}

std::uint64_t
AdaptiveVisualSelectionReceipt::policyRevision() const noexcept
{
  return valid() ? data_->evidence->policyRevision() : 0U;
}

AdaptiveSelectionReason
AdaptiveVisualSelectionReceipt::selectionReason() const noexcept
{
  return
    valid() ?
    data_->evidence->policyCapability().selectionReason() :
    AdaptiveSelectionReason::kNotApplicable;
}

bool
AdaptiveVisualSelectionReceipt::lidarInformationReady() const noexcept
{
  return
    valid() &&
    data_->lidar_token.lidarInformationReady();
}

const AdaptiveVisualPolicy::Matrix6 &
AdaptiveVisualSelectionReceipt::lidarPoseInformation() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual selection receipt");
  }
  return data_->lidar_token.lidarPoseInformation();
}

std::uint8_t
AdaptiveVisualSelectionReceipt::selectedCameraId() const noexcept
{
  return valid() ? data_->evidence->selectedCameraId() :
         static_cast<std::uint8_t>(kAdaptiveVisualCameraCount);
}

std::size_t
AdaptiveVisualSelectionReceipt::candidateBudget() const noexcept
{
  return valid() ? data_->evidence->candidateBudget() : 0U;
}

std::size_t
AdaptiveVisualSelectionReceipt::patchBudget() const noexcept
{
  return valid() ? data_->evidence->patchBudget() : 0U;
}

AdaptiveVisualSelectionStatus
AdaptiveVisualSelectionReceipt::selectionStatus() const noexcept
{
  return
    valid() ?
    data_->selection_status :
    AdaptiveVisualSelectionStatus::
    kInvalidSelectionCapability;
}

const std::array<bool, kAdaptiveVisualCameraCount> &
AdaptiveVisualSelectionReceipt::candidateMask() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual selection receipt");
  }
  return data_->evidence->candidateMask();
}

const std::array<bool, kAdaptiveVisualCameraCount> &
AdaptiveVisualSelectionReceipt::selectedMask() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual selection receipt");
  }
  return data_->evidence->selectedMask();
}

bool AdaptiveVisualSelectionReceipt::cameraPresent(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() &&
    camera_id < kAdaptiveVisualCameraCount &&
    data_->evidence->candidateMask()[camera_id];
}

std::uint64_t
AdaptiveVisualSelectionReceipt::
candidateObservationGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() && camera_id < kAdaptiveVisualCameraCount ?
    data_->evidence->observationGeneration(camera_id) : 0U;
}

std::uint64_t
AdaptiveVisualSelectionReceipt::candidateImageGeneration(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() && camera_id < kAdaptiveVisualCameraCount ?
    data_->evidence->imageGeneration(camera_id) :
    0U;
}

double
AdaptiveVisualSelectionReceipt::
candidateExposureTimestampSec(
  const std::uint8_t camera_id) const noexcept
{
  return
    valid() && camera_id < kAdaptiveVisualCameraCount ?
    data_->evidence->exposureTimestampSec(camera_id) :
    std::numeric_limits<double>::quiet_NaN();
}

std::uint64_t
AdaptiveVisualSelectionReceipt::observationGeneration() const noexcept
{
  return candidateObservationGeneration(
    selectedCameraId());
}

std::uint64_t
AdaptiveVisualSelectionReceipt::imageGeneration() const noexcept
{
  return candidateImageGeneration(selectedCameraId());
}

double
AdaptiveVisualSelectionReceipt::exposureTimestampSec() const noexcept
{
  return
    valid() ?
    candidateExposureTimestampSec(selectedCameraId()) :
    std::numeric_limits<double>::quiet_NaN();
}

AdaptiveVisualRuntimeAttemptPermit::
AdaptiveVisualRuntimeAttemptPermit(
  std::shared_ptr<AdaptiveVisualSelectionAuthority> authority,
  AuthoritativeLidarGenerationToken lidar_token,
  const std::uint64_t attempt_nonce)
: authority_(std::move(authority)),
  lidar_token_(std::move(lidar_token)),
  attempt_nonce_(attempt_nonce)
{
}

bool AdaptiveVisualRuntimeAttemptPermit::valid() const noexcept
{
  return
    static_cast<bool>(authority_) &&
    lidar_token_.valid() &&
    lidar_token_.issuedBy(authority_->identity()) &&
    attempt_nonce_ != 0U &&
    lidar_token_.data_->runtime_attempted.load(
      std::memory_order_acquire) &&
    lidar_token_.data_->runtime_attempt_nonce.load(
      std::memory_order_acquire) == attempt_nonce_;
}

std::uint64_t
AdaptiveVisualRuntimeAttemptPermit::attemptNonce() const noexcept
{
  return valid() ? attempt_nonce_ : 0U;
}

AdaptiveVisualSelectionIssueResult
AdaptiveVisualRuntimeAttemptPermit::issueSelection(
  const AdaptiveVisualPolicy & policy,
  const CameraObservationBatch & batch,
  const AdaptiveVisualCandidatePlanCapability &
    policy_capability)
{
  if (phase_ == Phase::kSelectionIssued)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kRuntimeAttemptSelectionAlreadyIssued,
      std::nullopt};
  }
  if (phase_ == Phase::kSelectionConsumed)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kRuntimeAttemptSelectionAlreadyConsumed,
      std::nullopt};
  }
  if (phase_ != Phase::kFresh || !valid())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidRuntimeAttemptPermit,
      std::nullopt};
  }
  const AdaptiveVisualSelectionStatus permit_status =
    authority_->validateRuntimeAttemptPermit(
    lidar_token_, attempt_nonce_);
  if (permit_status != AdaptiveVisualSelectionStatus::kReady)
  {
    phase_ = Phase::kFailed;
    return {permit_status, std::nullopt};
  }

  AdaptiveVisualSelectionIssueResult result =
    authority_->issue(
    policy, batch, policy_capability, lidar_token_);
  phase_ = Phase::kFailed;
  if (
    result.status == AdaptiveVisualSelectionStatus::kReady &&
    result.capability.has_value())
  {
    issued_capability_ = result.capability->data_;
    phase_ = Phase::kSelectionIssued;
  }
  else if (
    result.status ==
      AdaptiveVisualSelectionStatus::kNoSelectedCamera &&
    !result.capability.has_value() &&
    std::none_of(
      policy_capability.decisionPlan().cameras.begin(),
      policy_capability.decisionPlan().cameras.end(),
      [](const AdaptiveCameraDecision & decision)
      {
        return decision.enabled;
      }))
  {
    zero_selection_health_capability_ =
      policy_capability.data_;
    phase_ = Phase::kZeroSelectionHealthReady;
  }
  return result;
}

AdaptiveVisualSelectionConsumeResult
AdaptiveVisualRuntimeAttemptPermit::consumeSelection(
  const AdaptiveVisualPolicy & policy,
  const CameraObservationBatch & batch,
  const AdaptiveVisualSelectionCapability & capability)
{
  const auto cumulative_consumption =
    [this]() -> std::size_t
    {
      return
        lidar_token_.valid() &&
        lidar_token_.data_->consumed.load(
          std::memory_order_acquire) ? 1U : 0U;
    };
  if (phase_ == Phase::kFresh)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kRuntimeAttemptSelectionNotIssued,
      std::nullopt, cumulative_consumption(), false};
  }
  if (phase_ == Phase::kSelectionConsumed)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kRuntimeAttemptSelectionAlreadyConsumed,
      std::nullopt, cumulative_consumption(), false};
  }
  if (
    phase_ != Phase::kSelectionIssued ||
    !valid() ||
    !issued_capability_ ||
    capability.data_ != issued_capability_)
  {
    phase_ = Phase::kFailed;
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidRuntimeAttemptPermit,
      std::nullopt, cumulative_consumption(), false};
  }
  const AdaptiveVisualSelectionStatus permit_status =
    authority_->validateRuntimeAttemptPermit(
    lidar_token_, attempt_nonce_);
  if (permit_status != AdaptiveVisualSelectionStatus::kReady)
  {
    phase_ = Phase::kFailed;
    return {
      permit_status, std::nullopt,
      cumulative_consumption(), false};
  }

  AdaptiveVisualSelectionConsumeResult result =
    authority_->consume(
    policy, batch, capability, lidar_token_);
  phase_ = Phase::kFailed;
  result.selection_consumption_count =
    cumulative_consumption();
  result.consumed_by_this_call =
    result.status == AdaptiveVisualSelectionStatus::kReady &&
    result.receipt.has_value();
  if (result.consumed_by_this_call)
  {
    phase_ = Phase::kSelectionConsumed;
  }
  return result;
}

AdaptiveVisualSelectionAuthority::AdaptiveVisualSelectionAuthority(
  CameraObservationBatchAuthorityIdentity batch_authority)
: batch_authority_(std::move(batch_authority)),
  owner_(std::make_shared<const std::uint8_t>(0U))
{
}

AdaptiveVisualSelectionAuthorityIdentity
AdaptiveVisualSelectionAuthority::identity() const noexcept
{
  return AdaptiveVisualSelectionAuthorityIdentity{owner_};
}

AdaptiveVisualRuntimeAuthorityInput::
AdaptiveVisualRuntimeAuthorityInput(
  std::shared_ptr<AdaptiveVisualSelectionAuthority> authority,
  AuthoritativeLidarGenerationToken lidar_token)
: authority_(std::move(authority)),
  lidar_token_(std::move(lidar_token))
{
}

bool AdaptiveVisualRuntimeAuthorityInput::valid() const noexcept
{
  return
    static_cast<bool>(authority_) &&
    lidar_token_.valid() &&
    lidar_token_.issuedBy(authority_->identity());
}

std::uint64_t
AdaptiveVisualRuntimeAuthorityInput::lidarGeneration() const noexcept
{
  return valid() ? lidar_token_.generation() : 0U;
}

bool AdaptiveVisualRuntimeAuthorityInput::
lidarInformationReady() const noexcept
{
  return valid() && lidar_token_.lidarInformationReady();
}

const AdaptiveVisualPolicy::Matrix6 &
AdaptiveVisualRuntimeAuthorityInput::
lidarPoseInformation() const
{
  if (!valid())
  {
    throw std::logic_error(
      "invalid adaptive visual runtime authority input");
  }
  return lidar_token_.lidarPoseInformation();
}

AdaptiveVisualSelectionStatus
AdaptiveVisualRuntimeAuthorityInput::preflightBatch(
  const CameraObservationBatch & batch) const
{
  return
    valid() ?
    authority_->preflightRuntimeBatch(batch) :
    AdaptiveVisualSelectionStatus::kInvalidAuthority;
}

AdaptiveVisualRuntimeAttemptResult
AdaptiveVisualRuntimeAuthorityInput::beginRuntimeAttempt() const
{
  if (!valid())
  {
    return {};
  }
  return authority_->beginRuntimeAttempt(lidar_token_);
}

std::optional<AdaptiveVisualRuntimeAuthorityInput>
AdaptiveVisualSelectionAuthority::bindRuntimeInput(
  const AuthoritativeLidarGenerationToken & lidar_token)
{
  if (
    !batch_authority_.valid() ||
    !owner_ ||
    !lidar_token.valid() ||
    !lidar_token.issuedBy(identity()))
  {
    return std::nullopt;
  }
  std::shared_ptr<AdaptiveVisualSelectionAuthority> self =
    self_.lock();
  if (!self)
  {
    return std::nullopt;
  }
  return AdaptiveVisualRuntimeAuthorityInput{
    std::move(self), lidar_token};
}

AdaptiveVisualSelectionStatus
AdaptiveVisualSelectionAuthority::preflightRuntimeBatch(
  const CameraObservationBatch & batch) const
{
  if (!batch_authority_.valid() || !owner_)
  {
    return AdaptiveVisualSelectionStatus::kInvalidAuthority;
  }
  if (
    !batch.validatedReceipt().has_value() ||
    !batch.validatedReceipt()->valid())
  {
    return
      AdaptiveVisualSelectionStatus::
      kInvalidCameraBatchReceipt;
  }
  if (
    !batch.validatedReceipt()->issuedBy(batch_authority_))
  {
    return
      AdaptiveVisualSelectionStatus::
      kForeignCameraBatchOwner;
  }
  const std::size_t present_count =
    static_cast<std::size_t>(
    std::count(
      batch.present.begin(), batch.present.end(), true));
  if (
    batch.camera_count == 0U ||
    batch.camera_count > kAdaptiveVisualCameraCount ||
    present_count != batch.camera_count)
  {
    return
      AdaptiveVisualSelectionStatus::
      kInvalidCandidatePresentMask;
  }
  if (!batch.validatedReceipt()->matchesAndValidates(batch))
  {
    return
      AdaptiveVisualSelectionStatus::
      kObservationImageIdentityMismatch;
  }
  return AdaptiveVisualSelectionStatus::kReady;
}

AdaptiveVisualRuntimeAttemptResult
AdaptiveVisualSelectionAuthority::beginRuntimeAttempt(
  const AuthoritativeLidarGenerationToken & lidar_token)
{
  if (
    !batch_authority_.valid() ||
    !owner_ ||
    !lidar_token.valid() ||
    !lidar_token.issuedBy(identity()))
  {
    return {};
  }
  const std::lock_guard<std::mutex> lock(
    transaction_mutex_);
  if (
    lidar_token.generation() !=
    latest_lidar_generation_)
  {
    return {
      AdaptiveVisualRuntimeAttemptStatus::kStaleGeneration,
      lidar_token.data_->runtime_attempted.load(
        std::memory_order_acquire) ? 1U : 0U,
      lidar_token.data_->consumed.load(
        std::memory_order_acquire) ? 1U : 0U,
      std::nullopt};
  }
  if (
    lidar_token.data_->runtime_attempted.load(
      std::memory_order_acquire))
  {
    return {
      AdaptiveVisualRuntimeAttemptStatus::kAlreadyAttempted,
      1U,
      lidar_token.data_->consumed.load(
        std::memory_order_acquire) ? 1U : 0U,
      std::nullopt};
  }
  std::shared_ptr<AdaptiveVisualSelectionAuthority> self =
    self_.lock();
  if (!self)
  {
    return {};
  }
  std::uint64_t attempt_nonce = 0U;
  if (
    !checkedReserve(
      next_runtime_attempt_nonce_, attempt_nonce))
  {
    return {};
  }
  bool expected = false;
  if (
    !lidar_token.data_->runtime_attempted.
    compare_exchange_strong(
      expected, true, std::memory_order_acq_rel))
  {
    return {
      AdaptiveVisualRuntimeAttemptStatus::kAlreadyAttempted,
      1U,
      lidar_token.data_->consumed.load(
        std::memory_order_acquire) ? 1U : 0U,
      std::nullopt};
  }
  lidar_token.data_->runtime_attempt_nonce.store(
    attempt_nonce, std::memory_order_release);
  return {
    AdaptiveVisualRuntimeAttemptStatus::kReady,
    1U,
    lidar_token.data_->consumed.load(
      std::memory_order_acquire) ? 1U : 0U,
    AdaptiveVisualRuntimeAttemptPermit{
      std::move(self), lidar_token, attempt_nonce}};
}

AdaptiveVisualSelectionStatus
AdaptiveVisualSelectionAuthority::validateRuntimeAttemptPermit(
  const AuthoritativeLidarGenerationToken & lidar_token,
  const std::uint64_t attempt_nonce)
{
  if (
    !batch_authority_.valid() ||
    !owner_ ||
    !lidar_token.valid() ||
    !lidar_token.issuedBy(identity()) ||
    attempt_nonce == 0U)
  {
    return
      AdaptiveVisualSelectionStatus::
      kInvalidRuntimeAttemptPermit;
  }
  const std::lock_guard<std::mutex> lock(
    transaction_mutex_);
  if (lidar_token.generation() != latest_lidar_generation_)
  {
    return
      AdaptiveVisualSelectionStatus::kStaleLidarGeneration;
  }
  if (
    !lidar_token.data_->runtime_attempted.load(
      std::memory_order_acquire) ||
    lidar_token.data_->runtime_attempt_nonce.load(
      std::memory_order_acquire) != attempt_nonce)
  {
    return
      AdaptiveVisualSelectionStatus::
      kInvalidRuntimeAttemptPermit;
  }
  return AdaptiveVisualSelectionStatus::kReady;
}

AuthoritativeLidarGenerationIssueResult
AdaptiveVisualSelectionAuthority::issueLidarGeneration(
  const std::uint64_t generation,
  const AdaptiveVisualPolicy::Matrix6 &
    lidar_pose_information,
  const bool lidar_information_ready)
{
  if (generation == 0U)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kZeroLidarGeneration,
      std::nullopt};
  }
  if (!lidar_pose_information.allFinite())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidLidarInformation,
      std::nullopt};
  }
  const std::lock_guard<std::mutex> lock(
    transaction_mutex_);
  if (generation <= latest_lidar_generation_)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kNonMonotonicLidarGeneration,
      std::nullopt};
  }
  auto data =
    std::make_shared<
    AuthoritativeLidarGenerationToken::Data>(
    owner_, generation, lidar_pose_information,
    lidar_information_ready);
  latest_lidar_generation_ = generation;
  return {
    AdaptiveVisualSelectionStatus::kReady,
    AuthoritativeLidarGenerationToken{std::move(data)}};
}

AdaptiveVisualSelectionIssueResult
AdaptiveVisualSelectionAuthority::issue(
  const AdaptiveVisualPolicy & policy,
  const CameraObservationBatch & batch,
  const AdaptiveVisualCandidatePlanCapability & policy_capability,
  const AuthoritativeLidarGenerationToken & lidar_token)
{
  if (!batch_authority_.valid() || !owner_)
  {
    return {
      AdaptiveVisualSelectionStatus::kInvalidAuthority,
      std::nullopt};
  }
  if (!lidar_token.valid())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidLidarGenerationToken,
      std::nullopt};
  }
  if (!sameSharedOwner(lidar_token.data_->owner, owner_))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kForeignLidarGenerationOwner,
      std::nullopt};
  }
  if (!policy_capability.valid())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidPolicyCapability,
      std::nullopt};
  }
  if (!policy.ownsCandidatePlanCapability(policy_capability))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kForeignPolicyOwner,
      std::nullopt};
  }
  if (
    policy.revision() !=
    policy_capability.basePolicyRevision() ||
    !policy.accepts(policy_capability))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidPolicyCapability,
      std::nullopt};
  }
  const auto & plan = policy_capability.decisionPlan();
  if (
    lidar_token.lidarInformationReady() !=
    policy_capability.lidarInformationReady())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kLidarInformationReadinessMismatch,
      std::nullopt};
  }
  if (
    !lidar_token.matchesLidarInformation(
      policy_capability.lidarPoseInformation()))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kLidarInformationIdentityMismatch,
      std::nullopt};
  }
  if (!lidar_token.lidarInformationReady())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kLidarInformationNotReady,
      std::nullopt};
  }
  if (!plan.lidar_information_valid)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidLidarInformation,
      std::nullopt};
  }
  if (
    !batch.validatedReceipt().has_value() ||
    !batch.validatedReceipt()->valid())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidCameraBatchReceipt,
      std::nullopt};
  }
  if (
    !batch.validatedReceipt()->issuedBy(batch_authority_))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kForeignCameraBatchOwner,
      std::nullopt};
  }
  const std::size_t present_count =
    static_cast<std::size_t>(
    std::count(
      batch.present.begin(), batch.present.end(),
      true));
  if (
    batch.camera_count == 0U ||
    batch.camera_count > kAdaptiveVisualCameraCount ||
    present_count != batch.camera_count)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidCandidatePresentMask,
      std::nullopt};
  }
  if (
    batch.validatedReceipt()->batchSequence() !=
      policy_capability.cameraBatchSequence() ||
    !batch.validatedReceipt()->sameSealedBatch(
      policy_capability.cameraBatchReceipt()) ||
    batch.present != policy_capability.candidateMask())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kCameraBatchIdentityMismatch,
      std::nullopt};
  }
  if (!batch.validatedReceipt()->matchesAndValidates(batch))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kObservationImageIdentityMismatch,
      std::nullopt};
  }

  for (std::size_t camera_id = 0U;
       camera_id < kAdaptiveVisualCameraCount;
       ++camera_id)
  {
    if (!policy_capability.candidateMask()[camera_id]) {
      continue;
    }
    const auto & policy_observation =
      policy_capability.observations()[camera_id];
    if (
      !policy_observation.frame_received ||
      policy_observation.observation_generation == 0U ||
      policy_observation.image_generation == 0U ||
      policy_observation.observation_generation !=
        policy_capability.cameraBatchReceipt().
        observationGeneration(
          static_cast<std::uint8_t>(camera_id)) ||
      policy_observation.image_generation !=
        policy_capability.cameraBatchReceipt().
        imageGeneration(
          static_cast<std::uint8_t>(camera_id)))
    {
      return {
        AdaptiveVisualSelectionStatus::
        kPolicyObservationIdentityMismatch,
        std::nullopt};
    }
  }

  std::size_t enabled_count = 0U;
  std::uint8_t selected_camera = 0U;
  for (std::size_t camera_id = 0U;
       camera_id < kAdaptiveVisualCameraCount; ++camera_id)
  {
    if (plan.cameras[camera_id].enabled)
    {
      ++enabled_count;
      selected_camera = static_cast<std::uint8_t>(camera_id);
    }
  }
  if (enabled_count == 0U)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kNoSelectedCamera,
      std::nullopt};
  }
  if (enabled_count > 1U)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kMultipleSelectedCameras,
      std::nullopt};
  }
  if (!batch.present[selected_camera])
  {
    return {
      AdaptiveVisualSelectionStatus::
      kSelectedCameraNotCandidate,
      std::nullopt};
  }

  const auto & decision = plan.cameras[selected_camera];
  bool budgets_match =
    decision.candidate_budget ==
      policy.config().total_candidate_budget &&
    decision.patch_budget ==
      policy.config().total_patch_budget;
  for (std::size_t camera_id = 0U;
       camera_id < kAdaptiveVisualCameraCount;
       ++camera_id)
  {
    if (camera_id == selected_camera) {
      continue;
    }
    budgets_match =
      budgets_match &&
      plan.cameras[camera_id].candidate_budget == 0U &&
      plan.cameras[camera_id].patch_budget == 0U;
  }
  if (!budgets_match)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kPlanBudgetMismatch,
      std::nullopt};
  }
  auto evidence =
    BoundCandidateEvidence::bind(policy_capability);
  if (!evidence.has_value())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidSelectionCapability,
      std::nullopt};
  }
  auto data =
    std::make_shared<
    const AdaptiveVisualSelectionCapability::Data>(
    owner_, lidar_token, std::move(*evidence));
  AdaptiveVisualSelectionCapability capability{
    std::move(data)};
  if (!capability.valid()) {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidSelectionCapability,
      std::nullopt};
  }
  return {
    AdaptiveVisualSelectionStatus::kReady,
    std::move(capability)};
}

AdaptiveVisualSelectionConsumeResult
AdaptiveVisualSelectionAuthority::consume(
  const AdaptiveVisualPolicy & policy,
  const CameraObservationBatch & batch,
  const AdaptiveVisualSelectionCapability & capability,
  const AuthoritativeLidarGenerationToken & lidar_token)
{
  if (!batch_authority_.valid() || !owner_)
  {
    return {
      AdaptiveVisualSelectionStatus::kInvalidAuthority,
      std::nullopt};
  }
  if (!capability.valid())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidSelectionCapability,
      std::nullopt};
  }
  if (
    !sameSharedOwner(capability.data_->owner, owner_))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kForeignSelectionOwner,
      std::nullopt};
  }
  if (!lidar_token.valid())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidLidarGenerationToken,
      std::nullopt};
  }
  if (!sameSharedOwner(lidar_token.data_->owner, owner_))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kForeignLidarGenerationOwner,
      std::nullopt};
  }
  if (
    capability.data_->lidar_token.data_ !=
    lidar_token.data_)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kLidarTokenMismatch,
      std::nullopt};
  }
  if (
    !policy.ownsCandidatePlanCapability(
      capability.data_->evidence.policyCapability()))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kForeignPolicyOwner,
      std::nullopt};
  }
  if (
    policy.revision() != capability.policyRevision())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kStalePolicyRevision,
      std::nullopt};
  }
  if (
    !policy.accepts(
      capability.data_->evidence.policyCapability()))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidPolicyCapability,
      std::nullopt};
  }
  if (
    !batch.validatedReceipt().has_value() ||
    !batch.validatedReceipt()->valid())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kInvalidCameraBatchReceipt,
      std::nullopt};
  }
  if (!batch.validatedReceipt()->issuedBy(batch_authority_))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kForeignCameraBatchOwner,
      std::nullopt};
  }
  if (
    batch.validatedReceipt()->batchSequence() !=
      capability.cameraBatchSequence() ||
    !batch.validatedReceipt()->sameSealedBatch(
      capability.data_->evidence.cameraBatchReceipt()))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kCameraBatchIdentityMismatch,
      std::nullopt};
  }
  if (!batch.validatedReceipt()->matchesAndValidates(batch))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kObservationImageIdentityMismatch,
      std::nullopt};
  }

#ifdef LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING
  if (fail_next_receipt_allocation.exchange(
      false, std::memory_order_acq_rel))
  {
    throw std::bad_alloc();
  }
#endif
  std::shared_ptr<const BoundCandidateEvidence> evidence{
    capability.data_, &capability.data_->evidence};
  auto receipt_data =
    std::make_shared<
    const AdaptiveVisualSelectionReceipt::Data>(
    owner_, capability.data_->lidar_token,
    std::move(evidence),
    AdaptiveVisualSelectionStatus::kReady);
  AdaptiveVisualSelectionConsumeResult ready_result{
    AdaptiveVisualSelectionStatus::kReady,
    AdaptiveVisualSelectionReceipt{
      std::move(receipt_data)},
    1U,
    true};

  const std::lock_guard<std::mutex> lock(
    transaction_mutex_);
  if (capability.lidarGeneration() !=
      latest_lidar_generation_)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kStaleLidarGeneration,
      std::nullopt};
  }
  if (lidar_token.data_->consumed.load(
      std::memory_order_acquire))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kLidarGenerationAlreadyConsumed,
      std::nullopt};
  }
  if (capability.cameraBatchSequence() <=
      last_consumed_batch_sequence_)
  {
    return {
      AdaptiveVisualSelectionStatus::
      kCameraBatchAlreadyConsumed,
      std::nullopt};
  }
  if (
    !capability.data_->evidence.cameraBatchReceipt().
    claimAdaptiveVisualSelection())
  {
    return {
      AdaptiveVisualSelectionStatus::
      kCameraBatchAlreadyConsumed,
      std::nullopt};
  }
  bool expected = false;
  if (!lidar_token.data_->consumed.compare_exchange_strong(
      expected, true, std::memory_order_acq_rel))
  {
    return {
      AdaptiveVisualSelectionStatus::
      kLidarGenerationAlreadyConsumed,
      std::nullopt};
  }
  last_consumed_batch_sequence_ =
    capability.cameraBatchSequence();
  return ready_result;
}

AdaptiveCandidateHealthCommitStatus
AdaptiveVisualPolicy::commitHealthObservation(
  const AdaptiveVisualCandidatePlanCapability & capability,
  const AdaptiveVisualSelectionReceipt & receipt,
  const bool scout_trusted)
{
  const std::lock_guard<std::mutex> transaction_lock(
    candidate_commit_transaction_mutex_);
  if (!scout_trusted)
  {
    return
      AdaptiveCandidateHealthCommitStatus::kUntrustedScout;
  }
  if (!ownsCandidatePlanCapability(capability))
  {
    return AdaptiveCandidateHealthCommitStatus::
           kInvalidOrForeignCapability;
  }
  if (!receipt.valid())
  {
    return AdaptiveCandidateHealthCommitStatus::
           kInvalidOrForeignReceipt;
  }
  std::size_t selected_count = 0U;
  std::uint8_t selected_camera = 0U;
  for (std::size_t camera_id = 0U;
       camera_id < kAdaptiveVisualCameraCount;
       ++camera_id)
  {
    if (capability.decisionPlan().cameras[camera_id].enabled)
    {
      ++selected_count;
      selected_camera =
        static_cast<std::uint8_t>(camera_id);
    }
  }
  if (
    capability.data_ !=
      receipt.data_->evidence->
      policyCapability().data_ ||
    receipt.cameraBatchSequence() !=
      capability.cameraBatchSequence() ||
    receipt.policyRevision() !=
      capability.basePolicyRevision() ||
    receipt.selectionReason() !=
      capability.selectionReason() ||
    receipt.candidateMask() != capability.candidateMask() ||
    !receipt.data_->evidence->
    cameraBatchReceipt().sameSealedBatch(
      capability.cameraBatchReceipt()) ||
    receipt.lidarInformationReady() !=
      capability.lidarInformationReady() ||
    !(receipt.lidarPoseInformation().array() ==
      capability.lidarPoseInformation().array()).all() ||
    selected_count != 1U ||
    receipt.selectedCameraId() != selected_camera ||
    receipt.observationGeneration() !=
      capability.cameraBatchReceipt().
      observationGeneration(selected_camera) ||
    receipt.imageGeneration() !=
      capability.cameraBatchReceipt().
      imageGeneration(selected_camera))
  {
    return
      AdaptiveCandidateHealthCommitStatus::kBindingMismatch;
  }
  if (
    receipt.lidarGeneration() <=
    last_committed_candidate_lidar_generation_)
  {
    return AdaptiveCandidateHealthCommitStatus::
           kNonMonotonicOrReplayedLidarGeneration;
  }
  if (
    revision_ ==
    std::numeric_limits<std::uint64_t>::max())
  {
    return
      AdaptiveCandidateHealthCommitStatus::kRevisionExhausted;
  }
  if (!accepts(capability))
  {
    return
      AdaptiveCandidateHealthCommitStatus::kStalePolicyRevision;
  }
  if (!capability.decisionPlan().lidar_information_valid)
  {
    return AdaptiveCandidateHealthCommitStatus::
           kInvalidOrForeignCapability;
  }

  hysteresis_ =
    capability.decisionPlan().proposed_hysteresis;
  primary_camera_ =
    capability.decisionPlan().proposed_primary_camera;
  selection_challenger_ =
    capability.decisionPlan().
    proposed_selection_challenger;
  last_committed_candidate_lidar_generation_ =
    receipt.lidarGeneration();
  ++revision_;
  return AdaptiveCandidateHealthCommitStatus::kCommitted;
}

AdaptiveCandidateHealthCommitStatus
AdaptiveVisualPolicy::commitZeroSelectionHealthObservation(
  const AdaptiveVisualCandidatePlanCapability & capability,
  AdaptiveVisualRuntimeAttemptPermit && permit)
{
  const std::lock_guard<std::mutex> transaction_lock(
    candidate_commit_transaction_mutex_);
  const auto reject =
    [&permit](
      const AdaptiveCandidateHealthCommitStatus status)
    {
      permit.phase_ =
        AdaptiveVisualRuntimeAttemptPermit::Phase::kFailed;
      permit.issued_capability_.reset();
      permit.zero_selection_health_capability_.reset();
      return status;
    };

  if (!ownsCandidatePlanCapability(capability))
  {
    return reject(
      AdaptiveCandidateHealthCommitStatus::
      kInvalidOrForeignCapability);
  }
  if (
    permit.phase_ !=
      AdaptiveVisualRuntimeAttemptPermit::Phase::
      kZeroSelectionHealthReady ||
    !permit.valid() ||
    !permit.zero_selection_health_capability_ ||
    permit.zero_selection_health_capability_ !=
      capability.data_)
  {
    return reject(
      AdaptiveCandidateHealthCommitStatus::
      kBindingMismatch);
  }
  if (
    permit.authority_->validateRuntimeAttemptPermit(
      permit.lidar_token_, permit.attempt_nonce_) !=
      AdaptiveVisualSelectionStatus::kReady)
  {
    return reject(
      AdaptiveCandidateHealthCommitStatus::
      kNonMonotonicOrReplayedLidarGeneration);
  }
  if (!accepts(capability))
  {
    return reject(
      AdaptiveCandidateHealthCommitStatus::
      kStalePolicyRevision);
  }

  const AdaptiveVisualPolicyPlan & plan =
    capability.decisionPlan();
  const std::size_t enabled_count =
    static_cast<std::size_t>(
    std::count_if(
      plan.cameras.begin(), plan.cameras.end(),
      [](const AdaptiveCameraDecision & decision)
      {
        return decision.enabled;
      }));
  if (
    enabled_count != 0U ||
    !plan.adaptive_enabled ||
    !plan.lidar_information_valid ||
    !capability.lidarInformationReady() ||
    !permit.lidar_token_.lidarInformationReady() ||
    !capability.lidarPoseInformation().allFinite() ||
    (capability.lidarPoseInformation().array() !=
    permit.lidar_token_.lidarPoseInformation().array()).any() ||
    !capability.cameraBatchReceipt().valid() ||
    capability.cameraBatchReceipt().
    adaptiveVisualSelectionClaimed())
  {
    return reject(
      AdaptiveCandidateHealthCommitStatus::
      kBindingMismatch);
  }
  for (std::size_t camera_id = 0U;
       camera_id < kAdaptiveVisualCameraCount;
       ++camera_id)
  {
    if (!capability.candidateMask()[camera_id])
    {
      continue;
    }
    const auto candidate_id =
      static_cast<std::uint8_t>(camera_id);
    const AdaptiveCameraObservation & observation =
      capability.observations()[camera_id];
    if (
      !observation.frame_received ||
      observation.camera_id != candidate_id ||
      observation.observation_generation == 0U ||
      observation.image_generation == 0U ||
      observation.observation_generation !=
      capability.cameraBatchReceipt().
      observationGeneration(candidate_id) ||
      observation.image_generation !=
      capability.cameraBatchReceipt().
      imageGeneration(candidate_id))
    {
      return reject(
        AdaptiveCandidateHealthCommitStatus::
        kBindingMismatch);
    }
  }

  const std::uint64_t lidar_generation =
    permit.lidar_token_.generation();
  if (
    lidar_generation == 0U ||
    lidar_generation <=
    last_committed_candidate_lidar_generation_)
  {
    return reject(
      AdaptiveCandidateHealthCommitStatus::
      kNonMonotonicOrReplayedLidarGeneration);
  }
  if (
    revision_ ==
    std::numeric_limits<std::uint64_t>::max())
  {
    return reject(
      AdaptiveCandidateHealthCommitStatus::
      kRevisionExhausted);
  }

  hysteresis_ = plan.proposed_hysteresis;
  primary_camera_ = plan.proposed_primary_camera;
  selection_challenger_ =
    plan.proposed_selection_challenger;
  last_committed_candidate_lidar_generation_ =
    lidar_generation;
  ++revision_;
  permit.phase_ =
    AdaptiveVisualRuntimeAttemptPermit::Phase::
    kZeroSelectionHealthCommitted;
  permit.zero_selection_health_capability_.reset();
  return AdaptiveCandidateHealthCommitStatus::kCommitted;
}

}  // namespace livo
