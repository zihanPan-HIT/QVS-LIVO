#ifndef LIVO_ADAPTIVE_VISUAL_SELECTION_AUTHORITY_H_
#define LIVO_ADAPTIVE_VISUAL_SELECTION_AUTHORITY_H_

#include "adaptive_visual_policy.h"
#include "camera_observation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace livo
{

enum class AdaptiveVisualSelectionStatus
{
  kReady,
  kForeignLidarGenerationOwner,
  kInvalidCandidatePresentMask,
  kNoSelectedCamera,
  kMultipleSelectedCameras,
  kSelectedCameraNotCandidate,
  kObservationImageIdentityMismatch,
  kStalePolicyRevision,
  kZeroLidarGeneration,
  kStaleLidarGeneration,
  kLidarGenerationAlreadyConsumed,
  kForeignSelectionOwner,
  kForeignCameraBatchOwner,
  kForeignPolicyOwner,
  kNonMonotonicLidarGeneration,
  kCameraBatchAlreadyConsumed,
  kPolicyObservationIdentityMismatch,
  kInvalidAuthority,
  kInvalidLidarGenerationToken,
  kInvalidPolicyCapability,
  kInvalidCameraBatchReceipt,
  kCameraBatchIdentityMismatch,
  kInvalidSelectionCapability,
  kLidarTokenMismatch,
  kPlanBudgetMismatch,
  kInvalidLidarInformation,
  kLidarInformationIdentityMismatch,
  kLidarInformationReadinessMismatch,
  kLidarInformationNotReady,
  kInvalidRuntimeAttemptPermit,
  kRuntimeAttemptSelectionAlreadyIssued,
  kRuntimeAttemptSelectionNotIssued,
  kRuntimeAttemptSelectionAlreadyConsumed,
};

enum class AdaptiveVisualRuntimeAttemptStatus
{
  kReady,
  kInvalidInput,
  kStaleGeneration,
  kAlreadyAttempted,
};

class AdaptiveVisualSelectionAuthority;
class AdaptiveVisualRuntimeAuthorityInput;
class AdaptiveVisualRuntimeAttemptPermit;

class AdaptiveVisualSelectionAuthorityIdentity
{
public:
  AdaptiveVisualSelectionAuthorityIdentity() = default;

  bool valid() const noexcept;
  bool sameAuthority(
    const AdaptiveVisualSelectionAuthorityIdentity &
      other) const noexcept;

private:
  explicit AdaptiveVisualSelectionAuthorityIdentity(
    std::shared_ptr<const std::uint8_t> owner);

  std::shared_ptr<const std::uint8_t> owner_;

  friend class AdaptiveVisualSelectionAuthority;
  friend class AuthoritativeLidarGenerationToken;
  friend class AdaptiveVisualSelectionCapability;
  friend class AdaptiveVisualSelectionReceipt;
};

class AuthoritativeLidarGenerationToken
{
public:
  AuthoritativeLidarGenerationToken(
    const AuthoritativeLidarGenerationToken &) = default;
  AuthoritativeLidarGenerationToken(
    AuthoritativeLidarGenerationToken &&) = default;
  AuthoritativeLidarGenerationToken & operator=(
    const AuthoritativeLidarGenerationToken &) = default;
  AuthoritativeLidarGenerationToken & operator=(
    AuthoritativeLidarGenerationToken &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const AdaptiveVisualSelectionAuthorityIdentity &
      identity) const noexcept;
  std::uint64_t generation() const noexcept;
  bool lidarInformationReady() const noexcept;
  const AdaptiveVisualPolicy::Matrix6 &
  lidarPoseInformation() const;

private:
  struct Data;

  explicit AuthoritativeLidarGenerationToken(
    std::shared_ptr<Data> data);
  bool matchesLidarInformation(
    const AdaptiveVisualPolicy::Matrix6 &
      lidar_pose_information) const noexcept;

  std::shared_ptr<Data> data_;

  friend class AdaptiveVisualSelectionAuthority;
  friend class AdaptiveVisualSelectionReceipt;
  friend class AdaptiveVisualRuntimeAttemptPermit;
};

class AdaptiveVisualSelectionCapability
{
public:
  AdaptiveVisualSelectionCapability(
    const AdaptiveVisualSelectionCapability &) = default;
  AdaptiveVisualSelectionCapability(
    AdaptiveVisualSelectionCapability &&) = default;
  AdaptiveVisualSelectionCapability & operator=(
    const AdaptiveVisualSelectionCapability &) = default;
  AdaptiveVisualSelectionCapability & operator=(
    AdaptiveVisualSelectionCapability &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const AdaptiveVisualSelectionAuthorityIdentity &
      identity) const noexcept;
  std::uint64_t lidarGeneration() const noexcept;
  std::uint64_t cameraBatchSequence() const noexcept;
  std::uint64_t policyRevision() const noexcept;
  AdaptiveSelectionReason selectionReason() const noexcept;
  bool lidarInformationReady() const noexcept;
  const AdaptiveVisualPolicy::Matrix6 &
  lidarPoseInformation() const;
  std::uint8_t selectedCameraId() const noexcept;
  const std::array<bool, kAdaptiveVisualCameraCount> &
  candidateMask() const;
  const std::array<bool, kAdaptiveVisualCameraCount> &
  selectedMask() const;
  std::uint64_t candidateObservationGeneration(
    std::uint8_t camera_id) const noexcept;
  std::uint64_t candidateImageGeneration(
    std::uint8_t camera_id) const noexcept;
  double candidateExposureTimestampSec(
    std::uint8_t camera_id) const noexcept;
  std::size_t candidateBudget() const noexcept;
  std::size_t patchBudget() const noexcept;

private:
  struct Data;

  explicit AdaptiveVisualSelectionCapability(
    std::shared_ptr<const Data> data);

  std::shared_ptr<const Data> data_;

  friend class AdaptiveVisualSelectionAuthority;
  friend class AdaptiveVisualRuntimeAttemptPermit;
};

class AdaptiveVisualSelectionReceipt
{
public:
  AdaptiveVisualSelectionReceipt(
    const AdaptiveVisualSelectionReceipt &) = default;
  AdaptiveVisualSelectionReceipt(
    AdaptiveVisualSelectionReceipt &&) = default;
  AdaptiveVisualSelectionReceipt & operator=(
    const AdaptiveVisualSelectionReceipt &) = default;
  AdaptiveVisualSelectionReceipt & operator=(
    AdaptiveVisualSelectionReceipt &&) = default;

  bool valid() const noexcept;
  bool issuedBy(
    const AdaptiveVisualSelectionAuthorityIdentity &
      identity) const noexcept;
  std::uint64_t lidarGeneration() const noexcept;
  std::uint64_t cameraBatchSequence() const noexcept;
  std::uint64_t policyRevision() const noexcept;
  AdaptiveSelectionReason selectionReason() const noexcept;
  bool lidarInformationReady() const noexcept;
  const AdaptiveVisualPolicy::Matrix6 &
  lidarPoseInformation() const;
  std::uint8_t selectedCameraId() const noexcept;
  std::size_t candidateBudget() const noexcept;
  std::size_t patchBudget() const noexcept;
  AdaptiveVisualSelectionStatus selectionStatus() const noexcept;
  const std::array<bool, kAdaptiveVisualCameraCount> &
  candidateMask() const;
  const std::array<bool, kAdaptiveVisualCameraCount> &
  selectedMask() const;
  bool cameraPresent(std::uint8_t camera_id) const noexcept;
  std::uint64_t candidateObservationGeneration(
    std::uint8_t camera_id) const noexcept;
  std::uint64_t candidateImageGeneration(
    std::uint8_t camera_id) const noexcept;
  double candidateExposureTimestampSec(
    std::uint8_t camera_id) const noexcept;
  std::uint64_t observationGeneration() const noexcept;
  std::uint64_t imageGeneration() const noexcept;
  double exposureTimestampSec() const noexcept;

private:
  struct Data;

  explicit AdaptiveVisualSelectionReceipt(
    std::shared_ptr<const Data> data);

  std::shared_ptr<const Data> data_;

  friend class AdaptiveVisualPolicy;
  friend class AdaptiveVisualSelectionAuthority;
};

struct AuthoritativeLidarGenerationIssueResult
{
  AdaptiveVisualSelectionStatus status{
    AdaptiveVisualSelectionStatus::
    kInvalidLidarGenerationToken};
  std::optional<AuthoritativeLidarGenerationToken> token;
};

struct AdaptiveVisualSelectionIssueResult
{
  AdaptiveVisualSelectionStatus status{
    AdaptiveVisualSelectionStatus::
    kInvalidSelectionCapability};
  std::optional<AdaptiveVisualSelectionCapability> capability;
};

struct AdaptiveVisualSelectionConsumeResult
{
  AdaptiveVisualSelectionStatus status{
    AdaptiveVisualSelectionStatus::
    kInvalidSelectionCapability};
  std::optional<AdaptiveVisualSelectionReceipt> receipt;
  std::size_t selection_consumption_count{0U};
  bool consumed_by_this_call{false};
};

class AdaptiveVisualRuntimeAttemptPermit
{
public:
  AdaptiveVisualRuntimeAttemptPermit(
    const AdaptiveVisualRuntimeAttemptPermit &) = delete;
  AdaptiveVisualRuntimeAttemptPermit & operator=(
    const AdaptiveVisualRuntimeAttemptPermit &) = delete;
  AdaptiveVisualRuntimeAttemptPermit(
    AdaptiveVisualRuntimeAttemptPermit &&) = default;
  AdaptiveVisualRuntimeAttemptPermit & operator=(
    AdaptiveVisualRuntimeAttemptPermit &&) = default;

  bool valid() const noexcept;
  std::uint64_t attemptNonce() const noexcept;
  AdaptiveVisualSelectionIssueResult issueSelection(
    const AdaptiveVisualPolicy & policy,
    const CameraObservationBatch & batch,
    const AdaptiveVisualCandidatePlanCapability &
      policy_capability);
  AdaptiveVisualSelectionConsumeResult consumeSelection(
    const AdaptiveVisualPolicy & policy,
    const CameraObservationBatch & batch,
    const AdaptiveVisualSelectionCapability &
      capability);

private:
  enum class Phase
  {
    kFresh,
    kSelectionIssued,
    kSelectionConsumed,
    kZeroSelectionHealthReady,
    kZeroSelectionHealthCommitted,
    kFailed,
  };

  AdaptiveVisualRuntimeAttemptPermit(
    std::shared_ptr<AdaptiveVisualSelectionAuthority> authority,
    AuthoritativeLidarGenerationToken lidar_token,
    std::uint64_t attempt_nonce);

  std::shared_ptr<AdaptiveVisualSelectionAuthority> authority_;
  AuthoritativeLidarGenerationToken lidar_token_;
  std::uint64_t attempt_nonce_{0U};
  Phase phase_{Phase::kFresh};
  std::shared_ptr<
    const AdaptiveVisualSelectionCapability::Data>
    issued_capability_;
  std::shared_ptr<
    const AdaptiveVisualCandidatePlanCapability::Data>
    zero_selection_health_capability_;

  friend class AdaptiveVisualSelectionAuthority;
  friend class AdaptiveVisualPolicy;
};

struct AdaptiveVisualRuntimeAttemptResult
{
  AdaptiveVisualRuntimeAttemptStatus status{
    AdaptiveVisualRuntimeAttemptStatus::kInvalidInput};
  std::size_t runtime_attempt_count{0U};
  std::size_t selection_consumption_count{0U};
  std::optional<AdaptiveVisualRuntimeAttemptPermit> permit;
};

class AdaptiveVisualRuntimeAuthorityInput
{
public:
  AdaptiveVisualRuntimeAuthorityInput(
    const AdaptiveVisualRuntimeAuthorityInput &) = default;
  AdaptiveVisualRuntimeAuthorityInput(
    AdaptiveVisualRuntimeAuthorityInput &&) = default;
  AdaptiveVisualRuntimeAuthorityInput & operator=(
    const AdaptiveVisualRuntimeAuthorityInput &) = default;
  AdaptiveVisualRuntimeAuthorityInput & operator=(
    AdaptiveVisualRuntimeAuthorityInput &&) = default;

  bool valid() const noexcept;
  std::uint64_t lidarGeneration() const noexcept;
  bool lidarInformationReady() const noexcept;
  const AdaptiveVisualPolicy::Matrix6 &
  lidarPoseInformation() const;
  AdaptiveVisualSelectionStatus preflightBatch(
    const CameraObservationBatch & batch) const;
  AdaptiveVisualRuntimeAttemptResult
  beginRuntimeAttempt() const;

private:
  AdaptiveVisualRuntimeAuthorityInput(
    std::shared_ptr<AdaptiveVisualSelectionAuthority> authority,
    AuthoritativeLidarGenerationToken lidar_token);

  std::shared_ptr<AdaptiveVisualSelectionAuthority> authority_;
  AuthoritativeLidarGenerationToken lidar_token_;

  friend class AdaptiveVisualSelectionAuthority;
};

class AdaptiveVisualSelectionAuthority
{
public:
  AdaptiveVisualSelectionAuthority(
    const AdaptiveVisualSelectionAuthority &) = delete;
  AdaptiveVisualSelectionAuthority & operator=(
    const AdaptiveVisualSelectionAuthority &) = delete;
  AdaptiveVisualSelectionAuthority(
    AdaptiveVisualSelectionAuthority &&) = delete;
  AdaptiveVisualSelectionAuthority & operator=(
    AdaptiveVisualSelectionAuthority &&) = delete;

  AuthoritativeLidarGenerationIssueResult issueLidarGeneration(
    std::uint64_t generation,
    const AdaptiveVisualPolicy::Matrix6 &
      lidar_pose_information,
    bool lidar_information_ready);
  AdaptiveVisualSelectionAuthorityIdentity identity() const noexcept;
  std::optional<AdaptiveVisualRuntimeAuthorityInput>
  bindRuntimeInput(
    const AuthoritativeLidarGenerationToken & lidar_token);
  AdaptiveVisualSelectionIssueResult issue(
    const AdaptiveVisualPolicy & policy,
    const CameraObservationBatch & batch,
    const AdaptiveVisualCandidatePlanCapability & policy_capability,
    const AuthoritativeLidarGenerationToken & lidar_token);
  AdaptiveVisualSelectionConsumeResult consume(
    const AdaptiveVisualPolicy & policy,
    const CameraObservationBatch & batch,
    const AdaptiveVisualSelectionCapability & capability,
    const AuthoritativeLidarGenerationToken & lidar_token);

private:
  explicit AdaptiveVisualSelectionAuthority(
    CameraObservationBatchAuthorityIdentity batch_authority);
  AdaptiveVisualSelectionStatus preflightRuntimeBatch(
    const CameraObservationBatch & batch) const;
  AdaptiveVisualRuntimeAttemptResult beginRuntimeAttempt(
    const AuthoritativeLidarGenerationToken & lidar_token);
  AdaptiveVisualSelectionStatus validateRuntimeAttemptPermit(
    const AuthoritativeLidarGenerationToken & lidar_token,
    std::uint64_t attempt_nonce);

  CameraObservationBatchAuthorityIdentity batch_authority_;
  std::shared_ptr<const std::uint8_t> owner_;
  std::weak_ptr<AdaptiveVisualSelectionAuthority> self_;
  std::mutex transaction_mutex_;
  std::uint64_t latest_lidar_generation_{0U};
  std::uint64_t last_consumed_batch_sequence_{0U};
  std::uint64_t next_runtime_attempt_nonce_{1U};

  friend class ::CameraObservationQueue;
  friend class AdaptiveVisualPolicy;
  friend class AdaptiveVisualRuntimeAuthorityInput;
  friend class AdaptiveVisualRuntimeAttemptPermit;
};

}  // namespace livo

#endif  // LIVO_ADAPTIVE_VISUAL_SELECTION_AUTHORITY_H_
