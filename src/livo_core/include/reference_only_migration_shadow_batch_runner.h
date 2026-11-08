#ifndef LIVO_REFERENCE_ONLY_MIGRATION_SHADOW_BATCH_RUNNER_H_
#define LIVO_REFERENCE_ONLY_MIGRATION_SHADOW_BATCH_RUNNER_H_

#include "reference_only_migration_shadow.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace livo
{

inline constexpr std::size_t
  kReferenceOnlyMigrationShadowDirectedPairCount = 12U;
inline constexpr std::size_t
  kReferenceOnlyMigrationShadowPerPairBudget = 4U;
inline constexpr std::size_t
  kReferenceOnlyMigrationShadowTotalBudget = 32U;
inline constexpr std::size_t
  kReferenceOnlyMigrationShadowStatusCount =
  static_cast<std::size_t>(MigrationStatus::kNonFinite) + 1U;

enum class ReferenceOnlyMigrationShadowBatchReason
{
  kEmptyInput,
  kNoEligibleDirectedPair,
  kEvaluated
};

class ReferenceOnlyMigrationShadowBatchCandidate
{
public:
  explicit ReferenceOnlyMigrationShadowBatchCandidate(
    ReferenceOnlyMigrationShadowRequest request)
  : request_(std::move(request))
  {
  }

  const ReferenceOnlyMigrationShadowRequest & request() const noexcept
  {
    return request_;
  }

  std::uint8_t sourceCameraId() const noexcept
  {
    return request_.source_endpoint.camera_id;
  }

  std::uint8_t targetCameraId() const noexcept
  {
    return request_.target_endpoint.camera_id;
  }

  const GeometrySupportKey & supportKey() const noexcept
  {
    return request_.source_support_key;
  }

  std::uint64_t geometryRevision() const noexcept
  {
    return request_.source_geometry_version.revision;
  }

  const Eigen::Vector2d & sourcePixel() const noexcept
  {
    return request_.patch.source_pixel;
  }

  const Eigen::Vector3d & planePointWorld() const noexcept
  {
    return request_.plane.point_world_m;
  }

private:
  ReferenceOnlyMigrationShadowRequest request_;
};

struct ReferenceOnlyMigrationShadowBatchSelection
{
  std::uint8_t source_camera_id{0U};
  std::uint8_t target_camera_id{0U};
  GeometrySupportKey support_key;
  GeometryVersion geometry_version;
  Eigen::Vector2d source_pixel{Eigen::Vector2d::Zero()};
  Eigen::Vector3d plane_point_world_m{Eigen::Vector3d::Zero()};
  MigrationStatus migration_status{MigrationStatus::kNonFinite};
};

class ReferenceOnlyMigrationShadowBatchResult;

ReferenceOnlyMigrationShadowBatchResult
runReferenceOnlyMigrationShadowBatch(
  std::uint64_t lidar_feature_generation,
  std::vector<ReferenceOnlyMigrationShadowBatchCandidate> candidates);

class ReferenceOnlyMigrationShadowBatchResult
{
public:
  ReferenceOnlyMigrationShadowBatchResult(
    const ReferenceOnlyMigrationShadowBatchResult &) = default;
  ReferenceOnlyMigrationShadowBatchResult(
    ReferenceOnlyMigrationShadowBatchResult &&) = default;
  ReferenceOnlyMigrationShadowBatchResult & operator=(
    const ReferenceOnlyMigrationShadowBatchResult &) = default;
  ReferenceOnlyMigrationShadowBatchResult & operator=(
    ReferenceOnlyMigrationShadowBatchResult &&) = default;
  ~ReferenceOnlyMigrationShadowBatchResult() = default;

  ReferenceOnlyMigrationShadowBatchReason reason() const noexcept
  {
    return reason_;
  }

  std::size_t inputCandidateCount() const noexcept
  {
    return input_candidate_count_;
  }

  std::size_t attemptedCount() const noexcept
  {
    return attempted_count_;
  }

  std::size_t uniqueCandidateCount() const noexcept
  {
    return unique_candidate_count_;
  }

  std::size_t duplicateCandidateCount() const noexcept
  {
    return duplicate_candidate_count_;
  }

  std::size_t ambiguousDuplicateCount() const noexcept
  {
    return ambiguous_duplicate_count_;
  }

  std::size_t retainedCandidateCount() const noexcept
  {
    return retained_candidate_count_;
  }

  std::size_t pairBudgetTruncatedCount() const noexcept
  {
    return pair_budget_truncated_count_;
  }

  std::size_t totalBudgetTruncatedCount() const noexcept
  {
    return total_budget_truncated_count_;
  }

  std::size_t readyCount() const noexcept
  {
    return ready_count_;
  }

  std::size_t rejectedCount() const noexcept
  {
    return attempted_count_ - ready_count_;
  }

  const std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowDirectedPairCount> &
  pairAttemptCounts() const noexcept
  {
    return pair_attempt_counts_;
  }

  const std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowDirectedPairCount> &
  pairReadyCounts() const noexcept
  {
    return pair_ready_counts_;
  }

  std::size_t statusCount(
    const MigrationStatus status) const noexcept
  {
    const std::size_t index =
      static_cast<std::size_t>(status);
    return
      index < migration_status_counts_.size() ?
      migration_status_counts_[index] : 0U;
  }

  const std::vector<ReferenceOnlyMigrationShadowBatchSelection> &
  selectedSequence() const noexcept
  {
    return selected_sequence_;
  }

  const std::optional<double> &
  readyWarpConditionP50() const noexcept
  {
    return ready_warp_condition_p50_;
  }

  const std::optional<double> &
  readyWarpConditionP95() const noexcept
  {
    return ready_warp_condition_p95_;
  }

  static constexpr bool experimental() noexcept
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

  static constexpr bool acceptanceReceiptIssued() noexcept
  {
    return false;
  }

  static constexpr bool independenceProofIssued() noexcept
  {
    return false;
  }

private:
  ReferenceOnlyMigrationShadowBatchResult() = default;

  ReferenceOnlyMigrationShadowBatchReason reason_{
    ReferenceOnlyMigrationShadowBatchReason::kEmptyInput};
  std::size_t input_candidate_count_{0U};
  std::size_t unique_candidate_count_{0U};
  std::size_t duplicate_candidate_count_{0U};
  std::size_t ambiguous_duplicate_count_{0U};
  std::size_t retained_candidate_count_{0U};
  std::size_t pair_budget_truncated_count_{0U};
  std::size_t total_budget_truncated_count_{0U};
  std::size_t attempted_count_{0U};
  std::size_t ready_count_{0U};
  std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowDirectedPairCount>
    pair_attempt_counts_{};
  std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowDirectedPairCount>
    pair_ready_counts_{};
  std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowStatusCount>
    migration_status_counts_{};
  std::vector<ReferenceOnlyMigrationShadowBatchSelection>
    selected_sequence_;
  std::optional<double> ready_warp_condition_p50_;
  std::optional<double> ready_warp_condition_p95_;

  friend ReferenceOnlyMigrationShadowBatchResult
  runReferenceOnlyMigrationShadowBatch(
    std::uint64_t lidar_feature_generation,
    std::vector<
      ReferenceOnlyMigrationShadowBatchCandidate> candidates);
};

}  // namespace livo

#endif  // LIVO_REFERENCE_ONLY_MIGRATION_SHADOW_BATCH_RUNNER_H_
