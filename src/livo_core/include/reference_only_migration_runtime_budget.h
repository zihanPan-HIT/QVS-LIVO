#ifndef LIVO_REFERENCE_ONLY_MIGRATION_RUNTIME_BUDGET_H_
#define LIVO_REFERENCE_ONLY_MIGRATION_RUNTIME_BUDGET_H_

#include "cross_camera_patch_migration.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace livo
{

inline constexpr std::size_t
  kReferenceOnlyMigrationRuntimePairCount = 12U;
inline constexpr std::size_t
  kReferenceOnlyMigrationRuntimePerPairMaterializationBudget = 4U;
inline constexpr std::size_t
  kReferenceOnlyMigrationRuntimeMaterializationBudget =
  kReferenceOnlyMigrationRuntimePairCount *
  kReferenceOnlyMigrationRuntimePerPairMaterializationBudget;

struct ReferenceOnlyMigrationLightweightDescriptor
{
  std::size_t payload_index{0U};
  std::uint8_t source_camera_id{0U};
  std::uint8_t target_camera_id{0U};
  GeometrySupportKey support_key;
  GeometryVersion geometry_version;
  Eigen::Vector2d source_pixel{Eigen::Vector2d::Zero()};
  Eigen::Vector3d plane_point_world_m{Eigen::Vector3d::Zero()};
  // Exact-value fingerprint for inputs which are intentionally omitted from
  // this lightweight descriptor. It is used only to quarantine order ties;
  // it is not an authority or acceptance token.
  std::array<std::uint8_t, 32U> evaluator_fingerprint{};
};

struct ReferenceOnlyMigrationRuntimeBudgetPlan
{
  std::size_t input_descriptor_count{0U};
  std::size_t canonical_descriptor_count{0U};
  std::size_t duplicate_descriptor_count{0U};
  std::size_t ambiguous_duplicate_count{0U};
  std::size_t pair_budget_truncated_count{0U};
  std::vector<ReferenceOnlyMigrationLightweightDescriptor>
    retained_descriptors;
};

ReferenceOnlyMigrationRuntimeBudgetPlan
planReferenceOnlyMigrationRuntimeBudget(
  std::vector<ReferenceOnlyMigrationLightweightDescriptor>
    descriptors);

std::array<std::uint8_t, 32U>
referenceOnlyMigrationEvaluatorFingerprint(
  const GeometrySupportKey & support_key,
  const GeometryVersion & geometry_version,
  const MigrationPlane & plane);

}  // namespace livo

#endif  // LIVO_REFERENCE_ONLY_MIGRATION_RUNTIME_BUDGET_H_
