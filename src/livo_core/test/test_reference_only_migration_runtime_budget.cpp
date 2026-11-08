#include "reference_only_migration_runtime_budget.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace
{

livo::ReferenceOnlyMigrationLightweightDescriptor descriptor(
  const std::size_t payload_index,
  const std::uint8_t source_camera_id,
  const std::uint8_t target_camera_id,
  const std::uint64_t support_id)
{
  livo::ReferenceOnlyMigrationLightweightDescriptor value;
  value.payload_index = payload_index;
  value.source_camera_id = source_camera_id;
  value.target_camera_id = target_camera_id;
  value.support_key.plane = {support_id, 1U};
  value.support_key.cell = {
    static_cast<std::int32_t>(support_id), 0, 0U};
  value.geometry_version = {
    {support_id, 1U}, 3U};
  value.source_pixel =
    Eigen::Vector2d(
    100.0 + static_cast<double>(support_id),
    120.0 + static_cast<double>(support_id));
  value.plane_point_world_m =
    Eigen::Vector3d(
    0.01 * static_cast<double>(support_id),
    0.0, 5.0);
  value.evaluator_fingerprint[0U] =
    static_cast<std::uint8_t>(support_id);
  return value;
}

std::vector<std::uint64_t> retainedSupportIds(
  const livo::ReferenceOnlyMigrationRuntimeBudgetPlan & plan)
{
  std::vector<std::uint64_t> ids;
  for (const auto & selected : plan.retained_descriptors)
  {
    ids.push_back(selected.support_key.plane.id);
  }
  return ids;
}

TEST(
  ReferenceOnlyMigrationRuntimeBudget,
  FiveHundredTwelveInputsNeverMaterializeMoreThanFortyEight)
{
  std::vector<
    livo::ReferenceOnlyMigrationLightweightDescriptor> input;
  input.reserve(512U);
  for (std::size_t index = 0U; index < 512U; ++index)
  {
    const std::uint8_t source =
      static_cast<std::uint8_t>(index % 4U);
    const std::uint8_t target =
      static_cast<std::uint8_t>(
      (source + 1U + (index / 4U) % 3U) % 4U);
    input.push_back(descriptor(
        index, source, target,
        static_cast<std::uint64_t>(index + 1U)));
  }

  const auto plan =
    livo::planReferenceOnlyMigrationRuntimeBudget(
    std::move(input));

  EXPECT_EQ(plan.input_descriptor_count, 512U);
  EXPECT_LE(
    plan.retained_descriptors.size(),
    livo::kReferenceOnlyMigrationRuntimeMaterializationBudget);
  EXPECT_EQ(plan.retained_descriptors.size(), 48U);
  EXPECT_EQ(
    plan.pair_budget_truncated_count,
    plan.canonical_descriptor_count - 48U);
}

TEST(
  ReferenceOnlyMigrationRuntimeBudget,
  SelectionIsPermutationInvariantAndExactDuplicatesCollapse)
{
  std::vector<
    livo::ReferenceOnlyMigrationLightweightDescriptor> input;
  for (std::size_t index = 0U; index < 80U; ++index)
  {
    input.push_back(descriptor(
        index, 0U, 1U,
        static_cast<std::uint64_t>(index + 1U)));
  }
  input.push_back(input[7U]);
  const auto baseline =
    livo::planReferenceOnlyMigrationRuntimeBudget(input);

  std::mt19937 generator(123456U);
  for (int iteration = 0; iteration < 32; ++iteration)
  {
    std::shuffle(input.begin(), input.end(), generator);
    const auto permuted =
      livo::planReferenceOnlyMigrationRuntimeBudget(input);
    EXPECT_EQ(
      retainedSupportIds(permuted),
      retainedSupportIds(baseline));
    EXPECT_EQ(permuted.duplicate_descriptor_count, 1U);
    EXPECT_EQ(permuted.ambiguous_duplicate_count, 0U);
    EXPECT_EQ(permuted.retained_descriptors.size(), 4U);
  }
}

TEST(
  ReferenceOnlyMigrationRuntimeBudget,
  ConflictingPublicOrderTieIsQuarantined)
{
  auto first = descriptor(1U, 2U, 3U, 9U);
  auto conflicting = first;
  conflicting.payload_index = 2U;
  conflicting.evaluator_fingerprint[3U] = 1U;

  const auto plan =
    livo::planReferenceOnlyMigrationRuntimeBudget(
    {first, conflicting});

  EXPECT_EQ(plan.input_descriptor_count, 2U);
  EXPECT_EQ(plan.duplicate_descriptor_count, 1U);
  EXPECT_EQ(plan.ambiguous_duplicate_count, 1U);
  EXPECT_EQ(plan.canonical_descriptor_count, 0U);
  EXPECT_TRUE(plan.retained_descriptors.empty());
}

}  // namespace
