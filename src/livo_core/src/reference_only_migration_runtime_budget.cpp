#include "reference_only_migration_runtime_budget.h"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

namespace livo
{
namespace
{

std::optional<std::size_t> directedPairIndex(
  const std::uint8_t source_camera_id,
  const std::uint8_t target_camera_id)
{
  if (
    source_camera_id >= 4U ||
    target_camera_id >= 4U ||
    source_camera_id == target_camera_id)
  {
    return std::nullopt;
  }
  const std::size_t target_offset =
    target_camera_id < source_camera_id ?
    static_cast<std::size_t>(target_camera_id) :
    static_cast<std::size_t>(target_camera_id - 1U);
  return
    static_cast<std::size_t>(source_camera_id) * 3U +
    target_offset;
}

std::uint64_t normalizedDoubleOrderKey(const double value)
{
  if (std::isnan(value)) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  const double normalized = value == 0.0 ? 0.0 : value;
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(normalized));
  std::memcpy(&bits, &normalized, sizeof(bits));
  constexpr std::uint64_t kSign =
    std::uint64_t{1U} << 63U;
  return
    (bits & kSign) != 0U ?
    ~bits : bits | kSign;
}

bool descriptorLess(
  const ReferenceOnlyMigrationLightweightDescriptor & left,
  const ReferenceOnlyMigrationLightweightDescriptor & right)
{
  if (left.support_key < right.support_key) {
    return true;
  }
  if (right.support_key < left.support_key) {
    return false;
  }
  if (
    left.geometry_version.revision !=
    right.geometry_version.revision)
  {
    return
      left.geometry_version.revision >
      right.geometry_version.revision;
  }
  const std::array<std::uint64_t, 5U> left_values{{
      normalizedDoubleOrderKey(left.source_pixel.y()),
      normalizedDoubleOrderKey(left.source_pixel.x()),
      normalizedDoubleOrderKey(left.plane_point_world_m.x()),
      normalizedDoubleOrderKey(left.plane_point_world_m.y()),
      normalizedDoubleOrderKey(left.plane_point_world_m.z())}};
  const std::array<std::uint64_t, 5U> right_values{{
      normalizedDoubleOrderKey(right.source_pixel.y()),
      normalizedDoubleOrderKey(right.source_pixel.x()),
      normalizedDoubleOrderKey(right.plane_point_world_m.x()),
      normalizedDoubleOrderKey(right.plane_point_world_m.y()),
      normalizedDoubleOrderKey(right.plane_point_world_m.z())}};
  return left_values < right_values;
}

bool orderEquivalent(
  const ReferenceOnlyMigrationLightweightDescriptor & left,
  const ReferenceOnlyMigrationLightweightDescriptor & right)
{
  return
    !descriptorLess(left, right) &&
    !descriptorLess(right, left);
}

}  // namespace

std::array<std::uint8_t, 32U>
referenceOnlyMigrationEvaluatorFingerprint(
  const GeometrySupportKey & support_key,
  const GeometryVersion & geometry_version,
  const MigrationPlane & plane)
{
  std::array<std::uint8_t, 32U> fingerprint{};
  EVP_MD_CTX * context = EVP_MD_CTX_new();
  if (context == nullptr) {
    return fingerprint;
  }
  bool valid =
    EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1;
  const auto append =
    [&context, &valid](
      const void * data, const std::size_t size)
    {
      valid =
        valid &&
        EVP_DigestUpdate(context, data, size) == 1;
    };
  append(&support_key.plane.id, sizeof(support_key.plane.id));
  append(
    &support_key.plane.epoch,
    sizeof(support_key.plane.epoch));
  append(&support_key.cell.u, sizeof(support_key.cell.u));
  append(&support_key.cell.v, sizeof(support_key.cell.v));
  append(&support_key.cell.level, sizeof(support_key.cell.level));
  append(
    &geometry_version.plane.id,
    sizeof(geometry_version.plane.id));
  append(
    &geometry_version.plane.epoch,
    sizeof(geometry_version.plane.epoch));
  append(
    &geometry_version.revision,
    sizeof(geometry_version.revision));
  append(
    plane.point_world_m.data(),
    sizeof(double) * 3U);
  append(
    plane.point_covariance_m2.data(),
    sizeof(double) * 9U);
  append(
    plane.normal_world.data(),
    sizeof(double) * 3U);
  append(
    plane.normal_covariance.data(),
    sizeof(double) * 9U);
  unsigned int length = 0U;
  valid =
    valid &&
    EVP_DigestFinal_ex(
      context, fingerprint.data(), &length) == 1 &&
    length == fingerprint.size();
  EVP_MD_CTX_free(context);
  if (!valid) {
    fingerprint.fill(0U);
  }
  return fingerprint;
}

ReferenceOnlyMigrationRuntimeBudgetPlan
planReferenceOnlyMigrationRuntimeBudget(
  std::vector<ReferenceOnlyMigrationLightweightDescriptor>
    descriptors)
{
  ReferenceOnlyMigrationRuntimeBudgetPlan plan;
  plan.input_descriptor_count = descriptors.size();
  std::array<
    std::vector<ReferenceOnlyMigrationLightweightDescriptor>,
    kReferenceOnlyMigrationRuntimePairCount> buckets;
  for (auto & descriptor : descriptors)
  {
    const auto pair_index = directedPairIndex(
      descriptor.source_camera_id,
      descriptor.target_camera_id);
    if (pair_index.has_value())
    {
      buckets[*pair_index].push_back(
        std::move(descriptor));
    }
  }

  plan.retained_descriptors.reserve(
    kReferenceOnlyMigrationRuntimeMaterializationBudget);
  for (auto & bucket : buckets)
  {
    std::sort(bucket.begin(), bucket.end(), descriptorLess);
    std::vector<ReferenceOnlyMigrationLightweightDescriptor>
      canonical;
    canonical.reserve(bucket.size());
    std::size_t support_begin = 0U;
    while (support_begin < bucket.size())
    {
      std::size_t support_end = support_begin + 1U;
      while (
        support_end < bucket.size() &&
        bucket[support_begin].support_key ==
          bucket[support_end].support_key)
      {
        ++support_end;
      }
      plan.duplicate_descriptor_count +=
        support_end - support_begin - 1U;
      bool ambiguous = false;
      for (std::size_t index = support_begin + 1U;
           index < support_end; ++index)
      {
        if (
          orderEquivalent(
            bucket[index - 1U], bucket[index]) &&
          bucket[index - 1U].evaluator_fingerprint !=
            bucket[index].evaluator_fingerprint)
        {
          ambiguous = true;
          break;
        }
      }
      if (ambiguous)
      {
        ++plan.ambiguous_duplicate_count;
      }
      else
      {
        canonical.push_back(bucket[support_begin]);
      }
      support_begin = support_end;
    }

    plan.canonical_descriptor_count += canonical.size();
    const std::size_t retained = std::min(
      canonical.size(),
      kReferenceOnlyMigrationRuntimePerPairMaterializationBudget);
    plan.pair_budget_truncated_count +=
      canonical.size() - retained;
    plan.retained_descriptors.insert(
      plan.retained_descriptors.end(),
      canonical.begin(),
      canonical.begin() +
      static_cast<std::ptrdiff_t>(retained));
  }
  return plan;
}

}  // namespace livo
