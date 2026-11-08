#include "plane_lifecycle.h"
#include "voxel_map.h"
#include "checked_increment.h"

#include <atomic>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

namespace
{

livo::PlaneGeometryEstimate estimate(
  const Eigen::Vector3d & center,
  const Eigen::Vector3d & normal,
  const Eigen::Vector3d & principal_axis,
  const double planarity)
{
  livo::PlaneGeometryEstimate value;
  value.stable = true;
  value.center_world_m = center;
  value.normal_world = normal;
  value.principal_axis_world = principal_axis;
  value.planarity = planarity;
  return value;
}

std::vector<pointWithVar> planePoints(const double x_offset)
{
  std::vector<pointWithVar> points;
  for (int x = -2; x <= 2; ++x) {
    for (int y = -1; y <= 1; ++y) {
      pointWithVar point;
      point.point_w = Eigen::Vector3d(
        x_offset + 0.20 * static_cast<double>(x),
        0.10 * static_cast<double>(y),
        0.0);
      point.var = 1e-4 * Eigen::Matrix3d::Identity();
      points.push_back(point);
    }
  }
  return points;
}

TEST(PlaneLifecycle, FirstStableAndMinorRefitKeepIdentityAndBasis)
{
  livo::PlaneReplacementGateConfig config;
  config.maximum_normal_change_rad = 0.20;
  config.maximum_center_shift_m = 0.30;
  config.maximum_planarity_change = 0.05;

  const livo::PlaneLifecycleUpdate initialized =
    livo::advancePlaneLifecycle(
      livo::PlaneLifecycleSnapshot{},
      estimate(
        Eigen::Vector3d(1.0, 2.0, 3.0),
        Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 0.01),
      71U, 10U, config);

  ASSERT_EQ(
    initialized.transition,
    livo::PlaneLifecycleTransition::kInitialized);
  EXPECT_TRUE(initialized.snapshot.stable);
  EXPECT_EQ(initialized.snapshot.identity.id, 71U);
  EXPECT_EQ(initialized.snapshot.identity.epoch, 1U);
  EXPECT_EQ(initialized.snapshot.revision, 1U);
  EXPECT_FALSE(initialized.tombstone.has_value());
  const Eigen::Vector3d frozen_origin =
    initialized.snapshot.identity_origin_world_m;
  const Eigen::Vector3d frozen_u =
    initialized.snapshot.identity_u_axis_world;
  const Eigen::Vector3d frozen_v =
    initialized.snapshot.identity_v_axis_world;

  const livo::PlaneLifecycleUpdate minor =
    livo::advancePlaneLifecycle(
      initialized.snapshot,
      estimate(
        Eigen::Vector3d(1.02, 2.0, 3.0),
        Eigen::Vector3d(0.01, 0.0, 1.0).normalized(),
        Eigen::Vector3d::UnitY(), 0.02),
      999U, 11U, config);

  EXPECT_EQ(
    minor.transition,
    livo::PlaneLifecycleTransition::kMinorRefit);
  EXPECT_EQ(minor.snapshot.identity.id, 71U);
  EXPECT_EQ(minor.snapshot.identity.epoch, 1U);
  EXPECT_EQ(minor.snapshot.revision, 2U);
  EXPECT_TRUE(
    minor.snapshot.identity_origin_world_m.isApprox(
      frozen_origin));
  EXPECT_TRUE(
    minor.snapshot.identity_u_axis_world.isApprox(frozen_u));
  EXPECT_TRUE(
    minor.snapshot.identity_v_axis_world.isApprox(frozen_v));
  EXPECT_FALSE(minor.tombstone.has_value());
}

TEST(PlaneLifecycle, InvalidReplacementThresholdsFailFastWhenEnabled)
{
  livo::PlaneReplacementGateConfig valid;
  EXPECT_TRUE(valid.valid());

  VoxelMapConfig voxel_config{};
  voxel_config.plane_lifecycle_shadow_enable_ = true;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;

  voxel_config.plane_replacement_gate_ = valid;
  voxel_config.plane_replacement_gate_.
    maximum_normal_change_rad =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
    VoxelMapManager manager(voxel_config, voxel_map),
    std::invalid_argument);

  voxel_config.plane_replacement_gate_ = valid;
  voxel_config.plane_replacement_gate_.
    maximum_center_shift_m = -0.1;
  EXPECT_THROW(
    VoxelMapManager manager(voxel_config, voxel_map),
    std::invalid_argument);

  voxel_config.plane_replacement_gate_ = valid;
  voxel_config.plane_replacement_gate_.
    maximum_planarity_change = -0.1;
  EXPECT_THROW(
    VoxelMapManager manager(voxel_config, voxel_map),
    std::invalid_argument);

  voxel_config.plane_replacement_gate_ = valid;
  voxel_config.plane_replacement_gate_.
    maximum_normal_change_rad = M_PI + 0.01;
  EXPECT_THROW(
    VoxelMapManager manager(voxel_config, voxel_map),
    std::invalid_argument);
}

TEST(
  PlaneLifecycle,
  VoxelManagerRegistryTracksMajorReplacementAndDeletion)
{
  VoxelMapConfig voxel_config{};
  voxel_config.plane_lifecycle_shadow_enable_ = true;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  VoxelMapManager manager(voxel_config, voxel_map);
  auto * octree = new VoxelOctoTree();
  octree->plane_ptr_ = new VoxelPlane();
  octree->plane_ptr_->is_plane_ = true;
  octree->plane_ptr_->is_init_ = true;
  octree->plane_ptr_->geometry_version_ =
    livo::GeometryVersion{
    livo::PlaneIdentity{700U, 1U}, 4U};
  octree->plane_ptr_->geometry_generation_ = 10U;
  for (VoxelOctoTree *& child : octree->leaves_)
  {
    child = nullptr;
  }
  manager.voxel_map_.emplace(
    VOXEL_LOCATION{0, 0, 0}, octree);

  manager.UpdateVoxelMap({});
  const auto old_snapshot =
    manager.snapshotActivePlaneIdentities();
  const auto old_lease = old_snapshot.leaseFor(
    octree->plane_ptr_->geometry_version_);
  ASSERT_TRUE(old_lease.has_value());
  ASSERT_TRUE(old_lease->valid());

  octree->plane_ptr_->geometry_version_ =
    livo::GeometryVersion{
    livo::PlaneIdentity{700U, 2U}, 1U};
  octree->plane_ptr_->geometry_generation_ = 11U;
  manager.UpdateVoxelMap({});
  EXPECT_FALSE(old_snapshot.valid());
  EXPECT_FALSE(old_lease->valid());
  const auto replacement =
    manager.snapshotActivePlaneIdentities();
  ASSERT_EQ(replacement.size(), 1U);
  EXPECT_TRUE(replacement.leaseFor(
    octree->plane_ptr_->geometry_version_).has_value());

  manager.clearMemOutOfMap(
    -1, -2, -1, -2, -1, -2);
  EXPECT_EQ(
    manager.snapshotActivePlaneIdentities().size(), 0U);
}

TEST(
  PlaneLifecycle,
  NearMaximumEpochRevisionAndZeroGenerationFailClosed)
{
  livo::PlaneReplacementGateConfig config;
  livo::PlaneLifecycleSnapshot previous;
  previous.stable = true;
  previous.identity =
    livo::PlaneIdentity{
    710U, std::numeric_limits<std::uint32_t>::max()};
  previous.revision = 4U;
  previous.center_world_m = Eigen::Vector3d::Zero();
  previous.normal_world = Eigen::Vector3d::UnitZ();
  previous.planarity = 0.01;

  const auto major = livo::advancePlaneLifecycle(
    previous,
    estimate(
      Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d::UnitZ(),
      Eigen::Vector3d::UnitX(), 0.01),
    999U, 100U, config);
  EXPECT_EQ(
    major.transition,
    livo::PlaneLifecycleTransition::kNoChange);
  EXPECT_EQ(
    major.snapshot.identity.epoch,
    std::numeric_limits<std::uint32_t>::max());
  EXPECT_FALSE(major.tombstone.has_value());

  previous.identity.epoch = 1U;
  previous.revision =
    std::numeric_limits<std::uint64_t>::max();
  const auto minor = livo::advancePlaneLifecycle(
    previous,
    estimate(
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::UnitZ(),
      Eigen::Vector3d::UnitX(), 0.01),
    999U, 101U, config);
  EXPECT_EQ(
    minor.transition,
    livo::PlaneLifecycleTransition::kNoChange);
  EXPECT_EQ(
    minor.snapshot.revision,
    std::numeric_limits<std::uint64_t>::max());

  const auto zero_generation = livo::advancePlaneLifecycle(
    livo::PlaneLifecycleSnapshot{},
    estimate(
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::UnitZ(),
      Eigen::Vector3d::UnitX(), 0.01),
    711U, 0U, config);
  EXPECT_EQ(
    zero_generation.transition,
    livo::PlaneLifecycleTransition::kNoChange);
  EXPECT_FALSE(zero_generation.snapshot.stable);
}

TEST(
  PlaneLifecycle,
  NearMaximumVoxelIdentityAndGenerationCountersFailClosed)
{
  const auto maximum =
    std::numeric_limits<std::uint64_t>::max();
  std::atomic<std::uint64_t> next_plane_id{maximum};
  std::uint64_t reserved_plane_id = 0U;
  EXPECT_FALSE(livo::checkedReserve(
    next_plane_id, reserved_plane_id));
  EXPECT_EQ(next_plane_id.load(), maximum);
  EXPECT_EQ(reserved_plane_id, 0U);

  std::uint64_t geometry_generation = maximum;
  EXPECT_FALSE(
    livo::checkedIncrementNonzero(geometry_generation));
  EXPECT_EQ(geometry_generation, maximum);
}

TEST(PlaneLifecycle, MajorReplacementGateCreatesNewEpochAndOldIdentityTombstone)
{
  livo::PlaneReplacementGateConfig config;
  config.maximum_normal_change_rad = 0.20;
  config.maximum_center_shift_m = 0.30;
  config.maximum_planarity_change = 0.05;

  const livo::PlaneLifecycleUpdate initialized =
    livo::advancePlaneLifecycle(
      livo::PlaneLifecycleSnapshot{},
      estimate(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 0.01),
      71U, 10U, config);

  const std::array<livo::PlaneGeometryEstimate, 3U> replacements{
    estimate(
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.35, 0.0, 1.0).normalized(),
      Eigen::Vector3d::UnitY(), 0.01),
    estimate(
      Eigen::Vector3d(0.31, 0.0, 0.0),
      Eigen::Vector3d::UnitZ(),
      Eigen::Vector3d::UnitY(), 0.01),
    estimate(
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d::UnitZ(),
      Eigen::Vector3d::UnitY(), 0.061)};

  for (const auto & replacement : replacements) {
    const livo::PlaneLifecycleUpdate major =
      livo::advancePlaneLifecycle(
        initialized.snapshot, replacement, 999U, 42U, config);

    EXPECT_EQ(
      major.transition,
      livo::PlaneLifecycleTransition::kMajorReplacement);
    EXPECT_EQ(major.snapshot.identity.id, 71U);
    EXPECT_EQ(major.snapshot.identity.epoch, 2U);
    EXPECT_EQ(major.snapshot.revision, 1U);
    EXPECT_TRUE(
      major.snapshot.identity_origin_world_m.isApprox(
        replacement.center_world_m));
    EXPECT_FALSE(
      major.snapshot.identity_u_axis_world.isApprox(
        initialized.snapshot.identity_u_axis_world));
    ASSERT_TRUE(major.tombstone.has_value());
    EXPECT_EQ(major.tombstone->old_identity.id, 71U);
    EXPECT_EQ(major.tombstone->old_identity.epoch, 1U);
    EXPECT_EQ(major.tombstone->last_revision, 1U);
    EXPECT_EQ(major.tombstone->generation, 42U);
    EXPECT_EQ(
      major.tombstone->reason,
      livo::PlaneTombstoneReason::kMajorReplacement);
  }
}

TEST(PlaneLifecycle, NormalSignFlipIsAMinorRefit)
{
  livo::PlaneReplacementGateConfig config;
  config.maximum_normal_change_rad = 0.20;
  config.maximum_center_shift_m = 0.30;
  config.maximum_planarity_change = 0.05;

  const livo::PlaneLifecycleUpdate initialized =
    livo::advancePlaneLifecycle(
      livo::PlaneLifecycleSnapshot{},
      estimate(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 0.01),
      71U, 10U, config);
  const livo::PlaneLifecycleUpdate sign_flipped =
    livo::advancePlaneLifecycle(
      initialized.snapshot,
      estimate(
        Eigen::Vector3d::Zero(),
        -Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 0.01),
      999U, 11U, config);

  EXPECT_EQ(
    sign_flipped.transition,
    livo::PlaneLifecycleTransition::kMinorRefit);
  EXPECT_EQ(sign_flipped.snapshot.identity.epoch, 1U);
  EXPECT_FALSE(sign_flipped.tombstone.has_value());
}

TEST(PlaneLifecycle, InvalidationEmitsOnceAndRecoveryAdvancesEpoch)
{
  livo::PlaneReplacementGateConfig config;
  config.maximum_normal_change_rad = 0.20;
  config.maximum_center_shift_m = 0.30;
  config.maximum_planarity_change = 0.05;

  const livo::PlaneLifecycleUpdate initialized =
    livo::advancePlaneLifecycle(
      livo::PlaneLifecycleSnapshot{},
      estimate(
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d::UnitZ(),
        Eigen::Vector3d::UnitX(), 0.01),
      71U, 10U, config);

  const livo::PlaneLifecycleUpdate invalidated =
    livo::advancePlaneLifecycle(
      initialized.snapshot, livo::PlaneGeometryEstimate{},
      999U, 42U, config);
  EXPECT_EQ(
    invalidated.transition,
    livo::PlaneLifecycleTransition::kInvalidated);
  EXPECT_FALSE(invalidated.snapshot.stable);
  ASSERT_TRUE(invalidated.tombstone.has_value());
  EXPECT_EQ(
    invalidated.tombstone->reason,
    livo::PlaneTombstoneReason::kPlaneInvalidated);
  EXPECT_EQ(invalidated.tombstone->old_identity.id, 71U);
  EXPECT_EQ(invalidated.tombstone->old_identity.epoch, 1U);
  EXPECT_EQ(invalidated.tombstone->last_revision, 1U);

  const livo::PlaneLifecycleUpdate still_invalid =
    livo::advancePlaneLifecycle(
      invalidated.snapshot, livo::PlaneGeometryEstimate{},
      999U, 43U, config);
  EXPECT_EQ(
    still_invalid.transition,
    livo::PlaneLifecycleTransition::kNoChange);
  EXPECT_FALSE(still_invalid.tombstone.has_value());

  const livo::PlaneLifecycleUpdate recovered =
    livo::advancePlaneLifecycle(
      still_invalid.snapshot,
      estimate(
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d::UnitY(),
        Eigen::Vector3d::UnitZ(), 0.01),
      999U, 44U, config);
  EXPECT_TRUE(recovered.snapshot.stable);
  EXPECT_EQ(recovered.snapshot.identity.id, 71U);
  EXPECT_EQ(recovered.snapshot.identity.epoch, 2U);
  EXPECT_EQ(recovered.snapshot.revision, 1U);
  EXPECT_FALSE(recovered.tombstone.has_value());
}

TEST(PlaneTombstoneEventQueue, DeduplicatesAndSupportsSnapshotAndConsume)
{
  livo::PlaneTombstoneEventQueue queue;
  livo::PlaneTombstoneEvent deletion;
  deletion.old_identity.id = 71U;
  deletion.old_identity.epoch = 1U;
  deletion.last_revision = 4U;
  deletion.generation = 42U;
  deletion.reason = livo::PlaneTombstoneReason::kVoxelDeleted;

  EXPECT_TRUE(queue.record(deletion));
  livo::PlaneTombstoneEvent duplicate = deletion;
  duplicate.generation = 43U;
  duplicate.reason = livo::PlaneTombstoneReason::kPlaneInvalidated;
  EXPECT_FALSE(queue.record(duplicate));

  const std::vector<livo::PlaneTombstoneEvent> first_snapshot =
    queue.snapshot();
  ASSERT_EQ(first_snapshot.size(), 1U);
  EXPECT_EQ(first_snapshot.front().generation, 42U);
  EXPECT_EQ(
    first_snapshot.front().reason,
    livo::PlaneTombstoneReason::kVoxelDeleted);
  EXPECT_EQ(queue.snapshot().size(), 1U);

  const std::vector<livo::PlaneTombstoneEvent> consumed =
    queue.consume();
  ASSERT_EQ(consumed.size(), 1U);
  EXPECT_TRUE(queue.snapshot().empty());
  EXPECT_FALSE(queue.record(deletion));

  livo::PlaneTombstoneEvent next_epoch = deletion;
  next_epoch.old_identity.epoch = 2U;
  EXPECT_TRUE(queue.record(next_epoch));
  ASSERT_EQ(queue.consume().size(), 1U);
}

TEST(PlaneTombstoneEventQueue, WatermarkIsBoundedByLivePlanes)
{
  livo::PlaneTombstoneEventQueue queue;
  for (std::uint32_t epoch = 1U; epoch <= 5000U; ++epoch) {
    livo::PlaneTombstoneEvent replacement;
    replacement.old_identity.id = 71U;
    replacement.old_identity.epoch = epoch;
    replacement.last_revision = 4U;
    replacement.generation = epoch;
    replacement.reason =
      livo::PlaneTombstoneReason::kMajorReplacement;
    ASSERT_TRUE(queue.record(replacement));
    ASSERT_EQ(queue.consume().size(), 1U);
  }
  EXPECT_EQ(queue.deduplicationWatermarkCount(), 1U);

  for (std::uint64_t plane_id = 100U;
    plane_id < 1100U; ++plane_id)
  {
    livo::PlaneTombstoneEvent deletion;
    deletion.old_identity.id = plane_id;
    deletion.old_identity.epoch = 1U;
    deletion.last_revision = 4U;
    deletion.generation = plane_id;
    deletion.reason = livo::PlaneTombstoneReason::kVoxelDeleted;
    ASSERT_TRUE(queue.record(deletion));
    queue.retirePlane(plane_id);
  }
  EXPECT_EQ(queue.deduplicationWatermarkCount(), 1U);
  EXPECT_EQ(queue.consume().size(), 1000U);
}

TEST(
  PlaneRefitEventQueue,
  SlowConsumerCoalescesFiveThousandAndPreservesNewerEvent)
{
  livo::PlaneRefitEventQueue queue;
  const livo::PlaneIdentity identity{72U, 1U};
  auto event = [&](
      const std::uint64_t old_revision,
      const std::uint64_t new_revision) {
      livo::PlaneRefitEvent value;
      value.expected_geometry_version =
        livo::GeometryVersion{identity, old_revision};
      value.replacement_geometry_version =
        livo::GeometryVersion{identity, new_revision};
      value.generation = new_revision;
      value.center_world_m =
        Eigen::Vector3d(
        0.001 * static_cast<double>(new_revision),
        0.0, 0.0);
      value.normal_world = Eigen::Vector3d::UnitZ();
      value.position_covariance_m2 =
        0.01 * Eigen::Matrix3d::Identity();
      value.plane_covariance =
        0.001 *
        Eigen::Matrix<double, 6, 6>::Identity();
      return value;
    };

  for (std::uint64_t revision = 1U;
       revision <= 5000U; ++revision)
  {
    EXPECT_TRUE(queue.record(event(revision, revision + 1U)));
  }
  EXPECT_EQ(queue.pendingEventCount(), 1U);
  const auto slow_snapshot = queue.snapshot();
  ASSERT_TRUE(slow_snapshot.valid());
  ASSERT_EQ(slow_snapshot.events.size(), 1U);
  EXPECT_EQ(
    slow_snapshot.events.front().
    expected_geometry_version.revision,
    1U);
  EXPECT_EQ(
    slow_snapshot.events.front().
    replacement_geometry_version.revision,
    5001U);

  ASSERT_TRUE(queue.record(event(5001U, 5002U)));
  livo::PlaneRefitEventBatch wrong_delivery = slow_snapshot;
  ++wrong_delivery.delivery_token;
  EXPECT_FALSE(queue.acknowledge(wrong_delivery));
  EXPECT_TRUE(queue.acknowledge(slow_snapshot));
  const auto newer = queue.snapshot();
  ASSERT_TRUE(newer.valid());
  ASSERT_NE(
    newer.delivery_token,
    slow_snapshot.delivery_token);
  ASSERT_EQ(newer.events.size(), 1U);
  EXPECT_EQ(
    newer.events.front().
    expected_geometry_version.revision,
    5001U);
  EXPECT_EQ(
    newer.events.front().
    replacement_geometry_version.revision,
    5002U);

  EXPECT_TRUE(queue.acknowledge(newer));
  EXPECT_FALSE(queue.snapshot().valid());
  EXPECT_EQ(queue.pendingEventCount(), 0U);
}

TEST(
  PlaneRefitEventQueue,
  ReceiptRejectsHeaderOrPayloadTamperingWithoutLosingPendingEvent)
{
  livo::PlaneRefitEventQueue queue;
  const livo::PlaneIdentity identity{73U, 1U};
  livo::PlaneRefitEvent refit;
  refit.expected_geometry_version =
    livo::GeometryVersion{identity, 1U};
  refit.replacement_geometry_version =
    livo::GeometryVersion{identity, 2U};
  refit.generation = 9U;
  refit.center_world_m = Eigen::Vector3d(1.0, 2.0, 3.0);
  refit.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  refit.normal_world = Eigen::Vector3d::UnitZ();
  refit.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  ASSERT_TRUE(queue.record(refit));

  const livo::PlaneRefitEventBatch receipt = queue.snapshot();
  ASSERT_TRUE(receipt.valid());
  EXPECT_GT(receipt.queue_generation, 0U);
  EXPECT_EQ(receipt.event_count, 1U);

  livo::PlaneRefitEventBatch wrong_generation = receipt;
  ++wrong_generation.queue_generation;
  EXPECT_FALSE(queue.acknowledge(wrong_generation));

  livo::PlaneRefitEventBatch wrong_count = receipt;
  ++wrong_count.event_count;
  EXPECT_FALSE(queue.acknowledge(wrong_count));

  livo::PlaneRefitEventBatch wrong_digest = receipt;
  ++wrong_digest.payload_digest.front();
  EXPECT_FALSE(queue.acknowledge(wrong_digest));

  livo::PlaneRefitEventBatch changed_payload = receipt;
  changed_payload.events.front().center_world_m.x() += 0.25;
  EXPECT_FALSE(queue.acknowledge(changed_payload));

  EXPECT_EQ(queue.pendingEventCount(), 1U);
  EXPECT_EQ(queue.snapshot().delivery_token, receipt.delivery_token);
  EXPECT_TRUE(queue.acknowledge(receipt));
  EXPECT_EQ(queue.pendingEventCount(), 0U);
}

TEST(
  PlaneRefitEventQueue,
  RetiringOneIdentityInvalidatesItsReceiptAndPreservesLaterEpoch)
{
  livo::PlaneRefitEventQueue queue;
  auto refit = [](
      const livo::PlaneIdentity identity,
      const std::uint64_t generation) {
      livo::PlaneRefitEvent value;
      value.expected_geometry_version =
        livo::GeometryVersion{identity, 1U};
      value.replacement_geometry_version =
        livo::GeometryVersion{identity, 2U};
      value.generation = generation;
      value.center_world_m =
        Eigen::Vector3d(
        static_cast<double>(identity.epoch), 0.0, 0.0);
      value.position_covariance_m2 =
        0.01 * Eigen::Matrix3d::Identity();
      value.normal_world = Eigen::Vector3d::UnitZ();
      value.plane_covariance =
        0.001 *
        Eigen::Matrix<double, 6, 6>::Identity();
      return value;
    };

  const livo::PlaneIdentity retired_identity{74U, 1U};
  const livo::PlaneIdentity later_epoch{74U, 2U};
  const livo::PlaneIdentity other_identity{75U, 1U};
  ASSERT_TRUE(queue.record(refit(retired_identity, 11U)));
  ASSERT_TRUE(queue.record(refit(later_epoch, 12U)));
  ASSERT_TRUE(queue.record(refit(other_identity, 13U)));
  const livo::PlaneRefitEventBatch old_receipt =
    queue.snapshot();
  ASSERT_EQ(old_receipt.events.size(), 3U);

  queue.retirePlane(retired_identity);
  EXPECT_FALSE(queue.acknowledge(old_receipt));
  EXPECT_EQ(queue.pendingEventCount(), 2U);

  const livo::PlaneRefitEventBatch replacement_receipt =
    queue.snapshot();
  ASSERT_TRUE(replacement_receipt.valid());
  EXPECT_NE(
    replacement_receipt.delivery_token,
    old_receipt.delivery_token);
  EXPECT_GT(
    replacement_receipt.queue_generation,
    old_receipt.queue_generation);
  ASSERT_EQ(replacement_receipt.events.size(), 2U);
  EXPECT_EQ(
    replacement_receipt.events.front().
    expected_geometry_version.plane,
    later_epoch);
  EXPECT_EQ(
    replacement_receipt.events.back().
    expected_geometry_version.plane,
    other_identity);
  EXPECT_TRUE(queue.acknowledge(replacement_receipt));
  EXPECT_FALSE(queue.snapshot().valid());
}

TEST(VoxelPlaneLifecycle, RefitUsesEpochPolicyAndPublishesMajorTombstone)
{
  livo::PlaneReplacementGateConfig config;
  config.maximum_normal_change_rad = 0.20;
  config.maximum_center_shift_m = 0.10;
  config.maximum_planarity_change = 0.05;
  livo::PlaneTombstoneEventQueue queue;
  livo::PlaneRefitEventQueue refit_queue;
  std::uint64_t generation = 10U;
  VoxelOctoTree tree(
    0, 0, 3, 100, 0.01F, config, &queue, &generation,
    &refit_queue);

  tree.init_plane(planePoints(0.0), tree.plane_ptr_);
  ASSERT_TRUE(tree.plane_ptr_->is_plane_);
  const std::uint64_t plane_id =
    tree.plane_ptr_->geometry_version_.plane.id;
  EXPECT_GT(plane_id, 0);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.plane.epoch, 1U);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.revision, 1U);
  EXPECT_FALSE(refit_queue.snapshot().valid());

  generation = 11U;
  tree.init_plane(planePoints(0.02), tree.plane_ptr_);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.plane.id, plane_id);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.plane.epoch, 1U);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.revision, 2U);
  EXPECT_TRUE(queue.snapshot().empty());
  const auto minor_refits = refit_queue.snapshot();
  ASSERT_TRUE(minor_refits.valid());
  ASSERT_EQ(minor_refits.events.size(), 1U);
  EXPECT_EQ(
    minor_refits.events.front().
    expected_geometry_version.revision,
    1U);
  EXPECT_EQ(
    minor_refits.events.front().
    replacement_geometry_version.revision,
    2U);

  generation = 12U;
  tree.init_plane(planePoints(0.50), tree.plane_ptr_);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.plane.id, plane_id);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.plane.epoch, 2U);
  EXPECT_EQ(tree.plane_ptr_->geometry_version_.revision, 1U);
  const std::vector<livo::PlaneTombstoneEvent> events =
    queue.snapshot();
  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(
    events.front().old_identity.id,
    plane_id);
  EXPECT_EQ(events.front().old_identity.epoch, 1U);
  EXPECT_EQ(events.front().last_revision, 2U);
  EXPECT_EQ(events.front().generation, 12U);
  EXPECT_FALSE(refit_queue.acknowledge(minor_refits));
  EXPECT_FALSE(refit_queue.snapshot().valid());
}

TEST(VoxelPlaneLifecycle, RecursiveVoxelDeletionEmitsEachIdentityOnce)
{
  livo::PlaneTombstoneEventQueue queue;
  VoxelOctoTree root(1, 0, 3, 100, 0.01F);
  root.plane_ptr_->is_init_ = true;
  root.plane_ptr_->is_plane_ = true;
  root.plane_ptr_->geometry_version_.plane =
    livo::PlaneIdentity{71U, 1U};
  root.plane_ptr_->geometry_version_.revision = 4U;
  root.leaves_[0] = new VoxelOctoTree(1, 1, 3, 100, 0.01F);
  root.leaves_[0]->plane_ptr_->is_init_ = true;
  root.leaves_[0]->plane_ptr_->is_plane_ = true;
  root.leaves_[0]->plane_ptr_->geometry_version_.plane =
    livo::PlaneIdentity{72U, 2U};
  root.leaves_[0]->plane_ptr_->geometry_version_.revision = 7U;

  root.emitTombstonesForRemoval(
    queue, 55U, livo::PlaneTombstoneReason::kVoxelDeleted);
  root.emitTombstonesForRemoval(
    queue, 56U, livo::PlaneTombstoneReason::kVoxelDeleted);

  const std::vector<livo::PlaneTombstoneEvent> events =
    queue.snapshot();
  ASSERT_EQ(events.size(), 2U);
  EXPECT_EQ(events[0].old_identity.id, 71U);
  EXPECT_EQ(events[0].last_revision, 4U);
  EXPECT_EQ(events[1].old_identity.id, 72U);
  EXPECT_EQ(events[1].old_identity.epoch, 2U);
  EXPECT_EQ(events[1].last_revision, 7U);
}

VoxelOctoTree * identifiedTree(const std::uint64_t plane_id)
{
  VoxelOctoTree * tree =
    new VoxelOctoTree(0, 0, 3, 100, 0.01F);
  tree->plane_ptr_->is_init_ = true;
  tree->plane_ptr_->is_plane_ = true;
  tree->plane_ptr_->geometry_version_.plane =
    livo::PlaneIdentity{plane_id, 1U};
  tree->plane_ptr_->geometry_version_.revision = 4U;
  return tree;
}

TEST(VoxelMapManagerTombstones, RemovalIsDefaultOff)
{
  VoxelMapManager manager;
  manager.config_setting_.plane_lifecycle_shadow_enable_ = false;
  manager.voxel_map_[VOXEL_LOCATION(20, 0, 0)] =
    identifiedTree(71);

  manager.clearMemOutOfMap(1, -1, 1, -1, 1, -1);

  EXPECT_TRUE(manager.snapshotPlaneTombstoneEvents().empty());
}

TEST(VoxelMapManagerTombstones, RemovalEmitsBeforeDeleteWhenEnabled)
{
  VoxelMapManager manager;
  manager.config_setting_.plane_lifecycle_shadow_enable_ = true;
  manager.voxel_map_[VOXEL_LOCATION(20, 0, 0)] =
    identifiedTree(71);

  manager.clearMemOutOfMap(1, -1, 1, -1, 1, -1);

  const std::vector<livo::PlaneTombstoneEvent> events =
    manager.snapshotPlaneTombstoneEvents();
  ASSERT_EQ(events.size(), 1U);
  EXPECT_EQ(events.front().old_identity.id, 71U);
  EXPECT_EQ(
    events.front().reason,
    livo::PlaneTombstoneReason::kVoxelDeleted);
}

TEST(VoxelMapManagerTombstones, DisabledConsumerPreservesPendingEvents)
{
  VoxelMapManager manager;
  manager.config_setting_.plane_lifecycle_shadow_enable_ = true;
  manager.voxel_map_[VOXEL_LOCATION(20, 0, 0)] =
    identifiedTree(71);
  manager.clearMemOutOfMap(1, -1, 1, -1, 1, -1);
  ASSERT_EQ(manager.snapshotPlaneTombstoneEvents().size(), 1U);

  EXPECT_TRUE(manager.consumePlaneTombstoneEvents(false).empty());
  EXPECT_EQ(manager.snapshotPlaneTombstoneEvents().size(), 1U);
  EXPECT_EQ(manager.consumePlaneTombstoneEvents(true).size(), 1U);
  EXPECT_TRUE(manager.snapshotPlaneTombstoneEvents().empty());
}

}  // namespace
