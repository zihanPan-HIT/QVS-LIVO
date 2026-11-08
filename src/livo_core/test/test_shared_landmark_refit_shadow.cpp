#include "shared_landmark_refit_shadow.h"

#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace
{

livo::PlaneRefitEvent event(
  const livo::PlaneIdentity identity)
{
  livo::PlaneRefitEvent value;
  value.expected_geometry_version =
    livo::GeometryVersion{identity, 1U};
  value.replacement_geometry_version =
    livo::GeometryVersion{identity, 2U};
  value.generation = 2U;
  value.center_world_m = Eigen::Vector3d::Zero();
  value.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  value.normal_world = Eigen::Vector3d::UnitZ();
  value.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  return value;
}

TEST(
  SharedLandmarkRefitShadow,
  DisabledAndMissingLidarAuthorityNeverWrite)
{
  livo::MultiViewLandmarkStore store;
  const std::vector<livo::PlaneRefitEvent> events{
    event(livo::PlaneIdentity{501U, 2U})};

  livo::SharedLandmarkRefitShadow disabled;
  const auto disabled_result = disabled.apply(
    store, events, nullptr, livo::BatchId{1U});
  EXPECT_EQ(
    disabled_result.status,
    livo::SharedLandmarkRefitShadowStatus::kDisabled);
  EXPECT_FALSE(disabled_result.acknowledge_events);
  EXPECT_EQ(disabled_result.geometry_refits, 0U);

  livo::SharedLandmarkRefitShadowConfig config;
  config.enabled = true;
  const livo::SharedLandmarkRefitShadow enabled(config);
  const auto unauthorized = enabled.apply(
    store, events, nullptr, livo::BatchId{2U});
  EXPECT_EQ(
    unauthorized.status,
    livo::SharedLandmarkRefitShadowStatus::
      kMissingLidarToken);
  EXPECT_FALSE(unauthorized.acknowledge_events);
  EXPECT_EQ(unauthorized.geometry_refits, 0U);
  EXPECT_EQ(
    store.snapshot(
      livo::GeometrySnapshotId{1U},
      livo::CommitRevisionSet{}).
    store_revision.value,
    0U);
}

TEST(
  PlaneRefitEventQueue,
  InvalidCovarianceIsRejectedBeforeReceiptCreation)
{
  const livo::PlaneIdentity identity{502U, 1U};
  livo::PlaneRefitEventQueue queue;

  livo::PlaneRefitEvent negative_position = event(identity);
  negative_position.position_covariance_m2 =
    -Eigen::Matrix3d::Identity();
  EXPECT_FALSE(queue.record(negative_position));

  livo::PlaneRefitEvent asymmetric_position = event(identity);
  asymmetric_position.position_covariance_m2(0, 1) = 0.25;
  EXPECT_FALSE(queue.record(asymmetric_position));

  livo::PlaneRefitEvent nonfinite_position = event(identity);
  nonfinite_position.position_covariance_m2(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(queue.record(nonfinite_position));

  livo::PlaneRefitEvent negative_plane = event(identity);
  negative_plane.plane_covariance =
    -Eigen::Matrix<double, 6, 6>::Identity();
  EXPECT_FALSE(queue.record(negative_plane));

  livo::PlaneRefitEvent asymmetric_plane = event(identity);
  asymmetric_plane.plane_covariance(0, 1) = 0.25;
  EXPECT_FALSE(queue.record(asymmetric_plane));

  EXPECT_EQ(queue.pendingEventCount(), 0U);
  EXPECT_FALSE(queue.snapshot().valid());

  EXPECT_TRUE(queue.record(event(identity)));
  EXPECT_EQ(queue.pendingEventCount(), 1U);
  EXPECT_TRUE(queue.snapshot().valid());
}

}  // namespace
