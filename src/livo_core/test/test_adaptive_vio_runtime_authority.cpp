#include "adaptive_visual_selection_authority.h"
#include "camera_model_factory.h"
#include "vio.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{

static_assert(
  !std::is_default_constructible<
    livo::AdaptiveVisualRuntimeAuthorityInput>::value,
  "Runtime authority input must not be caller-constructible");
static_assert(
  !std::is_constructible<
    livo::AdaptiveVisualRuntimeAuthorityInput,
    livo::AdaptiveVisualSelectionAuthority &,
    livo::AuthoritativeLidarGenerationToken>::value,
  "Runtime authority input must not permit mixed-owner assembly");

livo::CameraModelParameters pinholeParameters(
  const std::uint8_t camera_id)
{
  livo::CameraModelParameters parameters;
  parameters.model = "Pinhole";
  parameters.width = 640;
  parameters.height = 480;
  parameters.scale = 1.0;
  parameters.fx = 400.0 + static_cast<double>(camera_id);
  parameters.fy = 400.0 + static_cast<double>(camera_id);
  parameters.cx = 320.0;
  parameters.cy = 240.0;
  return parameters;
}

cv::Mat texturedImage()
{
  cv::Mat image(480, 640, CV_8UC1);
  for (int row = 0; row < image.rows; ++row)
  {
    auto * pixels = image.ptr<std::uint8_t>(row);
    for (int column = 0; column < image.cols; ++column)
    {
      pixels[column] =
        ((row / 8 + column / 8) % 2 == 0) ? 32U : 224U;
    }
  }
  return image;
}

cv::Mat weakTexturedImage()
{
  cv::Mat image(480, 640, CV_8UC1);
  for (int row = 0; row < image.rows; ++row)
  {
    auto * pixels = image.ptr<std::uint8_t>(row);
    for (int column = 0; column < image.cols; ++column)
    {
      pixels[column] =
        ((row / 8 + column / 8) % 2 == 0) ? 88U : 112U;
    }
  }
  return image;
}

std::vector<pointWithVar> lidarPoints(
  const std::size_t maximum_points = 35U)
{
  std::vector<pointWithVar> points;
  points.reserve(35U);
  for (int row = 0; row < 5; ++row)
  {
    for (int column = 0; column < 7; ++column)
    {
      pointWithVar point;
      point.point_w = Eigen::Vector3d(
        0.45 * static_cast<double>(column) - 1.35,
        0.45 * static_cast<double>(row) - 0.90,
        5.0);
      point.var = 1.0e-4 * Eigen::Matrix3d::Identity();
      point.normal = Eigen::Vector3d::UnitZ();
      points.push_back(std::move(point));
    }
  }
  if (maximum_points < points.size())
  {
    std::vector<pointWithVar> selected;
    selected.reserve(maximum_points);
    for (std::size_t index = 0U; index < maximum_points; ++index)
    {
      const std::size_t source_index = std::min(
        static_cast<std::size_t>(
          (static_cast<double>(index) + 0.5) *
          static_cast<double>(points.size()) /
          static_cast<double>(maximum_points)),
        points.size() - 1U);
      selected.push_back(points[source_index]);
    }
    points = std::move(selected);
  }
  return points;
}

void initializeManager(
  VIOManager & manager, StatesGroup & state,
  StatesGroup & propagated,
  const bool adaptive_enabled = true,
  const bool single_camera_runtime = true,
  const bool configure_camera_two = true,
  const int pyramid_levels = 1)
{
  constexpr std::array<const char *, 4U> kNames{{
    "front", "left", "back", "right"}};
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    if (!configure_camera_two && camera_id == 2U)
    {
      continue;
    }
    manager.registerCamera(
      camera_id, kNames[camera_id],
      livo::createCameraModel(pinholeParameters(camera_id)),
      M3D::Identity(), V3D::Zero());
  }

  state.cov.setIdentity();
  propagated = state;
  manager.grid_size = 20;
  manager.patch_size = 8;
  manager.outlier_threshold = 400;
  manager.state = &state;
  manager.state_propagat = &propagated;
  manager.max_iterations = 1;
  manager.img_point_cov = 100.0;
  manager.joint_img_point_cov = 1000.0;
  manager.normal_en = true;
  manager.inverse_composition_en = false;
  manager.raycast_en = false;
  manager.grid_n_width = 32;
  manager.grid_n_height = 24;
  manager.patch_pyrimid_level = pyramid_levels;
  manager.exposure_estimate_en = false;
  manager.colmap_output_en = false;
  manager.plot_flag = false;
  manager.adaptive_policy_config.enabled = adaptive_enabled;
  if (adaptive_enabled && single_camera_runtime)
  {
    manager.adaptive_policy_config.minimum_selected_cameras = 1U;
    manager.adaptive_policy_config.maximum_selected_cameras = 1U;
    manager.joint_update_config.min_joint_cameras = 1U;
  }
  manager.initializeVIO();
}

void seedCameraMap(
  VIOManager & manager, const std::uint8_t camera_id,
  const cv::Mat & image,
  const Eigen::Vector3d & translation = Eigen::Vector3d::Zero(),
  const std::size_t maximum_points = 35U)
{
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = 10.0;
  observation.image = image.clone();
  observation.photometric_status =
    livo::PhotometricStatus::kValid;
  auto points = lidarPoints(maximum_points);
  for (pointWithVar & point : points)
  {
    point.point_w += translation;
  }
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;
  manager.processFrame(
    observation, points, feature_map, observation.timestamp);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  SealedSingleCameraInitializationPreservesConfiguredBaseTrackingFloor)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  manager.adaptive_policy_config.minimum_selection_tracked_points = 1U;
  manager.adaptive_policy_config.minimum_inliers = 13U;
  manager.joint_update_config.min_tracked_points_per_camera = 4U;
  manager.joint_update_config.min_joint_tracked_points = 12U;

  initializeManager(manager, state, propagated);

  EXPECT_EQ(
    manager.adaptive_policy_config.minimum_selection_tracked_points,
    1U);
  EXPECT_EQ(manager.adaptive_policy_config.minimum_inliers, 13U);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  NonSealedInitializationKeepsLegacyJointTrackingFloors)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  manager.adaptive_policy_config.minimum_selection_tracked_points = 1U;
  manager.adaptive_policy_config.minimum_inliers = 13U;
  manager.joint_update_config.min_tracked_points_per_camera = 4U;
  manager.joint_update_config.min_joint_tracked_points = 12U;

  initializeManager(manager, state, propagated, true, false);

  EXPECT_EQ(
    manager.adaptive_policy_config.minimum_selection_tracked_points,
    4U);
  EXPECT_EQ(
    manager.joint_update_config.min_tracked_points_per_camera,
    4U);
  EXPECT_EQ(manager.joint_update_config.min_joint_tracked_points, 12U);
  EXPECT_EQ(manager.adaptive_policy_config.minimum_inliers, 13U);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  NonFrontScoutPopLeavesFrontForIndependentConsumption)
{
  CameraObservationQueue queue;
  ASSERT_TRUE(queue.setBatchAuthoritySealingEnabled(true));

  auto make_observation = [](
      const std::uint8_t camera_id, const double timestamp)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = timestamp;
      observation.image = texturedImage();
      observation.photometric_status =
        livo::PhotometricStatus::kValid;
      return observation;
    };
  ASSERT_EQ(
    queue.push(make_observation(0U, 11.000)),
    CameraObservationPushResult::ACCEPTED);
  ASSERT_EQ(
    queue.push(make_observation(2U, 11.004)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch scout;
  EXPECT_EQ(
    queue.popAdaptiveNonFrontSynchronized(
      11.100, 0.010, 0.030, 1U, scout),
    AdaptiveCameraBatchResult::READY);
  ASSERT_EQ(scout.camera_count, 1U);
  EXPECT_FALSE(scout.present[0U]);
  EXPECT_TRUE(scout.present[2U]);
  EXPECT_TRUE(queue.hasCameraObservation(0U));
  ASSERT_NE(queue.nextReadyForCamera(0U, 11.100), nullptr);

  CameraObservation front;
  ASSERT_TRUE(queue.popReadyForCamera(0U, 11.100, front));
  EXPECT_EQ(front.camera_id, 0U);
  EXPECT_FALSE(queue.hasCameraObservation(0U));
  EXPECT_FALSE(queue.nextReadyForCamera(0U, 11.100));

  // The scout watermark is not a global watermark: a later Front frame is
  // accepted and can be consumed independently.
  ASSERT_EQ(
    queue.push(make_observation(0U, 11.100)),
    CameraObservationPushResult::ACCEPTED);
  EXPECT_TRUE(queue.hasCameraObservation(0U));
}

CameraObservationBatch sealTwoCandidates(
  CameraObservationQueue & queue,
  const bool later_camera_arrives_first)
{
  if (!queue.setBatchAuthoritySealingEnabled(true))
  {
    throw std::runtime_error("failed to enable batch sealing");
  }
  auto make_observation =
    [](const std::uint8_t camera_id, const double timestamp,
       cv::Mat image)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = timestamp;
      observation.image = std::move(image);
      observation.photometric_status =
        livo::PhotometricStatus::kValid;
      return observation;
    };
  CameraObservation earlier =
    make_observation(
    0U, 11.000,
    cv::Mat(480, 640, CV_8UC1, cv::Scalar(96U)));
  CameraObservation later =
    make_observation(2U, 11.008, texturedImage());
  if (later_camera_arrives_first)
  {
    EXPECT_EQ(
      queue.push(std::move(later)),
      CameraObservationPushResult::ACCEPTED);
    EXPECT_EQ(
      queue.push(std::move(earlier)),
      CameraObservationPushResult::ACCEPTED);
  }
  else
  {
    EXPECT_EQ(
      queue.push(std::move(earlier)),
      CameraObservationPushResult::ACCEPTED);
    EXPECT_EQ(
      queue.push(std::move(later)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      11.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

CameraObservationBatch sealFourCandidates(
  CameraObservationQueue & queue,
  const bool only_camera_three_healthy = true,
  const bool camera_zero_overexposed = false)
{
  if (!queue.setBatchAuthoritySealingEnabled(true))
  {
    throw std::runtime_error("failed to enable batch sealing");
  }
  constexpr std::array<std::uint8_t, 4U>
    kPushOrder{{3U, 1U, 2U, 0U}};
  constexpr std::array<double, 4U>
    kTimestamps{{12.000, 12.003, 12.006, 12.009}};
  for (const std::uint8_t camera_id : kPushOrder)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp = kTimestamps[camera_id];
    if (camera_zero_overexposed && camera_id == 0U)
    {
      observation.image =
        cv::Mat(480, 640, CV_8UC1, cv::Scalar(255U));
    }
    else
    {
      observation.image =
        !only_camera_three_healthy || camera_id == 3U ?
        texturedImage() :
        cv::Mat(480, 640, CV_8UC1, cv::Scalar(96U));
    }
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      12.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

CameraObservationBatch sealFourHealthyCandidatesAt(
  CameraObservationQueue & queue, const double timestamp,
  const std::array<std::uint8_t, 4U> & push_order)
{
  if (!queue.setBatchAuthoritySealingEnabled(true))
  {
    throw std::runtime_error("failed to enable batch sealing");
  }
  for (const std::uint8_t camera_id : push_order)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      timestamp + 0.002 * static_cast<double>(camera_id);
    observation.image = texturedImage();
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

CameraObservationBatch sealThreeCandidates(
  CameraObservationQueue & queue,
  const bool all_candidates_healthy = false,
  const double first_timestamp_sec = 13.0)
{
  if (!queue.setBatchAuthoritySealingEnabled(true))
  {
    throw std::runtime_error("failed to enable batch sealing");
  }
  constexpr std::array<std::uint8_t, 3U>
    kPushOrder{{3U, 0U, 1U}};
  constexpr std::array<double, 4U>
    kTimestampOffsets{{0.000, 0.004, 0.0, 0.008}};
  for (const std::uint8_t camera_id : kPushOrder)
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      first_timestamp_sec +
      kTimestampOffsets[camera_id];
    observation.image =
      all_candidates_healthy || camera_id == 1U ?
      texturedImage() :
      cv::Mat(480, 640, CV_8UC1, cv::Scalar(96U));
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      first_timestamp_sec + 0.100,
      0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

CameraObservationBatch sealTwoCandidateImages(
  CameraObservationQueue & queue, const double timestamp,
  const cv::Mat & camera_zero_image,
  const cv::Mat & camera_two_image)
{
  if (!queue.setBatchAuthoritySealingEnabled(true))
  {
    throw std::runtime_error("failed to enable batch sealing");
  }
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 2U>{{0U, 2U}})
  {
    CameraObservation observation;
    observation.camera_id = camera_id;
    observation.timestamp =
      timestamp + (camera_id == 0U ? 0.0 : 0.008);
    observation.image =
      (camera_id == 0U ? camera_zero_image :
       camera_two_image).clone();
    observation.photometric_status =
      livo::PhotometricStatus::kValid;
    EXPECT_EQ(
      queue.push(std::move(observation)),
      CameraObservationPushResult::ACCEPTED);
  }

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

CameraObservationBatch sealSingleCandidate(
  CameraObservationQueue & queue, const double timestamp,
  const std::uint8_t camera_id, const cv::Mat & image)
{
  if (!queue.setBatchAuthoritySealingEnabled(true))
  {
    throw std::runtime_error("failed to enable batch sealing");
  }
  CameraObservation observation;
  observation.camera_id = camera_id;
  observation.timestamp = timestamp;
  observation.image = image.clone();
  observation.photometric_status =
    livo::PhotometricStatus::kValid;
  EXPECT_EQ(
    queue.push(std::move(observation)),
    CameraObservationPushResult::ACCEPTED);

  CameraObservationBatch batch;
  EXPECT_EQ(
    queue.popAdaptiveSynchronized(
      timestamp + 0.100, 0.010, 0.030, 1U, batch),
    AdaptiveCameraBatchResult::READY);
  return batch;
}

void processAuthorizedWithLidar(
  VIOManager & manager, CameraObservationBatch & batch,
  const std::uint64_t lidar_generation,
  const livo::AdaptiveVisualPolicy::Matrix6 &
    lidar_pose_information,
  const bool lidar_information_ready,
  const livo::AdaptiveVisualRuntimeAuthorityInput *
    runtime_input,
  const std::optional<double> image_time_sec =
    std::nullopt)
{
  auto points = lidarPoints();
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;
  manager.processFrames(
    batch, points, feature_map,
    image_time_sec.value_or(batch.timestamp),
    lidar_pose_information, lidar_information_ready,
    lidar_generation, nullptr, nullptr, nullptr,
    runtime_input);
}

void processAuthorized(
  VIOManager & manager, CameraObservationBatch & batch,
  const std::uint64_t lidar_generation,
  const livo::AdaptiveVisualRuntimeAuthorityInput *
    runtime_input)
{
  processAuthorizedWithLidar(
    manager, batch, lidar_generation,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true,
    runtime_input);
}

void expectEstimationIdentity(
  const livo::VIOReadOnlyStructuralAudit & before,
  const livo::VIOReadOnlyStructuralAudit & after,
  const bool expect_policy_revision_identity = true)
{
  ASSERT_TRUE(before.active_state_available);
  ASSERT_TRUE(after.active_state_available);
  EXPECT_EQ(
    std::memcmp(
      before.active_state.data(), after.active_state.data(),
      sizeof(before.active_state)),
    0);
  ASSERT_TRUE(before.propagated_state_available);
  ASSERT_TRUE(after.propagated_state_available);
  EXPECT_EQ(
    std::memcmp(
      before.propagated_state.data(),
      after.propagated_state.data(),
      sizeof(before.propagated_state)),
    0);
  EXPECT_EQ(
    before.shared_landmark_shadow.geometry.value,
    after.shared_landmark_shadow.geometry.value);
  EXPECT_EQ(
    before.shared_landmark_shadow.store_revision,
    after.shared_landmark_shadow.store_revision);
  EXPECT_EQ(
    before.shared_landmark_shadow.landmark_count,
    after.shared_landmark_shadow.landmark_count);
  EXPECT_EQ(
    before.shared_landmark_shadow.observation_count,
    after.shared_landmark_shadow.observation_count);
  EXPECT_EQ(
    before.legacy_feature_voxel_counts,
    after.legacy_feature_voxel_counts);
  EXPECT_EQ(
    before.legacy_feature_point_counts,
    after.legacy_feature_point_counts);
  EXPECT_EQ(
    before.last_joint_solver.accepted,
    after.last_joint_solver.accepted);
  EXPECT_EQ(
    before.last_joint_solver.reason,
    after.last_joint_solver.reason);
  EXPECT_EQ(
    before.last_joint_solver.contributing_cameras,
    after.last_joint_solver.contributing_cameras);
  EXPECT_EQ(
    before.last_joint_solver.tracked_points,
    after.last_joint_solver.tracked_points);
  EXPECT_EQ(
    before.active_camera_id,
    after.active_camera_id);
  EXPECT_EQ(
    before.adaptive_camera_initialized,
    after.adaptive_camera_initialized);
  EXPECT_EQ(
    before.adaptive_visual_map_lifecycle,
    after.adaptive_visual_map_lifecycle);
  EXPECT_EQ(before.frame_count, after.frame_count);
  if (expect_policy_revision_identity)
  {
    EXPECT_EQ(
      before.adaptive_policy_revision,
      after.adaptive_policy_revision);
  }
}

void expectPersistentIdentity(
  const livo::VIOReadOnlyStructuralAudit & before,
  const livo::VIOReadOnlyStructuralAudit & after)
{
  expectEstimationIdentity(before, after);
  EXPECT_EQ(before.active_camera_id, after.active_camera_id);
  EXPECT_EQ(
    before.adaptive_prepare_call_count,
    after.adaptive_prepare_call_count);
  EXPECT_EQ(
    before.adaptive_scout_call_count,
    after.adaptive_scout_call_count);
  EXPECT_EQ(
    before.adaptive_finalize_call_count,
    after.adaptive_finalize_call_count);
  EXPECT_EQ(
    before.shared_landmark_shadow_entry_count,
    after.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    before.reference_only_migration_shadow_entry_count,
    after.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    before.multi_camera_linearization_shadow_entry_count,
    after.multi_camera_linearization_shadow_entry_count);
}

void expectLioOnlyWithoutSolver(
  const livo::AdaptiveVisualRuntimeAudit & audit)
{
  EXPECT_EQ(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_FALSE(audit.solver_called);
  EXPECT_FALSE(audit.legacy_fallback_used);
  EXPECT_EQ(
    audit.policy_revision_before,
    audit.policy_revision_after);
}

livo::AdaptiveVisualRuntimeAudit runGeneration(
  VIOManager & manager,
  livo::AdaptiveVisualSelectionAuthority & authority,
  CameraObservationBatch & batch,
  const std::uint64_t lidar_generation,
  const std::optional<double> image_time_sec =
    std::nullopt)
{
  const auto lidar_issue = authority.issueLidarGeneration(
    lidar_generation,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  EXPECT_EQ(
    lidar_issue.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(lidar_issue.token.has_value());
  if (!lidar_issue.token.has_value())
  {
    return {};
  }
  const auto runtime_input =
    authority.bindRuntimeInput(*lidar_issue.token);
  EXPECT_TRUE(runtime_input.has_value());
  if (!runtime_input.has_value())
  {
    return {};
  }
  processAuthorizedWithLidar(
    manager, batch, lidar_generation,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true,
    &*runtime_input, image_time_sec);
  return manager.adaptiveVisualRuntimeAudit();
}

livo::AdaptiveVisualRuntimeAudit runGenerationWithPointLimit(
  VIOManager & manager,
  livo::AdaptiveVisualSelectionAuthority & authority,
  CameraObservationBatch & batch,
  const std::uint64_t lidar_generation,
  const std::size_t maximum_points)
{
  const auto lidar_issue = authority.issueLidarGeneration(
    lidar_generation,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  EXPECT_EQ(
    lidar_issue.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(lidar_issue.token.has_value());
  if (!lidar_issue.token.has_value())
  {
    return {};
  }
  const auto runtime_input =
    authority.bindRuntimeInput(*lidar_issue.token);
  EXPECT_TRUE(runtime_input.has_value());
  if (!runtime_input.has_value())
  {
    return {};
  }

  auto points = lidarPoints(maximum_points);
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;
  manager.processFrames(
    batch, points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true,
    lidar_generation, nullptr, nullptr, nullptr,
    &*runtime_input);
  return manager.adaptiveVisualRuntimeAudit();
}

livo::AdaptiveVisualRuntimeAudit runTranslatedGeneration(
  VIOManager & manager,
  livo::AdaptiveVisualSelectionAuthority & authority,
  CameraObservationBatch & batch,
  const std::uint64_t lidar_generation,
  const Eigen::Vector3d & translation,
  const std::size_t maximum_points = 35U)
{
  const auto lidar_issue = authority.issueLidarGeneration(
    lidar_generation,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  EXPECT_EQ(
    lidar_issue.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_TRUE(lidar_issue.token.has_value());
  if (!lidar_issue.token.has_value())
  {
    return {};
  }
  const auto runtime_input =
    authority.bindRuntimeInput(*lidar_issue.token);
  EXPECT_TRUE(runtime_input.has_value());
  if (!runtime_input.has_value())
  {
    return {};
  }

  auto points = lidarPoints(maximum_points);
  for (pointWithVar & point : points)
  {
    point.point_w += translation;
  }
  const std::unordered_map<
    VOXEL_LOCATION, VoxelOctoTree *> feature_map;
  manager.processFrames(
    batch, points, feature_map, batch.timestamp,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true,
    lidar_generation, nullptr, nullptr, nullptr,
    &*runtime_input);
  return manager.adaptiveVisualRuntimeAudit();
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  SeededHealthPreparationPreservesPixelsAndSerialFrameCreation)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    seedCameraMap(manager, camera_id, reference);
  }
  CameraObservationQueue queue;
  auto authority = queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  for (std::uint64_t generation = 1U; generation <= 3U; ++generation)
  {
    auto batch = sealFourHealthyCandidatesAt(
      queue, 20.0 + generation * 0.1, {{3U, 1U, 2U, 0U}});
    if (generation == 2U)
    {
      // Keep the sealed logical pixels while exercising a strided input view.
      for (auto & observation : batch.observations)
      {
        cv::Mat backing(reference.rows + 2, reference.cols + 4, CV_8UC1, cv::Scalar(197U));
        cv::Mat roi = backing(cv::Rect(2, 1, reference.cols, reference.rows));
        reference.copyTo(roi);
        observation.image = roi;
        ASSERT_FALSE(observation.image.isContinuous());
      }
    }
    const int frames_before = Frame::frame_counter_;
    const auto audit = runGeneration(manager, *authority, batch, generation);
    EXPECT_EQ(audit.outcome, livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
    EXPECT_EQ(audit.prepare_call_count, 4U);
    EXPECT_EQ(Frame::frame_counter_, frames_before + 4);
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      EXPECT_EQ(cv::norm(batch.observations[camera_id].image, reference, cv::NORM_INF), 0.0);
    }
  }
}

class FailWorkerImageAllocation final : public cv::MatAllocator
{
public:
  FailWorkerImageAllocation()
  : original_(cv::Mat::getDefaultAllocator()),
    caller_(std::this_thread::get_id()),
    original_threads_(cv::getNumThreads())
  {
    cv::setNumThreads(1);
    cv::Mat::setDefaultAllocator(this);
  }

  ~FailWorkerImageAllocation() override
  {
    cv::Mat::setDefaultAllocator(original_);
    cv::setNumThreads(original_threads_);
  }

  cv::UMatData * allocate(
    int dims, const int * sizes, int type, void * data, std::size_t * step,
    cv::AccessFlag flags, cv::UMatUsageFlags usage) const override
  {
    if (std::this_thread::get_id() != caller_ && !injected_.exchange(true))
    {
      CV_Error(cv::Error::StsNoMem, "health worker allocation failure");
    }
    return original_->allocate(dims, sizes, type, data, step, flags, usage);
  }

  bool allocate(cv::UMatData * data, cv::AccessFlag flags, cv::UMatUsageFlags usage) const override
  {
    return original_->allocate(data, flags, usage);
  }

  void deallocate(cv::UMatData * data) const override { original_->deallocate(data); }
  bool injected() const noexcept { return injected_.load(); }

private:
  cv::MatAllocator * original_;
  std::thread::id caller_;
  int original_threads_;
  mutable std::atomic<bool> injected_{false};
};

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  HealthWorkerFailurePropagatesBeforeSelectionOrStateCommit)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  manager.adaptive_camera_threads = 4;
  const cv::Mat reference = texturedImage();
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    seedCameraMap(manager, camera_id, reference);
  }
  CameraObservationQueue queue;
  auto authority = queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  auto batch = sealFourHealthyCandidatesAt(queue, 21.0, {{3U, 1U, 2U, 0U}});
  cv::Mat backing(reference.rows + 2, reference.cols + 4, CV_8UC1, cv::Scalar(197U));
  cv::Mat roi = backing(cv::Rect(2, 1, reference.cols, reference.rows));
  reference.copyTo(roi);
  batch.observations[0U].image = roi;
  ASSERT_FALSE(batch.observations[0U].image.isContinuous());
  const auto before = manager.readOnlyStructuralAudit();
  bool failed = false;
  {
    FailWorkerImageAllocation allocation;
    try
    {
      (void)runGeneration(manager, *authority, batch, 1U);
    }
    catch (const cv::Exception & error)
    {
      EXPECT_EQ(error.code, cv::Error::StsNoMem);
      failed = true;
    }
    EXPECT_TRUE(allocation.injected());
  }
  EXPECT_TRUE(failed);
  const auto after = manager.readOnlyStructuralAudit();
  EXPECT_EQ(after.active_state, before.active_state);
  EXPECT_EQ(after.adaptive_policy_revision, before.adaptive_policy_revision);
  EXPECT_FALSE(batch.validatedReceipt()->adaptiveVisualSelectionClaimed());
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_EQ(cv::norm(batch.observations[camera_id].image, reference, cv::NORM_INF), 0.0);
  }
  // This failed generation is not retried; production stops the mapper.
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  SealedOneToThreePointCandidatesCommitOnceAcrossCameraIds)
{
  const cv::Mat reference = texturedImage();
  for (std::size_t tracked_points = 1U;
       tracked_points <= 3U; ++tracked_points)
  {
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      SCOPED_TRACE(
        "tracked=" + std::to_string(tracked_points) +
        " camera=" + std::to_string(camera_id));
      VIOManager manager;
      StatesGroup state;
      StatesGroup propagated;
      manager.adaptive_policy_config.
        minimum_selection_tracked_points = 1U;
      manager.adaptive_policy_config.minimum_inliers = 13U;
      manager.adaptive_policy_config.total_candidate_budget =
        tracked_points;
      manager.adaptive_policy_config.minimum_candidate_budget = 1U;
      manager.adaptive_policy_config.total_patch_budget = tracked_points;
      manager.adaptive_policy_config.minimum_patch_budget = 1U;
      manager.joint_update_config.min_tracked_points_per_camera = 4U;
      manager.joint_update_config.min_joint_tracked_points = 12U;
      initializeManager(manager, state, propagated);

      EXPECT_EQ(
        manager.adaptive_policy_config.
        minimum_selection_tracked_points,
        1U);
      EXPECT_EQ(manager.adaptive_policy_config.minimum_inliers, 13U);
      seedCameraMap(manager, camera_id, reference);
      CameraObservationQueue queue;
      auto authority =
        queue.issueAdaptiveVisualSelectionAuthority();
      ASSERT_NE(authority, nullptr);
      const double timestamp =
        11.0 + static_cast<double>(tracked_points) +
        0.1 * static_cast<double>(camera_id);
      CameraObservationBatch batch =
        sealSingleCandidate(
        queue, timestamp, camera_id, reference);
      const auto before = manager.readOnlyStructuralAudit();
      const auto covariance_before = state.cov;

      const auto audit = runGeneration(
        manager, *authority, batch,
        90U + tracked_points * 4U + camera_id);
      const auto after = manager.readOnlyStructuralAudit();

      EXPECT_EQ(
        audit.outcome,
        livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
      EXPECT_EQ(
        std::count(
          audit.selected_mask.begin(), audit.selected_mask.end(), true),
        1);
      ASSERT_TRUE(audit.selected_camera_id.has_value());
      EXPECT_EQ(*audit.selected_camera_id, camera_id);
      ASSERT_TRUE(audit.fusion_camera_id.has_value());
      EXPECT_EQ(*audit.fusion_camera_id, camera_id);
      EXPECT_EQ(
        audit.selection_reason,
        livo::AdaptiveSelectionReason::kSelectedBestHealthy);
      EXPECT_EQ(audit.solver_call_count, 1U);
      EXPECT_EQ(audit.finalize_call_count, 1U);
      ASSERT_TRUE(audit.generation_update_receipt.has_value());
      EXPECT_TRUE(
        audit.generation_update_receipt->visual_update_accepted);
      EXPECT_EQ(
        audit.generation_update_receipt->solver_tracked_points,
        tracked_points);
      EXPECT_EQ(
        audit.generation_update_receipt->solver_contributing_cameras,
        1U);
      ASSERT_TRUE(audit.health_commit_status.has_value());
      EXPECT_EQ(
        *audit.health_commit_status,
        livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
      EXPECT_EQ(
        after.adaptive_policy_revision,
        before.adaptive_policy_revision + 1U);
      EXPECT_EQ(
        after.adaptive_finalize_call_count,
        before.adaptive_finalize_call_count + 1U);
      EXPECT_FALSE(
        state.cov.isApprox(covariance_before, 0.0));
    }
  }
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  FineLevelTrackingCanRecoverAfterSparseCoarseLevel)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(
    manager, state, propagated, true, true, true, 2);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 2U, reference);

  CameraObservationQueue queue;
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  CameraObservationBatch batch =
    sealSingleCandidate(queue, 11.0, 2U, reference);
  const auto audit = runGeneration(
    manager, *authority, batch, 90U);

  EXPECT_TRUE(audit.solver_called);
  ASSERT_TRUE(audit.generation_update_receipt.has_value());
  EXPECT_TRUE(
    audit.generation_update_receipt->visual_update_accepted);
  EXPECT_EQ(
    audit.generation_update_receipt->solver_reason,
    "accepted_dry_run");
}

std::size_t countOccurrences(
  const std::string & text, const std::string & needle)
{
  std::size_t count = 0U;
  std::size_t position = 0U;
  while ((position = text.find(needle, position)) !=
         std::string::npos)
  {
    ++count;
    position += needle.size();
  }
  return count;
}

std::string cameraMetricRow(
  const std::string & output,
  const std::size_t camera_id)
{
  const std::string prefix =
    "[ADAPTIVE_VIO_CAMERA_METRIC]";
  const std::string camera_token =
    " camera_id=" + std::to_string(camera_id) + " ";
  std::size_t row_begin = 0U;
  while (
    (row_begin = output.find(prefix, row_begin)) !=
    std::string::npos)
  {
    const std::size_t row_end =
      output.find('\n', row_begin);
    const std::size_t camera_position =
      output.find(camera_token, row_begin);
    if (
      camera_position != std::string::npos &&
      (row_end == std::string::npos ||
      camera_position < row_end))
    {
      return output.substr(
        row_begin,
        row_end == std::string::npos ?
        std::string::npos : row_end - row_begin);
    }
    if (row_end == std::string::npos)
    {
      break;
    }
    row_begin = row_end + 1U;
  }
  return {};
}

std::string mapMaintenanceMetricRow(
  const std::string & output,
  const std::uint64_t generation,
  const std::size_t camera_id)
{
  const std::string prefix =
    "[ADAPTIVE_VIO_MAP_MAINTENANCE]";
  const std::string generation_token =
    " generation=" + std::to_string(generation) + " ";
  const std::string camera_token =
    " camera_id=" + std::to_string(camera_id) + " ";
  std::size_t row_begin = 0U;
  while (
    (row_begin = output.find(prefix, row_begin)) !=
    std::string::npos)
  {
    const std::size_t row_end =
      output.find('\n', row_begin);
    const bool generation_matches =
      output.find(generation_token, row_begin) !=
        std::string::npos &&
      (row_end == std::string::npos ||
      output.find(generation_token, row_begin) < row_end);
    const bool camera_matches =
      output.find(camera_token, row_begin) !=
        std::string::npos &&
      (row_end == std::string::npos ||
      output.find(camera_token, row_begin) < row_end);
    if (generation_matches && camera_matches)
    {
      return output.substr(
        row_begin,
        row_end == std::string::npos ?
        std::string::npos : row_end - row_begin);
    }
    if (row_end == std::string::npos)
    {
      break;
    }
    row_begin = row_end + 1U;
  }
  return {};
}

std::optional<double> structuredDoubleField(
  const std::string & row,
  const std::string & field_name)
{
  const std::string field_token =
    field_name + "=";
  const std::size_t field_position =
    row.find(field_token);
  if (field_position == std::string::npos)
  {
    return std::nullopt;
  }
  const std::size_t value_begin =
    field_position + field_token.size();
  const std::size_t value_end =
    row.find(' ', value_begin);
  return std::stod(
    row.substr(value_begin, value_end - value_begin));
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  PostInitializationZeroHysteresisFailsClosedBeforeRuntimeAttempt)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  constexpr std::uint8_t kCameraId = 2U;
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, kCameraId, reference);
  CameraObservationQueue queue;
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  CameraObservationBatch batch =
    sealSingleCandidate(queue, 12.0, kCameraId, reference);

  manager.adaptive_policy_config.on_hysteresis_frames = 0U;
  const auto before = manager.readOnlyStructuralAudit();
  testing::internal::CaptureStdout();
  const auto audit = runGeneration(
    manager, *authority, batch, 95U);
  const std::string output =
    testing::internal::GetCapturedStdout();

  EXPECT_EQ(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kInvalidRuntimeInput);
  EXPECT_FALSE(audit.runtime_attempt_status.has_value());
  EXPECT_EQ(audit.runtime_attempt_count, 0U);
  EXPECT_EQ(audit.prepare_call_count, 0U);
  EXPECT_EQ(audit.scouted_camera_count, 0U);
  EXPECT_FALSE(audit.capability_issued);
  EXPECT_FALSE(audit.receipt_consumed);
  EXPECT_EQ(audit.selection_consumption_count, 0U);
  EXPECT_FALSE(audit.solver_called);
  EXPECT_EQ(audit.solver_call_count, 0U);
  EXPECT_EQ(audit.finalize_call_count, 0U);
  EXPECT_FALSE(audit.generation_update_receipt.has_value());
  EXPECT_EQ(
    output.find("[ADAPTIVE_VIO_MAP_MAINTENANCE]"),
    std::string::npos);
  EXPECT_EQ(
    output.find("[ADAPTIVE_VIO_UPDATE_METRIC]"),
    std::string::npos);

  const auto after = manager.readOnlyStructuralAudit();
  expectPersistentIdentity(before, after);
  EXPECT_EQ(
    before.adaptive_zero_map_reentry_healthy_scouts,
    after.adaptive_zero_map_reentry_healthy_scouts);
  EXPECT_EQ(
    before.adaptive_zero_map_reseed_latched,
    after.adaptive_zero_map_reseed_latched);
  EXPECT_EQ(
    before.adaptive_zero_map_last_trusted_generation,
    after.adaptive_zero_map_last_trusted_generation);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  OwnerSealedTwoCandidateBatchSelectsBestCameraIndependentOfOrder)
{
  for (const bool later_camera_arrives_first :
       std::array<bool, 2U>{{false, true}})
  {
    VIOManager manager;
    StatesGroup state;
    StatesGroup propagated;
    initializeManager(manager, state, propagated);
    const cv::Mat reference = texturedImage();
    seedCameraMap(manager, 0U, reference);
    seedCameraMap(manager, 2U, reference);

    CameraObservationQueue queue;
    CameraObservationBatch batch =
      sealTwoCandidates(queue, later_camera_arrives_first);
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const std::uint64_t lidar_generation =
      later_camera_arrives_first ? 102U : 101U;
    const auto lidar_issue = authority->issueLidarGeneration(
      lidar_generation,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
    ASSERT_EQ(
      lidar_issue.status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(lidar_issue.token.has_value());
    const auto runtime_input =
      authority->bindRuntimeInput(*lidar_issue.token);
    ASSERT_TRUE(runtime_input.has_value());

    processAuthorized(
      manager, batch, lidar_generation, &*runtime_input);

    const livo::AdaptiveVisualRuntimeAudit & audit =
      manager.adaptiveVisualRuntimeAudit();
    EXPECT_NE(
      audit.outcome,
      livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
    ASSERT_TRUE(audit.lidar_status.has_value());
    EXPECT_EQ(
      *audit.lidar_status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(audit.selection_issue_status.has_value());
    EXPECT_EQ(
      *audit.selection_issue_status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(audit.selection_consume_status.has_value());
    EXPECT_EQ(
      *audit.selection_consume_status,
      livo::AdaptiveVisualSelectionStatus::kReady);
    ASSERT_TRUE(audit.health_commit_status.has_value());
    EXPECT_EQ(
      *audit.health_commit_status,
      livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
    EXPECT_EQ(audit.lidar_generation, lidar_generation);
    ASSERT_TRUE(batch.validatedReceipt().has_value());
    EXPECT_EQ(
      audit.camera_batch_sequence,
      batch.validatedReceipt()->batchSequence());
    EXPECT_EQ(
      audit.candidate_mask,
      (std::array<bool, 4U>{{true, false, true, false}}));
    EXPECT_EQ(
      audit.selected_mask,
      (std::array<bool, 4U>{{false, false, true, false}}));
    ASSERT_TRUE(audit.selected_camera_id.has_value());
    EXPECT_EQ(*audit.selected_camera_id, 2U);
    EXPECT_EQ(
      audit.selection_reason,
      livo::AdaptiveSelectionReason::kSelectedBestInformationGain);
    ASSERT_TRUE(
      audit.selected_exposure_timestamp_sec.has_value());
    EXPECT_DOUBLE_EQ(
      *audit.selected_exposure_timestamp_sec,
      batch.validatedReceipt()->
      cameraExposureTimestampSec(2U));
    EXPECT_TRUE(audit.capability_issued);
    EXPECT_TRUE(audit.receipt_consumed);
    EXPECT_TRUE(audit.solver_called);
    EXPECT_FALSE(audit.legacy_fallback_used);
    EXPECT_EQ(audit.runtime_attempt_count, 1U);
    EXPECT_EQ(audit.selection_consumption_count, 1U);
    EXPECT_EQ(audit.prepare_call_count, 2U);
    EXPECT_EQ(audit.scouted_camera_count, 2U);
    EXPECT_EQ(audit.finalize_call_count, 1U);
    EXPECT_EQ(audit.policy_revision_before, 0U);
    EXPECT_EQ(audit.policy_revision_after, 1U);

    const auto structural = manager.readOnlyStructuralAudit();
    EXPECT_LE(
      structural.last_joint_solver.contributing_cameras, 1U);
  }
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  RuntimeInputKeepsAuthorityAliveAfterExternalOwnersAreReleased)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  seedCameraMap(manager, 2U, reference);

  CameraObservationBatch batch;
  std::optional<livo::AdaptiveVisualRuntimeAuthorityInput>
    runtime_input;
  constexpr std::uint64_t kGeneration = 150U;
  {
    CameraObservationQueue queue;
    batch = sealTwoCandidates(queue, false);
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);
    const auto lidar_issue = authority->issueLidarGeneration(
      kGeneration,
      livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
    ASSERT_TRUE(lidar_issue.token.has_value());
    runtime_input =
      authority->bindRuntimeInput(*lidar_issue.token);
    ASSERT_TRUE(runtime_input.has_value());
    authority.reset();
  }

  ASSERT_TRUE(runtime_input->valid());
  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  EXPECT_NE(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_TRUE(audit.receipt_consumed);
  EXPECT_EQ(audit.runtime_attempt_count, 1U);
  EXPECT_EQ(audit.selection_consumption_count, 1U);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  MissingRuntimeInputFailsClosedWithoutPersistentMutation)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealTwoCandidates(queue, false);
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorized(manager, batch, 201U, nullptr);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(audit);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kMissingRuntimeInput);
  EXPECT_FALSE(audit.capability_issued);
  EXPECT_FALSE(audit.receipt_consumed);
  EXPECT_EQ(audit.runtime_attempt_count, 0U);
  EXPECT_EQ(audit.selection_consumption_count, 0U);
  expectPersistentIdentity(
    before, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  ValidatedLidarInformationAppearsOnlyAfterAllBindingsMatch)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealTwoCandidates(queue, false);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kGeneration = 200U;
  const auto information =
    livo::AdaptiveVisualPolicy::Matrix6::Identity();
  const auto lidar_issue = authority->issueLidarGeneration(
    kGeneration, information, true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());

  processAuthorizedWithLidar(
    manager, batch, kGeneration + 1U,
    information, true, &*runtime_input);
  EXPECT_EQ(
    manager.adaptiveVisualRuntimeAudit().lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kLidarGenerationMismatch);
  EXPECT_FALSE(
    manager.adaptiveVisualRuntimeAudit().
    validated_lidar_pose_information.has_value());

  processAuthorizedWithLidar(
    manager, batch, kGeneration,
    2.0 * information, true, &*runtime_input);
  EXPECT_EQ(
    manager.adaptiveVisualRuntimeAudit().lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kLidarInformationMismatch);
  EXPECT_FALSE(
    manager.adaptiveVisualRuntimeAudit().
    validated_lidar_pose_information.has_value());

  processAuthorizedWithLidar(
    manager, batch, kGeneration,
    information, false, &*runtime_input);
  EXPECT_EQ(
    manager.adaptiveVisualRuntimeAudit().lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kLidarInformationMismatch);
  EXPECT_FALSE(
    manager.adaptiveVisualRuntimeAudit().
    validated_lidar_pose_information.has_value());

  processAuthorizedWithLidar(
    manager, batch, kGeneration,
    information, true, &*runtime_input);
  ASSERT_TRUE(
    manager.adaptiveVisualRuntimeAudit().
    validated_lidar_pose_information.has_value());
  EXPECT_TRUE(
    manager.adaptiveVisualRuntimeAudit().
    validated_lidar_pose_information->isApprox(
      information, 0.0));
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  MatchingNotReadyLidarFailsClosedBeforeAttemptAndPrepare)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  seedCameraMap(manager, 2U, reference);
  CameraObservationQueue queue;
  CameraObservationBatch accepted_batch =
    sealTwoCandidateImages(
    queue, 16.0, reference, reference);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto accepted =
    runGeneration(
    manager, *authority, accepted_batch, 310U);
  EXPECT_NE(
    accepted.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);

  ASSERT_TRUE(manager.activateCamera(1U));
  const auto before = manager.readOnlyStructuralAudit();
  CameraObservationBatch not_ready_batch =
    sealTwoCandidateImages(
    queue, 17.0, reference, reference);
  const auto lidar_issue = authority->issueLidarGeneration(
    311U,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), false);
  ASSERT_EQ(
    lidar_issue.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());

  processAuthorizedWithLidar(
    manager, not_ready_batch, 311U,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), false,
    &*runtime_input);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(audit);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kLidarInformationNotReady);
  ASSERT_TRUE(audit.lidar_status.has_value());
  EXPECT_EQ(
    *audit.lidar_status,
    livo::AdaptiveVisualSelectionStatus::
    kLidarInformationNotReady);
  EXPECT_FALSE(audit.runtime_attempt_status.has_value());
  EXPECT_EQ(audit.runtime_attempt_count, 0U);
  EXPECT_EQ(audit.selection_consumption_count, 0U);
  EXPECT_EQ(audit.prepare_call_count, 0U);
  EXPECT_EQ(audit.scouted_camera_count, 0U);
  EXPECT_EQ(audit.finalize_call_count, 0U);
  expectPersistentIdentity(
    before, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  PreSolverLioOnlyPreservesPreviousSolverAudit)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 2U, reference);
  CameraObservationQueue queue;
  CameraObservationBatch accepted_batch =
    sealSingleCandidate(queue, 14.0, 2U, reference);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kAcceptedGeneration = 300U;
  const auto lidar_issue = authority->issueLidarGeneration(
    kAcceptedGeneration,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  processAuthorized(
    manager, accepted_batch, kAcceptedGeneration,
    &*runtime_input);
  const auto before = manager.readOnlyStructuralAudit();
  EXPECT_NE(before.last_joint_solver.reason, "not_run");

  CameraObservationBatch missing_input_batch =
    sealSingleCandidate(queue, 15.0, 2U, reference);
  processAuthorized(
    manager, missing_input_batch,
    kAcceptedGeneration + 1U, nullptr);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(audit);
  const auto after = manager.readOnlyStructuralAudit();
  EXPECT_EQ(
    after.last_joint_solver.accepted,
    before.last_joint_solver.accepted);
  EXPECT_EQ(
    after.last_joint_solver.reason,
    before.last_joint_solver.reason);
  EXPECT_EQ(
    after.last_joint_solver.contributing_cameras,
    before.last_joint_solver.contributing_cameras);
  EXPECT_EQ(
    after.last_joint_solver.tracked_points,
    before.last_joint_solver.tracked_points);
  expectPersistentIdentity(before, after);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  ForeignBatchOwnerFailsClosedBeforeSolver)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  CameraObservationQueue batch_queue;
  CameraObservationBatch batch =
    sealTwoCandidates(batch_queue, false);
  CameraObservationQueue foreign_queue;
  auto foreign_authority =
    foreign_queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(foreign_authority, nullptr);
  const auto lidar_issue =
    foreign_authority->issueLidarGeneration(
    202U, livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    foreign_authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorized(
    manager, batch, 202U, &*runtime_input);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(audit);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kBatchPreflightRejected);
  ASSERT_TRUE(audit.batch_preflight_status.has_value());
  EXPECT_EQ(
    *audit.batch_preflight_status,
    livo::AdaptiveVisualSelectionStatus::
    kForeignCameraBatchOwner);
  EXPECT_FALSE(audit.selection_issue_status.has_value());
  EXPECT_FALSE(audit.capability_issued);
  EXPECT_FALSE(audit.receipt_consumed);
  EXPECT_EQ(audit.runtime_attempt_count, 0U);
  EXPECT_EQ(audit.selection_consumption_count, 0U);
  EXPECT_EQ(audit.prepare_call_count, 0U);
  EXPECT_EQ(audit.scouted_camera_count, 0U);
  expectPersistentIdentity(
    before, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  UnpreparedSealedCandidateIsRejectedBeforeReceiptWithoutMutation)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(
    manager, state, propagated, true, true, false);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealTwoCandidateImages(
    queue, 13.0, reference, reference);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kGeneration = 205U;
  const auto lidar_issue = authority->issueLidarGeneration(
    kGeneration,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  ASSERT_TRUE(manager.activateCamera(1U));
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(audit);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kCameraPreflightRejected);
  ASSERT_TRUE(audit.batch_preflight_status.has_value());
  EXPECT_EQ(
    *audit.batch_preflight_status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  EXPECT_FALSE(audit.runtime_attempt_status.has_value());
  EXPECT_FALSE(audit.selection_issue_status.has_value());
  EXPECT_FALSE(audit.capability_issued);
  EXPECT_FALSE(audit.receipt_consumed);
  EXPECT_EQ(audit.runtime_attempt_count, 0U);
  EXPECT_EQ(audit.selection_consumption_count, 0U);
  expectPersistentIdentity(
    before, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  StaleAuthorityTokenFailsClosedBeforeSolver)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  seedCameraMap(manager, 2U, reference);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealTwoCandidates(queue, false);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto stale_issue = authority->issueLidarGeneration(
    203U, livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true);
  ASSERT_TRUE(stale_issue.token.has_value());
  const auto latest_issue = authority->issueLidarGeneration(
    204U, livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    true);
  ASSERT_TRUE(latest_issue.token.has_value());
  const auto stale_input =
    authority->bindRuntimeInput(*stale_issue.token);
  ASSERT_TRUE(stale_input.has_value());
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorized(
    manager, batch, 203U, &*stale_input);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(audit);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kStaleRuntimeGeneration);
  ASSERT_TRUE(audit.runtime_attempt_status.has_value());
  EXPECT_EQ(
    *audit.runtime_attempt_status,
    livo::AdaptiveVisualRuntimeAttemptStatus::
    kStaleGeneration);
  EXPECT_FALSE(audit.selection_consume_status.has_value());
  EXPECT_FALSE(audit.capability_issued);
  EXPECT_FALSE(audit.receipt_consumed);
  EXPECT_EQ(audit.runtime_attempt_count, 0U);
  EXPECT_EQ(audit.selection_consumption_count, 0U);
  expectPersistentIdentity(
    before, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  RejectedSelectionLocksGenerationWithoutPolicyOrVisualWrites)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  CameraObservationQueue queue;
  const cv::Mat hard_failed_image(
    480, 640, CV_8UC1, cv::Scalar(255U));
  CameraObservationBatch batch =
    sealTwoCandidateImages(
    queue, 11.0, hard_failed_image,
    hard_failed_image);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kGeneration = 209U;
  const auto lidar_issue = authority->issueLidarGeneration(
    kGeneration,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  ASSERT_TRUE(manager.activateCamera(1U));
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);

  const auto & first_audit =
    manager.adaptiveVisualRuntimeAudit();
  EXPECT_EQ(
    first_audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_FALSE(first_audit.solver_called);
  EXPECT_FALSE(first_audit.legacy_fallback_used);
  EXPECT_EQ(
    first_audit.policy_revision_after,
    first_audit.policy_revision_before);
  EXPECT_EQ(
    first_audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kSelectionIssueRejected);
  ASSERT_TRUE(
    first_audit.selection_issue_status.has_value());
  EXPECT_EQ(
    *first_audit.selection_issue_status,
    livo::AdaptiveVisualSelectionStatus::
    kNoSelectedCamera);
  EXPECT_EQ(first_audit.runtime_attempt_count, 1U);
  EXPECT_EQ(first_audit.selection_consumption_count, 0U);
  EXPECT_EQ(first_audit.solver_call_count, 0U);
  EXPECT_FALSE(first_audit.health_commit_status.has_value());
  EXPECT_EQ(first_audit.prepare_call_count, 2U);
  EXPECT_EQ(first_audit.scouted_camera_count, 2U);
  EXPECT_EQ(first_audit.finalize_call_count, 0U);
  const auto after_first =
    manager.readOnlyStructuralAudit();
  expectEstimationIdentity(
    before, after_first, false);
  EXPECT_EQ(
    after_first.adaptive_policy_revision,
    before.adaptive_policy_revision);
  EXPECT_EQ(
    after_first.adaptive_prepare_call_count,
    before.adaptive_prepare_call_count + 2U);
  EXPECT_EQ(
    after_first.adaptive_scout_call_count,
    before.adaptive_scout_call_count + 2U);
  EXPECT_EQ(
    after_first.adaptive_finalize_call_count,
    before.adaptive_finalize_call_count);
  EXPECT_EQ(
    after_first.shared_landmark_shadow_entry_count,
    before.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    after_first.reference_only_migration_shadow_entry_count,
    before.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    after_first.multi_camera_linearization_shadow_entry_count,
    before.multi_camera_linearization_shadow_entry_count);

  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);

  const auto & replay_audit =
    manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(replay_audit);
  EXPECT_EQ(
    replay_audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kGenerationAlreadyProcessed);
  EXPECT_EQ(replay_audit.runtime_attempt_count, 1U);
  EXPECT_EQ(replay_audit.selection_consumption_count, 0U);
  EXPECT_EQ(replay_audit.prepare_call_count, 0U);
  EXPECT_EQ(replay_audit.scouted_camera_count, 0U);
  expectPersistentIdentity(
    after_first, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  SemanticallyInvalidLidarInformationRejectsAfterScoutAndLocksAttempt)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  seedCameraMap(manager, 2U, reference);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealTwoCandidateImages(
    queue, 18.0, reference, reference);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kGeneration = 212U;
  const auto zero_information =
    livo::AdaptiveVisualPolicy::Matrix6::Zero();
  const auto lidar_issue = authority->issueLidarGeneration(
    kGeneration, zero_information, true);
  ASSERT_EQ(
    lidar_issue.status,
    livo::AdaptiveVisualSelectionStatus::kReady);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  ASSERT_TRUE(manager.activateCamera(1U));
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorizedWithLidar(
    manager, batch, kGeneration, zero_information, true,
    &*runtime_input);

  const auto & first = manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(first);
  EXPECT_EQ(
    first.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kSelectionIssueRejected);
  ASSERT_TRUE(first.selection_issue_status.has_value());
  EXPECT_EQ(
    *first.selection_issue_status,
    livo::AdaptiveVisualSelectionStatus::
    kInvalidLidarInformation);
  EXPECT_FALSE(first.health_commit_status.has_value());
  EXPECT_EQ(first.runtime_attempt_count, 1U);
  EXPECT_EQ(first.selection_consumption_count, 0U);
  EXPECT_EQ(first.prepare_call_count, 2U);
  EXPECT_EQ(first.scouted_camera_count, 2U);
  EXPECT_EQ(first.finalize_call_count, 0U);
  const auto after_first = manager.readOnlyStructuralAudit();
  expectEstimationIdentity(before, after_first);
  EXPECT_EQ(
    after_first.adaptive_prepare_call_count,
    before.adaptive_prepare_call_count + 2U);
  EXPECT_EQ(
    after_first.adaptive_scout_call_count,
    before.adaptive_scout_call_count + 2U);
  EXPECT_EQ(
    after_first.adaptive_finalize_call_count,
    before.adaptive_finalize_call_count);

  processAuthorizedWithLidar(
    manager, batch, kGeneration, zero_information, true,
    &*runtime_input);
  const auto & replay = manager.adaptiveVisualRuntimeAudit();
  EXPECT_EQ(
    replay.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kGenerationAlreadyProcessed);
  EXPECT_EQ(replay.runtime_attempt_count, 1U);
  EXPECT_EQ(replay.selection_consumption_count, 0U);
  EXPECT_EQ(replay.prepare_call_count, 0U);
  EXPECT_EQ(replay.scouted_camera_count, 0U);
  expectPersistentIdentity(
    after_first, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  CandidateImageAndGenerationTamperingFailsClosed)
{
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 2U>{{0U, 2U}})
  {
    for (const bool replace_generation_identity :
         std::array<bool, 2U>{{false, true}})
    {
      VIOManager manager;
      StatesGroup state;
      StatesGroup propagated;
      initializeManager(manager, state, propagated);
      const cv::Mat reference = texturedImage();
      seedCameraMap(manager, 0U, reference);
      seedCameraMap(manager, 2U, reference);
      CameraObservationQueue queue;
      CameraObservationBatch batch =
        sealTwoCandidates(queue, false);
      auto authority =
        queue.issueAdaptiveVisualSelectionAuthority();
      ASSERT_NE(authority, nullptr);
      const std::uint64_t lidar_generation =
        210U + 2U * camera_id +
        (replace_generation_identity ? 1U : 0U);
      const auto lidar_issue =
        authority->issueLidarGeneration(
        lidar_generation,
        livo::AdaptiveVisualPolicy::Matrix6::Identity(),
        true);
      ASSERT_TRUE(lidar_issue.token.has_value());
      const auto runtime_input =
        authority->bindRuntimeInput(*lidar_issue.token);
      ASSERT_TRUE(runtime_input.has_value());
      CameraObservationBatch tampered_batch = batch;

      if (replace_generation_identity)
      {
        CameraObservation replacement;
        replacement.camera_id = camera_id;
        replacement.timestamp =
          tampered_batch.observations[camera_id].timestamp;
        replacement.image =
          tampered_batch.observations[camera_id].image.clone();
        replacement.photometric_status =
          tampered_batch.observations[camera_id].
          photometric_status;
        tampered_batch.observations[camera_id] =
          std::move(replacement);
      }
      else
      {
        tampered_batch.observations[camera_id].image =
          batch.observations[camera_id].image.clone();
        auto & pixel =
          tampered_batch.observations[camera_id].
          image.at<std::uint8_t>(0, 0);
        pixel = static_cast<std::uint8_t>(pixel ^ 0x01U);
      }
      const auto before =
        manager.readOnlyStructuralAudit();

      processAuthorized(
        manager, tampered_batch, lidar_generation,
        &*runtime_input);

      const auto & audit =
        manager.adaptiveVisualRuntimeAudit();
      expectLioOnlyWithoutSolver(audit);
      EXPECT_EQ(
        audit.lio_only_reason,
        livo::AdaptiveVisualLioOnlyReason::
        kBatchPreflightRejected);
      ASSERT_TRUE(audit.batch_preflight_status.has_value());
      EXPECT_EQ(
        *audit.batch_preflight_status,
        livo::AdaptiveVisualSelectionStatus::
        kObservationImageIdentityMismatch);
      EXPECT_FALSE(audit.capability_issued);
      EXPECT_FALSE(audit.receipt_consumed);
      EXPECT_EQ(audit.runtime_attempt_count, 0U);
      EXPECT_EQ(audit.selection_consumption_count, 0U);
      expectPersistentIdentity(
        before, manager.readOnlyStructuralAudit());

      processAuthorized(
        manager, batch, lidar_generation,
        &*runtime_input);
      const auto & retry_audit =
        manager.adaptiveVisualRuntimeAudit();
      EXPECT_NE(
        retry_audit.outcome,
        livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
      EXPECT_EQ(retry_audit.runtime_attempt_count, 1U);
      EXPECT_EQ(
        retry_audit.selection_consumption_count, 1U);
    }
  }
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  ConsumedGenerationCannotStartSecondTransaction)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  seedCameraMap(manager, 2U, reference);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealTwoCandidates(queue, false);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kGeneration = 220U;
  const auto lidar_issue = authority->issueLidarGeneration(
    kGeneration,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);
  ASSERT_EQ(
    manager.adaptiveVisualRuntimeAudit().
    runtime_attempt_count,
    1U);
  ASSERT_EQ(
    manager.adaptiveVisualRuntimeAudit().
    selection_consumption_count,
    1U);
  const auto before_replay =
    manager.readOnlyStructuralAudit();

  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);

  const auto & replay_audit =
    manager.adaptiveVisualRuntimeAudit();
  expectLioOnlyWithoutSolver(replay_audit);
  EXPECT_EQ(
    replay_audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::
    kGenerationAlreadyProcessed);
  ASSERT_TRUE(
    replay_audit.runtime_attempt_status.has_value());
  EXPECT_EQ(
    *replay_audit.runtime_attempt_status,
    livo::AdaptiveVisualRuntimeAttemptStatus::
    kAlreadyAttempted);
  EXPECT_EQ(replay_audit.runtime_attempt_count, 1U);
  EXPECT_EQ(replay_audit.selection_consumption_count, 1U);
  EXPECT_EQ(replay_audit.prepare_call_count, 0U);
  EXPECT_EQ(replay_audit.scouted_camera_count, 0U);
  EXPECT_EQ(replay_audit.finalize_call_count, 0U);
  EXPECT_FALSE(replay_audit.receipt_consumed);
  expectPersistentIdentity(
    before_replay, manager.readOnlyStructuralAudit());
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  PartialMultiCandidateBatchStillSelectsExactlyOne)
{
  constexpr std::array<std::size_t, 3U>
    kCandidateCounts{{2U, 3U, 4U}};
  constexpr std::array<
    std::array<bool, 4U>, 3U> kCandidateMasks{{
      {{true, false, true, false}},
      {{true, true, false, true}},
      {{true, true, true, true}},
    }};
  const cv::Mat reference = texturedImage();

  for (std::size_t case_index = 0U;
       case_index < kCandidateCounts.size(); ++case_index)
  {
    VIOManager manager;
    StatesGroup state;
    StatesGroup propagated;
    initializeManager(manager, state, propagated);
    ASSERT_EQ(
      manager.adaptive_policy_config.maximum_selected_cameras,
      1U);
    for (std::uint8_t camera_id = 0U; camera_id < 4U;
         ++camera_id)
    {
      seedCameraMap(manager, camera_id, reference);
    }

    CameraObservationQueue queue;
    CameraObservationBatch batch;
    if (kCandidateCounts[case_index] == 2U)
    {
      batch = sealTwoCandidateImages(
        queue, 40.0, reference, reference);
    }
    else if (kCandidateCounts[case_index] == 3U)
    {
      batch = sealThreeCandidates(queue, true);
    }
    else
    {
      batch = sealFourCandidates(queue, false);
    }
    ASSERT_EQ(
      batch.camera_count, kCandidateCounts[case_index]);
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);

    const auto audit = runGeneration(
      manager, *authority, batch,
      270U + case_index);

    EXPECT_EQ(
      audit.candidate_mask, kCandidateMasks[case_index]);
    EXPECT_EQ(
      audit.scouted_camera_count,
      kCandidateCounts[case_index]);
    EXPECT_EQ(
      audit.prepare_call_count,
      kCandidateCounts[case_index]);
    EXPECT_EQ(
      std::count(
        audit.selected_mask.begin(), audit.selected_mask.end(),
        true),
      1);
    for (std::size_t camera_id = 0U; camera_id < 4U;
         ++camera_id)
    {
      EXPECT_EQ(
        audit.eligible_mask[camera_id],
        kCandidateMasks[case_index][camera_id]);
      EXPECT_EQ(
        audit.reject_reasons[camera_id],
        kCandidateMasks[case_index][camera_id] ?
        livo::AdaptiveCameraRejectReason::kNone :
        livo::AdaptiveCameraRejectReason::kFrameMissing);
    }
  }
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  ZeroMapCandidateSeedsFromLioWithoutSelectionOrSolver)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealSingleCandidate(queue, 13.0, 2U, texturedImage());
  ASSERT_EQ(batch.camera_count, 1U);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto before = manager.readOnlyStructuralAudit();

  const auto audit = runGeneration(
    manager, *authority, batch, 283U);
  const auto after = manager.readOnlyStructuralAudit();

  EXPECT_EQ(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    audit.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kVisualMapBootstrap);
  EXPECT_FALSE(audit.selection_issue_status.has_value());
  EXPECT_FALSE(audit.selection_consume_status.has_value());
  EXPECT_EQ(
    audit.candidate_mask,
    (std::array<bool, 4U>{{false, false, true, false}}));
  EXPECT_EQ(
    std::count(
      audit.selected_mask.begin(), audit.selected_mask.end(),
      true),
    0);
  EXPECT_FALSE(audit.selected_camera_id.has_value());
  EXPECT_EQ(audit.prepare_call_count, 1U);
  EXPECT_EQ(audit.scouted_camera_count, 0U);
  EXPECT_FALSE(audit.solver_called);
  EXPECT_EQ(audit.solver_call_count, 0U);
  EXPECT_EQ(audit.finalize_call_count, 1U);
  ASSERT_TRUE(audit.generation_update_receipt.has_value());
  EXPECT_FALSE(
    audit.generation_update_receipt->
    visual_update_accepted);
  EXPECT_FALSE(
    audit.generation_update_receipt->solver_called);
  EXPECT_EQ(
    audit.generation_update_receipt->solver_reason,
    "not_called");
  EXPECT_EQ(
    audit.generation_update_receipt->map_action,
    livo::AdaptiveVisualMapAction::kColdStartSeed);
  EXPECT_TRUE(
    audit.generation_update_receipt->seed_new_points);
  EXPECT_FALSE(
    audit.generation_update_receipt->
    update_existing_points);
  for (std::size_t camera_id = 0U; camera_id < 4U;
       ++camera_id)
  {
    EXPECT_EQ(
      after.legacy_feature_point_counts[camera_id] >
      before.legacy_feature_point_counts[camera_id],
      camera_id == 2U);
  }
  EXPECT_EQ(
    after.adaptive_visual_map_lifecycle[2U],
    livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated);
  EXPECT_FALSE(after.adaptive_camera_initialized[2U]);

}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  RuntimeAuditCopiesAuthoritativePlanForOneBadThreeHealthy)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  for (std::uint8_t camera_id = 0U; camera_id < 4U;
       ++camera_id)
  {
    seedCameraMap(manager, camera_id, reference);
  }
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealFourCandidates(queue, false, true);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  const auto audit = runGeneration(
    manager, *authority, batch, 280U);

  EXPECT_EQ(
    audit.candidate_mask,
    (std::array<bool, 4U>{{true, true, true, true}}));
  EXPECT_EQ(
    audit.eligible_mask,
    (std::array<bool, 4U>{{false, true, true, true}}));
  EXPECT_EQ(
    audit.reject_reasons[0U],
    livo::AdaptiveCameraRejectReason::kImageHardFailure);
  for (std::size_t camera_id = 1U; camera_id < 4U;
       ++camera_id)
  {
    EXPECT_EQ(
      audit.reject_reasons[camera_id],
      livo::AdaptiveCameraRejectReason::kNone);
  }
  EXPECT_FALSE(audit.selected_mask[0U]);
  EXPECT_EQ(
    std::count(
      audit.selected_mask.begin(), audit.selected_mask.end(),
      true),
    1);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  FourHealthMetricEmitsFourRowsAndMarksMissingRoute)
{
  constexpr double kSourceEpochSec = 1784713075.0;
  constexpr double kRelativeImageTimeSec = 41.25;
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  for (std::uint8_t camera_id = 0U; camera_id < 4U;
       ++camera_id)
  {
    seedCameraMap(manager, camera_id, reference);
  }
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealThreeCandidates(
    queue, true, kSourceEpochSec);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  testing::internal::CaptureStdout();
  const auto audit = runGeneration(
    manager, *authority, batch, 281U,
    kRelativeImageTimeSec);
  const std::string output =
    testing::internal::GetCapturedStdout();

  ASSERT_EQ(
    audit.reject_reasons[2U],
    livo::AdaptiveCameraRejectReason::kFrameMissing);
  EXPECT_EQ(
    countOccurrences(
      output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    4U);
  EXPECT_NE(
    output.find(
      "camera_id=2 camera=back metrics_mask=0 "
      "frame_received=0 available=0 eligible=0 selected=0 "
      "reject_reason=frame_missing"),
    std::string::npos);
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 3U>{{0U, 1U, 3U}})
  {
    const std::string row =
      cameraMetricRow(output, camera_id);
    ASSERT_FALSE(row.empty());
    const auto relative_time =
      structuredDoubleField(row, "t");
    const auto source_stamp =
      structuredDoubleField(
      row, "source_stamp_sec");
    const auto batch_source_stamp =
      structuredDoubleField(
      row, "batch_source_stamp_sec");
    ASSERT_TRUE(relative_time.has_value());
    ASSERT_TRUE(source_stamp.has_value());
    ASSERT_TRUE(batch_source_stamp.has_value());
    EXPECT_DOUBLE_EQ(
      *relative_time, kRelativeImageTimeSec);
    EXPECT_TRUE(std::isfinite(*source_stamp));
    EXPECT_TRUE(std::isfinite(*batch_source_stamp));
    EXPECT_GT(*source_stamp, 1.0e9);
    EXPECT_GT(*batch_source_stamp, 1.0e9);
    EXPECT_DOUBLE_EQ(
      *source_stamp,
      batch.observations[camera_id].timestamp);
    EXPECT_DOUBLE_EQ(
      *batch_source_stamp, batch.timestamp);
    if (camera_id == 0U)
    {
      EXPECT_NE(*source_stamp, *batch_source_stamp);
      EXPECT_NEAR(
        std::abs(
          *source_stamp - *batch_source_stamp),
        0.004, 1.0e-6);
    }
  }
  const std::string missing_row =
    cameraMetricRow(output, 2U);
  ASSERT_FALSE(missing_row.empty());
  const auto missing_source_stamp =
    structuredDoubleField(
    missing_row, "source_stamp_sec");
  const auto missing_batch_source_stamp =
    structuredDoubleField(
    missing_row, "batch_source_stamp_sec");
  ASSERT_TRUE(missing_source_stamp.has_value());
  ASSERT_TRUE(
    missing_batch_source_stamp.has_value());
  EXPECT_TRUE(std::isfinite(*missing_source_stamp));
  EXPECT_TRUE(
    std::isfinite(*missing_batch_source_stamp));
  EXPECT_GT(*missing_source_stamp, 1.0e9);
  EXPECT_GT(*missing_batch_source_stamp, 1.0e9);
  EXPECT_DOUBLE_EQ(
    *missing_source_stamp, batch.timestamp);
  EXPECT_DOUBLE_EQ(
    *missing_batch_source_stamp, batch.timestamp);
  EXPECT_DOUBLE_EQ(
    *missing_source_stamp,
    *missing_batch_source_stamp);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  SingleCameraCandidatePublishesCompleteAuthoritativeHealthRows)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealSingleCandidate(queue, 42.0, 0U, reference);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  testing::internal::CaptureStdout();
  const auto audit = runGeneration(
    manager, *authority, batch, 282U);
  const std::string output =
    testing::internal::GetCapturedStdout();

  EXPECT_FALSE(audit.legacy_fallback_used);
  EXPECT_TRUE(audit.solver_called);
  EXPECT_EQ(audit.solver_call_count, 1U);

  EXPECT_EQ(
    countOccurrences(
      output, "[ADAPTIVE_VIO_CAMERA_METRIC]"),
    4U);
  for (std::uint8_t camera_id = 0U; camera_id < 4U;
       ++camera_id)
  {
    const std::string row =
      cameraMetricRow(output, camera_id);
    ASSERT_FALSE(row.empty());
    const bool is_front = camera_id == 0U;
    EXPECT_NE(
      row.find(is_front ? "enabled=1" : "enabled=0"),
      std::string::npos);
    EXPECT_NE(
      row.find(is_front ? "selected=1" : "selected=0"),
      std::string::npos);
  }
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  FourCandidatesScoutAllButRunOnlyTopOneWithoutFullFourShadows)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  for (std::uint8_t camera_id = 0U; camera_id < 4U;
       ++camera_id)
  {
    seedCameraMap(manager, camera_id, reference);
  }
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealFourCandidates(queue);
  ASSERT_EQ(batch.camera_count, 4U);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kGeneration = 230U;
  const auto lidar_issue = authority->issueLidarGeneration(
    kGeneration,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  EXPECT_NE(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(audit.scouted_camera_count, 4U);
  EXPECT_EQ(audit.solver_call_count, 1U);
  EXPECT_EQ(
    std::count(
      audit.selected_mask.begin(), audit.selected_mask.end(),
      true),
    1);
  ASSERT_TRUE(audit.selected_camera_id.has_value());
  EXPECT_EQ(*audit.selected_camera_id, 3U);
  EXPECT_FALSE(audit.shared_landmark_shadow_called);
  EXPECT_FALSE(
    audit.reference_only_migration_shadow_called);
  EXPECT_FALSE(
    audit.multi_camera_linearization_shadow_called);
  const auto after = manager.readOnlyStructuralAudit();
  EXPECT_EQ(
    before.shared_landmark_shadow_entry_count,
    after.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    before.reference_only_migration_shadow_entry_count,
    after.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    before.multi_camera_linearization_shadow_entry_count,
    after.multi_camera_linearization_shadow_entry_count);
  EXPECT_LE(
    after.last_joint_solver.contributing_cameras,
    1U);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  WeakNoiseDoesNotTriggerPostUpdateFrontRollback)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, 0U, reference);
  seedCameraMap(manager, 2U, reference);
  manager.joint_img_point_cov = 1.0e12;

  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealTwoCandidates(queue, false);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  const auto audit = runGeneration(
    manager, *authority, batch, 283U);

  ASSERT_TRUE(audit.selected_camera_id.has_value());
  EXPECT_EQ(*audit.selected_camera_id, 2U);
  EXPECT_TRUE(audit.solver_called);
  EXPECT_EQ(audit.solver_call_count, 1U);
  EXPECT_FALSE(audit.legacy_fallback_used);
  ASSERT_TRUE(audit.fusion_camera_id.has_value());
  EXPECT_EQ(*audit.fusion_camera_id, 2U);
  EXPECT_FALSE(audit.pose_covariance_volume_reduction.has_value());
  ASSERT_TRUE(audit.generation_update_receipt.has_value());
  EXPECT_EQ(
    audit.generation_update_receipt->solver_reason,
    "accepted_dry_run");
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  ThreeCandidatesScoutAllButRunOnlyTopOne)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  for (const std::uint8_t camera_id :
       std::array<std::uint8_t, 3U>{{0U, 1U, 3U}})
  {
    seedCameraMap(manager, camera_id, reference);
  }
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealThreeCandidates(queue);
  ASSERT_EQ(batch.camera_count, 3U);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  constexpr std::uint64_t kGeneration = 231U;
  const auto lidar_issue = authority->issueLidarGeneration(
    kGeneration,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(), true);
  ASSERT_TRUE(lidar_issue.token.has_value());
  const auto runtime_input =
    authority->bindRuntimeInput(*lidar_issue.token);
  ASSERT_TRUE(runtime_input.has_value());
  const auto before = manager.readOnlyStructuralAudit();

  processAuthorized(
    manager, batch, kGeneration, &*runtime_input);

  const auto & audit = manager.adaptiveVisualRuntimeAudit();
  EXPECT_NE(
    audit.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(audit.scouted_camera_count, 3U);
  EXPECT_EQ(audit.solver_call_count, 1U);
  EXPECT_EQ(
    std::count(
      audit.selected_mask.begin(), audit.selected_mask.end(),
      true),
    1);
  ASSERT_TRUE(audit.selected_camera_id.has_value());
  EXPECT_EQ(*audit.selected_camera_id, 1U);
  const auto after = manager.readOnlyStructuralAudit();
  EXPECT_LE(
    after.last_joint_solver.contributing_cameras, 1U);
  EXPECT_EQ(
    before.shared_landmark_shadow_entry_count,
    after.shared_landmark_shadow_entry_count);
  EXPECT_EQ(
    before.reference_only_migration_shadow_entry_count,
    after.reference_only_migration_shadow_entry_count);
  EXPECT_EQ(
    before.multi_camera_linearization_shadow_entry_count,
    after.multi_camera_linearization_shadow_entry_count);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  HardImageFailureSwitchesToHealthyReplacementImmediately)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat strong = texturedImage();
  seedCameraMap(manager, 0U, strong);
  seedCameraMap(manager, 2U, strong);
  CameraObservationQueue queue;
  CameraObservationBatch initial_batch =
    sealTwoCandidateImages(
    queue, 20.0, strong,
    cv::Mat(480, 640, CV_8UC1, cv::Scalar(96U)));
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  const auto initial = runGeneration(
    manager, *authority, initial_batch, 240U);
  ASSERT_TRUE(initial.selected_camera_id.has_value());
  ASSERT_EQ(*initial.selected_camera_id, 0U);

  CameraObservationBatch replacement_batch =
    sealTwoCandidateImages(
    queue, 21.0,
    cv::Mat(480, 640, CV_8UC1, cv::Scalar(255U)),
    strong);
  const auto replacement = runGeneration(
    manager, *authority, replacement_batch, 241U);

  ASSERT_TRUE(replacement.selected_camera_id.has_value());
  EXPECT_EQ(*replacement.selected_camera_id, 2U);
  EXPECT_EQ(
    replacement.selection_reason,
    livo::AdaptiveSelectionReason::kSwitchedAfterHardFault);
  ASSERT_TRUE(replacement.health_commit_status.has_value());
  EXPECT_EQ(
    *replacement.health_commit_status,
    livo::AdaptiveCandidateHealthCommitStatus::kCommitted);
  EXPECT_EQ(replacement.policy_revision_before, 1U);
  EXPECT_EQ(replacement.policy_revision_after, 2U);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  BetterInformationGainRequiresThreeChallengerConfirmations)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat weak = weakTexturedImage();
  const cv::Mat strong = texturedImage();
  seedCameraMap(manager, 0U, weak);
  seedCameraMap(manager, 2U, strong);
  CameraObservationQueue queue;
  CameraObservationBatch initial_batch =
    sealSingleCandidate(queue, 30.0, 0U, weak);
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);
  const auto initial = runGeneration(
    manager, *authority, initial_batch, 250U);
  ASSERT_TRUE(initial.selected_camera_id.has_value());
  ASSERT_EQ(*initial.selected_camera_id, 0U);

  std::array<std::uint8_t, 3U> selected{};
  std::array<livo::AdaptiveSelectionReason, 3U> reasons{};
  for (std::size_t index = 0U; index < 3U; ++index)
  {
    CameraObservationBatch challenger_batch =
      sealTwoCandidateImages(
      queue, 31.0 + static_cast<double>(index),
      weak, strong);
    const auto audit = runGeneration(
      manager, *authority, challenger_batch,
      251U + index);
    ASSERT_TRUE(audit.selected_camera_id.has_value());
    selected[index] = *audit.selected_camera_id;
    reasons[index] = audit.selection_reason;
    EXPECT_EQ(
      audit.policy_revision_after,
      audit.policy_revision_before + 1U);
  }

  EXPECT_EQ(
    selected,
    (std::array<std::uint8_t, 3U>{{0U, 0U, 2U}}));
  EXPECT_EQ(
    reasons[0U],
    livo::AdaptiveSelectionReason::kRetainedHealthyPrimary);
  EXPECT_EQ(
    reasons[1U],
    livo::AdaptiveSelectionReason::kRetainedHealthyPrimary);
  EXPECT_EQ(
    reasons[2U],
    livo::AdaptiveSelectionReason::
    kSwitchedAfterChallengerConfirmation);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  InitializedZeroMapReentryReseedsFromLioAndRecoversVisualCommit)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  constexpr std::uint8_t kCameraId = 2U;
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, kCameraId, reference);
  CameraObservationQueue queue;
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  CameraObservationBatch initialization_batch =
    sealSingleCandidate(queue, 70.0, kCameraId, reference);
  const auto initialization = runGeneration(
    manager, *authority, initialization_batch, 300U);
  ASSERT_EQ(
    initialization.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
  ASSERT_TRUE(initialization.selected_camera_id.has_value());
  EXPECT_EQ(*initialization.selected_camera_id, kCameraId);
  ASSERT_EQ(
    manager.readOnlyStructuralAudit().
    adaptive_visual_map_lifecycle[kCameraId],
    livo::AdaptiveVisualMapLifecycle::kInitialized);

  const Eigen::Vector3d translation(100.0, 0.0, 0.0);
  state.pos_end = translation;
  propagated = state;
  for (std::size_t scout = 0U;
       scout < manager.adaptive_policy_config.
         on_hysteresis_frames;
       ++scout)
  {
    CameraObservationBatch scout_batch =
      sealSingleCandidate(
        queue, 71.0 + static_cast<double>(scout),
        kCameraId, reference);
    const auto scout_audit = runTranslatedGeneration(
      manager, *authority, scout_batch, 301U + scout,
      translation);
    EXPECT_EQ(
      scout_audit.outcome,
      livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
    EXPECT_FALSE(scout_audit.solver_called);
    EXPECT_EQ(scout_audit.finalize_call_count, 0U);
  }

  const auto before_reseed =
    manager.readOnlyStructuralAudit();
  CameraObservationBatch reseed_batch =
    sealSingleCandidate(
      queue,
      71.0 + static_cast<double>(
        manager.adaptive_policy_config.
        on_hysteresis_frames),
      kCameraId, reference);
  testing::internal::CaptureStdout();
  const auto reseed = runTranslatedGeneration(
    manager, *authority, reseed_batch,
    301U + manager.adaptive_policy_config.
      on_hysteresis_frames,
    translation);
  const std::string reseed_output =
    testing::internal::GetCapturedStdout();

  EXPECT_EQ(
    reseed.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    reseed.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kVisualMapReseed);
  EXPECT_FALSE(reseed.selected_camera_id.has_value());
  EXPECT_FALSE(reseed.solver_called);
  EXPECT_EQ(reseed.solver_call_count, 0U);
  EXPECT_EQ(reseed.finalize_call_count, 1U);
  ASSERT_TRUE(reseed.generation_update_receipt.has_value());
  EXPECT_FALSE(
    reseed.generation_update_receipt->
    visual_update_accepted);
  EXPECT_FALSE(
    reseed.generation_update_receipt->solver_called);
  EXPECT_EQ(
    reseed.generation_update_receipt->map_action,
    livo::AdaptiveVisualMapAction::kLidarAnchoredReseed);
  EXPECT_TRUE(
    reseed.generation_update_receipt->seed_new_points);
  EXPECT_FALSE(
    reseed.generation_update_receipt->
    update_existing_points);
  EXPECT_EQ(
    countOccurrences(
      reseed_output, "[ADAPTIVE_VIO_MAP_MAINTENANCE]"),
    1U);
  const std::string reseed_metric =
    mapMaintenanceMetricRow(reseed_output, 306U, kCameraId);
  ASSERT_FALSE(reseed_metric.empty());
  EXPECT_NE(
    reseed_metric.find(
      "schema_version=2 runtime_path=adaptive_single"),
    std::string::npos);
  EXPECT_NE(
    reseed_metric.find(
      "map_action=lio_anchored_reseed seed_new_points=1 "
      "update_existing_points=0"),
    std::string::npos);
  ASSERT_TRUE(
    structuredDoubleField(
      reseed_metric, "attempted").has_value());
  ASSERT_TRUE(
    structuredDoubleField(
      reseed_metric, "succeeded").has_value());
  ASSERT_TRUE(
    structuredDoubleField(
      reseed_metric, "generated_points").has_value());
  EXPECT_DOUBLE_EQ(
    *structuredDoubleField(reseed_metric, "attempted"), 1.0);
  EXPECT_DOUBLE_EQ(
    *structuredDoubleField(reseed_metric, "succeeded"), 1.0);
  EXPECT_GT(
    *structuredDoubleField(reseed_metric, "generated_points"),
    0.0);

  const auto after_reseed =
    manager.readOnlyStructuralAudit();
  ASSERT_TRUE(before_reseed.active_state_available);
  ASSERT_TRUE(after_reseed.active_state_available);
  EXPECT_EQ(
    std::memcmp(
      before_reseed.active_state.data(),
      after_reseed.active_state.data(),
      sizeof(before_reseed.active_state)),
    0);
  ASSERT_TRUE(before_reseed.propagated_state_available);
  ASSERT_TRUE(after_reseed.propagated_state_available);
  EXPECT_EQ(
    std::memcmp(
      before_reseed.propagated_state.data(),
      after_reseed.propagated_state.data(),
      sizeof(before_reseed.propagated_state)),
    0);
  EXPECT_EQ(
    after_reseed.adaptive_policy_revision,
    before_reseed.adaptive_policy_revision);
  EXPECT_EQ(
    after_reseed.adaptive_visual_map_lifecycle,
    before_reseed.adaptive_visual_map_lifecycle);
  EXPECT_GT(
    after_reseed.legacy_feature_point_counts[kCameraId],
    before_reseed.legacy_feature_point_counts[kCameraId]);
  EXPECT_TRUE(
    after_reseed.adaptive_zero_map_reseed_latched[kCameraId]);
  EXPECT_EQ(
    after_reseed.
    adaptive_zero_map_reentry_healthy_scouts[kCameraId],
    0U);

  const double accepted_residual_rmse =
    manager.joint_update_config.max_residual_rmse;
  manager.joint_update_config.max_residual_rmse = 1.0e-9;
  state.pos_end = translation;
  propagated = state;
  cv::Mat mismatched_reference;
  cv::bitwise_not(reference, mismatched_reference);
  CameraObservationBatch rejected_recovery_batch =
    sealSingleCandidate(
    queue, 77.0, kCameraId, mismatched_reference);
  testing::internal::CaptureStdout();
  const auto rejected_recovery = runTranslatedGeneration(
    manager, *authority, rejected_recovery_batch, 307U,
    translation);
  const std::string rejected_recovery_output =
    testing::internal::GetCapturedStdout();
  const std::string rejected_recovery_row =
    cameraMetricRow(rejected_recovery_output, kCameraId);
  ASSERT_FALSE(rejected_recovery_row.empty());
  const auto rejected_recovery_candidates =
    structuredDoubleField(
      rejected_recovery_row, "candidates");
  const auto rejected_recovery_tracked =
    structuredDoubleField(
      rejected_recovery_row, "tracked");
  ASSERT_TRUE(rejected_recovery_candidates.has_value());
  ASSERT_TRUE(rejected_recovery_tracked.has_value());
  EXPECT_DOUBLE_EQ(*rejected_recovery_candidates, 0.0);
  EXPECT_DOUBLE_EQ(*rejected_recovery_tracked, 0.0);
  EXPECT_EQ(
    rejected_recovery.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(rejected_recovery.finalize_call_count, 0U);
  EXPECT_TRUE(
    manager.readOnlyStructuralAudit().
    adaptive_zero_map_reseed_latched[kCameraId]);
  manager.joint_update_config.max_residual_rmse =
    accepted_residual_rmse;

  const Eigen::Vector3d continued_empty_translation(
    200.0, 0.0, 0.0);
  state.pos_end = continued_empty_translation;
  propagated = state;
  CameraObservationBatch repeated_batch =
    sealSingleCandidate(queue, 78.0, kCameraId, reference);
  testing::internal::CaptureStdout();
  const auto repeated = runTranslatedGeneration(
    manager, *authority, repeated_batch, 308U,
    continued_empty_translation);
  const std::string repeated_output =
    testing::internal::GetCapturedStdout();
  EXPECT_EQ(
    repeated.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(repeated.finalize_call_count, 0U);
  EXPECT_EQ(
    repeated_output.find("[ADAPTIVE_VIO_MAP_MAINTENANCE]"),
    std::string::npos);
  EXPECT_TRUE(
    manager.readOnlyStructuralAudit().
    adaptive_zero_map_reseed_latched[kCameraId]);

  state.pos_end = translation;
  propagated = state;
  CameraObservationBatch recovery_batch =
    sealSingleCandidate(queue, 79.0, kCameraId, reference);
  const auto recovery = runTranslatedGeneration(
    manager, *authority, recovery_batch, 309U,
    translation);
  EXPECT_EQ(
    recovery.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
  EXPECT_TRUE(recovery.solver_called);
  EXPECT_EQ(recovery.solver_call_count, 1U);
  EXPECT_FALSE(
    manager.readOnlyStructuralAudit().
    adaptive_zero_map_reseed_latched[kCameraId]);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  AcceptedWeakSupportWinnerDoesNotBypassSharedReseedCooldown)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  manager.adaptive_policy_config.on_hysteresis_frames = 2U;
  manager.adaptive_policy_config.total_candidate_budget = 12U;
  manager.adaptive_policy_config.minimum_candidate_budget = 1U;
  manager.adaptive_policy_config.total_patch_budget = 12U;
  manager.adaptive_policy_config.minimum_patch_budget = 1U;
  initializeManager(manager, state, propagated);
  constexpr std::uint8_t kCameraId = 3U;
  const cv::Mat reference = texturedImage();
  seedCameraMap(
    manager, kCameraId, reference,
    Eigen::Vector3d::Zero(), 12U);
  CameraObservationQueue queue;
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  CameraObservationBatch initialization_batch =
    sealSingleCandidate(queue, 90.0, kCameraId, reference);
  ASSERT_EQ(
    runGenerationWithPointLimit(
      manager, *authority, initialization_batch,
      450U, 12U).outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);

  const Eigen::Vector3d translation(100.0, 0.0, 0.0);
  state.pos_end = translation;
  propagated = state;
  livo::AdaptiveVisualRuntimeAudit reseed;
  for (std::size_t offset = 1U; offset <= 3U; ++offset)
  {
    CameraObservationBatch batch = sealSingleCandidate(
      queue, 90.0 + static_cast<double>(offset),
      kCameraId, reference);
    reseed = runTranslatedGeneration(
      manager, *authority, batch, 450U + offset,
      translation, 12U);
  }
  ASSERT_EQ(
    reseed.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kVisualMapReseed);
  ASSERT_TRUE(
    manager.readOnlyStructuralAudit().
    adaptive_zero_map_reseed_latched[kCameraId]);

  CameraObservationBatch weak_winner_batch =
    sealSingleCandidate(queue, 94.0, kCameraId, reference);
  const auto weak_winner = runTranslatedGeneration(
    manager, *authority, weak_winner_batch, 454U,
    translation, 12U);
  ASSERT_EQ(
    weak_winner.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
  ASSERT_TRUE(weak_winner.selected_camera_id.has_value());
  EXPECT_EQ(*weak_winner.selected_camera_id, kCameraId);
  EXPECT_EQ(weak_winner.solver_call_count, 1U);
  ASSERT_TRUE(weak_winner.generation_update_receipt.has_value());
  EXPECT_TRUE(
    weak_winner.generation_update_receipt->visual_update_accepted);
  EXPECT_GT(
    weak_winner.generation_update_receipt->solver_tracked_points,
    0U);
  EXPECT_LT(
    weak_winner.generation_update_receipt->solver_tracked_points,
    manager.adaptive_policy_config.minimum_inliers);

  const auto structure = manager.readOnlyStructuralAudit();
  EXPECT_TRUE(
    structure.adaptive_zero_map_reseed_latched[kCameraId]);
  EXPECT_EQ(
    structure.adaptive_zero_map_reentry_healthy_scouts[kCameraId],
    1U);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  EmptyReseedGenerationReaccumulatesEligibilityWithoutStorm)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  manager.adaptive_policy_config.on_hysteresis_frames = 2U;
  initializeManager(manager, state, propagated);
  constexpr std::uint8_t kCameraId = 0U;
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, kCameraId, reference);
  CameraObservationQueue queue;
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  CameraObservationBatch initialization_batch =
    sealSingleCandidate(queue, 100.0, kCameraId, reference);
  ASSERT_EQ(
    runGeneration(
      manager, *authority, initialization_batch, 500U).outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);

  const Eigen::Vector3d first_translation(100.0, 0.0, 0.0);
  state.pos_end = first_translation;
  propagated = state;
  livo::AdaptiveVisualRuntimeAudit first_reseed;
  for (std::size_t offset = 1U; offset <= 3U; ++offset)
  {
    CameraObservationBatch batch = sealSingleCandidate(
      queue, 100.0 + static_cast<double>(offset),
      kCameraId, reference);
    first_reseed = runTranslatedGeneration(
      manager, *authority, batch, 500U + offset,
      first_translation);
  }
  ASSERT_EQ(
    first_reseed.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kVisualMapReseed);
  ASSERT_TRUE(
    manager.readOnlyStructuralAudit().
    adaptive_zero_map_reseed_latched[kCameraId]);

  const Eigen::Vector3d second_translation(200.0, 0.0, 0.0);
  state.pos_end = second_translation;
  propagated = state;
  for (std::size_t offset = 1U; offset <= 4U; ++offset)
  {
    CameraObservationBatch batch = sealSingleCandidate(
      queue, 104.0 + static_cast<double>(offset),
      kCameraId, reference);
    const auto audit = runTranslatedGeneration(
      manager, *authority, batch, 503U + offset,
      second_translation);
    EXPECT_FALSE(audit.solver_called);
    EXPECT_EQ(audit.finalize_call_count, 0U);
  }

  CameraObservationBatch empty_batch =
    sealSingleCandidate(queue, 109.0, kCameraId, reference);
  const auto before_empty = manager.readOnlyStructuralAudit();
  testing::internal::CaptureStdout();
  const auto empty_attempt = runTranslatedGeneration(
    manager, *authority, empty_batch, 508U,
    second_translation, 0U);
  const std::string empty_output =
    testing::internal::GetCapturedStdout();
  EXPECT_EQ(
    empty_attempt.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_FALSE(empty_attempt.solver_called);
  EXPECT_EQ(empty_attempt.solver_call_count, 0U);
  EXPECT_EQ(empty_attempt.finalize_call_count, 1U);
  EXPECT_FALSE(empty_attempt.generation_update_receipt.has_value());
  auto structure = manager.readOnlyStructuralAudit();
  EXPECT_TRUE(
    structure.adaptive_zero_map_reseed_latched[kCameraId]);
  EXPECT_EQ(
    structure.adaptive_zero_map_reentry_healthy_scouts[kCameraId],
    0U);
  EXPECT_EQ(
    structure.legacy_feature_point_counts[kCameraId],
    before_empty.legacy_feature_point_counts[kCameraId]);
  EXPECT_EQ(
    countOccurrences(
      empty_output, "[ADAPTIVE_VIO_MAP_MAINTENANCE]"),
    1U);
  const std::string empty_metric =
    mapMaintenanceMetricRow(empty_output, 508U, kCameraId);
  ASSERT_FALSE(empty_metric.empty());
  EXPECT_NE(
    empty_metric.find(
      "schema_version=2 runtime_path=adaptive_single"),
    std::string::npos);
  EXPECT_NE(
    empty_metric.find(
      "map_action=lio_anchored_reseed seed_new_points=1 "
      "update_existing_points=0"),
    std::string::npos);
  ASSERT_TRUE(
    structuredDoubleField(
      empty_metric, "attempted").has_value());
  ASSERT_TRUE(
    structuredDoubleField(
      empty_metric, "succeeded").has_value());
  ASSERT_TRUE(
    structuredDoubleField(
      empty_metric, "generated_points").has_value());
  EXPECT_DOUBLE_EQ(
    *structuredDoubleField(empty_metric, "attempted"), 1.0);
  EXPECT_DOUBLE_EQ(
    *structuredDoubleField(empty_metric, "succeeded"), 0.0);
  EXPECT_DOUBLE_EQ(
    *structuredDoubleField(empty_metric, "generated_points"),
    0.0);

  for (std::size_t offset = 1U; offset <= 4U; ++offset)
  {
    CameraObservationBatch batch = sealSingleCandidate(
      queue, 109.0 + static_cast<double>(offset),
      kCameraId, reference);
    const auto audit = runTranslatedGeneration(
      manager, *authority, batch, 508U + offset,
      second_translation);
    EXPECT_FALSE(audit.solver_called);
    EXPECT_EQ(audit.finalize_call_count, 0U);
    EXPECT_FALSE(audit.generation_update_receipt.has_value());
    structure = manager.readOnlyStructuralAudit();
    if (offset < 2U)
    {
      EXPECT_TRUE(
        structure.adaptive_zero_map_reseed_latched[kCameraId]);
      EXPECT_EQ(
        structure.
        adaptive_zero_map_reentry_healthy_scouts[kCameraId],
        offset);
    }
    else
    {
      EXPECT_FALSE(
        structure.adaptive_zero_map_reseed_latched[kCameraId]);
      EXPECT_EQ(
        structure.
        adaptive_zero_map_reentry_healthy_scouts[kCameraId],
        offset - 2U);
    }
  }

  CameraObservationBatch retry_batch =
    sealSingleCandidate(queue, 114.0, kCameraId, reference);
  const auto retry = runTranslatedGeneration(
    manager, *authority, retry_batch, 513U,
    second_translation);
  EXPECT_EQ(
    retry.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kVisualMapReseed);
  EXPECT_FALSE(retry.solver_called);
  EXPECT_EQ(retry.finalize_call_count, 1U);
  ASSERT_TRUE(retry.generation_update_receipt.has_value());
  EXPECT_EQ(
    retry.generation_update_receipt->map_action,
    livo::AdaptiveVisualMapAction::kLidarAnchoredReseed);
  structure = manager.readOnlyStructuralAudit();
  EXPECT_TRUE(
    structure.adaptive_zero_map_reseed_latched[kCameraId]);
  EXPECT_EQ(
    structure.adaptive_zero_map_reentry_healthy_scouts[kCameraId],
    0U);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  LatchedLowSupportMapRearmsAfterTwoHysteresisWindows)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  manager.adaptive_policy_config.on_hysteresis_frames = 2U;
  initializeManager(manager, state, propagated);
  constexpr std::uint8_t kCameraId = 1U;
  const cv::Mat reference = texturedImage();
  seedCameraMap(manager, kCameraId, reference);
  CameraObservationQueue queue;
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  CameraObservationBatch initialization_batch =
    sealSingleCandidate(queue, 80.0, kCameraId, reference);
  ASSERT_EQ(
    runGeneration(
      manager, *authority, initialization_batch, 400U).outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);

  const Eigen::Vector3d first_translation(100.0, 0.0, 0.0);
  state.pos_end = first_translation;
  propagated = state;
  livo::AdaptiveVisualRuntimeAudit first_reseed;
  for (std::size_t offset = 1U; offset <= 3U; ++offset)
  {
    CameraObservationBatch batch = sealSingleCandidate(
      queue, 80.0 + static_cast<double>(offset),
      kCameraId, reference);
    first_reseed = runTranslatedGeneration(
      manager, *authority, batch, 400U + offset,
      first_translation);
  }
  ASSERT_EQ(
    first_reseed.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kVisualMapReseed);
  ASSERT_TRUE(
    manager.readOnlyStructuralAudit().
    adaptive_zero_map_reseed_latched[kCameraId]);

  const Eigen::Vector3d second_translation(200.0, 0.0, 0.0);
  state.pos_end = second_translation;
  propagated = state;
  for (std::size_t offset = 1U; offset <= 4U; ++offset)
  {
    CameraObservationBatch batch = sealSingleCandidate(
      queue, 84.0 + static_cast<double>(offset),
      kCameraId, reference);
    const auto audit = runTranslatedGeneration(
      manager, *authority, batch, 403U + offset,
      second_translation);
    EXPECT_EQ(
      audit.outcome,
      livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
    EXPECT_FALSE(audit.solver_called);
    EXPECT_EQ(audit.solver_call_count, 0U);
    EXPECT_EQ(audit.finalize_call_count, 0U);

    const auto structure = manager.readOnlyStructuralAudit();
    if (offset < 2U)
    {
      EXPECT_TRUE(
        structure.adaptive_zero_map_reseed_latched[kCameraId]);
      EXPECT_EQ(
        structure.
        adaptive_zero_map_reentry_healthy_scouts[kCameraId],
        offset);
    }
    else
    {
      EXPECT_FALSE(
        structure.adaptive_zero_map_reseed_latched[kCameraId]);
      EXPECT_EQ(
        structure.
        adaptive_zero_map_reentry_healthy_scouts[kCameraId],
        offset - 2U);
    }
  }

  CameraObservationBatch retry_batch = sealSingleCandidate(
    queue, 89.0, kCameraId, reference);
  const auto retry = runTranslatedGeneration(
    manager, *authority, retry_batch, 408U,
    second_translation);
  EXPECT_EQ(
    retry.outcome,
    livo::AdaptiveVisualRuntimeOutcome::kLioOnly);
  EXPECT_EQ(
    retry.lio_only_reason,
    livo::AdaptiveVisualLioOnlyReason::kVisualMapReseed);
  EXPECT_FALSE(retry.solver_called);
  EXPECT_EQ(retry.solver_call_count, 0U);
  EXPECT_EQ(retry.finalize_call_count, 1U);
  ASSERT_TRUE(retry.generation_update_receipt.has_value());
  EXPECT_FALSE(
    retry.generation_update_receipt->visual_update_accepted);
  EXPECT_EQ(
    retry.generation_update_receipt->map_action,
    livo::AdaptiveVisualMapAction::kLidarAnchoredReseed);
  EXPECT_TRUE(
    manager.readOnlyStructuralAudit().
    adaptive_zero_map_reseed_latched[kCameraId]);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  LioAnchoredRecoveryIsSymmetricAcrossPermutedCameraIds)
{
  const cv::Mat reference = texturedImage();
  for (std::uint8_t winner_id = 0U; winner_id < 4U; ++winner_id)
  {
    VIOManager manager;
    StatesGroup state;
    StatesGroup propagated;
    initializeManager(manager, state, propagated);
    CameraObservationQueue queue;
    auto authority =
      queue.issueAdaptiveVisualSelectionAuthority();
    ASSERT_NE(authority, nullptr);

    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      const std::size_t point_limit =
        camera_id == winner_id ? 35U : 12U;
      seedCameraMap(
        manager, camera_id, reference,
        Eigen::Vector3d::Zero(), point_limit);
      CameraObservationBatch initialization_batch =
        sealSingleCandidate(
          queue, 100.0 + static_cast<double>(camera_id),
          camera_id, reference);
      ASSERT_EQ(
        runGenerationWithPointLimit(
          manager, *authority, initialization_batch,
          600U + camera_id, point_limit).outcome,
        livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
    }
    const auto initialized = manager.readOnlyStructuralAudit();

    const std::size_t attempts =
      manager.adaptive_policy_config.on_hysteresis_frames + 1U;
    for (std::size_t attempt = 0U; attempt < attempts; ++attempt)
    {
      constexpr std::array<std::array<std::uint8_t, 4U>, 4U>
        kPushOrders{{
          {{0U, 1U, 2U, 3U}},
          {{1U, 2U, 3U, 0U}},
          {{2U, 3U, 0U, 1U}},
          {{3U, 0U, 1U, 2U}},
        }};
      CameraObservationBatch batch = sealFourHealthyCandidatesAt(
        queue, 110.0 + static_cast<double>(attempt),
        kPushOrders[
          (attempt + winner_id) % kPushOrders.size()]);
      const auto audit = runGeneration(
        manager, *authority, batch, 604U + attempt);
      ASSERT_EQ(
        audit.outcome,
        livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
      ASSERT_TRUE(audit.selected_camera_id.has_value());
      EXPECT_EQ(*audit.selected_camera_id, winner_id);
      EXPECT_EQ(audit.solver_call_count, 1U);

      const auto structure = manager.readOnlyStructuralAudit();
      for (std::size_t camera_id = 0U; camera_id < 4U; ++camera_id)
      {
        if (camera_id == winner_id)
        {
          EXPECT_EQ(
            structure.
            adaptive_zero_map_reentry_healthy_scouts[camera_id],
            0U);
          continue;
        }
        if (attempt + 1U < attempts)
        {
          EXPECT_EQ(
            structure.
            adaptive_zero_map_reentry_healthy_scouts[camera_id],
            attempt + 1U);
          EXPECT_FALSE(
            structure.adaptive_zero_map_reseed_latched[camera_id]);
        }
        else
        {
          EXPECT_EQ(
            structure.
            adaptive_zero_map_reentry_healthy_scouts[camera_id],
            0U);
          EXPECT_TRUE(
            structure.adaptive_zero_map_reseed_latched[camera_id]);
          EXPECT_GT(
            structure.legacy_feature_point_counts[camera_id],
            initialized.legacy_feature_point_counts[camera_id]);
        }
      }
      EXPECT_EQ(
        audit.finalize_call_count,
        attempt + 1U < attempts ? 1U : 4U);
    }
  }
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  VisualWinnerAllowsLioAnchoredSideMapMaintenanceWithoutSecondSolver)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  initializeManager(manager, state, propagated);
  const cv::Mat reference = texturedImage();
  seedCameraMap(
    manager, 2U, reference, Eigen::Vector3d::Zero(), 12U);
  CameraObservationQueue queue;
  auto authority =
    queue.issueAdaptiveVisualSelectionAuthority();
  ASSERT_NE(authority, nullptr);

  CameraObservationBatch back_initialization =
    sealSingleCandidate(queue, 90.0, 2U, reference);
  ASSERT_EQ(
    runGenerationWithPointLimit(
      manager, *authority, back_initialization, 500U,
      12U).outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);

  seedCameraMap(manager, 0U, reference);
  CameraObservationBatch front_initialization =
    sealSingleCandidate(queue, 91.0, 0U, reference);
  ASSERT_EQ(
    runGeneration(
      manager, *authority, front_initialization, 501U).outcome,
    livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);

  const auto initialized = manager.readOnlyStructuralAudit();
  ASSERT_EQ(
    initialized.adaptive_visual_map_lifecycle[0U],
    livo::AdaptiveVisualMapLifecycle::kInitialized);
  ASSERT_EQ(
    initialized.adaptive_visual_map_lifecycle[2U],
    livo::AdaptiveVisualMapLifecycle::kInitialized);
  const std::size_t attempts =
    manager.adaptive_policy_config.on_hysteresis_frames + 1U;
  for (std::size_t attempt = 0U; attempt < attempts; ++attempt)
  {
    CameraObservationBatch batch = sealTwoCandidateImages(
      queue, 92.0 + static_cast<double>(attempt),
      reference, reference);
    testing::internal::CaptureStdout();
    const auto audit = runGeneration(
      manager, *authority, batch, 502U + attempt);
    const std::string output =
      testing::internal::GetCapturedStdout();
    ASSERT_EQ(
      audit.outcome,
      livo::AdaptiveVisualRuntimeOutcome::kVisualAccepted);
    ASSERT_TRUE(audit.selected_camera_id.has_value());
    EXPECT_EQ(*audit.selected_camera_id, 0U);
    ASSERT_TRUE(audit.fusion_camera_id.has_value());
    EXPECT_EQ(*audit.fusion_camera_id, 0U);
    EXPECT_TRUE(audit.solver_called);
    EXPECT_EQ(audit.solver_call_count, 1U);
    ASSERT_TRUE(audit.generation_update_receipt.has_value());
    EXPECT_TRUE(
      audit.generation_update_receipt->visual_update_accepted);
    EXPECT_EQ(
      audit.generation_update_receipt->map_action,
      livo::AdaptiveVisualMapAction::kAcceptedVisualUpdate);
    const std::string side_row = cameraMetricRow(output, 2U);
    ASSERT_FALSE(side_row.empty());
    const auto side_inliers =
      structuredDoubleField(side_row, "inliers");
    ASSERT_TRUE(side_inliers.has_value());
    EXPECT_GT(*side_inliers, 0.0);
    EXPECT_LT(
      *side_inliers,
      static_cast<double>(
        manager.adaptive_policy_config.minimum_inliers));
    const auto structure = manager.readOnlyStructuralAudit();
    if (attempt + 1U < attempts)
    {
      EXPECT_EQ(audit.finalize_call_count, 1U);
      EXPECT_EQ(
        output.find("[ADAPTIVE_VIO_MAP_MAINTENANCE]"),
        std::string::npos);
      EXPECT_EQ(
        structure.
        adaptive_zero_map_reentry_healthy_scouts[2U],
        attempt + 1U);
      EXPECT_FALSE(
        structure.adaptive_zero_map_reseed_latched[2U]);
    }
    else
    {
      EXPECT_EQ(audit.finalize_call_count, 2U);
      EXPECT_NE(
        output.find(
          "runtime_path=adaptive_single generation=507 camera_id=2 "
          "map_action=lio_anchored_reseed"),
        std::string::npos);
      EXPECT_EQ(
        structure.
        adaptive_zero_map_reentry_healthy_scouts[2U],
        0U);
      EXPECT_TRUE(
        structure.adaptive_zero_map_reseed_latched[2U]);
      EXPECT_GT(
        structure.legacy_feature_point_counts[2U],
        initialized.legacy_feature_point_counts[2U]);
    }
  }
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  GenericAdaptiveBootstrapPrecedesRejectedSolverTransaction)
{
  VIOManager manager;
  StatesGroup state;
  StatesGroup propagated;
  manager.adaptive_policy_config.minimum_selected_cameras = 2U;
  manager.adaptive_policy_config.maximum_selected_cameras = 2U;
  manager.joint_update_config.min_joint_cameras = 2U;
  initializeManager(
    manager, state, propagated, true, false);

  const auto before = manager.readOnlyStructuralAudit();
  CameraObservationQueue queue;
  CameraObservationBatch batch =
    sealFourCandidates(queue, false);

  testing::internal::CaptureStdout();
  processAuthorized(manager, batch, 400U, nullptr);
  const std::string output =
    testing::internal::GetCapturedStdout();
  const auto after = manager.readOnlyStructuralAudit();

  EXPECT_EQ(
    after.last_joint_solver.reason,
    before.last_joint_solver.reason);
  EXPECT_EQ(
    after.last_joint_solver.accepted,
    before.last_joint_solver.accepted);

  ASSERT_TRUE(before.active_state_available);
  ASSERT_TRUE(after.active_state_available);
  EXPECT_EQ(
    std::memcmp(
      before.active_state.data(), after.active_state.data(),
      sizeof(before.active_state)),
    0);
  ASSERT_TRUE(before.propagated_state_available);
  ASSERT_TRUE(after.propagated_state_available);
  EXPECT_EQ(
    std::memcmp(
      before.propagated_state.data(),
      after.propagated_state.data(),
      sizeof(before.propagated_state)),
    0);

  std::size_t selected_count = 0U;
  std::size_t seeded_count = 0U;
  for (std::size_t camera_id = 0U; camera_id < 4U;
       ++camera_id)
  {
    const std::string row =
      cameraMetricRow(output, camera_id);
    ASSERT_FALSE(row.empty());
    const auto selected =
      structuredDoubleField(row, "selected");
    const auto enabled =
      structuredDoubleField(row, "enabled");
    ASSERT_TRUE(selected.has_value());
    ASSERT_TRUE(enabled.has_value());
    EXPECT_DOUBLE_EQ(*selected, *enabled);

    const bool route_authorized = *enabled == 1.0;
    const bool map_grew =
      after.legacy_feature_point_counts[camera_id] >
      before.legacy_feature_point_counts[camera_id];
    EXPECT_EQ(map_grew, route_authorized);
    if (!route_authorized)
    {
      EXPECT_EQ(
        after.legacy_feature_point_counts[camera_id],
        before.legacy_feature_point_counts[camera_id]);
    }
    EXPECT_FALSE(after.adaptive_camera_initialized[camera_id]);
    EXPECT_EQ(
      after.adaptive_visual_map_lifecycle[camera_id],
      route_authorized ?
      livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated :
      livo::AdaptiveVisualMapLifecycle::kEmpty);
    selected_count += route_authorized ? 1U : 0U;
    seeded_count += map_grew ? 1U : 0U;
  }
  EXPECT_EQ(selected_count, 2U);
  EXPECT_EQ(seeded_count, 2U);

  manager.joint_update_config.min_tracked_points_per_camera =
    1000U;
  manager.joint_update_config.min_joint_tracked_points = 1000U;
  CameraObservationQueue rejected_queue;
  CameraObservationBatch rejected_batch =
    sealFourCandidates(rejected_queue, false);
  const auto before_rejected = manager.readOnlyStructuralAudit();
  processAuthorized(manager, rejected_batch, 401U, nullptr);
  const auto after_rejected = manager.readOnlyStructuralAudit();

  EXPECT_FALSE(after_rejected.last_joint_solver.accepted);
  EXPECT_EQ(
    after_rejected.last_joint_solver.reason,
    "insufficient_tracking");
  EXPECT_EQ(
    after_rejected.legacy_feature_point_counts,
    before_rejected.legacy_feature_point_counts);
  EXPECT_EQ(
    after_rejected.adaptive_visual_map_lifecycle,
    before_rejected.adaptive_visual_map_lifecycle);
  EXPECT_EQ(
    after_rejected.adaptive_camera_initialized,
    before_rejected.adaptive_camera_initialized);
}

TEST(
  AdaptiveVioRuntimeAuthorityTest,
  DisabledAndNonMaxFullFourPathsRemainOutsideCandidateRuntime)
{
  for (const std::pair<bool, bool> mode :
       std::array<std::pair<bool, bool>, 2U>{{
         {false, false}, {true, false}}})
  {
    VIOManager manager;
    StatesGroup state;
    StatesGroup propagated;
    initializeManager(
      manager, state, propagated, mode.first, mode.second);
    const cv::Mat reference = texturedImage();
    for (std::uint8_t camera_id = 0U; camera_id < 4U;
         ++camera_id)
    {
      seedCameraMap(manager, camera_id, reference);
    }
    CameraObservationQueue queue;
    CameraObservationBatch batch =
      sealFourCandidates(queue, false);

    processAuthorized(manager, batch, 260U, nullptr);

    const auto & runtime_audit =
      manager.adaptiveVisualRuntimeAudit();
    EXPECT_EQ(
      runtime_audit.outcome,
      livo::AdaptiveVisualRuntimeOutcome::kNotApplicable);
    EXPECT_FALSE(runtime_audit.solver_called);
    EXPECT_EQ(runtime_audit.solver_call_count, 0U);
    const auto structure = manager.readOnlyStructuralAudit();
    EXPECT_NE(
      structure.last_joint_solver.reason,
      "adaptive_runtime_lio_only");
    EXPECT_NE(
      structure.last_joint_solver.reason,
      "incomplete_camera_batch");
    EXPECT_GE(
      structure.last_joint_solver.contributing_cameras, 2U);
  }
}

}  // namespace
