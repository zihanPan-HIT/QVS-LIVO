#include "camera_model_factory.h"
#include "vio.h"

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

namespace
{

livo::CameraModelParameters pinholeParameters(
  const std::uint8_t camera_id)
{
  livo::CameraModelParameters parameters;
  parameters.model = "Pinhole";
  parameters.width = 640;
  parameters.height = 480;
  parameters.scale = 1.0;
  parameters.fx = 600.0 + static_cast<double>(camera_id);
  parameters.fy = 601.0 + static_cast<double>(camera_id);
  parameters.cx = 320.0;
  parameters.cy = 240.0;
  return parameters;
}

M3D quarterTurnAboutZ(const int quarter_turns)
{
  const double angle =
    static_cast<double>(quarter_turns) * M_PI / 2.0;
  return Eigen::AngleAxisd(
    angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

}  // namespace

TEST(
  VioCameraRegistryTest,
  CrossCameraMigrationShadowIsOffWithoutStructuralOutput)
{
  VIOManager manager;
  livo::CrossCameraMigrationShadowInitializationConfig config;

  manager.initializeCrossCameraMigrationShadow(config);

  const livo::CrossCameraMigrationShadowInitAudit & audit =
    manager.crossCameraMigrationShadowInitAudit();
  EXPECT_FALSE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(audit.reason, "disabled");
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_TRUE(std::all_of(
    audit.cameras.begin(),
    audit.cameras.end(),
    [](const auto & camera) {return !camera.has_value();}));
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  VioCameraRegistryTest,
  FourCameraShadowOwnsIdSortedPinholeRigAndReportsTic)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  constexpr std::array<std::uint64_t, 4U>
    kExpectedModelRevisions{{
      15898773398903609459ULL,
      15257817696514483107ULL,
      5761291065440577945ULL,
      9512502533081254680ULL,
    }};
  const std::array<M3D, 4U> rotations_camera_from_lidar{{
    quarterTurnAboutZ(0),
    quarterTurnAboutZ(1),
    quarterTurnAboutZ(2),
    quarterTurnAboutZ(-1),
  }};
  const std::array<V3D, 4U> positions_camera_from_lidar{{
    V3D(1.0, 0.0, 0.0),
    V3D(0.0, 2.0, 0.0),
    V3D(-3.0, 0.0, 0.0),
    V3D(0.0, -4.0, 0.0),
  }};

  VIOManager manager;
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());
  std::array<livo::CameraModelParameters, 4U> parameters;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    parameters[camera_id] = pinholeParameters(camera_id);
    manager.registerCamera(
      camera_id,
      kNames[camera_id],
      livo::createCameraModel(parameters[camera_id]),
      rotations_camera_from_lidar[camera_id],
      positions_camera_from_lidar[camera_id]);
  }

  livo::CrossCameraMigrationShadowInitializationConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  constexpr std::array<std::uint8_t, 4U> kInputOrder{
    {3U, 0U, 2U, 1U}};
  for (std::size_t index = 0U; index < kInputOrder.size(); ++index)
  {
    const std::uint8_t camera_id = kInputOrder[index];
    config.cameras[index].camera_id = camera_id;
    config.cameras[index].camera_name = kNames[camera_id];
    config.cameras[index].projection = parameters[camera_id];
    config.cameras[index].projection_calibration_present = true;
    config.cameras[index].extrinsic_calibration_present = true;
  }

  manager.initializeCrossCameraMigrationShadow(config);

  const auto & audit = manager.crossCameraMigrationShadowInitAudit();
  ASSERT_TRUE(audit.enabled);
  ASSERT_TRUE(audit.ready);
  EXPECT_EQ(audit.reason, "shadow_owner_ready");
  EXPECT_NE(audit.rig_revision, 0U);
  EXPECT_TRUE(audit.rig_fingerprint.has_value());
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    ASSERT_TRUE(audit.cameras[camera_id].has_value());
    const auto & camera = *audit.cameras[camera_id];
    EXPECT_EQ(camera.camera_id, camera_id);
    EXPECT_EQ(camera.camera_name, kNames[camera_id]);
    EXPECT_EQ(
      camera.model_revision,
      kExpectedModelRevisions[camera_id]);
    EXPECT_TRUE(camera.rotation_imu_from_camera.isApprox(
      rotations_camera_from_lidar[camera_id].transpose(),
      1.0e-15));
    EXPECT_TRUE(camera.position_imu_from_camera.isApprox(
      V3D(-static_cast<double>(camera_id + 1U), 0.0, 0.0),
      1.0e-15));
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  VioCameraRegistryTest,
  FourCameraShadowDoesNotConstructBeforeImuLidarExtrinsic)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  VIOManager manager;
  livo::CrossCameraMigrationShadowInitializationConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const auto parameters = pinholeParameters(camera_id);
    manager.registerCamera(
      camera_id, kNames[camera_id],
      livo::createCameraModel(parameters),
      M3D::Identity(), V3D::Zero());
    config.cameras[camera_id].camera_id = camera_id;
    config.cameras[camera_id].camera_name = kNames[camera_id];
    config.cameras[camera_id].projection = parameters;
  }

  manager.initializeCrossCameraMigrationShadow(config);

  const auto & audit = manager.crossCameraMigrationShadowInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(audit.reason, "missing_imu_lidar_extrinsic");
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  EXPECT_TRUE(std::all_of(
    audit.cameras.begin(),
    audit.cameras.end(),
    [](const auto & camera) {return !camera.has_value();}));
}

TEST(
  VioCameraRegistryTest,
  MissingProjectionCalibrationEvidenceFailsClosedAndLeavesVioCamerasAlive)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  VIOManager manager;
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());
  livo::CrossCameraMigrationShadowInitializationConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const auto parameters = pinholeParameters(camera_id);
    manager.registerCamera(
      camera_id, kNames[camera_id],
      livo::createCameraModel(parameters),
      M3D::Identity(), V3D::Zero());
    config.cameras[camera_id].camera_id = camera_id;
    config.cameras[camera_id].camera_name = kNames[camera_id];
    config.cameras[camera_id].projection = parameters;
  }

  manager.initializeCrossCameraMigrationShadow(config);

  const auto & audit = manager.crossCameraMigrationShadowInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(
    audit.reason, "missing_camera_projection_calibration");
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_FALSE(audit.cameras[camera_id].has_value());
    EXPECT_NE(manager.cameraModel(camera_id), nullptr);
    EXPECT_EQ(manager.cameraName(camera_id), kNames[camera_id]);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  VioCameraRegistryTest,
  MissingExtrinsicCalibrationEvidenceFailsClosedAndLeavesVioCamerasAlive)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  VIOManager manager;
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());
  livo::CrossCameraMigrationShadowInitializationConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const auto parameters = pinholeParameters(camera_id);
    manager.registerCamera(
      camera_id, kNames[camera_id],
      livo::createCameraModel(parameters),
      M3D::Identity(), V3D::Zero());
    config.cameras[camera_id].camera_id = camera_id;
    config.cameras[camera_id].camera_name = kNames[camera_id];
    config.cameras[camera_id].projection = parameters;
    config.cameras[camera_id].projection_calibration_present = true;
    config.cameras[camera_id].extrinsic_calibration_present =
      camera_id != 3U;
  }

  manager.initializeCrossCameraMigrationShadow(config);

  const auto & audit = manager.crossCameraMigrationShadowInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(
    audit.reason, "missing_camera_extrinsic_calibration");
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_FALSE(audit.cameras[camera_id].has_value());
    EXPECT_NE(manager.cameraModel(camera_id), nullptr);
    EXPECT_EQ(manager.cameraName(camera_id), kNames[camera_id]);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(
  VioCameraRegistryTest,
  InvalidRigExtrinsicFailsClosedAndLeavesRegisteredVioCamerasAlive)
{
  constexpr std::array<const char *, 4U> kNames{
    {"front", "left", "back", "right"}};
  VIOManager manager;
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());
  livo::CrossCameraMigrationShadowInitializationConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const auto parameters = pinholeParameters(camera_id);
    M3D rotation = M3D::Identity();
    if (camera_id == 2U)
    {
      rotation(0, 0) = 2.0;
    }
    manager.registerCamera(
      camera_id, kNames[camera_id],
      livo::createCameraModel(parameters),
      rotation, V3D::Zero());
    config.cameras[camera_id].camera_id = camera_id;
    config.cameras[camera_id].camera_name = kNames[camera_id];
    config.cameras[camera_id].projection = parameters;
    config.cameras[camera_id].projection_calibration_present = true;
    config.cameras[camera_id].extrinsic_calibration_present = true;
  }

  EXPECT_NO_THROW(
    manager.initializeCrossCameraMigrationShadow(config));

  const auto & audit = manager.crossCameraMigrationShadowInitAudit();
  EXPECT_TRUE(audit.enabled);
  EXPECT_FALSE(audit.ready);
  EXPECT_EQ(audit.reason, "invalid_rig_configuration");
  EXPECT_EQ(audit.rig_revision, 0U);
  EXPECT_FALSE(audit.rig_fingerprint.has_value());
  EXPECT_TRUE(std::all_of(
    audit.cameras.begin(),
    audit.cameras.end(),
    [](const auto & camera) {return !camera.has_value();}));
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_NE(manager.cameraModel(camera_id), nullptr);
    EXPECT_EQ(manager.cameraName(camera_id), kNames[camera_id]);
  }
  EXPECT_FALSE(audit.production_ready);
  EXPECT_FALSE(audit.solver_applied);
}

TEST(VioCameraRegistryTest, OwnsRegisteredCameraUntilManagerDestruction)
{
  {
    livo::CameraModelParameters parameters;
    parameters.model = "Pinhole";
    parameters.width = 640;
    parameters.height = 480;
    parameters.fx = 603.1;
    parameters.fy = 603.1;
    parameters.cx = 330.1;
    parameters.cy = 246.6;
    vk::AbstractCamera *camera = livo::createCameraModel(parameters);
    ASSERT_NE(camera, nullptr);

    VIOManager manager;
    manager.setImuToLidarExtrinsic(V3D::Zero(), M3D::Identity());
    manager.registerCamera(
      0U, "front", camera, M3D::Identity(), V3D::Zero());

    ASSERT_NE(manager.cameraModel(0U), nullptr);
    EXPECT_EQ(manager.cameraName(0U), "front");
    EXPECT_TRUE(manager.activateCamera(0U));
  }
}

TEST(
  VioCameraRegistryTest,
  SameSizeFxFyCxCyMustMatchTheRegisteredVioCamera)
{
  constexpr std::array<const char *, 4U> kNames{{
    "front", "left", "back", "right"}};
  for (std::size_t intrinsic_index = 0U;
       intrinsic_index < 4U; ++intrinsic_index)
  {
    VIOManager manager;
    manager.setImuToLidarExtrinsic(
      V3D::Zero(), M3D::Identity());
    livo::CrossCameraMigrationShadowInitializationConfig config;
    config.enabled = true;
    config.four_camera_mode = true;
    for (std::uint8_t camera_id = 0U;
         camera_id < 4U; ++camera_id)
    {
      const auto parameters = pinholeParameters(camera_id);
      manager.registerCamera(
        camera_id, kNames[camera_id],
        livo::createCameraModel(parameters),
        M3D::Identity(), V3D::Zero());
      config.cameras[camera_id].camera_id = camera_id;
      config.cameras[camera_id].camera_name = kNames[camera_id];
      config.cameras[camera_id].projection = parameters;
      config.cameras[camera_id].
        projection_calibration_present = true;
      config.cameras[camera_id].
        extrinsic_calibration_present = true;
    }
    auto & mismatched = config.cameras[2U].projection;
    switch (intrinsic_index)
    {
      case 0U:
        mismatched.fx += 0.25;
        break;
      case 1U:
        mismatched.fy += 0.25;
        break;
      case 2U:
        mismatched.cx += 0.25;
        break;
      default:
        mismatched.cy += 0.25;
        break;
    }

    manager.initializeCrossCameraMigrationShadow(config);

    EXPECT_FALSE(
      manager.crossCameraMigrationShadowInitAudit().ready);
    EXPECT_EQ(
      manager.crossCameraMigrationShadowInitAudit().reason,
      "canonical_camera_model_mismatch");
    EXPECT_EQ(
      manager.crossCameraMigrationShadowInitAudit().
      rig_revision, 0U);
  }
}

TEST(
  VioCameraRegistryTest,
  ShadowDimensionsMustMatchTheRegisteredVioCamera)
{
  constexpr std::array<const char *, 4U> kNames{{
    "front", "left", "back", "right"}};
  VIOManager manager;
  manager.setImuToLidarExtrinsic(
    V3D::Zero(), M3D::Identity());
  livo::CrossCameraMigrationShadowInitializationConfig config;
  config.enabled = true;
  config.four_camera_mode = true;
  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const auto parameters = pinholeParameters(camera_id);
    manager.registerCamera(
      camera_id, kNames[camera_id],
      livo::createCameraModel(parameters),
      M3D::Identity(), V3D::Zero());
    config.cameras[camera_id].camera_id = camera_id;
    config.cameras[camera_id].camera_name = kNames[camera_id];
    config.cameras[camera_id].projection = parameters;
    config.cameras[camera_id].
      projection_calibration_present = true;
    config.cameras[camera_id].
      extrinsic_calibration_present = true;
  }
  config.cameras[1U].projection.width = 800;

  manager.initializeCrossCameraMigrationShadow(config);

  EXPECT_FALSE(
    manager.crossCameraMigrationShadowInitAudit().ready);
  EXPECT_EQ(
    manager.crossCameraMigrationShadowInitAudit().reason,
    "canonical_camera_model_mismatch");
}
