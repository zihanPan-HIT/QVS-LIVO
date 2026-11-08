#include "reference_only_migration_shadow.h"
#include "camera_batch_time_transfer.h"
#include "pinhole_camera_projection_adapter.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

namespace
{

static_assert(
  !std::is_convertible_v<
    livo::ReferenceOnlyMigrationShadowRequest,
    livo::MigrationRequest>);
static_assert(
  !std::is_constructible_v<
    livo::MigrationRequest,
    livo::ReferenceOnlyMigrationShadowRequest>);
static_assert(
  !std::is_convertible_v<
    livo::ReferenceOnlyMigrationShadowResult,
    livo::WhitenedPatchContribution>);
static_assert(
  !std::is_constructible_v<
    livo::WhitenedPatchContribution,
    livo::ReferenceOnlyMigrationShadowResult>);

livo::CameraTimeTransferImuSample imuSample(
  const double timestamp_sec)
{
  livo::CameraTimeTransferImuSample sample;
  sample.timestamp_sec = timestamp_sec;
  sample.angular_velocity_radps = Eigen::Vector3d::Zero();
  sample.specific_force_mps2 = Eigen::Vector3d::Zero();
  return sample;
}

livo::CameraBatchTimeTransferBundle targetTimeBundle(
  const double batch_timestamp_sec)
{
  livo::CameraTimeTransferConfig config;
  config.max_abs_transfer_sec = 0.05;
  config.max_imu_gap_sec = 0.01;
  livo::CameraTimeTransfer transfer(config);
  for (int index = 0; index <= 8; ++index)
  {
    EXPECT_EQ(
      transfer.appendImu(
        imuSample(
          batch_timestamp_sec - 0.02 +
          0.005 * static_cast<double>(index))),
      livo::CameraTimeTransferAppendStatus::kAccepted);
  }

  CameraObservationBatch batch;
  batch.present[1U] = true;
  batch.camera_count = 1U;
  batch.partial = true;
  batch.timestamp = batch_timestamp_sec;
  batch.observations[1U].camera_id = 1U;
  batch.observations[1U].timestamp = batch_timestamp_sec;
  batch.observations[1U].image =
    cv::Mat::ones(2, 2, CV_8UC1);

  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = batch_timestamp_sec;
  return livo::buildCameraBatchTimeTransferBundle(
    batch, anchor, 0.0, transfer);
}

std::shared_ptr<const livo::PinholeCameraProjectionAdapter>
cameraModel(const std::uint64_t revision)
{
  livo::PinholeCameraProjectionConfig config;
  config.model = "Pinhole";
  config.width = 640;
  config.height = 480;
  config.fx = 400.0;
  config.fy = 400.0;
  config.cx = 320.0;
  config.cy = 240.0;
  config.model_revision = revision;
  return std::make_shared<
    const livo::PinholeCameraProjectionAdapter>(config);
}

livo::MigrationEndpoint endpoint(
  const std::uint8_t camera_id,
  const std::uint64_t model_revision)
{
  livo::MigrationEndpoint result;
  result.camera_id = camera_id;
  result.camera_model = cameraModel(model_revision);
  result.camera_model_revision = model_revision;
  result.rig_revision = 77U;
  result.rotation_imu_from_camera =
    Eigen::Matrix3d::Identity();
  result.position_imu_from_camera =
    Eigen::Vector3d::Zero();
  return result;
}

livo::ReferenceOnlyMigrationShadowRequest validRequest(
  const livo::CameraBatchTimeTransferBundle & bundle)
{
  EXPECT_TRUE(bundle.validatedReceipt().has_value());
  EXPECT_TRUE(bundle.transfers[1U].has_value());
  livo::ReferenceOnlyMigrationShadowRequest request(
    livo::ReferenceOnlyMigrationTargetTime(
      *bundle.validatedReceipt(), bundle.transfers[1U]));

  request.source_support_key.plane = {9U, 2U};
  request.source_support_key.cell = {3, 4, 0U};
  request.source_geometry_version = {{9U, 2U}, 5U};
  request.source_endpoint = endpoint(0U, 101U);
  request.source_pose.exposure_timestamp_sec =
    bundle.transfers[1U].targetTimestampSec() - 0.01;
  request.source_pose.rotation_world_from_imu =
    Eigen::Matrix3d::Identity();
  request.source_pose.position_world_m =
    Eigen::Vector3d::Zero();
  request.source_pose.pose_covariance =
    Eigen::Matrix<double, 6, 6>::Identity() * 1.0e-4;
  request.patch.source_pixel = Eigen::Vector2d(320.0, 240.0);
  request.patch.source_level = 0U;
  request.patch.patch_width = 8;
  request.target_endpoint = endpoint(1U, 102U);
  request.target_image_bounds =
    livo::ReferenceOnlyMigrationTargetImageBounds{640, 480};
  request.plane.support_key = request.source_support_key;
  request.plane.geometry_version =
    request.source_geometry_version;
  request.plane.point_world_m =
    Eigen::Vector3d(0.0, 0.0, 5.0);
  request.plane.point_covariance_m2 =
    Eigen::Matrix3d::Identity() * 1.0e-4;
  request.plane.normal_world = Eigen::Vector3d::UnitZ();
  request.plane.normal_covariance =
    Eigen::Matrix3d::Identity() * 1.0e-4;
  return request;
}

TEST(
  ReferenceOnlyMigrationShadow,
  ProjectsBootstrapPatchWithoutCreatingConsumableInformation)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  const livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);

  const livo::ReferenceOnlyMigrationShadowResult result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.status(),
    livo::ReferenceOnlyMigrationShadowStatus::
      kGeometryReadyVisibilityUnverified);
  EXPECT_EQ(result.migrationStatus(), livo::MigrationStatus::kReady);
  EXPECT_TRUE(result.targetPixel().isApprox(
    Eigen::Vector2d(320.0, 240.0), 1.0e-12));
  EXPECT_NEAR(result.warpConditionNumber(), 1.0, 1.0e-12);
  EXPECT_EQ(result.validProjectedPixelCount(), 64U);
  EXPECT_EQ(
    result.visibility(),
    livo::ReferenceOnlyMigrationVisibility::kUnverified);
  EXPECT_TRUE(result.experimental());
  EXPECT_FALSE(result.consumable());
  EXPECT_FALSE(result.productionReady());
  EXPECT_FALSE(result.solverApplied());
  EXPECT_FALSE(result.acceptanceReceiptIssued());
  EXPECT_FALSE(result.independenceProofIssued());
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsSameCameraSourceAndTarget)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);
  request.source_endpoint = endpoint(1U, 103U);

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.status(),
    livo::ReferenceOnlyMigrationShadowStatus::kRejected);
  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kInvalidSourceModel);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsEntryFromIndependentReceiptAuthority)
{
  const livo::CameraBatchTimeTransferBundle first =
    targetTimeBundle(10.0);
  const livo::CameraBatchTimeTransferBundle foreign =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(first);
  request.target_time.entry = foreign.transfers[1U];

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kInvalidTimeTransfer);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsEntryFromAnotherBatch)
{
  const livo::CameraBatchTimeTransferBundle first =
    targetTimeBundle(10.0);
  const livo::CameraBatchTimeTransferBundle later =
    targetTimeBundle(10.01);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(first);
  request.target_time.entry = later.transfers[1U];

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kInvalidTimeTransfer);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsStaleSourceGeometry)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);
  ++request.source_geometry_version.revision;

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kStaleLandmark);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsCameraModelRevisionMismatch)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);
  ++request.source_endpoint.camera_model_revision;

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kModelRevisionMismatch);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsRigRevisionMismatch)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);
  ++request.target_endpoint.rig_revision;

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kRigRevisionMismatch);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsProjectionOutsideProvidedTargetBounds)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);
  request.target_image_bounds =
    livo::ReferenceOnlyMigrationTargetImageBounds{100, 100};

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kOutOfFrame);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsRayParallelToPlane)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);
  request.plane.normal_world = Eigen::Vector3d::UnitX();

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kRayPlaneDegenerate);
}

TEST(
  ReferenceOnlyMigrationShadow,
  RejectsNonFiniteSourcePose)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest request =
    validRequest(bundle);
  request.source_pose.position_world_m.x() =
    std::numeric_limits<double>::quiet_NaN();

  const auto result =
    livo::evaluateReferenceOnlyMigrationShadow(request);

  EXPECT_EQ(
    result.migrationStatus(),
    livo::MigrationStatus::kNonFinite);
}

TEST(
  ReferenceOnlyMigrationShadow,
  PlaneNormalSignDoesNotChangeGeometry)
{
  const livo::CameraBatchTimeTransferBundle bundle =
    targetTimeBundle(10.0);
  livo::ReferenceOnlyMigrationShadowRequest positive =
    validRequest(bundle);
  livo::ReferenceOnlyMigrationShadowRequest negative =
    positive;
  negative.plane.normal_world *= -1.0;

  const auto positive_result =
    livo::evaluateReferenceOnlyMigrationShadow(positive);
  const auto negative_result =
    livo::evaluateReferenceOnlyMigrationShadow(negative);

  ASSERT_EQ(
    positive_result.migrationStatus(),
    livo::MigrationStatus::kReady);
  ASSERT_EQ(
    negative_result.migrationStatus(),
    livo::MigrationStatus::kReady);
  EXPECT_TRUE(positive_result.targetPixel().isApprox(
    negative_result.targetPixel(), 1.0e-12));
  EXPECT_NEAR(
    positive_result.warpConditionNumber(),
    negative_result.warpConditionNumber(),
    1.0e-12);
  EXPECT_EQ(
    positive_result.validProjectedPixelCount(),
    negative_result.validProjectedPixelCount());
}

}  // namespace
