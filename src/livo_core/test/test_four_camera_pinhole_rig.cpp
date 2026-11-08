#include "four_camera_pinhole_rig.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

namespace
{

livo::FourCameraPinholeRigConfig simpleRigConfig()
{
  livo::FourCameraPinholeRigConfig config;
  for (std::size_t index = 0U; index < config.cameras.size(); ++index)
  {
    auto & camera = config.cameras[index];
    camera.camera_id = static_cast<std::uint8_t>(index);
    camera.projection.model = "Pinhole";
    camera.projection.width = 640;
    camera.projection.height = 480;
    camera.projection.fx = 600.0 + static_cast<double>(index);
    camera.projection.fy = 601.0 + static_cast<double>(index);
    camera.projection.cx = 320.0;
    camera.projection.cy = 240.0;
    camera.rotation_camera_from_imu =
      Eigen::Matrix3d::Identity();
    camera.position_camera_from_imu =
      Eigen::Vector3d(0.1 * static_cast<double>(index), 0.0, 0.0);
  }
  return config;
}

Eigen::Matrix3d rowMajorMatrix(
  const std::array<double, 9U> & values)
{
  Eigen::Matrix3d result;
  for (std::size_t row = 0U; row < 3U; ++row)
  {
    for (std::size_t column = 0U; column < 3U; ++column)
    {
      result(
        static_cast<Eigen::Index>(row),
        static_cast<Eigen::Index>(column)) =
        values[row * 3U + column];
    }
  }
  return result;
}

livo::FourCameraPinholeRigConfig outdoorAiryRigConfig()
{
  const std::array<std::array<double, 4U>, 4U> intrinsics{{
    {{603.106567, 603.102905, 330.111053, 246.598862}},
    {{604.419189, 604.081299, 327.659485, 250.951019}},
    {{604.789978, 604.661682, 326.815796, 244.573853}},
    {{604.105245, 603.948629, 328.195445, 247.374578}},
  }};
  // These are the exact outdoor YAML T_CI values after VIOManager applies
  // R_CI = R_CL * R_LI and p_CI = R_CL * p_LI + p_CL.
  const std::array<std::array<double, 9U>, 4U>
    rotations_camera_from_imu{{
      {{
        0.0031916438524545324,
        0.99999348814234101,
        -0.0016843641034486483,
        0.0050081472031502138,
        -0.0017003358397726313,
        -0.99998601356200167,
        -0.99998236578007249,
        0.0031831636693519043,
        -0.0050135414572385585,
      }},
      {{
        -0.99982795897656507,
        -0.011402097122951621,
        0.014630264179260499,
        -0.014464221993487875,
        -0.014511318976066556,
        -0.99979008191904872,
        0.011612008046814171,
        -0.9998296923990474,
        0.014343899969920915,
      }},
      {{
        -0.0016205753677488634,
        -0.99977697666663712,
        -0.021056368699190493,
        0.0044431698369502559,
        0.021048989597706597,
        -0.99976857235998173,
        0.99998881592609712,
        -0.0017137573440959807,
        0.0044080674323445646,
      }},
      {{
        0.99969405042166892,
        0.015445004958050595,
        -0.019319869910627285,
        -0.019778473062576601,
        0.030112931589163727,
        -0.99935080094750484,
        -0.014853200154535728,
        0.99942716751794991,
        0.030409197164955909,
      }},
    }};
  const std::array<std::array<double, 3U>, 4U>
    positions_camera_from_imu{{
      {{
        0.023638716220863235,
        -0.27175954406836622,
        -0.16185822972119865,
      }},
      {{
        0.025737681317899299,
        -0.27445913635877212,
        -0.15690723200309248,
      }},
      {{
        0.018101941301579442,
        -0.27367422775645012,
        -0.15932627532306756,
      }},
      {{
        0.015833011976899757,
        -0.27765219669322555,
        -0.1525408097148524,
      }},
    }};

  livo::FourCameraPinholeRigConfig config;
  for (std::size_t camera_id = 0U;
       camera_id < config.cameras.size();
       ++camera_id)
  {
    auto & camera = config.cameras[camera_id];
    camera.camera_id = static_cast<std::uint8_t>(camera_id);
    camera.projection.model = "Pinhole";
    camera.projection.width = 640;
    camera.projection.height = 480;
    camera.projection.fx = intrinsics[camera_id][0U];
    camera.projection.fy = intrinsics[camera_id][1U];
    camera.projection.cx = intrinsics[camera_id][2U];
    camera.projection.cy = intrinsics[camera_id][3U];
    camera.rotation_camera_from_imu =
      rowMajorMatrix(rotations_camera_from_imu[camera_id]);
    camera.position_camera_from_imu =
      Eigen::Vector3d(
        positions_camera_from_imu[camera_id][0U],
        positions_camera_from_imu[camera_id][1U],
        positions_camera_from_imu[camera_id][2U]);
  }
  return config;
}

template<typename Action>
void expectInvalidArgumentContaining(
  Action action,
  const std::string & expected_reason)
{
  try
  {
    action();
    FAIL() << "expected std::invalid_argument";
  }
  catch (const std::invalid_argument & error)
  {
    EXPECT_NE(
      std::string(error.what()).find(expected_reason),
      std::string::npos)
      << error.what();
  }
}

template<typename Rig, typename = void>
struct CanBorrowCameraModel : std::false_type
{
};

template<typename Rig>
struct CanBorrowCameraModel<
  Rig,
  std::void_t<decltype(
      std::declval<Rig>().cameraModel(std::uint8_t{}))>>
: std::true_type
{
};

template<typename Rig, typename = void>
struct CanCreateShadowEndpoint : std::false_type
{
};

template<typename Rig>
struct CanCreateShadowEndpoint<
  Rig,
  std::void_t<decltype(
      std::declval<Rig>().shadowEndpoint(
        std::uint8_t{},
        livo::FourCameraPinholeRigRevisionExpectation{}))>>
: std::true_type
{
};

std::string fingerprintHex(
  const livo::FourCameraPinholeRigFingerprint & fingerprint)
{
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result;
  result.reserve(fingerprint.size() * 2U);
  for (const std::uint8_t value : fingerprint)
  {
    result.push_back(kHexDigits[value >> 4U]);
    result.push_back(kHexDigits[value & 0x0fU]);
  }
  return result;
}

TEST(
  FourCameraPinholeRig,
  OwnsStableModelsAndCreatesOnlyStructuralShadowEndpoints)
{
  static_assert(
    !std::is_copy_constructible_v<livo::FourCameraPinholeRig>);
  static_assert(
    !std::is_move_constructible_v<livo::FourCameraPinholeRig>);
  static_assert(
    CanBorrowCameraModel<
      const livo::FourCameraPinholeRig &>::value);
  static_assert(
    !CanBorrowCameraModel<
      const livo::FourCameraPinholeRig &&>::value);
  static_assert(
    CanCreateShadowEndpoint<
      const livo::FourCameraPinholeRig &>::value);
  static_assert(
    !CanCreateShadowEndpoint<
      const livo::FourCameraPinholeRig &&>::value);

  const livo::FourCameraPinholeRig rig(simpleRigConfig());
  ASSERT_NE(rig.rigRevision(), 0U);

  std::array<
    std::shared_ptr<const livo::CameraProjectionAdapter>,
    4U> first_models{};
  for (std::size_t index = 0U; index < first_models.size(); ++index)
  {
    const auto camera_id = static_cast<std::uint8_t>(index);
    const std::shared_ptr<const livo::PinholeCameraProjectionAdapter> model =
      rig.cameraModel(camera_id);
    ASSERT_NE(model, nullptr);
    const std::uint64_t model_revision =
      model->modelRevision();
    ASSERT_NE(model_revision, 0U);
    const livo::FourCameraPinholeRigEndpointResult first =
      rig.shadowEndpoint(
        camera_id,
        livo::FourCameraPinholeRigRevisionExpectation{
          model_revision, rig.rigRevision()});
    ASSERT_TRUE(first.endpoint.has_value());
    EXPECT_EQ(
      first.status,
      livo::FourCameraPinholeRigEndpointStatus::
      kShadowOnlyMissingUncertaintyReceipt);
    EXPECT_EQ(
      first.reason,
      "missing_measured_endpoint_uncertainty_receipt");
    EXPECT_TRUE(first.shadowReady());
    EXPECT_FALSE(first.productionReady());

    const livo::MigrationEndpoint & endpoint = *first.endpoint;
    EXPECT_EQ(endpoint.camera_id, camera_id);
    EXPECT_EQ(endpoint.camera_model_revision, model_revision);
    EXPECT_EQ(endpoint.rig_revision, rig.rigRevision());
    EXPECT_EQ(endpoint.camera_model, rig.cameraModel(camera_id));
    EXPECT_TRUE(endpoint.modelRevisionMatchesAdapter());
    EXPECT_TRUE(
      endpoint.rotation_imu_from_camera.isApprox(
        Eigen::Matrix3d::Identity(), 0.0));
    EXPECT_TRUE(
      endpoint.position_imu_from_camera.isApprox(
        Eigen::Vector3d(
          -0.1 * static_cast<double>(index), 0.0, 0.0),
        1.0e-15));
    EXPECT_EQ(
      endpoint.uncertainty.camera_model_revision, 0U);
    EXPECT_EQ(endpoint.uncertainty.rig_revision, 0U);
    EXPECT_EQ(endpoint.uncertainty.revision, 0U);
    EXPECT_TRUE(
      endpoint.uncertainty.
      observation_pixel_covariance_px2.isZero(0.0));
    EXPECT_TRUE(
      endpoint.uncertainty.
      camera_model_equivalent_pixel_covariance_px2.isZero(0.0));
    EXPECT_TRUE(
      endpoint.uncertainty.extrinsic_covariance.isZero(0.0));
    EXPECT_FALSE(endpoint.uncertainty.productionReady(endpoint));

    first_models[index] = endpoint.camera_model;
    const livo::FourCameraPinholeRigEndpointResult second =
      rig.shadowEndpoint(
        camera_id,
        livo::FourCameraPinholeRigRevisionExpectation{
          model_revision, rig.rigRevision()});
    ASSERT_TRUE(second.endpoint.has_value());
    EXPECT_EQ(
      second.endpoint->camera_model,
      first_models[index]);
  }
}

TEST(
  FourCameraPinholeRig,
  EndpointCopiesKeepTheImmutableModelAliveAfterOwnerDestruction)
{
  std::optional<livo::MigrationEndpoint> saved_endpoint;
  {
    const livo::FourCameraPinholeRig rig(simpleRigConfig());
    const std::shared_ptr<const livo::PinholeCameraProjectionAdapter> model =
      rig.cameraModel(2U);
    ASSERT_NE(model, nullptr);
    const auto result = rig.shadowEndpoint(
      2U,
      livo::FourCameraPinholeRigRevisionExpectation{
        model->modelRevision(), rig.rigRevision()});
    ASSERT_TRUE(result.shadowReady());
    saved_endpoint = result.endpoint;
  }

  ASSERT_TRUE(saved_endpoint.has_value());
  ASSERT_NE(saved_endpoint->camera_model, nullptr);
  EXPECT_TRUE(saved_endpoint->camera_model->valid());
  EXPECT_TRUE(saved_endpoint->modelRevisionMatchesAdapter());
  EXPECT_EQ(
    saved_endpoint->camera_model->modelRevision(),
    saved_endpoint->camera_model_revision);
  const livo::CameraProjectionResult projection =
    saved_endpoint->camera_model->project(
      Eigen::Vector3d(0.0, 0.0, 1.0));
  EXPECT_EQ(projection.status, livo::ProjectionAdapterStatus::kValid);

  livo::MigrationEndpoint copied = *saved_endpoint;
  livo::MigrationEndpoint moved = std::move(copied);
  EXPECT_TRUE(moved.modelRevisionMatchesAdapter());
  EXPECT_TRUE(moved.camera_model->valid());
  EXPECT_TRUE(moved.sharesCameraModelIdentityWith(*saved_endpoint));

  livo::MigrationEndpoint null_model = *saved_endpoint;
  null_model.camera_model.reset();
  EXPECT_FALSE(null_model.modelRevisionMatchesAdapter());
  EXPECT_FALSE(
    saved_endpoint->sharesCameraModelIdentityWith(null_model));

  livo::MigrationEndpoint foreign_revision = *saved_endpoint;
  ++foreign_revision.camera_model_revision;
  EXPECT_FALSE(foreign_revision.modelRevisionMatchesAdapter());
  EXPECT_FALSE(
    saved_endpoint->sharesCameraModelIdentityWith(foreign_revision));

  const livo::FourCameraPinholeRig foreign_rig(simpleRigConfig());
  const auto foreign_endpoint_result = foreign_rig.shadowEndpoint(
    1U,
    livo::FourCameraPinholeRigRevisionExpectation{
      foreign_rig.cameraModel(1U)->modelRevision(),
      foreign_rig.rigRevision()});
  ASSERT_TRUE(foreign_endpoint_result.endpoint.has_value());
  const livo::MigrationEndpoint & foreign_endpoint =
    *foreign_endpoint_result.endpoint;
  EXPECT_FALSE(
    saved_endpoint->sharesCameraModelIdentityWith(foreign_endpoint));

  livo::MigrationEndpoint foreign_pointer_same_owner =
    *saved_endpoint;
  foreign_pointer_same_owner.camera_model =
    std::shared_ptr<const livo::CameraProjectionAdapter>(
      saved_endpoint->camera_model,
      foreign_endpoint.camera_model.get());
  EXPECT_FALSE(
    saved_endpoint->sharesCameraModelIdentityWith(
      foreign_pointer_same_owner));

  const auto unrelated_owner = std::make_shared<int>(7);
  livo::MigrationEndpoint same_pointer_foreign_owner =
    *saved_endpoint;
  same_pointer_foreign_owner.camera_model =
    std::shared_ptr<const livo::CameraProjectionAdapter>(
      unrelated_owner,
      saved_endpoint->camera_model.get());
  EXPECT_FALSE(
    saved_endpoint->sharesCameraModelIdentityWith(
      same_pointer_foreign_owner));
}

TEST(
  FourCameraPinholeRig,
  StructuralFingerprintChangesWithRigModelOrExtrinsicContent)
{
  const livo::FourCameraPinholeRigConfig baseline_config =
    simpleRigConfig();
  const livo::FourCameraPinholeRig baseline(baseline_config);
  const livo::FourCameraPinholeRigFingerprint
    baseline_fingerprint =
    baseline.configurationFingerprint();
  EXPECT_TRUE(std::any_of(
    baseline_fingerprint.begin(),
    baseline_fingerprint.end(),
    [](const std::uint8_t value) {return value != 0U;}));

  for (std::size_t camera_id = 0U;
       camera_id < baseline_config.cameras.size();
       ++camera_id)
  {
    for (std::size_t field = 0U; field < 6U; ++field)
    {
      auto changed_intrinsic = baseline_config;
      auto & projection =
        changed_intrinsic.cameras[camera_id].projection;
      switch (field)
      {
        case 0U:
          projection.width += 1;
          break;
        case 1U:
          projection.height += 1;
          break;
        case 2U:
          projection.fx += 0.25;
          break;
        case 3U:
          projection.fy += 0.25;
          break;
        case 4U:
          projection.cx += 0.25;
          break;
        case 5U:
          projection.cy += 0.25;
          break;
      }
      const livo::FourCameraPinholeRig changed(
        changed_intrinsic);
      EXPECT_NE(
        baseline.configurationFingerprint(),
        changed.configurationFingerprint());
      EXPECT_NE(
        baseline.rigRevision(), changed.rigRevision());
      EXPECT_NE(
        baseline.cameraModel(
          static_cast<std::uint8_t>(camera_id))->
        modelRevision(),
        changed.cameraModel(
          static_cast<std::uint8_t>(camera_id))->
        modelRevision());
    }

    for (Eigen::Index component = 0; component < 3; ++component)
    {
      auto changed_translation = baseline_config;
      changed_translation.cameras[camera_id].
        position_camera_from_imu(component) += 0.01;
      const livo::FourCameraPinholeRig changed(
        changed_translation);
      EXPECT_NE(
        baseline.configurationFingerprint(),
        changed.configurationFingerprint());
      EXPECT_NE(
        baseline.rigRevision(), changed.rigRevision());
      EXPECT_EQ(
        baseline.cameraModel(
          static_cast<std::uint8_t>(camera_id))->
        modelRevision(),
        changed.cameraModel(
          static_cast<std::uint8_t>(camera_id))->
        modelRevision());
    }

    auto changed_rotation = baseline_config;
    changed_rotation.cameras[camera_id].
      rotation_camera_from_imu =
      Eigen::AngleAxisd(
        0.001, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      changed_rotation.cameras[camera_id].
      rotation_camera_from_imu;
    const livo::FourCameraPinholeRig rotation_changed(
      changed_rotation);
    EXPECT_NE(
      baseline.configurationFingerprint(),
      rotation_changed.configurationFingerprint());
    EXPECT_NE(
      baseline.rigRevision(), rotation_changed.rigRevision());
  }

  auto reordered = baseline_config;
  std::swap(reordered.cameras[0U], reordered.cameras[3U]);
  const livo::FourCameraPinholeRig reordered_rig(reordered);
  EXPECT_EQ(
    baseline.configurationFingerprint(),
    reordered_rig.configurationFingerprint());

  auto negative_zero = baseline_config;
  for (auto & camera : negative_zero.cameras)
  {
    camera.projection.distortion_coefficients =
      {{-0.0, -0.0, -0.0, -0.0}};
    camera.position_camera_from_imu.y() = -0.0;
    camera.position_camera_from_imu.z() = -0.0;
    camera.rotation_camera_from_imu(0, 1) = -0.0;
    camera.rotation_camera_from_imu(0, 2) = -0.0;
    camera.rotation_camera_from_imu(1, 0) = -0.0;
  }
  const livo::FourCameraPinholeRig negative_zero_rig(
    negative_zero);
  EXPECT_EQ(
    baseline.configurationFingerprint(),
    negative_zero_rig.configurationFingerprint());
  EXPECT_EQ(
    baseline.rigRevision(), negative_zero_rig.rigRevision());
}

TEST(
  FourCameraPinholeRig,
  RejectsForeignRevisionsWithoutPublishingAnEndpoint)
{
  const livo::FourCameraPinholeRig rig(simpleRigConfig());
  const std::uint64_t model_revision =
    rig.cameraModel(1U)->modelRevision();

  const auto foreign_model = rig.shadowEndpoint(
    1U,
    livo::FourCameraPinholeRigRevisionExpectation{
      model_revision ^ 1U, rig.rigRevision()});
  EXPECT_EQ(
    foreign_model.status,
    livo::FourCameraPinholeRigEndpointStatus::
    kCameraModelRevisionMismatch);
  EXPECT_EQ(
    foreign_model.reason, "foreign_camera_model_revision");
  EXPECT_FALSE(foreign_model.endpoint.has_value());
  EXPECT_FALSE(foreign_model.shadowReady());
  EXPECT_FALSE(foreign_model.productionReady());

  const auto foreign_rig = rig.shadowEndpoint(
    1U,
    livo::FourCameraPinholeRigRevisionExpectation{
      model_revision, rig.rigRevision() ^ 1U});
  EXPECT_EQ(
    foreign_rig.status,
    livo::FourCameraPinholeRigEndpointStatus::
    kRigRevisionMismatch);
  EXPECT_EQ(foreign_rig.reason, "foreign_rig_revision");
  EXPECT_FALSE(foreign_rig.endpoint.has_value());

  const auto invalid_camera = rig.shadowEndpoint(
    4U,
    livo::FourCameraPinholeRigRevisionExpectation{
      model_revision, rig.rigRevision()});
  EXPECT_EQ(
    invalid_camera.status,
    livo::FourCameraPinholeRigEndpointStatus::
    kInvalidCameraId);
  EXPECT_EQ(invalid_camera.reason, "invalid_camera_id");
  EXPECT_FALSE(invalid_camera.endpoint.has_value());
}

TEST(
  FourCameraPinholeRig,
  RejectsIncompleteIdsInvalidExtrinsicsAndUnsupportedProjection)
{
  auto duplicate_id = simpleRigConfig();
  duplicate_id.cameras[3U].camera_id = 2U;
  expectInvalidArgumentContaining(
    [&duplicate_id]() {
      const livo::FourCameraPinholeRig rejected(duplicate_id);
    },
    "unique 0..3");

  auto out_of_range_id = simpleRigConfig();
  out_of_range_id.cameras[3U].camera_id = 4U;
  expectInvalidArgumentContaining(
    [&out_of_range_id]() {
      const livo::FourCameraPinholeRig rejected(out_of_range_id);
    },
    "unique 0..3");

  auto invalid_rotation = simpleRigConfig();
  invalid_rotation.cameras[1U].
    rotation_camera_from_imu(0, 0) = -1.0;
  expectInvalidArgumentContaining(
    [&invalid_rotation]() {
      const livo::FourCameraPinholeRig rejected(invalid_rotation);
    },
    "finite SO(3)");

  auto nonfinite_translation = simpleRigConfig();
  nonfinite_translation.cameras[1U].
    position_camera_from_imu.x() =
    std::numeric_limits<double>::infinity();
  expectInvalidArgumentContaining(
    [&nonfinite_translation]() {
      const livo::FourCameraPinholeRig rejected(
        nonfinite_translation);
    },
    "finite SO(3)");

  auto unsupported_model = simpleRigConfig();
  unsupported_model.cameras[1U].projection.model =
    "EquidistantCamera";
  expectInvalidArgumentContaining(
    [&unsupported_model]() {
      const livo::FourCameraPinholeRig rejected(unsupported_model);
    },
    "invalid Pinhole projection for camera 1");

  auto nonzero_distortion = simpleRigConfig();
  nonzero_distortion.cameras[1U].
    projection.distortion_coefficients[0U] = 0.01;
  expectInvalidArgumentContaining(
    [&nonzero_distortion]() {
      const livo::FourCameraPinholeRig rejected(
        nonzero_distortion);
    },
    "invalid Pinhole projection for camera 1");

  auto invalid_dimensions = simpleRigConfig();
  invalid_dimensions.cameras[1U].projection.width = 0;
  expectInvalidArgumentContaining(
    [&invalid_dimensions]() {
      const livo::FourCameraPinholeRig rejected(invalid_dimensions);
    },
    "invalid Pinhole projection for camera 1");
}

TEST(
  FourCameraPinholeRig,
  OutdoorYamlValuesProduceTheExpectedFourImuFromCameraEndpoints)
{
  const std::array<std::array<double, 9U>, 4U>
    expected_rotations_imu_from_camera{{
      {{
        0.0031916438524545324,
        0.0050081472031502138,
        -0.99998236578007249,
        0.99999348814234101,
        -0.0017003358397726313,
        0.0031831636693519043,
        -0.0016843641034486483,
        -0.99998601356200167,
        -0.0050135414572385585,
      }},
      {{
        -0.99982795897656507,
        -0.014464221993487875,
        0.011612008046814171,
        -0.011402097122951621,
        -0.014511318976066556,
        -0.9998296923990474,
        0.014630264179260499,
        -0.99979008191904872,
        0.014343899969920915,
      }},
      {{
        -0.0016205753677488634,
        0.0044431698369502559,
        0.99998881592609712,
        -0.99977697666663712,
        0.021048989597706597,
        -0.0017137573440959807,
        -0.021056368699190493,
        -0.99976857235998173,
        0.0044080674323445646,
      }},
      {{
        0.99969405042166892,
        -0.019778473062576601,
        -0.014853200154535728,
        0.015445004958050595,
        0.030112931589163727,
        0.99942716751794991,
        -0.019319869910627285,
        -0.99935080094750484,
        0.030409197164955909,
      }},
    }};
  const std::array<std::array<double, 3U>, 4U>
    expected_positions_imu_from_camera{{
      {{
        -0.16056981004032952,
        -0.023585423545053538,
        -0.27252740986020108,
      }},
      {{
        0.023585423545053535,
        -0.16056981004032955,
        -0.27252740986020108,
      }},
      {{
        0.16056981004032955,
        0.023585423545053563,
        -0.27252740986020113,
      }},
      {{
        -0.023585423545053545,
        0.16056981004032955,
        -0.27252740986020108,
      }},
    }};
  const std::array<std::array<double, 4U>, 4U> intrinsics{{
    {{603.106567, 603.102905, 330.111053, 246.598862}},
    {{604.419189, 604.081299, 327.659485, 250.951019}},
    {{604.789978, 604.661682, 326.815796, 244.573853}},
    {{604.105245, 603.948629, 328.195445, 247.374578}},
  }};

  const livo::FourCameraPinholeRig rig(outdoorAiryRigConfig());
  EXPECT_EQ(rig.rigRevision(), 8206638252039177439ULL);
  EXPECT_EQ(
    fingerprintHex(rig.configurationFingerprint()),
    "71e3d60274ed84df4237393f2d79db6f"
    "1d48ceef9d0f0bbfb97fb0fa84fcd048");
  const std::array<std::uint64_t, 4U>
    expected_model_revisions{{
      1172720795579671296ULL,
      14974416105296971232ULL,
      10800054445785174959ULL,
      11386666025749495921ULL,
    }};
  for (std::size_t camera_id = 0U;
       camera_id < livo::kFourCameraPinholeRigCameraCount;
       ++camera_id)
  {
    const auto model =
      rig.cameraModel(static_cast<std::uint8_t>(camera_id));
    ASSERT_NE(model, nullptr);
    EXPECT_DOUBLE_EQ(model->fx(), intrinsics[camera_id][0U]);
    EXPECT_DOUBLE_EQ(model->fy(), intrinsics[camera_id][1U]);
    EXPECT_DOUBLE_EQ(model->cx(), intrinsics[camera_id][2U]);
    EXPECT_DOUBLE_EQ(model->cy(), intrinsics[camera_id][3U]);
    EXPECT_EQ(
      model->modelRevision(),
      expected_model_revisions[camera_id]);

    const auto result = rig.shadowEndpoint(
      static_cast<std::uint8_t>(camera_id),
      livo::FourCameraPinholeRigRevisionExpectation{
        model->modelRevision(), rig.rigRevision()});
    ASSERT_TRUE(result.shadowReady());
    ASSERT_TRUE(result.endpoint.has_value());
    EXPECT_TRUE(
      result.endpoint->rotation_imu_from_camera.isApprox(
        rowMajorMatrix(
          expected_rotations_imu_from_camera[camera_id]),
        1.0e-15));
    EXPECT_TRUE(
      result.endpoint->position_imu_from_camera.isApprox(
        Eigen::Vector3d(
          expected_positions_imu_from_camera[camera_id][0U],
          expected_positions_imu_from_camera[camera_id][1U],
          expected_positions_imu_from_camera[camera_id][2U]),
        1.0e-15));
    EXPECT_TRUE(result.endpoint->modelRevisionMatchesAdapter());
    EXPECT_FALSE(result.productionReady());
  }
}

}  // namespace
