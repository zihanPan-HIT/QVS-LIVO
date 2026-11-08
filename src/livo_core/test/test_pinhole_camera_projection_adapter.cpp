#include "pinhole_camera_projection_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace
{

static_assert(std::is_copy_constructible_v<
  livo::PinholeCameraProjectionAdapter>);
static_assert(!std::is_default_constructible_v<
  livo::PinholeCameraProjectionAdapter>);
static_assert(!std::is_copy_assignable_v<
  livo::PinholeCameraProjectionAdapter>);
static_assert(!std::is_move_assignable_v<
  livo::PinholeCameraProjectionAdapter>);

livo::PinholeCameraProjectionConfig outdoorFrontConfig()
{
  livo::PinholeCameraProjectionConfig config;
  config.model = "Pinhole";
  config.width = 640;
  config.height = 480;
  config.fx = 603.106567;
  config.fy = 603.102905;
  config.cx = 330.111053;
  config.cy = 246.598862;
  config.distortion_coefficients = {
    {0.0, 0.0, 0.0, 0.0}};
  config.model_revision = 1U;
  return config;
}

std::vector<livo::PinholeCameraProjectionConfig>
outdoorCameraConfigs()
{
  struct Intrinsics
  {
    double fx;
    double fy;
    double cx;
    double cy;
  };
  const std::array<Intrinsics, 4U> intrinsics{{
    {603.106567, 603.102905, 330.111053, 246.598862},
    {604.419189, 604.081299, 327.659485, 250.951019},
    {604.789978, 604.661682, 326.815796, 244.573853},
    {604.105245, 603.948629, 328.195445, 247.374578},
  }};
  std::vector<livo::PinholeCameraProjectionConfig> configs;
  configs.reserve(intrinsics.size());
  for (std::size_t index = 0U;
       index < intrinsics.size(); ++index)
  {
    auto config = outdoorFrontConfig();
    config.fx = intrinsics[index].fx;
    config.fy = intrinsics[index].fy;
    config.cx = intrinsics[index].cx;
    config.cy = intrinsics[index].cy;
    config.model_revision =
      static_cast<std::uint64_t>(index + 1U);
    configs.push_back(config);
  }
  return configs;
}

TEST(
  PinholeCameraProjectionAdapter,
  OutdoorFrontUnprojectsAnUnnormalizedRayWithMatchingJacobian)
{
  const livo::PinholeCameraProjectionAdapter adapter(
    outdoorFrontConfig());
  ASSERT_TRUE(adapter.valid());
  EXPECT_EQ(adapter.modelRevision(), 1U);

  const Eigen::Vector2d pixel(
    480.88769475000004, 171.210998875);
  const auto ray = adapter.unproject(pixel);
  ASSERT_EQ(
    ray.status, livo::ProjectionAdapterStatus::kValid);
  EXPECT_TRUE(ray.direction_camera.isApprox(
    Eigen::Vector3d(0.25, -0.125, 1.0), 1.0e-14));
  EXPECT_NEAR(ray.direction_camera.norm(), 1.0383279828647594, 1.0e-14);

  const auto jacobian = adapter.unprojectJacobian(pixel);
  ASSERT_EQ(
    jacobian.status, livo::ProjectionAdapterStatus::kValid);
  Eigen::Matrix<double, 3, 2> expected =
    Eigen::Matrix<double, 3, 2>::Zero();
  expected(0, 0) = 0.001658081763185311;
  expected(1, 1) = 0.0016580918309455001;
  EXPECT_TRUE(jacobian.jacobian.isApprox(expected, 1.0e-15));
}

TEST(
  PinholeCameraProjectionAdapter,
  RejectsUnsupportedOrNumericallyInvalidConfiguration)
{
  const auto expect_rejected = [](
    const livo::PinholeCameraProjectionConfig & config) {
      EXPECT_THROW(
        {
          const livo::PinholeCameraProjectionAdapter rejected(
            config);
          static_cast<void>(rejected);
        },
        std::invalid_argument);
    };

  auto config = outdoorFrontConfig();
  config.model = "EquidistantCamera";
  expect_rejected(config);

  config = outdoorFrontConfig();
  config.width = 1;
  expect_rejected(config);
  config = outdoorFrontConfig();
  config.height = 1;
  expect_rejected(config);
  config = outdoorFrontConfig();
  config.width = std::numeric_limits<int>::max();
  config.height = std::numeric_limits<int>::max();
  expect_rejected(config);

  for (const double invalid_focal : {
      0.0,
      -0.0,
      -1.0,
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()})
  {
    config = outdoorFrontConfig();
    config.fx = invalid_focal;
    expect_rejected(config);
    config = outdoorFrontConfig();
    config.fy = invalid_focal;
    expect_rejected(config);
  }

  config = outdoorFrontConfig();
  config.cx = std::numeric_limits<double>::infinity();
  expect_rejected(config);
  config = outdoorFrontConfig();
  config.cy = std::numeric_limits<double>::quiet_NaN();
  expect_rejected(config);

  config = outdoorFrontConfig();
  config.distortion_coefficients[2U] = 1.0e-15;
  expect_rejected(config);
  config = outdoorFrontConfig();
  config.distortion_coefficients[0U] =
    std::numeric_limits<double>::quiet_NaN();
  expect_rejected(config);

  config = outdoorFrontConfig();
  config.model_revision = 0U;
  expect_rejected(config);

  config = outdoorFrontConfig();
  config.distortion_coefficients.fill(-0.0);
  const livo::PinholeCameraProjectionAdapter negative_zero_distortion(
    config);
  EXPECT_TRUE(negative_zero_distortion.valid());
}

TEST(
  PinholeCameraProjectionAdapter,
  RejectsAnyConfigurationWithUnsafeDeclaredImageRays)
{
  const auto expect_rejected = [](
    const livo::PinholeCameraProjectionConfig & config) {
      EXPECT_THROW(
        {
          const livo::PinholeCameraProjectionAdapter rejected(
            config);
          static_cast<void>(rejected);
        },
        std::invalid_argument);
    };

  auto config = outdoorFrontConfig();
  config.fx = std::numeric_limits<double>::denorm_min();
  expect_rejected(config);

  config = outdoorFrontConfig();
  config.fy = std::numeric_limits<double>::min();
  expect_rejected(config);

  config = outdoorFrontConfig();
  config.cx = std::numeric_limits<double>::max();
  expect_rejected(config);

  config = outdoorFrontConfig();
  const double square_root_maximum =
    std::sqrt(std::numeric_limits<double>::max());
  config.fx = 1.0;
  config.fy = 1.0;
  config.cx = square_root_maximum;
  config.cy = square_root_maximum;
  expect_rejected(config);

  config = outdoorFrontConfig();
  config.fx = 1.0;
  config.fy = 1.0;
  config.cx = 0.5 * square_root_maximum;
  config.cy = 0.5 * square_root_maximum;
  const livo::PinholeCameraProjectionAdapter near_norm_limit(
    config);
  ASSERT_TRUE(near_norm_limit.valid());
  const auto corner =
    near_norm_limit.unproject(Eigen::Vector2d(0.0, 0.0));
  ASSERT_EQ(
    corner.status, livo::ProjectionAdapterStatus::kValid);
  EXPECT_TRUE(corner.direction_camera.allFinite());
  EXPECT_TRUE(std::isfinite(
    corner.direction_camera.squaredNorm()));
  EXPECT_TRUE(std::isfinite(corner.direction_camera.norm()));

  config = outdoorFrontConfig();
  config.fx = std::numeric_limits<double>::max();
  config.fy = std::numeric_limits<double>::max();
  const livo::PinholeCameraProjectionAdapter
    near_reciprocal_limit(config);
  const auto jacobian =
    near_reciprocal_limit.unprojectJacobian(
      Eigen::Vector2d(config.cx, config.cy));
  ASSERT_EQ(
    jacobian.status, livo::ProjectionAdapterStatus::kValid);
  EXPECT_GT(jacobian.jacobian(0, 0), 0.0);
  EXPECT_GT(jacobian.jacobian(1, 1), 0.0);
  EXPECT_TRUE(jacobian.jacobian.allFinite());
}

TEST(
  PinholeCameraProjectionAdapter,
  ContainsUsesCheckedHalfOpenPixelBounds)
{
  const livo::PinholeCameraProjectionAdapter adapter(
    outdoorFrontConfig());
  const double inside_right = std::nextafter(640.0, 0.0);
  const double inside_bottom = std::nextafter(480.0, 0.0);

  EXPECT_TRUE(adapter.contains(Eigen::Vector2d(0.0, 0.0), 0));
  EXPECT_TRUE(adapter.contains(
    Eigen::Vector2d(-0.0, -0.0), 0));
  EXPECT_TRUE(adapter.contains(
    Eigen::Vector2d(inside_right, inside_bottom), 0));
  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(640.0, 200.0), 0));
  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(200.0, 480.0), 0));
  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(
      -std::numeric_limits<double>::denorm_min(), 200.0), 0));

  EXPECT_TRUE(adapter.contains(Eigen::Vector2d(1.0, 1.0), 1));
  EXPECT_TRUE(adapter.contains(
    Eigen::Vector2d(
      std::nextafter(639.0, 0.0),
      std::nextafter(479.0, 0.0)), 1));
  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(639.0, 200.0), 1));
  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(200.0, 479.0), 1));

  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(320.0, 240.0), -1));
  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(320.0, 240.0),
    std::numeric_limits<int>::max()));
  EXPECT_FALSE(adapter.contains(
    Eigen::Vector2d(
      std::numeric_limits<double>::quiet_NaN(), 240.0), 0));
}

TEST(
  PinholeCameraProjectionAdapter,
  UnprojectionFailsClosedForNonFiniteOrOutOfDomainPixels)
{
  const livo::PinholeCameraProjectionAdapter adapter(
    outdoorFrontConfig());
  const auto expect_failure = [](
    const livo::PinholeCameraProjectionAdapter & camera,
    const Eigen::Vector2d & pixel,
    const livo::ProjectionAdapterStatus expected_status) {
      const auto ray = camera.unproject(pixel);
      EXPECT_EQ(ray.status, expected_status);
      EXPECT_TRUE(ray.direction_camera.array().isNaN().all());
      const auto jacobian = camera.unprojectJacobian(pixel);
      EXPECT_EQ(jacobian.status, expected_status);
      EXPECT_TRUE(jacobian.jacobian.array().isNaN().all());
    };

  expect_failure(
    adapter,
    Eigen::Vector2d(640.0, 100.0),
    livo::ProjectionAdapterStatus::kOutOfDomain);
  expect_failure(
    adapter,
    Eigen::Vector2d(-1.0, 100.0),
    livo::ProjectionAdapterStatus::kOutOfDomain);
  expect_failure(
    adapter,
    Eigen::Vector2d(
      std::numeric_limits<double>::quiet_NaN(), 100.0),
    livo::ProjectionAdapterStatus::kNonFinite);
  expect_failure(
    adapter,
    Eigen::Vector2d(
      100.0, std::numeric_limits<double>::infinity()),
    livo::ProjectionAdapterStatus::kNonFinite);
}

TEST(
  PinholeCameraProjectionAdapter,
  ProjectsWithScaleInvariantRatiosAndMatchingJacobian)
{
  const livo::PinholeCameraProjectionAdapter adapter(
    outdoorFrontConfig());
  const Eigen::Vector3d point(0.5, -0.25, 2.0);
  const Eigen::Vector2d expected_pixel(
    480.88769475000004, 171.210998875);

  for (const double scale : {
      1.0e-200, 1.0e-100, 1.0, 1.0e100, 1.0e200})
  {
    const auto projection = adapter.project(scale * point);
    ASSERT_EQ(
      projection.status,
      livo::ProjectionAdapterStatus::kValid);
    EXPECT_TRUE(
      projection.pixel.isApprox(expected_pixel, 1.0e-12));
  }

  const auto jacobian = adapter.projectJacobian(point);
  ASSERT_EQ(
    jacobian.status, livo::ProjectionAdapterStatus::kValid);
  Eigen::Matrix<double, 2, 3> expected_jacobian;
  expected_jacobian <<
    301.5532835, 0.0, -75.388320875,
    0.0, 301.5514525, 37.6939315625;
  EXPECT_TRUE(
    jacobian.jacobian.isApprox(expected_jacobian, 1.0e-12));
}

TEST(
  PinholeCameraProjectionAdapter,
  ProjectLeavesFiniteOutOfFrameClassificationToContains)
{
  const livo::PinholeCameraProjectionAdapter adapter(
    outdoorFrontConfig());
  const Eigen::Vector3d point(2.0, 0.0, 1.0);

  const auto projection = adapter.project(point);
  ASSERT_EQ(
    projection.status, livo::ProjectionAdapterStatus::kValid);
  EXPECT_TRUE(projection.pixel.isApprox(
    Eigen::Vector2d(1536.324187, 246.598862), 1.0e-12));
  EXPECT_FALSE(adapter.contains(projection.pixel, 0));

  const auto jacobian = adapter.projectJacobian(point);
  EXPECT_EQ(
    jacobian.status, livo::ProjectionAdapterStatus::kValid);
  EXPECT_TRUE(jacobian.jacobian.allFinite());
}

TEST(
  PinholeCameraProjectionAdapter,
  ProjectionDistinguishesBehindNumericalDomainAndNonFinite)
{
  const livo::PinholeCameraProjectionAdapter adapter(
    outdoorFrontConfig());
  const auto expect_failure = [&adapter](
    const Eigen::Vector3d & point,
    const livo::ProjectionAdapterStatus expected_status) {
      const auto projection = adapter.project(point);
      EXPECT_EQ(projection.status, expected_status);
      EXPECT_TRUE(projection.pixel.array().isNaN().all());
      const auto jacobian = adapter.projectJacobian(point);
      EXPECT_EQ(jacobian.status, expected_status);
      EXPECT_TRUE(jacobian.jacobian.array().isNaN().all());
    };

  expect_failure(
    Eigen::Vector3d(0.0, 0.0, 0.0),
    livo::ProjectionAdapterStatus::kBehindCamera);
  expect_failure(
    Eigen::Vector3d(0.0, 0.0, -0.0),
    livo::ProjectionAdapterStatus::kBehindCamera);
  expect_failure(
    Eigen::Vector3d(0.0, 0.0, -1.0),
    livo::ProjectionAdapterStatus::kBehindCamera);
  expect_failure(
    Eigen::Vector3d(
      0.0, 0.0, std::numeric_limits<double>::denorm_min()),
    livo::ProjectionAdapterStatus::kOutOfDomain);
  expect_failure(
    Eigen::Vector3d(
      std::numeric_limits<double>::infinity(), 0.0, 1.0),
    livo::ProjectionAdapterStatus::kNonFinite);
  expect_failure(
    Eigen::Vector3d(
      0.0, std::numeric_limits<double>::quiet_NaN(), 1.0),
    livo::ProjectionAdapterStatus::kNonFinite);

  const auto minimum_normal_projection = adapter.project(
    Eigen::Vector3d(
      0.0, 0.0, std::numeric_limits<double>::min()));
  ASSERT_EQ(
    minimum_normal_projection.status,
    livo::ProjectionAdapterStatus::kValid);
  EXPECT_TRUE(minimum_normal_projection.pixel.isApprox(
    Eigen::Vector2d(330.111053, 246.598862), 0.0));
  const auto minimum_normal_jacobian = adapter.projectJacobian(
    Eigen::Vector3d(
      0.0, 0.0, std::numeric_limits<double>::min()));
  EXPECT_EQ(
    minimum_normal_jacobian.status,
    livo::ProjectionAdapterStatus::kNonFinite);
  EXPECT_TRUE(
    minimum_normal_jacobian.jacobian.array().isNaN().all());

  auto maximum_focal_config = outdoorFrontConfig();
  maximum_focal_config.fx =
    std::numeric_limits<double>::max();
  const livo::PinholeCameraProjectionAdapter maximum_focal(
    maximum_focal_config);
  const auto overflowed_projection = maximum_focal.project(
    Eigen::Vector3d(2.0, 0.0, 1.0));
  EXPECT_EQ(
    overflowed_projection.status,
    livo::ProjectionAdapterStatus::kNonFinite);
  EXPECT_TRUE(
    overflowed_projection.pixel.array().isNaN().all());
}

TEST(
  PinholeCameraProjectionAdapter,
  CopiesConfigurationIntoAReadOnlyRevisionedValue)
{
  auto config = outdoorFrontConfig();
  config.model_revision = 41U;
  const livo::PinholeCameraProjectionAdapter adapter(config);

  config.width = 2;
  config.height = 2;
  config.fx = 1.0;
  config.fy = 1.0;
  config.cx = 0.0;
  config.cy = 0.0;
  config.distortion_coefficients.fill(1.0);
  config.model_revision = 99U;

  EXPECT_EQ(adapter.width(), 640);
  EXPECT_EQ(adapter.height(), 480);
  EXPECT_DOUBLE_EQ(adapter.fx(), 603.106567);
  EXPECT_DOUBLE_EQ(adapter.fy(), 603.102905);
  EXPECT_DOUBLE_EQ(adapter.cx(), 330.111053);
  EXPECT_DOUBLE_EQ(adapter.cy(), 246.598862);
  EXPECT_EQ(
    adapter.distortionCoefficients(),
    (std::array<double, 4U>{{0.0, 0.0, 0.0, 0.0}}));
  EXPECT_EQ(adapter.modelRevision(), 41U);
}

TEST(
  CameraProjectionAdapter,
  ExposesTheConcreteModelsReadOnlyRevision)
{
  auto config = outdoorFrontConfig();
  config.model_revision = 73U;
  const livo::PinholeCameraProjectionAdapter pinhole(config);
  const livo::CameraProjectionAdapter & adapter = pinhole;

  EXPECT_EQ(adapter.modelRevision(), 73U);
}

TEST(
  PinholeCameraProjectionAdapter,
  AllOutdoorCameraIntrinsicsRoundTripInsideImagePixels)
{
  const std::array<Eigen::Vector2d, 5U> pixels{{
    Eigen::Vector2d(0.0, 0.0),
    Eigen::Vector2d(639.0, 479.0),
    Eigen::Vector2d(320.0, 240.0),
    Eigen::Vector2d(80.25, 399.75),
    Eigen::Vector2d(511.125, 96.875),
  }};

  for (const auto & config : outdoorCameraConfigs())
  {
    const livo::PinholeCameraProjectionAdapter adapter(config);
    for (const Eigen::Vector2d & pixel : pixels)
    {
      const auto ray = adapter.unproject(pixel);
      ASSERT_EQ(
        ray.status, livo::ProjectionAdapterStatus::kValid);
      EXPECT_DOUBLE_EQ(ray.direction_camera.z(), 1.0);

      const auto projection =
        adapter.project(7.25 * ray.direction_camera);
      ASSERT_EQ(
        projection.status,
        livo::ProjectionAdapterStatus::kValid);
      EXPECT_LE(
        (projection.pixel - pixel).cwiseAbs().maxCoeff(),
        1.0e-10);
    }
  }
}

TEST(
  PinholeCameraProjectionAdapter,
  AnalyticJacobiansMatchDeterministicFiniteDifferences)
{
  std::mt19937_64 generator(0x50494E484F4C45ULL);
  std::uniform_real_distribution<double> pixel_x(80.0, 560.0);
  std::uniform_real_distribution<double> pixel_y(80.0, 400.0);
  std::uniform_real_distribution<double> depth(0.2, 50.0);
  constexpr double kPixelStep = 1.0e-4;

  for (const auto & config : outdoorCameraConfigs())
  {
    const livo::PinholeCameraProjectionAdapter adapter(config);
    for (std::size_t sample = 0U; sample < 64U; ++sample)
    {
      const Eigen::Vector2d pixel(
        pixel_x(generator), pixel_y(generator));
      const auto analytic_unprojection =
        adapter.unprojectJacobian(pixel);
      ASSERT_EQ(
        analytic_unprojection.status,
        livo::ProjectionAdapterStatus::kValid);
      Eigen::Matrix<double, 3, 2> numeric_unprojection;
      for (Eigen::Index axis = 0; axis < 2; ++axis)
      {
        Eigen::Vector2d positive = pixel;
        Eigen::Vector2d negative = pixel;
        positive(axis) += kPixelStep;
        negative(axis) -= kPixelStep;
        const auto positive_ray = adapter.unproject(positive);
        const auto negative_ray = adapter.unproject(negative);
        ASSERT_EQ(
          positive_ray.status,
          livo::ProjectionAdapterStatus::kValid);
        ASSERT_EQ(
          negative_ray.status,
          livo::ProjectionAdapterStatus::kValid);
        numeric_unprojection.col(axis) =
          (positive_ray.direction_camera -
           negative_ray.direction_camera) /
          (2.0 * kPixelStep);
      }
      EXPECT_LE(
        (analytic_unprojection.jacobian -
         numeric_unprojection).cwiseAbs().maxCoeff(),
        2.0e-10);

      const auto ray = adapter.unproject(pixel);
      ASSERT_EQ(
        ray.status, livo::ProjectionAdapterStatus::kValid);
      const Eigen::Vector3d point =
        depth(generator) * ray.direction_camera;
      const auto analytic_projection =
        adapter.projectJacobian(point);
      ASSERT_EQ(
        analytic_projection.status,
        livo::ProjectionAdapterStatus::kValid);
      Eigen::Matrix<double, 2, 3> numeric_projection;
      for (Eigen::Index axis = 0; axis < 3; ++axis)
      {
        const double step =
          1.0e-7 * std::max(1.0, std::abs(point(axis)));
        Eigen::Vector3d positive = point;
        Eigen::Vector3d negative = point;
        positive(axis) += step;
        negative(axis) -= step;
        const auto positive_projection =
          adapter.project(positive);
        const auto negative_projection =
          adapter.project(negative);
        ASSERT_EQ(
          positive_projection.status,
          livo::ProjectionAdapterStatus::kValid);
        ASSERT_EQ(
          negative_projection.status,
          livo::ProjectionAdapterStatus::kValid);
        numeric_projection.col(axis) =
          (positive_projection.pixel -
           negative_projection.pixel) /
          (2.0 * step);
      }
      EXPECT_LE(
        (analytic_projection.jacobian -
         numeric_projection).cwiseAbs().maxCoeff(),
        2.0e-5);
    }
  }
}

}  // namespace
