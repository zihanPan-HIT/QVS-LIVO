#include "joint_visual_update.h"
#include "symmetric_information_validation.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>

namespace
{

TEST(
  SymmetricInformationValidationTest,
  ReportsComparableFailuresAndScaleForFixedSixAndNineteenDimensions)
{
  using Matrix6 = JointVisualNormalEquation::Matrix6;
  using Matrix19 = Eigen::Matrix<double, 19, 19>;

  const auto accepted =
    livo::validateSymmetricInformation(
    Matrix6::Identity(),
    livo::SymmetricInformationRequirement::kPositiveSemidefinite);
  EXPECT_TRUE(accepted.accepted());
  EXPECT_EQ(
    accepted.failure,
    livo::SymmetricInformationFailure::kNone);
  EXPECT_DOUBLE_EQ(accepted.scale, 1.0);
  EXPECT_TRUE(accepted.symmetric.isIdentity(0.0));

  Matrix6 non_symmetric = Matrix6::Identity();
  non_symmetric(0, 1) = 0.25;
  const auto asymmetric =
    livo::validateSymmetricInformation(
    non_symmetric,
    livo::SymmetricInformationRequirement::kPositiveSemidefinite);
  EXPECT_EQ(
    asymmetric.failure,
    livo::SymmetricInformationFailure::kNonSymmetric);
  EXPECT_DOUBLE_EQ(asymmetric.scale, 1.0);

  Matrix6 indefinite = Matrix6::Identity();
  indefinite(0, 0) = -0.25;
  const auto non_psd =
    livo::validateSymmetricInformation(
    indefinite,
    livo::SymmetricInformationRequirement::kPositiveSemidefinite);
  EXPECT_EQ(
    non_psd.failure,
    livo::SymmetricInformationFailure::kNotPositiveSemidefinite);
  EXPECT_DOUBLE_EQ(non_psd.scale, 1.0);

  Matrix6 singular = Matrix6::Identity();
  singular(5, 5) = 0.0;
  const auto non_spd =
    livo::validateSymmetricInformation(
    singular,
    livo::SymmetricInformationRequirement::kPositiveDefinite);
  EXPECT_EQ(
    non_spd.failure,
    livo::SymmetricInformationFailure::kNotPositiveDefinite);
  EXPECT_DOUBLE_EQ(non_spd.scale, 1.0);

  Matrix19 non_finite = Matrix19::Identity();
  non_finite(18, 18) =
    std::numeric_limits<double>::infinity();
  const auto invalid_19 =
    livo::validateSymmetricInformation(
    non_finite,
    livo::SymmetricInformationRequirement::kPositiveSemidefinite);
  EXPECT_EQ(
    invalid_19.failure,
    livo::SymmetricInformationFailure::kNonFinite);

  Matrix19 conditioned = Matrix19::Identity();
  conditioned(18, 18) = 1.0e-8;
  const auto conditioned_19 =
    livo::validateSymmetricInformation(
    conditioned,
    livo::SymmetricInformationRequirement::kPositiveDefinite);
  ASSERT_TRUE(conditioned_19.accepted());
  EXPECT_NEAR(conditioned_19.condition_number, 1.0e8, 1.0);
}

TEST(JointVisualUpdateTest, UsesConservativeDefaultPoseGates)
{
  const JointVisualUpdateConfig config;
  EXPECT_NEAR(
    config.max_rotation_increment_rad, 0.1 * M_PI / 180.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(config.max_translation_increment_m, 0.01);
}

TEST(JointVisualUpdateTest, InvalidConfigurationFailsFast)
{
  const auto expect_invalid = [](
    const JointVisualUpdateConfig & invalid_config)
    {
      EXPECT_THROW(
        JointVisualNormalEquation equation(invalid_config),
        std::invalid_argument);
    };

  JointVisualUpdateConfig config;
  config.min_tracked_points_per_camera = 0U;
  expect_invalid(config);

  config = JointVisualUpdateConfig{};
  config.min_joint_tracked_points = 0U;
  expect_invalid(config);

  config = JointVisualUpdateConfig{};
  config.min_joint_cameras = 0U;
  expect_invalid(config);

  config = JointVisualUpdateConfig{};
  config.min_joint_cameras = kVioCameraCount + 1U;
  expect_invalid(config);

  const std::array<double, 3> invalid_nonnegative{
    -1.0,
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN()};
  for (const double invalid : invalid_nonnegative)
  {
    config = JointVisualUpdateConfig{};
    config.max_residual_rmse = invalid;
    expect_invalid(config);

    config = JointVisualUpdateConfig{};
    config.max_rotation_increment_rad = invalid;
    expect_invalid(config);

    config = JointVisualUpdateConfig{};
    config.max_translation_increment_m = invalid;
    expect_invalid(config);
  }

  const std::array<double, 3> invalid_covariance_upper_bound{
    0.5,
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN()};
  for (const double invalid : invalid_covariance_upper_bound)
  {
    config = JointVisualUpdateConfig{};
    config.max_measurement_covariance_scale = invalid;
    expect_invalid(config);
  }

  config = JointVisualUpdateConfig{};
  config.max_residual_rmse = 0.0;
  config.max_rotation_increment_rad = 0.0;
  config.max_translation_increment_m = 0.0;
  config.max_measurement_covariance_scale = 1.0;
  EXPECT_NO_THROW(JointVisualNormalEquation equation(config));
}

TEST(JointVisualUpdateTest, NormalizesCorrelatedPixelsToOnePatchInformationUnit)
{
  JointVisualUpdateConfig config;
  config.min_tracked_points_per_camera = 1U;
  config.min_joint_tracked_points = 4U;
  config.min_joint_cameras = 4U;
  JointVisualNormalEquation equation(config);

  const JointVisualNormalEquation::Matrix6 pixel_hessian =
    64.0 * JointVisualNormalEquation::Matrix6::Identity();
  const JointVisualNormalEquation::Vector6 pixel_gradient =
    64.0 * JointVisualNormalEquation::Vector6::Ones();

  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_TRUE(equation.addCamera(
      camera_id, 1U, 64U, 64.0, pixel_hessian, pixel_gradient, 64U));
  }

  EXPECT_TRUE(equation.trackingReady());
  EXPECT_TRUE(equation.innovationAccepted());
  EXPECT_EQ(equation.contributingCameras(), 4U);
  EXPECT_EQ(equation.trackedPoints(), 4U);
  EXPECT_TRUE(equation.hessian().isApprox(
    4.0 * JointVisualNormalEquation::Matrix6::Identity()));
  EXPECT_TRUE(equation.gradient().isApprox(
    4.0 * JointVisualNormalEquation::Vector6::Ones()));
}

TEST(
  JointVisualUpdateTest,
  RejectsNonSymmetricOrIndefiniteHessianWithoutMutation)
{
  JointVisualUpdateConfig config;
  config.min_tracked_points_per_camera = 1U;
  JointVisualNormalEquation equation(config);
  const auto gradient =
    JointVisualNormalEquation::Vector6::Ones();

  JointVisualNormalEquation::Matrix6 non_symmetric =
    JointVisualNormalEquation::Matrix6::Identity();
  non_symmetric(0, 1) = 0.25;
  EXPECT_FALSE(equation.addCamera(
    0U, 1U, 1U, 1.0, non_symmetric, gradient, 1U));

  JointVisualNormalEquation::Matrix6 indefinite =
    JointVisualNormalEquation::Matrix6::Identity();
  indefinite(0, 0) = -0.25;
  EXPECT_FALSE(equation.addCamera(
    0U, 1U, 1U, 1.0, indefinite, gradient, 1U));

  EXPECT_EQ(equation.contributingCameras(), 0U);
  EXPECT_EQ(equation.trackedPoints(), 0U);
  EXPECT_EQ(equation.residualCount(), 0U);
  EXPECT_TRUE(equation.hessian().isZero());
  EXPECT_TRUE(equation.gradient().isZero());

  JointVisualNormalEquation::Matrix6 positive_semidefinite =
    JointVisualNormalEquation::Matrix6::Identity();
  positive_semidefinite(5, 5) = 0.0;
  EXPECT_TRUE(equation.addCamera(
    0U, 1U, 1U, 1.0,
    positive_semidefinite, gradient, 1U));
}

TEST(
  JointVisualUpdateTest,
  RejectsNonFiniteOrOverflowingAccumulationsAtomically)
{
  JointVisualUpdateConfig config;
  config.min_tracked_points_per_camera = 1U;
  JointVisualNormalEquation equation(config);

  JointVisualNormalEquation::Matrix6 non_finite =
    JointVisualNormalEquation::Matrix6::Identity();
  non_finite(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(equation.addCamera(
    0U, 1U, 1U, 1.0, non_finite,
    JointVisualNormalEquation::Vector6::Zero(), 1U));
  EXPECT_EQ(equation.contributingCameras(), 0U);
  EXPECT_TRUE(equation.hessian().isZero());

  const double large_finite =
    0.75 * std::numeric_limits<double>::max();
  JointVisualNormalEquation::Matrix6 large_hessian =
    JointVisualNormalEquation::Matrix6::Zero();
  large_hessian(0, 0) = large_finite;
  ASSERT_TRUE(equation.addCamera(
    0U, 1U, 1U, 1.0, large_hessian,
    JointVisualNormalEquation::Vector6::Zero(), 1U));
  const auto hessian_before_overflow = equation.hessian();
  const double rmse_before_overflow = equation.residualRmse();

  EXPECT_FALSE(equation.addCamera(
    1U, 1U, 1U, 1.0, large_hessian,
    JointVisualNormalEquation::Vector6::Zero(), 1U));
  EXPECT_TRUE((
    equation.hessian().array() ==
    hessian_before_overflow.array()).all());
  EXPECT_DOUBLE_EQ(
    equation.residualRmse(), rmse_before_overflow);
  EXPECT_EQ(equation.contributingCameras(), 1U);
  EXPECT_EQ(equation.trackedPoints(), 1U);
  EXPECT_EQ(equation.residualCount(), 1U);

  JointVisualNormalEquation gradient_equation(config);
  JointVisualNormalEquation::Vector6 large_gradient =
    JointVisualNormalEquation::Vector6::Zero();
  large_gradient(0) = large_finite;
  ASSERT_TRUE(gradient_equation.addCamera(
    0U, 1U, 1U, 1.0,
    JointVisualNormalEquation::Matrix6::Zero(),
    large_gradient, 1U));
  const auto gradient_before_overflow =
    gradient_equation.gradient();
  EXPECT_FALSE(gradient_equation.addCamera(
    1U, 1U, 1U, 1.0,
    JointVisualNormalEquation::Matrix6::Zero(),
    large_gradient, 1U));
  EXPECT_TRUE((
    gradient_equation.gradient().array() ==
    gradient_before_overflow.array()).all());
  EXPECT_EQ(gradient_equation.contributingCameras(), 1U);

  JointVisualNormalEquation residual_equation(config);
  ASSERT_TRUE(residual_equation.addCamera(
    0U, 1U, 1U, large_finite,
    JointVisualNormalEquation::Matrix6::Identity(),
    JointVisualNormalEquation::Vector6::Zero(), 1U));
  const auto residual_hessian_before_overflow =
    residual_equation.hessian();
  EXPECT_FALSE(residual_equation.addCamera(
    1U, 1U, 1U, large_finite,
    JointVisualNormalEquation::Matrix6::Identity(),
    JointVisualNormalEquation::Vector6::Zero(), 1U));
  EXPECT_TRUE((
    residual_equation.hessian().array() ==
    residual_hessian_before_overflow.array()).all());
  EXPECT_EQ(residual_equation.residualCount(), 1U);

  JointVisualNormalEquation count_equation(config);
  const std::size_t maximum_count =
    std::numeric_limits<std::size_t>::max();
  ASSERT_TRUE(count_equation.addCamera(
    0U, maximum_count, maximum_count, 1.0,
    JointVisualNormalEquation::Matrix6::Identity(),
    JointVisualNormalEquation::Vector6::Zero(), 1U));
  EXPECT_FALSE(count_equation.addCamera(
    1U, 1U, 1U, 1.0,
    JointVisualNormalEquation::Matrix6::Identity(),
    JointVisualNormalEquation::Vector6::Zero(), 1U));
  EXPECT_EQ(count_equation.contributingCameras(), 1U);
  EXPECT_EQ(count_equation.trackedPoints(), maximum_count);
  EXPECT_EQ(count_equation.residualCount(), maximum_count);
}

TEST(
  JointVisualUpdateTest,
  CovarianceScaleOnlyInflatesInformationAndHasSafeUpperBound)
{
  JointVisualUpdateConfig config;
  config.min_tracked_points_per_camera = 1U;
  JointVisualNormalEquation equation(config);
  const auto hessian =
    JointVisualNormalEquation::Matrix6::Identity();
  const auto gradient =
    JointVisualNormalEquation::Vector6::Ones();

  EXPECT_FALSE(equation.addCamera(
    0U, 1U, 1U, 1.0, hessian, gradient, 1U, 0.5));
  EXPECT_FALSE(equation.addCamera(
    0U, 1U, 1U, 1.0, hessian, gradient, 1U,
    std::numeric_limits<double>::max()));
  EXPECT_FALSE(equation.addCamera(
    0U, 1U, 1U, 1.0, hessian, gradient, 1U,
    std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(equation.addCamera(
    0U, 1U, 1U, 1.0, hessian, gradient, 1U,
    std::numeric_limits<double>::quiet_NaN()));
  EXPECT_EQ(equation.contributingCameras(), 0U);
  EXPECT_EQ(equation.trackedPoints(), 0U);
  EXPECT_EQ(equation.residualCount(), 0U);
  EXPECT_TRUE(equation.hessian().isZero());
  EXPECT_TRUE(equation.gradient().isZero());

  EXPECT_TRUE(equation.addCamera(
    0U, 1U, 1U, 1.0, hessian, gradient, 1U, 1.0));

  JointVisualUpdateConfig bounded_config = config;
  bounded_config.max_measurement_covariance_scale = 2.0;
  JointVisualNormalEquation bounded_equation(bounded_config);
  ASSERT_TRUE(bounded_equation.addCamera(
    0U, 1U, 1U, 1.0, hessian, gradient, 1U, 2.0));
  const auto bounded_hessian = bounded_equation.hessian();
  EXPECT_FALSE(bounded_equation.addCamera(
    1U, 1U, 1U, 1.0, hessian, gradient, 1U, 2.01));
  EXPECT_TRUE((
    bounded_equation.hessian().array() ==
    bounded_hessian.array()).all());
  EXPECT_EQ(bounded_equation.contributingCameras(), 1U);
}

TEST(JointVisualUpdateTest, RejectsSparseOrHighInnovationMeasurements)
{
  JointVisualUpdateConfig config;
  config.min_tracked_points_per_camera = 4U;
  config.min_joint_tracked_points = 8U;
  config.min_joint_cameras = 2U;
  config.max_residual_rmse = 20.0;
  JointVisualNormalEquation equation(config);

  const auto hessian = JointVisualNormalEquation::Matrix6::Identity();
  const auto gradient = JointVisualNormalEquation::Vector6::Zero();

  EXPECT_FALSE(equation.addCamera(0U, 3U, 192U, 192.0, hessian, gradient, 64U));
  EXPECT_TRUE(equation.addCamera(1U, 4U, 256U, 256.0, hessian, gradient, 64U));
  EXPECT_FALSE(equation.trackingReady());

  EXPECT_TRUE(equation.addCamera(
    2U, 4U, 256U, 256.0 * 30.0 * 30.0, hessian, gradient, 64U));
  EXPECT_TRUE(equation.trackingReady());
  EXPECT_FALSE(equation.innovationAccepted());
}

TEST(JointVisualUpdateTest, RejectsNonFiniteOrExcessivePoseIncrement)
{
  JointVisualUpdateConfig config;
  config.max_rotation_increment_rad = 2.0 * M_PI / 180.0;
  config.max_translation_increment_m = 0.10;
  JointVisualNormalEquation equation(config);

  EXPECT_TRUE(equation.incrementAccepted(0.01, 0.02));
  EXPECT_FALSE(equation.incrementAccepted(-0.01, 0.02));
  EXPECT_FALSE(equation.incrementAccepted(0.01, -0.02));
  EXPECT_FALSE(equation.incrementAccepted(0.01, 0.11));
  EXPECT_FALSE(equation.incrementAccepted(3.0 * M_PI / 180.0, 0.02));
  EXPECT_FALSE(equation.incrementAccepted(
    std::numeric_limits<double>::quiet_NaN(), 0.02));
}

}  // namespace
