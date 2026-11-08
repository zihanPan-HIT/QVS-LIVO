#include "full_state_visual_update.h"
#include "joint_visual_update.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace
{

template<typename T, typename = void>
struct HasConfigurableInformationUnits : std::false_type
{
};

template<typename T>
struct HasConfigurableInformationUnits<
  T,
  std::void_t<
    decltype(
      std::declval<T>().
      maximum_information_units_per_landmark)>>
: std::true_type
{
};

livo::CameraTimeTransferImuSample imuSample(
  const double timestamp_sec,
  const Eigen::Vector3d & angular_velocity,
  const Eigen::Vector3d & specific_force)
{
  livo::CameraTimeTransferImuSample sample;
  sample.timestamp_sec = timestamp_sec;
  sample.angular_velocity_radps = angular_velocity;
  sample.specific_force_mps2 = specific_force;
  return sample;
}

livo::FullStateVisualUpdateConfig permissiveConfig()
{
  livo::FullStateVisualUpdateConfig config;
  config.maximum_residual_rmse = 100.0;
  config.maximum_rotation_increment_rad = 10.0;
  config.maximum_translation_increment_m = 10.0;
  config.maximum_auxiliary_increment_norm = 10.0;
  config.maximum_prior_condition_number = 1.0e14;
  config.maximum_system_condition_number = 1.0e14;
  return config;
}

livo::FullStateVisualContribution simpleContribution(
  const livo::LandmarkId landmark_id,
  const std::uint8_t camera_id)
{
  livo::FullStateVisualContribution contribution;
  contribution.landmark_id = landmark_id;
  contribution.camera_id = camera_id;
  contribution.information_hessian(0, 0) = 1.0;
  contribution.information_gradient(0) = 0.1;
  contribution.squared_residual_sum = 1.0;
  contribution.residual_count = 1U;
  return contribution;
}

void expectAtomicFailure(
  const livo::FullStateVisualUpdateResult & result,
  const livo::FullStateVisualUpdateStatus expected_status)
{
  EXPECT_EQ(result.status, expected_status);
  EXPECT_FALSE(result.accepted());
  EXPECT_TRUE(result.increment.isZero(0.0));
  EXPECT_TRUE(result.posterior_covariance.isZero(0.0));
}

TEST(
  FullStateVisualUpdate,
  PoseOnlyZeroDtMatchesEquivalentSixDimensionalMapSolve)
{
  livo::FullStateVisualUpdateRequest request;
  request.batch_id = 101U;
  request.prior_covariance =
    2.0 * livo::FullStateVisualMatrix::Identity();

  livo::FullStateVisualContribution contribution;
  contribution.landmark_id = 11U;
  contribution.camera_id = 0U;
  Eigen::Matrix<double, 6, 6> pose_hessian =
    Eigen::Matrix<double, 6, 6>::Zero();
  pose_hessian.diagonal() <<
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  Eigen::Matrix<double, 6, 1> pose_gradient;
  pose_gradient << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6;
  contribution.information_hessian.topLeftCorner<6, 6>() =
    pose_hessian;
  contribution.information_gradient.head<6>() =
    pose_gradient;
  contribution.squared_residual_sum = 6.0;
  contribution.residual_count = 6U;
  request.contributions.push_back(contribution);

  const Eigen::Matrix<double, 6, 6> prior_pose_covariance =
    request.prior_covariance.topLeftCorner<6, 6>();
  const Eigen::Matrix<double, 6, 6> prior_pose_information =
    prior_pose_covariance.llt().solve(
      Eigen::Matrix<double, 6, 6>::Identity());
  const Eigen::Matrix<double, 6, 6> reference_system =
    prior_pose_information + pose_hessian;
  const Eigen::LLT<Eigen::Matrix<double, 6, 6>>
    reference_solver(reference_system);
  ASSERT_EQ(reference_solver.info(), Eigen::Success);
  const Eigen::Matrix<double, 6, 1> reference_increment =
    reference_solver.solve(-pose_gradient);
  const Eigen::Matrix<double, 6, 6> reference_posterior =
    reference_solver.solve(
      Eigen::Matrix<double, 6, 6>::Identity());

  const livo::FullStateVisualUpdate updater(permissiveConfig());
  const livo::FullStateVisualUpdateResult result =
    updater.solve(request);

  ASSERT_TRUE(result.accepted());
  EXPECT_TRUE(result.increment.head<6>().isApprox(
    reference_increment, 1.0e-12));
  EXPECT_TRUE(result.increment.tail<13>().isZero(0.0));
  EXPECT_TRUE((
    result.posterior_covariance.topLeftCorner<6, 6>().
    isApprox(reference_posterior, 1.0e-12)));
  EXPECT_DOUBLE_EQ(
    result.increment(livo::kFullStateVisualExposureOffset),
    0.0);
  EXPECT_TRUE(
    result.posterior_covariance.row(
      livo::kFullStateVisualExposureOffset).isApprox(
      request.prior_covariance.row(
        livo::kFullStateVisualExposureOffset),
      1.0e-12));
}

TEST(FullStateVisualUpdate, InvalidConfigurationFailsFast)
{
  static_assert(
    !HasConfigurableInformationUnits<
      livo::FullStateVisualUpdateConfig>::value,
    "one information unit per landmark must not be configurable");

  const auto expect_invalid = [](
    const livo::FullStateVisualUpdateConfig & config)
    {
      EXPECT_THROW(
        livo::FullStateVisualUpdate updater(config),
        std::invalid_argument);
    };

  livo::FullStateVisualUpdateConfig config = permissiveConfig();
  config.minimum_unique_landmarks = 0U;
  expect_invalid(config);

  config = permissiveConfig();
  config.minimum_cameras = 0U;
  expect_invalid(config);

  config = permissiveConfig();
  config.minimum_cameras = kVioCameraCount + 1U;
  expect_invalid(config);

  const std::array<double, 3> invalid_nonnegative{
    -1.0,
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN()};
  for (const double invalid : invalid_nonnegative)
  {
    config = permissiveConfig();
    config.maximum_residual_rmse = invalid;
    expect_invalid(config);

    config = permissiveConfig();
    config.maximum_rotation_increment_rad = invalid;
    expect_invalid(config);

    config = permissiveConfig();
    config.maximum_translation_increment_m = invalid;
    expect_invalid(config);

    config = permissiveConfig();
    config.maximum_auxiliary_increment_norm = invalid;
    expect_invalid(config);
  }

  const std::array<double, 3> invalid_condition_upper_bound{
    0.5,
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN()};
  for (const double invalid : invalid_condition_upper_bound)
  {
    config = permissiveConfig();
    config.maximum_prior_condition_number = invalid;
    expect_invalid(config);

    config = permissiveConfig();
    config.maximum_system_condition_number = invalid;
    expect_invalid(config);
  }

  config = permissiveConfig();
  config.maximum_residual_rmse = 0.0;
  config.maximum_rotation_increment_rad = 0.0;
  config.maximum_translation_increment_m = 0.0;
  config.maximum_auxiliary_increment_norm = 0.0;
  config.maximum_prior_condition_number = 1.0;
  config.maximum_system_condition_number = 1.0;
  EXPECT_NO_THROW(livo::FullStateVisualUpdate updater(config));
}

TEST(
  FullStateVisualUpdate,
  RequiresExactlyOneInformationUnitPerLandmark)
{
  const livo::FullStateVisualUpdate updater(permissiveConfig());
  for (const std::size_t invalid_units : {0U, 2U})
  {
    livo::FullStateVisualUpdateRequest request;
    request.batch_id = 129U + invalid_units;
    auto contribution =
      simpleContribution(15U + invalid_units, 0U);
    contribution.information_units = invalid_units;
    request.contributions.push_back(contribution);
    expectAtomicFailure(
      updater.solve(request),
      livo::FullStateVisualUpdateStatus::kInvalidContribution);
  }
}

TEST(
  FullStateVisualUpdate,
  InvalidRequestAndLayoutRevisionFailClosedAtomically)
{
  const livo::FullStateVisualUpdate updater(permissiveConfig());

  livo::FullStateVisualUpdateRequest zero_batch;
  zero_batch.contributions.push_back(simpleContribution(12U, 0U));
  expectAtomicFailure(
    updater.solve(zero_batch),
    livo::FullStateVisualUpdateStatus::kInvalidRequest);

  livo::FullStateVisualUpdateRequest empty_batch;
  empty_batch.batch_id = 126U;
  expectAtomicFailure(
    updater.solve(empty_batch),
    livo::FullStateVisualUpdateStatus::kInvalidRequest);

  livo::FullStateVisualUpdateRequest wrong_layout;
  wrong_layout.batch_id = 127U;
  wrong_layout.state_layout_id =
    std::string(livo::kFullStateVisualLayoutId) + " ";
  wrong_layout.contributions.push_back(simpleContribution(13U, 0U));
  expectAtomicFailure(
    updater.solve(wrong_layout),
    livo::FullStateVisualUpdateStatus::kInvalidLayout);

  livo::FullStateVisualUpdateRequest wrong_revision;
  wrong_revision.batch_id = 128U;
  wrong_revision.state_layout_revision =
    livo::kFullStateVisualLayoutRevision + 1U;
  wrong_revision.contributions.push_back(
    simpleContribution(14U, 0U));
  expectAtomicFailure(
    updater.solve(wrong_revision),
    livo::FullStateVisualUpdateStatus::kInvalidLayout);
}

TEST(
  FullStateVisualUpdate,
  AsynchronousTransferCreatesVelocityBiasAndGravityCoupling)
{
  livo::CameraTimeTransferConfig transfer_config;
  transfer_config.max_abs_transfer_sec = 0.1;
  transfer_config.max_imu_gap_sec = 0.02;
  livo::CameraTimeTransfer transfer_engine(transfer_config);
  livo::CameraTimeTransferState anchor;
  anchor.timestamp_sec = 0.0;
  const Eigen::Vector3d angular_velocity(0.2, -0.1, 0.3);
  const Eigen::Vector3d specific_force(1.0, -2.0, 0.5);
  const std::vector<livo::CameraTimeTransferImuSample> imu{
    imuSample(-0.01, angular_velocity, specific_force),
    imuSample(0.00, angular_velocity, specific_force),
    imuSample(0.01, angular_velocity, specific_force),
    imuSample(0.02, angular_velocity, specific_force),
    imuSample(0.03, angular_velocity, specific_force)};
  const livo::CameraPoseTransferData transfer_data =
    transfer_engine.transferWithSamples(anchor, imu, 0.02);
  const std::optional<livo::CameraPoseTransfer> transfer =
    transfer_data.validatedValue();
  ASSERT_TRUE(transfer.has_value());

  Eigen::Matrix<double, 6, 1> pose_residual;
  pose_residual << 0.2, -0.1, 0.15, -0.3, 0.25, -0.2;
  const auto & active_jacobian =
    transfer->raw().target_pose_wrt_active_anchor_state;
  livo::FullStateVisualContribution contribution;
  contribution.landmark_id = 21U;
  contribution.camera_id = 1U;
  contribution.information_hessian =
    active_jacobian.transpose() * active_jacobian;
  contribution.information_gradient =
    active_jacobian.transpose() * pose_residual;
  contribution.squared_residual_sum =
    pose_residual.squaredNorm();
  contribution.residual_count = 6U;

  EXPECT_TRUE(contribution.information_hessian.row(
    livo::kFullStateVisualExposureOffset).isZero(0.0));
  EXPECT_DOUBLE_EQ(
    contribution.information_gradient(
      livo::kFullStateVisualExposureOffset),
    0.0);

  livo::FullStateVisualUpdateRequest request;
  request.batch_id = 102U;
  request.prior_covariance =
    10.0 * livo::FullStateVisualMatrix::Identity();
  request.contributions.push_back(contribution);
  const livo::FullStateVisualUpdateResult result =
    livo::FullStateVisualUpdate(permissiveConfig()).solve(request);

  ASSERT_TRUE(result.accepted());
  EXPECT_GT(
    result.increment.segment<3>(
      livo::kFullStateVisualVelocityOffset).norm(),
    1.0e-12);
  EXPECT_GT(
    result.increment.segment<3>(
      livo::kFullStateVisualGyroBiasOffset).norm(),
    1.0e-12);
  EXPECT_GT(
    result.increment.segment<3>(
      livo::kFullStateVisualAccelBiasOffset).norm(),
    1.0e-12);
  EXPECT_GT(
    result.increment.segment<3>(
      livo::kFullStateVisualGravityOffset).norm(),
    1.0e-12);
  EXPECT_DOUBLE_EQ(
    result.increment(livo::kFullStateVisualExposureOffset),
    0.0);
}

TEST(
  FullStateVisualUpdate,
  GeometricContributionCannotUpdateInverseExposure)
{
  const livo::FullStateVisualUpdate updater(permissiveConfig());

  livo::FullStateVisualUpdateRequest direct_exposure;
  direct_exposure.batch_id = 103U;
  auto invalid_contribution = simpleContribution(31U, 0U);
  invalid_contribution.information_hessian(
    livo::kFullStateVisualExposureOffset,
    livo::kFullStateVisualExposureOffset) = 1.0;
  invalid_contribution.information_gradient(
    livo::kFullStateVisualExposureOffset) = 0.5;
  direct_exposure.contributions.push_back(invalid_contribution);
  const livo::FullStateVisualUpdateResult direct_result =
    updater.solve(direct_exposure);
  EXPECT_EQ(
    direct_result.status,
    livo::FullStateVisualUpdateStatus::kInvalidContribution);
  EXPECT_TRUE(direct_result.increment.isZero(0.0));
  EXPECT_TRUE(direct_result.posterior_covariance.isZero(0.0));

  livo::FullStateVisualUpdateRequest correlated_prior;
  correlated_prior.batch_id = 104U;
  correlated_prior.prior_covariance =
    livo::FullStateVisualMatrix::Identity();
  correlated_prior.prior_covariance(
    0, livo::kFullStateVisualExposureOffset) = 0.25;
  correlated_prior.prior_covariance(
    livo::kFullStateVisualExposureOffset, 0) = 0.25;
  correlated_prior.contributions.push_back(
    simpleContribution(32U, 0U));
  const livo::FullStateVisualUpdateResult coupled_result =
    updater.solve(correlated_prior);
  EXPECT_EQ(
    coupled_result.status,
    livo::FullStateVisualUpdateStatus::
    kExposureCouplingRejected);
  EXPECT_TRUE(coupled_result.increment.isZero(0.0));
  EXPECT_TRUE(coupled_result.posterior_covariance.isZero(0.0));
}

TEST(
  FullStateVisualUpdate,
  MultipleCamerasStackIntoOnePosteriorUpdate)
{
  livo::FullStateVisualUpdateConfig config = permissiveConfig();
  config.minimum_unique_landmarks = 2U;
  config.minimum_cameras = 2U;
  livo::FullStateVisualUpdateRequest request;
  request.batch_id = 105U;
  auto first = simpleContribution(41U, 0U);
  auto second = simpleContribution(42U, 1U);
  second.information_hessian.setZero();
  second.information_gradient.setZero();
  second.information_hessian(3, 3) = 2.0;
  second.information_gradient(3) = -0.2;
  request.contributions = {first, second};

  const livo::FullStateVisualUpdateResult result =
    livo::FullStateVisualUpdate(config).solve(request);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.unique_landmarks, 2U);
  EXPECT_EQ(result.contributing_cameras, 2U);
  EXPECT_EQ(result.information_units, 2U);
  EXPECT_NEAR(result.increment(0), -0.05, 1.0e-14);
  EXPECT_NEAR(result.increment(3), 0.2 / 3.0, 1.0e-14);
  EXPECT_NEAR(result.posterior_covariance(0, 0), 0.5, 1.0e-14);
  EXPECT_NEAR(
    result.posterior_covariance(3, 3), 1.0 / 3.0, 1.0e-14);
}

TEST(
  FullStateVisualUpdate,
  DuplicateLandmarkAcrossCamerasRejectsWholeBatch)
{
  livo::FullStateVisualUpdateRequest request;
  request.batch_id = 106U;
  request.contributions = {
    simpleContribution(51U, 0U),
    simpleContribution(51U, 1U)};

  const livo::FullStateVisualUpdateResult result =
    livo::FullStateVisualUpdate(permissiveConfig()).solve(request);

  EXPECT_EQ(
    result.status,
    livo::FullStateVisualUpdateStatus::kDuplicateLandmark);
  EXPECT_TRUE(result.increment.isZero(0.0));
  EXPECT_TRUE(result.posterior_covariance.isZero(0.0));
}

TEST(
  FullStateVisualUpdate,
  NonSymmetricNonPsdOrNonFiniteContributionFailsAtomically)
{
  const livo::FullStateVisualUpdate updater(permissiveConfig());
  const auto solve_one = [&updater](
    const livo::FullStateVisualContribution & contribution,
    const std::uint64_t batch_id)
    {
      livo::FullStateVisualUpdateRequest request;
      request.batch_id = batch_id;
      request.contributions.push_back(contribution);
      return updater.solve(request);
    };

  auto non_symmetric = simpleContribution(61U, 0U);
  non_symmetric.information_hessian(0, 1) = 0.25;
  expectAtomicFailure(
    solve_one(non_symmetric, 107U),
    livo::FullStateVisualUpdateStatus::kInvalidContribution);

  auto non_psd = simpleContribution(62U, 0U);
  non_psd.information_hessian(0, 0) = -2.0;
  expectAtomicFailure(
    solve_one(non_psd, 108U),
    livo::FullStateVisualUpdateStatus::kInvalidContribution);

  auto non_finite_hessian = simpleContribution(63U, 0U);
  non_finite_hessian.information_hessian(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  expectAtomicFailure(
    solve_one(non_finite_hessian, 109U),
    livo::FullStateVisualUpdateStatus::kInvalidContribution);

  auto non_finite_gradient = simpleContribution(64U, 0U);
  non_finite_gradient.information_gradient(0) =
    std::numeric_limits<double>::infinity();
  expectAtomicFailure(
    solve_one(non_finite_gradient, 110U),
    livo::FullStateVisualUpdateStatus::kInvalidContribution);
}

TEST(
  FullStateVisualUpdate,
  SixAndNineteenStateModulesRejectTheSameMaliciousMatrices)
{
  JointVisualUpdateConfig joint_config;
  joint_config.min_tracked_points_per_camera = 1U;
  const auto expect_both_reject = [&joint_config](
    const JointVisualNormalEquation::Matrix6 & malicious,
    const livo::LandmarkId landmark_id,
    const std::uint64_t batch_id)
    {
      JointVisualNormalEquation joint(joint_config);
      EXPECT_FALSE(joint.addCamera(
        0U, 1U, 1U, 1.0, malicious,
        JointVisualNormalEquation::Vector6::Zero(), 1U));
      EXPECT_TRUE(joint.hessian().isZero(0.0));

      livo::FullStateVisualContribution contribution;
      contribution.landmark_id = landmark_id;
      contribution.camera_id = 0U;
      contribution.information_hessian.topLeftCorner<6, 6>() =
        malicious;
      contribution.squared_residual_sum = 1.0;
      contribution.residual_count = 1U;
      livo::FullStateVisualUpdateRequest request;
      request.batch_id = batch_id;
      request.contributions.push_back(contribution);
      expectAtomicFailure(
        livo::FullStateVisualUpdate(permissiveConfig()).solve(
          request),
        livo::FullStateVisualUpdateStatus::kInvalidContribution);
    };

  JointVisualNormalEquation::Matrix6 non_finite =
    JointVisualNormalEquation::Matrix6::Identity();
  non_finite(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  expect_both_reject(non_finite, 501U, 131U);

  JointVisualNormalEquation::Matrix6 non_symmetric =
    JointVisualNormalEquation::Matrix6::Identity();
  non_symmetric(0, 1) = 0.25;
  expect_both_reject(non_symmetric, 502U, 132U);

  JointVisualNormalEquation::Matrix6 indefinite =
    JointVisualNormalEquation::Matrix6::Identity();
  indefinite(0, 0) = -0.25;
  expect_both_reject(indefinite, 503U, 133U);
}

TEST(
  FullStateVisualUpdate,
  PriorMustBeFiniteSymmetricSpdAndWellConditioned)
{
  livo::FullStateVisualUpdateConfig config = permissiveConfig();
  config.maximum_prior_condition_number = 1.0e8;
  const livo::FullStateVisualUpdate updater(config);
  const auto solve_with_prior = [&updater](
    const livo::FullStateVisualMatrix & prior,
    const std::uint64_t batch_id)
    {
      livo::FullStateVisualUpdateRequest request;
      request.batch_id = batch_id;
      request.prior_covariance = prior;
      request.contributions.push_back(
        simpleContribution(70U + batch_id, 0U));
      return updater.solve(request);
    };

  livo::FullStateVisualMatrix non_symmetric =
    livo::FullStateVisualMatrix::Identity();
  non_symmetric(0, 1) = 0.1;
  expectAtomicFailure(
    solve_with_prior(non_symmetric, 111U),
    livo::FullStateVisualUpdateStatus::kInvalidPrior);

  livo::FullStateVisualMatrix non_spd =
    livo::FullStateVisualMatrix::Identity();
  non_spd(0, 0) = -1.0;
  expectAtomicFailure(
    solve_with_prior(non_spd, 112U),
    livo::FullStateVisualUpdateStatus::kInvalidPrior);

  livo::FullStateVisualMatrix non_finite =
    livo::FullStateVisualMatrix::Identity();
  non_finite(0, 0) =
    std::numeric_limits<double>::quiet_NaN();
  expectAtomicFailure(
    solve_with_prior(non_finite, 113U),
    livo::FullStateVisualUpdateStatus::kInvalidPrior);

  livo::FullStateVisualMatrix ill_conditioned =
    livo::FullStateVisualMatrix::Identity();
  ill_conditioned(0, 0) = 1.0e-10;
  expectAtomicFailure(
    solve_with_prior(ill_conditioned, 114U),
    livo::FullStateVisualUpdateStatus::kIllConditionedPrior);
}

TEST(
  FullStateVisualUpdate,
  MatrixScalarAndCounterOverflowRejectWholeBatch)
{
  const livo::FullStateVisualUpdate updater(permissiveConfig());
  const auto solve_pair = [&updater](
    livo::FullStateVisualContribution first,
    livo::FullStateVisualContribution second,
    const std::uint64_t batch_id)
    {
      livo::FullStateVisualUpdateRequest request;
      request.batch_id = batch_id;
      request.contributions = {
        std::move(first), std::move(second)};
      return updater.solve(request);
    };
  const double large =
    0.75 * std::numeric_limits<double>::max();

  auto hessian_first = simpleContribution(201U, 0U);
  auto hessian_second = simpleContribution(202U, 1U);
  hessian_first.information_hessian.setZero();
  hessian_second.information_hessian.setZero();
  hessian_first.information_hessian(0, 0) = large;
  hessian_second.information_hessian(0, 0) = large;
  expectAtomicFailure(
    solve_pair(hessian_first, hessian_second, 115U),
    livo::FullStateVisualUpdateStatus::kAccumulationOverflow);

  auto gradient_first = simpleContribution(203U, 0U);
  auto gradient_second = simpleContribution(204U, 1U);
  gradient_first.information_hessian.setZero();
  gradient_second.information_hessian.setZero();
  gradient_first.information_gradient.setZero();
  gradient_second.information_gradient.setZero();
  gradient_first.information_gradient(0) = large;
  gradient_second.information_gradient(0) = large;
  expectAtomicFailure(
    solve_pair(gradient_first, gradient_second, 116U),
    livo::FullStateVisualUpdateStatus::kAccumulationOverflow);

  auto residual_first = simpleContribution(205U, 0U);
  auto residual_second = simpleContribution(206U, 1U);
  residual_first.squared_residual_sum = large;
  residual_second.squared_residual_sum = large;
  expectAtomicFailure(
    solve_pair(residual_first, residual_second, 117U),
    livo::FullStateVisualUpdateStatus::kAccumulationOverflow);

  auto count_first = simpleContribution(207U, 0U);
  auto count_second = simpleContribution(208U, 1U);
  count_first.residual_count =
    std::numeric_limits<std::size_t>::max();
  count_second.residual_count = 1U;
  expectAtomicFailure(
    solve_pair(count_first, count_second, 118U),
    livo::FullStateVisualUpdateStatus::kAccumulationOverflow);

}

TEST(
  FullStateVisualUpdate,
  InformationInnovationConditionAndIncrementGatesFailAtomically)
{
  livo::FullStateVisualUpdateRequest request;
  request.batch_id = 120U;
  request.contributions.push_back(simpleContribution(301U, 0U));

  livo::FullStateVisualUpdateConfig insufficient =
    permissiveConfig();
  insufficient.minimum_unique_landmarks = 2U;
  insufficient.minimum_cameras = 2U;
  expectAtomicFailure(
    livo::FullStateVisualUpdate(insufficient).solve(request),
    livo::FullStateVisualUpdateStatus::kInsufficientInformation);

  livo::FullStateVisualUpdateConfig innovation =
    permissiveConfig();
  innovation.maximum_residual_rmse = 0.5;
  expectAtomicFailure(
    livo::FullStateVisualUpdate(innovation).solve(request),
    livo::FullStateVisualUpdateStatus::kInnovationRejected);

  livo::FullStateVisualUpdateConfig rotation =
    permissiveConfig();
  rotation.maximum_rotation_increment_rad = 0.01;
  expectAtomicFailure(
    livo::FullStateVisualUpdate(rotation).solve(request),
    livo::FullStateVisualUpdateStatus::kIncrementRejected);

  livo::FullStateVisualUpdateRequest translation_request;
  translation_request.batch_id = 121U;
  auto translation_contribution = simpleContribution(302U, 0U);
  translation_contribution.information_hessian.setZero();
  translation_contribution.information_gradient.setZero();
  translation_contribution.information_hessian(3, 3) = 1.0;
  translation_contribution.information_gradient(3) = 0.1;
  translation_request.contributions.push_back(
    translation_contribution);
  livo::FullStateVisualUpdateConfig translation =
    permissiveConfig();
  translation.maximum_translation_increment_m = 0.01;
  expectAtomicFailure(
    livo::FullStateVisualUpdate(translation).solve(
      translation_request),
    livo::FullStateVisualUpdateStatus::kIncrementRejected);

  livo::FullStateVisualUpdateRequest auxiliary_request;
  auxiliary_request.batch_id = 122U;
  auto auxiliary_contribution = simpleContribution(303U, 0U);
  auxiliary_contribution.information_hessian.setZero();
  auxiliary_contribution.information_gradient.setZero();
  auxiliary_contribution.information_hessian(
    livo::kFullStateVisualVelocityOffset,
    livo::kFullStateVisualVelocityOffset) = 1.0;
  auxiliary_contribution.information_gradient(
    livo::kFullStateVisualVelocityOffset) = 0.1;
  auxiliary_request.contributions.push_back(
    auxiliary_contribution);
  livo::FullStateVisualUpdateConfig auxiliary =
    permissiveConfig();
  auxiliary.maximum_auxiliary_increment_norm = 0.01;
  expectAtomicFailure(
    livo::FullStateVisualUpdate(auxiliary).solve(
      auxiliary_request),
    livo::FullStateVisualUpdateStatus::kIncrementRejected);

  livo::FullStateVisualUpdateRequest conditioned_request;
  conditioned_request.batch_id = 123U;
  auto strong = simpleContribution(304U, 0U);
  strong.information_hessian(0, 0) = 100.0;
  conditioned_request.contributions.push_back(strong);
  livo::FullStateVisualUpdateConfig conditioned =
    permissiveConfig();
  conditioned.maximum_system_condition_number = 10.0;
  expectAtomicFailure(
    livo::FullStateVisualUpdate(conditioned).solve(
      conditioned_request),
    livo::FullStateVisualUpdateStatus::kIllConditionedSystem);
}

TEST(
  FullStateVisualUpdate,
  WeakAndStrongInformationMatchHighPrecisionReferenceAndContractPrior)
{
  livo::FullStateVisualMatrix prior_cholesky =
    livo::FullStateVisualMatrix::Zero();
  for (Eigen::Index row = 0; row < prior_cholesky.rows(); ++row)
  {
    prior_cholesky(row, row) =
      0.5 + 0.03 * static_cast<double>(row);
    for (Eigen::Index column = 0; column < row; ++column)
    {
      if (row != livo::kFullStateVisualExposureOffset &&
          column != livo::kFullStateVisualExposureOffset &&
          (row + column) % 4 == 0)
      {
        prior_cholesky(row, column) =
          0.005 * static_cast<double>(1 + (row + column) % 3);
      }
    }
  }
  const livo::FullStateVisualMatrix prior =
    prior_cholesky * prior_cholesky.transpose();

  Eigen::Matrix<double, 8, livo::kFullStateVisualDimension>
    jacobian;
  for (Eigen::Index row = 0; row < jacobian.rows(); ++row)
  {
    for (Eigen::Index column = 0; column < jacobian.cols(); ++column)
    {
      jacobian(row, column) =
        column == livo::kFullStateVisualExposureOffset ?
        0.0 :
        0.01 * static_cast<double>(
        (row + 1) * (1 + (column % 5)));
    }
  }
  Eigen::Matrix<double, 8, 1> residual;
  residual <<
    0.2, -0.1, 0.3, -0.25, 0.15, -0.2, 0.1, -0.05;

  using LongMatrix =
    Eigen::Matrix<long double, livo::kFullStateVisualDimension,
      livo::kFullStateVisualDimension>;
  using LongVector =
    Eigen::Matrix<long double, livo::kFullStateVisualDimension, 1>;
  const auto high_precision_reference = [&prior](
    const livo::FullStateVisualMatrix & hessian,
    const livo::FullStateVisualVector & gradient)
    {
      const LongMatrix prior_long = prior.cast<long double>();
      const LongMatrix hessian_long = hessian.cast<long double>();
      const LongVector gradient_long = gradient.cast<long double>();
      const Eigen::LLT<LongMatrix> prior_solver(prior_long);
      const LongMatrix prior_information = prior_solver.solve(
        LongMatrix::Identity());
      const LongMatrix system =
        prior_information + hessian_long;
      const Eigen::LLT<LongMatrix> system_solver(system);
      const livo::FullStateVisualVector increment =
        system_solver.solve(-gradient_long).
        template cast<double>();
      const livo::FullStateVisualMatrix posterior =
        system_solver.solve(LongMatrix::Identity()).
        template cast<double>();
      return std::make_pair(increment, posterior);
    };

  livo::FullStateVisualUpdateConfig config = permissiveConfig();
  config.maximum_rotation_increment_rad = 1.0e6;
  config.maximum_translation_increment_m = 1.0e6;
  config.maximum_auxiliary_increment_norm = 1.0e6;
  const livo::FullStateVisualUpdate updater(config);
  double weak_trace = 0.0;
  double strong_trace = 0.0;
  for (const double strength : {1.0e-6, 1.0e4})
  {
    livo::FullStateVisualContribution contribution;
    contribution.landmark_id =
      strength < 1.0 ? 401U : 402U;
    contribution.camera_id = 0U;
    contribution.information_hessian =
      strength * jacobian.transpose() * jacobian;
    contribution.information_gradient =
      strength * jacobian.transpose() * residual;
    contribution.squared_residual_sum = residual.squaredNorm();
    contribution.residual_count =
      static_cast<std::size_t>(residual.rows());

    livo::FullStateVisualUpdateRequest request;
    request.batch_id = strength < 1.0 ? 124U : 125U;
    request.prior_covariance = prior;
    request.contributions.push_back(contribution);
    const auto reference = high_precision_reference(
      contribution.information_hessian,
      contribution.information_gradient);
    const livo::FullStateVisualUpdateResult result =
      updater.solve(request);

    ASSERT_TRUE(result.accepted());
    EXPECT_LT(
      (result.increment - reference.first).
      cwiseAbs().maxCoeff(),
      1.0e-10);
    EXPECT_LT(
      (result.posterior_covariance - reference.second).
      cwiseAbs().maxCoeff(),
      1.0e-10);
    EXPECT_TRUE(result.posterior_covariance.isApprox(
      result.posterior_covariance.transpose(), 1.0e-12));
    const Eigen::SelfAdjointEigenSolver<
      livo::FullStateVisualMatrix> posterior_solver(
      result.posterior_covariance);
    ASSERT_EQ(posterior_solver.info(), Eigen::Success);
    EXPECT_GT(posterior_solver.eigenvalues().minCoeff(), 0.0);
    const livo::FullStateVisualMatrix reduction =
      0.5 * (prior - result.posterior_covariance) +
      0.5 * (prior - result.posterior_covariance).transpose();
    const Eigen::SelfAdjointEigenSolver<
      livo::FullStateVisualMatrix> reduction_solver(reduction);
    ASSERT_EQ(reduction_solver.info(), Eigen::Success);
    EXPECT_GE(
      reduction_solver.eigenvalues().minCoeff(), -1.0e-10);
    EXPECT_TRUE(std::isfinite(result.prior_condition_number));
    EXPECT_TRUE(std::isfinite(result.system_condition_number));
    if (strength < 1.0)
    {
      weak_trace = result.posterior_covariance.trace();
    }
    else
    {
      strong_trace = result.posterior_covariance.trace();
    }
  }
  EXPECT_LT(strong_trace, weak_trace);
}

}  // namespace
