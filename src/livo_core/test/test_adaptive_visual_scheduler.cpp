#include "adaptive_visual_policy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace
{

using livo::AdaptiveCameraObservation;
using livo::AdaptiveCameraHealthState;
using livo::AdaptiveCameraTransition;
using livo::AdaptiveVisualPolicy;
using livo::AdaptiveVisualPolicyConfig;

AdaptiveCameraObservation healthyObservation(const std::uint8_t camera_id)
{
  AdaptiveCameraObservation observation;
  observation.camera_id = camera_id;
  observation.frame_received = true;
  observation.available_metrics = livo::kAdaptiveFirstVersionMetricMask;
  observation.candidate_points = 40U;
  observation.tracked_points = 24U;
  observation.residual_count = 24U * 64U;
  observation.squared_residual_sum =
    static_cast<double>(observation.residual_count) * 25.0;
  observation.saturation_fraction = 0.0;
  observation.dark_fraction = 0.0;
  observation.mean_gradient = 16.0;
  observation.spatial_coverage = 0.30;
  observation.visual_information =
    AdaptiveCameraObservation::Matrix6::Identity();
  observation.available_metrics |=
    livo::kAdaptiveRobustAdmissionMetricMask;
  observation.inlier_count = 24U;
  observation.total_patch_count = 32U;
  observation.inlier_ratio = 0.75;
  observation.nis = 1.0;
  observation.nis_dof = 24U;
  observation.normalized_pose_correction = 0.01;
  observation.robust_information_gain = 1.0;
  return observation;
}

AdaptiveCameraObservation unhealthyButAvailableObservation(
  const std::uint8_t camera_id)
{
  AdaptiveCameraObservation observation = healthyObservation(camera_id);
  observation.candidate_points = 100U;
  observation.tracked_points = 1U;
  observation.residual_count = 64U;
  observation.squared_residual_sum = 64.0 * 40.0 * 40.0;
  observation.saturation_fraction = 0.25;
  observation.dark_fraction = 0.35;
  observation.mean_gradient = 0.1;
  observation.spatial_coverage = 0.001;
  return observation;
}

AdaptiveCameraObservation droppedObservation(const std::uint8_t camera_id)
{
  AdaptiveCameraObservation observation;
  observation.camera_id = camera_id;
  observation.frame_received = false;
  return observation;
}

AdaptiveCameraObservation imageOnlyObservation(
  const std::uint8_t camera_id,
  const double saturation_fraction = 0.0,
  const double dark_fraction = 0.0,
  const double mean_gradient = 16.0)
{
  AdaptiveCameraObservation observation;
  observation.camera_id = camera_id;
  observation.frame_received = true;
  observation.available_metrics =
    livo::adaptiveMetricMask(
      livo::AdaptiveHealthMetric::kCandidateCount) |
    livo::adaptiveMetricMask(
      livo::AdaptiveHealthMetric::kTrackedCount) |
    livo::adaptiveMetricMask(
      livo::AdaptiveHealthMetric::kSaturationFraction) |
    livo::adaptiveMetricMask(
      livo::AdaptiveHealthMetric::kDarkFraction) |
    livo::adaptiveMetricMask(
      livo::AdaptiveHealthMetric::kMeanGradient) |
    livo::adaptiveMetricMask(
      livo::AdaptiveHealthMetric::kSpatialCoverage);
  observation.saturation_fraction = saturation_fraction;
  observation.dark_fraction = dark_fraction;
  observation.mean_gradient = mean_gradient;
  return observation;
}

AdaptiveCameraObservation trackingCollapsedObservation(
  const std::uint8_t camera_id,
  const double mean_gradient = 16.0)
{
  AdaptiveCameraObservation observation =
    imageOnlyObservation(camera_id, 0.0, 0.0, mean_gradient);
  observation.candidate_points = 40U;
  observation.tracked_points = 0U;
  return observation;
}

AdaptiveCameraObservation uninitializedZeroMapObservation(
  const std::uint8_t camera_id,
  const double saturation_fraction = 0.0,
  const double dark_fraction = 0.0,
  const double mean_gradient = 16.0)
{
  AdaptiveCameraObservation observation =
    imageOnlyObservation(
    camera_id, saturation_fraction, dark_fraction,
    mean_gradient);
  observation.tracking_map_initialized = false;
  return observation;
}

AdaptiveVisualPolicy::CameraObservations healthyRig()
{
  AdaptiveVisualPolicy::CameraObservations observations;
  for (std::size_t camera_id = 0U; camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] =
      healthyObservation(static_cast<std::uint8_t>(camera_id));
  }
  return observations;
}

AdaptiveVisualPolicy::Matrix6 physicalInformationForNormalizedDiagonal(
  const AdaptiveVisualPolicyConfig & config,
  const std::array<double, 6U> & normalized_diagonal)
{
  AdaptiveVisualPolicy::Matrix6 information =
    AdaptiveVisualPolicy::Matrix6::Zero();
  for (std::size_t index = 0U; index < normalized_diagonal.size(); ++index)
  {
    const double scale =
      index < 3U ? config.rotation_scale_rad : config.translation_scale_m;
    information(
      static_cast<Eigen::Index>(index),
      static_cast<Eigen::Index>(index)) =
      normalized_diagonal[index] / (scale * scale);
  }
  return information;
}

std::size_t candidateBudgetSum(
  const livo::AdaptiveVisualPolicyPlan & plan)
{
  return std::accumulate(
    plan.cameras.begin(), plan.cameras.end(), std::size_t{0U},
    [](const std::size_t sum, const livo::AdaptiveCameraDecision & camera)
    {
      return sum + camera.candidate_budget;
    });
}

std::size_t patchBudgetSum(const livo::AdaptiveVisualPolicyPlan & plan)
{
  return std::accumulate(
    plan.cameras.begin(), plan.cameras.end(), std::size_t{0U},
    [](const std::size_t sum, const livo::AdaptiveCameraDecision & camera)
    {
      return sum + camera.patch_budget;
    });
}

TEST(AdaptiveVisualSchedulerTest, DetectsWeakDirectionAfterUnitNormalization)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.rotation_scale_rad = 0.01;
  config.translation_scale_m = 0.10;
  config.weak_relative_eigenvalue = 0.05;
  const AdaptiveVisualPolicy policy(config);
  const auto lidar_information = physicalInformationForNormalizedDiagonal(
    config, {0.1, 10.0, 20.0, 30.0, 40.0, 50.0});

  const auto plan = policy.plan(lidar_information, healthyRig());

  ASSERT_TRUE(plan.lidar_information_valid);
  EXPECT_EQ(plan.weak_direction_count, 1U);
  EXPECT_NEAR(plan.lidar_eigenvalues.minCoeff(), 0.1, 1.0e-9);
  EXPECT_NEAR(plan.lidar_eigenvalues.maxCoeff(), 50.0, 1.0e-9);
  EXPECT_GT(plan.weak_direction_weight(0, 0), 0.9);
  EXPECT_NEAR(plan.weak_direction_weight(5, 5), 0.0, 1.0e-12);
}

TEST(AdaptiveVisualSchedulerTest, DisabledPolicyPassesAllCamerasThrough)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = false;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  for (std::size_t camera_id = 0U; camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] =
      unhealthyButAvailableObservation(
        static_cast<std::uint8_t>(camera_id));
  }

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Zero(), observations);

  EXPECT_FALSE(plan.adaptive_enabled);
  for (const auto & camera : plan.cameras)
  {
    EXPECT_TRUE(camera.enabled);
    EXPECT_FALSE(camera.isolated);
    EXPECT_TRUE(camera.metrics_complete);
    EXPECT_FALSE(camera.health_degraded);
    EXPECT_DOUBLE_EQ(camera.health, 1.0);
    EXPECT_DOUBLE_EQ(camera.covariance_scale, 1.0);
  }
  EXPECT_FALSE(policy.commit(plan, true));
  for (const auto & state : policy.hysteresis())
  {
    EXPECT_FALSE(state.isolated);
    EXPECT_EQ(state.consecutive_unhealthy, 0U);
    EXPECT_EQ(state.consecutive_healthy, 0U);
  }
}

TEST(
  AdaptiveVisualSchedulerTest,
  RejectsNonFiniteDoubleConfigurationValues)
{
  constexpr std::array<
    double AdaptiveVisualPolicyConfig::*, 18U> kDoubleFields{
    &AdaptiveVisualPolicyConfig::rotation_scale_rad,
    &AdaptiveVisualPolicyConfig::translation_scale_m,
    &AdaptiveVisualPolicyConfig::weak_relative_eigenvalue,
    &AdaptiveVisualPolicyConfig::weak_absolute_eigenvalue,
    &AdaptiveVisualPolicyConfig::health_off_threshold,
    &AdaptiveVisualPolicyConfig::health_on_threshold,
    &AdaptiveVisualPolicyConfig::hard_reject_health,
    &AdaptiveVisualPolicyConfig::minimum_probation_duration_sec,
    &AdaptiveVisualPolicyConfig::covariance_health_floor,
    &AdaptiveVisualPolicyConfig::maximum_covariance_scale,
    &AdaptiveVisualPolicyConfig::complement_score_gain,
    &AdaptiveVisualPolicyConfig::selection_switch_margin,
    &AdaptiveVisualPolicyConfig::healthy_tracking_ratio,
    &AdaptiveVisualPolicyConfig::healthy_residual_rmse,
    &AdaptiveVisualPolicyConfig::maximum_saturation_fraction,
    &AdaptiveVisualPolicyConfig::maximum_dark_fraction,
    &AdaptiveVisualPolicyConfig::healthy_mean_gradient,
    &AdaptiveVisualPolicyConfig::healthy_spatial_coverage};
  const std::array<double, 3U> non_finite_values{
    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::infinity()};

  for (std::size_t field_index = 0U;
       field_index < kDoubleFields.size(); ++field_index)
  {
    SCOPED_TRACE(field_index);
    for (const double value : non_finite_values)
    {
      SCOPED_TRACE(value);
      AdaptiveVisualPolicyConfig config;
      config.*kDoubleFields[field_index] = value;
      EXPECT_THROW(
        {AdaptiveVisualPolicy policy(config);},
        std::invalid_argument);
    }
  }
}

TEST(
  AdaptiveVisualSchedulerTest,
  HardRejectHealthCannotExceedIsolationThreshold)
{
  AdaptiveVisualPolicyConfig config;
  config.hard_reject_health = config.health_off_threshold;
  EXPECT_NO_THROW({AdaptiveVisualPolicy policy(config);});

  config.hard_reject_health = std::nextafter(
    config.health_off_threshold,
    std::numeric_limits<double>::infinity());
  EXPECT_THROW(
    {AdaptiveVisualPolicy policy(config);},
    std::invalid_argument);
}

TEST(
  AdaptiveVisualSchedulerTest,
  RejectsZeroSelectionChallengerFrames)
{
  AdaptiveVisualPolicyConfig config;
  config.selection_challenger_frames = 0U;

  EXPECT_THROW(
    {AdaptiveVisualPolicy policy(config);},
    std::invalid_argument);
}

TEST(
  AdaptiveVisualSchedulerTest,
  BudgetValidationRejectsOverflowAndAcceptsLargestSafeProduct)
{
  constexpr std::size_t kMaximum =
    std::numeric_limits<std::size_t>::max();
  constexpr std::size_t kLargestSafeMinimum = kMaximum / 2U;
  constexpr std::size_t kOverflowingMinimum =
    kLargestSafeMinimum + 1U;

  AdaptiveVisualPolicyConfig safe_candidate;
  safe_candidate.minimum_selected_cameras = 2U;
  safe_candidate.minimum_candidate_budget = kLargestSafeMinimum;
  safe_candidate.total_candidate_budget =
    kLargestSafeMinimum * 2U;
  EXPECT_NO_THROW({AdaptiveVisualPolicy policy(safe_candidate);});

  AdaptiveVisualPolicyConfig safe_patch;
  safe_patch.minimum_selected_cameras = 2U;
  safe_patch.minimum_patch_budget = kLargestSafeMinimum;
  safe_patch.total_patch_budget = kLargestSafeMinimum * 2U;
  EXPECT_NO_THROW({AdaptiveVisualPolicy policy(safe_patch);});

  AdaptiveVisualPolicyConfig overflowing_candidate;
  overflowing_candidate.minimum_selected_cameras = 2U;
  overflowing_candidate.minimum_candidate_budget =
    kOverflowingMinimum;
  overflowing_candidate.total_candidate_budget = kMaximum;
  EXPECT_THROW(
    {AdaptiveVisualPolicy policy(overflowing_candidate);},
    std::invalid_argument);

  AdaptiveVisualPolicyConfig overflowing_patch;
  overflowing_patch.minimum_selected_cameras = 2U;
  overflowing_patch.minimum_patch_budget = kOverflowingMinimum;
  overflowing_patch.total_patch_budget = kMaximum;
  EXPECT_THROW(
    {AdaptiveVisualPolicy policy(overflowing_patch);},
    std::invalid_argument);
}

TEST(
  AdaptiveVisualSchedulerTest,
  DuplicateOrMisplacedCameraIdsFailClosedWithoutAdvancingHealthState)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  observations[1U].camera_id = 0U;

  const auto invalid_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_EQ(
    invalid_plan.reason,
    livo::AdaptivePolicyReason::kInvalidCameraObservations);
  EXPECT_FALSE(invalid_plan.lidar_information_valid);
  for (const auto & camera : invalid_plan.cameras)
  {
    EXPECT_FALSE(camera.enabled);
  }
  for (const auto & state : invalid_plan.proposed_hysteresis)
  {
    EXPECT_FALSE(state.isolated);
    EXPECT_EQ(state.consecutive_unhealthy, 0U);
    EXPECT_EQ(state.consecutive_healthy, 0U);
  }
  EXPECT_FALSE(
    policy.commitHealthObservation(invalid_plan, true));
  const auto next_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), healthyRig());
  EXPECT_EQ(next_plan.base_revision, 0U);
}

TEST(AdaptiveVisualSchedulerTest, GivesMoreBudgetToCameraComplementingLidar)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.total_candidate_budget = 80U;
  config.minimum_candidate_budget = 16U;
  config.total_patch_budget = 40U;
  config.minimum_patch_budget = 8U;
  config.complement_score_gain = 4.0;
  AdaptiveVisualPolicy policy(config);
  const auto lidar_information = physicalInformationForNormalizedDiagonal(
    config, {0.1, 10.0, 20.0, 30.0, 40.0, 50.0});
  auto observations = healthyRig();
  observations[0].visual_information =
    physicalInformationForNormalizedDiagonal(
      config, {100.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  observations[1].visual_information =
    physicalInformationForNormalizedDiagonal(
      config, {0.1, 100.0, 1.0, 1.0, 1.0, 1.0});

  const auto plan = policy.plan(lidar_information, observations);

  EXPECT_GT(
    plan.cameras[0].complement_score,
    plan.cameras[1].complement_score);
  EXPECT_GT(plan.cameras[0].selection_score, plan.cameras[1].selection_score);
  EXPECT_GT(
    plan.cameras[0].candidate_budget,
    plan.cameras[1].candidate_budget);
  EXPECT_GT(plan.cameras[0].patch_budget, plan.cameras[1].patch_budget);
}

TEST(
  AdaptiveVisualSchedulerTest,
  ExcludedCamerasCannotDistortHealthySelectionOrBudgets)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 2U;
  config.maximum_selected_cameras = 2U;
  config.total_candidate_budget = 81U;
  config.minimum_candidate_budget = 16U;
  config.total_patch_budget = 41U;
  config.minimum_patch_budget = 8U;
  config.complement_score_gain = 4.0;
  AdaptiveVisualPolicy clean_policy(config);
  AdaptiveVisualPolicy poisoned_policy(config);
  const auto lidar_information = physicalInformationForNormalizedDiagonal(
    config, {0.1, 10.0, 20.0, 30.0, 40.0, 50.0});

  auto isolation_observations = healthyRig();
  isolation_observations[2U].saturation_fraction = 0.95;
  const auto clean_isolation_plan = clean_policy.plan(
    lidar_information, isolation_observations);
  const auto poisoned_isolation_plan = poisoned_policy.plan(
    lidar_information, isolation_observations);
  ASSERT_TRUE(
    clean_policy.commitHealthObservation(
      clean_isolation_plan, true));
  ASSERT_TRUE(
    poisoned_policy.commitHealthObservation(
      poisoned_isolation_plan, true));
  ASSERT_TRUE(clean_policy.hysteresis()[2U].isolated);
  ASSERT_TRUE(poisoned_policy.hysteresis()[2U].isolated);

  auto clean_observations = healthyRig();
  clean_observations[0U].visual_information =
    physicalInformationForNormalizedDiagonal(
    config, {100.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  clean_observations[1U].visual_information =
    physicalInformationForNormalizedDiagonal(
    config, {20.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  clean_observations[3U].tracked_points = 0U;
  clean_observations[3U].saturation_fraction = 0.95;
  auto poisoned_observations = clean_observations;
  const auto forged_information =
    physicalInformationForNormalizedDiagonal(
    config, {1.0e9, 1.0, 1.0, 1.0, 1.0, 1.0});
  poisoned_observations[2U].visual_information =
    forged_information;
  poisoned_observations[3U].visual_information =
    forged_information;

  const auto clean_plan = clean_policy.plan(
    lidar_information, clean_observations);
  const auto poisoned_plan = poisoned_policy.plan(
    lidar_information, poisoned_observations);

  ASSERT_EQ(clean_plan.reason, livo::AdaptivePolicyReason::kReady);
  ASSERT_EQ(poisoned_plan.reason, livo::AdaptivePolicyReason::kReady);
  for (const std::size_t camera_id : {0U, 1U})
  {
    EXPECT_TRUE(clean_plan.cameras[camera_id].enabled);
    EXPECT_TRUE(poisoned_plan.cameras[camera_id].enabled);
    EXPECT_DOUBLE_EQ(
      poisoned_plan.cameras[camera_id].selection_score,
      clean_plan.cameras[camera_id].selection_score);
    EXPECT_EQ(
      poisoned_plan.cameras[camera_id].candidate_budget,
      clean_plan.cameras[camera_id].candidate_budget);
    EXPECT_EQ(
      poisoned_plan.cameras[camera_id].patch_budget,
      clean_plan.cameras[camera_id].patch_budget);
  }
  EXPECT_GT(
    poisoned_plan.cameras[0U].selection_score,
    poisoned_plan.cameras[1U].selection_score);
  EXPECT_TRUE(poisoned_plan.cameras[2U].isolated);
  EXPECT_FALSE(poisoned_plan.cameras[2U].enabled);
  EXPECT_TRUE(poisoned_plan.cameras[3U].image_hard_failure);
  EXPECT_FALSE(poisoned_plan.cameras[3U].available);
  EXPECT_FALSE(poisoned_plan.cameras[3U].enabled);
  EXPECT_DOUBLE_EQ(poisoned_plan.cameras[2U].selection_score, 0.0);
  EXPECT_DOUBLE_EQ(poisoned_plan.cameras[3U].selection_score, 0.0);
}

TEST(
  AdaptiveVisualSchedulerTest,
  IncompleteHealthEvidenceReportsDegradedState)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  const AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  observations[0U] = imageOnlyObservation(0U);

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_TRUE(plan.cameras[0U].health_degraded);
  EXPECT_EQ(
    plan.cameras[0U].health_state,
    AdaptiveCameraHealthState::kDegraded);
  EXPECT_FALSE(plan.cameras[0U].isolated);
}

TEST(
  AdaptiveVisualSchedulerTest,
  ProbationRequiresBothDistinctCommitsAndMinimumDuration)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.on_hysteresis_frames = 2U;
  config.minimum_probation_duration_sec = 0.5;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();

  observations[0U].saturation_fraction = 0.95;
  observations[0U].source_timestamp_sec = 10.0;
  const auto isolated_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);
  ASSERT_EQ(
    isolated_plan.cameras[0U].health_state,
    AdaptiveCameraHealthState::kIsolated);
  ASSERT_TRUE(policy.commitHealthObservation(isolated_plan, true));

  observations[0U] = healthyObservation(0U);
  observations[0U].source_timestamp_sec = 10.1;
  const auto probation_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);
  ASSERT_EQ(
    probation_plan.cameras[0U].health_state,
    AdaptiveCameraHealthState::kProbation);
  ASSERT_TRUE(probation_plan.cameras[0U].isolated);
  ASSERT_TRUE(policy.commitHealthObservation(probation_plan, true));

  observations[0U].source_timestamp_sec = 10.59;
  const auto too_early_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);
  EXPECT_EQ(
    too_early_plan.cameras[0U].health_state,
    AdaptiveCameraHealthState::kProbation);
  EXPECT_TRUE(too_early_plan.cameras[0U].isolated);
  ASSERT_TRUE(policy.commitHealthObservation(too_early_plan, true));

  observations[0U].source_timestamp_sec = 10.61;
  const auto recovered_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);
  EXPECT_EQ(
    recovered_plan.cameras[0U].health_state,
    AdaptiveCameraHealthState::kHealthy);
  EXPECT_FALSE(recovered_plan.cameras[0U].isolated);
  EXPECT_EQ(
    recovered_plan.cameras[0U].transition,
    AdaptiveCameraTransition::kRecovered);
}

TEST(
  AdaptiveVisualSchedulerTest,
  OneDroppedCameraLeavesThreeEligibleAndSelectsExactlyOne)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  const AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  for (std::size_t camera_id = 1U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id].robust_information_gain =
      camera_id == 2U ? 2.0 : 1.0;
  }
  observations[0U] = droppedObservation(0U);

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_FALSE(plan.cameras[0U].eligible);
  EXPECT_EQ(
    plan.cameras[0U].reject_reason,
    livo::AdaptiveCameraRejectReason::kFrameMissing);
  EXPECT_FALSE(plan.cameras[0U].enabled);
  EXPECT_EQ(plan.cameras[0U].candidate_budget, 0U);
  EXPECT_EQ(plan.cameras[0U].patch_budget, 0U);
  for (std::size_t camera_id = 1U;
       camera_id < plan.cameras.size(); ++camera_id)
  {
    EXPECT_TRUE(plan.cameras[camera_id].eligible);
    EXPECT_EQ(
      plan.cameras[camera_id].reject_reason,
      livo::AdaptiveCameraRejectReason::kNone);
  }
  EXPECT_EQ(
    std::count_if(
      plan.cameras.begin(), plan.cameras.end(),
      [](const livo::AdaptiveCameraDecision & camera)
      {
        return camera.eligible;
      }),
    3);
  EXPECT_EQ(
    std::count_if(
      plan.cameras.begin(), plan.cameras.end(),
      [](const livo::AdaptiveCameraDecision & camera)
      {
        return camera.enabled;
      }),
    1);
  EXPECT_TRUE(plan.cameras[2U].enabled);
  EXPECT_EQ(candidateBudgetSum(plan), config.total_candidate_budget);
  EXPECT_EQ(patchBudgetSum(plan), config.total_patch_budget);
}

TEST(AdaptiveVisualSchedulerTest, ConservesGlobalBudgets)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.total_candidate_budget = 96U;
  config.minimum_candidate_budget = 16U;
  config.total_patch_budget = 48U;
  config.minimum_patch_budget = 8U;
  const AdaptiveVisualPolicy policy(config);
  const auto lidar_information = physicalInformationForNormalizedDiagonal(
    config, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});

  const auto plan = policy.plan(lidar_information, healthyRig());

  EXPECT_EQ(candidateBudgetSum(plan), config.total_candidate_budget);
  EXPECT_EQ(patchBudgetSum(plan), config.total_patch_budget);
  for (const auto & camera : plan.cameras)
  {
    ASSERT_TRUE(camera.enabled);
    EXPECT_GE(camera.candidate_budget, config.minimum_candidate_budget);
    EXPECT_GE(camera.patch_budget, config.minimum_patch_budget);
  }
}

TEST(AdaptiveVisualSchedulerTest, ConservesGlobalScoutCandidateBudget)
{
  constexpr std::size_t kTotalBudget = 97U;
  std::size_t scout_budget_sum = 0U;
  for (std::size_t camera_id = 0U;
       camera_id < livo::kAdaptiveVisualCameraCount; ++camera_id)
  {
    scout_budget_sum +=
      livo::adaptiveScoutCandidateBudget(kTotalBudget, camera_id);
  }

  EXPECT_EQ(scout_budget_sum, kTotalBudget);
  EXPECT_EQ(livo::adaptiveScoutCandidateBudget(kTotalBudget, 0U), 25U);
  EXPECT_EQ(livo::adaptiveScoutCandidateBudget(kTotalBudget, 1U), 24U);
  EXPECT_EQ(
    livo::adaptiveScoutCandidateBudget(
      kTotalBudget, livo::kAdaptiveVisualCameraCount),
    0U);
}

TEST(AdaptiveVisualSchedulerTest, InvalidLidarInformationFailsClosed)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  const AdaptiveVisualPolicy policy(config);

  const auto not_ready = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), healthyRig(), false);
  EXPECT_FALSE(not_ready.lidar_information_valid);
  EXPECT_EQ(
    not_ready.reason, livo::AdaptivePolicyReason::kLidarNotReady);

  const auto insufficient = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Zero(), healthyRig());
  EXPECT_FALSE(insufficient.lidar_information_valid);
  EXPECT_EQ(
    insufficient.reason,
    livo::AdaptivePolicyReason::kLidarInsufficientInformation);

  AdaptiveVisualPolicy::Matrix6 non_finite =
    AdaptiveVisualPolicy::Matrix6::Identity();
  non_finite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  const auto non_finite_plan = policy.plan(non_finite, healthyRig());
  EXPECT_FALSE(non_finite_plan.lidar_information_valid);
  EXPECT_EQ(
    non_finite_plan.reason,
    livo::AdaptivePolicyReason::kLidarNonFinite);

  AdaptiveVisualPolicy::Matrix6 non_symmetric =
    AdaptiveVisualPolicy::Matrix6::Identity();
  non_symmetric(0, 1) = 0.1;
  const auto non_symmetric_plan =
    policy.plan(non_symmetric, healthyRig());
  EXPECT_FALSE(non_symmetric_plan.lidar_information_valid);
  EXPECT_EQ(
    non_symmetric_plan.reason,
    livo::AdaptivePolicyReason::kLidarNonSymmetric);

  AdaptiveVisualPolicy::Matrix6 indefinite =
    AdaptiveVisualPolicy::Matrix6::Identity();
  indefinite(0, 0) = -1.0;
  const auto indefinite_plan = policy.plan(indefinite, healthyRig());
  EXPECT_FALSE(indefinite_plan.lidar_information_valid);
  EXPECT_EQ(
    indefinite_plan.reason,
    livo::AdaptivePolicyReason::kLidarInsufficientInformation);

  for (const auto & invalid_plan :
       {not_ready, insufficient, non_finite_plan,
        non_symmetric_plan, indefinite_plan})
  {
    EXPECT_EQ(candidateBudgetSum(invalid_plan), 0U);
    EXPECT_EQ(patchBudgetSum(invalid_plan), 0U);
    EXPECT_TRUE(std::none_of(
      invalid_plan.cameras.begin(), invalid_plan.cameras.end(),
      [](const livo::AdaptiveCameraDecision & camera)
      {
        return camera.enabled;
      }));
  }
}

TEST(AdaptiveVisualSchedulerTest, RequiresConfiguredMinimumCameraCount)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 2U;
  const AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  for (std::size_t camera_id = 1U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] = AdaptiveCameraObservation{};
    observations[camera_id].camera_id =
      static_cast<std::uint8_t>(camera_id);
  }

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_TRUE(plan.lidar_information_valid);
  EXPECT_EQ(
    plan.reason, livo::AdaptivePolicyReason::kInsufficientCameras);
  EXPECT_EQ(candidateBudgetSum(plan), 0U);
  EXPECT_EQ(patchBudgetSum(plan), 0U);
  EXPECT_TRUE(std::none_of(
    plan.cameras.begin(), plan.cameras.end(),
    [](const livo::AdaptiveCameraDecision & camera)
    {
      return camera.enabled;
    }));
}

TEST(AdaptiveVisualSchedulerTest, LowHealthInflatesMeasurementCovariance)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.hard_reject_health = 0.0;
  const AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  observations[1] = unhealthyButAvailableObservation(1U);

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_LT(plan.cameras[1].health, plan.cameras[0].health);
  EXPECT_GT(
    plan.cameras[1].covariance_scale,
    plan.cameras[0].covariance_scale);
  EXPECT_GE(plan.cameras[0].covariance_scale, 1.0);
  EXPECT_LE(
    plan.cameras[1].covariance_scale,
    config.maximum_covariance_scale);
}

TEST(AdaptiveVisualSchedulerTest, AllBadCamerasFallBackToLio)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  for (std::size_t camera_id = 0U; camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] =
      unhealthyButAvailableObservation(
        static_cast<std::uint8_t>(camera_id));
  }

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_TRUE(std::none_of(
    plan.cameras.begin(), plan.cameras.end(),
    [](const livo::AdaptiveCameraDecision & camera)
    {
      return camera.enabled;
    }));
  EXPECT_EQ(candidateBudgetSum(plan), 0U);
  EXPECT_EQ(patchBudgetSum(plan), 0U);
}

TEST(
  AdaptiveVisualSchedulerTest,
  IncompleteFourCameraColdStartDoesNotAccumulateIsolation)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.hard_reject_health = 0.0;
  config.off_hysteresis_frames = 3U;
  AdaptiveVisualPolicy policy(config);
  AdaptiveVisualPolicy::CameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] =
      imageOnlyObservation(static_cast<std::uint8_t>(camera_id));
  }

  for (std::size_t frame = 0U;
       frame < 2U * config.off_hysteresis_frames; ++frame)
  {
    const auto plan = policy.plan(
      AdaptiveVisualPolicy::Matrix6::Identity(), observations);
    for (const auto & camera : plan.cameras)
    {
      EXPECT_FALSE(camera.isolated);
      EXPECT_EQ(camera.transition, AdaptiveCameraTransition::kNone);
      EXPECT_FALSE(camera.health_evidence_complete);
      EXPECT_FALSE(camera.image_hard_failure);
      EXPECT_TRUE(camera.image_recovery_healthy);
      const auto map_action =
        livo::planAdaptiveLioAnchoredMapMaintenance(
        livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
          kBootstrap,
        livo::AdaptiveVisualMapLifecycle::kEmpty,
        true, camera.isolated, true, true);
      EXPECT_TRUE(map_action.seed_new_points);
      EXPECT_FALSE(map_action.update_existing_points);
    }
    ASSERT_TRUE(policy.commitHealthObservation(plan, true));
  }

  for (const auto & state : policy.hysteresis())
  {
    EXPECT_FALSE(state.isolated);
    EXPECT_EQ(state.consecutive_unhealthy, 0U);
  }
}

TEST(
  AdaptiveVisualSchedulerTest,
  HardFailedCameraRecoversFromHealthyImageOnlyScoutsWithoutMapMutation)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.hard_reject_health = 0.0;
  config.off_hysteresis_frames = 2U;
  config.on_hysteresis_frames = 2U;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  observations[0] = imageOnlyObservation(0U, 0.95, 0.0, 16.0);

  const auto hard_failure_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);
  EXPECT_FALSE(hard_failure_plan.cameras[0].health_evidence_complete);
  EXPECT_TRUE(hard_failure_plan.cameras[0].image_hard_failure);
  EXPECT_FALSE(hard_failure_plan.cameras[0].image_recovery_healthy);
  EXPECT_EQ(
    hard_failure_plan.cameras[0].transition,
    AdaptiveCameraTransition::kIsolated);
  ASSERT_TRUE(
    policy.commitHealthObservation(hard_failure_plan, true));
  ASSERT_TRUE(policy.hysteresis()[0].isolated);

  observations[0] = imageOnlyObservation(0U);
  for (std::size_t frame = 0U;
       frame < config.on_hysteresis_frames; ++frame)
  {
    const auto plan = policy.plan(
      AdaptiveVisualPolicy::Matrix6::Identity(), observations);
    EXPECT_FALSE(plan.cameras[0].health_evidence_complete);
    EXPECT_FALSE(plan.cameras[0].image_hard_failure);
    EXPECT_TRUE(plan.cameras[0].image_recovery_healthy);
    const auto map_action =
      livo::planAdaptiveVisualSolverMapCommit(
      true, true, false);
    EXPECT_FALSE(map_action.seed_new_points);
    EXPECT_FALSE(map_action.update_existing_points);
    ASSERT_TRUE(policy.commitHealthObservation(plan, true));
  }
  EXPECT_FALSE(policy.hysteresis()[0].isolated);
}

TEST(AdaptiveVisualSchedulerTest, DisabledModePreservesLegacyMapGrowth)
{
  const auto decision =
    livo::planAdaptiveVisualSolverMapCommit(
    false, false, false);

  EXPECT_TRUE(decision.seed_new_points);
  EXPECT_TRUE(decision.update_existing_points);
}

TEST(
  AdaptiveVisualSchedulerTest,
  LioAnchoredMaintenanceRequiresAuthorizedLifecycleEvidence)
{
  const auto cold_start =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kBootstrap,
    livo::AdaptiveVisualMapLifecycle::kEmpty,
    true, false, true, true);
  EXPECT_TRUE(cold_start.seed_new_points);
  EXPECT_FALSE(cold_start.update_existing_points);

  const auto unsafe_image =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kBootstrap,
    livo::AdaptiveVisualMapLifecycle::kEmpty,
    true, false, false, true);
  EXPECT_FALSE(unsafe_image.seed_new_points);
  EXPECT_FALSE(unsafe_image.update_existing_points);

  const auto bootstrap_of_seeded_map =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kBootstrap,
    livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated,
    true, false, true, true);
  EXPECT_TRUE(bootstrap_of_seeded_map.seed_new_points);
  EXPECT_FALSE(bootstrap_of_seeded_map.update_existing_points);

  const auto bootstrap_of_initialized_map =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kBootstrap,
    livo::AdaptiveVisualMapLifecycle::kInitialized,
    true, false, true, true);
  EXPECT_FALSE(bootstrap_of_initialized_map.seed_new_points);
  EXPECT_FALSE(bootstrap_of_initialized_map.update_existing_points);

  const auto invalid_lidar =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kBootstrap,
    livo::AdaptiveVisualMapLifecycle::kEmpty,
    true, false, true, false);
  EXPECT_FALSE(invalid_lidar.seed_new_points);
  EXPECT_FALSE(invalid_lidar.update_existing_points);

  const auto isolated_cold_start =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kBootstrap,
    livo::AdaptiveVisualMapLifecycle::kEmpty,
    true, true, true, true);
  EXPECT_FALSE(isolated_cold_start.seed_new_points);
  EXPECT_FALSE(isolated_cold_start.update_existing_points);

  const auto selected_zero_map_reseed =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kReseed,
    livo::AdaptiveVisualMapLifecycle::kInitialized,
    true, false, true, true);
  EXPECT_TRUE(selected_zero_map_reseed.seed_new_points);
  EXPECT_FALSE(selected_zero_map_reseed.update_existing_points);

  const auto unvalidated_zero_map_reseed =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kReseed,
    livo::AdaptiveVisualMapLifecycle::kSeededUnvalidated,
    true, false, true, true);
  EXPECT_TRUE(unvalidated_zero_map_reseed.seed_new_points);
  EXPECT_FALSE(unvalidated_zero_map_reseed.update_existing_points);

  const auto isolated_zero_map_reseed =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kReseed,
    livo::AdaptiveVisualMapLifecycle::kInitialized,
    true, true, true, true);
  EXPECT_FALSE(isolated_zero_map_reseed.seed_new_points);
  EXPECT_FALSE(isolated_zero_map_reseed.update_existing_points);

  const auto invalid_lidar_zero_map_reseed =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kReseed,
    livo::AdaptiveVisualMapLifecycle::kInitialized,
    true, false, true, false);
  EXPECT_FALSE(invalid_lidar_zero_map_reseed.seed_new_points);
  EXPECT_FALSE(
    invalid_lidar_zero_map_reseed.update_existing_points);

  const auto unsafe_image_zero_map_reseed =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kReseed,
    livo::AdaptiveVisualMapLifecycle::kInitialized,
    true, false, false, true);
  EXPECT_FALSE(unsafe_image_zero_map_reseed.seed_new_points);
  EXPECT_FALSE(
    unsafe_image_zero_map_reseed.update_existing_points);
}

TEST(
  AdaptiveVisualSchedulerTest,
  AdaptiveMapMaintenanceRejectsEveryUnselectedCameraWrite)
{
  const auto accepted_but_unselected =
    livo::planAdaptiveVisualSolverMapCommit(
      true, false, true);
  EXPECT_FALSE(accepted_but_unselected.seed_new_points);
  EXPECT_FALSE(accepted_but_unselected.update_existing_points);

  const auto cold_start_but_unselected =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kBootstrap,
    livo::AdaptiveVisualMapLifecycle::kEmpty,
    false, false, true, true);
  EXPECT_FALSE(cold_start_but_unselected.seed_new_points);
  EXPECT_FALSE(cold_start_but_unselected.update_existing_points);

  const auto reseed_but_unselected =
    livo::planAdaptiveLioAnchoredMapMaintenance(
    livo::AdaptiveVisualLioAnchoredMaintenanceRequest::
      kReseed,
    livo::AdaptiveVisualMapLifecycle::kInitialized,
    false, false, true, true);
  EXPECT_FALSE(reseed_but_unselected.seed_new_points);
  EXPECT_FALSE(reseed_but_unselected.update_existing_points);

  const auto rejected_solver =
    livo::planAdaptiveVisualSolverMapCommit(
      true, true, false);
  EXPECT_FALSE(rejected_solver.seed_new_points);
  EXPECT_FALSE(rejected_solver.update_existing_points);
}

TEST(
  AdaptiveVisualSchedulerTest,
  LegacyMutablePlanRemainsExplicitlyUntrustedAndNonExclusive)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  AdaptiveVisualPolicy policy(config);
  const auto legacy_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), healthyRig());

  EXPECT_TRUE(
    AdaptiveVisualPolicy::legacyMutablePlanCompatibilityEnabled());
  EXPECT_FALSE(
    AdaptiveVisualPolicy::ownerSealedRuntimeExclusive());
  EXPECT_FALSE(
    AdaptiveVisualPolicy::expectedRuntimeAuthorityPinned());
  EXPECT_TRUE(policy.commitHealthObservation(legacy_plan, true));
}

TEST(
  AdaptiveVisualSchedulerTest,
  SealedRuntimeSelectsHighestRobustGainWithPerCameraBudgets)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  config.total_candidate_budget = 240U;
  config.total_patch_budget = 120U;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id].robust_information_gain =
      static_cast<double>(camera_id + 1U);
  }

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  ASSERT_EQ(plan.reason, livo::AdaptivePolicyReason::kReady);
  EXPECT_EQ(
    plan.selection_reason,
    livo::AdaptiveSelectionReason::kSelectedBestInformationGain);
  EXPECT_TRUE(plan.cameras[3U].enabled);
  EXPECT_EQ(
    std::count_if(
      plan.cameras.begin(), plan.cameras.end(),
      [](const livo::AdaptiveCameraDecision & camera)
      {
        return camera.enabled;
      }),
    1);
  EXPECT_EQ(candidateBudgetSum(plan), config.total_candidate_budget);
  EXPECT_EQ(patchBudgetSum(plan), config.total_patch_budget);
  EXPECT_EQ(plan.cameras[3U].candidate_budget,
    config.total_candidate_budget);
  EXPECT_EQ(plan.cameras[3U].patch_budget,
    config.total_patch_budget);
}

TEST(
  AdaptiveVisualSchedulerTest,
  SealedRuntimeUsesCameraIdOnlyForExactRobustGainTies)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  AdaptiveVisualPolicy policy(config);
  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), healthyRig());

  ASSERT_EQ(plan.reason, livo::AdaptivePolicyReason::kReady);
  EXPECT_EQ(
    plan.selection_reason,
    livo::AdaptiveSelectionReason::kSelectedBestInformationGain);
  EXPECT_TRUE(plan.cameras[0U].enabled);
  EXPECT_EQ(
    std::count_if(
      plan.cameras.begin(), plan.cameras.end(),
      [](const livo::AdaptiveCameraDecision & camera)
      {
        return camera.enabled;
      }),
    1);
}

TEST(
  AdaptiveVisualSchedulerTest,
  SealedRuntimeFallsBackToBaseCandidatesWhenRobustEvidenceIsMissing)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  for (auto & observation : observations)
  {
    observation.available_metrics &=
      ~livo::kAdaptiveRobustAdmissionMetricMask;
  }

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_EQ(
    plan.reason, livo::AdaptivePolicyReason::kReady);
  EXPECT_EQ(
    plan.selection_reason,
    livo::AdaptiveSelectionReason::kSelectedBestHealthy);
  EXPECT_EQ(
    std::count_if(
      plan.cameras.begin(), plan.cameras.end(),
      [](const livo::AdaptiveCameraDecision & camera)
      {
        return camera.enabled;
      }),
    1);
  for (const auto & camera : plan.cameras)
  {
    EXPECT_TRUE(camera.eligible);
    EXPECT_FALSE(camera.robust_gate_passed);
    EXPECT_EQ(
      camera.reject_reason,
      livo::AdaptiveCameraRejectReason::kNone);
  }
}

TEST(
  AdaptiveVisualSchedulerTest,
  SealedRuntimeFallsBackToBaseCandidatesForEachFailedRobustGate)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  AdaptiveVisualPolicy policy(config);

  for (std::size_t gate = 0U; gate < 5U; ++gate)
  {
    SCOPED_TRACE(gate);
    auto observations = healthyRig();
    for (auto & observation : observations)
    {
      switch (gate)
      {
        case 0U:
          observation.inlier_count = config.minimum_inliers - 1U;
          break;
        case 1U:
          observation.inlier_ratio =
            config.minimum_inlier_ratio - 0.01;
          break;
        case 2U:
          observation.spatial_coverage =
            config.healthy_spatial_coverage - 0.01;
          break;
        case 3U:
          observation.nis = 1.0e6;
          break;
        case 4U:
          observation.normalized_pose_correction =
            config.maximum_normalized_pose_correction + 1.0;
          break;
        default:
          FAIL();
      }
    }

    const auto plan = policy.plan(
      AdaptiveVisualPolicy::Matrix6::Identity(), observations);
    EXPECT_EQ(
      plan.reason, livo::AdaptivePolicyReason::kReady);
    EXPECT_EQ(
      plan.selection_reason,
      livo::AdaptiveSelectionReason::kSelectedBestHealthy);
    EXPECT_EQ(
      std::count_if(
        plan.cameras.begin(), plan.cameras.end(),
        [](const livo::AdaptiveCameraDecision & camera)
        {
          return camera.enabled;
        }),
      1);
    for (const auto & camera : plan.cameras)
    {
      EXPECT_TRUE(camera.eligible);
      EXPECT_FALSE(camera.robust_gate_passed);
      EXPECT_EQ(
        camera.reject_reason,
        livo::AdaptiveCameraRejectReason::kNone);
    }
  }
}

TEST(
  AdaptiveVisualSchedulerTest,
  NonSealedCompleteRobustTelemetryRejectsEachSingleFailedGate)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 2U;
  AdaptiveVisualPolicy policy(config);

  for (std::size_t gate = 0U; gate < 6U; ++gate)
  {
    SCOPED_TRACE(gate);
    auto observations = healthyRig();
    auto & failed = observations[0U];
    switch (gate)
    {
      case 0U:
        failed.inlier_count = config.minimum_inliers - 1U;
        break;
      case 1U:
        failed.inlier_ratio =
          config.minimum_inlier_ratio - 0.01;
        break;
      case 2U:
        failed.spatial_coverage =
          config.healthy_spatial_coverage - 0.01;
        break;
      case 3U:
        failed.nis = 1.0e6;
        break;
      case 4U:
        failed.normalized_pose_correction =
          config.maximum_normalized_pose_correction + 1.0;
        break;
      case 5U:
        failed.robust_information_gain = -1.0;
        break;
      default:
        FAIL();
    }

    const auto plan = policy.plan(
      AdaptiveVisualPolicy::Matrix6::Identity(), observations);

    EXPECT_EQ(
      failed.available_metrics &
      livo::kAdaptiveRobustAdmissionMetricMask,
      livo::kAdaptiveRobustAdmissionMetricMask);
    EXPECT_FALSE(plan.cameras[0U].available);
    EXPECT_FALSE(plan.cameras[0U].eligible);
    EXPECT_FALSE(plan.cameras[0U].enabled);
    EXPECT_EQ(
      plan.cameras[0U].reject_reason,
      livo::AdaptiveCameraRejectReason::kTrackingUnavailable);
  }
}

TEST(
  AdaptiveVisualSchedulerTest,
  NonSealedMissingRobustTelemetryKeepsBaseMeasurementAvailable)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 2U;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  observations[0U].available_metrics &=
    ~livo::kAdaptiveRobustAdmissionMetricMask;

  const auto plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);

  EXPECT_TRUE(plan.cameras[0U].available);
  EXPECT_TRUE(plan.cameras[0U].eligible);
  EXPECT_EQ(
    plan.cameras[0U].reject_reason,
    livo::AdaptiveCameraRejectReason::kNone);
}

TEST(
  AdaptiveVisualSchedulerTest,
  SealedRawPlanCommitCannotAdvancePrimaryOrChallengerHistory)
{
  AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  config.minimum_selected_cameras = 1U;
  config.maximum_selected_cameras = 1U;
  config.selection_switch_margin = 1.0;
  config.selection_challenger_frames = 99U;
  AdaptiveVisualPolicy policy(config);
  auto observations = healthyRig();
  observations[0U].robust_information_gain = 5.0;
  observations[3U].saturation_fraction = 0.95;
  const auto before = policy.hysteresis();

  const auto first_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);
  ASSERT_TRUE(first_plan.cameras[0U].enabled);
  ASSERT_TRUE(
    first_plan.proposed_hysteresis[3U].isolated);
  ASSERT_TRUE(first_plan.proposed_primary_camera.has_value());
  EXPECT_EQ(*first_plan.proposed_primary_camera, 0U);
  const std::uint64_t revision_before = policy.revision();
  ASSERT_TRUE(policy.commitHealthObservation(first_plan, true));
  EXPECT_EQ(policy.revision(), revision_before + 1U);

  auto after_first = policy.hysteresis();
  for (std::size_t camera_id = 0U;
       camera_id < before.size(); ++camera_id)
  {
    EXPECT_EQ(after_first[camera_id].state, before[camera_id].state);
    EXPECT_EQ(after_first[camera_id].isolated,
      before[camera_id].isolated);
    EXPECT_EQ(after_first[camera_id].consecutive_unhealthy,
      before[camera_id].consecutive_unhealthy);
    EXPECT_EQ(after_first[camera_id].consecutive_healthy,
      before[camera_id].consecutive_healthy);
    EXPECT_TRUE(
      after_first[camera_id].probation_started_at_sec ==
        before[camera_id].probation_started_at_sec ||
      (std::isnan(after_first[camera_id].probation_started_at_sec) &&
       std::isnan(before[camera_id].probation_started_at_sec)));
  }

  observations[0U].robust_information_gain = 1.0;
  observations[2U].robust_information_gain = 6.0;
  const auto second_plan = policy.plan(
    AdaptiveVisualPolicy::Matrix6::Identity(), observations);
  EXPECT_FALSE(second_plan.cameras[0U].enabled);
  EXPECT_TRUE(second_plan.cameras[2U].enabled);
  ASSERT_TRUE(second_plan.proposed_primary_camera.has_value());
  EXPECT_EQ(*second_plan.proposed_primary_camera, 2U);
  EXPECT_EQ(
    second_plan.selection_reason,
    livo::AdaptiveSelectionReason::kSelectedBestInformationGain);
  EXPECT_FALSE(
    second_plan.proposed_selection_challenger.camera_id.has_value());
  EXPECT_EQ(
    second_plan.proposed_selection_challenger.
    consecutive_confirmations,
    0U);
}

}  // namespace
