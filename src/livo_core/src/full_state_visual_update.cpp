#include "full_state_visual_update.h"
#include "symmetric_information_validation.h"

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>

namespace livo
{

FullStateVisualUpdate::FullStateVisualUpdate(
  FullStateVisualUpdateConfig config)
: config_(std::move(config))
{
  if (config_.minimum_unique_landmarks == 0U ||
      config_.minimum_cameras == 0U ||
      config_.minimum_cameras > kVioCameraCount ||
      !std::isfinite(config_.maximum_residual_rmse) ||
      config_.maximum_residual_rmse < 0.0 ||
      !std::isfinite(config_.maximum_rotation_increment_rad) ||
      config_.maximum_rotation_increment_rad < 0.0 ||
      !std::isfinite(config_.maximum_translation_increment_m) ||
      config_.maximum_translation_increment_m < 0.0 ||
      !std::isfinite(
        config_.maximum_auxiliary_increment_norm) ||
      config_.maximum_auxiliary_increment_norm < 0.0 ||
      !std::isfinite(
        config_.maximum_prior_condition_number) ||
      config_.maximum_prior_condition_number < 1.0 ||
      !std::isfinite(
        config_.maximum_system_condition_number) ||
      config_.maximum_system_condition_number < 1.0)
  {
    throw std::invalid_argument(
      "invalid full-state visual update configuration");
  }
}

FullStateVisualUpdateResult FullStateVisualUpdate::solve(
  const FullStateVisualUpdateRequest & request) const
{
  FullStateVisualUpdateResult result;
  result.batch_id = request.batch_id;
  if (request.batch_id == 0U || request.contributions.empty())
  {
    return result;
  }
  if (request.state_layout_id != kFullStateVisualLayoutId ||
      request.state_layout_revision !=
        kFullStateVisualLayoutRevision)
  {
    result.status = FullStateVisualUpdateStatus::kInvalidLayout;
    return result;
  }

  FullStateVisualMatrix information_hessian =
    FullStateVisualMatrix::Zero();
  FullStateVisualVector information_gradient =
    FullStateVisualVector::Zero();
  double squared_residual_sum = 0.0;
  std::size_t residual_count = 0U;
  std::set<std::uint8_t> cameras;
  std::set<LandmarkId> landmarks;
  for (const FullStateVisualContribution & contribution :
       request.contributions)
  {
    if (contribution.landmark_id == 0U ||
        contribution.camera_id >= kVioCameraCount ||
        contribution.residual_count == 0U ||
        contribution.information_units != 1U ||
        !std::isfinite(contribution.squared_residual_sum) ||
        contribution.squared_residual_sum < 0.0 ||
        !contribution.information_gradient.allFinite())
    {
      result.status =
        FullStateVisualUpdateStatus::kInvalidContribution;
      return result;
    }
    const auto hessian_validation =
      validateSymmetricInformation(
      contribution.information_hessian,
      SymmetricInformationRequirement::kPositiveSemidefinite);
    if (!hessian_validation.accepted())
    {
      result.status =
        FullStateVisualUpdateStatus::kInvalidContribution;
      return result;
    }
    if (!landmarks.insert(contribution.landmark_id).second)
    {
      result.status =
        FullStateVisualUpdateStatus::kDuplicateLandmark;
      return result;
    }
    if (!contribution.information_hessian.row(
          kFullStateVisualExposureOffset).isZero(0.0) ||
        !contribution.information_hessian.col(
          kFullStateVisualExposureOffset).isZero(0.0) ||
        contribution.information_gradient(
          kFullStateVisualExposureOffset) != 0.0)
    {
      result.status =
        FullStateVisualUpdateStatus::kInvalidContribution;
      return result;
    }
    const FullStateVisualMatrix next_hessian =
      information_hessian + hessian_validation.symmetric;
    const FullStateVisualVector next_gradient =
      information_gradient + contribution.information_gradient;
    const double next_squared_residual_sum =
      squared_residual_sum + contribution.squared_residual_sum;
    if (!next_hessian.allFinite() ||
        !next_gradient.allFinite() ||
        !std::isfinite(next_squared_residual_sum) ||
        contribution.residual_count >
          std::numeric_limits<std::size_t>::max() -
          residual_count)
    {
      result.status =
        FullStateVisualUpdateStatus::kAccumulationOverflow;
      return result;
    }
    information_hessian = next_hessian;
    information_gradient = next_gradient;
    squared_residual_sum = next_squared_residual_sum;
    residual_count += contribution.residual_count;
    cameras.insert(contribution.camera_id);
  }
  if (landmarks.size() < config_.minimum_unique_landmarks ||
      cameras.size() < config_.minimum_cameras)
  {
    result.status =
      FullStateVisualUpdateStatus::kInsufficientInformation;
    return result;
  }
  const double residual_rmse = std::sqrt(
    squared_residual_sum /
    static_cast<double>(residual_count));
  if (!std::isfinite(residual_rmse) ||
      residual_rmse > config_.maximum_residual_rmse)
  {
    result.status =
      FullStateVisualUpdateStatus::kInnovationRejected;
    return result;
  }

  const auto prior_validation =
    validateSymmetricInformation(
    request.prior_covariance,
    SymmetricInformationRequirement::kPositiveDefinite);
  if (!prior_validation.accepted())
  {
    result.status = FullStateVisualUpdateStatus::kInvalidPrior;
    return result;
  }
  const FullStateVisualMatrix & symmetric_prior =
    prior_validation.symmetric;
  const double prior_condition_number =
    prior_validation.condition_number;
  if (prior_condition_number >
      config_.maximum_prior_condition_number)
  {
    result.status =
      FullStateVisualUpdateStatus::kIllConditionedPrior;
    return result;
  }
  const Eigen::LLT<FullStateVisualMatrix> prior_solver(
    symmetric_prior);
  if (prior_solver.info() != Eigen::Success)
  {
    result.status = FullStateVisualUpdateStatus::kInvalidPrior;
    return result;
  }
  const FullStateVisualMatrix prior_cholesky =
    prior_solver.matrixL();
  if (!prior_cholesky.allFinite())
  {
    result.status = FullStateVisualUpdateStatus::kInvalidPrior;
    return result;
  }
  const FullStateVisualMatrix raw_whitened_system =
    FullStateVisualMatrix::Identity() +
    prior_cholesky.transpose() *
    information_hessian *
    prior_cholesky;
  const auto system_validation =
    validateSymmetricInformation(
    raw_whitened_system,
    SymmetricInformationRequirement::kPositiveDefinite);
  const double system_condition_number =
    system_validation.condition_number;
  if (!system_validation.accepted() ||
      system_condition_number >
        config_.maximum_system_condition_number)
  {
    result.status =
      FullStateVisualUpdateStatus::kIllConditionedSystem;
    return result;
  }
  const FullStateVisualMatrix & whitened_system =
    system_validation.symmetric;
  const Eigen::LLT<FullStateVisualMatrix> system_solver(
    whitened_system);
  if (system_solver.info() != Eigen::Success)
  {
    result.status =
      FullStateVisualUpdateStatus::kIllConditionedSystem;
    return result;
  }

  const FullStateVisualVector whitened_gradient =
    prior_cholesky.transpose() * information_gradient;
  const FullStateVisualVector whitened_increment =
    system_solver.solve(-whitened_gradient);
  const FullStateVisualVector increment =
    prior_cholesky * whitened_increment;
  const FullStateVisualMatrix whitened_posterior =
    system_solver.solve(FullStateVisualMatrix::Identity());
  const FullStateVisualMatrix raw_posterior =
    prior_cholesky *
    whitened_posterior *
    prior_cholesky.transpose();
  if (system_solver.info() != Eigen::Success ||
      !whitened_gradient.allFinite() ||
      !whitened_increment.allFinite() ||
      !increment.allFinite())
  {
    result.status =
      FullStateVisualUpdateStatus::kIncrementRejected;
    return result;
  }
  const auto posterior_validation =
    validateSymmetricInformation(
    raw_posterior,
    SymmetricInformationRequirement::kPositiveSemidefinite);
  if (!whitened_posterior.allFinite() ||
      !posterior_validation.accepted())
  {
    result.status =
      FullStateVisualUpdateStatus::kInvalidPosterior;
    return result;
  }
  const FullStateVisualMatrix & posterior =
    posterior_validation.symmetric;
  const auto contraction_validation =
    validateSymmetricInformation(
    symmetric_prior - posterior,
    SymmetricInformationRequirement::kPositiveSemidefinite);
  if (!contraction_validation.accepted())
  {
    result.status =
      FullStateVisualUpdateStatus::kInvalidPosterior;
    return result;
  }
  constexpr double kExposureTolerance = 1.0e-12;
  if (std::abs(increment(kFullStateVisualExposureOffset)) >
        kExposureTolerance ||
      !posterior.row(kFullStateVisualExposureOffset).isApprox(
        symmetric_prior.row(
          kFullStateVisualExposureOffset),
        kExposureTolerance) ||
      !posterior.col(kFullStateVisualExposureOffset).isApprox(
        symmetric_prior.col(
          kFullStateVisualExposureOffset),
        kExposureTolerance))
  {
    result.status =
      FullStateVisualUpdateStatus::kExposureCouplingRejected;
    return result;
  }
  const double rotation_increment =
    increment.segment<3>(kFullStateVisualThetaOffset).norm();
  const double translation_increment =
    increment.segment<3>(kFullStateVisualPositionOffset).norm();
  const double auxiliary_increment =
    increment.segment<12>(kFullStateVisualVelocityOffset).norm();
  if (!std::isfinite(rotation_increment) ||
      !std::isfinite(translation_increment) ||
      !std::isfinite(auxiliary_increment) ||
      rotation_increment >
        config_.maximum_rotation_increment_rad ||
      translation_increment >
        config_.maximum_translation_increment_m ||
      auxiliary_increment >
        config_.maximum_auxiliary_increment_norm)
  {
    result.status =
      FullStateVisualUpdateStatus::kIncrementRejected;
    return result;
  }

  result.increment = increment;
  result.posterior_covariance = posterior;
  result.unique_landmarks = landmarks.size();
  result.contributing_cameras = cameras.size();
  result.residual_rmse = residual_rmse;
  result.residual_count = residual_count;
  result.information_units = landmarks.size();
  result.prior_condition_number =
    prior_condition_number;
  result.system_condition_number = system_condition_number;
  result.status = FullStateVisualUpdateStatus::kAccepted;
  return result;
}

const char * fullStateVisualUpdateStatusName(
  const FullStateVisualUpdateStatus status)
{
  switch (status)
  {
    case FullStateVisualUpdateStatus::kAccepted:
      return "accepted";
    case FullStateVisualUpdateStatus::kInvalidRequest:
      return "invalid_request";
    case FullStateVisualUpdateStatus::kInvalidLayout:
      return "invalid_layout";
    case FullStateVisualUpdateStatus::kInvalidPrior:
      return "invalid_prior";
    case FullStateVisualUpdateStatus::kDuplicateLandmark:
      return "duplicate_landmark";
    case FullStateVisualUpdateStatus::kInvalidContribution:
      return "invalid_contribution";
    case FullStateVisualUpdateStatus::kAccumulationOverflow:
      return "accumulation_overflow";
    case FullStateVisualUpdateStatus::kInsufficientInformation:
      return "insufficient_information";
    case FullStateVisualUpdateStatus::kInnovationRejected:
      return "innovation_rejected";
    case FullStateVisualUpdateStatus::kIllConditionedPrior:
      return "ill_conditioned_prior";
    case FullStateVisualUpdateStatus::kIllConditionedSystem:
      return "ill_conditioned_system";
    case FullStateVisualUpdateStatus::kIncrementRejected:
      return "increment_rejected";
    case FullStateVisualUpdateStatus::kExposureCouplingRejected:
      return "exposure_coupling_rejected";
    case FullStateVisualUpdateStatus::kInvalidPosterior:
      return "invalid_posterior";
  }
  return "unknown";
}

}  // namespace livo
