#ifndef LIVO_JOINT_VISUAL_UPDATE_H_
#define LIVO_JOINT_VISUAL_UPDATE_H_

#include <Eigen/Core>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "camera_observation.h"
#include "symmetric_information_validation.h"

struct JointVisualUpdateConfig
{
  std::size_t min_tracked_points_per_camera{4U};
  std::size_t min_joint_tracked_points{12U};
  std::size_t min_joint_cameras{2U};
  double max_residual_rmse{20.0};
  double max_rotation_increment_rad{0.1 * 3.14159265358979323846 / 180.0};
  double max_translation_increment_m{0.01};
  double max_measurement_covariance_scale{1.0e6};
};

class JointVisualNormalEquation
{
public:
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  using Vector6 = Eigen::Matrix<double, 6, 1>;

  explicit JointVisualNormalEquation(
    const JointVisualUpdateConfig & config = JointVisualUpdateConfig{})
  : config_(config)
  {
    if (config_.min_tracked_points_per_camera == 0U ||
        config_.min_joint_tracked_points == 0U ||
        config_.min_joint_cameras == 0U ||
        config_.min_joint_cameras > kVioCameraCount ||
        !std::isfinite(config_.max_residual_rmse) ||
        config_.max_residual_rmse < 0.0 ||
        !std::isfinite(config_.max_rotation_increment_rad) ||
        config_.max_rotation_increment_rad < 0.0 ||
        !std::isfinite(config_.max_translation_increment_m) ||
        config_.max_translation_increment_m < 0.0 ||
        !std::isfinite(
          config_.max_measurement_covariance_scale) ||
        config_.max_measurement_covariance_scale < 1.0)
    {
      throw std::invalid_argument(
        "invalid joint visual update configuration");
    }
    hessian_.setZero();
    gradient_.setZero();
    camera_contributed_.fill(false);
  }

  bool addCamera(
    const std::uint8_t camera_id, const std::size_t tracked_points,
    const std::size_t residual_count, const double squared_residual_sum,
    const Matrix6 & pixel_hessian, const Vector6 & pixel_gradient,
    const std::size_t pixels_per_patch,
    const double measurement_covariance_scale = 1.0)
  {
    if (camera_id >= kVioCameraCount ||
        camera_contributed_[camera_id] ||
        tracked_points < config_.min_tracked_points_per_camera ||
        residual_count == 0U || pixels_per_patch == 0U ||
        !std::isfinite(measurement_covariance_scale) ||
        measurement_covariance_scale < 1.0 ||
        measurement_covariance_scale >
          config_.max_measurement_covariance_scale ||
        !std::isfinite(squared_residual_sum) ||
        squared_residual_sum < 0.0 ||
        !pixel_gradient.allFinite())
    {
      return false;
    }

    const auto hessian_validation =
      livo::validateSymmetricInformation(
      pixel_hessian,
      livo::SymmetricInformationRequirement::
      kPositiveSemidefinite);
    if (!hessian_validation.accepted())
    {
      return false;
    }
    const Matrix6 & symmetric_hessian =
      hessian_validation.symmetric;

    // Pixels inside one direct-method patch are strongly correlated. Scaling
    // their summed normal equation by the patch area makes one patch carry
    // approximately one independent unit of information instead of 64.
    const double normalization_denominator =
      static_cast<double>(pixels_per_patch) *
      measurement_covariance_scale;
    if (!std::isfinite(normalization_denominator) ||
        normalization_denominator <= 0.0 ||
        tracked_points >
          std::numeric_limits<std::size_t>::max() -
          tracked_points_ ||
        residual_count >
          std::numeric_limits<std::size_t>::max() -
          residual_count_)
    {
      return false;
    }
    const double information_scale =
      1.0 / normalization_denominator;
    const Matrix6 scaled_hessian =
      information_scale * symmetric_hessian;
    const Vector6 scaled_gradient =
      information_scale * pixel_gradient;
    const Matrix6 next_hessian = hessian_ + scaled_hessian;
    const Vector6 next_gradient = gradient_ + scaled_gradient;
    const double next_squared_residual_sum =
      squared_residual_sum_ + squared_residual_sum;
    if (!std::isfinite(information_scale) ||
        information_scale <= 0.0 ||
        !scaled_hessian.allFinite() ||
        !scaled_gradient.allFinite() ||
        !next_hessian.allFinite() ||
        !next_gradient.allFinite() ||
        !std::isfinite(next_squared_residual_sum))
    {
      return false;
    }

    hessian_ = next_hessian;
    gradient_ = next_gradient;
    squared_residual_sum_ = next_squared_residual_sum;
    residual_count_ += residual_count;
    tracked_points_ += tracked_points;
    camera_contributed_[camera_id] = true;
    ++contributing_cameras_;
    return true;
  }

  bool trackingReady() const
  {
    return contributing_cameras_ >= config_.min_joint_cameras &&
           tracked_points_ >= config_.min_joint_tracked_points;
  }

  bool innovationAccepted() const
  {
    const double rmse = residualRmse();
    return std::isfinite(rmse) && rmse <= config_.max_residual_rmse;
  }

  bool incrementAccepted(
    const double rotation_increment_rad,
    const double translation_increment_m) const
  {
    return std::isfinite(rotation_increment_rad) &&
           std::isfinite(translation_increment_m) &&
           rotation_increment_rad >= 0.0 &&
           translation_increment_m >= 0.0 &&
           rotation_increment_rad <=
             config_.max_rotation_increment_rad &&
           translation_increment_m <=
             config_.max_translation_increment_m;
  }

  double residualRmse() const
  {
    if (residual_count_ == 0U)
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return std::sqrt(
      squared_residual_sum_ / static_cast<double>(residual_count_));
  }

  const Matrix6 & hessian() const { return hessian_; }
  const Vector6 & gradient() const { return gradient_; }
  std::size_t contributingCameras() const { return contributing_cameras_; }
  std::size_t trackedPoints() const { return tracked_points_; }
  std::size_t residualCount() const { return residual_count_; }

private:
  JointVisualUpdateConfig config_;
  Matrix6 hessian_;
  Vector6 gradient_;
  std::array<bool, kVioCameraCount> camera_contributed_;
  std::size_t contributing_cameras_{0U};
  std::size_t tracked_points_{0U};
  std::size_t residual_count_{0U};
  double squared_residual_sum_{0.0};
};

#endif  // LIVO_JOINT_VISUAL_UPDATE_H_
