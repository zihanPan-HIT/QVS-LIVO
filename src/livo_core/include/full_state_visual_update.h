#ifndef LIVO_FULL_STATE_VISUAL_UPDATE_H_
#define LIVO_FULL_STATE_VISUAL_UPDATE_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "camera_observation.h"
#include "camera_time_transfer.h"
#include "multi_view_landmark.h"

namespace livo
{

inline constexpr Eigen::Index kFullStateVisualDimension = 19;
inline constexpr Eigen::Index kFullStateVisualThetaOffset = 0;
inline constexpr Eigen::Index kFullStateVisualPositionOffset = 3;
inline constexpr Eigen::Index kFullStateVisualExposureOffset = 6;
inline constexpr Eigen::Index kFullStateVisualVelocityOffset = 7;
inline constexpr Eigen::Index kFullStateVisualGyroBiasOffset = 10;
inline constexpr Eigen::Index kFullStateVisualAccelBiasOffset = 13;
inline constexpr Eigen::Index kFullStateVisualGravityOffset = 16;
inline constexpr std::uint32_t kFullStateVisualLayoutRevision =
  kCameraTimeTransferActiveJacobianRevision;
inline constexpr const char * kFullStateVisualLayoutId =
  kCameraTimeTransferActiveStateLayoutId;

static_assert(
  kFullStateVisualDimension ==
  kCameraTimeTransferActiveStateDimension);
static_assert(
  kFullStateVisualExposureOffset ==
  kCameraTimeTransferActiveExposureOffset);
static_assert(
  kFullStateVisualThetaOffset ==
  kCameraTimeTransferThetaOffset);
static_assert(
  kFullStateVisualPositionOffset ==
  kCameraTimeTransferPositionOffset);
static_assert(
  kFullStateVisualVelocityOffset ==
  kCameraTimeTransferVelocityOffset + 1);
static_assert(
  kFullStateVisualGyroBiasOffset ==
  kCameraTimeTransferGyroBiasOffset + 1);
static_assert(
  kFullStateVisualAccelBiasOffset ==
  kCameraTimeTransferAccelBiasOffset + 1);
static_assert(
  kFullStateVisualGravityOffset ==
  kCameraTimeTransferGravityOffset + 1);
static_assert(
  kFullStateVisualLayoutRevision ==
  kCameraTimeTransferActiveJacobianRevision);
static_assert(
  kFullStateVisualLayoutId ==
  kCameraTimeTransferActiveStateLayoutId);

using FullStateVisualMatrix =
  Eigen::Matrix<double, kFullStateVisualDimension,
    kFullStateVisualDimension>;
using FullStateVisualVector =
  Eigen::Matrix<double, kFullStateVisualDimension, 1>;

struct FullStateVisualUpdateConfig
{
  std::size_t minimum_unique_landmarks{1U};
  std::size_t minimum_cameras{1U};
  double maximum_residual_rmse{20.0};
  double maximum_rotation_increment_rad{
    0.1 * 3.14159265358979323846 / 180.0};
  double maximum_translation_increment_m{0.01};
  double maximum_auxiliary_increment_norm{1.0};
  double maximum_prior_condition_number{1.0e12};
  double maximum_system_condition_number{1.0e12};
};

struct FullStateVisualContribution
{
  LandmarkId landmark_id{0U};
  std::uint8_t camera_id{0U};
  FullStateVisualMatrix information_hessian{
    FullStateVisualMatrix::Zero()};
  FullStateVisualVector information_gradient{
    FullStateVisualVector::Zero()};
  double squared_residual_sum{0.0};
  std::size_t residual_count{0U};
  std::size_t information_units{1U};
};

struct FullStateVisualUpdateRequest
{
  std::uint64_t batch_id{0U};
  std::string state_layout_id{kFullStateVisualLayoutId};
  std::uint32_t state_layout_revision{
    kFullStateVisualLayoutRevision};
  FullStateVisualMatrix prior_covariance{
    FullStateVisualMatrix::Identity()};
  std::vector<FullStateVisualContribution> contributions;
};

enum class FullStateVisualUpdateStatus
{
  kAccepted,
  kInvalidRequest,
  kInvalidLayout,
  kInvalidPrior,
  kDuplicateLandmark,
  kInvalidContribution,
  kAccumulationOverflow,
  kInsufficientInformation,
  kInnovationRejected,
  kIllConditionedPrior,
  kIllConditionedSystem,
  kIncrementRejected,
  kExposureCouplingRejected,
  kInvalidPosterior
};

struct FullStateVisualUpdateResult
{
  FullStateVisualUpdateStatus status{
    FullStateVisualUpdateStatus::kInvalidRequest};
  std::uint64_t batch_id{0U};
  FullStateVisualVector increment{
    FullStateVisualVector::Zero()};
  FullStateVisualMatrix posterior_covariance{
    FullStateVisualMatrix::Zero()};
  std::size_t unique_landmarks{0U};
  std::size_t contributing_cameras{0U};
  std::size_t residual_count{0U};
  std::size_t information_units{0U};
  double residual_rmse{
    std::numeric_limits<double>::infinity()};
  double prior_condition_number{
    std::numeric_limits<double>::infinity()};
  double system_condition_number{
    std::numeric_limits<double>::infinity()};

  bool accepted() const
  {
    return status == FullStateVisualUpdateStatus::kAccepted;
  }
};

class FullStateVisualUpdate
{
public:
  explicit FullStateVisualUpdate(
    FullStateVisualUpdateConfig config =
      FullStateVisualUpdateConfig{});

  FullStateVisualUpdateResult solve(
    const FullStateVisualUpdateRequest & request) const;

private:
  FullStateVisualUpdateConfig config_;
};

const char * fullStateVisualUpdateStatusName(
  FullStateVisualUpdateStatus status);

}  // namespace livo

#endif  // LIVO_FULL_STATE_VISUAL_UPDATE_H_
