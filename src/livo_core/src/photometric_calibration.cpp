#include "photometric_calibration.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace livo
{
namespace
{

constexpr double kNumericalEpsilon = 1.0e-12;

bool finitePositive(const double value)
{
  return std::isfinite(value) && value > 0.0;
}

bool finiteNonnegative(const double value)
{
  return std::isfinite(value) && value >= 0.0;
}

PhotometricStatus validateCalibrationStructure(
  const PhotometricCameraCalibration & calibration)
{
  if (
    calibration.calibration_revision == 0U ||
    calibration.source_id.empty() ||
    calibration.image_width <= 0 ||
    calibration.image_height <= 0 ||
    !std::isfinite(calibration.black_level) ||
    !std::isfinite(calibration.white_level) ||
    calibration.white_level <= calibration.black_level ||
    !finitePositive(calibration.radiance_scale) ||
    !std::isfinite(calibration.radiance_offset) ||
    !std::isfinite(calibration.covariance_scale) ||
    calibration.covariance_scale < 1.0 ||
    calibration.inverse_response_lut.size() < 2U)
  {
    return PhotometricStatus::kInvalidCalibration;
  }

  double previous_response = -std::numeric_limits<double>::infinity();
  for (const double response : calibration.inverse_response_lut)
  {
    if (
      !finiteNonnegative(response) ||
      response + kNumericalEpsilon < previous_response)
    {
      return PhotometricStatus::kInvalidCalibration;
    }
    previous_response = response;
  }
  if (
    calibration.inverse_response_lut.back() -
    calibration.inverse_response_lut.front() <= kNumericalEpsilon)
  {
    return PhotometricStatus::kInvalidCalibration;
  }

  const std::size_t expected_vignetting_size =
    static_cast<std::size_t>(calibration.image_width) *
    static_cast<std::size_t>(calibration.image_height);
  if (
    calibration.vignetting_transmission.size() !=
    expected_vignetting_size)
  {
    return PhotometricStatus::kInvalidCalibration;
  }
  for (const double transmission : calibration.vignetting_transmission)
  {
    if (!finitePositive(transmission))
    {
      return PhotometricStatus::kInvalidCalibration;
    }
  }
  return PhotometricStatus::kValid;
}

PhotometricStatus validateMeasuredCalibration(
  const PhotometricCameraCalibration & calibration)
{
  if (!calibration.measuredCalibrationReady())
  {
    return PhotometricStatus::kCalibrationUnavailable;
  }
  return validateCalibrationStructure(calibration);
}

PhotometricStatus validateExposureGain(
  const PhotometricFrameMetadata & metadata)
{
  if (!metadata.exposure_sec || !metadata.sensor_gain)
  {
    return PhotometricStatus::kMissingExposureGain;
  }
  if (
    !finitePositive(*metadata.exposure_sec) ||
    !finitePositive(*metadata.sensor_gain) ||
    !finitePositive(
      *metadata.exposure_sec * *metadata.sensor_gain))
  {
    return PhotometricStatus::kInvalidExposureGain;
  }
  if (!metadata.hardwareExposureGainReady())
  {
    return PhotometricStatus::kUntrustedExposureGain;
  }
  return PhotometricStatus::kValid;
}

struct ResponseInterpolation
{
  double value{std::numeric_limits<double>::quiet_NaN()};
  double derivative_wrt_normalized_raw{
    std::numeric_limits<double>::quiet_NaN()};
};

ResponseInterpolation interpolateResponse(
  const PhotometricCameraCalibration & calibration,
  const double normalized_raw)
{
  ResponseInterpolation result;
  const double lut_position =
    normalized_raw *
    static_cast<double>(calibration.inverse_response_lut.size() - 1U);
  const std::size_t left_index =
    static_cast<std::size_t>(std::floor(lut_position));
  const std::size_t right_index =
    std::min(
      left_index + 1U,
      calibration.inverse_response_lut.size() - 1U);
  const double fraction =
    lut_position - static_cast<double>(left_index);
  result.value =
    (1.0 - fraction) *
    calibration.inverse_response_lut[left_index] +
    fraction * calibration.inverse_response_lut[right_index];
  result.derivative_wrt_normalized_raw =
    static_cast<double>(
      calibration.inverse_response_lut.size() - 1U) *
    (calibration.inverse_response_lut[right_index] -
     calibration.inverse_response_lut[left_index]);
  return result;
}

struct VignettingInterpolation
{
  double value{std::numeric_limits<double>::quiet_NaN()};
  double derivative_wrt_pixel_x{
    std::numeric_limits<double>::quiet_NaN()};
  double derivative_wrt_pixel_y{
    std::numeric_limits<double>::quiet_NaN()};
};

VignettingInterpolation interpolateVignetting(
  const PhotometricCameraCalibration & calibration,
  const double pixel_x,
  const double pixel_y)
{
  VignettingInterpolation result;
  const int left = static_cast<int>(std::floor(pixel_x));
  const int top = static_cast<int>(std::floor(pixel_y));
  const int right = std::min(left + 1, calibration.image_width - 1);
  const int bottom = std::min(top + 1, calibration.image_height - 1);
  const double horizontal_fraction = pixel_x - static_cast<double>(left);
  const double vertical_fraction = pixel_y - static_cast<double>(top);

  const auto at = [&calibration](const int x, const int y) {
      return calibration.vignetting_transmission[
        static_cast<std::size_t>(y * calibration.image_width + x)];
    };
  const double top_value =
    (1.0 - horizontal_fraction) * at(left, top) +
    horizontal_fraction * at(right, top);
  const double bottom_value =
    (1.0 - horizontal_fraction) * at(left, bottom) +
    horizontal_fraction * at(right, bottom);
  result.value =
    (1.0 - vertical_fraction) * top_value +
    vertical_fraction * bottom_value;
  result.derivative_wrt_pixel_x =
    (1.0 - vertical_fraction) *
      (at(right, top) - at(left, top)) +
    vertical_fraction *
      (at(right, bottom) - at(left, bottom));
  result.derivative_wrt_pixel_y = bottom_value - top_value;
  return result;
}

bool linearizeSampleValue(
  const PhotometricCameraCalibration & calibration,
  const PhotometricSample & sample,
  const bool absolute_radiance,
  const double exposure_gain,
  double * const value,
  double * const derivative_wrt_raw = nullptr,
  double * const derivative_wrt_pixel_x = nullptr,
  double * const derivative_wrt_pixel_y = nullptr)
{
  if (
    value == nullptr ||
    !std::isfinite(sample.raw_value) ||
    !std::isfinite(sample.pixel_x) ||
    !std::isfinite(sample.pixel_y) ||
    sample.pixel_x < 0.0 ||
    sample.pixel_y < 0.0 ||
    sample.pixel_x >
      static_cast<double>(calibration.image_width - 1) ||
    sample.pixel_y >
      static_cast<double>(calibration.image_height - 1) ||
    sample.raw_value <= calibration.black_level ||
    sample.raw_value >= calibration.white_level)
  {
    return false;
  }

  const double normalized_raw =
    (sample.raw_value - calibration.black_level) /
    (calibration.white_level - calibration.black_level);
  const ResponseInterpolation response =
    interpolateResponse(calibration, normalized_raw);
  const VignettingInterpolation transmission =
    interpolateVignetting(
      calibration, sample.pixel_x, sample.pixel_y);
  const double raw_range =
    calibration.white_level - calibration.black_level;
  double value_scale = 1.0;
  if (absolute_radiance)
  {
    value_scale =
      calibration.radiance_scale / exposure_gain;
  }
  const double linear_value =
    value_scale * response.value / transmission.value +
    (absolute_radiance ? calibration.radiance_offset : 0.0);
  const double raw_derivative =
    value_scale *
    response.derivative_wrt_normalized_raw /
    (raw_range * transmission.value);
  const double transmission_derivative_scale =
    -value_scale * response.value /
    (transmission.value * transmission.value);
  const double pixel_x_derivative =
    transmission_derivative_scale *
    transmission.derivative_wrt_pixel_x;
  const double pixel_y_derivative =
    transmission_derivative_scale *
    transmission.derivative_wrt_pixel_y;
  if (
    !std::isfinite(linear_value) ||
    !std::isfinite(raw_derivative) ||
    !std::isfinite(pixel_x_derivative) ||
    !std::isfinite(pixel_y_derivative))
  {
    return false;
  }
  *value = linear_value;
  if (derivative_wrt_raw != nullptr)
  {
    *derivative_wrt_raw = raw_derivative;
  }
  if (derivative_wrt_pixel_x != nullptr)
  {
    *derivative_wrt_pixel_x = pixel_x_derivative;
  }
  if (derivative_wrt_pixel_y != nullptr)
  {
    *derivative_wrt_pixel_y = pixel_y_derivative;
  }
  return true;
}

PhotometricPatchResult makeResult(
  const PhotometricPatchRequest & request,
  const PhotometricStatus status)
{
  PhotometricPatchResult result;
  result.status = status;
  result.mode = request.mode;
  result.experimental =
    request.mode == PhotometricMode::kRobustAffineExperimental;
  result.cross_camera_comparable = false;
  result.valid_mask.assign(
    request.reference.samples.size(), std::uint8_t{0U});
  result.reference_values.assign(
    request.reference.samples.size(),
    std::numeric_limits<double>::quiet_NaN());
  result.current_values.assign(
    request.reference.samples.size(),
    std::numeric_limits<double>::quiet_NaN());
  result.diagnostic = photometricStatusName(status);
  return result;
}

struct PairStatistics
{
  double reference_mean{0.0};
  double current_mean{0.0};
  double reference_standard_deviation{0.0};
  double current_standard_deviation{0.0};
  double zncc{std::numeric_limits<double>::quiet_NaN()};
};

PairStatistics computePairStatistics(
  const std::vector<double> & reference_values,
  const std::vector<double> & current_values)
{
  PairStatistics statistics;
  if (
    reference_values.empty() ||
    reference_values.size() != current_values.size())
  {
    return statistics;
  }

  double reference_scale = 0.0;
  double current_scale = 0.0;
  for (std::size_t index = 0U;
       index < reference_values.size(); ++index)
  {
    if (
      !std::isfinite(reference_values[index]) ||
      !std::isfinite(current_values[index]))
    {
      return statistics;
    }
    reference_scale = std::max(
      reference_scale, std::abs(reference_values[index]));
    current_scale = std::max(
      current_scale, std::abs(current_values[index]));
  }
  if (
    !finitePositive(reference_scale) ||
    !finitePositive(current_scale))
  {
    return statistics;
  }

  double reference_mean_scaled = 0.0;
  double current_mean_scaled = 0.0;
  double reference_m2_scaled = 0.0;
  double current_m2_scaled = 0.0;
  double cross_m2_scaled = 0.0;
  double sample_count = 0.0;
  for (std::size_t index = 0U;
       index < reference_values.size(); ++index)
  {
    const double reference =
      reference_values[index] / reference_scale;
    const double current =
      current_values[index] / current_scale;
    sample_count += 1.0;
    const double reference_delta =
      reference - reference_mean_scaled;
    reference_mean_scaled +=
      reference_delta / sample_count;
    const double current_delta =
      current - current_mean_scaled;
    current_mean_scaled += current_delta / sample_count;
    reference_m2_scaled +=
      reference_delta *
      (reference - reference_mean_scaled);
    current_m2_scaled +=
      current_delta * (current - current_mean_scaled);
    cross_m2_scaled +=
      reference_delta *
      (current - current_mean_scaled);
  }
  statistics.reference_mean =
    reference_mean_scaled * reference_scale;
  statistics.current_mean =
    current_mean_scaled * current_scale;
  statistics.reference_standard_deviation =
    std::sqrt(
      std::max(0.0, reference_m2_scaled) / sample_count) *
    reference_scale;
  statistics.current_standard_deviation =
    std::sqrt(
      std::max(0.0, current_m2_scaled) / sample_count) *
    current_scale;
  const double denominator =
    std::sqrt(std::max(0.0, reference_m2_scaled)) *
    std::sqrt(std::max(0.0, current_m2_scaled));
  if (
    std::isfinite(statistics.reference_mean) &&
    std::isfinite(statistics.current_mean) &&
    std::isfinite(
      statistics.reference_standard_deviation) &&
    std::isfinite(
      statistics.current_standard_deviation) &&
    finitePositive(denominator))
  {
    statistics.zncc = std::clamp(
      cross_m2_scaled / denominator, -1.0, 1.0);
  }
  return statistics;
}

double rootMeanSquareDifference(
  const std::vector<double> & reference_values,
  const std::vector<double> & current_values)
{
  if (
    reference_values.empty() ||
    reference_values.size() != current_values.size())
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double scale = 0.0;
  double scaled_squared_sum = 1.0;
  for (std::size_t index = 0U;
       index < reference_values.size(); ++index)
  {
    const double residual =
      current_values[index] - reference_values[index];
    if (!std::isfinite(residual))
    {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double absolute_residual = std::abs(residual);
    if (absolute_residual == 0.0)
    {
      continue;
    }
    if (scale < absolute_residual)
    {
      const double ratio = scale / absolute_residual;
      scaled_squared_sum =
        1.0 + scaled_squared_sum * ratio * ratio;
      scale = absolute_residual;
    }
    else
    {
      const double ratio = absolute_residual / scale;
      scaled_squared_sum += ratio * ratio;
    }
  }
  if (scale == 0.0)
  {
    return 0.0;
  }
  const double result =
    scale * std::sqrt(
      scaled_squared_sum /
      static_cast<double>(reference_values.size()));
  return
    std::isfinite(result) ?
    result : std::numeric_limits<double>::quiet_NaN();
}

double matrixConditionNumber(
  const double hessian_gain_gain,
  const double hessian_gain_bias,
  const double hessian_bias_bias)
{
  const double trace =
    hessian_gain_gain + hessian_bias_bias;
  const double discriminant = std::hypot(
    hessian_gain_gain - hessian_bias_bias,
    2.0 * hessian_gain_bias);
  const double maximum_eigenvalue = 0.5 * (trace + discriminant);
  const double minimum_eigenvalue = 0.5 * (trace - discriminant);
  if (
    !finitePositive(maximum_eigenvalue) ||
    !finitePositive(minimum_eigenvalue))
  {
    return std::numeric_limits<double>::infinity();
  }
  return maximum_eigenvalue / minimum_eigenvalue;
}

struct AffineFit
{
  PhotometricStatus status{PhotometricStatus::kInvalidInput};
  double gain{1.0};
  double bias{0.0};
  double condition_number{
    std::numeric_limits<double>::infinity()};
};

AffineFit fitRobustAffine(
  const std::vector<double> & reference_values,
  const std::vector<double> & current_values,
  const RobustAffinePhotometricConfig & config)
{
  AffineFit fit;
  const double gain_prior_information =
    1.0 / (config.gain_prior_stddev * config.gain_prior_stddev);
  const double bias_prior_information =
    1.0 / (config.bias_prior_stddev * config.bias_prior_stddev);

  double gain = 1.0;
  double bias = 0.0;
  double final_hessian_gain_gain = 0.0;
  double final_hessian_gain_bias = 0.0;
  double final_hessian_bias_bias = 0.0;
  for (std::size_t iteration = 0U;
       iteration < config.maximum_iterations; ++iteration)
  {
    double hessian_gain_gain = gain_prior_information;
    double hessian_gain_bias = 0.0;
    double hessian_bias_bias = bias_prior_information;
    double rhs_gain = gain_prior_information;
    double rhs_bias = 0.0;

    for (std::size_t index = 0U;
         index < reference_values.size(); ++index)
    {
      const double current = current_values[index];
      const double reference = reference_values[index];
      const double residual = gain * current + bias - reference;
      const double absolute_residual = std::abs(residual);
      const double weight =
        absolute_residual <= config.huber_delta ?
        1.0 : config.huber_delta / absolute_residual;
      hessian_gain_gain += weight * current * current;
      hessian_gain_bias += weight * current;
      hessian_bias_bias += weight;
      rhs_gain += weight * current * reference;
      rhs_bias += weight * reference;
    }

    const double schur_complement =
      hessian_gain_gain -
      hessian_gain_bias * hessian_gain_bias /
      hessian_bias_bias;
    if (
      !finitePositive(hessian_bias_bias) ||
      !finitePositive(schur_complement))
    {
      fit.status = PhotometricStatus::kAffineIllConditioned;
      return fit;
    }

    const double next_gain =
      (rhs_gain -
       hessian_gain_bias * rhs_bias / hessian_bias_bias) /
      schur_complement;
    const double next_bias =
      (rhs_bias - hessian_gain_bias * next_gain) /
      hessian_bias_bias;
    if (!std::isfinite(next_gain) || !std::isfinite(next_bias))
    {
      fit.status = PhotometricStatus::kAffineIllConditioned;
      return fit;
    }

    gain = next_gain;
    bias = next_bias;
    final_hessian_gain_gain = hessian_gain_gain;
    final_hessian_gain_bias = hessian_gain_bias;
    final_hessian_bias_bias = hessian_bias_bias;
  }

  fit.condition_number = matrixConditionNumber(
    final_hessian_gain_gain,
    final_hessian_gain_bias,
    final_hessian_bias_bias);
  if (
    !std::isfinite(fit.condition_number) ||
    fit.condition_number > config.maximum_normal_condition_number)
  {
    fit.status = PhotometricStatus::kAffineIllConditioned;
    return fit;
  }
  if (
    gain < config.minimum_gain ||
    gain > config.maximum_gain ||
    std::abs(bias) > config.maximum_absolute_bias)
  {
    fit.status = PhotometricStatus::kAffineOutOfBounds;
    return fit;
  }

  fit.status = PhotometricStatus::kValid;
  fit.gain = gain;
  fit.bias = bias;
  return fit;
}

}  // namespace

PhotometricCalibrationReceipt::PhotometricCalibrationReceipt(
  const PhotometricCameraCalibration & calibration)
: camera_id_(calibration.camera_id),
  calibration_revision_(calibration.calibration_revision),
  image_width_(calibration.image_width),
  image_height_(calibration.image_height),
  black_level_(calibration.black_level),
  white_level_(calibration.white_level),
  inverse_response_lut_(calibration.inverse_response_lut),
  vignetting_transmission_(calibration.vignetting_transmission),
  radiance_scale_(calibration.radiance_scale),
  radiance_offset_(calibration.radiance_offset),
  covariance_scale_(calibration.covariance_scale),
  source_id_(calibration.source_id)
{
}

bool PhotometricCalibrationReceipt::matches(
  const PhotometricCameraCalibration & calibration) const
{
  return
    camera_id_ == calibration.camera_id &&
    calibration_revision_ == calibration.calibration_revision &&
    image_width_ == calibration.image_width &&
    image_height_ == calibration.image_height &&
    black_level_ == calibration.black_level &&
    white_level_ == calibration.white_level &&
    inverse_response_lut_ == calibration.inverse_response_lut &&
    vignetting_transmission_ ==
      calibration.vignetting_transmission &&
    radiance_scale_ == calibration.radiance_scale &&
    radiance_offset_ == calibration.radiance_offset &&
    covariance_scale_ == calibration.covariance_scale &&
    source_id_ == calibration.source_id;
}

void PhotometricCameraCalibration::sealMeasuredReceipt()
{
  measured_receipt_ = PhotometricCalibrationReceipt(*this);
}

bool PhotometricCameraCalibration::measuredCalibrationReady() const
{
  return
    measured_receipt_ &&
    measured_receipt_->matches(*this);
}

PhotometricFrameMetadataReceipt::PhotometricFrameMetadataReceipt(
  const PhotometricFrameMetadata & metadata)
: exposure_sec_(metadata.exposure_sec),
  sensor_gain_(metadata.sensor_gain),
  saturated_pixel_fraction_(metadata.saturated_pixel_fraction),
  dark_pixel_fraction_(metadata.dark_pixel_fraction),
  quality_statistics_revision_(metadata.quality_statistics_revision)
{
}

bool PhotometricFrameMetadataReceipt::matchesExposureGain(
  const PhotometricFrameMetadata & metadata) const
{
  return
    exposure_sec_ == metadata.exposure_sec &&
    sensor_gain_ == metadata.sensor_gain;
}

bool PhotometricFrameMetadataReceipt::matchesQualityStatistics(
  const PhotometricFrameMetadata & metadata) const
{
  return
    saturated_pixel_fraction_ ==
      metadata.saturated_pixel_fraction &&
    dark_pixel_fraction_ == metadata.dark_pixel_fraction &&
    quality_statistics_revision_ ==
      metadata.quality_statistics_revision;
}

void PhotometricFrameMetadata::sealHardwareReceipt()
{
  hardware_receipt_ = PhotometricFrameMetadataReceipt(*this);
}

bool PhotometricFrameMetadata::hardwareExposureGainReady() const
{
  return
    hardware_receipt_ &&
    hardware_receipt_->matchesExposureGain(*this) &&
    exposure_sec &&
    sensor_gain;
}

bool PhotometricFrameMetadata::hardwareQualityStatisticsReady() const
{
  return
    hardware_receipt_ &&
    hardware_receipt_->matchesQualityStatistics(*this) &&
    quality_statistics_revision != 0U &&
    saturated_pixel_fraction &&
    dark_pixel_fraction;
}

PhotometricCalibration::PhotometricCalibration(
  std::vector<PhotometricCameraCalibration> camera_calibrations,
  RobustAffinePhotometricConfig robust_affine_config)
: camera_calibrations_(std::move(camera_calibrations)),
  robust_affine_config_(robust_affine_config)
{
}

bool PhotometricCalibration::configValid() const
{
  const RobustAffinePhotometricConfig & config =
    robust_affine_config_;
  if (
    config.minimum_valid_samples < 3U ||
    !finitePositive(config.minimum_standard_deviation) ||
    !std::isfinite(config.minimum_zncc) ||
    config.minimum_zncc < -1.0 ||
    config.minimum_zncc > 1.0 ||
    !finitePositive(config.minimum_gain) ||
    !std::isfinite(config.maximum_gain) ||
    config.maximum_gain <= config.minimum_gain ||
    !finiteNonnegative(config.maximum_absolute_bias) ||
    !finitePositive(config.gain_prior_stddev) ||
    !finitePositive(config.bias_prior_stddev) ||
    !finitePositive(config.huber_delta) ||
    config.maximum_iterations == 0U ||
    !std::isfinite(config.maximum_normal_condition_number) ||
    config.maximum_normal_condition_number <= 1.0 ||
    !std::isfinite(config.minimum_covariance_scale) ||
    config.minimum_covariance_scale < 1.0 ||
    !std::isfinite(config.maximum_covariance_scale) ||
    config.maximum_covariance_scale <
      config.minimum_covariance_scale)
  {
    return false;
  }

  for (std::size_t first = 0U;
       first < camera_calibrations_.size(); ++first)
  {
    for (std::size_t second = first + 1U;
         second < camera_calibrations_.size(); ++second)
    {
      if (
        camera_calibrations_[first].camera_id ==
        camera_calibrations_[second].camera_id)
      {
        return false;
      }
    }
  }
  return true;
}

const PhotometricCameraCalibration * PhotometricCalibration::findCamera(
  const std::size_t camera_id) const
{
  const auto iterator = std::find_if(
    camera_calibrations_.begin(), camera_calibrations_.end(),
    [camera_id](const PhotometricCameraCalibration & calibration) {
      return calibration.camera_id == camera_id;
    });
  return
    iterator == camera_calibrations_.end() ?
    nullptr : &(*iterator);
}

PhotometricStatus PhotometricCalibration::cameraCalibrationStatus(
  const std::size_t camera_id) const
{
  if (!configValid())
  {
    return PhotometricStatus::kInvalidConfiguration;
  }
  const PhotometricCameraCalibration * const calibration =
    findCamera(camera_id);
  if (calibration == nullptr)
  {
    return PhotometricStatus::kUnknownCamera;
  }
  return validateMeasuredCalibration(*calibration);
}

std::uint64_t PhotometricCalibration::cameraCalibrationRevision(
  const std::size_t camera_id) const
{
  if (!configValid())
  {
    return 0U;
  }
  const PhotometricCameraCalibration * const calibration =
    findCamera(camera_id);
  return
    calibration != nullptr &&
    validateMeasuredCalibration(*calibration) ==
      PhotometricStatus::kValid ?
    calibration->calibration_revision : 0U;
}

PhotometricSampleLinearization
PhotometricCalibration::linearizeSample(
  const std::size_t camera_id,
  const PhotometricFrameMetadata & metadata,
  const PhotometricSample & sample,
  const PhotometricMode mode) const
{
  PhotometricSampleLinearization result;
  result.mode = mode;
  result.experimental =
    mode == PhotometricMode::kRobustAffineExperimental;
  if (!configValid())
  {
    result.status = PhotometricStatus::kInvalidConfiguration;
    return result;
  }
  const PhotometricCameraCalibration * const calibration =
    findCamera(camera_id);
  if (calibration == nullptr)
  {
    result.status = PhotometricStatus::kUnknownCamera;
    return result;
  }
  const PhotometricStatus calibration_status =
    mode == PhotometricMode::kCalibratedDirect ?
    validateMeasuredCalibration(*calibration) :
    validateCalibrationStructure(*calibration);
  if (calibration_status != PhotometricStatus::kValid)
  {
    result.status = calibration_status;
    return result;
  }

  const bool calibrated_direct =
    mode == PhotometricMode::kCalibratedDirect;
  double exposure_gain = 1.0;
  if (calibrated_direct)
  {
    const PhotometricStatus metadata_status =
      validateExposureGain(metadata);
    if (metadata_status != PhotometricStatus::kValid)
    {
      result.status = metadata_status;
      return result;
    }
    exposure_gain =
      *metadata.exposure_sec * *metadata.sensor_gain;
  }
  else if (mode != PhotometricMode::kRobustAffineExperimental)
  {
    result.status = PhotometricStatus::kInvalidInput;
    return result;
  }

  if (!linearizeSampleValue(
      *calibration, sample, calibrated_direct, exposure_gain,
      &result.value, &result.derivative_wrt_raw,
      &result.derivative_wrt_pixel_x,
      &result.derivative_wrt_pixel_y))
  {
    result.status = PhotometricStatus::kInvalidInput;
    return result;
  }
  result.status = PhotometricStatus::kValid;
  result.cross_camera_comparable = calibrated_direct;
  result.experimental = !calibrated_direct;
  result.covariance_scale = calibration->covariance_scale;
  return result;
}

PhotometricPatchResult PhotometricCalibration::compare(
  const PhotometricPatchRequest & request) const
{
  if (!configValid())
  {
    return makeResult(
      request, PhotometricStatus::kInvalidConfiguration);
  }
  if (
    request.reference.samples.size() !=
    request.current.samples.size())
  {
    return makeResult(
      request, PhotometricStatus::kPatchSizeMismatch);
  }
  if (request.reference.samples.empty())
  {
    return makeResult(
      request, PhotometricStatus::kInsufficientSamples);
  }

  const PhotometricCameraCalibration * const reference_calibration =
    findCamera(request.reference.camera_id);
  const PhotometricCameraCalibration * const current_calibration =
    findCamera(request.current.camera_id);
  if (
    reference_calibration == nullptr ||
    current_calibration == nullptr)
  {
    return makeResult(
      request, PhotometricStatus::kUnknownCamera);
  }

  const bool calibrated_direct =
    request.mode == PhotometricMode::kCalibratedDirect;
  const PhotometricStatus reference_calibration_status =
    calibrated_direct ?
    validateMeasuredCalibration(*reference_calibration) :
    validateCalibrationStructure(*reference_calibration);
  if (reference_calibration_status != PhotometricStatus::kValid)
  {
    return makeResult(request, reference_calibration_status);
  }
  const PhotometricStatus current_calibration_status =
    calibrated_direct ?
    validateMeasuredCalibration(*current_calibration) :
    validateCalibrationStructure(*current_calibration);
  if (current_calibration_status != PhotometricStatus::kValid)
  {
    return makeResult(request, current_calibration_status);
  }

  double reference_exposure_gain = 1.0;
  double current_exposure_gain = 1.0;
  if (calibrated_direct)
  {
    const PhotometricStatus reference_metadata_status =
      validateExposureGain(request.reference.metadata);
    if (reference_metadata_status != PhotometricStatus::kValid)
    {
      return makeResult(request, reference_metadata_status);
    }
    const PhotometricStatus current_metadata_status =
      validateExposureGain(request.current.metadata);
    if (current_metadata_status != PhotometricStatus::kValid)
    {
      return makeResult(request, current_metadata_status);
    }
    reference_exposure_gain =
      *request.reference.metadata.exposure_sec *
      *request.reference.metadata.sensor_gain;
    current_exposure_gain =
      *request.current.metadata.exposure_sec *
      *request.current.metadata.sensor_gain;
  }
  else if (
    request.mode !=
    PhotometricMode::kRobustAffineExperimental)
  {
    return makeResult(
      request, PhotometricStatus::kInvalidInput);
  }

  PhotometricPatchResult result =
    makeResult(request, PhotometricStatus::kValid);
  std::vector<double> compact_reference_values;
  std::vector<double> compact_current_values;
  compact_reference_values.reserve(request.reference.samples.size());
  compact_current_values.reserve(request.current.samples.size());

  for (std::size_t index = 0U;
       index < request.reference.samples.size(); ++index)
  {
    double reference_value = 0.0;
    double current_value = 0.0;
    if (
      !linearizeSampleValue(
        *reference_calibration,
        request.reference.samples[index],
        calibrated_direct,
        reference_exposure_gain,
        &reference_value) ||
      !linearizeSampleValue(
        *current_calibration,
        request.current.samples[index],
        calibrated_direct,
        current_exposure_gain,
        &current_value))
    {
      continue;
    }
    result.valid_mask[index] = std::uint8_t{1U};
    result.reference_values[index] = reference_value;
    result.current_values[index] = current_value;
    compact_reference_values.push_back(reference_value);
    compact_current_values.push_back(current_value);
  }
  result.valid_sample_count = compact_reference_values.size();

  if (calibrated_direct)
  {
    if (compact_reference_values.empty())
    {
      return makeResult(
        request, PhotometricStatus::kInsufficientSamples);
    }

    const PairStatistics statistics = computePairStatistics(
      compact_reference_values, compact_current_values);
    const double residual_rmse = rootMeanSquareDifference(
      compact_reference_values, compact_current_values);
    if (
      !std::isfinite(statistics.zncc) ||
      !std::isfinite(residual_rmse))
    {
      return makeResult(
        request, PhotometricStatus::kDegeneratePatch);
    }
    result.status = PhotometricStatus::kValid;
    result.cross_camera_comparable = true;
    result.experimental = false;
    result.estimated_gain = 1.0;
    result.estimated_bias = 0.0;
    result.zncc = statistics.zncc;
    result.residual_rmse = residual_rmse;
    result.normal_condition_number = 1.0;
    result.covariance_scale = std::max(
      reference_calibration->covariance_scale,
      current_calibration->covariance_scale);
    result.diagnostic = "calibrated_direct";
    return result;
  }

  if (
    compact_reference_values.size() <
    robust_affine_config_.minimum_valid_samples)
  {
    return makeResult(
      request, PhotometricStatus::kInsufficientSamples);
  }
  const PairStatistics statistics = computePairStatistics(
    compact_reference_values, compact_current_values);
  if (
    !std::isfinite(statistics.zncc) ||
    statistics.reference_standard_deviation <
      robust_affine_config_.minimum_standard_deviation ||
    statistics.current_standard_deviation <
      robust_affine_config_.minimum_standard_deviation)
  {
    return makeResult(
      request, PhotometricStatus::kDegeneratePatch);
  }
  if (statistics.zncc < robust_affine_config_.minimum_zncc)
  {
    return makeResult(
      request, PhotometricStatus::kLowZncc);
  }

  const AffineFit fit = fitRobustAffine(
    compact_reference_values,
    compact_current_values,
    robust_affine_config_);
  if (fit.status != PhotometricStatus::kValid)
  {
    PhotometricPatchResult failed =
      makeResult(request, fit.status);
    failed.normal_condition_number = fit.condition_number;
    return failed;
  }

  for (std::size_t index = 0U;
       index < result.valid_mask.size(); ++index)
  {
    if (result.valid_mask[index] == std::uint8_t{0U})
    {
      continue;
    }
    result.current_values[index] =
      fit.gain * result.current_values[index] + fit.bias;
  }
  std::vector<double> fitted_reference_values;
  std::vector<double> fitted_current_values;
  fitted_reference_values.reserve(result.valid_sample_count);
  fitted_current_values.reserve(result.valid_sample_count);
  for (std::size_t index = 0U;
       index < result.valid_mask.size(); ++index)
  {
    if (result.valid_mask[index] == std::uint8_t{0U})
    {
      continue;
    }
    fitted_reference_values.push_back(
      result.reference_values[index]);
    fitted_current_values.push_back(
      result.current_values[index]);
  }
  const double residual_rmse = rootMeanSquareDifference(
    fitted_reference_values, fitted_current_values);
  if (!std::isfinite(residual_rmse))
  {
    return makeResult(
      request, PhotometricStatus::kInvalidInput);
  }
  const double quality_inflation =
    1.0 +
    residual_rmse / robust_affine_config_.huber_delta +
    std::max(0.0, 1.0 - statistics.zncc);
  const double calibration_covariance_scale = std::max(
    reference_calibration->covariance_scale,
    current_calibration->covariance_scale);

  result.status = PhotometricStatus::kValid;
  result.cross_camera_comparable = false;
  result.experimental = true;
  result.estimated_gain = fit.gain;
  result.estimated_bias = fit.bias;
  result.zncc = statistics.zncc;
  result.residual_rmse = residual_rmse;
  result.normal_condition_number = fit.condition_number;
  result.covariance_scale = std::clamp(
    std::max(
      robust_affine_config_.minimum_covariance_scale,
      calibration_covariance_scale * quality_inflation),
    robust_affine_config_.minimum_covariance_scale,
    robust_affine_config_.maximum_covariance_scale);
  result.diagnostic =
    "robust_affine_experimental_non_radiometric";
  return result;
}

const char * photometricStatusName(const PhotometricStatus status)
{
  switch (status)
  {
    case PhotometricStatus::kValid:
      return "VALID";
    case PhotometricStatus::kInvalidConfiguration:
      return "INVALID_CONFIGURATION";
    case PhotometricStatus::kInvalidInput:
      return "INVALID_INPUT";
    case PhotometricStatus::kUnknownCamera:
      return "UNKNOWN_CAMERA";
    case PhotometricStatus::kCalibrationUnavailable:
      return "CALIBRATION_UNAVAILABLE";
    case PhotometricStatus::kInvalidCalibration:
      return "INVALID_CALIBRATION";
    case PhotometricStatus::kMissingExposureGain:
      return "MISSING_EXPOSURE_GAIN";
    case PhotometricStatus::kInvalidExposureGain:
      return "INVALID_EXPOSURE_GAIN";
    case PhotometricStatus::kUntrustedExposureGain:
      return "UNTRUSTED_EXPOSURE_GAIN";
    case PhotometricStatus::kPatchSizeMismatch:
      return "PATCH_SIZE_MISMATCH";
    case PhotometricStatus::kInsufficientSamples:
      return "INSUFFICIENT_SAMPLES";
    case PhotometricStatus::kDegeneratePatch:
      return "DEGENERATE_PATCH";
    case PhotometricStatus::kAffineIllConditioned:
      return "AFFINE_ILL_CONDITIONED";
    case PhotometricStatus::kAffineOutOfBounds:
      return "AFFINE_OUT_OF_BOUNDS";
    case PhotometricStatus::kLowZncc:
      return "LOW_ZNCC";
  }
  return "UNKNOWN";
}

}  // namespace livo
