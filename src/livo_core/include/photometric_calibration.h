#ifndef LIVO_PHOTOMETRIC_CALIBRATION_H_
#define LIVO_PHOTOMETRIC_CALIBRATION_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace livo
{

enum class PhotometricMode
{
  kCalibratedDirect,
  kRobustAffineExperimental
};

enum class PhotometricStatus
{
  kValid,
  kInvalidConfiguration,
  kInvalidInput,
  kUnknownCamera,
  kCalibrationUnavailable,
  kInvalidCalibration,
  kMissingExposureGain,
  kInvalidExposureGain,
  kUntrustedExposureGain,
  kPatchSizeMismatch,
  kInsufficientSamples,
  kDegeneratePatch,
  kAffineIllConditioned,
  kAffineOutOfBounds,
  kLowZncc
};

struct PhotometricCameraCalibration;
struct PhotometricFrameMetadata;
class TrustedPhotometricCalibrationLoader;

// Opaque receipt issued by the calibration-loading boundary. Public code can
// copy a receipt carried by a DTO, but cannot mint one or alter its sealed
// payload.
class PhotometricCalibrationReceipt
{
public:
  PhotometricCalibrationReceipt(
    const PhotometricCalibrationReceipt &) = default;
  PhotometricCalibrationReceipt(
    PhotometricCalibrationReceipt &&) = default;
  PhotometricCalibrationReceipt & operator=(
    const PhotometricCalibrationReceipt &) = default;
  PhotometricCalibrationReceipt & operator=(
    PhotometricCalibrationReceipt &&) = default;

private:
  explicit PhotometricCalibrationReceipt(
    const PhotometricCameraCalibration & calibration);
  bool matches(
    const PhotometricCameraCalibration & calibration) const;

  std::size_t camera_id_{0U};
  std::uint64_t calibration_revision_{0U};
  int image_width_{0};
  int image_height_{0};
  double black_level_{0.0};
  double white_level_{0.0};
  std::vector<double> inverse_response_lut_;
  std::vector<double> vignetting_transmission_;
  double radiance_scale_{0.0};
  double radiance_offset_{0.0};
  double covariance_scale_{0.0};
  std::string source_id_;

  friend struct PhotometricCameraCalibration;
};

// Opaque receipt issued where exposure/gain and image-quality statistics are
// captured from the hardware frame. It seals the exact metadata values.
class PhotometricFrameMetadataReceipt
{
public:
  PhotometricFrameMetadataReceipt(
    const PhotometricFrameMetadataReceipt &) = default;
  PhotometricFrameMetadataReceipt(
    PhotometricFrameMetadataReceipt &&) = default;
  PhotometricFrameMetadataReceipt & operator=(
    const PhotometricFrameMetadataReceipt &) = default;
  PhotometricFrameMetadataReceipt & operator=(
    PhotometricFrameMetadataReceipt &&) = default;

private:
  explicit PhotometricFrameMetadataReceipt(
    const PhotometricFrameMetadata & metadata);
  bool matchesExposureGain(
    const PhotometricFrameMetadata & metadata) const;
  bool matchesQualityStatistics(
    const PhotometricFrameMetadata & metadata) const;

  std::optional<double> exposure_sec_;
  std::optional<double> sensor_gain_;
  std::optional<double> saturated_pixel_fraction_;
  std::optional<double> dark_pixel_fraction_;
  std::uint64_t quality_statistics_revision_{0U};

  friend struct PhotometricFrameMetadata;
};

struct PhotometricFrameMetadata
{
  std::optional<double> exposure_sec;
  std::optional<double> sensor_gain;
  std::optional<double> saturated_pixel_fraction;
  std::optional<double> dark_pixel_fraction;
  std::uint64_t quality_statistics_revision{0U};

  bool hardwareExposureGainReady() const;
  bool hardwareQualityStatisticsReady() const;

private:
  std::optional<PhotometricFrameMetadataReceipt>
    hardware_receipt_;

  void sealHardwareReceipt();
};

struct PhotometricCameraCalibration
{
  std::size_t camera_id{0U};
  std::uint64_t calibration_revision{0U};
  int image_width{0};
  int image_height{0};
  double black_level{0.0};
  double white_level{255.0};

  // Maps normalized raw intensity [0, 1] to a linear response. Values must be
  // finite, nonnegative, and monotonically nondecreasing.
  std::vector<double> inverse_response_lut;

  // Per-pixel optical transmission in row-major order. Values must be finite
  // and strictly positive; radiance is divided by this map.
  std::vector<double> vignetting_transmission;

  // Optional fixed cross-camera radiance alignment obtained during the same
  // calibration procedure.
  double radiance_scale{1.0};
  double radiance_offset{0.0};
  double covariance_scale{1.0};
  std::string source_id;

  bool measuredCalibrationReady() const;

private:
  std::optional<PhotometricCalibrationReceipt>
    measured_receipt_;

  void sealMeasuredReceipt();

  friend class TrustedPhotometricCalibrationLoader;
};

struct PhotometricSample
{
  double raw_value{0.0};
  double pixel_x{0.0};
  double pixel_y{0.0};
};

struct PhotometricPatch
{
  std::size_t camera_id{0U};
  PhotometricFrameMetadata metadata;
  std::vector<PhotometricSample> samples;
};

struct RobustAffinePhotometricConfig
{
  std::size_t minimum_valid_samples{8U};
  double minimum_standard_deviation{1.0e-3};
  double minimum_zncc{0.55};
  double minimum_gain{0.25};
  double maximum_gain{4.0};
  double maximum_absolute_bias{1.0};
  double gain_prior_stddev{0.5};
  double bias_prior_stddev{0.25};
  double huber_delta{0.05};
  std::size_t maximum_iterations{5U};
  double maximum_normal_condition_number{1.0e8};
  double minimum_covariance_scale{4.0};
  double maximum_covariance_scale{100.0};
};

struct PhotometricPatchRequest
{
  PhotometricMode mode{PhotometricMode::kCalibratedDirect};
  PhotometricPatch reference;
  PhotometricPatch current;
};

struct PhotometricPatchResult
{
  PhotometricStatus status{PhotometricStatus::kInvalidInput};
  PhotometricMode mode{PhotometricMode::kCalibratedDirect};
  bool cross_camera_comparable{false};
  bool experimental{false};
  std::vector<std::uint8_t> valid_mask;
  std::vector<double> reference_values;
  std::vector<double> current_values;
  std::size_t valid_sample_count{0U};
  double estimated_gain{
    std::numeric_limits<double>::quiet_NaN()};
  double estimated_bias{
    std::numeric_limits<double>::quiet_NaN()};
  double zncc{
    std::numeric_limits<double>::quiet_NaN()};
  double residual_rmse{
    std::numeric_limits<double>::quiet_NaN()};
  double normal_condition_number{
    std::numeric_limits<double>::quiet_NaN()};
  double covariance_scale{
    std::numeric_limits<double>::infinity()};
  std::string diagnostic;

  bool valid() const
  {
    if (
      status != PhotometricStatus::kValid ||
      valid_mask.size() != reference_values.size() ||
      valid_mask.size() != current_values.size() ||
      !std::isfinite(estimated_gain) ||
      !std::isfinite(estimated_bias) ||
      !std::isfinite(zncc) ||
      !std::isfinite(residual_rmse) ||
      !std::isfinite(normal_condition_number) ||
      !std::isfinite(covariance_scale) ||
      covariance_scale < 1.0)
    {
      return false;
    }
    std::size_t counted_valid_samples = 0U;
    for (std::size_t index = 0U;
         index < valid_mask.size(); ++index)
    {
      if (valid_mask[index] == std::uint8_t{0U})
      {
        continue;
      }
      if (
        valid_mask[index] != std::uint8_t{1U} ||
        !std::isfinite(reference_values[index]) ||
        !std::isfinite(current_values[index]))
      {
        return false;
      }
      ++counted_valid_samples;
    }
    return counted_valid_samples == valid_sample_count;
  }
};

struct PhotometricSampleLinearization
{
  PhotometricStatus status{PhotometricStatus::kInvalidInput};
  PhotometricMode mode{PhotometricMode::kCalibratedDirect};
  bool cross_camera_comparable{false};
  bool experimental{false};
  double value{std::numeric_limits<double>::quiet_NaN()};
  double derivative_wrt_raw{
    std::numeric_limits<double>::quiet_NaN()};
  double derivative_wrt_pixel_x{
    std::numeric_limits<double>::quiet_NaN()};
  double derivative_wrt_pixel_y{
    std::numeric_limits<double>::quiet_NaN()};
  double covariance_scale{
    std::numeric_limits<double>::infinity()};

  bool valid() const
  {
    return
      status == PhotometricStatus::kValid &&
      std::isfinite(value) &&
      std::isfinite(derivative_wrt_raw) &&
      std::isfinite(derivative_wrt_pixel_x) &&
      std::isfinite(derivative_wrt_pixel_y) &&
      std::isfinite(covariance_scale) &&
      covariance_scale >= 1.0;
  }
};

class PhotometricCalibration
{
public:
  explicit PhotometricCalibration(
    std::vector<PhotometricCameraCalibration> camera_calibrations,
    RobustAffinePhotometricConfig robust_affine_config =
      RobustAffinePhotometricConfig{});

  bool configValid() const;

  PhotometricStatus cameraCalibrationStatus(
    std::size_t camera_id) const;

  std::uint64_t cameraCalibrationRevision(
    std::size_t camera_id) const;

  PhotometricPatchResult compare(
    const PhotometricPatchRequest & request) const;

  // Linearizes the calibrated scalar value while holding frame metadata
  // fixed. Pixel derivatives include only the calibration field (vignetting);
  // callers compose derivative_wrt_raw with their image interpolation
  // gradient.
  PhotometricSampleLinearization linearizeSample(
    std::size_t camera_id,
    const PhotometricFrameMetadata & metadata,
    const PhotometricSample & sample,
    PhotometricMode mode) const;

private:
  const PhotometricCameraCalibration * findCamera(
    std::size_t camera_id) const;

  std::vector<PhotometricCameraCalibration> camera_calibrations_;
  RobustAffinePhotometricConfig robust_affine_config_;
};

const char * photometricStatusName(PhotometricStatus status);

}  // namespace livo

#endif  // LIVO_PHOTOMETRIC_CALIBRATION_H_
