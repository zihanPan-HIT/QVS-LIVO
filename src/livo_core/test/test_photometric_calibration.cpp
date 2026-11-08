#include "photometric_calibration.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace livo
{

namespace
{

static_assert(
  !std::is_default_constructible_v<
    PhotometricCalibrationReceipt>);
static_assert(
  !std::is_aggregate_v<PhotometricCalibrationReceipt>);
static_assert(
  !std::is_default_constructible_v<
    PhotometricFrameMetadataReceipt>);
static_assert(
  !std::is_aggregate_v<PhotometricFrameMetadataReceipt>);
static_assert(
  !std::is_aggregate_v<PhotometricCameraCalibration>);
static_assert(
  !std::is_aggregate_v<PhotometricFrameMetadata>);

PhotometricCameraCalibration makeCalibration(
  const std::size_t camera_id)
{
  PhotometricCameraCalibration calibration;
  calibration.camera_id = camera_id;
  calibration.calibration_revision =
    1U + static_cast<std::uint64_t>(camera_id);
  calibration.image_width = 4;
  calibration.image_height = 4;
  calibration.black_level = 0.0;
  calibration.white_level = 255.0;
  calibration.inverse_response_lut.resize(256U);
  for (std::size_t index = 0U;
       index < calibration.inverse_response_lut.size(); ++index)
  {
    calibration.inverse_response_lut[index] =
      static_cast<double>(index) / 255.0;
  }
  calibration.vignetting_transmission.assign(16U, 1.0);
  calibration.source_id =
    "unit-test-camera-" + std::to_string(camera_id);
  return calibration;
}

PhotometricSample makeSample(const double raw_value)
{
  PhotometricSample sample;
  sample.raw_value = raw_value;
  sample.pixel_x = 1.5;
  sample.pixel_y = 1.5;
  return sample;
}

PhotometricPatchRequest makeRequest(
  const PhotometricMode mode,
  const std::vector<double> & reference_raw_values,
  const std::vector<double> & current_raw_values)
{
  PhotometricPatchRequest request;
  request.mode = mode;
  request.reference.camera_id = 0U;
  request.current.camera_id = 1U;
  for (const double raw_value : reference_raw_values)
  {
    request.reference.samples.push_back(makeSample(raw_value));
  }
  for (const double raw_value : current_raw_values)
  {
    request.current.samples.push_back(makeSample(raw_value));
  }
  return request;
}

TEST(
  PhotometricCalibrationTest,
  MissingMeasuredCalibrationFailsClosedForDirectMode)
{
  PhotometricCameraCalibration unavailable;
  unavailable.camera_id = 0U;
  PhotometricCalibration calibration({unavailable});

  PhotometricPatchRequest direct = makeRequest(
    PhotometricMode::kCalibratedDirect, {40.0}, {40.0});
  direct.current.camera_id = 0U;
  EXPECT_EQ(
    calibration.compare(direct).status,
    PhotometricStatus::kCalibrationUnavailable);

}

TEST(
  PhotometricCalibrationTest,
  CallerFilledCalibrationCannotMintMeasuredReceipt)
{
  PhotometricCameraCalibration raw_only = makeCalibration(0U);
  PhotometricCalibration calibration({raw_only});

  EXPECT_TRUE(
    raw_only.calibration_revision != 0U &&
    !raw_only.source_id.empty());
  EXPECT_FALSE(raw_only.measuredCalibrationReady());
  EXPECT_EQ(
    calibration.cameraCalibrationStatus(0U),
    PhotometricStatus::kCalibrationUnavailable);
}

TEST(
  PhotometricCalibrationTest,
  CallerFilledCalibrationAndMetadataCannotEnableDirectMode)
{
  PhotometricCalibration calibration(
    {makeCalibration(0U), makeCalibration(1U)});
  PhotometricPatchRequest request = makeRequest(
    PhotometricMode::kCalibratedDirect,
    {40.0, 60.0, 80.0},
    {80.0, 120.0, 160.0});
  request.reference.metadata.exposure_sec = 0.01;
  request.reference.metadata.sensor_gain = 1.0;
  request.current.metadata.exposure_sec = 0.02;
  request.current.metadata.sensor_gain = 1.0;

  const PhotometricPatchResult result = calibration.compare(request);

  EXPECT_EQ(
    result.status,
    PhotometricStatus::kCalibrationUnavailable);
  EXPECT_FALSE(result.cross_camera_comparable);
  EXPECT_FALSE(result.experimental);
  EXPECT_FALSE(
    request.reference.metadata.hardwareExposureGainReady());
  EXPECT_FALSE(
    request.current.metadata.hardwareExposureGainReady());
}

TEST(
  PhotometricCalibrationTest,
  RobustAffineRecoversLocalTransferAndInflatesCovariance)
{
  RobustAffinePhotometricConfig config;
  config.gain_prior_stddev = 1000.0;
  config.bias_prior_stddev = 1000.0;
  config.maximum_normal_condition_number = 1.0e10;
  PhotometricCalibration calibration(
    {makeCalibration(0U), makeCalibration(1U)}, config);

  const std::vector<double> current{
    20.0, 35.0, 50.0, 65.0, 80.0, 95.0,
    110.0, 125.0, 140.0, 155.0, 170.0, 185.0};
  std::vector<double> reference;
  reference.reserve(current.size());
  for (const double value : current)
  {
    reference.push_back(1.2 * value + 8.0);
  }
  const PhotometricPatchRequest request = makeRequest(
    PhotometricMode::kRobustAffineExperimental,
    reference, current);

  const PhotometricPatchResult result = calibration.compare(request);

  ASSERT_TRUE(result.valid());
  EXPECT_FALSE(result.cross_camera_comparable);
  EXPECT_TRUE(result.experimental);
  EXPECT_EQ(
    result.diagnostic,
    "robust_affine_experimental_non_radiometric");
  EXPECT_NEAR(result.estimated_gain, 1.2, 1.0e-5);
  EXPECT_NEAR(result.estimated_bias, 8.0 / 255.0, 1.0e-5);
  EXPECT_NEAR(result.zncc, 1.0, 1.0e-12);
  EXPECT_LT(result.residual_rmse, 1.0e-5);
  EXPECT_GE(
    result.covariance_scale,
    config.minimum_covariance_scale);
}

TEST(
  PhotometricCalibrationTest,
  RobustAffineOperatesWithoutExposureMetadataButNeedsCalibration)
{
  RobustAffinePhotometricConfig config;
  config.minimum_valid_samples = 3U;
  PhotometricCalibration calibration(
    {makeCalibration(0U), makeCalibration(1U)}, config);
  const PhotometricPatchRequest request = makeRequest(
    PhotometricMode::kRobustAffineExperimental,
    {30.0, 50.0, 70.0, 90.0},
    {30.0, 50.0, 70.0, 90.0});

  const PhotometricPatchResult result = calibration.compare(request);

  EXPECT_TRUE(result.valid());
  EXPECT_FALSE(result.cross_camera_comparable);
  EXPECT_TRUE(result.experimental);
}

TEST(PhotometricCalibrationTest, RobustAffineRejectsDegeneratePatch)
{
  PhotometricCalibration calibration(
    {makeCalibration(0U), makeCalibration(1U)});
  const std::vector<double> constant_values(8U, 80.0);
  const PhotometricPatchRequest request = makeRequest(
    PhotometricMode::kRobustAffineExperimental,
    constant_values, constant_values);

  const PhotometricPatchResult result = calibration.compare(request);

  EXPECT_EQ(result.status, PhotometricStatus::kDegeneratePatch);
  EXPECT_FALSE(result.cross_camera_comparable);
}

TEST(PhotometricCalibrationTest, RobustAffineRejectsLowZncc)
{
  PhotometricCalibration calibration(
    {makeCalibration(0U), makeCalibration(1U)});
  const PhotometricPatchRequest request = makeRequest(
    PhotometricMode::kRobustAffineExperimental,
    {160.0, 140.0, 120.0, 100.0, 80.0, 60.0, 40.0, 20.0},
    {20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0});

  const PhotometricPatchResult result = calibration.compare(request);

  EXPECT_EQ(result.status, PhotometricStatus::kLowZncc);
  EXPECT_FALSE(result.cross_camera_comparable);
}

TEST(PhotometricCalibrationTest, InvalidResponseLutFailsClosed)
{
  PhotometricCameraCalibration invalid = makeCalibration(0U);
  invalid.inverse_response_lut[128U] = -1.0;
  PhotometricCalibration calibration({invalid});

  EXPECT_EQ(
    calibration.cameraCalibrationStatus(0U),
    PhotometricStatus::kCalibrationUnavailable);

  PhotometricPatchRequest request = makeRequest(
    PhotometricMode::kRobustAffineExperimental,
    {20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0},
    {20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0});
  request.current.camera_id = 0U;
  EXPECT_EQ(
    calibration.compare(request).status,
    PhotometricStatus::kInvalidCalibration);
}

TEST(
  PhotometricCalibrationTest,
  SampleLinearizationMatchesRawAndPixelCentralDifferences)
{
  PhotometricCameraCalibration camera = makeCalibration(0U);
  for (int y = 0; y < camera.image_height; ++y)
  {
    for (int x = 0; x < camera.image_width; ++x)
    {
      camera.vignetting_transmission[
        static_cast<std::size_t>(
          y * camera.image_width + x)] =
        0.8 + 0.02 * static_cast<double>(x) +
        0.03 * static_cast<double>(y);
    }
  }
  PhotometricCalibration calibration({camera});
  PhotometricSample sample;
  sample.raw_value = 100.25;
  sample.pixel_x = 1.25;
  sample.pixel_y = 1.40;
  const PhotometricFrameMetadata metadata;

  const PhotometricSampleLinearization nominal =
    calibration.linearizeSample(
      0U, metadata, sample,
      PhotometricMode::kRobustAffineExperimental);

  ASSERT_TRUE(nominal.valid());
  EXPECT_FALSE(nominal.cross_camera_comparable);
  EXPECT_TRUE(nominal.experimental);
  constexpr double kRawEpsilon = 1.0e-5;
  constexpr double kPixelEpsilon = 1.0e-6;
  const auto value_at =
    [&calibration, &metadata, &sample](
      const double raw_delta,
      const double x_delta,
      const double y_delta)
    {
      PhotometricSample perturbed = sample;
      perturbed.raw_value += raw_delta;
      perturbed.pixel_x += x_delta;
      perturbed.pixel_y += y_delta;
      return calibration.linearizeSample(
        0U, metadata, perturbed,
        PhotometricMode::kRobustAffineExperimental).value;
    };
  const double raw_numeric =
    (value_at(kRawEpsilon, 0.0, 0.0) -
     value_at(-kRawEpsilon, 0.0, 0.0)) /
    (2.0 * kRawEpsilon);
  const double x_numeric =
    (value_at(0.0, kPixelEpsilon, 0.0) -
     value_at(0.0, -kPixelEpsilon, 0.0)) /
    (2.0 * kPixelEpsilon);
  const double y_numeric =
    (value_at(0.0, 0.0, kPixelEpsilon) -
     value_at(0.0, 0.0, -kPixelEpsilon)) /
    (2.0 * kPixelEpsilon);

  EXPECT_NEAR(
    nominal.derivative_wrt_raw, raw_numeric, 1.0e-8);
  EXPECT_NEAR(
    nominal.derivative_wrt_pixel_x, x_numeric, 1.0e-8);
  EXPECT_NEAR(
    nominal.derivative_wrt_pixel_y, y_numeric, 1.0e-8);
}

}  // namespace
}  // namespace livo
