#include "photometric_noise_authority.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace livo
{
namespace
{

constexpr std::size_t kTrustedCameraCount = 4U;
constexpr char kMeasuredProvenance[] = "laboratory_measured";
constexpr char kConstantNoiseModel[] = "constant_sigma_v1";
constexpr char kNoiseEstimatorRevision[] = "temporal_mad_p99_v1";

}  // namespace

bool RawIntensityDomain::valid() const noexcept
{
  return
    bit_depth == 8U &&
    unit == RawIntensityUnit::kCanonicalGrayCodeValue &&
    (
      canonicalization ==
        RawIntensityCanonicalization::kMono8IdentityV1 ||
      canonicalization ==
        RawIntensityCanonicalization::kRgb8Bt601LumaUint8V1
    ) &&
    image_width > 0 &&
    image_height > 0;
}

bool operator==(
  const RawIntensityDomain & lhs,
  const RawIntensityDomain & rhs) noexcept
{
  return
    lhs.bit_depth == rhs.bit_depth &&
    lhs.unit == rhs.unit &&
    lhs.canonicalization == rhs.canonicalization &&
    lhs.image_width == rhs.image_width &&
    lhs.image_height == rhs.image_height;
}

bool operator!=(
  const RawIntensityDomain & lhs,
  const RawIntensityDomain & rhs) noexcept
{
  return !(lhs == rhs);
}

TrustedRawIntensityNoiseCapability::
TrustedRawIntensityNoiseCapability(
  const std::uint8_t camera_id,
  const std::uint64_t calibration_revision,
  RawIntensityDomain domain,
  const double standard_deviation_code_value,
  std::string source_id,
  std::string provenance,
  std::string model,
  std::string estimator_revision,
  const std::uint64_t source_sample_count)
: camera_id_(camera_id),
  calibration_revision_(calibration_revision),
  domain_(domain),
  standard_deviation_code_value_(
    standard_deviation_code_value),
  source_id_(std::move(source_id)),
  provenance_(std::move(provenance)),
  model_(std::move(model)),
  estimator_revision_(std::move(estimator_revision)),
  source_sample_count_(source_sample_count)
{
}

bool TrustedRawIntensityNoiseCapability::valid() const noexcept
{
  return
    camera_id_ < kTrustedCameraCount &&
    calibration_revision_ != 0U &&
    domain_.valid() &&
    std::isfinite(standard_deviation_code_value_) &&
    standard_deviation_code_value_ > 0.0 &&
    !source_id_.empty() &&
    provenance_ == kMeasuredProvenance &&
    model_ == kConstantNoiseModel &&
    estimator_revision_ == kNoiseEstimatorRevision &&
    source_sample_count_ > 0U;
}

bool TrustedRawIntensityNoiseCapability::validFor(
  const std::uint8_t expected_camera_id,
  const std::uint64_t expected_calibration_revision,
  const RawIntensityDomain & expected_domain) const noexcept
{
  return
    valid() &&
    camera_id_ == expected_camera_id &&
    calibration_revision_ == expected_calibration_revision &&
    domain_ == expected_domain;
}

std::uint8_t
TrustedRawIntensityNoiseCapability::cameraId() const noexcept
{
  return camera_id_;
}

std::uint64_t
TrustedRawIntensityNoiseCapability::calibrationRevision() const noexcept
{
  return calibration_revision_;
}

const RawIntensityDomain &
TrustedRawIntensityNoiseCapability::domain() const noexcept
{
  return domain_;
}

double TrustedRawIntensityNoiseCapability::
standardDeviationCodeValue() const noexcept
{
  return standard_deviation_code_value_;
}

const std::string &
TrustedRawIntensityNoiseCapability::sourceId() const noexcept
{
  return source_id_;
}

const std::string &
TrustedRawIntensityNoiseCapability::provenance() const noexcept
{
  return provenance_;
}

const std::string &
TrustedRawIntensityNoiseCapability::model() const noexcept
{
  return model_;
}

const std::string &
TrustedRawIntensityNoiseCapability::estimatorRevision() const noexcept
{
  return estimator_revision_;
}

std::uint64_t TrustedRawIntensityNoiseCapability::
sourceSampleCount() const noexcept
{
  return source_sample_count_;
}

TrustedPhotometricCalibrationBundle::
TrustedPhotometricCalibrationBundle(
  const std::uint64_t calibration_revision,
  std::vector<TrustedRawIntensityNoiseCapability> capabilities)
: calibration_revision_(calibration_revision),
  capabilities_(std::move(capabilities))
{
}

bool TrustedPhotometricCalibrationBundle::valid() const noexcept
{
  if (
    calibration_revision_ == 0U ||
    capabilities_.size() != kTrustedCameraCount)
  {
    return false;
  }
  for (std::size_t camera_id = 0U;
       camera_id < capabilities_.size(); ++camera_id)
  {
    const TrustedRawIntensityNoiseCapability & capability =
      capabilities_[camera_id];
    if (
      !capability.valid() ||
      capability.cameraId() != camera_id ||
      capability.calibrationRevision() !=
        calibration_revision_)
    {
      return false;
    }
  }
  return true;
}

bool TrustedPhotometricCalibrationBundle::accepts(
  const PhotometricCalibration & calibration) const
{
  if (!valid())
  {
    return false;
  }
  for (std::size_t camera_id = 0U;
       camera_id < capabilities_.size(); ++camera_id)
  {
    if (
      calibration.cameraCalibrationStatus(camera_id) !=
        PhotometricStatus::kValid ||
      calibration.cameraCalibrationRevision(camera_id) !=
        calibration_revision_)
    {
      return false;
    }
  }
  return true;
}

std::uint64_t TrustedPhotometricCalibrationBundle::
calibrationRevision() const noexcept
{
  return calibration_revision_;
}

std::size_t TrustedPhotometricCalibrationBundle::
noiseCapabilityCount() const noexcept
{
  return valid() ? capabilities_.size() : 0U;
}

const TrustedRawIntensityNoiseCapability *
TrustedPhotometricCalibrationBundle::noiseCapability(
  const std::uint8_t camera_id) const noexcept
{
  if (
    !valid() ||
    camera_id >= capabilities_.size())
  {
    return nullptr;
  }
  return &capabilities_[camera_id];
}

const char * rawIntensityUnitName(
  const RawIntensityUnit unit) noexcept
{
  switch (unit)
  {
    case RawIntensityUnit::kInvalid:
      return "invalid";
    case RawIntensityUnit::kCanonicalGrayCodeValue:
      return "canonical_gray_code_value";
  }
  return "invalid";
}

const char * rawIntensityCanonicalizationName(
  const RawIntensityCanonicalization canonicalization) noexcept
{
  switch (canonicalization)
  {
    case RawIntensityCanonicalization::kInvalid:
      return "invalid";
    case RawIntensityCanonicalization::kMono8IdentityV1:
      return "mono8_identity_v1";
    case RawIntensityCanonicalization::kRgb8Bt601LumaUint8V1:
      return "rgb8_bt601_luma_uint8_v1";
  }
  return "invalid";
}

}  // namespace livo
