#ifndef LIVO_PHOTOMETRIC_NOISE_AUTHORITY_H_
#define LIVO_PHOTOMETRIC_NOISE_AUTHORITY_H_

#include "photometric_calibration.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace livo
{

class TrustedPhotometricCalibrationLoader;

enum class RawIntensityUnit
{
  kInvalid,
  kCanonicalGrayCodeValue
};

enum class RawIntensityCanonicalization
{
  kInvalid,
  kMono8IdentityV1,
  kRgb8Bt601LumaUint8V1
};

struct RawIntensityDomain
{
  std::uint8_t bit_depth{0U};
  RawIntensityUnit unit{RawIntensityUnit::kInvalid};
  RawIntensityCanonicalization canonicalization{
    RawIntensityCanonicalization::kInvalid};
  int image_width{0};
  int image_height{0};

  bool valid() const noexcept;
};

bool operator==(
  const RawIntensityDomain & lhs,
  const RawIntensityDomain & rhs) noexcept;

bool operator!=(
  const RawIntensityDomain & lhs,
  const RawIntensityDomain & rhs) noexcept;

// Opaque capability issued only after one signed schema-v2 calibration
// payload has supplied complete measured evidence for all four cameras.
class TrustedRawIntensityNoiseCapability
{
public:
  TrustedRawIntensityNoiseCapability(
    const TrustedRawIntensityNoiseCapability &) = default;
  TrustedRawIntensityNoiseCapability(
    TrustedRawIntensityNoiseCapability &&) = default;
  TrustedRawIntensityNoiseCapability & operator=(
    const TrustedRawIntensityNoiseCapability &) = delete;
  TrustedRawIntensityNoiseCapability & operator=(
    TrustedRawIntensityNoiseCapability &&) = delete;

  bool valid() const noexcept;
  bool validFor(
    std::uint8_t expected_camera_id,
    std::uint64_t expected_calibration_revision,
    const RawIntensityDomain & expected_domain) const noexcept;

  std::uint8_t cameraId() const noexcept;
  std::uint64_t calibrationRevision() const noexcept;
  const RawIntensityDomain & domain() const noexcept;
  double standardDeviationCodeValue() const noexcept;
  const std::string & sourceId() const noexcept;
  const std::string & provenance() const noexcept;
  const std::string & model() const noexcept;
  const std::string & estimatorRevision() const noexcept;
  std::uint64_t sourceSampleCount() const noexcept;

private:
  TrustedRawIntensityNoiseCapability(
    std::uint8_t camera_id,
    std::uint64_t calibration_revision,
    RawIntensityDomain domain,
    double standard_deviation_code_value,
    std::string source_id,
    std::string provenance,
    std::string model,
    std::string estimator_revision,
    std::uint64_t source_sample_count);

  std::uint8_t camera_id_{0U};
  std::uint64_t calibration_revision_{0U};
  RawIntensityDomain domain_;
  double standard_deviation_code_value_{0.0};
  std::string source_id_;
  std::string provenance_;
  std::string model_;
  std::string estimator_revision_;
  std::uint64_t source_sample_count_{0U};

  friend class TrustedPhotometricCalibrationLoader;
};

// Atomic four-camera noise authority. It owns no second calibration copy;
// every capability binds the payload-derived calibration revision.
class TrustedPhotometricCalibrationBundle
{
public:
  TrustedPhotometricCalibrationBundle(
    const TrustedPhotometricCalibrationBundle &) = default;
  TrustedPhotometricCalibrationBundle(
    TrustedPhotometricCalibrationBundle &&) = default;
  TrustedPhotometricCalibrationBundle & operator=(
    const TrustedPhotometricCalibrationBundle &) = delete;
  TrustedPhotometricCalibrationBundle & operator=(
    TrustedPhotometricCalibrationBundle &&) = delete;

  bool valid() const noexcept;
  bool accepts(
    const PhotometricCalibration & calibration) const;
  std::uint64_t calibrationRevision() const noexcept;
  std::size_t noiseCapabilityCount() const noexcept;
  const TrustedRawIntensityNoiseCapability *
  noiseCapability(std::uint8_t camera_id) const noexcept;

private:
  TrustedPhotometricCalibrationBundle(
    std::uint64_t calibration_revision,
    std::vector<TrustedRawIntensityNoiseCapability> capabilities);

  std::uint64_t calibration_revision_{0U};
  std::vector<TrustedRawIntensityNoiseCapability> capabilities_;

  friend class TrustedPhotometricCalibrationLoader;
};

const char * rawIntensityUnitName(
  RawIntensityUnit unit) noexcept;
const char * rawIntensityCanonicalizationName(
  RawIntensityCanonicalization canonicalization) noexcept;

}  // namespace livo

#endif  // LIVO_PHOTOMETRIC_NOISE_AUTHORITY_H_
