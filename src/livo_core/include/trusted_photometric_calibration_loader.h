#ifndef LIVO_TRUSTED_PHOTOMETRIC_CALIBRATION_LOADER_H_
#define LIVO_TRUSTED_PHOTOMETRIC_CALIBRATION_LOADER_H_

#include "photometric_noise_authority.h"
#include "photometric_calibration.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace livo
{

enum class TrustedPhotometricCalibrationLoadStatus
{
  kLoaded,
  kTrustRootUnavailable,
  kInvalidBundlePath,
  kIoFailure,
  kResourceLimitExceeded,
  kInvalidJson,
  kInvalidBundleContract,
  kPayloadHashMismatch,
  kTrustRootMismatch,
  kSignatureInvalid,
  kArtifactInvalid,
  kCalibrationInvalid
};

struct TrustedPhotometricCalibrationLoadResult
{
  TrustedPhotometricCalibrationLoadStatus status{
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract};
  bool structurally_valid{false};
  std::uint64_t calibration_revision{0U};
  std::optional<PhotometricCalibration> calibration;
  std::optional<TrustedPhotometricCalibrationBundle>
    trusted_noise_bundle;
  std::string diagnostic;

  bool trustedCalibrationReady() const
  {
    return
      status == TrustedPhotometricCalibrationLoadStatus::kLoaded &&
      structurally_valid &&
      calibration_revision != 0U &&
      calibration.has_value();
  }

  // Backward-compatible calibration-only readiness.
  bool trustedReady() const
  {
    return trustedCalibrationReady();
  }

  bool trustedNoiseReady() const
  {
    return
      trustedCalibrationReady() &&
      trusted_noise_bundle.has_value() &&
      trusted_noise_bundle->valid() &&
      trusted_noise_bundle->calibrationRevision() ==
        calibration_revision &&
      trusted_noise_bundle->accepts(*calibration);
  }

  std::size_t trustedNoiseCapabilityCount() const
  {
    return trustedNoiseReady() ?
      trusted_noise_bundle->noiseCapabilityCount() : 0U;
  }
};

// The trust root is owned by this translation unit's build target. There is
// intentionally no public constructor, key parameter, filename parameter, or
// mutable global hook through which a caller could authorize its own signer.
class TrustedPhotometricCalibrationLoader final
{
public:
  TrustedPhotometricCalibrationLoader() = delete;

  static TrustedPhotometricCalibrationLoadResult load(
    const std::filesystem::path & bundle_directory);
};

const char * trustedPhotometricCalibrationLoadStatusName(
  TrustedPhotometricCalibrationLoadStatus status);

}  // namespace livo

#endif  // LIVO_TRUSTED_PHOTOMETRIC_CALIBRATION_LOADER_H_
