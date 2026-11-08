#ifndef LIVO_TEST_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_SUPPORT_H_
#define LIVO_TEST_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_SUPPORT_H_

#include "exposure_timing_attestation.h"

#include <cstdint>

namespace livo::test
{

class ExposureTimingCalibrationTestAccess
{
public:
  static TrustedExposureTimingCalibrationResult mintFixture(
    const ExposureTimingCalibrationDocument & document) noexcept;
};

void failNextExposureTimingCalibrationAllocation();
void failNextExposureTimingSourceAuthorityFactoryAllocation();
void failNextExposureFrameTimingReceiptAllocation();
void failNextExposureTimingAuthorityFactoryAllocation();
void failNextExposureTimingAttestationAllocation();
void exhaustNextExposureTimingSourceSequence(
  std::uint8_t camera_id);
void exhaustNextExposureTimingAttestationSequence();

void pauseNextExposureFrameTimingReceiptBeforeRevalidation();
bool waitForExposureFrameTimingReceiptPausedBeforeRevalidation();
void releaseExposureFrameTimingReceiptBeforeRevalidation();

void pauseNextExposureTimingAttestationBeforeRevalidation();
bool waitForExposureTimingAttestationPausedBeforeRevalidation();
void releaseExposureTimingAttestationBeforeRevalidation();

}  // namespace livo::test

#endif  // LIVO_TEST_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_SUPPORT_H_
