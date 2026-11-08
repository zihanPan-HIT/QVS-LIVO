#ifndef LIVO_TEST_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_SUPPORT_H_
#define LIVO_TEST_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_SUPPORT_H_

#include <cstdint>

namespace livo::test
{

void failNextCameraDriverEnvelopeIdentityAllocation();
void failNextCameraDriverEnvelopeImplementationAllocation();
void failNextCameraDriverEnvelopePayloadAllocation();
void failNextAttestedFrameIngressIdentityAllocation();
void failNextAttestedFrameIngressImplementationAllocation();
void failNextAttestedFramePayloadAllocation();
void failNextCameraDriverEnvelopeDigestComputation();
void exhaustNextCameraDriverEnvelopeSequence(
  std::uint8_t camera_id);
void exhaustNextAttestedFrameIngressSequence(
  std::uint8_t camera_id);
void corruptNextCameraDriverEnvelopeDigest();
void corruptNextCameraDriverEnvelopeMetadata();
void corruptNextCameraDriverEnvelopePixel();

}  // namespace livo::test

#endif  // LIVO_TEST_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_SUPPORT_H_
