#ifndef LIVO_TEST_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_SUPPORT_H_
#define LIVO_TEST_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_SUPPORT_H_

namespace livo::test
{

void failNextCameraTimeTransferFactoryAllocation();
void failNextCameraTimeTransferAppendAllocation();
void failNextCameraTimeTransferClearResource();
void failNextCameraTimeTransferCaptureAllocation();
void failNextCameraTimeTransferNumericalAllocation();
void setNextCameraTimeTransferEpochToMaximum();
void setNextCameraTimeTransferEpochNearMaximum();
void setNextCameraTimeTransferRevisionToMaximum();
void setNextCameraTimeTransferRevisionNearMaximum();
void setNextCameraTimeTransferSnapshotSequenceToMaximum();
void setNextCameraTimeTransferSnapshotSequenceNearMaximum();

}  // namespace livo::test

#endif  // LIVO_TEST_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_SUPPORT_H_
