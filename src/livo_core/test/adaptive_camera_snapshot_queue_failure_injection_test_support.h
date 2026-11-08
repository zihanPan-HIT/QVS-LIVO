#ifndef LIVO_TEST_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_SUPPORT_H_
#define LIVO_TEST_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_SUPPORT_H_

#include <cstdint>

namespace livo::test
{

void failNextAdaptiveCameraSnapshotQueueFactoryAllocation();
void failNextAdaptiveCameraSnapshotRequestStorageAllocation();
void exhaustNextAdaptiveCameraSnapshotSequence();
void exhaustNextAdaptiveCameraSnapshotObservationGeneration(
  std::uint8_t camera_id);
void exhaustNextAdaptiveCameraSnapshotImageGeneration(
  std::uint8_t camera_id);

void pauseNextAdaptiveCameraSnapshotPushAfterFreeze();
bool waitForAdaptiveCameraSnapshotPushPausedAfterFreeze();
void releaseAdaptiveCameraSnapshotPushAfterFreeze();

void pauseNextAdaptiveCameraSnapshotRequestBeforeRevalidation();
bool waitForAdaptiveCameraSnapshotRequestPausedBeforeRevalidation();
void releaseAdaptiveCameraSnapshotRequestBeforeRevalidation();

}  // namespace livo::test

#endif  // LIVO_TEST_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_SUPPORT_H_
