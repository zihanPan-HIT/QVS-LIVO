#ifndef LIVO_RUNTIME_FRAME_PROVENANCE_TEST_SUPPORT_H_
#define LIVO_RUNTIME_FRAME_PROVENANCE_TEST_SUPPORT_H_

#include "reference_only_migration_runtime_provenance.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <opencv2/core.hpp>

namespace livo::test_support
{

class RuntimeFrameProvenanceTestSupport
{
public:
  explicit RuntimeFrameProvenanceTestSupport(
    const double timestamp_sec = 10.0,
    const std::uint8_t pixel = 37U)
  : publisher_(queue_.batchAuthorityIdentity())
  {
    valid_ = queue_.setBatchAuthoritySealingEnabled(true);
    for (
      std::uint8_t camera_id = 0U;
      valid_ && camera_id < 4U; ++camera_id)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = timestamp_sec;
      observation.image =
        cv::Mat(8, 12, CV_8UC1, cv::Scalar(pixel));
      valid_ =
        queue_.push(std::move(observation)) ==
        CameraObservationPushResult::ACCEPTED;
    }
    valid_ =
      valid_ &&
      queue_.popSynchronized(
        timestamp_sec + 0.01, 0.01, batch_);
    if (!valid_)
    {
      return;
    }

    StatesGroup state;
    state.cov =
      1.0e-3 *
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    runtime_ = publisher_.publish(
      batch_, state, batch_.timestamp, 0.0,
      CameraTimeTransfer{});
    valid_ =
      runtime_.has_value() &&
      runtime_->validatedComposition().issuedBy(
        queue_.batchAuthorityIdentity(),
        publisher_.identity()) &&
      runtime_->validatedComposition().accepts(*runtime_);
  }

  bool valid() const noexcept
  {
    return valid_;
  }

  const CameraObservationBatch & batch() const noexcept
  {
    return batch_;
  }

  const ReferenceOnlyMigrationRuntimeBundle &
  runtime() const
  {
    return *runtime_;
  }

  CameraObservationBatchAuthorityIdentity
  batchAuthority() const noexcept
  {
    return queue_.batchAuthorityIdentity();
  }

  ReferenceOnlyMigrationRuntimePublisherIdentity
  runtimeAuthority() const noexcept
  {
    return publisher_.identity();
  }

  std::shared_ptr<
    const ReferenceOnlyMigrationFrameProvenance>
  frame(const std::uint8_t camera_id) const
  {
    if (!valid_ || !runtime_.has_value())
    {
      return {};
    }
    const auto provenance =
      runtime_->frameProvenance(camera_id);
    if (!provenance.has_value())
    {
      return {};
    }
    return std::make_shared<
      const ReferenceOnlyMigrationFrameProvenance>(
      *provenance);
  }

private:
  CameraObservationQueue queue_;
  ReferenceOnlyMigrationRuntimePublisher publisher_;
  CameraObservationBatch batch_;
  std::optional<ReferenceOnlyMigrationRuntimeBundle>
    runtime_;
  bool valid_{false};
};

}  // namespace livo::test_support

#endif  // LIVO_RUNTIME_FRAME_PROVENANCE_TEST_SUPPORT_H_
