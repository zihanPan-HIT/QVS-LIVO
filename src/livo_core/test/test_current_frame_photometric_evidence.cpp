#include "current_frame_photometric_evidence.h"
#include "runtime_frame_provenance_test_support.h"
#include "trusted_photometric_evidence_test_support.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

namespace livo
{
namespace
{

struct CurrentFrameEvidencePipelineConfig
{
  int image_type{CV_8UC1};
  int image_rows{2};
  int image_columns{2};
  cv::Scalar pixel{cv::Scalar::all(37)};
  PhotometricStatus photometric_status{
    PhotometricStatus::kMissingExposureGain};
};

class CurrentFrameEvidencePipeline
{
public:
  explicit CurrentFrameEvidencePipeline(
    const CurrentFrameEvidencePipelineConfig & config = {})
  : config_(config),
    publisher_(queue_.batchAuthorityIdentity())
  {
    valid_ = queue_.setBatchAuthoritySealingEnabled(true);
    for (
      std::uint8_t camera_id = 0U;
      valid_ && camera_id < 4U; ++camera_id)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = 10.0;
      observation.image =
        cv::Mat(
        config.image_rows, config.image_columns,
        config.image_type, config.pixel);
      observation.photometric_status =
        config.photometric_status;
      valid_ =
        queue_.push(std::move(observation)) ==
        CameraObservationPushResult::ACCEPTED;
    }
    valid_ =
      valid_ &&
      queue_.popSynchronized(10.01, 0.01, batch_);
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

  CameraObservationBatch & mutableBatch() noexcept
  {
    return batch_;
  }

  const ReferenceOnlyMigrationRuntimeBundle & runtime() const
  {
    return *runtime_;
  }

  std::optional<ReferenceOnlyMigrationFrameProvenance>
  frame(const std::uint8_t camera_id) const
  {
    return runtime_->frameProvenance(camera_id);
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

  std::optional<ReferenceOnlyMigrationRuntimeBundle>
  publishAgain()
  {
    StatesGroup state;
    state.cov =
      1.0e-3 *
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    return publisher_.publish(
      batch_, state, batch_.timestamp, 0.0,
      CameraTimeTransfer{});
  }

  bool sealNextBatch(const double timestamp_sec = 11.0)
  {
    for (std::uint8_t camera_id = 0U;
         camera_id < 4U; ++camera_id)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = timestamp_sec;
      observation.image =
        cv::Mat(
        config_.image_rows, config_.image_columns,
        config_.image_type, config_.pixel);
      observation.photometric_status =
        config_.photometric_status;
      if (
        queue_.push(std::move(observation)) !=
        CameraObservationPushResult::ACCEPTED)
      {
        return false;
      }
    }
    if (
      !queue_.popSynchronized(
        timestamp_sec + 0.01, 0.01,
        next_batch_))
    {
      return false;
    }
    return true;
  }

  bool publishNextBatch(const double timestamp_sec = 11.0)
  {
    if (!sealNextBatch(timestamp_sec)) {
      return false;
    }
    StatesGroup state;
    state.cov =
      1.0e-3 *
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    next_runtime_ = publisher_.publish(
      next_batch_, state, next_batch_.timestamp, 0.0,
      CameraTimeTransfer{});
    return
      next_runtime_.has_value() &&
      next_runtime_->validatedComposition().issuedBy(
        queue_.batchAuthorityIdentity(),
        publisher_.identity()) &&
      next_runtime_->validatedComposition().accepts(
        *next_runtime_);
  }

  const CameraObservationBatch & nextBatch() const noexcept
  {
    return next_batch_;
  }

  const ReferenceOnlyMigrationRuntimeBundle &
  nextRuntime() const
  {
    return *next_runtime_;
  }

private:
  CurrentFrameEvidencePipelineConfig config_;
  CameraObservationQueue queue_;
  ReferenceOnlyMigrationRuntimePublisher publisher_;
  CameraObservationBatch batch_;
  std::optional<ReferenceOnlyMigrationRuntimeBundle> runtime_;
  CameraObservationBatch next_batch_;
  std::optional<ReferenceOnlyMigrationRuntimeBundle>
    next_runtime_;
  bool valid_{false};
};

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  DefaultOffReturnsUnavailableNonProductionDecision)
{
  test_support::RuntimeFrameProvenanceTestSupport pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_NE(frame, nullptr);

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    CurrentFramePhotometricEvidenceConfig{});
  const CurrentFramePhotometricEvidenceDecision decision =
    evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::kUnavailable);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::kDisabled);
  EXPECT_FALSE(decision.experimental());
  EXPECT_FALSE(decision.crossCameraComparable());
  EXPECT_FALSE(decision.trustedNoiseMatched());
  EXPECT_FALSE(
    CurrentFramePhotometricEvidenceDecision::consumable());
  EXPECT_FALSE(
    CurrentFramePhotometricEvidenceDecision::productionReady());
  EXPECT_FALSE(
    CurrentFramePhotometricEvidenceDecision::solverApplied());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  TrustedBundleHelperBindsRequestedImageDimensions)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle(
    3, 2);
  const TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();

  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  ASSERT_TRUE(calibration.trusted_noise_bundle.has_value());
  for (std::uint8_t camera_id = 0U;
       camera_id < 4U; ++camera_id)
  {
    const TrustedRawIntensityNoiseCapability * const noise =
      calibration.trusted_noise_bundle->noiseCapability(
      camera_id);
    ASSERT_NE(noise, nullptr);
    EXPECT_EQ(noise->domain().image_width, 3);
    EXPECT_EQ(noise->domain().image_height, 2);
  }
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  TrustedBundleHelperRejectsInvalidOrExcessiveDimensions)
{
  EXPECT_THROW(
    {
      test_support::TrustedPhotometricEvidenceTestBundle bundle(
        0, 2);
    },
    std::invalid_argument);
  EXPECT_THROW(
    {
      test_support::TrustedPhotometricEvidenceTestBundle bundle(
        2, -1);
    },
    std::invalid_argument);
  EXPECT_THROW(
    {
      test_support::TrustedPhotometricEvidenceTestBundle bundle(
        1048577, 1);
    },
    std::invalid_argument);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  TrustedBundleConstructionFailureRemovesTemporaryDirectory)
{
  const auto ticks =
    std::chrono::steady_clock::now().
    time_since_epoch().count();
  const std::filesystem::path directory =
    std::filesystem::temp_directory_path() /
    (
      "livo-current-frame-photometric-failure-" +
      std::to_string(ticks));
  ASSERT_FALSE(std::filesystem::exists(directory));

  EXPECT_THROW(
    {
      test_support::TrustedPhotometricEvidenceTestBundle bundle(
        2, 2,
        test_support::
        TrustedPhotometricEvidenceTestFailurePoint::
        kAfterDirectoryCreation,
        directory);
    },
    std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  TrustedBundlePublishFailureRemovesPartiallyWrittenDirectory)
{
  const auto ticks =
    std::chrono::steady_clock::now().
    time_since_epoch().count();
  const std::filesystem::path directory =
    std::filesystem::temp_directory_path() /
    (
      "livo-current-frame-photometric-publish-failure-" +
      std::to_string(ticks));
  ASSERT_FALSE(std::filesystem::exists(directory));

  EXPECT_THROW(
    {
      test_support::TrustedPhotometricEvidenceTestBundle bundle(
        2, 2,
        test_support::
        TrustedPhotometricEvidenceTestFailurePoint::
        kAfterFirstPublishedFile,
        directory);
    },
    std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  TrustedBundleNormalScopeLeavesNoTemporaryDirectory)
{
  const auto ticks =
    std::chrono::steady_clock::now().
    time_since_epoch().count();
  const std::filesystem::path directory =
    std::filesystem::temp_directory_path() /
    (
      "livo-current-frame-photometric-normal-scope-" +
      std::to_string(ticks));
  ASSERT_FALSE(std::filesystem::exists(directory));

  {
    test_support::TrustedPhotometricEvidenceTestBundle bundle(
      2, 2,
      test_support::TrustedPhotometricEvidenceTestFailurePoint::
      kNone,
      directory);
    EXPECT_TRUE(std::filesystem::is_directory(directory));
  }
  EXPECT_FALSE(std::filesystem::exists(directory));
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  ValidFrameWithoutCalibrationBindsIdentityThenFailsClosed)
{
  test_support::RuntimeFrameProvenanceTestSupport pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_NE(frame, nullptr);
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const CurrentFramePhotometricEvidenceDecision decision =
    evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kCalibrationUnavailable);
  EXPECT_EQ(decision.cameraId(), 0U);
  EXPECT_EQ(
    decision.cameraBatchSequence(),
    frame->cameraBatchSequence());
  EXPECT_EQ(
    decision.runtimePublicationSequence(),
    frame->runtimePublicationSequence());
  EXPECT_EQ(
    decision.observationGeneration(),
    frame->observationGeneration());
  EXPECT_EQ(
    decision.imageGeneration(),
    frame->imageGeneration());
  EXPECT_DOUBLE_EQ(
    decision.exposureTimestampSec(),
    frame->exposureTimestampSec());
  EXPECT_EQ(decision.calibrationRevision(), 0U);
  EXPECT_TRUE(decision.observedDomain().valid());
  EXPECT_EQ(
    decision.observedDomain().canonicalization,
    ObservedRawIntensityCanonicalization::kMono8IdentityV1);
  EXPECT_EQ(decision.observedDomain().image_width, 12);
  EXPECT_EQ(decision.observedDomain().image_height, 8);
  EXPECT_FALSE(
    decision.observedDomain().canonical_gray_owner_attested);
  EXPECT_FALSE(
    decision.observedDomain().native_resolution_attested);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  ExactMonoNoiseMatchesButStrictRemainsUnavailable)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const CurrentFramePhotometricEvidenceDecision decision =
    evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::kUnavailable);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kCanonicalGrayAuthorityUnavailable);
  EXPECT_EQ(
    decision.strictRejectionStatus(),
    CurrentFramePhotometricEvidenceStatus::
    kCanonicalGrayAuthorityUnavailable);
  EXPECT_EQ(
    decision.calibrationRevision(),
    calibration.calibration_revision);
  EXPECT_TRUE(decision.trustedNoiseMatched());
  EXPECT_TRUE(
    decision.hasEvidenceGap(
      CurrentFramePhotometricEvidenceGap::
      kCanonicalGrayAuthorityUnavailable));
  EXPECT_TRUE(
    decision.hasEvidenceGap(
      CurrentFramePhotometricEvidenceGap::
      kNativeResolutionAuthorityUnavailable));
  EXPECT_TRUE(
    decision.hasEvidenceGap(
      CurrentFramePhotometricEvidenceGap::
      kHardwareMetadataUnavailable));
  EXPECT_FALSE(
    decision.hasEvidenceGap(
      CurrentFramePhotometricEvidenceGap::
      kHardwareMetadataUnbound));
  EXPECT_FALSE(decision.experimental());
  EXPECT_FALSE(decision.crossCameraComparable());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  RobustAffineRequiresExplicitOptInAndNeverMatchesTrustedNoise)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  const TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;
  config.allow_robust_affine_experiment = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::
    kRobustAffineExperimental);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kRobustAffineExperimentalReady);
  EXPECT_EQ(
    decision.strictRejectionStatus(),
    CurrentFramePhotometricEvidenceStatus::
    kCanonicalGrayAuthorityUnavailable);
  EXPECT_TRUE(decision.experimental());
  EXPECT_FALSE(decision.crossCameraComparable());
  EXPECT_FALSE(decision.trustedNoiseMatched());
  EXPECT_FALSE(
    CurrentFramePhotometricEvidenceDecision::hasLinearization());
  EXPECT_FALSE(
    CurrentFramePhotometricEvidenceDecision::consumable());
  EXPECT_FALSE(
    CurrentFramePhotometricEvidenceDecision::productionReady());
  EXPECT_FALSE(
    CurrentFramePhotometricEvidenceDecision::solverApplied());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  BgrOpenCvCanonicalizationDoesNotAliasSignedRgbDomain)
{
  cv::Mat bgr(
    1, 1, CV_8UC3, cv::Scalar(201, 1, 0));
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  ASSERT_EQ(gray.type(), CV_8UC1);
  EXPECT_EQ(gray.at<std::uint8_t>(0, 0), 23U);
  EXPECT_EQ(
    static_cast<std::uint8_t>(
      std::rint(
        0.299 * 0.0 +
        0.587 * 1.0 +
        0.114 * 201.0)),
    24U);

  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  const TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  CurrentFrameEvidencePipelineConfig pipeline_config;
  pipeline_config.image_type = CV_8UC3;
  pipeline_config.pixel = cv::Scalar(201, 1, 0);
  CurrentFrameEvidencePipeline pipeline(pipeline_config);
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(1U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kRuntimeCanonicalizationMismatch);
  EXPECT_EQ(
    decision.observedDomain().canonicalization,
    ObservedRawIntensityCanonicalization::
    kBgr8OpenCvBt601LumaUint8V1);
  EXPECT_FALSE(decision.trustedNoiseMatched());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  BgrMismatchCanOnlyEnterExplicitRobustExperimentalPath)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  const TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  CurrentFrameEvidencePipelineConfig pipeline_config;
  pipeline_config.image_type = CV_8UC3;
  CurrentFrameEvidencePipeline pipeline(pipeline_config);
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(1U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;
  config.allow_robust_affine_experiment = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::
    kRobustAffineExperimental);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kRobustAffineExperimentalReady);
  EXPECT_EQ(
    decision.strictRejectionStatus(),
    CurrentFramePhotometricEvidenceStatus::
    kRuntimeCanonicalizationMismatch);
  EXPECT_FALSE(decision.trustedNoiseMatched());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  InvalidExpectedAuthorityPrecedesRequestValidation)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    CameraObservationBatchAuthorityIdentity{},
    ReferenceOnlyMigrationRuntimePublisherIdentity{},
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kInvalidAuthorityConfiguration);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  RepeatedBatchCannotMintAnotherRuntimeComposition)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const std::uint64_t validation_hash_count_before =
    pipeline.runtime().cameraBatchReceipt().
    validationImageHashCount();

  EXPECT_FALSE(pipeline.publishAgain().has_value());
  EXPECT_EQ(
    pipeline.runtime().cameraBatchReceipt().
    validationImageHashCount(),
    validation_hash_count_before);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  ForeignBatchAuthorityFailsClosed)
{
  CurrentFrameEvidencePipeline expected;
  CurrentFrameEvidencePipeline foreign;
  ASSERT_TRUE(expected.valid());
  ASSERT_TRUE(foreign.valid());
  const auto foreign_frame = foreign.frame(0U);
  ASSERT_TRUE(foreign_frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    expected.batchAuthority(),
    expected.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      foreign.runtime().validatedComposition(),
      foreign.runtime(), *foreign_frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kBatchAuthorityRejected);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  PublishedCompositionIsStableAfterCallerImageMutation)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  const std::uint64_t validation_hash_count_before =
    pipeline.runtime().cameraBatchReceipt().
    validationImageHashCount();
  pipeline.mutableBatch().
    observations[0U].image.at<std::uint8_t>(0, 0) = 99U;
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kCalibrationUnavailable);
  EXPECT_EQ(
    pipeline.runtime().cameraBatchReceipt().
    validationImageHashCount(),
    validation_hash_count_before);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  FrameCannotBeComposedWithAnotherValidPublication)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto first_frame = pipeline.frame(0U);
  ASSERT_TRUE(first_frame.has_value());
  ASSERT_TRUE(pipeline.publishNextBatch());
  const auto & second_runtime = pipeline.nextRuntime();
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      second_runtime.validatedComposition(),
      second_runtime, *first_frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kFrameCompositionMismatch);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  ForeignRuntimePublisherIsRejectedAfterBatchValidation)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  ASSERT_TRUE(pipeline.sealNextBatch());
  ReferenceOnlyMigrationRuntimePublisher foreign_publisher(
    pipeline.batchAuthority());
  StatesGroup state;
  state.cov =
    1.0e-3 *
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
  const auto foreign_runtime = foreign_publisher.publish(
    pipeline.nextBatch(), state, pipeline.nextBatch().timestamp,
    0.0, CameraTimeTransfer{});
  ASSERT_TRUE(foreign_runtime.has_value());
  const auto foreign_frame =
    foreign_runtime->frameProvenance(0U);
  ASSERT_TRUE(foreign_frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      foreign_runtime->validatedComposition(),
      *foreign_runtime, *foreign_frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kRuntimeAuthorityRejected);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  RuntimeCannotBeComposedWithAnotherOwnerIssuedBatch)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto first_frame = pipeline.frame(0U);
  ASSERT_TRUE(first_frame.has_value());
  ASSERT_TRUE(pipeline.publishNextBatch());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.nextRuntime().validatedComposition(),
      pipeline.runtime(),
      *first_frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kRuntimeCompositionRejected);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  ForeignFramePublisherIsRejectedAfterRuntimeValidation)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  ASSERT_TRUE(pipeline.sealNextBatch());
  ReferenceOnlyMigrationRuntimePublisher foreign_publisher(
    pipeline.batchAuthority());
  StatesGroup state;
  state.cov =
    1.0e-3 *
    Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
  const auto foreign_runtime = foreign_publisher.publish(
    pipeline.nextBatch(), state, pipeline.nextBatch().timestamp,
    0.0, CameraTimeTransfer{});
  ASSERT_TRUE(foreign_runtime.has_value());
  const auto foreign_frame =
    foreign_runtime->frameProvenance(0U);
  ASSERT_TRUE(foreign_frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *foreign_frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kFrameAuthorityRejected);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  UnsupportedImageTypesFailBeforeCalibration)
{
  const std::array<int, 3U> unsupported_types{{
      CV_16UC1, CV_32FC1, CV_8UC4}};
  for (const int image_type : unsupported_types)
  {
    SCOPED_TRACE(image_type);
    CurrentFrameEvidencePipelineConfig pipeline_config;
    pipeline_config.image_type = image_type;
    CurrentFrameEvidencePipeline pipeline(pipeline_config);
    ASSERT_TRUE(pipeline.valid());
    const auto frame = pipeline.frame(0U);
    ASSERT_TRUE(frame.has_value());
    CurrentFramePhotometricEvidenceConfig config;
    config.enabled = true;

    CurrentFramePhotometricEvidenceEvaluator evaluator(
      pipeline.batchAuthority(),
      pipeline.runtimeAuthority(),
      TrustedPhotometricCalibrationLoadResult{},
      config);
    const auto decision = evaluator.evaluate(
      CurrentFramePhotometricEvidenceRequest{
        pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

    EXPECT_EQ(
      decision.status(),
      CurrentFramePhotometricEvidenceStatus::
      kUnsupportedImageDomain);
  }
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  CalibrationRevisionMismatchPrecedesNoiseValidation)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  ++calibration.calibration_revision;
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kCalibrationRevisionMismatch);
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  MissingNoiseCapabilityBlocksRobustWithoutTrustedDomainEvidence)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  calibration.trusted_noise_bundle.reset();
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;
  config.allow_robust_affine_experiment = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::kUnavailable);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kNoiseCapabilityUnavailable);
  EXPECT_EQ(
    decision.strictRejectionStatus(),
    CurrentFramePhotometricEvidenceStatus::
    kNoiseCapabilityUnavailable);
  EXPECT_FALSE(decision.experimental());
  EXPECT_FALSE(decision.trustedNoiseMatched());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  MonoDimensionsMustExactlyMatchTrustedNoiseDomainEvenForRobust)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  const TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  CurrentFrameEvidencePipelineConfig pipeline_config;
  pipeline_config.image_columns = 3;
  CurrentFrameEvidencePipeline pipeline(pipeline_config);
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;
  config.allow_robust_affine_experiment = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::kUnavailable);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kNoiseDomainMismatch);
  EXPECT_EQ(
    decision.strictRejectionStatus(),
    CurrentFramePhotometricEvidenceStatus::
    kNoiseDomainMismatch);
  EXPECT_FALSE(decision.experimental());
  EXPECT_FALSE(decision.trustedNoiseMatched());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  SignedRgbCapabilityDoesNotAliasMonoRuntimeCanonicalization)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  const TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(1U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::kUnavailable);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kRuntimeCanonicalizationMismatch);
  EXPECT_FALSE(decision.trustedNoiseMatched());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  ConstructorOwnsValidatedCopiesOfAuthoritiesAndCalibration)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  const std::uint64_t signed_revision =
    calibration.calibration_revision;
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CameraObservationBatchAuthorityIdentity batch_authority =
    pipeline.batchAuthority();
  ReferenceOnlyMigrationRuntimePublisherIdentity
    runtime_authority = pipeline.runtimeAuthority();
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    batch_authority,
    runtime_authority,
    calibration,
    config);
  batch_authority =
    CameraObservationBatchAuthorityIdentity{};
  runtime_authority =
    ReferenceOnlyMigrationRuntimePublisherIdentity{};
  ++calibration.calibration_revision;
  calibration.trusted_noise_bundle.reset();

  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kCanonicalGrayAuthorityUnavailable);
  EXPECT_EQ(
    decision.calibrationRevision(), signed_revision);
  EXPECT_TRUE(decision.trustedNoiseMatched());
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  PublicPhotometricStatusCannotAuthorizeStrictMode)
{
  test_support::TrustedPhotometricEvidenceTestBundle bundle;
  const TrustedPhotometricCalibrationLoadResult calibration =
    bundle.load();
  ASSERT_TRUE(calibration.trustedNoiseReady())
    << calibration.diagnostic;
  CurrentFrameEvidencePipelineConfig pipeline_config;
  pipeline_config.photometric_status =
    PhotometricStatus::kValid;
  CurrentFrameEvidencePipeline pipeline(pipeline_config);
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(0U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;

  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    calibration,
    config);
  const auto decision = evaluator.evaluate(
    CurrentFramePhotometricEvidenceRequest{
      pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame});

  EXPECT_EQ(
    decision.path(),
    CurrentFramePhotometricEvidencePath::kUnavailable);
  EXPECT_EQ(
    decision.status(),
    CurrentFramePhotometricEvidenceStatus::
    kCanonicalGrayAuthorityUnavailable);
  EXPECT_FALSE(
    decision.hasEvidenceGap(
      CurrentFramePhotometricEvidenceGap::
      kHardwareMetadataUnavailable));
  EXPECT_TRUE(
    decision.hasEvidenceGap(
      CurrentFramePhotometricEvidenceGap::
      kHardwareMetadataUnbound));
}

TEST(
  CurrentFramePhotometricEvidenceEvaluatorTest,
  RepeatedEvaluationIsDeterministicAndDoesNotMintGeneration)
{
  CurrentFrameEvidencePipeline pipeline;
  ASSERT_TRUE(pipeline.valid());
  const auto frame = pipeline.frame(2U);
  ASSERT_TRUE(frame.has_value());
  CurrentFramePhotometricEvidenceConfig config;
  config.enabled = true;
  CurrentFramePhotometricEvidenceEvaluator evaluator(
    pipeline.batchAuthority(),
    pipeline.runtimeAuthority(),
    TrustedPhotometricCalibrationLoadResult{},
    config);
  const CurrentFramePhotometricEvidenceRequest request{
    pipeline.runtime().validatedComposition(),
      pipeline.runtime(), *frame};
  const CameraObservationBatchReceipt & batch_receipt =
    pipeline.runtime().cameraBatchReceipt();
  const std::uint64_t validation_hash_count_before =
    batch_receipt.validationImageHashCount();

  const auto first = evaluator.evaluate(request);
  EXPECT_EQ(
    batch_receipt.validationImageHashCount(),
    validation_hash_count_before);
  const auto second = evaluator.evaluate(request);
  EXPECT_EQ(
    batch_receipt.validationImageHashCount(),
    validation_hash_count_before);

  EXPECT_EQ(first.path(), second.path());
  EXPECT_EQ(first.status(), second.status());
  EXPECT_EQ(
    first.strictRejectionStatus(),
    second.strictRejectionStatus());
  EXPECT_EQ(first.cameraId(), second.cameraId());
  EXPECT_EQ(
    first.cameraBatchSequence(),
    second.cameraBatchSequence());
  EXPECT_EQ(
    first.runtimePublicationSequence(),
    second.runtimePublicationSequence());
  EXPECT_EQ(
    first.observationGeneration(),
    second.observationGeneration());
  EXPECT_EQ(
    first.imageGeneration(),
    second.imageGeneration());
  EXPECT_DOUBLE_EQ(
    first.exposureTimestampSec(),
    second.exposureTimestampSec());
  EXPECT_EQ(
    first.calibrationRevision(),
    second.calibrationRevision());
  EXPECT_EQ(first.observedDomain(), second.observedDomain());
  EXPECT_EQ(first.evidenceGapMask(), second.evidenceGapMask());
  EXPECT_EQ(
    first.trustedNoiseMatched(),
    second.trustedNoiseMatched());
  EXPECT_EQ(first.experimental(), second.experimental());
  EXPECT_EQ(
    first.crossCameraComparable(),
    second.crossCameraComparable());
}

}  // namespace
}  // namespace livo
