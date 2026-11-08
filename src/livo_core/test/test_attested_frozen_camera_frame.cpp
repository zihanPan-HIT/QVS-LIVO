#include "attested_frozen_camera_frame.h"
#include "attested_frozen_camera_frame_failure_injection_test_support.h"
#include "exposure_timing_attestation_failure_injection_test_support.h"
#include "frozen_camera_image_failure_injection_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace livo
{
namespace
{

CameraDriverFrameEnvelopeAuthorityConfiguration
validEnvelopeConfiguration()
{
  CameraDriverFrameEnvelopeAuthorityConfiguration configuration;
  for (std::size_t camera_id = 0U;
    camera_id < configuration.routes.size();
    ++camera_id)
  {
    auto & route = configuration.routes[camera_id];
    route.source_binding.camera_id =
      static_cast<std::uint8_t>(camera_id);
    route.source_binding.source_id =
      "driver-source-" + std::to_string(camera_id);
    route.source_binding.topic =
      "/camera/" + std::to_string(camera_id) + "/image";
    route.source_binding.sensor_serial =
      "serial-" + std::to_string(camera_id);
    route.source_binding.frame_id =
      "camera_" + std::to_string(camera_id);
    route.driver_epoch = 17U + camera_id;
    route.clock_domain =
      CameraDriverClockDomain::SENSOR_HARDWARE;
    route.hardware_frame_counter_policy =
      CameraDriverHardwareFrameCounterPolicy::
      REQUIRED_MONOTONIC;
    route.width = 4U;
    route.height = 3U;
    route.encoding = "mono8";
    route.is_bigendian = false;
    route.step = 4U;
    route.metadata_revision =
      "metadata-r" + std::to_string(camera_id + 1U);
    route.transform_revision =
      "transform-r" + std::to_string(camera_id + 1U);
  }
  return configuration;
}

CameraDriverFrameDocument validFrameDocument(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &
    configuration,
  const std::uint8_t camera_id,
  const std::uint64_t frame_number,
  const double raw_header_timestamp_sec)
{
  const auto & route = configuration.routes[camera_id];
  CameraDriverFrameDocument document;
  document.source_binding = route.source_binding;
  document.driver_epoch = route.driver_epoch;
  document.driver_frame_number = frame_number;
  document.hardware_frame_counter_policy =
    route.hardware_frame_counter_policy;
  if (
    route.hardware_frame_counter_policy ==
    CameraDriverHardwareFrameCounterPolicy::
    REQUIRED_MONOTONIC)
  {
    document.hardware_frame_counter = 1000U + frame_number;
  }
  document.clock_domain = route.clock_domain;
  document.raw_header_timestamp_sec =
    raw_header_timestamp_sec;
  document.sensor_timestamp_sec =
    raw_header_timestamp_sec - 0.001;
  document.exposure_duration_sec = 0.004;
  document.width = route.width;
  document.height = route.height;
  document.encoding = route.encoding;
  document.is_bigendian = route.is_bigendian;
  document.step = route.step;
  document.data = std::vector<unsigned char>(
    static_cast<std::size_t>(route.step) * route.height,
    static_cast<unsigned char>(20U + camera_id));
  document.metadata_revision = route.metadata_revision;
  document.transform_revision = route.transform_revision;
  return document;
}

ExposureTimingCalibrationDocument timingDocument(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &
    configuration)
{
  ExposureTimingCalibrationDocument document;
  document.mode = ExposureTimingAuthorityMode::SIMULATION_TEST;
  document.provenance =
    ExposureTimingProvenanceKind::TEST_FIXTURE;
  document.canonical_payload =
    "{\"fixture\":\"attested-camera-frame\"}";
  document.canonical_payload_sha256.fill(0xB7U);
  document.artifact_id = "attested-frame-fixture-only";
  document.calibration_revision = 11U;
  document.global_offset_sec = 0.010;
  for (std::size_t camera_id = 0U;
    camera_id < document.cameras.size();
    ++camera_id)
  {
    auto & camera = document.cameras[camera_id];
    camera.source_binding =
      configuration.routes[camera_id].source_binding;
    camera.timestamp_convention =
      ExposureTimestampConvention::MIDPOINT;
    camera.per_camera_offset_sec =
      static_cast<double>(camera_id) * 0.001;
    camera.duration_source = "driver-envelope";
  }
  return document;
}

struct TimingRig final
{
  std::unique_ptr<ExposureTimingSourceAuthority> source;
  std::unique_ptr<ExposureTimingAuthority> timing;
};

std::optional<TimingRig> makeTimingRig(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &
    configuration,
  const ExposureTimingCalibrationDocument & document)
{
  auto calibration =
    test::ExposureTimingCalibrationTestAccess::
    mintFixture(document);
  if (!calibration.ready())
  {
    return std::nullopt;
  }
  ExposureTimingSourceAuthorityConfiguration
    source_configuration;
  for (std::size_t camera_id = 0U;
    camera_id < source_configuration.cameras.size();
    ++camera_id)
  {
    source_configuration.cameras[camera_id] =
      configuration.routes[camera_id].source_binding;
  }
  auto source =
    ExposureTimingSourceAuthority::create(
    source_configuration);
  if (!source.ready())
  {
    return std::nullopt;
  }
  auto timing = ExposureTimingAuthority::create(
    *calibration.calibration,
    source.authority->identity());
  if (!timing.ready())
  {
    return std::nullopt;
  }
  return TimingRig{
    std::move(source.authority),
    std::move(timing.authority)};
}

std::optional<TimingRig> makeTimingRig(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &
    configuration)
{
  return makeTimingRig(
    configuration, timingDocument(configuration));
}

TEST(
  AttestedFrozenCameraFrameTest,
  FixtureEnvelopeAuthorityBindsFourRouteContract)
{
  auto result = CameraDriverFrameEnvelopeAuthority::create(
    validEnvelopeConfiguration());

  ASSERT_TRUE(result.ready());
  ASSERT_TRUE(result.authority);
  EXPECT_TRUE(result.authority->identity().valid());
}

TEST(
  AttestedFrozenCameraFrameTest,
  EnvelopeOwnsOneCanonicalDocumentAndItsBytes)
{
  const auto configuration = validEnvelopeConfiguration();
  auto authority_result =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(authority_result.ready());
  const auto identity = authority_result.authority->identity();
  auto document =
    validFrameDocument(configuration, 2U, 41U, 120.25);

  auto issue_result =
    authority_result.authority->issue(document);

  ASSERT_TRUE(issue_result.ready());
  ASSERT_TRUE(issue_result.envelope.has_value());
  EXPECT_TRUE(issue_result.envelope->valid());
  EXPECT_TRUE(issue_result.envelope->issuedBy(identity));
  const auto audit_before = issue_result.envelope->audit();
  ASSERT_TRUE(audit_before.has_value());
  EXPECT_EQ(2U, audit_before->cameraId());
  EXPECT_EQ(1U, audit_before->envelopeSequence());
  EXPECT_EQ(41U, audit_before->driverFrameNumber());
  EXPECT_EQ(
    CameraDriverHardwareFrameCounterPolicy::
    REQUIRED_MONOTONIC,
    audit_before->hardwareFrameCounterPolicy());
  EXPECT_EQ(12U, audit_before->dataSize());
  const auto digest_before = audit_before->canonicalDigest();
  EXPECT_NE(
    CameraDriverFrameEnvelopeDigest{},
    digest_before);

  document.data.assign(document.data.size(), 255U);
  document.metadata_revision = "mutated";
  const auto audit_after = issue_result.envelope->audit();
  ASSERT_TRUE(audit_after.has_value());
  EXPECT_EQ(12U, audit_after->dataSize());
  EXPECT_EQ(
    configuration.routes[2U].metadata_revision,
    audit_after->metadataRevision());
  EXPECT_EQ(digest_before, audit_after->canonicalDigest());
}

TEST(
  AttestedFrozenCameraFrameTest,
  IngestProducesOneImageTimingFrameCapability)
{
  const auto envelope_configuration =
    validEnvelopeConfiguration();
  auto envelope_authority =
    CameraDriverFrameEnvelopeAuthority::create(
    envelope_configuration);
  ASSERT_TRUE(envelope_authority.ready());
  auto timing_rig = makeTimingRig(envelope_configuration);
  ASSERT_TRUE(timing_rig.has_value());
  AttestedCameraFrameIngressConfiguration
    ingress_configuration;
  ingress_configuration.routes =
    envelope_configuration.routes;

  auto ingress = AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(timing_rig->source),
    std::move(timing_rig->timing));
  ASSERT_TRUE(ingress.ready());
  const auto ingress_identity = ingress.ingress->identity();
  auto document =
    validFrameDocument(
    envelope_configuration, 3U, 80U, 70.0);
  auto envelope =
    envelope_authority.authority->issue(document);
  ASSERT_TRUE(envelope.ready());

  auto result =
    ingress.ingress->ingest(*envelope.envelope);

  ASSERT_TRUE(result.ready());
  ASSERT_TRUE(result.frame.has_value());
  EXPECT_TRUE(result.frame->valid());
  EXPECT_TRUE(result.frame->issuedBy(ingress_identity));
  EXPECT_TRUE(result.frame->sameFrame(*envelope.envelope));
  const auto audit = result.frame->audit();
  ASSERT_TRUE(audit.has_value());
  EXPECT_EQ(3U, audit->cameraId());
  EXPECT_EQ(1U, audit->ingressSequence());
  EXPECT_EQ(3, audit->imageRows());
  EXPECT_EQ(4, audit->imageColumns());
  EXPECT_NE(
    FrozenCameraImage::Digest{},
    audit->imageDigest());
  ASSERT_TRUE(audit->effectiveExposureTime().has_value());
  EXPECT_DOUBLE_EQ(
    70.013,
    audit->effectiveExposureTime()->seconds());
  ASSERT_TRUE(audit->envelope().has_value());
  EXPECT_EQ(
    80U,
    audit->envelope()->driverFrameNumber());
  ASSERT_TRUE(audit->timing().has_value());
  EXPECT_EQ(
    3U,
    audit->timing()->cameraId());
}

TEST(
  AttestedFrozenCameraFrameTest,
  EnvelopeRejectsDriverAndHardwareCounterReplayAtomically)
{
  const auto configuration = validEnvelopeConfiguration();
  auto authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(authority.ready());
  auto first =
    validFrameDocument(configuration, 1U, 10U, 10.0);
  first.hardware_frame_counter = 500U;
  ASSERT_TRUE(authority.authority->issue(first).ready());

  auto replay = first;
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::
    STALE_OR_REPLAYED_DRIVER_FRAME,
    authority.authority->issue(replay).status);

  auto hardware_replay =
    validFrameDocument(configuration, 1U, 11U, 11.0);
  hardware_replay.hardware_frame_counter = 500U;
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::
    STALE_OR_REPLAYED_HARDWARE_COUNTER,
    authority.authority->issue(hardware_replay).status);

  hardware_replay.hardware_frame_counter = 501U;
  const auto retry =
    authority.authority->issue(hardware_replay);
  ASSERT_TRUE(retry.ready());
  ASSERT_TRUE(retry.envelope->audit().has_value());
  EXPECT_EQ(2U, retry.envelope->audit()->envelopeSequence());

  auto missing_counter =
    validFrameDocument(configuration, 1U, 12U, 12.0);
  missing_counter.hardware_frame_counter.reset();
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::
    MISSING_REQUIRED_HARDWARE_FRAME_COUNTER,
    authority.authority->issue(missing_counter).status);

  auto policy_mismatch = missing_counter;
  policy_mismatch.hardware_frame_counter_policy =
    CameraDriverHardwareFrameCounterPolicy::UNAVAILABLE;
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::
    FOREIGN_HARDWARE_FRAME_COUNTER_POLICY,
    authority.authority->issue(policy_mismatch).status);

  auto unavailable_configuration = configuration;
  unavailable_configuration.routes[1U]
  .hardware_frame_counter_policy =
    CameraDriverHardwareFrameCounterPolicy::UNAVAILABLE;
  auto unavailable_authority =
    CameraDriverFrameEnvelopeAuthority::create(
    unavailable_configuration);
  ASSERT_TRUE(unavailable_authority.ready());
  auto without_counter = validFrameDocument(
    unavailable_configuration, 1U, 1U, 20.0);
  ASSERT_FALSE(without_counter.hardware_frame_counter.has_value());
  const auto unavailable_first =
    unavailable_authority.authority->issue(without_counter);
  ASSERT_TRUE(unavailable_first.ready());
  ASSERT_TRUE(unavailable_first.envelope->audit().has_value());
  EXPECT_EQ(
    CameraDriverHardwareFrameCounterPolicy::UNAVAILABLE,
    unavailable_first.envelope->audit()->
    hardwareFrameCounterPolicy());
  EXPECT_FALSE(
    unavailable_first.envelope->audit()->
    hardwareFrameCounter().has_value());

  auto unexpected_counter = validFrameDocument(
    unavailable_configuration, 1U, 2U, 21.0);
  unexpected_counter.hardware_frame_counter = 901U;
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::
    UNEXPECTED_HARDWARE_FRAME_COUNTER,
    unavailable_authority.authority->issue(
      unexpected_counter).status);
  const auto unavailable_second =
    unavailable_authority.authority->issue(
    validFrameDocument(
      unavailable_configuration, 1U, 2U, 21.0));
  ASSERT_TRUE(unavailable_second.ready());
  EXPECT_EQ(
    2U,
    unavailable_second.envelope->audit()->envelopeSequence());
}

TEST(
  AttestedFrozenCameraFrameTest,
  IngressCreationRejectsForeignTimingAndRouteContracts)
{
  const auto configuration = validEnvelopeConfiguration();
  auto envelope_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(envelope_authority.ready());

  auto first_timing = makeTimingRig(configuration);
  auto second_timing = makeTimingRig(configuration);
  ASSERT_TRUE(first_timing.has_value());
  ASSERT_TRUE(second_timing.has_value());
  AttestedCameraFrameIngressConfiguration ingress_configuration;
  ingress_configuration.routes = configuration.routes;
  auto foreign_pair = AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(first_timing->source),
    std::move(second_timing->timing));
  EXPECT_EQ(
    AttestedCameraFrameIngressCreateStatus::
    TIMING_SOURCE_AUTHORITY_MISMATCH,
    foreign_pair.status);
  EXPECT_FALSE(foreign_pair.ingress);

  auto mismatched_configuration = configuration;
  mismatched_configuration.routes[1U].metadata_revision =
    "different-metadata";
  auto route_timing = makeTimingRig(configuration);
  ASSERT_TRUE(route_timing.has_value());
  AttestedCameraFrameIngressConfiguration mismatched_ingress;
  mismatched_ingress.routes =
    mismatched_configuration.routes;
  auto route_mismatch = AttestedCameraFrameIngress::create(
    mismatched_ingress,
    envelope_authority.authority->identity(),
    std::move(route_timing->source),
    std::move(route_timing->timing));
  EXPECT_EQ(
    AttestedCameraFrameIngressCreateStatus::
    ENVELOPE_CONTRACT_MISMATCH,
    route_mismatch.status);

  auto foreign_source_configuration = configuration;
  foreign_source_configuration.routes[2U]
  .source_binding.sensor_serial = "other-serial";
  auto foreign_source_timing =
    makeTimingRig(foreign_source_configuration);
  ASSERT_TRUE(foreign_source_timing.has_value());
  auto source_contract_mismatch =
    AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(foreign_source_timing->source),
    std::move(foreign_source_timing->timing));
  EXPECT_EQ(
    AttestedCameraFrameIngressCreateStatus::
    TIMING_SOURCE_CONTRACT_MISMATCH,
    source_contract_mismatch.status);
}

TEST(
  AttestedFrozenCameraFrameTest,
  EnvelopeAuthorityRejectsNonCanonicalRouteContracts)
{
  const auto expect_invalid =
    [](const CameraDriverFrameEnvelopeAuthorityConfiguration &
      configuration)
    {
      const auto result =
        CameraDriverFrameEnvelopeAuthority::create(configuration);
      EXPECT_EQ(
        CameraDriverFrameEnvelopeAuthorityCreateStatus::
        INVALID_CONFIGURATION,
        result.status);
      EXPECT_FALSE(result.authority);
    };

  expect_invalid(
    CameraDriverFrameEnvelopeAuthorityConfiguration{});
  auto configuration = validEnvelopeConfiguration();
  configuration.routes[1U].source_binding.camera_id = 0U;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[1U].source_binding.sensor_serial =
    configuration.routes[0U].source_binding.sensor_serial;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].source_binding.source_id =
    std::string(
    kAttestedCameraFrameIdentifierMaxLength + 1U, 'a');
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].source_binding.topic =
    "/" + std::string(
    kAttestedCameraFrameTopicMaxLength, 'a');
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].metadata_revision =
    "metadata revision";
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].driver_epoch = 0U;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].clock_domain =
    CameraDriverClockDomain::UNKNOWN;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].hardware_frame_counter_policy =
    CameraDriverHardwareFrameCounterPolicy::UNKNOWN;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].width = 0U;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].width =
    kAttestedCameraFrameMaximumImageDimension + 1U;
  configuration.routes[0U].step =
    configuration.routes[0U].width;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].width = 8193U;
  configuration.routes[0U].height = 8192U;
  configuration.routes[0U].step = 8193U;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].encoding = "rgb8";
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].is_bigendian = true;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].step = 5U;
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].metadata_revision.clear();
  expect_invalid(configuration);
  configuration = validEnvelopeConfiguration();
  configuration.routes[0U].transform_revision.clear();
  expect_invalid(configuration);
}

TEST(
  AttestedFrozenCameraFrameTest,
  EnvelopeIssueRejectsInvalidDocumentWithoutSequenceAdvance)
{
  const auto configuration = validEnvelopeConfiguration();
  auto authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(authority.ready());
  const auto base =
    validFrameDocument(configuration, 0U, 1U, 10.0);
  const auto expect_status =
    [&authority](
      const CameraDriverFrameDocument & document,
      const CameraDriverFrameEnvelopeIssueStatus status)
    {
      const auto result = authority.authority->issue(document);
      EXPECT_EQ(status, result.status);
      EXPECT_FALSE(result.envelope.has_value());
    };

  auto invalid = base;
  invalid.source_binding.camera_id = 4U;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::INVALID_CAMERA_ID);
  invalid = base;
  invalid.source_binding.sensor_serial = "foreign";
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    FOREIGN_SOURCE_BINDING);
  invalid = base;
  invalid.driver_epoch += 1U;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    FOREIGN_DRIVER_EPOCH);
  invalid = base;
  invalid.driver_frame_number = 0U;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_DRIVER_FRAME_NUMBER);
  invalid = base;
  invalid.driver_frame_number =
    std::numeric_limits<std::uint64_t>::max();
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_DRIVER_FRAME_NUMBER);
  invalid = base;
  invalid.hardware_frame_counter_policy =
    CameraDriverHardwareFrameCounterPolicy::UNAVAILABLE;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    FOREIGN_HARDWARE_FRAME_COUNTER_POLICY);
  invalid = base;
  invalid.hardware_frame_counter.reset();
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    MISSING_REQUIRED_HARDWARE_FRAME_COUNTER);
  invalid = base;
  invalid.hardware_frame_counter = 0U;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_HARDWARE_FRAME_COUNTER);
  invalid = base;
  invalid.hardware_frame_counter =
    std::numeric_limits<std::uint64_t>::max();
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_HARDWARE_FRAME_COUNTER);
  invalid = base;
  invalid.clock_domain = CameraDriverClockDomain::UNKNOWN;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_CLOCK_DOMAIN);
  invalid = base;
  invalid.raw_header_timestamp_sec =
    std::numeric_limits<double>::quiet_NaN();
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::INVALID_TIMESTAMP);
  invalid = base;
  invalid.sensor_timestamp_sec = 0.0;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::INVALID_TIMESTAMP);
  invalid = base;
  invalid.exposure_duration_sec = 0.0;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_EXPOSURE_DURATION);
  invalid = base;
  invalid.width += 1U;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_IMAGE_CONTRACT);
  invalid = base;
  invalid.encoding = "rgb8";
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_IMAGE_CONTRACT);
  invalid = base;
  invalid.step += 1U;
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_IMAGE_CONTRACT);
  invalid = base;
  invalid.data.pop_back();
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_FRAME_DATA);
  invalid = base;
  invalid.metadata_revision = "foreign";
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_METADATA_REVISION);
  invalid = base;
  invalid.transform_revision = "foreign";
  expect_status(
    invalid,
    CameraDriverFrameEnvelopeIssueStatus::
    INVALID_TRANSFORM_REVISION);

  const auto first = authority.authority->issue(base);
  ASSERT_TRUE(first.ready());
  ASSERT_TRUE(first.envelope->audit().has_value());
  EXPECT_EQ(1U, first.envelope->audit()->envelopeSequence());
}

TEST(
  AttestedFrozenCameraFrameTest,
  ResourceFailuresRemainTypedAtomicAndOrdered)
{
  const auto configuration = validEnvelopeConfiguration();

  test::failNextCameraDriverEnvelopeIdentityAllocation();
  EXPECT_EQ(
    CameraDriverFrameEnvelopeAuthorityCreateStatus::
    RESOURCE_FAILURE,
    CameraDriverFrameEnvelopeAuthority::create(
      configuration).status);
  test::failNextCameraDriverEnvelopeImplementationAllocation();
  EXPECT_EQ(
    CameraDriverFrameEnvelopeAuthorityCreateStatus::
    RESOURCE_FAILURE,
    CameraDriverFrameEnvelopeAuthority::create(
      configuration).status);

  auto envelope_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(envelope_authority.ready());
  auto document =
    validFrameDocument(configuration, 0U, 1U, 100.0);
  test::failNextCameraDriverEnvelopePayloadAllocation();
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::RESOURCE_FAILURE,
    envelope_authority.authority->issue(document).status);
  auto envelope =
    envelope_authority.authority->issue(document);
  ASSERT_TRUE(envelope.ready());
  ASSERT_TRUE(envelope.envelope->audit().has_value());
  EXPECT_EQ(
    1U,
    envelope.envelope->audit()->envelopeSequence());

  auto exhausted_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(exhausted_authority.ready());
  test::exhaustNextCameraDriverEnvelopeSequence(0U);
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::
    AUTHORITY_EXHAUSTED,
    exhausted_authority.authority->issue(document).status);
  auto other_route_document =
    validFrameDocument(configuration, 1U, 1U, 200.0);
  const auto other_route =
    exhausted_authority.authority->issue(
    other_route_document);
  ASSERT_TRUE(other_route.ready());
  ASSERT_TRUE(other_route.envelope->audit().has_value());
  EXPECT_EQ(
    1U,
    other_route.envelope->audit()->envelopeSequence());

  AttestedCameraFrameIngressConfiguration
    ingress_configuration;
  ingress_configuration.routes = configuration.routes;
  auto timing_for_identity = makeTimingRig(configuration);
  ASSERT_TRUE(timing_for_identity.has_value());
  test::failNextAttestedFrameIngressIdentityAllocation();
  EXPECT_EQ(
    AttestedCameraFrameIngressCreateStatus::RESOURCE_FAILURE,
    AttestedCameraFrameIngress::create(
      ingress_configuration,
      envelope_authority.authority->identity(),
      std::move(timing_for_identity->source),
      std::move(timing_for_identity->timing)).status);
  auto timing_for_implementation = makeTimingRig(configuration);
  ASSERT_TRUE(timing_for_implementation.has_value());
  test::failNextAttestedFrameIngressImplementationAllocation();
  EXPECT_EQ(
    AttestedCameraFrameIngressCreateStatus::RESOURCE_FAILURE,
    AttestedCameraFrameIngress::create(
      ingress_configuration,
      envelope_authority.authority->identity(),
      std::move(timing_for_implementation->source),
      std::move(timing_for_implementation->timing)).status);

  auto timing = makeTimingRig(configuration);
  ASSERT_TRUE(timing.has_value());
  auto ingress = AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(timing->source),
    std::move(timing->timing));
  ASSERT_TRUE(ingress.ready());

  test::failNextAttestedFramePayloadAllocation();
  test::failNextFrozenCameraImageStorageAllocation();
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::RESOURCE_FAILURE,
    ingress.ingress->ingest(*envelope.envelope).status);
  const auto freeze_failure =
    ingress.ingress->ingest(*envelope.envelope);
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::IMAGE_FREEZE_FAILURE,
    freeze_failure.status);
  ASSERT_TRUE(freeze_failure.image_freeze_status.has_value());
  EXPECT_EQ(
    FrozenCameraImageFreezeStatus::RESOURCE_FAILURE,
    *freeze_failure.image_freeze_status);
  const auto first_ingest =
    ingress.ingress->ingest(*envelope.envelope);
  ASSERT_TRUE(first_ingest.ready());
  ASSERT_TRUE(first_ingest.frame->audit().has_value());
  EXPECT_EQ(
    1U,
    first_ingest.frame->audit()->ingressSequence());
  ASSERT_TRUE(first_ingest.frame->audit()->timing().has_value());
  EXPECT_EQ(
    1U,
    first_ingest.frame->audit()->timing()->sourceSequence());

  document =
    validFrameDocument(configuration, 0U, 2U, 101.0);
  auto second_envelope =
    envelope_authority.authority->issue(document);
  ASSERT_TRUE(second_envelope.ready());
  test::failNextExposureFrameTimingReceiptAllocation();
  const auto issue_failure =
    ingress.ingress->ingest(*second_envelope.envelope);
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::TIMING_ISSUE_FAILURE,
    issue_failure.status);
  ASSERT_TRUE(issue_failure.timing_issue_status.has_value());
  EXPECT_EQ(
    ExposureFrameTimingIssueStatus::RESOURCE_FAILURE,
    *issue_failure.timing_issue_status);
  const auto second_ingest =
    ingress.ingress->ingest(*second_envelope.envelope);
  ASSERT_TRUE(second_ingest.ready());
  EXPECT_EQ(
    2U,
    second_ingest.frame->audit()->ingressSequence());

  document =
    validFrameDocument(configuration, 0U, 3U, 102.0);
  auto third_envelope =
    envelope_authority.authority->issue(document);
  ASSERT_TRUE(third_envelope.ready());
  test::failNextExposureTimingAttestationAllocation();
  const auto attest_failure =
    ingress.ingress->ingest(*third_envelope.envelope);
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::TIMING_ATTEST_FAILURE,
    attest_failure.status);
  ASSERT_TRUE(attest_failure.timing_attest_status.has_value());
  EXPECT_EQ(
    ExposureTimingAttestStatus::RESOURCE_FAILURE,
    *attest_failure.timing_attest_status);
  const auto third_ingest =
    ingress.ingress->ingest(*third_envelope.envelope);
  ASSERT_TRUE(third_ingest.ready());
  EXPECT_EQ(
    3U,
    third_ingest.frame->audit()->ingressSequence());
  EXPECT_EQ(
    4U,
    third_ingest.frame->audit()->timing()->sourceSequence());

  auto route_one_envelope =
    envelope_authority.authority->issue(
    validFrameDocument(configuration, 1U, 1U, 500.0));
  ASSERT_TRUE(route_one_envelope.ready());
  test::exhaustNextAttestedFrameIngressSequence(1U);
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::AUTHORITY_EXHAUSTED,
    ingress.ingress->ingest(
      *route_one_envelope.envelope).status);
  document =
    validFrameDocument(configuration, 0U, 4U, 103.0);
  auto fourth_envelope =
    envelope_authority.authority->issue(document);
  ASSERT_TRUE(fourth_envelope.ready());
  const auto fourth_ingest =
    ingress.ingress->ingest(*fourth_envelope.envelope);
  ASSERT_TRUE(fourth_ingest.ready());
  EXPECT_EQ(
    4U,
    fourth_ingest.frame->audit()->ingressSequence());
}

TEST(
  AttestedFrozenCameraFrameTest,
  EstablishedReplaySkipsDigestWorkAndFreshRetryRemainsAtomic)
{
  const auto configuration = validEnvelopeConfiguration();
  auto envelope_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(envelope_authority.ready());
  auto first_document =
    validFrameDocument(configuration, 0U, 1U, 10.0);
  auto first_envelope =
    envelope_authority.authority->issue(first_document);
  ASSERT_TRUE(first_envelope.ready());

  test::failNextCameraDriverEnvelopeDigestComputation();
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::
    STALE_OR_REPLAYED_DRIVER_FRAME,
    envelope_authority.authority->issue(first_document).status);
  auto second_document =
    validFrameDocument(configuration, 0U, 2U, 11.0);
  EXPECT_EQ(
    CameraDriverFrameEnvelopeIssueStatus::RESOURCE_FAILURE,
    envelope_authority.authority->issue(second_document).status);
  auto second_envelope =
    envelope_authority.authority->issue(second_document);
  ASSERT_TRUE(second_envelope.ready());
  EXPECT_EQ(
    2U,
    second_envelope.envelope->audit()->envelopeSequence());

  auto timing = makeTimingRig(configuration);
  ASSERT_TRUE(timing.has_value());
  AttestedCameraFrameIngressConfiguration ingress_configuration;
  ingress_configuration.routes = configuration.routes;
  auto ingress = AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(timing->source),
    std::move(timing->timing));
  ASSERT_TRUE(ingress.ready());
  ASSERT_TRUE(
    ingress.ingress->ingest(*first_envelope.envelope).ready());

  test::failNextCameraDriverEnvelopeDigestComputation();
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::
    STALE_OR_REPLAYED_ENVELOPE,
    ingress.ingress->ingest(*first_envelope.envelope).status);
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::RESOURCE_FAILURE,
    ingress.ingress->ingest(*second_envelope.envelope).status);
  const auto second_ingest =
    ingress.ingress->ingest(*second_envelope.envelope);
  ASSERT_TRUE(second_ingest.ready());
  EXPECT_EQ(
    2U,
    second_ingest.frame->audit()->ingressSequence());
}

TEST(
  AttestedFrozenCameraFrameTest,
  IngressRejectsForeignTamperedAndReplayedEnvelopesPerRoute)
{
  const auto configuration = validEnvelopeConfiguration();
  auto expected_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  auto foreign_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(expected_authority.ready());
  ASSERT_TRUE(foreign_authority.ready());
  auto timing = makeTimingRig(configuration);
  ASSERT_TRUE(timing.has_value());
  AttestedCameraFrameIngressConfiguration
    ingress_configuration;
  ingress_configuration.routes = configuration.routes;
  auto ingress = AttestedCameraFrameIngress::create(
    ingress_configuration,
    expected_authority.authority->identity(),
    std::move(timing->source),
    std::move(timing->timing));
  ASSERT_TRUE(ingress.ready());

  auto foreign_envelope = foreign_authority.authority->issue(
    validFrameDocument(configuration, 0U, 1U, 10.0));
  ASSERT_TRUE(foreign_envelope.ready());
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::
    FOREIGN_ENVELOPE_AUTHORITY,
    ingress.ingress->ingest(*foreign_envelope.envelope).status);

  test::corruptNextCameraDriverEnvelopeMetadata();
  auto metadata_tampered = expected_authority.authority->issue(
    validFrameDocument(configuration, 0U, 1U, 11.0));
  ASSERT_TRUE(metadata_tampered.ready());
  ASSERT_TRUE(metadata_tampered.envelope->valid());
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::FOREIGN_ROUTE_CONTRACT,
    ingress.ingress->ingest(
      *metadata_tampered.envelope).status);

  test::corruptNextCameraDriverEnvelopePixel();
  auto pixel_tampered = expected_authority.authority->issue(
    validFrameDocument(configuration, 0U, 2U, 12.0));
  ASSERT_TRUE(pixel_tampered.ready());
  ASSERT_TRUE(pixel_tampered.envelope->valid());
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::
    INVALID_ENVELOPE_DIGEST,
    ingress.ingress->ingest(*pixel_tampered.envelope).status);

  test::corruptNextCameraDriverEnvelopeDigest();
  auto digest_tampered = expected_authority.authority->issue(
    validFrameDocument(configuration, 0U, 3U, 13.0));
  ASSERT_TRUE(digest_tampered.ready());
  ASSERT_TRUE(digest_tampered.envelope->valid());
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::
    INVALID_ENVELOPE_DIGEST,
    ingress.ingress->ingest(*digest_tampered.envelope).status);

  auto good_envelope = expected_authority.authority->issue(
    validFrameDocument(configuration, 0U, 4U, 14.0));
  ASSERT_TRUE(good_envelope.ready());
  const auto good =
    ingress.ingress->ingest(*good_envelope.envelope);
  ASSERT_TRUE(good.ready());
  EXPECT_EQ(1U, good.frame->audit()->ingressSequence());
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::
    STALE_OR_REPLAYED_ENVELOPE,
    ingress.ingress->ingest(*good_envelope.envelope).status);

  auto independent_route =
    expected_authority.authority->issue(
    validFrameDocument(configuration, 1U, 1U, 10000.0));
  ASSERT_TRUE(independent_route.ready());
  const auto independent =
    ingress.ingress->ingest(*independent_route.envelope);
  ASSERT_TRUE(independent.ready());
  EXPECT_EQ(
    1U,
    independent.frame->audit()->ingressSequence());
}

TEST(
  AttestedFrozenCameraFrameTest,
  TimingConventionsFailClosedWithoutCrossCameraSyncGate)
{
  const auto configuration = validEnvelopeConfiguration();
  auto timing_document = timingDocument(configuration);
  timing_document.cameras[0U].timestamp_convention =
    ExposureTimestampConvention::MIDPOINT;
  timing_document.cameras[1U].timestamp_convention =
    ExposureTimestampConvention::START;
  timing_document.cameras[2U].timestamp_convention =
    ExposureTimestampConvention::END;
  timing_document.cameras[3U].timestamp_convention =
    ExposureTimestampConvention::MIDPOINT;
  auto timing = makeTimingRig(configuration, timing_document);
  ASSERT_TRUE(timing.has_value());
  auto envelope_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(envelope_authority.ready());
  AttestedCameraFrameIngressConfiguration ingress_configuration;
  ingress_configuration.routes = configuration.routes;
  auto ingress = AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(timing->source),
    std::move(timing->timing));
  ASSERT_TRUE(ingress.ready());

  auto midpoint_document =
    validFrameDocument(configuration, 0U, 1U, 1.0);
  midpoint_document.exposure_duration_sec.reset();
  auto start_missing =
    validFrameDocument(configuration, 1U, 1U, 1000.0);
  start_missing.exposure_duration_sec.reset();
  auto end_missing =
    validFrameDocument(configuration, 2U, 1U, 2000.0);
  end_missing.exposure_duration_sec.reset();
  auto remote_midpoint =
    validFrameDocument(configuration, 3U, 1U, 1000000.0);
  remote_midpoint.exposure_duration_sec.reset();

  auto midpoint_envelope =
    envelope_authority.authority->issue(midpoint_document);
  auto start_missing_envelope =
    envelope_authority.authority->issue(start_missing);
  auto end_missing_envelope =
    envelope_authority.authority->issue(end_missing);
  auto remote_envelope =
    envelope_authority.authority->issue(remote_midpoint);
  ASSERT_TRUE(midpoint_envelope.ready());
  ASSERT_TRUE(start_missing_envelope.ready());
  ASSERT_TRUE(end_missing_envelope.ready());
  ASSERT_TRUE(remote_envelope.ready());

  const auto midpoint =
    ingress.ingress->ingest(*midpoint_envelope.envelope);
  const auto start_failure =
    ingress.ingress->ingest(*start_missing_envelope.envelope);
  const auto end_failure =
    ingress.ingress->ingest(*end_missing_envelope.envelope);
  const auto remote =
    ingress.ingress->ingest(*remote_envelope.envelope);
  ASSERT_TRUE(midpoint.ready());
  ASSERT_TRUE(remote.ready());
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::TIMING_ATTEST_FAILURE,
    start_failure.status);
  EXPECT_EQ(
    AttestedCameraFrameIngestStatus::TIMING_ATTEST_FAILURE,
    end_failure.status);
  ASSERT_TRUE(start_failure.timing_attest_status.has_value());
  ASSERT_TRUE(end_failure.timing_attest_status.has_value());
  EXPECT_EQ(
    ExposureTimingAttestStatus::EXPOSURE_DURATION_REQUIRED,
    *start_failure.timing_attest_status);
  EXPECT_EQ(
    ExposureTimingAttestStatus::EXPOSURE_DURATION_REQUIRED,
    *end_failure.timing_attest_status);
  EXPECT_DOUBLE_EQ(
    1.010,
    midpoint.frame->audit()->
    effectiveExposureTime()->seconds());
  EXPECT_DOUBLE_EQ(
    1000000.013,
    remote.frame->audit()->
    effectiveExposureTime()->seconds());

  auto start_retry =
    validFrameDocument(configuration, 1U, 2U, 1001.0);
  auto end_retry =
    validFrameDocument(configuration, 2U, 2U, 2001.0);
  auto start_retry_envelope =
    envelope_authority.authority->issue(start_retry);
  auto end_retry_envelope =
    envelope_authority.authority->issue(end_retry);
  ASSERT_TRUE(start_retry_envelope.ready());
  ASSERT_TRUE(end_retry_envelope.ready());
  const auto start =
    ingress.ingress->ingest(*start_retry_envelope.envelope);
  const auto end =
    ingress.ingress->ingest(*end_retry_envelope.envelope);
  ASSERT_TRUE(start.ready());
  ASSERT_TRUE(end.ready());
  EXPECT_EQ(1U, start.frame->audit()->ingressSequence());
  EXPECT_EQ(1U, end.frame->audit()->ingressSequence());
  EXPECT_DOUBLE_EQ(
    1001.013,
    start.frame->audit()->
    effectiveExposureTime()->seconds());
  EXPECT_DOUBLE_EQ(
    2001.010,
    end.frame->audit()->
    effectiveExposureTime()->seconds());
}

TEST(
  AttestedFrozenCameraFrameTest,
  FourRoutesIssueAndIngestConcurrentlyWithoutCrossRouteState)
{
  const auto configuration = validEnvelopeConfiguration();
  auto timing = makeTimingRig(configuration);
  ASSERT_TRUE(timing.has_value());
  auto envelope_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(envelope_authority.ready());
  AttestedCameraFrameIngressConfiguration ingress_configuration;
  ingress_configuration.routes = configuration.routes;
  auto ingress = AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(timing->source),
    std::move(timing->timing));
  ASSERT_TRUE(ingress.ready());

  std::atomic<bool> failed{false};
  std::array<
    std::optional<AttestedFrozenCameraFrame>,
    kAttestedCameraFrameCameraCount> final_frames;
  std::array<std::thread, kAttestedCameraFrameCameraCount>
    workers;
  for (std::uint8_t camera_id = 0U;
    camera_id < workers.size();
    ++camera_id)
  {
    workers[camera_id] = std::thread(
      [&, camera_id]()
      {
        for (std::uint64_t frame_number = 1U;
          frame_number <= 100U;
          ++frame_number)
        {
          auto document = validFrameDocument(
            configuration,
            camera_id,
            frame_number,
            static_cast<double>(
              1000000U * camera_id + frame_number));
          auto envelope =
            envelope_authority.authority->issue(document);
          if (!envelope.ready())
          {
            failed.store(true, std::memory_order_release);
            return;
          }
          auto frame =
            ingress.ingress->ingest(*envelope.envelope);
          if (!frame.ready())
          {
            failed.store(true, std::memory_order_release);
            return;
          }
          final_frames[camera_id] = *frame.frame;
        }
      });
  }
  for (auto & worker : workers)
  {
    worker.join();
  }

  EXPECT_FALSE(failed.load(std::memory_order_acquire));
  for (std::uint8_t camera_id = 0U;
    camera_id < final_frames.size();
    ++camera_id)
  {
    ASSERT_TRUE(final_frames[camera_id].has_value());
    ASSERT_TRUE(final_frames[camera_id]->audit().has_value());
    EXPECT_EQ(
      camera_id,
      final_frames[camera_id]->audit()->cameraId());
    EXPECT_EQ(
      100U,
      final_frames[camera_id]->
      audit()->ingressSequence());
  }
}

TEST(
  AttestedFrozenCameraFrameTest,
  ConcurrentIssueOfOneDocumentPublishesExactlyOnce)
{
  const auto configuration = validEnvelopeConfiguration();
  auto authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(authority.ready());
  const auto document =
    validFrameDocument(configuration, 2U, 1U, 10.0);

  std::array<CameraDriverFrameEnvelopeIssueResult, 2U> results;
  std::array<std::thread, 2U> workers;
  for (std::size_t index = 0U; index < workers.size(); ++index)
  {
    workers[index] = std::thread(
      [&, index]()
      {
        results[index] = authority.authority->issue(document);
      });
  }
  for (auto & worker : workers)
  {
    worker.join();
  }

  std::size_t ready_count = 0U;
  std::size_t replay_count = 0U;
  for (const auto & result : results)
  {
    ready_count += result.ready() ? 1U : 0U;
    replay_count +=
      result.status ==
      CameraDriverFrameEnvelopeIssueStatus::
      STALE_OR_REPLAYED_DRIVER_FRAME ?
      1U : 0U;
    if (result.ready())
    {
      ASSERT_TRUE(result.envelope->audit().has_value());
      EXPECT_EQ(
        1U,
        result.envelope->audit()->envelopeSequence());
    }
  }
  EXPECT_EQ(1U, ready_count);
  EXPECT_EQ(1U, replay_count);
}

TEST(
  AttestedFrozenCameraFrameTest,
  ConcurrentReplayOfOneEnvelopePublishesExactlyOnce)
{
  const auto configuration = validEnvelopeConfiguration();
  auto timing = makeTimingRig(configuration);
  auto envelope_authority =
    CameraDriverFrameEnvelopeAuthority::create(configuration);
  ASSERT_TRUE(timing.has_value());
  ASSERT_TRUE(envelope_authority.ready());
  AttestedCameraFrameIngressConfiguration ingress_configuration;
  ingress_configuration.routes = configuration.routes;
  auto ingress = AttestedCameraFrameIngress::create(
    ingress_configuration,
    envelope_authority.authority->identity(),
    std::move(timing->source),
    std::move(timing->timing));
  ASSERT_TRUE(ingress.ready());
  auto envelope = envelope_authority.authority->issue(
    validFrameDocument(configuration, 0U, 1U, 10.0));
  ASSERT_TRUE(envelope.ready());

  std::array<AttestedCameraFrameIngestResult, 2U> results;
  std::array<std::thread, 2U> workers;
  for (std::size_t index = 0U; index < workers.size(); ++index)
  {
    workers[index] = std::thread(
      [&, index]()
      {
        results[index] =
          ingress.ingress->ingest(*envelope.envelope);
      });
  }
  for (auto & worker : workers)
  {
    worker.join();
  }

  std::size_t ready_count = 0U;
  std::size_t replay_count = 0U;
  for (const auto & result : results)
  {
    ready_count += result.ready() ? 1U : 0U;
    replay_count +=
      result.status ==
      AttestedCameraFrameIngestStatus::
      STALE_OR_REPLAYED_ENVELOPE ?
      1U : 0U;
  }
  EXPECT_EQ(1U, ready_count);
  EXPECT_EQ(1U, replay_count);
}

TEST(
  AttestedFrozenCameraFrameTest,
  TokensRetainExactAuditAfterAllAuthoritiesAreDestroyed)
{
  const auto configuration = validEnvelopeConfiguration();
  std::optional<TrustedCameraDriverFrameEnvelope>
    retained_envelope;
  std::optional<AttestedFrozenCameraFrame> retained_frame;
  std::optional<AttestedCameraFrameIngressIdentity>
    retained_ingress_identity;
  {
    auto envelope_authority =
      CameraDriverFrameEnvelopeAuthority::create(configuration);
    auto timing = makeTimingRig(configuration);
    ASSERT_TRUE(envelope_authority.ready());
    ASSERT_TRUE(timing.has_value());
    AttestedCameraFrameIngressConfiguration
      ingress_configuration;
    ingress_configuration.routes = configuration.routes;
    auto ingress = AttestedCameraFrameIngress::create(
      ingress_configuration,
      envelope_authority.authority->identity(),
      std::move(timing->source),
      std::move(timing->timing));
    ASSERT_TRUE(ingress.ready());
    retained_ingress_identity = ingress.ingress->identity();
    auto envelope = envelope_authority.authority->issue(
      validFrameDocument(configuration, 2U, 9U, 42.0));
    ASSERT_TRUE(envelope.ready());
    retained_envelope = *envelope.envelope;
    auto frame =
      ingress.ingress->ingest(*envelope.envelope);
    ASSERT_TRUE(frame.ready());
    retained_frame = *frame.frame;
  }

  ASSERT_TRUE(retained_envelope.has_value());
  ASSERT_TRUE(retained_frame.has_value());
  ASSERT_TRUE(retained_ingress_identity.has_value());
  EXPECT_TRUE(retained_envelope->valid());
  EXPECT_TRUE(retained_frame->valid());
  EXPECT_TRUE(
    retained_frame->issuedBy(*retained_ingress_identity));
  EXPECT_TRUE(
    retained_frame->sameFrame(*retained_envelope));
  ASSERT_TRUE(retained_frame->audit().has_value());
  EXPECT_EQ(2U, retained_frame->audit()->cameraId());
  EXPECT_EQ(
    retained_envelope->audit()->canonicalDigest(),
    retained_frame->audit()->
    envelope()->canonicalDigest());
}

}  // namespace
}  // namespace livo
