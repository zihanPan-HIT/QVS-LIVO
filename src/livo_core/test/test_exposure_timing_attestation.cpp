#include "exposure_timing_attestation.h"
#include "exposure_timing_attestation_failure_injection_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

livo::ExposureTimingCalibrationDocument fixtureDocument()
{
  livo::ExposureTimingCalibrationDocument document;
  document.mode =
    livo::ExposureTimingAuthorityMode::SIMULATION_TEST;
  document.provenance =
    livo::ExposureTimingProvenanceKind::TEST_FIXTURE;
  document.canonical_payload =
    "{\"fixture\":\"not-a-real-calibration\"}";
  document.canonical_payload_sha256.fill(0xA5U);
  document.artifact_id = "test-fixture-only";
  document.calibration_revision = 7U;
  document.global_offset_sec = 0.010;
  document.legacy_exposure_time_init_sec = 0.0;

  for (std::uint8_t camera_id = 0U;
    camera_id < document.cameras.size();
    ++camera_id)
  {
    auto & camera = document.cameras[camera_id];
    camera.source_binding.camera_id = camera_id;
    camera.source_binding.source_id =
      "source-" + std::to_string(camera_id);
    camera.source_binding.topic =
      "/camera/" + std::to_string(camera_id) + "/image";
    camera.source_binding.sensor_serial =
      "serial-" + std::to_string(camera_id);
    camera.source_binding.frame_id =
      "camera-" + std::to_string(camera_id);
    camera.timestamp_convention =
      livo::ExposureTimestampConvention::MIDPOINT;
    camera.per_camera_offset_sec =
      static_cast<double>(camera_id) * 0.001;
    camera.duration_source = "driver-frame-metadata";
  }
  return document;
}

livo::ExposureTimingSourceAuthorityConfiguration sourceConfiguration(
  const livo::ExposureTimingCalibrationDocument & document)
{
  livo::ExposureTimingSourceAuthorityConfiguration configuration;
  for (std::size_t camera_id = 0U;
    camera_id < configuration.cameras.size();
    ++camera_id)
  {
    configuration.cameras[camera_id] =
      document.cameras[camera_id].source_binding;
  }
  return configuration;
}

struct FixtureRig final
{
  livo::TrustedExposureTimingCalibration calibration;
  std::unique_ptr<livo::ExposureTimingSourceAuthority> source;
  std::unique_ptr<livo::ExposureTimingAuthority> timing;
};

std::optional<FixtureRig> makeFixtureRig(
  const livo::ExposureTimingCalibrationDocument & document,
  const std::optional<
    livo::ExposureTimingSourceAuthorityConfiguration> &
    source_configuration = std::nullopt)
{
  auto minted =
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document);
  if (!minted.ready())
  {
    return std::nullopt;
  }
  auto source_created =
    livo::ExposureTimingSourceAuthority::create(
    source_configuration.value_or(
      sourceConfiguration(document)));
  if (!source_created.ready())
  {
    return std::nullopt;
  }
  const auto source_identity =
    source_created.authority->identity();
  auto timing_created =
    livo::ExposureTimingAuthority::create(
    *minted.calibration, source_identity);
  if (!timing_created.ready())
  {
    return std::nullopt;
  }
  return FixtureRig{
    *minted.calibration,
    std::move(source_created.authority),
    std::move(timing_created.authority)};
}

TEST(
  ExposureTimingAttestationTest,
  FixtureCalibrationAttestsStrongTimeAndRetainsOneAuditView)
{
  auto rig = makeFixtureRig(fixtureDocument());
  ASSERT_TRUE(rig.has_value());
  const auto source_identity = rig->source->identity();
  const auto timing_identity = rig->timing->identity();
  const auto issued =
    rig->source->issue(2U, 100.0, std::nullopt);
  ASSERT_TRUE(issued.ready());

  const auto result =
    rig->timing->attest(*issued.receipt);

  ASSERT_TRUE(result.ready());
  const auto & attestation = *result.attestation;
  EXPECT_TRUE(attestation.issuedBy(timing_identity));
  EXPECT_EQ(attestation.cameraId(), 2U);
  EXPECT_EQ(attestation.sourceSequence(), 1U);
  EXPECT_TRUE(
    attestation.sourceAuthorityIdentity()
    .sameAuthority(source_identity));
  EXPECT_TRUE(
    attestation.sourceFrameIdentity().sameFrame(
      issued.receipt->frameIdentity()));
  ASSERT_TRUE(attestation.effectiveExposureTime().has_value());
  EXPECT_DOUBLE_EQ(
    attestation.effectiveExposureTime()->seconds(), 100.012);

  ASSERT_TRUE(attestation.audit().has_value());
  const auto audit = *attestation.audit();
  EXPECT_EQ(audit.attestationSequence(), 1U);
  EXPECT_EQ(audit.sourceId(), "source-2");
  EXPECT_EQ(audit.topic(), "/camera/2/image");
  EXPECT_EQ(audit.sensorSerial(), "serial-2");
  EXPECT_EQ(audit.frameId(), "camera-2");
  EXPECT_EQ(
    audit.timestampConvention(),
    livo::ExposureTimestampConvention::MIDPOINT);
  EXPECT_DOUBLE_EQ(audit.rawHeaderTimestampSec(), 100.0);
  EXPECT_FALSE(audit.exposureDurationSec().has_value());
  EXPECT_EQ(audit.durationSource(), "driver-frame-metadata");
  EXPECT_DOUBLE_EQ(audit.midpointTimestampSec(), 100.0);
  EXPECT_DOUBLE_EQ(audit.globalOffsetSec(), 0.010);
  EXPECT_DOUBLE_EQ(audit.perCameraOffsetSec(), 0.002);
  EXPECT_DOUBLE_EQ(audit.totalOffsetSec(), 0.012);
  EXPECT_DOUBLE_EQ(audit.effectiveTimestampSec(), 100.012);
  EXPECT_EQ(
    audit.authorityMode(),
    livo::ExposureTimingAuthorityMode::SIMULATION_TEST);
  EXPECT_EQ(
    audit.provenanceKind(),
    livo::ExposureTimingProvenanceKind::TEST_FIXTURE);
  EXPECT_EQ(audit.calibrationRevision(), 7U);
  EXPECT_EQ(audit.artifactId(), "test-fixture-only");
  EXPECT_EQ(
    audit.canonicalPayload(),
    "{\"fixture\":\"not-a-real-calibration\"}");
  livo::ExposureTimingArtifactDigest expected_digest{};
  expected_digest.fill(0xA5U);
  EXPECT_EQ(audit.canonicalPayloadSha256(), expected_digest);
}

TEST(
  ExposureTimingAttestationTest,
  TimingAuthorityProvesItsExactSourceAuthorityBinding)
{
  auto rig = makeFixtureRig(fixtureDocument());
  auto foreign_rig = makeFixtureRig(fixtureDocument());
  ASSERT_TRUE(rig.has_value());
  ASSERT_TRUE(foreign_rig.has_value());
  const auto source_identity = rig->source->identity();
  auto invalid_identity = source_identity;
  auto retained_identity = std::move(invalid_identity);
  ASSERT_TRUE(retained_identity.valid());
  ASSERT_FALSE(invalid_identity.valid());

  EXPECT_TRUE(rig->timing->boundTo(source_identity));
  EXPECT_FALSE(
    rig->timing->boundTo(
      foreign_rig->source->identity()));
  EXPECT_FALSE(rig->timing->boundTo(invalid_identity));

  const auto document = fixtureDocument();
  auto expected_configuration = sourceConfiguration(document);
  EXPECT_TRUE(
    rig->source->configuredFor(
      expected_configuration.cameras[1U]));
  expected_configuration.cameras[2U].sensor_serial =
    "foreign-serial";
  EXPECT_FALSE(
    rig->source->configuredFor(
      expected_configuration.cameras[2U]));
}

TEST(
  ExposureTimingAttestationTest,
  FourCameraContractsApplyMidpointStartEndAndSignedOffsets)
{
  auto document = fixtureDocument();
  document.cameras[0U].timestamp_convention =
    livo::ExposureTimestampConvention::MIDPOINT;
  document.cameras[0U].per_camera_offset_sec = 0.0;
  document.cameras[1U].timestamp_convention =
    livo::ExposureTimestampConvention::START;
  document.cameras[1U].per_camera_offset_sec = -0.002;
  document.cameras[2U].timestamp_convention =
    livo::ExposureTimestampConvention::END;
  document.cameras[2U].per_camera_offset_sec = 0.003;
  document.cameras[3U].timestamp_convention =
    livo::ExposureTimestampConvention::MIDPOINT;
  document.cameras[3U].per_camera_offset_sec = -0.010;
  auto rig = makeFixtureRig(document);
  ASSERT_TRUE(rig.has_value());

  const auto midpoint_frame =
    rig->source->issue(0U, 10.0, std::nullopt);
  const auto start_frame =
    rig->source->issue(1U, 20.0, 0.020);
  const auto end_frame =
    rig->source->issue(2U, 30.0, 0.040);
  const auto calibrated_zero_frame =
    rig->source->issue(3U, 40.0, std::nullopt);
  ASSERT_TRUE(midpoint_frame.ready());
  ASSERT_TRUE(start_frame.ready());
  ASSERT_TRUE(end_frame.ready());
  ASSERT_TRUE(calibrated_zero_frame.ready());

  const auto midpoint =
    rig->timing->attest(*midpoint_frame.receipt);
  const auto start =
    rig->timing->attest(*start_frame.receipt);
  const auto end =
    rig->timing->attest(*end_frame.receipt);
  const auto calibrated_zero =
    rig->timing->attest(*calibrated_zero_frame.receipt);
  ASSERT_TRUE(midpoint.ready());
  ASSERT_TRUE(start.ready());
  ASSERT_TRUE(end.ready());
  ASSERT_TRUE(calibrated_zero.ready());

  ASSERT_TRUE(midpoint.attestation->audit().has_value());
  ASSERT_TRUE(start.attestation->audit().has_value());
  ASSERT_TRUE(end.attestation->audit().has_value());
  EXPECT_DOUBLE_EQ(
    midpoint.attestation->audit()->midpointTimestampSec(),
    10.0);
  EXPECT_NEAR(
    midpoint.attestation->effectiveExposureTime()->seconds(),
    10.010, 1.0e-14);
  EXPECT_NEAR(
    start.attestation->audit()->midpointTimestampSec(),
    20.010, 1.0e-14);
  EXPECT_NEAR(
    start.attestation->effectiveExposureTime()->seconds(),
    20.018, 1.0e-14);
  EXPECT_NEAR(
    end.attestation->audit()->midpointTimestampSec(),
    29.980, 1.0e-14);
  EXPECT_NEAR(
    end.attestation->effectiveExposureTime()->seconds(),
    29.993, 1.0e-14);
  EXPECT_DOUBLE_EQ(
    calibrated_zero.attestation
    ->effectiveExposureTime()->seconds(),
    40.0);
}

TEST(
  ExposureTimingAttestationTest,
  StartAndEndRequireStrictMovementForNonzeroHalfDuration)
{
  for (const auto convention : {
      livo::ExposureTimestampConvention::START,
      livo::ExposureTimestampConvention::END})
  {
    auto document = fixtureDocument();
    document.global_offset_sec = 0.0;
    for (auto & camera : document.cameras)
    {
      camera.per_camera_offset_sec = 0.0;
    }
    document.cameras[0U].timestamp_convention = convention;
    auto rig = makeFixtureRig(document);
    ASSERT_TRUE(rig.has_value());
    const auto frame = rig->source->issue(
      0U,
      1.0,
      2.0 * std::numeric_limits<double>::denorm_min());
    ASSERT_TRUE(frame.ready());

    const auto result =
      rig->timing->attest(*frame.receipt);

    EXPECT_EQ(
      result.status,
      livo::ExposureTimingAttestStatus::
      TIMESTAMP_ARITHMETIC_FAILURE);
    EXPECT_FALSE(result.ready());
  }
}

TEST(
  ExposureTimingAttestationTest,
  HalfUnderflowAndNonpositiveResultsAreArithmeticFailures)
{
  auto document = fixtureDocument();
  document.global_offset_sec = 0.0;
  for (auto & camera : document.cameras)
  {
    camera.per_camera_offset_sec = 0.0;
  }
  document.cameras[0U].timestamp_convention =
    livo::ExposureTimestampConvention::START;
  document.cameras[1U].timestamp_convention =
    livo::ExposureTimestampConvention::END;
  auto rig = makeFixtureRig(document);
  ASSERT_TRUE(rig.has_value());

  const auto underflow = rig->source->issue(
    0U, 1.0, std::numeric_limits<double>::denorm_min());
  const auto nonpositive =
    rig->source->issue(1U, 0.25, 1.0);
  ASSERT_TRUE(underflow.ready());
  ASSERT_TRUE(nonpositive.ready());
  EXPECT_EQ(
    rig->timing->attest(*underflow.receipt).status,
    livo::ExposureTimingAttestStatus::
    TIMESTAMP_ARITHMETIC_FAILURE);
  EXPECT_EQ(
    rig->timing->attest(*nonpositive.receipt).status,
    livo::ExposureTimingAttestStatus::
    TIMESTAMP_ARITHMETIC_FAILURE);
}

TEST(
  ExposureTimingAttestationTest,
  OnlyNonfiniteTimestampResultsAreClassifiedAsOverflow)
{
  auto document = fixtureDocument();
  document.global_offset_sec = 0.0;
  for (auto & camera : document.cameras)
  {
    camera.per_camera_offset_sec = 0.0;
  }
  document.cameras[0U].timestamp_convention =
    livo::ExposureTimestampConvention::START;
  auto rig = makeFixtureRig(document);
  ASSERT_TRUE(rig.has_value());
  const auto frame = rig->source->issue(
    0U,
    std::numeric_limits<double>::max(),
    std::numeric_limits<double>::max());
  ASSERT_TRUE(frame.ready());
  EXPECT_EQ(
    rig->timing->attest(*frame.receipt).status,
    livo::ExposureTimingAttestStatus::TIMESTAMP_OVERFLOW);

  document = fixtureDocument();
  document.global_offset_sec =
    std::numeric_limits<double>::max();
  for (auto & camera : document.cameras)
  {
    camera.per_camera_offset_sec = 0.0;
  }
  rig = makeFixtureRig(document);
  ASSERT_TRUE(rig.has_value());
  const auto offset_frame = rig->source->issue(
    0U, std::numeric_limits<double>::max(), std::nullopt);
  ASSERT_TRUE(offset_frame.ready());
  EXPECT_EQ(
    rig->timing->attest(*offset_frame.receipt).status,
    livo::ExposureTimingAttestStatus::TIMESTAMP_OVERFLOW);
}

TEST(
  ExposureTimingAttestationTest,
  NonzeroStaticOffsetMustMoveInItsDeclaredDirection)
{
  for (const double offset : {
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min()})
  {
    auto document = fixtureDocument();
    document.global_offset_sec = offset;
    for (auto & camera : document.cameras)
    {
      camera.per_camera_offset_sec = 0.0;
    }
    auto rig = makeFixtureRig(document);
    ASSERT_TRUE(rig.has_value());
    const auto frame =
      rig->source->issue(0U, 1.0, std::nullopt);
    ASSERT_TRUE(frame.ready());

    EXPECT_EQ(
      rig->timing->attest(*frame.receipt).status,
      livo::ExposureTimingAttestStatus::
      TIMESTAMP_ARITHMETIC_FAILURE);
  }
}

TEST(
  ExposureTimingAttestationTest,
  MissingDurationRejectsAtTimingWithoutInvalidatingReceipt)
{
  auto document = fixtureDocument();
  document.cameras[0U].timestamp_convention =
    livo::ExposureTimestampConvention::START;
  document.cameras[1U].timestamp_convention =
    livo::ExposureTimestampConvention::END;
  auto rig = makeFixtureRig(document);
  ASSERT_TRUE(rig.has_value());
  const auto start =
    rig->source->issue(0U, 10.0, std::nullopt);
  const auto end =
    rig->source->issue(1U, 10.0, std::nullopt);
  ASSERT_TRUE(start.ready());
  ASSERT_TRUE(end.ready());
  EXPECT_EQ(
    rig->timing->attest(*start.receipt).status,
    livo::ExposureTimingAttestStatus::
    EXPOSURE_DURATION_REQUIRED);
  EXPECT_EQ(
    rig->timing->attest(*end.receipt).status,
    livo::ExposureTimingAttestStatus::
    EXPOSURE_DURATION_REQUIRED);
  EXPECT_TRUE(start.receipt->valid());
  EXPECT_TRUE(end.receipt->valid());
}

TEST(
  ExposureTimingAttestationTest,
  CalibrationMintValidatesProvenanceOffsetsLegacyAndConvention)
{
  const auto expect_status =
    [](livo::ExposureTimingCalibrationDocument document,
      const livo::ExposureTimingCalibrationStatus expected)
    {
      const auto result =
        livo::test::ExposureTimingCalibrationTestAccess::
        mintFixture(document);
      EXPECT_EQ(result.status, expected);
      EXPECT_FALSE(result.ready());
    };

  auto document = fixtureDocument();
  document.mode = livo::ExposureTimingAuthorityMode::UNKNOWN;
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_AUTHORITY_MODE);
  document = fixtureDocument();
  document.provenance =
    livo::ExposureTimingProvenanceKind::UNKNOWN;
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    MISSING_CALIBRATION_PROVENANCE);
  document = fixtureDocument();
  document.mode = livo::ExposureTimingAuthorityMode::PRODUCTION;
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    PROVENANCE_MODE_MISMATCH);
  document.provenance =
    livo::ExposureTimingProvenanceKind::HARDWARE_CALIBRATION;
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    PROVENANCE_MODE_MISMATCH);
  document = fixtureDocument();
  document.calibration_revision = 0U;
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_CALIBRATION_REVISION);
  document = fixtureDocument();
  document.canonical_payload_sha256.fill(0U);
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_CANONICAL_PAYLOAD_HASH);
  document = fixtureDocument();
  document.cameras[0U].timestamp_convention =
    static_cast<livo::ExposureTimestampConvention>(99);
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    UNKNOWN_TIMESTAMP_CONVENTION);
  document = fixtureDocument();
  document.global_offset_sec =
    std::numeric_limits<double>::infinity();
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_STATIC_OFFSET);
  document = fixtureDocument();
  document.cameras[0U].per_camera_offset_sec =
    std::numeric_limits<double>::quiet_NaN();
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_STATIC_OFFSET);
  for (const double legacy_value : {
      0.001,
      -0.001,
      std::numeric_limits<double>::quiet_NaN()})
  {
    document = fixtureDocument();
    document.legacy_exposure_time_init_sec = legacy_value;
    expect_status(
      document,
      livo::ExposureTimingCalibrationStatus::
      NONZERO_OR_INVALID_LEGACY_EXPOSURE_OFFSET);
  }
}

TEST(
  ExposureTimingAttestationTest,
  CalibrationMintRejectsControlNulWhitespaceAndOversizeStrings)
{
  const auto expect_status =
    [](livo::ExposureTimingCalibrationDocument document,
      const livo::ExposureTimingCalibrationStatus expected)
    {
      const auto result =
        livo::test::ExposureTimingCalibrationTestAccess::
        mintFixture(document);
      EXPECT_EQ(result.status, expected);
      EXPECT_FALSE(result.ready());
    };

  auto document = fixtureDocument();
  document.canonical_payload = "payload with space";
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_CANONICAL_PAYLOAD);
  document = fixtureDocument();
  document.canonical_payload =
    std::string("abc\0def", 7U);
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_CANONICAL_PAYLOAD);
  document = fixtureDocument();
  document.canonical_payload = std::string(4097U, 'x');
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_CANONICAL_PAYLOAD);
  document = fixtureDocument();
  document.artifact_id = "fixture\nid";
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_ARTIFACT_ID);
  document = fixtureDocument();
  document.cameras[0U].source_binding.source_id =
    "source\t0";
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_SOURCE_BINDING);
  document = fixtureDocument();
  document.cameras[0U].source_binding.topic =
    std::string("/camera\0/image", 13U);
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_SOURCE_BINDING);
  document = fixtureDocument();
  document.cameras[0U].source_binding.sensor_serial =
    std::string(129U, 's');
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_SOURCE_BINDING);
  document = fixtureDocument();
  document.cameras[0U].duration_source = "driver metadata";
  expect_status(
    document,
    livo::ExposureTimingCalibrationStatus::
    INVALID_SOURCE_BINDING);
}

TEST(
  ExposureTimingAttestationTest,
  CalibrationAndSourceRejectDuplicateOrOutOfRangeBindings)
{
  auto document = fixtureDocument();
  document.cameras[0U].source_binding.camera_id = 1U;
  EXPECT_EQ(
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document).status,
    livo::ExposureTimingCalibrationStatus::INVALID_CAMERA_ID);
  document = fixtureDocument();
  document.cameras[1U].source_binding.topic =
    document.cameras[0U].source_binding.topic;
  EXPECT_EQ(
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document).status,
    livo::ExposureTimingCalibrationStatus::
    INVALID_SOURCE_BINDING);

  auto configuration =
    sourceConfiguration(fixtureDocument());
  configuration.cameras[0U].camera_id = 4U;
  EXPECT_EQ(
    livo::ExposureTimingSourceAuthority::create(
      configuration).status,
    livo::ExposureTimingSourceAuthorityCreateStatus::
    INVALID_CAMERA_ID);
  configuration = sourceConfiguration(fixtureDocument());
  configuration.cameras[0U].source_id = "source\n0";
  EXPECT_EQ(
    livo::ExposureTimingSourceAuthority::create(
      configuration).status,
    livo::ExposureTimingSourceAuthorityCreateStatus::
    INVALID_SOURCE_BINDING);
  configuration = sourceConfiguration(fixtureDocument());
  configuration.cameras[0U].topic = std::string(257U, '/');
  EXPECT_EQ(
    livo::ExposureTimingSourceAuthority::create(
      configuration).status,
    livo::ExposureTimingSourceAuthorityCreateStatus::
    INVALID_SOURCE_BINDING);
  configuration = sourceConfiguration(fixtureDocument());
  configuration.cameras[1U].frame_id =
    configuration.cameras[0U].frame_id;
  EXPECT_EQ(
    livo::ExposureTimingSourceAuthority::create(
      configuration).status,
    livo::ExposureTimingSourceAuthorityCreateStatus::
    INVALID_SOURCE_BINDING);
}

TEST(
  ExposureTimingAttestationTest,
  TimingFactoryPinsExactSourceBindings)
{
  const auto document = fixtureDocument();
  auto minted =
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document);
  ASSERT_TRUE(minted.ready());
  auto source_configuration = sourceConfiguration(document);
  source_configuration.cameras[0U].source_id =
    "other-source-0";
  auto source_created =
    livo::ExposureTimingSourceAuthority::create(
    source_configuration);
  ASSERT_TRUE(source_created.ready());

  const auto timing_created =
    livo::ExposureTimingAuthority::create(
    *minted.calibration,
    source_created.authority->identity());

  EXPECT_EQ(
    timing_created.status,
    livo::ExposureTimingAuthorityCreateStatus::
    CALIBRATION_SOURCE_BINDING_MISMATCH);
  EXPECT_FALSE(timing_created.ready());
}

TEST(
  ExposureTimingAttestationTest,
  ForeignReceiptWithSameScalarsAndSequenceIsRejected)
{
  const auto document = fixtureDocument();
  auto first_source =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(document));
  auto second_source =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(document));
  auto minted =
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document);
  ASSERT_TRUE(first_source.ready());
  ASSERT_TRUE(second_source.ready());
  ASSERT_TRUE(minted.ready());
  auto timing =
    livo::ExposureTimingAuthority::create(
    *minted.calibration,
    first_source.authority->identity());
  ASSERT_TRUE(timing.ready());
  const auto first =
    first_source.authority->issue(0U, 42.0, std::nullopt);
  const auto second =
    second_source.authority->issue(0U, 42.0, std::nullopt);
  ASSERT_TRUE(first.ready());
  ASSERT_TRUE(second.ready());
  EXPECT_EQ(first.receipt->sourceSequence(), 1U);
  EXPECT_EQ(second.receipt->sourceSequence(), 1U);
  EXPECT_FALSE(first.receipt->sameFrame(*second.receipt));
  EXPECT_FALSE(
    first.receipt->frameIdentity().sameFrame(
      second.receipt->frameIdentity()));
  EXPECT_EQ(
    timing.authority->attest(*second.receipt).status,
    livo::ExposureTimingAttestStatus::
    FOREIGN_SOURCE_RECEIPT);
  EXPECT_TRUE(timing.authority->attest(*first.receipt).ready());
}

TEST(
  ExposureTimingAttestationTest,
  SourceAuthorityAssignsIndependentPerCameraSequences)
{
  auto created =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(fixtureDocument()));
  ASSERT_TRUE(created.ready());
  const auto camera_zero_first =
    created.authority->issue(0U, 1.0, std::nullopt);
  const auto camera_zero_second =
    created.authority->issue(0U, 2.0, std::nullopt);
  const auto camera_one_first =
    created.authority->issue(1U, 3.0, std::nullopt);
  ASSERT_TRUE(camera_zero_first.ready());
  ASSERT_TRUE(camera_zero_second.ready());
  ASSERT_TRUE(camera_one_first.ready());
  EXPECT_EQ(camera_zero_first.receipt->sourceSequence(), 1U);
  EXPECT_EQ(camera_zero_second.receipt->sourceSequence(), 2U);
  EXPECT_EQ(camera_one_first.receipt->sourceSequence(), 1U);
}

TEST(
  ExposureTimingAttestationTest,
  SourceRejectsInvalidMeasurementsWithoutConsumingSequence)
{
  auto created =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(fixtureDocument()));
  ASSERT_TRUE(created.ready());
  EXPECT_EQ(
    created.authority->issue(
      4U, 1.0, std::nullopt).status,
    livo::ExposureFrameTimingIssueStatus::INVALID_CAMERA_ID);
  for (const double invalid_header : {
      0.0,
      -1.0,
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    EXPECT_EQ(
      created.authority->issue(
        0U, invalid_header, std::nullopt).status,
      livo::ExposureFrameTimingIssueStatus::
      INVALID_RAW_HEADER_TIMESTAMP);
  }
  for (const double invalid_duration : {
      0.0,
      -1.0,
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    EXPECT_EQ(
      created.authority->issue(
        0U, 1.0, invalid_duration).status,
      livo::ExposureFrameTimingIssueStatus::
      INVALID_EXPOSURE_DURATION);
  }
  const auto valid =
    created.authority->issue(0U, 1.0, std::nullopt);
  ASSERT_TRUE(valid.ready());
  EXPECT_EQ(valid.receipt->sourceSequence(), 1U);
}

TEST(
  ExposureTimingAttestationTest,
  ConcurrentSourceIssueAllocatesUniqueSequences)
{
  auto created =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(fixtureDocument()));
  ASSERT_TRUE(created.ready());
  constexpr std::size_t thread_count = 16U;
  std::array<std::uint64_t, thread_count> sequences{};
  std::array<
    livo::ExposureFrameTimingIssueStatus,
    thread_count> statuses{};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t index = 0U;
    index < thread_count;
    ++index)
  {
    threads.emplace_back(
      [&, index]()
      {
        const auto result = created.authority->issue(
          2U,
          10.0 + static_cast<double>(index),
          std::nullopt);
        statuses[index] = result.status;
        if (result.ready())
        {
          sequences[index] =
            result.receipt->sourceSequence();
        }
      });
  }
  for (auto & thread : threads)
  {
    thread.join();
  }
  EXPECT_TRUE(std::all_of(
      statuses.begin(), statuses.end(),
      [](const auto status)
      {
        return
          status ==
          livo::ExposureFrameTimingIssueStatus::READY;
      }));
  std::sort(sequences.begin(), sequences.end());
  for (std::size_t index = 0U;
    index < thread_count;
    ++index)
  {
    EXPECT_EQ(sequences[index], index + 1U);
  }
}

TEST(
  ExposureTimingAttestationTest,
  EveryCameraFailsClosedBeforeSourceSequenceWrap)
{
  auto created =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(fixtureDocument()));
  ASSERT_TRUE(created.ready());
  for (std::uint8_t camera_id = 0U;
    camera_id < livo::kExposureTimingCameraCount;
    ++camera_id)
  {
    livo::test::exhaustNextExposureTimingSourceSequence(
      camera_id);
    const auto result = created.authority->issue(
      camera_id, 10.0, std::nullopt);
    EXPECT_EQ(
      result.status,
      livo::ExposureFrameTimingIssueStatus::
      AUTHORITY_EXHAUSTED);
    EXPECT_FALSE(result.ready());
    EXPECT_EQ(
      created.authority->issue(
        camera_id, 11.0, std::nullopt).status,
      livo::ExposureFrameTimingIssueStatus::
      AUTHORITY_EXHAUSTED);
  }
}

TEST(
  ExposureTimingAttestationTest,
  SourceReceiptAndIdentityOutliveAuthority)
{
  auto created =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(fixtureDocument()));
  ASSERT_TRUE(created.ready());
  const auto source_identity = created.authority->identity();
  const auto issued =
    created.authority->issue(1U, 10.0, std::nullopt);
  ASSERT_TRUE(issued.ready());
  const auto frame_identity =
    issued.receipt->frameIdentity();
  created.authority.reset();

  EXPECT_TRUE(source_identity.valid());
  EXPECT_TRUE(issued.receipt->valid());
  EXPECT_TRUE(issued.receipt->issuedBy(source_identity));
  EXPECT_TRUE(frame_identity.valid());
  EXPECT_TRUE(
    issued.receipt->frameIdentity().sameFrame(frame_identity));
  EXPECT_EQ(issued.receipt->sourceSequence(), 1U);
}

TEST(
  ExposureTimingAttestationTest,
  SourceRevalidationReservesAfterConcurrentWinner)
{
  auto created =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(fixtureDocument()));
  ASSERT_TRUE(created.ready());
  livo::test::
  pauseNextExposureFrameTimingReceiptBeforeRevalidation();
  livo::ExposureFrameTimingIssueResult delayed;
  std::thread thread(
    [&]()
    {
      delayed =
        created.authority->issue(2U, 10.0, std::nullopt);
    });
  const bool paused =
    livo::test::
    waitForExposureFrameTimingReceiptPausedBeforeRevalidation();
  if (!paused)
  {
    livo::test::
    releaseExposureFrameTimingReceiptBeforeRevalidation();
    thread.join();
    FAIL() << "source issue did not reach revalidation seam";
  }
  const auto winner =
    created.authority->issue(2U, 11.0, std::nullopt);
  livo::test::
  releaseExposureFrameTimingReceiptBeforeRevalidation();
  thread.join();

  ASSERT_TRUE(winner.ready());
  ASSERT_TRUE(delayed.ready());
  EXPECT_EQ(winner.receipt->sourceSequence(), 1U);
  EXPECT_EQ(delayed.receipt->sourceSequence(), 2U);
}

TEST(
  ExposureTimingAttestationTest,
  TimingRejectsDuplicateAndReorderedSourceReceipts)
{
  auto rig = makeFixtureRig(fixtureDocument());
  ASSERT_TRUE(rig.has_value());
  const auto older =
    rig->source->issue(0U, 10.0, std::nullopt);
  const auto newer =
    rig->source->issue(0U, 11.0, std::nullopt);
  ASSERT_TRUE(older.ready());
  ASSERT_TRUE(newer.ready());
  ASSERT_TRUE(rig->timing->attest(*newer.receipt).ready());
  EXPECT_EQ(
    rig->timing->attest(*newer.receipt).status,
    livo::ExposureTimingAttestStatus::
    STALE_OR_REPLAYED_SOURCE_SEQUENCE);
  EXPECT_EQ(
    rig->timing->attest(*older.receipt).status,
    livo::ExposureTimingAttestStatus::
    STALE_OR_REPLAYED_SOURCE_SEQUENCE);
}

TEST(
  ExposureTimingAttestationTest,
  ConcurrentReplayOfOneReceiptHasExactlyOneWinner)
{
  auto rig = makeFixtureRig(fixtureDocument());
  ASSERT_TRUE(rig.has_value());
  const auto frame =
    rig->source->issue(3U, 10.0, std::nullopt);
  ASSERT_TRUE(frame.ready());
  constexpr std::size_t thread_count = 16U;
  std::array<
    livo::ExposureTimingAttestStatus,
    thread_count> statuses{};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::size_t index = 0U;
    index < thread_count;
    ++index)
  {
    threads.emplace_back(
      [&, index]()
      {
        statuses[index] =
          rig->timing->attest(*frame.receipt).status;
      });
  }
  for (auto & thread : threads)
  {
    thread.join();
  }
  const auto ready_count = std::count(
    statuses.begin(),
    statuses.end(),
    livo::ExposureTimingAttestStatus::READY);
  const auto stale_count = std::count(
    statuses.begin(),
    statuses.end(),
    livo::ExposureTimingAttestStatus::
    STALE_OR_REPLAYED_SOURCE_SEQUENCE);
  EXPECT_EQ(ready_count, 1);
  EXPECT_EQ(stale_count, static_cast<int>(thread_count - 1U));
}

TEST(
  ExposureTimingAttestationTest,
  AttestationRevalidationRejectsReceiptOvertakenDuringAllocation)
{
  auto rig = makeFixtureRig(fixtureDocument());
  ASSERT_TRUE(rig.has_value());
  const auto older =
    rig->source->issue(2U, 10.0, std::nullopt);
  const auto newer =
    rig->source->issue(2U, 11.0, std::nullopt);
  ASSERT_TRUE(older.ready());
  ASSERT_TRUE(newer.ready());
  livo::test::
  pauseNextExposureTimingAttestationBeforeRevalidation();
  livo::ExposureTimingAttestResult delayed;
  std::thread thread(
    [&]()
    {
      delayed = rig->timing->attest(*older.receipt);
    });
  const bool paused =
    livo::test::
    waitForExposureTimingAttestationPausedBeforeRevalidation();
  if (!paused)
  {
    livo::test::
    releaseExposureTimingAttestationBeforeRevalidation();
    thread.join();
    FAIL() << "attestation did not reach revalidation seam";
  }
  const auto winner =
    rig->timing->attest(*newer.receipt);
  livo::test::
  releaseExposureTimingAttestationBeforeRevalidation();
  thread.join();

  ASSERT_TRUE(winner.ready());
  EXPECT_EQ(
    delayed.status,
    livo::ExposureTimingAttestStatus::
    STALE_OR_REPLAYED_SOURCE_SEQUENCE);
  ASSERT_TRUE(winner.attestation->audit().has_value());
  EXPECT_EQ(
    winner.attestation->audit()->attestationSequence(), 1U);
}

TEST(
  ExposureTimingAttestationTest,
  AttestationAndIdentityOutliveBothAuthorities)
{
  auto rig = makeFixtureRig(fixtureDocument());
  ASSERT_TRUE(rig.has_value());
  const auto frame =
    rig->source->issue(1U, 10.0, std::nullopt);
  ASSERT_TRUE(frame.ready());
  const auto timing_identity = rig->timing->identity();
  const auto issued =
    rig->timing->attest(*frame.receipt);
  ASSERT_TRUE(issued.ready());
  const auto survivor = *issued.attestation;
  rig->timing.reset();
  rig->source.reset();

  EXPECT_TRUE(timing_identity.valid());
  EXPECT_TRUE(survivor.valid());
  EXPECT_TRUE(survivor.issuedBy(timing_identity));
  EXPECT_TRUE(
    survivor.sourceFrameIdentity().sameFrame(
      frame.receipt->frameIdentity()));
  ASSERT_TRUE(survivor.audit().has_value());
  EXPECT_EQ(survivor.audit()->sourceId(), "source-1");
}

TEST(
  ExposureTimingAttestationTest,
  AllocationFailuresDoNotConsumeCalibrationOrSequences)
{
  const auto document = fixtureDocument();
  livo::test::failNextExposureTimingCalibrationAllocation();
  EXPECT_EQ(
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document).status,
    livo::ExposureTimingCalibrationStatus::RESOURCE_FAILURE);
  auto minted =
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(document);
  ASSERT_TRUE(minted.ready());

  livo::test::
  failNextExposureTimingSourceAuthorityFactoryAllocation();
  EXPECT_EQ(
    livo::ExposureTimingSourceAuthority::create(
      sourceConfiguration(document)).status,
    livo::ExposureTimingSourceAuthorityCreateStatus::
    RESOURCE_FAILURE);
  auto source =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(document));
  ASSERT_TRUE(source.ready());

  livo::test::failNextExposureFrameTimingReceiptAllocation();
  EXPECT_EQ(
    source.authority->issue(
      3U, 10.0, std::nullopt).status,
    livo::ExposureFrameTimingIssueStatus::RESOURCE_FAILURE);
  const auto frame =
    source.authority->issue(3U, 10.0, std::nullopt);
  ASSERT_TRUE(frame.ready());
  EXPECT_EQ(frame.receipt->sourceSequence(), 1U);

  livo::test::failNextExposureTimingAuthorityFactoryAllocation();
  EXPECT_EQ(
    livo::ExposureTimingAuthority::create(
      *minted.calibration,
      source.authority->identity()).status,
    livo::ExposureTimingAuthorityCreateStatus::
    RESOURCE_FAILURE);
  auto timing =
    livo::ExposureTimingAuthority::create(
    *minted.calibration,
    source.authority->identity());
  ASSERT_TRUE(timing.ready());

  livo::test::failNextExposureTimingAttestationAllocation();
  EXPECT_EQ(
    timing.authority->attest(*frame.receipt).status,
    livo::ExposureTimingAttestStatus::RESOURCE_FAILURE);
  const auto retry =
    timing.authority->attest(*frame.receipt);
  ASSERT_TRUE(retry.ready());
  ASSERT_TRUE(retry.attestation->audit().has_value());
  EXPECT_EQ(
    retry.attestation->audit()->attestationSequence(), 1U);
}

TEST(
  ExposureTimingAttestationTest,
  TimingAuthorityFailsClosedBeforeInternalSequenceWrap)
{
  auto rig = makeFixtureRig(fixtureDocument());
  ASSERT_TRUE(rig.has_value());
  const auto frame =
    rig->source->issue(0U, 10.0, std::nullopt);
  ASSERT_TRUE(frame.ready());
  livo::test::exhaustNextExposureTimingAttestationSequence();
  EXPECT_EQ(
    rig->timing->attest(*frame.receipt).status,
    livo::ExposureTimingAttestStatus::AUTHORITY_EXHAUSTED);
  EXPECT_EQ(
    rig->timing->attest(*frame.receipt).status,
    livo::ExposureTimingAttestStatus::AUTHORITY_EXHAUSTED);
}

TEST(
  ExposureTimingAttestationTest,
  OpaqueTokensAreConstantTimeValuesWithOwnerIdentity)
{
  auto minted =
    livo::test::ExposureTimingCalibrationTestAccess::
    mintFixture(fixtureDocument());
  ASSERT_TRUE(minted.ready());
  auto calibration_copy = *minted.calibration;
  auto calibration_move = std::move(calibration_copy);
  EXPECT_FALSE(calibration_copy.valid());
  EXPECT_TRUE(calibration_move.valid());

  auto source =
    livo::ExposureTimingSourceAuthority::create(
    sourceConfiguration(fixtureDocument()));
  ASSERT_TRUE(source.ready());
  auto source_identity = source.authority->identity();
  auto source_identity_copy = source_identity;
  auto source_identity_move = std::move(source_identity_copy);
  EXPECT_FALSE(source_identity_copy.valid());
  EXPECT_TRUE(
    source_identity.sameAuthority(source_identity_move));

  const auto frame =
    source.authority->issue(0U, 10.0, std::nullopt);
  ASSERT_TRUE(frame.ready());
  auto receipt_copy = *frame.receipt;
  auto receipt_move = std::move(receipt_copy);
  EXPECT_FALSE(receipt_copy.valid());
  EXPECT_TRUE(receipt_move.sameFrame(*frame.receipt));
  auto frame_identity = receipt_move.frameIdentity();
  auto frame_identity_copy = frame_identity;
  auto frame_identity_move = std::move(frame_identity_copy);
  EXPECT_FALSE(frame_identity_copy.valid());
  EXPECT_TRUE(frame_identity.sameFrame(frame_identity_move));

  auto timing =
    livo::ExposureTimingAuthority::create(
    calibration_move, source_identity);
  ASSERT_TRUE(timing.ready());
  const auto result =
    timing.authority->attest(receipt_move);
  ASSERT_TRUE(result.ready());
  auto attestation_copy = *result.attestation;
  auto attestation_move = std::move(attestation_copy);
  EXPECT_FALSE(attestation_copy.valid());
  EXPECT_TRUE(
    attestation_move.sameAttestation(*result.attestation));
  auto audit_copy = *attestation_move.audit();
  auto audit_move = std::move(audit_copy);
  EXPECT_FALSE(audit_copy.valid());
  EXPECT_TRUE(audit_move.valid());
}

TEST(
  ExposureTimingAttestationTest,
  AttestationRetainsExactFrameAndCalibrationAfterAuthoritiesDie)
{
  std::optional<livo::ExposureTimingAttestation> survivor;
  std::optional<livo::ExposureFrameTimingIdentity>
    expected_frame_identity;
  {
    auto document = fixtureDocument();
    auto rig = makeFixtureRig(document);
    ASSERT_TRUE(rig.has_value());
    const auto frame =
      rig->source->issue(0U, 9.0, std::nullopt);
    ASSERT_TRUE(frame.ready());
    expected_frame_identity =
      frame.receipt->frameIdentity();
    const auto result =
      rig->timing->attest(*frame.receipt);
    ASSERT_TRUE(result.ready());
    survivor = *result.attestation;
    document.artifact_id = "mutated";
    document.cameras[0U].source_binding.source_id = "mutated";
  }

  ASSERT_TRUE(survivor.has_value());
  ASSERT_TRUE(expected_frame_identity.has_value());
  EXPECT_TRUE(survivor->valid());
  EXPECT_TRUE(
    survivor->sourceFrameIdentity().sameFrame(
      *expected_frame_identity));
  ASSERT_TRUE(survivor->audit().has_value());
  EXPECT_EQ(survivor->audit()->artifactId(), "test-fixture-only");
  EXPECT_EQ(survivor->audit()->sourceId(), "source-0");
  EXPECT_NEAR(
    survivor->effectiveExposureTime()->seconds(),
    9.010, 1.0e-14);
}

}  // namespace
