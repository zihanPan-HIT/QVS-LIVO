#include "trusted_photometric_calibration_loader.h"

#include <openssl/evp.h>

#include <json/json.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace livo
{
namespace
{

constexpr char kTestTrustDomain[] =
  "livo-core-photometric-test-v1";
constexpr char kTestPublicKeyHex[] =
  "d75a980182b10ab7d54bfed3c964073a0"
  "ee172f3daa62325af021a68f707511a";
constexpr char kTestPublicKeySha256[] =
  "21fe31dfa154a261626bf854046fd2271"
  "b7bed4b6abe45aa58877ef47f9721b9";
constexpr char kRuntimePayloadFilename[] =
  "photometric_calibration_runtime_payload.json";
constexpr char kRuntimeSignatureFilename[] =
  "photometric_calibration_runtime_signature.json";
constexpr char kReceiptFilename[] =
  "photometric_calibration_receipt.json";
constexpr char kReceiptHashScheme[] =
  "sha256_runtime_payload_lf_ed25519_signature_v1";
constexpr unsigned char kReceiptHashDomainSeparator[] =
  "livo_core.photometric_calibration.receipt_hash.v1\0";

static_assert(
  std::is_same_v<
    decltype(TrustedPhotometricCalibrationLoader::load(
      std::declval<const std::filesystem::path &>())),
    TrustedPhotometricCalibrationLoadResult>);
static_assert(
  !std::is_default_constructible_v<
    TrustedPhotometricCalibrationLoader>);
static_assert(
  !std::is_default_constructible_v<
    TrustedRawIntensityNoiseCapability>);
static_assert(
  !std::is_default_constructible_v<
    TrustedPhotometricCalibrationBundle>);
static_assert(
  !std::is_copy_assignable_v<
    TrustedRawIntensityNoiseCapability>);
static_assert(
  !std::is_move_assignable_v<
    TrustedRawIntensityNoiseCapability>);
static_assert(
  !std::is_copy_assignable_v<
    TrustedPhotometricCalibrationBundle>);
static_assert(
  !std::is_move_assignable_v<
    TrustedPhotometricCalibrationBundle>);

std::string compactJson(const Json::Value & value)
{
  Json::StreamWriterBuilder builder;
  builder["commentStyle"] = "None";
  builder["indentation"] = "";
  builder["emitUTF8"] = true;
  return Json::writeString(builder, value) + "\n";
}

std::string sha256Hex(const std::string & bytes)
{
  std::array<unsigned char, 32U> digest{};
  unsigned int digest_size = 0U;
  EXPECT_EQ(
    EVP_Digest(
      bytes.data(),
      bytes.size(),
      digest.data(),
      &digest_size,
      EVP_sha256(),
      nullptr),
    1);
  EXPECT_EQ(digest_size, digest.size());
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const unsigned char byte : digest)
  {
    stream << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return stream.str();
}

std::vector<unsigned char> fromHex(const std::string & encoded)
{
  std::vector<unsigned char> result(encoded.size() / 2U, 0U);
  for (std::size_t index = 0U; index < result.size(); ++index)
  {
    result[index] = static_cast<unsigned char>(
      std::stoul(encoded.substr(index * 2U, 2U), nullptr, 16));
  }
  return result;
}

struct TestSignature
{
  std::array<unsigned char, 64U> raw{};
  std::string base64;
};

TestSignature signPayload(const std::string & payload)
{
  const std::vector<unsigned char> private_seed = fromHex(
    "9d61b19deffd5a60ba844af492ec2cc4"
    "4449c5697b326919703bac031cae7f60");
  EVP_PKEY * const raw_key = EVP_PKEY_new_raw_private_key_ex(
    nullptr,
    "ED25519",
    nullptr,
    private_seed.data(),
    private_seed.size());
  EXPECT_NE(raw_key, nullptr);
  std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
    raw_key, &EVP_PKEY_free);
  EVP_MD_CTX * const raw_context = EVP_MD_CTX_new();
  EXPECT_NE(raw_context, nullptr);
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
    raw_context, &EVP_MD_CTX_free);
  EXPECT_EQ(
    EVP_DigestSignInit(
      context.get(), nullptr, nullptr, nullptr, key.get()),
    1);
  TestSignature signature;
  std::size_t signature_size = signature.raw.size();
  EXPECT_EQ(
    EVP_DigestSign(
      context.get(),
      signature.raw.data(),
      &signature_size,
      reinterpret_cast<const unsigned char *>(payload.data()),
      payload.size()),
    1);
  EXPECT_EQ(signature_size, signature.raw.size());
  signature.base64.assign(
    4U * ((signature_size + 2U) / 3U), '\0');
  const int encoded_size = EVP_EncodeBlock(
    reinterpret_cast<unsigned char *>(signature.base64.data()),
    signature.raw.data(),
    static_cast<int>(signature_size));
  EXPECT_EQ(
    encoded_size, static_cast<int>(signature.base64.size()));
  return signature;
}

std::array<unsigned char, 8U> bigEndianLength(
  const std::size_t size)
{
  std::array<unsigned char, 8U> encoded{};
  std::uint64_t value = static_cast<std::uint64_t>(size);
  for (std::size_t index = encoded.size(); index > 0U; --index)
  {
    encoded[index - 1U] =
      static_cast<unsigned char>(value & 0xffU);
    value >>= 8U;
  }
  return encoded;
}

std::string receiptHash(
  const std::string & payload,
  const std::array<unsigned char, 64U> & signature)
{
  const std::array<unsigned char, 8U> payload_length =
    bigEndianLength(payload.size());
  const std::array<unsigned char, 8U> signature_length =
    bigEndianLength(signature.size());
  std::string preimage(
    reinterpret_cast<const char *>(kReceiptHashDomainSeparator),
    sizeof(kReceiptHashDomainSeparator) - 1U);
  preimage.append(
    reinterpret_cast<const char *>(payload_length.data()),
    payload_length.size());
  preimage.append(payload);
  preimage.append(
    reinterpret_cast<const char *>(signature_length.data()),
    signature_length.size());
  preimage.append(
    reinterpret_cast<const char *>(signature.data()),
    signature.size());
  return sha256Hex(preimage);
}

[[maybe_unused]] std::array<unsigned char, 64U> decodeSignature(
  const std::string & encoded)
{
  std::array<unsigned char, 66U> decoded{};
  const int decoded_size = EVP_DecodeBlock(
    decoded.data(),
    reinterpret_cast<const unsigned char *>(encoded.data()),
    static_cast<int>(encoded.size()));
  EXPECT_EQ(decoded_size, 66);
  std::array<unsigned char, 64U> signature{};
  std::copy_n(decoded.begin(), signature.size(), signature.begin());
  return signature;
}

class SyntheticBundle
{
public:
  explicit SyntheticBundle(const bool trusted_noise_schema = false)
  {
    static std::uint64_t sequence = 0U;
    const auto ticks =
      std::chrono::steady_clock::now().time_since_epoch().count();
    directory_ = std::filesystem::temp_directory_path() /
      (
        "livo-trusted-photometric-" +
        std::to_string(ticks) + "-" +
        std::to_string(++sequence));
    EXPECT_TRUE(std::filesystem::create_directory(directory_));
    buildDocuments(trusted_noise_schema);
    publishSignedDocuments();
  }

  ~SyntheticBundle()
  {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  SyntheticBundle(const SyntheticBundle &) = delete;
  SyntheticBundle & operator=(const SyntheticBundle &) = delete;

  const std::filesystem::path & directory() const
  {
    return directory_;
  }

  Json::Value & payload()
  {
    return payload_;
  }

  Json::Value & detached()
  {
    return detached_;
  }

  Json::Value & receipt()
  {
    return receipt_;
  }

  std::filesystem::path path(const std::string & filename) const
  {
    return directory_ / filename;
  }

  void write(const std::string & filename, const std::string & bytes)
  {
    std::ofstream stream(
      path(filename), std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(stream.good());
    stream.write(
      bytes.data(), static_cast<std::streamsize>(bytes.size()));
    ASSERT_TRUE(stream.good());
  }

  void publishSignedDocuments()
  {
    const std::string payload_bytes = compactJson(payload_);
    const TestSignature signature = signPayload(payload_bytes);
    const std::string receipt_sha256 =
      receiptHash(payload_bytes, signature.raw);
    detached_["schema_version"] = 1;
    detached_["signature_type"] =
      "four_camera_photometric_calibration_runtime_signature";
    detached_["algorithm"] = "ed25519";
    detached_["solver"] = payload_["solver"];
    detached_["trust_domain"] =
      payload_["trust_root"]["trust_domain"];
    detached_["payload_file"] = kRuntimePayloadFilename;
    detached_["payload_sha256"] = sha256Hex(payload_bytes);
    detached_["public_key_sha256"] =
      payload_["trust_root"]["public_key_sha256"];
    detached_["receipt_hash_scheme"] =
      payload_["receipt_hash_scheme"];
    detached_["receipt_sha256"] = receipt_sha256;
    detached_["signature_base64"] = signature.base64;
    receipt_ = payload_;
    receipt_["signature"]["algorithm"] = "ed25519";
    receipt_["signature"]["public_key_sha256"] =
      detached_["public_key_sha256"];
    receipt_["signature"]["signature_base64"] = signature.base64;
    receipt_["receipt_sha256"] = receipt_sha256;
    write(kRuntimePayloadFilename, payload_bytes);
    write(kRuntimeSignatureFilename, compactJson(detached_));
    write(kReceiptFilename, compactJson(receipt_));
  }

private:
  void buildDocuments(const bool trusted_noise_schema)
  {
    const Json::Value trust_root = [] {
        Json::Value value(Json::objectValue);
        value["trust_domain"] = kTestTrustDomain;
        value["public_key_sha256"] = kTestPublicKeySha256;
        return value;
      }();
    Json::Value solver(Json::objectValue);
    solver["name"] =
      "livo_core_four_camera_photometric_calibration";
    solver["version"] = "2.0.0";
    const std::string source_bundle_hash(64U, 'a');

    Json::Value config(Json::objectValue);
    config["schema_version"] = trusted_noise_schema ? 2 : 1;
    config["calibration_set_id"] = "trusted-synthetic-four-camera";
    config["dataset_scope"] = Json::Value(Json::arrayValue);
    config["status"] = "available";
    config["frame_metadata"] = Json::Value(Json::objectValue);
    config["source_bundle_file"] =
      "photometric_calibration_source_bundle.json";
    config["source_bundle_sha256"] = source_bundle_hash;
    config["calibration_receipt_file"] = kReceiptFilename;
    config["runtime_payload_file"] = kRuntimePayloadFilename;
    config["runtime_signature_file"] = kRuntimeSignatureFilename;
    config["trust_root"] = trust_root;

    Json::Value artifacts(Json::objectValue);
    Json::Value cameras(Json::arrayValue);
    for (int camera_id = 0; camera_id < 4; ++camera_id)
    {
      const std::string prefix =
        "camera_" + std::to_string(camera_id);
      const std::string lut_filename =
        prefix + "_inverse_response_lut.txt";
      const std::string vignetting_filename =
        prefix + "_vignetting_transmission.txt";
      std::ostringstream lut_stream;
      lut_stream << std::setprecision(17);
      for (std::size_t index = 0U; index < 256U; ++index)
      {
        lut_stream <<
          static_cast<double>(index) / 255.0 <<
          (index + 1U == 256U ? '\n' : ' ');
      }
      const std::string lut = lut_stream.str();
      const std::string vignetting = "1 0.9 0.8 1\n";
      write(lut_filename, lut);
      write(vignetting_filename, vignetting);
      artifacts[lut_filename] = sha256Hex(lut);
      artifacts[vignetting_filename] = sha256Hex(vignetting);

      Json::Value camera(Json::objectValue);
      camera["camera_id"] = camera_id;
      camera["name"] =
        std::array<const char *, 4U>{
          "front", "left", "back", "right"}[
          static_cast<std::size_t>(camera_id)];
      camera["status"] = "calibrated";
      camera["provenance"] = "laboratory_measured";
      camera["source_id"] =
        "source-bundle-sha256:" + source_bundle_hash;
      camera["source_bundle_sha256"] = source_bundle_hash;
      camera["image_width"] = 2;
      camera["image_height"] = 2;
      camera["black_level"] = 0.0;
      camera["white_level"] = 255.0;
      camera["inverse_response_lut_file"] = lut_filename;
      camera["vignetting_transmission_file"] =
        vignetting_filename;
      camera["radiance_scale"] =
        1.0 + 0.05 * static_cast<double>(camera_id);
      camera["radiance_offset"] = 0.0;
      camera["covariance_scale"] = 1.0;
      if (trusted_noise_schema)
      {
        Json::Value domain(Json::objectValue);
        domain["bit_depth"] = 8;
        domain["unit"] = "canonical_gray_code_value";
        domain["canonicalization"] =
          camera_id % 2 == 0 ?
          "mono8_identity_v1" :
          "rgb8_bt601_luma_uint8_v1";
        domain["image_width"] = camera["image_width"];
        domain["image_height"] = camera["image_height"];
        camera["raw_intensity_domain"] = domain;

        Json::Value noise(Json::objectValue);
        noise["model"] = "constant_sigma_v1";
        noise["standard_deviation_code_value"] =
          0.75 + 0.25 * static_cast<double>(camera_id);
        noise["provenance"] = "laboratory_measured";
        noise["estimator_revision"] = "temporal_mad_p99_v1";
        noise["source_sample_count"] = 64;
        camera["raw_intensity_noise"] = noise;
      }
      cameras.append(camera);
    }
    config["cameras"] = cameras;

    payload_["schema_version"] = trusted_noise_schema ? 2 : 1;
    payload_["receipt_type"] =
      "four_camera_photometric_calibration";
    payload_["receipt_hash_scheme"] = kReceiptHashScheme;
    payload_["solver"] = solver;
    payload_["calibration_set_id"] =
      config["calibration_set_id"];
    payload_["declared_source_id"] = "synthetic-lab-run";
    payload_["manifest_sha256"] = std::string(64U, 'b');
    payload_["source_bundle_file"] =
      config["source_bundle_file"];
    payload_["source_bundle_sha256"] =
      config["source_bundle_sha256"];
    payload_["source_counts"] = Json::Value(Json::objectValue);
    payload_["frame_metadata"] = config["frame_metadata"];
    payload_["cameras"] = cameras;
    payload_["calibration_config_payload"] = config;
    payload_["calibration_config_sha256"] = std::string(64U, 'c');
    payload_["held_out_metrics"] =
      Json::Value(Json::objectValue);
    payload_["thresholds"] = Json::Value(Json::objectValue);
    payload_["artifacts_sha256"] = artifacts;
    payload_["integrity_scope"] =
      "source_snapshot_config_solver_metrics_numeric_artifacts";
    payload_["trust_boundary"] =
      "ed25519_signature_requires_explicit_external_public_key_pin";
    payload_["trust_root"] = trust_root;
  }

  std::filesystem::path directory_;
  Json::Value payload_{Json::objectValue};
  Json::Value detached_{Json::objectValue};
  Json::Value receipt_{Json::objectValue};
};

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  ReceiptHashV1MatchesCrossLanguageGolden)
{
  const std::string payload =
    "{\"receipt_hash_scheme\":"
    "\"sha256_runtime_payload_lf_ed25519_signature_v1\"}\n";
  std::array<unsigned char, 64U> signature{};
  for (std::size_t index = 0U; index < signature.size(); ++index)
  {
    signature[index] = static_cast<unsigned char>(index);
  }

  EXPECT_EQ(
    receiptHash(payload, signature),
    "dfe47714ab74419a2c7a32aa7507909b"
    "4c7f4f2a291aa82e5dced94175fae74f");
}

#ifdef LIVO_TEST_EXPECT_TRUSTED_PHOTOMETRIC_ROOT

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  TrustedSyntheticFourCameraBundleLoadsAndLinearizes)
{
  SyntheticBundle bundle;

  const TrustedPhotometricCalibrationLoadResult result =
    TrustedPhotometricCalibrationLoader::load(bundle.directory());

  ASSERT_TRUE(result.trustedReady()) << result.diagnostic;
  EXPECT_TRUE(result.trustedCalibrationReady());
  EXPECT_FALSE(result.trustedNoiseReady());
  EXPECT_EQ(result.trustedNoiseCapabilityCount(), 0U);
  EXPECT_FALSE(result.trusted_noise_bundle.has_value());
  EXPECT_EQ(
    result.status,
    TrustedPhotometricCalibrationLoadStatus::kLoaded);
  ASSERT_NE(result.calibration_revision, 0U);
  ASSERT_TRUE(result.calibration.has_value());
  for (std::size_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    EXPECT_EQ(
      result.calibration->cameraCalibrationStatus(camera_id),
      PhotometricStatus::kValid);
    EXPECT_EQ(
      result.calibration->cameraCalibrationRevision(camera_id),
      result.calibration_revision);
  }
  PhotometricSample sample;
  sample.raw_value = 128.0;
  sample.pixel_x = 0.5;
  sample.pixel_y = 0.5;
  PhotometricFrameMetadata metadata;
  metadata.exposure_sec = 0.01;
  metadata.sensor_gain = 1.0;
  const PhotometricSampleLinearization robust =
    result.calibration->linearizeSample(
      3U,
      metadata,
      sample,
      PhotometricMode::kRobustAffineExperimental);
  EXPECT_TRUE(robust.valid());
  EXPECT_EQ(
    result.calibration->linearizeSample(
      3U,
      metadata,
      sample,
      PhotometricMode::kCalibratedDirect).status,
    PhotometricStatus::kUntrustedExposureGain);
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  SignedSchemaV2AtomicallyIssuesFourNoiseCapabilities)
{
  SyntheticBundle bundle(true);

  const TrustedPhotometricCalibrationLoadResult result =
    TrustedPhotometricCalibrationLoader::load(bundle.directory());

  ASSERT_TRUE(result.trustedCalibrationReady()) << result.diagnostic;
  ASSERT_TRUE(result.trustedNoiseReady()) << result.diagnostic;
  ASSERT_TRUE(result.trusted_noise_bundle.has_value());
  EXPECT_EQ(result.trustedNoiseCapabilityCount(), 4U);
  EXPECT_EQ(
    result.trusted_noise_bundle->calibrationRevision(),
    result.calibration_revision);
  ASSERT_TRUE(result.calibration.has_value());
  EXPECT_TRUE(
    result.trusted_noise_bundle->accepts(*result.calibration));

  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const TrustedRawIntensityNoiseCapability * const capability =
      result.trusted_noise_bundle->noiseCapability(camera_id);
    ASSERT_NE(capability, nullptr);
    EXPECT_TRUE(capability->valid());
    EXPECT_EQ(capability->cameraId(), camera_id);
    EXPECT_EQ(
      capability->calibrationRevision(),
      result.calibration_revision);
    EXPECT_EQ(capability->domain().bit_depth, 8U);
    EXPECT_EQ(
      capability->domain().unit,
      RawIntensityUnit::kCanonicalGrayCodeValue);
    EXPECT_EQ(
      capability->domain().canonicalization,
      camera_id % 2U == 0U ?
      RawIntensityCanonicalization::kMono8IdentityV1 :
      RawIntensityCanonicalization::
      kRgb8Bt601LumaUint8V1);
    EXPECT_EQ(capability->domain().image_width, 2);
    EXPECT_EQ(capability->domain().image_height, 2);
    EXPECT_TRUE(
      capability->validFor(
        camera_id,
        result.calibration_revision,
        capability->domain()));
    EXPECT_FALSE(
      capability->validFor(
        camera_id,
        result.calibration_revision + 1U,
        capability->domain()));
    EXPECT_DOUBLE_EQ(
      capability->standardDeviationCodeValue(),
      0.75 + 0.25 * static_cast<double>(camera_id));
    EXPECT_EQ(
      capability->sourceId(),
      "source-bundle-sha256:" + std::string(64U, 'a'));
    EXPECT_EQ(capability->provenance(), "laboratory_measured");
    EXPECT_EQ(capability->model(), "constant_sigma_v1");
    EXPECT_EQ(
      capability->estimatorRevision(),
      "temporal_mad_p99_v1");
    EXPECT_EQ(capability->sourceSampleCount(), 64U);
  }
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  SignedSchemaV2RevisionBindsExactPayloadAndRejectsCrossBundleUse)
{
  SyntheticBundle first_bundle(true);
  SyntheticBundle second_bundle(true);
  Json::Value & second_config_camera =
    second_bundle.payload()["calibration_config_payload"][
    "cameras"][0];
  second_config_camera["raw_intensity_noise"][
    "standard_deviation_code_value"] = 0.875;
  second_bundle.payload()["cameras"][0] = second_config_camera;
  ASSERT_NE(
    compactJson(first_bundle.payload()),
    compactJson(second_bundle.payload()));
  second_bundle.publishSignedDocuments();

  const TrustedPhotometricCalibrationLoadResult first =
    TrustedPhotometricCalibrationLoader::load(
    first_bundle.directory());
  const TrustedPhotometricCalibrationLoadResult second =
    TrustedPhotometricCalibrationLoader::load(
    second_bundle.directory());

  ASSERT_TRUE(first.trustedNoiseReady()) << first.diagnostic;
  ASSERT_TRUE(second.trustedNoiseReady()) << second.diagnostic;
  ASSERT_NE(first.calibration_revision, 0U);
  ASSERT_NE(second.calibration_revision, 0U);
  ASSERT_NE(
    first.calibration_revision,
    second.calibration_revision);
  ASSERT_TRUE(first.calibration.has_value());
  ASSERT_TRUE(second.calibration.has_value());
  ASSERT_TRUE(first.trusted_noise_bundle.has_value());
  ASSERT_TRUE(second.trusted_noise_bundle.has_value());
  EXPECT_EQ(first.trustedNoiseCapabilityCount(), 4U);
  EXPECT_EQ(second.trustedNoiseCapabilityCount(), 4U);
  EXPECT_TRUE(
    first.trusted_noise_bundle->accepts(*first.calibration));
  EXPECT_TRUE(
    second.trusted_noise_bundle->accepts(*second.calibration));
  EXPECT_FALSE(
    first.trusted_noise_bundle->accepts(*second.calibration));
  EXPECT_FALSE(
    second.trusted_noise_bundle->accepts(*first.calibration));

  for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
  {
    const TrustedRawIntensityNoiseCapability * const
      first_capability =
      first.trusted_noise_bundle->noiseCapability(camera_id);
    const TrustedRawIntensityNoiseCapability * const
      second_capability =
      second.trusted_noise_bundle->noiseCapability(camera_id);
    ASSERT_NE(first_capability, nullptr);
    ASSERT_NE(second_capability, nullptr);
    EXPECT_EQ(
      first_capability->calibrationRevision(),
      first.calibration_revision);
    EXPECT_EQ(
      second_capability->calibrationRevision(),
      second.calibration_revision);
    EXPECT_TRUE(
      first_capability->validFor(
        camera_id,
        first.calibration_revision,
        first_capability->domain()));
    EXPECT_TRUE(
      second_capability->validFor(
        camera_id,
        second.calibration_revision,
        second_capability->domain()));
    EXPECT_FALSE(
      first_capability->validFor(
        camera_id,
        second.calibration_revision,
        first_capability->domain()));
    EXPECT_FALSE(
      second_capability->validFor(
        camera_id,
        first.calibration_revision,
        second_capability->domain()));
  }
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  SignedSchemaV2InvalidEvidenceKeepsCalibrationButIssuesNothing)
{
  constexpr int kMutationCount = 12;
  for (int mutation = 0; mutation < kMutationCount; ++mutation)
  {
    SCOPED_TRACE(mutation);
    SyntheticBundle bundle(true);
    Json::Value & config_camera =
      bundle.payload()["calibration_config_payload"]["cameras"][2];
    switch (mutation)
    {
      case 0:
        config_camera["raw_intensity_noise"][
          "standard_deviation_code_value"] = 0.0;
        break;
      case 1:
        config_camera["raw_intensity_noise"][
          "standard_deviation_code_value"] = "nan";
        break;
      case 2:
        config_camera["raw_intensity_domain"]["unit"] =
          "normalized_radiance";
        break;
      case 3:
        config_camera["raw_intensity_domain"]["bit_depth"] = 16;
        break;
      case 4:
        config_camera["raw_intensity_domain"]["canonicalization"] =
          "implementation_defined";
        break;
      case 5:
        config_camera["raw_intensity_domain"]["image_width"] = 3;
        break;
      case 6:
        config_camera["raw_intensity_noise"]["model"] =
          "caller_selected";
        break;
      case 7:
        config_camera["raw_intensity_noise"]["provenance"] =
          "synthetic_default";
        break;
      case 8:
        config_camera["raw_intensity_noise"]["estimator_revision"] =
          "unknown";
        break;
      case 9:
        config_camera["raw_intensity_noise"]["source_sample_count"] =
          0;
        break;
      case 10:
        config_camera["raw_intensity_noise"].removeMember(
          "source_sample_count");
        break;
      case 11:
        config_camera["raw_intensity_noise"]["default_sigma"] = 1.0;
        break;
    }
    bundle.payload()["cameras"][2] = config_camera;
    bundle.publishSignedDocuments();

    const TrustedPhotometricCalibrationLoadResult result =
      TrustedPhotometricCalibrationLoader::load(bundle.directory());

    EXPECT_TRUE(result.trustedCalibrationReady())
      << result.diagnostic;
    EXPECT_TRUE(result.calibration.has_value());
    EXPECT_FALSE(result.trustedNoiseReady());
    EXPECT_EQ(result.trustedNoiseCapabilityCount(), 0U);
    EXPECT_FALSE(result.trusted_noise_bundle.has_value());
  }
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  SignedSchemaV2MissingOrUnequalCameraEvidenceFailsClosed)
{
  SyntheticBundle missing(true);
  missing.payload()["calibration_config_payload"]["cameras"][1].
    removeMember("raw_intensity_noise");
  missing.payload()["cameras"][1].
    removeMember("raw_intensity_noise");
  missing.publishSignedDocuments();

  const TrustedPhotometricCalibrationLoadResult missing_result =
    TrustedPhotometricCalibrationLoader::load(missing.directory());

  EXPECT_EQ(
    missing_result.status,
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);
  EXPECT_FALSE(missing_result.trustedCalibrationReady());
  EXPECT_FALSE(missing_result.trustedNoiseReady());
  EXPECT_EQ(missing_result.trustedNoiseCapabilityCount(), 0U);
  EXPECT_FALSE(missing_result.trusted_noise_bundle.has_value());

  SyntheticBundle unequal(true);
  unequal.payload()["calibration_config_payload"]["cameras"][1][
    "raw_intensity_noise"]["standard_deviation_code_value"] = 1.5;
  unequal.publishSignedDocuments();

  const TrustedPhotometricCalibrationLoadResult unequal_result =
    TrustedPhotometricCalibrationLoader::load(unequal.directory());

  EXPECT_EQ(
    unequal_result.status,
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);
  EXPECT_FALSE(unequal_result.trustedCalibrationReady());
  EXPECT_FALSE(unequal_result.trustedNoiseReady());
  EXPECT_EQ(unequal_result.trustedNoiseCapabilityCount(), 0U);
  EXPECT_FALSE(unequal_result.trusted_noise_bundle.has_value());
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  ExactPayloadByteTamperFailsSignature)
{
  SyntheticBundle bundle(true);
  std::ifstream input(
    bundle.path(kRuntimePayloadFilename), std::ios::binary);
  std::string bytes(
    (std::istreambuf_iterator<char>(input)),
    std::istreambuf_iterator<char>());
  ASSERT_FALSE(bytes.empty());
  ASSERT_EQ(bytes.back(), '\n');
  bytes.insert(bytes.size() - 1U, 1U, ' ');
  bundle.write(kRuntimePayloadFilename, bytes);
  bundle.detached()["payload_sha256"] = sha256Hex(bytes);
  const std::array<unsigned char, 64U> signature = decodeSignature(
    bundle.detached()["signature_base64"].asString());
  const std::string receipt_sha256 =
    receiptHash(bytes, signature);
  bundle.detached()["receipt_sha256"] = receipt_sha256;
  bundle.write(
    kRuntimeSignatureFilename, compactJson(bundle.detached()));
  bundle.receipt()["receipt_sha256"] = receipt_sha256;
  bundle.write(kReceiptFilename, compactJson(bundle.receipt()));

  const auto result =
    TrustedPhotometricCalibrationLoader::load(bundle.directory());

  EXPECT_EQ(
    result.status,
    TrustedPhotometricCalibrationLoadStatus::kSignatureInvalid);
  EXPECT_TRUE(result.structurally_valid);
  EXPECT_FALSE(result.calibration.has_value());
  EXPECT_FALSE(result.trustedCalibrationReady());
  EXPECT_FALSE(result.trustedNoiseReady());
  EXPECT_EQ(result.trustedNoiseCapabilityCount(), 0U);
  EXPECT_FALSE(result.trusted_noise_bundle.has_value());
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  AssetTamperAndSymlinkFailClosed)
{
  SyntheticBundle tampered;
  tampered.write(
    "camera_0_inverse_response_lut.txt", "0 0.5 1\n");
  const auto tampered_result =
    TrustedPhotometricCalibrationLoader::load(tampered.directory());
  EXPECT_EQ(
    tampered_result.status,
    TrustedPhotometricCalibrationLoadStatus::kArtifactInvalid);
  EXPECT_FALSE(tampered_result.calibration.has_value());

  SyntheticBundle linked;
  const std::filesystem::path external =
    linked.directory().parent_path() /
    (linked.directory().filename().string() + "-external.txt");
  {
    std::ofstream stream(external);
    stream << "0 0.5 1\n";
  }
  std::filesystem::remove(
    linked.path("camera_0_inverse_response_lut.txt"));
  std::filesystem::create_symlink(
    external,
    linked.path("camera_0_inverse_response_lut.txt"));
  const auto linked_result =
    TrustedPhotometricCalibrationLoader::load(linked.directory());
  EXPECT_EQ(
    linked_result.status,
    TrustedPhotometricCalibrationLoadStatus::kArtifactInvalid);
  EXPECT_FALSE(linked_result.calibration.has_value());
  std::filesystem::remove(external);
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  SignedTraversalExtraCameraAndTrustDomainFailClosed)
{
  SyntheticBundle traversal;
  traversal.payload()["calibration_config_payload"]["cameras"][0][
    "inverse_response_lut_file"] = "../escape.txt";
  traversal.payload()["cameras"][0][
    "inverse_response_lut_file"] = "../escape.txt";
  traversal.publishSignedDocuments();
  EXPECT_EQ(
    TrustedPhotometricCalibrationLoader::load(
      traversal.directory()).status,
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);

  SyntheticBundle extra_camera;
  extra_camera.payload()["calibration_config_payload"]["cameras"].append(
    extra_camera.payload()["calibration_config_payload"]["cameras"][0]);
  extra_camera.payload()["cameras"].append(
    extra_camera.payload()["cameras"][0]);
  extra_camera.publishSignedDocuments();
  EXPECT_EQ(
    TrustedPhotometricCalibrationLoader::load(
      extra_camera.directory()).status,
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);

  SyntheticBundle wrong_domain;
  wrong_domain.payload()["trust_root"]["trust_domain"] =
    "attacker-controlled-domain";
  wrong_domain.payload()["calibration_config_payload"]["trust_root"] =
    wrong_domain.payload()["trust_root"];
  wrong_domain.publishSignedDocuments();
  const auto wrong_domain_result =
    TrustedPhotometricCalibrationLoader::load(
      wrong_domain.directory());
  EXPECT_EQ(
    wrong_domain_result.status,
    TrustedPhotometricCalibrationLoadStatus::kTrustRootMismatch);
  EXPECT_TRUE(wrong_domain_result.structurally_valid);
  EXPECT_FALSE(wrong_domain_result.calibration.has_value());

  SyntheticBundle unknown_scheme;
  unknown_scheme.payload()["receipt_hash_scheme"] =
    "unknown_receipt_hash_scheme";
  unknown_scheme.publishSignedDocuments();
  EXPECT_EQ(
    TrustedPhotometricCalibrationLoader::load(
      unknown_scheme.directory()).status,
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  FixedFilesReceiptAndBundleDirectorySymlinkFailClosed)
{
  SyntheticBundle missing;
  std::filesystem::remove(
    missing.path(kRuntimeSignatureFilename));
  EXPECT_EQ(
    TrustedPhotometricCalibrationLoader::load(
      missing.directory()).status,
    TrustedPhotometricCalibrationLoadStatus::kIoFailure);

  SyntheticBundle receipt_tamper;
  receipt_tamper.receipt()["calibration_set_id"] = "other";
  receipt_tamper.write(
    kReceiptFilename, compactJson(receipt_tamper.receipt()));
  EXPECT_EQ(
    TrustedPhotometricCalibrationLoader::load(
      receipt_tamper.directory()).status,
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);

  SyntheticBundle real_bundle;
  const std::filesystem::path link =
    real_bundle.directory().parent_path() /
    (real_bundle.directory().filename().string() + "-link");
  std::filesystem::create_directory_symlink(
    real_bundle.directory(), link);
  EXPECT_EQ(
    TrustedPhotometricCalibrationLoader::load(link).status,
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundlePath);
  std::filesystem::remove(link);
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  ReceiptSelfHashTamperingFailsClosed)
{
  for (const bool synchronize_detached : {false, true})
  {
    for (const std::string & invalid_hash : {
        std::string(64U, '0'),
        std::string(64U, 'g'),
        std::string(63U, '0'),
        std::string(65U, '0'),
        std::string(64U, '2')})
    {
      SyntheticBundle bundle;
      bundle.receipt()["receipt_sha256"] = invalid_hash;
      bundle.write(
        kReceiptFilename, compactJson(bundle.receipt()));
      if (synchronize_detached)
      {
        bundle.detached()["receipt_sha256"] = invalid_hash;
        bundle.write(
          kRuntimeSignatureFilename,
          compactJson(bundle.detached()));
      }

      const auto result =
        TrustedPhotometricCalibrationLoader::load(bundle.directory());

      EXPECT_EQ(
        result.status,
        TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);
      EXPECT_FALSE(result.structurally_valid);
      EXPECT_FALSE(result.calibration.has_value());
    }
  }
}

#else

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  ProductionBuildWithoutTrustRootNeverSeals)
{
  SyntheticBundle bundle;

  const TrustedPhotometricCalibrationLoadResult result =
    TrustedPhotometricCalibrationLoader::load(bundle.directory());

  EXPECT_EQ(
    result.status,
    TrustedPhotometricCalibrationLoadStatus::kTrustRootUnavailable);
  EXPECT_TRUE(result.structurally_valid) << result.diagnostic;
  EXPECT_NE(result.calibration_revision, 0U);
  EXPECT_FALSE(result.calibration.has_value());
  EXPECT_FALSE(result.trustedReady());
  EXPECT_FALSE(result.trustedCalibrationReady());
  EXPECT_FALSE(result.trustedNoiseReady());
  EXPECT_EQ(result.trustedNoiseCapabilityCount(), 0U);
  EXPECT_FALSE(result.trusted_noise_bundle.has_value());
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  ProductionSchemaV2WithoutTrustRootNeverIssuesNoiseAuthority)
{
  SyntheticBundle bundle(true);

  const TrustedPhotometricCalibrationLoadResult result =
    TrustedPhotometricCalibrationLoader::load(bundle.directory());

  EXPECT_EQ(
    result.status,
    TrustedPhotometricCalibrationLoadStatus::kTrustRootUnavailable);
  EXPECT_TRUE(result.structurally_valid) << result.diagnostic;
  EXPECT_NE(result.calibration_revision, 0U);
  EXPECT_FALSE(result.calibration.has_value());
  EXPECT_FALSE(result.trustedReady());
  EXPECT_FALSE(result.trustedCalibrationReady());
  EXPECT_FALSE(result.trustedNoiseReady());
  EXPECT_EQ(result.trustedNoiseCapabilityCount(), 0U);
  EXPECT_FALSE(result.trusted_noise_bundle.has_value());
}

TEST(
  TrustedPhotometricCalibrationLoaderTest,
  ProductionBuildRejectsBadReceiptSelfHashStructurally)
{
  for (const bool synchronize_detached : {false, true})
  {
    SyntheticBundle bundle;
    bundle.receipt()["receipt_sha256"] = std::string(64U, '0');
    bundle.write(kReceiptFilename, compactJson(bundle.receipt()));
    if (synchronize_detached)
    {
      bundle.detached()["receipt_sha256"] = std::string(64U, '0');
      bundle.write(
        kRuntimeSignatureFilename,
        compactJson(bundle.detached()));
    }

    const TrustedPhotometricCalibrationLoadResult result =
      TrustedPhotometricCalibrationLoader::load(bundle.directory());

    EXPECT_EQ(
      result.status,
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract);
    EXPECT_FALSE(result.structurally_valid);
    EXPECT_FALSE(result.calibration.has_value());
  }
}

#endif

}  // namespace
}  // namespace livo
