#ifndef LIVO_TRUSTED_PHOTOMETRIC_EVIDENCE_TEST_SUPPORT_H_
#define LIVO_TRUSTED_PHOTOMETRIC_EVIDENCE_TEST_SUPPORT_H_

#include "trusted_photometric_calibration_loader.h"

#include <json/json.h>
#include <openssl/evp.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace livo::test_support
{
namespace photometric_evidence_detail
{

inline constexpr char kTrustDomain[] =
  "livo-core-photometric-test-v1";
inline constexpr char kPublicKeySha256[] =
  "21fe31dfa154a261626bf854046fd2271"
  "b7bed4b6abe45aa58877ef47f9721b9";
inline constexpr char kPayloadFilename[] =
  "photometric_calibration_runtime_payload.json";
inline constexpr char kSignatureFilename[] =
  "photometric_calibration_runtime_signature.json";
inline constexpr char kReceiptFilename[] =
  "photometric_calibration_receipt.json";
inline constexpr char kReceiptHashScheme[] =
  "sha256_runtime_payload_lf_ed25519_signature_v1";
inline constexpr unsigned char kReceiptHashDomainSeparator[] =
  "livo_core.photometric_calibration.receipt_hash.v1\0";
inline constexpr std::uint64_t kMaximumTestImagePixels =
  1024U * 1024U;

inline void require(
  const bool condition,
  const char * const diagnostic)
{
  if (!condition)
  {
    throw std::runtime_error(diagnostic);
  }
}

inline std::string compactJson(const Json::Value & value)
{
  Json::StreamWriterBuilder builder;
  builder["commentStyle"] = "None";
  builder["indentation"] = "";
  builder["emitUTF8"] = true;
  return Json::writeString(builder, value) + "\n";
}

inline std::string sha256Hex(const std::string & bytes)
{
  std::array<unsigned char, 32U> digest{};
  unsigned int digest_size = 0U;
  require(
    EVP_Digest(
      bytes.data(), bytes.size(), digest.data(),
      &digest_size, EVP_sha256(), nullptr) == 1,
    "test SHA-256 failed");
  require(
    digest_size == digest.size(),
    "test SHA-256 size mismatch");
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const unsigned char byte : digest)
  {
    stream <<
      std::setw(2) <<
      static_cast<unsigned int>(byte);
  }
  return stream.str();
}

inline std::vector<unsigned char> fromHex(
  const std::string & encoded)
{
  std::vector<unsigned char> result(encoded.size() / 2U, 0U);
  for (std::size_t index = 0U; index < result.size(); ++index)
  {
    result[index] = static_cast<unsigned char>(
      std::stoul(
        encoded.substr(index * 2U, 2U),
        nullptr, 16));
  }
  return result;
}

struct Signature
{
  std::array<unsigned char, 64U> raw{};
  std::string base64;
};

inline Signature signPayload(const std::string & payload)
{
  const std::vector<unsigned char> private_seed = fromHex(
    "9d61b19deffd5a60ba844af492ec2cc4"
    "4449c5697b326919703bac031cae7f60");
  EVP_PKEY * const raw_key =
    EVP_PKEY_new_raw_private_key_ex(
    nullptr, "ED25519", nullptr,
    private_seed.data(), private_seed.size());
  require(raw_key != nullptr, "test Ed25519 key failed");
  std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
    raw_key, &EVP_PKEY_free);
  EVP_MD_CTX * const raw_context = EVP_MD_CTX_new();
  require(
    raw_context != nullptr,
    "test Ed25519 context failed");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>
    context(raw_context, &EVP_MD_CTX_free);
  require(
    EVP_DigestSignInit(
      context.get(), nullptr, nullptr, nullptr,
      key.get()) == 1,
    "test Ed25519 init failed");
  Signature signature;
  std::size_t signature_size = signature.raw.size();
  require(
    EVP_DigestSign(
      context.get(), signature.raw.data(),
      &signature_size,
      reinterpret_cast<const unsigned char *>(
        payload.data()),
      payload.size()) == 1,
    "test Ed25519 sign failed");
  require(
    signature_size == signature.raw.size(),
    "test Ed25519 signature size mismatch");
  signature.base64.assign(
    4U * ((signature_size + 2U) / 3U), '\0');
  const int encoded_size = EVP_EncodeBlock(
    reinterpret_cast<unsigned char *>(
      signature.base64.data()),
    signature.raw.data(),
    static_cast<int>(signature_size));
  require(
    encoded_size ==
      static_cast<int>(signature.base64.size()),
    "test Ed25519 base64 failed");
  return signature;
}

inline std::array<unsigned char, 8U> bigEndianLength(
  const std::size_t size)
{
  std::array<unsigned char, 8U> encoded{};
  std::uint64_t value = static_cast<std::uint64_t>(size);
  for (std::size_t index = encoded.size();
       index > 0U; --index)
  {
    encoded[index - 1U] =
      static_cast<unsigned char>(value & 0xffU);
    value >>= 8U;
  }
  return encoded;
}

inline std::string receiptHash(
  const std::string & payload,
  const std::array<unsigned char, 64U> & signature)
{
  const auto payload_length =
    bigEndianLength(payload.size());
  const auto signature_length =
    bigEndianLength(signature.size());
  std::string preimage(
    reinterpret_cast<const char *>(
      kReceiptHashDomainSeparator),
    sizeof(kReceiptHashDomainSeparator) - 1U);
  preimage.append(
    reinterpret_cast<const char *>(
      payload_length.data()),
    payload_length.size());
  preimage.append(payload);
  preimage.append(
    reinterpret_cast<const char *>(
      signature_length.data()),
    signature_length.size());
  preimage.append(
    reinterpret_cast<const char *>(signature.data()),
    signature.size());
  return sha256Hex(preimage);
}

}  // namespace photometric_evidence_detail

enum class TrustedPhotometricEvidenceTestFailurePoint
{
  kNone,
  kAfterDirectoryCreation,
  kAfterFirstPublishedFile
};

class ScopedTemporaryDirectory
{
public:
  explicit ScopedTemporaryDirectory(
    std::filesystem::path directory)
  : directory_(std::move(directory))
  {
    photometric_evidence_detail::require(
      std::filesystem::create_directory(directory_),
      "test bundle directory creation failed");
  }

  ~ScopedTemporaryDirectory()
  {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  ScopedTemporaryDirectory(
    const ScopedTemporaryDirectory &) = delete;
  ScopedTemporaryDirectory & operator=(
    const ScopedTemporaryDirectory &) = delete;
  ScopedTemporaryDirectory(
    ScopedTemporaryDirectory &&) = delete;
  ScopedTemporaryDirectory & operator=(
    ScopedTemporaryDirectory &&) = delete;

  const std::filesystem::path & path() const noexcept
  {
    return directory_;
  }

private:
  std::filesystem::path directory_;
};

class TrustedPhotometricEvidenceTestBundle
{
public:
  TrustedPhotometricEvidenceTestBundle()
  : TrustedPhotometricEvidenceTestBundle(2, 2)
  {
  }

  TrustedPhotometricEvidenceTestBundle(
    const int image_width,
    const int image_height)
  : TrustedPhotometricEvidenceTestBundle(
      image_width, image_height,
      TrustedPhotometricEvidenceTestFailurePoint::kNone,
      nextDirectoryPath())
  {
  }

  TrustedPhotometricEvidenceTestBundle(
    const int image_width,
    const int image_height,
    const TrustedPhotometricEvidenceTestFailurePoint
      failure_point,
    std::filesystem::path directory)
  : directory_(std::move(directory)),
    image_width_(image_width),
    image_height_(image_height)
  {
    if (image_width_ <= 0 || image_height_ <= 0)
    {
      throw std::invalid_argument(
        "test bundle image dimensions must be positive");
    }
    const std::uint64_t pixel_count =
      static_cast<std::uint64_t>(image_width_) *
      static_cast<std::uint64_t>(image_height_);
    if (
      pixel_count >
      photometric_evidence_detail::kMaximumTestImagePixels)
    {
      throw std::invalid_argument(
        "test bundle image dimensions exceed resource cap");
    }
    image_pixel_count_ =
      static_cast<std::size_t>(pixel_count);

    if (
      failure_point ==
      TrustedPhotometricEvidenceTestFailurePoint::
      kAfterDirectoryCreation)
    {
      throw std::runtime_error(
              "injected test bundle construction failure");
    }
    build();
    publish(failure_point);
  }

  ~TrustedPhotometricEvidenceTestBundle() = default;

  TrustedPhotometricEvidenceTestBundle(
    const TrustedPhotometricEvidenceTestBundle &) = delete;
  TrustedPhotometricEvidenceTestBundle & operator=(
    const TrustedPhotometricEvidenceTestBundle &) = delete;

  TrustedPhotometricCalibrationLoadResult load() const
  {
    return TrustedPhotometricCalibrationLoader::load(
      directory_.path());
  }

private:
  static std::filesystem::path nextDirectoryPath()
  {
    static std::uint64_t sequence = 0U;
    const auto ticks =
      std::chrono::steady_clock::now().
      time_since_epoch().count();
    return
      std::filesystem::temp_directory_path() /
      (
        "livo-current-frame-photometric-" +
        std::to_string(ticks) + "-" +
        std::to_string(++sequence));
  }

  void write(
    const std::string & filename,
    const std::string & bytes)
  {
    std::ofstream stream(
      directory_.path() / filename,
      std::ios::binary | std::ios::trunc);
    photometric_evidence_detail::require(
      stream.good(), "test bundle open failed");
    stream.write(
      bytes.data(),
      static_cast<std::streamsize>(bytes.size()));
    photometric_evidence_detail::require(
      stream.good(), "test bundle write failed");
  }

  void build()
  {
    Json::Value trust_root(Json::objectValue);
    trust_root["trust_domain"] =
      photometric_evidence_detail::kTrustDomain;
    trust_root["public_key_sha256"] =
      photometric_evidence_detail::kPublicKeySha256;

    Json::Value solver(Json::objectValue);
    solver["name"] =
      "livo_core_four_camera_photometric_calibration";
    solver["version"] = "2.0.0";
    const std::string source_bundle_hash(64U, 'a');

    Json::Value config(Json::objectValue);
    config["schema_version"] = 2;
    config["calibration_set_id"] =
      "current-frame-evidence-test";
    config["dataset_scope"] =
      Json::Value(Json::arrayValue);
    config["status"] = "available";
    config["frame_metadata"] =
      Json::Value(Json::objectValue);
    config["source_bundle_file"] =
      "photometric_calibration_source_bundle.json";
    config["source_bundle_sha256"] =
      source_bundle_hash;
    config["calibration_receipt_file"] =
      photometric_evidence_detail::kReceiptFilename;
    config["runtime_payload_file"] =
      photometric_evidence_detail::kPayloadFilename;
    config["runtime_signature_file"] =
      photometric_evidence_detail::kSignatureFilename;
    config["trust_root"] = trust_root;

    Json::Value artifacts(Json::objectValue);
    Json::Value cameras(Json::arrayValue);
    std::ostringstream vignetting_stream;
    const std::array<const char *, 4U> vignetting_pattern{
      "1", "0.9", "0.8", "1"};
    for (std::size_t index = 0U;
         index < image_pixel_count_; ++index)
    {
      vignetting_stream <<
        vignetting_pattern[index % vignetting_pattern.size()] <<
        (index + 1U == image_pixel_count_ ? '\n' : ' ');
    }
    const std::string vignetting = vignetting_stream.str();

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
      for (std::size_t index = 0U;
           index < 256U; ++index)
      {
        lut_stream <<
          static_cast<double>(index) / 255.0 <<
          (index + 1U == 256U ? '\n' : ' ');
      }
      const std::string lut = lut_stream.str();
      write(lut_filename, lut);
      write(vignetting_filename, vignetting);
      artifacts[lut_filename] =
        photometric_evidence_detail::sha256Hex(lut);
      artifacts[vignetting_filename] =
        photometric_evidence_detail::sha256Hex(
        vignetting);

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
      camera["source_bundle_sha256"] =
        source_bundle_hash;
      camera["image_width"] = image_width_;
      camera["image_height"] = image_height_;
      camera["black_level"] = 0.0;
      camera["white_level"] = 255.0;
      camera["inverse_response_lut_file"] =
        lut_filename;
      camera["vignetting_transmission_file"] =
        vignetting_filename;
      camera["radiance_scale"] =
        1.0 + 0.05 * static_cast<double>(camera_id);
      camera["radiance_offset"] = 0.0;
      camera["covariance_scale"] = 1.0;

      Json::Value domain(Json::objectValue);
      domain["bit_depth"] = 8;
      domain["unit"] = "canonical_gray_code_value";
      domain["canonicalization"] =
        camera_id % 2 == 0 ?
        "mono8_identity_v1" :
        "rgb8_bt601_luma_uint8_v1";
      domain["image_width"] = image_width_;
      domain["image_height"] = image_height_;
      camera["raw_intensity_domain"] = domain;

      Json::Value noise(Json::objectValue);
      noise["model"] = "constant_sigma_v1";
      noise["standard_deviation_code_value"] =
        0.75 + 0.25 * static_cast<double>(camera_id);
      noise["provenance"] = "laboratory_measured";
      noise["estimator_revision"] =
        "temporal_mad_p99_v1";
      noise["source_sample_count"] = Json::UInt64(64U);
      camera["raw_intensity_noise"] = noise;
      cameras.append(camera);
    }
    config["cameras"] = cameras;

    payload_["schema_version"] = 2;
    payload_["receipt_type"] =
      "four_camera_photometric_calibration";
    payload_["receipt_hash_scheme"] =
      photometric_evidence_detail::kReceiptHashScheme;
    payload_["solver"] = solver;
    payload_["calibration_set_id"] =
      config["calibration_set_id"];
    payload_["declared_source_id"] =
      "current-frame-evidence-test";
    payload_["manifest_sha256"] = std::string(64U, 'b');
    payload_["source_bundle_file"] =
      config["source_bundle_file"];
    payload_["source_bundle_sha256"] =
      config["source_bundle_sha256"];
    payload_["source_counts"] =
      Json::Value(Json::objectValue);
    payload_["frame_metadata"] =
      config["frame_metadata"];
    payload_["cameras"] = cameras;
    payload_["calibration_config_payload"] = config;
    payload_["calibration_config_sha256"] =
      std::string(64U, 'c');
    payload_["held_out_metrics"] =
      Json::Value(Json::objectValue);
    payload_["thresholds"] =
      Json::Value(Json::objectValue);
    payload_["artifacts_sha256"] = artifacts;
    payload_["integrity_scope"] =
      "source_snapshot_config_solver_metrics_numeric_artifacts";
    payload_["trust_boundary"] =
      "ed25519_signature_requires_explicit_external_public_key_pin";
    payload_["trust_root"] = trust_root;
  }

  void publish(
    const TrustedPhotometricEvidenceTestFailurePoint
      failure_point)
  {
    using namespace photometric_evidence_detail;
    const std::string payload_bytes = compactJson(payload_);
    const Signature signature = signPayload(payload_bytes);
    const std::string receipt_sha256 =
      receiptHash(payload_bytes, signature.raw);

    Json::Value detached(Json::objectValue);
    detached["schema_version"] = 1;
    detached["signature_type"] =
      "four_camera_photometric_calibration_runtime_signature";
    detached["algorithm"] = "ed25519";
    detached["solver"] = payload_["solver"];
    detached["trust_domain"] =
      payload_["trust_root"]["trust_domain"];
    detached["payload_file"] = kPayloadFilename;
    detached["payload_sha256"] = sha256Hex(payload_bytes);
    detached["public_key_sha256"] =
      payload_["trust_root"]["public_key_sha256"];
    detached["receipt_hash_scheme"] =
      payload_["receipt_hash_scheme"];
    detached["receipt_sha256"] = receipt_sha256;
    detached["signature_base64"] = signature.base64;

    Json::Value receipt = payload_;
    receipt["signature"]["algorithm"] = "ed25519";
    receipt["signature"]["public_key_sha256"] =
      detached["public_key_sha256"];
    receipt["signature"]["signature_base64"] =
      signature.base64;
    receipt["receipt_sha256"] = receipt_sha256;
    write(kPayloadFilename, payload_bytes);
    if (
      failure_point ==
      TrustedPhotometricEvidenceTestFailurePoint::
      kAfterFirstPublishedFile)
    {
      throw std::runtime_error(
              "injected test bundle publish failure");
    }
    write(kSignatureFilename, compactJson(detached));
    write(kReceiptFilename, compactJson(receipt));
  }

  ScopedTemporaryDirectory directory_;
  int image_width_{2};
  int image_height_{2};
  std::size_t image_pixel_count_{4U};
  Json::Value payload_{Json::objectValue};
};

}  // namespace livo::test_support

#endif  // LIVO_TRUSTED_PHOTOMETRIC_EVIDENCE_TEST_SUPPORT_H_
