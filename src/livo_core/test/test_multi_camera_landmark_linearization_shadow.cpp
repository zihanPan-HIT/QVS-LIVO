#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreorder"
#endif
#include "active_plane_authority_test_support.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include "adaptive_visual_policy.h"
#include "reference_only_migration_runtime_provenance.h"
#include "camera_observation.h"
#include "multi_camera_landmark_linearization_shadow.h"
#include "shared_landmark_shadow_session.h"
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
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{

static_assert(
  !std::is_default_constructible_v<
    livo::MultiCameraLandmarkSelectionTrace>);
static_assert(
  !std::is_copy_constructible_v<
    livo::MultiCameraLandmarkSelectionTrace>);
static_assert(
  !std::is_copy_assignable_v<
    livo::MultiCameraLandmarkSelectionTrace>);
static_assert(
  !std::is_move_constructible_v<
    livo::MultiCameraLandmarkSelectionTrace>);
static_assert(
  !std::is_move_assignable_v<
    livo::MultiCameraLandmarkSelectionTrace>);
static_assert(
  !std::is_default_constructible_v<
    livo::MultiCameraLandmarkShadowDecision>);
static_assert(
  !std::is_default_constructible_v<
    livo::MultiCameraLandmarkShadowBatchResult>);
static_assert(
  !std::is_copy_constructible_v<
    livo::MultiCameraLandmarkLinearizationShadow>);

constexpr char kTestTrustDomain[] =
  "livo-core-photometric-test-v1";
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

std::string compactJson(const Json::Value & value)
{
  Json::StreamWriterBuilder builder;
  builder["commentStyle"] = "None";
  builder["indentation"] = "";
  builder["emitUTF8"] = true;
  return Json::writeString(builder, value) + "\n";
}

std::vector<unsigned char> fromHex(const std::string & encoded)
{
  if (encoded.size() % 2U != 0U)
  {
    throw std::runtime_error("invalid hex fixture");
  }
  std::vector<unsigned char> result(encoded.size() / 2U, 0U);
  for (std::size_t index = 0U; index < result.size(); ++index)
  {
    result[index] = static_cast<unsigned char>(
      std::stoul(encoded.substr(index * 2U, 2U), nullptr, 16));
  }
  return result;
}

std::string sha256Hex(const std::string & bytes)
{
  std::array<unsigned char, 32U> digest{};
  unsigned int digest_size = 0U;
  if (
    EVP_Digest(
      bytes.data(), bytes.size(), digest.data(), &digest_size,
      EVP_sha256(), nullptr) != 1 ||
    digest_size != digest.size())
  {
    throw std::runtime_error("fixture sha256 failed");
  }
  std::ostringstream stream;
  stream << std::hex << std::setfill('0');
  for (const unsigned char byte : digest)
  {
    stream << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return stream.str();
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
    nullptr, "ED25519", nullptr,
    private_seed.data(), private_seed.size());
  if (raw_key == nullptr)
  {
    throw std::runtime_error("fixture private key failed");
  }
  std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
    raw_key, &EVP_PKEY_free);
  EVP_MD_CTX * const raw_context = EVP_MD_CTX_new();
  if (raw_context == nullptr)
  {
    throw std::runtime_error("fixture signature context failed");
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
    raw_context, &EVP_MD_CTX_free);
  if (
    EVP_DigestSignInit(
      context.get(), nullptr, nullptr, nullptr, key.get()) != 1)
  {
    throw std::runtime_error("fixture signature init failed");
  }
  TestSignature signature;
  std::size_t signature_size = signature.raw.size();
  if (
    EVP_DigestSign(
      context.get(), signature.raw.data(), &signature_size,
      reinterpret_cast<const unsigned char *>(payload.data()),
      payload.size()) != 1 ||
    signature_size != signature.raw.size())
  {
    throw std::runtime_error("fixture signature failed");
  }
  signature.base64.assign(
    4U * ((signature_size + 2U) / 3U), '\0');
  const int encoded_size = EVP_EncodeBlock(
    reinterpret_cast<unsigned char *>(signature.base64.data()),
    signature.raw.data(), static_cast<int>(signature_size));
  if (encoded_size != static_cast<int>(signature.base64.size()))
  {
    throw std::runtime_error("fixture base64 failed");
  }
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
  const auto payload_length = bigEndianLength(payload.size());
  const auto signature_length = bigEndianLength(signature.size());
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

class TrustedCalibrationBundle
{
public:
  TrustedCalibrationBundle()
  {
    static std::uint64_t sequence = 0U;
    const auto ticks =
      std::chrono::steady_clock::now().time_since_epoch().count();
    directory_ = std::filesystem::temp_directory_path() /
      (
        "livo-multicam-selection-calibration-" +
        std::to_string(ticks) + "-" +
        std::to_string(++sequence));
    if (!std::filesystem::create_directory(directory_))
    {
      throw std::runtime_error(
        "failed to create trusted calibration fixture");
    }
    buildAndWrite();
  }

  ~TrustedCalibrationBundle()
  {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  TrustedCalibrationBundle(
    const TrustedCalibrationBundle &) = delete;
  TrustedCalibrationBundle & operator=(
    const TrustedCalibrationBundle &) = delete;

  const std::filesystem::path & directory() const noexcept
  {
    return directory_;
  }

private:
  void write(
    const std::string & filename,
    const std::string & bytes) const
  {
    std::ofstream stream(
      directory_ / filename,
      std::ios::binary | std::ios::trunc);
    stream.write(
      bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!stream.good())
    {
      throw std::runtime_error(
        "failed to write trusted calibration fixture");
    }
  }

  void buildAndWrite()
  {
    Json::Value trust_root(Json::objectValue);
    trust_root["trust_domain"] = kTestTrustDomain;
    trust_root["public_key_sha256"] = kTestPublicKeySha256;

    Json::Value solver(Json::objectValue);
    solver["name"] =
      "livo_core_four_camera_photometric_calibration";
    solver["version"] = "2.0.0";
    const std::string source_bundle_hash(64U, 'a');

    Json::Value config(Json::objectValue);
    config["schema_version"] = 1;
    config["calibration_set_id"] =
      "trusted-synthetic-four-camera";
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
      cameras.append(camera);
    }
    config["cameras"] = cameras;

    Json::Value payload(Json::objectValue);
    payload["schema_version"] = 1;
    payload["receipt_type"] =
      "four_camera_photometric_calibration";
    payload["receipt_hash_scheme"] = kReceiptHashScheme;
    payload["solver"] = solver;
    payload["calibration_set_id"] =
      config["calibration_set_id"];
    payload["declared_source_id"] = "synthetic-lab-run";
    payload["manifest_sha256"] = std::string(64U, 'b');
    payload["source_bundle_file"] =
      config["source_bundle_file"];
    payload["source_bundle_sha256"] =
      config["source_bundle_sha256"];
    payload["source_counts"] = Json::Value(Json::objectValue);
    payload["frame_metadata"] = config["frame_metadata"];
    payload["cameras"] = cameras;
    payload["calibration_config_payload"] = config;
    payload["calibration_config_sha256"] = std::string(64U, 'c');
    payload["held_out_metrics"] = Json::Value(Json::objectValue);
    payload["thresholds"] = Json::Value(Json::objectValue);
    payload["artifacts_sha256"] = artifacts;
    payload["integrity_scope"] =
      "source_snapshot_config_solver_metrics_numeric_artifacts";
    payload["trust_boundary"] =
      "ed25519_signature_requires_explicit_external_public_key_pin";
    payload["trust_root"] = trust_root;

    const std::string payload_bytes = compactJson(payload);
    const TestSignature signature = signPayload(payload_bytes);
    const std::string receipt_sha256 =
      receiptHash(payload_bytes, signature.raw);

    Json::Value detached(Json::objectValue);
    detached["schema_version"] = 1;
    detached["signature_type"] =
      "four_camera_photometric_calibration_runtime_signature";
    detached["algorithm"] = "ed25519";
    detached["solver"] = payload["solver"];
    detached["trust_domain"] =
      payload["trust_root"]["trust_domain"];
    detached["payload_file"] = kRuntimePayloadFilename;
    detached["payload_sha256"] = sha256Hex(payload_bytes);
    detached["public_key_sha256"] =
      payload["trust_root"]["public_key_sha256"];
    detached["receipt_hash_scheme"] =
      payload["receipt_hash_scheme"];
    detached["receipt_sha256"] = receipt_sha256;
    detached["signature_base64"] = signature.base64;

    Json::Value receipt = payload;
    receipt["signature"]["algorithm"] = "ed25519";
    receipt["signature"]["public_key_sha256"] =
      detached["public_key_sha256"];
    receipt["signature"]["signature_base64"] =
      signature.base64;
    receipt["receipt_sha256"] = receipt_sha256;

    write(kRuntimePayloadFilename, payload_bytes);
    write(kRuntimeSignatureFilename, compactJson(detached));
    write(kReceiptFilename, compactJson(receipt));
  }

  std::filesystem::path directory_;
};

struct RuntimePublication
{
  CameraObservationBatch batch;
  std::optional<livo::ReferenceOnlyMigrationRuntimeBundle> runtime;
};

class RuntimePipeline
{
public:
  RuntimePipeline()
  : publisher_(queue_.batchAuthorityIdentity())
  {
    if (!queue_.setBatchAuthoritySealingEnabled(true))
    {
      throw std::runtime_error("batch authority setup failed");
    }
  }

  RuntimePublication publish(
    const double timestamp_sec,
    const std::uint8_t pixel = 37U)
  {
    RuntimePublication result;
    result.batch = sealBatch(timestamp_sec, pixel);
    result.runtime = publishRuntime(result.batch);
    if (!result.runtime.has_value())
    {
      throw std::runtime_error("runtime publication failed");
    }
    return result;
  }

  CameraObservationBatch sealBatch(
    const double timestamp_sec,
    const std::uint8_t pixel = 37U)
  {
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      CameraObservation observation;
      observation.camera_id = camera_id;
      observation.timestamp = timestamp_sec;
      observation.image =
        cv::Mat(8, 12, CV_8UC1, cv::Scalar(pixel));
      observation.photometric_status =
        livo::PhotometricStatus::kValid;
      if (
        queue_.push(std::move(observation)) !=
        CameraObservationPushResult::ACCEPTED)
      {
        throw std::runtime_error("camera observation push failed");
      }
    }
    CameraObservationBatch batch;
    if (!queue_.popSynchronized(
        timestamp_sec + 0.01, 0.01, batch))
    {
      throw std::runtime_error("camera batch seal failed");
    }
    return batch;
  }

  livo::CameraObservationBatchAuthorityIdentity
  batchAuthority() const noexcept
  {
    return queue_.batchAuthorityIdentity();
  }

  livo::ReferenceOnlyMigrationRuntimePublisherIdentity
  runtimeAuthority() const noexcept
  {
    return publisher_.identity();
  }

private:
  std::optional<livo::ReferenceOnlyMigrationRuntimeBundle>
  publishRuntime(const CameraObservationBatch & batch)
  {
    StatesGroup state;
    state.pos_end.x() = batch.timestamp;
    state.cov =
      1.0e-3 *
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    return publisher_.publish(
      batch, state, batch.timestamp, 0.0,
      livo::CameraTimeTransfer{});
  }

  CameraObservationQueue queue_;
  livo::ReferenceOnlyMigrationRuntimePublisher publisher_;
};

livo::AdaptiveCameraObservation healthyObservation(
  const std::uint8_t camera_id)
{
  livo::AdaptiveCameraObservation observation;
  observation.camera_id = camera_id;
  observation.frame_received = true;
  observation.available_metrics =
    livo::kAdaptiveFirstVersionMetricMask;
  observation.candidate_points = 40U;
  observation.tracked_points = 28U;
  observation.residual_count = 28U * 64U;
  observation.squared_residual_sum =
    static_cast<double>(observation.residual_count) * 16.0;
  observation.saturation_fraction = 0.0;
  observation.dark_fraction = 0.0;
  observation.mean_gradient = 18.0;
  observation.spatial_coverage = 0.35;
  observation.visual_information =
    (1.0 + static_cast<double>(camera_id)) *
    livo::AdaptiveCameraObservation::Matrix6::Identity();
  return observation;
}

livo::AdaptiveVisualPolicy::CameraObservations healthyRig()
{
  livo::AdaptiveVisualPolicy::CameraObservations observations;
  for (std::size_t camera_id = 0U;
       camera_id < observations.size(); ++camera_id)
  {
    observations[camera_id] =
      healthyObservation(static_cast<std::uint8_t>(camera_id));
  }
  return observations;
}

void disableCamera(
  livo::AdaptiveVisualPolicy::CameraObservations & observations,
  const std::uint8_t camera_id)
{
  livo::AdaptiveCameraObservation & observation =
    observations.at(camera_id);
  observation.candidate_points = 0U;
  observation.tracked_points = 0U;
  observation.residual_count = 0U;
  observation.squared_residual_sum = 0.0;
  observation.saturation_fraction = 1.0;
  observation.mean_gradient = 0.0;
  observation.spatial_coverage = 0.0;
  observation.visual_information.setZero();
}

livo::AdaptiveVisualPolicy makePolicy()
{
  livo::AdaptiveVisualPolicyConfig config;
  config.enabled = true;
  return livo::AdaptiveVisualPolicy(config);
}

struct TargetInput
{
  CameraObservationBatch batch;
  std::optional<livo::ReferenceOnlyMigrationRuntimeBundle> runtime;
  std::optional<livo::AdaptiveVisualPolicyPlanCapability> capability;
};

TargetInput issueTarget(
  livo::AdaptiveVisualPolicy & policy,
  RuntimePublication publication,
  const livo::AdaptiveVisualPolicy::CameraObservations & observations,
  const bool lidar_information_ready = true)
{
  TargetInput target;
  target.batch = std::move(publication.batch);
  target.runtime = std::move(publication.runtime);
  if (!target.runtime.has_value())
  {
    throw std::runtime_error("missing target runtime");
  }
  target.capability = policy.issuePlan(
    *target.runtime,
    livo::AdaptiveVisualPolicy::Matrix6::Identity(),
    observations, lidar_information_ready);
  if (!target.capability.has_value())
  {
    throw std::runtime_error("policy capability issue failed");
  }
  return target;
}

livo::LegacyPlaneLandmarkSample geometrySample()
{
  livo::LegacyPlaneLandmarkSample sample;
  sample.plane = livo::PlaneIdentity{700U, 1U};
  sample.geometry_revision = 1U;
  sample.identity_origin_world_m = Eigen::Vector3d::Zero();
  sample.identity_u_axis_world = Eigen::Vector3d::UnitX();
  sample.identity_v_axis_world = Eigen::Vector3d::UnitY();
  sample.position_world_m = Eigen::Vector3d(0.1, 0.1, 5.0);
  sample.position_covariance_m2 =
    0.01 * Eigen::Matrix3d::Identity();
  sample.normal_world = -Eigen::Vector3d::UnitZ();
  sample.plane_covariance =
    0.001 * Eigen::Matrix<double, 6, 6>::Identity();
  return sample;
}

class StoreFixture
{
public:
  enum class SourceMode
  {
    kAcceptedVisual,
    kReferenceOnly,
    kNone,
  };

  explicit StoreFixture(
    const bool reserve_older_target = false,
    const SourceMode source_mode = SourceMode::kAcceptedVisual,
    const bool bootstrap_geometry = true)
  : session_(sessionConfig())
  {
    const auto loaded =
      livo::TrustedPhotometricCalibrationLoader::load(
      calibration_bundle_.directory());
    if (!loaded.trustedReady() || !loaded.calibration.has_value())
    {
      throw std::runtime_error(
        "trusted photometric test calibration failed: " +
        loaded.diagnostic);
    }
    calibration_.emplace(*loaded.calibration);
    if (!session_.bindRuntimeObservationAuthority(
        pipeline_.batchAuthority(),
        pipeline_.runtimeAuthority()))
    {
      throw std::runtime_error("runtime authority bind failed");
    }
    for (std::uint8_t camera_id = 0U; camera_id < 4U; ++camera_id)
    {
      if (!session_.recordCameraConfiguration(camera_id).changed)
      {
        throw std::runtime_error("camera configuration failed");
      }
      livo::ShadowPhotometricCapabilityEvidence evidence;
      evidence.mode = livo::PhotometricMode::kCalibratedDirect;
      evidence.strict_calibration = &*calibration_;
      if (
        session_.configurePhotometric(camera_id, evidence).status !=
        livo::ShadowPhotometricCapabilityStatus::kStrictCalibrated)
      {
        throw std::runtime_error(
          "strict photometric capability failed");
      }
    }
    session_.advanceVisualStateRevision();
    if (
      session_.configureTimeTransfer(
        livo::CameraTimeTransferConfig{}).status !=
      livo::TimeTransferConfigurationUpdateStatus::kUpdated)
    {
      throw std::runtime_error("time configuration failed");
    }
    if (bootstrap_geometry)
    {
      bootstrapGeometry();
    }
    else if (source_mode != SourceMode::kNone)
    {
      throw std::runtime_error(
        "source requires bootstrapped geometry");
    }
    if (reserve_older_target)
    {
      older_target_.emplace(pipeline_.publish(9.9, 31U));
    }
    if (source_mode != SourceMode::kNone)
    {
      reference_source_.emplace(
        pipeline_.publish(9.95, 35U));
      commitSource(*reference_source_);
    }
    if (source_mode == SourceMode::kAcceptedVisual)
    {
      source_.emplace(pipeline_.publish(10.0, 37U));
      commitSource(*source_, false);
    }
  }

  livo::SharedLandmarkShadowSession & session() noexcept
  {
    return session_;
  }

  RuntimePipeline & pipeline() noexcept
  {
    return pipeline_;
  }

  livo::CameraObservationBatchAuthorityIdentity
  batchAuthority() const noexcept
  {
    return pipeline_.batchAuthority();
  }

  livo::ReferenceOnlyMigrationRuntimePublisherIdentity
  runtimeAuthority() const noexcept
  {
    return pipeline_.runtimeAuthority();
  }

  const RuntimePublication & source() const
  {
    return source_.value();
  }

  const RuntimePublication & olderTarget() const
  {
    return older_target_.value();
  }

  livo::LandmarkSnapshot landmark() const
  {
    const auto value = session_.findLandmark(support_);
    if (!value.has_value())
    {
      throw std::runtime_error("fixture landmark missing");
    }
    return *value;
  }

  const livo::ObservationSnapshot & acceptedObservation(
    const livo::LandmarkSnapshot & value) const
  {
    const auto accepted = std::find_if(
      value.observations.begin(), value.observations.end(),
      [](const livo::ObservationSnapshot & observation) {
        return !observation.reference_only;
      });
    if (accepted == value.observations.end())
    {
      throw std::runtime_error(
        "fixture accepted observation missing");
    }
    return *accepted;
  }

  void refit()
  {
    const livo::LandmarkSnapshot current = landmark();
    session_.advanceLioPosteriorRevision();
    livo::SharedLandmarkGeometryRefitSample sample;
    sample.support_key = support_;
    sample.position_world_m =
      current.position_world_m + Eigen::Vector3d(0.01, 0.0, 0.0);
    sample.position_covariance_m2 =
      current.position_covariance_m2;
    sample.normal_world = current.normal_world;
    sample.plane_covariance = current.plane_covariance;
    sample.local_basis = current.local_basis;
    livo::SharedLandmarkGeometryRefitRequest request;
    request.samples.push_back(std::move(sample));
    auto staged = session_.stageGeometryRefits(request);
    if (
      staged.status !=
        livo::SharedLandmarkShadowOperationStatus::
        kGeometryRefitStaged ||
      !staged.transaction.valid())
    {
      throw std::runtime_error("refit stage failed");
    }
    if (
      session_.commitGeometryRefits(
        std::move(staged.transaction)).status !=
      livo::SharedLandmarkShadowOperationStatus::
      kGeometryRefitCommitted)
    {
      throw std::runtime_error("refit commit failed");
    }
  }

  void tombstone()
  {
    livo::PlaneTombstoneEvent event;
    event.old_identity = geometry_sample_.plane;
    event.last_revision = geometry_sample_.geometry_revision;
    event.generation = 2U;
    event.reason =
      livo::PlaneTombstoneReason::kMajorReplacement;
    if (
      session_.applyTombstones({event}).status !=
      livo::SharedLandmarkShadowOperationStatus::
      kTombstoneCommitted)
    {
      throw std::runtime_error("tombstone commit failed");
    }
  }

  void replaceActivePlanesForCurrentGeometry(
    const std::uint64_t generation)
  {
    const livo::GeometryVersion version{
      geometry_sample_.plane,
      geometry_sample_.geometry_revision};
    if (!active_planes_.replace({
        livo::ActivePlaneDescriptor{version, generation}}))
    {
      throw std::runtime_error(
        "active plane replacement failed");
    }
  }

  void appendAcceptedSource(const double timestamp_sec)
  {
    source_.emplace(
      pipeline_.publish(timestamp_sec, 39U));
    commitSource(*source_, false);
  }

  std::uint64_t currentActivePlaneRevision() const
  {
    return active_planes_.snapshot().revision();
  }

  livo::SharedLandmarkShadowOperationResult
  confirmCurrentGeometry()
  {
    livo::SharedLandmarkGeometryBootstrapRequest request;
    request.geometry = livo::GeometrySnapshotId{77U};
    request.active_planes = active_planes_.snapshot();
    request.samples.push_back(geometry_sample_);
    return session_.bootstrapGeometry(request);
  }

  livo::SharedLandmarkShadowOperationResult
  replaceAndConfirmAlternateGeometry()
  {
    livo::LegacyPlaneLandmarkSample alternate =
      geometry_sample_;
    alternate.plane = livo::PlaneIdentity{701U, 1U};
    alternate.position_world_m =
      Eigen::Vector3d(0.2, 0.1, 5.0);
    const livo::GeometryVersion version{
      alternate.plane, alternate.geometry_revision};
    if (!active_planes_.replace({
        livo::ActivePlaneDescriptor{version, 1U}}))
    {
      throw std::runtime_error(
        "alternate active plane replacement failed");
    }
    livo::SharedLandmarkGeometryBootstrapRequest request;
    request.geometry = livo::GeometrySnapshotId{77U};
    request.active_planes = active_planes_.snapshot();
    request.samples.push_back(std::move(alternate));
    return session_.bootstrapGeometry(request);
  }

private:
  static livo::SharedLandmarkShadowSessionConfig sessionConfig()
  {
    livo::SharedLandmarkShadowSessionConfig config;
    config.geometry.enabled = true;
    config.observation.enabled = true;
    config.tombstone.enabled = true;
    config.refit.enabled = true;
    return config;
  }

  void bootstrapGeometry()
  {
    geometry_sample_ = geometrySample();
    const livo::GeometryVersion version{
      geometry_sample_.plane,
      geometry_sample_.geometry_revision};
    if (!active_planes_.replace({
        livo::ActivePlaneDescriptor{version, 77U}}))
    {
      throw std::runtime_error("active plane publication failed");
    }
    livo::SharedLandmarkGeometryBootstrapRequest request;
    request.geometry = livo::GeometrySnapshotId{77U};
    request.active_planes = active_planes_.snapshot();
    request.samples.push_back(geometry_sample_);
    if (
      session_.bootstrapGeometry(request).status !=
      livo::SharedLandmarkShadowOperationStatus::
      kGeometryCommitted)
    {
      throw std::runtime_error("geometry bootstrap failed");
    }
    livo::GeometryVersion resolved_version;
    if (!session_.resolveGeometryIdentity(
        geometry_sample_, support_, resolved_version))
    {
      throw std::runtime_error("geometry identity resolution failed");
    }
  }

  void commitSource(
    const RuntimePublication & source,
    const bool is_bootstrap = true)
  {
    if (!source.runtime.has_value())
    {
      throw std::runtime_error("source runtime missing");
    }
    const auto frame = source.runtime->frameProvenance(0U);
    const auto & time_bundle =
      source.runtime->timeTransferBundle();
    if (
      !frame.has_value() ||
      !time_bundle.validatedReceipt().has_value())
    {
      throw std::runtime_error("source provenance missing");
    }
    livo::SharedLandmarkShadowObservationSample sample;
    sample.support_key = support_;
    sample.camera_id = 0U;
    sample.time_transfer = time_bundle.transfers[0U];
    sample.frame_provenance =
      std::make_shared<
      const livo::ReferenceOnlyMigrationFrameProvenance>(
      *frame);
    sample.exposure_timestamp_sec =
      frame->exposureTimestampSec();
    sample.pixel = Eigen::Vector2d(100.0, 80.0);
    sample.level = 0;
    sample.owned_patch_pyramid = {
      {11.0F, 12.0F, 13.0F, 14.0F}};
    sample.frame_photometric_status =
      livo::PhotometricStatus::kValid;
    sample.anchor_pose_covariance =
      1.0e-5 *
      Eigen::Matrix<double, 6, 6>::Identity();
    sample.visual_update_accepted = !is_bootstrap;
    sample.is_bootstrap = is_bootstrap;

    livo::SharedLandmarkShadowObservationRequest request;
    request.active_planes = active_planes_.snapshot();
    request.time_transfer_batch =
      time_bundle.validatedReceipt();
    request.samples.push_back(std::move(sample));
    auto staged = session_.stageObservations(request);
    if (
      staged.status !=
        livo::SharedLandmarkShadowOperationStatus::
        kObservationStaged ||
      !staged.transaction.valid())
    {
      throw std::runtime_error("observation stage failed");
    }
    if (
      session_.commitObservations(
        std::move(staged.transaction)).status !=
      livo::SharedLandmarkShadowOperationStatus::
      kObservationCommitted)
    {
      throw std::runtime_error("observation commit failed");
    }
  }

  TrustedCalibrationBundle calibration_bundle_;
  std::optional<livo::PhotometricCalibration> calibration_;
  RuntimePipeline pipeline_;
  livo::test_support::ActivePlaneAuthority active_planes_;
  livo::SharedLandmarkShadowSession session_;
  livo::LegacyPlaneLandmarkSample geometry_sample_;
  livo::GeometrySupportKey support_;
  std::optional<RuntimePublication> reference_source_;
  std::optional<RuntimePublication> source_;
  std::optional<RuntimePublication> older_target_;
};

livo::MultiCameraLandmarkLinearizationShadowConfig enabledConfig()
{
  livo::MultiCameraLandmarkLinearizationShadowConfig config;
  config.enabled = true;
  return config;
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  DefaultOffIsInactiveAndCannotReadOrWriteSharedStore)
{
  StoreFixture fixture;
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, {});
  const auto before = fixture.session().audit();

  EXPECT_FALSE(shadow.active());
  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);
  const auto after = fixture.session().audit();

  EXPECT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kDisabled);
  EXPECT_EQ(result.acceptedTargetGeneration(), 0U);
  EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
  EXPECT_TRUE(result.decisions().empty());
  EXPECT_EQ(after.store_revision, before.store_revision);
  EXPECT_EQ(after.landmark_count, before.landmark_count);
  EXPECT_EQ(after.observation_count, before.observation_count);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  ReferenceOnlyObservationIsSkippedAndTargetStillAdvances)
{
  StoreFixture fixture(
    false, StoreFixture::SourceMode::kReferenceOnly);
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());
  const std::uint64_t validation_hash_count_before =
    target.runtime->cameraBatchReceipt().
    validationImageHashCount();

  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);

  EXPECT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_EQ(result.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(result.inspectedSourceObservationCount(), 1U);
  EXPECT_EQ(result.skippedReferenceOnlySourceCount(), 1U);
  EXPECT_EQ(result.candidateEvaluationCount(), 0U);
  EXPECT_TRUE(result.decisions().empty());
  EXPECT_EQ(
    target.runtime->cameraBatchReceipt().
    validationImageHashCount(),
    validation_hash_count_before);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  MissingCurrentActivePlaneSnapshotFailsClosed)
{
  StoreFixture fixture(
    false, StoreFixture::SourceMode::kNone, false);
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());

  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);

  EXPECT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::
    kMissingCurrentActivePlaneSnapshot);
  EXPECT_EQ(result.acceptedTargetGeneration(), 0U);
  EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
  EXPECT_TRUE(result.decisions().empty());
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  FirstOwnerValidatedTargetDirectlySelectsCurrentStoreSource)
{
  StoreFixture fixture;
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1, 41U), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());
  const auto before = fixture.session().audit();

  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);

  ASSERT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_TRUE(result.accepted());
  EXPECT_EQ(result.configurationGeneration(), 1U);
  EXPECT_EQ(result.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(shadow.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(result.targetCameraBatchSequence(), 3U);
  EXPECT_EQ(result.targetRuntimePublicationSequence(), 3U);
  EXPECT_EQ(result.skippedReferenceOnlySourceCount(), 1U);
  EXPECT_EQ(result.candidateEvaluationCount(), 4U);
  ASSERT_EQ(result.decisions().size(), 1U);
  const auto & decision = result.decisions().front();
  EXPECT_EQ(decision.sourceCameraId(), 0U);
  EXPECT_EQ(decision.targetCameraId(), 0U);
  EXPECT_STREQ(decision.photometricPathLabel(), "legacy_unattested");
  ASSERT_NE(decision.selectionTrace(), nullptr);
  const auto & trace = *decision.selectionTrace();
  ASSERT_TRUE(trace.valid());
  EXPECT_TRUE(shadow.accepts(trace));
  EXPECT_EQ(trace.configurationGeneration(), 1U);
  EXPECT_EQ(trace.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(trace.sourceCameraBatchSequence(), 2U);
  EXPECT_EQ(trace.targetCameraBatchSequence(), 3U);
  EXPECT_EQ(trace.targetRuntimePublicationSequence(), 3U);
  EXPECT_TRUE(
    trace.targetFrameProvenance().issuedBy(
      fixture.runtimeAuthority()));
  EXPECT_TRUE(
    trace.targetFrameProvenance().accepts(
      target.runtime->validatedComposition()));
  const livo::LandmarkSnapshot landmark = fixture.landmark();
  const auto & source = fixture.acceptedObservation(landmark);
  ASSERT_TRUE(source.store_receipt.has_value());
  EXPECT_TRUE(
    trace.sourceStoreReceipt().accepts(
      landmark.landmark_id, source));
  EXPECT_EQ(
    trace.sourceObservationRevision(),
    source.observation_revision);
  EXPECT_EQ(trace.policyBaseRevision(), policy.revision());
  const auto & camera_decision =
    target.capability->decisionPlan().cameras[0U];
  EXPECT_DOUBLE_EQ(
    trace.healthScore(), camera_decision.health);
  EXPECT_DOUBLE_EQ(
    trace.lidarComplementScore(),
    camera_decision.complement_score);
  EXPECT_EQ(
    trace.candidateBudget(),
    camera_decision.candidate_budget);
  EXPECT_EQ(trace.patchBudget(), camera_decision.patch_budget);
  EXPECT_FALSE(trace.hasLinearization());
  EXPECT_FALSE(trace.consumable());
  EXPECT_FALSE(trace.productionReady());
  EXPECT_FALSE(trace.solverApplied());
  const auto after = fixture.session().audit();
  EXPECT_EQ(after.store_revision, before.store_revision);
  EXPECT_EQ(after.observation_count, before.observation_count);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  ActivePlaneRevisionInvalidatesTraceUntilReconfirmed)
{
  StoreFixture fixture;
  auto policy = makePolicy();
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());
  const auto first_target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  const auto first = shadow.processBatch(
    *first_target.runtime,
    *first_target.capability);
  ASSERT_EQ(first.decisions().size(), 1U);
  ASSERT_NE(first.decisions().front().selectionTrace(), nullptr);
  const auto & old_trace =
    *first.decisions().front().selectionTrace();
  const std::uint64_t old_registry_revision =
    old_trace.sourceActivePlaneLease().registryRevision();
  ASSERT_TRUE(shadow.accepts(old_trace));

  fixture.replaceActivePlanesForCurrentGeometry(2U);

  EXPECT_FALSE(shadow.accepts(old_trace));
  const auto stale_target = issueTarget(
    policy, fixture.pipeline().publish(10.2), healthyRig());
  const auto stale = shadow.processBatch(
    *stale_target.runtime,
    *stale_target.capability);
  EXPECT_EQ(
    stale.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::
    kStaleCurrentActivePlaneSnapshot);
  EXPECT_EQ(stale.acceptedTargetGeneration(), 1U);
  EXPECT_TRUE(stale.decisions().empty());

  const auto confirmed = fixture.confirmCurrentGeometry();
  ASSERT_EQ(
    confirmed.status,
    livo::SharedLandmarkShadowOperationStatus::
    kNoGeometryChange);
  const auto recovered_target = issueTarget(
    policy, fixture.pipeline().publish(10.3), healthyRig());
  const auto recovered = shadow.processBatch(
    *recovered_target.runtime,
    *recovered_target.capability);
  ASSERT_EQ(
    recovered.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  ASSERT_EQ(recovered.decisions().size(), 1U);
  ASSERT_NE(
    recovered.decisions().front().selectionTrace(), nullptr);
  const auto & recovered_trace =
    *recovered.decisions().front().selectionTrace();
  EXPECT_TRUE(shadow.accepts(recovered_trace));
  EXPECT_NE(
    recovered_trace.sourceActivePlaneLease().
    registryRevision(),
    old_registry_revision);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  CurrentActivePlaneSnapshotSkipsSourcesWithoutLease)
{
  StoreFixture fixture;
  const auto confirmed =
    fixture.replaceAndConfirmAlternateGeometry();
  ASSERT_EQ(
    confirmed.status,
    livo::SharedLandmarkShadowOperationStatus::
    kGeometryCommitted);
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());

  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);

  EXPECT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_EQ(result.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(result.skippedReferenceOnlySourceCount(), 1U);
  EXPECT_EQ(result.skippedInactivePlaneSourceCount(), 1U);
  EXPECT_EQ(result.candidateEvaluationCount(), 0U);
  EXPECT_TRUE(result.decisions().empty());
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  SuccessfulObservationCommitRefreshesCurrentActivePlaneSnapshot)
{
  StoreFixture fixture;
  fixture.replaceActivePlanesForCurrentGeometry(2U);
  fixture.appendAcceptedSource(10.05);
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());

  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);

  ASSERT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  ASSERT_EQ(result.decisions().size(), 1U);
  ASSERT_NE(result.decisions().front().selectionTrace(), nullptr);
  EXPECT_EQ(
    result.decisions().front().selectionTrace()->
    sourceActivePlaneLease().registryRevision(),
    fixture.currentActivePlaneRevision());
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  InternalStoreBatchIdDoesNotAuthorizeTargetOrdering)
{
  StoreFixture fixture;
  const livo::LandmarkSnapshot landmark = fixture.landmark();
  const auto & source = fixture.acceptedObservation(landmark);
  ASSERT_EQ(source.accepted_batch_id.value, 3U);
  ASSERT_EQ(source.accepted_camera_batch_sequence, 2U);
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  ASSERT_EQ(
    target.runtime->cameraBatchReceipt().batchSequence(), 3U);
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());

  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);

  ASSERT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  ASSERT_EQ(result.decisions().size(), 1U);
  EXPECT_EQ(
    result.decisions().front().sourceObservationId(),
    source.observation_id);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  EqualAndNewerSourceCameraBatchesAreCanonicalIneligible)
{
  {
    StoreFixture fixture;
    auto policy = makePolicy();
    const auto equal_target = issueTarget(
      policy, fixture.source(), healthyRig());
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());

    const auto result = shadow.processBatch(
      *equal_target.runtime,
      *equal_target.capability);

    EXPECT_EQ(
      result.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
    EXPECT_EQ(result.acceptedTargetGeneration(), 1U);
    EXPECT_EQ(result.skippedSourceOrderCount(), 1U);
    EXPECT_TRUE(result.decisions().empty());
  }
  {
    StoreFixture fixture(true);
    auto policy = makePolicy();
    const auto older_target = issueTarget(
      policy, fixture.olderTarget(), healthyRig());
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());

    const auto result = shadow.processBatch(
      *older_target.runtime,
      *older_target.capability);

    EXPECT_EQ(
      result.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
    EXPECT_EQ(result.skippedSourceOrderCount(), 1U);
    EXPECT_TRUE(result.decisions().empty());
  }
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  CallerBatchMutationDoesNotAffectCompositionAndReplayDoesNotAdvance)
{
  StoreFixture fixture;
  auto policy = makePolicy();
  auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());
  target.batch.observations[2U].image =
    target.batch.observations[2U].image.clone();
  target.batch.observations[2U].image.
    at<std::uint8_t>(0, 0) = 99U;
  const std::uint64_t validation_hash_count_before =
    target.runtime->cameraBatchReceipt().
    validationImageHashCount();

  const auto accepted = shadow.processBatch(
    *target.runtime, *target.capability);
  ASSERT_EQ(
    accepted.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_EQ(accepted.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(
    target.runtime->cameraBatchReceipt().
    validationImageHashCount(),
    validation_hash_count_before);

  const auto replay = shadow.processBatch(
    *target.runtime, *target.capability);
  EXPECT_EQ(
    replay.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::
    kTargetOrderRejected);
  EXPECT_EQ(replay.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(shadow.acceptedTargetGeneration(), 1U);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  ForeignBatchRuntimeAndPolicyOwnersFailWithoutGenerationAdvance)
{
  {
    StoreFixture fixture;
    RuntimePipeline foreign_pipeline;
    auto policy = makePolicy();
    const auto foreign_target = issueTarget(
      policy, foreign_pipeline.publish(10.1), healthyRig());
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());

    const auto rejected = shadow.processBatch(
      *foreign_target.runtime,
      *foreign_target.capability);

    EXPECT_EQ(
      rejected.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::
      kTargetBatchAuthorityRejected);
    EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
  }
  {
    StoreFixture fixture;
    auto expected_policy = makePolicy();
    auto foreign_policy = makePolicy();
    const RuntimePublication publication =
      fixture.pipeline().publish(10.1);
    const auto foreign_target = issueTarget(
      foreign_policy, publication, healthyRig());
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), expected_policy,
      enabledConfig());

    const auto rejected = shadow.processBatch(
      *foreign_target.runtime,
      *foreign_target.capability);

    EXPECT_EQ(
      rejected.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::
      kPolicyCapabilityRejected);
    EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
  }
  {
    StoreFixture fixture;
    auto policy = makePolicy();
    const RuntimePublication publication =
      fixture.pipeline().publish(10.1);
    const auto target = issueTarget(
      policy, publication, healthyRig());
    livo::ReferenceOnlyMigrationRuntimePublisher foreign_publisher(
      fixture.batchAuthority());
    const CameraObservationBatch foreign_batch =
      fixture.pipeline().sealBatch(10.2);
    StatesGroup state;
    state.cov =
      1.0e-3 *
      Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity();
    const auto foreign_runtime = foreign_publisher.publish(
      foreign_batch, state, foreign_batch.timestamp, 0.0,
      livo::CameraTimeTransfer{});
    ASSERT_TRUE(foreign_runtime.has_value());
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());

    const auto rejected = shadow.processBatch(
      *foreign_runtime, *target.capability);

    EXPECT_EQ(
      rejected.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::
      kRuntimeAuthorityRejected);
    EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
  }
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  CapabilityRuntimeMustBeTheExactTargetPublication)
{
  StoreFixture fixture;
  auto policy = makePolicy();
  const RuntimePublication original =
    fixture.pipeline().publish(10.1);
  ASSERT_TRUE(original.runtime.has_value());
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.2),
    healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());

  const auto result = shadow.processBatch(
    *original.runtime, *target.capability);

  EXPECT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::
    kTargetCompositionRejected);
  EXPECT_EQ(result.acceptedTargetGeneration(), 0U);
  EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  DisabledAndNonReadyPolicyPlansFailWithoutGenerationAdvance)
{
  {
    StoreFixture fixture;
    livo::AdaptiveVisualPolicy disabled_policy;
    const auto target = issueTarget(
      disabled_policy, fixture.pipeline().publish(10.1),
      healthyRig());
    ASSERT_FALSE(
      target.capability->decisionPlan().adaptive_enabled);
    ASSERT_EQ(
      target.capability->decisionPlan().reason,
      livo::AdaptivePolicyReason::kDisabled);
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), disabled_policy,
      enabledConfig());

    const auto rejected = shadow.processBatch(
      *target.runtime, *target.capability);

    EXPECT_EQ(
      rejected.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::
      kPolicyCapabilityRejected);
    EXPECT_EQ(rejected.acceptedTargetGeneration(), 0U);
    EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
  }
  {
    StoreFixture fixture;
    auto policy = makePolicy();
    const auto target = issueTarget(
      policy, fixture.pipeline().publish(10.1),
      healthyRig(), false);
    ASSERT_TRUE(
      target.capability->decisionPlan().adaptive_enabled);
    ASSERT_FALSE(
      target.capability->decisionPlan().
      lidar_information_valid);
    ASSERT_EQ(
      target.capability->decisionPlan().reason,
      livo::AdaptivePolicyReason::kLidarNotReady);
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());

    const auto rejected = shadow.processBatch(
      *target.runtime, *target.capability);

    EXPECT_EQ(
      rejected.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::
      kPolicyCapabilityRejected);
    EXPECT_EQ(rejected.acceptedTargetGeneration(), 0U);
    EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
  }
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  CandidateBudgetFailureDoesNotAdvanceAndNextSmallerPlanStartsAtOne)
{
  StoreFixture fixture;
  auto policy = makePolicy();
  auto config = enabledConfig();
  config.allow_relative_affine_experiment = true;
  config.maximum_candidate_evaluations_per_batch = 3U;
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, config);
  const auto too_large = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());

  const auto rejected = shadow.processBatch(
    *too_large.runtime, *too_large.capability);

  EXPECT_EQ(
    rejected.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::
    kCandidateBudgetExceeded);
  EXPECT_EQ(rejected.acceptedTargetGeneration(), 0U);

  auto three_cameras = healthyRig();
  disableCamera(three_cameras, 3U);
  const auto bounded = issueTarget(
    policy, fixture.pipeline().publish(10.2), three_cameras);
  const auto accepted = shadow.processBatch(
    *bounded.runtime, *bounded.capability);
  EXPECT_EQ(
    accepted.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_EQ(accepted.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(accepted.candidateEvaluationCount(), 3U);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  LegalTargetWithoutAnySourceIsAcceptedAndAdvancesGeneration)
{
  StoreFixture fixture(
    false, StoreFixture::SourceMode::kNone);
  auto policy = makePolicy();
  const auto target = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, enabledConfig());

  const auto result = shadow.processBatch(
    *target.runtime, *target.capability);

  EXPECT_EQ(
    result.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
  EXPECT_EQ(result.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(result.inspectedSourceObservationCount(), 0U);
  EXPECT_TRUE(result.decisions().empty());
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  RefitAndTombstoneInvalidateLiveTraceButRemainNormalCandidateSkips)
{
  {
    StoreFixture fixture;
    auto policy = makePolicy();
    const auto first = issueTarget(
      policy, fixture.pipeline().publish(10.1), healthyRig());
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());
    const auto selected = shadow.processBatch(
      *first.runtime, *first.capability);
    ASSERT_EQ(selected.decisions().size(), 1U);
    ASSERT_NE(
      selected.decisions().front().selectionTrace(), nullptr);
    const auto & trace =
      *selected.decisions().front().selectionTrace();
    const auto before_live_accepts = fixture.session().audit();
    const auto generation_before_live_accepts =
      shadow.acceptedTargetGeneration();
    ASSERT_TRUE(shadow.accepts(trace));
    const auto after_live_accepts = fixture.session().audit();
    EXPECT_EQ(
      after_live_accepts.store_revision,
      before_live_accepts.store_revision);
    EXPECT_EQ(
      shadow.acceptedTargetGeneration(),
      generation_before_live_accepts);
    fixture.refit();
    const auto before_stale_accepts = fixture.session().audit();
    const auto generation_before_stale_accepts =
      shadow.acceptedTargetGeneration();
    EXPECT_FALSE(shadow.accepts(trace));
    const auto after_stale_accepts = fixture.session().audit();
    EXPECT_EQ(
      after_stale_accepts.store_revision,
      before_stale_accepts.store_revision);
    EXPECT_EQ(
      shadow.acceptedTargetGeneration(),
      generation_before_stale_accepts);
    EXPECT_EQ(shadow.acceptedTargetGeneration(), 1U);

    const auto second = issueTarget(
      policy, fixture.pipeline().publish(10.2), healthyRig());
    const auto result = shadow.processBatch(
      *second.runtime, *second.capability);
    EXPECT_EQ(
      result.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
    EXPECT_EQ(result.acceptedTargetGeneration(), 2U);
    EXPECT_EQ(result.skippedSamplingGeometryCount(), 1U);
    EXPECT_TRUE(result.decisions().empty());
  }
  {
    StoreFixture fixture;
    auto policy = makePolicy();
    const auto first = issueTarget(
      policy, fixture.pipeline().publish(10.1), healthyRig());
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());
    const auto selected = shadow.processBatch(
      *first.runtime, *first.capability);
    ASSERT_EQ(selected.decisions().size(), 1U);
    const auto & trace =
      *selected.decisions().front().selectionTrace();
    fixture.tombstone();
    const auto before_stale_accepts = fixture.session().audit();
    const auto generation_before_stale_accepts =
      shadow.acceptedTargetGeneration();
    EXPECT_FALSE(shadow.accepts(trace));
    const auto after_stale_accepts = fixture.session().audit();
    EXPECT_EQ(
      after_stale_accepts.store_revision,
      before_stale_accepts.store_revision);
    EXPECT_EQ(
      shadow.acceptedTargetGeneration(),
      generation_before_stale_accepts);

    const auto second = issueTarget(
      policy, fixture.pipeline().publish(10.2), healthyRig());
    const auto result = shadow.processBatch(
      *second.runtime, *second.capability);
    EXPECT_EQ(
      result.status(),
      livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
    EXPECT_EQ(result.skippedTombstonedSourceCount(), 1U);
    EXPECT_TRUE(result.decisions().empty());
  }
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  SelectionTraceIsRejectedByForeignModuleAndStore)
{
  StoreFixture owner_fixture;
  auto owner_policy = makePolicy();
  const auto target = issueTarget(
    owner_policy, owner_fixture.pipeline().publish(10.1),
    healthyRig());
  livo::MultiCameraLandmarkLinearizationShadow owner(
    owner_fixture.session(), owner_fixture.batchAuthority(),
    owner_fixture.runtimeAuthority(), owner_policy,
    enabledConfig());
  const auto selected = owner.processBatch(
    *target.runtime, *target.capability);
  ASSERT_EQ(selected.decisions().size(), 1U);
  const auto & trace =
    *selected.decisions().front().selectionTrace();

  StoreFixture foreign_fixture;
  auto foreign_policy = makePolicy();
  livo::MultiCameraLandmarkLinearizationShadow foreign(
    foreign_fixture.session(), foreign_fixture.batchAuthority(),
    foreign_fixture.runtimeAuthority(), foreign_policy,
    enabledConfig());

  const auto owner_before = owner_fixture.session().audit();
  const auto foreign_before = foreign_fixture.session().audit();
  EXPECT_TRUE(owner.accepts(trace));
  EXPECT_FALSE(foreign.accepts(trace));
  const auto owner_after = owner_fixture.session().audit();
  const auto foreign_after = foreign_fixture.session().audit();
  EXPECT_EQ(
    owner_after.store_revision, owner_before.store_revision);
  EXPECT_EQ(
    foreign_after.store_revision, foreign_before.store_revision);
  EXPECT_EQ(owner.acceptedTargetGeneration(), 1U);
  EXPECT_EQ(foreign.acceptedTargetGeneration(), 0U);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  StrictCrossCameraMissingEvidenceRejectsAndRelativeFallbackIsExplicit)
{
  {
    StoreFixture fixture;
    auto policy = makePolicy();
    auto observations = healthyRig();
    disableCamera(observations, 0U);
    const auto target = issueTarget(
      policy, fixture.pipeline().publish(10.1), observations);
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());

    const auto result = shadow.processBatch(
      *target.runtime, *target.capability);

    ASSERT_EQ(result.decisions().size(), 1U);
    const auto & decision = result.decisions().front();
    EXPECT_EQ(
      decision.status(),
      livo::MultiCameraLandmarkShadowDecisionStatus::
      kStrictCrossCameraEvidenceUnavailable);
    EXPECT_STREQ(
      decision.photometricPathLabel(),
      "calibrated_evidence_unavailable");
    EXPECT_EQ(decision.selectionTrace(), nullptr);
    EXPECT_EQ(result.strictPhotometricRejectedCount(), 1U);
  }
  {
    StoreFixture fixture;
    auto policy = makePolicy();
    auto observations = healthyRig();
    disableCamera(observations, 0U);
    auto config = enabledConfig();
    config.allow_relative_affine_experiment = true;
    const auto target = issueTarget(
      policy, fixture.pipeline().publish(10.1), observations);
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, config);

    const auto result = shadow.processBatch(
      *target.runtime, *target.capability);

    ASSERT_EQ(result.decisions().size(), 1U);
    const auto * trace =
      result.decisions().front().selectionTrace();
    ASSERT_NE(trace, nullptr);
    EXPECT_STREQ(
      trace->photometricPathLabel(),
      "experimental_non_radiometric");
    EXPECT_TRUE(trace->experimental());
    EXPECT_FALSE(trace->crossCameraComparable());
    EXPECT_TRUE(trace->configuredRelativeAffineExperiment());
  }
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  FourCameraPolicyAcceptsCanonicalSlotsAndRejectsMisplacedPermutations)
{
  StoreFixture fixture;
  const RuntimePublication publication =
    fixture.pipeline().publish(10.1);
  const std::array<std::uint8_t, 4U> canonical_order{{
    0U, 1U, 2U, 3U}};
  std::array<std::uint8_t, 4U> order{{0U, 1U, 2U, 3U}};
  std::size_t permutation_count = 0U;
  std::size_t accepted_count = 0U;
  std::size_t rejected_count = 0U;
  do
  {
    auto policy = makePolicy();
    auto observations = healthyRig();
    livo::AdaptiveVisualPolicy::CameraObservations permuted;
    for (std::size_t index = 0U; index < order.size(); ++index)
    {
      permuted[index] = observations[order[index]];
    }
    const auto target = issueTarget(
      policy, publication, permuted);
    const std::uint64_t policy_revision_before =
      policy.revision();
    const auto hysteresis_before = policy.hysteresis();
    livo::MultiCameraLandmarkLinearizationShadow shadow(
      fixture.session(), fixture.batchAuthority(),
      fixture.runtimeAuthority(), policy, enabledConfig());
    const auto result = shadow.processBatch(
      *target.runtime, *target.capability);

    if (order == canonical_order)
    {
      EXPECT_EQ(
        target.capability->decisionPlan().reason,
        livo::AdaptivePolicyReason::kReady);
      EXPECT_EQ(
        result.status(),
        livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);
      ASSERT_EQ(result.decisions().size(), 1U);
      EXPECT_NE(
        result.decisions().front().selectionTrace(), nullptr);
      EXPECT_EQ(result.acceptedTargetGeneration(), 1U);
      EXPECT_EQ(shadow.acceptedTargetGeneration(), 1U);
      ++accepted_count;
    }
    else
    {
      EXPECT_EQ(
        target.capability->decisionPlan().reason,
        livo::AdaptivePolicyReason::kInvalidCameraObservations);
      EXPECT_EQ(
        result.status(),
        livo::MultiCameraLandmarkShadowBatchStatus::
        kPolicyCapabilityRejected);
      EXPECT_TRUE(result.decisions().empty());
      EXPECT_EQ(result.acceptedTargetGeneration(), 0U);
      EXPECT_EQ(shadow.acceptedTargetGeneration(), 0U);
      EXPECT_EQ(policy.revision(), policy_revision_before);
      for (std::size_t camera_id = 0U;
           camera_id < hysteresis_before.size(); ++camera_id)
      {
        const auto & before = hysteresis_before[camera_id];
        const auto & after = policy.hysteresis()[camera_id];
        EXPECT_EQ(after.isolated, before.isolated);
        EXPECT_EQ(
          after.consecutive_unhealthy,
          before.consecutive_unhealthy);
        EXPECT_EQ(
          after.consecutive_healthy,
          before.consecutive_healthy);
      }
      ++rejected_count;
    }
    ++permutation_count;
  }
  while (std::next_permutation(order.begin(), order.end()));
  EXPECT_EQ(permutation_count, 24U);
  EXPECT_EQ(accepted_count, 1U);
  EXPECT_EQ(rejected_count, 23U);
}

TEST(
  MultiCameraLandmarkLinearizationShadow,
  AcceptedGenerationExhaustionIsPermanentAndFailClosed)
{
  StoreFixture fixture;
  auto policy = makePolicy();
  auto config = enabledConfig();
  config.maximum_accepted_target_generations = 1U;
  livo::MultiCameraLandmarkLinearizationShadow shadow(
    fixture.session(), fixture.batchAuthority(),
    fixture.runtimeAuthority(), policy, config);
  const auto first = issueTarget(
    policy, fixture.pipeline().publish(10.1), healthyRig());
  ASSERT_EQ(
    shadow.processBatch(
      *first.runtime, *first.capability).status(),
    livo::MultiCameraLandmarkShadowBatchStatus::kAccepted);

  const auto second = issueTarget(
    policy, fixture.pipeline().publish(10.2), healthyRig());
  const auto exhausted = shadow.processBatch(
    *second.runtime, *second.capability);
  EXPECT_EQ(
    exhausted.status(),
    livo::MultiCameraLandmarkShadowBatchStatus::
    kGenerationExhausted);
  EXPECT_EQ(exhausted.acceptedTargetGeneration(), 1U);
  EXPECT_FALSE(shadow.active());

  const auto third = issueTarget(
    policy, fixture.pipeline().publish(10.3), healthyRig());
  EXPECT_EQ(
    shadow.processBatch(
      *third.runtime, *third.capability).status(),
    livo::MultiCameraLandmarkShadowBatchStatus::
    kGenerationExhausted);
  EXPECT_EQ(shadow.acceptedTargetGeneration(), 1U);
}

}  // namespace
