#include "four_camera_pinhole_rig.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <openssl/evp.h>

namespace livo
{
namespace
{

bool validRotation(const Eigen::Matrix3d & rotation)
{
  return
    rotation.allFinite() &&
    std::abs(rotation.determinant() - 1.0) <= 1.0e-6 &&
    (rotation.transpose() * rotation -
     Eigen::Matrix3d::Identity()).norm() <= 1.0e-6;
}

std::array<
  FourCameraPinholeCameraConfig,
  kFourCameraPinholeRigCameraCount>
validatedCamerasById(
  const FourCameraPinholeRigConfig & config)
{
  std::array<
    FourCameraPinholeCameraConfig,
    kFourCameraPinholeRigCameraCount> cameras_by_id;
  std::array<bool, kFourCameraPinholeRigCameraCount> seen{
    {false, false, false, false}};
  for (const FourCameraPinholeCameraConfig & camera : config.cameras)
  {
    const std::size_t camera_id =
      static_cast<std::size_t>(camera.camera_id);
    if (
      camera_id >= kFourCameraPinholeRigCameraCount ||
      seen[camera_id])
    {
      throw std::invalid_argument(
        "four-camera Pinhole rig ids must be unique 0..3");
    }
    if (
      !validRotation(camera.rotation_camera_from_imu) ||
      !camera.position_camera_from_imu.allFinite())
    {
      throw std::invalid_argument(
        "four-camera Pinhole rig rotation must be finite SO(3) "
        "and translation must be finite");
    }
    seen[camera_id] = true;
    cameras_by_id[camera_id] = camera;
  }
  return cameras_by_id;
}

class CanonicalBytes
{
public:
  void appendByte(const std::uint8_t value)
  {
    bytes_.push_back(value);
  }

  void appendUnsigned(const std::uint64_t value)
  {
    // Canonical unsigned integers are always big-endian, independent of the
    // host architecture.
    for (int shift = 56; shift >= 0; shift -= 8)
    {
      appendByte(
        static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
  }

  void appendSigned(const int value)
  {
    appendUnsigned(
      static_cast<std::uint64_t>(
        static_cast<std::int64_t>(value)));
  }

  void appendDouble(const double value)
  {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    // Validated IEEE-754 binary64 values are encoded by their bits in the
    // same big-endian integer form. Signed zero is semantically canonical.
    const double canonical_value = value == 0.0 ? 0.0 : value;
    std::uint64_t bits = 0U;
    std::memcpy(&bits, &canonical_value, sizeof(bits));
    appendUnsigned(bits);
  }

  void appendString(const std::string & value)
  {
    appendUnsigned(static_cast<std::uint64_t>(value.size()));
    for (const char character : value)
    {
      appendByte(static_cast<std::uint8_t>(character));
    }
  }

  void appendFingerprint(
    const FourCameraPinholeRigFingerprint & fingerprint)
  {
    bytes_.insert(
      bytes_.end(), fingerprint.begin(), fingerprint.end());
  }

  const std::vector<std::uint8_t> & value() const noexcept
  {
    return bytes_;
  }

private:
  std::vector<std::uint8_t> bytes_;
};

void appendProjection(
  CanonicalBytes & canonical,
  const FourCameraPinholeProjectionConfig & projection)
{
  // This explicit field order is the v1 wire contract. Never hash a raw
  // struct, std::string object, or Eigen storage.
  canonical.appendString(projection.model);
  canonical.appendSigned(projection.width);
  canonical.appendSigned(projection.height);
  canonical.appendDouble(projection.fx);
  canonical.appendDouble(projection.fy);
  canonical.appendDouble(projection.cx);
  canonical.appendDouble(projection.cy);
  for (const double coefficient :
       projection.distortion_coefficients)
  {
    canonical.appendDouble(coefficient);
  }
}

FourCameraPinholeRigFingerprint sha256(
  const CanonicalBytes & canonical)
{
  FourCameraPinholeRigFingerprint digest{};
  const std::vector<std::uint8_t> & bytes = canonical.value();
  const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>
    context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  unsigned int digest_size = 0U;
  if (
    context == nullptr ||
    EVP_DigestInit_ex(
      context.get(), EVP_sha256(), nullptr) != 1 ||
    EVP_DigestUpdate(
      context.get(), bytes.data(), bytes.size()) != 1 ||
    EVP_DigestFinal_ex(
      context.get(), digest.data(), &digest_size) != 1 ||
    digest_size != digest.size())
  {
    throw std::runtime_error(
      "failed to compute four-camera rig SHA-256");
  }
  return digest;
}

FourCameraPinholeRigFingerprint modelFingerprint(
  const FourCameraPinholeProjectionConfig & projection)
{
  CanonicalBytes canonical;
  canonical.appendString(
    "livo/four-camera-pinhole-model/v1");
  appendProjection(canonical, projection);
  return sha256(canonical);
}

std::uint64_t revisionFromFingerprint(
  const FourCameraPinholeRigFingerprint & fingerprint) noexcept
{
  std::uint64_t revision = 0U;
  for (std::size_t index = 0U; index < sizeof(revision); ++index)
  {
    revision =
      (revision << 8U) |
      static_cast<std::uint64_t>(fingerprint[index]);
  }
  return revision == 0U ? 1U : revision;
}

FourCameraPinholeRigFingerprint rigFingerprint(
  const std::array<
    FourCameraPinholeCameraConfig,
    kFourCameraPinholeRigCameraCount> & cameras_by_id,
  const std::array<
    FourCameraPinholeRigFingerprint,
    kFourCameraPinholeRigCameraCount> & model_fingerprints)
{
  CanonicalBytes canonical;
  canonical.appendString("livo/four-camera-pinhole-rig/v1");
  canonical.appendUnsigned(kFourCameraPinholeRigCameraCount);
  for (std::size_t camera_id = 0U;
       camera_id < kFourCameraPinholeRigCameraCount;
       ++camera_id)
  {
    const FourCameraPinholeCameraConfig & camera =
      cameras_by_id[camera_id];
    canonical.appendByte(camera.camera_id);
    canonical.appendFingerprint(model_fingerprints[camera_id]);
    for (Eigen::Index row = 0; row < 3; ++row)
    {
      for (Eigen::Index column = 0; column < 3; ++column)
      {
        canonical.appendDouble(
          camera.rotation_camera_from_imu(row, column));
      }
    }
    for (Eigen::Index index = 0; index < 3; ++index)
    {
      canonical.appendDouble(
        camera.position_camera_from_imu(index));
    }
  }
  return sha256(canonical);
}

PinholeCameraProjectionConfig adapterConfig(
  const FourCameraPinholeProjectionConfig & projection,
  const std::uint64_t model_revision)
{
  PinholeCameraProjectionConfig config;
  config.model = projection.model;
  config.width = projection.width;
  config.height = projection.height;
  config.fx = projection.fx;
  config.fy = projection.fy;
  config.cx = projection.cx;
  config.cy = projection.cy;
  config.distortion_coefficients =
    projection.distortion_coefficients;
  config.model_revision = model_revision;
  return config;
}

}  // namespace

bool FourCameraPinholeRigEndpointResult::shadowReady() const noexcept
{
  return
    status ==
    FourCameraPinholeRigEndpointStatus::
    kShadowOnlyMissingUncertaintyReceipt &&
    endpoint.has_value();
}

bool FourCameraPinholeRigEndpointResult::productionReady() const
{
  return
    endpoint.has_value() &&
    endpoint->uncertainty.productionReady(*endpoint);
}

FourCameraPinholeRig::FourCameraPinholeRig(
  const FourCameraPinholeRigConfig & config)
{
  const auto cameras_by_id = validatedCamerasById(config);
  std::array<
    FourCameraPinholeRigFingerprint,
    kFourCameraPinholeRigCameraCount> model_fingerprints;
  for (std::size_t camera_id = 0U;
       camera_id < kFourCameraPinholeRigCameraCount;
       ++camera_id)
  {
    const FourCameraPinholeCameraConfig & camera =
      cameras_by_id[camera_id];
    model_fingerprints[camera_id] =
      modelFingerprint(camera.projection);
    const std::uint64_t model_revision =
      revisionFromFingerprint(model_fingerprints[camera_id]);
    try
    {
      camera_models_[camera_id] =
        std::make_shared<const PinholeCameraProjectionAdapter>(
          adapterConfig(camera.projection, model_revision));
    }
    catch (const std::invalid_argument & error)
    {
      throw std::invalid_argument(
        "invalid Pinhole projection for camera " +
        std::to_string(camera_id) + ": " + error.what());
    }
    rotations_camera_from_imu_[camera_id] =
      camera.rotation_camera_from_imu;
    positions_camera_from_imu_[camera_id] =
      camera.position_camera_from_imu;
  }
  configuration_fingerprint_ =
    rigFingerprint(cameras_by_id, model_fingerprints);
  rig_revision_ =
    revisionFromFingerprint(configuration_fingerprint_);
}

std::uint64_t FourCameraPinholeRig::rigRevision() const noexcept
{
  return rig_revision_;
}

FourCameraPinholeRigFingerprint
FourCameraPinholeRig::configurationFingerprint() const noexcept
{
  return configuration_fingerprint_;
}

std::shared_ptr<const PinholeCameraProjectionAdapter>
FourCameraPinholeRig::cameraModel(
  const std::uint8_t camera_id) const & noexcept
{
  if (
    static_cast<std::size_t>(camera_id) >=
    kFourCameraPinholeRigCameraCount)
  {
    return {};
  }
  return camera_models_[camera_id];
}

FourCameraPinholeRigEndpointResult
FourCameraPinholeRig::shadowEndpoint(
  const std::uint8_t camera_id,
  const FourCameraPinholeRigRevisionExpectation
    expected_revision) const &
{
  FourCameraPinholeRigEndpointResult result;
  const std::shared_ptr<const PinholeCameraProjectionAdapter> model =
    cameraModel(camera_id);
  if (!model)
  {
    result.status =
      FourCameraPinholeRigEndpointStatus::kInvalidCameraId;
    result.reason = "invalid_camera_id";
    return result;
  }
  if (
    expected_revision.camera_model_revision !=
    model->modelRevision())
  {
    result.status =
      FourCameraPinholeRigEndpointStatus::
      kCameraModelRevisionMismatch;
    result.reason = "foreign_camera_model_revision";
    return result;
  }
  if (expected_revision.rig_revision != rig_revision_)
  {
    result.status =
      FourCameraPinholeRigEndpointStatus::kRigRevisionMismatch;
    result.reason = "foreign_rig_revision";
    return result;
  }

  MigrationEndpoint endpoint;
  endpoint.camera_id = camera_id;
  endpoint.camera_model = model;
  endpoint.camera_model_revision = model->modelRevision();
  endpoint.rig_revision = rig_revision_;
  endpoint.rotation_imu_from_camera =
    rotations_camera_from_imu_[camera_id].transpose();
  endpoint.position_imu_from_camera =
    -endpoint.rotation_imu_from_camera *
    positions_camera_from_imu_[camera_id];

  result.status =
    FourCameraPinholeRigEndpointStatus::
    kShadowOnlyMissingUncertaintyReceipt;
  result.reason =
    "missing_measured_endpoint_uncertainty_receipt";
  result.endpoint = std::move(endpoint);
  return result;
}

const char * fourCameraPinholeRigEndpointStatusName(
  const FourCameraPinholeRigEndpointStatus status) noexcept
{
  switch (status)
  {
    case FourCameraPinholeRigEndpointStatus::
      kShadowOnlyMissingUncertaintyReceipt:
      return "shadow_only_missing_uncertainty_receipt";
    case FourCameraPinholeRigEndpointStatus::kInvalidCameraId:
      return "invalid_camera_id";
    case FourCameraPinholeRigEndpointStatus::
      kCameraModelRevisionMismatch:
      return "camera_model_revision_mismatch";
    case FourCameraPinholeRigEndpointStatus::kRigRevisionMismatch:
      return "rig_revision_mismatch";
  }
  return "unknown";
}

}  // namespace livo
