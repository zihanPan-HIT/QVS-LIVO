#include "trusted_photometric_calibration_loader.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <json/json.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX
#define LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX ""
#endif

#ifndef LIVO_PHOTOMETRIC_TRUST_DOMAIN
#define LIVO_PHOTOMETRIC_TRUST_DOMAIN ""
#endif

namespace livo
{
namespace
{

constexpr char kRuntimePayloadFilename[] =
  "photometric_calibration_runtime_payload.json";
constexpr char kRuntimeSignatureFilename[] =
  "photometric_calibration_runtime_signature.json";
constexpr char kReceiptFilename[] =
  "photometric_calibration_receipt.json";
constexpr char kSignatureType[] =
  "four_camera_photometric_calibration_runtime_signature";
constexpr char kReceiptHashScheme[] =
  "sha256_runtime_payload_lf_ed25519_signature_v1";
constexpr unsigned char kReceiptHashDomainSeparator[] =
  "livo_core.photometric_calibration.receipt_hash.v1\0";
constexpr char kSolverName[] =
  "livo_core_four_camera_photometric_calibration";
constexpr char kSolverVersion[] = "2.0.0";
constexpr std::size_t kCameraCount = 4U;
constexpr std::size_t kSha256Bytes = 32U;
constexpr std::size_t kEd25519PublicKeyBytes = 32U;
constexpr std::size_t kEd25519SignatureBytes = 64U;
constexpr std::size_t kMaximumPayloadBytes = 16U * 1024U * 1024U;
constexpr std::size_t kMaximumSignatureDocumentBytes = 64U * 1024U;
constexpr std::size_t kMaximumReceiptBytes = 16U * 1024U * 1024U;
constexpr std::size_t kMaximumArtifactBytes = 32U * 1024U * 1024U;
constexpr std::size_t kMaximumTotalArtifactBytes =
  128U * 1024U * 1024U;
constexpr double kMonotonicTolerance = 1.0e-12;

using Sha256Digest = std::array<unsigned char, kSha256Bytes>;

struct FileSnapshot
{
  std::string bytes;
};

struct DirectoryHandle
{
  int descriptor{-1};

  DirectoryHandle() = default;
  explicit DirectoryHandle(const int value)
  : descriptor(value)
  {
  }
  DirectoryHandle(const DirectoryHandle &) = delete;
  DirectoryHandle & operator=(const DirectoryHandle &) = delete;
  DirectoryHandle(DirectoryHandle && other) noexcept
  : descriptor(other.descriptor)
  {
    other.descriptor = -1;
  }
  DirectoryHandle & operator=(DirectoryHandle && other) noexcept
  {
    if (this != &other)
    {
      if (descriptor >= 0)
      {
        ::close(descriptor);
      }
      descriptor = other.descriptor;
      other.descriptor = -1;
    }
    return *this;
  }
  ~DirectoryHandle()
  {
    if (descriptor >= 0)
    {
      ::close(descriptor);
    }
  }
};

struct InternalLoad
{
  struct ParsedNoiseEvidence
  {
    std::uint8_t camera_id{0U};
    RawIntensityDomain domain;
    double standard_deviation_code_value{0.0};
    std::string source_id;
    std::string provenance;
    std::string model;
    std::string estimator_revision;
    std::uint64_t source_sample_count{0U};
  };

  TrustedPhotometricCalibrationLoadStatus status{
    TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract};
  bool structurally_valid{false};
  std::uint64_t revision{0U};
  std::vector<PhotometricCameraCalibration> cameras;
  std::vector<ParsedNoiseEvidence> noise_evidence;
  std::string diagnostic;
};

TrustedPhotometricCalibrationLoadResult makePublicResult(
  InternalLoad && internal,
  std::optional<TrustedPhotometricCalibrationBundle>
    trusted_noise_bundle = std::nullopt)
{
  TrustedPhotometricCalibrationLoadResult result;
  result.status = internal.status;
  result.structurally_valid = internal.structurally_valid;
  result.calibration_revision = internal.revision;
  result.diagnostic = std::move(internal.diagnostic);
  if (
    result.status ==
      TrustedPhotometricCalibrationLoadStatus::kLoaded &&
    result.structurally_valid &&
    result.calibration_revision != 0U &&
    internal.cameras.size() == kCameraCount)
  {
    result.calibration.emplace(std::move(internal.cameras));
    if (
      trusted_noise_bundle.has_value() &&
      trusted_noise_bundle->valid() &&
      trusted_noise_bundle->calibrationRevision() ==
        result.calibration_revision &&
      trusted_noise_bundle->accepts(*result.calibration))
    {
      result.trusted_noise_bundle.emplace(
        std::move(*trusted_noise_bundle));
    }
  }
  return result;
}

InternalLoad failure(
  const TrustedPhotometricCalibrationLoadStatus status,
  std::string diagnostic,
  const bool structurally_valid = false,
  const std::uint64_t revision = 0U)
{
  InternalLoad result;
  result.status = status;
  result.structurally_valid = structurally_valid;
  result.revision = revision;
  result.diagnostic = std::move(diagnostic);
  return result;
}

bool isBasename(const std::string & filename)
{
  return
    !filename.empty() &&
    filename != "." &&
    filename != ".." &&
    filename.size() <= 255U &&
    filename.find('/') == std::string::npos &&
    filename.find('\\') == std::string::npos &&
    filename.find('\0') == std::string::npos;
}

bool sameSnapshot(
  const struct stat & first,
  const struct stat & second)
{
  return
    first.st_dev == second.st_dev &&
    first.st_ino == second.st_ino &&
    first.st_mode == second.st_mode &&
    first.st_nlink == second.st_nlink &&
    first.st_size == second.st_size &&
    first.st_mtim.tv_sec == second.st_mtim.tv_sec &&
    first.st_mtim.tv_nsec == second.st_mtim.tv_nsec &&
    first.st_ctim.tv_sec == second.st_ctim.tv_sec &&
    first.st_ctim.tv_nsec == second.st_ctim.tv_nsec;
}

std::optional<DirectoryHandle> openBundleDirectory(
  const std::filesystem::path & path,
  std::string * const error)
{
  if (error == nullptr || path.empty())
  {
    return std::nullopt;
  }
  struct stat before{};
  if (::lstat(path.c_str(), &before) != 0)
  {
    *error = "bundle directory lstat failed: " +
      std::string(std::strerror(errno));
    return std::nullopt;
  }
  if (S_ISLNK(before.st_mode) || !S_ISDIR(before.st_mode))
  {
    *error = "bundle path must be a non-symlink directory";
    return std::nullopt;
  }
  int flags = O_RDONLY | O_DIRECTORY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  const int descriptor = ::open(path.c_str(), flags);
  if (descriptor < 0)
  {
    *error = "bundle directory open failed: " +
      std::string(std::strerror(errno));
    return std::nullopt;
  }
  DirectoryHandle handle(descriptor);
  struct stat opened{};
  struct stat after{};
  if (
    ::fstat(descriptor, &opened) != 0 ||
    ::lstat(path.c_str(), &after) != 0 ||
    !S_ISDIR(opened.st_mode) ||
    S_ISLNK(after.st_mode) ||
    opened.st_dev != before.st_dev ||
    opened.st_ino != before.st_ino ||
    opened.st_dev != after.st_dev ||
    opened.st_ino != after.st_ino)
  {
    *error = "bundle directory changed while opening";
    return std::nullopt;
  }
  return handle;
}

std::optional<FileSnapshot> readRegularFileOnce(
  const int directory_descriptor,
  const std::string & filename,
  const std::size_t maximum_bytes,
  std::string * const error)
{
  if (error == nullptr || !isBasename(filename))
  {
    if (error != nullptr)
    {
      *error = "artifact filename must be one safe basename";
    }
    return std::nullopt;
  }
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  const int descriptor =
    ::openat(directory_descriptor, filename.c_str(), flags);
  if (descriptor < 0)
  {
    *error = "cannot open " + filename + ": " +
      std::string(std::strerror(errno));
    return std::nullopt;
  }
  const auto close_descriptor = [descriptor](void *) {
      ::close(descriptor);
    };
  std::unique_ptr<void, decltype(close_descriptor)> guard(
    reinterpret_cast<void *>(1), close_descriptor);

  struct stat before{};
  if (
    ::fstat(descriptor, &before) != 0 ||
    !S_ISREG(before.st_mode) ||
    before.st_nlink != 1 ||
    before.st_size < 0)
  {
    *error = filename +
      " must be a single-link regular file";
    return std::nullopt;
  }
  const auto unsigned_size =
    static_cast<std::uintmax_t>(before.st_size);
  if (unsigned_size > maximum_bytes)
  {
    *error = filename + " exceeds its size cap";
    return std::nullopt;
  }
  std::string bytes(static_cast<std::size_t>(unsigned_size), '\0');
  std::size_t offset = 0U;
  while (offset < bytes.size())
  {
    const ssize_t count = ::read(
      descriptor,
      bytes.data() + offset,
      bytes.size() - offset);
    if (count < 0)
    {
      if (errno == EINTR)
      {
        continue;
      }
      *error = "cannot read " + filename + ": " +
        std::string(std::strerror(errno));
      return std::nullopt;
    }
    if (count == 0)
    {
      *error = filename + " was truncated while reading";
      return std::nullopt;
    }
    offset += static_cast<std::size_t>(count);
  }
  unsigned char trailing_byte = 0U;
  ssize_t trailing_count = -1;
  do
  {
    trailing_count = ::read(descriptor, &trailing_byte, 1U);
  } while (trailing_count < 0 && errno == EINTR);
  if (trailing_count != 0)
  {
    *error = filename + " grew while reading";
    return std::nullopt;
  }
  struct stat after{};
  struct stat path_after{};
  if (
    ::fstat(descriptor, &after) != 0 ||
    ::fstatat(
      directory_descriptor,
      filename.c_str(),
      &path_after,
      AT_SYMLINK_NOFOLLOW) != 0 ||
    !sameSnapshot(before, after) ||
    after.st_dev != path_after.st_dev ||
    after.st_ino != path_after.st_ino ||
    S_ISLNK(path_after.st_mode))
  {
    *error = filename + " changed while reading";
    return std::nullopt;
  }
  return FileSnapshot{std::move(bytes)};
}

std::optional<Json::Value> parseStrictObject(
  const std::string & bytes,
  std::string * const error)
{
  Json::CharReaderBuilder builder;
  Json::CharReaderBuilder::strictMode(&builder.settings_);
  builder["collectComments"] = false;
  builder["rejectDupKeys"] = true;
  builder["failIfExtra"] = true;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  Json::Value root;
  std::string parse_errors;
  if (
    !reader->parse(
      bytes.data(),
      bytes.data() + bytes.size(),
      &root,
      &parse_errors) ||
    !root.isObject())
  {
    *error = "strict JSON object parse failed: " + parse_errors;
    return std::nullopt;
  }
  return root;
}

std::set<std::string> memberSet(const Json::Value & value)
{
  const std::vector<std::string> names = value.getMemberNames();
  return std::set<std::string>(names.begin(), names.end());
}

bool exactMembers(
  const Json::Value & value,
  const std::set<std::string> & expected)
{
  return value.isObject() && memberSet(value) == expected;
}

std::optional<Sha256Digest> sha256(const std::string_view bytes)
{
  Sha256Digest digest{};
  unsigned int output_bytes = 0U;
  if (
    EVP_Digest(
      bytes.data(),
      bytes.size(),
      digest.data(),
      &output_bytes,
      EVP_sha256(),
      nullptr) != 1 ||
    output_bytes != digest.size())
  {
    return std::nullopt;
  }
  return digest;
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

std::optional<Sha256Digest> receiptSha256V1(
  const std::string & payload,
  const std::vector<unsigned char> & signature)
{
  if (
    payload.empty() ||
    payload.back() != '\n' ||
    (
      payload.size() > 1U &&
      payload[payload.size() - 2U] == '\n'
    ) ||
    signature.size() != kEd25519SignatureBytes)
  {
    return std::nullopt;
  }
  EVP_MD_CTX * const raw_context = EVP_MD_CTX_new();
  if (raw_context == nullptr)
  {
    return std::nullopt;
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
    raw_context, &EVP_MD_CTX_free);
  Sha256Digest digest{};
  unsigned int digest_size = 0U;
  const std::array<unsigned char, 8U> payload_length =
    bigEndianLength(payload.size());
  const std::array<unsigned char, 8U> signature_length =
    bigEndianLength(signature.size());
  if (
    EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
    EVP_DigestUpdate(
      context.get(),
      kReceiptHashDomainSeparator,
      sizeof(kReceiptHashDomainSeparator) - 1U) != 1 ||
    EVP_DigestUpdate(
      context.get(),
      payload_length.data(),
      payload_length.size()) != 1 ||
    EVP_DigestUpdate(
      context.get(), payload.data(), payload.size()) != 1 ||
    EVP_DigestUpdate(
      context.get(),
      signature_length.data(),
      signature_length.size()) != 1 ||
    EVP_DigestUpdate(
      context.get(), signature.data(), signature.size()) != 1 ||
    EVP_DigestFinal_ex(
      context.get(), digest.data(), &digest_size) != 1 ||
    digest_size != digest.size())
  {
    return std::nullopt;
  }
  return digest;
}

bool isLowercaseSha256(const std::string & value)
{
  if (value.size() != 64U)
  {
    return false;
  }
  for (const char character : value)
  {
    if (
      !(
        (character >= '0' && character <= '9') ||
        (character >= 'a' && character <= 'f')
      ))
    {
      return false;
    }
  }
  return true;
}

std::string hexEncode(
  const unsigned char * const bytes,
  const std::size_t size)
{
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::hex << std::setfill('0');
  for (std::size_t index = 0U; index < size; ++index)
  {
    stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return stream.str();
}

std::optional<std::vector<unsigned char>> hexDecode(
  const std::string & encoded)
{
  if (encoded.empty() || encoded.size() % 2U != 0U)
  {
    return std::nullopt;
  }
  std::vector<unsigned char> decoded(encoded.size() / 2U, 0U);
  for (std::size_t index = 0U; index < decoded.size(); ++index)
  {
    const std::string byte_text = encoded.substr(index * 2U, 2U);
    char * end = nullptr;
    errno = 0;
    const unsigned long value =
      std::strtoul(byte_text.c_str(), &end, 16);
    if (
      errno != 0 ||
      end != byte_text.c_str() + byte_text.size() ||
      value > 255UL)
    {
      return std::nullopt;
    }
    decoded[index] = static_cast<unsigned char>(value);
  }
  return decoded;
}

std::optional<std::vector<unsigned char>> decodeBase64(
  const std::string & encoded)
{
  if (
    encoded.empty() ||
    encoded.size() % 4U != 0U ||
    encoded.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    return std::nullopt;
  }
  std::vector<unsigned char> decoded(
    3U * (encoded.size() / 4U), 0U);
  const int count = EVP_DecodeBlock(
    decoded.data(),
    reinterpret_cast<const unsigned char *>(encoded.data()),
    static_cast<int>(encoded.size()));
  if (count < 0)
  {
    return std::nullopt;
  }
  std::size_t actual_size = static_cast<std::size_t>(count);
  if (!encoded.empty() && encoded.back() == '=')
  {
    --actual_size;
  }
  if (encoded.size() >= 2U && encoded[encoded.size() - 2U] == '=')
  {
    --actual_size;
  }
  decoded.resize(actual_size);
  return decoded;
}

bool verifyEd25519(
  const std::vector<unsigned char> & public_key,
  const std::vector<unsigned char> & signature,
  const std::string & payload)
{
  if (
    public_key.size() != kEd25519PublicKeyBytes ||
    signature.size() != kEd25519SignatureBytes)
  {
    return false;
  }
  EVP_PKEY * const raw_key = EVP_PKEY_new_raw_public_key_ex(
    nullptr,
    "ED25519",
    nullptr,
    public_key.data(),
    public_key.size());
  if (raw_key == nullptr)
  {
    return false;
  }
  std::unique_ptr<EVP_PKEY, decltype(&EVP_PKEY_free)> key(
    raw_key, &EVP_PKEY_free);
  EVP_MD_CTX * const raw_context = EVP_MD_CTX_new();
  if (raw_context == nullptr)
  {
    return false;
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
    raw_context, &EVP_MD_CTX_free);
  return
    EVP_DigestVerifyInit(
      context.get(), nullptr, nullptr, nullptr, key.get()) == 1 &&
    EVP_DigestVerify(
      context.get(),
      signature.data(),
      signature.size(),
      reinterpret_cast<const unsigned char *>(payload.data()),
      payload.size()) == 1;
}

std::uint64_t revisionFromDigest(const Sha256Digest & digest)
{
  for (std::size_t block = 0U; block < 4U; ++block)
  {
    std::uint64_t revision = 0U;
    for (std::size_t index = 0U; index < 8U; ++index)
    {
      revision =
        (revision << 8U) |
        static_cast<std::uint64_t>(digest[block * 8U + index]);
    }
    if (revision != 0U)
    {
      return revision;
    }
  }
  return 1U;
}

bool finiteJsonNumber(const Json::Value & value)
{
  return value.isNumeric() && std::isfinite(value.asDouble());
}

bool finitePositiveJsonNumber(const Json::Value & value)
{
  return finiteJsonNumber(value) && value.asDouble() > 0.0;
}

std::optional<std::vector<double>> parseNumericArtifact(
  const std::string & bytes)
{
  std::istringstream stream(bytes);
  stream.imbue(std::locale::classic());
  std::vector<double> values;
  double value = 0.0;
  while (stream >> value)
  {
    if (!std::isfinite(value))
    {
      return std::nullopt;
    }
    values.push_back(value);
  }
  if (!stream.eof())
  {
    return std::nullopt;
  }
  return values;
}

bool valuesMatch(
  const Json::Value & first,
  const Json::Value & second,
  const std::set<std::string> & fields)
{
  if (!first.isObject() || !second.isObject())
  {
    return false;
  }
  for (const std::string & field : fields)
  {
    if (!first.isMember(field) || first[field] != second[field])
    {
      return false;
    }
  }
  return true;
}

InternalLoad loadInternal(const std::filesystem::path & bundle_directory)
{
  std::string io_error;
  std::optional<DirectoryHandle> directory =
    openBundleDirectory(bundle_directory, &io_error);
  if (!directory)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundlePath,
      io_error);
  }

  const auto read_fixed =
    [&directory, &io_error](
      const char * const filename,
      const std::size_t maximum_bytes)
      -> std::optional<FileSnapshot>
    {
      return readRegularFileOnce(
        directory->descriptor,
        filename,
        maximum_bytes,
        &io_error);
    };
  const std::optional<FileSnapshot> payload_snapshot =
    read_fixed(kRuntimePayloadFilename, kMaximumPayloadBytes);
  if (!payload_snapshot)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kIoFailure,
      io_error);
  }
  const std::optional<FileSnapshot> signature_snapshot =
    read_fixed(
      kRuntimeSignatureFilename,
      kMaximumSignatureDocumentBytes);
  if (!signature_snapshot)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kIoFailure,
      io_error);
  }
  const std::optional<FileSnapshot> receipt_snapshot =
    read_fixed(kReceiptFilename, kMaximumReceiptBytes);
  if (!receipt_snapshot)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kIoFailure,
      io_error);
  }

  std::string parse_error;
  const std::optional<Json::Value> payload =
    parseStrictObject(payload_snapshot->bytes, &parse_error);
  if (!payload)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidJson,
      "runtime payload " + parse_error);
  }
  const std::optional<Json::Value> detached =
    parseStrictObject(signature_snapshot->bytes, &parse_error);
  if (!detached)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidJson,
      "detached signature " + parse_error);
  }
  const std::optional<Json::Value> receipt =
    parseStrictObject(receipt_snapshot->bytes, &parse_error);
  if (!receipt)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidJson,
      "receipt " + parse_error);
  }

  if (
    !exactMembers(
      *detached,
      {
        "algorithm",
        "payload_file",
        "payload_sha256",
        "public_key_sha256",
        "receipt_hash_scheme",
        "receipt_sha256",
        "schema_version",
        "signature_base64",
        "signature_type",
        "solver",
        "trust_domain",
      }) ||
    (*detached)["schema_version"].asInt() != 1 ||
    (*detached)["signature_type"].asString() != kSignatureType ||
    (*detached)["algorithm"].asString() != "ed25519" ||
    (*detached)["receipt_hash_scheme"].asString() !=
      kReceiptHashScheme ||
    (*detached)["payload_file"].asString() !=
      kRuntimePayloadFilename ||
    !exactMembers((*detached)["solver"], {"name", "version"}) ||
    (*detached)["solver"]["name"].asString() != kSolverName ||
    (*detached)["solver"]["version"].asString() != kSolverVersion)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "detached signature metadata is not the exact allowlisted shape");
  }
  const std::optional<Sha256Digest> payload_digest =
    sha256(payload_snapshot->bytes);
  if (!payload_digest)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kIoFailure,
      "SHA-256 provider is unavailable");
  }
  const std::string payload_digest_hex =
    hexEncode(payload_digest->data(), payload_digest->size());
  if ((*detached)["payload_sha256"].asString() != payload_digest_hex)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kPayloadHashMismatch,
      "runtime payload hash does not match detached metadata");
  }

  const int payload_schema_version =
    (*payload)["schema_version"].isInt() ?
    (*payload)["schema_version"].asInt() : 0;
  if (
    !exactMembers(
      *payload,
      {
        "artifacts_sha256",
        "calibration_config_payload",
        "calibration_config_sha256",
        "calibration_set_id",
        "cameras",
        "declared_source_id",
        "frame_metadata",
        "held_out_metrics",
        "integrity_scope",
        "manifest_sha256",
        "receipt_type",
        "receipt_hash_scheme",
        "schema_version",
        "solver",
        "source_bundle_file",
        "source_bundle_sha256",
        "source_counts",
        "thresholds",
        "trust_boundary",
        "trust_root",
      }) ||
    (
      payload_schema_version != 1 &&
      payload_schema_version != 2
    ) ||
    (*payload)["receipt_type"].asString() !=
      "four_camera_photometric_calibration" ||
    (*payload)["receipt_hash_scheme"].asString() !=
      kReceiptHashScheme ||
    (*payload)["receipt_hash_scheme"] !=
      (*detached)["receipt_hash_scheme"] ||
    (*payload)["solver"] != (*detached)["solver"])
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "runtime payload is not the exact signed receipt shape");
  }
  const Json::Value & trust_root = (*payload)["trust_root"];
  if (
    !exactMembers(
      trust_root, {"public_key_sha256", "trust_domain"}) ||
    trust_root["public_key_sha256"].asString() !=
      (*detached)["public_key_sha256"].asString() ||
    trust_root["trust_domain"].asString() !=
      (*detached)["trust_domain"].asString())
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "payload and detached trust roots differ");
  }
  const Json::Value & config =
    (*payload)["calibration_config_payload"];
  if (
    !exactMembers(
      config,
      {
        "calibration_receipt_file",
        "calibration_set_id",
        "cameras",
        "dataset_scope",
        "frame_metadata",
        "runtime_payload_file",
        "runtime_signature_file",
        "schema_version",
        "source_bundle_file",
        "source_bundle_sha256",
        "status",
        "trust_root",
      }) ||
    !config["schema_version"].isInt() ||
    config["schema_version"].asInt() !=
      payload_schema_version ||
    config["status"].asString() != "available" ||
    config["runtime_payload_file"].asString() !=
      kRuntimePayloadFilename ||
    config["runtime_signature_file"].asString() !=
      kRuntimeSignatureFilename ||
    config["calibration_receipt_file"].asString() !=
      kReceiptFilename ||
    config["trust_root"] != trust_root ||
    config["calibration_set_id"] != (*payload)["calibration_set_id"] ||
    config["frame_metadata"] != (*payload)["frame_metadata"] ||
    config["source_bundle_file"] != (*payload)["source_bundle_file"] ||
    config["source_bundle_sha256"] !=
      (*payload)["source_bundle_sha256"])
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "signed calibration config does not match its receipt envelope");
  }

  Json::Value expected_receipt = *payload;
  Json::Value embedded_signature(Json::objectValue);
  embedded_signature["algorithm"] = "ed25519";
  embedded_signature["public_key_sha256"] =
    (*detached)["public_key_sha256"];
  embedded_signature["signature_base64"] =
    (*detached)["signature_base64"];
  expected_receipt["signature"] = embedded_signature;
  Json::Value receipt_without_hash = *receipt;
  if (
    !receipt_without_hash.isMember("receipt_sha256") ||
    !receipt_without_hash["receipt_sha256"].isString() ||
    !(*detached)["receipt_sha256"].isString())
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "receipt structural hash field is missing");
  }
  receipt_without_hash.removeMember("receipt_sha256");
  if (receipt_without_hash != expected_receipt)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "receipt is not exactly equivalent to payload and detached signature");
  }
  const std::optional<std::vector<unsigned char>> signature =
    decodeBase64((*detached)["signature_base64"].asString());
  if (
    !signature ||
    signature->size() != kEd25519SignatureBytes)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "receipt signature encoding is invalid");
  }
  const std::optional<Sha256Digest> calculated_receipt_hash =
    receiptSha256V1(payload_snapshot->bytes, *signature);
  const std::string embedded_receipt_hash =
    (*receipt)["receipt_sha256"].asString();
  if (
    !calculated_receipt_hash ||
    !isLowercaseSha256(embedded_receipt_hash) ||
    (*detached)["receipt_sha256"].asString() !=
      embedded_receipt_hash)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "receipt structural self-hash contract is invalid");
  }
  const std::string calculated_receipt_hash_hex =
    hexEncode(
      calculated_receipt_hash->data(),
      calculated_receipt_hash->size());
  if (
    CRYPTO_memcmp(
      embedded_receipt_hash.data(),
      calculated_receipt_hash_hex.data(),
      calculated_receipt_hash_hex.size()) != 0)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "receipt structural self-hash does not match");
  }

  const Json::Value & config_cameras = config["cameras"];
  const Json::Value & receipt_cameras = (*payload)["cameras"];
  const Json::Value & artifacts = (*payload)["artifacts_sha256"];
  if (
    !config_cameras.isArray() ||
    config_cameras.size() != kCameraCount ||
    !receipt_cameras.isArray() ||
    receipt_cameras.size() != kCameraCount ||
    !artifacts.isObject())
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "signed bundle must contain exactly four cameras and artifacts");
  }

  std::map<int, const Json::Value *> receipt_by_id;
  for (const Json::Value & receipt_camera : receipt_cameras)
  {
    if (
      !receipt_camera.isObject() ||
      !receipt_camera["camera_id"].isInt() ||
      !receipt_by_id.emplace(
        receipt_camera["camera_id"].asInt(),
        &receipt_camera).second)
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
        "receipt camera ids are invalid or duplicated");
    }
  }

  const std::set<std::string> legacy_matched_camera_fields{
    "black_level",
    "camera_id",
    "covariance_scale",
    "image_height",
    "image_width",
    "inverse_response_lut_file",
    "radiance_offset",
    "radiance_scale",
    "vignetting_transmission_file",
    "white_level",
  };
  const std::set<std::string> base_config_camera_fields{
    "black_level",
    "camera_id",
    "covariance_scale",
    "image_height",
    "image_width",
    "inverse_response_lut_file",
    "name",
    "provenance",
    "radiance_offset",
    "radiance_scale",
    "source_bundle_sha256",
    "source_id",
    "status",
    "vignetting_transmission_file",
    "white_level",
  };
  std::set<std::string> schema_v2_camera_fields =
    base_config_camera_fields;
  schema_v2_camera_fields.insert("raw_intensity_domain");
  schema_v2_camera_fields.insert("raw_intensity_noise");
  const std::set<std::string> & expected_camera_fields =
    payload_schema_version == 2 ?
    schema_v2_camera_fields : base_config_camera_fields;
  std::set<std::string> referenced_artifacts;
  std::array<std::optional<PhotometricCameraCalibration>, kCameraCount>
    parsed_cameras;
  std::array<
    std::optional<InternalLoad::ParsedNoiseEvidence>,
    kCameraCount> parsed_noise_evidence;
  bool noise_evidence_complete =
    payload_schema_version == 2;
  std::map<std::string, FileSnapshot> artifact_snapshots;
  std::size_t total_artifact_bytes = 0U;
  for (const Json::Value & camera : config_cameras)
  {
    if (
      !exactMembers(
        camera, expected_camera_fields) ||
      !camera["camera_id"].isInt())
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
        "config camera shape is not exact");
    }
    const int camera_id = camera["camera_id"].asInt();
    if (
      camera_id < 0 ||
      camera_id >= static_cast<int>(kCameraCount) ||
      parsed_cameras[static_cast<std::size_t>(camera_id)].has_value() ||
      camera["status"].asString() != "calibrated" ||
      camera["provenance"].asString() != "laboratory_measured")
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
        "config must contain camera ids 0..3 exactly once");
    }
    const auto receipt_camera = receipt_by_id.find(camera_id);
    if (
      receipt_camera == receipt_by_id.end() ||
      (
        payload_schema_version == 2 ?
        (
          !exactMembers(
            *receipt_camera->second,
            schema_v2_camera_fields) ||
          camera != *receipt_camera->second
        ) :
        !valuesMatch(
          camera,
          *receipt_camera->second,
          legacy_matched_camera_fields)
      ))
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
        "config and receipt camera values differ");
    }
    if (
      !camera["image_width"].isInt() ||
      !camera["image_height"].isInt() ||
      camera["image_width"].asInt() <= 0 ||
      camera["image_height"].asInt() <= 0 ||
      !finiteJsonNumber(camera["black_level"]) ||
      !finiteJsonNumber(camera["white_level"]) ||
      camera["white_level"].asDouble() <=
        camera["black_level"].asDouble() ||
      !finitePositiveJsonNumber(camera["radiance_scale"]) ||
      !finiteJsonNumber(camera["radiance_offset"]) ||
      !finitePositiveJsonNumber(camera["covariance_scale"]) ||
      camera["covariance_scale"].asDouble() < 1.0 ||
      !camera["source_id"].isString() ||
      camera["source_id"].asString().empty() ||
      camera["source_bundle_sha256"] !=
        config["source_bundle_sha256"])
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kCalibrationInvalid,
        "camera scalar calibration is invalid");
    }
    if (payload_schema_version == 2)
    {
      const Json::Value & domain =
        camera["raw_intensity_domain"];
      const Json::Value & noise =
        camera["raw_intensity_noise"];
      const bool domain_shape_valid =
        exactMembers(
          domain,
          {
            "bit_depth",
            "canonicalization",
            "image_height",
            "image_width",
            "unit",
          });
      const bool noise_shape_valid =
        exactMembers(
          noise,
          {
            "estimator_revision",
            "model",
            "provenance",
            "source_sample_count",
            "standard_deviation_code_value",
          });
      RawIntensityCanonicalization canonicalization =
        RawIntensityCanonicalization::kInvalid;
      if (
        domain_shape_valid &&
        domain["canonicalization"].isString())
      {
        const std::string value =
          domain["canonicalization"].asString();
        if (value == "mono8_identity_v1")
        {
          canonicalization =
            RawIntensityCanonicalization::kMono8IdentityV1;
        }
        else if (value == "rgb8_bt601_luma_uint8_v1")
        {
          canonicalization =
            RawIntensityCanonicalization::
            kRgb8Bt601LumaUint8V1;
        }
      }
      const bool source_sample_count_valid =
        noise_shape_valid &&
        noise["source_sample_count"].isUInt64() &&
        noise["source_sample_count"].asUInt64() > 0U;
      RawIntensityDomain parsed_domain;
      if (
        domain_shape_valid &&
        domain["bit_depth"].isInt() &&
        domain["bit_depth"].asInt() >= 0 &&
        domain["bit_depth"].asInt() <=
          std::numeric_limits<std::uint8_t>::max())
      {
        parsed_domain.bit_depth =
          static_cast<std::uint8_t>(
          domain["bit_depth"].asInt());
      }
      parsed_domain.unit =
        domain_shape_valid &&
        domain["unit"].isString() &&
        domain["unit"].asString() ==
          "canonical_gray_code_value" ?
        RawIntensityUnit::kCanonicalGrayCodeValue :
        RawIntensityUnit::kInvalid;
      parsed_domain.canonicalization = canonicalization;
      parsed_domain.image_width =
        domain_shape_valid &&
          domain["image_width"].isInt() ?
        domain["image_width"].asInt() : 0;
      parsed_domain.image_height =
        domain_shape_valid &&
          domain["image_height"].isInt() ?
        domain["image_height"].asInt() : 0;
      const bool evidence_valid =
        domain_shape_valid &&
        noise_shape_valid &&
        domain["bit_depth"].isInt() &&
        domain["bit_depth"].asInt() == 8 &&
        parsed_domain.valid() &&
        parsed_domain.image_width ==
          camera["image_width"].asInt() &&
        parsed_domain.image_height ==
          camera["image_height"].asInt() &&
        finitePositiveJsonNumber(
          noise["standard_deviation_code_value"]) &&
        noise["provenance"].isString() &&
        noise["provenance"].asString() ==
          "laboratory_measured" &&
        noise["model"].isString() &&
        noise["model"].asString() == "constant_sigma_v1" &&
        noise["estimator_revision"].isString() &&
        noise["estimator_revision"].asString() ==
          "temporal_mad_p99_v1" &&
        source_sample_count_valid;
      if (evidence_valid)
      {
        InternalLoad::ParsedNoiseEvidence parsed_noise;
        parsed_noise.camera_id =
          static_cast<std::uint8_t>(camera_id);
        parsed_noise.domain = parsed_domain;
        parsed_noise.standard_deviation_code_value =
          noise["standard_deviation_code_value"].asDouble();
        parsed_noise.source_id = camera["source_id"].asString();
        parsed_noise.provenance =
          noise["provenance"].asString();
        parsed_noise.model = noise["model"].asString();
        parsed_noise.estimator_revision =
          noise["estimator_revision"].asString();
        parsed_noise.source_sample_count =
          noise["source_sample_count"].asUInt64();
        parsed_noise_evidence[
          static_cast<std::size_t>(camera_id)] =
          std::move(parsed_noise);
      }
      else
      {
        noise_evidence_complete = false;
      }
    }

    const std::string lut_filename =
      camera["inverse_response_lut_file"].asString();
    const std::string vignetting_filename =
      camera["vignetting_transmission_file"].asString();
    if (
      !isBasename(lut_filename) ||
      !isBasename(vignetting_filename) ||
      lut_filename == vignetting_filename ||
      lut_filename == kRuntimePayloadFilename ||
      lut_filename == kRuntimeSignatureFilename ||
      lut_filename == kReceiptFilename ||
      vignetting_filename == kRuntimePayloadFilename ||
      vignetting_filename == kRuntimeSignatureFilename ||
      vignetting_filename == kReceiptFilename ||
      !referenced_artifacts.insert(lut_filename).second ||
      !referenced_artifacts.insert(vignetting_filename).second)
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
        "numeric artifact names are unsafe or aliased");
    }
    const auto read_artifact =
      [&directory, &artifacts, &artifact_snapshots,
       &total_artifact_bytes, &io_error](
        const std::string & filename)
        -> bool
      {
        if (
          !artifacts.isMember(filename) ||
          !artifacts[filename].isString())
        {
          io_error = "artifact hash is missing: " + filename;
          return false;
        }
        std::optional<FileSnapshot> snapshot =
          readRegularFileOnce(
            directory->descriptor,
            filename,
            kMaximumArtifactBytes,
            &io_error);
        if (!snapshot)
        {
          return false;
        }
        if (
          snapshot->bytes.size() >
            kMaximumTotalArtifactBytes - total_artifact_bytes)
        {
          io_error = "numeric artifacts exceed cumulative size cap";
          return false;
        }
        total_artifact_bytes += snapshot->bytes.size();
        const std::optional<Sha256Digest> digest =
          sha256(snapshot->bytes);
        if (
          !digest ||
          hexEncode(digest->data(), digest->size()) !=
            artifacts[filename].asString())
        {
          io_error = "numeric artifact hash mismatch: " + filename;
          return false;
        }
        artifact_snapshots.emplace(
          filename, std::move(*snapshot));
        return true;
      };
    if (!read_artifact(lut_filename) ||
        !read_artifact(vignetting_filename))
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kArtifactInvalid,
        io_error);
    }
    const std::optional<std::vector<double>> lut =
      parseNumericArtifact(
        artifact_snapshots.at(lut_filename).bytes);
    const std::optional<std::vector<double>> vignetting =
      parseNumericArtifact(
        artifact_snapshots.at(vignetting_filename).bytes);
    const int width = camera["image_width"].asInt();
    const int height = camera["image_height"].asInt();
    if (
      !lut ||
      lut->size() != 256U ||
      !vignetting ||
      static_cast<std::uintmax_t>(width) *
        static_cast<std::uintmax_t>(height) !=
        vignetting->size())
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kCalibrationInvalid,
        "numeric artifact dimensions are invalid");
    }
    for (std::size_t index = 0U; index < lut->size(); ++index)
    {
      if (
        (*lut)[index] < 0.0 ||
        (
          index > 0U &&
          (*lut)[index] + kMonotonicTolerance <
            (*lut)[index - 1U]))
      {
        return failure(
          TrustedPhotometricCalibrationLoadStatus::kCalibrationInvalid,
          "inverse response LUT is invalid");
      }
    }
    if (
      lut->back() - lut->front() <= kMonotonicTolerance)
    {
      return failure(
        TrustedPhotometricCalibrationLoadStatus::kCalibrationInvalid,
        "inverse response LUT has no usable range");
    }
    for (const double value : *vignetting)
    {
      if (!(value > 0.0) || !std::isfinite(value))
      {
        return failure(
          TrustedPhotometricCalibrationLoadStatus::kCalibrationInvalid,
          "vignetting transmission is invalid");
      }
    }

    PhotometricCameraCalibration parsed;
    parsed.camera_id = static_cast<std::size_t>(camera_id);
    parsed.calibration_revision = revisionFromDigest(*payload_digest);
    parsed.image_width = width;
    parsed.image_height = height;
    parsed.black_level = camera["black_level"].asDouble();
    parsed.white_level = camera["white_level"].asDouble();
    parsed.inverse_response_lut = *lut;
    parsed.vignetting_transmission = *vignetting;
    parsed.radiance_scale = camera["radiance_scale"].asDouble();
    parsed.radiance_offset = camera["radiance_offset"].asDouble();
    parsed.covariance_scale = camera["covariance_scale"].asDouble();
    parsed.source_id = camera["source_id"].asString();
    parsed_cameras[parsed.camera_id] = std::move(parsed);
  }
  if (
    referenced_artifacts != memberSet(artifacts) ||
    receipt_by_id.size() != kCameraCount)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "artifact or camera allowlist is not exact");
  }

  const std::uint64_t revision = revisionFromDigest(*payload_digest);
  const std::string compiled_key_hex =
    LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX;
  const std::string compiled_domain = LIVO_PHOTOMETRIC_TRUST_DOMAIN;
  if (compiled_key_hex.empty() && compiled_domain.empty())
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kTrustRootUnavailable,
      "application build has no photometric trust root",
      true,
      revision);
  }
  const std::optional<std::vector<unsigned char>> compiled_key =
    hexDecode(compiled_key_hex);
  if (
    !compiled_key ||
    compiled_key->size() != kEd25519PublicKeyBytes ||
    compiled_domain.empty())
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kTrustRootMismatch,
      "compiled photometric trust root is malformed",
      true,
      revision);
  }
  const std::optional<Sha256Digest> key_digest = sha256(
    std::string_view(
      reinterpret_cast<const char *>(compiled_key->data()),
      compiled_key->size()));
  if (
    !key_digest ||
    trust_root["public_key_sha256"].asString() !=
      hexEncode(key_digest->data(), key_digest->size()) ||
    trust_root["trust_domain"].asString() != compiled_domain)
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kTrustRootMismatch,
      "signed trust root does not match application trust root",
      true,
      revision);
  }
  if (
    !verifyEd25519(
      *compiled_key, *signature, payload_snapshot->bytes))
  {
    return failure(
      TrustedPhotometricCalibrationLoadStatus::kSignatureInvalid,
      "Ed25519 signature over exact runtime payload bytes is invalid",
      true,
      revision);
  }

  InternalLoad loaded;
  loaded.status = TrustedPhotometricCalibrationLoadStatus::kLoaded;
  loaded.structurally_valid = true;
  loaded.revision = revision;
  loaded.diagnostic = "trusted four-camera photometric calibration loaded";
  loaded.cameras.reserve(kCameraCount);
  for (std::optional<PhotometricCameraCalibration> & camera :
       parsed_cameras)
  {
    loaded.cameras.push_back(std::move(*camera));
  }
  if (noise_evidence_complete)
  {
    loaded.noise_evidence.reserve(kCameraCount);
    for (std::optional<InternalLoad::ParsedNoiseEvidence> & evidence :
         parsed_noise_evidence)
    {
      if (!evidence.has_value())
      {
        loaded.noise_evidence.clear();
        break;
      }
      loaded.noise_evidence.push_back(std::move(*evidence));
    }
  }
  return loaded;
}

}  // namespace

TrustedPhotometricCalibrationLoadResult
TrustedPhotometricCalibrationLoader::load(
  const std::filesystem::path & bundle_directory)
{
  InternalLoad internal;
  try
  {
    internal = loadInternal(bundle_directory);
  }
  catch (const Json::Exception & error)
  {
    internal = failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidJson,
      "JSON value conversion failed: " + std::string(error.what()));
  }
  catch (const std::exception & error)
  {
    internal = failure(
      TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract,
      "bundle validation failed: " + std::string(error.what()));
  }
  if (
    internal.status ==
      TrustedPhotometricCalibrationLoadStatus::kLoaded &&
    internal.structurally_valid &&
    internal.revision != 0U &&
    internal.cameras.size() == kCameraCount)
  {
    for (PhotometricCameraCalibration & camera : internal.cameras)
    {
      camera.sealMeasuredReceipt();
    }
  }
  std::optional<TrustedPhotometricCalibrationBundle>
    trusted_noise_bundle;
  if (
    internal.status ==
      TrustedPhotometricCalibrationLoadStatus::kLoaded &&
    internal.structurally_valid &&
    internal.revision != 0U &&
    internal.noise_evidence.size() == kCameraCount)
  {
    std::vector<TrustedRawIntensityNoiseCapability> capabilities;
    capabilities.reserve(kCameraCount);
    for (InternalLoad::ParsedNoiseEvidence & evidence :
         internal.noise_evidence)
    {
      capabilities.push_back(
        TrustedRawIntensityNoiseCapability(
          evidence.camera_id,
          internal.revision,
          evidence.domain,
          evidence.standard_deviation_code_value,
          std::move(evidence.source_id),
          std::move(evidence.provenance),
          std::move(evidence.model),
          std::move(evidence.estimator_revision),
          evidence.source_sample_count));
    }
    TrustedPhotometricCalibrationBundle issued(
      internal.revision, std::move(capabilities));
    if (issued.valid())
    {
      trusted_noise_bundle.emplace(std::move(issued));
    }
  }
  return makePublicResult(
    std::move(internal), std::move(trusted_noise_bundle));
}

const char * trustedPhotometricCalibrationLoadStatusName(
  const TrustedPhotometricCalibrationLoadStatus status)
{
  switch (status)
  {
    case TrustedPhotometricCalibrationLoadStatus::kLoaded:
      return "loaded";
    case TrustedPhotometricCalibrationLoadStatus::kTrustRootUnavailable:
      return "trust_root_unavailable";
    case TrustedPhotometricCalibrationLoadStatus::kInvalidBundlePath:
      return "invalid_bundle_path";
    case TrustedPhotometricCalibrationLoadStatus::kIoFailure:
      return "io_failure";
    case TrustedPhotometricCalibrationLoadStatus::kResourceLimitExceeded:
      return "resource_limit_exceeded";
    case TrustedPhotometricCalibrationLoadStatus::kInvalidJson:
      return "invalid_json";
    case TrustedPhotometricCalibrationLoadStatus::kInvalidBundleContract:
      return "invalid_bundle_contract";
    case TrustedPhotometricCalibrationLoadStatus::kPayloadHashMismatch:
      return "payload_hash_mismatch";
    case TrustedPhotometricCalibrationLoadStatus::kTrustRootMismatch:
      return "trust_root_mismatch";
    case TrustedPhotometricCalibrationLoadStatus::kSignatureInvalid:
      return "signature_invalid";
    case TrustedPhotometricCalibrationLoadStatus::kArtifactInvalid:
      return "artifact_invalid";
    case TrustedPhotometricCalibrationLoadStatus::kCalibrationInvalid:
      return "calibration_invalid";
  }
  return "unknown";
}

}  // namespace livo
