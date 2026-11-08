#ifndef LIVO_LEGACY_LANDMARK_OBSERVATION_SHADOW_ADAPTER_H_
#define LIVO_LEGACY_LANDMARK_OBSERVATION_SHADOW_ADAPTER_H_

#include "camera_time_transfer.h"
#include "multi_view_landmark.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

namespace livo
{

struct LegacyLandmarkObservationShadowConfig
{
  bool enabled{false};
  std::size_t maximum_observations_per_batch{512U};
  std::size_t maximum_patch_levels{4U};
  std::size_t maximum_patch_values_per_observation{1024U};
};

// Value-owned boundary object. It intentionally contains no cv::Mat, Feature,
// or other legacy-map pointer.
struct LegacyLandmarkObservationShadowSample
{
  GeometrySupportKey support_key;
  ActivePlaneIdentityLease active_plane_lease;
  std::uint8_t camera_id{0U};
  double exposure_timestamp_sec{0.0};
  Eigen::Matrix3d rotation_world_from_imu{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_world_m{Eigen::Vector3d::Zero()};
  std::uint64_t rig_calibration_revision{0U};
  std::uint64_t camera_model_revision{0U};
  Eigen::Vector2d pixel{Eigen::Vector2d::Zero()};
  int level{0};
  std::vector<std::vector<float>> owned_patch_pyramid;
  ObservationPhotometricMode photometric_mode{
    ObservationPhotometricMode::kInvalid};
  PhotometricStatus photometric_status{
    PhotometricStatus::kInvalidInput};
  bool photometric_cross_camera_comparable{false};
  bool photometric_experimental{false};
  std::uint64_t photometric_calibration_revision{0U};
  CameraTimeTransferStatus time_status{
    CameraTimeTransferStatus::kInvalidInput};
  std::uint64_t time_transfer_revision{0U};
  std::shared_ptr<const ReferenceOnlyMigrationFrameProvenance>
    frame_provenance;
  Eigen::Matrix<double, 6, 6> pose_covariance{
    Eigen::Matrix<double, 6, 6>::Zero()};
  bool visual_update_accepted{false};
  bool is_bootstrap{false};
  bool isolated{false};
};

struct LegacyLandmarkObservationShadowRequest
{
  BatchId batch;
  GeometrySnapshotId geometry;
  ActivePlaneIdentitySnapshot active_planes;
  std::optional<LidarAnchoredCommitToken> bootstrap_token;
  std::optional<AcceptedStateCommitToken> accepted_state_token;

  std::vector<LegacyLandmarkObservationShadowSample> samples;
};

enum class LegacyLandmarkObservationShadowStatus
{
  kDisabled,
  kInvalidConfiguration,
  kInvalidRequest,
  kBatchLimitExceeded,
  kMixedCommitKinds,
  kNoEligibleObservation,
  kPrepareRejected,
  kProofRejected,
  kStaged
};

class StagedLegacyLandmarkObservationShadowBatch
{
public:
  StagedLegacyLandmarkObservationShadowBatch() = default;
  StagedLegacyLandmarkObservationShadowBatch(
    const StagedLegacyLandmarkObservationShadowBatch &) = delete;
  StagedLegacyLandmarkObservationShadowBatch & operator=(
    const StagedLegacyLandmarkObservationShadowBatch &) = delete;
  StagedLegacyLandmarkObservationShadowBatch(
    StagedLegacyLandmarkObservationShadowBatch &&) = default;
  StagedLegacyLandmarkObservationShadowBatch & operator=(
    StagedLegacyLandmarkObservationShadowBatch &&) = default;

  bool valid() const;
  CommitResult commit(MultiViewLandmarkStore & store);

private:
  friend class LegacyLandmarkObservationShadowAdapter;

  StagedLandmarkDelta delta_;
  std::optional<MapCommitProof> proof_;
  bool valid_{false};
};

struct LegacyLandmarkObservationShadowStageResult
{
  LegacyLandmarkObservationShadowStatus status{
    LegacyLandmarkObservationShadowStatus::kDisabled};
  std::size_t input_sample_count{0U};
  std::size_t eligible_sample_count{0U};
  std::size_t staged_observation_count{0U};
  StagedLegacyLandmarkObservationShadowBatch staged;
};

class LegacyLandmarkObservationShadowAdapter
{
public:
  explicit LegacyLandmarkObservationShadowAdapter(
    LegacyLandmarkObservationShadowConfig config =
      LegacyLandmarkObservationShadowConfig{});

  bool configValid() const;

  LegacyLandmarkObservationShadowStageResult stage(
    const MultiViewLandmarkStore & store,
    const LegacyLandmarkObservationShadowRequest & request) const;

private:
  LegacyLandmarkObservationShadowConfig config_;
};

const char * legacyLandmarkObservationShadowStatusName(
  LegacyLandmarkObservationShadowStatus status);

}  // namespace livo

#endif  // LIVO_LEGACY_LANDMARK_OBSERVATION_SHADOW_ADAPTER_H_
