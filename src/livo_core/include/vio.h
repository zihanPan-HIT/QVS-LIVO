/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VIO_H_
#define VIO_H_

#include "adaptive_visual_policy.h"
#include "adaptive_visual_selection_authority.h"
#include "camera_model_factory.h"
#include "camera_observation.h"
#include "camera_batch_time_transfer.h"
#include "four_camera_pinhole_rig.h"
#include "joint_visual_update.h"
#include "multi_camera_landmark_linearization_shadow.h"
#include "reference_only_migration_shadow_batch_runner.h"
#include "reference_only_migration_runtime_provenance.h"
#include "shared_landmark_shadow_session.h"
#include "voxel_map.h"
#include "feature.h"
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <opencv2/imgproc/imgproc_c.h>
#include <pcl/filters/voxel_grid.h>
#include <set>
#include <string>
#include <vector>
#include <vikit/math_utils.h>
#include <vikit/robust_cost.h>
#include <vikit/vision.h>
#include <vikit/pinhole_camera.h>

struct SubSparseMap
{
  vector<float> propa_errors;
  vector<float> errors;
  vector<vector<float>> warp_patch;
  vector<int> search_levels;
  vector<VisualPoint *> voxel_points;
  vector<double> inv_expo_list;
  vector<pointWithVar> add_from_voxel_map;

  SubSparseMap()
  {
    propa_errors.reserve(SIZE_LARGE);
    errors.reserve(SIZE_LARGE);
    warp_patch.reserve(SIZE_LARGE);
    search_levels.reserve(SIZE_LARGE);
    voxel_points.reserve(SIZE_LARGE);
    inv_expo_list.reserve(SIZE_LARGE);
    add_from_voxel_map.reserve(SIZE_SMALL);
  };

  void reset()
  {
    propa_errors.clear();
    errors.clear();
    warp_patch.clear();
    search_levels.clear();
    voxel_points.clear();
    inv_expo_list.clear();
    add_from_voxel_map.clear();
  }
};

namespace livo
{

struct CrossCameraMigrationShadowCameraConfig
{
  std::uint8_t camera_id{
    std::numeric_limits<std::uint8_t>::max()};
  std::string camera_name;
  CameraModelParameters projection;
  bool projection_calibration_present{false};
  bool extrinsic_calibration_present{false};
};

struct CrossCameraMigrationShadowInitializationConfig
{
  bool enabled{false};
  bool four_camera_mode{false};
  std::array<
    CrossCameraMigrationShadowCameraConfig,
    kFourCameraPinholeRigCameraCount> cameras;
};

struct CrossCameraMigrationShadowCameraAudit
{
  std::uint8_t camera_id{0U};
  std::string camera_name;
  std::uint64_t model_revision{0U};
  Eigen::Matrix3d rotation_imu_from_camera{
    Eigen::Matrix3d::Identity()};
  Eigen::Vector3d position_imu_from_camera{
    Eigen::Vector3d::Zero()};
};

struct CrossCameraMigrationShadowInitAudit
{
  bool enabled{false};
  bool ready{false};
  std::string reason{"not_initialized"};
  std::uint64_t rig_revision{0U};
  std::uint64_t owner_generation{0U};
  std::optional<FourCameraPinholeRigFingerprint>
    rig_fingerprint;
  std::array<
    std::optional<CrossCameraMigrationShadowCameraAudit>,
    kFourCameraPinholeRigCameraCount> cameras;
  bool production_ready{false};
  bool solver_applied{false};
};

struct ReferenceOnlyMigrationShadowRuntimeConfig
{
  bool enabled{false};
  bool four_camera_mode{false};
};

enum class ReferenceOnlyMigrationShadowRuntimeInitReason
{
  kNotInitialized,
  kDisabled,
  kRequiresFourCameraMode,
  kMissingCrossCameraShadowOwner,
  kCrossCameraShadowOwnerNotReady,
  kMissingSharedGeometryShadow,
  kMissingTimeTransferConfiguration,
  kMissingBatchAuthority,
  kMissingRuntimePublisherAuthority,
  kRuntimeAuthorityOwnerMismatch,
  kRuntimeObservationAuthorityBindingRejected,
  kMetricsOnlyReady
};

struct ReferenceOnlyMigrationShadowRuntimeInitAudit
{
  bool enabled{false};
  bool effective{false};
  ReferenceOnlyMigrationShadowRuntimeInitReason reason{
    ReferenceOnlyMigrationShadowRuntimeInitReason::
    kNotInitialized};
  std::uint64_t bound_cross_camera_owner_generation{0U};
  std::uint64_t bound_rig_revision{0U};
  std::optional<FourCameraPinholeRigFingerprint>
    bound_rig_fingerprint;
  std::uint64_t bound_geometry_configuration_generation{0U};
  std::uint64_t bound_time_transfer_revision{0U};
  std::uint64_t bound_authority_generation{0U};
  bool visibility_verified{false};
  bool experimental{true};
  bool consumable{false};
  bool production_ready{false};
  bool solver_applied{false};
  bool acceptance_receipt_issued{false};
  bool independence_proof_issued{false};
};

enum class ReferenceOnlyMigrationShadowRuntimeBatchReason
{
  kNotRun,
  kMissingTimeTransferBundle,
  kMissingRuntimeProvenance,
  kRuntimeProvenanceMismatch,
  kInvalidFullStateTimeTransferBundle,
  kTimeTransferBatchBindingMismatch,
  kInvalidActiveStateSnapshot,
  kInvalidSourceCovariance,
  kDependencyStale,
  kImageGeometryMismatch,
  kSequenceExhausted,
  kIncompleteFourCameraContexts,
  kNoCandidate,
  kNoEligibleDirectedPair,
  kEvaluated
};

const char *
referenceOnlyMigrationShadowRuntimeInitReasonName(
  ReferenceOnlyMigrationShadowRuntimeInitReason reason) noexcept;

const char *
referenceOnlyMigrationShadowRuntimeBatchReasonName(
  ReferenceOnlyMigrationShadowRuntimeBatchReason reason) noexcept;

struct ReferenceOnlyMigrationShadowRuntimeBatchAudit
{
  ReferenceOnlyMigrationShadowRuntimeBatchReason reason{
    ReferenceOnlyMigrationShadowRuntimeBatchReason::kNotRun};
  std::uint64_t batch_sequence{0U};
  std::uint64_t runtime_publication_sequence{0U};
  std::uint64_t active_state_snapshot_revision{0U};
  std::uint64_t lidar_feature_generation{0U};
  double batch_timestamp_sec{0.0};
  std::uint64_t runner_call_count{0U};
  std::size_t captured_seed_count{0U};
  std::size_t descriptor_count{0U};
  std::size_t canonical_descriptor_count{0U};
  std::size_t materialized_candidate_count{0U};
  std::size_t source_covariance_evaluation_count{0U};
  std::uint64_t seal_image_hash_count{0U};
  std::uint64_t validation_image_hash_count{0U};
  std::size_t candidate_count{0U};
  std::size_t unique_candidate_count{0U};
  std::size_t duplicate_candidate_count{0U};
  std::size_t ambiguous_duplicate_count{0U};
  std::size_t retained_candidate_count{0U};
  std::size_t pair_budget_truncated_count{0U};
  std::size_t total_budget_truncated_count{0U};
  std::size_t attempted_count{0U};
  std::size_t ready_count{0U};
  std::size_t rejected_count{0U};
  std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowDirectedPairCount>
    pair_attempt_counts{};
  std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowDirectedPairCount>
    pair_ready_counts{};
  std::array<
    std::size_t,
    kReferenceOnlyMigrationShadowStatusCount>
    migration_status_counts{};
  std::vector<
    ReferenceOnlyMigrationShadowBatchSelection>
    selected_sequence;
  std::optional<double> ready_warp_condition_p50;
  std::optional<double> ready_warp_condition_p95;
  double source_pose_covariance_trace_sum{0.0};
  double runtime_sec{0.0};
  bool visibility_verified{false};
  bool experimental{true};
  bool consumable{false};
  bool production_ready{false};
  bool solver_applied{false};
  bool acceptance_receipt_issued{false};
  bool independence_proof_issued{false};
};

enum class VIORuntimeObservationAuthorityStatus
{
  kNotEvaluated,
  kMissingRuntimeBundle,
  kRuntimePublisherAuthorityMismatch,
  kCameraBatchAuthorityMismatch,
  kActiveStateSnapshotMismatch,
  kInvalidTimeTransferBundle,
  kReplayedRuntimeBundle,
  kInvalidFrameProvenance,
  kReady
};

const char * vioRuntimeObservationAuthorityStatusName(
  VIORuntimeObservationAuthorityStatus status) noexcept;

enum class VIOMultiCameraLandmarkShadowInvocation
{
  kDisabled,
  kNotEvaluated,
  kModuleUnavailable,
  kMissingRuntimeBundle,
  kRuntimeAuthorityNotReady,
  kMissingPlanCapability,
  kProcessed
};

const char * vioMultiCameraLandmarkShadowInvocationName(
  VIOMultiCameraLandmarkShadowInvocation invocation) noexcept;

struct VIOMultiCameraLandmarkLinearizationShadowBatchAudit
{
  VIOMultiCameraLandmarkShadowInvocation invocation{
    VIOMultiCameraLandmarkShadowInvocation::kDisabled};
  VIORuntimeObservationAuthorityStatus runtime_authority{
    VIORuntimeObservationAuthorityStatus::kNotEvaluated};
  std::optional<MultiCameraLandmarkShadowBatchStatus>
    batch_status;
  bool configured_enabled{false};
  bool module_active{false};
  bool plan_capability_present{false};
  bool plan_capability_validated{false};
  bool process_called{false};
  std::uint64_t configuration_generation{0U};
  std::uint64_t accepted_target_generation{0U};
  std::uint64_t target_camera_batch_sequence{0U};
  std::uint64_t target_runtime_publication_sequence{0U};
  std::uint64_t store_revision{0U};
  std::size_t inspected_source_observations{0U};
  std::size_t store_authority_violations{0U};
  std::size_t skipped_reference_only_sources{0U};
  std::size_t skipped_tombstoned_sources{0U};
  std::size_t skipped_inactive_plane_sources{0U};
  std::size_t skipped_sampling_geometry{0U};
  std::size_t skipped_source_order{0U};
  std::size_t skipped_source_runtime_authority{0U};
  std::size_t skipped_source_age{0U};
  std::size_t candidate_evaluations{0U};
  std::size_t decision_count{0U};
  std::size_t selection_trace_count{0U};
  std::size_t strict_photometric_rejected{0U};
  std::size_t same_camera_trace_count{0U};
  std::size_t cross_camera_strict_rejected_count{0U};
  std::size_t relative_affine_experimental_count{0U};
  bool has_linearization{false};
  bool consumable{false};
  bool production_ready{false};
  bool solver_applied{false};
};

inline constexpr std::size_t kVioReadOnlyStateScalarCount =
  25U + DIM_STATE * DIM_STATE;

struct VIOJointSolverReadOnlyAudit
{
  bool accepted{false};
  std::string reason{"not_run"};
  std::size_t contributing_cameras{0U};
  std::size_t tracked_points{0U};
};

enum class AdaptiveVisualRuntimeOutcome
{
  kNotRun,
  kNotApplicable,
  kLioOnly,
  kVisualRejected,
  kVisualAccepted,
};

enum class AdaptiveVisualLioOnlyReason
{
  kNone,
  kMissingRuntimeInput,
  kInvalidRuntimeInput,
  kLidarGenerationMismatch,
  kLidarInformationMismatch,
  kLidarInformationNotReady,
  kStaleRuntimeGeneration,
  kGenerationAlreadyProcessed,
  kBatchPreflightRejected,
  kCameraPreflightRejected,
  kCandidatePlanRejected,
  kSelectionIssueRejected,
  kSelectionConsumeRejected,
  kInvalidSelectionReceipt,
  kHealthCommitRejected,
  kVisualMapBootstrap,
  kVisualMapReseed,
};

enum class AdaptiveVisualMapAction
{
  kNone,
  kColdStartSeed,
  kAcceptedVisualUpdate,
  kLidarAnchoredReseed,
};

struct AdaptiveVisualGenerationUpdateReceipt
{
  bool visual_update_accepted{false};
  bool solver_called{false};
  std::string solver_reason{"not_called"};
  std::size_t solver_tracked_points{0U};
  std::size_t solver_contributing_cameras{0U};
  AdaptiveVisualMapAction map_action{
    AdaptiveVisualMapAction::kNone};
  bool seed_new_points{false};
  bool update_existing_points{false};
};

struct AdaptiveVisualRuntimeAudit
{
  AdaptiveVisualRuntimeOutcome outcome{
    AdaptiveVisualRuntimeOutcome::kNotRun};
  AdaptiveVisualLioOnlyReason lio_only_reason{
    AdaptiveVisualLioOnlyReason::kNone};
  std::optional<AdaptiveVisualSelectionStatus> lidar_status;
  std::optional<AdaptiveVisualSelectionStatus>
    batch_preflight_status;
  std::optional<AdaptiveVisualRuntimeAttemptStatus>
    runtime_attempt_status;
  std::optional<AdaptiveVisualSelectionStatus>
    selection_issue_status;
  std::optional<AdaptiveVisualSelectionStatus>
    selection_consume_status;
  std::optional<AdaptiveCandidateHealthCommitStatus>
    health_commit_status;
  std::optional<AdaptiveVisualGenerationUpdateReceipt>
    generation_update_receipt;
  std::uint64_t lidar_generation{0U};
  std::optional<AdaptiveVisualPolicy::Matrix6>
    validated_lidar_pose_information;
  std::uint64_t camera_batch_sequence{0U};
  std::array<bool, kAdaptiveVisualCameraCount>
    candidate_mask{{false, false, false, false}};
  std::array<bool, kAdaptiveVisualCameraCount>
    eligible_mask{{false, false, false, false}};
  std::array<AdaptiveCameraRejectReason, kAdaptiveVisualCameraCount>
    reject_reasons{{
      AdaptiveCameraRejectReason::kNone,
      AdaptiveCameraRejectReason::kNone,
      AdaptiveCameraRejectReason::kNone,
      AdaptiveCameraRejectReason::kNone}};
  std::array<bool, kAdaptiveVisualCameraCount>
    selected_mask{{false, false, false, false}};
  std::optional<std::uint8_t> selected_camera_id;
  std::optional<std::uint8_t> fusion_camera_id;
  std::optional<double> selected_exposure_timestamp_sec;
  std::optional<double> pose_covariance_volume_reduction;
  AdaptiveSelectionReason selection_reason{
    AdaptiveSelectionReason::kNotApplicable};
  bool capability_issued{false};
  bool receipt_consumed{false};
  bool solver_called{false};
  bool legacy_fallback_used{false};
  std::size_t scouted_camera_count{0U};
  std::size_t solver_call_count{0U};
  bool shared_landmark_shadow_called{false};
  bool reference_only_migration_shadow_called{false};
  bool multi_camera_linearization_shadow_called{false};
  std::size_t runtime_attempt_count{0U};
  std::size_t selection_consumption_count{0U};
  std::size_t prepare_call_count{0U};
  std::size_t finalize_call_count{0U};
  std::uint64_t policy_revision_before{0U};
  std::uint64_t policy_revision_after{0U};
};

struct VIOReadOnlyStructuralAudit
{
  bool active_state_available{false};
  std::array<double, kVioReadOnlyStateScalarCount>
    active_state{};
  bool propagated_state_available{false};
  std::array<double, kVioReadOnlyStateScalarCount>
    propagated_state{};
  SharedLandmarkShadowAudit shared_landmark_shadow;
  std::array<std::size_t, kFourCameraPinholeRigCameraCount>
    legacy_feature_voxel_counts{};
  std::array<std::size_t, kFourCameraPinholeRigCameraCount>
    legacy_feature_point_counts{};
  VIOJointSolverReadOnlyAudit last_joint_solver;
  std::uint8_t active_camera_id{0U};
  std::array<bool, kAdaptiveVisualCameraCount>
    adaptive_camera_initialized{{false, false, false, false}};
  std::array<AdaptiveVisualMapLifecycle,
    kAdaptiveVisualCameraCount> adaptive_visual_map_lifecycle{{
      AdaptiveVisualMapLifecycle::kEmpty,
      AdaptiveVisualMapLifecycle::kEmpty,
      AdaptiveVisualMapLifecycle::kEmpty,
      AdaptiveVisualMapLifecycle::kEmpty}};
  std::array<std::size_t, kAdaptiveVisualCameraCount>
    adaptive_zero_map_reentry_healthy_scouts{{0U, 0U, 0U, 0U}};
  std::array<bool, kAdaptiveVisualCameraCount>
    adaptive_zero_map_reseed_latched{{false, false, false, false}};
  std::array<std::uint64_t, kAdaptiveVisualCameraCount>
    adaptive_zero_map_last_trusted_generation{{0U, 0U, 0U, 0U}};
  std::size_t frame_count{0U};
  std::uint64_t adaptive_policy_revision{0U};
  std::size_t adaptive_prepare_call_count{0U};
  std::size_t adaptive_scout_call_count{0U};
  std::size_t adaptive_finalize_call_count{0U};
  std::size_t shared_landmark_shadow_entry_count{0U};
  std::size_t
    reference_only_migration_shadow_entry_count{0U};
  std::size_t
    multi_camera_linearization_shadow_entry_count{0U};
};

}  // namespace livo

class Warp
{
public:
  Matrix2d A_cur_ref;
  int search_level;
  Warp(int level, Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
  ~Warp() {}
};

class VOXEL_POINTS
{
public:
  std::vector<VisualPoint *> voxel_points;
  int count;
  VOXEL_POINTS(int num) : count(num) {}
  ~VOXEL_POINTS() 
  { 
    for (VisualPoint* vp : voxel_points) 
    {
      if (vp != nullptr) { delete vp; vp = nullptr; }
    }
  }
};

class VIOManager
{
public:
  static constexpr std::size_t kCameraCount = CameraObservationQueue::kCameraCount;

  int adaptive_camera_threads{1};
  int grid_size;
  vk::AbstractCamera *cam = nullptr;
  vk::PinholeCamera *pinhole_cam = nullptr;
  StatesGroup *state = nullptr;
  StatesGroup *state_propagat = nullptr;
  M3D Rli, Rci, Rcl, Rcw, Jdphi_dR, Jdp_dt, Jdp_dR;
  V3D Pli, Pci, Pcl, Pcw;
  vector<int> grid_num;
  vector<int> map_index;
  vector<int> border_flag;
  vector<int> update_flag;
  vector<float> map_dist;
  vector<float> scan_value;
  vector<float> patch_buffer;
  bool normal_en, inverse_composition_en, exposure_estimate_en, raycast_en, has_ref_patch_cache;
  bool ncc_en = false, colmap_output_en = false;
  JointVisualUpdateConfig joint_update_config;
  livo::AdaptiveVisualPolicyConfig adaptive_policy_config;

  int width, height, grid_n_width, grid_n_height, length;
  double image_resize_factor;
  double fx, fy, cx, cy;
  int patch_pyrimid_level, patch_size, patch_size_total, patch_size_half, border, warp_len;
  int max_iterations, total_points;

  double img_point_cov, joint_img_point_cov, outlier_threshold, ncc_thre;
  
  SubSparseMap *visual_submap = nullptr;
  std::vector<std::vector<V3D>> rays_with_sample_points;

  double compute_jacobian_time, update_ekf_time;
  double ave_total = 0;
  // double ave_build_residual_time = 0;
  // double ave_ekf_time = 0;

  int frame_count = 0;
  bool plot_flag{false};
  std::uint8_t active_camera_id_ = 0U;

  Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H;
  MatrixXd K, H_sub_inv;

  ofstream fout_camera, fout_colmap;
  unordered_map<VOXEL_LOCATION, int> sub_feat_map; 
  unordered_map<int, Warp *> warp_map;
  vector<VisualPoint *> retrieve_voxel_points;
  vector<pointWithVar> append_voxel_points;
  FramePtr new_frame_;
  cv::Mat img_cp, img_rgb, img_test;

  enum CellType
  {
    TYPE_MAP = 1,
    TYPE_POINTCLOUD,
    TYPE_UNKNOWN
  };

  VIOManager();
  ~VIOManager();
  void updateStateInverse(cv::Mat img, int level);
  void updateState(cv::Mat img, int level);
  void processFrame(CameraObservation &observation, vector<pointWithVar> &pg,
                    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
                    double img_time);
  void processFrames(
    CameraObservationBatch &batch, vector<pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
    double img_time,
    const livo::AdaptiveVisualPolicy::Matrix6 & lidar_pose_information,
    bool lidar_pose_information_valid,
    std::uint64_t lidar_feature_generation,
    const livo::ActivePlaneIdentitySnapshot *
      active_plane_snapshot = nullptr,
    const livo::CameraBatchTimeTransferBundle *
      time_transfer_bundle = nullptr,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      migration_runtime_bundle = nullptr,
    const livo::AdaptiveVisualRuntimeAuthorityInput *
      adaptive_runtime_input = nullptr);
  void configureSharedLandmarkShadow(
    const livo::LegacyLandmarkShadowConfig & config);
  void configureSharedLandmarkObservationShadow(
    const livo::LegacyLandmarkObservationShadowConfig & config);
  void configureSharedLandmarkTombstoneShadow(
    const livo::SharedLandmarkTombstoneShadowConfig & config);
  void configureSharedLandmarkRefitShadow(
    const livo::SharedLandmarkGeometryRefitShadowConfig & config);
  void configureSharedLandmarkTimeTransfer(
    const livo::CameraTimeTransferConfig & config);
  void configureSharedLandmarkPhotometric(
    std::uint8_t camera_id,
    const livo::ShadowPhotometricCapabilityEvidence & evidence);
  bool configureMultiCameraLandmarkLinearizationShadow(
    const livo::MultiCameraLandmarkLinearizationShadowConfig &
      config);
  void initializeCrossCameraMigrationShadow(
    const livo::CrossCameraMigrationShadowInitializationConfig &
      config);
  const livo::CrossCameraMigrationShadowInitAudit &
  crossCameraMigrationShadowInitAudit() const noexcept;
  void configureReferenceOnlyMigrationShadowRuntime(
    const livo::ReferenceOnlyMigrationShadowRuntimeConfig &
      config);
  void configureReferenceOnlyMigrationRuntimeAuthorities(
    const livo::CameraObservationBatchAuthorityIdentity &
      batch_authority,
    const livo::ReferenceOnlyMigrationRuntimePublisherIdentity &
      publisher_authority);
  const livo::ReferenceOnlyMigrationShadowRuntimeInitAudit &
  referenceOnlyMigrationShadowRuntimeInitAudit() const noexcept;
  const livo::ReferenceOnlyMigrationShadowRuntimeBatchAudit &
  referenceOnlyMigrationShadowRuntimeBatchAudit() const noexcept;
  const livo::
    VIOMultiCameraLandmarkLinearizationShadowBatchAudit &
  multiCameraLandmarkLinearizationShadowBatchAudit() const noexcept;
  livo::VIOReadOnlyStructuralAudit
  readOnlyStructuralAudit() const;
  const livo::AdaptiveVisualRuntimeAudit &
  adaptiveVisualRuntimeAudit() const noexcept;
  livo::SharedLandmarkShadowOperationResult
  applySharedLandmarkTombstonesShadow(
    const std::vector<livo::PlaneTombstoneEvent> & events);
  livo::SharedLandmarkShadowOperationResult
  applySharedLandmarkRefitsShadow(
    const std::vector<livo::PlaneRefitEvent> & events);
  void retrieveFromVisualSparseMap(
    cv::Mat img, vector<pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
    bool allow_cache_writes = true);
  void retrieveCameraVisualMap(
    cv::Mat img, vector<pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map,
    bool allow_cache_writes, Frame *frame, std::uint8_t camera_id,
    SubSparseMap &submap, vector<int> &grid,
    vector<float> &distances, vector<VisualPoint *> &points);
  void emitVisualMapResources() const;
  void generateVisualMapPoints(
    cv::Mat img, vector<pointWithVar> &pg,
    std::vector<VisualPoint *> * generated_points = nullptr);
  void setImuToLidarExtrinsic(const V3D &transl, const M3D &rot);
  void setLidarToCameraExtrinsic(vector<double> &R, vector<double> &P);
  void registerCamera(std::uint8_t camera_id, const std::string &name,
                      vk::AbstractCamera *camera, const M3D &lidar_to_camera_rotation,
                      const V3D &lidar_to_camera_translation);
  bool activateCamera(std::uint8_t camera_id);
  vk::AbstractCamera *cameraModel(std::uint8_t camera_id) const;
  bool cameraFromImuExtrinsic(
    std::uint8_t camera_id, M3D & rotation_camera_from_imu,
    V3D & translation_camera_from_imu) const noexcept;
  const std::string &cameraName(std::uint8_t camera_id) const;
  void initializeVIO();
  void getImagePatch(cv::Mat img, V2D pc, float *patch_tmp, int level);
  void computeProjectionJacobian(V3D p, MD(2, 3) & J);
  void computeJacobianAndUpdateEKF(cv::Mat img);
  void resetGrid();
  void updateVisualMapPoints(cv::Mat img);
  void getWarpMatrixAffine(const vk::AbstractCamera &cam, const Vector2d &px_ref, const Vector3d &f_ref, const double depth_ref, const SE3<double> &T_cur_ref,
                           const int level_ref, 
                           const int pyramid_level, const int halfpatch_size, Matrix2d &A_cur_ref);
  void getWarpMatrixAffineHomography(const vk::AbstractCamera &cam, const V2D &px_ref,
                                     const V3D &xyz_ref, const V3D &normal_ref, const SE3<double> &T_cur_ref, const int level_ref, Matrix2d &A_cur_ref);
  void warpAffine(const Matrix2d &A_cur_ref, const cv::Mat &img_ref, const Vector2d &px_ref, const int level_ref, const int search_level,
                  const int pyramid_level, const int halfpatch_size, float *patch);
  void insertPointIntoVoxelMap(VisualPoint *pt_new);
  void plotTrackedPoints();
  void updateFrameState(StatesGroup state);
  void projectPatchFromRefToCur(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void updateReferencePatch(const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &plane_map);
  void precomputeReferencePatches(int level);
  void dumpDataForColmap();
  double calculateNCC(float *ref_patch, float *cur_patch, int patch_size);
  int getBestSearchLevel(const Matrix2d &A_cur_ref, const int max_level);
  V3F getInterpolatedPixel(cv::Mat img, V2D pc);

  // void resetRvizDisplay();
  // deque<VisualPoint *> map_cur_frame;
  // deque<VisualPoint *> sub_map_ray;
  // deque<VisualPoint *> sub_map_ray_fov;
  // deque<VisualPoint *> visual_sub_map_cur;
  // deque<VisualPoint *> visual_converged_point;
  // std::vector<std::vector<V3D>> sample_points;

  // PointCloudXYZI::Ptr pg_down;
  // pcl::VoxelGrid<PointType> downSizeFilter;

private:
  struct ReferenceOnlyMigrationGeometrySeed
  {
    livo::GeometrySupportKey support_key;
    livo::GeometryVersion geometry_version;
    livo::MigrationPlane plane;
  };

  struct JointCameraContext
  {
    std::uint8_t camera_id{0U};
    double timestamp{0.0};
    cv::Mat gray_image;
    cv::Mat color_image;
    FramePtr frame;
    SubSparseMap submap;
    std::size_t unbounded_candidate_points{0U};
    double saturation_fraction{0.0};
    double dark_fraction{0.0};
    double mean_gradient{0.0};
    livo::PhotometricStatus photometric_status{
      livo::PhotometricStatus::kInvalidInput};
    std::optional<livo::CameraBatchTimeTransferEntry>
      time_transfer_entry;
    std::vector<VisualPoint *> generated_points;
    // Populated by the sealed adaptive scout. It is a dry-run posterior
    // rooted at the same LIO state and is consumed directly by the winner.
    std::optional<StatesGroup> tentative_state;
    std::vector<std::size_t> tentative_accepted_point_indices;
    std::vector<VisualPoint *> tentative_inlier_points;
  };

  struct JointCameraLinearization
  {
    JointVisualNormalEquation::Matrix6 hessian{
      JointVisualNormalEquation::Matrix6::Zero()};
    JointVisualNormalEquation::Vector6 gradient{
      JointVisualNormalEquation::Vector6::Zero()};
    std::size_t tracked_points{0U};
    std::size_t residual_count{0U};
    std::size_t inlier_count{0U};
    std::size_t total_patch_count{0U};
    double squared_residual_sum{0.0};
    double nis{std::numeric_limits<double>::quiet_NaN()};
    std::size_t nis_dof{0U};
    double normalized_pose_correction{
      std::numeric_limits<double>::quiet_NaN()};
    double robust_information_gain{0.0};
    double spatial_coverage{0.0};
    std::vector<std::size_t> accepted_point_indices;
  };

  struct AdaptiveVisualRuntimePreflightResult
  {
    livo::AdaptiveVisualRuntimeAudit audit;
    std::optional<
      livo::AdaptiveVisualRuntimeAttemptPermit> permit;
    bool ready{false};
  };

  struct JointUpdateResult
  {
    bool accepted{false};
    std::string reason{"not_attempted"};
    std::size_t contributing_cameras{0U};
    std::size_t tracked_points{0U};
    double residual_rmse{std::numeric_limits<double>::quiet_NaN()};
    double rotation_increment_rad{0.0};
    double translation_increment_m{0.0};
    double adaptive_policy_time_sec{0.0};
    double visual_update_time_sec{0.0};
    bool adaptive_policy_committed{false};
    livo::AdaptiveVisualPolicyPlan adaptive_plan;
    std::optional<
      livo::AdaptiveVisualPolicyPlanCapability>
      validated_adaptive_plan_capability;
    bool adaptive_legacy_fallback_used{false};
    // A sealed adaptive update is evaluated against an isolated state copy.
    // The caller commits this value exactly once after the receipt is sealed.
    std::optional<StatesGroup> tentative_state;
    std::array<livo::AdaptiveCameraObservation, kCameraCount>
      adaptive_observations;
    std::array<std::vector<std::size_t>, kCameraCount>
      accepted_point_indices;
  };

  struct SharedLandmarkShadowMetrics
  {
    std::size_t geometry_writes{0U};
    std::size_t bootstrapped_observations{0U};
    std::size_t accepted_observations{0U};
    std::size_t rejected_observations{0U};
    std::size_t stale_cas{0U};
    std::size_t adapter_stage_calls{0U};
    std::size_t lifecycle_lidar_seeded{0U};
    std::size_t lifecycle_same_camera{0U};
    std::size_t lifecycle_migration_shadow{0U};
    std::size_t lifecycle_migration_active{0U};
    std::size_t lifecycle_tombstoned{0U};
    bool photometric_revision_valid{false};
    bool time_transfer_revision_valid{false};
    std::string bootstrap_status{"not_called"};
    std::string observation_status{"not_called"};
    std::string revision_reason{"not_evaluated"};
    livo::VIORuntimeObservationAuthorityStatus
      runtime_observation_authority{
        livo::VIORuntimeObservationAuthorityStatus::
        kNotEvaluated};
    std::size_t runtime_observation_frames{0U};
  };

  using VisualFeatureMap =
    unordered_map<VOXEL_LOCATION, VOXEL_POINTS *>;

  void updateActiveCameraGeometry();
  bool prepareJointCameraImage(
    const CameraObservation & observation, JointCameraContext & context) const;
  bool prepareJointCamera(
    CameraObservation &observation, vector<pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
    JointCameraContext &context);
  JointCameraLinearization linearizeJointCamera(
    JointCameraContext &context, int level,
    std::size_t maximum_tracked_patches =
      std::numeric_limits<std::size_t>::max());
  JointCameraLinearization linearizeJointCameraAtState(
    JointCameraContext & context, int level,
    std::size_t maximum_tracked_patches, const StatesGroup & prior) const;
  livo::AdaptiveCameraObservations buildAdaptiveObservations(
    const CameraObservationBatch & batch,
    std::vector<JointCameraContext> & contexts,
    bool bind_candidate_generations,
    const livo::AdaptiveVisualPolicy::Matrix6 * lidar_pose_information = nullptr,
    bool lidar_information_valid = false);
  void advanceTrustedZeroMapReentryState(
    const livo::AdaptiveVisualCandidatePlanCapability &
      candidate_plan,
    std::uint64_t lidar_generation);
  AdaptiveVisualRuntimePreflightResult
  preflightAdaptiveSingleCameraRuntime(
    const CameraObservationBatch & batch,
    const livo::AdaptiveVisualPolicy::Matrix6 &
      lidar_pose_information,
    bool lidar_pose_information_valid,
    std::uint64_t lidar_feature_generation,
    const livo::AdaptiveVisualRuntimeAuthorityInput *
      adaptive_runtime_input);
  livo::AdaptiveVisualRuntimeAudit
  processAdaptiveSingleCameraRuntime(
    CameraObservationBatch & batch,
    vector<pointWithVar> & pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &
      feat_map,
    double img_time,
    double process_start_time,
    const livo::AdaptiveVisualRuntimeAuthorityInput &
      adaptive_runtime_input,
    livo::AdaptiveVisualRuntimeAttemptPermit
      runtime_attempt_permit,
    livo::AdaptiveVisualRuntimeAudit audit);
  std::optional<livo::AdaptiveVisualPolicyPlanCapability>
  issueValidatedAdaptiveRuntimePlan(
    const CameraObservationBatch & batch,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle,
    bool runtime_observation_authority_ready,
    const livo::AdaptiveVisualPolicy::Matrix6 &
      lidar_pose_information,
    bool lidar_pose_information_valid,
    const livo::AdaptiveCameraObservations & observations,
    const livo::AdaptiveVisualPolicyPlan & plan);
  std::optional<JointUpdateResult>
  executeAuthorizedAdaptiveMapBootstrap(
    const CameraObservationBatch & batch,
    std::vector<JointCameraContext> & contexts,
    vector<pointWithVar> & pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &
      feat_map,
    double img_time,
    const livo::AdaptiveVisualPolicy::Matrix6 &
      lidar_pose_information,
    bool lidar_pose_information_valid,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle,
    bool runtime_observation_authority_ready);
  void commitAdaptiveHealthObservation(
    JointUpdateResult & update_result);
  JointUpdateResult computeJointJacobianAndUpdateEKF(
    std::vector<JointCameraContext> &contexts,
    const livo::AdaptiveVisualPolicy::Matrix6 & lidar_pose_information,
    bool lidar_pose_information_valid,
    const CameraObservationBatch & batch,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle,
    bool runtime_observation_authority_ready,
    const livo::AdaptiveVisualCandidatePlanCapability *
      candidate_plan_capability,
    bool commit_state = true);
  void runMultiCameraLandmarkLinearizationShadow(
    const CameraObservationBatch & batch,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle,
    livo::VIORuntimeObservationAuthorityStatus
      runtime_authority,
    const JointUpdateResult & update_result,
    double timestamp_sec,
    std::uint64_t lidar_feature_generation);
  void emitMultiCameraLandmarkLinearizationShadowMetric(
    double timestamp_sec,
    std::uint64_t lidar_feature_generation,
    const livo::
      VIOMultiCameraLandmarkLinearizationShadowBatchAudit &
        audit) const;
  void limitJointCameraCandidates(
    JointCameraContext & context, std::size_t maximum_candidates);
  void finalizeJointCamera(
    JointCameraContext &context, vector<pointWithVar> &pg,
    const unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &feat_map,
    double img_time, double camera_time_offset_sec,
    const livo::AdaptiveVisualMapMaintenanceDecision &
      map_maintenance);
  std::optional<
    std::vector<ReferenceOnlyMigrationGeometrySeed>>
  bootstrapSharedLandmarksShadow(
    const CameraObservationBatch & batch,
    const std::vector<JointCameraContext> & contexts,
    const std::vector<pointWithVar> & lidar_points,
    std::uint64_t lidar_feature_generation,
    const livo::ActivePlaneIdentitySnapshot *
      active_plane_snapshot,
    const livo::CameraBatchTimeTransferBundle *
      time_transfer_bundle,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle,
    SharedLandmarkShadowMetrics & metrics);
  void runReferenceOnlyMigrationShadowRuntime(
    const CameraObservationBatch & batch,
    const std::vector<JointCameraContext> & contexts,
    const std::vector<ReferenceOnlyMigrationGeometrySeed> &
      geometry_seeds,
    double batch_timestamp_sec,
    std::uint64_t lidar_feature_generation,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle);
  void mirrorSharedLandmarkObservationsShadow(
    const CameraObservationBatch & batch,
    const std::vector<JointCameraContext> & contexts,
    const JointUpdateResult & update_result,
    std::uint64_t lidar_feature_generation,
    const livo::ActivePlaneIdentitySnapshot *
      active_plane_snapshot,
    const livo::CameraBatchTimeTransferBundle *
      time_transfer_bundle,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle,
    SharedLandmarkShadowMetrics & metrics);
  livo::VIORuntimeObservationAuthorityStatus
  validateRuntimeObservationAuthority(
    const CameraObservationBatch & batch,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle);
  bool collectRuntimeObservationFrameProvenances(
    const CameraObservationBatch & batch,
    const livo::ReferenceOnlyMigrationRuntimeBundle *
      runtime_bundle,
    SharedLandmarkShadowMetrics & metrics,
    std::array<std::shared_ptr<
      const livo::ReferenceOnlyMigrationFrameProvenance>,
      kCameraCount> & frame_provenances) const;
  void emitSharedLandmarkShadowMetric(
    double timestamp_sec,
    std::uint64_t lidar_feature_generation,
    const SharedLandmarkShadowMetrics & metrics) const;
  bool attachSharedLandmarkIdentity(
    VisualPoint & visual_point,
    const pointWithVar & lidar_point) const;
  void refreshSharedLandmarkRevisionSet(
    const std::vector<JointCameraContext> & contexts,
    std::uint64_t lidar_feature_generation,
    const livo::CameraBatchTimeTransferBundle *
      time_transfer_bundle,
    SharedLandmarkShadowMetrics & metrics);
  void refreshSharedLandmarkLifecycleMetrics(
    SharedLandmarkShadowMetrics & metrics) const;
  std::optional<livo::MigrationEndpoint>
  crossCameraMigrationShadowEndpoint(
    std::uint8_t camera_id) const;

  std::array<livo::CameraModelPtr, kCameraCount> cameras_;
  std::array<VisualFeatureMap, kCameraCount> camera_feature_maps_;
  VisualFeatureMap *active_visual_feature_map_ = nullptr;
  std::array<M3D, kCameraCount> lidar_to_camera_rotations_;
  std::array<V3D, kCameraCount> lidar_to_camera_translations_;
  std::array<std::string, kCameraCount> camera_names_{{"front", "left", "back", "right"}};
  std::array<bool, kCameraCount> camera_configured_{{false, false, false, false}};
  std::array<
    std::optional<livo::CameraModelParameters>,
    kCameraCount> registered_camera_parameters_;
  std::array<livo::AdaptiveVisualMapLifecycle, kCameraCount>
    adaptive_visual_map_lifecycle_{{
      livo::AdaptiveVisualMapLifecycle::kEmpty,
      livo::AdaptiveVisualMapLifecycle::kEmpty,
      livo::AdaptiveVisualMapLifecycle::kEmpty,
      livo::AdaptiveVisualMapLifecycle::kEmpty}};
  std::array<std::size_t, kCameraCount>
    adaptive_zero_map_reentry_healthy_scouts_
      {{0U, 0U, 0U, 0U}};
  std::array<bool, kCameraCount>
    adaptive_zero_map_reseed_latched_
      {{false, false, false, false}};
  std::array<std::uint64_t, kCameraCount>
    adaptive_zero_map_last_trusted_generation_
      {{0U, 0U, 0U, 0U}};
  std::unique_ptr<livo::AdaptiveVisualPolicy> adaptive_visual_policy_;
  livo::SharedLandmarkShadowSession
    shared_landmark_shadow_session_;
  livo::MultiCameraLandmarkLinearizationShadowConfig
    multi_camera_landmark_linearization_shadow_config_;
  std::unique_ptr<
    livo::MultiCameraLandmarkLinearizationShadow>
    multi_camera_landmark_linearization_shadow_;
  livo::CrossCameraMigrationShadowInitAudit
    cross_camera_migration_shadow_init_audit_;
  livo::ReferenceOnlyMigrationShadowRuntimeInitAudit
    reference_only_migration_shadow_runtime_init_audit_;
  livo::ReferenceOnlyMigrationShadowRuntimeBatchAudit
    reference_only_migration_shadow_runtime_batch_audit_;
  livo::VIOMultiCameraLandmarkLinearizationShadowBatchAudit
    multi_camera_landmark_linearization_shadow_batch_audit_;
  livo::VIOJointSolverReadOnlyAudit
    last_joint_solver_audit_;
  livo::AdaptiveVisualRuntimeAudit
    adaptive_visual_runtime_audit_;
  std::size_t adaptive_prepare_call_count_{0U};
  std::size_t adaptive_scout_call_count_{0U};
  std::size_t adaptive_finalize_call_count_{0U};
  std::size_t shared_landmark_shadow_entry_count_{0U};
  std::size_t
    reference_only_migration_shadow_entry_count_{0U};
  std::size_t
    multi_camera_linearization_shadow_entry_count_{0U};
  std::unique_ptr<livo::FourCameraPinholeRig>
    cross_camera_migration_shadow_rig_;
  std::uint64_t cross_camera_migration_owner_generation_{0U};
  livo::CameraObservationBatchAuthorityIdentity
    reference_only_migration_batch_authority_;
  livo::ReferenceOnlyMigrationRuntimePublisherIdentity
    reference_only_migration_publisher_authority_;
  bool reference_only_migration_authorities_configured_{false};
  bool reference_only_migration_authority_owner_match_{false};
  std::uint64_t
    reference_only_migration_authority_generation_{0U};
  std::uint64_t
    last_runtime_observation_batch_sequence_{0U};
  std::uint64_t
    last_runtime_observation_publication_sequence_{0U};
  bool imu_to_lidar_extrinsic_configured_{false};
};
typedef std::shared_ptr<VIOManager> VIOManagerPtr;

#endif // VIO_H_
