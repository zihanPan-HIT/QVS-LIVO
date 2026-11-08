from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_four_camera_frames_reach_one_joint_vio_update():
    header = (ROOT / "include" / "LIVMapper.h").read_text()
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()

    assert "CameraObservationQueue camera_observation_queue_" in header
    assert "std::optional<CameraObservationBatch> pending_vio_batch_" in header
    assert 'vio_camera_mode_ == "four"' in mapper
    assert "camera_transport_cbk(msg, 0U" in mapper
    assert "camera_transport_cbk(msg, 1U" in mapper
    assert "camera_transport_cbk(msg, 2U" in mapper
    assert "camera_transport_cbk(msg, 3U" in mapper
    assert "drainCameraTransportBuffer();" in mapper
    assert "handleMultiCameraImage(" in mapper
    assert "camera_observation_queue_.nextSynchronized(" in mapper
    assert "camera_observation_queue_.popSynchronized(" in mapper
    assert "camera_observation_queue_.synchronizedWatermark(" in mapper
    assert "pending_vio_batch_" in mapper
    assert "if (_pv_list.empty())" in mapper
    assert "if (!pcl_w_wait_pub->empty())" in mapper
    assert "processFrames(" in mapper
    assert "processFrames(" in vio_header
    assert "computeJointJacobianAndUpdateEKF" in vio_header
    assert "JointVisualNormalEquation" in vio_source
    assert "[VIO_JOINT_METRIC]" in vio_source
    assert "processFrame(CameraObservation &observation" in vio_header
    assert "activateCamera(observation.camera_id)" in vio_source
    assert "[VIO_METRIC]" in vio_source


def test_adaptive_four_camera_uses_equal_queue_and_authority_path():
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()

    adaptive_dispatch = mapper[
        mapper.index("bool LIVMapper::dispatchAdaptiveSingleCameraVio()") :
        mapper.index("void LIVMapper::readParameters")
    ]
    assert "camera_observation_queue_.nextAdaptiveSynchronized(" in mapper
    assert "camera_observation_queue_.popAdaptiveSynchronized(" in mapper
    assert "nextReadyForCamera(" not in mapper
    assert "popReadyForCamera(" not in mapper
    assert "popAdaptiveNonFrontSynchronized(" not in mapper
    assert "vio_manager->processFrame(" not in adaptive_dispatch
    assert adaptive_dispatch.index("issueLidarGeneration(") < adaptive_dispatch.index(
        "bindRuntimeInput("
    )
    assert adaptive_dispatch.index("bindRuntimeInput(") < adaptive_dispatch.index(
        "vio_manager->processFrames("
    )


def test_camera_supply_receipts_are_opt_in_and_source_bound():
    header = (ROOT / "include" / "LIVMapper.h").read_text()
    observation = (ROOT / "include" / "camera_observation.h").read_text()
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()

    assert "std::uint64_t source_stamp_ns{0U};" in observation
    assert "bool camera_supply_receipt_enable_ = false;" in header
    assert '"vio.camera_supply_receipt_enable", false' in mapper
    assert '"vio.camera_supply_receipt_enable",' in mapper
    assert "sourceStampNanoseconds(" in mapper
    assert "emitCameraSupplyReceipt(" in mapper
    for stage in (
        '"queue_result"',
        '"batch_outcome"',
    ):
        assert stage in mapper
    for result in ('"batch_admitted"', '"stale_rejected"'):
        assert result in mapper
    # adaptive3 counts pointer-only ingress separately from decoded admission.
    assert "[CAMERA_TRANSPORT_METRIC]" in mapper
    assert "[CAMERA_TRANSPORT_SUMMARY]" in mapper


def test_visual_patches_never_cross_physical_cameras():
    patches = (ROOT / "src" / "vio_patches.cpp").read_text()
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()

    assert patches.count("pt->cam_source != active_camera_id_") >= 2
    assert "pt_new->cam_source = active_camera_id_" in patches
    assert "std::array<livo::CameraModelPtr" in vio_header
    assert "std::array<VisualFeatureMap, kCameraCount>" in vio_header
    assert "active_visual_feature_map_ = &camera_feature_maps_[camera_id]" in vio_source
    assert "CameraModelDeleter" in (
        ROOT / "include" / "camera_model_factory.h"
    ).read_text()
    assert "registerCamera(" in vio_source
    assert "addExtraCameraFeatures" not in vio_header
    assert "addExtraCameraFeatures" not in vio_source


def test_joint_vio_start_is_timestamp_deterministic():
    header = (ROOT / "include" / "LIVMapper.h").read_text()
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()

    assert "joint_vio_warmup_sec_" in header
    assert '"vio.joint_warmup_sec"' in mapper
    assert "joint_warmup_end_time" in mapper
    assert "before_joint_warmup" in mapper
    warmup_guard = mapper[mapper.index("const auto before_joint_warmup") :]
    assert "!adaptive_policy_config_.enabled" in warmup_guard
    assert "!adaptive_front_baseline" not in warmup_guard


def test_joint_vio_consumes_each_lidar_feature_generation_once():
    header = (ROOT / "include" / "LIVMapper.h").read_text()
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()

    assert "lio_feature_generation_" in header
    assert "last_joint_vio_feature_generation_" in header
    assert "++lio_feature_generation_" not in mapper
    checked_increment = mapper.index("livo::checkedIncrement(")
    checked_generation = mapper.index(
        "lio_feature_generation_", checked_increment
    )
    exhausted_generation = mapper.index(
        "throw std::overflow_error", checked_generation
    )
    assert checked_increment < checked_generation < exhausted_generation
    assert "no_fresh_lidar" in mapper
    assert (
        "last_joint_vio_feature_generation_ = lio_feature_generation_"
        in mapper
    )


def test_mapper_dispatch_audit_has_typed_exception_phases():
    header = (ROOT / "include" / "LIVMapper.h").read_text()
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()

    for required_field in (
        "lidar_posterior",
        "generation_issued",
        "runtime_bound",
        "vio_dispatch_started",
        "vio_dispatch_completed",
        "kRuntimeInvariantViolation",
        "kResourceFailure",
        "kUnexpectedFailure",
    ):
        assert required_field in header
    for removed_field in (
        "snapshot_capture_count",
        "runtime_bind_count",
        "vio_dispatch_count",
        "legacy_generation_guard_used",
        "kBindRejected",
    ):
        assert removed_field not in header

    issue = mapper.index("issueLidarGeneration(")
    issue_resource_catch = mapper.index(
        "catch (const std::bad_alloc &)", issue
    )
    issue_resource_status = mapper.index(
        "kResourceFailure", issue_resource_catch
    )
    issue_retryable_return = mapper.index(
        "return false;", issue_resource_status
    )
    generation_issued = mapper.index(
        "generation_issued = true", issue_retryable_return
    )
    assert (
        issue
        < issue_resource_catch
        < issue_resource_status
        < issue_retryable_return
        < generation_issued
    )

    dispatch_started = mapper.index(
        "vio_dispatch_started = true", generation_issued
    )
    process_frames = mapper.index(
        "vio_manager->processFrames(", dispatch_started
    )
    process_resource_catch = mapper.index(
        "catch (const std::bad_alloc &)", process_frames
    )
    process_resource_status = mapper.index(
        "kResourceFailure", process_resource_catch
    )
    process_resource_rethrow = mapper.index(
        "throw;", process_resource_status
    )
    unexpected_catch = mapper.index(
        "catch (...)", process_resource_rethrow
    )
    unexpected_status = mapper.index(
        "kUnexpectedFailure", unexpected_catch
    )
    unexpected_rethrow = mapper.index(
        "throw;", unexpected_status
    )
    dispatch_completed = mapper.index(
        "vio_dispatch_completed = true", unexpected_rethrow
    )
    dispatched_status = mapper.index(
        "kVioDispatched", dispatch_completed
    )
    assert (
        dispatch_started
        < process_frames
        < process_resource_catch
        < process_resource_status
        < process_resource_rethrow
        < unexpected_catch
        < unexpected_status
        < unexpected_rethrow
        < dispatch_completed
        < dispatched_status
    )


def test_adaptive_scout_and_isolated_map_paths_are_read_only():
    vio_header = (ROOT / "include" / "vio.h").read_text()
    policy_header = (
        ROOT / "include" / "adaptive_visual_policy.h"
    ).read_text()
    policy_source = (
        ROOT / "src" / "adaptive_visual_policy.cpp"
    ).read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    patches = (ROOT / "src" / "vio_patches.cpp").read_text()

    assert "bool allow_cache_writes = true" in vio_header
    assert "!adaptive_policy_config.enabled" in vio_source
    assert "const bool map_action" in vio_source
    assert "if (map_action)" in vio_source
    assert "decision.isolated," in vio_source
    assert "bool camera_route_authorized" in policy_header
    assert "if (!camera_route_authorized)" in policy_source
    assert "planAdaptiveLioAnchoredMapMaintenance(" in vio_source
    assert "planAdaptiveVisualSolverMapCommit(" in vio_source
    assert (
        "if (!camera_route_authorized || !visual_update_accepted)"
        in policy_source
    )
    assert "kSeededUnvalidated" in policy_header
    assert "map_maintenance.update_existing_points);" in vio_source
    assert "if (!normal_en && allow_cache_writes) warp_map.clear();" in patches
    assert patches.count("if (allow_cache_writes)") >= 3
    assert "pt->ref_patch = ref_ftr;" in patches
    assert "warp_map[ref_ftr->id_] = ot;" in patches


def test_structured_joint_and_adaptive_metrics_use_one_atomic_write():
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    metric_source = (ROOT / "src" / "structured_metric.cpp").read_text()
    prefixes = (
        "ADAPTIVE_VIO_LIDAR_METRIC",
        "ADAPTIVE_VIO_CAMERA_METRIC",
        "ADAPTIVE_VIO_EVENT",
        "ADAPTIVE_VIO_UPDATE_METRIC",
        "VIO_JOINT_METRIC",
    )

    assert "emitStructuredMetric(" in vio_source
    assert "::write(STDOUT_FILENO" in metric_source
    assert "required_size" in metric_source
    assert "PIPE_BUF" in metric_source
    for prefix in prefixes:
        assert not re.search(
            rf'printf\(\s*"\[{re.escape(prefix)}\]', vio_source
        )
    assert vio_source.count("rotation_deg=%.6f") >= 2
    assert "total_s=%.6f generation=%llu" in vio_source


def test_observation_shadow_is_default_off_and_accepted_only():
    mapper_header = (ROOT / "include" / "LIVMapper.h").read_text()
    mapper_source = (ROOT / "src" / "LIVMapper.cpp").read_text()
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    default_config = (
        ROOT / "config" / "airy_d435_vio_ablation_ros2.yaml"
    ).read_text()

    assert (
        '"vio.multiview.shared_landmarks.observation_shadow_enable", false'
        in mapper_source
    )
    assert re.search(
        r"time_transfer:\s*\n\s+shadow_enable:\s*false",
        default_config,
    )
    assert re.search(
        r"shared_landmarks:\s*\n\s+shadow_enable:\s*false"
        r"\s*\n\s+observation_shadow_enable:\s*false",
        default_config,
    )
    assert "refit_shadow_enable: false" in default_config
    assert "observation_shadow_config_" in mapper_header
    assert "configureSharedLandmarkObservationShadow(" in vio_header
    assert "SharedLandmarkShadowSession" in vio_header
    assert "mirrorSharedLandmarkObservationsShadow(" in vio_header
    assert "if (!update_result.accepted)" in vio_source
    assert "decision.isolated" in vio_source
    assert "shared_landmark_shadow_session_.stageObservations(" in vio_source
    assert "[VIO_LANDMARK_OBSERVATION_SHADOW]" in vio_source
    for field in (
        "geometry=",
        "bootstrapped=",
        "accepted=",
        "rejected=",
        "stale_cas=",
        "lifecycle_lidar_seeded=",
        "lifecycle_same_camera=",
        "adapter_stage_calls=",
        "preview_stage_calls=0",
        "photo_revision_valid=",
        "time_revision_valid=",
        "revision_reason=",
        "consumable=0",
        "solver_applied=0",
    ):
        assert field in vio_source
    assert "shared_landmark_shadow_session_" in vio_header
    assert mapper_source.index(
        "configureSharedLandmarkShadow("
    ) < mapper_source.index("registerCamera(")
    assert "snapshotPlaneRefitEvents()" in mapper_source
    assert "acknowledgePlaneRefitEvents(" in mapper_source
    assert "[VIO_LANDMARK_REFIT_SHADOW]" in vio_source
    for removed_owner in (
        "shared_landmark_shadow_store_",
        "shared_landmark_revision_set_",
        "shared_landmark_lio_commit_token_",
        "shared_landmark_shadow_batch_id_",
        "shared_landmark_identity_cache_",
        "LegacyLandmarkObservationShadowAdapter",
    ):
        assert removed_owner not in vio_header
    assert "photometric_calibration_revisions.fill(1U)" not in vio_source
    assert "time_transfer_revision = 1U" not in vio_source
    assert "CrossCameraPatchMigration" not in vio_source
    assert "LandmarkReferenceRequest" not in vio_source


def test_legacy_visual_identity_is_value_owned():
    visual_point = (ROOT / "include" / "visual_point.h").read_text()
    feature = (ROOT / "include" / "feature.h").read_text()

    for source in (visual_point, feature):
        assert "livo::GeometrySupportKey shadow_support_key_" in source
        assert "livo::LandmarkId shadow_landmark_id_{0U}" in source
        assert "MultiViewLandmarkStore *" not in source


def test_active_plane_snapshot_flows_from_voxel_map_into_shadow_session_only():
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    session_header = (
        ROOT / "include" / "shared_landmark_shadow_session.h"
    ).read_text()

    assert "SharedLandmarkLifecycleRouting" in mapper
    assert "activePlaneLifecycleRequired()" in mapper
    assert re.search(
        r"if \(shared_landmark_lifecycle_routing\."
        r"\s*activePlaneSnapshotRequired\(\)\)\s*\{"
        r"\s*active_plane_snapshot ="
        r"\s*voxelmap_manager->snapshotActivePlaneIdentities\(\);",
        mapper,
    )
    assert re.search(
        r"const livo::ActivePlaneIdentitySnapshot\s*\*\s*"
        r"active_plane_snapshot\s*=\s*nullptr",
        vio_header,
    )
    snapshot_index = mapper.index(
        "voxelmap_manager->snapshotActivePlaneIdentities()"
    )
    legacy_process_index = mapper.index(
        "vio_manager->processFrames(", snapshot_index
    )
    snapshot_argument_index = mapper.index(
        "&*active_plane_snapshot", legacy_process_index
    )
    assert snapshot_index < legacy_process_index < snapshot_argument_index
    assert re.search(
        r"request\.active_planes\s*=\s*\*active_plane_snapshot\s*;",
        vio_source,
    )
    assert re.search(
        r"observation_request\.active_planes\s*=\s*"
        r"\*active_plane_snapshot\s*;",
        vio_source,
    )
    assert re.search(
        r"ActivePlaneIdentitySnapshot\s+active_planes\s*;",
        session_header,
    )
    assert "MultiViewLandmarkStore" not in vio_header


def test_vio_consumes_only_authoritative_per_camera_time_transfers():
    mapper = (ROOT / "src" / "LIVMapper.cpp").read_text()
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()

    assert (
        "std::optional<livo::CameraBatchTimeTransferEntry>"
        in vio_header
    )
    assert re.search(
        r"effective_time_transfer_bundle->transfers\[\s*"
        r"context\.camera_id\]\.has_value\(\)",
        vio_source,
    )
    assert re.search(
        r"context\.time_transfer_entry\s*=\s*"
        r"effective_time_transfer_bundle->transfers\[\s*"
        r"context\.camera_id\];",
        vio_source,
    )
    assert re.search(
        r"effective_time_transfer_bundle\s*=\s*"
        r"&migration_runtime_bundle->timeTransferBundle\(\);",
        vio_source,
    )
    assert re.search(
        r"const bool runtime_authority_required\s*="
        r".*reference_only_migration_shadow_runtime_init_audit_"
        r"\.effective\s*&&\s*migration_runtime_bundle\s*!=\s*nullptr"
        r".*if \(runtime_authority_required\)",
        vio_source,
        re.DOTALL,
    )
    assert not re.search(
        r"context\.time_transfer_entry\s*=\s*"
        r"time_transfer_bundle->transfers",
        vio_source,
    )
    assert "context.time_transfer_entry->value().raw()" in vio_source
    assert "observation.time_transfer =" in vio_source
    assert "request.time_transfer_batch =" in vio_source
    assert "bundle.transfers[camera_id].data()" in mapper


def test_multi_camera_landmark_linearization_shadow_is_explicitly_default_off():
    mapper_header = (ROOT / "include" / "LIVMapper.h").read_text()
    mapper_source = (ROOT / "src" / "LIVMapper.cpp").read_text()
    vio_header = (ROOT / "include" / "vio.h").read_text()
    cmake = (ROOT / "CMakeLists.txt").read_text()
    default_config = (
        ROOT / "config" / "airy_d435_vio_ablation_ros2.yaml"
    ).read_text()

    assert (
        "livo::MultiCameraLandmarkLinearizationShadowConfig\n"
        "    multi_camera_landmark_linearization_shadow_config_;"
        in mapper_header
    )
    assert (
        '"vio.multiview.shared_landmarks.linearization_shadow.enable", false'
        in mapper_source
    )
    assert (
        '"vio.multiview.shared_landmarks.linearization_shadow.'
        'allow_relative_affine_experiment", false'
        in mapper_source
    )
    assert (
        '"vio.multiview.shared_landmarks.linearization_shadow.'
        'maximum_source_age_ms", 500.0'
        in mapper_source
    )
    assert (
        '"vio.multiview.shared_landmarks.linearization_shadow.'
        'maximum_candidate_evaluations_per_batch", 16384'
        in mapper_source
    )
    assert re.search(
        r"linearization_shadow:\s*\n"
        r"\s+enable:\s*false\s*\n"
        r"\s+allow_relative_affine_experiment:\s*false",
        default_config,
    )
    assert "configureMultiCameraLandmarkLinearizationShadow(" in vio_header
    assert re.search(
        r"std::unique_ptr<\s*"
        r"livo::MultiCameraLandmarkLinearizationShadow\s*>",
        vio_header,
    )
    vio_links = re.search(
        r"target_link_libraries\(\s*vio(?P<body>.*?)\)",
        cmake,
        re.DOTALL,
    )
    assert vio_links is not None
    assert "multi_camera_landmark_linearization_shadow" in vio_links.group("body")
    for dependency in (
        'vio_camera_mode_ != "four"',
        "!adaptive_policy_config_.enabled",
        "!reference_only_migration_shadow_enable_",
        "!cross_camera_migration_shadow_enable_",
        "!camera_time_transfer_shadow_enable_",
        "!shared_landmark_shadow_config_.enabled",
        "!observation_shadow_config_.enabled",
    ):
        assert dependency in mapper_source


def test_multi_camera_landmark_shadow_uses_exact_owner_composition_only():
    vio_source = (ROOT / "src" / "vio.cpp").read_text()

    for exact_composition_check in (
        "capability->runtimeBundle().valid()",
        "capability->runtimeBundle().issuedBy(",
        "capability->runtimeBundle().validatedComposition().",
        "capability->runtimeBundle().\n        activeStateSnapshotRevision()",
        "sameReferenceOnlyMigrationFrameBindings(",
        "(capability->lidarPoseInformation().array() ==",
        "sameAdaptiveCameraObservations(",
        "sameAdaptiveVisualPolicyPlan(",
    ):
        assert exact_composition_check in vio_source
    for observation_field in (
        "left.camera_id == right.camera_id",
        "left.available_metrics == right.available_metrics",
        "left.candidate_points == right.candidate_points",
        "left.tracked_points == right.tracked_points",
        "left.residual_count == right.residual_count",
        "left.squared_residual_sum == right.squared_residual_sum",
        "left.saturation_fraction == right.saturation_fraction",
        "left.dark_fraction == right.dark_fraction",
        "left.mean_gradient == right.mean_gradient",
        "left.spatial_coverage == right.spatial_coverage",
        "(left.visual_information.array() ==",
    ):
        assert observation_field in vio_source


def test_adaptive_switch_confirmation_state_and_configuration_are_bound():
    policy_header = (
        ROOT / "include" / "adaptive_visual_policy.h"
    ).read_text()
    mapper_source = (ROOT / "src" / "LIVMapper.cpp").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    default_config = (
        ROOT / "config" / "airy_d435_vio_ablation_ros2.yaml"
    ).read_text()

    assert "kAdaptiveVisualPolicyConfigRevision = 7U" in policy_header
    assert (
        '"vio.adaptive.selection_challenger_frames", 3'
        in mapper_source
    )
    assert (
        'read_positive_size("vio.adaptive.selection_challenger_frames")'
        in mapper_source
    )
    assert (
        '"vio.adaptive.minimum_selection_tracked_points", 1'
        in mapper_source
    )
    assert (
        'read_positive_size("vio.adaptive.minimum_selection_tracked_points")'
        in mapper_source
    )
    assert re.search(
        r"selection_challenger_frames:\s*3",
        default_config,
    )
    assert (
        "left.proposed_selection_challenger.camera_id !="
        in vio_source
    )
    assert (
        "left.proposed_selection_challenger.consecutive_confirmations !="
        in vio_source
    )


def test_multi_camera_landmark_shadow_has_one_validated_capability_state():
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    joint_update_result = vio_header[
        vio_header.index("  struct JointUpdateResult") :
        vio_header.index("  struct SharedLandmarkShadowMetrics")
    ]

    assert "validated_adaptive_plan_capability" in joint_update_result
    assert "adaptive_plan_capability_validated" not in joint_update_result
    assert "kUnvalidatedPlanCapability" not in vio_header
    assert "kUnvalidatedPlanCapability" not in vio_source
    assert (
        "update_result.adaptive_plan_capability_validated"
        not in vio_source
    )


def test_multi_camera_landmark_shadow_is_metrics_only_and_runs_before_commits():
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    process_frames = vio_source[
        vio_source.index("void VIOManager::processFrames(") :
        vio_source.index("void VIOManager::processFrame(")
    ]
    shadow_runner = vio_source[
        vio_source.index(
            "void VIOManager::runMultiCameraLandmarkLinearizationShadow("
        ) :
        vio_source.index(
            "void VIOManager::\n"
            "emitMultiCameraLandmarkLinearizationShadowMetric("
        )
    ]
    initialize_vio = vio_source[
        vio_source.index("void VIOManager::initializeVIO()") :
        vio_source.index("void VIOManager::resetGrid()")
    ]

    common_path_start = process_frames.index(
        "\n  adaptive_visual_runtime_audit_ = {};"
    )
    single_camera_path = process_frames[:common_path_start]
    common_path = process_frames[common_path_start:]
    assert "if (adaptive_single_camera_runtime)" in single_camera_path
    assert (
        "runMultiCameraLandmarkLinearizationShadow("
        not in single_camera_path
    )
    assert (
        "mirrorSharedLandmarkObservationsShadow("
        not in single_camera_path
    )
    assert re.search(
        r"adaptive_visual_runtime_audit_\s*=\s*"
        r"processAdaptiveSingleCameraRuntime\(",
        single_camera_path,
    )

    invocation = common_path.index(
        "runMultiCameraLandmarkLinearizationShadow("
    )
    assert invocation < common_path.index(
        "advanceVisualStateRevision()"
    )
    assert invocation < common_path.index(
        "commitAdaptiveHealthObservation(update_result);"
    )
    assert invocation < common_path.index(
        "mirrorSharedLandmarkObservationsShadow("
    )
    assert "MultiCameraLandmarkSelectionTrace" not in vio_header
    assert "MultiCameraLandmarkSelectionTrace" not in vio_source
    assert "selectionTrace()" not in vio_source
    assert (
        "multi_camera_landmark_linearization_shadow_->accepts("
        not in vio_source
    )
    assert "decision.hasSelectionTrace()" in vio_source
    process_batch = shadow_runner.index("->processBatch(")
    for fail_closed_gate in (
        "if (!audit.configured_enabled)",
        "if (!audit.module_active)",
        "if (runtime_bundle == nullptr)",
        "runtime_authority !=",
        "if (!update_result.validated_adaptive_plan_capability.has_value())",
    ):
        assert shadow_runner.index(fail_closed_gate) < process_batch
    assert "result.skippedInactivePlaneSourceCount()" in vio_source
    assert "skipped_inactive_plane=%zu" in vio_source
    assert "skipped_inactive_plane_sources" in vio_header
    assert initialize_vio.index(
        "multi_camera_landmark_linearization_shadow_.reset()"
    ) < initialize_vio.index("adaptive_visual_policy_.reset()")
    assert initialize_vio.index(
        "adaptive_visual_policy_ ="
    ) < initialize_vio.index(
        "multi_camera_landmark_linearization_shadow_ ="
    )
    for field in (
        "has_linearization=0",
        "consumable=0",
        "production_ready=0",
        "solver_applied=0",
    ):
        assert field in vio_source


def test_sealed_single_camera_runtime_is_deep_and_reuses_one_scout_builder():
    vio_header = (ROOT / "include" / "vio.h").read_text()
    vio_source = (ROOT / "src" / "vio.cpp").read_text()
    process_frames = vio_source[
        vio_source.index("void VIOManager::processFrames(") :
        vio_source.index("void VIOManager::processFrame(")
    ]
    joint_update = vio_source[
        vio_source.index(
            "VIOManager::computeJointJacobianAndUpdateEKF("
        ) :
        vio_source.index(
            "void VIOManager::runMultiCameraLandmarkLinearizationShadow("
        )
    ]
    single_camera_runtime = vio_source[
        vio_source.index(
            "VIOManager::processAdaptiveSingleCameraRuntime("
        ) :
        vio_source.index(
            "void VIOManager::commitAdaptiveHealthObservation("
        )
    ]

    assert "buildAdaptiveObservations(" in vio_header
    assert "preflightAdaptiveSingleCameraRuntime(" in vio_header
    assert "processAdaptiveSingleCameraRuntime(" in vio_header
    assert "buildAdaptiveObservations(" in joint_update
    assert (
        "preflightAdaptiveSingleCameraRuntime(" in process_frames
    )
    assert "processAdaptiveSingleCameraRuntime(" in process_frames
    assert (
        "selected_context.frame->exposure_timestamp_sec_ =\n"
        "    receipt.candidateExposureTimestampSec("
        "selected_camera);"
        in single_camera_runtime
    )
    for deep_runtime_detail in (
        "linearizeJointCamera(",
        "issueCandidatePlan(",
        "issueSelection(",
        "consumeSelection(",
        "commitHealthObservation(",
    ):
        assert deep_runtime_detail not in process_frames
