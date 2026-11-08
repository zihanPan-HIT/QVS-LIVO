cmake_minimum_required(VERSION 3.10)
project(livo_core)

option(
  LIVO_BUILD_EXPERIMENTAL_SHADOWS
  "Allow the P3-P5 shadow/reference-only research runtime"
  OFF
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

if("$ENV{ROS_DISTRO}" STRLESS "iron")
  add_definitions(-DPRE_ROS_IRON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(vikit_common REQUIRED)
find_package(vikit_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)
if(NOT TARGET JsonCpp::JsonCpp)
  find_package(jsoncpp CONFIG REQUIRED)
endif()
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenMP QUIET)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${vikit_common_INCLUDE_DIRS}
  ${vikit_ros_INCLUDE_DIRS}
  ${livox_ros_driver2_INCLUDE_DIRS}
)

set(ROS_DEPS
  rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  cv_bridge
  image_transport
  livox_ros_driver2
  vikit_common
  vikit_ros
)

add_library(structured_metric
  src/structured_metric.cpp
)

add_library(adaptive_lio_mode_ledger
  src/adaptive_lio_mode_ledger.cpp
)

add_library(adaptive_visual_policy
  src/adaptive_visual_policy.cpp
)
target_link_libraries(adaptive_visual_policy Eigen3::Eigen)

add_library(adaptive_visual_selection_authority
  src/adaptive_visual_selection_authority.cpp
)
target_link_libraries(
  adaptive_visual_selection_authority
  adaptive_visual_policy
  camera_observation_batch_authority
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)

add_library(geometry_identity
  src/active_plane_identity_registry.cpp
  src/geometry_identity.cpp
)
target_link_libraries(geometry_identity Eigen3::Eigen)

add_library(multi_view_landmark
  src/multi_view_landmark.cpp
)
target_link_libraries(
  multi_view_landmark
  geometry_identity
  Eigen3::Eigen
)

add_library(plane_lifecycle
  src/plane_lifecycle.cpp
)
target_link_libraries(
  plane_lifecycle
  geometry_identity
  Eigen3::Eigen
)

add_library(shared_landmark_tombstone_shadow
  src/shared_landmark_tombstone_shadow.cpp
)
target_link_libraries(
  shared_landmark_tombstone_shadow
  multi_view_landmark
  plane_lifecycle
  Eigen3::Eigen
)

add_library(shared_landmark_refit_shadow
  src/shared_landmark_refit_shadow.cpp
)
target_link_libraries(
  shared_landmark_refit_shadow
  multi_view_landmark
  plane_lifecycle
  Eigen3::Eigen
)

add_library(legacy_landmark_shadow_adapter
  src/legacy_landmark_shadow_adapter.cpp
)
target_link_libraries(
  legacy_landmark_shadow_adapter
  multi_view_landmark
  Eigen3::Eigen
)

add_library(legacy_landmark_observation_shadow_adapter
  src/legacy_landmark_observation_shadow_adapter.cpp
)
target_link_libraries(
  legacy_landmark_observation_shadow_adapter
  camera_time_transfer
  multi_view_landmark
  Eigen3::Eigen
)

add_library(shared_landmark_shadow_session
  src/shared_landmark_shadow_session.cpp
)
target_link_libraries(
  shared_landmark_shadow_session
  camera_batch_time_transfer
  camera_time_transfer
  legacy_landmark_observation_shadow_adapter
  legacy_landmark_shadow_adapter
  multi_view_landmark
  photometric_calibration
  shared_landmark_refit_shadow
  shared_landmark_tombstone_shadow
  Eigen3::Eigen
)

add_library(camera_time_transfer
  STATIC
  src/camera_time_transfer.cpp
)
target_link_libraries(camera_time_transfer Eigen3::Eigen)
target_compile_options(
  camera_time_transfer
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)

add_library(exposure_timing_attestation
  STATIC
  src/exposure_timing_attestation.cpp
)
# Variant compatibility is enforced for supported CMake targets. Direct raw
# linkage of these lazy static archives bypasses usage requirements and is
# unsupported.
set_property(
  TARGET exposure_timing_attestation
  PROPERTY INTERFACE_LIVO_ATTESTED_FRAME_VARIANT production
)
set_property(
  TARGET exposure_timing_attestation
  APPEND PROPERTY COMPATIBLE_INTERFACE_STRING
  LIVO_ATTESTED_FRAME_VARIANT
)
target_compile_options(
  exposure_timing_attestation
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)

add_library(frozen_camera_image
  STATIC
  src/frozen_camera_image.cpp
)
set_property(
  TARGET frozen_camera_image
  PROPERTY INTERFACE_LIVO_ATTESTED_FRAME_VARIANT production
)
set_property(
  TARGET frozen_camera_image
  APPEND PROPERTY COMPATIBLE_INTERFACE_STRING
  LIVO_ATTESTED_FRAME_VARIANT
)
target_link_libraries(
  frozen_camera_image
  OpenSSL::Crypto
  ${OpenCV_LIBRARIES}
)

add_library(attested_frozen_camera_frame
  STATIC
  src/attested_frozen_camera_frame.cpp
)
set_property(
  TARGET attested_frozen_camera_frame
  PROPERTY INTERFACE_LIVO_ATTESTED_FRAME_VARIANT production
)
set_property(
  TARGET attested_frozen_camera_frame
  APPEND PROPERTY COMPATIBLE_INTERFACE_STRING
  LIVO_ATTESTED_FRAME_VARIANT
)
target_link_libraries(
  attested_frozen_camera_frame
  exposure_timing_attestation
  frozen_camera_image
  OpenSSL::Crypto
  ${OpenCV_LIBRARIES}
)
target_compile_options(
  attested_frozen_camera_frame
  PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
)

add_library(adaptive_camera_snapshot_queue
  src/adaptive_camera_snapshot_queue.cpp
)
target_link_libraries(
  adaptive_camera_snapshot_queue
  frozen_camera_image
  ${OpenCV_LIBRARIES}
)

add_library(camera_observation_batch_authority
  src/camera_observation_batch_authority.cpp
)
target_link_libraries(
  camera_observation_batch_authority
  frozen_camera_image
  ${OpenCV_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(camera_observation_batch_authority OpenMP::OpenMP_CXX)
endif()

add_library(full_state_visual_update
  src/full_state_visual_update.cpp
)
target_link_libraries(
  full_state_visual_update
  Eigen3::Eigen
)

add_library(camera_batch_time_transfer
  src/camera_batch_time_transfer.cpp
)
target_link_libraries(
  camera_batch_time_transfer
  camera_time_transfer
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)

add_library(reference_only_migration_runtime_provenance
  src/reference_only_migration_runtime_provenance.cpp
)
ament_target_dependencies(
  reference_only_migration_runtime_provenance
  ${ROS_DEPS}
)
target_link_libraries(
  reference_only_migration_runtime_provenance
  camera_observation_batch_authority
  camera_batch_time_transfer
  camera_time_transfer
  ${OpenCV_LIBRARIES}
  Eigen3::Eigen
)
target_link_libraries(
  adaptive_visual_policy
  reference_only_migration_runtime_provenance
)
target_link_libraries(
  multi_view_landmark
  reference_only_migration_runtime_provenance
)
target_link_libraries(
  legacy_landmark_observation_shadow_adapter
  reference_only_migration_runtime_provenance
)
target_link_libraries(
  shared_landmark_shadow_session
  reference_only_migration_runtime_provenance
)

add_library(multi_camera_landmark_linearization_shadow
  src/multi_camera_landmark_linearization_shadow.cpp
)
target_link_libraries(
  multi_camera_landmark_linearization_shadow
  adaptive_visual_policy
  camera_observation_batch_authority
  multi_view_landmark
  reference_only_migration_runtime_provenance
  shared_landmark_shadow_session
  Eigen3::Eigen
)

add_library(photometric_calibration
  src/photometric_calibration.cpp
)

add_library(photometric_noise_authority
  src/photometric_noise_authority.cpp
)
target_link_libraries(
  photometric_noise_authority
  photometric_calibration
)

set(
  LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX
  ""
  CACHE STRING
  "Application-owned Ed25519 photometric trust-root raw public key"
)
set(
  LIVO_PHOTOMETRIC_TRUST_DOMAIN
  ""
  CACHE STRING
  "Application-owned photometric calibration trust domain"
)
if(LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX STREQUAL "")
  if(NOT LIVO_PHOTOMETRIC_TRUST_DOMAIN STREQUAL "")
    message(FATAL_ERROR
      "Photometric trust-root key and domain must be configured together"
    )
  endif()
elseif(LIVO_PHOTOMETRIC_TRUST_DOMAIN STREQUAL "")
  message(FATAL_ERROR
    "Photometric trust-root key and domain must be configured together"
  )
endif()
if(NOT LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX STREQUAL "")
  string(
    LENGTH
    "${LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX}"
    LIVO_PHOTOMETRIC_TRUST_ROOT_KEY_LENGTH
  )
  string(
    LENGTH
    "${LIVO_PHOTOMETRIC_TRUST_DOMAIN}"
    LIVO_PHOTOMETRIC_TRUST_DOMAIN_LENGTH
  )
  if(
    NOT LIVO_PHOTOMETRIC_TRUST_ROOT_KEY_LENGTH EQUAL 64
    OR NOT LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX
      MATCHES "^[0-9a-f]+$"
    OR LIVO_PHOTOMETRIC_TRUST_DOMAIN_LENGTH GREATER 128
    OR NOT LIVO_PHOTOMETRIC_TRUST_DOMAIN
      MATCHES "^[A-Za-z0-9][A-Za-z0-9._:-]*$"
  )
    message(FATAL_ERROR
      "Photometric trust root must be 64 lowercase hex digits and its "
      "domain must be 1..128 allowlisted ASCII characters"
    )
  endif()
endif()

add_library(trusted_photometric_calibration_loader
  src/trusted_photometric_calibration_loader.cpp
)
target_link_libraries(
  trusted_photometric_calibration_loader
  photometric_calibration
  photometric_noise_authority
  OpenSSL::Crypto
  JsonCpp::JsonCpp
)
if(NOT LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX STREQUAL "")
  target_compile_definitions(
    trusted_photometric_calibration_loader
    PRIVATE
    LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX="${LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX}"
    LIVO_PHOTOMETRIC_TRUST_DOMAIN="${LIVO_PHOTOMETRIC_TRUST_DOMAIN}"
  )
endif()

add_library(current_frame_photometric_evidence
  src/current_frame_photometric_evidence.cpp
)
target_link_libraries(
  current_frame_photometric_evidence
  camera_observation_batch_authority
  photometric_calibration
  photometric_noise_authority
  reference_only_migration_runtime_provenance
  ${OpenCV_LIBRARIES}
)

add_library(cross_camera_patch_migration
  src/cross_camera_patch_migration.cpp
)
target_link_libraries(
  cross_camera_patch_migration
  camera_time_transfer
  photometric_calibration
  multi_view_landmark
  Eigen3::Eigen
)

add_library(reference_only_migration_shadow
  src/reference_only_migration_shadow.cpp
)
target_link_libraries(
  reference_only_migration_shadow
  camera_batch_time_transfer
  cross_camera_patch_migration
  geometry_identity
  Eigen3::Eigen
)

add_library(reference_only_migration_shadow_batch_runner
  src/reference_only_migration_shadow_batch_runner.cpp
)
target_link_libraries(
  reference_only_migration_shadow_batch_runner
  reference_only_migration_shadow
  Eigen3::Eigen
)

add_library(reference_only_migration_runtime_budget
  src/reference_only_migration_runtime_budget.cpp
)
target_link_libraries(
  reference_only_migration_runtime_budget
  cross_camera_patch_migration
  geometry_identity
  OpenSSL::Crypto
  Eigen3::Eigen
)

add_library(pinhole_camera_projection_adapter
  src/pinhole_camera_projection_adapter.cpp
)
target_link_libraries(
  pinhole_camera_projection_adapter
  Eigen3::Eigen
)

add_library(four_camera_pinhole_rig
  src/four_camera_pinhole_rig.cpp
)
target_link_libraries(
  four_camera_pinhole_rig
  cross_camera_patch_migration
  pinhole_camera_projection_adapter
  OpenSSL::Crypto
  Eigen3::Eigen
)

add_library(vio
  src/vio.cpp
  src/vio_patches.cpp
  src/vio_visualization.cpp
  src/frame.cpp
  src/visual_point.cpp
)
target_link_libraries(
  vio
  adaptive_visual_policy
  adaptive_visual_selection_authority
  four_camera_pinhole_rig
  multi_camera_landmark_linearization_shadow
  reference_only_migration_runtime_budget
  reference_only_migration_shadow_batch_runner
  reference_only_migration_runtime_provenance
  shared_landmark_shadow_session
  structured_metric
)
add_library(camera_model_factory src/camera_model_factory.cpp)
add_library(lio src/voxel_map.cpp)
target_link_libraries(lio plane_lifecycle)
add_library(pre src/preprocess.cpp)
add_library(imu_proc src/IMU_Processing.cpp)
add_library(laser_mapping src/LIVMapper.cpp)
add_executable(livo_mapping src/main.cpp)

foreach(target vio lio laser_mapping)
  target_compile_definitions(
    ${target}
    PUBLIC
    LIVO_BUILD_EXPERIMENTAL_SHADOWS=$<BOOL:${LIVO_BUILD_EXPERIMENTAL_SHADOWS}>
  )
endforeach()

target_link_libraries(
  laser_mapping
  adaptive_lio_mode_ledger
  adaptive_visual_selection_authority
  camera_observation_batch_authority
  camera_batch_time_transfer
  camera_time_transfer
  reference_only_migration_runtime_provenance
  structured_metric
)

foreach(target vio lio pre imu_proc laser_mapping livo_mapping)
  target_compile_definitions(${target} PRIVATE EIGEN_DONT_PARALLELIZE)
  ament_target_dependencies(${target} ${ROS_DEPS})
  target_link_libraries(${target}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${Boost_LIBRARIES}
  )
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} OpenMP::OpenMP_CXX)
  endif()
endforeach()

# vikit_common was built as C++14. Its Eigen-aligned camera allocation ABI
# differs from C++17, so camera construction must remain in a C++14 TU.
ament_target_dependencies(camera_model_factory
  vikit_common
)
set_target_properties(camera_model_factory PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
set_source_files_properties(src/camera_model_factory.cpp PROPERTIES
  COMPILE_OPTIONS "-std=c++14"
)

target_link_libraries(livo_mapping
  laser_mapping
  camera_model_factory
  vio
  lio
  pre
  imu_proc
)

install(PROGRAMS scripts/camera_rate_gate.py DESTINATION lib/livo_core)
install(TARGETS livo_mapping DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(CODE [[
  set(_livo_history_prefix "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
  file(SHA256 "${_livo_history_prefix}/lib/livo_core/livo_mapping" _livo_history_sha256)
  file(WRITE "${_livo_history_prefix}/share/livo_core/imu_history_contract.json"
    "{\"contract_version\":1,\"executable_sha256\":\"${_livo_history_sha256}\"}\n")
]])

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_local_scan_contract test/test_local_scan_contract.cpp)
  ament_target_dependencies(test_local_scan_contract sensor_msgs)
  ament_add_gtest(
    test_imu_first_propagation
    test/test_imu_first_propagation.cpp
  )
  target_link_libraries(test_imu_first_propagation imu_proc)
  ament_target_dependencies(test_imu_first_propagation ${ROS_DEPS})
  find_program(VALGRIND_EXECUTABLE valgrind)
  if(VALGRIND_EXECUTABLE)
    add_test(
      NAME test_imu_first_propagation_valgrind
      COMMAND
        ${VALGRIND_EXECUTABLE}
        --tool=memcheck
        --track-origins=yes
        --undef-value-errors=yes
        --error-exitcode=99
        $<TARGET_FILE:test_imu_first_propagation>
    )
    set_tests_properties(
      test_imu_first_propagation_valgrind
      PROPERTIES LABELS "first-frame;memcheck"
    )
  endif()
  ament_add_gtest(
    test_adaptive_lio_mode_ledger
    test/test_adaptive_lio_mode_ledger.cpp
  )
  target_link_libraries(
    test_adaptive_lio_mode_ledger
    adaptive_lio_mode_ledger
  )
  add_library(
    camera_time_transfer_failure_injection_test
    STATIC
    src/camera_time_transfer.cpp
  )
  target_compile_definitions(
    camera_time_transfer_failure_injection_test
    PRIVATE
    LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING=1
  )
  target_link_libraries(
    camera_time_transfer_failure_injection_test
    Eigen3::Eigen
  )
  target_compile_options(
    camera_time_transfer_failure_injection_test
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
  )
  add_library(
    exposure_timing_attestation_failure_injection_test
    STATIC
    src/exposure_timing_attestation.cpp
  )
  set_property(
    TARGET exposure_timing_attestation_failure_injection_test
    PROPERTY INTERFACE_LIVO_ATTESTED_FRAME_VARIANT
    failure_injection
  )
  set_property(
    TARGET exposure_timing_attestation_failure_injection_test
    APPEND PROPERTY COMPATIBLE_INTERFACE_STRING
    LIVO_ATTESTED_FRAME_VARIANT
  )
  target_compile_definitions(
    exposure_timing_attestation_failure_injection_test
    PRIVATE
    LIVO_EXPOSURE_TIMING_ATTESTATION_FAILURE_INJECTION_TESTING=1
  )
  target_compile_options(
    exposure_timing_attestation_failure_injection_test
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
  )
  ament_add_gtest(
    test_exposure_timing_attestation
    test/test_exposure_timing_attestation.cpp
  )
  target_link_libraries(
    test_exposure_timing_attestation
    exposure_timing_attestation_failure_injection_test
  )
  add_library(
    frozen_camera_image_failure_injection_test
    STATIC
    src/frozen_camera_image.cpp
  )
  set_property(
    TARGET frozen_camera_image_failure_injection_test
    PROPERTY INTERFACE_LIVO_ATTESTED_FRAME_VARIANT
    failure_injection
  )
  set_property(
    TARGET frozen_camera_image_failure_injection_test
    APPEND PROPERTY COMPATIBLE_INTERFACE_STRING
    LIVO_ATTESTED_FRAME_VARIANT
  )
  target_compile_definitions(
    frozen_camera_image_failure_injection_test
    PRIVATE
    LIVO_FROZEN_CAMERA_IMAGE_FAILURE_INJECTION_TESTING=1
  )
  target_link_libraries(
    frozen_camera_image_failure_injection_test
    OpenSSL::Crypto
    ${OpenCV_LIBRARIES}
  )
  add_library(
    attested_frozen_camera_frame_failure_injection_test
    STATIC
    src/attested_frozen_camera_frame.cpp
  )
  set_property(
    TARGET attested_frozen_camera_frame_failure_injection_test
    PROPERTY INTERFACE_LIVO_ATTESTED_FRAME_VARIANT
    failure_injection
  )
  set_property(
    TARGET attested_frozen_camera_frame_failure_injection_test
    APPEND PROPERTY COMPATIBLE_INTERFACE_STRING
    LIVO_ATTESTED_FRAME_VARIANT
  )
  target_compile_definitions(
    attested_frozen_camera_frame_failure_injection_test
    PRIVATE
    LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING=1
  )
  target_link_libraries(
    attested_frozen_camera_frame_failure_injection_test
    exposure_timing_attestation_failure_injection_test
    frozen_camera_image_failure_injection_test
    OpenSSL::Crypto
    ${OpenCV_LIBRARIES}
  )
  target_compile_options(
    attested_frozen_camera_frame_failure_injection_test
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
  )
  add_library(
    adaptive_camera_snapshot_queue_failure_injection_test
    src/adaptive_camera_snapshot_queue.cpp
    src/frozen_camera_image.cpp
  )
  target_compile_definitions(
    adaptive_camera_snapshot_queue_failure_injection_test
    PRIVATE
    LIVO_ADAPTIVE_CAMERA_SNAPSHOT_QUEUE_FAILURE_INJECTION_TESTING=1
    LIVO_FROZEN_CAMERA_IMAGE_FAILURE_INJECTION_TESTING=1
  )
  target_link_libraries(
    adaptive_camera_snapshot_queue_failure_injection_test
    OpenSSL::Crypto
    ${OpenCV_LIBRARIES}
  )
  add_library(
    adaptive_visual_policy_failure_injection_test
    src/adaptive_visual_policy.cpp
  )
  target_compile_definitions(
    adaptive_visual_policy_failure_injection_test
    PRIVATE
    LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING=1
  )
  target_link_libraries(
    adaptive_visual_policy_failure_injection_test
    reference_only_migration_runtime_provenance
    Eigen3::Eigen
  )
  add_library(
    adaptive_visual_selection_authority_failure_injection_test
    src/adaptive_visual_selection_authority.cpp
  )
  target_compile_definitions(
    adaptive_visual_selection_authority_failure_injection_test
    PRIVATE
    LIVO_ADAPTIVE_VISUAL_FAILURE_INJECTION_TESTING=1
  )
  target_link_libraries(
    adaptive_visual_selection_authority_failure_injection_test
    adaptive_visual_policy
    camera_observation_batch_authority
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(test_camera_observation_queue test/test_camera_observation_queue.cpp)
  target_link_libraries(
    test_camera_observation_queue
    camera_observation_batch_authority
    ${OpenCV_LIBRARIES}
  )
  ament_add_gtest(
    test_frozen_camera_image
    test/test_frozen_camera_image.cpp
  )
  target_link_libraries(
    test_frozen_camera_image
    frozen_camera_image_failure_injection_test
    ${OpenCV_LIBRARIES}
  )
  ament_add_gtest(
    test_attested_frozen_camera_frame_production
    test/test_attested_frozen_camera_frame_production.cpp
  )
  target_link_libraries(
    test_attested_frozen_camera_frame_production
    attested_frozen_camera_frame
  )
  ament_add_gtest(
    test_attested_frozen_camera_frame
    test/test_attested_frozen_camera_frame.cpp
  )
  target_link_libraries(
    test_attested_frozen_camera_frame
    attested_frozen_camera_frame_failure_injection_test
  )
  ament_add_gtest(
    test_adaptive_camera_snapshot_queue
    test/test_adaptive_camera_snapshot_queue.cpp
  )
  target_link_libraries(
    test_adaptive_camera_snapshot_queue
    adaptive_camera_snapshot_queue_failure_injection_test
    ${OpenCV_LIBRARIES}
  )
  ament_add_gtest(
    test_reference_only_migration_runtime_provenance
    test/test_reference_only_migration_runtime_provenance.cpp
  )
  target_link_libraries(
    test_reference_only_migration_runtime_provenance
    reference_only_migration_runtime_provenance
    camera_observation_batch_authority
    camera_batch_time_transfer
    camera_time_transfer
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_reference_only_migration_runtime_budget
    test/test_reference_only_migration_runtime_budget.cpp
  )
  target_link_libraries(
    test_reference_only_migration_runtime_budget
    reference_only_migration_runtime_budget
    Eigen3::Eigen
  )
  ament_add_gtest(test_joint_visual_update test/test_joint_visual_update.cpp)
  ament_add_gtest(
    test_full_state_visual_update
    test/test_full_state_visual_update.cpp
  )
  target_link_libraries(
    test_full_state_visual_update
    full_state_visual_update
    camera_time_transfer
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_camera_time_transfer
    test/test_camera_time_transfer.cpp
  )
  target_link_libraries(
    test_camera_time_transfer
    camera_time_transfer
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_camera_time_transfer_epoch_receipt
    test/test_camera_time_transfer_epoch_receipt.cpp
  )
  target_link_libraries(
    test_camera_time_transfer_epoch_receipt
    camera_time_transfer_failure_injection_test
    Eigen3::Eigen
  )
  ament_add_pytest_test(
    test_camera_time_transfer_epoch_receipt_compile_contract
    test/test_camera_time_transfer_epoch_receipt_compile_contract.py
    TIMEOUT 120
    ENV
    "LIVO_CAMERA_TIME_TRANSFER_BUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
  )
  ament_add_gtest(
    test_camera_batch_time_transfer
    test/test_camera_batch_time_transfer.cpp
  )
  target_link_libraries(
    test_camera_batch_time_transfer
    camera_batch_time_transfer
    camera_time_transfer
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_photometric_calibration
    test/test_photometric_calibration.cpp
  )
  target_link_libraries(
    test_photometric_calibration
    photometric_calibration
  )
  add_library(
    trusted_photometric_calibration_loader_test_root
    src/trusted_photometric_calibration_loader.cpp
  )
  target_compile_definitions(
    trusted_photometric_calibration_loader_test_root
    PRIVATE
    LIVO_PHOTOMETRIC_TRUST_ROOT_PUBLIC_KEY_HEX="d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    LIVO_PHOTOMETRIC_TRUST_DOMAIN="livo-core-photometric-test-v1"
  )
  target_link_libraries(
    trusted_photometric_calibration_loader_test_root
    photometric_calibration
    photometric_noise_authority
    OpenSSL::Crypto
    JsonCpp::JsonCpp
  )
  ament_add_gtest(
    test_trusted_photometric_calibration_loader
    test/test_trusted_photometric_calibration_loader.cpp
  )
  target_compile_definitions(
    test_trusted_photometric_calibration_loader
    PRIVATE
    LIVO_TEST_EXPECT_TRUSTED_PHOTOMETRIC_ROOT=1
  )
  target_link_libraries(
    test_trusted_photometric_calibration_loader
    trusted_photometric_calibration_loader_test_root
  )
  ament_add_gtest(
    test_production_photometric_calibration_loader
    test/test_trusted_photometric_calibration_loader.cpp
  )
  target_link_libraries(
    test_production_photometric_calibration_loader
    trusted_photometric_calibration_loader
  )
  ament_add_gtest(
    test_current_frame_photometric_evidence
    test/test_current_frame_photometric_evidence.cpp
  )
  target_link_libraries(
    test_current_frame_photometric_evidence
    current_frame_photometric_evidence
    reference_only_migration_runtime_provenance
    trusted_photometric_calibration_loader_test_root
    camera_observation_batch_authority
    camera_batch_time_transfer
    camera_time_transfer
    OpenSSL::Crypto
    JsonCpp::JsonCpp
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_adaptive_visual_scheduler
    test/test_adaptive_visual_scheduler.cpp
  )
  target_link_libraries(
    test_adaptive_visual_scheduler
    adaptive_visual_policy
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_adaptive_visual_policy_plan_capability
    test/test_adaptive_visual_policy_plan_capability.cpp
  )
  target_link_libraries(
    test_adaptive_visual_policy_plan_capability
    adaptive_visual_policy
    reference_only_migration_runtime_provenance
    camera_observation_batch_authority
    camera_batch_time_transfer
    camera_time_transfer
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_adaptive_visual_selection_capability
    test/test_adaptive_visual_selection_capability.cpp
  )
  target_link_libraries(
    test_adaptive_visual_selection_capability
    adaptive_visual_policy
    adaptive_visual_selection_authority
    reference_only_migration_runtime_provenance
    camera_observation_batch_authority
    camera_batch_time_transfer
    camera_time_transfer
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_adaptive_visual_policy_reset_allocation_failure
    test/test_adaptive_visual_policy_reset_allocation_failure.cpp
  )
  target_link_libraries(
    test_adaptive_visual_policy_reset_allocation_failure
    adaptive_visual_policy_failure_injection_test
    adaptive_visual_selection_authority
  )
  ament_add_gtest(
    test_adaptive_visual_selection_receipt_allocation_failure
    test/test_adaptive_visual_selection_receipt_allocation_failure.cpp
  )
  target_link_libraries(
    test_adaptive_visual_selection_receipt_allocation_failure
    adaptive_visual_selection_authority_failure_injection_test
  )
  ament_add_gtest(
    test_multi_view_landmark
    test/test_multi_view_landmark.cpp
  )
  target_link_libraries(
    test_multi_view_landmark
    legacy_landmark_shadow_adapter
    lio
    multi_view_landmark
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_plane_lifecycle
    test/test_plane_lifecycle.cpp
  )
  ament_add_gtest(
    test_residual_correspondence
    test/test_residual_correspondence.cpp
  )
  target_link_libraries(test_residual_correspondence lio)
  target_link_libraries(
    test_plane_lifecycle
    lio
    plane_lifecycle
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_shared_landmark_tombstone_shadow
    test/test_shared_landmark_tombstone_shadow.cpp
  )
  target_link_libraries(
    test_shared_landmark_tombstone_shadow
    legacy_landmark_shadow_adapter
    lio
    shared_landmark_tombstone_shadow
    multi_view_landmark
    plane_lifecycle
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_shared_landmark_refit_shadow
    test/test_shared_landmark_refit_shadow.cpp
  )
  target_link_libraries(
    test_shared_landmark_refit_shadow
    shared_landmark_refit_shadow
    multi_view_landmark
    plane_lifecycle
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_legacy_landmark_shadow_adapter
    test/test_legacy_landmark_shadow_adapter.cpp
  )
  target_link_libraries(
    test_legacy_landmark_shadow_adapter
    legacy_landmark_shadow_adapter
    lio
    multi_view_landmark
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_legacy_landmark_observation_shadow_adapter
    test/test_legacy_landmark_observation_shadow_adapter.cpp
  )
  target_link_libraries(
    test_legacy_landmark_observation_shadow_adapter
    legacy_landmark_shadow_adapter
    legacy_landmark_observation_shadow_adapter
    camera_time_transfer
    lio
    multi_view_landmark
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_shared_landmark_shadow_session
    test/test_shared_landmark_shadow_session.cpp
  )
  target_link_libraries(
    test_shared_landmark_shadow_session
    shared_landmark_shadow_session
    lio
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_multi_camera_landmark_linearization_shadow
    test/test_multi_camera_landmark_linearization_shadow.cpp
  )
  target_link_libraries(
    test_multi_camera_landmark_linearization_shadow
    multi_camera_landmark_linearization_shadow
    trusted_photometric_calibration_loader_test_root
    lio
    OpenSSL::Crypto
    JsonCpp::JsonCpp
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_cross_camera_patch_migration
    test/test_cross_camera_patch_migration.cpp
  )
  target_link_libraries(
    test_cross_camera_patch_migration
    cross_camera_patch_migration
    pinhole_camera_projection_adapter
    photometric_calibration
    camera_time_transfer
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_reference_only_migration_shadow
    test/test_reference_only_migration_shadow.cpp
  )
  target_link_libraries(
    test_reference_only_migration_shadow
    reference_only_migration_shadow
    camera_batch_time_transfer
    camera_time_transfer
    pinhole_camera_projection_adapter
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_reference_only_migration_shadow_batch_runner
    test/test_reference_only_migration_shadow_batch_runner.cpp
  )
  target_link_libraries(
    test_reference_only_migration_shadow_batch_runner
    reference_only_migration_shadow_batch_runner
    camera_batch_time_transfer
    camera_time_transfer
    pinhole_camera_projection_adapter
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_pinhole_camera_projection_adapter
    test/test_pinhole_camera_projection_adapter.cpp
  )
  target_link_libraries(
    test_pinhole_camera_projection_adapter
    pinhole_camera_projection_adapter
    Eigen3::Eigen
  )
  ament_add_gtest(
    test_four_camera_pinhole_rig
    test/test_four_camera_pinhole_rig.cpp
  )
  target_link_libraries(
    test_four_camera_pinhole_rig
    four_camera_pinhole_rig
    Eigen3::Eigen
  )
  ament_add_gtest(test_vio_camera_registry test/test_vio_camera_registry.cpp)
  ament_target_dependencies(test_vio_camera_registry ${ROS_DEPS})
  target_link_libraries(test_vio_camera_registry
    vio
    lio
    camera_model_factory
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${Boost_LIBRARIES}
  )
  ament_add_gtest(
    test_reference_only_migration_shadow_runtime
    test/test_reference_only_migration_shadow_runtime.cpp
  )
  ament_target_dependencies(
    test_reference_only_migration_shadow_runtime
    ${ROS_DEPS}
  )
  target_link_libraries(
    test_reference_only_migration_shadow_runtime
    vio
    lio
    camera_model_factory
    reference_only_migration_shadow_batch_runner
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${Boost_LIBRARIES}
  )
  ament_add_gtest(
    test_vio_multi_camera_landmark_linearization_shadow
    test/test_vio_multi_camera_landmark_linearization_shadow.cpp
  )
  ament_target_dependencies(
    test_vio_multi_camera_landmark_linearization_shadow
    ${ROS_DEPS}
  )
  target_link_libraries(
    test_vio_multi_camera_landmark_linearization_shadow
    vio
    lio
    camera_model_factory
    trusted_photometric_calibration_loader_test_root
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${Boost_LIBRARIES}
  )
  ament_add_gtest(
    test_adaptive_vio_runtime_authority
    test/test_adaptive_vio_runtime_authority.cpp
  )
  ament_target_dependencies(
    test_adaptive_vio_runtime_authority
    ${ROS_DEPS}
  )
  target_link_libraries(
    test_adaptive_vio_runtime_authority
    vio
    lio
    camera_model_factory
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${Boost_LIBRARIES}
  )
  ament_add_gtest(
    test_cross_camera_migration_shadow_runtime
    test/test_cross_camera_migration_shadow_runtime.cpp
  )
  ament_target_dependencies(
    test_cross_camera_migration_shadow_runtime
    ${ROS_DEPS}
  )
  target_compile_definitions(
    test_cross_camera_migration_shadow_runtime
    PRIVATE
    LIVO_TEST_AIRY_VIO_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/airy_d435_vio_ablation_ros2.yaml"
  )
  target_link_libraries(
    test_cross_camera_migration_shadow_runtime
    laser_mapping
    camera_model_factory
    vio
    lio
    pre
    imu_proc
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${Boost_LIBRARIES}
  )
  ament_add_gtest(
    test_liv_mapper_adaptive_visual_runtime
    test/test_liv_mapper_adaptive_visual_runtime.cpp
  )
  ament_target_dependencies(
    test_liv_mapper_adaptive_visual_runtime
    ${ROS_DEPS}
  )
  target_compile_definitions(
    test_liv_mapper_adaptive_visual_runtime
    PRIVATE
    LIVO_TEST_AIRY_VIO_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/airy_d435_vio_ablation_ros2.yaml"
  )
  target_link_libraries(
    test_liv_mapper_adaptive_visual_runtime
    laser_mapping
    camera_model_factory
    vio
    lio
    pre
    imu_proc
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${Boost_LIBRARIES}
  )
  ament_add_pytest_test(test_body_cloud_contract test/test_body_cloud_contract.py)
  ament_add_pytest_test(test_multi_camera_vio_contract test/test_multi_camera_vio_contract.py)
  ament_add_pytest_test(
    test_sealed_receipt_compile_contract
    test/test_sealed_receipt_compile_contract.py
  )
  ament_add_pytest_test(
    test_landmark_authority_compile_contract
    test/test_landmark_authority_compile_contract.py
  )
  ament_add_pytest_test(
    test_camera_projection_adapter_compile_contract
    test/test_camera_projection_adapter_compile_contract.py
  )
  ament_add_pytest_test(
    test_reference_only_migration_shadow_compile_contract
    test/test_reference_only_migration_shadow_compile_contract.py
  )
  ament_add_pytest_test(
    test_reference_only_migration_shadow_batch_runner_compile_contract
    test/test_reference_only_migration_shadow_batch_runner_compile_contract.py
  )
  ament_add_pytest_test(
    test_reference_only_migration_shadow_runtime_compile_contract
    test/test_reference_only_migration_shadow_runtime_compile_contract.py
  )
  ament_add_pytest_test(
    test_reference_only_migration_runtime_provenance_compile_contract
    test/test_reference_only_migration_runtime_provenance_compile_contract.py
  )
  ament_add_pytest_test(
    test_adaptive_visual_policy_plan_compile_contract
    test/test_adaptive_visual_policy_plan_compile_contract.py
  )
  ament_add_pytest_test(
    test_adaptive_visual_selection_capability_compile_contract
    test/test_adaptive_visual_selection_capability_compile_contract.py
  )
  ament_add_pytest_test(
    test_adaptive_camera_snapshot_queue_compile_contract
    test/test_adaptive_camera_snapshot_queue_compile_contract.py
    ENV
    "LIVO_ADAPTIVE_SNAPSHOT_BUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
  )
  ament_add_pytest_test(
    test_exposure_timing_attestation_compile_contract
    test/test_exposure_timing_attestation_compile_contract.py
    ENV
    "LIVO_EXPOSURE_TIMING_BUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
  )
  ament_add_pytest_test(
    test_attested_frozen_camera_frame_compile_contract
    test/test_attested_frozen_camera_frame_compile_contract.py
    ENV
    "LIVO_ATTESTED_FRAME_BUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
  )
  ament_add_pytest_test(
    test_multi_camera_landmark_linearization_shadow_compile_contract
    test/test_multi_camera_landmark_linearization_shadow_compile_contract.py
  )
  ament_add_pytest_test(
    test_current_frame_photometric_evidence_compile_contract
    test/test_current_frame_photometric_evidence_compile_contract.py
  )
  ament_add_pytest_test(
    test_pinhole_camera_projection_config
    test/test_pinhole_camera_projection_config.py
  )
  ament_add_pytest_test(
    test_four_camera_pinhole_rig_yaml_contract
    test/test_four_camera_pinhole_rig_yaml_contract.py
  )
  ament_add_pytest_test(test_camera_fault_injector test/test_camera_fault_injector.py)
endif()

ament_export_dependencies(OpenSSL jsoncpp)
ament_package()
