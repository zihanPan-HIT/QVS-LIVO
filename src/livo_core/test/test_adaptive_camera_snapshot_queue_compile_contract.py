from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = Path(
    os.environ.get(
        "LIVO_ADAPTIVE_SNAPSHOT_BUILD_ROOT",
        PACKAGE_ROOT.parents[1] / "build" / PACKAGE_ROOT.name,
    )
)
HEADER = PACKAGE_ROOT / "include" / "adaptive_camera_snapshot_queue.h"
PRODUCTION_FLAGS = (
    BUILD_ROOT
    / "CMakeFiles"
    / "adaptive_camera_snapshot_queue.dir"
    / "flags.make"
)
FAILURE_FLAGS = (
    BUILD_ROOT
    / "CMakeFiles"
    / "adaptive_camera_snapshot_queue_failure_injection_test.dir"
    / "flags.make"
)
PRODUCTION_LIBRARY = (
    BUILD_ROOT / "libadaptive_camera_snapshot_queue.a"
)
FAILURE_LIBRARY = (
    BUILD_ROOT
    / "libadaptive_camera_snapshot_queue_failure_injection_test.a"
)
QUEUE_TEST = BUILD_ROOT / "test_adaptive_camera_snapshot_queue"


def cmake_flags(path: Path, variable: str) -> list[str]:
    prefix = f"{variable} = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return shlex.split(line.removeprefix(prefix))
    raise AssertionError(f"{variable} missing from {path}")


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("c++")
    assert compiler is not None
    assert PRODUCTION_FLAGS.is_file()
    with tempfile.TemporaryDirectory(
        prefix="livo-adaptive-snapshot-contract-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        source_path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                compiler,
                *cmake_flags(PRODUCTION_FLAGS, "CXX_DEFINES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_INCLUDES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_FLAGS"),
                "-fsyntax-only",
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def defined_symbols(binary: Path) -> str:
    nm = shutil.which("nm")
    assert nm is not None
    assert binary.is_file(), f"missing symbol input: {binary}"
    result = subprocess.run(
        [nm, "-C", "--defined-only", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_receipt_is_copyable_but_not_publicly_mintable() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_camera_snapshot_queue.h"
#include <memory>
#include <type_traits>
#include <utility>

using Queue = livo::AdaptiveCameraSnapshotQueue;
using Identity = livo::AdaptiveCameraSnapshotQueueIdentity;
using Receipt = livo::AdaptiveCameraSnapshotReceipt;
using Digest = livo::AdaptiveCameraSnapshotImageDigest;

static_assert(!std::is_default_constructible_v<Receipt>);
static_assert(std::is_copy_constructible_v<Receipt>);
static_assert(std::is_nothrow_move_constructible_v<Receipt>);
static_assert(std::is_nothrow_move_assignable_v<Receipt>);
static_assert(!std::is_copy_constructible_v<Queue>);
static_assert(!std::is_move_constructible_v<Queue>);
static_assert(std::is_default_constructible_v<Identity>);
static_assert(!std::is_constructible_v<
  Receipt, std::shared_ptr<const void>>);
static_assert(std::is_same_v<
  decltype(std::declval<const Receipt &>().imageDigest(0U)),
  Digest>);
static_assert(std::is_same_v<
  decltype(std::declval<Queue &>().takeFinalized()),
  std::optional<Receipt>>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_receipt_surface_has_no_pixel_or_frozen_handle_escape() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_camera_snapshot_queue.h"
#include <type_traits>
#include <utility>

using Receipt = livo::AdaptiveCameraSnapshotReceipt;

template<typename T, typename = void>
struct HasImage : std::false_type {};
template<typename T>
struct HasImage<
  T, std::void_t<decltype(std::declval<const T &>().image(0U))>>
: std::true_type {};

template<typename T, typename = void>
struct HasFrozenImage : std::false_type {};
template<typename T>
struct HasFrozenImage<
  T,
  std::void_t<decltype(
    std::declval<const T &>().frozenImage(0U))>>
: std::true_type {};

template<typename T, typename = void>
struct HasPixels : std::false_type {};
template<typename T>
struct HasPixels<
  T, std::void_t<decltype(std::declval<const T &>().pixels(0U))>>
: std::true_type {};

template<typename T, typename = void>
struct HasData : std::false_type {};
template<typename T>
struct HasData<
  T, std::void_t<decltype(std::declval<const T &>().data(0U))>>
: std::true_type {};

static_assert(!HasImage<Receipt>::value);
static_assert(!HasFrozenImage<Receipt>::value);
static_assert(!HasPixels<Receipt>::value);
static_assert(!HasData<Receipt>::value);
"""
    )
    assert result.returncode == 0, result.stderr

    header = HEADER.read_text(encoding="utf-8")
    receipt_start = header.index(
        "class AdaptiveCameraSnapshotReceipt final"
    )
    receipt_end = header.index(
        "class AdaptiveCameraSnapshotQueue final"
    )
    receipt_surface = header[receipt_start:receipt_end]
    assert "cv::Mat" not in receipt_surface
    assert "FrozenCameraImage" not in receipt_surface
    assert "TestPeer" not in header


def test_ordinary_consumer_cannot_construct_sealed_receipt() -> None:
    result = compile_translation_unit(
        """
#include "adaptive_camera_snapshot_queue.h"
#include <memory>
void forge() {
  using Sealed =
    livo::AdaptiveCameraSnapshotReceipt::SealedData;
  std::shared_ptr<const Sealed> sealed;
  livo::AdaptiveCameraSnapshotReceipt receipt(sealed);
}
"""
    )
    assert result.returncode != 0
    assert "private" in result.stderr


def test_failure_hooks_are_isolated_from_production() -> None:
    queue_hooks = (
        "failNextAdaptiveCameraSnapshotQueueFactoryAllocation",
        "failNextAdaptiveCameraSnapshotRequestStorageAllocation",
        "exhaustNextAdaptiveCameraSnapshotSequence",
        "exhaustNextAdaptiveCameraSnapshotObservationGeneration",
        "exhaustNextAdaptiveCameraSnapshotImageGeneration",
        "pauseNextAdaptiveCameraSnapshotPushAfterFreeze",
        "waitForAdaptiveCameraSnapshotPushPausedAfterFreeze",
        "releaseAdaptiveCameraSnapshotPushAfterFreeze",
        "pauseNextAdaptiveCameraSnapshotRequestBeforeRevalidation",
        "waitForAdaptiveCameraSnapshotRequestPausedBeforeRevalidation",
        "releaseAdaptiveCameraSnapshotRequestBeforeRevalidation",
    )
    frozen_hook = "failNextFrozenCameraImageStorageAllocation"

    production_symbols = defined_symbols(PRODUCTION_LIBRARY)
    failure_symbols = defined_symbols(FAILURE_LIBRARY)
    test_symbols = defined_symbols(QUEUE_TEST)
    for hook in queue_hooks:
        assert hook not in production_symbols
        assert hook in failure_symbols
        assert hook in test_symbols
    assert frozen_hook not in production_symbols
    assert frozen_hook in failure_symbols
    assert frozen_hook in test_symbols

    production_defines = cmake_flags(
        PRODUCTION_FLAGS, "CXX_DEFINES"
    )
    failure_defines = cmake_flags(FAILURE_FLAGS, "CXX_DEFINES")
    assert not any(
        "SNAPSHOT_QUEUE_FAILURE_INJECTION" in item
        for item in production_defines
    )
    assert not any(
        "FROZEN_CAMERA_IMAGE_FAILURE_INJECTION" in item
        for item in production_defines
    )
    assert any(
        "SNAPSHOT_QUEUE_FAILURE_INJECTION" in item
        for item in failure_defines
    )
    assert any(
        "FROZEN_CAMERA_IMAGE_FAILURE_INJECTION" in item
        for item in failure_defines
    )


def test_test_variant_does_not_colink_production_variants() -> None:
    link_file = (
        BUILD_ROOT
        / "CMakeFiles"
        / "test_adaptive_camera_snapshot_queue.dir"
        / "link.txt"
    )
    link = shlex.split(link_file.read_text(encoding="utf-8"))
    names = {Path(item).name for item in link}
    assert (
        "libadaptive_camera_snapshot_queue_failure_injection_test.a"
        in names
    )
    assert "libadaptive_camera_snapshot_queue.a" not in names
    assert "libfrozen_camera_image.a" not in names
    assert "libfrozen_camera_image_failure_injection_test.a" not in names


def test_standalone_queue_is_not_linked_into_runtime_or_selection() -> None:
    target_names = (
        "livo_mapping",
        "adaptive_visual_selection_authority",
        "test_adaptive_visual_selection_capability",
    )
    forbidden = (
        "libadaptive_camera_snapshot_queue.a",
        "libadaptive_camera_snapshot_queue_failure_injection_test.a",
    )
    for target in target_names:
        link_file = (
            BUILD_ROOT / "CMakeFiles" / f"{target}.dir" / "link.txt"
        )
        assert link_file.is_file(), f"missing link graph: {link_file}"
        link = link_file.read_text(encoding="utf-8")
        for library in forbidden:
            assert library not in link
