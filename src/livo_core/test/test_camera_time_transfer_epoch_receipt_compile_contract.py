import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from unittest import mock


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORY = PACKAGE_ROOT / "include"
TEST_DIRECTORY = PACKAGE_ROOT / "test"
BUILD_ROOT_ENVIRONMENT_VARIABLE = (
    "LIVO_CAMERA_TIME_TRANSFER_BUILD_ROOT"
)
BUILD_ROOT_VALUE = os.environ.get(
    BUILD_ROOT_ENVIRONMENT_VARIABLE
)
BUILD_ROOT = (
    Path(BUILD_ROOT_VALUE.strip()).resolve()
    if BUILD_ROOT_VALUE is not None and BUILD_ROOT_VALUE.strip()
    else None
)
CMAKE_CACHE = (
    BUILD_ROOT / "CMakeCache.txt"
    if BUILD_ROOT is not None
    else None
)
PRODUCTION_TARGET = "camera_time_transfer"
FAILURE_TARGET = "camera_time_transfer_failure_injection_test"
PRODUCTION_ARCHIVE = (
    BUILD_ROOT / f"lib{PRODUCTION_TARGET}.a"
    if BUILD_ROOT is not None
    else None
)
FAILURE_ARCHIVE = (
    BUILD_ROOT / f"lib{FAILURE_TARGET}.a"
    if BUILD_ROOT is not None
    else None
)


def require_build_root() -> Path:
    assert BUILD_ROOT is not None, (
        f"{BUILD_ROOT_ENVIRONMENT_VARIABLE} must name the exact "
        "configured livo_core build root"
    )
    return BUILD_ROOT


def cmake_cache_value(variable: str) -> str:
    assert CMAKE_CACHE is not None
    assert CMAKE_CACHE.is_file(), f"missing cache: {CMAKE_CACHE}"
    prefix = f"{variable}:"
    for line in CMAKE_CACHE.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            _, separator, value = line.partition("=")
            assert separator
            return value
    raise AssertionError(f"{variable} missing from {CMAKE_CACHE}")


def optional_cmake_cache_value(variable: str) -> str:
    assert CMAKE_CACHE is not None
    prefix = f"{variable}:"
    for line in CMAKE_CACHE.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.partition("=")[2]
    return ""


def make_flags(target: str, variable: str) -> list[str]:
    build_root = require_build_root()
    path = (
        build_root / "CMakeFiles" / f"{target}.dir" / "flags.make"
    )
    assert path.is_file(), f"missing target flags: {path}"
    assert path.stat().st_size > 0, f"empty target flags: {path}"
    prefix = f"{variable} = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return shlex.split(line.removeprefix(prefix))
    raise AssertionError(f"{variable} missing from {path}")


def generator_name() -> str:
    name = cmake_cache_value("CMAKE_GENERATOR")
    assert name
    return name


def ninja_target_commands(target: str) -> list[str]:
    build_root = require_build_root()
    ninja = shutil.which("ninja")
    assert ninja is not None, (
        "CMAKE_GENERATOR=Ninja requires the ninja executable "
        "for compile-contract inspection"
    )
    result = subprocess.run(
        [ninja, "-C", str(build_root), "-t", "commands", target],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [
        line for line in result.stdout.splitlines() if line.strip()
    ]
    assert commands, f"empty Ninja command graph for {target}"
    return commands


def all_ninja_commands() -> list[str]:
    build_root = require_build_root()
    ninja = shutil.which("ninja")
    assert ninja is not None, (
        "CMAKE_GENERATOR=Ninja requires the ninja executable "
        "for full link-graph inspection"
    )
    result = subprocess.run(
        [ninja, "-C", str(build_root), "-t", "commands"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = [
        line for line in result.stdout.splitlines() if line.strip()
    ]
    assert commands, "empty full Ninja command graph"
    return commands


def consumer_flags() -> list[str]:
    if generator_name() == "Unix Makefiles":
        flags = [
            *make_flags(PRODUCTION_TARGET, "CXX_DEFINES"),
            *make_flags(PRODUCTION_TARGET, "CXX_INCLUDES"),
            *make_flags(PRODUCTION_TARGET, "CXX_FLAGS"),
            f"-I{TEST_DIRECTORY}",
        ]
        return [
            flag
            for flag in flags
            if not re.fullmatch(
                r"-O(?:0|1|2|3|g|s|fast)", flag
            )
        ]
    if generator_name() == "Ninja":
        build_type = cmake_cache_value("CMAKE_BUILD_TYPE").upper()
        flags = shlex.split(cmake_cache_value("CMAKE_CXX_FLAGS"))
        if build_type:
            flags.extend(
                shlex.split(
                    cmake_cache_value(
                        f"CMAKE_CXX_FLAGS_{build_type}"
                    )
                )
            )
        return [
            *flags,
            "-std=c++17",
            f"-I{INCLUDE_DIRECTORY}",
            f"-I{TEST_DIRECTORY}",
            "-isystem",
            "/usr/include/eigen3",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
        ]
    raise AssertionError(
        "unsupported CMake generator for camera-time-transfer "
        f"compile contract: {generator_name()}"
    )


def compiler_command() -> list[str]:
    compiler = cmake_cache_value("CMAKE_CXX_COMPILER")
    resolved = shutil.which(compiler)
    assert resolved is not None, f"C++ compiler not found: {compiler}"
    command = [resolved]
    argument = optional_cmake_cache_value("CMAKE_CXX_COMPILER_ARG1")
    if argument:
        command.extend(shlex.split(argument))
    return command


def compile_source(
    source: str,
    *,
    link_archive: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(
        prefix="livo-camera-epoch-receipt-contract-"
    ) as directory:
        root = Path(directory)
        source_path = root / "consumer.cpp"
        binary_path = root / "consumer"
        source_path.write_text(source, encoding="utf-8")
        command = [
            *compiler_command(),
            *consumer_flags(),
            str(source_path),
        ]
        if link_archive is None:
            command.append("-fsyntax-only")
        else:
            assert link_archive.is_file()
            assert link_archive.stat().st_size > 0
            command.extend(
                [str(link_archive), "-pthread", "-o", str(binary_path)]
            )
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )


def relocatable_link_source(
    source: str,
    archive: Path,
) -> tuple[
    subprocess.CompletedProcess[str],
    subprocess.CompletedProcess[str] | None,
]:
    linker = shutil.which("ld")
    archiver = shutil.which("ar")
    objcopy = shutil.which("objcopy")
    assert linker is not None
    assert archiver is not None
    assert objcopy is not None
    assert archive.is_file()
    with tempfile.TemporaryDirectory(
        prefix="livo-camera-epoch-receipt-override-"
    ) as directory:
        root = Path(directory)
        source_path = root / "consumer.cpp"
        object_path = root / "consumer.o"
        extracted_path = root / "archive-member.o"
        stripped_path = root / "archive-member-stripped.o"
        stripped_archive = root / "security-archive-stripped.a"
        combined_path = root / "combined.o"
        source_path.write_text(source, encoding="utf-8")
        listed = subprocess.run(
            [archiver, "t", str(archive)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert listed.returncode == 0, listed.stderr
        members = [
            line for line in listed.stdout.splitlines() if line.strip()
        ]
        assert len(members) == 1, (
            f"security archive must have exactly one member: "
            f"{archive}: {members}"
        )
        with extracted_path.open("wb") as extracted:
            extracted_result = subprocess.run(
                [archiver, "p", str(archive), members[0]],
                check=False,
                stdout=extracted,
                stderr=subprocess.PIPE,
            )
        assert extracted_result.returncode == 0, (
            extracted_result.stderr.decode(
                "utf-8", errors="replace"
            )
        )
        assert extracted_path.stat().st_size > 0
        stripped = subprocess.run(
            [
                objcopy,
                "--strip-debug",
                str(extracted_path),
                str(stripped_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert stripped.returncode == 0, stripped.stderr
        assert stripped_path.stat().st_size > 0
        repacked = subprocess.run(
            [
                archiver,
                "rcs",
                str(stripped_archive),
                str(stripped_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert repacked.returncode == 0, repacked.stderr
        assert stripped_archive.stat().st_size > 0
        compiled = subprocess.run(
            [
                *compiler_command(),
                *consumer_flags(),
                "-g0",
                "-O0",
                "-c",
                str(source_path),
                "-o",
                str(object_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if compiled.returncode != 0:
            return compiled, None
        linked = subprocess.run(
            [
                linker,
                "-r",
                str(object_path),
                str(stripped_archive),
                "-o",
                str(combined_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return compiled, linked


def global_defined_records(
    archive: Path,
) -> set[tuple[str, str, str]]:
    nm = shutil.which("nm")
    assert nm is not None
    result = subprocess.run(
        [nm, "-g", "-C", "--defined-only", str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    records: set[tuple[str, str, str]] = set()
    for line in result.stdout.splitlines():
        match = re.match(
            r"^\s*([0-9a-fA-F]+)\s+([A-Za-z])\s+(.+)$", line
        )
        if match is not None:
            records.add(
                (match.group(1), match.group(2), match.group(3))
            )
    assert records, f"no global definitions in {archive}"
    return records


def global_undefined_names(archive: Path) -> set[str]:
    nm = shutil.which("nm")
    assert nm is not None
    result = subprocess.run(
        [nm, "-g", "-C", "--undefined-only", str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    names: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*U\s+(.+)$", line)
        if match is not None:
            names.add(match.group(1))
    return names


def assert_exact_strong_definitions(
    records: set[tuple[str, str, str]],
    required: tuple[str, ...],
    archive: Path,
) -> None:
    prefixes = (
        "livo::CameraTimeTransfer::",
        "livo::CameraTimeTransferAuthorityIdentity::",
        "livo::CameraTimeTransferEpochReceipt::",
        "livo::EpochBoundCameraPoseTransfer::",
    )
    actual_names = {
        record[2]
        for record in records
        if record[1] == "T" and record[2].startswith(prefixes)
    }
    assert actual_names == set(required), (
        f"out-of-line security member inventory mismatch in {archive}: "
        f"missing={set(required) - actual_names}, "
        f"unexpected={actual_names - set(required)}"
    )
    for name in required:
        matches = [
            record
            for record in records
            if record[1] == "T" and record[2] == name
        ]
        assert len(matches) == 1, (
            f"expected exactly one strong definition of {name} "
            f"in {archive}, found {matches}"
        )


def add_library_arguments(target: str) -> list[str]:
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(
        encoding="utf-8"
    )
    declaration = re.search(
        rf"add_library\s*\(\s*{re.escape(target)}\b([^)]*)\)",
        cmake,
        re.DOTALL,
    )
    assert declaration is not None, f"missing add_library({target})"
    return declaration.group(1).split()


def build_graph_commands() -> list[str]:
    build_root = require_build_root()
    generator = generator_name()
    if generator == "Unix Makefiles":
        link_files = tuple(build_root.rglob("link.txt"))
        assert link_files, f"no Makefile link graph under {build_root}"
        commands = []
        for path in link_files:
            assert path.stat().st_size > 0, f"empty link graph: {path}"
            commands.extend(
                line
                for line in path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        assert commands
        return commands
    if generator == "Ninja":
        build_ninja = build_root / "build.ninja"
        assert build_ninja.is_file(), f"missing {build_ninja}"
        assert build_ninja.stat().st_size > 0, f"empty {build_ninja}"
        return all_ninja_commands()
    raise AssertionError(
        "unsupported CMake generator for camera-time-transfer "
        f"compile contract: {generator}"
    )


def validate_security_link_commands(
    commands: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    production = f"lib{PRODUCTION_TARGET}.a"
    failure = f"lib{FAILURE_TARGET}.a"
    nonempty = tuple(command for command in commands if command.strip())
    assert nonempty, "empty link command graph"
    for command in nonempty:
        assert not (
            production in command and failure in command
        ), f"mixed trust variants in link graph: {command}"
    downstream = tuple(
        command
        for command in nonempty
        if (
            (production in command or failure in command)
            and re.search(r"(?:^|\s)-o(?:\s|$)", command)
            and not re.search(r"(?:^|\s)-c(?:\s|$)", command)
        )
    )
    assert downstream, (
        "no real downstream link consumer references a "
        "camera-time-transfer security archive"
    )
    return downstream


def test_build_root_is_explicit_configured_and_nonempty() -> None:
    assert_blank_build_root_environment_never_resolves_to_cwd()
    assert_link_graph_checker_rejects_a_mixed_downstream_command()
    assert_ninja_full_graph_query_is_unscoped()
    build_root = require_build_root()
    assert CMAKE_CACHE is not None
    assert PRODUCTION_ARCHIVE is not None
    assert FAILURE_ARCHIVE is not None
    assert build_root.is_dir(), f"missing build root: {build_root}"
    assert CMAKE_CACHE.stat().st_size > 0
    assert (
        Path(cmake_cache_value("CMAKE_HOME_DIRECTORY")).resolve()
        == PACKAGE_ROOT.resolve()
    )
    assert cmake_cache_value("BUILD_TESTING").upper() in ("ON", "TRUE", "1")
    assert generator_name() in ("Unix Makefiles", "Ninja")
    for archive in (PRODUCTION_ARCHIVE, FAILURE_ARCHIVE):
        assert archive.is_file(), f"missing archive: {archive}"
        assert archive.stat().st_size > 0, f"empty archive: {archive}"

    required_flags = ("-Wall", "-Wextra", "-Wpedantic", "-Werror")
    if generator_name() == "Unix Makefiles":
        production_flags = make_flags(
            PRODUCTION_TARGET, "CXX_FLAGS"
        )
        failure_flags = make_flags(FAILURE_TARGET, "CXX_FLAGS")
        production_defines = make_flags(
            PRODUCTION_TARGET, "CXX_DEFINES"
        )
        failure_defines = make_flags(
            FAILURE_TARGET, "CXX_DEFINES"
        )
        for flags in (production_flags, failure_flags):
            for required in required_flags:
                assert required in flags
        assert not any(
            "LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING"
            in define
            for define in production_defines
        )
        assert any(
            "LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING"
            in define
            for define in failure_defines
        )
    else:
        production_commands = "\n".join(
            ninja_target_commands(PRODUCTION_TARGET)
        )
        failure_commands = "\n".join(
            ninja_target_commands(FAILURE_TARGET)
        )
        for commands in (production_commands, failure_commands):
            for required in required_flags:
                assert required in commands
        macro = (
            "LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING"
        )
        assert macro not in production_commands
        assert macro in failure_commands

    graph = build_graph_commands()
    downstream = validate_security_link_commands(graph)
    assert any(
        PRODUCTION_ARCHIVE.name in command
        for command in downstream
    )
    assert any(
        FAILURE_ARCHIVE.name in command
        for command in downstream
    )


def assert_blank_build_root_environment_never_resolves_to_cwd() -> None:
    environment = os.environ.copy()
    environment[BUILD_ROOT_ENVIRONMENT_VARIABLE] = ""
    probe = subprocess.run(
        [
            shutil.which("python3") or "python3",
            "-c",
            (
                "import importlib.util\n"
                f"p={str(Path(__file__).resolve())!r}\n"
                "s=importlib.util.spec_from_file_location('probe', p)\n"
                "m=importlib.util.module_from_spec(s)\n"
                "s.loader.exec_module(m)\n"
                "assert m.BUILD_ROOT is None\n"
                "try:\n"
                "  m.require_build_root()\n"
                "except AssertionError:\n"
                "  pass\n"
                "else:\n"
                "  raise AssertionError('blank root was accepted')\n"
            ),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def assert_link_graph_checker_rejects_a_mixed_downstream_command() -> None:
    production_only = (
        "c++ consumer.o libcamera_time_transfer.a -o consumer",
    )
    downstream = validate_security_link_commands(production_only)
    assert downstream == production_only

    mixed = (
        "c++ consumer.o libcamera_time_transfer.a "
        "libcamera_time_transfer_failure_injection_test.a "
        "-o compromised",
    )
    try:
        validate_security_link_commands(mixed)
    except AssertionError as error:
        assert "mixed trust variants" in str(error)
    else:
        raise AssertionError("mixed security archives escaped detection")


def assert_ninja_full_graph_query_is_unscoped() -> None:
    fake_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=(
            "c++ consumer.o libcamera_time_transfer.a "
            "-o consumer\n"
        ),
        stderr="",
    )
    with (
        mock.patch("shutil.which", return_value="/usr/bin/ninja"),
        mock.patch("subprocess.run", return_value=fake_result)
        as run,
    ):
        commands = all_ninja_commands()
    assert commands == [
        "c++ consumer.o libcamera_time_transfer.a -o consumer"
    ]
    run.assert_called_once_with(
        [
            "/usr/bin/ninja",
            "-C",
            str(require_build_root()),
            "-t",
            "commands",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_epoch_capability_types_are_opaque_and_locking_is_honest() -> None:
    result = compile_source(
        """
#include "camera_time_transfer.h"
#include <type_traits>
#include <utility>
#include <vector>

using Authority = livo::CameraTimeTransfer;
using Identity = livo::CameraTimeTransferAuthorityIdentity;
using Receipt = livo::CameraTimeTransferEpochReceipt;
using Bound = livo::EpochBoundCameraPoseTransfer;

static_assert(!std::is_default_constructible_v<Identity>);
static_assert(!std::is_default_constructible_v<Receipt>);
static_assert(!std::is_default_constructible_v<Bound>);
static_assert(std::is_copy_constructible_v<Identity>);
static_assert(std::is_copy_constructible_v<Receipt>);
static_assert(!std::is_copy_constructible_v<Authority>);
static_assert(!std::is_copy_assignable_v<Authority>);
static_assert(std::is_nothrow_move_constructible_v<Authority>);
static_assert(std::is_nothrow_move_assignable_v<Authority>);
static_assert(!std::is_constructible_v<
  Bound, livo::CameraPoseTransfer>);
static_assert(!std::is_convertible_v<
  livo::CameraPoseTransfer, Bound>);
static_assert(!std::is_constructible_v<
  Bound, livo::CameraPoseTransferData>);
static_assert(!std::is_convertible_v<
  livo::CameraPoseTransferData, Bound>);

static_assert(!noexcept(std::declval<const Authority &>().configValid()));
static_assert(!noexcept(std::declval<Authority &>().appendImu(
  std::declval<const livo::CameraTimeTransferImuSample &>())));
static_assert(!noexcept(std::declval<Authority &>().clear()));
static_assert(!noexcept(std::declval<const Authority &>().epoch()));
static_assert(!noexcept(
  std::declval<const Authority &>().bufferedSamples()));
static_assert(!noexcept(
  std::declval<const Authority &>().authorityIdentity()));
static_assert(!noexcept(std::declval<Authority &>().captureEpoch()));
static_assert(!noexcept(std::declval<Authority &>().captureWindow(
  std::declval<double>(), std::declval<double>())));
static_assert(!noexcept(std::declval<const Authority &>().transfer(
  std::declval<const livo::CameraTimeTransferState &>(),
  std::declval<double>(), std::declval<double>())));
static_assert(!noexcept(
  std::declval<const Authority &>().transferWithSamples(
    std::declval<const livo::CameraTimeTransferState &>(),
    std::declval<const std::vector<
      livo::CameraTimeTransferImuSample> &>(),
    std::declval<double>(), std::declval<double>())));

static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kValid) == 0);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kInvalidConfiguration) == 1);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kInvalidInput) == 2);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kTargetOutOfRange) == 3);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kImuNotStrictlyOrdered) == 4);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kImuCoverageMissing) == 5);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kImuGapTooLarge) == 6);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kMissingLeftBracket) == 7);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kMissingRightBracket) == 8);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kCrossesImuEpoch) == 9);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kInvalidNumericalResult) == 10);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kIllConditionedTransition) == 11);
static_assert(static_cast<int>(
  livo::CameraTimeTransferAppendStatus::kAccepted) == 0);
static_assert(static_cast<int>(
  livo::CameraTimeTransferAppendStatus::kNewEpoch) == 1);
static_assert(static_cast<int>(
  livo::CameraTimeTransferAppendStatus::kInvalidSample) == 2);
static_assert(static_cast<int>(
  livo::CameraTimeTransferStatus::kResourceFailure) >
  static_cast<int>(
    livo::CameraTimeTransferStatus::kIllConditionedTransition));
"""
    )
    assert result.returncode == 0, result.stderr


def test_ordinary_consumer_cannot_call_private_epoch_constructors() -> None:
    result = compile_source(
        """
#include "camera_time_transfer.h"
#include <memory>
int main() {
  livo::CameraTimeTransferEpochReceipt receipt(
    std::shared_ptr<const livo::CameraTimeTransferEpochReceipt::Data>{});
  return receipt.valid() ? 0 : 1;
}
"""
    )
    assert result.returncode != 0
    assert "private" in result.stderr


def test_security_members_have_exactly_one_strong_definition() -> None:
    required = (
        (
            "livo::CameraTimeTransfer::CameraTimeTransfer("
            "livo::CameraTimeTransferConfig)"
        ),
        (
            "livo::CameraTimeTransfer::CameraTimeTransfer("
            "livo::CameraTimeTransfer&&)"
        ),
        "livo::CameraTimeTransfer::~CameraTimeTransfer()",
        (
            "livo::CameraTimeTransfer::operator=("
            "livo::CameraTimeTransfer&&)"
        ),
        "livo::CameraTimeTransfer::configValid() const",
        (
            "livo::CameraTimeTransfer::appendImu("
            "livo::CameraTimeTransferImuSample const&)"
        ),
        "livo::CameraTimeTransfer::clear()",
        "livo::CameraTimeTransfer::epoch() const",
        "livo::CameraTimeTransfer::bufferedSamples() const",
        "livo::CameraTimeTransfer::captureEpoch()",
        "livo::CameraTimeTransfer::captureWindow(double, double)",
        "livo::CameraTimeTransfer::authorityIdentity() const",
        (
            "livo::CameraTimeTransfer::transfer("
            "livo::CameraTimeTransferState const&, double, double) const"
        ),
        (
            "livo::CameraTimeTransfer::transferWithSamples("
            "livo::CameraTimeTransferState const&, "
            "std::vector<livo::CameraTimeTransferImuSample, "
            "std::allocator<livo::CameraTimeTransferImuSample> > "
            "const&, double, double) const"
        ),
        (
            "livo::CameraTimeTransferAuthorityIdentity::"
            "CameraTimeTransferAuthorityIdentity("
            "std::shared_ptr<"
            "livo::CameraTimeTransferAuthorityIdentity::Data const>)"
        ),
        "livo::CameraTimeTransferAuthorityIdentity::valid() const",
        (
            "livo::CameraTimeTransferEpochReceipt::"
            "CameraTimeTransferEpochReceipt(std::shared_ptr<"
            "livo::CameraTimeTransferEpochReceipt::Data const>)"
        ),
        "livo::CameraTimeTransferEpochReceipt::valid() const",
        (
            "livo::CameraTimeTransferEpochReceipt::belongsTo("
            "livo::CameraTimeTransferAuthorityIdentity const&) const"
        ),
        "livo::CameraTimeTransferEpochReceipt::epoch() const",
        (
            "livo::CameraTimeTransferEpochReceipt::"
            "streamRevision() const"
        ),
        (
            "livo::CameraTimeTransferEpochReceipt::"
            "snapshotSequence() const"
        ),
        (
            "livo::CameraTimeTransferEpochReceipt::"
            "bufferedSamples() const"
        ),
        (
            "livo::CameraTimeTransferEpochReceipt::transfer("
            "livo::CameraTimeTransferState const&, double, double) const"
        ),
        (
            "livo::EpochBoundCameraPoseTransfer::"
            "EpochBoundCameraPoseTransfer("
            "livo::CameraPoseTransfer, std::shared_ptr<void const>)"
        ),
        "livo::EpochBoundCameraPoseTransfer::valid() const",
        (
            "livo::EpochBoundCameraPoseTransfer::wasProducedBy("
            "livo::CameraTimeTransferEpochReceipt const&) const"
        ),
        "livo::EpochBoundCameraPoseTransfer::value() const",
        "livo::EpochBoundCameraPoseTransfer::raw() const",
    )
    for archive in (PRODUCTION_ARCHIVE, FAILURE_ARCHIVE):
        records = global_defined_records(archive)
        assert_exact_strong_definitions(records, required, archive)


def test_failure_hooks_are_exact_and_isolated_from_production() -> None:
    hooks = (
        "livo::test::failNextCameraTimeTransferFactoryAllocation()",
        "livo::test::failNextCameraTimeTransferAppendAllocation()",
        "livo::test::failNextCameraTimeTransferClearResource()",
        "livo::test::failNextCameraTimeTransferCaptureAllocation()",
        "livo::test::failNextCameraTimeTransferNumericalAllocation()",
        "livo::test::setNextCameraTimeTransferEpochToMaximum()",
        "livo::test::setNextCameraTimeTransferEpochNearMaximum()",
        "livo::test::setNextCameraTimeTransferRevisionToMaximum()",
        "livo::test::setNextCameraTimeTransferRevisionNearMaximum()",
        (
            "livo::test::"
            "setNextCameraTimeTransferSnapshotSequenceToMaximum()"
        ),
        (
            "livo::test::"
            "setNextCameraTimeTransferSnapshotSequenceNearMaximum()"
        ),
    )
    production_defined = global_defined_records(PRODUCTION_ARCHIVE)
    production_undefined = global_undefined_names(PRODUCTION_ARCHIVE)
    failure_defined = global_defined_records(FAILURE_ARCHIVE)
    for hook in hooks:
        assert not any(
            record[2] == hook for record in production_defined
        )
        assert hook not in production_undefined
        matches = [
            record
            for record in failure_defined
            if record[1] == "T" and record[2] == hook
        ]
        assert len(matches) == 1, (
            f"expected exactly one failure-hook definition "
            f"for {hook}: {matches}"
        )

    source = """
#include "camera_time_transfer.h"
#include "camera_time_transfer_failure_injection_test_support.h"
int main() {
  livo::test::failNextCameraTimeTransferFactoryAllocation();
  livo::test::failNextCameraTimeTransferAppendAllocation();
  livo::test::failNextCameraTimeTransferClearResource();
  livo::test::failNextCameraTimeTransferCaptureAllocation();
  livo::test::failNextCameraTimeTransferNumericalAllocation();
  livo::test::setNextCameraTimeTransferEpochToMaximum();
  livo::test::setNextCameraTimeTransferEpochNearMaximum();
  livo::test::setNextCameraTimeTransferRevisionToMaximum();
  livo::test::setNextCameraTimeTransferRevisionNearMaximum();
  livo::test::setNextCameraTimeTransferSnapshotSequenceToMaximum();
  livo::test::setNextCameraTimeTransferSnapshotSequenceNearMaximum();
  return 0;
}
"""
    production = compile_source(source, link_archive=PRODUCTION_ARCHIVE)
    assert production.returncode != 0, (
        "production archive unexpectedly exported failure hooks"
    )
    failure = compile_source(source, link_archive=FAILURE_ARCHIVE)
    assert failure.returncode == 0, failure.stderr


def test_consumer_cannot_override_identity_or_receipt_minting() -> None:
    malicious_consumer = """
#include "camera_time_transfer.h"
#include <memory>
#include <optional>

namespace livo
{

struct CameraTimeTransferAuthorityIdentity::Data final {};
struct CameraTimeTransferEpochReceipt::Data final {};

std::optional<CameraTimeTransferAuthorityIdentity>
CameraTimeTransfer::authorityIdentity() const
{
  auto data =
    std::make_shared<const CameraTimeTransferAuthorityIdentity::Data>();
  return CameraTimeTransferAuthorityIdentity(std::move(data));
}

CameraTimeTransferEpochCaptureResult
CameraTimeTransfer::captureEpoch()
{
  auto data =
    std::make_shared<const CameraTimeTransferEpochReceipt::Data>();
  CameraTimeTransferEpochCaptureResult result;
  result.status = CameraTimeTransferEpochCaptureStatus::kCaptured;
  result.receipt =
    CameraTimeTransferEpochReceipt(std::move(data));
  return result;
}

}  // namespace livo

int main()
{
  (void)livo::cameraTimeTransferStatusName(
    livo::CameraTimeTransferStatus::kValid);
  livo::CameraTimeTransfer authority;
  return authority.captureEpoch().receipt.has_value() ? 0 : 1;
}
    """
    for archive in (PRODUCTION_ARCHIVE, FAILURE_ARCHIVE):
        compiled, linked = relocatable_link_source(
            malicious_consumer, archive
        )
        assert compiled.returncode == 0, compiled.stderr
        assert linked is not None
        assert linked.returncode != 0, (
            f"consumer mint override linked against {archive}"
        )
        diagnostic = linked.stderr.lower()
        assert "multiple definition" in diagnostic, linked.stderr
        assert "captureepoch" in diagnostic, linked.stderr

    append_side_effect_consumer = """
#include "camera_time_transfer.h"
#include <memory>
#include <optional>

namespace attack
{

std::optional<livo::CameraTimeTransferEpochReceipt> escaped_receipt;

}  // namespace attack

namespace livo
{

struct CameraTimeTransferEpochReceipt::Data final {};

CameraTimeTransferAppendStatus CameraTimeTransfer::appendImu(
  const CameraTimeTransferImuSample & sample)
{
  (void)sample;
  auto data =
    std::make_shared<const CameraTimeTransferEpochReceipt::Data>();
  attack::escaped_receipt =
    CameraTimeTransferEpochReceipt(std::move(data));
  return CameraTimeTransferAppendStatus::kAccepted;
}

}  // namespace livo

int main()
{
  (void)livo::cameraTimeTransferStatusName(
    livo::CameraTimeTransferStatus::kValid);
  livo::CameraTimeTransfer authority;
  (void)authority.appendImu(
    livo::CameraTimeTransferImuSample{});
  return attack::escaped_receipt.has_value() ? 0 : 1;
}
    """
    for archive in (PRODUCTION_ARCHIVE, FAILURE_ARCHIVE):
        compiled, linked = relocatable_link_source(
            append_side_effect_consumer, archive
        )
        assert compiled.returncode == 0, compiled.stderr
        assert linked is not None
        assert linked.returncode != 0, (
            f"consumer append override linked against {archive}"
        )
        diagnostic = linked.stderr.lower()
        assert "multiple definition" in diagnostic, linked.stderr
        assert "appendimu" in diagnostic, linked.stderr


def test_consumer_cannot_override_receipt_to_bound_sealing() -> None:
    malicious_consumer = """
#include "camera_time_transfer.h"
#include <memory>
#include <vector>

namespace livo
{

struct CameraTimeTransferEpochReceipt::Data final {};

CameraTimeTransferEpochTransferResult
CameraTimeTransferEpochReceipt::transfer(
  const CameraTimeTransferState & anchor,
  const double target_timestamp_sec,
  const double target_timestamp_stddev_sec) const
{
  CameraTimeTransfer numerical_authority;
  const auto raw = numerical_authority.transferWithSamples(
    anchor,
    std::vector<CameraTimeTransferImuSample>{},
    target_timestamp_sec,
    target_timestamp_stddev_sec);
  const auto sealed = raw.validatedValue();
  CameraTimeTransferEpochTransferResult result;
  result.status = sealed.has_value() ?
    CameraTimeTransferStatus::kValid : raw.status;
  if (sealed.has_value())
  {
    result.value = EpochBoundCameraPoseTransfer(
      *sealed, std::shared_ptr<const void>(data_));
  }
  return result;
}

}  // namespace livo

int main()
{
  (void)livo::cameraTimeTransferStatusName(
    livo::CameraTimeTransferStatus::kValid);
  livo::CameraTimeTransfer authority;
  return authority.configValid() ? 0 : 1;
}
    """
    for archive in (PRODUCTION_ARCHIVE, FAILURE_ARCHIVE):
        compiled, linked = relocatable_link_source(
            malicious_consumer, archive
        )
        assert compiled.returncode == 0, compiled.stderr
        assert linked is not None
        assert linked.returncode != 0, (
            f"consumer receipt sealing override linked against {archive}"
        )
        diagnostic = linked.stderr.lower()
        assert "multiple definition" in diagnostic, linked.stderr
        assert "cameratimetransferepochreceipt" in diagnostic, (
            linked.stderr
        )


def test_header_token_is_identical_for_production_and_failure_builds() -> None:
    header = (
        INCLUDE_DIRECTORY / "camera_time_transfer.h"
    ).read_text(encoding="utf-8")
    assert "LIVO_CAMERA_TIME_TRANSFER_FAILURE" not in header
    source = """
#include "camera_time_transfer.h"
#include <type_traits>
static_assert(!std::is_copy_constructible_v<
  livo::CameraTimeTransfer>);
static_assert(!std::is_convertible_v<
  livo::CameraPoseTransfer,
  livo::EpochBoundCameraPoseTransfer>);
static_assert(!std::is_convertible_v<
  livo::CameraPoseTransferData,
  livo::EpochBoundCameraPoseTransfer>);
"""
    production = compile_source(source)
    assert production.returncode == 0, production.stderr
    with_macro = compile_source(
        "#define "
        "LIVO_CAMERA_TIME_TRANSFER_FAILURE_INJECTION_TESTING 1\n"
        + source
    )
    assert with_macro.returncode == 0, with_macro.stderr


def test_security_archives_remain_static_with_build_shared_libs_on(
) -> None:
    targets = (PRODUCTION_TARGET, FAILURE_TARGET)
    for target in targets:
        arguments = add_library_arguments(target)
        assert arguments[0] == "STATIC", (
            f"{target} must explicitly select STATIC: {arguments}"
        )

    cmake = shutil.which("cmake")
    assert cmake is not None
    dependency_arguments = []
    for variable in (
        "livox_ros_driver2_DIR",
        "vikit_common_DIR",
        "vikit_ros_DIR",
    ):
        value = cmake_cache_value(variable)
        assert value and not value.endswith("-NOTFOUND"), (
            f"configured dependency is unavailable: {variable}={value}"
        )
        dependency_arguments.append(f"-D{variable}={value}")

    with tempfile.TemporaryDirectory(
        prefix="livo-camera-time-transfer-shared-on-contract-"
    ) as directory:
        configured = subprocess.run(
            [
                cmake,
                "-S",
                str(PACKAGE_ROOT),
                "-B",
                directory,
                "-DBUILD_TESTING=ON",
                "-DBUILD_SHARED_LIBS=ON",
                "-DCMAKE_BUILD_TYPE=Release",
                *dependency_arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert configured.returncode == 0, configured.stderr
        built = subprocess.run(
            [
                cmake,
                "--build",
                directory,
                "--target",
                *targets,
                "--parallel",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert built.returncode == 0, built.stderr
        root = Path(directory)
        for target in targets:
            archive = root / f"lib{target}.a"
            assert archive.is_file(), f"missing {archive}"
            assert archive.stat().st_size > 0, f"empty {archive}"
            assert not tuple(root.glob(f"lib{target}.so*"))
