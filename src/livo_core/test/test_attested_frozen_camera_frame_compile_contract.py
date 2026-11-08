from pathlib import Path
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def explicit_build_root(environment: dict[str, str]) -> Path | None:
    value = environment.get("LIVO_ATTESTED_FRAME_BUILD_ROOT")
    if value is None or not value.strip():
        return None
    return Path(value.strip())


BUILD_ROOT = explicit_build_root(dict(os.environ))
HEADER = PACKAGE_ROOT / "include" / "attested_frozen_camera_frame.h"
SOURCE = PACKAGE_ROOT / "src" / "attested_frozen_camera_frame.cpp"
TIMING_SOURCE = PACKAGE_ROOT / "src" / "exposure_timing_attestation.cpp"
FROZEN_SOURCE = PACKAGE_ROOT / "src" / "frozen_camera_image.cpp"
CMAKE_FILE = PACKAGE_ROOT / "CMakeLists.txt"
PRODUCTION_TARGET = "attested_frozen_camera_frame"
FAILURE_TARGET = (
    "attested_frozen_camera_frame_failure_injection_test"
)
PRODUCTION_FLAGS = (
    BUILD_ROOT / "CMakeFiles" / f"{PRODUCTION_TARGET}.dir" / "flags.make"
    if BUILD_ROOT is not None
    else None
)
FAILURE_FLAGS = (
    BUILD_ROOT / "CMakeFiles" / f"{FAILURE_TARGET}.dir" / "flags.make"
    if BUILD_ROOT is not None
    else None
)
PRODUCTION_LIBRARY = (
    BUILD_ROOT / f"lib{PRODUCTION_TARGET}.a"
    if BUILD_ROOT is not None
    else None
)
FAILURE_LIBRARY = (
    BUILD_ROOT / f"lib{FAILURE_TARGET}.a"
    if BUILD_ROOT is not None
    else None
)
PRODUCTION_TIMING = (
    BUILD_ROOT / "libexposure_timing_attestation.a"
    if BUILD_ROOT is not None
    else None
)
FAILURE_TIMING = (
    BUILD_ROOT
    / "libexposure_timing_attestation_failure_injection_test.a"
    if BUILD_ROOT is not None
    else None
)
PRODUCTION_FROZEN = (
    BUILD_ROOT / "libfrozen_camera_image.a"
    if BUILD_ROOT is not None
    else None
)
FAILURE_FROZEN = (
    BUILD_ROOT / "libfrozen_camera_image_failure_injection_test.a"
    if BUILD_ROOT is not None
    else None
)
BEHAVIOR_TEST = (
    BUILD_ROOT / "test_attested_frozen_camera_frame"
    if BUILD_ROOT is not None
    else None
)
PRODUCTION_TEST = (
    BUILD_ROOT / "test_attested_frozen_camera_frame_production"
    if BUILD_ROOT is not None
    else None
)


def require_build_root() -> Path:
    assert BUILD_ROOT is not None, (
        "LIVO_ATTESTED_FRAME_BUILD_ROOT must explicitly name "
        "the audited build; cwd/canonical fallback is forbidden"
    )
    return BUILD_ROOT


def require_artifact(path: Path | None) -> Path:
    require_build_root()
    assert path is not None
    return path


def cmake_flags(path: Path | None, variable: str) -> list[str]:
    require_build_root()
    assert path is not None
    prefix = f"{variable} = "
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return shlex.split(line.removeprefix(prefix))
    raise AssertionError(f"{variable} missing from {path}")


def compile_translation_unit(
    source: str,
    extra_flags: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    compiler = shutil.which("c++")
    assert compiler is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-attested-frame-contract-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        source_path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                compiler,
                *cmake_flags(PRODUCTION_FLAGS, "CXX_DEFINES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_INCLUDES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_FLAGS"),
                *extra_flags,
                "-fsyntax-only",
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def external_link_libraries() -> list[str]:
    pkg_config = shutil.which("pkg-config")
    assert pkg_config is not None
    result = subprocess.run(
        [pkg_config, "--libs", "opencv4", "openssl"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout)


def link_translation_unit(
    source: str,
    libraries: list[Path | None],
    extra_link_flags: tuple[str, ...] = (),
) -> tuple[
    subprocess.CompletedProcess[str],
    subprocess.CompletedProcess[str] | None,
]:
    compiler = shutil.which("c++")
    assert compiler is not None
    require_build_root()
    for library in libraries:
        assert library is not None
        assert library.is_file(), library
    with tempfile.TemporaryDirectory(
        prefix="livo-attested-frame-link-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        executable = Path(directory) / "consumer"
        source_path.write_text(source, encoding="utf-8")
        linked = subprocess.run(
            [
                compiler,
                *cmake_flags(PRODUCTION_FLAGS, "CXX_DEFINES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_INCLUDES"),
                *cmake_flags(PRODUCTION_FLAGS, "CXX_FLAGS"),
                str(source_path),
                *extra_link_flags,
                *(str(library) for library in libraries),
                *(
                    ("-Wl,--no-whole-archive",)
                    if "-Wl,--whole-archive" in extra_link_flags
                    else ()
                ),
                *external_link_libraries(),
                "-pthread",
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if linked.returncode != 0:
            return linked, None
        executed = subprocess.run(
            [str(executable)],
            check=False,
            capture_output=True,
            text=True,
        )
        return linked, executed


def symbols(
    binary: Path | None,
    global_only: bool = False,
) -> str:
    require_build_root()
    assert binary is not None
    nm = shutil.which("nm")
    assert nm is not None
    command = [nm]
    if global_only:
        command.append("-g")
    command.extend(["-C", "--defined-only", str(binary)])
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def e0_public_definitions(
    binary: Path | None,
) -> dict[str, set[str]]:
    require_build_root()
    assert binary is not None
    class_names = (
        "CameraDriverFrameEnvelopeAuthorityIdentity",
        "CameraDriverFrameEnvelopeAuditView",
        "TrustedCameraDriverFrameEnvelope",
        "CameraDriverFrameEnvelopeAuthority",
        "AttestedCameraFrameIngressIdentity",
        "AttestedFrozenCameraFrameAuditView",
        "AttestedFrozenCameraFrame",
        "AttestedCameraFrameIngress",
    )
    definitions: dict[str, set[str]] = {}
    for line in symbols(binary, global_only=True).splitlines():
        match = re.match(
            r"^([0-9a-fA-F]+)\s+([A-Za-z])\s+(livo::.*)$",
            line,
        )
        if match is None:
            continue
        address, symbol_type, name = match.groups()
        if not any(
            f"livo::{class_name}::" in name
            for class_name in class_names
        ):
            continue
        assert symbol_type == "T", (binary, line)
        definitions.setdefault(name, set()).add(address)
    return definitions


def undefined_symbols(binary: Path | None) -> str:
    require_build_root()
    assert binary is not None
    nm = shutil.which("nm")
    assert nm is not None
    result = subprocess.run(
        [nm, "-C", "--undefined-only", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def raw_symbol_matching(
    binary: Path | None,
    fragment: str,
) -> str:
    require_build_root()
    assert binary is not None
    nm = shutil.which("nm")
    cxxfilt = shutil.which("c++filt")
    assert nm is not None
    assert cxxfilt is not None
    result = subprocess.run(
        [nm, "-g", "--defined-only", "-P", str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    raw_names = [
        line.split()[0]
        for line in result.stdout.splitlines()
        if line and not line.endswith(":")
    ]
    demangled = subprocess.run(
        [cxxfilt, *raw_names],
        check=False,
        capture_output=True,
        text=True,
    )
    assert demangled.returncode == 0, demangled.stderr
    matches = [
        raw
        for raw, readable in zip(
            raw_names,
            demangled.stdout.splitlines(),
            strict=True,
        )
        if fragment in readable
    ]
    assert len(matches) == 1, (binary, fragment, matches)
    return matches[0]


def stripped_rearchive(
    library: Path | None,
    symbol_fragments: tuple[str, ...],
    directory: Path,
) -> Path:
    require_build_root()
    assert library is not None
    archiver = shutil.which("ar")
    objcopy = shutil.which("objcopy")
    assert archiver is not None
    assert objcopy is not None
    members_result = subprocess.run(
        [archiver, "t", str(library)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert members_result.returncode == 0, members_result.stderr
    members = [
        line.strip()
        for line in members_result.stdout.splitlines()
        if line.strip()
    ]
    assert len(members) == 1, (library, members)
    extracted = subprocess.run(
        [archiver, "x", str(library)],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    assert extracted.returncode == 0, extracted.stderr
    original = directory / members[0]
    stripped = directory / "stripped.o"
    command = [objcopy]
    for fragment in symbol_fragments:
        command.append(
            f"--strip-symbol={raw_symbol_matching(library, fragment)}"
        )
    command.extend([str(original), str(stripped)])
    stripped_result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert stripped_result.returncode == 0, stripped_result.stderr
    rearchive = directory / "rearchive.a"
    archived = subprocess.run(
        [archiver, "rcs", str(rearchive), str(stripped)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert archived.returncode == 0, archived.stderr
    return rearchive


def add_library_arguments(target: str) -> list[str]:
    declaration = re.search(
        rf"add_library\s*\(\s*{re.escape(target)}\b([^)]*)\)",
        CMAKE_FILE.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert declaration is not None, target
    return declaration.group(1).split()


def link_names(target: str) -> set[str]:
    build_root = require_build_root()
    production_library = require_artifact(PRODUCTION_LIBRARY)
    failure_library = require_artifact(FAILURE_LIBRARY)
    link_file = (
        build_root / "CMakeFiles" / f"{target}.dir" / "link.txt"
    )
    if link_file.is_file():
        command = link_file.read_text(encoding="utf-8")
    else:
        commands, generator = all_link_commands()
        assert generator == "ninja"
        candidates = [
            item
            for item in commands
            if command_output_basename(item) == target
            and (
                production_library.name in item
                or failure_library.name in item
            )
        ]
        assert len(candidates) == 1, (target, candidates)
        command = candidates[0]
    return {
        Path(item).name
        for item in shlex.split(command)
    }


def command_output_basename(command: str) -> str | None:
    tokens = shlex.split(command)
    for index, token in enumerate(tokens):
        if token == "-o" and index + 1 < len(tokens):
            return Path(tokens[index + 1]).name
        if token.startswith("-o") and len(token) > 2:
            return Path(token[2:]).name
    return None


def command_variant_mixes(commands: list[str]) -> list[str]:
    production_e0 = require_artifact(PRODUCTION_LIBRARY).name
    failure_e0 = require_artifact(FAILURE_LIBRARY).name
    production_names = {
        production_e0,
        require_artifact(PRODUCTION_TIMING).name,
        require_artifact(PRODUCTION_FROZEN).name,
    }
    failure_names = {
        failure_e0,
        require_artifact(FAILURE_TIMING).name,
        require_artifact(FAILURE_FROZEN).name,
    }
    return [
        command
        for command in commands
        if (production_e0 in command or failure_e0 in command)
        and any(name in command for name in production_names)
        and any(name in command for name in failure_names)
    ]


def all_link_commands() -> tuple[list[str], str]:
    build_root = require_build_root()
    make_links = sorted(build_root.rglob("link.txt"))
    if make_links:
        return (
            [
                link.read_text(encoding="utf-8")
                for link in make_links
            ],
            "make",
        )
    ninja_file = build_root / "build.ninja"
    if ninja_file.is_file():
        ninja = shutil.which("ninja")
        assert ninja is not None
        result = subprocess.run(
            [ninja, "-C", str(build_root), "-t", "commands"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return (result.stdout.splitlines(), "ninja")
    raise AssertionError(
        f"no Make link.txt or Ninja graph under {build_root}"
    )


def test_build_root_requires_explicit_nonblank_environment() -> None:
    assert explicit_build_root({}) is None
    assert explicit_build_root(
        {"LIVO_ATTESTED_FRAME_BUILD_ROOT": "   "}
    ) is None
    assert explicit_build_root(
        {"LIVO_ATTESTED_FRAME_BUILD_ROOT": "/tmp/e0-audit"}
    ) == Path("/tmp/e0-audit")


def test_synthetic_ninja_selector_uses_complete_graph_and_exact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = "test_attested_frozen_camera_frame"
    production = tmp_path / f"lib{PRODUCTION_TARGET}.a"
    failure = tmp_path / f"lib{FAILURE_TARGET}.a"
    (tmp_path / "build.ninja").write_text(
        "# synthetic graph marker\n",
        encoding="utf-8",
    )
    commands = [
        (
            f"/usr/bin/c++ exact.o -o {tmp_path / target} "
            f"{production}"
        ),
        (
            f"/usr/bin/c++ suffix.o "
            f"-o{tmp_path / (target + '_production')} {failure}"
        ),
        (
            f"/usr/bin/c++ prefix.o -o "
            f"{tmp_path / ('prefix_' + target)} {failure}"
        ),
    ]
    calls: list[list[str]] = []

    def fake_which(program: str) -> str | None:
        assert program == "ninja"
        return "/synthetic/bin/ninja"

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs == {
            "check": False,
            "capture_output": True,
            "text": True,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(commands),
            stderr="",
        )

    monkeypatch.setitem(globals(), "BUILD_ROOT", tmp_path)
    monkeypatch.setitem(
        globals(), "PRODUCTION_LIBRARY", production
    )
    monkeypatch.setitem(globals(), "FAILURE_LIBRARY", failure)
    monkeypatch.setattr(shutil, "which", fake_which)
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert command_output_basename(
        "c++ input.o -o /tmp/bin/separate"
    ) == "separate"
    assert command_output_basename(
        "c++ input.o -o/tmp/bin/attached"
    ) == "attached"
    selected = link_names(target)

    assert production.name in selected
    assert failure.name not in selected
    assert calls == [[
        "/synthetic/bin/ninja",
        "-C",
        str(tmp_path),
        "-t",
        "commands",
    ]]


def test_tokens_are_opaque_noexcept_value_handles() -> None:
    result = compile_translation_unit(
        """
#include "attested_frozen_camera_frame.h"
#include "checked_increment.h"
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

using EnvelopeIdentity =
  livo::CameraDriverFrameEnvelopeAuthorityIdentity;
using EnvelopeAudit = livo::CameraDriverFrameEnvelopeAuditView;
using Envelope = livo::TrustedCameraDriverFrameEnvelope;
using IngressIdentity = livo::AttestedCameraFrameIngressIdentity;
using FrameAudit = livo::AttestedFrozenCameraFrameAuditView;
using Frame = livo::AttestedFrozenCameraFrame;
using EnvelopeAuthority =
  livo::CameraDriverFrameEnvelopeAuthority;
using Ingress = livo::AttestedCameraFrameIngress;

static_assert(!std::is_default_constructible_v<EnvelopeIdentity>);
static_assert(!std::is_default_constructible_v<EnvelopeAudit>);
static_assert(!std::is_default_constructible_v<Envelope>);
static_assert(!std::is_default_constructible_v<IngressIdentity>);
static_assert(!std::is_default_constructible_v<FrameAudit>);
static_assert(!std::is_default_constructible_v<Frame>);
static_assert(std::is_copy_constructible_v<Envelope>);
static_assert(std::is_nothrow_copy_constructible_v<Envelope>);
static_assert(std::is_nothrow_move_constructible_v<Envelope>);
static_assert(std::is_nothrow_copy_assignable_v<Envelope>);
static_assert(std::is_copy_constructible_v<Frame>);
static_assert(std::is_nothrow_copy_constructible_v<Frame>);
static_assert(std::is_nothrow_move_constructible_v<Frame>);
static_assert(std::is_nothrow_copy_assignable_v<Frame>);
static_assert(std::is_nothrow_constructible_v<
  std::optional<Frame>, Frame &&>);
static_assert(std::is_nothrow_move_constructible_v<
  livo::AttestedCameraFrameIngestResult>);
static_assert(!std::is_copy_constructible_v<EnvelopeAuthority>);
static_assert(!std::is_move_constructible_v<EnvelopeAuthority>);
static_assert(!std::is_copy_constructible_v<Ingress>);
static_assert(!std::is_move_constructible_v<Ingress>);
static_assert(livo::kAttestedCameraFrameCameraCount == 4U);
static_assert(
  livo::kAttestedCameraFrameIdentifierMaxLength == 128U);
static_assert(livo::kAttestedCameraFrameTopicMaxLength == 256U);
static_assert(
  livo::kAttestedCameraFrameMaximumImageDimension == 16384U);
static_assert(
  livo::kAttestedCameraFrameMaximumPayloadBytes ==
  64U * 1024U * 1024U);
static_assert(sizeof(Envelope) == sizeof(std::shared_ptr<const void>));
static_assert(sizeof(Frame) == sizeof(std::shared_ptr<const void>));
static_assert(std::is_same_v<
  decltype(std::declval<const Frame &>().audit()),
  std::optional<FrameAudit>>);
static_assert(std::is_same_v<
  decltype(std::declval<const FrameAudit &>().cameraId()),
  std::uint8_t>);
static_assert(std::is_same_v<
  decltype(
    std::declval<const FrameAudit &>().effectiveExposureTime()),
  std::optional<livo::EffectiveExposureTime>>);
static_assert(std::is_same_v<
  decltype(std::declval<const FrameAudit &>().imageDigest()),
  livo::FrozenCameraImage::Digest>);
static_assert(std::is_same_v<
  decltype(
    std::declval<const EnvelopeAudit &>().
    hardwareFrameCounterPolicy()),
  livo::CameraDriverHardwareFrameCounterPolicy>);
static_assert(noexcept(livo::checkedReserve(
  std::declval<std::uint64_t &>(),
  std::declval<std::uint64_t &>())));
static_assert(!noexcept(
  std::declval<Ingress &>().ingest(
    std::declval<const Envelope &>())));
"""
    )
    assert result.returncode == 0, result.stderr


def test_public_api_cannot_combine_independent_image_and_timing() -> None:
    result = compile_translation_unit(
        """
#include "attested_frozen_camera_frame.h"

void combine(
  livo::AttestedCameraFrameIngress & ingress,
  const livo::TrustedCameraDriverFrameEnvelope & envelope,
  const livo::FrozenCameraImage & image,
  const livo::ExposureTimingAttestation & timing)
{
  (void)ingress.ingest(envelope, image, timing);
}
"""
    )
    assert result.returncode != 0

    result = compile_translation_unit(
        """
#include "attested_frozen_camera_frame.h"

void forge()
{
  livo::TrustedCameraDriverFrameEnvelope envelope;
  livo::AttestedFrozenCameraFrame frame;
  (void)envelope;
  (void)frame;
}
"""
    )
    assert result.returncode != 0
    assert "private" in result.stderr or "no matching" in result.stderr


def test_hardware_counter_policy_is_explicit_and_digest_bound() -> None:
    header = HEADER.read_text(encoding="utf-8")
    for required in (
        "CameraDriverHardwareFrameCounterPolicy",
        "REQUIRED_MONOTONIC",
        "UNAVAILABLE",
        "FOREIGN_HARDWARE_FRAME_COUNTER_POLICY",
        "MISSING_REQUIRED_HARDWARE_FRAME_COUNTER",
        "UNEXPECTED_HARDWARE_FRAME_COUNTER",
        "hardware_frame_counter_policy",
        "hardwareFrameCounterPolicy",
    ):
        assert required in header
    source = SOURCE.read_text(encoding="utf-8")
    digest_body = source[
        source.index("[[maybe_unused]] bool canonicalDigestFor"):
        source.index("bool allZero")
    ]
    assert "hardware_frame_counter_policy" in digest_body
    assert "livo.camera-driver-frame-envelope.v2" in digest_body


def test_token_surface_has_no_claim_queue_or_raw_image_escape() -> None:
    header = HEADER.read_text(encoding="utf-8")
    for forbidden in (
        "takeReady",
        "claimFor",
        "claimedFor",
        "mutableClaim",
        "bindImage",
        "bindTiming",
        "mintAttested",
        "friend class AdaptiveCameraSnapshotQueue",
        "friend class CameraObservationBatch",
        "LIVO_ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION_TESTING",
        "issuer_identity",
        "issuer_seal",
    ):
        assert forbidden not in header
    assert "cv::Mat" not in header
    assert "sensor_msgs" not in header
    assert "rclcpp" not in header

    result = compile_translation_unit(
        """
#include "attested_frozen_camera_frame.h"
#include <type_traits>
#include <utility>

using Frame = livo::AttestedFrozenCameraFrame;
#define HAS_METHOD(Trait, Method)                                \\
template<typename T, typename = void>                            \\
struct Trait : std::false_type {};                              \\
template<typename T>                                             \\
struct Trait<T, std::void_t<                                    \\
  decltype(std::declval<T &>().Method())>>                       \\
: std::true_type {}

HAS_METHOD(HasImage, image);
HAS_METHOD(HasClaim, claim);
HAS_METHOD(HasTakeReady, takeReady);
HAS_METHOD(HasMutableData, data);
static_assert(!HasImage<Frame>::value);
static_assert(!HasClaim<Frame>::value);
static_assert(!HasTakeReady<Frame>::value);
static_assert(!HasMutableData<Frame>::value);
"""
    )
    assert result.returncode == 0, result.stderr


def test_dependency_direction_is_one_way() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert '#include "attested_frozen_camera_frame.h"' in source
    assert '#include "exposure_timing_attestation.h"' in (
        HEADER.read_text(encoding="utf-8")
    )
    assert '#include "frozen_camera_image.h"' in (
        HEADER.read_text(encoding="utf-8")
    )
    assert "attested_frozen_camera_frame" not in (
        TIMING_SOURCE.read_text(encoding="utf-8")
    )
    assert "attested_frozen_camera_frame" not in (
        FROZEN_SOURCE.read_text(encoding="utf-8")
    )


def test_security_archives_are_explicit_static_targets() -> None:
    targets = (
        PRODUCTION_TARGET,
        FAILURE_TARGET,
        "exposure_timing_attestation",
        "exposure_timing_attestation_failure_injection_test",
        "frozen_camera_image",
        "frozen_camera_image_failure_injection_test",
    )
    for target in targets:
        arguments = add_library_arguments(target)
        assert arguments[0] == "STATIC", (target, arguments)


def test_supported_cmake_targets_publish_one_compatible_variant(
) -> None:
    cmake = CMAKE_FILE.read_text(encoding="utf-8")
    variants = {
        PRODUCTION_TARGET: "production",
        "exposure_timing_attestation": "production",
        "frozen_camera_image": "production",
        FAILURE_TARGET: "failure_injection",
        "exposure_timing_attestation_failure_injection_test":
            "failure_injection",
        "frozen_camera_image_failure_injection_test":
            "failure_injection",
    }
    for target, variant in variants.items():
        value = re.search(
            rf"set_property\s*\(\s*TARGET\s+{re.escape(target)}\s+"
            rf"PROPERTY\s+INTERFACE_LIVO_ATTESTED_FRAME_VARIANT\s+"
            rf"{variant}\s*\)",
            cmake,
            re.DOTALL,
        )
        assert value is not None, (target, variant)
        compatibility = re.search(
            rf"set_property\s*\(\s*TARGET\s+{re.escape(target)}\s+"
            r"APPEND\s+PROPERTY\s+COMPATIBLE_INTERFACE_STRING\s+"
            r"LIVO_ATTESTED_FRAME_VARIANT\s*\)",
            cmake,
            re.DOTALL,
        )
        assert compatibility is not None, target

    failure_targets = (
        FAILURE_TARGET,
        "exposure_timing_attestation_failure_injection_test",
        "frozen_camera_image_failure_injection_test",
    )
    testing_index = cmake.index("if(BUILD_TESTING)")
    for target in failure_targets:
        assert cmake.index(f"add_library(\n    {target}") > (
            testing_index
        )
    install_bodies = re.findall(
        r"install\s*\((.*?)\)", cmake, re.DOTALL
    )
    for target in failure_targets:
        assert all(target not in body for body in install_bodies)
    assert "ament_export_targets" not in cmake
    assert "Direct raw" in cmake
    assert "unsupported" in cmake


def configure_mixed_variant_consumer(
    first: str,
    second: str,
) -> subprocess.CompletedProcess[str]:
    cmake = shutil.which("cmake")
    assert cmake is not None
    production = require_artifact(PRODUCTION_LIBRARY)
    failure = require_artifact(FAILURE_LIBRARY)
    with tempfile.TemporaryDirectory(
        prefix="livo-attested-frame-cmake-variant-"
    ) as directory_text:
        directory = Path(directory_text)
        source = directory / "source"
        build = directory / "build"
        source.mkdir()
        (source / "main.cpp").write_text(
            "int main() { return 0; }\n",
            encoding="utf-8",
        )
        (source / "CMakeLists.txt").write_text(
            f"""
cmake_minimum_required(VERSION 3.10)
project(attested_frame_variant_probe LANGUAGES CXX)
add_library(production STATIC IMPORTED)
set_property(TARGET production PROPERTY
  IMPORTED_LOCATION "{production.as_posix()}")
set_property(TARGET production PROPERTY
  INTERFACE_LIVO_ATTESTED_FRAME_VARIANT production)
set_property(TARGET production APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING LIVO_ATTESTED_FRAME_VARIANT)
add_library(failure_injection STATIC IMPORTED)
set_property(TARGET failure_injection PROPERTY
  IMPORTED_LOCATION "{failure.as_posix()}")
set_property(TARGET failure_injection PROPERTY
  INTERFACE_LIVO_ATTESTED_FRAME_VARIANT failure_injection)
set_property(TARGET failure_injection APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING LIVO_ATTESTED_FRAME_VARIANT)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE {first} {second})
""",
            encoding="utf-8",
        )
        return subprocess.run(
            [
                cmake,
                "-S",
                str(source),
                "-B",
                str(build),
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def test_supported_cmake_target_colink_is_rejected_in_both_orders(
) -> None:
    for first, second in (
        ("production", "failure_injection"),
        ("failure_injection", "production"),
    ):
        result = configure_mixed_variant_consumer(first, second)
        assert result.returncode != 0, (
            first,
            second,
            result.stdout,
            result.stderr,
        )
        diagnostics = (result.stdout + result.stderr).lower()
        assert "livo_attested_frame_variant" in diagnostics
        assert (
            "disagree" in diagnostics
            or "compatible" in diagnostics
            or "does not agree" in diagnostics
            or "not agree" in diagnostics
            or "already determined" in diagnostics
        )


def test_variants_do_not_colink() -> None:
    production_library = require_artifact(PRODUCTION_LIBRARY)
    failure_library = require_artifact(FAILURE_LIBRARY)
    production_timing = require_artifact(PRODUCTION_TIMING)
    failure_timing = require_artifact(FAILURE_TIMING)
    production_frozen = require_artifact(PRODUCTION_FROZEN)
    failure_frozen = require_artifact(FAILURE_FROZEN)
    behavior = link_names("test_attested_frozen_camera_frame")
    assert failure_library.name in behavior
    assert failure_timing.name in behavior
    assert failure_frozen.name in behavior
    assert production_library.name not in behavior
    assert production_timing.name not in behavior
    assert production_frozen.name not in behavior

    production = link_names(
        "test_attested_frozen_camera_frame_production"
    )
    assert production_library.name in production
    assert production_timing.name in production
    assert production_frozen.name in production
    assert failure_library.name not in production
    assert failure_timing.name not in production
    assert failure_frozen.name not in production


def test_variant_mix_parser_detects_all_synthetic_families() -> None:
    production_library = require_artifact(PRODUCTION_LIBRARY)
    failure_library = require_artifact(FAILURE_LIBRARY)
    production_timing = require_artifact(PRODUCTION_TIMING)
    failure_timing = require_artifact(FAILURE_TIMING)
    production_frozen = require_artifact(PRODUCTION_FROZEN)
    failure_frozen = require_artifact(FAILURE_FROZEN)
    safe = [
        (
            f"c++ app.o {production_library.name} "
            f"{production_timing.name} {production_frozen.name} -o app"
        ),
        (
            f"c++ test.o {failure_library.name} "
            f"{failure_timing.name} {failure_frozen.name} -o test"
        ),
    ]
    violating = [
        (
            f"c++ bad-e0.o {production_library.name} "
            f"{failure_library.name} -o bad-e0"
        ),
        (
            f"c++ bad-timing.o {production_library.name} "
            f"{failure_timing.name} -o bad-timing"
        ),
        (
            f"c++ bad-frozen.o {failure_library.name} "
            f"{production_frozen.name} -o bad-frozen"
        ),
    ]
    assert command_variant_mixes(safe) == []
    assert len(command_variant_mixes(violating)) == len(violating)


def test_all_actual_downstream_link_commands_avoid_variant_mix(
) -> None:
    commands, generator = all_link_commands()
    assert commands
    assert not command_variant_mixes(commands)
    if generator != "ninja":
        pytest.skip(
            "audited complete Make link.txt graph; no real Ninja root "
            "was provided"
        )


def test_production_create_is_typed_unavailable_without_parsing() -> None:
    consumer = """
#include "attested_frozen_camera_frame.h"
#include <cstdlib>
#include <new>

namespace
{
bool forbid_allocation = false;
}

void * operator new(const std::size_t size)
{
  if (forbid_allocation)
  {
    std::abort();
  }
  if (void * memory = std::malloc(size))
  {
    return memory;
  }
  throw std::bad_alloc();
}

void operator delete(void * memory) noexcept
{
  std::free(memory);
}

void operator delete(void * memory, std::size_t) noexcept
{
  std::free(memory);
}

int main()
{
  livo::CameraDriverFrameEnvelopeAuthorityConfiguration config;
  config.routes[0U].metadata_revision = "must-not-be-parsed";
  forbid_allocation = true;
  const auto result =
    livo::CameraDriverFrameEnvelopeAuthority::create(config);
  forbid_allocation = false;
  return
    result.status ==
      livo::CameraDriverFrameEnvelopeAuthorityCreateStatus::
      DRIVER_CONTRACT_UNAVAILABLE &&
    !result.authority &&
    !result.ready() ?
    0 : 1;
}
"""
    linked, executed = link_translation_unit(
        consumer,
        [
            PRODUCTION_LIBRARY,
            PRODUCTION_TIMING,
            PRODUCTION_FROZEN,
        ],
    )
    assert linked.returncode == 0, linked.stderr
    assert executed is not None
    assert executed.returncode == 0, executed.stderr


def test_production_issue_and_ingest_stubs_do_not_allocate_or_parse(
) -> None:
    allocation_trap = """
#include <cstdlib>
#include <new>
namespace
{
bool forbid_allocation = false;
}
void * operator new(const std::size_t size)
{
  if (forbid_allocation) { std::abort(); }
  if (void * memory = std::malloc(size)) { return memory; }
  throw std::bad_alloc();
}
void operator delete(void * memory) noexcept { std::free(memory); }
void operator delete(void * memory, std::size_t) noexcept
{
  std::free(memory);
}
"""
    issue_probe = (
        """
#include "attested_frozen_camera_frame.h"
#include <memory>
#include <utility>
"""
        + allocation_trap
        + """
namespace livo
{
CameraDriverFrameEnvelopeAuthorityCreateResult
CameraDriverFrameEnvelopeAuthority::create(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &) noexcept
{
  auto authority =
    std::unique_ptr<CameraDriverFrameEnvelopeAuthority>(
      new CameraDriverFrameEnvelopeAuthority(
        std::shared_ptr<Implementation>{}));
  return {
    CameraDriverFrameEnvelopeAuthorityCreateStatus::READY,
    std::move(authority)};
}
}  // namespace livo
int main()
{
  livo::CameraDriverFrameEnvelopeAuthorityConfiguration config;
  auto authority =
    livo::CameraDriverFrameEnvelopeAuthority::create(config);
  if (!authority.authority) { return 2; }
  livo::CameraDriverFrameDocument document;
  document.metadata_revision = "must-not-be-parsed";
  forbid_allocation = true;
  const auto result = authority.authority->issue(document);
  forbid_allocation = false;
  return
    result.status ==
      livo::CameraDriverFrameEnvelopeIssueStatus::
      DRIVER_CONTRACT_UNAVAILABLE &&
    !result.envelope.has_value() ?
    0 : 1;
}
"""
    )
    with tempfile.TemporaryDirectory(
        prefix="livo-production-issue-stub-"
    ) as directory_text:
        rearchive = stripped_rearchive(
            PRODUCTION_LIBRARY,
            (
                "CameraDriverFrameEnvelopeAuthority::create",
            ),
            Path(directory_text),
        )
        linked, executed = link_translation_unit(
            issue_probe,
            [rearchive, PRODUCTION_TIMING, PRODUCTION_FROZEN],
        )
        assert linked.returncode == 0, linked.stderr
        assert executed is not None
        assert executed.returncode == 0, executed.stderr

    ingest_probe = (
        """
#include "attested_frozen_camera_frame.h"
#include <memory>
#include <utility>
"""
        + allocation_trap
        + """
namespace livo
{
CameraDriverFrameEnvelopeAuthorityCreateResult
CameraDriverFrameEnvelopeAuthority::create(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &) noexcept
{
  auto authority =
    std::unique_ptr<CameraDriverFrameEnvelopeAuthority>(
      new CameraDriverFrameEnvelopeAuthority(
        std::shared_ptr<Implementation>{}));
  return {
    CameraDriverFrameEnvelopeAuthorityCreateStatus::READY,
    std::move(authority)};
}

CameraDriverFrameEnvelopeIssueResult
CameraDriverFrameEnvelopeAuthority::issue(
  const CameraDriverFrameDocument &)
{
  std::shared_ptr<
    const CameraDriverFrameEnvelopeAuditView::Data> data;
  return {
    CameraDriverFrameEnvelopeIssueStatus::READY,
    TrustedCameraDriverFrameEnvelope{std::move(data)}};
}

AttestedCameraFrameIngressCreateResult
AttestedCameraFrameIngress::create(
  const AttestedCameraFrameIngressConfiguration &,
  const CameraDriverFrameEnvelopeAuthorityIdentity &,
  std::unique_ptr<ExposureTimingSourceAuthority>,
  std::unique_ptr<ExposureTimingAuthority>) noexcept
{
  auto ingress = std::unique_ptr<AttestedCameraFrameIngress>(
    new AttestedCameraFrameIngress(
      std::shared_ptr<Implementation>{}));
  return {
    AttestedCameraFrameIngressCreateStatus::READY,
    std::move(ingress)};
}
}  // namespace livo

int main()
{
  livo::CameraDriverFrameEnvelopeAuthorityConfiguration config;
  auto authority =
    livo::CameraDriverFrameEnvelopeAuthority::create(config);
  livo::CameraDriverFrameDocument document;
  auto envelope = authority.authority->issue(document);
  if (!envelope.envelope.has_value()) { return 2; }
  auto identity = authority.authority->identity();
  livo::AttestedCameraFrameIngressConfiguration ingress_config;
  auto ingress = livo::AttestedCameraFrameIngress::create(
    ingress_config,
    identity,
    std::unique_ptr<livo::ExposureTimingSourceAuthority>{},
    std::unique_ptr<livo::ExposureTimingAuthority>{});
  if (!ingress.ingress) { return 3; }
  forbid_allocation = true;
  const auto result =
    ingress.ingress->ingest(*envelope.envelope);
  forbid_allocation = false;
  return
    result.status ==
      livo::AttestedCameraFrameIngestStatus::
      DRIVER_CONTRACT_UNAVAILABLE &&
    !result.frame.has_value() ?
    0 : 1;
}
"""
    )
    with tempfile.TemporaryDirectory(
        prefix="livo-production-ingest-stub-"
    ) as directory_text:
        rearchive = stripped_rearchive(
            PRODUCTION_LIBRARY,
            (
                "CameraDriverFrameEnvelopeAuthority::create",
                "CameraDriverFrameEnvelopeAuthority::issue",
                "AttestedCameraFrameIngress::create",
            ),
            Path(directory_text),
        )
        linked, executed = link_translation_unit(
            ingest_probe,
            [rearchive, PRODUCTION_TIMING, PRODUCTION_FROZEN],
        )
        assert linked.returncode == 0, linked.stderr
        assert executed is not None
        assert executed.returncode == 0, executed.stderr


def test_public_issuer_methods_have_one_strong_definition_per_archive(
) -> None:
    required = (
        "CameraDriverFrameEnvelopeAuthorityIdentity::"
        "CameraDriverFrameEnvelopeAuthorityIdentity",
        "CameraDriverFrameEnvelopeAuthorityIdentity::valid",
        "CameraDriverFrameEnvelopeAuthorityIdentity::sameAuthority",
        "CameraDriverFrameEnvelopeAuditView::"
        "CameraDriverFrameEnvelopeAuditView",
        "CameraDriverFrameEnvelopeAuditView::valid",
        "CameraDriverFrameEnvelopeAuditView::cameraId",
        "CameraDriverFrameEnvelopeAuditView::envelopeSequence",
        "CameraDriverFrameEnvelopeAuditView::driverEpoch",
        "CameraDriverFrameEnvelopeAuditView::driverFrameNumber",
        "CameraDriverFrameEnvelopeAuditView::"
        "hardwareFrameCounterPolicy",
        "CameraDriverFrameEnvelopeAuditView::hardwareFrameCounter",
        "CameraDriverFrameEnvelopeAuditView::clockDomain",
        "CameraDriverFrameEnvelopeAuditView::rawHeaderTimestampSec",
        "CameraDriverFrameEnvelopeAuditView::sensorTimestampSec",
        "CameraDriverFrameEnvelopeAuditView::exposureDurationSec",
        "CameraDriverFrameEnvelopeAuditView::width",
        "CameraDriverFrameEnvelopeAuditView::height",
        "CameraDriverFrameEnvelopeAuditView::encoding",
        "CameraDriverFrameEnvelopeAuditView::isBigendian",
        "CameraDriverFrameEnvelopeAuditView::step",
        "CameraDriverFrameEnvelopeAuditView::dataSize",
        "CameraDriverFrameEnvelopeAuditView::sourceId",
        "CameraDriverFrameEnvelopeAuditView::topic",
        "CameraDriverFrameEnvelopeAuditView::sensorSerial",
        "CameraDriverFrameEnvelopeAuditView::frameId",
        "CameraDriverFrameEnvelopeAuditView::metadataRevision",
        "CameraDriverFrameEnvelopeAuditView::transformRevision",
        "CameraDriverFrameEnvelopeAuditView::canonicalDigest",
        "TrustedCameraDriverFrameEnvelope::"
        "TrustedCameraDriverFrameEnvelope",
        "TrustedCameraDriverFrameEnvelope::valid",
        "TrustedCameraDriverFrameEnvelope::issuedBy",
        "TrustedCameraDriverFrameEnvelope::sameFrame",
        "TrustedCameraDriverFrameEnvelope::audit",
        "CameraDriverFrameEnvelopeAuthority::"
        "CameraDriverFrameEnvelopeAuthority",
        "CameraDriverFrameEnvelopeAuthority::"
        "~CameraDriverFrameEnvelopeAuthority",
        "CameraDriverFrameEnvelopeAuthority::create",
        "CameraDriverFrameEnvelopeAuthority::issue",
        "CameraDriverFrameEnvelopeAuthority::identity",
        "AttestedCameraFrameIngressIdentity::"
        "AttestedCameraFrameIngressIdentity",
        "AttestedCameraFrameIngressIdentity::valid",
        "AttestedCameraFrameIngressIdentity::sameIngress",
        "AttestedFrozenCameraFrameAuditView::"
        "AttestedFrozenCameraFrameAuditView",
        "AttestedFrozenCameraFrameAuditView::valid",
        "AttestedFrozenCameraFrameAuditView::cameraId",
        "AttestedFrozenCameraFrameAuditView::ingressSequence",
        "AttestedFrozenCameraFrameAuditView::imageRows",
        "AttestedFrozenCameraFrameAuditView::imageColumns",
        "AttestedFrozenCameraFrameAuditView::imageType",
        "AttestedFrozenCameraFrameAuditView::imageDigest",
        "AttestedFrozenCameraFrameAuditView::"
        "effectiveExposureTime",
        "AttestedFrozenCameraFrameAuditView::envelope",
        "AttestedFrozenCameraFrameAuditView::timing",
        "AttestedFrozenCameraFrame::AttestedFrozenCameraFrame",
        "AttestedFrozenCameraFrame::valid",
        "AttestedFrozenCameraFrame::issuedBy",
        "AttestedFrozenCameraFrame::sameFrame",
        "AttestedFrozenCameraFrame::audit",
        "AttestedCameraFrameIngress::AttestedCameraFrameIngress",
        "AttestedCameraFrameIngress::~AttestedCameraFrameIngress",
        "AttestedCameraFrameIngress::create",
        "AttestedCameraFrameIngress::ingest",
        "AttestedCameraFrameIngress::identity",
    )
    signature_sets: list[set[str]] = []
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        definitions = e0_public_definitions(library)
        assert definitions
        for name, addresses in definitions.items():
            assert len(addresses) == 1, (
                library,
                name,
                addresses,
            )
        for symbol in required:
            matches = [
                name for name in definitions if symbol in name
            ]
            assert matches, (
                library,
                symbol,
                sorted(definitions),
            )
        signature_sets.append(set(definitions))
        assert not undefined_symbols(library).count(
            "livo::test::"
        )
    assert signature_sets[0] == signature_sets[1]


def test_archive_issuer_seals_are_internal_and_variant_local() -> None:
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        all_symbols = symbols(library)
        global_symbols = symbols(library, global_only=True)
        for seal in (
            "envelope_issuer_seal",
            "attested_frame_issuer_seal",
        ):
            matches = [
                line for line in all_symbols.splitlines()
                if seal in line
            ]
            assert len(matches) == 1, (library, seal, matches)
            assert "(anonymous namespace)" in matches[0]
            assert seal not in global_symbols


def test_failure_hooks_are_not_in_production_archive() -> None:
    hooks = (
        "failNextCameraDriverEnvelopeIdentityAllocation",
        "failNextCameraDriverEnvelopeImplementationAllocation",
        "failNextCameraDriverEnvelopePayloadAllocation",
        "failNextAttestedFrameIngressIdentityAllocation",
        "failNextAttestedFrameIngressImplementationAllocation",
        "failNextAttestedFramePayloadAllocation",
        "failNextCameraDriverEnvelopeDigestComputation",
        "exhaustNextCameraDriverEnvelopeSequence",
        "exhaustNextAttestedFrameIngressSequence",
        "corruptNextCameraDriverEnvelopeDigest",
        "corruptNextCameraDriverEnvelopeMetadata",
        "corruptNextCameraDriverEnvelopePixel",
    )
    production = symbols(PRODUCTION_LIBRARY)
    failure = symbols(FAILURE_LIBRARY)
    behavior = symbols(BEHAVIOR_TEST)
    for hook in hooks:
        assert hook not in production
        assert hook in failure
        assert hook in behavior
    production_defines = cmake_flags(
        PRODUCTION_FLAGS, "CXX_DEFINES"
    )
    failure_defines = cmake_flags(
        FAILURE_FLAGS, "CXX_DEFINES"
    )
    assert not any(
        "ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION"
        in flag
        for flag in production_defines
    )
    assert any(
        "ATTESTED_FROZEN_CAMERA_FRAME_FAILURE_INJECTION"
        in flag
        for flag in failure_defines
    )


def test_consumer_issuer_override_cannot_launder_seal() -> None:
    malicious_definitions = (
        """
#include "attested_frozen_camera_frame.h"

namespace livo
{
CameraDriverFrameEnvelopeAuthorityCreateResult
CameraDriverFrameEnvelopeAuthority::create(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &) noexcept
{
  return {
    CameraDriverFrameEnvelopeAuthorityCreateStatus::READY,
    nullptr};
}
}  // namespace livo

int main() { return 0; }
""",
        """
#include "attested_frozen_camera_frame.h"

namespace livo
{
CameraDriverFrameEnvelopeIssueResult
CameraDriverFrameEnvelopeAuthority::issue(
  const CameraDriverFrameDocument &)
{
  return {
    CameraDriverFrameEnvelopeIssueStatus::READY,
    std::nullopt};
}
}  // namespace livo

int main() { return 0; }
""",
        """
#include "attested_frozen_camera_frame.h"

namespace livo
{
AttestedCameraFrameIngestResult
AttestedCameraFrameIngress::ingest(
  const TrustedCameraDriverFrameEnvelope &)
{
  return {
    AttestedCameraFrameIngestStatus::READY,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt};
}
}  // namespace livo

int main() { return 0; }
""",
    )
    for library, timing, frozen in (
        (PRODUCTION_LIBRARY, PRODUCTION_TIMING, PRODUCTION_FROZEN),
        (FAILURE_LIBRARY, FAILURE_TIMING, FAILURE_FROZEN),
    ):
        for malicious in malicious_definitions:
            linked, executed = link_translation_unit(
                malicious,
                [library, timing, frozen],
                ("-Wl,--whole-archive",),
            )
            assert linked.returncode != 0
            assert executed is None
            assert "multiple definition" in linked.stderr.lower()


def test_frozen_image_member_override_is_duplicate_link_failure() -> None:
    malicious = """
#include "frozen_camera_image.h"

namespace livo
{
FrozenCameraImageFreezeResult FrozenCameraImage::freeze(
  const cv::Mat &) noexcept
{
  return {
    FrozenCameraImageFreezeStatus::READY,
    std::nullopt};
}
}  // namespace livo

int main()
{
  auto official = &livo::FrozenCameraImage::digest;
  return official == nullptr ? 1 : 0;
}
"""
    for frozen in (PRODUCTION_FROZEN, FAILURE_FROZEN):
        linked, executed = link_translation_unit(
            malicious,
            [frozen],
            ("-Wl,--whole-archive",),
        )
        assert linked.returncode != 0
        assert executed is None
        assert "multiple definition" in linked.stderr.lower()
        assert "freeze" in linked.stderr.lower()


def test_production_and_failure_variants_cannot_be_whole_linked() -> None:
    consumer = "int main() { return 0; }\n"
    linked, executed = link_translation_unit(
        consumer,
        [
            PRODUCTION_LIBRARY,
            FAILURE_LIBRARY,
            PRODUCTION_TIMING,
            PRODUCTION_FROZEN,
        ],
        (
            "-Wl,--whole-archive",
        ),
    )
    assert linked.returncode != 0
    assert executed is None
    assert "multiple definition" in linked.stderr.lower()


def test_stripped_rearchive_still_rejects_issuer_redefinition_ld_r(
) -> None:
    compiler = shutil.which("c++")
    archiver = shutil.which("ar")
    objcopy = shutil.which("objcopy")
    linker = shutil.which("ld")
    assert compiler is not None
    assert archiver is not None
    assert objcopy is not None
    assert linker is not None
    malicious = """
#include "attested_frozen_camera_frame.h"
namespace livo
{
CameraDriverFrameEnvelopeAuthorityCreateResult
CameraDriverFrameEnvelopeAuthority::create(
  const CameraDriverFrameEnvelopeAuthorityConfiguration &) noexcept
{
  return {
    CameraDriverFrameEnvelopeAuthorityCreateStatus::READY,
    nullptr};
}
CameraDriverFrameEnvelopeIssueResult
CameraDriverFrameEnvelopeAuthority::issue(
  const CameraDriverFrameDocument &)
{
  return {
    CameraDriverFrameEnvelopeIssueStatus::READY,
    std::nullopt};
}
AttestedCameraFrameIngestResult
AttestedCameraFrameIngress::ingest(
  const TrustedCameraDriverFrameEnvelope &)
{
  return {
    AttestedCameraFrameIngestStatus::READY,
    std::nullopt,
    std::nullopt,
    std::nullopt,
    std::nullopt};
}
}  // namespace livo
"""
    for library in (PRODUCTION_LIBRARY, FAILURE_LIBRARY):
        with tempfile.TemporaryDirectory(
            prefix="livo-attested-frame-rearchive-"
        ) as directory_text:
            directory = Path(directory_text)
            listed = subprocess.run(
                [archiver, "t", str(library)],
                check=False,
                capture_output=True,
                text=True,
            )
            assert listed.returncode == 0, listed.stderr
            members = [
                line.strip()
                for line in listed.stdout.splitlines()
                if line.strip()
            ]
            assert len(members) == 1, (library, members)
            extracted = subprocess.run(
                [archiver, "x", str(library)],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
            )
            assert extracted.returncode == 0, extracted.stderr
            original_object = directory / members[0]
            stripped_object = directory / "stripped.o"
            stripped = subprocess.run(
                [
                    objcopy,
                    "--strip-debug",
                    str(original_object),
                    str(stripped_object),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert stripped.returncode == 0, stripped.stderr
            rearchive = directory / "rearchive.a"
            archived = subprocess.run(
                [archiver, "rcs", str(rearchive), str(stripped_object)],
                check=False,
                capture_output=True,
                text=True,
            )
            assert archived.returncode == 0, archived.stderr
            assert rearchive.is_file()

            malicious_source = directory / "malicious.cpp"
            malicious_object = directory / "malicious.o"
            malicious_source.write_text(malicious, encoding="utf-8")
            compiled = subprocess.run(
                [
                    compiler,
                    *cmake_flags(PRODUCTION_FLAGS, "CXX_DEFINES"),
                    *cmake_flags(PRODUCTION_FLAGS, "CXX_INCLUDES"),
                    *cmake_flags(PRODUCTION_FLAGS, "CXX_FLAGS"),
                    "-c",
                    str(malicious_source),
                    "-o",
                    str(malicious_object),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert compiled.returncode == 0, compiled.stderr
            combined = subprocess.run(
                [
                    linker,
                    "-r",
                    "--whole-archive",
                    str(rearchive),
                    "--no-whole-archive",
                    str(malicious_object),
                    "-o",
                    str(directory / "combined.o"),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert combined.returncode != 0
            assert "multiple definition" in combined.stderr.lower()


def test_module_targets_are_werror_clean() -> None:
    for flags_file in (PRODUCTION_FLAGS, FAILURE_FLAGS):
        flags = cmake_flags(flags_file, "CXX_FLAGS")
        for required in (
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
        ):
            assert required in flags


def test_expected_binaries_are_present() -> None:
    assert require_artifact(PRODUCTION_TEST).is_file()
    assert require_artifact(BEHAVIOR_TEST).is_file()
