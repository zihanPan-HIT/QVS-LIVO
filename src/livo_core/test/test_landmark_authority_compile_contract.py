import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRECTORY = PACKAGE_ROOT / "include"


def compile_translation_unit(source: str) -> subprocess.CompletedProcess[str]:
    configured_compiler = shlex.split(os.environ.get("CXX", "c++"))
    compiler = shutil.which(configured_compiler[0])
    assert compiler is not None
    with tempfile.TemporaryDirectory(
        prefix="livo-landmark-authority-contract-"
    ) as directory:
        source_path = Path(directory) / "consumer.cpp"
        source_path.write_text(source, encoding="utf-8")
        return subprocess.run(
            [
                compiler,
                *configured_compiler[1:],
                "-std=c++17",
                "-fsyntax-only",
                f"-I{INCLUDE_DIRECTORY}",
                "-isystem",
                "/usr/include/eigen3",
                str(source_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )


def assert_private_contract_rejects(source: str) -> None:
    result = compile_translation_unit(source)
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly obtained landmark authority\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "private" in result.stderr


def test_production_headers_do_not_declare_test_peers() -> None:
    offenders = [
        header.relative_to(INCLUDE_DIRECTORY)
        for header in INCLUDE_DIRECTORY.rglob("*.h")
        if "TestPeer" in header.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_ordinary_peer_cannot_publish_active_plane_identities() -> None:
    assert_private_contract_rejects(
        """
#include "active_plane_identity_registry.h"
namespace livo {
class MapCommitTestPeer {
public:
  static bool publish(ActivePlaneIdentityRegistry & registry) {
    return registry.replaceActivePlanes({});
  }
};
}
"""
    )


def test_ordinary_peer_cannot_mint_commit_tokens() -> None:
    assert_private_contract_rejects(
        """
#include "multi_view_landmark.h"
namespace livo {
class MapCommitTestPeer {
public:
  static LidarAnchoredCommitToken lidar(
      GeometrySnapshotId geometry,
      CommitRevisionSet revisions) {
    return LidarAnchoredCommitToken(geometry, revisions);
  }
  static LidarGeometryRefitCommitToken refit(
      GeometrySnapshotId geometry,
      LioPosteriorRevision revision) {
    return LidarGeometryRefitCommitToken(geometry, revision);
  }
  static AcceptedStateCommitToken state(
      GeometrySnapshotId geometry,
      CommitRevisionSet revisions) {
    return AcceptedStateCommitToken(geometry, revisions);
  }
};
}
"""
    )


def test_ordinary_peer_cannot_mutate_store_counters() -> None:
    assert_private_contract_rejects(
        """
#include "multi_view_landmark.h"
namespace livo {
class MapCommitTestPeer {
public:
  static void setCounters(MultiViewLandmarkStore & store) {
    store.store_revision_.value = 7U;
    store.next_landmark_id_ = 8U;
    store.next_observation_id_ = 9U;
  }
};
}
"""
    )
