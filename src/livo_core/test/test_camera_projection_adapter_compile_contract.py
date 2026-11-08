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
        prefix="livo-camera-projection-contract-"
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


def test_ordinary_adapter_can_expose_a_read_only_structural_revision() -> None:
    result = compile_translation_unit(
        """
#include "cross_camera_patch_migration.h"
#include <memory>
#include <type_traits>

class OrdinaryAdapter final : public livo::CameraProjectionAdapter {
public:
  bool valid() const override { return true; }
  std::uint64_t modelRevision() const noexcept override { return 17U; }
  livo::CameraRayResult unproject(
      const Eigen::Vector2d &) const override {
    livo::CameraRayResult result;
    result.status = livo::ProjectionAdapterStatus::kValid;
    result.direction_camera = Eigen::Vector3d(0.0, 0.0, 1.0);
    return result;
  }
  livo::CameraProjectionResult project(
      const Eigen::Vector3d &) const override {
    livo::CameraProjectionResult result;
    result.status = livo::ProjectionAdapterStatus::kValid;
    result.pixel = Eigen::Vector2d::Zero();
    return result;
  }
  livo::CameraProjectionJacobianResult projectJacobian(
      const Eigen::Vector3d &) const override {
    livo::CameraProjectionJacobianResult result;
    result.status = livo::ProjectionAdapterStatus::kValid;
    result.jacobian.setZero();
    return result;
  }
  bool contains(const Eigen::Vector2d &, int) const override {
    return true;
  }
};

static_assert(!std::is_abstract_v<OrdinaryAdapter>);
static_assert(std::is_same_v<
  decltype(std::declval<const OrdinaryAdapter &>().modelRevision()),
  std::uint64_t>);

int main() {
  auto adapter = std::make_shared<const OrdinaryAdapter>();
  livo::MigrationEndpoint endpoint;
  endpoint.camera_model = adapter;
  endpoint.camera_model_revision = adapter->modelRevision();
  return endpoint.modelRevisionMatchesAdapter() ? 0 : 1;
}
"""
    )
    assert result.returncode == 0, result.stderr


def test_adapter_without_revision_override_remains_abstract() -> None:
    result = compile_translation_unit(
        """
#include "cross_camera_patch_migration.h"
#include <type_traits>

class MissingRevisionAdapter : public livo::CameraProjectionAdapter {
public:
  bool valid() const override { return true; }
  livo::CameraRayResult unproject(
      const Eigen::Vector2d &) const override {
    return {};
  }
  livo::CameraProjectionResult project(
      const Eigen::Vector3d &) const override {
    return {};
  }
  livo::CameraProjectionJacobianResult projectJacobian(
      const Eigen::Vector3d &) const override {
    return {};
  }
  bool contains(const Eigen::Vector2d &, int) const override {
    return true;
  }
};

static_assert(std::is_abstract_v<MissingRevisionAdapter>);
"""
    )
    assert result.returncode == 0, result.stderr


def test_ordinary_consumer_cannot_mutate_pinhole_model_revision() -> None:
    result = compile_translation_unit(
        """
#include "pinhole_camera_projection_adapter.h"

void mutate(livo::PinholeCameraProjectionAdapter & adapter) {
  adapter.modelRevision() = 9U;
}
"""
    )
    assert result.returncode != 0, (
        "ordinary consumer unexpectedly mutated an immutable model revision\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
