#include "voxel_map.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

namespace
{

std::atomic<int> allocations_until_failure{-1};
std::atomic<bool> injected_failure{false};

bool consumeAllocationFailure() noexcept
{
  int remaining = allocations_until_failure.load();
  while (remaining >= 0)
  {
    if (allocations_until_failure.compare_exchange_weak(remaining, remaining - 1))
    {
      if (remaining == 0) injected_failure.store(true);
      return remaining == 0;
    }
  }
  return false;
}

void * allocateBytes(
  const std::size_t size,
  const std::size_t alignment = alignof(std::max_align_t))
{
  if (consumeAllocationFailure())
  {
    throw std::bad_alloc();
  }
  void * allocation = nullptr;
  const std::size_t nonzero_size = size == 0U ? 1U : size;
  if (alignment <= alignof(std::max_align_t))
  {
    allocation = std::malloc(nonzero_size);
  }
  else if (
    posix_memalign(
      &allocation, alignment, nonzero_size) != 0)
  {
    allocation = nullptr;
  }
  if (allocation == nullptr)
  {
    throw std::bad_alloc();
  }
  return allocation;
}


}  // namespace

void * operator new(const std::size_t size)
{
  return allocateBytes(size);
}

void * operator new[](const std::size_t size)
{
  return allocateBytes(size);
}

void * operator new(
  const std::size_t size, const std::align_val_t alignment)
{
  return allocateBytes(
    size, static_cast<std::size_t>(alignment));
}

void * operator new[](
  const std::size_t size, const std::align_val_t alignment)
{
  return allocateBytes(
    size, static_cast<std::size_t>(alignment));
}

void operator delete(void * allocation) noexcept
{
  std::free(allocation);
}

void operator delete[](void * allocation) noexcept
{
  std::free(allocation);
}

void operator delete(
  void * allocation, const std::size_t) noexcept
{
  std::free(allocation);
}

void operator delete[](
  void * allocation, const std::size_t) noexcept
{
  std::free(allocation);
}

void operator delete(
  void * allocation, const std::align_val_t) noexcept
{
  std::free(allocation);
}

void operator delete[](
  void * allocation, const std::align_val_t) noexcept
{
  std::free(allocation);
}

void operator delete(
  void * allocation, const std::size_t,
  const std::align_val_t) noexcept
{
  std::free(allocation);
}

void operator delete[](
  void * allocation, const std::size_t,
  const std::align_val_t) noexcept
{
  std::free(allocation);
}

namespace
{

struct ResidualFixture
{
  VoxelMapManager manager;
  std::vector<std::unique_ptr<VoxelOctoTree>> roots;

  VoxelOctoTree * node(const double x, const std::uint64_t id, const bool planar = true)
  {
    auto * tree = new VoxelOctoTree(2, 0, 5, 100, 0.01F);
    tree->voxel_center_[0] = x + 0.5;
    tree->voxel_center_[1] = 0.5;
    tree->voxel_center_[2] = 0.0;
    tree->quater_length_ = 0.25F;
    auto & plane = *tree->plane_ptr_;
    plane.is_plane_ = planar;
    plane.center_ = Eigen::Vector3d(x + 0.5, 0.5, 0.0);
    plane.normal_ = Eigen::Vector3d::UnitZ();
    plane.radius_ = 2.0F;
    plane.plane_var_ = Eigen::Matrix<double, 6, 6>::Identity() * 0.0001;
    plane.geometry_version_ = {livo::PlaneIdentity{id, 2U}, 3U};
    plane.identity_basis_.origin_world_m = plane.center_;
    plane.identity_basis_.u_axis_world = Eigen::Vector3d::UnitX();
    plane.identity_basis_.v_axis_world = Eigen::Vector3d::UnitY();
    return tree;
  }

  void insert(const int x, VoxelOctoTree * tree)
  {
    roots.emplace_back(tree);
    manager.voxel_map_.emplace(VOXEL_LOCATION(x, 0, 0), tree);
  }

  ResidualFixture()
  {
    manager.config_setting_ = {};
    manager.config_setting_.max_voxel_size_ = 1.0;
    manager.config_setting_.max_layer_ = 2;
    manager.config_setting_.sigma_num_ = 3.0;
    insert(0, node(0, 10U));
    insert(20, node(20, 20U, false));
    insert(21, node(20, 21U));
    auto * root = node(30, 30U, false);
    root->leaves_[0] = node(30, 31U, false);
    root->leaves_[0]->leaves_[0] = node(30, 32U);
    root->leaves_[0]->leaves_[7] = node(30, 33U);
    insert(30, root);
  }

  std::vector<pointWithVar> points(const std::size_t count)
  {
    std::vector<pointWithVar> result(count);
    for (std::size_t i = 0; i < count; ++i)
    {
      const double x = i % 4U == 0U ? 0.4 :
        i % 4U == 1U ? 10.4 : i % 4U == 2U ? 20.9 : 30.4;
      result[i].point_w = Eigen::Vector3d(x, 0.4, 0.02);
      result[i].point_b = Eigen::Vector3d(static_cast<double>(i), 0.2, 0.1);
      result[i].body_var = Eigen::Matrix3d::Identity() * 0.01;
      result[i].var = Eigen::Matrix3d::Identity() * 0.01;
    }
    return result;
  }

  void expectCorrespondences(
    const std::vector<pointWithVar> & input,
    const std::vector<PointToPlane> & residuals)
  {
    std::size_t output = 0U;
    for (std::size_t i = 0; i < input.size(); ++i)
    {
      const auto & point = input[i];
      if (i % 4U == 1U)
      {
        EXPECT_FALSE(point.plane_support_valid);
        EXPECT_EQ(point.plane_id, 0U);
        continue;
      }
      ASSERT_LT(output, residuals.size());
      const auto & residual = residuals[output++];
      EXPECT_EQ(residual.point_b_.x(), static_cast<double>(i));
      EXPECT_EQ(residual.point_b_, point.point_b);
      EXPECT_EQ(residual.point_w_, point.point_w);
      EXPECT_EQ(residual.normal_, Eigen::Vector3d::UnitZ());
      EXPECT_EQ(residual.dis_to_plane_, static_cast<float>(0.02));
      EXPECT_EQ(residual.d_, 0.0);
      EXPECT_EQ(residual.body_cov_, point.body_var);
      EXPECT_EQ(residual.plane_var_, (Eigen::Matrix<double, 6, 6>::Identity() * 0.0001));
      EXPECT_TRUE(point.plane_support_valid);
      EXPECT_EQ(point.plane_id, i % 4U == 0U ? 10U : i % 4U == 2U ? 21U : 32U);
      EXPECT_EQ(point.plane_epoch, 2U);
      EXPECT_EQ(point.plane_revision, 3U);
      EXPECT_EQ(point.plane_level, i % 4U == 3U ? 2U : 0U);
      EXPECT_EQ(residual.layer_, point.plane_level);
      EXPECT_EQ(point.normal, residual.normal_);
      EXPECT_EQ(point.plane_identity_origin, residual.center_);
      EXPECT_EQ(point.plane_identity_u_axis, Eigen::Vector3d::UnitX());
      EXPECT_EQ(point.plane_identity_v_axis, Eigen::Vector3d::UnitY());
      EXPECT_EQ(point.plane_covariance, residual.plane_var_);
    }
    EXPECT_EQ(output, residuals.size());
    EXPECT_EQ(manager.voxel_map_.size(), 4U);
  }
};

TEST(ResidualCorrespondenceTest, EmptySmallAndBothPartitionsPreserveSupportAndInputOrder)
{
  ResidualFixture fixture;
  for (const std::size_t count : {0U, 1U, 17U, 1024U, 4097U})
  {
    SCOPED_TRACE(count);
    auto input = fixture.points(count);
    std::vector<PointToPlane> output;
    fixture.manager.BuildResidualListOMP(input, output);
    fixture.expectCorrespondences(input, output);
  }
}

TEST(ResidualCorrespondenceTest, ScratchAllocationFailureLeavesNoPartialOutputAndCanBeRetried)
{
  ResidualFixture fixture;
  auto input = fixture.points(4097U);
  std::vector<PointToPlane> output;
  output.reserve(input.size());
  bool allocation_escaped = false;
  injected_failure.store(false);
  // Fail the scratch vector allocation before any correspondence is written.
  allocations_until_failure.store(0);
  try
  {
    fixture.manager.BuildResidualListOMP(input, output);
  }
  catch (const std::bad_alloc &)
  {
    allocation_escaped = true;
  }
  allocations_until_failure.store(-1);
  EXPECT_TRUE(injected_failure.load());
  ASSERT_TRUE(allocation_escaped);
  EXPECT_TRUE(output.empty());
  fixture.manager.BuildResidualListOMP(input, output);
  fixture.expectCorrespondences(input, output);
}

}  // namespace
