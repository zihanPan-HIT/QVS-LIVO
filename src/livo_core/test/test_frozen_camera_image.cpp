#include "frozen_camera_image.h"
#include "frozen_camera_image_failure_injection_test_support.h"

#include <array>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace
{

static_assert(
  !std::is_default_constructible_v<livo::FrozenCameraImage>);
static_assert(
  std::is_copy_constructible_v<livo::FrozenCameraImage>);
static_assert(
  std::is_move_constructible_v<livo::FrozenCameraImage>);

cv::Mat legacyGoldenSource()
{
  cv::Mat source(2, 3, CV_8UC1);
  const std::array<unsigned char, 6U> pixels{{
      1U, 2U, 3U, 4U, 5U, 6U}};
  for (int row = 0; row < source.rows; ++row)
  {
    for (int column = 0; column < source.cols; ++column)
    {
      source.at<unsigned char>(row, column) =
        pixels[static_cast<std::size_t>(
          row * source.cols + column)];
    }
  }
  return source;
}

livo::FrozenCameraImage::Digest legacyGoldenDigest()
{
  return {{
      0xcfU, 0x1aU, 0xc5U, 0x94U, 0x0fU, 0x11U, 0x25U, 0xd0U,
      0xc1U, 0x7dU, 0xdcU, 0x30U, 0xe0U, 0x74U, 0x78U, 0xa3U,
      0x57U, 0x04U, 0xbfU, 0x50U, 0x24U, 0xa2U, 0x86U, 0x57U,
      0x55U, 0x57U, 0xd9U, 0xf9U, 0xf0U, 0x11U, 0x35U, 0x31U}};
}

TEST(FrozenCameraImageTest, FreezesContinuousImageWithLegacyGoldenDigest)
{
  const cv::Mat source = legacyGoldenSource();

  const livo::FrozenCameraImageFreezeResult result =
    livo::FrozenCameraImage::freeze(source);

  EXPECT_EQ(
    result.status,
    livo::FrozenCameraImageFreezeStatus::READY);
  ASSERT_TRUE(result.ready());
  ASSERT_TRUE(result.image.has_value());
  EXPECT_TRUE(result.image->valid());
  EXPECT_EQ(result.image->rows(), 2);
  EXPECT_EQ(result.image->columns(), 3);
  EXPECT_EQ(result.image->type(), CV_8UC1);
  EXPECT_EQ(result.image->digest(), legacyGoldenDigest());
}

TEST(
  FrozenCameraImageTest,
  ReportsStorageResourceFailureWithoutPublishingPartialValueAndAllowsRetry)
{
  const cv::Mat source = legacyGoldenSource();
  livo::test::failNextFrozenCameraImageStorageAllocation();

  const livo::FrozenCameraImageFreezeResult failed =
    livo::FrozenCameraImage::freeze(source);

  EXPECT_EQ(
    failed.status,
    livo::FrozenCameraImageFreezeStatus::RESOURCE_FAILURE);
  EXPECT_FALSE(failed.ready());
  EXPECT_FALSE(failed.image.has_value());

  const livo::FrozenCameraImageFreezeResult retried =
    livo::FrozenCameraImage::freeze(source);
  EXPECT_EQ(
    retried.status,
    livo::FrozenCameraImageFreezeStatus::READY);
  EXPECT_TRUE(retried.ready());
  EXPECT_TRUE(retried.image.has_value());
  EXPECT_EQ(retried.image->digest(), legacyGoldenDigest());
}

TEST(FrozenCameraImageTest, RejectsEmptyAndNonTwoDimensionalInputs)
{
  const livo::FrozenCameraImageFreezeResult empty =
    livo::FrozenCameraImage::freeze(cv::Mat{});
  EXPECT_EQ(
    empty.status,
    livo::FrozenCameraImageFreezeStatus::INVALID_INPUT);
  EXPECT_FALSE(empty.ready());
  EXPECT_FALSE(empty.image.has_value());

  const std::array<int, 3U> sizes{{2, 2, 2}};
  const cv::Mat three_dimensional(
    static_cast<int>(sizes.size()), sizes.data(),
    CV_8UC1, cv::Scalar(3U));
  const livo::FrozenCameraImageFreezeResult non_two_dimensional =
    livo::FrozenCameraImage::freeze(three_dimensional);
  EXPECT_EQ(
    non_two_dimensional.status,
    livo::FrozenCameraImageFreezeStatus::INVALID_INPUT);
  EXPECT_FALSE(non_two_dimensional.ready());
  EXPECT_FALSE(non_two_dimensional.image.has_value());
}

TEST(
  FrozenCameraImageTest,
  HashesNonContinuousRegionByLogicalRowsWithLegacyGoldenDigest)
{
  cv::Mat backing(3, 5, CV_8UC1, cv::Scalar(0U));
  cv::Mat roi = backing(cv::Rect(1, 1, 3, 2));
  ASSERT_FALSE(roi.isContinuous());
  legacyGoldenSource().copyTo(roi);

  const livo::FrozenCameraImageFreezeResult result =
    livo::FrozenCameraImage::freeze(roi);

  ASSERT_TRUE(result.ready());
  ASSERT_TRUE(result.image.has_value());
  EXPECT_EQ(result.image->rows(), 2);
  EXPECT_EQ(result.image->columns(), 3);
  EXPECT_EQ(result.image->type(), CV_8UC1);
  EXPECT_EQ(result.image->digest(), legacyGoldenDigest());
}

TEST(
  FrozenCameraImageTest,
  CopyAndMovePreserveValueWhileMovedFromObjectIsExplicitlyInvalid)
{
  cv::Mat source = legacyGoldenSource();
  const livo::FrozenCameraImageFreezeResult result =
    livo::FrozenCameraImage::freeze(source);
  ASSERT_TRUE(result.ready());
  ASSERT_TRUE(result.image.has_value());

  livo::FrozenCameraImage copied = *result.image;
  source.setTo(cv::Scalar(255U));
  EXPECT_TRUE(copied.valid());
  EXPECT_EQ(copied.rows(), 2);
  EXPECT_EQ(copied.columns(), 3);
  EXPECT_EQ(copied.type(), CV_8UC1);
  EXPECT_EQ(copied.digest(), legacyGoldenDigest());

  livo::FrozenCameraImage moved = std::move(copied);
  EXPECT_TRUE(moved.valid());
  EXPECT_EQ(moved.digest(), legacyGoldenDigest());
  EXPECT_FALSE(copied.valid());
  EXPECT_EQ(copied.rows(), 0);
  EXPECT_EQ(copied.columns(), 0);
  EXPECT_EQ(
    copied.digest(),
    livo::FrozenCameraImage::Digest{});
}

}  // namespace
