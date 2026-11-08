#include "attested_frozen_camera_frame.h"

#include <gtest/gtest.h>

namespace livo
{
namespace
{

TEST(
  AttestedFrozenCameraFrameProductionTest,
  EnvelopeAuthorityIsUnavailableWithoutDriverAdapter)
{
  const CameraDriverFrameEnvelopeAuthorityConfiguration configuration{};

  auto result =
    CameraDriverFrameEnvelopeAuthority::create(configuration);

  EXPECT_EQ(
    CameraDriverFrameEnvelopeAuthorityCreateStatus::
    DRIVER_CONTRACT_UNAVAILABLE,
    result.status);
  EXPECT_FALSE(result.authority);
  EXPECT_FALSE(result.ready());
}

}  // namespace
}  // namespace livo
