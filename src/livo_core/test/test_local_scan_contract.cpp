#include "local_scan_contract.h"
#include <gtest/gtest.h>

sensor_msgs::msg::PointCloud2 nativeCloud()
{
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp.sec=1750000000; msg.header.stamp.nanosec=123456789;
  msg.width=2; msg.height=1; msg.point_step=8; msg.row_step=16;
  sensor_msgs::msg::PointField f; f.name="timestamp"; f.datatype=f.FLOAT64; f.count=1;
  msg.fields.push_back(f); msg.data.resize(16);
  const double points[]{100.0,100.125};
  std::memcpy(msg.data.data(), points, sizeof(points));
  return msg;
}
TEST(LocalScanContract, ExactHeaderLowBitsAndRawTail)
{
  EXPECT_EQ(*livo::nativeScanEnd(nativeCloud()),1750000000248456789LL);
}
TEST(LocalScanContract, InstantaneousRequiresOptInAndPreservesExactHeader)
{
  auto msg=nativeCloud();
  const double points[]{1.0916666984558105,1.0916666984558105};
  std::memcpy(msg.data.data(),points,sizeof(points));
  EXPECT_FALSE(livo::nativeScanEnd(msg));
  ASSERT_TRUE(livo::nativeScanEnd(msg,true));
  EXPECT_EQ(*livo::nativeScanEnd(msg,true),1750000000123456789LL);
  const double invalid[]{100.0,102.0};
  std::memcpy(msg.data.data(),invalid,sizeof(invalid));
  EXPECT_FALSE(livo::nativeScanEnd(msg,true));
  double nan=std::numeric_limits<double>::quiet_NaN();
  std::memcpy(msg.data.data()+8,&nan,8);
  EXPECT_FALSE(livo::nativeScanEnd(msg,true));
}
TEST(LocalScanContract, RejectMissingEmptyMalformedAndNonfinite)
{
  auto msg=nativeCloud(); msg.width=0; EXPECT_FALSE(livo::nativeScanEnd(msg));
  msg=nativeCloud(); msg.fields.clear(); EXPECT_FALSE(livo::nativeScanEnd(msg));
  msg=nativeCloud(); msg.data.resize(8); EXPECT_FALSE(livo::nativeScanEnd(msg));
  msg=nativeCloud(); double nan=std::numeric_limits<double>::quiet_NaN();
  std::memcpy(msg.data.data()+8,&nan,8); EXPECT_FALSE(livo::nativeScanEnd(msg));
}
TEST(LocalScanContract, AcceptChannelUnorderedPointsWithoutShiftingAnchor)
{
  auto msg=nativeCloud();
  msg.width=3; msg.row_step=24; msg.data.resize(24);
  const double times[]{1.0,0.9998,1.0998};
  std::memcpy(msg.data.data(),times,sizeof(times));
  const auto expected=1750000000123456789LL+99800000LL;
  ASSERT_TRUE(livo::nativeScanEnd(msg));
  EXPECT_EQ(*livo::nativeScanEnd(msg),expected);
  const double reordered[]{1.0,1.0998,0.9998};
  std::memcpy(msg.data.data(),reordered,sizeof(reordered));
  EXPECT_EQ(*livo::nativeScanEnd(msg),expected);
}
