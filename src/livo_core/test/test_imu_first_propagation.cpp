#include "IMU_Processing.h"

#include <cmath>
#include <memory>

#include <gtest/gtest.h>

namespace
{

sensor_msgs::msg::Imu::SharedPtr imuSample(
  const double timestamp, const double acceleration_x)
{
  auto sample = std::make_shared<sensor_msgs::msg::Imu>();
  sample->header.stamp = sec2Stamp(timestamp);
  sample->linear_acceleration.x = acceleration_x;
  sample->linear_acceleration.z = -G_m_s2;
  return sample;
}

PointCloudXYZI::Ptr scanForPropagation()
{
  auto scan = std::make_shared<PointCloudXYZI>();
  PointType begin;
  begin.x = 1.0F;
  begin.normal_x = 0.0F;
  begin.normal_y = 0.0F;
  begin.normal_z = 0.0F;
  begin.curvature = 0.0F;
  scan->push_back(begin);

  PointType end = begin;
  end.x = 2.0F;
  end.curvature = 100.0F;
  scan->push_back(end);
  return scan;
}

TEST(ImuFirstPropagationTest, AnchorsPropagationAtInitializationFrameEnd)
{
  ImuProcess process;
  process.lidar_type = ROBOSENSE;
  process.set_imu_init_frame_num(1);
  StatesGroup state;
  auto output = std::make_shared<PointCloudXYZI>();

  LidarMeasureGroup initialization;
  initialization.lio_vio_flg = LIO;
  MeasureGroup initialization_measurement;
  initialization_measurement.lio_time = 10.15;
  initialization_measurement.imu.push_back(imuSample(10.10, 0.0));
  initialization.measures.push_back(initialization_measurement);
  process.Process2(initialization, state, output);
  ASSERT_FALSE(process.imu_need_init);

  LidarMeasureGroup propagation;
  propagation.lio_vio_flg = LIO;
  propagation.pcl_proc_cur = scanForPropagation();
  MeasureGroup propagation_measurement;
  propagation_measurement.lio_time = 10.25;
  propagation_measurement.imu.push_back(imuSample(10.20, 1.0));
  propagation.measures.push_back(propagation_measurement);
  process.Process2(propagation, state, output);

  EXPECT_NEAR(state.vel_end.x(), 0.025, 1.0e-9);
  EXPECT_NEAR(state.pos_end.x(), 0.000625, 1.0e-9);
  EXPECT_TRUE(state.vel_end.allFinite());
  EXPECT_TRUE(state.pos_end.allFinite());
  EXPECT_TRUE(state.rot_end.allFinite());
  EXPECT_FALSE(output->empty());
}

}  // namespace
