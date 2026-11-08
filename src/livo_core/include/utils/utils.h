#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

inline std::vector<int> convertToIntVectorSafe(const std::vector<int64_t> &int64_vector)
{
  std::vector<int> int_vector;
  int_vector.reserve(int64_vector.size());
  for (const int64_t value : int64_vector)
  {
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
    {
      throw std::out_of_range("Value is out of range for int");
    }
    int_vector.push_back(static_cast<int>(value));
  }
  return int_vector;
}

inline double stamp2Sec(const builtin_interfaces::msg::Time &stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

inline rclcpp::Time sec2Stamp(double timestamp)
{
  const int32_t sec = static_cast<int32_t>(std::floor(timestamp));
  const uint32_t nanosec = static_cast<uint32_t>((timestamp - std::floor(timestamp)) * 1e9);
  return rclcpp::Time(sec, nanosec);
}

namespace tf
{
inline geometry_msgs::msg::Quaternion createQuaternionMsgFromYaw(double yaw)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(q);
}

inline geometry_msgs::msg::Quaternion createQuaternionMsgFromRollPitchYaw(double roll, double pitch, double yaw)
{
  tf2::Quaternion q;
  q.setRPY(roll, pitch, yaw);
  return tf2::toMsg(q);
}
}  // namespace tf

inline geometry_msgs::msg::TransformStamped createTransformStamped(
  const tf2::Transform &transform,
  const builtin_interfaces::msg::Time &stamp,
  const std::string &frame_id,
  const std::string &child_frame_id)
{
  geometry_msgs::msg::TransformStamped transform_stamped;
  transform_stamped.header.stamp = stamp;
  transform_stamped.header.frame_id = frame_id;
  transform_stamped.child_frame_id = child_frame_id;
  transform_stamped.transform = tf2::toMsg(transform);
  return transform_stamped;
}

#endif  // UTILS_H
