#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)


def make_transform(parent, child, xyz, yaw=0.0):
    msg = TransformStamped()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(xyz[0])
    msg.transform.translation.y = float(xyz[1])
    msg.transform.translation.z = float(xyz[2])
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    msg.transform.rotation.x = qx
    msg.transform.rotation.y = qy
    msg.transform.rotation.z = qz
    msg.transform.rotation.w = qw
    return msg


def main():
    rclpy.init()
    node = rclpy.create_node("indoor_hall_static_tf_ros2")
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform([
        make_transform("camera_init", "aft_mapped", (0.0, 0.0, 0.0)),
        make_transform("aft_mapped", "base_link", (0.0, 0.0, -0.32)),
        make_transform("base_link", "livox_frame", (0.0, 0.0, 0.32)),
    ])
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
