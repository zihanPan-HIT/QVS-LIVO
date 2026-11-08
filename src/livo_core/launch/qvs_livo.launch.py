"""启动无界面的 QVS-LIVO 四相机定位节点。"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=PathJoinSubstitution([
            FindPackageShare('livo_core'), 'config', 'qvs_livo_airy_four.yaml'])),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('camera_gate', default_value='true'),
        Node(package='livo_core', executable='camera_rate_gate.py',
             name='qvs_camera_rate_gate', output='screen',
             condition=IfCondition(LaunchConfiguration('camera_gate')),
             parameters=[{'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}]),
        Node(package='livo_core', executable='livo_mapping',
             name='qvs_livo', output='screen',
             additional_env={'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'OMP_WAIT_POLICY': 'PASSIVE'},
             parameters=[
                 LaunchConfiguration('config'),
                 {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}]),
    ])
