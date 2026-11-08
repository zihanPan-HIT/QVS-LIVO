# QVS-LIVO

基于 FAST-LIVO2 的 ROS 2 激光–惯性–视觉定位后端，采用 **adaptive3 四相机自适应视觉更新**。各相机维护自己的跟踪状态，共享激光／IMU 估计状态；依据观测质量选择视觉更新，在视觉观测不可用时继续执行 LIO。

此仓库发布实车算法主线，包含定位、融合、时间戳保留的相机输入整形和必要依赖。定位使用的局部几何地图属于估计器内部状态。网页地图、RViz 配置、语义分割、2.5D 地形／风险建图、导航控制、仿真 GPS 后端及实验数据不在本仓库内。

## 环境

- Ubuntu 22.04、ROS 2 Humble、C++17。
- Eigen、PCL、OpenCV、Sophus、Boost、OpenSSL、JsonCpp、fmt。
- 默认配置示例：AIRY 激光雷达、IMU 和 front/left/back/right 四相机。

在已安装 ROS 2 Humble 的系统中安装构建依赖：

```bash
sudo apt update
sudo apt install build-essential cmake python3-colcon-common-extensions \
  python3-pytest python3-yaml libeigen3-dev libpcl-dev libopencv-dev \
  libboost-thread-dev libssl-dev libjsoncpp-dev libfmt-dev \
  ros-humble-ros-base ros-humble-pcl-ros ros-humble-pcl-conversions \
  ros-humble-cv-bridge ros-humble-image-transport ros-humble-sophus \
  ros-humble-tf2-geometry-msgs ros-humble-visualization-msgs \
  ros-humble-rosidl-default-generators ros-humble-ament-cmake-gtest \
  ros-humble-ament-cmake-pytest
```

## 构建

建议将工作区放在不含空格和非 ASCII 字符的路径中；Humble 的接口生成工具在部分版本上不能正确读取中文路径。

```bash
git clone https://github.com/zihanPan-HIT/QVS-LIVO.git
cd QVS-LIVO
source /opt/ros/humble/setup.bash
export CMAKE_BUILD_PARALLEL_LEVEL=2
colcon build --parallel-workers 2 --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DLIVO_BUILD_EXPERIMENTAL_SHADOWS=OFF
source install/setup.bash
```

工作区包含 `livo_core`、ROS 2 适配的 `vikit_common` / `vikit_ros`，以及精简的 `livox_ros_driver2` **消息定义包**。消息包不启动设备、不依赖 Livox SDK；如使用实际 Livox 设备，在驱动工作区使用官方完整驱动，避免将两个同名包放入同一构建工作区。AIRY 驱动需单独启动。

## 启动

```bash
source install/setup.bash
ros2 launch livo_core qvs_livo.launch.py
```

默认入口启动定位节点和 10 Hz 相机输入整形节点，不启动图形界面。输入整形保留原始传感器时间戳，在 35 ms 批次跨度内选取图像，并允许缺失相机时其余相机继续供应。

| 输入 | 默认话题 | 消息 |
|---|---|---|
| 激光 | `/rslidar_points` | `sensor_msgs/PointCloud2` |
| IMU | `/rslidar_imu_data` | `sensor_msgs/Imu` |
| 前相机 | `/camera/camera/color/image_raw` | `sensor_msgs/Image` |
| 左相机 | `/left/camera/color/image_raw` | `sensor_msgs/Image` |
| 后相机 | `/back/camera/color/image_raw` | `sensor_msgs/Image` |
| 右相机 | `/right/camera/color/image_raw` | `sensor_msgs/Image` |

相机输入整形后发布到 `/qvs/input/{front,left,back,right}/image`。已有等价输入时可用 `camera_gate:=false` 关闭该节点，并通过配置指定相机话题。

```bash
ros2 launch livo_core qvs_livo.launch.py \
  config:=/absolute/path/to/your_calibration.yaml camera_gate:=false
```

示例标定文件为 [`qvs_livo_airy_four.yaml`](src/livo_core/config/qvs_livo_airy_four.yaml)。**运行前按自己的传感器修改内参、外参、时间偏移、图像尺寸和激光格式。** 所有传感器应使用一致的时钟基准；输入整形节点不校正传感器时钟。示例采用 96 线 AIRY 配置，不能直接代替其他雷达的标定。

该配置明确启用 `vio.camera_mode: four`、`vio.adaptive.enable: true` 和部分相机批次支持。`multi_camera.enable: false` 关闭的是附加四路点云着色，不会关闭四相机定位。持久全局地图、PCD 和 COLMAP 导出默认关闭。

主要输出为 `/aft_mapped_to_init` 里程计、TF 和 `/livo/runtime_ready` 就绪信号。保留算法原有的 ROS 点云及诊断接口，便于接入其他系统。

回放数据时使用仿真时钟，并先启动后端、等待就绪，再播放 bag：

```bash
ros2 launch livo_core qvs_livo.launch.py use_sim_time:=true
# 在另一个终端，确认 /livo/runtime_ready 为 true 后：
ros2 bag play /absolute/path/to/bag --clock
```

## 测试

```bash
source install/setup.bash
colcon test --packages-select livo_core --event-handlers console_direct+
colcon test-result --verbose
```

测试覆盖自适应调度、观测队列、视觉更新权限、IMU 传播、几何及配置契约。历史实验报告解析器和依赖私有数据的实验运行脚本不随此后端发布。实际发布检查见 [验证记录](docs/VALIDATION.md)。

## 版本与许可

源码来自实车 adaptive3 主线及其 2026-10-05 部署修复，来源与发布整理差异见 [源码说明](docs/SOURCE.md)。本次发布保留估计器的更新逻辑，不混入仿真 GPS／轮速实验分支。

FAST-LIVO2 衍生核心保留上游 GPLv2 许可和作者声明；Vikit 和 Livox 消息定义使用各自的上游许可证。具体文件归属见 [第三方说明](THIRD_PARTY_NOTICES.md)，各子包中的许可证优先适用于相应代码。
