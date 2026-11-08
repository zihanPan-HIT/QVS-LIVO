# 第三方代码与许可证

本仓库保留原作者署名及上游许可，不将所有组件重新标注为同一种许可证。

| 组件 | 来源 | 随附许可 |
|---|---|---|
| `src/livo_core` | [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) 衍生的 ROS 2 / QVS-LIVO adaptive3 实现 | [GPLv2](src/livo_core/LICENSE)；保留文件头中的作者与联系信息 |
| `src/vikit_common`、`src/vikit_ros` | [Vikit](https://github.com/uzh-rpg/rpg_vikit)，含本地 ROS 2 适配 | 各包原有 `package.xml` 标注 GPLv3；随附 [GPLv3 正文](src/vikit_common/LICENSE) |
| `src/livox_ros_driver2/msg` | [Livox ROS Driver 2](https://github.com/Livox-SDK/livox_ros_driver2) | [上游 MIT 许可](src/livox_ros_driver2/LICENSE.txt)；仅收录消息定义 |

根目录 `LICENSE` 为 FAST-LIVO2 衍生核心的上游 GPLv2 正文。原 ROS 2 核心包的 `package.xml` 曾标为 BSD，本次依据原核心许可证与文件头改正为 GPL-2.0。

上述许可版本分别保留；本发布只提供源码，没有为不同 GPL 版本的组合授予额外许可，也不提供重许可声明。分发组合二进制前应核对所用依赖版本与适用许可。

ROS 2、Eigen、PCL、OpenCV、Sophus、Boost、OpenSSL、JsonCpp 和 fmt 通过系统依赖安装，不随本仓库复制其实现。
