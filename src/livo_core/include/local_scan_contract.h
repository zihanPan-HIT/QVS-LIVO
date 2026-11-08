#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>

namespace livo {
// AIRY 原生逐点秒时间；保留 header 整型身份，末点不依赖几何滤波结果。
inline std::optional<int64_t> nativeScanEnd(const sensor_msgs::msg::PointCloud2 & msg,
                                          bool allow_instantaneous = false)
{
  const sensor_msgs::msg::PointField * field = nullptr;
  for (const auto & f : msg.fields) if (f.name == "timestamp") field = &f;
  if (!field || field->datatype != sensor_msgs::msg::PointField::FLOAT64 ||
      msg.is_bigendian || msg.width == 0 || msg.height == 0 ||
      field->offset + sizeof(double) > msg.point_step ||
      msg.row_step < uint64_t(msg.width) * msg.point_step ||
      msg.data.size() < uint64_t(msg.row_step) * msg.height) return std::nullopt;
  double first = 0.0, last = 0.0, minimum = 0.0;
  std::memcpy(&first, msg.data.data() + field->offset, sizeof(double));
  if (!std::isfinite(first)) return std::nullopt;
  last = minimum = first;
  for (uint32_t row = 0; row < msg.height; ++row)
    for (uint32_t col = 0; col < msg.width; ++col) {
      double time;
      std::memcpy(&time, msg.data.data() + size_t(row) * msg.row_step +
        size_t(col) * msg.point_step + field->offset, sizeof(double));
      if (!std::isfinite(time)) return std::nullopt;
      minimum = std::min(minimum, time);
      last = std::max(last, time);
    }
  // AIRY 点按通道打包，正常扫描可能有早于首元素约数百微秒的点。
  // 跨度用 min/max 校验；原始首点锚不随排序变化，不能把 max-min 加到 header。
  // PhysX瞬时帧需显式启用；原生实车默认仍拒绝零时长，不伪造点时间。
  if (last - minimum > 1.0 ||
      (last <= first && !(allow_instantaneous && last == minimum))) return std::nullopt;
  return int64_t(msg.header.stamp.sec) * 1000000000LL + msg.header.stamp.nanosec +
    int64_t(std::llround((last - first) * 1e9));
}
}  // namespace livo
