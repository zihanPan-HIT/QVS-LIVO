#include "structured_metric.h"

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <limits.h>
#include <string>

#include <unistd.h>

namespace livo
{

namespace
{

// Standalone policy/VIO tests historically emit metrics without a mapper.
// The production mapper applies its explicit parameter before processing.
std::atomic<bool> structured_metric_enabled{true};

}  // namespace

void setStructuredMetricEnabled(const bool enabled) noexcept
{
  structured_metric_enabled.store(enabled, std::memory_order_relaxed);
}

bool structuredMetricEnabled() noexcept
{
  return structured_metric_enabled.load(std::memory_order_relaxed);
}

bool emitStructuredMetric(const char * format, ...)
{
  if (!structuredMetricEnabled() || format == nullptr) {
    return false;
  }

  va_list arguments;
  va_start(arguments, format);
  va_list length_arguments;
  va_copy(length_arguments, arguments);
  const int required_size =
    std::vsnprintf(nullptr, 0, format, length_arguments);
  va_end(length_arguments);
  if (required_size <= 0 ||
      static_cast<unsigned long>(required_size) >=
        static_cast<unsigned long>(PIPE_BUF))
  {
    va_end(arguments);
    return false;
  }

  std::string line(
    static_cast<std::size_t>(required_size) + 1U, '\0');
  const int formatted_size =
    std::vsnprintf(line.data(), line.size(), format, arguments);
  va_end(arguments);
  if (formatted_size != required_size) {
    return false;
  }
  line.resize(static_cast<std::size_t>(required_size));

  ssize_t write_result;
  do
  {
    write_result =
      ::write(STDOUT_FILENO, line.data(), line.size());
  }
  while (write_result < 0 && errno == EINTR);
  return
    write_result == static_cast<ssize_t>(line.size());
}

}  // namespace livo
