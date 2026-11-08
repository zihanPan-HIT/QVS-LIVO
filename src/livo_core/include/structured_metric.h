#ifndef LIVO_STRUCTURED_METRIC_H_
#define LIVO_STRUCTURED_METRIC_H_

namespace livo
{

void setStructuredMetricEnabled(bool enabled) noexcept;
bool structuredMetricEnabled() noexcept;

// Formats one bounded metric line and emits it with exactly one write(2).
// Lines at or above PIPE_BUF are rejected to preserve pipe atomicity.
bool emitStructuredMetric(const char * format, ...);

}  // namespace livo

#endif  // LIVO_STRUCTURED_METRIC_H_
