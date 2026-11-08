#ifndef LIVO_CHECKED_INCREMENT_H_
#define LIVO_CHECKED_INCREMENT_H_

#include <atomic>
#include <limits>
#include <type_traits>

namespace livo
{

template<typename Unsigned>
bool checkedIncrement(Unsigned & value)
{
  static_assert(
    std::is_integral_v<Unsigned> &&
    std::is_unsigned_v<Unsigned>);
  if (
    value ==
    std::numeric_limits<Unsigned>::max())
  {
    return false;
  }
  ++value;
  return value != 0U;
}

template<typename Unsigned>
bool checkedIncrementNonzero(Unsigned & value)
{
  static_assert(
    std::is_integral_v<Unsigned> &&
    std::is_unsigned_v<Unsigned>);
  if (
    value == 0U ||
    value == std::numeric_limits<Unsigned>::max())
  {
    return false;
  }
  ++value;
  return true;
}

template<typename Unsigned>
bool checkedSuccessor(
  const Unsigned value,
  Unsigned & successor)
{
  successor = value;
  return checkedIncrement(successor);
}

template<typename Unsigned>
bool checkedReserve(
  Unsigned & next,
  Unsigned & reserved) noexcept
{
  static_assert(
    std::is_integral_v<Unsigned> &&
    std::is_unsigned_v<Unsigned>);
  if (
    next == 0U ||
    next == std::numeric_limits<Unsigned>::max())
  {
    return false;
  }
  reserved = next;
  ++next;
  return true;
}

template<typename Unsigned>
bool checkedReserve(
  std::atomic<Unsigned> & next,
  Unsigned & reserved) noexcept
{
  static_assert(
    std::is_integral_v<Unsigned> &&
    std::is_unsigned_v<Unsigned>);
  Unsigned current = next.load(std::memory_order_relaxed);
  while (
    current != 0U &&
    current != std::numeric_limits<Unsigned>::max())
  {
    if (next.compare_exchange_weak(
        current, static_cast<Unsigned>(current + 1U),
        std::memory_order_relaxed,
        std::memory_order_relaxed))
    {
      reserved = current;
      return true;
    }
  }
  return false;
}

}  // namespace livo

#endif  // LIVO_CHECKED_INCREMENT_H_
