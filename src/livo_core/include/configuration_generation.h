#ifndef LIVO_CONFIGURATION_GENERATION_H_
#define LIVO_CONFIGURATION_GENERATION_H_

#include "checked_increment.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace livo
{

class CheckedConfigurationGeneration;

class ConfigurationGenerationReceipt
{
public:
  ConfigurationGenerationReceipt(
    const ConfigurationGenerationReceipt &) = default;
  ConfigurationGenerationReceipt(
    ConfigurationGenerationReceipt &&) = default;
  ConfigurationGenerationReceipt & operator=(
    const ConfigurationGenerationReceipt &) = default;
  ConfigurationGenerationReceipt & operator=(
    ConfigurationGenerationReceipt &&) = default;

  std::uint64_t value() const
  {
    return value_;
  }

private:
  ConfigurationGenerationReceipt(
    std::weak_ptr<const void> authority,
    const std::uint64_t value)
  : authority_(std::move(authority)), value_(value)
  {
  }

  std::weak_ptr<const void> authority_;
  std::uint64_t value_{0U};

  friend class CheckedConfigurationGeneration;
};

// Monotonic fail-closed generation authority. Receipts are bound to an
// independent lifetime control block as well as its checked generation value.
// Moving the authority owner intentionally exhausts both owner objects.
class CheckedConfigurationGeneration
{
public:
  CheckedConfigurationGeneration()
  : authority_(std::make_shared<const Authority>())
  {
  }

  CheckedConfigurationGeneration(
    const CheckedConfigurationGeneration &) = delete;
  CheckedConfigurationGeneration & operator=(
    const CheckedConfigurationGeneration &) = delete;

  CheckedConfigurationGeneration(
    CheckedConfigurationGeneration && other) noexcept
  : value_(other.value_), exhausted_(true)
  {
    other.failClosed();
  }

  CheckedConfigurationGeneration & operator=(
    CheckedConfigurationGeneration && other) noexcept
  {
    if (this == &other)
    {
      failClosed();
      return *this;
    }
    failClosed();
    value_ = other.value_;
    other.failClosed();
    return *this;
  }

  bool advance()
  {
    if (
      exhausted_ ||
      authority_ == nullptr ||
      !checkedIncrement(value_))
    {
      failClosed();
      return false;
    }
    return true;
  }

  std::optional<ConfigurationGenerationReceipt> receipt() const
  {
    if (
      exhausted_ ||
      authority_ == nullptr ||
      value_ == 0U)
    {
      return std::nullopt;
    }
    return ConfigurationGenerationReceipt(authority_, value_);
  }

  bool accepts(
    const ConfigurationGenerationReceipt & receipt) const
  {
    const std::shared_ptr<const void> receipt_authority =
      receipt.authority_.lock();
    return
      !exhausted_ &&
      authority_ != nullptr &&
      receipt_authority != nullptr &&
      value_ != 0U &&
      sameAuthority(authority_, receipt_authority) &&
      receipt.value_ == value_;
  }

  std::uint64_t value() const
  {
    return value_;
  }

  bool exhausted() const
  {
    return exhausted_;
  }

  void failClosed()
  {
    exhausted_ = true;
    authority_.reset();
  }

private:
  struct Authority final
  {
  };

  static bool sameAuthority(
    const std::shared_ptr<const void> & left,
    const std::shared_ptr<const void> & right)
  {
    return
      !left.owner_before(right) &&
      !right.owner_before(left);
  }

  std::shared_ptr<const void> authority_;
  std::uint64_t value_{0U};
  bool exhausted_{false};
};

}  // namespace livo

#endif  // LIVO_CONFIGURATION_GENERATION_H_
