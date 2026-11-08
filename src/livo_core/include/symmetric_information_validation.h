#ifndef LIVO_SYMMETRIC_INFORMATION_VALIDATION_H_
#define LIVO_SYMMETRIC_INFORMATION_VALIDATION_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

namespace livo
{

inline constexpr double kSymmetricInformationRelativeTolerance =
  1.0e-10;

enum class SymmetricInformationRequirement
{
  kPositiveSemidefinite,
  kPositiveDefinite
};

enum class SymmetricInformationFailure
{
  kNone,
  kNonFinite,
  kNonSymmetric,
  kEigenDecompositionFailed,
  kNotPositiveSemidefinite,
  kNotPositiveDefinite,
  kInvalidConditionNumber
};

template<typename MatrixType>
struct SymmetricInformationValidationResult
{
  SymmetricInformationFailure failure{
    SymmetricInformationFailure::kNonFinite};
  MatrixType symmetric{MatrixType::Zero()};
  double scale{1.0};
  double minimum_eigenvalue{
    std::numeric_limits<double>::quiet_NaN()};
  double maximum_eigenvalue{
    std::numeric_limits<double>::quiet_NaN()};
  double condition_number{
    std::numeric_limits<double>::infinity()};

  bool accepted() const
  {
    return failure == SymmetricInformationFailure::kNone;
  }
};

template<typename Derived>
SymmetricInformationValidationResult<typename Derived::PlainObject>
validateSymmetricInformation(
  const Eigen::MatrixBase<Derived> & input_expression,
  const SymmetricInformationRequirement requirement)
{
  using MatrixType = typename Derived::PlainObject;
  static_assert(
    MatrixType::RowsAtCompileTime != Eigen::Dynamic &&
    MatrixType::ColsAtCompileTime != Eigen::Dynamic,
    "symmetric information validation requires fixed dimensions");
  static_assert(
    MatrixType::RowsAtCompileTime == MatrixType::ColsAtCompileTime,
    "symmetric information validation requires a square matrix");
  static_assert(
    std::is_floating_point<typename MatrixType::Scalar>::value,
    "symmetric information validation requires floating-point data");

  SymmetricInformationValidationResult<MatrixType> result;
  const MatrixType input = input_expression;
  if (!input.allFinite())
  {
    return result;
  }

  for (Eigen::Index row = 0; row < input.rows(); ++row)
  {
    for (Eigen::Index column = row + 1;
         column < input.cols(); ++column)
    {
      const double upper =
        static_cast<double>(input(row, column));
      const double lower =
        static_cast<double>(input(column, row));
      const double entry_scale = std::max(
        1.0, std::max(std::abs(upper), std::abs(lower)));
      result.scale = std::max(result.scale, entry_scale);
      const double difference = std::abs(upper - lower);
      if (!std::isfinite(difference) ||
          difference >
            kSymmetricInformationRelativeTolerance *
            entry_scale)
      {
        result.failure =
          SymmetricInformationFailure::kNonSymmetric;
        return result;
      }
    }
  }

  result.symmetric =
    0.5 * input + 0.5 * input.transpose();
  if (!result.symmetric.allFinite())
  {
    result.failure = SymmetricInformationFailure::kNonFinite;
    return result;
  }
  const Eigen::SelfAdjointEigenSolver<MatrixType> solver(
    result.symmetric, Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success ||
      !solver.eigenvalues().allFinite())
  {
    result.failure =
      SymmetricInformationFailure::kEigenDecompositionFailed;
    return result;
  }

  result.minimum_eigenvalue =
    static_cast<double>(solver.eigenvalues().minCoeff());
  result.maximum_eigenvalue =
    static_cast<double>(solver.eigenvalues().maxCoeff());
  result.scale = std::max(
    1.0,
    std::max(
      std::abs(result.minimum_eigenvalue),
      std::abs(result.maximum_eigenvalue)));
  if (requirement ==
      SymmetricInformationRequirement::kPositiveSemidefinite)
  {
    if (result.minimum_eigenvalue <
        -kSymmetricInformationRelativeTolerance * result.scale)
    {
      result.failure =
        SymmetricInformationFailure::kNotPositiveSemidefinite;
      return result;
    }
  }
  else
  {
    if (result.minimum_eigenvalue <= 0.0)
    {
      result.failure =
        SymmetricInformationFailure::kNotPositiveDefinite;
      return result;
    }
    result.condition_number =
      result.maximum_eigenvalue / result.minimum_eigenvalue;
    if (!std::isfinite(result.condition_number) ||
        result.condition_number < 1.0)
    {
      result.failure =
        SymmetricInformationFailure::kInvalidConditionNumber;
      return result;
    }
  }

  result.failure = SymmetricInformationFailure::kNone;
  return result;
}

}  // namespace livo

#endif  // LIVO_SYMMETRIC_INFORMATION_VALIDATION_H_
