#ifndef TIANCHENRV_TARGET_BINARYSELFCHECKEXPECTATION_H
#define TIANCHENRV_TARGET_BINARYSELFCHECKEXPECTATION_H

#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>

namespace tianchenrv::target {

enum class BinarySelfCheckArithmeticKind {
  Add,
  Sub,
  Mul,
};

struct BinarySelfCheckExpectation {
  BinarySelfCheckArithmeticKind arithmetic =
      BinarySelfCheckArithmeticKind::Add;
  std::string scalarCType;
  std::string provenance;

  llvm::StringRef getCOperator() const {
    switch (arithmetic) {
    case BinarySelfCheckArithmeticKind::Add:
      return "+";
    case BinarySelfCheckArithmeticKind::Sub:
      return "-";
    case BinarySelfCheckArithmeticKind::Mul:
      return "*";
    }
    return "?";
  }

  std::string formatExpression(llvm::StringRef lhs,
                               llvm::StringRef rhs) const {
    return (llvm::Twine(lhs) + " " + getCOperator() + " " + rhs).str();
  }
};

inline llvm::Error
makeBinarySelfCheckExpectationError(llvm::StringRef context,
                                    llvm::Twine message) {
  return llvm::createStringError(
      llvm::errc::invalid_argument,
      (llvm::Twine("TianChen-RV binary self-check expectation for ") +
       context + " failed: " + message)
          .str());
}

inline llvm::Expected<std::string>
getRuntimeABIPointeeScalarCType(llvm::StringRef cType,
                                llvm::StringRef context,
                                llvm::StringRef roleName) {
  cType = cType.trim();
  if (!cType.ends_with("*"))
    return makeBinarySelfCheckExpectationError(
        context, llvm::Twine(roleName) +
                     " ABI parameter c_type must be a pointer type");

  llvm::StringRef pointee = cType.drop_back().rtrim();
  if (pointee.consume_front("const "))
    pointee = pointee.ltrim();
  if (pointee.empty())
    return makeBinarySelfCheckExpectationError(
        context, llvm::Twine(roleName) +
                     " ABI parameter pointer c_type has no scalar pointee");
  return pointee.str();
}

inline llvm::Expected<BinarySelfCheckExpectation>
buildBinarySelfCheckExpectationFromRuntimeABI(
    llvm::ArrayRef<support::RuntimeABIParameter> runtimeABIParameters,
    BinarySelfCheckArithmeticKind arithmetic, llvm::StringRef provenance,
    llvm::StringRef context) {
  using Role = support::RuntimeABIParameterRole;
  llvm::Expected<const support::RuntimeABIParameter *> lhs =
      support::findUniqueRuntimeABIParameterByRole(
          runtimeABIParameters, Role::LHSInputBuffer, context);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<const support::RuntimeABIParameter *> rhs =
      support::findUniqueRuntimeABIParameterByRole(
          runtimeABIParameters, Role::RHSInputBuffer, context);
  if (!rhs)
    return rhs.takeError();
  llvm::Expected<const support::RuntimeABIParameter *> out =
      support::findUniqueRuntimeABIParameterByRole(
          runtimeABIParameters, Role::OutputBuffer, context);
  if (!out)
    return out.takeError();

  llvm::Expected<std::string> lhsType = getRuntimeABIPointeeScalarCType(
      (*lhs)->cType, context,
      support::stringifyRuntimeABIParameterRole(Role::LHSInputBuffer));
  if (!lhsType)
    return lhsType.takeError();
  llvm::Expected<std::string> rhsType = getRuntimeABIPointeeScalarCType(
      (*rhs)->cType, context,
      support::stringifyRuntimeABIParameterRole(Role::RHSInputBuffer));
  if (!rhsType)
    return rhsType.takeError();
  llvm::Expected<std::string> outType = getRuntimeABIPointeeScalarCType(
      (*out)->cType, context,
      support::stringifyRuntimeABIParameterRole(Role::OutputBuffer));
  if (!outType)
    return outType.takeError();

  if (*lhsType != *rhsType || *lhsType != *outType)
    return makeBinarySelfCheckExpectationError(
        context,
        "lhs, rhs, and output ABI pointer scalar types must agree before "
        "self-check harness emission");

  if (provenance.empty())
    return makeBinarySelfCheckExpectationError(
        context,
        "requires non-empty typed provenance before self-check harness "
        "emission");

  BinarySelfCheckExpectation expectation;
  expectation.arithmetic = arithmetic;
  expectation.scalarCType = std::move(*lhsType);
  expectation.provenance = provenance.str();
  return expectation;
}

} // namespace tianchenrv::target

#endif // TIANCHENRV_TARGET_BINARYSELFCHECKEXPECTATION_H
