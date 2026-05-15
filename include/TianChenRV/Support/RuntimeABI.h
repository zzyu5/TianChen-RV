#ifndef TIANCHENRV_SUPPORT_RUNTIMEABI_H
#define TIANCHENRV_SUPPORT_RUNTIMEABI_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace tianchenrv::support {

class FiniteBinaryRuntimeABIContract;

inline constexpr llvm::StringLiteral kRuntimeABIParameterCNameAttrName(
    "c_name");
inline constexpr llvm::StringLiteral kRuntimeABIParameterCTypeAttrName(
    "c_type");
inline constexpr llvm::StringLiteral kRuntimeABIParameterRoleAttrName("role");
inline constexpr llvm::StringLiteral kRuntimeABIParameterOwnershipAttrName(
    "ownership");

enum class RuntimeABIParameterRole {
  LHSInputBuffer,
  RHSInputBuffer,
  OutputBuffer,
  RuntimeElementCount,
  DispatchAvailabilityGuard,
};

enum class RuntimeABIParameterOwnership {
  IRModeled,
  TargetExportABIOwned,
};

struct RuntimeABIParameter {
  RuntimeABIParameter() = default;
  RuntimeABIParameter(llvm::StringRef cName, llvm::StringRef cType,
                      RuntimeABIParameterRole role,
                      RuntimeABIParameterOwnership ownership)
      : cName(cName.str()), cType(cType.str()), role(role),
        ownership(ownership) {}

  std::string cName;
  std::string cType;
  RuntimeABIParameterRole role = RuntimeABIParameterRole::LHSInputBuffer;
  RuntimeABIParameterOwnership ownership =
      RuntimeABIParameterOwnership::TargetExportABIOwned;
};

struct FiniteBinaryCallableRuntimeABIParameterBindings {
  const RuntimeABIParameter *lhs = nullptr;
  const RuntimeABIParameter *rhs = nullptr;
  const RuntimeABIParameter *out = nullptr;
  const RuntimeABIParameter *runtimeElementCount = nullptr;
};

inline llvm::StringRef stringifyRuntimeABIParameterRole(
    RuntimeABIParameterRole role) {
  switch (role) {
  case RuntimeABIParameterRole::LHSInputBuffer:
    return "lhs-input-buffer";
  case RuntimeABIParameterRole::RHSInputBuffer:
    return "rhs-input-buffer";
  case RuntimeABIParameterRole::OutputBuffer:
    return "output-buffer";
  case RuntimeABIParameterRole::RuntimeElementCount:
    return "runtime-element-count";
  case RuntimeABIParameterRole::DispatchAvailabilityGuard:
    return "dispatch-availability-guard";
  }
  return "unknown";
}

inline std::optional<RuntimeABIParameterRole>
symbolizeRuntimeABIParameterRole(llvm::StringRef role) {
  if (role == "lhs-input-buffer")
    return RuntimeABIParameterRole::LHSInputBuffer;
  if (role == "rhs-input-buffer")
    return RuntimeABIParameterRole::RHSInputBuffer;
  if (role == "output-buffer")
    return RuntimeABIParameterRole::OutputBuffer;
  if (role == "runtime-element-count")
    return RuntimeABIParameterRole::RuntimeElementCount;
  if (role == "dispatch-availability-guard")
    return RuntimeABIParameterRole::DispatchAvailabilityGuard;
  return std::nullopt;
}

inline llvm::StringRef stringifyRuntimeABIParameterOwnership(
    RuntimeABIParameterOwnership ownership) {
  switch (ownership) {
  case RuntimeABIParameterOwnership::IRModeled:
    return "ir-modeled";
  case RuntimeABIParameterOwnership::TargetExportABIOwned:
    return "target-export-abi-owned";
  }
  return "unknown";
}

inline std::optional<RuntimeABIParameterOwnership>
symbolizeRuntimeABIParameterOwnership(llvm::StringRef ownership) {
  if (ownership == "ir-modeled")
    return RuntimeABIParameterOwnership::IRModeled;
  if (ownership == "target-export-abi-owned")
    return RuntimeABIParameterOwnership::TargetExportABIOwned;
  return std::nullopt;
}

inline RuntimeABIParameter makeTargetExportABIParameter(
    llvm::StringRef cName, llvm::StringRef cType,
    RuntimeABIParameterRole role) {
  return RuntimeABIParameter(cName, cType, role,
                             RuntimeABIParameterOwnership::TargetExportABIOwned);
}

inline RuntimeABIParameter makeTargetExportABIRoleRequirement(
    llvm::StringRef cType, RuntimeABIParameterRole role) {
  return RuntimeABIParameter("", cType, role,
                             RuntimeABIParameterOwnership::TargetExportABIOwned);
}

inline void appendDispatchRuntimeABIParameters(
    llvm::SmallVectorImpl<RuntimeABIParameter> &out,
    llvm::ArrayRef<RuntimeABIParameter> callableParameters,
    const RuntimeABIParameter &guardParameter) {
  out.append(callableParameters.begin(), callableParameters.end());
  out.push_back(guardParameter);
}

inline bool runtimeABIParametersEqual(
    llvm::ArrayRef<RuntimeABIParameter> lhs,
    llvm::ArrayRef<RuntimeABIParameter> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs)) {
    if (left.cName != right.cName || left.cType != right.cType ||
        left.role != right.role || left.ownership != right.ownership)
      return false;
  }
  return true;
}

llvm::Expected<const RuntimeABIParameter *> findUniqueRuntimeABIParameterByRole(
    llvm::ArrayRef<RuntimeABIParameter> parameters,
    RuntimeABIParameterRole role, llvm::StringRef context);

llvm::Expected<FiniteBinaryCallableRuntimeABIParameterBindings>
bindFiniteBinaryCallableRuntimeABIParametersByRole(
    llvm::ArrayRef<RuntimeABIParameter> parameters, llvm::StringRef context,
    const FiniteBinaryRuntimeABIContract &contract);

inline void printRuntimeABIParameterCDeclaration(
    llvm::raw_ostream &os, const RuntimeABIParameter &parameter) {
  os << parameter.cType;
  if (!llvm::StringRef(parameter.cType).ends_with("*"))
    os << " ";
  os << parameter.cName;
}

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABI_H
