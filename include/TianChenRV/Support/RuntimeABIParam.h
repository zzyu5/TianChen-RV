#ifndef TIANCHENRV_SUPPORT_RUNTIMEABIPARAM_H
#define TIANCHENRV_SUPPORT_RUNTIMEABIPARAM_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace tianchenrv::support {

inline constexpr llvm::StringLiteral kRuntimeParamPurposeAttrName("purpose");
inline constexpr llvm::StringLiteral kRuntimeParamABIRoleAttrName("abi_role");
inline constexpr llvm::StringLiteral kRuntimeParamCNameAttrName("c_name");
inline constexpr llvm::StringLiteral kRuntimeParamCTypeAttrName("c_type");
inline constexpr llvm::StringLiteral kRuntimeParamOwnershipAttrName(
    "ownership");

inline constexpr llvm::StringLiteral kRuntimeABIScalarParamPurpose(
    "runtime-abi-scalar");

struct RuntimeABIParamSpec {
  RuntimeABIParamSpec() = default;
  RuntimeABIParamSpec(llvm::StringRef symbolName,
                      RuntimeABIParameterRole role, llvm::StringRef cName,
                      llvm::StringRef cType, llvm::StringRef ownership)
      : symbolName(symbolName.str()), role(role), cName(cName.str()),
        cType(cType.str()), ownership(ownership.str()) {}

  std::string symbolName;
  RuntimeABIParameterRole role = RuntimeABIParameterRole::RuntimeElementCount;
  std::string cName;
  std::string cType;
  std::string ownership;
};

RuntimeABIParamSpec getI32BinaryRuntimeElementCountParamSpec(
    llvm::StringRef cName = "n");

RuntimeABIParamSpec getI32BinaryRuntimeElementCountParamSpecForFamily(
    llvm::StringRef familyID, llvm::StringRef cName = "n");

RuntimeABIParamSpec getDispatchAvailabilityGuardParamSpec(
    llvm::StringRef cName = "rvv_available");

RuntimeABIParamSpec getI32BinaryDispatchAvailabilityGuardParamSpec(
    llvm::StringRef cName = "rvv_available");

RuntimeABIParamSpec
getI32BinaryDispatchAvailabilityGuardParamSpecForFamily(
    llvm::StringRef familyID, llvm::StringRef cName = "rvv_available");

llvm::SmallVector<RuntimeABIParamSpec, 1>
getI32BinaryRuntimeElementCountParamSpecs(llvm::StringRef cName = "n");

llvm::SmallVector<RuntimeABIParamSpec, 1>
getI32BinaryRuntimeElementCountParamSpecsForFamily(
    llvm::StringRef familyID, llvm::StringRef cName = "n");

llvm::SmallVector<RuntimeABIParamSpec, 2>
getI32BinaryDispatchRuntimeParamSpecs(llvm::StringRef runtimeCountCName = "n",
                                      llvm::StringRef guardCName =
                                          "rvv_available");

llvm::SmallVector<RuntimeABIParamSpec, 2>
getI32BinaryDispatchRuntimeParamSpecsForFamily(
    llvm::StringRef familyID, llvm::StringRef runtimeCountCName = "n",
    llvm::StringRef guardCName = "rvv_available");

llvm::Error ensureRuntimeABIParams(
    tcrv::exec::KernelOp kernel, mlir::OpBuilder &builder,
    llvm::ArrayRef<RuntimeABIParamSpec> specs);

llvm::Error ensureRuntimeABIParamsAllowingExistingCNames(
    tcrv::exec::KernelOp kernel, mlir::OpBuilder &builder,
    llvm::ArrayRef<RuntimeABIParamSpec> specs);

llvm::Error collectRuntimeABIParams(
    tcrv::exec::KernelOp kernel, llvm::ArrayRef<RuntimeABIParamSpec> specs,
    llvm::SmallVectorImpl<tcrv::exec::RuntimeParamOp> &out);

llvm::Error validateRuntimeABIParams(
    tcrv::exec::KernelOp kernel, llvm::ArrayRef<RuntimeABIParamSpec> specs);

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABIPARAM_H
