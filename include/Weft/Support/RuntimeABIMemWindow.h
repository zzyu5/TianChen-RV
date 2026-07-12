#ifndef WEFT_SUPPORT_RUNTIMEABIMEMWINDOW_H
#define WEFT_SUPPORT_RUNTIMEABIMEMWINDOW_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Support/RuntimeABI.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace weft::support {

inline constexpr llvm::StringLiteral kMemWindowPurposeAttrName("purpose");
inline constexpr llvm::StringLiteral kMemWindowBindingAttrName("binding");
inline constexpr llvm::StringLiteral kMemWindowMemorySpaceAttrName(
    "memory_space");
inline constexpr llvm::StringLiteral kMemWindowABIRoleAttrName("abi_role");
inline constexpr llvm::StringLiteral kMemWindowAccessAttrName("access");
inline constexpr llvm::StringLiteral kMemWindowOwnershipAttrName("ownership");
inline constexpr llvm::StringLiteral kMemWindowCTypeAttrName("c_type");

inline constexpr llvm::StringLiteral kRuntimeABIBufferPurpose(
    "runtime-abi-buffer");
inline constexpr llvm::StringLiteral kRuntimeABIKernelArgumentBinding(
    "kernel-argument");
inline constexpr llvm::StringLiteral kRuntimeABIHostMemorySpace("host");
inline constexpr llvm::StringLiteral kRuntimeABIReadAccess("read");
inline constexpr llvm::StringLiteral kRuntimeABIWriteAccess("write");

struct RuntimeABIMemWindowSpec {
  RuntimeABIMemWindowSpec() = default;
  RuntimeABIMemWindowSpec(llvm::StringRef symbolName,
                          RuntimeABIParameterRole role,
                          llvm::StringRef access, llvm::StringRef ownership,
                          llvm::StringRef cType)
      : symbolName(symbolName.str()), role(role), access(access.str()),
        ownership(ownership.str()), cType(cType.str()) {}

  std::string symbolName;
  RuntimeABIParameterRole role = RuntimeABIParameterRole::LHSInputBuffer;
  std::string access;
  std::string ownership;
  std::string cType;
};

llvm::Error ensureRuntimeABIBufferMemWindows(
    weft::exec::KernelOp kernel, mlir::OpBuilder &builder,
    llvm::ArrayRef<RuntimeABIMemWindowSpec> specs);

llvm::Error collectRuntimeABIBufferMemWindows(
    weft::exec::KernelOp kernel, llvm::ArrayRef<RuntimeABIMemWindowSpec> specs,
    llvm::SmallVectorImpl<weft::exec::MemWindowOp> &out);

llvm::Error validateRuntimeABIBufferMemWindows(
    weft::exec::KernelOp kernel, llvm::ArrayRef<RuntimeABIMemWindowSpec> specs);

} // namespace weft::support

#endif // WEFT_SUPPORT_RUNTIMEABIMEMWINDOW_H
