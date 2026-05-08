#ifndef TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H
#define TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"
#include "TianChenRV/Support/RuntimeABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::support {

struct I32VAddCallableABIPlan {
  llvm::SmallVector<RuntimeABIParameter, 4> parameters;
  llvm::SmallVector<tcrv::exec::MemWindowOp, 3> bufferWindows;
  tcrv::exec::RuntimeParamOp runtimeElementCountParam;
};

// Bounded bridge for the current i32-vadd callable ABI only.  It is not a
// general ABI subsystem.
llvm::Expected<I32VAddCallableABIPlan>
buildI32VAddCallableABIPlan(tcrv::exec::KernelOp kernel);

llvm::Error validateI32VAddCallableABIParameterMirror(
    tcrv::exec::KernelOp kernel,
    llvm::ArrayRef<RuntimeABIParameter> metadataParameters,
    llvm::ArrayRef<RuntimeABIParameter> irBackedParameters,
    llvm::StringRef metadataSource);

} // namespace tianchenrv::support

#endif // TIANCHENRV_SUPPORT_RUNTIMEABICALLABLEPLAN_H
