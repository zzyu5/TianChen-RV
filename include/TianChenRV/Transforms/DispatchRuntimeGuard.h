#ifndef TIANCHENRV_TRANSFORMS_DISPATCHRUNTIMEGUARD_H
#define TIANCHENRV_TRANSFORMS_DISPATCHRUNTIMEGUARD_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "llvm/Support/Error.h"

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace tianchenrv::transforms {

llvm::Error materializeDispatchRuntimeGuards(
    tcrv::exec::KernelOp kernel, mlir::OpBuilder &builder);

} // namespace tianchenrv::transforms

#endif // TIANCHENRV_TRANSFORMS_DISPATCHRUNTIMEGUARD_H
