#ifndef WEFT_TRANSFORMS_DISPATCHRUNTIMEGUARD_H
#define WEFT_TRANSFORMS_DISPATCHRUNTIMEGUARD_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "llvm/Support/Error.h"

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace weft::transforms {

llvm::Error materializeDispatchRuntimeGuards(
    weft::exec::KernelOp kernel, mlir::OpBuilder &builder);

} // namespace weft::transforms

#endif // WEFT_TRANSFORMS_DISPATCHRUNTIMEGUARD_H
