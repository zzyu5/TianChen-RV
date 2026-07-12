#ifndef WEFT_TRANSFORMS_VARIANTDISPATCHSYNTHESIS_H
#define WEFT_TRANSFORMS_VARIANTDISPATCHSYNTHESIS_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace weft::transforms {

mlir::LogicalResult synthesizeVariantDispatch(
    mlir::OpBuilder &builder, weft::exec::KernelOp kernel,
    weft::exec::DispatchOp *createdDispatch = nullptr);

} // namespace weft::transforms

#endif // WEFT_TRANSFORMS_VARIANTDISPATCHSYNTHESIS_H
