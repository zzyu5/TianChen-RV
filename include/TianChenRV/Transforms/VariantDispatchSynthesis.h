#ifndef TIANCHENRV_TRANSFORMS_VARIANTDISPATCHSYNTHESIS_H
#define TIANCHENRV_TRANSFORMS_VARIANTDISPATCHSYNTHESIS_H

#include "TianChenRV/Dialect/Exec/IR/ExecOps.h"

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class OpBuilder;
} // namespace mlir

namespace tianchenrv::transforms {

mlir::LogicalResult synthesizeVariantDispatch(
    mlir::OpBuilder &builder, tcrv::exec::KernelOp kernel,
    tcrv::exec::DispatchOp *createdDispatch = nullptr);

} // namespace tianchenrv::transforms

#endif // TIANCHENRV_TRANSFORMS_VARIANTDISPATCHSYNTHESIS_H
