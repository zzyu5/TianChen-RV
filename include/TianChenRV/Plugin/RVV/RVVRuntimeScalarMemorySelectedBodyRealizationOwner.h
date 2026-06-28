#ifndef TIANCHENRV_PLUGIN_RVV_RVVRUNTIMESCALARMEMORYSELECTEDBODYREALIZATIONOWNER_H
#define TIANCHENRV_PLUGIN_RVV_RVVRUNTIMESCALARMEMORYSELECTEDBODYREALIZATIONOWNER_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

bool isPreRealizedRVVRuntimeScalarSplatStoreOwnerOp(mlir::Operation *op);

bool isPreRealizedRVVRuntimeScalarComputedMaskStoreOwnerOp(
    mlir::Operation *op);

bool isPreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwnerOp(
    mlir::Operation *op);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarSplatStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVRuntimeScalarComputedMaskLoadStoreOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVRUNTIMESCALARMEMORYSELECTEDBODYREALIZATIONOWNER_H
