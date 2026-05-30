#ifndef TIANCHENRV_PLUGIN_RVV_RVVSTANDALONEREDUCTIONSELECTEDBODYREALIZATIONOWNER_H
#define TIANCHENRV_PLUGIN_RVV_RVVSTANDALONEREDUCTIONSELECTEDBODYREALIZATIONOWNER_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

bool isPreRealizedRVVStandaloneReductionClusterOp(mlir::Operation *op);

llvm::Expected<tcrv::rvv::WithVLOp>
realizePreRealizedRVVStandaloneReductionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVSTANDALONEREDUCTIONSELECTEDBODYREALIZATIONOWNER_H
