#ifndef TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONSELECTEDBODYREALIZATIONOWNER_H
#define TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONSELECTEDBODYREALIZATIONOWNER_H

#include "TianChenRV/Dialect/RVV/IR/RVVDialect.h"
#include "TianChenRV/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace tianchenrv::plugin::rvv {

bool isPreRealizedRVVContractionClusterOp(mlir::Operation *op);

llvm::Expected<tcrv::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

} // namespace tianchenrv::plugin::rvv

#endif // TIANCHENRV_PLUGIN_RVV_RVVCONTRACTIONSELECTEDBODYREALIZATIONOWNER_H
