#ifndef WEFT_PLUGIN_RVV_RVVCONTRACTIONSELECTEDBODYREALIZATIONOWNER_H
#define WEFT_PLUGIN_RVV_RVVCONTRACTIONSELECTEDBODYREALIZATIONOWNER_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "Weft/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::rvv {

bool isPreRealizedRVVContractionClusterOp(mlir::Operation *op);

llvm::Expected<weft::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

llvm::Expected<weft::rvv::WithVLOp> realizePreRealizedRVVContractionOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp,
    const RVVLowPrecisionProductionPressureProfile &pressureProfile);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVCONTRACTIONSELECTEDBODYREALIZATIONOWNER_H
