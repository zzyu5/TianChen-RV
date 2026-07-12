#ifndef WEFT_PLUGIN_RVV_RVVSEGMENT2MEMORYSELECTEDBODYREALIZATIONOWNER_H
#define WEFT_PLUGIN_RVV_RVVSEGMENT2MEMORYSELECTEDBODYREALIZATIONOWNER_H

#include "Weft/Dialect/RVV/IR/RVVDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"

#include "mlir/IR/Operation.h"
#include "llvm/Support/Error.h"

namespace weft::plugin::rvv {

bool isPreRealizedRVVSegment2MemoryOwnerOp(mlir::Operation *op);

llvm::Expected<weft::rvv::WithVLOp> realizePreRealizedRVVSegment2MemoryOwner(
    const VariantLoweringBoundaryRequest &request, mlir::Operation *bodyOp);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVSEGMENT2MEMORYSELECTEDBODYREALIZATIONOWNER_H
