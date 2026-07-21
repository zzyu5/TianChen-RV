#ifndef WEFT_CONVERSION_RVV_RVVCODEBOOKGATHERPLANMATERIALIZATION_H
#define WEFT_CONVERSION_RVV_RVVCODEBOOKGATHERPLANMATERIALIZATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace weft::conversion::rvv {

/// The single pre-emission owner for the A3 small-codebook decision slice.
/// It constructs any still-abstract production codebook dequant row, projects
/// typed g + the uniquely selected canonical capability c, runs the bounded
/// formula decision, and stamps the complete selected plan on the typed decode
/// core. Existing stamps must exactly match the recomputed result. No EmitC is
/// created here.
mlir::LogicalResult
materializeRVVCodebookGatherPlans(mlir::ModuleOp module);

} // namespace weft::conversion::rvv

#endif // WEFT_CONVERSION_RVV_RVVCODEBOOKGATHERPLANMATERIALIZATION_H
