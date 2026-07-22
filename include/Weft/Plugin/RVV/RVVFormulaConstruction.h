#ifndef WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H
#define WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace weft::plugin::rvv {

/// The unique RVV pre-emission construction cut. It evaluates family-local
/// typed formulas and atomically creates or validates the final typed plans.
/// It emits no target code and has no partial-plan or replay mode.
mlir::LogicalResult constructRVVFormulaBodies(mlir::ModuleOp module);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H
