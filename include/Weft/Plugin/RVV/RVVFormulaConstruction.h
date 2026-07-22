#ifndef WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H
#define WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace weft::plugin::rvv {

/// Domain-local owners shared by the explicit inspection front doors and the
/// project-level pre-emission lifecycle.  Each function constructs the final
/// typed body atomically for only its own abstract source domain.
mlir::LogicalResult constructRVVQuantizeRowFormulaBodies(mlir::ModuleOp module);
mlir::LogicalResult
constructRVVDequantizeRowFormulaBodies(mlir::ModuleOp module);

/// The unique RVV pre-emission construction cut. It evaluates family-local
/// typed formulas and atomically creates or validates the final typed plans.
/// It emits no target code and has no partial-plan or replay mode.
mlir::LogicalResult constructRVVFormulaBodies(mlir::ModuleOp module);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H
