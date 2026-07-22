#ifndef WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H
#define WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Support/CapabilityModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

namespace weft::plugin::rvv {

/// Domain-local owners shared by the explicit inspection front doors and the
/// project-level pre-emission lifecycle.  Each function constructs the final
/// typed body atomically for only its own abstract source domain.
mlir::LogicalResult constructRVVQuantizeRowFormulaBodies(mlir::ModuleOp module);
mlir::LogicalResult
constructRVVDequantizeRowFormulaBodies(mlir::ModuleOp module);

/// Recursively qualify only the exact typed body returned by the bound RVV
/// construction invocation.  This is family-local structural qualification;
/// common orchestration and artifact code never rediscover a body by scanning
/// metadata.
mlir::LogicalResult validateRVVConstructedTypedBody(mlir::Operation *body);

/// Evaluate RVV family-local formulas only inside the explicitly bound variant,
/// consuming the capability set already projected by the common binding gate.
/// Selected-body realization remains the family owner's next step; artifact
/// lowering is construction-blind.
mlir::LogicalResult constructRVVFormulaPlansForVariant(
    weft::exec::VariantOp variant,
    const support::TargetCapabilitySet &capabilities);

} // namespace weft::plugin::rvv

#endif // WEFT_PLUGIN_RVV_RVVFORMULACONSTRUCTION_H
