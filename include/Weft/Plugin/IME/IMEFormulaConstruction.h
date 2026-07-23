#ifndef WEFT_PLUGIN_IME_IMEFORMULACONSTRUCTION_H
#define WEFT_PLUGIN_IME_IMEFORMULACONSTRUCTION_H

#include "Weft/Dialect/Exec/IR/ExecOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
class Operation;
} // namespace mlir

namespace weft::plugin::ime {

/// Compile-time inventory and conversion-local plan keys. Formula evaluation
/// remains in the typed IME family construction call graph; these strings must
/// never become a dynamic dispatcher or a way to recover a body.
inline constexpr llvm::StringLiteral kIMEConstructionFormulaID(
    "weft.ime.matmul.construct");

/// Artifact-neutral result of the whole-matrix IME construction formula.
/// The target instruction identity remains carried by the typed IME op; this
/// plan records only computation geometry selected before artifact lowering.
struct IMEMatMulComputationPlan {
  int64_t matM = 0;
  int64_t matN = 0;
  int64_t matK = 0;
};

/// Artifact-neutral result of the quantized-tile IME construction formulas.
/// `macBatched` and `wideNJW` are computation/schedule decisions.  The
/// remaining wide fields are numeric construction provenance used to project
/// diagnostics; none names a helper, mnemonic, artifact kind, or backend.
struct IMEQuantComputationPlan : IMEMatMulComputationPlan {
  bool macBatched = false;
  int64_t wideNJW = 1;
  int64_t wideVlenBits = 0;
  int64_t wideInputFragmentVRegs = 0;
  int64_t wideAccumulatorVRegs = 0;
  int64_t wideVRegFloor = 0;
};

/// Complete one explicitly bound IME final body's computation plan.  This is
/// the sole owner of IME formula evaluation; it never scans another variant or
/// rebuilds the target capability set.
mlir::LogicalResult constructIMEFormulaPlan(
    mlir::Operation *body, weft::exec::VariantOp variant,
    weft::exec::KernelOp kernel, int64_t targetVlenBits);

/// Typed consumers for family-owned exact bodies. Quantized bodies reject
/// missing or partial typed schedule fields; there is no parallel plan
/// dictionary for the artifact backend to replay.
mlir::LogicalResult requireIMESimpleComputationPlan(mlir::Operation *op);
mlir::FailureOr<IMEMatMulComputationPlan>
readIMEMatMulComputationPlan(mlir::Operation *op);
mlir::FailureOr<IMEQuantComputationPlan>
readIMEQuantComputationPlan(mlir::Operation *op);

} // namespace weft::plugin::ime

#endif // WEFT_PLUGIN_IME_IMEFORMULACONSTRUCTION_H
