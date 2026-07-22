#ifndef WEFT_PLUGIN_SCALAR_SCALARFORMULACONSTRUCTION_H
#define WEFT_PLUGIN_SCALAR_SCALARFORMULACONSTRUCTION_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class Operation;
}

namespace weft::exec {
class KernelOp;
class VariantOp;
}

namespace weft::support {
class TargetCapabilitySet;
}

namespace weft::plugin::scalar {

/// Formula identities are compile-time inventory keys only.  Evaluation stays
/// in the typed family-local construction functions; callers must never use
/// these strings to dispatch or recover an implementation.
inline constexpr llvm::StringLiteral kScalarFallbackConstructionFormulaID(
    "weft.scalar.fallback.construct");
inline constexpr llvm::StringLiteral kScalarTernaryBlockDotFormulaID(
    "weft.scalar.tq2-q8.block-dot.construct");
inline constexpr llvm::StringLiteral kScalarQ40DequantizeRowFormulaID(
    "weft.scalar.q4-0.dequantize-row.construct");

/// Evaluate the Scalar formula for one explicitly bound variant, consume its
/// typed source problem, and return the distinct final family-local body.  A
/// null operation is an honest unsupported fallback envelope.  Formula IDs
/// remain catalog inventory keys; they are never copied into the IR or read by
/// artifact lowering.
llvm::Expected<mlir::Operation *> constructScalarFinalBody(
    weft::exec::VariantOp variant, weft::exec::KernelOp kernel,
    const weft::support::TargetCapabilitySet &capabilities);

} // namespace weft::plugin::scalar

#endif // WEFT_PLUGIN_SCALAR_SCALARFORMULACONSTRUCTION_H
