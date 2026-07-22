#ifndef WEFT_PLUGIN_SCALAR_SCALARFORMULACONSTRUCTION_H
#define WEFT_PLUGIN_SCALAR_SCALARFORMULACONSTRUCTION_H

#include "llvm/ADT/StringRef.h"

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

inline constexpr llvm::StringLiteral kScalarFinalPlanAttrName(
    "weft.scalar.final_plan");

} // namespace weft::plugin::scalar

#endif // WEFT_PLUGIN_SCALAR_SCALARFORMULACONSTRUCTION_H
