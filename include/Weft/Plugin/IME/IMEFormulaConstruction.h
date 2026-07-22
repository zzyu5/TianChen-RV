#ifndef WEFT_PLUGIN_IME_IMEFORMULACONSTRUCTION_H
#define WEFT_PLUGIN_IME_IMEFORMULACONSTRUCTION_H

#include "llvm/ADT/StringRef.h"

namespace weft::plugin::ime {

/// Compile-time inventory and conversion-local plan keys. Formula evaluation
/// remains in the typed IME family construction call graph; these strings must
/// never become a dynamic dispatcher or a way to recover a body.
inline constexpr llvm::StringLiteral kIMEConstructionFormulaID(
    "weft.ime.matmul.construct");
inline constexpr llvm::StringLiteral kIMEFinalPlanAttrName(
    "weft.ime.final_plan");

} // namespace weft::plugin::ime

#endif // WEFT_PLUGIN_IME_IMEFORMULACONSTRUCTION_H
