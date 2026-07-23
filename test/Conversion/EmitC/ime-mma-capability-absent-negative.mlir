// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries | FileCheck %s --implicit-check-not="weft_ime.mma" --implicit-check-not="vmadot" --implicit-check-not="spacemit.ime"

// Complementary arm (dispatch is capability-FACT-driven, NOT hardcoded to IME):
// a target that exposes a NON-IME capability and does NOT provide spacemit.ime
// does not satisfy lookupProviderByID("spacemit.ime"), so the IME plugin
// declines and no weft_ime.mma / vmadot is ever materialized. The kernel is left
// without an IME boundary. This proves the IME path is gated on the capability
// FACT, not on a family-name match.
module {
  // CHECK-LABEL: weft.exec.kernel @non_ime_kernel
  weft.exec.kernel @non_ime_kernel attributes {problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
