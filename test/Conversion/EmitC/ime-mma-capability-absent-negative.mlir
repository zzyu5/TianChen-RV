// RUN: not weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=ABSENT --implicit-check-not="weft_ime.mma" --implicit-check-not="vmadot" --implicit-check-not="spacemit.ime"

// Complementary arm (dispatch is capability-FACT-driven, NOT hardcoded to IME):
// a target that exposes a NON-IME capability and does NOT provide spacemit.ime
// does not satisfy lookupProviderByID("spacemit.ime"), so the IME plugin
// declines and no weft_ime.mma / vmadot is ever materialized. With no other owner
// for this canonical P, proposal collection fails closed. This proves the IME path is gated on the capability
// FACT, not on a family-name match.
module {
  // ABSENT: collected no viable plugin proposals
  weft.exec.kernel @non_ime_kernel attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
