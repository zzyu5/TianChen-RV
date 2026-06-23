// RUN: tcrv-opt %s --tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --implicit-check-not="tcrv_ime.mma" --implicit-check-not="vmadot" --implicit-check-not="spacemit.ime"

// Complementary arm (dispatch is capability-FACT-driven, NOT hardcoded to IME):
// a target that exposes a NON-IME capability and does NOT provide spacemit.ime
// does not satisfy lookupProviderByID("spacemit.ime"), so the IME plugin
// declines and no tcrv_ime.mma / vmadot is ever materialized. The kernel is left
// without an IME boundary. This proves the IME path is gated on the capability
// FACT, not on a family-name match.
module {
  // CHECK-LABEL: tcrv.exec.kernel @non_ime_kernel
  tcrv.exec.kernel @non_ime_kernel {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}
