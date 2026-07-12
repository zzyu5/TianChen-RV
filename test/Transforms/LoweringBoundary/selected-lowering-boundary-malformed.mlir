// RUN: not weft-opt %s --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  weft.exec.kernel @duplicate_selected_lowering_boundary_ref {
    weft.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    weft.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    weft.exec.dispatch {
      weft.exec.case @scalar_fallback_first_slice
      weft.exec.fallback @scalar_fallback_first_slice
    }
  }
}

// CHECK: Weft-RV selected lowering-boundary materialization failed
// CHECK-SAME: dispatch reference validation failed before plugin lowering-boundary routing
// CHECK-SAME: duplicate selected lowering-boundary reference to variant @scalar_fallback_first_slice
