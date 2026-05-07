// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @duplicate_selected_lowering_boundary_ref {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @scalar_fallback_first_slice
      tcrv.exec.fallback @scalar_fallback_first_slice
    }
  }
}

// CHECK: TianChen-RV selected lowering-boundary materialization failed
// CHECK-SAME: dispatch reference validation failed before plugin lowering-boundary routing
// CHECK-SAME: duplicate selected lowering-boundary reference to variant @scalar_fallback_first_slice
