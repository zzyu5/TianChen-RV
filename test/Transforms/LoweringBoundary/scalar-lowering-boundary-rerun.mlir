// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @scalar_boundary_deleted {
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
    tcrv.exec.diagnostic {
      message = "scalar fallback direct selected path is unsupported",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
  }
}

// CHECK: TianChen-RV selected lowering-boundary materialization failed
// CHECK-SAME: origin plugin 'scalar-plugin' reported unsupported lowering-boundary materialization
// CHECK-SAME: no longer materializes a legacy metadata selected lowering boundary
