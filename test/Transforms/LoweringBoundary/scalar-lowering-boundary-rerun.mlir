// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s

module {
  weft.exec.kernel @scalar_boundary_deleted {
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
    weft.exec.diagnostic {
      message = "scalar fallback direct selected path has no source problem",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
  }
}

// CHECK: weft.exec.kernel @scalar_boundary_deleted
// CHECK: weft.exec.variant @scalar_fallback_first_slice
// CHECK-NOT: weft_scalar.lowering_boundary
