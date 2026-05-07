// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @scalar_descriptor_bad_count {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback],
      tcrv_scalar.element_count = 0 : i64,
      tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected finite scalar i32-vadd descriptor fixture",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
  }
}

// CHECK: selected scalar fallback variant @scalar_fallback_first_slice failed plugin legality before boundary materialization
// CHECK-SAME: finite scalar i32-vadd lowering descriptor on variant @scalar_fallback_first_slice requires tcrv_scalar.element_count in the bounded smoke range [1, 64]
