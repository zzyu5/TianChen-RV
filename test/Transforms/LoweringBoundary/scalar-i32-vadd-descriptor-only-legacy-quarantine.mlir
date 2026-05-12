// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @scalar_descriptor_only_i32_vadd {
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
      tcrv_scalar.element_count = 16 : i64,
      tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"
    } {
    }
    tcrv.exec.diagnostic {
      message = "legacy descriptor-only scalar i32-vadd fixture must be quarantined",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
  }
}

// CHECK: legacy descriptor-only metadata
// CHECK-SAME: tcrv_scalar.lowering_descriptor
// CHECK-SAME: tcrv_scalar.element_count
// CHECK-SAME: typed scalar microkernel body
// CHECK-SAME: descriptorless typed default materialization
// CHECK-SAME: cannot create tcrv_scalar.lowering_boundary
