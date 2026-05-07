// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @scalar_boundary_rerun {
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
      message = "scalar fallback selected by generic planner",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv_scalar.lowering_boundary {
      fallback_reason = "existing scalar metadata boundary",
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_boundary_rerun",
      status = "metadata-only"
    }
  }
}

// CHECK: TianChen-RV selected lowering-boundary materialization failed
// CHECK-SAME: origin plugin 'scalar-plugin' failed lowering-boundary materialization
// CHECK-SAME: requires no pre-existing tcrv_scalar.lowering_boundary for target @scalar_fallback_first_slice
