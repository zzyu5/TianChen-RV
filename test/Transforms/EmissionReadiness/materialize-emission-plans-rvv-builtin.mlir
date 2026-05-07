// RUN: tcrv-opt %s --tcrv-materialize-emission-plans --split-input-file | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_rvv_plan
  tcrv.exec.kernel @public_rvv_plan {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: message = "RVV metadata-only first slice has no RVV lowering pipeline
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: plan_kind = "plugin-emission-plan"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
    // CHECK-SAME: runtime_abi_name = "rvv-executable-runtime-abi-deferred"
    // CHECK-SAME: runtime_glue_role = "deferred-rvv-runtime-glue"
    // CHECK-SAME: severity = "error"
    // CHECK-SAME: status = "unsupported"
    // CHECK-SAME: target = @rvv_first_slice
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @selected_marker_routes_only_selected
  tcrv.exec.kernel @selected_marker_routes_only_selected {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    // CHECK: tcrv.exec.variant @rvv_unselected
    tcrv.exec.variant @rvv_unselected attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    // CHECK: tcrv.exec.variant @rvv_selected
    tcrv.exec.variant @rvv_selected attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    tcrv.exec.diagnostic {
      message = "select only one RVV first-slice variant",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_selected
    }
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_selected,
      source_kernel = "selected_marker_routes_only_selected",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: target = @rvv_selected
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: selected_variant = @rvv_selected
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
    // CHECK-SAME: runtime_abi_name = "rvv-executable-runtime-abi-deferred"
    // CHECK-SAME: runtime_glue_role = "deferred-rvv-runtime-glue"
    // CHECK-SAME: target = @rvv_selected
    // CHECK-NOT: target = @rvv_unselected
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @dispatch_order_is_stable
  tcrv.exec.kernel @dispatch_order_is_stable {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_case_b attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    tcrv.exec.variant @rvv_case_a attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // CHECK: tcrv.exec.dispatch
    tcrv.exec.dispatch {
      // CHECK: tcrv.exec.case @rvv_case_b
      tcrv.exec.case @rvv_case_b {condition = "case_b_guard"}
      // CHECK: tcrv.exec.case @rvv_case_a
      tcrv.exec.case @rvv_case_a {condition = "case_a_guard"}
      // CHECK: tcrv.exec.fallback @scalar_fallback_first_slice
      tcrv.exec.fallback @scalar_fallback_first_slice
    }
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "dispatch case",
      selected_variant = @rvv_case_b,
      source_kernel = "dispatch_order_is_stable",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "dispatch case",
      selected_variant = @rvv_case_a,
      source_kernel = "dispatch_order_is_stable",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
    tcrv_scalar.lowering_boundary {
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "dispatch_order_is_stable",
      status = "metadata-only"
    }
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
    // CHECK-SAME: runtime_abi_name = "rvv-executable-runtime-abi-deferred"
    // CHECK-SAME: runtime_glue_role = "deferred-rvv-runtime-glue"
    // CHECK-SAME: target = @rvv_case_b
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
    // CHECK-SAME: runtime_abi_name = "rvv-executable-runtime-abi-deferred"
    // CHECK-SAME: runtime_glue_role = "deferred-rvv-runtime-glue"
    // CHECK-SAME: target = @rvv_case_a
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "dispatch fallback"
    // CHECK-SAME: runtime_abi_kind = "host-scalar-fallback-metadata"
    // CHECK-SAME: runtime_abi_name = "portable-scalar-fallback-metadata-abi.v1"
    // CHECK-SAME: runtime_glue_role = "metadata-only-host-fallback-boundary"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}
