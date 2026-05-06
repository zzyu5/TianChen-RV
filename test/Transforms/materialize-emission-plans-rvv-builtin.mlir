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
    // CHECK-SAME: role = "direct variant"
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
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: target = @rvv_selected
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "emission_plan"
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
    tcrv.exec.variant @rvv_fallback attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    // CHECK: tcrv.exec.dispatch
    tcrv.exec.dispatch {
      // CHECK: tcrv.exec.case @rvv_case_b
      tcrv.exec.case @rvv_case_b {condition = "case_b_guard"}
      // CHECK: tcrv.exec.case @rvv_case_a
      tcrv.exec.case @rvv_case_a {condition = "case_a_guard"}
      // CHECK: tcrv.exec.fallback @rvv_fallback
      tcrv.exec.fallback @rvv_fallback
    }
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: target = @rvv_case_b
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: target = @rvv_case_a
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: role = "dispatch fallback"
    // CHECK-SAME: target = @rvv_fallback
  }
}
