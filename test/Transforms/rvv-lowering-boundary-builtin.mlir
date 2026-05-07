// RUN: tcrv-opt %s --split-input-file --tcrv-select-variants --tcrv-materialize-rvv-lowering-boundary --tcrv-materialize-emission-plans | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_rvv_plus_scalar_boundary
  tcrv.exec.kernel @public_rvv_plus_scalar_boundary {
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
    // CHECK: tcrv.exec.variant @rvv_first_slice
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // CHECK: tcrv.exec.variant @scalar_fallback_first_slice
    // CHECK-SAME: fallback_role = "conservative"
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    // CHECK: tcrv.exec.dispatch
    // CHECK: tcrv.exec.case @rvv_first_slice
    // CHECK-SAME: condition = "capability_available"
    // CHECK-SAME: guard = "plugin_local_rvv_first_slice"
    // CHECK-SAME: policy = "metadata_only_first_slice"
    // CHECK: tcrv.exec.fallback @scalar_fallback_first_slice
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: capability_summary = "rvv"
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "public_rvv_plus_scalar_boundary"
    // CHECK-SAME: status = "unsupported"
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: role = "dispatch case"
    // CHECK-SAME: status = "unsupported"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: artifact_kind = "metadata-diagnostic"
    // CHECK-SAME: emission_kind = "portable-scalar-fallback-metadata-route"
    // CHECK-SAME: lowering_pipeline = "none-executable-metadata-only"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: role = "dispatch fallback"
    // CHECK-SAME: runtime_abi = "none-metadata-only"
    // CHECK-SAME: status = "metadata-only"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_scalar_only_no_rvv_boundary
  tcrv.exec.kernel @public_scalar_only_no_rvv_boundary {
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
    // CHECK-NOT: tcrv_rvv.lowering_boundary
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: selection_kind = "fallback-only"
    // CHECK-SAME: target = @scalar_fallback_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: status = "metadata-only"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_rvv_without_scalar_boundary
  tcrv.exec.kernel @public_rvv_without_scalar_boundary {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    // CHECK-NOT: tcrv.exec.dispatch
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "static-variant"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "fallback-coverage-missing"
    // CHECK-SAME: selection_kind = "missing-conservative-fallback"
    // CHECK-SAME: status = "missing"
    // CHECK-SAME: target = @rvv_first_slice
    // CHECK: tcrv_rvv.lowering_boundary
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "public_rvv_without_scalar_boundary"
    // CHECK-SAME: status = "unsupported"
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: status = "unsupported"
    // CHECK-SAME: target = @rvv_first_slice
  }
}
