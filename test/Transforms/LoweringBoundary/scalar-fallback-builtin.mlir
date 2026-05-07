// RUN: tcrv-opt %s --tcrv-verify-plugin-variant-legality --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-check-emission-paths --tcrv-materialize-emission-plans | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @public_scalar_fallback
  tcrv.exec.kernel @public_scalar_fallback {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // CHECK: tcrv.exec.variant @scalar_fallback_first_slice
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback],
      policy = "portable_scalar_fallback_first_slice"
    } {
    }
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: fallback_role = "conservative"
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: preference_available = true
    // CHECK-SAME: preference_rank = 0
    // CHECK-SAME: preference_score
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "fallback-only"
    // CHECK-SAME: status = "selected"
    // CHECK-SAME: target = @scalar_fallback_first_slice
    // CHECK: tcrv_scalar.lowering_boundary
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK-SAME: source_kernel = "public_scalar_fallback"
    // CHECK-SAME: status = "metadata-only"
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: artifact_kind = "metadata-diagnostic"
    // CHECK-SAME: emission_kind = "portable-scalar-fallback-metadata-route"
    // CHECK-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
    // CHECK-SAME: lowering_pipeline = "none-executable-metadata-only"
    // CHECK-SAME: message = "scalar fallback first slice records a portable fallback metadata route
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: plan_kind = "plugin-emission-plan"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: runtime_abi = "none-metadata-only"
    // CHECK-SAME: runtime_abi_kind = "host-scalar-fallback-metadata"
    // CHECK-SAME: runtime_abi_name = "portable-scalar-fallback-metadata-abi.v1"
    // CHECK-SAME: runtime_glue_role = "metadata-only-host-fallback-boundary"
    // CHECK-SAME: severity = "note"
    // CHECK-SAME: status = "metadata-only"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}
