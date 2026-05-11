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
    // CHECK: tcrv_scalar.i32_vadd_microkernel
    // CHECK-SAME: element_count = 16 : i64
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK-SAME: source_kernel = "public_scalar_fallback"
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: artifact_kind = "runtime-callable-c-source"
    // CHECK-SAME: emission_kind = "scalar-explicit-i32-vadd-microkernel-c-source"
    // CHECK-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
    // CHECK-SAME: lowering_pipeline = "tcrv-export-scalar-microkernel-c"
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: plan_kind = "plugin-emission-plan"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: runtime_abi = "scalar-i32-vadd-runtime-callable-c-abi.v1"
    // CHECK-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
    // CHECK-SAME: runtime_abi_name = "scalar-i32-vadd-runtime-callable-c-function.v1"
    // CHECK-SAME: runtime_glue_role = "runtime-callable-i32-vadd-fallback-function"
    // CHECK-SAME: selected_plan_metadata = [{{.*}}name = "tcrv_scalar.emitc_source_op"{{.*}}value = "tcrv_scalar.i32_vadd_microkernel"{{.*}}name = "tcrv_scalar.emitc_lowerable_op_interface"{{.*}}value = "TCRVEmitCLowerableOpInterface"
    // CHECK-SAME: status = "supported"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}
