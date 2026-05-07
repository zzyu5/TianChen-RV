// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=runtime_success --implicit-check-not=password --implicit-check-not=token

module @scalar_microkernel_plan_input {
  tcrv.exec.kernel @scalar_microkernel_direct {
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
      message = "static scalar microkernel fallback path selected by test fixture",
      origin = "scalar-plugin",
      reason = "variant-selected",
      selection_kind = "fallback-only",
      severity = "note",
      status = "selected",
      target = @scalar_fallback_first_slice
    }
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_microkernel_direct"
    }
  }
}

// CHECK-LABEL: tcrv.exec.kernel @scalar_microkernel_direct
// CHECK: tcrv_scalar.lowering_boundary
// CHECK-SAME: selected_variant = @scalar_fallback_first_slice
// CHECK-SAME: status = "metadata-only"
// CHECK: tcrv.exec.diagnostic
// CHECK-SAME: artifact_kind = "standalone-c-source"
// CHECK-SAME: emission_kind = "scalar-explicit-i32-vadd-microkernel-c-source"
// CHECK-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
// CHECK-SAME: lowering_pipeline = "tcrv-export-scalar-microkernel-c"
// CHECK-SAME: message = "explicit scalar i32 vector-add microkernel C source export is available for this selected fallback path; this is not generic scalar lowering, runtime ABI integration, arbitrary kernel emission, correctness, or performance evidence"
// CHECK-SAME: origin = "scalar-plugin"
// CHECK-SAME: reason = "emission_plan"
// CHECK-SAME: required_capabilities = [@scalar_fallback]
// CHECK-SAME: role = "direct variant"
// CHECK-SAME: runtime_abi = "scalar-i32-vadd-standalone-c-self-check.v1"
// CHECK-SAME: runtime_abi_kind = "scalar-standalone-c-source-export"
// CHECK-SAME: runtime_abi_name = "scalar-i32-vadd-microkernel-standalone-c.v1"
// CHECK-SAME: runtime_glue_role = "standalone-self-check-main"
// CHECK-SAME: severity = "info"
// CHECK-SAME: status = "supported"
// CHECK-SAME: target = @scalar_fallback_first_slice
