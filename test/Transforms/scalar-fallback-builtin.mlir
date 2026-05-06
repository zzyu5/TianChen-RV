// RUN: tcrv-opt %s --tcrv-select-variants --tcrv-check-emission-paths --tcrv-materialize-emission-plans | FileCheck %s

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
    // CHECK-SAME: reason = "variant-selected"
    // CHECK-SAME: selection_kind = "fallback-only"
    // CHECK-SAME: status = "selected"
    // CHECK-SAME: target = @scalar_fallback_first_slice
    // CHECK: tcrv.exec.diagnostic
    // CHECK-SAME: artifact_kind = "mlir-lowering-plan"
    // CHECK-SAME: emission_kind = "portable-scalar-fallback"
    // CHECK-SAME: lowering_pipeline = "mlir-default-scalar-lowering"
    // CHECK-SAME: message = "scalar fallback first slice records a portable fallback emission route
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: plan_kind = "plugin-emission-plan"
    // CHECK-SAME: reason = "emission_plan"
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: runtime_abi = "none-required"
    // CHECK-SAME: severity = "info"
    // CHECK-SAME: status = "supported"
    // CHECK-SAME: target = @scalar_fallback_first_slice
  }
}
