// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @scalar_i64_vadd_microkernel_valid
  tcrv.exec.kernel @scalar_i64_vadd_microkernel_valid {
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
      tcrv_scalar.lowering_descriptor = "i64-vadd-microkernel.v1"
    } {
    }
    // CHECK: tcrv_scalar.i64_vadd_microkernel
    // CHECK-SAME: element_count = 16 : i64
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "dispatch fallback"
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK-SAME: source_kernel = "scalar_i64_vadd_microkernel_valid"
    tcrv_scalar.i64_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_i64_vadd_microkernel_valid"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_i64_vadd_bad_count {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{element_count must be in the bounded smoke range [1, 64]}}
    tcrv_scalar.i64_vadd_microkernel {
      element_count = 65 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_i64_vadd_bad_count"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_i64_vadd_wrong_origin {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{origin must be 'scalar-plugin' because this executable microkernel is scalar plugin-local}}
    tcrv_scalar.i64_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_i64_vadd_wrong_origin"
    }
  }
}
