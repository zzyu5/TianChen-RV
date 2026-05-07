// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @scalar_microkernel_valid
  tcrv.exec.kernel @scalar_microkernel_valid {
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
    // CHECK: tcrv_scalar.i32_vadd_microkernel
    // CHECK-SAME: element_count = 16 : i64
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: required_capabilities = [@scalar_fallback]
    // CHECK-SAME: role = "dispatch fallback"
    // CHECK-SAME: selected_variant = @scalar_fallback_first_slice
    // CHECK-SAME: source_kernel = "scalar_microkernel_valid"
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_microkernel_valid"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_microkernel_unknown_attr {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute '"shape"'}}
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      shape = "generic_tensor",
      source_kernel = "scalar_microkernel_unknown_attr"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_microkernel_bad_count {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{element_count must be in the bounded smoke range [1, 64]}}
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 0 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_microkernel_bad_count"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_microkernel_unavailable_capability {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "unavailable"
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{requires unavailable capability @scalar_fallback}}
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "scalar_microkernel_unavailable_capability"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_microkernel_secret_string {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{attribute 'source_kernel' must not contain secret-like or raw credential text}}
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "token_leak"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @scalar_microkernel_stale_variant {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback"}
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    // expected-error@+1 {{selected_variant @old_scalar must resolve to a direct sibling tcrv.exec.variant}}
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "direct variant",
      selected_variant = @old_scalar,
      source_kernel = "scalar_microkernel_stale_variant"
    }
  }
}
