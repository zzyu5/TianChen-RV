// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_i32_vadd_microkernel_valid
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_valid {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // CHECK: tcrv_rvv.i32_vadd_microkernel
    // CHECK-SAME: element_count = 16 : i64
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: required_capabilities = [@rvv]
    // CHECK-SAME: required_march = "rv64gcv"
    // CHECK-SAME: role = "direct variant"
    // CHECK-SAME: selected_mabi = "lp64d"
    // CHECK-SAME: selected_variant = @rvv_first_slice
    // CHECK-SAME: source_kernel = "rvv_i32_vadd_microkernel_valid"
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_valid"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_bad_source {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{source_kernel must match enclosing tcrv.exec.kernel symbol @rvv_i32_vadd_microkernel_bad_source}}
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "other_kernel"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_bad_count {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{element_count must be in the bounded smoke range [1, 64]}}
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 0 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_bad_count"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_missing_caps {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{requires non-empty array attribute 'required_capabilities'}}
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_missing_caps"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_unavailable_rvv {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "unavailable"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{requires unavailable capability @rvv}}
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_unavailable_rvv"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_stale_variant {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{selected_variant @rvv_stale must resolve to a direct sibling tcrv.exec.variant}}
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_stale,
      source_kernel = "rvv_i32_vadd_microkernel_stale_variant"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_bad_march {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{attribute 'required_march' must not contain secret-like or raw credential text}}
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv_token",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_bad_march"
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_tensor_attr {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{does not accept generic tensor/tile/benchmark or unknown attribute '"tensor_shape"'}}
    tcrv_rvv.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_tensor_attr",
      tensor_shape = "16xi32"
    }
  }
}
