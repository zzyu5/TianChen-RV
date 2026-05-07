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
    // CHECK: ^bb0(%[[N:.*]]: index):
    // CHECK: %[[VL:.*]] = tcrv_rvv.setvl %[[N]]
    // CHECK-SAME: lmul = "m1"
    // CHECK-SAME: policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // CHECK-SAME: sew = 32 : i64
    // CHECK: tcrv_rvv.with_vl %[[VL]] attributes
    // CHECK-SAME: lmul = "m1"
    // CHECK-SAME: policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // CHECK-SAME: sew = 32 : i64
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_valid"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_i32_vadd_microkernel_missing_with_vl {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    // expected-error@+1 {{requires exactly one tcrv_rvv.with_vl in the structured control-plane body}}
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_missing_with_vl"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "other_kernel"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 0 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_bad_count"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_missing_caps"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_unavailable_rvv"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_stale,
      source_kernel = "rvv_i32_vadd_microkernel_stale_variant"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv_token",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_bad_march"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
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
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "rvv_i32_vadd_microkernel_tensor_attr",
      tensor_shape = "16xi32"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      } : !tcrv_rvv.vl
    }
  }
}
