// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_setvl_valid
  tcrv.exec.kernel @rvv_setvl_valid {
    // CHECK: %[[AVL:[0-9]+]] = builtin.unrealized_conversion_cast to index
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // CHECK: %{{.*}} = tcrv_rvv.setvl %[[AVL]]
    // CHECK-SAME: lmul = "m1"
    // CHECK-SAME: policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // CHECK-SAME: sew = 32 : i64
    // CHECK-SAME: : index -> !tcrv_rvv.vl
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_avl_attr {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{requires AVL to be a runtime SSA operand; attribute 'avl' is not accepted as an AVL substitute}}
    %vl = tcrv_rvv.setvl %avl {
      avl = 16 : i64,
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_sew {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{requires bounded RVV first-slice compile-time config to be SEW32 with LMUL "m1" or "m2", or SEW64 with LMUL "m1"}}
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 16 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_lmul {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{requires bounded RVV first-slice compile-time config to be SEW32 with LMUL "m1" or "m2", or SEW64 with LMUL "m1"}}
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m4",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_vlen {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{does not accept attribute '"vlen"'}}
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64,
      vlen = 128 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_vlenb {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{does not accept attribute '"vlenb"'}}
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64,
      vlenb = 16 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_element_count {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{does not accept attribute '"element_count"'}}
    %vl = tcrv_rvv.setvl %avl {
      element_count = 16 : i64,
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_required_march {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{does not accept attribute '"required_march"'}}
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      required_march = "rv64gcv",
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_setvl_reject_required_capabilities {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{does not accept attribute '"required_capabilities"'}}
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      required_capabilities = [@rvv],
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
  }
}
