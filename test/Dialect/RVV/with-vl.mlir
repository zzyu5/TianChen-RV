// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_with_vl_valid
  weft.exec.kernel @rvv_with_vl_valid {
    // CHECK: %[[AVL:[0-9]+]] = builtin.unrealized_conversion_cast to index
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // CHECK: %[[VL:[0-9]+]] = weft_rvv.setvl %[[AVL]]
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    // CHECK: weft_rvv.with_vl %[[VL]] attributes {
    // CHECK-SAME: lmul = "m1"
    // CHECK-SAME: policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    // CHECK-SAME: sew = 32 : i64
    // CHECK: } : !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
    } : !weft_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_with_vl_exact_body_attrs
  weft.exec.kernel @rvv_with_vl_exact_body_attrs {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @rvv_selected attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %avl = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = weft_rvv.setvl %avl {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !weft_rvv.vl
      // CHECK: weft_rvv.with_vl
      // CHECK-SAME: lmul = "m1"
      // CHECK-SAME: policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
      // CHECK-SAME: sew = 32 : i64
      // CHECK-NOT: selected_variant
      // CHECK-NOT: rvv_construction_protocol
      // CHECK: } : !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {
        lmul = "m1",
        policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } {
      } : !weft_rvv.vl
    }
  }
}

// -----

module {
  weft.exec.kernel @rvv_with_vl_reject_retired_route_mirror {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    // expected-error@+1 {{source/selection/capability/protocol/route mirrors belong outside the exact typed body; unexpected attribute '"rvv_emitc_route_mapping"'}}
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      rvv_emitc_route_mapping = "retired-second-authority",
      sew = 32 : i64
    } {
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_with_vl_reject_operand_type {
    %bad = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{requires runtime VL operand to have !weft_rvv.vl type}}
    weft_rvv.with_vl %bad {
    } : index
  }
}

// -----

module {
  weft.exec.kernel @rvv_with_vl_reject_element_count {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    // expected-error@+1 {{does not accept attribute '"element_count"'}}
    weft_rvv.with_vl %vl attributes {
      element_count = 16 : i64
    } {
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_with_vl_reject_policy_mismatch {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    // expected-error@+1 {{requires optional 'policy' metadata to match defining weft_rvv.setvl}}
    weft_rvv.with_vl %vl attributes {
      policy = #weft_rvv.policy<tail = undisturbed, mask = agnostic>
    } {
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_with_vl_reject_sew_mismatch {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    // SEW16 LMUL m8 on the with_vl is rejected by the config check (which fires
    // before the setvl/with_vl-match check): not a first-slice dataflow config
    // and not a deferred-wide dot-reduce strip rung (the i16 source rungs cap at
    // m4). The setvl above stays a valid SEW32/m1 first-slice config.
    // expected-error@+1 {{requires bounded RVV first-slice compile-time config to be SEW32 with LMUL "m1" or "m2", or SEW64 with LMUL "m1"}}
    weft_rvv.with_vl %vl attributes {
      lmul = "m8",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 16 : i64
    } {
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_with_vl_reject_lmul_mismatch {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    // expected-error@+1 {{requires optional 'lmul' metadata to match defining weft_rvv.setvl}}
    weft_rvv.with_vl %vl attributes {
      lmul = "m2",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
    } : !weft_rvv.vl
  }
}
