// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_with_vl_valid
  tcrv.exec.kernel @rvv_with_vl_valid {
    // CHECK: %[[AVL:[0-9]+]] = builtin.unrealized_conversion_cast to index
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // CHECK: %[[VL:[0-9]+]] = tcrv_rvv.setvl %[[AVL]]
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    // CHECK: tcrv_rvv.with_vl %[[VL]] attributes {
    // CHECK-SAME: lmul = "m1"
    // CHECK-SAME: policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // CHECK-SAME: sew = 32 : i64
    // CHECK: } : !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_with_vl_selected_boundary_attrs
  tcrv.exec.kernel @rvv_with_vl_selected_boundary_attrs {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_selected attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %avl = "builtin.unrealized_conversion_cast"() : () -> index
      %vl = tcrv_rvv.setvl %avl {
        lmul = "m1",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        sew = 32 : i64
      } : index -> !tcrv_rvv.vl
      // CHECK: tcrv_rvv.with_vl
      // CHECK-SAME: origin = "rvv-plugin"
      // CHECK-SAME: required_capabilities = [@rvv]
      // CHECK-SAME: rvv_construction_protocol = "extension-family-construction-protocol.v1"
      // CHECK-SAME: rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family"
      // CHECK-SAME: selected_path_role = "direct variant"
      // CHECK-SAME: selected_variant = @rvv_selected
      // CHECK-SAME: source_kernel = "rvv_with_vl_selected_boundary_attrs"
      // CHECK-SAME: status = "selected-lowering-boundary"
      tcrv_rvv.with_vl %vl attributes {
        lmul = "m1",
        origin = "rvv-plugin",
        policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
        required_capabilities = [@rvv],
        rvv_construction_protocol = "extension-family-construction-protocol.v1",
        rvv_emitc_route_mapping = "rvv-generic-typed-body-emitc-route-family",
        selected_path_role = "direct variant",
        selected_variant = @rvv_selected,
        sew = 32 : i64,
        source_kernel = "rvv_with_vl_selected_boundary_attrs",
        status = "selected-lowering-boundary"
      } {
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_with_vl_reject_operand_type {
    %bad = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{requires runtime VL operand to have !tcrv_rvv.vl type}}
    tcrv_rvv.with_vl %bad {
    } : index
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_with_vl_reject_element_count {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    // expected-error@+1 {{does not accept attribute '"element_count"'}}
    tcrv_rvv.with_vl %vl attributes {
      element_count = 16 : i64
    } {
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_with_vl_reject_policy_mismatch {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    // expected-error@+1 {{requires optional 'policy' metadata to match defining tcrv_rvv.setvl}}
    tcrv_rvv.with_vl %vl attributes {
      policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>
    } {
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_with_vl_reject_sew_mismatch {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    // expected-error@+1 {{requires bounded RVV first-slice compile-time config to be SEW32 with LMUL "m1" or "m2"}}
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 16 : i64
    } {
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_with_vl_reject_lmul_mismatch {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    // expected-error@+1 {{requires optional 'lmul' metadata to match defining tcrv_rvv.setvl}}
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m2",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
    } : !tcrv_rvv.vl
  }
}
