// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// Deprecated Stage1 inventory only: finite weft_rvv.i32_* ops and
// !weft_rvv.i32m* types are retained here to exercise parser/verifier
// diagnostics. This file must not be used as positive route, EmitC, target
// artifact, source-front-door, runtime, correctness, or performance evidence.

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_legacy_i32_dataflow_deprecated_parse_inventory
  weft.exec.kernel @rvv_legacy_i32_dataflow_deprecated_parse_inventory {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // CHECK: weft_rvv.i32_load
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
      %rhs = weft_rvv.i32_load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
      // CHECK: weft_rvv.i32_broadcast_load
      %rhs_broadcast = weft_rvv.i32_broadcast_load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
      // CHECK: weft_rvv.i32_add
      %sum = weft_rvv.i32_add %lhs, %rhs, %vl : !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1
      // CHECK: weft_rvv.i32_sub
      %diff = weft_rvv.i32_sub %lhs, %rhs_broadcast, %vl : !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1
      // CHECK: weft_rvv.i32_mul
      %product = weft_rvv.i32_mul %sum, %diff, %vl : !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1
      // CHECK: weft_rvv.i32_cmp_eq
      %mask = weft_rvv.i32_cmp_eq %sum, %product, %vl : !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1_mask
      // CHECK: weft_rvv.i32_select
      %selected = weft_rvv.i32_select %mask, %sum, %product, %vl : !weft_rvv.i32m1_mask, !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1
      // CHECK: weft_rvv.i32_store
      weft_rvv.i32_store %out_ptr, %selected, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.i32m1, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_legacy_i32m2_dataflow_deprecated_parse_inventory
  weft.exec.kernel @rvv_legacy_i32m2_dataflow_deprecated_parse_inventory {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m2",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m2",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // CHECK: weft_rvv.i32_load
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m2
      %rhs = weft_rvv.i32_load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m2
      // CHECK: weft_rvv.i32_sub
      %diff = weft_rvv.i32_sub %lhs, %rhs, %vl : !weft_rvv.i32m2, !weft_rvv.i32m2, !weft_rvv.vl -> !weft_rvv.i32m2
      // CHECK: weft_rvv.i32_store
      weft_rvv.i32_store %out_ptr, %diff, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.i32m2, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_select_reject_mask_not_typed_compare {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
      %rhs = weft_rvv.i32_load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
      %mask = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.i32m1_mask
      // expected-error@+1 {{requires mask operand to be produced by weft_rvv.i32_cmp_eq inside the selected RVV typed body}}
      %selected = weft_rvv.i32_select %mask, %lhs, %rhs, %vl : !weft_rvv.i32m1_mask, !weft_rvv.i32m1, !weft_rvv.i32m1, !weft_rvv.vl -> !weft_rvv.i32m1
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_i32m2_reject_m1_dataflow {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m2",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m2",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // expected-error@+1 {{requires result type '!weft_rvv.i32m1' to agree with enclosing weft_rvv.with_vl LMUL metadata 'm2'}}
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_compare_reject_m2_predicate_form {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m2",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m2",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m2
      %rhs = weft_rvv.i32_load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m2
      // expected-error@+1 {{requires lhs type to be !weft_rvv.i32m1}}
      %mask = weft_rvv.i32_cmp_eq %lhs, %rhs, %vl : !weft_rvv.i32m2, !weft_rvv.i32m2, !weft_rvv.vl -> !weft_rvv.i32m1_mask
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_broadcast_load_reject_lhs_role {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // expected-error@+1 {{requires broadcast RHS buffer operand to bind runtime ABI role 'rhs-input-buffer'}}
      %rhs = weft_rvv.i32_broadcast_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_dataflow_reject_outside_with_vl {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    // expected-error@+1 {{must be nested within a weft_rvv.with_vl body}}
    %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
  }
}

// -----

module {
  weft.exec.kernel @rvv_dataflow_reject_wrong_vl_token {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    %other = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // expected-error@+1 {{requires RVV dataflow op to consume the !weft_rvv.vl token owned by the surrounding weft_rvv.with_vl}}
      %lhs = weft_rvv.i32_load %lhs_ptr, %other : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_dataflow_reject_missing_with_vl_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl {
      // expected-error@+1 {{requires enclosing weft_rvv.with_vl to carry explicit SEW metadata for bounded RVV i32 dataflow}}
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_dataflow_reject_missing_with_vl_policy {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      sew = 32 : i64
    } {
      // expected-error@+1 {{requires enclosing weft_rvv.with_vl to carry explicit policy metadata for bounded RVV i32 dataflow}}
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_dataflow_reject_element_count {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // expected-error@+1 {{does not accept attribute '"element_count"'}}
      %lhs = weft_rvv.i32_load %lhs_ptr, %vl {element_count = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.i32m1
    } : !weft_rvv.vl
  }
}
