// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_i32_dataflow_valid
  tcrv.exec.kernel @rvv_i32_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // CHECK: tcrv_rvv.i32_load
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      // CHECK: tcrv_rvv.i32_broadcast_load
      %rhs_broadcast = tcrv_rvv.i32_broadcast_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      // CHECK: tcrv_rvv.i32_add
      %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      // CHECK: tcrv_rvv.i32_sub
      %diff = tcrv_rvv.i32_sub %lhs, %rhs_broadcast, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      // CHECK: tcrv_rvv.i32_mul
      %product = tcrv_rvv.i32_mul %sum, %diff, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      // CHECK: tcrv_rvv.i32_cmp_eq
      %mask = tcrv_rvv.i32_cmp_eq %sum, %product, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1_mask
      // CHECK: tcrv_rvv.i32_select
      %selected = tcrv_rvv.i32_select %mask, %sum, %product, %vl : !tcrv_rvv.i32m1_mask, !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      // CHECK: tcrv_rvv.i32_store
      tcrv_rvv.i32_store %out_ptr, %selected, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m1, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_i32m2_dataflow_valid
  tcrv.exec.kernel @rvv_i32m2_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m2",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m2",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // CHECK: tcrv_rvv.i32_load
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
      %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
      // CHECK: tcrv_rvv.i32_sub
      %diff = tcrv_rvv.i32_sub %lhs, %rhs, %vl : !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
      // CHECK: tcrv_rvv.i32_store
      tcrv_rvv.i32_store %out_ptr, %diff, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.i32m2, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_select_reject_mask_not_typed_compare {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
      %mask = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.i32m1_mask
      // expected-error@+1 {{requires mask operand to be produced by tcrv_rvv.i32_cmp_eq inside the selected RVV typed body}}
      %selected = tcrv_rvv.i32_select %mask, %lhs, %rhs, %vl : !tcrv_rvv.i32m1_mask, !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_compare_reject_m2_predicate_form {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m2",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m2",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
      %rhs = tcrv_rvv.i32_load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
      // expected-error@+1 {{requires lhs type to be !tcrv_rvv.i32m1}}
      %mask = tcrv_rvv.i32_cmp_eq %lhs, %rhs, %vl : !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m1_mask
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_broadcast_load_reject_lhs_role {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // expected-error@+1 {{requires broadcast RHS buffer operand to bind runtime ABI role 'rhs-input-buffer'}}
      %rhs = tcrv_rvv.i32_broadcast_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_dataflow_reject_outside_with_vl {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    // expected-error@+1 {{must be nested directly in a tcrv_rvv.with_vl body}}
    %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_dataflow_reject_wrong_vl_token {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    %other = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // expected-error@+1 {{requires RVV dataflow op to consume the !tcrv_rvv.vl token owned by the surrounding tcrv_rvv.with_vl}}
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %other : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_dataflow_reject_missing_with_vl_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl {
      // expected-error@+1 {{requires enclosing tcrv_rvv.with_vl to carry explicit SEW metadata for bounded RVV i32 dataflow}}
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_dataflow_reject_missing_with_vl_policy {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      sew = 32 : i64
    } {
      // expected-error@+1 {{requires enclosing tcrv_rvv.with_vl to carry explicit policy metadata for bounded RVV i32 dataflow}}
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_dataflow_reject_element_count {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {
      lmul = "m1",
      policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      sew = 32 : i64
    } {
      // expected-error@+1 {{does not accept attribute '"element_count"'}}
      %lhs = tcrv_rvv.i32_load %lhs_ptr, %vl {element_count = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.i32m1
    } : !tcrv_rvv.vl
  }
}
