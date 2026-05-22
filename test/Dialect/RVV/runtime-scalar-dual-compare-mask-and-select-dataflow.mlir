// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_runtime_scalar_dual_compare_mask_and_select_valid
  tcrv.exec.kernel @rvv_runtime_scalar_dual_compare_mask_and_select_valid {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_dual_compare_mask_and_select attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs_a = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_a = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_b = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-secondary-scalar-value"} : i32
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // CHECK: tcrv_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body
      tcrv_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body %cmp_lhs_a, %rhs_scalar_a, %cmp_lhs_b, %rhs_scalar_b, %true_value, %false_value, %out, %n {lmul = "m1", mask_composition = "and", mask_memory_form = "composed-compare-produced-mask", mask_role = "predicate-mask-produced-by-mask-and", mask_source = "mask-and-of-two-runtime-scalar-compare-produced-masks", memory_form = "runtime-scalar-dual-cmp-mask-and-select", op_kind = "runtime_scalar_dual_cmp_mask_and_select", predicate_kind_a = "sle", predicate_kind_b = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_runtime_scalar_dual_compare_mask_and_select_explicit_valid
  tcrv.exec.kernel @rvv_runtime_scalar_dual_compare_mask_and_select_explicit_valid {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_dual_compare_mask_and_select_explicit attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs_a = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_a = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_b = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-secondary-scalar-value"} : i32
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %cmp_lhs_a_vec = tcrv_rvv.load %cmp_lhs_a, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_a_vec = tcrv_rvv.splat %rhs_scalar_a, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %cmp_lhs_b_vec = tcrv_rvv.load %cmp_lhs_b, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_b_vec = tcrv_rvv.splat %rhs_scalar_b, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %true_vec = tcrv_rvv.load %true_value, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %false_vec = tcrv_rvv.load %false_value, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask_a = tcrv_rvv.compare %cmp_lhs_a_vec, %rhs_a_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %mask_b = tcrv_rvv.compare %cmp_lhs_b_vec, %rhs_b_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        // CHECK: tcrv_rvv.mask_and
        %composed = tcrv_rvv.mask_and %mask_a, %mask_b, %vl {kind = "and"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %selected = tcrv_rvv.select %composed, %true_vec, %false_vec, %vl : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %selected, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_dual_compare_mask_and_select_reject_rhs_b_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_dual_compare_mask_and_select_bad_rhs_b_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs_a = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_a = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_b = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires rhs scalar threshold B operand to bind runtime ABI role 'rhs-secondary-scalar-value'}}
      tcrv_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body %cmp_lhs_a, %rhs_scalar_a, %cmp_lhs_b, %rhs_scalar_b, %true_value, %false_value, %out, %n {lmul = "m1", mask_composition = "and", mask_memory_form = "composed-compare-produced-mask", mask_role = "predicate-mask-produced-by-mask-and", mask_source = "mask-and-of-two-runtime-scalar-compare-produced-masks", memory_form = "runtime-scalar-dual-cmp-mask-and-select", op_kind = "runtime_scalar_dual_cmp_mask_and_select", predicate_kind_a = "sle", predicate_kind_b = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_dual_compare_mask_and_select_reject_mask_composition {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_dual_compare_mask_and_select_bad_mask_composition attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs_a = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_a", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_a = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_a", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %cmp_lhs_b = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs_b", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar_b = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar_b", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-secondary-scalar-value"} : i32
      %true_value = tcrv_rvv.runtime_abi_value {c_name = "true_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "true-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %false_value = tcrv_rvv.runtime_abi_value {c_name = "false_value", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "false-value-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only mask_composition "and" for the bounded runtime scalar dual-compare mask-and select hook}}
      tcrv_rvv.typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body %cmp_lhs_a, %rhs_scalar_a, %cmp_lhs_b, %rhs_scalar_b, %true_value, %false_value, %out, %n {lmul = "m1", mask_composition = "or", mask_memory_form = "composed-compare-produced-mask", mask_role = "predicate-mask-produced-by-mask-and", mask_source = "mask-and-of-two-runtime-scalar-compare-produced-masks", memory_form = "runtime-scalar-dual-cmp-mask-and-select", op_kind = "runtime_scalar_dual_cmp_mask_and_select", predicate_kind_a = "sle", predicate_kind_b = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "select-true-value-when-mask-else-false-value", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_mask_and_reject_wrong_kind {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_mask_and_bad_kind attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask_a = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %mask_b = tcrv_rvv.compare %rhs_vec, %lhs_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        // expected-error@+1 {{currently supports only kind "and" for the bounded Stage 2 mask composition route}}
        %composed = tcrv_rvv.mask_and %mask_a, %mask_b, %vl {kind = "or"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
