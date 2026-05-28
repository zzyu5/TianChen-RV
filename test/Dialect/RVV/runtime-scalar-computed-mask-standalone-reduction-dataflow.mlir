// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_valid
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_valid {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // CHECK: tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_i64_valid
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_i64_valid {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_i64 attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int64_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i64
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // CHECK: tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i64-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 64 : i64} : (!tcrv_rvv.runtime_abi_value, i64, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_predicate {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_predicate attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only predicate_kind "sle" for the bounded selected-body runtime scalar computed-mask standalone reduction hook}}
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_wrong_op_kind {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_op_kind attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only op_kind "runtime_scalar_cmp_masked_standalone_reduce_add" for the bounded selected-body runtime scalar computed-mask standalone reduction hook}}
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "computed_mask_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_wrong_memory_form {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_memory_form attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only memory_form "runtime-scalar-computed-mask-unit-stride-standalone-reduction" for the bounded selected-body runtime scalar computed-mask standalone reduction hook}}
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_rhs_scalar_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_rhs_scalar_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      // expected-error@+1 {{requires runtime ABI role 'source-input-buffer' to use C type 'const int16_t *', 'const int32_t *', or 'const int64_t *'}}
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "source-input-buffer"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_source_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_source_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires source operand to bind runtime ABI role 'source-input-buffer'}}
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_output_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_output_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires scalar output operand to bind runtime ABI role 'output-buffer'}}
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_bad_config {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires bounded pre-realized runtime scalar computed-mask standalone reduction config to be SEW32 LMUL m1 or SEW64 LMUL m1; SEW32 LMUL m2 requires a separate LMUL m1 scalar reduction accumulator/result channel and fails closed in this bounded route family}}
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_runtime_scalar_computed_mask_standalone_reduce_reject_authority_attr {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_runtime_scalar_computed_mask_standalone_reduce_bad_authority_attr attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      tcrv_rvv.typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body %cmp_lhs, %rhs_scalar, %src, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-standalone-reduction", op_kind = "runtime_scalar_cmp_masked_standalone_reduce_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-standalone-reduction-lane0-to-output-scalar", route_id = "rvv-i32m1", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}
