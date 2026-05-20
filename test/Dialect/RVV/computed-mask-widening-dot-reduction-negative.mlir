// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

module {
  tcrv.exec.kernel @rvv_pre_realized_masked_dot_reject_mask_source {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_masked_dot_bad_mask_source attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{currently supports only mask_source "compare-produced-mask-same-vl-scope" for the bounded computed-mask widening dot-product reduction hook}}
      tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "runtime_abi:mask", memory_form = "computed-mask-unit-stride-widening-dot-reduce", op_kind = "signed_masked_widening_dot_reduce_add", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_masked_dot_reject_route_id {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_masked_dot_bad_route_id attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-widening-dot-reduce", op_kind = "signed_masked_widening_dot_reduce_add", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, route_id = "rvv-i32m1", source_lmul = "mf2", source_sew = 16 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_masked_dot_reject_dot_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_masked_dot_bad_dot_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires dot lhs operand to bind runtime ABI role 'dot-lhs-input-buffer'}}
      tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-widening-dot-reduce", op_kind = "signed_masked_widening_dot_reduce_add", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_masked_dot_reject_source_config {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_masked_dot_bad_source_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires typed computed-mask widening dot-product reduction config/relation to match op_kind "signed_masked_widening_dot_reduce_add" with source SEW16 LMUL mf2}}
      tcrv_rvv.typed_computed_mask_widening_dot_reduce_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-widening-dot-reduce", op_kind = "signed_masked_widening_dot_reduce_add", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "m1", source_sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}
