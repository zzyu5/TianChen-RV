// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

module {
  tcrv.exec.kernel @rvv_pre_realized_strided_dot_reject_stride_unit {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_strided_dot_bad_stride_unit attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      // expected-error@+1 {{currently supports only stride_unit "element" for the bounded selected-body strided-input widening dot-product reduction hook}}
      tcrv_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n, %lhs_stride, %rhs_stride {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "strided-input-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64, stride_unit = "byte"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_strided_dot_reject_lhs_stride_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_strided_dot_bad_lhs_stride_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      // expected-error@+1 {{requires lhs stride operand to bind runtime ABI role 'lhs-input-stride'}}
      tcrv_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n, %lhs_stride, %rhs_stride {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "strided-input-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64, stride_unit = "element"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_strided_dot_reject_missing_n_role {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_strided_dot_bad_n_role attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      // expected-error@+1 {{requires runtime n/AVL operand to bind runtime ABI role 'runtime-element-count'}}
      tcrv_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n, %lhs_stride, %rhs_stride {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "strided-input-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64, stride_unit = "element"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_strided_dot_reject_source_config {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_strided_dot_bad_source_config attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      // expected-error@+1 {{requires typed strided-input widening dot-product reduction config/relation to match op_kind "signed_widening_dot_reduce_add" with source SEW16 LMUL mf2}}
      tcrv_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n, %lhs_stride, %rhs_stride {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "strided-input-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "m1", source_sew = 32 : i64, stride_unit = "element"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index) -> ()
    }
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_strided_dot_reject_route_id {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_pre_realized_strided_dot_bad_route_id attributes {origin = "rvv-plugin", requires = [@rvv]} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      %lhs_stride = tcrv_rvv.runtime_abi_value {c_name = "lhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "lhs-input-stride"} : index
      %rhs_stride = tcrv_rvv.runtime_abi_value {c_name = "rhs_stride", c_type = "size_t", ownership = "target-export-abi-owned", role = "rhs-input-stride"} : index
      // expected-error@+1 {{does not accept authority metadata attribute '"route_id"'}}
      tcrv_rvv.typed_strided_input_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n, %lhs_stride, %rhs_stride {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "strided-input-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, route_id = "rvv-i32m1", source_lmul = "mf2", source_sew = 16 : i64, stride_unit = "element"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index) -> ()
    }
  }
}
