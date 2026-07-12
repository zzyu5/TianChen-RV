// RUN: weft-opt %s --split-input-file --verify-diagnostics

// The N3 resource-aware max-legal-LMUL deferred-wide low-precision contraction
// typed body (the measured ssh-rvv winner var_v_m2_a1.c modeled as first-class
// structure): i8m2 strip-load -> i16m4 widening product -> a DEFERRED i32m8
// vector accumulate (weft_rvv.widening_accumulate, vwadd.wv) -> ONE trailing
// weft_rvv.standalone_reduce (i32m8 -> i32m1, kind "add") -> i32->f32 dequant.
// The deferred-accumulate op is the structural marker (I5): emission follows op
// identity, not metadata. The enclosing with_vl is the strip config SEW8/m2.
module {
  weft.exec.kernel @rvv_deferred_wide_accumulate_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %acc_ptr = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
    %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
      %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
      %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !weft_rvv.vector<i8, "m2">, !weft_rvv.vector<i8, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m4">
      %acc_vec = weft_rvv.widening_accumulate %product, %vl {accumulate_relation = "signed-i16m4-into-i32m8-deferred-add", kind = "signed_widening_accumulate_add"} : !weft_rvv.vector<i16, "m4">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m8">
      %reduced = weft_rvv.standalone_reduce %acc_vec, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i32, "m8">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      %deq = weft_rvv.dequantize %reduced, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      weft_rvv.store %out_ptr, %deq, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

// The deferred-wide accumulate REQUIRES the i8m2 x i8m2 -> i16m4 product (the
// structural source of the wide schedule); a narrow i16mf2 product is rejected.
module {
  weft.exec.kernel @rvv_deferred_wide_accumulate_rejects_narrow_product {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %product = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i16, "mf2">
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
      // expected-error@+1 {{requires product operand to have type !weft_rvv.vector<i16, "m4"> for the deferred-wide widening accumulate route}}
      %acc_vec = weft_rvv.widening_accumulate %product, %vl {accumulate_relation = "signed-i16m4-into-i32m8-deferred-add", kind = "signed_widening_accumulate_add"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m8">
    } : !weft_rvv.vl
  }
}

// -----

// The deferred-wide trailing standalone_reduce over the i32m8 accumulator must
// use kind "add" (one vredsum), not the narrow widening-reduce kind.
module {
  weft.exec.kernel @rvv_deferred_wide_reduce_rejects_widening_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %acc_ptr = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
      %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
      %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i8, "m2">
      %product = weft_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !weft_rvv.vector<i8, "m2">, !weft_rvv.vector<i8, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i16, "m4">
      %acc_vec = weft_rvv.widening_accumulate %product, %vl {accumulate_relation = "signed-i16m4-into-i32m8-deferred-add", kind = "signed_widening_accumulate_add"} : !weft_rvv.vector<i16, "m4">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m8">
      // expected-error@+1 {{requires kind "add" for the deferred-wide trailing standalone reduction}}
      %reduced = weft_rvv.standalone_reduce %acc_vec, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !weft_rvv.vector<i32, "m8">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
    } : !weft_rvv.vl
  }
}
