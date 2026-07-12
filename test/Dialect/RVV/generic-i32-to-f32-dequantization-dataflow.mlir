// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_generic_dequantize_i32_to_f32_valid
  weft.exec.kernel @rvv_generic_dequantize_i32_to_f32_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i32, "m1">
      %source = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // CHECK: weft_rvv.dequantize
      // CHECK-SAME: dequant_relation = "signed-i32m1-to-f32m1-scale-f32"
      // CHECK-SAME: kind = "i32_to_f32_scaled"
      // CHECK-SAME: !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
      // CHECK: weft_rvv.store
      // CHECK-SAME: !weft_rvv.vector<f32, "m1">
      weft_rvv.store %out_ptr, %result, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_dequantize_reject_missing_scale_role {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %not_scale = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      %source = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // expected-error@+1 {{requires runtime scale operand to bind runtime ABI role 'dequant-scale-value'}}
      %result = weft_rvv.dequantize %source, %not_scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_dequantize_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i64, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires source vector type to be !weft_rvv.vector<i32, "m1">}}
      %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i64, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_dequantize_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires result vector type to be !weft_rvv.vector<f32, "m1">}}
      %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_dequantize_reject_relation {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{currently supports only dequant_relation "signed-i32m1-to-f32m1-scale-f32"}}
      %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "i32m1-to-f32m1-with-implicit-scale", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
    } : !weft_rvv.vl
  }
}
