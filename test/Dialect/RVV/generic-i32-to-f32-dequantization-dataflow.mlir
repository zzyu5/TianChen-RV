// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_generic_dequantize_i32_to_f32_valid
  tcrv.exec.kernel @rvv_generic_dequantize_i32_to_f32_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i32, "m1">
      %source = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.dequantize
      // CHECK-SAME: dequant_relation = "signed-i32m1-to-f32m1-scale-f32"
      // CHECK-SAME: kind = "i32_to_f32_scaled"
      // CHECK-SAME: !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      %result = tcrv_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
      // CHECK: tcrv_rvv.store
      // CHECK-SAME: !tcrv_rvv.vector<f32, "m1">
      tcrv_rvv.store %out_ptr, %result, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_dequantize_reject_missing_scale_role {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %not_scale = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      %source = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // expected-error@+1 {{requires runtime scale operand to bind runtime ABI role 'dequant-scale-value'}}
      %result = tcrv_rvv.dequantize %source, %not_scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_dequantize_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i64, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires source vector type to be !tcrv_rvv.vector<i32, "m1">}}
      %result = tcrv_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i64, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_dequantize_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires result vector type to be !tcrv_rvv.vector<f32, "m1">}}
      %result = tcrv_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_dequantize_reject_relation {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{currently supports only dequant_relation "signed-i32m1-to-f32m1-scale-f32"}}
      %result = tcrv_rvv.dequantize %source, %scale, %vl {dequant_relation = "i32m1-to-f32m1-with-implicit-scale", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
    } : !tcrv_rvv.vl
  }
}
