// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_generic_widening_macc_dataflow_valid
  weft.exec.kernel @rvv_generic_widening_macc_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %rhs_ptr = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %acc_ptr = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i16, "mf2">
      %lhs = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i16, "mf2">
      %rhs = weft_rvv.load %rhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i32, "m1">
      %acc = weft_rvv.load %acc_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // CHECK: weft_rvv.widening_macc
      %sum = weft_rvv.widening_macc %lhs, %rhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // CHECK: weft_rvv.store
      weft_rvv.store %out_ptr, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widening_macc_reject_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i16, "mf2">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i16, "mf2">
    %acc = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{currently supports only kind "signed_widening_macc_add" for the bounded Stage 2 widening multiply-accumulate route}}
      %sum = weft_rvv.widening_macc %lhs, %rhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "unsigned_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widening_macc_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i16, "mf2">
    %acc = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires lhs and rhs source vectors to have type !weft_rvv.vector<i16, "mf2"> for the bounded signed widening multiply-accumulate route}}
      %sum = weft_rvv.widening_macc %lhs, %rhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i16, "mf2">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_pre_realized_widening_macc_reject_accumulator_role {
    weft.exec.variant @rvv_pre_realized_widening_macc_bad_acc attributes {origin = "rvv-plugin", requires = []} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires accumulator operand to bind runtime ABI role 'accumulator-input-buffer'}}
      weft_rvv.typed_widening_macc_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", memory_form = "unit-stride-widening-macc", op_kind = "signed_widening_macc_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-widening-multiply-accumulate-result-to-output-buffer", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
  }
}
