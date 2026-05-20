// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_generic_widening_macc_dataflow_valid
  tcrv.exec.kernel @rvv_generic_widening_macc_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i16, "mf2">
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i16, "mf2">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i32, "m1">
      %acc = tcrv_rvv.load %acc_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.widening_macc
      %sum = tcrv_rvv.widening_macc %lhs, %rhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.store
      tcrv_rvv.store %out_ptr, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widening_macc_reject_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i16, "mf2">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i16, "mf2">
    %acc = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{currently supports only kind "signed_widening_macc_add" for the bounded Stage 2 widening multiply-accumulate route}}
      %sum = tcrv_rvv.widening_macc %lhs, %rhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "unsigned_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widening_macc_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %rhs = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i16, "mf2">
    %acc = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires lhs and rhs source vectors to have type !tcrv_rvv.vector<i16, "mf2"> for the bounded signed widening multiply-accumulate route}}
      %sum = tcrv_rvv.widening_macc %lhs, %rhs, %acc, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_pre_realized_widening_macc_reject_accumulator_role {
    tcrv.exec.variant @rvv_pre_realized_widening_macc_bad_acc attributes {origin = "rvv-plugin", requires = []} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"} : index
      // expected-error@+1 {{requires accumulator operand to bind runtime ABI role 'accumulator-input-buffer'}}
      tcrv_rvv.typed_widening_macc_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", memory_form = "unit-stride-widening-macc", op_kind = "signed_widening_macc_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-widening-multiply-accumulate-result-to-output-buffer", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
  }
}
