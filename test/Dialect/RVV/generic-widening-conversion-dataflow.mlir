// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_generic_widen_i32_to_i64_valid
  weft.exec.kernel @rvv_generic_widen_i32_to_i64_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i32, "m1">
      %source = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // CHECK: weft_rvv.widening_convert
      // CHECK-SAME: kind = "widen_i32_to_i64"
      // CHECK-SAME: !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
      %widened = weft_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
      // CHECK: weft_rvv.store
      // CHECK-SAME: !weft_rvv.vector<i64, "m2">
      weft_rvv.store %out_ptr, %widened, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m2">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: weft.exec.kernel @rvv_generic_widen_i16_to_i32_valid
  weft.exec.kernel @rvv_generic_widen_i16_to_i32_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
    %out_ptr = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !weft_rvv.runtime_abi_value
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: weft_rvv.load
      // CHECK-SAME: !weft_rvv.vector<i16, "mf2">
      %source = weft_rvv.load %lhs_ptr, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i16, "mf2">
      // CHECK: weft_rvv.widening_convert
      // CHECK-SAME: kind = "sign_extend_widen_vf2"
      // CHECK-SAME: !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      %widened = weft_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      // CHECK: weft_rvv.store
      // CHECK-SAME: !weft_rvv.vector<i32, "m1">
      weft_rvv.store %out_ptr, %widened, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widen_reject_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{currently supports only kind "widen_i32_to_i64" or "sign_extend_widen_vf2"}}
      %widened = weft_rvv.widening_convert %source, %vl {kind = "zero_extend_i32_to_i64"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widen_i16_to_i32_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires source vector type to be !weft_rvv.vector<i16, "mf2">}}
      %widened = weft_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widen_i16_to_i32_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i16, "mf2">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires result vector type to be !weft_rvv.vector<i32, "m1">}}
      %widened = weft_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widen_i16_to_i32_reject_destination_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i16, "mf2">
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires enclosing weft_rvv.with_vl destination config to be SEW32 LMUL m1}}
      %widened = weft_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widen_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i64, "m2">
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{requires source vector type to be !weft_rvv.vector<i32, "m1">}}
      %widened = weft_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !weft_rvv.vector<i64, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widen_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{requires result vector type to be !weft_rvv.vector<i64, "m2">}}
      %widened = weft_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m1">
    } : !weft_rvv.vl
  }
}

// -----

module {
  weft.exec.kernel @rvv_generic_widen_reject_destination_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !weft_rvv.vector<i32, "m1">
    %vl = weft_rvv.setvl %avl {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
    weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{requires enclosing weft_rvv.with_vl destination config to be SEW64 LMUL m2}}
      %widened = weft_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
    } : !weft_rvv.vl
  }
}
