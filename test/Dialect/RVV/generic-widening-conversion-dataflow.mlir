// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_generic_widen_i32_to_i64_valid
  tcrv.exec.kernel @rvv_generic_widen_i32_to_i64_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i32, "m1">
      %source = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.widening_convert
      // CHECK-SAME: kind = "widen_i32_to_i64"
      // CHECK-SAME: !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
      // CHECK: tcrv_rvv.store
      // CHECK-SAME: !tcrv_rvv.vector<i64, "m2">
      tcrv_rvv.store %out_ptr, %widened, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i64, "m2">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @rvv_generic_widen_i16_to_i32_valid
  tcrv.exec.kernel @rvv_generic_widen_i16_to_i32_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // CHECK: tcrv_rvv.load
      // CHECK-SAME: !tcrv_rvv.vector<i16, "mf2">
      %source = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
      // CHECK: tcrv_rvv.widening_convert
      // CHECK-SAME: kind = "sign_extend_widen_vf2"
      // CHECK-SAME: !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      // CHECK: tcrv_rvv.store
      // CHECK-SAME: !tcrv_rvv.vector<i32, "m1">
      tcrv_rvv.store %out_ptr, %widened, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widen_reject_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{currently supports only kind "widen_i32_to_i64" or "sign_extend_widen_vf2"}}
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "zero_extend_i32_to_i64"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widen_i16_to_i32_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires source vector type to be !tcrv_rvv.vector<i16, "mf2">}}
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widen_i16_to_i32_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i16, "mf2">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires result vector type to be !tcrv_rvv.vector<i32, "m1">}}
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widen_i16_to_i32_reject_destination_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i16, "mf2">
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
      // expected-error@+1 {{requires enclosing tcrv_rvv.with_vl destination config to be SEW32 LMUL m1}}
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "sign_extend_widen_vf2"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widen_reject_source_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i64, "m2">
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{requires source vector type to be !tcrv_rvv.vector<i32, "m1">}}
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !tcrv_rvv.vector<i64, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widen_reject_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{requires result vector type to be !tcrv_rvv.vector<i64, "m2">}}
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m1">
    } : !tcrv_rvv.vl
  }
}

// -----

module {
  tcrv.exec.kernel @rvv_generic_widen_reject_destination_config {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %source = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i32, "m1">
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} {
      // expected-error@+1 {{requires enclosing tcrv_rvv.with_vl destination config to be SEW64 LMUL m2}}
      %widened = tcrv_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i64, "m2">
    } : !tcrv_rvv.vl
  }
}
