// RUN: tcrv-opt %s --split-input-file --verify-diagnostics
//
// The Track B byte-anchor widening int8 dot-reduce typed body (the bar-A
// auto-lowered first block, here exercised DIRECTLY at the dialect layer to lock
// the additive verifier relaxations fail-closed). The integer-core anchor is the
// capability-selected LMUL: SEW8/m2 at VLEN128 (e8m2) and SEW8/m1 at VLEN256
// (e8m1). The chain is: i8<anchor> strip-load x2 -> i16<wider> signed widening
// product -> i32m1 widening standalone_reduce -> i32 lane0 store. These are the
// SAME structural ops the front-door materializer auto-constructs; this file
// pins that the relaxed Load/WideningProduct/StandaloneReduce/SetVL/WithVL
// verifier branches admit ONLY the byte-anchor window and reject every edge
// (parallel to generic-deferred-wide-dot-reduce-dataflow.mlir).

// ===== VALID: the e8m1 (VLEN256) byte anchor -- i8m1 -> i16m2 -> i32m1. =====
module {
  tcrv.exec.kernel @rvv_byte_anchor_dot_reduce_m1_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8m1xi8m1-to-i16m2"} : !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
      %reduced = tcrv_rvv.standalone_reduce %product, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      tcrv_rvv.store %out_ptr, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

// ===== VALID: the e8m2 (VLEN128) byte anchor -- i8m2 -> i16m4 -> i32m1. =====
// The m2 anchor reuses the deferred-wide "signed-i8m2xi8m2-to-i16m4" product
// relation; only the consumer (standalone_reduce, gated by the byte-anchor
// scope) differs. The SAME structure as the m1 body, one EMUL rung wider.
module {
  tcrv.exec.kernel @rvv_byte_anchor_dot_reduce_m2_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8m2xi8m2-to-i16m4"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
      %reduced = tcrv_rvv.standalone_reduce %product, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      tcrv_rvv.store %out_ptr, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

// ===== REJECT: an m4 byte anchor is NOT a legal integer-core anchor =====
// (only m1/m2 are admitted by the byte-anchor strip config). fail-closed (I7).
module {
  tcrv.exec.kernel @rvv_byte_anchor_dot_reduce_rejects_m4_anchor {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    // expected-error@+1 {{requires bounded RVV first-slice compile-time config to be SEW32 with LMUL "m1" or "m2", or SEW64 with LMUL "m1" or "m2", or a deferred-wide strip config (SEW8 LMUL "m2", or SEW16 LMUL "mf2"/"m1"/"m2"/"m4" for the budget-selected dot-reduce rung), or the byte-anchor dot-reduce strip config (SEW8 LMUL "m1"/"m2")}}
    %vl = tcrv_rvv.setvl %avl {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
  }
}

// -----

// ===== REJECT: under the m1 anchor the widening product result must be i16m2 =====
// (an i16m4 product widens the WRONG rung -- only the byte anchor's own one-step
// wider product is admitted). fail-closed.
module {
  tcrv.exec.kernel @rvv_byte_anchor_dot_reduce_rejects_wrong_product_lmul {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
      // expected-error@+1 {{requires result vector to have type !tcrv_rvv.vector<i16, "m2"> for the byte-anchor m1 widening-product dot-reduce rung}}
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8m1xi8m1-to-i16m2"} : !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
    } : !tcrv_rvv.vl
  }
}

// -----

// ===== REJECT: the byte-anchor widening reduce result must be i32m1 =====
// (an i16m1 result is rejected -- the widening reduction TARGET is the i32m1
// scalar accumulator channel, never the strip width). This is the branch that
// the relaxed SEW-agreement helper relies on to pin the result type, so the
// SEW8/SEW16-scope relaxation cannot admit a non-i32 result. fail-closed.
module {
  tcrv.exec.kernel @rvv_byte_anchor_dot_reduce_rejects_wrong_result_type {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i8m1xi8m1-to-i16m2"} : !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
      // expected-error@+1 {{requires the byte-anchor signed widening standalone reduction result vector to have type !tcrv_rvv.vector<i32, "m1">}}
      %reduced = tcrv_rvv.standalone_reduce %product, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m1">
    } : !tcrv_rvv.vl
  }
}
