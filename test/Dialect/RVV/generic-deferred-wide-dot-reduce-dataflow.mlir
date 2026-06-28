// RUN: tcrv-opt %s --split-input-file --verify-diagnostics

// The 2nd-family (i16 widening dot-reduce) N3 resource-aware max-legal-LMUL
// deferred-wide typed body (the measured ssh-rvv winner dot_wide_deferred
// modeled as first-class structure): i16m4 strip-load -> i32m8 SINGLE-step
// widening product -> a DEFERRED i32m8 NON-widening vector accumulate
// (tcrv_rvv.deferred_accumulate, vadd.vv) -> ONE trailing
// tcrv_rvv.standalone_reduce (i32m8 -> i32m1, kind "add") -> scalar acc[0] add
// -> i32 lane0 store. The deferred_accumulate op is the structural marker (I5):
// emission follows op identity. The enclosing with_vl is the strip config
// SEW16/m4.
module {
  tcrv.exec.kernel @rvv_deferred_wide_dot_reduce_dataflow_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i16m4xi16m4-to-i32m8"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
      %acc_vec = tcrv_rvv.deferred_accumulate %product, %vl {accumulate_relation = "signed-i32m8-into-i32m8-deferred-add", kind = "signed_deferred_accumulate_add"} : !tcrv_rvv.vector<i32, "m8">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
      %reduced = tcrv_rvv.standalone_reduce %acc_vec, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i32, "m8">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      tcrv_rvv.store %out_ptr, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

// The deferred-wide dot-reduce accumulate's operand type must MATCH its
// accumulate_relation: an i16mf2 operand under the "...i32m8..." relation is
// rejected (the relation-derived expected operand type is i32m8).
module {
  tcrv.exec.kernel @rvv_deferred_wide_dot_reduce_rejects_narrow_product {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %product = "builtin.unrealized_conversion_cast"() : () -> !tcrv_rvv.vector<i16, "mf2">
    %vl = tcrv_rvv.setvl %avl {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      // expected-error@+1 {{requires product operand to have type !tcrv_rvv.vector<i32, "m8"> matching the deferred-wide dot-reduce accumulate_relation}}
      %acc_vec = tcrv_rvv.deferred_accumulate %product, %vl {accumulate_relation = "signed-i32m8-into-i32m8-deferred-add", kind = "signed_deferred_accumulate_add"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
    } : !tcrv_rvv.vl
  }
}

// -----

// The all-compiler LMUL-width ablation: the SAME deferred-accumulate structure
// is ALSO well-formed at a NARROWER budget-selected rung -- i16m2 strip-load ->
// i32m4 SINGLE-step widening product -> deferred i32m4 vadd.vv -> ONE trailing
// standalone_reduce (i32m4 -> i32m1). The verifiers derive every expected LMUL
// from the relation strings (parametric), so this narrow rung passes with the
// SAME structure as the wide m4/m8 body above, only the LMUL width differs.
module {
  tcrv.exec.kernel @rvv_deferred_wide_dot_reduce_narrow_rung_valid {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %out_ptr = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i16m2xi16m2-to-i32m4"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m4">
      %acc_vec = tcrv_rvv.deferred_accumulate %product, %vl {accumulate_relation = "signed-i32m4-into-i32m4-deferred-add", kind = "signed_deferred_accumulate_add"} : !tcrv_rvv.vector<i32, "m4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m4">
      %reduced = tcrv_rvv.standalone_reduce %acc_vec, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i32, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      tcrv_rvv.store %out_ptr, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
    } : !tcrv_rvv.vl
  }
}

// -----

// A MISMATCHED rung is still rejected (fail-closed, I7): an i16m2 product
// (i32m4 result) under an "...i32m8..." accumulate_relation does not match -- the
// product/accumulate LMULs must agree with the relation.
module {
  tcrv.exec.kernel @rvv_deferred_wide_dot_reduce_rejects_mismatched_rung {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i16m2xi16m2-to-i32m4"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m4">
      // expected-error@+1 {{requires product operand to have type !tcrv_rvv.vector<i32, "m8"> matching the deferred-wide dot-reduce accumulate_relation}}
      %acc_vec = tcrv_rvv.deferred_accumulate %product, %vl {accumulate_relation = "signed-i32m8-into-i32m8-deferred-add", kind = "signed_deferred_accumulate_add"} : !tcrv_rvv.vector<i32, "m4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
    } : !tcrv_rvv.vl
  }
}

// -----

// The deferred-wide dot-reduce trailing standalone_reduce over the i32m8
// accumulator must use kind "add" (one vredsum), not a widening-reduce kind.
module {
  tcrv.exec.kernel @rvv_deferred_wide_dot_reduce_rejects_widening_kind {
    %avl = "builtin.unrealized_conversion_cast"() : () -> index
    %lhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %rhs_ptr = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %acc_ptr = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
    %vl = tcrv_rvv.setvl %avl {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} : index -> !tcrv_rvv.vl
    tcrv_rvv.with_vl %vl attributes {lmul = "m4", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64} {
      %lhs = tcrv_rvv.load %lhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
      %rhs = tcrv_rvv.load %rhs_ptr, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
      %product = tcrv_rvv.widening_product %lhs, %rhs, %vl {kind = "signed_widening_product", product_relation = "signed-i16m4xi16m4-to-i32m8"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
      %acc_vec = tcrv_rvv.deferred_accumulate %product, %vl {accumulate_relation = "signed-i32m8-into-i32m8-deferred-add", kind = "signed_deferred_accumulate_add"} : !tcrv_rvv.vector<i32, "m8">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m8">
      // expected-error@+1 {{requires kind "add" for the deferred-wide dot-reduce trailing standalone reduction}}
      %reduced = tcrv_rvv.standalone_reduce %acc_vec, %acc_ptr, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "signed_widening_reduce_add", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.vector<i32, "m8">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
    } : !tcrv_rvv.vl
  }
}
