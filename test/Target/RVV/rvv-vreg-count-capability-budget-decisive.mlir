// JUDGMENT — the register-pressure feasible set is driven by the in-IR `vreg_count`
// CAPABILITY fact, not a hardcoded 32. The gearbox budget stamp
// (materializeDeferredWideBudgetForDotReduceBody) PULLS the architectural vector-
// register count off the capability provider op (resolveRVVVectorRegisterBudget ->
// readRVVProviderVregCount; the architectural 32 is only the absent-fallback) and
// feeds it to the STEP ② register-pressure inequality
// (enumerateRVVDotReduceDeferredWideLMULRungs / ...MaxLegalLMULRung).
//
// The SAME pre-realized i16mf2 dot-reduce body realizes the WIDE deferred
// i16m4 -> i32m8 rung OR the NARROW i16mf2 -> i32m1 rung PURELY as a function of the
// provider `vreg_count` fact:
//   * vreg_count = 32 (a normal RVV board): the i32m8 rung fits (acc 8 + reserve 8 =
//                      16 <= 32) -> the WIDE accumulator.
//   * vreg_count = 9  (a hypothetical narrow-register profile a capability file
//                      declares): every wider rung is pruned (m2 -> 2+8=10 > 9); only
//                      mf2 -> i32m1 survives (1+8 = 9 <= 9) -> the NARROW accumulator.
// i32m8 (32) != i32m1 (9): the accumulator LMUL moved with the CAPABILITY fact.
//
// This MIRRORS rvv-register-pressure-inequality-decisive.mlir, but the budget is now
// read from the in-IR provider `vreg_count` capability object (the pulled pipe),
// proving the register budget follows the capability fact, not the hardcoded 32
// (core-invariant I1). A deployed 32-register board (default == 32) emits the
// WIDE body byte-for-byte (the fallback reproduces the historical constant).
//
// RUN: weft-opt %s --weft-rvv-materialize-gearbox-schedules=reduction-structure=deferred_accumulate \
// RUN:   --weft-materialize-selected-lowering-boundaries \
// RUN:   | FileCheck %s --check-prefix=WIDE
// RUN: sed 's/vreg_count = 32 : i64/vreg_count = 9 : i64/' %s \
// RUN:   | weft-opt --weft-rvv-materialize-gearbox-schedules=reduction-structure=deferred_accumulate \
// RUN:   --weft-materialize-selected-lowering-boundaries \
// RUN:   | FileCheck %s --check-prefix=NARROW

module {
  weft.exec.kernel @vreg_budget_decisive_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available", vreg_count = 32 : i64}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @vreg_budget_decisive_rvv attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "vreg-count-capability-budget-decisive-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "vreg-count-capability-budget-decisive-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "vreg-count-capability-budget-decisive-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "vreg-count-capability-budget-decisive-widening-dot-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "vreg-count-capability-budget-decisive-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "unit-stride-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @vreg_budget_decisive_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @vreg_budget_decisive_rvv {origin = "rvv-plugin", policy = "vreg-count-capability-budget-decisive-widening-dot-reduce-add-case"}
      weft.exec.fallback @vreg_budget_decisive_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "vreg-count-capability-budget-decisive-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// capability vreg_count = 32: the register-pressure inequality admits the WIDEST
// i16m4 -> i32m8 deferred-accumulate rung.
// WIDE: weft_rvv.widening_product
// WIDE-SAME: product_relation = "signed-i16m4xi16m4-to-i32m8"
// WIDE-SAME: -> !weft_rvv.vector<i32, "m8">
// WIDE: weft_rvv.deferred_accumulate
// WIDE-SAME: -> !weft_rvv.vector<i32, "m8">

// capability vreg_count = 9: the inequality prunes every wider rung; only the
// mf2 -> i32m1 rung survives. NO i32m8 accumulator (the budget prune binds).
// NARROW-NOT: vector<i32, "m8">
// NARROW: weft_rvv.widening_product
// NARROW-SAME: product_relation = "signed-i16mf2xi16mf2-to-i32m1"
// NARROW-SAME: -> !weft_rvv.vector<i32, "m1">
