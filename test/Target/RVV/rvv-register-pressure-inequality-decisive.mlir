// DECISIVE EXPERIMENT — the register-pressure feasibility inequality (width-
// selection STEP ②, the closed form立成显式 f in RVVGearboxSchedule.h:
//   rvvRegisterPressurePeakCost / rvvRegisterPressureLegal /
//   enumerateRVVRegisterPressureLegalCombinations).
//
//   peakCost(combo) = Σ_levels footprint(LMUL_level) · unroll · liveVars_level
//   legal(combo)    ⟺  peakCost ≤ vregBudget − fixedOccupancy
//
// The SAME deferred-accumulate dot-reduce body is realized under two DIFFERENT
// vector_register_budget inputs. The inequality's legal (unroll × chain) SET — and
// hence the WIDEST-legal accumulator LMUL the realizer picks — must change with the
// budget input. If the accumulator LMUL did NOT change, the "formula" would be a
// 摆设 (dead knob); the flip proves the inequality is load-bearing.
//
//   budget 32 : the i16m4 source rung fits (acc i32m8 = 8 vregs + 8 reserve = 16 ≤
//               32) → WIDEST legal accumulator = i32m8.
//   budget  9 : every wider rung is pruned (m1->m2 = 2+8=10 > 9); only the mf2->m1
//               rung fits (1+8 = 9 ≤ 9) → WIDEST legal accumulator = i32m1.
// i32m8 (budget 32) ≠ i32m1 (budget 9) == GREEN: the legal set moved with the input.

// RUN: sed 's/source_lmul = "mf2"/"weft_rvv.low_precision_resource.vector_register_budget" = 32 : i64, "weft_rvv.low_precision_resource.reduction_structure" = "deferred_accumulate", source_lmul = "mf2"/' %s \
// RUN:   | weft-opt --weft-materialize-selected-lowering-boundaries \
// RUN:   | FileCheck %s --check-prefix=WIDE
// RUN: sed 's/source_lmul = "mf2"/"weft_rvv.low_precision_resource.vector_register_budget" = 9 : i64, "weft_rvv.low_precision_resource.reduction_structure" = "deferred_accumulate", source_lmul = "mf2"/' %s \
// RUN:   | weft-opt --weft-materialize-selected-lowering-boundaries \
// RUN:   | FileCheck %s --check-prefix=NARROW

module {
  weft.exec.kernel @dot_reduce_regpressure_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @dot_reduce_regpressure_rvv attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "register-pressure-decisive-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "register-pressure-decisive-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "register-pressure-decisive-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "register-pressure-decisive-widening-dot-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "register-pressure-decisive-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "unit-stride-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @dot_reduce_regpressure_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @dot_reduce_regpressure_rvv {origin = "rvv-plugin", policy = "register-pressure-decisive-widening-dot-reduce-add-case"}
      weft.exec.fallback @dot_reduce_regpressure_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "register-pressure-decisive-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// budget 32: the register-pressure inequality admits the WIDEST i16m4 -> i32m8 rung.
// WIDE: i32m8
// budget 9: the inequality prunes every wider rung; only mf2 -> i32m1 survives.
// NARROW: i32m1
// NARROW-NOT: i32m8
