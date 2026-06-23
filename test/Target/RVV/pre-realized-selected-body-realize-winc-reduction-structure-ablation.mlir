// N3 Win-C — the reduction-STRUCTURE axis (deferred-accumulate vs per-iteration
// reduce), ISOLATED from the LMUL/budget (Win-A) axis, all-compiler-emitted.
//
// The reduction-structure pass option stamps an orthogonal body fact the
// realization owner reads BEFORE the budget rung logic. The fixed-LMUL ablation
// arms hold both bodies at i16mf2 source / i32m1 accumulator, so the ONLY
// difference between the two emitted bodies is the reduction STRUCTURE:
//
//   deferred_accumulate (ON)  : i16mf2 loads, vwmul_vv_i32m1 product, vadd.vv
//                               i32m1 DEFERRED accumulate into a loop-carried
//                               vector, ONE trailing vredsum i32m1->m1. NO
//                               per-iteration reduce.
//   per_iteration (OFF)       : i16mf2 loads, vwmul_vv_i32m1 product, a
//                               vredsum_vs_i32m1 EVERY iteration onto the running
//                               seed. NO deferred vadd, NO trailing standalone
//                               reduce.
//
// Both are m1 (e16mf2 VLMAX == e32m1 VLMAX at any VLEN), so this is a pure
// STRUCTURE flip -- the make-or-break ablation measured 3.0-3.3x on ssh rvv
// (VLEN=128). Absent the fact, the realizer keeps its existing budget-driven
// behavior unchanged (I7 fail-closed); see the autotuner-e2e companion test.
//
// (1) OPTION-MECHANISM coverage: the gearbox pass option stamps the body fact on
// a clean (no pre-stamped budget) body. Budget defaults to 32 here and is
// irrelevant -- this run only proves option -> body fact.
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules='reduction-structure=deferred_accumulate' | FileCheck %s --check-prefix=STAMP-DEFERRED
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules='reduction-structure=per_iteration' | FileCheck %s --check-prefix=STAMP-PERITER
// RUN: not tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules='reduction-structure=bogus' 2>&1 | FileCheck %s --check-prefix=BADOPT
//
// (2) REALIZER-at-fixed-m1 coverage: inject the structure fact directly (mirror
// the existing budget-injection arms, which skip the gearbox because its stale-
// fact check pins budget=32). The deferred arm also injects budget 9 (the
// smallest deferred rung mf2->m1) so its LMUL is m1; per_iteration ignores budget
// (the structure read short-circuits to the per-iter family realizer first).
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 9 : i64, "tcrv_rvv.low_precision_resource.reduction_structure" = "deferred_accumulate", source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=DEFERRED
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 9 : i64, "tcrv_rvv.low_precision_resource.reduction_structure" = "deferred_accumulate", source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=DEFERRED-EMITC
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.reduction_structure" = "per_iteration", source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=PERITER
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.reduction_structure" = "per_iteration", source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=PERITER-EMITC
//
// (3) ORTHOGONALITY: structure OVERRIDES budget. per_iteration at budget 32
// (which would otherwise pick the WIDE deferred m8 rung) still emits per-iter m1
// -- proving the structure axis is independent of (beats) the LMUL/budget axis.
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 32 : i64, "tcrv_rvv.low_precision_resource.reduction_structure" = "per_iteration", source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=PERITER
// (4) deferred at budget 8 (all rungs pruned) falls to the always-available
// minimal mf2->m1 deferred rung -- the structure request is honored at m1.
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 8 : i64, "tcrv_rvv.low_precision_resource.reduction_structure" = "deferred_accumulate", source_lmul = "mf2"/' %s | tcrv-opt --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=DEFERRED
// (5) I7: a directly-injected unrecognized structure fact fails closed at the
// realizer (bounded diagnostic), never silently falling through to budget logic.
// RUN: sed 's/source_lmul = "mf2"/"tcrv_rvv.low_precision_resource.reduction_structure" = "bogus", source_lmul = "mf2"/' %s | not tcrv-opt --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=BADFACT

module {
  tcrv.exec.kernel @dot_reduce_winc_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @dot_reduce_winc_rvv attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "unit-stride-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @dot_reduce_winc_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @dot_reduce_winc_rvv {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-dot-reduce-add-case"}
      tcrv.exec.fallback @dot_reduce_winc_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// STAMP: the gearbox pass option materializes the orthogonal reduction-structure
// body fact on the dot-reduce pre-realized body (the same op the budget fact
// lives on; consumed and erased at realization, never mirrored onto a realized
// op -- I5).
// STAMP-DEFERRED: tcrv_rvv.typed_widening_dot_reduce_pre_realized_body
// STAMP-DEFERRED-SAME: tcrv_rvv.low_precision_resource.reduction_structure = "deferred_accumulate"
// STAMP-PERITER: tcrv_rvv.typed_widening_dot_reduce_pre_realized_body
// STAMP-PERITER-SAME: tcrv_rvv.low_precision_resource.reduction_structure = "per_iteration"

// DEFERRED (ON): the reduction-structure fact forces the deferred-accumulate
// chain at the budget-9 m1 LMUL -- widening_product + deferred_accumulate + ONE
// trailing standalone_reduce. NO per-iteration widening_dot_reduce.
// DEFERRED-NOT: tcrv_rvv.widening_dot_reduce
// DEFERRED: tcrv_rvv.setvl %{{.*}} {lmul = "mf2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64}
// DEFERRED: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i16, "mf2">
// DEFERRED: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i16, "mf2">
// DEFERRED: tcrv_rvv.widening_product
// DEFERRED-SAME: product_relation = "signed-i16mf2xi16mf2-to-i32m1"
// DEFERRED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// DEFERRED: tcrv_rvv.deferred_accumulate
// DEFERRED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// DEFERRED: tcrv_rvv.standalone_reduce
// DEFERRED-SAME: -> !tcrv_rvv.vector<i32, "m1">

// DEFERRED EMITC: zero-seeded i32m1 vector accumulator, per-iteration vadd.vv
// (DEFERRED), ONE trailing vredsum. NO per-iteration vredsum inside the strip.
// DEFERRED-EMITC-LABEL: emitc.func @tcrv_emitc_dot_reduce_winc_kernel_dot_reduce_winc_rvv
// DEFERRED-EMITC: call_opaque "__riscv_vle16_v_i16mf2"
// DEFERRED-EMITC: call_opaque "__riscv_vwmul_vv_i32m1"
// DEFERRED-EMITC: call_opaque "__riscv_vadd_vv_i32m1"
// DEFERRED-EMITC: call_opaque "__riscv_vredsum_vs_i32m1_i32m1"
// DEFERRED-EMITC: call_opaque "__riscv_vse32_v_i32m1"

// PERITER (OFF): the reduction-structure fact forces the per-iteration
// widening_dot_reduce at the SAME m1 LMUL -- a per-strip reduce onto the running
// seed. NO deferred_accumulate, NO standalone_reduce.
// PERITER: tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// PERITER: tcrv_rvv.widening_dot_reduce
// PERITER-NOT: tcrv_rvv.deferred_accumulate
// PERITER-NOT: tcrv_rvv.standalone_reduce

// PERITER EMITC: the per-iteration reduce -- vwmul product then a vredsum EVERY
// iteration onto the running seed read from out[0]. NO vadd deferred accumulate.
// PERITER-EMITC-LABEL: emitc.func @tcrv_emitc_dot_reduce_winc_kernel_dot_reduce_winc_rvv
// PERITER-EMITC: call_opaque "__riscv_vle16_v_i16mf2"
// PERITER-EMITC: call_opaque "__riscv_vwmul_vv_i32m1"
// PERITER-EMITC: call_opaque "__riscv_vredsum_vs_i32m1_i32m1"
// PERITER-EMITC-NOT: call_opaque "__riscv_vadd_vv_i32m1"

// An unrecognized reduction-structure option value is a bounded pass error (no
// silent fall-through to a wrong structure).
// BADOPT: reduction-structure option must be 'deferred_accumulate' or 'per_iteration', got 'bogus'

// A directly-injected unrecognized reduction_structure body fact fails closed at
// the realizer (I7), never silently falling through to budget-driven emission.
// BADFACT: cannot consume unrecognized reduction_structure fact 'bogus'
