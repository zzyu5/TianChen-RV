// Formula-constructed dot-reduce schedule: reduction structure is a bounded
// typed direct-authoring constraint, while vreg_count remains capability c.
// The fixed-LMUL arms hold both bodies at i16mf2 source / i32m1 accumulator, so
// the only emitted difference is reduction structure:
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
// Both are m1, so this is a pure structure flip. Absent the optional constraint,
// the analytic prior uses the budget-legal deferred schedule; an invalid typed
// constraint fails closed at dialect verification.
// RUN: sed -e '/capability @rvv/s/status = "available"}/status = "available", vreg_count = 9 : i64}/' -e 's/source_lmul = "mf2"/reduction_structure = "deferred_accumulate", source_lmul = "mf2"/' %s | weft-opt --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=DEFERRED
// RUN: sed -e '/capability @rvv/s/status = "available"}/status = "available", vreg_count = 9 : i64}/' -e 's/source_lmul = "mf2"/reduction_structure = "deferred_accumulate", source_lmul = "mf2"/' %s | weft-opt --weft-materialize-selected-lowering-boundaries | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=DEFERRED-EMITC
// RUN: sed 's/source_lmul = "mf2"/reduction_structure = "per_iteration", source_lmul = "mf2"/' %s | weft-opt --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=PERITER
// RUN: sed 's/source_lmul = "mf2"/reduction_structure = "per_iteration", source_lmul = "mf2"/' %s | weft-opt --weft-materialize-selected-lowering-boundaries | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=PERITER-EMITC
// RUN: sed 's/source_lmul = "mf2"/reduction_structure = "bogus", source_lmul = "mf2"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=BADFACT

module {
  weft.exec.kernel @dot_reduce_winc_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @dot_reduce_winc_rvv attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-dot-reduce-add:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_dot_reduce_pre_realized_body %lhs, %rhs, %acc, %out, %n {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dot_product_relation = "signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32", memory_form = "unit-stride-widening-dot-reduce", op_kind = "signed_widening_dot_reduce_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-dot-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @dot_reduce_winc_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @dot_reduce_winc_rvv {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-dot-reduce-add-case"}
      weft.exec.fallback @dot_reduce_winc_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-dot-reduce-add-fallback-envelope"}
    }
  }
}

// DEFERRED (ON): the reduction-structure fact forces the deferred-accumulate
// chain at the budget-9 m1 LMUL -- widening_product + deferred_accumulate + ONE
// trailing standalone_reduce. NO per-iteration widening_dot_reduce.
// DEFERRED-NOT: weft_rvv.widening_dot_reduce
// DEFERRED: weft_rvv.setvl %{{.*}} {lmul = "mf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 16 : i64}
// DEFERRED: weft_rvv.load %{{.*}} -> !weft_rvv.vector<i16, "mf2">
// DEFERRED: weft_rvv.load %{{.*}} -> !weft_rvv.vector<i16, "mf2">
// DEFERRED: weft_rvv.widening_product
// DEFERRED-SAME: product_relation = "signed-i16mf2xi16mf2-to-i32m1"
// DEFERRED-SAME: -> !weft_rvv.vector<i32, "m1">
// DEFERRED: weft_rvv.deferred_accumulate
// DEFERRED-SAME: -> !weft_rvv.vector<i32, "m1">
// DEFERRED: weft_rvv.standalone_reduce
// DEFERRED-SAME: -> !weft_rvv.vector<i32, "m1">

// DEFERRED EMITC: zero-seeded i32m1 vector accumulator, per-iteration vadd.vv
// (DEFERRED), ONE trailing vredsum. NO per-iteration vredsum inside the strip.
// DEFERRED-EMITC-LABEL: emitc.func @weft_emitc_dot_reduce_winc_kernel_dot_reduce_winc_rvv
// DEFERRED-EMITC: call_opaque "__riscv_vle16_v_i16mf2"
// DEFERRED-EMITC: call_opaque "__riscv_vwmul_vv_i32m1"
// DEFERRED-EMITC: call_opaque "__riscv_vadd_vv_i32m1"
// DEFERRED-EMITC: call_opaque "__riscv_vredsum_vs_i32m1_i32m1"
// DEFERRED-EMITC: call_opaque "__riscv_vse32_v_i32m1"

// PERITER (OFF): the reduction-structure fact forces the per-iteration
// widening_dot_reduce at the SAME m1 LMUL -- a per-strip reduce onto the running
// seed. NO deferred_accumulate, NO standalone_reduce.
// PERITER: weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// PERITER: weft_rvv.widening_dot_reduce
// PERITER-NOT: weft_rvv.deferred_accumulate
// PERITER-NOT: weft_rvv.standalone_reduce

// PERITER EMITC: the per-iteration reduce -- vwmul product then a vredsum EVERY
// iteration onto the running seed read from out[0]. NO vadd deferred accumulate.
// PERITER-EMITC-LABEL: emitc.func @weft_emitc_dot_reduce_winc_kernel_dot_reduce_winc_rvv
// PERITER-EMITC: call_opaque "__riscv_vle16_v_i16mf2"
// PERITER-EMITC: call_opaque "__riscv_vwmul_vv_i32m1"
// PERITER-EMITC: call_opaque "__riscv_vredsum_vs_i32m1_i32m1"
// PERITER-EMITC-NOT: call_opaque "__riscv_vadd_vv_i32m1"

// BADFACT: requires optional reduction_structure to be "per_iteration" or "deferred_accumulate"
