// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=WIDE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=WIDE-EMITC
// RUN: sed 's/source_lmul = "mf4"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 16 : i64, source_lmul = "mf4"/' %s | tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=NARROW
// RUN: sed 's/source_lmul = "mf4"/"tcrv_rvv.low_precision_resource.vector_register_budget" = 16 : i64, source_lmul = "mf4"/' %s | tcrv-opt --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries | tcrv-opt --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NARROW-EMITC
//
// PRIZE — the N3 capability/resource DIVERGENCE for the byte low-precision
// product-reduce-dequantize contraction, END-TO-END from ONE kernel.
//
// The SAME pre-realized body (carrying the narrow i8mf4 schedule facts) realizes
// the WIDE deferred i8m2->i16m4->i32m8 rung OR the NARROW i8mf4/i16mf2/i32m1
// per-iteration-vwredsum body purely as a function of the architectural
// vector-register-budget resource fact, which the realization owner reads and
// feeds to the max-legal-LMUL selector (selectRVVLowPrecisionMaxLegalAccumulator
// LMULRung). The budget is a resource/capability INPUT to the Gearbox
// derivation, not a hardcoded constant: the default 32-vreg RVV fact admits the
// i32m8 rung (8+4+8reserve=20<=32) -> wide; a constrained 16-vreg profile prunes
// it (20>16) -> the legacy grouped narrow body. Every budget-derived fact the
// gearbox stamps (legality, the budget attr) flows from the one resolved budget,
// so the realizer consumes a self-consistent fact set at either budget. This
// MIRRORS the strip_elision / dot-reduce autotuner divergence tests: the budget
// GENUINELY drives narrow-vs-wide (the prune binds), proving the wide emission is
// resource-derived, not pinned to the kernel.

module {
  tcrv.exec.kernel @budget_divergence_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @budget_divergence_rvv attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @budget_divergence_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @budget_divergence_rvv {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
      tcrv.exec.fallback @budget_divergence_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// === Default budget (32) admits the wide rung: the deferred i8m2 -> i16m4 ->
// i32m8 body with ONE trailing standalone_reduce. No pre-realized survivor, no
// per-iteration vwredsum, no gearbox_cross_region_handoff. ===
// WIDE-NOT: tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body
// WIDE-NOT: tcrv_rvv.gearbox_cross_region_handoff
// WIDE: tcrv_rvv.setvl %{{.*}} {lmul = "m2", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64}
// WIDE: tcrv_rvv.with_vl
// WIDE-SAME: lmul = "m2"
// WIDE-SAME: sew = 8 : i64
// WIDE-SAME: unroll_factor = 1 : i64
// WIDE: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i8, "m2">
// WIDE: tcrv_rvv.load %{{.*}} -> !tcrv_rvv.vector<i8, "m2">
// WIDE: tcrv_rvv.widening_product
// WIDE-SAME: product_relation = "signed-i8m2xi8m2-to-i16m4"
// WIDE-SAME: -> !tcrv_rvv.vector<i16, "m4">
// WIDE: tcrv_rvv.widening_accumulate
// WIDE-SAME: accumulate_relation = "signed-i16m4-into-i32m8-deferred-add"
// WIDE-SAME: kind = "signed_widening_accumulate_add"
// WIDE-SAME: -> !tcrv_rvv.vector<i32, "m8">
// WIDE: tcrv_rvv.standalone_reduce
// WIDE-SAME: kind = "add"
// WIDE-SAME: -> !tcrv_rvv.vector<i32, "m1">
// WIDE: tcrv_rvv.dequantize
// WIDE-SAME: -> !tcrv_rvv.vector<f32, "m1">
// WIDE: tcrv_rvv.store

// === The wide compiler C: zero-seeded i32m8 vector accumulator, per-iteration
// vwadd.wv (deferred), ONE trailing vredsum -- the measured winner. ===
// WIDE-EMITC-LABEL: emitc.func @tcrv_emitc_budget_divergence_kernel_budget_divergence_rvv
// WIDE-EMITC: call_opaque "__riscv_vmv_v_x_i32m8"
// WIDE-EMITC: call_opaque "__riscv_vle8_v_i8m2"
// WIDE-EMITC: call_opaque "__riscv_vwmul_vv_i16m4"
// WIDE-EMITC: call_opaque "__riscv_vwadd_wv_i32m8"
// WIDE-EMITC-NOT: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// WIDE-EMITC: call_opaque "__riscv_vredsum_vs_i32m8_i32m1"

// === Constrained budget (16 < the 20 crossover) prunes the wide i32m8 rung; the
// realization falls through to the NARROW i8mf4 / i16mf2 grouped per-iteration
// vwredsum body. NO widening_accumulate, NO i32m8 accumulator. The budget
// genuinely drives the choice (N3 -- the prune binds). ===
// NARROW-NOT: tcrv_rvv.widening_accumulate
// NARROW-NOT: tcrv_rvv.vector<i32, "m8">
// NARROW: tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// NARROW: tcrv_rvv.widening_product
// NARROW-SAME: product_relation = "signed-i8mf4xi8mf4-to-i16mf2"
// NARROW-SAME: -> !tcrv_rvv.vector<i16, "mf2">
// NARROW: tcrv_rvv.standalone_reduce
// NARROW: tcrv_rvv.dequantize
// NARROW-SAME: -> !tcrv_rvv.vector<f32, "m1">

// === The narrow compiler C: per-iteration vwmul.vv i16mf2 + vwredsum (the
// legacy under-LMUL grouped path). NO wide i32m8 deferred accumulate. ===
// NARROW-EMITC-LABEL: emitc.func @tcrv_emitc_budget_divergence_kernel_budget_divergence_rvv
// NARROW-EMITC: call_opaque "__riscv_vle8_v_i8mf4"
// NARROW-EMITC: call_opaque "__riscv_vwmul_vv_i16mf2"
// NARROW-EMITC: call_opaque "__riscv_vwredsum_vs_i16mf2_i32m1"
// NARROW-EMITC-NOT: call_opaque "__riscv_vwadd_wv_i32m8"
// NARROW-EMITC-NOT: call_opaque "__riscv_vredsum_vs_i32m8_i32m1"
