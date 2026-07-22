// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=WIDE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
//
// P-B5 — the N3 autotuner finale, END-TO-END from a kernel (selector-driven).
//
// The pre-realized body carries the NARROW i8mf4 schedule facts + the
// architectural vector-register-budget resource fact (32). The realization owner
// reads that budget, runs the P-B2 resource-aware selector
// (selectRVVLowPrecisionMaxLegalAccumulatorLMULRung), which at budget 32 returns
// the i8m2 -> i16m4 -> i32m8 rung, and PRODUCES the deferred-wide typed body --
// the measured ssh-rvv winner (P-B4: 4.1-10.8x vs scalar, 3.3-5.4x vs naive).
// The accumulator LMUL (i32m8) is realized INTO the typed body's vector types
// from the budget-pruned selector choice, so the tune decision is STRUCTURAL
// (I5), not a constant. The byte source (operand_form = "unpacked-byte-elements")
// is what gates the wide branch; packed-i4 keeps the narrow path.
//
// The lowered C is byte-identical to the hand-validated wide-lmul winner.
// The wide body now DESCRIBES + EXPORTS through emission-plans: the description
// engine recognizes the deferred-wide chain and produces the parallel wide fact
// set (P-B6), so the selector-driven wide kernel reaches a deployable bundle.

module {
  weft.exec.kernel @autotuner_e2e_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @autotuner_e2e_rvv attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", operand_encoding = "unpacked_i8", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @autotuner_e2e_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @autotuner_e2e_rvv {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
      weft.exec.fallback @autotuner_e2e_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// The selector-driven realization produces the DEFERRED-WIDE typed body: the
// strip config is SEW8 LMUL m2, the product widens to i16m4, and the deferred
// accumulator is i32m8 (the budget-pruned wide rung), folded with ONE trailing
// standalone_reduce. There is NO narrow gearbox_cross_region_handoff and NO
// per-iteration vwredsum.
// WIDE-NOT: weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body
// WIDE-NOT: weft_rvv.gearbox_cross_region_handoff
// WIDE: weft_rvv.setvl %{{.*}} {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64}
// WIDE: weft_rvv.with_vl
// WIDE-SAME: lmul = "m2"
// WIDE-SAME: sew = 8 : i64
// WIDE-SAME: unroll_factor = 1 : i64
// WIDE: weft_rvv.load %{{.*}} -> !weft_rvv.vector<i8, "m2">
// WIDE: weft_rvv.load %{{.*}} -> !weft_rvv.vector<i8, "m2">
// WIDE: weft_rvv.widening_product
// WIDE-SAME: product_relation = "signed-i8m2xi8m2-to-i16m4"
// WIDE-SAME: -> !weft_rvv.vector<i16, "m4">
// WIDE: weft_rvv.widening_accumulate
// WIDE-SAME: accumulate_relation = "signed-i16m4-into-i32m8-deferred-add"
// WIDE-SAME: kind = "signed_widening_accumulate_add"
// WIDE-SAME: -> !weft_rvv.vector<i32, "m8">
// WIDE: weft_rvv.standalone_reduce
// WIDE-SAME: kind = "add"
// WIDE-SAME: -> !weft_rvv.vector<i32, "m1">
// WIDE: weft_rvv.dequantize
// WIDE-SAME: -> !weft_rvv.vector<f32, "m1">
// WIDE: weft_rvv.store

// The automatic compiler C: zero-seeded i32m8 vector accumulator, per-iteration
// vwadd.wv (deferred), ONE trailing vredsum -- the measured winner. NO
// per-iteration vwredsum.
// EMITC-LABEL: emitc.func @weft_emitc_autotuner_e2e_kernel_autotuner_e2e_rvv
// EMITC: call_opaque "__riscv_vmv_v_x_i32m8"
// EMITC: call_opaque "__riscv_vle8_v_i8m2"
// EMITC: call_opaque "__riscv_vwmul_vv_i16m4"
// EMITC: call_opaque "__riscv_vwadd_wv_i32m8"
// EMITC-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMITC: call_opaque "__riscv_vredsum_vs_i32m8_i32m1"

// The wide body describes + exports through emission-plans (P-B6): the PLAN
// metadata carries the deferred-wide typed-compute chain and the honest wide strip
// ladder on the primitive facts.
// PLAN-DAG: "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_product+weft_rvv.widening_accumulate+weft_rvv.standalone_reduce+weft_rvv.dequantize"
// PLAN-DAG: "weft_rvv.low_precision_primitive.source_lmul", value = "m2"
// PLAN-DAG: "weft_rvv.low_precision_primitive.product_lmul", value = "m4"
// PLAN-DAG: "weft_rvv.target_leaf_profile", value = "rvv-v1-i8mf4-i16mf2-i32m1-f32m1-product-reduction-dequantization-leaf-profile.v1"
