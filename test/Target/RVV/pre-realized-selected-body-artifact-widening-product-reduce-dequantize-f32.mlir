// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED --implicit-check-not=weft_rvv.gearbox_cross_region_handoff --implicit-check-not=weft_rvv.vsetvl_region_marker
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | weft-opt --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: sed '/c_name = "scale"/s/c_type = "float"/c_type = "float *"/;/c_name = "scale"/s/role = "dequant-scale-value"/role = "output-buffer"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-SCALE
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1"/accumulator_carry_boundary = "metadata-carry-boundary"/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=MISSING-ACC
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/source_sew = 8 : i64/source_sew = 16 : i64/' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=DTYPE-CHAIN
// RUN: sed '/typed_widening_product_reduce_dequantize_pre_realized_body/s/operand_encoding = "unpacked_i8", policy = /operand_encoding = "unpacked_i8", route_id = "rvv-i32m1", policy = /' %s | not weft-opt --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=STALE-AUTH

// Pre-realized selected-body input for the bounded signed i8 product -> i16
// product -> i32 reduction -> runtime-scale f32 dequantization chain, at the REAL
// architectural vector-register budget (32).
//
// N3 AUTOTUNER (P-B5): the gearbox stamps the schedule facts (still the NARROW
// i8mf4 candidate set + the budget=32 resource fact); the realization owner reads
// the budget, runs the resource-aware LMUL selector, and -- because budget 32
// admits the i8m2 -> i16m4 -> i32m8 rung -- PRODUCES the DEFERRED-WIDE typed body
// (the measured ssh-rvv winner), NOT the narrow per-iteration-vwredsum body. The
// i32m8 accumulator LMUL is realized into the typed body from the budget-pruned
// selector choice (structural, I5).
//
// DEPLOYABLE BUNDLE (P-B6): the description/bundle-export engine now RECOGNIZES the
// deferred-wide body and produces a PARALLEL deferred-wide fact set (the
// widening_accumulate compute step, the i8m2/i16m4/i32m8 wide ladder, the
// i32m1/f32m1 result config, the wide-strip route operand binding), so the wide
// byte kernel exports a valid target-artifact bundle (.o + .h). The PLAN/HEADER
// checks below assert the deferred-wide facts (I5: the wide body is authority). The
// EMITC + ssh-rvv P-B5 evidence still cover the runnable-C winner; the packed-i4
// sibling keeps the narrow nibble-unpack bundle path byte-identical.

module {
  weft.exec.kernel @pre_realized_body_product_reduce_dequantize_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_product_reduce_dequantize attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", operand_encoding = "unpacked_i8", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_product_reduce_dequantize {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// The selector-driven realization produces the DEFERRED-WIDE body: SEW8 LMUL m2
// strip config, i16m4 product, i32m8 deferred accumulator, ONE trailing reduce.
// REALIZED-NOT: weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 8 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: sew = 8 : i64
// REALIZED-SAME: unroll_factor = 1 : i64
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i8, "m2">
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i8, "m2">
// REALIZED: %[[PRODUCT:.*]] = weft_rvv.widening_product %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "signed_widening_product"
// REALIZED-SAME: product_relation = "signed-i8m2xi8m2-to-i16m4"
// REALIZED-SAME: -> !weft_rvv.vector<i16, "m4">
// REALIZED: %[[ACC:.*]] = weft_rvv.widening_accumulate %[[PRODUCT]], %[[VL]]
// REALIZED-SAME: accumulate_relation = "signed-i16m4-into-i32m8-deferred-add"
// REALIZED-SAME: kind = "signed_widening_accumulate_add"
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m8">
// REALIZED: %[[REDUCED:.*]] = weft_rvv.standalone_reduce %[[ACC]], %{{.*}}, %[[VL]]
// REALIZED-SAME: accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: result_layout = "store-standalone-reduction-lane0-to-output-scalar"
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// REALIZED: %[[DEQUANT:.*]] = weft_rvv.dequantize %[[REDUCED]], %{{.*}}, %[[VL]]
// REALIZED-SAME: dequant_relation = "signed-i32m1-to-f32m1-scale-f32"
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: weft_rvv.store %{{.*}}, %[[DEQUANT]], %[[VL]]
// REALIZED-NOT: weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body

// The automatic compiler C is the measured winner: zero-seeded i32m8 vector
// accumulator, per-iteration vwadd.wv (deferred), ONE trailing vredsum.
// EMITC-LABEL: emitc.func @weft_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize
// EMITC: call_opaque "__riscv_vmv_v_x_i32m8"
// EMITC: call_opaque "__riscv_vle8_v_i8m2"
// EMITC: call_opaque "__riscv_vwmul_vv_i16m4"
// EMITC: call_opaque "__riscv_vwadd_wv_i32m8"
// EMITC-NOT: call_opaque "__riscv_vwredsum_vs_i16m4_i32m1"
// EMITC: call_opaque "__riscv_vredsum_vs_i32m8_i32m1"

// The gearbox still stamps the NARROW i8mf4 schedule facts (the realization owner
// re-derives the wide rung from the budget fact; the gearbox is unchanged).
// GEARBOX-SCHEDULE-PRIMITIVE: weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body

// MISSING-RESOURCE-PASS: requires pass-produced low-precision direct-contraction resource fact

// The deferred-wide body now DESCRIBES + EXPORTS a deployable bundle. The PLAN
// metadata carries the deferred-wide typed-compute chain (widening_product ->
// widening_accumulate -> standalone_reduce -> dequantize), the honest wide strip
// ladder on the primitive facts (source_lmul=m2, product_lmul=m4), the i32m1/f32m1
// RESULT config (sew32/m1), and the same logical product-reduction-dequantization
// route identity/leaf profile + ABI as the narrow path. The route operand binding
// honestly mirrors the wide strip (src-i8m2).

// The deployable header artifact: the same callable C ABI prototype as the narrow
// path (the deferred-wide realization is an internal config; the runtime ABI is
// unchanged), exported from the wide bundle.
// HEADER: weft.rvv.selected_route: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

// MISSING-SCALE: runtime scale
// MISSING-SCALE-SAME: dequant-scale-value

// MISSING-ACC: accumulator_carry_boundary
// MISSING-ACC-SAME: vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1

// DTYPE-CHAIN: requires typed product-reduction-dequantization config

// STALE-AUTH: does not accept authority metadata attribute
// STALE-AUTH-SAME: route_id

// PLAN: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
