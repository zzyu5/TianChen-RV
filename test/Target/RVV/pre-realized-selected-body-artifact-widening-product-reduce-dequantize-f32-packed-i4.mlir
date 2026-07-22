// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED --implicit-check-not=weft_rvv.gearbox_cross_region_handoff --implicit-check-not=weft_rvv.vsetvl_region_marker
// Stage 3 single-scope flip: the resource decision now lives on the lone with_vl
// two-scope handoff op previously carried a bare resource_decision). Re-targeted to
// the with_vl occurrence; still fail-closed at emission-plan time with the specific
// "requires packed-i4 realization decision" provider-fact-gate reason.
// Stage 3 single-scope flip: the region count now lives on with_vl as
// markers previously carried region_count). Re-targeted; still fail-closed at
// emission-plan time with the specific "realized vsetvl region count" reason.
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-rvv-emitc-to-cpp | FileCheck %s --check-prefix=CPP

// Focused measurement-disposition fixture for the same product-reduction-dequant op kind with
// an explicit signed packed-i4 selected resource. The candidate is authority
// only because it is carried in the typed pre-realized weft_rvv body and then
// consumed by RVV selected-body realization/provider planning.

module {
  weft.exec.kernel @pre_realized_body_product_reduce_dequantize_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_product_reduce_dequantize attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-packed-i4:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_dequantize_pre_realized_body %lhs, %rhs, %acc, %scale, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-dequantized-f32-vector-to-output-buffer", memory_form = "unit-stride-widening-product-reduce-dequantize-f32", op_kind = "widening_product_reduce_dequantize_f32", operand_encoding = "packed_i4", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", source_lmul = "mf4", source_sew = 8 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_product_reduce_dequantize {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequantize-f32-fallback-envelope"}
    }
  }
}

// Stage 3 single-scope packed-i4 flip: the realized body is one weft_rvv.with_vl
// scope carrying a typed weft_rvv.packed_i4_nibble_unpack_product head + the inline
// dequant chain, with NO weft_rvv.gearbox_cross_region_handoff carrier and NO
// weft_rvv.vsetvl_region_marker placeholders. The structural unroll_factor (=1) the
// conversion reads is stamped on with_vl; the low_precision_resource.* facts above
// survive on the single scope. Numerics HW-validated on ssh rvv (tolerance=1e-05).
// REALIZED-DAG: unroll_factor = 1 : i64
// REALIZED-DAG: weft_rvv.packed_i4_nibble_unpack_product
// REALIZED-DAG: kind = "signed_packed_i4_nibble_unpack_product"
// REALIZED-DAG: weft_rvv.standalone_reduce
// REALIZED-DAG: weft_rvv.dequantize
// REALIZED-DAG: weft_rvv.store
// The deleted two-scope carrier/markers are genuinely absent across the WHOLE
// realized body (the REALIZED RUN line's --implicit-check-not enforces this
// globally, fail-closed: a stray handoff or marker anywhere = incomplete flip).


// HEADER: void weft_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float *out, size_t n);

// CPP: __riscv_vle8_v_i8mf4
// CPP: __riscv_vle8_v_i8mf4
// CPP: __riscv_vsll_vx_i8mf4
// CPP: __riscv_vsll_vx_i8mf4
// CPP: __riscv_vwmul_vv_i16mf2
// CPP: __riscv_vsra_vx_i16mf2
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vsra_vx_i8mf4
// CPP: __riscv_vwmacc_vv_i16mf2
// CPP: __riscv_vwredsum_vs_i16mf2_i32m1
// CPP: weft_emitc.assign target=dot_acc_vec
// CPP: __riscv_vmv_x_s_i32m1_i32
// Stage 3 single-scope flip: the conversion emits the UNIFIED vector dequant
// epilogue (lane-0 extract -> scalar dequant -> VL=1 f32 splat-store), the same as
// the grouped/unpacked candidate -- numerically a single-scalar write to out[0]
// (HW-validated on ssh rvv, tolerance=1e-05). The legacy packed-i4 scalar out[0]=
// store is retired (it was the other code path's form, numerically equivalent).
// CPP: __riscv_vfmv_v_f_f32m1
// CPP: __riscv_vse32_v_f32m1

// PLAN: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"

