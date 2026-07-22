// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED --implicit-check-not=weft_rvv.gearbox_cross_region_handoff --implicit-check-not=weft_rvv.vsetvl_region_marker
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-rvv-emitc-to-cpp | FileCheck %s --check-prefix=CPP

// Focused dequant-clamp realized-body consumption representative. The selected candidate is
// authority only because it is present on the typed RVV body and consumed by
// the RVV provider/resource planner; artifact names and measurement metadata
// are mirrors that must match the provider facts.

module {
  weft.exec.kernel @pre_realized_body_product_reduce_dequant_clamp_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_product_reduce_dequant_clamp attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int8_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:acc", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body %lhs, %rhs, %acc, %scale, %lower, %upper, %out, %n {accumulator_carry_boundary = "vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1", accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", accumulator_lmul = "m1", accumulator_role = "accumulator-input-buffer", accumulator_sew = 32 : i64, bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", dequant_store_boundary = "store-clamped-dequantized-f32-vector-to-output-buffer", lower_predicate_kind = "slt", memory_form = "unit-stride-widening-product-reduce-dequant-clamp-f32", op_kind = "widening_product_reduce_dequant_clamp_f32", operand_encoding = "packed_i4", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, product_lmul = "mf2", product_reduction_chain_relation = "signed-i8mf4xi8mf4-to-i16mf2-reduce-plus-i32-scalar-to-i32", product_relation = "signed-i8mf4xi8mf4-to-i16mf2", product_sew = 16 : i64, result_layout = "store-standalone-reduction-lane0-to-output-scalar", result_lmul = "m1", result_sew = 32 : i64, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", source_lmul = "mf4", source_sew = 8 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_product_reduce_dequant_clamp {origin = "rvv-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-fallback-envelope"}
    }
  }
}

// Stage 3 single-scope packed-i4 flip: one with_vl scope with a typed
// weft_rvv.packed_i4_nibble_unpack_product head + the inline dequant/clamp chain;
// NO weft_rvv.gearbox_cross_region_handoff carrier, NO weft_rvv.vsetvl_region_marker
// (the REALIZED RUN line's --implicit-check-not enforces their global absence,
// fail-closed). The structural unroll_factor (=1) is on with_vl; the
// low_precision_resource.* facts above survive. Numerics HW-validated (ssh rvv, 1e-05).
// REALIZED-DAG: unroll_factor = 1 : i64
// REALIZED-DAG: weft_rvv.packed_i4_nibble_unpack_product
// REALIZED-DAG: kind = "signed_packed_i4_nibble_unpack_product"
// REALIZED: weft_rvv.dequantize
// REALIZED: weft_rvv.compare
// REALIZED: weft_rvv.select

// PLAN: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs,acc,scale,lower_bound,upper_bound,out,n"}
// PLAN-DAG: {key = "weft_rvv.selected_dispatch_case_mirror", value = "selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequant_clamp;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-case"}
// PLAN-DAG: {key = "weft_rvv.selected_dispatch_fallback_mirror", value = "selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-fallback-envelope"}
// PLAN-SAME: status = "supported"

// HEADER: weft.rvv.runtime_abi_order: lhs,rhs,acc,scale,lower_bound,upper_bound,out,n
// HEADER-DAG: weft.rvv.selected_dispatch_case_mirror: selected_dispatch_case_mirror:@pre_realized_body_rvv_product_reduce_dequant_clamp;role=dispatch case;runtime_guard_required=false;runtime_guard=none;origin=rvv-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-case
// HEADER-DAG: weft.rvv.selected_dispatch_fallback_mirror: selected_dispatch_fallback_mirror:@pre_realized_body_scalar_fallback;role=dispatch fallback;fallback_role=conservative;origin=scalar-plugin;policy=pre-realized-selected-body-widening-product-reduce-dequant-clamp-f32-packed-i4-fallback-envelope
// HEADER: void weft_emitc_pre_realized_body_product_reduce_dequant_clamp_kernel_pre_realized_body_rvv_product_reduce_dequant_clamp(const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float lower_bound, float upper_bound, float *out, size_t n);

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
// CPP: __riscv_vmv_x_s_i32m1_i32
// Stage 3 single-scope flip: the conversion emits the UNIFIED vector dequant-clamp
// epilogue (lane-0 extract -> scalar dequant -> 3 VL=1 f32 splats -> lower then
// upper VL=1 compare/select clamp -> VL=1 store), the same as the grouped/unpacked
// candidate -- numerically a single-scalar lower-then-upper clamped write to out[0]
// (HW-validated on ssh rvv, tolerance=1e-05). The legacy packed-i4 scalar
// fmaxf/fminf + out[0]= clamp store is retired (numerically equivalent at VL=1).
// CPP: __riscv_vfmv_v_f_f32m1
// CPP: __riscv_vmflt_vv_f32m1_b32
// CPP: __riscv_vmerge_vvm_f32m1
// CPP: __riscv_vmflt_vv_f32m1_b32
// CPP: __riscv_vmerge_vvm_f32m1
// CPP: __riscv_vse32_v_f32m1
// CPP-NOT: __builtin_fmaxf
// CPP-NOT: __builtin_fminf
