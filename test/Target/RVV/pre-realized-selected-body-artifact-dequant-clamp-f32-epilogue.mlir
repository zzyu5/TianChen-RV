// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-dequant-clamp-f32-epilogue-plan-validated/s//provider_supported_mirror:rvv-script-derived-dequant-clamp/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lhs,scale,lower_bound,upper_bound,out,n/s//lhs,lower_bound,scale,upper_bound,out,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/__riscv_vle32_v_i32m1/s//__riscv_vle32_v_f32m1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SOURCE-LOAD
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/__riscv_vfcvt_f_x_v_f32m1/s//__riscv_vle32_v_f32m1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT

module {
  tcrv.exec.kernel @pre_realized_dequant_clamp_f32_epilogue_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_rvv_dequant_clamp_f32_epilogue attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %lower = tcrv_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:lower", role = "lower-bound-scalar-value"} : f32
      %upper = tcrv_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:upper", role = "upper-bound-scalar-value"} : f32
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, f32, f32, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_rvv_dequant_clamp_f32_epilogue {origin = "rvv-plugin", policy = "pre-realized-dequant-clamp-f32-epilogue-case"}
      tcrv.exec.fallback @pre_realized_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-dequant-clamp-f32-epilogue-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_rvv_dequant_clamp_f32_epilogue
// REALIZED: %[[SRC:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[DEQ:.*]] = tcrv_rvv.dequantize %[[SRC]]
// REALIZED-SAME: kind = "i32_to_f32_scaled"
// REALIZED-SAME: -> !tcrv_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER:.*]] = tcrv_rvv.splat
// REALIZED-SAME: -> !tcrv_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER:.*]] = tcrv_rvv.splat
// REALIZED-SAME: -> !tcrv_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER_MASK:.*]] = tcrv_rvv.compare %[[DEQ]], %[[LOWER]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[LOWER_CLAMPED:.*]] = tcrv_rvv.select %[[LOWER_MASK]], %[[LOWER]], %[[DEQ]], %[[VL]]
// REALIZED: %[[UPPER_MASK:.*]] = tcrv_rvv.compare %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[CLAMPED:.*]] = tcrv_rvv.select %[[UPPER_MASK]], %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED: tcrv_rvv.store %{{.*}}, %[[CLAMPED]], %[[VL]]
// REALIZED-NOT: tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "dequant_clamp_f32_epilogue"}
// PLAN-SAME: {key = "tcrv_rvv.element_type", value = "f32"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-stride-dequant-clamp-f32-epilogue"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,scale,lower_bound,upper_bound,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:dequant_clamp_f32_epilogue.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_select_route_family_plan", value = "rvv-dequant-clamp-f32-epilogue-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i32m1-to-f32m1-runtime-scale-lower-upper-clamp-select-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-dequant-clamp-f32-epilogue-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.lower_bound_role", value = "lower-bound-scalar-value"}
// PLAN-SAME: {key = "tcrv_rvv.upper_bound_role", value = "upper-bound-scalar-value"}
// PLAN-SAME: {key = "tcrv_rvv.bound_order", value = "lower-bound-before-upper-bound"}
// PLAN-SAME: {key = "tcrv_rvv.clamp_relation", value = "i32-input-runtime-scale-dequant-lower-select-then-upper-select-f32-runtime-bounds"}
// PLAN-SAME: {key = "tcrv_rvv.source_vector_type", value = "!tcrv_rvv.vector<i32, \22m1\22>"}
// PLAN-SAME: {key = "tcrv_rvv.source_vector_load_intrinsic", value = "__riscv_vle32_v_i32m1"}
// PLAN-SAME: {key = "tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "tcrv_rvv.dequantize_convert_intrinsic", value = "__riscv_vfcvt_f_x_v_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.dequantize_scale_intrinsic", value = "__riscv_vfmul_vf_f32m1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: target = @pre_realized_rvv_dequant_clamp_f32_epilogue

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_rvv_dequant_clamp_f32_epilogue
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-dequant-clamp-f32-epilogue-callable-c-abi.v1
// HEADER: tianchenrv.rvv.element_type: f32
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,scale,lower_bound,upper_bound,out,n
// HEADER: tianchenrv.rvv.dequantize_convert_intrinsic: __riscv_vfcvt_f_x_v_f32m1
// HEADER: tianchenrv.rvv.dequantize_scale_intrinsic: __riscv_vfmul_vf_f32m1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-dequant-clamp-f32-epilogue-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:dequant_clamp_f32_epilogue.v1
// HEADER: tianchenrv.rvv.computed_mask_select_route_family_plan: rvv-dequant-clamp-f32-epilogue-route-family-plan.v1
// HEADER: tianchenrv.rvv.lower_bound_role: lower-bound-scalar-value
// HEADER: tianchenrv.rvv.upper_bound_role: upper-bound-scalar-value
// HEADER: tianchenrv.rvv.clamp_relation: i32-input-runtime-scale-dequant-lower-select-then-upper-select-f32-runtime-bounds
// HEADER: void tcrv_emitc_pre_realized_dequant_clamp_f32_epilogue_kernel_pre_realized_rvv_dequant_clamp_f32_epilogue(const int32_t *lhs, float scale, float lower_bound, float upper_bound, float *out, size_t n);

// STALE-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-PROVIDER: candidate tcrv_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-dequant-clamp

// STALE-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ABI: tcrv_rvv.runtime_abi_order
// STALE-ABI-SAME: must mirror
// STALE-ABI-SAME: lhs,lower_bound,scale,upper_bound,out,n

// STALE-SOURCE-LOAD: RVV materialized EmitC target artifact bridge failed
// STALE-SOURCE-LOAD: tcrv_rvv.source_vector_load_intrinsic
// STALE-SOURCE-LOAD-SAME: must mirror
// STALE-SOURCE-LOAD-SAME: __riscv_vle32_v_f32m1

// STALE-DEQUANT: RVV materialized EmitC target artifact bridge failed
// STALE-DEQUANT: tcrv_rvv.dequantize_convert_intrinsic
// STALE-DEQUANT-SAME: must mirror
// STALE-DEQUANT-SAME: __riscv_vle32_v_f32m1
