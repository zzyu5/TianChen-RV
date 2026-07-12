// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-dequant-clamp-f32-epilogue-plan-validated/s//provider_supported_mirror:rvv-script-derived-dequant-clamp/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/lhs,scale,lower_bound,upper_bound,out,n/s//lhs,lower_bound,scale,upper_bound,out,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/scale=dequant-scale-value:scale:abi|scale|sf32|hdr/s//scale=dequant-scale-value:scale:abi|scale|hdr/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.computed_mask_select_route_family_plan", value = "rvv-dequant-clamp-f32-epilogue-route-family-plan.v1"/s//weft_rvv.computed_mask_select_route_family_plan", value = "rvv-computed-mask-select-route-family-plan.v1"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROUTE-FAMILY
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"/s//weft_rvv.required_header_declarations", value = "stddef.h,stdint.h"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/scale:float/s//scale:double/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CTYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.lower_bound_role", value = "lower-bound-scalar-value"/s//weft_rvv.lower_bound_role", value = "rhs-scalar-value"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-LOWER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.upper_bound_c_type", value = "float"/s//weft_rvv.upper_bound_c_type", value = "double"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-UPPER-CTYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/__riscv_vle32_v_i32m1/s//__riscv_vle32_v_f32m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SOURCE-LOAD
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/__riscv_vfcvt_f_x_v_f32m1/s//__riscv_vle32_v_f32m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/__riscv_vfmul_vf_f32m1/s//__riscv_vfadd_vf_f32m1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT-SCALE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.dequant_scale_role", value = "dequant-scale-value"/s//weft_rvv.dequant_scale_role", value = "runtime-scale-from-metadata"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SCALE-ROLE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.dequant_scale_c_type", value = "float"/s//weft_rvv.dequant_scale_c_type", value = "double"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SCALE-CTYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.dequant_scale_name", value = "scale"/s//weft_rvv.dequant_scale_name", value = "quant_scale"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SCALE-NAME

module {
  weft.exec.kernel @pre_realized_dequant_clamp_f32_epilogue_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_rvv_dequant_clamp_f32_epilogue attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-dequant-clamp:n", role = "runtime-element-count"} : index
      weft_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body %lhs, %scale, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", dequant_relation = "signed-i32m1-to-f32m1-scale-f32", lmul = "m1", lower_predicate_kind = "slt", memory_form = "unit-stride-dequant-clamp-f32-epilogue", op_kind = "dequant_clamp_f32_epilogue", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, scale_role = "dequant-scale-value", select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_rvv_dequant_clamp_f32_epilogue {origin = "rvv-plugin", policy = "pre-realized-dequant-clamp-f32-epilogue-case"}
      weft.exec.fallback @pre_realized_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-dequant-clamp-f32-epilogue-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_rvv_dequant_clamp_f32_epilogue
// REALIZED: %[[SRC:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m1">
// REALIZED: %[[DEQ:.*]] = weft_rvv.dequantize %[[SRC]]
// REALIZED-SAME: kind = "i32_to_f32_scaled"
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER_MASK:.*]] = weft_rvv.compare %[[DEQ]], %[[LOWER]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[LOWER_CLAMPED:.*]] = weft_rvv.select %[[LOWER_MASK]], %[[LOWER]], %[[DEQ]], %[[VL]]
// REALIZED: %[[UPPER_MASK:.*]] = weft_rvv.compare %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[CLAMPED:.*]] = weft_rvv.select %[[UPPER_MASK]], %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED: weft_rvv.store %{{.*}}, %[[CLAMPED]], %[[VL]]
// REALIZED-NOT: weft_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "dequant_clamp_f32_epilogue"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "f32"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-stride-dequant-clamp-f32-epilogue"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,scale,lower_bound,upper_bound,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:dequant_clamp_f32_epilogue.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:dequant_clamp_f32_epilogue.v1;lhs=lhs-input-buffer:lhs:abi|src-load|deq-src|i32m1|i32-f32-scale|lcmp|lselF|hdr;scale=dequant-scale-value:scale:abi|scale|sf32|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lsp|lcmp|lselT|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|usp|ucmp|uselT|hdr;out=output-buffer:out:abi|store|dequant-result|res-f32m1|i32-f32-scale|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_route_family_plan", value = "rvv-dequant-clamp-f32-epilogue-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_mask_producer_source", value = "i32-to-f32-dequant-then-two-compare-two-select-f32-clamp-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-i32m1-to-f32m1-runtime-scale-lower-upper-clamp-select-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-dequant-clamp-f32-epilogue-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float,lower:float,upper:float,mask:f32m1-predicate,result:f32m1"}
// PLAN-SAME: {key = "weft_rvv.lower_bound_role", value = "lower-bound-scalar-value"}
// PLAN-SAME: {key = "weft_rvv.upper_bound_role", value = "upper-bound-scalar-value"}
// PLAN-SAME: {key = "weft_rvv.lower_bound_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.upper_bound_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.bound_order", value = "lower-bound-before-upper-bound"}
// PLAN-SAME: {key = "weft_rvv.clamp_relation", value = "i32-input-runtime-scale-dequant-lower-select-then-upper-select-f32-runtime-bounds"}
// PLAN-SAME: {key = "weft_rvv.source_vector_type", value = "!weft_rvv.vector<i32, \22m1\22>"}
// PLAN-SAME: {key = "weft_rvv.source_vector_c_type", value = "vint32m1_t"}
// PLAN-SAME: {key = "weft_rvv.source_vector_load_intrinsic", value = "__riscv_vle32_v_i32m1"}
// PLAN-SAME: {key = "weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "weft_rvv.dequantize_convert_intrinsic", value = "__riscv_vfcvt_f_x_v_f32m1"}
// PLAN-SAME: {key = "weft_rvv.dequantize_scale_intrinsic", value = "__riscv_vfmul_vf_f32m1"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_name", value = "scale"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: target = @pre_realized_rvv_dequant_clamp_f32_epilogue

// HEADER: weft.rvv.selected_variant: @pre_realized_rvv_dequant_clamp_f32_epilogue
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-dequant-clamp-f32-epilogue-callable-c-abi.v1
// HEADER: weft.rvv.element_type: f32
// HEADER: weft.rvv.runtime_abi_order: lhs,scale,lower_bound,upper_bound,out,n
// HEADER: weft.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER: weft.rvv.dequantize_convert_intrinsic: __riscv_vfcvt_f_x_v_f32m1
// HEADER: weft.rvv.dequantize_scale_intrinsic: __riscv_vfmul_vf_f32m1
// HEADER: weft.rvv.dequant_scale_role: dequant-scale-value
// HEADER: weft.rvv.dequant_scale_c_type: float
// HEADER: weft.rvv.dequant_scale_name: scale
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-dequant-clamp-f32-epilogue-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:dequant_clamp_f32_epilogue.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:dequant_clamp_f32_epilogue.v1
// HEADER-SAME: scale=dequant-scale-value:scale:abi|scale|sf32|hdr
// HEADER: weft.rvv.computed_mask_select_route_family_plan: rvv-dequant-clamp-f32-epilogue-route-family-plan.v1
// HEADER: weft.rvv.computed_mask_select_mask_producer_source: i32-to-f32-dequant-then-two-compare-two-select-f32-clamp-same-vl-scope
// HEADER: weft.rvv.lower_bound_role: lower-bound-scalar-value
// HEADER: weft.rvv.upper_bound_role: upper-bound-scalar-value
// HEADER: weft.rvv.lower_bound_c_type: float
// HEADER: weft.rvv.upper_bound_c_type: float
// HEADER: weft.rvv.bound_order: lower-bound-before-upper-bound
// HEADER: weft.rvv.clamp_relation: i32-input-runtime-scale-dequant-lower-select-then-upper-select-f32-runtime-bounds
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float,lower:float,upper:float,mask:f32m1-predicate,result:f32m1
// HEADER: void weft_emitc_pre_realized_dequant_clamp_f32_epilogue_kernel_pre_realized_rvv_dequant_clamp_f32_epilogue(const int32_t *lhs, float scale, float lower_bound, float upper_bound, float *out, size_t n);

// STALE-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-PROVIDER: candidate weft_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-dequant-clamp

// STALE-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ABI: weft_rvv.runtime_abi_order
// STALE-ABI-SAME: must mirror
// STALE-ABI-SAME: lhs,lower_bound,scale,upper_bound,out,n

// STALE-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-BINDING: weft_rvv.route_operand_binding_operands
// STALE-BINDING-SAME: must mirror
// STALE-BINDING-SAME: scale=dequant-scale-value:scale:abi|scale|hdr

// STALE-ROUTE-FAMILY: RVV materialized EmitC target artifact bridge failed
// STALE-ROUTE-FAMILY: weft_rvv.computed_mask_select_route_family_plan
// STALE-ROUTE-FAMILY-SAME: must mirror
// STALE-ROUTE-FAMILY-SAME: rvv-computed-mask-select-route-family-plan.v1

// STALE-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-HEADER: weft_rvv.required_header_declarations
// STALE-HEADER-SAME: must mirror
// STALE-HEADER-SAME: stddef.h,stdint.h

// STALE-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-CTYPE: weft_rvv.c_type_mapping
// STALE-CTYPE-SAME: must mirror
// STALE-CTYPE-SAME: scale:double

// STALE-LOWER: metadata key '{{.*}}lower_bound_role'{{.*}}'lower-bound-scalar-value' but was 'rhs-scalar-value'

// STALE-UPPER-CTYPE: metadata key '{{.*}}upper_bound_c_type'{{.*}}'float' but was 'double'

// STALE-SOURCE-LOAD: RVV materialized EmitC target artifact bridge failed
// STALE-SOURCE-LOAD: weft_rvv.source_vector_load_intrinsic
// STALE-SOURCE-LOAD-SAME: must mirror
// STALE-SOURCE-LOAD-SAME: __riscv_vle32_v_f32m1

// STALE-DEQUANT: RVV materialized EmitC target artifact bridge failed
// STALE-DEQUANT: weft_rvv.dequantize_convert_intrinsic
// STALE-DEQUANT-SAME: must mirror
// STALE-DEQUANT-SAME: __riscv_vle32_v_f32m1

// STALE-DEQUANT-SCALE: RVV materialized EmitC target artifact bridge failed
// STALE-DEQUANT-SCALE: weft_rvv.dequantize_scale_intrinsic
// STALE-DEQUANT-SCALE-SAME: must mirror
// STALE-DEQUANT-SCALE-SAME: __riscv_vfadd_vf_f32m1

// STALE-SCALE-ROLE: RVV materialized EmitC target artifact bridge failed
// STALE-SCALE-ROLE: weft_rvv.dequant_scale_role
// STALE-SCALE-ROLE-SAME: must mirror
// STALE-SCALE-ROLE-SAME: runtime-scale-from-metadata

// STALE-SCALE-CTYPE: RVV materialized EmitC target artifact bridge failed
// STALE-SCALE-CTYPE: weft_rvv.dequant_scale_c_type
// STALE-SCALE-CTYPE-SAME: must mirror
// STALE-SCALE-CTYPE-SAME: double

// STALE-SCALE-NAME: RVV materialized EmitC target artifact bridge failed
// STALE-SCALE-NAME: weft_rvv.dequant_scale_name
// STALE-SCALE-NAME-SAME: must mirror
// STALE-SCALE-NAME-SAME: quant_scale
