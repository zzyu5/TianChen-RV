// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-f32-clamp-select-runtime-bounds-plan-validated/s//provider_supported_mirror:rvv-script-derived-f32-clamp-select/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/input,lower_bound,upper_bound,out,n/s//input,upper_bound,lower_bound,out,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.lower_bound_role\", value = \"lower-bound-scalar-value\"/s//weft_rvv.lower_bound_role\", value = \"rhs-scalar-value\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-LOWER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.bound_order\", value = \"lower-bound-before-upper-bound\"/s//weft_rvv.bound_order\", value = \"upper-bound-before-lower-bound\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-BOUND
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.route_operand_binding_operands\", value = \"rvv-route-operand-binding:f32_clamp_select.v1;input=lhs-input-buffer:input:abi|ld|lcmp|lselF|hdr/s//weft_rvv.route_operand_binding_operands\", value = \"rvv-route-operand-binding:f32_clamp_select.v1;input=lhs-input-buffer:input:abi|ld|stale-lcmp|lselF|hdr/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-f32-clamp-select-route-family-plan.v1/s//rvv-stale-f32-clamp-select-route-family-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-FAMILY
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.upper_bound_c_type\", value = \"float\"/s//weft_rvv.upper_bound_c_type\", value = \"double\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-UPPER-CTYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.secondary_compare_predicate_kind\", value = \"slt\"/s//weft_rvv.secondary_compare_predicate_kind\", value = \"sle\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-SECONDARY-PRED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.required_header_declarations\", value = \"stddef.h,stdint.h,riscv_vector.h\"/s//weft_rvv.required_header_declarations\", value = \"stddef.h,stdint.h\"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-F32-HEADER

module {
  weft.exec.kernel @pre_realized_f32_clamp_select_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_rvv_f32_clamp_select attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %input = weft_rvv.runtime_abi_value {c_name = "input", c_type = "const float *", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:input", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lower = weft_rvv.runtime_abi_value {c_name = "lower_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:lower", role = "lower-bound-scalar-value"} : f32
      %upper = weft_rvv.runtime_abi_value {c_name = "upper_bound", c_type = "float", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:upper", role = "upper-bound-scalar-value"} : f32
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-f32-clamp-select:n", role = "runtime-element-count"} : index
      weft_rvv.typed_f32_clamp_select_pre_realized_body %input, %lower, %upper, %out, %n {bound_order = "lower-bound-before-upper-bound", lmul = "m1", lower_predicate_kind = "slt", memory_form = "runtime-scalar-f32-clamp-select", op_kind = "f32_clamp_select", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, select_layout = "clamp-lower-then-upper", sew = 32 : i64, upper_predicate_kind = "slt"} : (!weft_rvv.runtime_abi_value, f32, f32, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_rvv_f32_clamp_select {origin = "rvv-plugin", policy = "pre-realized-f32-clamp-select-case"}
      weft.exec.fallback @pre_realized_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-f32-clamp-select-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_f32_clamp_select_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_rvv_f32_clamp_select
// REALIZED: %[[INPUT:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[LOWER_MASK:.*]] = weft_rvv.compare %[[INPUT]], %[[LOWER]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !weft_rvv.mask<f32, "m1">
// REALIZED: %[[LOWER_CLAMPED:.*]] = weft_rvv.select %[[LOWER_MASK]], %[[LOWER]], %[[INPUT]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: %[[UPPER_MASK:.*]] = weft_rvv.compare %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !weft_rvv.mask<f32, "m1">
// REALIZED: %[[CLAMPED:.*]] = weft_rvv.select %[[UPPER_MASK]], %[[UPPER]], %[[LOWER_CLAMPED]], %[[VL]]
// REALIZED-SAME: -> !weft_rvv.vector<f32, "m1">
// REALIZED: weft_rvv.store %{{.*}}, %[[CLAMPED]], %[[VL]]
// REALIZED-NOT: weft_rvv.typed_f32_clamp_select_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "f32_clamp_select"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "f32"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "runtime-scalar-f32-clamp-select"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "input,lower_bound,upper_bound,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:f32_clamp_select.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:f32_clamp_select.v1;input=lhs-input-buffer:input:abi|ld|lcmp|lselF|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lsp|lcmp|lselT|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|usp|ucmp|uselT|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_route_family_plan", value = "rvv-f32-clamp-select-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_select_mask_producer_source", value = "two-compare-two-select-f32-clamp-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_tail_policy_route_family_plan", value = "rvv-mask-tail-policy-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-f32m1-runtime-lower-upper-clamp-select-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-f32-clamp-select-runtime-bounds-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,input:f32m1,lower:float,upper:float,mask:f32m1-predicate,result:f32m1"}
// PLAN-SAME: {key = "weft_rvv.secondary_compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "weft_rvv.lower_bound_role", value = "lower-bound-scalar-value"}
// PLAN-SAME: {key = "weft_rvv.upper_bound_role", value = "upper-bound-scalar-value"}
// PLAN-SAME: {key = "weft_rvv.lower_bound_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.upper_bound_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.bound_order", value = "lower-bound-before-upper-bound"}
// PLAN-SAME: {key = "weft_rvv.clamp_relation", value = "input-lower-select-then-upper-select-f32-runtime-bounds"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: target = @pre_realized_rvv_f32_clamp_select

// HEADER: weft.rvv.selected_variant: @pre_realized_rvv_f32_clamp_select
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-f32-clamp-select-callable-c-abi.v1
// HEADER: weft.rvv.element_type: f32
// HEADER: weft.rvv.runtime_abi_order: input,lower_bound,upper_bound,out,n
// HEADER-DAG: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-f32-clamp-select-runtime-bounds-plan-validated
// HEADER-DAG: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:f32_clamp_select.v1
// HEADER-DAG: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:f32_clamp_select.v1;input=lhs-input-buffer:input:abi|ld|lcmp|lselF|hdr;lower_bound=lower-bound-scalar-value:lower_bound:abi|lsp|lcmp|lselT|hdr;upper_bound=upper-bound-scalar-value:upper_bound:abi|usp|ucmp|uselT|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER-DAG: weft.rvv.computed_mask_select_route_family_plan: rvv-f32-clamp-select-route-family-plan.v1
// HEADER-DAG: weft.rvv.computed_mask_select_mask_producer_source: two-compare-two-select-f32-clamp-same-vl-scope
// HEADER-DAG: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER-DAG: weft.rvv.c_type_mapping: vl:size_t,input:f32m1,lower:float,upper:float,mask:f32m1-predicate,result:f32m1
// HEADER-DAG: weft.rvv.lower_bound_role: lower-bound-scalar-value
// HEADER-DAG: weft.rvv.upper_bound_role: upper-bound-scalar-value
// HEADER-DAG: weft.rvv.lower_bound_c_type: float
// HEADER-DAG: weft.rvv.upper_bound_c_type: float
// HEADER-DAG: weft.rvv.secondary_compare_predicate_kind: slt
// HEADER-DAG: weft.rvv.bound_order: lower-bound-before-upper-bound
// HEADER-DAG: weft.rvv.clamp_relation: input-lower-select-then-upper-select-f32-runtime-bounds
// HEADER: void weft_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select(const float *input, float lower_bound, float upper_bound, float *out, size_t n);

// STALE-F32-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-F32-PROVIDER: candidate weft_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-F32-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-f32-clamp-select

// STALE-F32-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-F32-ABI: weft_rvv.runtime_abi_order
// STALE-F32-ABI-SAME: must mirror
// STALE-F32-ABI-SAME: input,upper_bound,lower_bound,out,n

// STALE-F32-LOWER: metadata key '{{.*}}lower_bound_role'{{.*}}'lower-bound-scalar-value' but was 'rhs-scalar-value'

// STALE-F32-BOUND: metadata key '{{.*}}bound_order'{{.*}}'lower-bound-before-upper-bound' but was 'upper-bound-before-lower-bound'

// STALE-F32-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-F32-BINDING: candidate weft_rvv.route_operand_binding_operands provenance
// STALE-F32-BINDING-SAME: must mirror
// STALE-F32-BINDING-SAME: stale-lcmp

// STALE-F32-FAMILY: metadata key '{{.*}}computed_mask_select_route_family_plan'{{.*}}'rvv-f32-clamp-select-route-family-plan.v1' but was 'rvv-stale-f32-clamp-select-route-family-plan.v1'

// STALE-F32-UPPER-CTYPE: metadata key '{{.*}}upper_bound_c_type'{{.*}}'float' but was 'double'

// STALE-F32-SECONDARY-PRED: metadata key '{{.*}}secondary_compare_predicate_kind'{{.*}}'slt' but was 'sle'

// STALE-F32-HEADER: metadata key '{{.*}}required_header_declarations'{{.*}}'stddef.h,stdint.h,riscv_vector.h' but was 'stddef.h,stdint.h'
