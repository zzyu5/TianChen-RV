// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 computed
// mask plus two-field segment2 masked load slice. The selected RVV body
// structurally carries compare-produced mask, interleaved source memory,
// field0/field1 old-vector passthrough loads, field0/field1 results, dual
// field stores, and runtime n/AVL through weft_rvv.masked_segment2_load.

module {
  weft.exec.kernel @explicit_selected_body_cmseg_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_cmseg_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-load:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-load:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-load:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %out0 = weft_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-load:out0", role = "segment-field0-output-buffer"} : !weft_rvv.runtime_abi_value
      %out1 = weft_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-load:out1", role = "segment-field1-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-load:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_cmseg_load, sew = 32 : i64, source_kernel = "explicit_selected_body_cmseg_load_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_vec = weft_rvv.load %cmp_rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old0 = weft_rvv.load %out0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old1 = weft_rvv.load %out1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask = weft_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %field0, %field1 = weft_rvv.masked_segment2_load %src, %mask, %old0, %old1, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !weft_rvv.runtime_abi_value, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out0, %field0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
        weft_rvv.store %out1, %field1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_cmseg_load {origin = "rvv-plugin", policy = "explicit-selected-body-computed-mask-segment2-load-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-computed-mask-segment2-load-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_segment2_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_segment2_load"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-segment2-load-unit-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src,out0,out1,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.mask_tail_policy_route_family_plan", value = "rvv-mask-tail-policy-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.mask_tail_policy_owner", value = "computed-mask memory mask/tail policy"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-segment2-load-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-segment2-load-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,compare/source/passthrough-fields:signed-e32m1,mask:b32,segment2:vint32m1x2,result:masked-segment2-load-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-compare-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "segment2-interleaved-unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.segment_memory_layout", value = "unit-stride-compare-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.segment_count", value = "2"}
// PLAN-SAME: {key = "weft_rvv.segment_load_intrinsic", value = "__riscv_vlseg2e32_v_i32m1x2_tumu"}
// PLAN-SAME: {key = "weft_rvv.segment_tuple_create_intrinsic", value = "__riscv_vcreate_v_i32m1x2"}
// PLAN-SAME: {key = "weft_rvv.segment_field_extract_intrinsic", value = "__riscv_vget_v_i32m1x2_i32m1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_cmseg_load

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_cmseg_load
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,out0,out1,n
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-e32m1-computed-mask-segment2-load-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-segment2-load-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.computed_mask_memory_route_family_plan: rvv-computed-mask-memory-route-family-plan.v1
// HEADER: weft.rvv.computed_mask_memory_mask_producer_source: vector-compare-rhs-load
// HEADER: weft.rvv.mask_tail_policy_route_family_plan: rvv-mask-tail-policy-route-family-plan.v1
// HEADER: weft.rvv.mask_tail_policy_owner: computed-mask memory mask/tail policy
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,compare/source/passthrough-fields:signed-e32m1,mask:b32,segment2:vint32m1x2,result:masked-segment2-load-store
// HEADER: void weft_emitc_explicit_selected_body_cmseg_load_kernel_explicit_selected_body_rvv_cmseg_load(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
