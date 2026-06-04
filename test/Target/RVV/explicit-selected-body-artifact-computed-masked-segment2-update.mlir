// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/rvv-generic-computed-masked-segment2-update-unit-load-emitc-route/s//rvv-script-derived-segment2-update-route/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROUTE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated/s//provider_supported_mirror:rvv-script-derived-segment2-update/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/rvv-route-operand-binding:cmseg2_update_unit_load.v1/s//rvv-route-operand-binding:script-derived-segment2-update.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/cmp_lhs,cmp_rhs,src0,src1,dst,n/s//cmp_lhs,src0,cmp_rhs,src1,dst,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-TYPE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"/s//tcrv_rvv.mask_source", value = "script-derived-mask-source"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MASK
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.inactive_lane_contract", value = "masked-store-false-lanes-preserve-output-buffer"/s//tcrv_rvv.inactive_lane_contract", value = "script-derived-passthrough"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-INACTIVE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.field0_role", value = "segment-field0-input-buffer"/s//tcrv_rvv.field0_role", value = "segment-field0-output-buffer"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-FIELD
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.segment2_update_arithmetic_kind", value = "add"/s//tcrv_rvv.segment2_update_arithmetic_kind", value = "script-derived-sub"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-UPDATE

// Hand-authored explicit selected-body input for one bounded Stage2 computed
// mask plus segment2 update slice. The selected RVV body structurally carries
// compare-produced mask, field0/field1 payload loads, add(field0, field1),
// interleaved destination memory, inactive-lane no-write policy, and runtime
// n/AVL through tcrv_rvv.masked_segment2_store.

module {
  tcrv.exec.kernel @explicit_selected_body_cmseg_update_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_cmseg_update attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-update:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-update:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src0 = tcrv_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-update:src0", role = "segment-field0-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src1 = tcrv_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-update:src1", role = "segment-field1-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-update:dst", role = "segment-interleaved-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-segment2-update:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_cmseg_update, sew = 32 : i64, source_kernel = "explicit_selected_body_cmseg_update_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.load %cmp_rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %field0 = tcrv_rvv.load %src0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %field1 = tcrv_rvv.load %src1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %updated = tcrv_rvv.binary %field0, %field1, %vl {kind = "add"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.masked_segment2_store %dst, %mask, %updated, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", inactive_lane_policy = "preserve-output-on-false-lanes", segment_count = 2 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_cmseg_update {origin = "rvv-plugin", policy = "explicit-selected-body-computed-mask-segment2-update-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-computed-mask-segment2-update-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_segment2_update_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.binary"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-unit-load-segment2-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src0,src1,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:cmseg2_update_unit_load.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:cmseg2_update_unit_load.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|add-lhs|tuple0|f0-role|src0-mem|hdr;src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|add-rhs|tuple1|f1-role|src1-mem|hdr;dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-update-store"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-compare-field-payloads-arithmetic-segment2-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "masked-store-has-no-passthrough-load"}
// PLAN-SAME: {key = "tcrv_rvv.segment_memory_layout", value = "unit-stride-compare-field-payloads-arithmetic-segment2-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.segment_count", value = "2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_tuple_c_type", value = "vint32m1x2_t"}
// PLAN-SAME: {key = "tcrv_rvv.segment_store_intrinsic", value = "__riscv_vsseg2e32_v_i32m1x2_m"}
// PLAN-SAME: {key = "tcrv_rvv.segment_tuple_create_intrinsic", value = "__riscv_vcreate_v_i32m1x2"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "segment2-interleaved-unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.field0_role", value = "segment-field0-input-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.field1_role", value = "segment-field1-input-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.field0_name", value = "masked_segment2_update_field0_vec"}
// PLAN-SAME: {key = "tcrv_rvv.field1_name", value = "masked_segment2_update_field1_vec"}
// PLAN-SAME: {key = "tcrv_rvv.field0_source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.field1_source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.segment2_update_arithmetic_kind", value = "add"}
// PLAN-SAME: {key = "tcrv_rvv.segment2_update_arithmetic_intrinsic", value = "__riscv_vadd_vv_i32m1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_cmseg_update

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_cmseg_update
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src0,src1,dst,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: slt
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:cmseg2_update_unit_load.v1
// HEADER: tianchenrv.rvv.computed_mask_memory_route_family_plan: rvv-computed-mask-memory-route-family-plan.v1
// HEADER: tianchenrv.rvv.computed_mask_memory_mask_producer_source: vector-compare-rhs-load
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-update-store
// HEADER: tianchenrv.rvv.segment2_update_arithmetic_kind: add
// HEADER: void tcrv_emitc_explicit_selected_body_cmseg_update_kernel_explicit_selected_body_rvv_cmseg_update(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);

// STALE-ROUTE: RVV materialized EmitC target artifact bridge failed
// STALE-ROUTE: candidate rvv_emitc_lowerable_route provenance must mirror selected typed RVV body route
// STALE-ROUTE-SAME: rvv-script-derived-segment2-update-route

// STALE-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-PROVIDER: candidate tcrv_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-segment2-update

// STALE-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-BINDING: candidate tcrv_rvv.route_operand_binding_plan provenance must mirror selected typed RVV body binding plan
// STALE-BINDING-SAME: rvv-route-operand-binding:script-derived-segment2-update.v1

// STALE-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ABI: candidate tcrv_rvv.runtime_abi_order provenance must mirror selected typed RVV computed-mask segment2 runtime ABI order
// STALE-ABI-SAME: cmp_lhs,src0,cmp_rhs,src1,dst,n

// STALE-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-HEADER: candidate tcrv_rvv.required_header_declarations provenance must mirror selected typed RVV computed-mask segment2 route header requirements
// STALE-HEADER-SAME: stddef.h,stdint.h

// STALE-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-TYPE: candidate tcrv_rvv.c_type_mapping provenance must mirror selected typed RVV computed-mask segment2 route type mapping summary
// STALE-TYPE-SAME: vl:uint64_t

// STALE-MASK: RVV materialized EmitC target artifact bridge failed
// STALE-MASK: candidate tcrv_rvv.mask_source provenance must mirror selected typed RVV computed-mask segment2 mask source
// STALE-MASK-SAME: script-derived-mask-source

// STALE-INACTIVE: RVV materialized EmitC target artifact bridge failed
// STALE-INACTIVE: candidate tcrv_rvv.inactive_lane_contract provenance must mirror selected typed RVV computed-mask segment2 inactive lane contract
// STALE-INACTIVE-SAME: script-derived-passthrough

// STALE-FIELD: RVV materialized EmitC target artifact bridge failed
// STALE-FIELD: candidate tcrv_rvv.field0_role provenance must mirror selected typed RVV computed-mask segment2 field0 role
// STALE-FIELD-SAME: segment-field0-output-buffer

// STALE-UPDATE: RVV materialized EmitC target artifact bridge failed
// STALE-UPDATE: candidate tcrv_rvv.segment2_update_arithmetic_kind provenance must mirror selected typed RVV computed-mask segment2 update arithmetic kind
// STALE-UPDATE-SAME: script-derived-sub
