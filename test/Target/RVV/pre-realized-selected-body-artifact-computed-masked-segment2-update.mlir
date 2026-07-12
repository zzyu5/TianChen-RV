// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-generic-computed-masked-segment2-update-unit-load-emitc-route/s//rvv-script-derived-segment2-update-route/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ROUTE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated/s//provider_supported_mirror:rvv-script-derived-segment2-update/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-route-operand-binding:cmseg2_update_unit_load.v1/s//rvv-route-operand-binding:script-derived-segment2-update.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed 's/src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|add-rhs|tuple1|f1-role|src1-mem|hdr/src1=segment-field0-input-buffer:src1:abi|f1-load|f1-payload|add-lhs|tuple1|f1-role|src1-mem|hdr/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SRC1-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/cmp_lhs,cmp_rhs,src0,src1,dst,n/s//cmp_lhs,src0,cmp_rhs,src1,dst,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-TYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-computed-mask-memory-route-family-plan.v1/s//rvv-script-derived-computed-mask-segment2-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CM-PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.segment_count", value = "2"/s//weft_rvv.segment_count", value = "3"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SEGCOUNT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"/s//weft_rvv.mask_role", value = "script-derived-mask-role"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MASK-ROLE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1"/s//weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1"}, {key = "weft_rvv.elementwise_arithmetic_route_family_plan", value = "rvv-script-derived-elementwise-plan.v1"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RESIDUE

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// segment2 update slice. The RVV plugin must realize compare lhs/rhs,
// field0/field1 payload loads, add(field0, field1), compare-produced mask,
// and masked_segment2_store into explicit typed structure before route
// planning.

module {
  weft.exec.kernel @pre_realized_body_cmseg_update_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_cmseg_update attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-update:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-update:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-update:src0", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-update:src1", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-update:dst", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-update:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_segment2_store_pre_realized_body %cmp_lhs, %cmp_rhs, %src0, %src1, %dst, %n {arithmetic_kind = "add", destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", inactive_lane_policy = "preserve-output-on-false-lanes", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-segment2-store", op_kind = "computed_masked_segment2_update_unit_load", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_cmseg_update {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-segment2-update-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-segment2-update-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_segment2_store_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmseg_update
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED: %[[FIELD0:.*]] = weft_rvv.load
// REALIZED: %[[FIELD1:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[UPDATED:.*]] = weft_rvv.binary %[[FIELD0]], %[[FIELD1]], %[[VL]]
// REALIZED-SAME: kind = "add"
// REALIZED: weft_rvv.masked_segment2_store %{{.*}}, %[[MASK]], %[[UPDATED]], %[[FIELD1]], %[[VL]]
// REALIZED-SAME: destination_memory_form = "segment2-interleaved-unit-stride-store"
// REALIZED-SAME: field0_role = "segment-field0-input-buffer"
// REALIZED-SAME: field1_role = "segment-field1-input-buffer"
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-NOT: weft_rvv.masked_indexed_store
// REALIZED-NOT: weft_rvv.strided_load
// REALIZED-NOT: weft_rvv.segment2_store
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.mask_load
// REALIZED-NOT: weft_rvv.typed_computed_mask_segment2_store_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_segment2_update_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.binary"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-unit-load-segment2-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src0,src1,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:cmseg2_update_unit_load.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:cmseg2_update_unit_load.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|add-lhs|tuple0|f0-role|src0-mem|hdr;src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|add-rhs|tuple1|f1-role|src1-mem|hdr;dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-update-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-compare-field-payloads-arithmetic-segment2-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.segment_memory_layout", value = "unit-stride-compare-field-payloads-arithmetic-segment2-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.segment_count", value = "2"}
// PLAN-SAME: {key = "weft_rvv.segment_store_intrinsic", value = "__riscv_vsseg2e32_v_i32m1x2_m"}
// PLAN-SAME: {key = "weft_rvv.segment_tuple_create_intrinsic", value = "__riscv_vcreate_v_i32m1x2"}
// PLAN-SAME: {key = "weft_rvv.segment2_update_arithmetic_kind", value = "add"}
// PLAN-SAME: {key = "weft_rvv.segment2_update_arithmetic_intrinsic", value = "__riscv_vadd_vv_i32m1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmseg_update

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_cmseg_update
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-masked-segment2-update-unit-load-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src0,src1,dst,n
// HEADER: weft.rvv.compare_predicate_kind: slt
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-e32m1-computed-mask-segment2-update-add-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:cmseg2_update_unit_load.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:cmseg2_update_unit_load.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|add-lhs|tuple0|f0-role|src0-mem|hdr;src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|add-rhs|tuple1|f1-role|src1-mem|hdr;dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.computed_mask_memory_route_family_plan: rvv-computed-mask-memory-route-family-plan.v1
// HEADER: weft.rvv.computed_mask_memory_mask_producer_source: vector-compare-rhs-load
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-update-store
// HEADER: weft.rvv.segment2_update_arithmetic_kind: add
// HEADER: void weft_emitc_pre_realized_body_cmseg_update_kernel_pre_realized_body_rvv_cmseg_update(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);

// STALE-ROUTE: RVV materialized EmitC target artifact bridge failed
// STALE-ROUTE: candidate rvv_emitc_lowerable_route provenance must mirror selected typed RVV body route
// STALE-ROUTE-SAME: rvv-script-derived-segment2-update-route

// STALE-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-PROVIDER: candidate weft_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-segment2-update

// STALE-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-BINDING: candidate weft_rvv.route_operand_binding_plan provenance must mirror selected typed RVV body binding plan
// STALE-BINDING-SAME: rvv-route-operand-binding:script-derived-segment2-update.v1

// STALE-SRC1-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-SRC1-BINDING: candidate weft_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary
// STALE-SRC1-BINDING-SAME: src1=segment-field1-input-buffer:src1:abi
// STALE-SRC1-BINDING-SAME: add-rhs
// STALE-SRC1-BINDING-SAME: but was
// STALE-SRC1-BINDING-SAME: src1=segment-field0-input-buffer:src1:abi
// STALE-SRC1-BINDING-SAME: add-lhs

// STALE-ABI: metadata key '{{.*}}runtime_abi_order'{{.*}}'cmp_lhs,cmp_rhs,src0,src1,dst,n' but was 'cmp_lhs,src0,cmp_rhs,src1,dst,n'

// STALE-HEADER: metadata key '{{.*}}required_header_declarations'{{.*}}'stddef.h,stdint.h,riscv_vector.h' but was 'stddef.h,stdint.h'

// STALE-TYPE: metadata key '{{.*}}c_type_mapping'{{.*}}'vl:size_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-update-store' but was 'vl:uint64_t,compare/field-payloads/update-add:signed-e32m1,mask:b32,segment2:vint32m1x2,dst:masked-segment2-update-store'

// STALE-CM-PLAN: metadata key '{{.*}}computed_mask_memory_route_family_plan'{{.*}}'rvv-computed-mask-memory-route-family-plan.v1' but was 'rvv-script-derived-computed-mask-segment2-plan.v1'

// STALE-SEGCOUNT: metadata key '{{.*}}segment_count'{{.*}}'2' but was '3'

// STALE-MASK-ROLE: metadata key '{{.*}}mask_role'{{.*}}'predicate-mask-produced-by-compare' but was 'script-derived-mask-role'

// STALE-RESIDUE: carry exactly 58{{.*}}metadata entries
