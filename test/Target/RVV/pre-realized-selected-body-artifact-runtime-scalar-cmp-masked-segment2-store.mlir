// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed 's/runtime-scalar-splat-compare-rhs/vector-compare-rhs-load/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRODUCER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed 's/src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|tuple1|f1-role|src1-mem|hdr/src1=segment-field0-input-buffer:src1:abi|f1-load|f1-payload|tuple1|f1-role|src1-mem|hdr/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING

// Pre-realized selected-body input for one bounded Stage2 runtime scalar
// compare plus two-field segment2 masked store slice. The RVV plugin must
// realize lhs load, rhs_scalar splat, field0/field1 payload loads,
// compare-produced mask, and masked_segment2_store into explicit typed
// structure before route planning.

module {
  weft.exec.kernel @pre_realized_body_rt_scalar_cmseg_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_rt_scalar_cmseg_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store:src0", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store:src1", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store:dst", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body %lhs, %rhs_scalar, %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", inactive_lane_policy = "preserve-output-on-false-lanes", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-load-segment2-store", op_kind = "runtime_scalar_cmp_masked_segment2_store_unit_load", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_rt_scalar_cmseg_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-cmp-masked-segment2-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_rt_scalar_cmseg_store
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED: %[[FIELD0:.*]] = weft_rvv.load
// REALIZED: %[[FIELD1:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: weft_rvv.masked_segment2_store %{{.*}}, %[[MASK]], %[[FIELD0]], %[[FIELD1]], %[[VL]]
// REALIZED-SAME: destination_memory_form = "segment2-interleaved-unit-stride-store"
// REALIZED-SAME: field0_role = "segment-field0-input-buffer"
// REALIZED-SAME: field1_role = "segment-field1-input-buffer"
// REALIZED-SAME: inactive_lane_policy = "preserve-output-on-false-lanes"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-NOT: weft_rvv.load %{{.*}}rhs_scalar
// REALIZED-NOT: weft_rvv.masked_indexed_store
// REALIZED-NOT: weft_rvv.strided_load
// REALIZED-NOT: weft_rvv.segment2_store
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_segment2_store_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_segment2_store"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-unit-load-segment2-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs_scalar,src0,src1,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_store_unit_load.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_store_unit_load.v1;lhs=lhs-input-buffer:lhs:abi|cmp-lhs-load|lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr;src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|tuple0|f0-role|src0-mem|hdr;src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|tuple1|f1-role|src1-mem|hdr;dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "weft_rvv.mask_tail_policy_route_family_plan", value = "rvv-mask-tail-policy-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.mask_tail_policy_owner", value = "computed-mask memory mask/tail policy"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-segment2-store-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-store-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/field-payloads:signed-e32m1,rhs_scalar:signed-scalar,mask:b32,segment2:vint32m1x2,dst:runtime-scalar-masked-segment2-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-field-payloads-segment2-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.segment_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-field-payloads-segment2-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.segment_count", value = "2"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_rt_scalar_cmseg_store

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_rt_scalar_cmseg_store
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-segment2-store-unit-load-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs_scalar,src0,src1,dst,n
// HEADER: weft.rvv.compare_predicate_kind: sle
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-segment2-store-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-store-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_store_unit_load.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_store_unit_load.v1;lhs=lhs-input-buffer:lhs:abi|cmp-lhs-load|lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr;src0=segment-field0-input-buffer:src0:abi|f0-load|f0-payload|tuple0|f0-role|src0-mem|hdr;src1=segment-field1-input-buffer:src1:abi|f1-load|f1-payload|tuple1|f1-role|src1-mem|hdr;dst=segment-interleaved-output-buffer:dst:abi|mseg-store|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,lhs/field-payloads:signed-e32m1,rhs_scalar:signed-scalar,mask:b32,segment2:vint32m1x2,dst:runtime-scalar-masked-segment2-store
// HEADER: void weft_emitc_pre_realized_body_rt_scalar_cmseg_store_kernel_pre_realized_body_rvv_rt_scalar_cmseg_store(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);

// STALE-PRODUCER: metadata key '{{.*}}computed_mask_memory_mask_producer_source'{{.*}}'runtime-scalar-splat-compare-rhs' but was 'vector-compare-rhs-load'
// STALE-BINDING: candidate weft_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary
// STALE-BINDING-SAME: src1=segment-field1-input-buffer:src1:abi
// STALE-BINDING-SAME: but was
// STALE-BINDING-SAME: src1=segment-field0-input-buffer:src1:abi
