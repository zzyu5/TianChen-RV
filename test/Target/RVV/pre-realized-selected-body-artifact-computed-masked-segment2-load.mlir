// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// segment2 masked load slice. The RVV plugin must realize compare lhs/rhs,
// interleaved source, field0/field1 old passthroughs, masked_segment2_load,
// and dual field stores into explicit typed structure before route construction.

module {
  tcrv.exec.kernel @pre_realized_body_cmseg_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_cmseg_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:out0", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:out1", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-segment2-load:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_segment2_load_pre_realized_body %cmp_lhs, %cmp_rhs, %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-segment2-load-unit-store", op_kind = "computed_masked_segment2_load_unit_store", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_cmseg_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-segment2-load-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-segment2-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_segment2_load_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_cmseg_load
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[OLD0:.*]] = tcrv_rvv.load
// REALIZED: %[[OLD1:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[FIELD0:.*]], %[[FIELD1:.*]] = tcrv_rvv.masked_segment2_load %{{.*}}, %[[MASK]], %[[OLD0]], %[[OLD1]], %[[VL]]
// REALIZED-SAME: field0_role = "segment-field0-output-buffer"
// REALIZED-SAME: field1_role = "segment-field1-output-buffer"
// REALIZED-SAME: inactive_lane_policy = "preserve-passthrough-on-false-lanes"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-SAME: source_memory_form = "segment2-interleaved-unit-stride-load"
// REALIZED: tcrv_rvv.store %{{.*}}, %[[FIELD0]], %[[VL]]
// REALIZED: tcrv_rvv.store %{{.*}}, %[[FIELD1]], %[[VL]]
// REALIZED-NOT: tcrv_rvv.masked_indexed_load
// REALIZED-NOT: tcrv_rvv.masked_indexed_store
// REALIZED-NOT: tcrv_rvv.strided_load
// REALIZED-NOT: tcrv_rvv.segment2_load
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.mask_load
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_computed_mask_segment2_load_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_segment2_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_segment2_load"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-segment2-load-unit-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src,out0,out1,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-compare-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.segment_memory_layout", value = "unit-stride-compare-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.segment_count", value = "2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_load_intrinsic", value = "__riscv_vlseg2e32_v_i32m1x2_tumu"}
// PLAN-SAME: {key = "tcrv_rvv.segment_store_intrinsic", value = "__riscv_vcreate_v_i32m1x2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_field_extract_intrinsic", value = "__riscv_vget_v_i32m1x2_i32m1"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_cmseg_load

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_cmseg_load
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,out0,out1,n
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: void tcrv_emitc_pre_realized_body_cmseg_load_kernel_pre_realized_body_rvv_cmseg_load(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
