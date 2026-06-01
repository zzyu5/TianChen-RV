// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized selected-body input for one bounded Stage2 segment2
// deinterleave memory movement slice. The RVV plugin must realize the
// interleaved source, field0 output, field1 output, and runtime n ABI operands
// into explicit segment2_load/move/move/store/store typed structure before the
// provider may construct the EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_segment2_deinterleave_unit_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_segment2_deinterleave_unit_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:src", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:out0", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:out1", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-deinterleave-unit-store:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body %src, %out0, %out1, %n {destination_memory_form = "unit-stride-store", field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", lmul = "m1", memory_form = "segment2-load-unit-store", op_kind = "segment2_deinterleave_unit_store", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_segment2_deinterleave_unit_store {origin = "rvv-plugin", policy = "pre-realized-selected-body-segment2-deinterleave-unit-store-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-segment2-deinterleave-unit-store-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_segment2_deinterleave_unit_store
// REALIZED: %[[FIELD0:.*]], %[[FIELD1:.*]] = tcrv_rvv.segment2_load
// REALIZED-SAME: field0_role = "segment-field0-output-buffer"
// REALIZED-SAME: field1_role = "segment-field1-output-buffer"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-SAME: source_memory_form = "segment2-interleaved-unit-stride-load"
// REALIZED: %[[MOVED0:.*]] = tcrv_rvv.move %[[FIELD0]], %[[VL]]
// REALIZED-SAME: kind = "copy"
// REALIZED: %[[MOVED1:.*]] = tcrv_rvv.move %[[FIELD1]], %[[VL]]
// REALIZED-SAME: kind = "copy"
// REALIZED: tcrv_rvv.store
// REALIZED-SAME: !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
// REALIZED: tcrv_rvv.store
// REALIZED-SAME: !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
// REALIZED-NOT: tcrv_rvv.indexed_store
// REALIZED-NOT: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "segment2_deinterleave_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.move"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "segment2-load-unit-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src,out0,out1,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:segment2_deinterleave_unit_store.v1;src=lhs-input-buffer:src:abi|seg-load-base|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|field0-store-base|field0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|field1-store-base|field1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.segment2_memory_route_family_plan", value = "rvv-segment2-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-segment2-deinterleave-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-segment2-deinterleave-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,segment2:vint32m1x2,field-outputs:signed-e32m1"}
// PLAN-SAME: {key = "tcrv_rvv.segment_memory_layout", value = "segment2-interleaved-source-dual-unit-stride-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.segment_count", value = "2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_tuple_c_type", value = "vint32m1x2_t"}
// PLAN-SAME: {key = "tcrv_rvv.segment_load_intrinsic", value = "__riscv_vlseg2e32_v_i32m1x2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_field_extract_intrinsic", value = "__riscv_vget_v_i32m1x2_i32m1"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "segment2-interleaved-unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.field0_role", value = "segment-field0-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.field1_role", value = "segment-field1-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.field0_destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.field1_destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-segment2-deinterleave-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_segment2_deinterleave_unit_store

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_segment2_deinterleave_unit_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-segment2-deinterleave-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: src,out0,out1,n
// HEADER: tianchenrv.rvv.memory_form: segment2-load-unit-store
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-segment2-deinterleave-leaf-profile.v1
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-segment2-deinterleave-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:segment2_deinterleave_unit_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:segment2_deinterleave_unit_store.v1;src=lhs-input-buffer:src:abi|seg-load-base|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|field0-store-base|field0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|field1-store-base|field1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: tianchenrv.rvv.segment2_memory_route_family_plan: rvv-segment2-memory-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,segment2:vint32m1x2,field-outputs:signed-e32m1
// HEADER: void tcrv_emitc_pre_realized_body_segment2_deinterleave_unit_store_kernel_pre_realized_body_rvv_segment2_deinterleave_unit_store(const int32_t *src, int32_t *out0, int32_t *out1, size_t n);
