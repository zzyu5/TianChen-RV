// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-segment2-interleave-plan-validated/s//provider_supported_mirror:rvv-script-derived-segment2-interleave/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PROVIDER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-segment2-memory-route-family-plan.v1/s//rvv-script-derived-plain-segment2-plan.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SEGMENT2-PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.destination_memory_form", value = "segment2-interleaved-unit-stride-store"/s//tcrv_rvv.destination_memory_form", value = "script-derived-segment2-destination"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEST-MEM

// Pre-realized selected-body input for one bounded Stage2 segment2 interleave
// memory movement slice. The RVV plugin must realize the field0 source, field1
// source, interleaved destination, and runtime n ABI operands into explicit
// load/load/segment2_store typed structure before the provider may construct
// the EmitC route.

module {
  tcrv.exec.kernel @pre_realized_body_segment2_interleave_unit_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_segment2_interleave_unit_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src0 = tcrv_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:src0", role = "segment-field0-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %src1 = tcrv_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:src1", role = "segment-field1-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:dst", role = "segment-interleaved-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-segment2-interleave-unit-load:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_segment2_interleave_memory_pre_realized_body %src0, %src1, %dst, %n {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", lmul = "m1", memory_form = "unit-load-segment2-store", op_kind = "segment2_interleave_unit_load", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, segment_count = 2 : i64, sew = 32 : i64, source0_memory_form = "unit-stride-load", source1_memory_form = "unit-stride-load"} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_segment2_interleave_unit_load {origin = "rvv-plugin", policy = "pre-realized-selected-body-segment2-interleave-unit-load-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-segment2-interleave-unit-load-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_segment2_interleave_memory_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl %{{.*}} {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_segment2_interleave_unit_load
// REALIZED: %[[FIELD0:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: %[[FIELD1:.*]] = tcrv_rvv.load
// REALIZED-SAME: !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
// REALIZED: tcrv_rvv.segment2_store
// REALIZED-SAME: destination_memory_form = "segment2-interleaved-unit-stride-store"
// REALIZED-SAME: field0_role = "segment-field0-input-buffer"
// REALIZED-SAME: field1_role = "segment-field1-input-buffer"
// REALIZED-SAME: segment_count = 2 : i64
// REALIZED-SAME: !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
// REALIZED-NOT: tcrv_rvv.segment2_load
// REALIZED-NOT: tcrv_rvv.indexed_store
// REALIZED-NOT: tcrv_rvv.strided_store
// REALIZED-NOT: tcrv_rvv.masked_move
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.typed_segment2_interleave_memory_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "segment2_interleave_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.segment2_store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-load-segment2-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "src0,src1,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:segment2_interleave_unit_load.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:segment2_interleave_unit_load.v1;src0=segment-field0-input-buffer:src0:abi|field0-load-base|field0-role|src0-mem|tuple-field0|hdr;src1=segment-field1-input-buffer:src1:abi|field1-load-base|field1-role|src1-mem|tuple-field1|hdr;dst=segment-interleaved-output-buffer:dst:abi|seg-store-base|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.segment2_memory_route_family_plan", value = "rvv-segment2-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-e32m1-segment2-interleave-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-segment2-interleave-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,field-inputs:signed-e32m1,segment2:vint32m1x2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_memory_layout", value = "dual-unit-stride-source-segment2-interleaved-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.segment_count", value = "2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_tuple_c_type", value = "vint32m1x2_t"}
// PLAN-SAME: {key = "tcrv_rvv.segment_store_intrinsic", value = "__riscv_vsseg2e32_v_i32m1x2"}
// PLAN-SAME: {key = "tcrv_rvv.segment_tuple_create_intrinsic", value = "__riscv_vcreate_v_i32m1x2"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "segment2-interleaved-unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.field0_role", value = "segment-field0-input-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.field1_role", value = "segment-field1-input-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.field0_source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.field1_source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_segment2_interleave_unit_load

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_segment2_interleave_unit_load
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_abi_order: src0,src1,dst,n
// HEADER: tianchenrv.rvv.memory_form: unit-load-segment2-store
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-e32m1-segment2-interleave-leaf-profile.v1
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-segment2-interleave-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:segment2_interleave_unit_load.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:segment2_interleave_unit_load.v1;src0=segment-field0-input-buffer:src0:abi|field0-load-base|field0-role|src0-mem|tuple-field0|hdr;src1=segment-field1-input-buffer:src1:abi|field1-load-base|field1-role|src1-mem|tuple-field1|hdr;dst=segment-interleaved-output-buffer:dst:abi|seg-store-base|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: tianchenrv.rvv.segment2_memory_route_family_plan: rvv-segment2-memory-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,field-inputs:signed-e32m1,segment2:vint32m1x2
// HEADER: void tcrv_emitc_pre_realized_body_segment2_interleave_unit_load_kernel_pre_realized_body_rvv_segment2_interleave_unit_load(const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);

// STALE-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-PROVIDER: candidate tcrv_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-segment2-interleave

// STALE-SEGMENT2-PLAN: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.segment2_memory_route_family_plan' must mirror provider route description value 'rvv-segment2-memory-route-family-plan.v1' but was 'rvv-script-derived-plain-segment2-plan.v1'

// STALE-DEST-MEM: candidate tcrv_rvv selected-body metadata key 'tcrv_rvv.destination_memory_form' must mirror provider route description value 'segment2-interleaved-unit-stride-store' but was 'script-derived-segment2-destination'
