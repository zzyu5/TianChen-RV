// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 plain
// segment2 interleave memory movement slice. The selected RVV body
// structurally carries field0/field1 unit loads, a segment2 store, and runtime
// n/AVL through typed weft_rvv ops.

module {
  weft.exec.kernel @explicit_seg2_interleave_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_rvv_seg2_interleave attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %src0 = weft_rvv.runtime_abi_value {c_name = "src0", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-interleave-unit-load:src0", role = "segment-field0-input-buffer"} : !weft_rvv.runtime_abi_value
      %src1 = weft_rvv.runtime_abi_value {c_name = "src1", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-interleave-unit-load:src1", role = "segment-field1-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-interleave-unit-load:dst", role = "segment-interleaved-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-segment2-interleave-unit-load:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_rvv_seg2_interleave, sew = 32 : i64, source_kernel = "explicit_seg2_interleave_kernel", status = "selected-lowering-boundary"} {
        %field0 = weft_rvv.load %src0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %field1 = weft_rvv.load %src1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.segment2_store %dst, %field0, %field1, %vl {destination_memory_form = "segment2-interleaved-unit-stride-store", field0_role = "segment-field0-input-buffer", field1_role = "segment-field1-input-buffer", segment_count = 2 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_rvv_seg2_interleave {origin = "rvv-plugin", policy = "explicit-selected-body-segment2-interleave-unit-load-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-segment2-interleave-unit-load-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "segment2_interleave_unit_load"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.segment2_store"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-load-segment2-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "src0,src1,dst,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:segment2_interleave_unit_load.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:segment2_interleave_unit_load.v1;src0=segment-field0-input-buffer:src0:abi|field0-load-base|field0-role|src0-mem|tuple-field0|hdr;src1=segment-field1-input-buffer:src1:abi|field1-load-base|field1-role|src1-mem|tuple-field1|hdr;dst=segment-interleaved-output-buffer:dst:abi|seg-store-base|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.segment2_memory_route_family_plan", value = "rvv-segment2-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-segment2-interleave-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-segment2-interleave-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,field-inputs:signed-e32m1,segment2:vint32m1x2"}
// PLAN-SAME: {key = "weft_rvv.segment_memory_layout", value = "dual-unit-stride-source-segment2-interleaved-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.segment_count", value = "2"}
// PLAN-SAME: {key = "weft_rvv.segment_tuple_c_type", value = "vint32m1x2_t"}
// PLAN-SAME: {key = "weft_rvv.segment_store_intrinsic", value = "__riscv_vsseg2e32_v_i32m1x2"}
// PLAN-SAME: {key = "weft_rvv.segment_tuple_create_intrinsic", value = "__riscv_vcreate_v_i32m1x2"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "segment2-interleaved-unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.field0_role", value = "segment-field0-input-buffer"}
// PLAN-SAME: {key = "weft_rvv.field1_role", value = "segment-field1-input-buffer"}
// PLAN-SAME: {key = "weft_rvv.field0_source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "weft_rvv.field1_source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_rvv_seg2_interleave

// HEADER: weft.rvv.selected_variant: @explicit_rvv_seg2_interleave
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-segment2-interleave-unit-load-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: src0,src1,dst,n
// HEADER: weft.rvv.memory_form: unit-load-segment2-store
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-e32m1-segment2-interleave-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-segment2-interleave-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:segment2_interleave_unit_load.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:segment2_interleave_unit_load.v1;src0=segment-field0-input-buffer:src0:abi|field0-load-base|field0-role|src0-mem|tuple-field0|hdr;src1=segment-field1-input-buffer:src1:abi|field1-load-base|field1-role|src1-mem|tuple-field1|hdr;dst=segment-interleaved-output-buffer:dst:abi|seg-store-base|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.segment2_memory_route_family_plan: rvv-segment2-memory-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,field-inputs:signed-e32m1,segment2:vint32m1x2
// HEADER: void weft_emitc_explicit_seg2_interleave_kernel_explicit_rvv_seg2_interleave(const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);
