// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed 's/runtime-scalar-splat-compare-rhs/vector-compare-rhs-load/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRODUCER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed 's/rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr/rhs_scalar=lhs-input-buffer:rhs_scalar:abi|cmp-lhs-load|rhs-call|hdr/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RHS-BINDING
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed 's/out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr/out1=segment-field0-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/lhs,rhs_scalar,src,out0,out1,n/s//lhs,src,rhs_scalar,out0,out1,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/masked-off-lanes-preserve-old-destination/s//script-derived-passthrough/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-INACTIVE
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.field0_role", value = "segment-field0-output-buffer"/s//tcrv_rvv.field0_role", value = "segment-field1-output-buffer"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-FIELD0

// Hand-authored explicit selected-body input for one bounded Stage2 runtime
// scalar compare plus two-field masked segment2 load slice. The selected RVV
// body structurally carries the runtime scalar ABI value through tcrv_rvv.splat
// into compare-produced mask construction before tcrv_rvv.masked_segment2_load.

module {
  tcrv.exec.kernel @explicit_selected_body_rt_scalar_cmseg_load_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_rt_scalar_cmseg_load attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out0 = tcrv_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:out0", role = "segment-field0-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %out1 = tcrv_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:out1", role = "segment-field1-output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_rt_scalar_cmseg_load, sew = 32 : i64, source_kernel = "explicit_selected_body_rt_scalar_cmseg_load_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %old0 = tcrv_rvv.load %out0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %old1 = tcrv_rvv.load %out1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %field0, %field1 = tcrv_rvv.masked_segment2_load %src, %mask, %old0, %old1, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out0, %field0, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
        tcrv_rvv.store %out1, %field1, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_rt_scalar_cmseg_load {origin = "rvv-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_segment2_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_segment2_load"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-segment2-load-unit-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs_scalar,src,out0,out1,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1;lhs=lhs-input-buffer:lhs:abi|cmp-lhs-load|lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-segment2-load-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-load-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs/source/passthrough-fields:signed-e32m1,rhs_scalar:signed-scalar,mask:b32,segment2:vint32m1x2,result:runtime-scalar-masked-segment2-load-store"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.segment_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_rt_scalar_cmseg_load

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_rt_scalar_cmseg_load
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs_scalar,src,out0,out1,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.mask_role: predicate-mask-produced-by-compare
// HEADER: tianchenrv.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: tianchenrv.rvv.mask_memory_form: compare-produced-mask
// HEADER: tianchenrv.rvv.inactive_lane_contract: masked-off-lanes-preserve-old-destination
// HEADER: tianchenrv.rvv.masked_passthrough_layout: old-destination-vector-preserves-inactive-lanes
// HEADER: tianchenrv.rvv.masked_memory_layout: unit-stride-lhs-runtime-scalar-threshold-segment2-masked-source-old-fields-destination-runtime-abi
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-segment2-load-leaf-profile.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-load-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1;lhs=lhs-input-buffer:lhs:abi|cmp-lhs-load|lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: tianchenrv.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: void tcrv_emitc_explicit_selected_body_rt_scalar_cmseg_load_kernel_explicit_selected_body_rvv_rt_scalar_cmseg_load(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, int32_t *out0, int32_t *out1, size_t n);

// STALE-PRODUCER: candidate tcrv_rvv.computed_mask_memory_mask_producer_source provenance must mirror selected typed RVV computed-mask segment2 producer source 'runtime-scalar-splat-compare-rhs' but was 'vector-compare-rhs-load'
// STALE-RHS-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-RHS-BINDING: candidate tcrv_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary
// STALE-RHS-BINDING-SAME: rhs_scalar=rhs-scalar-value:rhs_scalar:abi
// STALE-RHS-BINDING-SAME: splat
// STALE-RHS-BINDING-SAME: but was
// STALE-RHS-BINDING-SAME: rhs_scalar=lhs-input-buffer:rhs_scalar:abi
// STALE-RHS-BINDING-SAME: cmp-lhs-load
// STALE-BINDING: candidate tcrv_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary
// STALE-BINDING-SAME: out1=segment-field1-output-buffer:out1:abi
// STALE-BINDING-SAME: but was
// STALE-BINDING-SAME: out1=segment-field0-output-buffer:out1:abi
// STALE-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-ABI: candidate tcrv_rvv.runtime_abi_order provenance must mirror route-local runtime AVL/VL ABI order mirror
// STALE-ABI-SAME: lhs,rhs_scalar,src,out0,out1,n
// STALE-ABI-SAME: but was
// STALE-ABI-SAME: lhs,src,rhs_scalar,out0,out1,n
// STALE-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-HEADER: candidate tcrv_rvv.required_header_declarations provenance must mirror selected typed RVV computed-mask segment2 route header requirements
// STALE-HEADER-SAME: stddef.h,stdint.h,riscv_vector.h
// STALE-HEADER-SAME: but was
// STALE-HEADER-SAME: stddef.h,stdint.h
// STALE-INACTIVE: RVV materialized EmitC target artifact bridge failed
// STALE-INACTIVE: candidate tcrv_rvv.inactive_lane_contract provenance must mirror selected typed RVV computed-mask segment2 inactive lane contract
// STALE-INACTIVE-SAME: masked-off-lanes-preserve-old-destination
// STALE-INACTIVE-SAME: but was
// STALE-INACTIVE-SAME: script-derived-passthrough
// STALE-FIELD0: RVV materialized EmitC target artifact bridge failed
// STALE-FIELD0: candidate tcrv_rvv.field0_role provenance must mirror selected typed RVV computed-mask segment2 field0 role
// STALE-FIELD0-SAME: segment-field0-output-buffer
// STALE-FIELD0-SAME: but was
// STALE-FIELD0-SAME: segment-field1-output-buffer
