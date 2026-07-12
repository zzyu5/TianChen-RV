// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed 's/runtime-scalar-splat-compare-rhs/vector-compare-rhs-load/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PRODUCER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed 's/rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr/rhs_scalar=lhs-input-buffer:rhs_scalar:abi|cmp-lhs-load|rhs-call|hdr/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RHS-BINDING
// RUN: weft-opt %s --weft-materialize-emission-plans | sed 's/out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr/out1=segment-field0-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/lhs,rhs_scalar,src,out0,out1,n/s//lhs,src,rhs_scalar,out0,out1,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/masked-off-lanes-preserve-old-destination/s//script-derived-passthrough/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-INACTIVE
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.field0_role", value = "segment-field0-output-buffer"/s//weft_rvv.field0_role", value = "segment-field1-output-buffer"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-FIELD0

// Hand-authored explicit selected-body input for one bounded Stage2 runtime
// scalar compare plus two-field masked segment2 load slice. The selected RVV
// body structurally carries the runtime scalar ABI value through weft_rvv.splat
// into compare-produced mask construction before weft_rvv.masked_segment2_load.

module {
  weft.exec.kernel @explicit_selected_body_rt_scalar_cmseg_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_rt_scalar_cmseg_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %out0 = weft_rvv.runtime_abi_value {c_name = "out0", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:out0", role = "segment-field0-output-buffer"} : !weft_rvv.runtime_abi_value
      %out1 = weft_rvv.runtime_abi_value {c_name = "out1", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:out1", role = "segment-field1-output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_rt_scalar_cmseg_load, sew = 32 : i64, source_kernel = "explicit_selected_body_rt_scalar_cmseg_load_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_vec = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old0 = weft_rvv.load %out0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old1 = weft_rvv.load %out1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask = weft_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %field0, %field1 = weft_rvv.masked_segment2_load %src, %mask, %old0, %old1, %vl {field0_role = "segment-field0-output-buffer", field1_role = "segment-field1-output-buffer", inactive_lane_policy = "preserve-passthrough-on-false-lanes", segment_count = 2 : i64, source_memory_form = "segment2-interleaved-unit-stride-load"} : !weft_rvv.runtime_abi_value, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out0, %field0, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
        weft_rvv.store %out1, %field1, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_rt_scalar_cmseg_load {origin = "rvv-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-segment2-load-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_segment2_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_segment2_load"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-segment2-load-unit-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,rhs_scalar,src,out0,out1,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1;lhs=lhs-input-buffer:lhs:abi|cmp-lhs-load|lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-segment2-load-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-load-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,lhs/source/passthrough-fields:signed-e32m1,rhs_scalar:signed-scalar,mask:b32,segment2:vint32m1x2,result:runtime-scalar-masked-segment2-load-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.segment_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-segment2-masked-source-old-fields-destination-runtime-abi"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_rt_scalar_cmseg_load

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_rt_scalar_cmseg_load
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-segment2-load-unit-store-callable-c-abi.v1
// HEADER: weft.rvv.runtime_abi_order: lhs,rhs_scalar,src,out0,out1,n
// HEADER: weft.rvv.compare_predicate_kind: sle
// HEADER: weft.rvv.mask_role: predicate-mask-produced-by-compare
// HEADER: weft.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: weft.rvv.mask_memory_form: compare-produced-mask
// HEADER: weft.rvv.inactive_lane_contract: masked-off-lanes-preserve-old-destination
// HEADER: weft.rvv.masked_passthrough_layout: old-destination-vector-preserves-inactive-lanes
// HEADER: weft.rvv.masked_memory_layout: unit-stride-lhs-runtime-scalar-threshold-segment2-masked-source-old-fields-destination-runtime-abi
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-segment2-load-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-segment2-load-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1;lhs=lhs-input-buffer:lhs:abi|cmp-lhs-load|lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|rhs-call|hdr;src=source-input-buffer:src:abi|mseg-base|mseg-call|src-mem|hdr;out0=segment-field0-output-buffer:out0:abi|old0-load|f0-pass|f0-store|f0-role|dst-mem|hdr;out1=segment-field1-output-buffer:out1:abi|old1-load|f1-pass|f1-store|f1-role|dst-mem|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr
// HEADER: weft.rvv.computed_mask_memory_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: void weft_emitc_explicit_selected_body_rt_scalar_cmseg_load_kernel_explicit_selected_body_rvv_rt_scalar_cmseg_load(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, int32_t *out0, int32_t *out1, size_t n);

// STALE-PRODUCER: metadata key '{{.*}}computed_mask_memory_mask_producer_source'{{.*}}'runtime-scalar-splat-compare-rhs' but was 'vector-compare-rhs-load'
// STALE-RHS-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-RHS-BINDING: candidate weft_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary
// STALE-RHS-BINDING-SAME: rhs_scalar=rhs-scalar-value:rhs_scalar:abi
// STALE-RHS-BINDING-SAME: splat
// STALE-RHS-BINDING-SAME: but was
// STALE-RHS-BINDING-SAME: rhs_scalar=lhs-input-buffer:rhs_scalar:abi
// STALE-RHS-BINDING-SAME: cmp-lhs-load
// STALE-BINDING: candidate weft_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary
// STALE-BINDING-SAME: out1=segment-field1-output-buffer:out1:abi
// STALE-BINDING-SAME: but was
// STALE-BINDING-SAME: out1=segment-field0-output-buffer:out1:abi
// STALE-ABI: metadata key '{{.*}}runtime_abi_order'{{.*}}'lhs,rhs_scalar,src,out0,out1,n' but was 'lhs,src,rhs_scalar,out0,out1,n'
// STALE-HEADER: metadata key '{{.*}}required_header_declarations'{{.*}}'stddef.h,stdint.h,riscv_vector.h' but was 'stddef.h,stdint.h'
// STALE-INACTIVE: metadata key '{{.*}}inactive_lane_contract'{{.*}}'masked-off-lanes-preserve-old-destination' but was 'script-derived-passthrough'
// STALE-FIELD0: metadata key '{{.*}}field0_role'{{.*}}'segment-field0-output-buffer' but was 'segment-field1-output-buffer'
