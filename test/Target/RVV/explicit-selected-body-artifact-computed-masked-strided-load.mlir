// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 computed
// mask plus runtime byte-strided masked source-load slice. The selected RVV
// body structurally carries compare-produced mask, source input buffer,
// runtime source byte stride, old-destination passthrough for inactive lanes,
// loaded result, and unit-stride destination store through
// weft_rvv.masked_strided_load.

module {
  weft.exec.kernel @explicit_selected_body_computed_masked_strided_load_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_computed_masked_strided_load attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-load:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-load:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-load:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-load:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-load:n", role = "runtime-element-count"} : index
      %src_stride_bytes = weft_rvv.runtime_abi_value {c_name = "src_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-load:src-stride-bytes", role = "source-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_computed_masked_strided_load, sew = 32 : i64, source_kernel = "explicit_selected_body_computed_masked_strided_load_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_vec = weft_rvv.load %cmp_rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %old_dst = weft_rvv.load %dst, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask = weft_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        %loaded = weft_rvv.masked_strided_load %src, %mask, %old_dst, %src_stride_bytes, %vl {inactive_lane_policy = "preserve-passthrough-on-false-lanes", memory_form = "masked-strided-load", stride_unit = "byte"} : !weft_rvv.runtime_abi_value, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %dst, %loaded, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_computed_masked_strided_load {origin = "rvv-plugin", policy = "explicit-selected-body-computed-mask-strided-load-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-computed-mask-strided-load-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_strided_load_unit_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_strided_load"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-strided-load-unit-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src,dst,n,src_stride_bytes"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_strided_load_unit_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_strided_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src=source-input-buffer:src:abi|mstr-base|mstr-load-call|hdr;dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr;src_stride_bytes=source-byte-stride:src_stride_bytes:abi|mstr-stride|byte|hdr"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_route_family_plan", value = "rvv-computed-mask-memory-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.computed_mask_memory_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-e32m1-computed-mask-strided-load-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-strided-load-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-strided-load-store"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-compare-byte-strided-masked-source-old-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-old-destination"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "old-destination-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.source_memory_form", value = "masked-strided-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "weft_rvv.strided_memory_layout", value = "unit-stride-compare-byte-strided-masked-source-old-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.source_stride_source", value = "runtime_abi:src_stride_bytes"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-strided-load-unit-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_computed_masked_strided_load

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_computed_masked_strided_load
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-masked-strided-load-unit-store-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,dst,n,src_stride_bytes
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-e32m1-computed-mask-strided-load-leaf-profile.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-strided-load-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_strided_load_unit_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_strided_load_unit_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|rhs-call|hdr;src=source-input-buffer:src:abi|mstr-base|mstr-load-call|hdr;dst=output-buffer:dst:abi|old-dst-load|passthru-call|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr;src_stride_bytes=source-byte-stride:src_stride_bytes:abi|mstr-stride|byte|hdr
// HEADER: weft.rvv.computed_mask_memory_route_family_plan: rvv-computed-mask-memory-route-family-plan.v1
// HEADER: weft.rvv.computed_mask_memory_mask_producer_source: vector-compare-rhs-load
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,compare/source/passthrough:signed-e32m1,mask:b32,result:masked-strided-load-store
// HEADER: void weft_emitc_explicit_selected_body_computed_masked_strided_load_kernel_explicit_selected_body_rvv_computed_masked_strided_load(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *dst, size_t n, size_t src_stride_bytes);
