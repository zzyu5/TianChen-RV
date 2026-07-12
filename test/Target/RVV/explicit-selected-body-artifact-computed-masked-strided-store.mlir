// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage2 computed
// mask plus runtime byte-strided masked destination-store slice. The selected
// RVV body structurally carries compare-produced mask, source payload,
// destination output buffer, runtime destination byte stride, and inactive-lane
// no-write policy through weft_rvv.masked_strided_store.

module {
  weft.exec.kernel @explicit_selected_body_computed_masked_strided_store_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_computed_masked_strided_store attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-store:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-store:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %src = weft_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-store:src", role = "source-input-buffer"} : !weft_rvv.runtime_abi_value
      %dst = weft_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-store:dst", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-store:n", role = "runtime-element-count"} : index
      %dst_stride_bytes = weft_rvv.runtime_abi_value {c_name = "dst_stride_bytes", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-computed-mask-strided-store:dst-stride-bytes", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_computed_masked_strided_store, sew = 32 : i64, source_kernel = "explicit_selected_body_computed_masked_strided_store_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = weft_rvv.load %cmp_lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %rhs_vec = weft_rvv.load %cmp_rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %src_vec = weft_rvv.load %src, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %mask = weft_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "slt"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.mask<i32, "m1">
        weft_rvv.masked_strided_store %dst, %mask, %src_vec, %dst_stride_bytes, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-strided-store", stride_unit = "byte"} : !weft_rvv.runtime_abi_value, !weft_rvv.mask<i32, "m1">, !weft_rvv.vector<i32, "m1">, index, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_computed_masked_strided_store {origin = "rvv-plugin", policy = "explicit-selected-body-computed-mask-strided-store-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-computed-mask-strided-store-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_strided_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_strided_store"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-unit-load-strided-store"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_strided_store.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_strided_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|cmp-rhs-call|hdr;src=source-input-buffer:src:abi|src-load|mstr-store-src-call|hdr;dst=output-buffer:dst:abi|mstr-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:abi|mstr-store-stride|byte|hdr"}
// PLAN-SAME: {key = "weft_rvv.masked_memory_layout", value = "unit-stride-compare-source-byte-strided-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-strided-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "masked-strided-store-has-no-passthrough-load"}
// PLAN-SAME: {key = "weft_rvv.destination_memory_form", value = "masked-strided-store"}
// PLAN-SAME: {key = "weft_rvv.strided_memory_layout", value = "unit-stride-compare-source-byte-strided-masked-destination-runtime-abi"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-strided-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_computed_masked_strided_store

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_computed_masked_strided_store
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-masked-strided-store-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,src,dst,n,dst_stride_bytes
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_strided_store.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_strided_store.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs-load|cmp-rhs-call|hdr;src=source-input-buffer:src:abi|src-load|mstr-store-src-call|hdr;dst=output-buffer:dst:abi|mstr-store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop-control|hdr;dst_stride_bytes=destination-byte-stride:dst_stride_bytes:abi|mstr-store-stride|byte|hdr
// HEADER: void weft_emitc_explicit_selected_body_computed_masked_strided_store_kernel_explicit_selected_body_rvv_computed_masked_strided_store(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src, int32_t *dst, size_t n, size_t dst_stride_bytes);
