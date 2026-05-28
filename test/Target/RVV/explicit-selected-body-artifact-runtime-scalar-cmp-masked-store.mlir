// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @explicit_body_runtime_scalar_cmp_masked_store_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_body_rvv_runtime_scalar_cmp_masked_store attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-store:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-store:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-store:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %dst = tcrv_rvv.runtime_abi_value {c_name = "dst", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-store:dst", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-cmp-masked-store:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_body_rvv_runtime_scalar_cmp_masked_store, sew = 32 : i64, source_kernel = "explicit_body_runtime_scalar_cmp_masked_store_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %payload = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        tcrv_rvv.masked_store %dst, %mask, %payload, %vl {inactive_lane_policy = "preserve-output-on-false-lanes", memory_form = "masked-unit-store"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_body_rvv_runtime_scalar_cmp_masked_store {origin = "rvv-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-store-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-runtime-scalar-cmp-masked-store-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_store"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_store"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-undisturbed-mask-undisturbed.v1"}
// PLAN-SAME: {key = "tcrv_rvv.tail_policy", value = "undisturbed"}
// PLAN-SAME: {key = "tcrv_rvv.mask_policy", value = "undisturbed"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-computed-mask-store"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs_scalar,src,dst,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|cmp-lhs|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;src=source-input-buffer:src:abi|src-load|mstore-src|hdr;dst=output-buffer:dst:abi|mstore-base|mstore-dst|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-store-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-store-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,lhs_payload:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,dst:masked-store"}
// PLAN-SAME: {key = "tcrv_rvv.masked_memory_layout", value = "unit-stride-lhs-runtime-scalar-threshold-source-masked-destination-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-store-false-lanes-preserve-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "masked-store-has-no-passthrough-load"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "masked-unit-store"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_body_rvv_runtime_scalar_cmp_masked_store

// HEADER: tianchenrv.rvv.selected_variant: @explicit_body_rvv_runtime_scalar_cmp_masked_store
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-store-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: lhs,rhs_scalar,src,dst,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1;lhs=lhs-input-buffer:lhs:abi|lhs-load|cmp-lhs|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;src=source-input-buffer:src:abi|src-load|mstore-src|hdr;dst=output-buffer:dst:abi|mstore-base|mstore-dst|hdr;n=runtime-element-count:n:abi|setvl|loop|hdr
// HEADER: void tcrv_emitc_explicit_body_runtime_scalar_cmp_masked_store_kernel_explicit_body_rvv_runtime_scalar_cmp_masked_store(const int32_t *lhs, int32_t rhs_scalar, const int32_t *src, int32_t *dst, size_t n);
