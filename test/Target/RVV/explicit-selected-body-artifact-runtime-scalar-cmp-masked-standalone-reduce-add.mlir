// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @rt_scalar_cm_standalone_reduce_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_rt_scalar_cm_standalone_reduce attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add:rhs_scalar", role = "rhs-scalar-value"} : i32
      %src = tcrv_rvv.runtime_abi_value {c_name = "src", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add:src", role = "source-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add:seed", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add:scalar-output", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @rvv_rt_scalar_cm_standalone_reduce, sew = 32 : i64, source_kernel = "rt_scalar_cm_standalone_reduce_kernel", status = "selected-lowering-boundary"} {
        %lhs_vec = tcrv_rvv.load %cmp_lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %rhs_vec = tcrv_rvv.splat %rhs_scalar, %vl : i32, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %src_vec = tcrv_rvv.load %src, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %mask = tcrv_rvv.compare %lhs_vec, %rhs_vec, %vl {kind = "sle"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.mask<i32, "m1">
        %reduced = tcrv_rvv.masked_standalone_reduce %mask, %src_vec, %acc, %vl {accumulator_layout = "scalar-i32-seed-lane0-from-accumulator-input", kind = "add", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", result_layout = "store-standalone-reduction-lane0-to-output-scalar"} : !tcrv_rvv.mask<i32, "m1">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %reduced, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_rt_scalar_cm_standalone_reduce {origin = "rvv-plugin", policy = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-runtime-scalar-computed-mask-standalone-reduce-add-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_standalone_reduce_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_standalone_reduce"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-computed-mask-unit-stride-standalone-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,rhs_scalar,src,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs-call|hdr;src=source-input-buffer:src:abi|src-load|masked-reduce-input|zero-inactive|hdr;acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc;out=output-buffer:out:abi|acc-state|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_route_family_plan", value = "rvv-computed-mask-accumulation-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_compute_suffix", value = "scalar-horizontal-masked-standalone-reduce-add"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_accumulator_contract", value = "scalar-seed-input-feeds-masked-horizontal-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_result_contract", value = "scalar-horizontal-reduction-lane0-stored-to-output"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_scalar_carry_contract", value = "scalar-result-carries-across-runtime-vl-chunks"}
// PLAN-SAME: {key = "tcrv_rvv.standalone_reduction_route_family_plan", value = "rvv-standalone-reduction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-standalone-reduction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs/source:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,seed:typed-scalar,result:typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_zeroing_requirement", value = "masked-standalone-reduction-zero-inactive-lanes-before-reduction"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_accumulator_layout", value = "scalar-i32-seed-lane0-from-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.reduction_result_layout", value = "store-standalone-reduction-lane0-to-output-scalar"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_rt_scalar_cm_standalone_reduce

// HEADER: tianchenrv.rvv.selected_variant: @rvv_rt_scalar_cm_standalone_reduce
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-standalone-reduce-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,rhs_scalar,src,acc,out,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_standalone_reduce_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs-load|cmp-lhs-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs-call|hdr;src=source-input-buffer:src:abi|src-load|masked-reduce-input|zero-inactive|hdr;acc=accumulator-input-buffer:acc:abi|initial-seed|acc-state|masked-reduce-acc;out=output-buffer:out:abi|acc-state|store-base|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: tianchenrv.rvv.standalone_reduction_route_family_plan: rvv-standalone-reduction-route-family-plan.v1
// HEADER: tianchenrv.rvv.accumulation_route_family_plan: rvv-computed-mask-accumulation-route-family-plan.v1
// HEADER: tianchenrv.rvv.accumulation_compute_suffix: scalar-horizontal-masked-standalone-reduce-add
// HEADER: tianchenrv.rvv.accumulation_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: tianchenrv.rvv.accumulation_accumulator_contract: scalar-seed-input-feeds-masked-horizontal-reduction
// HEADER: tianchenrv.rvv.accumulation_result_contract: scalar-horizontal-reduction-lane0-stored-to-output
// HEADER: tianchenrv.rvv.accumulation_scalar_carry_contract: scalar-result-carries-across-runtime-vl-chunks
// HEADER: void tcrv_emitc_rt_scalar_cm_standalone_reduce_kernel_rvv_rt_scalar_cm_standalone_reduce(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *src, const int32_t *acc, int32_t *out, size_t n);
