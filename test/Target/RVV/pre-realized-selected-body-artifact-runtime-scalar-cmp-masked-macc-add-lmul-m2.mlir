// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pr_rt_scalar_masked_macc_m2_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @rvv_pr_rt_scalar_masked_macc_m2 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2:rhs_scalar", role = "rhs-scalar-value"} : i32
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2:lhs-payload", role = "dot-lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2:rhs-payload", role = "dot-rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2:n", role = "runtime-element-count"} : index
      weft_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body %cmp_lhs, %rhs_scalar, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-macc", op_kind = "runtime_scalar_cmp_masked_macc_add", predicate_kind = "sle", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @rvv_pr_rt_scalar_masked_macc_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: sew = 32
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: selected_variant = @rvv_pr_rt_scalar_masked_macc_m2
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[PAYLOAD_RHS:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED-SAME: -> !weft_rvv.mask<i32, "m2">
// REALIZED: %[[SUM:.*]] = weft_rvv.masked_macc %[[MASK]], %[[LHS]], %[[PAYLOAD_RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: mask_memory_form = "compare-produced-mask"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED-SAME: -> !weft_rvv.vector<i32, "m2">
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.macc
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.select
// REALIZED-NOT: weft_rvv.masked_store
// REALIZED-NOT: weft_rvv.masked_load
// REALIZED-NOT: weft_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_macc"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.element_type", value = "i32"}
// PLAN-SAME: {key = "weft_rvv.sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m2"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "runtime-scalar-computed-mask-unit-stride-macc"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.accumulation_route_family_plan", value = "rvv-computed-mask-accumulation-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.accumulation_compute_suffix", value = "vector-masked-macc-add"}
// PLAN-SAME: {key = "weft_rvv.accumulation_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "weft_rvv.accumulation_accumulator_contract", value = "vector-accumulator-input-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.accumulation_result_contract", value = "vector-macc-result-stored-to-output-buffer"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m2"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-macc-add-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs/lhs/rhs/acc:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-macc-false-lanes-preserve-accumulator"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "accumulator-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "weft_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pr_rt_scalar_masked_macc_m2

// HEADER: weft.rvv.selected_variant: @rvv_pr_rt_scalar_masked_macc_m2
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi.v1
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.element_type: i32
// HEADER: weft.rvv.sew: 32
// HEADER: weft.rvv.lmul: m2
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n
// HEADER: weft.rvv.compare_predicate_kind: sle
// HEADER: weft.rvv.mask_role: predicate-mask-produced-by-compare
// HEADER: weft.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: weft.rvv.mask_memory_form: compare-produced-mask
// HEADER: weft.rvv.inactive_lane_contract: masked-macc-false-lanes-preserve-accumulator
// HEADER: weft.rvv.masked_passthrough_layout: accumulator-vector-preserves-inactive-lanes
// HEADER: weft.rvv.macc_accumulator_layout: separate-i32-vector-accumulator-input
// HEADER: weft.rvv.macc_result_layout: store-multiply-accumulate-result-to-output-buffer
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-macc-add-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.accumulation_route_family_plan: rvv-computed-mask-accumulation-route-family-plan.v1
// HEADER: weft.rvv.accumulation_compute_suffix: vector-masked-macc-add
// HEADER: weft.rvv.accumulation_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: weft.rvv.accumulation_accumulator_contract: vector-accumulator-input-preserves-inactive-lanes
// HEADER: weft.rvv.accumulation_result_contract: vector-macc-result-stored-to-output-buffer
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,cmp_lhs/lhs/rhs/acc:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:typed-vector
// HEADER: void weft_emitc_pr_rt_scalar_masked_macc_m2_kernel_rvv_pr_rt_scalar_masked_macc_m2(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);
