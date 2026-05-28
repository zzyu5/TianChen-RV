// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  tcrv.exec.kernel @pre_realized_body_computed_masked_macc_add_m2_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @pre_realized_body_rvv_computed_masked_macc_add_m2 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %cmp_rhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:cmp_rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:lhs-payload", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:rhs-payload", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc-m2:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_computed_mask_macc_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m2", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-macc", op_kind = "computed_masked_macc_add", predicate_kind = "slt", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @pre_realized_body_rvv_computed_masked_macc_add_m2 {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-macc-m2-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-macc-m2-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_computed_mask_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: sew = 32
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_masked_macc_add_m2
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m2">
// REALIZED: %[[CMP_RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m2">
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m2">
// REALIZED: %[[RHS:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m2">
// REALIZED: %[[ACC:.*]] = tcrv_rvv.load
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m2">
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED-SAME: -> !tcrv_rvv.mask<i32, "m2">
// REALIZED: %[[SUM:.*]] = tcrv_rvv.masked_macc %[[MASK]], %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: mask_memory_form = "compare-produced-mask"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED-SAME: -> !tcrv_rvv.vector<i32, "m2">
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.macc
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.select
// REALIZED-NOT: tcrv_rvv.masked_store
// REALIZED-NOT: tcrv_rvv.masked_load
// REALIZED-NOT: tcrv_rvv.typed_computed_mask_macc_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_macc"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "computed-mask-unit-stride-macc"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-SAME: {key = "tcrv_rvv.target_capability_legality_mirror", value = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m2;tail=agnostic;mask=agnostic"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_macc_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_route_family_plan", value = "rvv-computed-mask-accumulation-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_compute_suffix", value = "vector-masked-macc-add"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_accumulator_contract", value = "vector-accumulator-input-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_result_contract", value = "vector-macc-result-stored-to-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-computed-mask-macc-add-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:typed-vector,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-macc-false-lanes-preserve-accumulator"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "accumulator-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_masked_macc_add_m2

// HEADER: tianchenrv.rvv.selected_variant: @pre_realized_body_rvv_computed_masked_macc_add_m2
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-computed-masked-macc-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1
// HEADER: tianchenrv.rvv.runtime_avl_source: runtime_abi:n
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: slt
// HEADER: tianchenrv.rvv.inactive_lane_contract: masked-macc-false-lanes-preserve-accumulator
// HEADER: tianchenrv.rvv.masked_passthrough_layout: accumulator-vector-preserves-inactive-lanes
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-computed-mask-macc-add-leaf-profile.v1
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated
// HEADER: tianchenrv.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact
// HEADER: tianchenrv.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m2;tail=agnostic;mask=agnostic
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_macc_add.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: tianchenrv.rvv.accumulation_route_family_plan: rvv-computed-mask-accumulation-route-family-plan.v1
// HEADER: tianchenrv.rvv.accumulation_compute_suffix: vector-masked-macc-add
// HEADER: tianchenrv.rvv.accumulation_mask_producer_source: vector-compare-rhs-load
// HEADER: tianchenrv.rvv.accumulation_accumulator_contract: vector-accumulator-input-preserves-inactive-lanes
// HEADER: tianchenrv.rvv.accumulation_result_contract: vector-macc-result-stored-to-output-buffer
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:typed-vector,mask:typed-mask,result:typed-vector
// HEADER: void tcrv_emitc_pre_realized_body_computed_masked_macc_add_m2_kernel_pre_realized_body_rvv_computed_masked_macc_add_m2(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);
