// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-validated/s//provider_supported_mirror:rvv-script-derived-runtime-scalar-macc/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-PROVIDER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1/s//rvv-route-operand-binding:script-derived-runtime-scalar-macc.v1/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-BINDING
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr/s//rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs|hdr/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-BINDING-SUMMARY
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n/s//cmp_lhs,lhs,rhs_scalar,rhs,acc,out,n/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-ABI
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.compare_predicate_kind", value = "sle"/s//tcrv_rvv.compare_predicate_kind", value = "slt"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-PREDICATE
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-HEADER
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-TYPE
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/runtime-scalar-splat-compare-rhs/s//script-derived-runtime-scalar-mask-producer/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-SCALAR
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"/s//tcrv_rvv.mask_source", value = "script-derived-mask-source"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-MASK-SOURCE
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.mask_memory_form", value = "compare-produced-mask"/s//tcrv_rvv.mask_memory_form", value = "script-derived-mask-form"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-MASK-FORM
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.inactive_lane_contract", value = "masked-macc-false-lanes-preserve-accumulator"/s//tcrv_rvv.inactive_lane_contract", value = "script-derived-inactive-lanes"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-INACTIVE
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.masked_passthrough_layout", value = "accumulator-vector-preserves-inactive-lanes"/s//tcrv_rvv.masked_passthrough_layout", value = "script-derived-passthrough"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-PASSTHROUGH
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"/s//tcrv_rvv.macc_accumulator_layout", value = "script-derived-accumulator-layout"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-ACC-LAYOUT
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"/s//tcrv_rvv.macc_result_layout", value = "script-derived-result-layout"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-LAYOUT
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.source_memory_form", value = "unit-stride-load"/s//tcrv_rvv.source_memory_form", value = "script-derived-source-memory"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RT-MACC-SOURCE-MEMORY

module {
  tcrv.exec.kernel @pr_rt_scalar_masked_macc_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @rvv_pr_rt_scalar_masked_macc attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = tcrv_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc:cmp_lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs_scalar = tcrv_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc:rhs_scalar", role = "rhs-scalar-value"} : i32
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc:lhs-payload", role = "dot-lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc:rhs-payload", role = "dot-rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc:accumulator", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-runtime-scalar-computed-mask-macc:n", role = "runtime-element-count"} : index
      tcrv_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body %cmp_lhs, %rhs_scalar, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "runtime-scalar-computed-mask-unit-stride-macc", op_kind = "runtime_scalar_cmp_masked_macc_add", predicate_kind = "sle", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!tcrv_rvv.runtime_abi_value, i32, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index) -> ()
    }
    tcrv.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_pr_rt_scalar_masked_macc {origin = "rvv-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-case"}
      tcrv.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-runtime-scalar-computed-mask-macc-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = tcrv_rvv.setvl
// REALIZED: tcrv_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @rvv_pr_rt_scalar_masked_macc
// REALIZED: %[[CMP_LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[RHS:.*]] = tcrv_rvv.splat
// REALIZED: %[[LHS:.*]] = tcrv_rvv.load
// REALIZED: %[[PAYLOAD_RHS:.*]] = tcrv_rvv.load
// REALIZED: %[[ACC:.*]] = tcrv_rvv.load
// REALIZED: %[[MASK:.*]] = tcrv_rvv.compare %[[CMP_LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "sle"
// REALIZED: %[[SUM:.*]] = tcrv_rvv.masked_macc %[[MASK]], %[[LHS]], %[[PAYLOAD_RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: mask_memory_form = "compare-produced-mask"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED: tcrv_rvv.store
// REALIZED-NOT: tcrv_rvv.macc
// REALIZED-NOT: tcrv_rvv.binary
// REALIZED-NOT: tcrv_rvv.select
// REALIZED-NOT: tcrv_rvv.masked_store
// REALIZED-NOT: tcrv_rvv.masked_load
// REALIZED-NOT: tcrv_rvv.typed_runtime_scalar_computed_mask_macc_pre_realized_body

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "runtime_scalar_cmp_masked_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.masked_macc"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.compare_predicate_kind", value = "sle"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "runtime-scalar-computed-mask-unit-stride-macc"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_route_family_plan", value = "rvv-computed-mask-accumulation-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_compute_suffix", value = "vector-masked-macc-add"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_mask_producer_source", value = "runtime-scalar-splat-compare-rhs"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_accumulator_contract", value = "vector-accumulator-input-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.accumulation_result_contract", value = "vector-macc-result-stored-to-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-typed-runtime-scalar-cmp-masked-macc-add-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs/lhs/rhs/acc:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "tcrv_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "tcrv_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "tcrv_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "tcrv_rvv.inactive_lane_contract", value = "masked-macc-false-lanes-preserve-accumulator"}
// PLAN-SAME: {key = "tcrv_rvv.masked_passthrough_layout", value = "accumulator-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "tcrv_rvv.source_memory_form", value = "unit-stride-load"}
// PLAN-SAME: {key = "tcrv_rvv.destination_memory_form", value = "unit-stride-store"}
// PLAN-SAME: {key = "tcrv_rvv.indexed_memory_layout", value = "unit-stride-compare-lhs-runtime-scalar-threshold-lhs-rhs-accumulator-masked-macc-output-runtime-abi"}
// PLAN-SAME: {key = "tcrv_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @rvv_pr_rt_scalar_masked_macc

// HEADER: tianchenrv.rvv.selected_variant: @rvv_pr_rt_scalar_masked_macc
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-runtime-scalar-cmp-masked-macc-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.runtime_abi_order: cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n
// HEADER: tianchenrv.rvv.compare_predicate_kind: sle
// HEADER: tianchenrv.rvv.source_memory_form: unit-stride-load
// HEADER: tianchenrv.rvv.destination_memory_form: unit-stride-store
// HEADER: tianchenrv.rvv.mask_role: predicate-mask-produced-by-compare
// HEADER: tianchenrv.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: tianchenrv.rvv.mask_memory_form: compare-produced-mask
// HEADER: tianchenrv.rvv.inactive_lane_contract: masked-macc-false-lanes-preserve-accumulator
// HEADER: tianchenrv.rvv.masked_passthrough_layout: accumulator-vector-preserves-inactive-lanes
// HEADER: tianchenrv.rvv.indexed_memory_layout: unit-stride-compare-lhs-runtime-scalar-threshold-lhs-rhs-accumulator-masked-macc-output-runtime-abi
// HEADER: tianchenrv.rvv.macc_accumulator_layout: separate-i32-vector-accumulator-input
// HEADER: tianchenrv.rvv.macc_result_layout: store-multiply-accumulate-result-to-output-buffer
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-typed-runtime-scalar-cmp-masked-macc-add-leaf-profile.v1
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: tianchenrv.rvv.accumulation_route_family_plan: rvv-computed-mask-accumulation-route-family-plan.v1
// HEADER: tianchenrv.rvv.accumulation_compute_suffix: vector-masked-macc-add
// HEADER: tianchenrv.rvv.accumulation_mask_producer_source: runtime-scalar-splat-compare-rhs
// HEADER: tianchenrv.rvv.accumulation_accumulator_contract: vector-accumulator-input-preserves-inactive-lanes
// HEADER: tianchenrv.rvv.accumulation_result_contract: vector-macc-result-stored-to-output-buffer
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,cmp_lhs/lhs/rhs/acc:typed-vector,rhs_scalar:typed-scalar,mask:typed-mask,result:typed-vector
// HEADER: void tcrv_emitc_pr_rt_scalar_masked_macc_kernel_rvv_pr_rt_scalar_masked_macc(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);

// STALE-RT-MACC-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-PROVIDER: candidate tcrv_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-RT-MACC-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-runtime-scalar-macc

// STALE-RT-MACC-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-BINDING: candidate tcrv_rvv.route_operand_binding_plan provenance must mirror selected typed RVV body binding plan
// STALE-RT-MACC-BINDING-SAME: rvv-route-operand-binding:script-derived-runtime-scalar-macc.v1

// STALE-RT-MACC-BINDING-SUMMARY: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-BINDING-SUMMARY: tcrv_rvv.route_operand_binding_operands
// STALE-RT-MACC-BINDING-SUMMARY-SAME: must mirror
// STALE-RT-MACC-BINDING-SUMMARY-SAME: rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|macc-rhs|hdr

// STALE-RT-MACC-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-ABI: tcrv_rvv.runtime_abi_order
// STALE-RT-MACC-ABI-SAME: must mirror
// STALE-RT-MACC-ABI-SAME: cmp_lhs,lhs,rhs_scalar,rhs,acc,out,n

// STALE-RT-MACC-PREDICATE: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-PREDICATE: tcrv_rvv.compare_predicate_kind
// STALE-RT-MACC-PREDICATE-SAME: must mirror
// STALE-RT-MACC-PREDICATE-SAME: slt

// STALE-RT-MACC-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-HEADER: tcrv_rvv.required_header_declarations
// STALE-RT-MACC-HEADER-SAME: must mirror
// STALE-RT-MACC-HEADER-SAME: stddef.h,stdint.h

// STALE-RT-MACC-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-TYPE: tcrv_rvv.c_type_mapping
// STALE-RT-MACC-TYPE-SAME: must mirror
// STALE-RT-MACC-TYPE-SAME: vl:uint64_t

// STALE-RT-MACC-SCALAR: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-SCALAR: tcrv_rvv.accumulation_mask_producer_source
// STALE-RT-MACC-SCALAR-SAME: must mirror
// STALE-RT-MACC-SCALAR-SAME: script-derived-runtime-scalar-mask-producer

// STALE-RT-MACC-MASK-SOURCE: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-MASK-SOURCE: tcrv_rvv.mask_source
// STALE-RT-MACC-MASK-SOURCE-SAME: must mirror
// STALE-RT-MACC-MASK-SOURCE-SAME: script-derived-mask-source

// STALE-RT-MACC-MASK-FORM: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-MASK-FORM: tcrv_rvv.mask_memory_form
// STALE-RT-MACC-MASK-FORM-SAME: must mirror
// STALE-RT-MACC-MASK-FORM-SAME: script-derived-mask-form

// STALE-RT-MACC-INACTIVE: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-INACTIVE: tcrv_rvv.inactive_lane_contract
// STALE-RT-MACC-INACTIVE-SAME: must mirror
// STALE-RT-MACC-INACTIVE-SAME: script-derived-inactive-lanes

// STALE-RT-MACC-PASSTHROUGH: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-PASSTHROUGH: tcrv_rvv.masked_passthrough_layout
// STALE-RT-MACC-PASSTHROUGH-SAME: must mirror
// STALE-RT-MACC-PASSTHROUGH-SAME: script-derived-passthrough

// STALE-RT-MACC-ACC-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-ACC-LAYOUT: tcrv_rvv.macc_accumulator_layout
// STALE-RT-MACC-ACC-LAYOUT-SAME: must mirror
// STALE-RT-MACC-ACC-LAYOUT-SAME: script-derived-accumulator-layout

// STALE-RT-MACC-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-LAYOUT: tcrv_rvv.macc_result_layout
// STALE-RT-MACC-LAYOUT-SAME: must mirror
// STALE-RT-MACC-LAYOUT-SAME: script-derived-result-layout

// STALE-RT-MACC-SOURCE-MEMORY: RVV materialized EmitC target artifact bridge failed
// STALE-RT-MACC-SOURCE-MEMORY: tcrv_rvv.source_memory_form
// STALE-RT-MACC-SOURCE-MEMORY-SAME: must mirror
// STALE-RT-MACC-SOURCE-MEMORY-SAME: script-derived-source-memory
