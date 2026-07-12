// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-generic-computed-masked-macc-add-emitc-route/s//rvv-script-derived-computed-mask-macc-route/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-ROUTE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated/s//provider_supported_mirror:rvv-script-derived-computed-mask-macc/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-route-operand-binding:computed_masked_macc_add.v1/s//rvv-route-operand-binding:script-derived-computed-mask-macc.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n/s//cmp_lhs,lhs,cmp_rhs,rhs,acc,out,n/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/stddef.h,stdint.h,riscv_vector.h/s//stddef.h,stdint.h/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/vl:size_t/s//vl:uint64_t/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-TYPE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-computed-mask-accumulation-route-family-plan.v1/s//rvv-script-derived-computed-mask-macc-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-ACCUM
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"/s//weft_rvv.macc_accumulator_layout", value = "script-derived-accumulator-layout"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-LAYOUT
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-route-operand-binding:computed_masked_macc_add.v1/s//rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-RUNTIME-SCALAR-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.compare_predicate_kind", value = "slt"/s//weft_rvv.compare_predicate_kind", value = "sle"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-PREDICATE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"/s//weft_rvv.mask_role", value = "script-derived-mask-role"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-MASK-ROLE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"/s//weft_rvv.mask_source", value = "runtime-scalar-splat-compare-rhs"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-MASK-SOURCE
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.mask_memory_form", value = "compare-produced-mask"/s//weft_rvv.mask_memory_form", value = "script-derived-mask-form"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-MASK-FORM
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.source_memory_form", value = "unit-stride-load"/s//weft_rvv.source_memory_form", value = "runtime-scalar-threshold"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-SOURCE-MEMORY
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.masked_passthrough_layout", value = "accumulator-vector-preserves-inactive-lanes"/s//weft_rvv.masked_passthrough_layout", value = "drop-inactive-lanes"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-PASSTHROUGH
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.macc_arithmetic_kind", value = "add"/s//weft_rvv.macc_arithmetic_kind", value = "script-derived-macc-kind"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-ARITHMETIC
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"/s//weft_rvv.macc_result_layout", value = "store-active-products-only"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-MACC-RESULT

// Pre-realized selected-body input for one bounded Stage2 computed-mask
// multiply-accumulate slice. The RVV plugin must realize compare lhs/rhs,
// payload lhs/rhs, accumulator passthrough/result, compare-produced mask, and
// masked_macc into explicit typed structure before route planning/provider use.

module {
  weft.exec.kernel @pre_realized_body_computed_masked_macc_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_computed_masked_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %cmp_lhs = weft_rvv.runtime_abi_value {c_name = "cmp_lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc:cmp_lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %cmp_rhs = weft_rvv.runtime_abi_value {c_name = "cmp_rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc:cmp_rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc:lhs-payload", role = "dot-lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc:rhs-payload", role = "dot-rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-computed-mask-macc:n", role = "runtime-element-count"} : index
      weft_rvv.typed_computed_mask_macc_pre_realized_body %cmp_lhs, %cmp_rhs, %lhs, %rhs, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", mask_memory_form = "compare-produced-mask", mask_role = "predicate-mask-produced-by-compare", mask_source = "compare-produced-mask-same-vl-scope", memory_form = "computed-mask-unit-stride-macc", op_kind = "computed_masked_macc_add", predicate_kind = "slt", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_computed_masked_macc_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-computed-mask-macc-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-computed-mask-macc-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_computed_mask_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_computed_masked_macc_add
// REALIZED: %[[CMP_LHS:.*]] = weft_rvv.load
// REALIZED: %[[CMP_RHS:.*]] = weft_rvv.load
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[CMP_LHS]], %[[CMP_RHS]], %[[VL]]
// REALIZED-SAME: kind = "slt"
// REALIZED: %[[SUM:.*]] = weft_rvv.masked_macc %[[MASK]], %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: mask_memory_form = "compare-produced-mask"
// REALIZED-SAME: mask_role = "predicate-mask-produced-by-compare"
// REALIZED-SAME: mask_source = "compare-produced-mask-same-vl-scope"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.macc
// REALIZED-NOT: weft_rvv.binary
// REALIZED-NOT: weft_rvv.masked_move
// REALIZED-NOT: weft_rvv.typed_computed_mask_macc_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "computed_masked_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_macc"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.compare_predicate_kind", value = "slt"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "computed-mask-unit-stride-macc"}
// PLAN-SAME: {key = "weft_rvv.runtime_vl_contract", value = "rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1"}
// PLAN-SAME: {key = "weft_rvv.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "weft_rvv.target_capability_provider_mirror", value = "selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact"}
// PLAN-SAME: {key = "weft_rvv.target_capability_legality_mirror", value = "selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:computed_masked_macc_add.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:computed_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.accumulation_route_family_plan", value = "rvv-computed-mask-accumulation-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.accumulation_compute_suffix", value = "vector-masked-macc-add"}
// PLAN-SAME: {key = "weft_rvv.accumulation_mask_producer_source", value = "vector-compare-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.accumulation_accumulator_contract", value = "vector-accumulator-input-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.accumulation_result_contract", value = "vector-macc-result-stored-to-output-buffer"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-typed-computed-mask-macc-add-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:typed-vector,mask:typed-mask,result:typed-vector"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.mask_memory_form", value = "compare-produced-mask"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-macc-false-lanes-preserve-accumulator"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "accumulator-vector-preserves-inactive-lanes"}
// PLAN-SAME: {key = "weft_rvv.macc_arithmetic_kind", value = "add"}
// PLAN-SAME: {key = "weft_rvv.macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "weft_rvv.macc_result_layout", value = "store-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-computed-masked-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_computed_masked_macc_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_computed_masked_macc_add
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-computed-masked-macc-add-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.runtime_vl_contract: rvv-runtime-avl-n-multivl-setvl-with-vl-loop.v1
// HEADER: weft.rvv.runtime_avl_source: runtime_abi:n
// HEADER: weft.rvv.runtime_abi_order: cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n
// HEADER: weft.rvv.compare_predicate_kind: slt
// HEADER: weft.rvv.mask_role: predicate-mask-produced-by-compare
// HEADER: weft.rvv.mask_source: compare-produced-mask-same-vl-scope
// HEADER: weft.rvv.mask_memory_form: compare-produced-mask
// HEADER: weft.rvv.inactive_lane_contract: masked-macc-false-lanes-preserve-accumulator
// HEADER: weft.rvv.masked_passthrough_layout: accumulator-vector-preserves-inactive-lanes
// HEADER: weft.rvv.macc_accumulator_layout: separate-i32-vector-accumulator-input
// HEADER: weft.rvv.macc_result_layout: store-multiply-accumulate-result-to-output-buffer
// HEADER: weft.rvv.macc_arithmetic_kind: add
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-typed-computed-mask-macc-add-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated
// HEADER: weft.rvv.target_capability_provider_mirror: selected_capability_provider_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact
// HEADER: weft.rvv.target_capability_legality_mirror: selected_target_capability_legality_mirror:@rvv;id=rvv;kind=isa-vector;rvv=exact;sew=32;lmul=m1;tail=agnostic;mask=agnostic
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:computed_masked_macc_add.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:computed_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;cmp_rhs=rhs-input-buffer:cmp_rhs:abi|cmp-rhs|cmp-call|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.accumulation_route_family_plan: rvv-computed-mask-accumulation-route-family-plan.v1
// HEADER: weft.rvv.accumulation_compute_suffix: vector-masked-macc-add
// HEADER: weft.rvv.accumulation_mask_producer_source: vector-compare-rhs-load
// HEADER: weft.rvv.accumulation_accumulator_contract: vector-accumulator-input-preserves-inactive-lanes
// HEADER: weft.rvv.accumulation_result_contract: vector-macc-result-stored-to-output-buffer
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:typed-vector,mask:typed-mask,result:typed-vector
// HEADER: void weft_emitc_pre_realized_body_computed_masked_macc_add_kernel_pre_realized_body_rvv_computed_masked_macc_add(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);

// STALE-MACC-ROUTE: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-ROUTE: candidate rvv_emitc_lowerable_route provenance must mirror selected typed RVV body route
// STALE-MACC-ROUTE-SAME: rvv-script-derived-computed-mask-macc-route

// STALE-MACC-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-PROVIDER: candidate weft_rvv.provider_supported_mirror provenance must mirror selected typed RVV body provider support
// STALE-MACC-PROVIDER-SAME: provider_supported_mirror:rvv-script-derived-computed-mask-macc

// STALE-MACC-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-BINDING: candidate weft_rvv.route_operand_binding_plan provenance must mirror selected typed RVV body binding plan
// STALE-MACC-BINDING-SAME: rvv-route-operand-binding:script-derived-computed-mask-macc.v1

// STALE-MACC-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-ABI: weft_rvv.runtime_abi_order
// STALE-MACC-ABI-SAME: must mirror
// STALE-MACC-ABI-SAME: cmp_lhs,lhs,cmp_rhs,rhs,acc,out,n

// STALE-MACC-HEADER: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-HEADER: weft_rvv.required_header_declarations
// STALE-MACC-HEADER-SAME: must mirror
// STALE-MACC-HEADER-SAME: stddef.h,stdint.h

// STALE-MACC-TYPE: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-TYPE: weft_rvv.c_type_mapping
// STALE-MACC-TYPE-SAME: must mirror
// STALE-MACC-TYPE-SAME: vl:uint64_t

// STALE-MACC-ACCUM: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-ACCUM: weft_rvv.accumulation_route_family_plan
// STALE-MACC-ACCUM-SAME: must mirror
// STALE-MACC-ACCUM-SAME: rvv-script-derived-computed-mask-macc-plan.v1

// STALE-MACC-LAYOUT: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-LAYOUT: weft_rvv.macc_accumulator_layout
// STALE-MACC-LAYOUT-SAME: must mirror
// STALE-MACC-LAYOUT-SAME: script-derived-accumulator-layout

// STALE-MACC-RUNTIME-SCALAR-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-RUNTIME-SCALAR-BINDING: candidate weft_rvv.route_operand_binding_plan provenance must mirror selected typed RVV body binding plan
// STALE-MACC-RUNTIME-SCALAR-BINDING-SAME: rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1

// STALE-MACC-PREDICATE: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-PREDICATE: weft_rvv.compare_predicate_kind
// STALE-MACC-PREDICATE-SAME: must mirror
// STALE-MACC-PREDICATE-SAME: sle

// STALE-MACC-MASK-ROLE: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-MASK-ROLE: weft_rvv.mask_role
// STALE-MACC-MASK-ROLE-SAME: must mirror
// STALE-MACC-MASK-ROLE-SAME: script-derived-mask-role

// STALE-MACC-MASK-SOURCE: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-MASK-SOURCE: weft_rvv.mask_source
// STALE-MACC-MASK-SOURCE-SAME: must mirror
// STALE-MACC-MASK-SOURCE-SAME: runtime-scalar-splat-compare-rhs

// STALE-MACC-MASK-FORM: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-MASK-FORM: weft_rvv.mask_memory_form
// STALE-MACC-MASK-FORM-SAME: must mirror
// STALE-MACC-MASK-FORM-SAME: script-derived-mask-form

// STALE-MACC-SOURCE-MEMORY: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-SOURCE-MEMORY: weft_rvv.source_memory_form
// STALE-MACC-SOURCE-MEMORY-SAME: must mirror
// STALE-MACC-SOURCE-MEMORY-SAME: runtime-scalar-threshold

// STALE-MACC-PASSTHROUGH: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-PASSTHROUGH: weft_rvv.masked_passthrough_layout
// STALE-MACC-PASSTHROUGH-SAME: must mirror
// STALE-MACC-PASSTHROUGH-SAME: drop-inactive-lanes

// STALE-MACC-ARITHMETIC: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-ARITHMETIC: weft_rvv.macc_arithmetic_kind
// STALE-MACC-ARITHMETIC-SAME: must mirror
// STALE-MACC-ARITHMETIC-SAME: script-derived-macc-kind

// STALE-MACC-RESULT: RVV materialized EmitC target artifact bridge failed
// STALE-MACC-RESULT: weft_rvv.macc_result_layout
// STALE-MACC-RESULT-SAME: must mirror
// STALE-MACC-RESULT-SAME: store-active-products-only
