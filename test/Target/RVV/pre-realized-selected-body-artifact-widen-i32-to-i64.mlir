// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/rvv-widening-conversion-route-family-plan.v1/s//rvv-script-derived-widening-conversion-plan.v1/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CONVERSION-PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.conversion_relation\", value = \"signed-i32m1-to-i64m2/s//weft_rvv.conversion_relation\", value = \"script-derived-relation/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CONVERSION-RELATION
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/provider_supported_mirror:rvv-widen-i32-to-i64-plan-validated/s//provider_supported_mirror:rvv-script-derived-widening-conversion/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CONVERSION-PROVIDER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_conversion_route_family_plan\", value = \"rvv-widening-conversion-route-family-plan.v1\"}/s//&, {key = \"weft_rvv.elementwise_arithmetic_route_family_plan\", value = \"script-derived-elementwise-plan\"}/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ELEMENTWISE-RESIDUE

// Pre-realized selected-body input for one bounded Stage 2 signed widening
// conversion slice. The RVV plugin must derive source i32/m1, destination
// i64/m2, memory, policy, route, and ABI facts from typed body/config facts.

module {
  weft.exec.kernel @pre_realized_body_widen_i32_to_i64_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_widen_i32_to_i64 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i32-to-i64:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i32-to-i64:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i32-to-i64:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i32m1-to-i64m2", dest_lmul = "m2", dest_sew = 64 : i64, memory_form = "unit-stride-conversion", op_kind = "widen_i32_to_i64", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "m1", source_sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_widen_i32_to_i64 {origin = "rvv-plugin", policy = "pre-realized-selected-body-widen-i32-to-i64-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widen-i32-to-i64-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m2"
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_widen_i32_to_i64
// REALIZED: weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.widening_convert
// REALIZED-SAME: kind = "widen_i32_to_i64"
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i64, "m2">
// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widen_i32_to_i64"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_convert"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew64-lmul-m2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.sew", value = "64"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m2"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-stride-conversion"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widen_i32_to_i64.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:widen_i32_to_i64.v1;lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i32m1|relation-signed-i32m1-to-i64m2|hdr;out=output-buffer:out:abi|res-store|convert-result|res-i64m2|relation-signed-i32m1-to-i64m2|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.widening_conversion_route_family_plan", value = "rvv-widening-conversion-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew64-lmul-m2"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-i32m1-i64m2-widening-conversion-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-widen-i32-to-i64-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,result:signed-e64m2"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.dest_sew", value = "64"}
// PLAN-SAME: {key = "weft_rvv.dest_lmul", value = "m2"}
// PLAN-SAME: {key = "weft_rvv.conversion_relation", value = "signed-i32m1-to-i64m2"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widen-i32-to-i64-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_widen_i32_to_i64

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_widen_i32_to_i64
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-widen-i32-to-i64-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew64-lmul-m2-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.memory_form: unit-stride-conversion
// HEADER: weft.rvv.source_sew: 32
// HEADER: weft.rvv.source_lmul: m1
// HEADER: weft.rvv.dest_sew: 64
// HEADER: weft.rvv.dest_lmul: m2
// HEADER: weft.rvv.conversion_relation: signed-i32m1-to-i64m2
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-i32m1-i64m2-widening-conversion-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-widen-i32-to-i64-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:widen_i32_to_i64.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:widen_i32_to_i64.v1;lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i32m1|relation-signed-i32m1-to-i64m2|hdr;out=output-buffer:out:abi|res-store|convert-result|res-i64m2|relation-signed-i32m1-to-i64m2|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.widening_conversion_route_family_plan: rvv-widening-conversion-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,result:signed-e64m2
// HEADER: void weft_emitc_pre_realized_body_widen_i32_to_i64_kernel_pre_realized_body_rvv_widen_i32_to_i64(const int32_t *lhs, int64_t *out, size_t n);

// STALE-CONVERSION-PLAN: RVV materialized EmitC target artifact bridge failed
// STALE-CONVERSION-PLAN: weft_rvv.widening_conversion_route_family_plan
// STALE-CONVERSION-PLAN-SAME: must mirror
// STALE-CONVERSION-PLAN-SAME: rvv-script-derived-widening-conversion-plan.v1

// STALE-CONVERSION-RELATION: RVV materialized EmitC target artifact bridge failed
// STALE-CONVERSION-RELATION: weft_rvv.conversion_relation
// STALE-CONVERSION-RELATION-SAME: must mirror
// STALE-CONVERSION-RELATION-SAME: script-derived-relation

// STALE-CONVERSION-PROVIDER: RVV materialized EmitC target artifact bridge failed
// STALE-CONVERSION-PROVIDER: weft_rvv.provider_supported_mirror
// STALE-CONVERSION-PROVIDER-SAME: must mirror
// STALE-CONVERSION-PROVIDER-SAME: rvv-script-derived-widening-conversion

// STALE-ELEMENTWISE-RESIDUE: carry exactly 43{{.*}}metadata entries
