// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.conversion_kind\", value = \"sign_extend_widen_vf2/s//weft_rvv.conversion_kind\", value = \"widen_i32_to_i64/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-CONVERSION-KIND
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.source_sew\", value = \"16/s//weft_rvv.source_sew\", value = \"32/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SOURCE-SEW
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.dest_lmul\", value = \"m1/s//weft_rvv.dest_lmul\", value = \"m2/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEST-LMUL
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.runtime_abi_order\", value = \"lhs,out,n/s//weft_rvv.runtime_abi_order\", value = \"lhs,n,out/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RUNTIME-ABI
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.route_operand_binding_operands\", value = \"rvv-route-operand-binding:widen_i16_to_i32.v1;lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i16mf2|relation-signed-i16mf2-to-i32m1|hdr;out=output-buffer:out:abi|res-store|convert-result|res-i32m1|relation-signed-i16mf2-to-i32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr/s//weft_rvv.route_operand_binding_operands\", value = \"metadata-derived-binding-summary/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-BINDING
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.required_header_declarations\", value = \"stddef.h,stdint.h,riscv_vector.h/s//weft_rvv.required_header_declarations\", value = \"stddef.h/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-HEADERS
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | sed '0,/weft_rvv.widening_conversion_route_family_plan\", value = \"rvv-widening-conversion-route-family-plan.v1\"}/s//&, {key = \"weft_rvv.dequantization_route_family_plan\", value = \"metadata-derived-dequantization-plan\"}/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEQUANT-RESIDUE

// Pre-realized selected-body input for one bounded Stage 2 signed widening
// conversion slice. The RVV plugin must derive source i16/mf2, destination
// i32/m1, conversion kind, memory, policy, route, and ABI facts from typed
// body/config facts.

module {
  weft.exec.kernel @pre_realized_body_widen_i16_to_i32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_widen_i16_to_i32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i16-to-i32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i16-to-i32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-widen-i16-to-i32:n", role = "runtime-element-count"} : index
      weft_rvv.typed_widening_conversion_pre_realized_body %lhs, %out, %n {conversion_relation = "signed-i16mf2-to-i32m1", dest_lmul = "m1", dest_sew = 32 : i64, memory_form = "unit-stride-conversion", op_kind = "sign_extend_widen_vf2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, source_lmul = "mf2", source_sew = 16 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_widen_i16_to_i32 {origin = "rvv-plugin", policy = "pre-realized-selected-body-widen-i16-to-i32-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-widen-i16-to-i32-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: lmul = "m1"
// REALIZED-SAME: origin = "rvv-plugin"
// REALIZED-SAME: selected_path_role = "dispatch case"
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_widen_i16_to_i32
// REALIZED: weft_rvv.load
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">
// REALIZED: weft_rvv.widening_convert
// REALIZED-SAME: kind = "sign_extend_widen_vf2"
// REALIZED-SAME: !weft_rvv.vector<i16, "mf2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
// REALIZED: weft_rvv.store
// REALIZED-SAME: !weft_rvv.vector<i32, "m1">
// REALIZED-NOT: weft_rvv.typed_widening_conversion_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widen_i16_to_i32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.widening_convert"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-stride-conversion"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widen_i16_to_i32.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:widen_i16_to_i32.v1;lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i16mf2|relation-signed-i16mf2-to-i32m1|hdr;out=output-buffer:out:abi|res-store|convert-result|res-i32m1|relation-signed-i16mf2-to-i32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "weft_rvv.widening_conversion_route_family_plan", value = "rvv-widening-conversion-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-i16mf2-i32m1-widening-conversion-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-widen-i16-to-i32-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:signed-e16mf2,result:signed-e32m1"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "16"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "mf2"}
// PLAN-SAME: {key = "weft_rvv.dest_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.dest_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.conversion_kind", value = "sign_extend_widen_vf2"}
// PLAN-SAME: {key = "weft_rvv.conversion_relation", value = "signed-i16mf2-to-i32m1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widen-i16-to-i32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_widen_i16_to_i32

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_widen_i16_to_i32
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-widen-i16-to-i32-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.memory_form: unit-stride-conversion
// HEADER: weft.rvv.source_sew: 16
// HEADER: weft.rvv.source_lmul: mf2
// HEADER: weft.rvv.dest_sew: 32
// HEADER: weft.rvv.dest_lmul: m1
// HEADER: weft.rvv.conversion_relation: signed-i16mf2-to-i32m1
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-i16mf2-i32m1-widening-conversion-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-widen-i16-to-i32-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:widen_i16_to_i32.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:widen_i16_to_i32.v1;lhs=lhs-input-buffer:lhs:abi|src-load|convert-src|src-i16mf2|relation-signed-i16mf2-to-i32m1|hdr;out=output-buffer:out:abi|res-store|convert-result|res-i32m1|relation-signed-i16mf2-to-i32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: weft.rvv.widening_conversion_route_family_plan: rvv-widening-conversion-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,source:signed-e16mf2,result:signed-e32m1
// HEADER: void weft_emitc_pre_realized_body_widen_i16_to_i32_kernel_pre_realized_body_rvv_widen_i16_to_i32(const int16_t *lhs, int32_t *out, size_t n);

// STALE-CONVERSION-KIND: RVV materialized EmitC target artifact bridge failed
// STALE-CONVERSION-KIND: weft_rvv.conversion_kind
// STALE-CONVERSION-KIND-SAME: must mirror
// STALE-CONVERSION-KIND-SAME: sign_extend_widen_vf2
// STALE-CONVERSION-KIND-SAME: widen_i32_to_i64

// STALE-SOURCE-SEW: RVV materialized EmitC target artifact bridge failed
// STALE-SOURCE-SEW: weft_rvv.source_sew
// STALE-SOURCE-SEW-SAME: must mirror
// STALE-SOURCE-SEW-SAME: 16
// STALE-SOURCE-SEW-SAME: 32

// STALE-DEST-LMUL: RVV materialized EmitC target artifact bridge failed
// STALE-DEST-LMUL: weft_rvv.dest_lmul
// STALE-DEST-LMUL-SAME: must mirror
// STALE-DEST-LMUL-SAME: m1
// STALE-DEST-LMUL-SAME: m2

// STALE-RUNTIME-ABI: RVV materialized EmitC target artifact bridge failed
// STALE-RUNTIME-ABI: weft_rvv.runtime_abi_order
// STALE-RUNTIME-ABI-SAME: must mirror
// STALE-RUNTIME-ABI-SAME: lhs,out,n
// STALE-RUNTIME-ABI-SAME: lhs,n,out

// STALE-BINDING: RVV materialized EmitC target artifact bridge failed
// STALE-BINDING: weft_rvv.route_operand_binding_operands
// STALE-BINDING-SAME: must mirror
// STALE-BINDING-SAME: metadata-derived-binding-summary

// STALE-HEADERS: RVV materialized EmitC target artifact bridge failed
// STALE-HEADERS: weft_rvv.required_header_declarations
// STALE-HEADERS-SAME: must mirror
// STALE-HEADERS-SAME: stddef.h,stdint.h,riscv_vector.h

// STALE-DEQUANT-RESIDUE: RVV materialized EmitC target artifact bridge failed
// STALE-DEQUANT-RESIDUE: carry exactly 43{{.*}}metadata entries
