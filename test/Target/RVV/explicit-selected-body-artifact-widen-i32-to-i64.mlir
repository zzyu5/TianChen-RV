// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Hand-authored explicit selected-body input for one bounded Stage 2 signed
// widening conversion slice. The typed weft_rvv body is already the route
// authority; route ids, helper strings, descriptors, source-front-door markers,
// and common EmitC/export code are not allowed to infer conversion semantics.

module {
  weft.exec.kernel @explicit_selected_body_widen_i32_to_i64_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_widen_i32_to_i64 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widen-i32-to-i64:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widen-i32-to-i64:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widen-i32-to-i64:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 64 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_widen_i32_to_i64, sew = 64 : i64, source_kernel = "explicit_selected_body_widen_i32_to_i64_kernel", status = "selected-lowering-boundary"} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %widened = weft_rvv.widening_convert %source, %vl {kind = "widen_i32_to_i64"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i64, "m2">
        weft_rvv.store %out, %widened, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i64, "m2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_widen_i32_to_i64 {origin = "rvv-plugin", policy = "explicit-selected-body-widen-i32-to-i64-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widen-i32-to-i64-fallback-envelope"}
    }
  }
}

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
// PLAN-SAME: target = @explicit_selected_body_rvv_widen_i32_to_i64

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_widen_i32_to_i64
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
// HEADER: void weft_emitc_explicit_selected_body_widen_i32_to_i64_kernel_explicit_selected_body_rvv_widen_i32_to_i64(const int32_t *lhs, int64_t *out, size_t n);
