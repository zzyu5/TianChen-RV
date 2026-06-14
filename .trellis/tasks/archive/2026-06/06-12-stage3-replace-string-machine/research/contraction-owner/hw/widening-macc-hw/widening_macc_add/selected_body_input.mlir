// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Explicit selected-body input for one bounded Stage 2 signed widening
// multiply-accumulate slice. The selected tcrv.exec RVV variant already
// carries the realized generic tcrv_rvv body; the RVV provider must still
// derive the contraction family plan and route binding closure from body facts.

module {
  tcrv.exec.kernel @explicit_selected_body_widening_macc_add_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_widening_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-macc-add:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %rhs = tcrv_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int16_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-macc-add:rhs", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %acc = tcrv_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-macc-add:acc", role = "accumulator-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-macc-add:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-widening-macc-add:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_widening_macc_add, sew = 32 : i64, source_kernel = "explicit_selected_body_widening_macc_add_kernel", status = "selected-lowering-boundary"} {
        %a = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %b = tcrv_rvv.load %rhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "mf2">
        %seed = tcrv_rvv.load %acc, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %sum = tcrv_rvv.widening_macc %a, %b, %seed, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "signed_widening_macc_add", macc_relation = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1", result_layout = "store-widening-multiply-accumulate-result-to-output-buffer"} : !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i16, "mf2">, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %out, %sum, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_widening_macc_add {origin = "rvv-plugin", policy = "explicit-selected-body-widening-macc-add-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-widening-macc-add-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "widening_macc_add"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.widening_macc"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,rhs,acc,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:widening_macc_add.v1"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:widening_macc_add.v1;lhs=lhs-input-buffer:lhs:abi|src-load|wmacc-lhs|src-i16mf2|hdr;rhs=rhs-input-buffer:rhs:abi|src-load|wmacc-rhs|src-i16mf2|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|wmacc-acc|acc-i32m1|hdr;out=output-buffer:out:abi|res-store|res-i32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr"}
// PLAN-SAME: {key = "tcrv_rvv.contraction_route_family_plan", value = "rvv-contraction-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-contraction-family-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "16"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "mf2"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.accumulator_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.result_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.result_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.widening_macc_accumulator_layout", value = "separate-i32-vector-accumulator-input"}
// PLAN-SAME: {key = "tcrv_rvv.widening_macc_result_layout", value = "store-widening-multiply-accumulate-result-to-output-buffer"}
// PLAN-SAME: {key = "tcrv_rvv.widening_macc_relation", value = "signed-i16mf2xi16mf2-plus-i32m1-to-i32m1"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-widening-macc-add-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_widening_macc_add

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_widening_macc_add
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-widening-macc-add-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.memory_form: vector-rhs-load
// HEADER: tianchenrv.rvv.widening_macc_relation: signed-i16mf2xi16mf2-plus-i32m1-to-i32m1
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:widening_macc_add.v1
// HEADER: tianchenrv.rvv.route_operand_binding_operands: rvv-route-operand-binding:widening_macc_add.v1;lhs=lhs-input-buffer:lhs:abi|src-load|wmacc-lhs|src-i16mf2|hdr;rhs=rhs-input-buffer:rhs:abi|src-load|wmacc-rhs|src-i16mf2|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|wmacc-acc|acc-i32m1|hdr;out=output-buffer:out:abi|res-store|res-i32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr
// HEADER: tianchenrv.rvv.contraction_route_family_plan: rvv-contraction-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32
// HEADER: void tcrv_emitc_explicit_selected_body_widening_macc_add_kernel_explicit_selected_body_rvv_widening_macc_add(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n);
