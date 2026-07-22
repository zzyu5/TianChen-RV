// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"/s//weft_rvv.dequantization_relation", value = "script-derived-dequant"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RELATION
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.dequant_scale_role", value = "dequant-scale-value"/s//weft_rvv.dequant_scale_role", value = "output-buffer"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SCALE
// RUN: weft-opt %s --weft-materialize-emission-plans | sed '0,/weft_rvv.dest_lmul", value = "m1"/s//weft_rvv.dest_lmul", value = "m2"/' | not weft-translate --weft-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEST-LMUL

// Hand-authored explicit selected-body input for one bounded Stage 2
// i32-to-f32 runtime-scale dequantization slice. The typed weft_rvv body and
// RVV provider-owned route facts are the route authority; route ids, artifact
// names, ABI names, q-names, and common EmitC/export code are mirrors only.

module {
  weft.exec.kernel @explicit_selected_body_dequantize_i32_to_f32_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_dequantize_i32_to_f32 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %scale = weft_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:scale", role = "dequant-scale-value"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_dequantize_i32_to_f32, sew = 32 : i64, source_kernel = "explicit_selected_body_dequantize_i32_to_f32_kernel", status = "selected-lowering-boundary"} {
        %source = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %result = weft_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<f32, "m1">
        weft_rvv.store %out, %result, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<f32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_dequantize_i32_to_f32 {origin = "rvv-plugin", policy = "explicit-selected-body-dequantize-i32-to-f32-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-dequantize-i32-to-f32-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "dequantize_i32_to_f32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.dequantize"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "unit-stride-dequantization"}
// PLAN-SAME: {key = "weft_rvv.runtime_abi_order", value = "lhs,scale,out,n"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:dequantize_i32_to_f32.v1"}
// PLAN-SAME: {key = "weft_rvv.dequantization_route_family_plan", value = "rvv-dequantization-route-family-plan.v1"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "weft_rvv.target_leaf_profile", value = "rvv-v1-i32m1-f32m1-runtime-scale-dequantization-leaf-profile.v1"}
// PLAN-SAME: {key = "weft_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-dequantize-i32-to-f32-runtime-scale-plan-validated"}
// PLAN-SAME: {key = "weft_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "weft_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float"}
// PLAN-SAME: {key = "weft_rvv.source_element_type", value = "i32"}
// PLAN-SAME: {key = "weft_rvv.result_element_type", value = "f32"}
// PLAN-SAME: {key = "weft_rvv.source_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.source_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.dest_sew", value = "32"}
// PLAN-SAME: {key = "weft_rvv.dest_lmul", value = "m1"}
// PLAN-SAME: {key = "weft_rvv.conversion_kind", value = "i32_to_f32_scaled"}
// PLAN-SAME: {key = "weft_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "weft_rvv.dequantize_convert_intrinsic", value = "__riscv_vfcvt_f_x_v_f32m1"}
// PLAN-SAME: {key = "weft_rvv.dequantize_scale_intrinsic", value = "__riscv_vfmul_vf_f32m1"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_c_type", value = "float"}
// PLAN-SAME: {key = "weft_rvv.dequant_scale_name", value = "scale"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-dequantize-i32-to-f32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_dequantize_i32_to_f32

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_dequantize_i32_to_f32
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-dequantize-i32-to-f32-callable-c-abi.v1
// HEADER: weft.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.memory_form: unit-stride-dequantization
// HEADER: weft.rvv.source_sew: 32
// HEADER: weft.rvv.source_lmul: m1
// HEADER: weft.rvv.dest_sew: 32
// HEADER: weft.rvv.dest_lmul: m1
// HEADER: weft.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER: weft.rvv.dequantize_convert_intrinsic: __riscv_vfcvt_f_x_v_f32m1
// HEADER: weft.rvv.dequantize_scale_intrinsic: __riscv_vfmul_vf_f32m1
// HEADER: weft.rvv.dequant_scale_role: dequant-scale-value
// HEADER: weft.rvv.dequant_scale_c_type: float
// HEADER: weft.rvv.dequant_scale_name: scale
// HEADER: weft.rvv.target_leaf_profile: rvv-v1-i32m1-f32m1-runtime-scale-dequantization-leaf-profile.v1
// HEADER: weft.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: weft.rvv.provider_supported_mirror: provider_supported_mirror:rvv-dequantize-i32-to-f32-runtime-scale-plan-validated
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:dequantize_i32_to_f32.v1
// HEADER: weft.rvv.dequantization_route_family_plan: rvv-dequantization-route-family-plan.v1
// HEADER: weft.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: weft.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float
// HEADER: void weft_emitc_explicit_selected_body_dequantize_i32_to_f32_kernel_explicit_selected_body_rvv_dequantize_i32_to_f32(const int32_t *lhs, float scale, float *out, size_t n);

// MISSING-GEARBOX-SAME: before provider route construction

// BAD-GEARBOX-CANDIDATE-SET: selected RVV Gearbox candidate
// BAD-GEARBOX-CANDIDATE-SET-SAME: belong to pass-produced legal candidate set

// UNSUPPORTED-GEARBOX-UNROLL-SAME: provider-derived '2' but found '3'

// STALE-RELATION: metadata key '{{.*}}dequantization_relation'{{.*}}'signed-i32m1-to-f32m1-scale-f32' but was 'script-derived-dequant'

// STALE-SCALE: metadata key '{{.*}}dequant_scale_role'{{.*}}'dequant-scale-value' but was 'output-buffer'

// STALE-DEST-LMUL: metadata key '{{.*}}dest_lmul'{{.*}}'m1' but was 'm2'

// STALE-GEARBOX-SELECTED: metadata key '{{.*}}gearbox.selected_candidate'{{.*}}'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but was 'artifact-name-derived-gear'

// STALE-GEARBOX: metadata key '{{.*}}gearbox.schedule_id'{{.*}}'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but was 'artifact-name-derived-gear'
