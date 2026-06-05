// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER
// RUN: not tcrv-opt %s --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=MISSING-GEARBOX
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"/s//tcrv_rvv.dequantization_relation", value = "script-derived-dequant"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-RELATION
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.dequant_scale_role", value = "dequant-scale-value"/s//tcrv_rvv.dequant_scale_role", value = "output-buffer"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SCALE
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.dest_lmul", value = "m1"/s//tcrv_rvv.dest_lmul", value = "m2"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DEST-LMUL
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed '0,/tcrv_rvv.gearbox.candidate_set = "rvv-gearbox-candidate-set.v1\[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1\]"/s//tcrv_rvv.gearbox.candidate_set = "rvv-gearbox-candidate-set.v1[artifact-name-derived-gear]"/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=BAD-GEARBOX-CANDIDATE-SET
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules | sed '0,/tcrv_rvv.gearbox.unroll = 2 : i64/s//tcrv_rvv.gearbox.unroll = 3 : i64/' | not tcrv-opt --tcrv-materialize-emission-plans 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-GEARBOX-UNROLL
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.gearbox.selected_candidate", value = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"/s//tcrv_rvv.gearbox.selected_candidate", value = "artifact-name-derived-gear"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-GEARBOX-SELECTED
// RUN: tcrv-opt %s --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-emission-plans | sed '0,/tcrv_rvv.gearbox.schedule_id", value = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"/s//tcrv_rvv.gearbox.schedule_id", value = "artifact-name-derived-gear"/' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-GEARBOX

// Hand-authored explicit selected-body input for one bounded Stage 2
// i32-to-f32 runtime-scale dequantization slice. The typed tcrv_rvv body and
// RVV provider-owned route facts are the route authority; route ids, artifact
// names, ABI names, q-names, and common EmitC/export code are mirrors only.

module {
  tcrv.exec.kernel @explicit_selected_body_dequantize_i32_to_f32_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv.exec.variant @explicit_selected_body_rvv_dequantize_i32_to_f32 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:lhs", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %scale = tcrv_rvv.runtime_abi_value {c_name = "scale", c_type = "float", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:scale", role = "dequant-scale-value"} : !tcrv_rvv.runtime_abi_value
      %out = tcrv_rvv.runtime_abi_value {c_name = "out", c_type = "float *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-dequantize-i32-to-f32:n", role = "runtime-element-count"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_dequantize_i32_to_f32, sew = 32 : i64, source_kernel = "explicit_selected_body_dequantize_i32_to_f32_kernel", status = "selected-lowering-boundary"} {
        %source = tcrv_rvv.load %lhs, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %result = tcrv_rvv.dequantize %source, %scale, %vl {dequant_relation = "signed-i32m1-to-f32m1-scale-f32", kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %out, %result, %vl : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
    tcrv.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @explicit_selected_body_rvv_dequantize_i32_to_f32 {origin = "rvv-plugin", policy = "explicit-selected-body-dequantize-i32-to-f32-case"}
      tcrv.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-dequantize-i32-to-f32-fallback-envelope"}
    }
  }
}

// PLAN: tcrv.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "dequantize_i32_to_f32"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "tcrv_rvv.dequantize"}
// PLAN-SAME: {key = "tcrv_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "tcrv_rvv.sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_control_plan", value = "rvv-runtime-avl-vl-control-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.memory_form", value = "unit-stride-dequantization"}
// PLAN-SAME: {key = "tcrv_rvv.runtime_abi_order", value = "lhs,scale,out,n"}
// PLAN-SAME: {key = "tcrv_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:dequantize_i32_to_f32.v1"}
// PLAN-SAME: {key = "tcrv_rvv.dequantization_route_family_plan", value = "rvv-dequantization-route-family-plan.v1"}
// PLAN-SAME: {key = "tcrv_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m1"}
// PLAN-SAME: {key = "tcrv_rvv.target_leaf_profile", value = "rvv-v1-i32m1-f32m1-runtime-scale-dequantization-leaf-profile.v1"}
// PLAN-SAME: {key = "tcrv_rvv.provider_supported_mirror", value = "provider_supported_mirror:rvv-dequantize-i32-to-f32-runtime-scale-plan-validated"}
// PLAN-SAME: {key = "tcrv_rvv.required_header_declarations", value = "stddef.h,stdint.h,riscv_vector.h"}
// PLAN-SAME: {key = "tcrv_rvv.c_type_mapping", value = "vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float"}
// PLAN-SAME: {key = "tcrv_rvv.source_element_type", value = "i32"}
// PLAN-SAME: {key = "tcrv_rvv.result_element_type", value = "f32"}
// PLAN-SAME: {key = "tcrv_rvv.source_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.source_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.dest_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.dest_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.conversion_kind", value = "i32_to_f32_scaled"}
// PLAN-SAME: {key = "tcrv_rvv.dequantization_relation", value = "signed-i32m1-to-f32m1-scale-f32"}
// PLAN-SAME: {key = "tcrv_rvv.dequantize_convert_intrinsic", value = "__riscv_vfcvt_f_x_v_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.dequantize_scale_intrinsic", value = "__riscv_vfmul_vf_f32m1"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_role", value = "dequant-scale-value"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_c_type", value = "float"}
// PLAN-SAME: {key = "tcrv_rvv.dequant_scale_name", value = "scale"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.candidate_set", value = "rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.selected_candidate", value = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.selection_reason", value = "select-bounded-u2-two-slice-route-plan-for-typed-dequantize-i32-to-f32-e32-m1-runtime-avl"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.legality_scope", value = "typed-dequantize-i32-to-f32-sew32-lmul-m1-runtime-avl"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.schedule_id", value = "rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.selector", value = "static-dequantize-i32-to-f32-e32-m1-u2"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.source", value = "rvv-gearbox-static-pass.v1"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.operation", value = "dequantize_i32_to_f32"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.unroll", value = "2"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.vl_policy", value = "runtime-avl-two-slice-setvl"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.source_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.source_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.dest_sew", value = "32"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.dest_lmul", value = "m1"}
// PLAN-SAME: {key = "tcrv_rvv.gearbox.runtime_avl_source", value = "runtime_abi:n"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "tcrv_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-dequantize-i32-to-f32-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_dequantize_i32_to_f32

// HEADER: tianchenrv.rvv.selected_variant: @explicit_selected_body_rvv_dequantize_i32_to_f32
// HEADER: tianchenrv.rvv.runtime_abi_name: rvv-generic-dequantize-i32-to-f32-callable-c-abi.v1
// HEADER: tianchenrv.rvv.emitc_route_mapping: rvv-generic-typed-body-emitc-route-family
// HEADER: tianchenrv.rvv.config_contract: rvv-selected-body-sew32-lmul-m1-tail-agnostic-mask-agnostic.v1
// HEADER: tianchenrv.rvv.memory_form: unit-stride-dequantization
// HEADER: tianchenrv.rvv.source_sew: 32
// HEADER: tianchenrv.rvv.source_lmul: m1
// HEADER: tianchenrv.rvv.dest_sew: 32
// HEADER: tianchenrv.rvv.dest_lmul: m1
// HEADER: tianchenrv.rvv.dequantization_relation: signed-i32m1-to-f32m1-scale-f32
// HEADER: tianchenrv.rvv.dequantize_convert_intrinsic: __riscv_vfcvt_f_x_v_f32m1
// HEADER: tianchenrv.rvv.dequantize_scale_intrinsic: __riscv_vfmul_vf_f32m1
// HEADER: tianchenrv.rvv.dequant_scale_role: dequant-scale-value
// HEADER: tianchenrv.rvv.dequant_scale_c_type: float
// HEADER: tianchenrv.rvv.dequant_scale_name: scale
// HEADER: tianchenrv.rvv.gearbox_candidate_set: rvv-gearbox-candidate-set.v1[rvv-gearbox-dequantize-i32-to-f32-e32-m1-u1.v1,rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1]
// HEADER: tianchenrv.rvv.gearbox_selected_candidate: rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1
// HEADER: tianchenrv.rvv.gearbox_selection_reason: select-bounded-u2-two-slice-route-plan-for-typed-dequantize-i32-to-f32-e32-m1-runtime-avl
// HEADER: tianchenrv.rvv.gearbox_legality_scope: typed-dequantize-i32-to-f32-sew32-lmul-m1-runtime-avl
// HEADER: tianchenrv.rvv.gearbox_schedule_id: rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1
// HEADER: tianchenrv.rvv.gearbox_selector: static-dequantize-i32-to-f32-e32-m1-u2
// HEADER: tianchenrv.rvv.gearbox_source: rvv-gearbox-static-pass.v1
// HEADER: tianchenrv.rvv.gearbox_operation: dequantize_i32_to_f32
// HEADER: tianchenrv.rvv.gearbox_unroll: 2
// HEADER: tianchenrv.rvv.gearbox_vl_policy: runtime-avl-two-slice-setvl
// HEADER: tianchenrv.rvv.gearbox_source_sew: 32
// HEADER: tianchenrv.rvv.gearbox_source_lmul: m1
// HEADER: tianchenrv.rvv.gearbox_dest_sew: 32
// HEADER: tianchenrv.rvv.gearbox_dest_lmul: m1
// HEADER: tianchenrv.rvv.gearbox_runtime_avl_source: runtime_abi:n
// HEADER: tianchenrv.rvv.target_leaf_profile: rvv-v1-i32m1-f32m1-runtime-scale-dequantization-leaf-profile.v1
// HEADER: tianchenrv.rvv.runtime_control_plan: rvv-runtime-avl-vl-control-plan.v1
// HEADER: tianchenrv.rvv.provider_supported_mirror: provider_supported_mirror:rvv-dequantize-i32-to-f32-runtime-scale-plan-validated
// HEADER: tianchenrv.rvv.route_operand_binding_plan: rvv-route-operand-binding:dequantize_i32_to_f32.v1
// HEADER: tianchenrv.rvv.dequantization_route_family_plan: rvv-dequantization-route-family-plan.v1
// HEADER: tianchenrv.rvv.required_header_declarations: stddef.h,stdint.h,riscv_vector.h
// HEADER: tianchenrv.rvv.c_type_mapping: vl:size_t,source:signed-e32m1,converted/scaled:float-e32m1,scale:float
// HEADER: void tcrv_emitc_explicit_selected_body_dequantize_i32_to_f32_kernel_explicit_selected_body_rvv_dequantize_i32_to_f32(const int32_t *lhs, float scale, float *out, size_t n);

// MISSING-GEARBOX: requires pass-produced RVV Gearbox candidate-selection fact 'tcrv_rvv.gearbox.candidate_set'
// MISSING-GEARBOX-SAME: before provider route construction

// BAD-GEARBOX-CANDIDATE-SET: selected RVV Gearbox candidate
// BAD-GEARBOX-CANDIDATE-SET-SAME: belong to pass-produced legal candidate set

// UNSUPPORTED-GEARBOX-UNROLL: requires RVV Gearbox schedule fact 'tcrv_rvv.gearbox.unroll'
// UNSUPPORTED-GEARBOX-UNROLL-SAME: provider-derived '2' but found '3'

// STALE-RELATION: target artifact candidate validation failed
// STALE-RELATION-SAME: candidate tcrv_rvv.dequantization_relation provenance must mirror selected typed RVV dequantization relation 'signed-i32m1-to-f32m1-scale-f32' but was 'script-derived-dequant'

// STALE-SCALE: target artifact candidate validation failed
// STALE-SCALE-SAME: candidate tcrv_rvv.dequant_scale_role provenance must mirror selected typed RVV dequantization scale ABI role 'dequant-scale-value' but was 'output-buffer'

// STALE-DEST-LMUL: target artifact candidate validation failed
// STALE-DEST-LMUL-SAME: candidate tcrv_rvv.dest_lmul provenance must mirror selected typed RVV conversion dtype-policy destination LMUL 'm1' but was 'm2'

// STALE-GEARBOX-SELECTED: target artifact candidate validation failed
// STALE-GEARBOX-SELECTED-SAME: candidate tcrv_rvv.gearbox.selected_candidate provenance must mirror selected typed RVV dequantization Gearbox selected candidate 'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but was 'artifact-name-derived-gear'

// STALE-GEARBOX: target artifact candidate validation failed
// STALE-GEARBOX-SAME: candidate tcrv_rvv.gearbox.schedule_id provenance must mirror selected typed RVV dequantization Gearbox schedule id 'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but was 'artifact-name-derived-gear'
