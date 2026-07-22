// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

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
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_dequantize_i32_to_f32

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_dequantize_i32_to_f32
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_explicit_selected_body_dequantize_i32_to_f32_kernel_explicit_selected_body_rvv_dequantize_i32_to_f32(const int32_t *lhs, float scale, float *out, size_t n);

// MISSING-GEARBOX-SAME: before provider route construction

// BAD-GEARBOX-CANDIDATE-SET: selected RVV Gearbox candidate
// BAD-GEARBOX-CANDIDATE-SET-SAME: belong to pass-produced legal candidate set

// UNSUPPORTED-GEARBOX-UNROLL-SAME: provider-derived '2' but found '3'




// STALE-GEARBOX-SELECTED: metadata key '{{.*}}gearbox.selected_candidate'{{.*}}'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but was 'artifact-name-derived-gear'

// STALE-GEARBOX: metadata key '{{.*}}gearbox.schedule_id'{{.*}}'rvv-gearbox-dequantize-i32-to-f32-e32-m1-u2.v1' but was 'artifact-name-derived-gear'
