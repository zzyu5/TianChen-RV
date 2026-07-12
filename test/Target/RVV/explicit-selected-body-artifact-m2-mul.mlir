// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @explicit_selected_body_m2_mul_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @explicit_selected_body_rvv_i32m2_mul attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-m2:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-m2:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-m2:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "explicit-selected-body-m2:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m2", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m2", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @explicit_selected_body_rvv_i32m2_mul, sew = 32 : i64, source_kernel = "explicit_selected_body_m2_mul_kernel", status = "selected-lowering-boundary"} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
        %b = weft_rvv.load %rhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
        %product = weft_rvv.binary %a, %b, %vl {kind = "mul"} : !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">
        weft_rvv.store %out, %product, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m2">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_i32m2_mul {origin = "rvv-plugin", policy = "explicit-selected-body-m2-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "mul"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.binary"}
// PLAN-SAME: {key = "weft_rvv.config_contract", value = "rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1"}
// PLAN-SAME: {key = "weft_rvv.lmul", value = "m2"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.bounded_slice", value = "multi-vl-selected-body-sew32-lmul-m2"}
// PLAN-SAME: emission_kind = "materialized-emitc-cpp-rvv-intrinsic-object"
// PLAN-SAME: lowering_boundary = "weft_rvv.with_vl"
// PLAN-SAME: origin = "rvv-plugin"
// PLAN-SAME: reason = "emission_plan"
// PLAN-SAME: role = "dispatch case"
// PLAN-SAME: runtime_abi_name = "rvv-generic-binary-mul-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_i32m2_mul

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_i32m2_mul
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-binary-mul-callable-c-abi.v1
// HEADER: weft.rvv.config_contract: rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1
// HEADER: weft.rvv.lmul: m2
// HEADER: weft.rvv.bounded_slice: multi-vl-selected-body-sew32-lmul-m2
// HEADER: void weft_emitc_explicit_selected_body_m2_mul_kernel_explicit_selected_body_rvv_i32m2_mul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
