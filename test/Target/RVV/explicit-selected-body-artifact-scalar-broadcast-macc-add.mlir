// RUN: weft-opt %s --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Stage2 multi-family selected-body input. The RHS vector is produced by a
// typed runtime scalar splat and consumed by typed weft_rvv.macc in the same
// selected RVV body; route authority must come from body/config/runtime facts.

module {
  weft.exec.target @rvv_profile {id = "rvv.profile.rv64gcv", target_kind = "profile", status = "available", construction_domain = "riscv-execution", relations = #weft.capability_relations<provides = ["rvv"]>, architecture = "riscv64", isa_vector_hints = "rv64gcv_zvl128b", supported_lmul = "m1", supported_sew = "32"}
  weft.exec.kernel @explicit_selected_body_scalar_broadcast_macc_add_kernel attributes {target = @rvv_profile} {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.variant @explicit_selected_body_rvv_scalar_broadcast_macc_add attributes {origin = "rvv-plugin", requires = [@rvv_profile], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, weft_rvv.require_exec_abi_bindings = true} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-macc:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-macc:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-macc:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-macc:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "explicit-selected-body-scalar-broadcast-macc:n", role = "runtime-element-count"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %a = weft_rvv.load %lhs, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %b = weft_rvv.splat %rhs_scalar, %vl : i32, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %acc_vec = weft_rvv.load %acc, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        %sum = weft_rvv.macc %a, %b, %acc_vec, %vl {accumulator_layout = "separate-i32-vector-accumulator-input", kind = "add", result_layout = "store-multiply-accumulate-result-to-output-buffer"} : !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
        weft_rvv.store %out, %sum, %vl : !weft_rvv.runtime_abi_value, !weft_rvv.vector<i32, "m1">, !weft_rvv.vl
      } : !weft_rvv.vl
    }
    weft.exec.variant @explicit_selected_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @explicit_selected_body_rvv_scalar_broadcast_macc_add {origin = "rvv-plugin", policy = "explicit-selected-body-scalar-broadcast-macc-case"}
      weft.exec.fallback @explicit_selected_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "explicit-selected-body-scalar-broadcast-macc-fallback-envelope"}
    }
  }
}

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @explicit_selected_body_rvv_scalar_broadcast_macc_add

// HEADER: weft.rvv.selected_variant: @explicit_selected_body_rvv_scalar_broadcast_macc_add
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_explicit_selected_body_scalar_broadcast_macc_add_kernel_explicit_selected_body_rvv_scalar_broadcast_macc_add(const int32_t *lhs, int32_t rhs_scalar, const int32_t *acc, int32_t *out, size_t n);
