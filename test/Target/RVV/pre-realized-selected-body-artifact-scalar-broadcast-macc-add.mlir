// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

// Pre-realized scalar-broadcast macc selected-body input. The RVV plugin must
// consume explicit typed operation/config/runtime facts into load/splat/load/
// macc/store structure before the provider route/common EmitC/target path can
// construct scalar_broadcast_macc_add route facts.

module {
  weft.exec.kernel @pre_realized_body_scalar_broadcast_macc_add_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.mem_window @abi_lhs_input_buffer {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_rhs_scalar_value {abi_role = "rhs-scalar-value", c_name = "rhs_scalar", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.mem_window @abi_accumulator_input_buffer {abi_role = "accumulator-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.mem_window @abi_output_buffer {abi_role = "output-buffer", access = "write", binding = "kernel-argument", c_type = "int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}
    weft.exec.runtime_param @abi_runtime_element_count {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}
    weft.exec.variant @pre_realized_body_rvv_scalar_broadcast_macc_add attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, weft_rvv.require_exec_abi_bindings = true} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", exec_binding = @abi_lhs_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs_scalar = weft_rvv.runtime_abi_value {c_name = "rhs_scalar", c_type = "int32_t", exec_binding = @abi_rhs_scalar_value, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:rhs-scalar", role = "rhs-scalar-value"} : i32
      %acc = weft_rvv.runtime_abi_value {c_name = "acc", c_type = "const int32_t *", exec_binding = @abi_accumulator_input_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:accumulator", role = "accumulator-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", exec_binding = @abi_output_buffer, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", exec_binding = @abi_runtime_element_count, ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-scalar-broadcast-macc:n", role = "runtime-element-count"} : index
      weft_rvv.typed_macc_pre_realized_body %lhs, %rhs_scalar, %acc, %out, %n {accumulator_layout = "separate-i32-vector-accumulator-input", accumulator_role = "accumulator-input-buffer", lmul = "m1", memory_form = "rhs-scalar-broadcast-macc", op_kind = "scalar_broadcast_macc_add", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, result_layout = "store-multiply-accumulate-result-to-output-buffer", sew = 32 : i64} : (!weft_rvv.runtime_abi_value, i32, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_scalar_broadcast_macc_add {origin = "rvv-plugin", policy = "pre-realized-selected-body-scalar-broadcast-macc-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-scalar-broadcast-macc-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_macc_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.splat
// REALIZED: %[[ACC:.*]] = weft_rvv.load
// REALIZED: %[[SUM:.*]] = weft_rvv.macc %[[LHS]], %[[RHS]], %[[ACC]], %[[VL]]
// REALIZED-SAME: accumulator_layout = "separate-i32-vector-accumulator-input"
// REALIZED-SAME: kind = "add"
// REALIZED-SAME: result_layout = "store-multiply-accumulate-result-to-output-buffer"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_macc_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: runtime_abi_name = "rvv-exact-typed-body-callable-c-abi.v2"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_scalar_broadcast_macc_add

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_scalar_broadcast_macc_add
// HEADER: weft.rvv.runtime_abi_name: rvv-exact-typed-body-callable-c-abi.v2
// HEADER: void weft_emitc_pre_realized_body_scalar_broadcast_macc_add_kernel_pre_realized_body_rvv_scalar_broadcast_macc_add(const int32_t *lhs, int32_t rhs_scalar, const int32_t *acc, int32_t *out, size_t n);





