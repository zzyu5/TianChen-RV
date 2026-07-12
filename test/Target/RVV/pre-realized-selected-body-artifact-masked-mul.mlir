// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REALIZED
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | FileCheck %s --check-prefix=PLAN
// RUN: weft-opt %s --weft-materialize-selected-lowering-boundaries --weft-materialize-emission-plans | weft-translate --weft-export-target-header-artifact | FileCheck %s --check-prefix=HEADER

module {
  weft.exec.kernel @pre_realized_body_masked_mul_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @pre_realized_body_rvv_masked_mul attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %lhs = weft_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-mul:lhs", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %rhs = weft_rvv.runtime_abi_value {c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-mul:rhs", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %out = weft_rvv.runtime_abi_value {c_name = "out", c_type = "int32_t *", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-mul:out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "pre-realized-selected-body-masked-mul:n", role = "runtime-element-count"} : index
      weft_rvv.typed_masked_binary_pre_realized_body %lhs, %rhs, %out, %n {lmul = "m1", mask_source = "compare-produced-mask-same-vl-scope", masked_passthrough = "passthrough-vector-preserves-inactive-lanes", memory_form = "masked-vector-rhs-load", op_kind = "masked_mul", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : (!weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index) -> ()
    }
    weft.exec.variant @pre_realized_body_scalar_fallback attributes {fallback_role = "conservative", origin = "scalar-plugin", policy = "portable_scalar_fallback_first_slice", requires = [@scalar_fallback]} {
    }
    weft.exec.dispatch {
      weft.exec.case @pre_realized_body_rvv_masked_mul {origin = "rvv-plugin", policy = "pre-realized-selected-body-masked-mul-case"}
      weft.exec.fallback @pre_realized_body_scalar_fallback {fallback_role = "conservative", origin = "scalar-plugin", policy = "pre-realized-selected-body-masked-mul-fallback-envelope"}
    }
  }
}

// REALIZED-NOT: weft_rvv.typed_masked_binary_pre_realized_body
// REALIZED: %[[VL:.*]] = weft_rvv.setvl %{{.*}} {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64}
// REALIZED: weft_rvv.with_vl %[[VL]] attributes
// REALIZED-SAME: selected_variant = @pre_realized_body_rvv_masked_mul
// REALIZED: %[[LHS:.*]] = weft_rvv.load
// REALIZED: %[[RHS:.*]] = weft_rvv.load
// REALIZED: %[[MASK:.*]] = weft_rvv.compare %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "eq"
// REALIZED: weft_rvv.masked_binary %[[MASK]], %[[LHS]], %[[LHS]], %[[RHS]], %[[VL]]
// REALIZED-SAME: kind = "mul"
// REALIZED: weft_rvv.store
// REALIZED-NOT: weft_rvv.typed_masked_binary_pre_realized_body

// PLAN: weft.exec.diagnostic
// PLAN-SAME: artifact_kind = "riscv-elf-relocatable-object"
// PLAN-SAME: {key = "rvv_selected_body_operation", value = "masked_mul"}
// PLAN-SAME: {key = "rvv_selected_body_typed_compute_op", value = "weft_rvv.masked_binary"}
// PLAN-SAME: {key = "weft_rvv.memory_form", value = "vector-rhs-load"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_plan", value = "rvv-route-operand-binding:masked_mul.v1"}
// PLAN-SAME: {key = "weft_rvv.route_operand_binding_operands", value = "rvv-route-operand-binding:masked_mul.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|masked-mul-lhs-call|masked-merge-passthrough-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|masked-mul-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header"}
// PLAN-SAME: {key = "weft_rvv.mask_role", value = "predicate-mask-produced-by-compare"}
// PLAN-SAME: {key = "weft_rvv.mask_source", value = "compare-produced-mask-same-vl-scope"}
// PLAN-SAME: {key = "weft_rvv.inactive_lane_contract", value = "masked-off-lanes-preserve-passthrough-vector"}
// PLAN-SAME: {key = "weft_rvv.masked_passthrough_layout", value = "passthrough-vector-preserves-inactive-lanes"}
// PLAN-SAME: runtime_abi_name = "rvv-generic-masked-mul-callable-c-abi.v1"
// PLAN-SAME: status = "supported"
// PLAN-SAME: target = @pre_realized_body_rvv_masked_mul

// HEADER: weft.rvv.selected_variant: @pre_realized_body_rvv_masked_mul
// HEADER: weft.rvv.runtime_abi_name: rvv-generic-masked-mul-callable-c-abi.v1
// HEADER: weft.rvv.route_operand_binding_plan: rvv-route-operand-binding:masked_mul.v1
// HEADER: weft.rvv.route_operand_binding_operands: rvv-route-operand-binding:masked_mul.v1;lhs=lhs-input-buffer:lhs:abi|load-base|compare-lhs-call|masked-mul-lhs-call|masked-merge-passthrough-call;rhs=rhs-input-buffer:rhs:abi|load-base|compare-rhs-call|masked-mul-rhs-call;out=output-buffer:out:abi|store-base|header;n=runtime-element-count:n:abi|setvl-avl|loop-control|header
// HEADER: void weft_emitc_pre_realized_body_masked_mul_kernel_pre_realized_body_rvv_masked_mul(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
