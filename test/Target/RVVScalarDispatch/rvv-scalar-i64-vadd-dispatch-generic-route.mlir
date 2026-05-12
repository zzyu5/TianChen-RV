// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=int32_t --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=GENERIC --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i32 --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=GENERIC-HDR --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not="out[index]" --implicit-check-not=int32_t --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i64-vadd-dispatch-self-check-c | FileCheck %s --check-prefix=SELF --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i32 --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c 2>&1 | FileCheck %s --check-prefix=I32-ROUTE-MISMATCH

module {
  tcrv.exec.kernel @conflict_planned_i64_dispatch attributes {
    tcrv_frontend_lowering = "i64-vadd"
  } {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i64_m1.sew64", "rvv.i64_m1.lmul_m1", "rvv.i64_m1.tail_policy.agnostic", "rvv.i64_m1.mask_policy.agnostic"],
      sew_bits = 64 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      selected_mabi = "lp64d",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @host_policy_no_vector {
      id = "host.policy.no-vector",
      kind = "toolchain",
      conflicts = ["rvv"],
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// IR: tcrv.exec.variant @rvv_first_slice
// IR-SAME: origin = "rvv-plugin"
// IR-NOT: tcrv_rvv.lowering_descriptor
// IR: tcrv.exec.variant @scalar_fallback_first_slice
// IR-SAME: origin = "scalar-plugin"
// IR-NOT: tcrv_scalar.lowering_descriptor
// IR: tcrv.exec.runtime_param @abi_dispatch_availability_guard
// IR-SAME: abi_role = "dispatch-availability-guard"
// IR-SAME: c_name = "rvv_available"
// IR-SAME: c_type = "int"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR-SAME: runtime_guard_required = true
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv.exec.mem_window @abi_lhs_input_buffer
// IR-SAME: c_type = "const int64_t *"
// IR: tcrv.exec.mem_window @abi_output_buffer
// IR-SAME: c_type = "int64_t *"
// IR: tcrv_rvv.lowering_boundary
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: selected_vector_shape = "i64m1"
// IR: tcrv_scalar.lowering_boundary
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: tcrv_scalar.i64_vadd_microkernel
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: lowering_pipeline = "tcrv-export-scalar-i64-vadd-microkernel-c"
// IR-SAME: runtime_abi_name = "scalar-i64-vadd-runtime-callable-c-function.v1"
// IR-SAME: {name = "tcrv_scalar.selected_binary_family"
// IR-SAME: role = "typed-scalar-binary-source"
// IR-SAME: value = "i64-vadd"
// IR-SAME: {name = "tcrv_scalar.emitc_lowerable_op_interface"
// IR-SAME: value = "TCRVEmitCLowerableOpInterface"

// GENERIC: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// GENERIC: /* Scope: one selected RVV i64-vadd dispatch case plus one scalar i64-vadd dispatch fallback. */
// GENERIC: /* rvv_artifact_route_id: tcrv-export-rvv-i64-vadd-microkernel-c */
// GENERIC: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.selected_binary_family, value=i64-vadd, role=typed-rvv-binary-source
// GENERIC: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_source_op, value=tcrv_rvv.i64_add, role=typed-rvv-emitc-source-op
// GENERIC: /* rvv_selected_plan_metadata{{.*}}name=tcrv_rvv.emitc_lowerable_op_interface, value=TCRVEmitCLowerableOpInterface
// GENERIC: /* scalar_artifact_route_id: tcrv-export-scalar-i64-vadd-microkernel-c */
// GENERIC: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.selected_binary_family, value=i64-vadd, role=typed-scalar-binary-source
// GENERIC: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.emitc_source_op, value=tcrv_scalar.i64_vadd_microkernel, role=typed-scalar-emitc-source-op
// GENERIC: /* scalar_selected_plan_metadata{{.*}}name=tcrv_scalar.emitc_lowerable_op_interface, value=TCRVEmitCLowerableOpInterface
// GENERIC: /* dispatch_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int64_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// GENERIC: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// GENERIC: void tcrv_rvv_i64_vadd_microkernel_conflict_planned_i64_dispatch_rvv_first_slice
// GENERIC: __riscv_vadd_vv_i64m1
// GENERIC: void tcrv_scalar_i64_vadd_microkernel_conflict_planned_i64_dispatch_scalar_fallback_first_slice
// GENERIC: // tcrv_emitc.source_op=tcrv_scalar.i64_vadd_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i64_add
// GENERIC: tcrv_scalar_i64_add
// GENERIC: // tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch
// GENERIC: // tcrv_emitc.dispatch_guard_value=rvv_available
// GENERIC-LABEL: {{^}}void tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch
// GENERIC: bool [[GENERIC_GUARD:v[0-9]+]] = {{v[0-9]+}} != 0;
// GENERIC: if ([[GENERIC_GUARD]])
// GENERIC: tcrv_rvv_i64_vadd_microkernel_conflict_planned_i64_dispatch_rvv_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});
// GENERIC: return;
// GENERIC: tcrv_scalar_i64_vadd_microkernel_conflict_planned_i64_dispatch_scalar_fallback_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});

// GENERIC-HDR: #ifndef TIANCHENRV_RVV_SCALAR_I64_VADD_DISPATCH_CONFLICT_PLANNED_I64_DISPATCH_H
// GENERIC-HDR: extern "C" {
// GENERIC-HDR: void tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n, int rvv_available);
// GENERIC-HDR: #endif /* TIANCHENRV_RVV_SCALAR_I64_VADD_DISPATCH_CONFLICT_PLANNED_I64_DISPATCH_H */

// SELF: static int tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch_self_check_one(size_t runtime_n, int rvv_available)
// SELF: int64_t lhs[kCapacity];
// SELF: int64_t rhs[kCapacity];
// SELF: int64_t out[kCapacity];
// SELF: tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch(lhs, rhs, out, runtime_n, rvv_available);
// SELF: if (tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch_self_check_one(7, 0))
// SELF: if (tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch_self_check_one(16, 0))
// SELF: if (tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch_self_check_one(7, 1))
// SELF: if (tcrv_dispatch_i64_vadd_conflict_planned_i64_dispatch_self_check_one(16, 1))
// SELF: tcrv_rvv_scalar_i64_vadd_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv

// I32-ROUTE-MISMATCH: TianChen-RV RVV+scalar binary dispatch C export failed
// I32-ROUTE-MISMATCH-SAME: self-check export route expected i32-vadd dispatch artifacts, got i64-vadd
