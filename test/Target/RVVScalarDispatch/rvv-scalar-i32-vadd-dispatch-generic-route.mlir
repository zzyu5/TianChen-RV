// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=GENERIC --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module {
  tcrv.exec.kernel @conflict_planned_dispatch {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
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

// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: policy = "metadata_only_first_slice"
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv.exec.mem_window @abi_lhs_input_buffer
// IR-SAME: abi_role = "lhs-input-buffer"
// IR-SAME: access = "read"
// IR-SAME: c_type = "const int32_t *"
// IR: tcrv.exec.mem_window @abi_rhs_input_buffer
// IR-SAME: abi_role = "rhs-input-buffer"
// IR-SAME: access = "read"
// IR-SAME: c_type = "const int32_t *"
// IR: tcrv.exec.mem_window @abi_output_buffer
// IR-SAME: abi_role = "output-buffer"
// IR-SAME: access = "write"
// IR-SAME: c_type = "int32_t *"
// IR: tcrv_rvv.lowering_boundary
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR: tcrv_scalar.lowering_boundary
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch case"
// IR-SAME: status = "supported"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: reason = "emission_plan"
// IR-SAME: role = "dispatch fallback"
// IR-SAME: status = "supported"
// IR-SAME: target = @scalar_fallback_first_slice

// GENERIC: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// GENERIC: /* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
// GENERIC: /* selected_kernel: @conflict_planned_dispatch */
// GENERIC: /* rvv_selected_variant: @rvv_first_slice */
// GENERIC: /* rvv_selected_role: dispatch case */
// GENERIC: /* rvv_artifact_route_id: tcrv-export-rvv-microkernel-c */
// GENERIC: /* scalar_selected_variant: @scalar_fallback_first_slice */
// GENERIC: /* scalar_selected_role: dispatch fallback */
// GENERIC: /* scalar_artifact_route_id: tcrv-export-scalar-microkernel-c */
// GENERIC: /* dispatch_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// GENERIC: /* dispatch_mem_window[2]: symbol=@abi_output_buffer, abi_role=output-buffer, access=write, ownership=target-export-abi-owned, c_type=int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// GENERIC: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// GENERIC: void tcrv_rvv_i32_vadd_microkernel_conflict_planned_dispatch_rvv_first_slice
// GENERIC: __riscv_vadd_vv_i32m1
// GENERIC: void tcrv_scalar_i32_vadd_microkernel_conflict_planned_dispatch_scalar_fallback_first_slice
// GENERIC: out[index] = lhs[index] + rhs[index];
// GENERIC-LABEL: {{^}}void tcrv_dispatch_i32_vadd_conflict_planned_dispatch
// GENERIC: if (rvv_available)
// GENERIC: tcrv_rvv_i32_vadd_microkernel_conflict_planned_dispatch_rvv_first_slice(lhs, rhs, out, n);
// GENERIC: return;
// GENERIC: tcrv_scalar_i32_vadd_microkernel_conflict_planned_dispatch_scalar_fallback_first_slice(lhs, rhs, out, n);
