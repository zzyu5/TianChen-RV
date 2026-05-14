// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=GENERIC-DELETED --implicit-check-not="__riscv" --implicit-check-not="tcrv_scalar_i32_add"
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=GENERIC-HDR-DELETED --implicit-check-not="__riscv" --implicit-check-not="out[index]"

module {
  tcrv.exec.kernel @conflict_planned_dispatch {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
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

// IR: tcrv.exec.runtime_param @abi_dispatch_availability_guard
// IR-SAME: abi_role = "dispatch-availability-guard"
// IR-SAME: c_name = "rvv_available"
// IR-SAME: c_type = "int"
// IR: tcrv.exec.dispatch
// IR: tcrv.exec.case @rvv_first_slice
// IR-SAME: policy = "metadata_only_first_slice"
// IR-SAME: runtime_guard = @abi_dispatch_availability_guard
// IR-SAME: runtime_guard_required = true
// IR: tcrv.exec.fallback @scalar_fallback_first_slice
// IR: tcrv.exec.mem_window @abi_lhs_input_buffer
// IR-SAME: abi_role = "lhs-input-buffer"
// IR: tcrv.exec.mem_window @abi_rhs_input_buffer
// IR-SAME: abi_role = "rhs-input-buffer"
// IR: tcrv.exec.mem_window @abi_output_buffer
// IR-SAME: abi_role = "output-buffer"
// IR: tcrv.exec.runtime_param @abi_runtime_element_count
// IR-SAME: abi_role = "runtime-element-count"
// IR: tcrv_rvv.lowering_boundary
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR: tcrv_scalar.lowering_boundary
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: message = "runtime-callable RVV direct C source exporter was deleted; rebuild requires a materialized MLIR EmitC module source route"
// IR-SAME: role = "dispatch case"
// IR-SAME: status = "unsupported"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: message = "runtime-callable scalar direct C source exporter was deleted; rebuild requires a materialized MLIR EmitC module source route"
// IR-SAME: role = "dispatch fallback"
// IR-SAME: status = "unsupported"
// IR-SAME: target = @scalar_fallback_first_slice

// GENERIC-DELETED: requires exactly one supported source artifact emission-plan route; found none
// GENERIC-HDR-DELETED: requires exactly one supported header artifact emission-plan route; found none
