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
// IR: tcrv_rvv.lowering_boundary {
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR: tcrv_scalar.lowering_boundary
// IR-SAME: role = "dispatch fallback"
// IR-SAME: selected_variant = @scalar_fallback_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "rvv-unsupported-metadata-boundary"
// IR-SAME: message = "RVV metadata-only first slice has no RVV lowering pipeline, runtime ABI, artifact contract, or executable emission path; this unsupported emission plan is a plugin-owned diagnostic boundary and not RVV hardware/toolchain/runtime/correctness/performance evidence"
// IR-SAME: role = "dispatch case"
// IR-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
// IR-SAME: status = "unsupported"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv.exec.diagnostic
// IR-SAME: emission_kind = "portable-scalar-fallback-metadata-route"
// IR-SAME: message = "scalar fallback first slice records a portable fallback metadata route for compiler decisions only; it does not emit objects, link a runtime, run hardware, prove correctness, or measure performance"
// IR-SAME: role = "dispatch fallback"
// IR-SAME: runtime_abi_kind = "host-scalar-fallback-metadata"
// IR-SAME: status = "metadata-only"
// IR-SAME: target = @scalar_fallback_first_slice

// GENERIC-DELETED: requires exactly one supported source artifact emission-plan route; found none
// GENERIC-HDR-DELETED: requires exactly one supported header artifact emission-plan route; found none
