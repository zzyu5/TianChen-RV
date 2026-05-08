// RUN: not tcrv-opt %s --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @pipeline_coherence_ambiguous_rvv_offload {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      conflicts = ["build.policy.no_rvv"],
      status = "available"
    }
    tcrv.exec.capability @no_rvv_policy {
      id = "generic.build.profile.rvv",
      kind = "build-policy",
      provides = ["build.policy.no_rvv"],
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
    tcrv.exec.capability @offload_runtime {
      id = "offload.runtime",
      kind = "runtime-offload",
      status = "available",
      runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
      conflicts = ["build.policy.no_offload"],
      handoff_kind = "runtime-offload"
    }
    tcrv.exec.capability @no_offload_policy {
      id = "generic.build.profile.offload",
      kind = "build-policy",
      provides = ["build.policy.no_offload"],
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: error: TianChen-RV execution plan coherence check failed for kernel @pipeline_coherence_ambiguous_rvv_offload
// CHECK-SAME: requires at most one supported target artifact emission-plan route
// CHECK-SAME: found multiple ambiguous supported artifacts without a registered composite route
// CHECK-SAME: @rvv_first_slice as dispatch case route 'tcrv-export-rvv-microkernel-c' artifact_kind 'runtime-callable-c-source'
// CHECK-SAME: @offload_runtime_first_slice as dispatch case route 'tcrv-export-offload-runtime-descriptor' artifact_kind 'runtime-offload-handoff-descriptor'
