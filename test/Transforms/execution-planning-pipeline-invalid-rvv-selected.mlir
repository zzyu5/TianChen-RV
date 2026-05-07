// RUN: not tcrv-opt %s --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @pipeline_invalid_selected_rvv_metadata {
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
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_bad_selected attributes {
      condition = "capability_available",
      guard = "plugin_local_rvv_first_slice",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gc"
    } {
    }
  }
}

// CHECK: error: TianChen-RV variant legality verification failed for variant @rvv_bad_selected in kernel @pipeline_invalid_selected_rvv_metadata
// CHECK-SAME: origin plugin 'rvv-plugin' rejected variant
// CHECK: 'tcrv_rvv.required_march' metadata is not satisfied by preserved capability property 'selected_march'
// CHECK-NOT: TianChen-RV variant selection failed
// CHECK-NOT: selected lowering-boundary materialization
