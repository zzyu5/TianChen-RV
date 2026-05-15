// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants | FileCheck %s --check-prefix=MAT
// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants --tcrv-check-capability-requires --tcrv-verify-plugin-variant-legality --tcrv-select-variants | FileCheck %s --check-prefix=PIPE
// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants --tcrv-materialize-plugin-variants | FileCheck %s --check-prefix=RERUN

module {
  // MAT-LABEL: tcrv.exec.kernel @scalar_only
  tcrv.exec.kernel @scalar_only {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: fallback_role = "conservative"
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: policy = "portable_scalar_fallback_first_slice"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @scalar_only
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "fallback-only"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @scalar_only
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // MAT-LABEL: tcrv.exec.kernel @rvv_capability_with_scalar_fallback
  tcrv.exec.kernel @rvv_capability_with_scalar_fallback {
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
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // MAT-NOT: origin = "rvv-plugin"
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @rvv_capability_with_scalar_fallback
    // PIPE-NOT: origin = "rvv-plugin"
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: selection_kind = "fallback-only"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @rvv_capability_with_scalar_fallback
    // RERUN-NOT: origin = "rvv-plugin"
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}
