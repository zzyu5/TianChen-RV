// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants | FileCheck %s --check-prefix=MAT
// RUN: tcrv-opt %s --split-input-file --tcrv-materialize-plugin-variants --tcrv-check-capability-requires --tcrv-select-variants | FileCheck %s --check-prefix=PIPE
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
  // MAT-LABEL: tcrv.exec.kernel @deterministic_builtin_order
  tcrv.exec.kernel @deterministic_builtin_order {
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

    // MAT: tcrv.exec.variant @rvv_first_slice
    // MAT-SAME: condition = "rvv_capability_properties_available"
    // MAT-SAME: guard = "plugin_local_rvv_property_evidence"
    // MAT-SAME: origin = "rvv-plugin"
    // MAT-SAME: policy = "metadata_only_first_slice"
    // MAT-SAME: requires = [@rvv]
    // MAT-SAME: tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // MAT-SAME: tcrv_rvv.required_march = "rv64gcv"
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: fallback_role = "conservative"
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: policy = "portable_scalar_fallback_first_slice"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @deterministic_builtin_order
    // PIPE: tcrv.exec.variant @rvv_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE: tcrv.exec.dispatch
    // PIPE: tcrv.exec.case @rvv_first_slice
    // PIPE-SAME: condition = "rvv_capability_properties_available"
    // PIPE-SAME: guard = "plugin_local_rvv_property_evidence"
    // PIPE-SAME: policy = "metadata_only_first_slice"
    // PIPE: tcrv.exec.fallback @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @deterministic_builtin_order
    // RERUN-COUNT-1: tcrv.exec.variant @rvv_first_slice
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // MAT-LABEL: tcrv.exec.kernel @scalar_independent_from_unavailable_rvv
  tcrv.exec.kernel @scalar_independent_from_unavailable_rvv {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "missing"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // MAT-NOT: tcrv.exec.variant @rvv_first_slice
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @scalar_independent_from_unavailable_rvv
    // PIPE-NOT: tcrv.exec.variant @rvv_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @scalar_independent_from_unavailable_rvv
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}

// -----

module {
  // MAT-LABEL: tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties
  tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      status = "available"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // MAT-NOT: tcrv.exec.variant @rvv_first_slice
    // MAT: tcrv.exec.variant @scalar_fallback_first_slice
    // MAT-SAME: fallback_role = "conservative"
    // MAT-SAME: origin = "scalar-plugin"
    // MAT-SAME: policy = "portable_scalar_fallback_first_slice"
    // MAT-SAME: requires = [@scalar_fallback]

    // PIPE-LABEL: tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties
    // PIPE-NOT: tcrv.exec.variant @rvv_first_slice
    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice

    // RERUN-LABEL: tcrv.exec.kernel @scalar_preserved_after_malformed_rvv_properties
    // RERUN-NOT: tcrv.exec.variant @rvv_first_slice
    // RERUN-COUNT-1: tcrv.exec.variant @scalar_fallback_first_slice
    // RERUN-NOT: tcrv.exec.variant @scalar_fallback_first_slice
  }
}
