// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-verify-plugin-variant-legality

module {
  // expected-error@+1 {{TianChen-RV variant legality verification failed for variant @unknown_path in kernel @legality_unknown_origin: unknown origin plugin 'missing-plugin'}}
  tcrv.exec.kernel @legality_unknown_origin {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @unknown_path attributes {
      origin = "missing-plugin",
      requires = [@scalar_fallback]
    } {
    }
  }
}

// -----

module {
  // expected-error@+1 {{origin plugin 'rvv-plugin' rejected variant}}
  tcrv.exec.kernel @legality_rvv_missing_policy {
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
    tcrv.exec.variant @rvv_missing_policy attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
  }
}

// -----

module {
  // expected-error@+1 {{'tcrv_rvv.required_march' metadata is not satisfied by preserved capability property 'selected_march'}}
  tcrv.exec.kernel @legality_rvv_march_conflict {
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
    tcrv.exec.variant @rvv_bad_march attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gc"
    } {
    }
  }
}

// -----

module {
  // expected-error@+1 {{materialized RVV variant requires an available capability id 'rvv'}}
  tcrv.exec.kernel @legality_rvv_without_rvv_capability {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    tcrv.exec.variant @rvv_without_rvv_capability attributes {
      origin = "rvv-plugin",
      requires = [@scalar_fallback],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
