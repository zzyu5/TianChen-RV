// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=DELETED --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=runtime_success --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_manifest_input {
  tcrv.exec.kernel @rvv_microkernel_manifest {
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
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV microkernel path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
  }
}

// DELETED: emission_kind = "rvv-unsupported-metadata-boundary"
// DELETED-SAME: lowering_pipeline = "rvv-none-executable-unsupported"
// DELETED-SAME: message = "RVV metadata-only first slice has no RVV lowering pipeline, runtime ABI, artifact contract, or executable emission path; this unsupported emission plan is a plugin-owned diagnostic boundary and not RVV hardware/toolchain/runtime/correctness/performance evidence"
// DELETED-SAME: reason = "emission_plan"
// DELETED-SAME: runtime_abi_kind = "rvv-plugin-deferred-runtime-abi"
// DELETED-SAME: runtime_abi_name = "rvv-executable-runtime-abi-deferred"
// DELETED-SAME: runtime_glue_role = "deferred-rvv-runtime-glue"
// DELETED-SAME: status = "unsupported"
