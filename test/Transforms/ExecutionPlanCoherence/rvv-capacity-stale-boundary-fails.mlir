// RUN: not tcrv-opt %s --verify-diagnostics --tcrv-check-execution-plan-coherence 2>&1 | FileCheck %s --check-prefix=NEG

module {
  tcrv.exec.kernel @stale_rvv_capacity_boundary {
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
    tcrv.exec.capability @rvv_vlenb_bytes {
      id = "rvv.vlenb_bytes",
      kind = "uarch",
      bytes = 16 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lanes {
      id = "rvv.i32_m1_lane_count",
      kind = "uarch",
      lanes = 4 : i64,
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
      value = "rv64gcv",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.vlenb_bytes = 16 : i64,
      tcrv_rvv.i32_m1_lanes = 4 : i64
    } {
    }
    tcrv.exec.diagnostic {
      reason = "variant-selected",
      message = "static RVV selection for stale-boundary negative",
      severity = "info",
      status = "selected",
      selection_kind = "static-variant",
      target = @rvv_first_slice,
      origin = "rvv-plugin"
    }
    tcrv_rvv.lowering_boundary {
      source_kernel = "stale_rvv_capacity_boundary",
      selected_variant = @rvv_first_slice,
      origin = "rvv-plugin",
      role = "direct variant",
      status = "unsupported",
      required_capabilities = [@rvv],
      vlenb_bytes = 32 : i64,
      i32_m1_lanes = 8 : i64,
      capability_summary = "rvv",
      unsupported_reason = "RVV lowering boundary is pre-executable metadata only"
    }
    tcrv.exec.diagnostic {
      reason = "emission_plan",
      message = "unsupported RVV diagnostic boundary",
      severity = "error",
      status = "unsupported",
      target = @rvv_first_slice,
      origin = "rvv-plugin",
      role = "direct variant",
      plan_kind = "plugin-emission-plan",
      runtime_abi_kind = "rvv-plugin-deferred-runtime-abi",
      runtime_abi_name = "rvv-executable-runtime-abi-deferred",
      runtime_glue_role = "deferred-rvv-runtime-glue",
      required_capabilities = [@rvv]
    }
  }
}

// NEG: failed lowering-boundary validation
// NEG: capacity metadata does not match the selected variant
