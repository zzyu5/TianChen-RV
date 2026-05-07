// RUN: not tcrv-translate --tcrv-export-rvv-microkernel-c %s 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @control_body_policy_mismatch {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
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
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected RVV microkernel path",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv_rvv.lowering_boundary {
      capability_summary = "rvv",
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "control_body_policy_mismatch",
      status = "unsupported",
      unsupported_reason = "RVV lowering boundary is pre-executable metadata only"
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "direct variant",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "control_body_policy_mismatch"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = undisturbed, mask = agnostic>, sew = 32 : i64} {
        tcrv_rvv.i32_vadd_dataflow {lhs = "lhs", out = "out", rhs = "rhs", runtime_n = "n"}
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK: control-plane setvl policy must match selected variant tcrv_rvv.policy metadata
