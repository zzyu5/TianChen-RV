// RUN: tcrv-opt %s --split-input-file --tcrv-verify-plugin-variant-legality | FileCheck %s

module {
  // CHECK-LABEL: tcrv.exec.kernel @legality_valid_scalar
  tcrv.exec.kernel @legality_valid_scalar {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
    // CHECK: tcrv.exec.variant @scalar_fallback_first_slice
    // CHECK-SAME: origin = "scalar-plugin"
    // CHECK-SAME: requires = [@scalar_fallback]
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback],
      policy = "portable_scalar_fallback_first_slice"
    } {
    }
  }
}

// -----

module {
  // CHECK-LABEL: tcrv.exec.kernel @legality_valid_rvv
  tcrv.exec.kernel @legality_valid_rvv {
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
    // CHECK: tcrv.exec.variant @rvv_first_slice
    // CHECK-SAME: origin = "rvv-plugin"
    // CHECK-SAME: requires = [@rvv]
    // CHECK-SAME: tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    // CHECK-SAME: tcrv_rvv.required_march = "rv64gcv"
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
  }
}
