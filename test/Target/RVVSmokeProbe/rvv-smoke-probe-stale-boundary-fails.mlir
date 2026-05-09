// RUN: not tcrv-translate --tcrv-export-rvv-smoke-probe-c %s 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>"

module {
  tcrv.exec.kernel @stale_boundary {
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
    tcrv.exec.variant @rvv_selected attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.variant @rvv_old attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.diagnostic {
      message = "selected RVV metadata fixture",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_selected
    }
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_old,
      source_kernel = "stale_boundary",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// CHECK: stale tcrv_rvv.lowering_boundary for @rvv_old as direct variant
// CHECK-SAME: is not selected by the current RVV smoke surface
