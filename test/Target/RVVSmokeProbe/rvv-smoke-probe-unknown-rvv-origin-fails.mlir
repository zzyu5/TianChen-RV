// RUN: not tcrv-translate --tcrv-export-rvv-smoke-probe-c %s 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>"

module {
  tcrv.exec.kernel @unknown_rvv_origin {
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
    tcrv.exec.variant @rvv_manual attributes {
      origin = "rvv-plugin-experimental",
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
      target = @rvv_manual
    }
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_manual,
      source_kernel = "unknown_rvv_origin",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// CHECK: selected RVV-like path @rvv_manual uses unknown origin
// CHECK-SAME: only accepts registered origin 'rvv-plugin'
