// RUN: not tcrv-opt %s --tcrv-check-emission-paths 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @public_rvv_readiness {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    tcrv.exec.diagnostic {
      message = "select RVV first-slice metadata path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_first_slice
    }
    tcrv_rvv.lowering_boundary {
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      role = "direct variant",
      selected_variant = @rvv_first_slice,
      source_kernel = "public_rvv_readiness",
      status = "unsupported",
      unsupported_reason = "unsupported RVV pre-executable boundary metadata only"
    }
  }
}

// CHECK: TianChen-RV variant emission readiness check failed
// CHECK-SAME: variant @rvv_first_slice in kernel @public_rvv_readiness as direct variant
// CHECK: origin plugin 'rvv-plugin' reported unsupported emission path
// CHECK: RVV first slice has no materialized EmitC lowering
// CHECK-NOT: unknown origin plugin 'rvv-plugin'
