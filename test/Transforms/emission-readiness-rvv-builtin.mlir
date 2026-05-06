// RUN: not tcrv-opt %s --tcrv-check-emission-paths 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @public_rvv_readiness {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
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
  }
}

// CHECK: TianChen-RV variant emission readiness check failed
// CHECK-SAME: variant @rvv_first_slice in kernel @public_rvv_readiness as direct variant
// CHECK: origin plugin 'rvv-plugin' reported unsupported emission path
// CHECK: RVV metadata-only first slice has no RVV lowering
// CHECK-NOT: unknown origin plugin 'rvv-plugin'
