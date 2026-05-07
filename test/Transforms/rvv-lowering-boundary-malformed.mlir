// RUN: not tcrv-opt %s --tcrv-materialize-rvv-lowering-boundary 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_missing_policy_for_boundary {
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

// CHECK: TianChen-RV RVV lowering-boundary materialization failed
// CHECK-SAME: selected RVV variant @rvv_first_slice failed plugin legality
// CHECK: tcrv_rvv.policy
