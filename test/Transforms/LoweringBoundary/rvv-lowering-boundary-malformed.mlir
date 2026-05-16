// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_missing_policy_for_boundary {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_deleted_metadata_path attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    tcrv.exec.diagnostic {
      message = "select deleted RVV metadata path",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_deleted_metadata_path
    }
  }
}

// CHECK: TianChen-RV selected lowering-boundary materialization failed
// CHECK-SAME: origin plugin 'rvv-plugin' failed lowering-boundary materialization
// CHECK: explicit typed RVV extension-family body
// CHECK: metadata-only RVV first-slice route has been deleted
