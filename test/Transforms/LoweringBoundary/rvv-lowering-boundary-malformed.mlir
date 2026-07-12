// RUN: not weft-opt %s --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  weft.exec.kernel @rvv_missing_policy_for_boundary {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.variant @rvv_missing_typed_body attributes {
      origin = "rvv-plugin",
      requires = [@rvv]
    } {
    }
    weft.exec.diagnostic {
      message = "select RVV path without typed body",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_missing_typed_body
    }
  }
}

// CHECK: Weft-RV selected lowering-boundary materialization failed
// CHECK-SAME: origin plugin 'rvv-plugin' failed lowering-boundary materialization
// CHECK: materialized RVV variant requires explicit typed RVV extension-family body
