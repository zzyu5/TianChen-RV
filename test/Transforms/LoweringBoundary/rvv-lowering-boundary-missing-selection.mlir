// RUN: not weft-opt %s --weft-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  weft.exec.kernel @rvv_without_selected_surface {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    weft.exec.variant @rvv_unselected_explicit_placeholder attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}

// CHECK: requires selected weft.exec.dispatch or direct selected-path diagnostic
