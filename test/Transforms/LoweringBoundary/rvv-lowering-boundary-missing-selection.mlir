// RUN: not tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_without_selected_surface {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_unselected_explicit_placeholder attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}

// CHECK: requires selected tcrv.exec.dispatch or direct selected-path diagnostic
