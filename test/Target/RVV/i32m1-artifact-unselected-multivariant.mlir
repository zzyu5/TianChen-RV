// RUN: not tcrv-translate --tcrv-export-target-artifact %s 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @rvv_unselected_multivariant {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
    tcrv.exec.variant @rvv_i32_add_a attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
    tcrv.exec.variant @rvv_i32_add_b attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}

// CHECK: requires a selected dispatch or selected-path diagnostic surface before execution-plan coherence checking
