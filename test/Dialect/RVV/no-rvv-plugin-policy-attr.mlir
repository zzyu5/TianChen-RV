// RUN: not tcrv-opt %s 2>&1 | FileCheck %s

// CHECK: tcrv_rvv
module {
  tcrv.exec.kernel @rvv_policy_without_plugin attributes {} {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_first_slice attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      policy = "metadata_only_first_slice",
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
