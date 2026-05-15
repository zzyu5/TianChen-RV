// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins 2>&1 | FileCheck %s

// CHECK: tcrv_rvv
module {
  tcrv.exec.kernel @rvv_policy_without_plugin attributes {} {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    tcrv.exec.variant @rvv_typed_policy_holder attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
