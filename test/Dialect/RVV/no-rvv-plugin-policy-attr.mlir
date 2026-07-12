// RUN: not weft-opt %s --weft-disable-builtin-plugins 2>&1 | FileCheck %s

// CHECK: weft_rvv
module {
  weft.exec.kernel @rvv_policy_without_plugin attributes {} {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector"}
    weft.exec.variant @rvv_typed_policy_holder attributes {
      origin = "rvv-plugin",
      requires = [@rvv],
      weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>
    } {
    }
  }
}
