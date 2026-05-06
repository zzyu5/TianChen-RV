// RUN: not tcrv-opt %s --tcrv-materialize-emission-plans 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @public_unknown_origin {
    tcrv.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    tcrv.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
  }
}

// CHECK: TianChen-RV variant emission plan collection failed
// CHECK-SAME: variant @fast in kernel @public_unknown_origin as direct variant
// CHECK: unknown origin plugin 'mock-emitter'
