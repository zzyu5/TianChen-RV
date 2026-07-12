// RUN: not weft-opt %s --weft-materialize-emission-plans 2>&1 | FileCheck %s

module {
  weft.exec.kernel @public_unknown_origin {
    weft.exec.capability @base {
      id = "generic.base",
      kind = "generic"
    }
    weft.exec.variant @fast attributes {
      origin = "mock-emitter",
      requires = [@base]
    } {
    }
  }
}

// CHECK: Weft-RV variant emission plan collection failed
// CHECK-SAME: variant @fast in kernel @public_unknown_origin as direct variant
// CHECK: unknown origin plugin 'mock-emitter'
