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

// CHECK: formula construction failed before emission planning
// CHECK-SAME: formula construction cannot bind unknown origin 'mock-emitter'
