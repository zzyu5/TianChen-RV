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

// CHECK: Weft-RV emission path check failed for kernel @public_unknown_origin:
// CHECK-SAME: variant emission plan collection failed during family construction
// CHECK-SAME: for variant @fast as direct variant:
// CHECK-SAME: selected owner construction cannot resolve unknown origin 'mock-emitter'
