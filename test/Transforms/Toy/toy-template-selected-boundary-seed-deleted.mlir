// RUN: not tcrv-opt %s --tcrv-toy-materialize-template-selected-boundary-seed 2>&1 | FileCheck %s

module {
  func.func @toy_seed() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}

// CHECK: Unknown command line argument
// CHECK-SAME: tcrv-toy-materialize-template-selected-boundary-seed
