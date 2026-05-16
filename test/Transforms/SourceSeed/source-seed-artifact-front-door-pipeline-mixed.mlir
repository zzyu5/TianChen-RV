// RUN: not tcrv-opt %s --tcrv-source-seed-artifact-front-door-pipeline 2>&1 | FileCheck %s

module {
  func.func @rvv_seed(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }

  func.func @toy_seed() attributes {tcrv_toy.lowering_seed = "template_compute"} {
    return
  }
}

// CHECK: bounded Toy template selected-boundary seed failed: source seed pass requires source-only MLIR input
// CHECK-SAME: pre-existing tcrv.exec/tcrv_toy selected-boundary or unselected variant residue is not accepted
