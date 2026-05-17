// RUN: not tcrv-opt %s --tcrv-disable-builtin-plugins --tcrv-source-artifact-front-door-pipeline 2>&1 | FileCheck %s
// RUN: not tcrv-opt %s --tcrv-source-seed-artifact-front-door-pipeline 2>&1 | FileCheck %s --check-prefix=OLD-PIPELINE-REMOVED

module {
  func.func @rvv_source(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
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
}

// CHECK: TianChen-RV execution plan coherence check failed for kernel <missing>: requires at least one tcrv.exec.kernel

// OLD-PIPELINE-REMOVED: Unknown command line argument
// OLD-PIPELINE-REMOVED-SAME: tcrv-source-seed-artifact-front-door-pipeline
