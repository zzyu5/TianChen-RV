// Generic source-only MLIR with no tcrv.exec.kernel. Used as input by the
// source-artifact bundle front-door tests to assert that a plain vector source
// (lacking any selected execution plan) fails the generic source-artifact
// front-door pipeline before any target artifact bundle is produced.
module {
  func.func @vector_source(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %remaining = arith.subi %n, %i : index
      %mask = vector.create_mask %remaining : vector<4xi1>
      %a = vector.transfer_read %lhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %b = vector.transfer_read %rhs[%i], %pad, %mask : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.transfer_write %sum, %out[%i], %mask : vector<4xi32>, memref<?xi32>
    }
    return
  }
}
