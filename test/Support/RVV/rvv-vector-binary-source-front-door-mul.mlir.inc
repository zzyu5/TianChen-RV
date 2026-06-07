module attributes {tcrv_rvv.source_front_door = "bounded_vector_source"} {
  func.func @source_vector_mul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %a = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %b = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %product = arith.muli %a, %b : vector<4xi32>
    vector.transfer_write %product, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}
