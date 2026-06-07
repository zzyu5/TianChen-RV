module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  func.func @source_vector_runtime_scalar_cmp_select_sle(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.splat %rhs_scalar : vector<4xi32>
    %true_vec = vector.transfer_read %true_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %false_vec = vector.transfer_read %false_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi sle, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %true_vec, %false_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}
