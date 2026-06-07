// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-rvv-materialize-vector-runtime-scalar-cmp-select-source-front-door

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  // expected-error@+1 {{bounded RVV vector-runtime-scalar-cmp-select source front door failed: source function inputs must be lhs rank-1 i32 memref, rhs_scalar i32, true_value/false_value/out rank-1 i32 memrefs, and one runtime n index}}
  func.func @bad_scalar_type(%lhs: memref<?xi32>, %rhs_scalar: index, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  // expected-error@+1 {{bounded RVV vector-runtime-scalar-cmp-select source front door failed: source function must have exactly six inputs and no results: lhs memref<?xi32>, rhs_scalar i32, true_value/false_value/out memref<?xi32>, plus n index}}
  func.func @missing_runtime_scalar(%lhs: memref<?xi32>, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  // expected-error@+1 {{bounded RVV vector-runtime-scalar-cmp-select source front door failed: source materializer requires RVV source-only MLIR input; pre-existing tcrv.exec/tcrv_rvv/tcrv_toy/tcrv_tensorext_lite selected-boundary or variant residue is not accepted}}
  tcrv.exec.kernel @stale_selected_boundary {
  }

  func.func @source_vector_runtime_scalar_cmp_select_sle(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}

// -----

// expected-error@+1 {{RVV vector source-front-door family registry failed: family 'bounded-vector-runtime-scalar-cmp-select-source-front-door' rejected stale tcrv_rvv.lowering_seed metadata as RVV source-route authority}}
module attributes {tcrv_rvv.lowering_seed = "stale-route", tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  func.func @stale_lowering_seed(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  func.func @unsupported_predicate(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.splat %rhs_scalar : vector<4xi32>
    %true_vec = vector.transfer_read %true_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %false_vec = vector.transfer_read %false_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    // expected-error@+1 {{bounded RVV vector-runtime-scalar-cmp-select source front door failed: only arith.cmpi predicates eq, slt, or sle plus arith.select are supported by this bounded runtime-scalar source path}}
    %mask = arith.cmpi ne, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %true_vec, %false_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  // expected-error@+1 {{bounded RVV vector-runtime-scalar-cmp-select source front door failed: source pattern must contain exactly three vector.transfer_read ops, one vector.splat of rhs_scalar, one supported arith.cmpi vector compare op, one arith.select op, and one vector.transfer_write}}
  func.func @missing_scalar_splat(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %true_vec = vector.transfer_read %true_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %false_vec = vector.transfer_read %false_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi sle, %lhs_vec, %true_vec : vector<4xi32>
    %selected = arith.select %mask, %true_vec, %false_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_runtime_scalar_cmp_select_source"} {
  func.func @unsupported_select_layout(%lhs: memref<?xi32>, %rhs_scalar: i32, %true_value: memref<?xi32>, %false_value: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.splat %rhs_scalar : vector<4xi32>
    %true_vec = vector.transfer_read %true_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %false_vec = vector.transfer_read %false_value[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi sle, %lhs_vec, %rhs_vec : vector<4xi32>
    // expected-error@+1 {{bounded RVV vector-runtime-scalar-cmp-select source front door failed: arith.select layout must select true_value when the runtime-scalar compare mask is true and false_value when it is false}}
    %selected = arith.select %mask, %false_vec, %true_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}
