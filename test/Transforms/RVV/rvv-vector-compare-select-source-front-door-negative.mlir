// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-rvv-materialize-vector-compare-select-source-front-door

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  // expected-error@+1 {{bounded RVV vector-compare-select source front door failed: source function inputs must be lhs/rhs/out rank-1 i32 memrefs and one runtime n index}}
  func.func @bad_dtype(%lhs: memref<?xf32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %lhs_pad = arith.constant 0.0 : f32
    %rhs_pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %lhs_pad {in_bounds = [true]} : memref<?xf32>, vector<4xf32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %rhs_pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_mask = arith.cmpi eq, %rhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %rhs_mask, %rhs_vec, %rhs_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  // expected-error@+1 {{bounded RVV vector-compare-select source front door failed: source function must have exactly four inputs and no results: lhs/rhs/out memref<?xi32> plus n index}}
  func.func @missing_runtime_n(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi eq, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %lhs_vec, %rhs_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  // expected-error@+1 {{bounded RVV vector-compare-select source front door failed: source materializer requires RVV source-only MLIR input; pre-existing tcrv.exec/tcrv_rvv/tcrv_toy/tcrv_tensorext_lite selected-boundary or variant residue is not accepted}}
  tcrv.exec.kernel @stale_selected_boundary {
  }

  func.func @source_vector_cmp_select_eq(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi eq, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %lhs_vec, %rhs_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

// expected-error@+1 {{bounded RVV vector-compare-select source front door failed: stale tcrv_rvv.lowering_seed metadata is not accepted as RVV source-route authority}}
module attributes {tcrv_rvv.lowering_seed = "stale-route", tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  func.func @stale_lowering_seed(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi eq, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %lhs_vec, %rhs_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  func.func @unsupported_predicate(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    // expected-error@+1 {{bounded RVV vector-compare-select source front door failed: only arith.cmpi predicates eq, slt, or sle plus arith.select are supported by this bounded source path}}
    %mask = arith.cmpi ne, %lhs_vec, %rhs_vec : vector<4xi32>
    %selected = arith.select %mask, %lhs_vec, %rhs_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}

// -----

module attributes {tcrv_rvv.source_front_door = "bounded_vector_compare_select_source"} {
  func.func @unsupported_select_layout(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<4xi32>
    %mask = arith.cmpi eq, %lhs_vec, %rhs_vec : vector<4xi32>
    // expected-error@+1 {{bounded RVV vector-compare-select source front door failed: arith.select layout must select lhs when the compare mask is true and rhs when it is false}}
    %selected = arith.select %mask, %rhs_vec, %lhs_vec : vector<4xi1>, vector<4xi32>
    vector.transfer_write %selected, %out[%c0] {in_bounds = [true]} : vector<4xi32>, memref<?xi32>
    return
  }
}
