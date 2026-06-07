// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-rvv-materialize-vector-binary-source-front-door
// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-rvv-materialize-vector-compare-select-source-front-door

// expected-error@+1 {{RVV vector source-front-door family registry failed: unknown tcrv_rvv.source_front_door marker 'bounded_vector_unknown_source'; registered RVV vector source-front-door markers are 'bounded_vector_source', 'bounded_vector_compare_select_source'}}
module attributes {tcrv_rvv.source_front_door = "bounded_vector_unknown_source"} {
  func.func @unknown_marker(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}
