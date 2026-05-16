// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-source-seed-artifact-front-door-pipeline

module {
  // expected-error@+1 {{bounded RVV i32m1 selected-boundary materializer failed: stale tcrv_rvv.lowering_seed metadata is not accepted as source-route authority}}
  func.func @stale_rvv_marker(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    return
  }
}

// -----

module {
  tcrv.exec.kernel @stale_unselected_variant {
    // expected-error@+1 {{bounded RVV i32m1 selected-boundary materializer failed: source materializer requires source-only MLIR input; pre-existing tcrv.exec/tcrv_rvv selected-boundary or unselected variant residue is not accepted}}
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
    }
  }
  func.func @rvv_with_stale_residue(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
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
