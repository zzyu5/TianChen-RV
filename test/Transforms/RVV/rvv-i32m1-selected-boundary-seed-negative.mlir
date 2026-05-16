// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-rvv-materialize-i32m1-selected-boundary-seed

module {
  // expected-error@+1 {{bounded RVV i32m1 selected-boundary seed failed: source function must expose exactly four runtime ABI operands: lhs, rhs, out, and n}}
  func.func @missing_n(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded RVV i32m1 selected-boundary seed failed: lhs, rhs, and out operands must be memref<?xi32>}}
  func.func @wrong_dtype(%lhs: memref<?xf32>, %rhs: memref<?xf32>, %out: memref<?xf32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xf32>, vector<4xf32>
      %b = vector.load %rhs[%i] : memref<?xf32>, vector<4xf32>
      %sum = arith.addf %a, %b : vector<4xf32>
      vector.store %sum, %out[%i] : memref<?xf32>, vector<4xf32>
    }
    return
  }
}

// -----

module {
  func.func @wrong_vector_shape(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{bounded RVV i32m1 selected-boundary seed failed: source vector loads must produce vector<4xi32>}}
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<8xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<8xi32>
      %sum = arith.addi %a, %b : vector<8xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<8xi32>
    }
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded RVV i32m1 selected-boundary seed failed: source function must contain one scf.for}}
  func.func @missing_loop(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    return
  }
}

// -----

module {
  tcrv.exec.kernel @stale_unselected_variant {
    // expected-error@+1 {{bounded RVV i32m1 selected-boundary seed failed: source seed pass requires source-only MLIR input; pre-existing tcrv.exec/tcrv_rvv selected-boundary or unselected variant residue is not accepted}}
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
    }
  }
  func.func @with_stale_residue(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
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
