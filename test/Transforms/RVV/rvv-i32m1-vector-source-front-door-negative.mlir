// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-rvv-materialize-i32m1-vector-source-front-door

module {
  // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: RVV Stage1 source-front-door materialization is disabled; use an explicit selected generic tcrv_rvv.load/tcrv_rvv.binary/tcrv_rvv.store body instead}}
  func.func @valid_add_source_is_fail_closed(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
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

// -----

module {
  tcrv.exec.kernel @preexisting_rvv_boundary {
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source materializer requires source-only MLIR input; pre-existing tcrv.exec/tcrv_rvv selected-boundary or unselected variant residue is not accepted}}
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
  }

  func.func @preexisting_exec_is_rejected(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}
