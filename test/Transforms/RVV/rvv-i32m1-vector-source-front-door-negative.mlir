// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-rvv-materialize-i32m1-vector-source-front-door

module {
  // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source function must expose exactly four runtime ABI operands: lhs, rhs, out, and n}}
  func.func @missing_n(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>) {
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source function must expose exactly four runtime ABI operands: lhs, rhs, out, and n}}
  func.func @extra_n(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index, %extra_n: index) {
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: stale tcrv_rvv.lowering_seed metadata is not accepted as source-route authority}}
  func.func @stale_seed_only(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) attributes {tcrv_rvv.lowering_seed = "i32m1_add"} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: lhs, rhs, and out operands must be memref<?xi32>}}
  func.func @wrong_abi_order(%n: index, %lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>) {
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: lhs, rhs, and out operands must be memref<?xi32>}}
  func.func @wrong_dtype(%lhs: memref<?xf32>, %rhs: memref<?xf32>, %out: memref<?xf32>, %n: index) {
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
  func.func @wrong_vector_shape(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source vector loads must produce vector<4xi32>}}
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
  func.func @wrong_arithmetic_op(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: scf.for body operation order must be vector.load, vector.load, arith.addi, vector.store}}
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.subi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// -----

module {
  func.func @extra_loop_op(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: scf.for body must contain exactly two vector.load ops, one arith.addi, and one vector.store}}
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      %unused = arith.addi %i, %i : index
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// -----

module {
  func.func @missing_store(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: scf.for body must contain exactly two vector.load ops, one arith.addi, and one vector.store}}
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
    }
    return
  }
}

// -----

module {
  func.func @wrong_memref_order(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: vector.store must write out at the loop iv}}
      vector.store %sum, %lhs[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// -----

module {
  func.func @wrong_lower_bound(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: scf.for lower bound must be constant index 0}}
    scf.for %i = %c1 to %n step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// -----

module {
  func.func @wrong_upper_bound(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: scf.for upper bound must be the runtime n operand}}
    scf.for %i = %c0 to %c8 step %c4 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// -----

module {
  func.func @wrong_step(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: scf.for step must match the bounded vector chunk}}
    scf.for %i = %c0 to %n step %c8 {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
    }
    return
  }
}

// -----

module {
  func.func @loop_carried_value(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: scf.for must not use loop-carried iter_args or yield values}}
    %result = scf.for %i = %c0 to %n step %c4 iter_args(%acc = %c0) -> (index) {
      %a = vector.load %lhs[%i] : memref<?xi32>, vector<4xi32>
      %b = vector.load %rhs[%i] : memref<?xi32>, vector<4xi32>
      %sum = arith.addi %a, %b : vector<4xi32>
      vector.store %sum, %out[%i] : memref<?xi32>, vector<4xi32>
      scf.yield %acc : index
    }
    return
  }
}

// -----

module {
  func.func @unrelated_body(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source function may contain only index constants, one scf.for, and one empty return}}
    %unused = arith.addi %n, %n : index
    return
  }
}

// -----

module {
  // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source function must contain one scf.for}}
  func.func @missing_loop(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    return
  }
}

// -----

module {
  func.func @with_stale_rvv_residue(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source materializer requires source-only MLIR input; pre-existing tcrv.exec/tcrv_rvv selected-boundary or unselected variant residue is not accepted}}
    %value = tcrv_rvv.runtime_abi_value {c_name = "lhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
    return
  }
}

// -----

module {
  tcrv.exec.kernel @stale_unselected_variant {
    // expected-error@+1 {{bounded RVV i32m1 vector-source front door failed: source materializer requires source-only MLIR input; pre-existing tcrv.exec/tcrv_rvv selected-boundary or unselected variant residue is not accepted}}
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @rvv_i32_add attributes {origin = "rvv-plugin", requires = [@rvv]} {
    }
  }
  func.func @with_stale_residue(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index) {
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
