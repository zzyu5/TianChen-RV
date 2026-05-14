// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-lower-vector-rvv-i32-vadd-to-exec

module {
  tcrv.exec.target @vector_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.frontend.invalid",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  // expected-error@+1 {{TianChen-RV vector i32-vadd frontend supports only marker 'i32-vadd'; marker 'i32-vsub' is not accepted because this pass is not a generic vector backend}}
  func.func @source_vector_vsub_marker(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_i32_vsub_marker",
        tcrv_frontend_lowering = "i32-vsub",
        tcrv_frontend_target = @vector_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
    vector.transfer_write %sum, %out[%c0] {in_bounds = [true]} : vector<16xi32>, memref<?xi32>
    return
  }
}

// -----

module {
  tcrv.exec.target @vector_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.frontend.invalid",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_vector_dynamic_inbounds_tail(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_inbounds_tail",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      // expected-error@+1 {{TianChen-RV dynamic vector i32 binary frontend expects lhs read to expose MLIR transfer tail semantics; in_bounds = [true] is stale for runtime %n tail iterations}}
      %lhs_vec = vector.transfer_read %lhs[%i], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %sum, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// -----

module {
  tcrv.exec.target @vector_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.frontend.invalid",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_vector_dynamic_wrong_step(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_wrong_step",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{TianChen-RV dynamic vector i32 binary frontend expects the second source operation to be arith.constant 16 : index}}
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %n step %c8 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
      %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %sum, %out[%i] {in_bounds = [true]} : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// -----

module {
  tcrv.exec.target @vector_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.frontend.invalid",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  // expected-error@+1 {{TianChen-RV vector frontend no longer accepts legacy descriptor metadata 'tcrv_rvv.lowering_descriptor'; the source vector/arith body and typed operands are the compute authority}}
  func.func @source_vector_legacy_descriptor(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_legacy_descriptor",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_frontend_profile,
        tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"
      } {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
    vector.transfer_write %sum, %out[%c0] {in_bounds = [true]} : vector<16xi32>, memref<?xi32>
    return
  }
}

// -----

module {
  tcrv.exec.target @vector_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.frontend.invalid",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_vector_sub_body(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_sub_body",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    // expected-error@+1 {{TianChen-RV vector i32-vadd frontend expects arith.addi over the two transfer-read vector values}}
    %diff = arith.subi %lhs_vec, %rhs_vec : vector<16xi32>
    vector.transfer_write %diff, %out[%c0] {in_bounds = [true]} : vector<16xi32>, memref<?xi32>
    return
  }
}
