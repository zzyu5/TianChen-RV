// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-lower-source-rvv-binary-to-exec

module {
  tcrv.exec.target @vector_dynamic_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.dynamic.frontend.invalid",
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

  // expected-error@+1 {{TianChen-RV vector source frontend has marker 'i32-vsub' requesting family 'i32-vsub' but source body infers family 'i32-vadd' from arith.addi; marker is only a bounded route request/cross-check}}
  func.func @source_vector_dynamic_vsub_marker_add_body(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_vsub_marker_add_body",
        tcrv_frontend_lowering = "i32-vsub",
        tcrv_frontend_target = @vector_dynamic_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %sum, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// -----

module {
  tcrv.exec.target @vector_dynamic_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.dynamic.frontend.invalid",
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

  // expected-error@+1 {{TianChen-RV vector source frontend supports only marker 'i32-vadd' or 'i32-vsub'; marker 'i32-vmul' is not accepted because this pass is not a generic vector backend}}
  func.func @source_vector_dynamic_vmul_marker_mul_body(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_vmul_marker_mul_body",
        tcrv_frontend_lowering = "i32-vmul",
        tcrv_frontend_target = @vector_dynamic_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %product = arith.muli %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %product, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}
// -----

module {
  tcrv.exec.target @vector_dynamic_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.dynamic.frontend.invalid",
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

  // expected-error@+1 {{TianChen-RV vector source frontend has marker 'i32-vadd' requesting family 'i32-vadd' but source body infers family 'i32-vsub' from arith.subi; marker is only a bounded route request/cross-check}}
  func.func @source_vector_dynamic_vadd_marker_sub_body(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_vadd_marker_sub_body",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_dynamic_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %diff = arith.subi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %diff, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}
