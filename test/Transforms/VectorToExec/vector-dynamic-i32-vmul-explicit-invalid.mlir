// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-lower-vector-rvv-i32-vmul-to-exec

module {
  tcrv.exec.target @vector_dynamic_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.dynamic.frontend.explicit.vmul.invalid",
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

  // expected-error@+1 {{TianChen-RV vector i32-vmul frontend supports only marker 'i32-vmul'; marker 'i32-vsub' is not accepted because this pass is not a generic vector backend}}
  func.func @source_vector_dynamic_vsub_marker_on_vmul_pass(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_vsub_marker_on_vmul_pass",
        tcrv_frontend_lowering = "i32-vsub",
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
    id = "rvv.profile.vector.dynamic.frontend.explicit.vmul.invalid",
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

  // expected-error@+1 {{TianChen-RV vector source frontend has marker 'i32-vmul' requesting family 'i32-vmul' but source body infers family 'i32-vadd' from arith.addi; marker is only a bounded route request/cross-check}}
  func.func @source_vector_dynamic_vmul_marker_add_body(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_vmul_marker_add_body",
        tcrv_frontend_lowering = "i32-vmul",
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
    id = "rvv.profile.vector.dynamic.frontend.explicit.vmul.invalid",
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

  // expected-error@+1 {{TianChen-RV vector i32-vmul frontend expects the dynamic three-buffer plus runtime %n: index SCF wrapper}}
  func.func @source_fixed_vadd_on_explicit_vmul_pass(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "source_fixed_vadd_on_explicit_vmul_pass",
        tcrv_frontend_lowering = "i32-vmul",
        tcrv_frontend_target = @vector_dynamic_frontend_profile
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
