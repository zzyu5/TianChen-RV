// RUN: tcrv-opt %s --split-input-file --verify-diagnostics --tcrv-lower-linalg-rvv-binary-to-exec

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.invalid",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run"],
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available"
  }

  func.func @bad_unknown_marker(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "bad_unknown_marker_kernel",
        tcrv_frontend_target = @frontend_profile
      } {
    // expected-error@+1 {{marked linalg.generic for TianChen-RV expects 'tcrv_frontend_lowering' to be 'i32-vadd', 'i32-vsub', 'i32-vmul', 'i64-vadd', 'i64-vsub', or 'i64-vmul'}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vxor"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %sum = arith.addi %a, %b : i32
      linalg.yield %sum : i32
    }
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.invalid",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run"],
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available"
  }

  func.func @bad_i16_element_type(%lhs: memref<?xi16>, %rhs: memref<?xi16>, %out: memref<?xi16>)
      attributes {
        tcrv_frontend_kernel = "bad_i16_kernel",
        tcrv_frontend_target = @frontend_profile
      } {
    // expected-error@+1 {{marked linalg.generic for TianChen-RV i32-vadd expects memref<?xi32> lhs/rhs/output operands}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi16>, memref<?xi16>)
      outs(%out : memref<?xi16>) {
    ^bb0(%a: i16, %b: i16, %old: i16):
      %sum = arith.addi %a, %b : i16
      linalg.yield %sum : i16
    }
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_i64_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.invalid.i64",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.i64_m1.sew64", "rvv.i64_m1.lmul_m1", "rvv.i64_m1.tail_policy.agnostic", "rvv.i64_m1.mask_policy.agnostic", "rvv.probe.compile_run"],
    sew_bits = 64 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available"
  }

  func.func @bad_i64_vsub_operator(%lhs: memref<?xi64>, %rhs: memref<?xi64>, %out: memref<?xi64>)
      attributes {
        tcrv_frontend_kernel = "bad_i64_vsub_kernel",
        tcrv_frontend_target = @frontend_i64_profile
      } {
    // expected-error@+1 {{marked linalg.generic for TianChen-RV i64-vsub expects one arith.subi feeding linalg.yield}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i64-vsub"
      }
      ins(%lhs, %rhs : memref<?xi64>, memref<?xi64>)
      outs(%out : memref<?xi64>) {
    ^bb0(%a: i64, %b: i64, %old: i64):
      %sum = arith.addi %a, %b : i64
      linalg.yield %sum : i64
    }
    return
  }
}
