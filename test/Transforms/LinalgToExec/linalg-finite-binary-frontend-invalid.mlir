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
    id = "rvv.profile.frontend.invalid.nonuniform",
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

  func.func @bad_nonuniform_source_dtype(%lhs: memref<?xi32>, %rhs: memref<?xi64>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "bad_nonuniform_source_dtype_kernel",
        tcrv_frontend_target = @frontend_profile
      } {
    // expected-error@+1 {{marked linalg.generic for TianChen-RV bounded binary expects lhs/rhs/output memref element types to match, got i32, i64, and i32}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi64>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i64, %old: i32):
      %sum = arith.addi %a, %old : i32
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
    // expected-error@+1 {{marked linalg.generic for TianChen-RV bounded binary supports only i32 or i64 source memref element types, got i16}}
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
    // expected-error@+1 {{marked linalg.generic for TianChen-RV has marker 'i64-vsub' requesting family 'i64-vsub' but source body infers family 'i64-vadd' from arith.addi; marker is only a bounded route request/cross-check}}
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

// -----

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.invalid.dtype",
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

  func.func @bad_marker_dtype(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "bad_marker_dtype_kernel",
        tcrv_frontend_target = @frontend_profile
      } {
    // expected-error@+1 {{marked linalg.generic for TianChen-RV has marker 'i64-vadd' requesting dtype 'i64' but source operands and region arguments infer dtype 'i32'; marker is only a bounded route request/cross-check}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i64-vadd"
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
    id = "rvv.profile.frontend.invalid.unsupported",
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

  func.func @bad_unsupported_arithmetic(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "bad_unsupported_arithmetic_kernel",
        tcrv_frontend_target = @frontend_profile
      } {
    // expected-error@+1 {{marked linalg.generic for TianChen-RV bounded binary expects source body arithmetic to be arith.addi, arith.subi, or arith.muli}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %and = arith.andi %a, %b : i32
      linalg.yield %and : i32
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
    id = "rvv.profile.frontend.invalid.yield",
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

  func.func @bad_yield_source(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "bad_yield_source_kernel",
        tcrv_frontend_target = @frontend_profile
      } {
    // expected-error@+1 {{marked linalg.generic for TianChen-RV bounded binary expects linalg.yield to return the arith.addi result}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %sum = arith.addi %a, %b : i32
      linalg.yield %old : i32
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
    id = "rvv.profile.frontend.invalid.legacy",
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

  func.func @bad_legacy_descriptor(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "bad_legacy_descriptor_kernel",
        tcrv_frontend_target = @frontend_profile
      } {
    // expected-error@+1 {{TianChen-RV linalg frontend no longer accepts legacy descriptor metadata 'tcrv_rvv.lowering_descriptor'; the source linalg body and typed operands are the compute authority}}
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_rvv.lowering_descriptor = "i32-vsub-microkernel.v1"
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
