// RUN: not tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec 2>&1 | FileCheck %s --implicit-check-not="tcrv.exec.kernel @bad_frontend_vadd"

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.probe.compile_run"],
    selected_march = "rv64gcv",
    status = "available"
  }

  func.func @bad_source_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "bad_frontend_vadd",
        tcrv_frontend_target = @frontend_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vadd"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %diff = arith.subi %a, %b : i32
      linalg.yield %diff : i32
    }
    return
  }
}

// CHECK: marked linalg.generic for TianChen-RV i32-vadd expects one arith.addi feeding linalg.yield
