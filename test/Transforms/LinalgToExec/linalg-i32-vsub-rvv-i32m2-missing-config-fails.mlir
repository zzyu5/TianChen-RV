// RUN: not tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s --implicit-check-not=tcrv_rvv.i32_vsub_microkernel --implicit-check-not="emission_plan" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_rvv_incomplete_m2_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.incomplete-m2",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m2.sew32", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    bytes = 16 : i64,
    lanes = 4 : i64,
    sew_bits = 32 : i64,
    lmul = "m2",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_vsub_missing_m2_config(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_i32_vsub_missing_m2_config",
        tcrv_frontend_target = @frontend_rvv_incomplete_m2_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vsub"
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

// CHECK: no viable plugin proposals
// CHECK: rvv-plugin:
// CHECK: RVV property decision requires either the finite i32m1 config capability ids or the finite i32m2 config capability ids
