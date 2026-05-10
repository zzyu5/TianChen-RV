// RUN: not tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s --implicit-check-not=tcrv_rvv.i64_vadd_microkernel --implicit-check-not="emission_plan" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_rvv_missing_i64_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.missing-i64",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    sew_bits = 32 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_i64_vadd_missing_i64_config(%lhs: memref<?xi64>, %rhs: memref<?xi64>, %out: memref<?xi64>)
      attributes {
        tcrv_frontend_kernel = "frontend_i64_vadd_missing_i64_config",
        tcrv_frontend_target = @frontend_rvv_missing_i64_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i64-vadd"
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

// CHECK: no viable plugin proposals
// CHECK: rvv-plugin:
// CHECK: RVV property decision requires capability id 'rvv.i64_m1.sew64'
