// RUN: rm -rf %t.frontend.marker_mismatch.bundle && mkdir %t.frontend.marker_mismatch.bundle
// RUN: not tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=%t.frontend.marker_mismatch.bundle %s 2>&1 | FileCheck %s --implicit-check-not=tianchenrv.target_artifact_bundle_export --implicit-check-not="#include <riscv_vector.h>" --implicit-check-not=tcrv-export-rvv-scalar-i32-vsub-dispatch-c
// RUN: not test -e %t.frontend.marker_mismatch.bundle/tianchenrv-target-artifact-bundle.index

#map = affine_map<(d0) -> (d0)>

module @plan_linalg_i32_vsub_marker_mismatch_no_bundle {
  tcrv.exec.capability @scalar_fallback {
    id = "scalar.fallback",
    kind = "fallback",
    status = "available"
  }

  tcrv.exec.target @frontend_marker_mismatch_profile {
    architecture = "riscv64",
    count = 64 : i64,
    capability_providers = [@scalar_fallback],
    id = "rvv.profile.frontend.marker_mismatch",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
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

  func.func @source_marker_mismatch(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_marker_mismatch",
        tcrv_frontend_target = @frontend_marker_mismatch_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vsub"
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

// CHECK: marked linalg.generic for TianChen-RV i32-vsub expects one arith.subi feeding linalg.yield
// CHECK: TianChen-RV plan-and-export target artifact bundle failed during bounded linalg RVV binary frontend lowering
