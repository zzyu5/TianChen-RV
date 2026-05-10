// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-binary-to-exec | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=func.func
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=func.func

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_profile {
    architecture = "riscv64",
    id = "rvv.profile.frontend.compat",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv"],
    status = "available"
  }

  func.func @source_compat_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_compat_i32_vadd",
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
      %sum = arith.addi %a, %b : i32
      linalg.yield %sum : i32
    }
    return
  }
}

// CHECK: tcrv.exec.kernel @frontend_compat_i32_vadd
// CHECK-SAME: target = @frontend_profile
// CHECK-SAME: tcrv_frontend_lowering = "i32-vadd"
// CHECK: tcrv.exec.mem_window @abi_lhs_input_buffer
// CHECK-SAME: abi_role = "lhs-input-buffer"
// CHECK: tcrv.exec.mem_window @abi_rhs_input_buffer
// CHECK-SAME: abi_role = "rhs-input-buffer"
// CHECK: tcrv.exec.mem_window @abi_output_buffer
// CHECK-SAME: abi_role = "output-buffer"
// CHECK: tcrv.exec.runtime_param @abi_runtime_element_count
// CHECK-SAME: abi_role = "runtime-element-count"
