// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-i32-vadd-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m1 --implicit-check-not=i32_vadd --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_rvv_scalar_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march", "scalar.fallback"],
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

  func.func @source_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_i32_vsub",
        tcrv_frontend_target = @frontend_rvv_scalar_profile
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

// PIPE-LABEL: tcrv.exec.kernel @frontend_i32_vsub
// PIPE-SAME: target = @frontend_rvv_scalar_profile
// PIPE-SAME: tcrv_frontend_lowering = "i32-vsub"
// PIPE: tcrv.exec.mem_window @abi_lhs_input_buffer
// PIPE-SAME: abi_role = "lhs-input-buffer"
// PIPE: tcrv.exec.runtime_param @abi_runtime_element_count
// PIPE-SAME: abi_role = "runtime-element-count"
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: requires = [@frontend_rvv_scalar_profile]
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-SAME: tcrv_rvv.i32_m1_lanes = 4 : i64
// PIPE-SAME: tcrv_rvv.lowering_descriptor = "i32-vsub-microkernel.v1"
// PIPE-SAME: tcrv_rvv.required_march = "rv64gcv"
// PIPE-SAME: tcrv_rvv.vlenb_bytes = 16 : i64
// PIPE: tcrv.exec.variant @scalar_fallback_first_slice
// PIPE-SAME: fallback_role = "conservative"
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: requires = [@frontend_rvv_scalar_profile]
// PIPE-SAME: tcrv_scalar.element_count = 16 : i64
// PIPE-SAME: tcrv_scalar.lowering_descriptor = "i32-vsub-microkernel.v1"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: reason = "variant-selected"
// PIPE-SAME: selection_kind = "static-variant"
// PIPE-SAME: target = @rvv_first_slice
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: required_capabilities = [@frontend_rvv_scalar_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: source_kernel = "frontend_i32_vsub"
// PIPE: tcrv_rvv.i32_vsub_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@frontend_rvv_scalar_profile]
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: source_kernel = "frontend_i32_vsub"
// PIPE: tcrv_rvv.i32_sub
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-i32-vsub-microkernel-c"
// PIPE-SAME: reason = "emission_plan"
// PIPE-SAME: runtime_abi = "rvv-i32-vsub-runtime-callable-c-abi.v1"
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @rvv_first_slice

// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// SOURCE: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_sub -> tcrv_rvv.i32_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec */
// SOURCE: void tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice
// SOURCE: __riscv_vsub_vv_i32m1
