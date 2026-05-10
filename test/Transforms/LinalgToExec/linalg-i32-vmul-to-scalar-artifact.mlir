// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=i32-vsub-microkernel.v1 --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=tcrv_scalar.i32_vsub_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=riscv_vector --implicit-check-not=__riscv --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_scalar_profile {
    id = "scalar.profile.frontend",
    kind = "profile",
    provides = ["scalar.fallback"],
    status = "available"
  }

  func.func @source_vmul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_i32_vmul_scalar",
        tcrv_frontend_target = @frontend_scalar_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i32-vmul"
      }
      ins(%lhs, %rhs : memref<?xi32>, memref<?xi32>)
      outs(%out : memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %old: i32):
      %product = arith.muli %a, %b : i32
      linalg.yield %product : i32
    }
    return
  }
}

// PIPE-LABEL: tcrv.exec.kernel @frontend_i32_vmul_scalar
// PIPE-SAME: target = @frontend_scalar_profile
// PIPE-SAME: tcrv_frontend_lowering = "i32-vmul"
// PIPE: tcrv.exec.variant @scalar_fallback_first_slice
// PIPE-SAME: fallback_role = "conservative"
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: requires = [@frontend_scalar_profile]
// PIPE-SAME: tcrv_scalar.element_count = 16 : i64
// PIPE-SAME: tcrv_scalar.lowering_descriptor = "i32-vmul-microkernel.v1"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: reason = "variant-selected"
// PIPE-SAME: selection_kind = "fallback-only"
// PIPE-SAME: target = @scalar_fallback_first_slice
// PIPE: tcrv_scalar.lowering_boundary
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: required_capabilities = [@frontend_scalar_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_variant = @scalar_fallback_first_slice
// PIPE-SAME: source_kernel = "frontend_i32_vmul_scalar"
// PIPE-SAME: status = "metadata-only"
// PIPE: tcrv_scalar.i32_vmul_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@frontend_scalar_profile]
// PIPE-SAME: selected_variant = @scalar_fallback_first_slice
// PIPE-SAME: source_kernel = "frontend_i32_vmul_scalar"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "scalar-explicit-i32-vmul-microkernel-c-source"
// PIPE-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
// PIPE-SAME: lowering_pipeline = "tcrv-export-scalar-i32-vmul-microkernel-c"
// PIPE-SAME: reason = "emission_plan"
// PIPE-SAME: runtime_abi = "scalar-i32-vmul-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
// PIPE-SAME: runtime_abi_name = "scalar-i32-vmul-runtime-callable-c-function.v1"
// PIPE-SAME: runtime_glue_role = "runtime-callable-i32-vmul-fallback-function"
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @scalar_fallback_first_slice

// SOURCE: /* Scope: library-style C source for exactly one tcrv_scalar.i32_vmul_microkernel. */
// SOURCE: /* selected_kernel: @frontend_i32_vmul_scalar */
// SOURCE: /* selected_variant: @scalar_fallback_first_slice */
// SOURCE: /* executable_microkernel: tcrv_scalar.i32_vmul_microkernel */
// SOURCE: /* runtime_abi_name: scalar-i32-vmul-runtime-callable-c-function.v1 */
// SOURCE: /* runtime_glue_role: runtime-callable-i32-vmul-fallback-function */
// SOURCE: void tcrv_scalar_i32_vmul_microkernel_frontend_i32_vmul_scalar_scalar_fallback_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)
// SOURCE: for (size_t index = 0; index < n; ++index)
// SOURCE: out[index] = lhs[index] * rhs[index];
