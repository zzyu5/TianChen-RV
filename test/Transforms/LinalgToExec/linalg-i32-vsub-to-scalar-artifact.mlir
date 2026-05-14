// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=tcrv_scalar.i32_vadd_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=riscv_vector --implicit-check-not=__riscv --implicit-check-not=i32_vadd --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_scalar_profile {
    id = "scalar.profile.frontend",
    kind = "profile",
    provides = ["scalar.fallback"],
    status = "available"
  }

  func.func @source_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_i32_vsub_scalar",
        tcrv_frontend_target = @frontend_scalar_profile
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

// PIPE-LABEL: tcrv.exec.kernel @frontend_i32_vsub_scalar
// PIPE-SAME: target = @frontend_scalar_profile
// PIPE-SAME: tcrv_frontend_lowering = "i32-vsub"
// PIPE: tcrv.exec.variant @scalar_fallback_first_slice
// PIPE-SAME: fallback_role = "conservative"
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: requires = [@frontend_scalar_profile]
// PIPE-NOT: tcrv_scalar.element_count
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: reason = "variant-selected"
// PIPE-SAME: selection_kind = "fallback-only"
// PIPE-SAME: target = @scalar_fallback_first_slice
// PIPE: tcrv_scalar.lowering_boundary
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: required_capabilities = [@frontend_scalar_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_variant = @scalar_fallback_first_slice
// PIPE-SAME: source_kernel = "frontend_i32_vsub_scalar"
// PIPE-SAME: status = "metadata-only"
// PIPE: tcrv_scalar.i32_vsub_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@frontend_scalar_profile]
// PIPE-SAME: selected_variant = @scalar_fallback_first_slice
// PIPE-SAME: source_kernel = "frontend_i32_vsub_scalar"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "scalar-explicit-i32-vsub-microkernel-c-source"
// PIPE-SAME: lowering_boundary = "tcrv_scalar.lowering_boundary"
// PIPE-SAME: lowering_pipeline = "tcrv-export-scalar-i32-vsub-microkernel-c"
// PIPE-SAME: reason = "emission_plan"
// PIPE-SAME: runtime_abi = "scalar-i32-vsub-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_kind = "scalar-runtime-callable-c-abi"
// PIPE-SAME: runtime_abi_name = "scalar-i32-vsub-runtime-callable-c-function.v1"
// PIPE-SAME: runtime_glue_role = "runtime-callable-i32-vsub-fallback-function"
// PIPE-SAME: selected_plan_metadata = [{name = "tcrv_scalar.selected_binary_dtype"
// PIPE-SAME: role = "typed-scalar-binary-source"
// PIPE-SAME: value = "i32"}
// PIPE-SAME: {name = "tcrv_scalar.selected_binary_family"
// PIPE-SAME: value = "i32-vsub"}
// PIPE-SAME: {name = "tcrv_scalar.emitc_source_op"
// PIPE-SAME: value = "tcrv_scalar.i32_vsub_microkernel"}
// PIPE-SAME: {name = "tcrv_scalar.emitc_lowerable_op_interface"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"}
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @scalar_fallback_first_slice

// SOURCE: /* Scope: library-style C source for exactly one tcrv_scalar.i32_vsub_microkernel. */
// SOURCE: /* Route: typed scalar family op builds the common EmitC lowerable route emitted by the common lower-to-EmitC source-authority boundary. */
// SOURCE: /* selected_kernel: @frontend_i32_vsub_scalar */
// SOURCE: /* selected_variant: @scalar_fallback_first_slice */
// SOURCE: /* executable_microkernel: tcrv_scalar.i32_vsub_microkernel */
// SOURCE: /* runtime_abi_name: scalar-i32-vsub-runtime-callable-c-function.v1 */
// SOURCE: /* runtime_glue_role: runtime-callable-i32-vsub-fallback-function */
// SOURCE: /* emitc_route: tcrv_scalar.i32_vsub_microkernel -> emitc.call_opaque -> scalar runtime C/C++ */
// SOURCE: /* emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_scalar_i32_vsub_microkernel_frontend_i32_vsub_scalar_scalar_fallback_first_slice__tcrv_emitc_body
// SOURCE: // tcrv_emitc.source_op=tcrv_scalar.i32_vsub_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_sub
// SOURCE: tcrv_scalar_i32_sub
// SOURCE: void tcrv_scalar_i32_vsub_microkernel_frontend_i32_vsub_scalar_scalar_fallback_first_slice
