// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=tcrv_rvv.i64_vadd_microkernel --implicit-check-not=tcrv_rvv.i64_vsub_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i64m1 --implicit-check-not=__riscv_vsub_vv_i64m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed 's/name = "tcrv_rvv.selected_vector_shape"/name = "tcrv_rvv.missing_selected_vector_shape"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-SELECTED --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/value = "64"/s//value = "32"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SEW --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/value = "runtime-element-count"/s//value = "stale-runtime-element-count"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-AVL --implicit-check-not="#include <riscv_vector.h>"

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_rvv_i64_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend.i64",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.i64_m1.sew64", "rvv.i64_m1.lmul_m1", "rvv.i64_m1.tail_policy.agnostic", "rvv.i64_m1.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march"],
    sew_bits = 64 : i64,
    lmul = "m1",
    tail_policy = "agnostic",
    mask_policy = "agnostic",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  func.func @source_i64_vmul(%lhs: memref<?xi64>, %rhs: memref<?xi64>, %out: memref<?xi64>)
      attributes {
        tcrv_frontend_kernel = "frontend_i64_vmul",
        tcrv_frontend_target = @frontend_rvv_i64_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i64-vmul"
      }
      ins(%lhs, %rhs : memref<?xi64>, memref<?xi64>)
      outs(%out : memref<?xi64>) {
    ^bb0(%a: i64, %b: i64, %old: i64):
      %product = arith.muli %a, %b : i64
      linalg.yield %product : i64
    }
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_i64_vmul
// LOWER-SAME: target = @frontend_rvv_i64_profile
// LOWER-SAME: tcrv_frontend_lowering = "i64-vmul"
// LOWER: tcrv.exec.mem_window @abi_lhs_input_buffer
// LOWER-SAME: abi_role = "lhs-input-buffer"
// LOWER-SAME: access = "read"
// LOWER-SAME: c_type = "const int64_t *"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER: tcrv.exec.mem_window @abi_rhs_input_buffer
// LOWER-SAME: abi_role = "rhs-input-buffer"
// LOWER-SAME: c_type = "const int64_t *"
// LOWER: tcrv.exec.mem_window @abi_output_buffer
// LOWER-SAME: abi_role = "output-buffer"
// LOWER-SAME: access = "write"
// LOWER-SAME: c_type = "int64_t *"
// LOWER: tcrv.exec.runtime_param @abi_runtime_element_count
// LOWER-SAME: abi_role = "runtime-element-count"
// LOWER-SAME: c_name = "n"
// LOWER-SAME: c_type = "size_t"

// PIPE-LABEL: tcrv.exec.kernel @frontend_i64_vmul
// PIPE-SAME: target = @frontend_rvv_i64_profile
// PIPE-SAME: tcrv_frontend_lowering = "i64-vmul"
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: requires = [@frontend_rvv_i64_profile]
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-SAME: tcrv_rvv.required_march = "rv64gcv"
// PIPE-SAME: tcrv_rvv.selected_setvl_suffix = "e64m1"
// PIPE-SAME: tcrv_rvv.selected_vector_lmul = "m1"
// PIPE-SAME: tcrv_rvv.selected_vector_sew = 64 : i64
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i64m1"
// PIPE-SAME: tcrv_rvv.selected_vector_suffix = "i64m1"
// PIPE-SAME: tcrv_rvv.selected_vector_type = "vint64m1_t"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: reason = "variant-selected"
// PIPE-SAME: selection_kind = "static-variant"
// PIPE-SAME: target = @rvv_first_slice
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: required_capabilities = [@frontend_rvv_i64_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_lmul = "m1"
// PIPE-SAME: selected_vector_sew = 64 : i64
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_type = "vint64m1_t"
// PIPE-SAME: source_kernel = "frontend_i64_vmul"
// PIPE: tcrv_rvv.i64_vmul_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@frontend_rvv_i64_profile]
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_suffix = "i64m1"
// PIPE-SAME: source_kernel = "frontend_i64_vmul"
// PIPE: tcrv_rvv.setvl
// PIPE-SAME: lmul = "m1"
// PIPE-SAME: sew = 64 : i64
// PIPE: tcrv_rvv.i64_load
// PIPE-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_mul
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_store
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.vl
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "rvv-explicit-i64-vmul-microkernel-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-i64-vmul-microkernel-c"
// PIPE-SAME: runtime_abi = "rvv-i64-vmul-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// PIPE-SAME: runtime_abi_name = "rvv-i64-vmul-runtime-callable-c-function.v1"
// PIPE-SAME: runtime_glue_role = "runtime-callable-i64-vmul-function"
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_dtype"
// PIPE-SAME: role = "typed-rvv-binary-source"
// PIPE-SAME: value = "i64"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_family"
// PIPE-SAME: value = "i64-vmul"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_operator"
// PIPE-SAME: value = "multiply"}
// PIPE-SAME: {name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: value = "tcrv_rvv.i64_mul"}
// PIPE-SAME: {name = "tcrv_rvv.emitc_lowerable_op_interface"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"}
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @rvv_first_slice

// SOURCE: /* executable_microkernel: tcrv_rvv.i64_vmul_microkernel */
// SOURCE: /* arithmetic_family: i64-vmul */
// SOURCE: /* dtype: i64 */
// SOURCE: /* active_route: tcrv-export-rvv-i64-vmul-microkernel-c */
// SOURCE: /* dataflow_body: tcrv_rvv.i64_load -> tcrv_rvv.i64_load -> tcrv_rvv.i64_mul -> tcrv_rvv.i64_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i64_mul, lhs=lhs_vec, rhs=rhs_vec, result=product_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* selected_vector_shape_config: dtype=i64, shape=i64m1, sew=64, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i64_m1.sew64 rvv.i64_m1.lmul_m1 rvv.i64_m1.tail_policy.agnostic rvv.i64_m1.mask_policy.agnostic */
// SOURCE: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int64_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[2]: c_name=out, c_type=int64_t *, role=output-buffer, ownership=target-export-abi-owned */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i64_vmul_microkernel_frontend_i64_vmul_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vsetvl_e64m1
// SOURCE: __riscv_vle64_v_i64m1
// SOURCE: __riscv_vmul_vv_i64m1
// SOURCE: __riscv_vse64_v_i64m1
// SOURCE: void tcrv_rvv_i64_vmul_microkernel_frontend_i64_vmul_rvv_first_slice

// MISSING-SELECTED: requires selected_plan_metadata 'tcrv_rvv.selected_vector_shape'
// STALE-SEW: selected_plan_metadata 'tcrv_rvv.selected_vector_sew'
// STALE-SEW-SAME: sew must be '64'
// STALE-AVL: selected_plan_metadata 'tcrv_rvv.runtime_avl_role'
// STALE-AVL-SAME: must use value 'runtime-element-count'
