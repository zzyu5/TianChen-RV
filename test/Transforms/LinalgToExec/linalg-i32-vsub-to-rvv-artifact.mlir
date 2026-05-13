// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=i32-vadd-microkernel.v1 --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=__riscv_vadd_vv_i32m2 --implicit-check-not=__riscv_vmul_vv_i32m2 --implicit-check-not=i32_vadd --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/tcrv_rvv.selected_vector_shape = "i32m2"/s//tcrv_rvv.selected_vector_shape = "i32m1"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-SHAPE --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact | FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not=__riscv --implicit-check-not=riscv_vector --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed 's#c_name = "rhs", c_type = "const int32_t \*", ownership = "target-export-abi-owned", role = "rhs-input-buffer"#c_name = "rhs", c_type = "const int32_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"#' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ABI --implicit-check-not="#ifndef"
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.linalg-i32m2-vsub.o
// RUN: llvm-readobj --file-headers --symbols %t.linalg-i32m2-vsub.o | FileCheck %s --check-prefix=OBJECT --implicit-check-not="Name: main"

#map = affine_map<(d0) -> (d0)>

module {
  tcrv.exec.target @frontend_rvv_scalar_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.frontend",
    isa_vector_hints = "rv64gcv_zvl128b",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.vlenb_bytes", "rvv.i32_m1_lane_count", "rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic", "rvv.probe.compile_run", "rvv.toolchain.march", "scalar.fallback"],
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

// PIPE: tcrv.exec.target @frontend_rvv_scalar_profile
// PIPE-SAME: lmul = "m2"
// PIPE-SAME: rvv.i32_m2.sew32
// PIPE-SAME: rvv.i32_m2.lmul_m2
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
// PIPE-SAME: tcrv_rvv.base_i32_m1_lanes = 4 : i64
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-SAME: tcrv_rvv.required_march = "rv64gcv"
// PIPE-SAME: tcrv_rvv.selected_mask_policy = "agnostic"
// PIPE-SAME: tcrv_rvv.selected_setvl_suffix = "e32m2"
// PIPE-SAME: tcrv_rvv.selected_tail_policy = "agnostic"
// PIPE-SAME: tcrv_rvv.selected_vector_lmul = "m2"
// PIPE-SAME: tcrv_rvv.selected_vector_sew = 32 : i64
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i32m2"
// PIPE-SAME: tcrv_rvv.selected_vector_suffix = "i32m2"
// PIPE-SAME: tcrv_rvv.selected_vector_type = "vint32m2_t"
// PIPE-SAME: tcrv_rvv.vlenb_bytes = 16 : i64
// PIPE: tcrv.exec.variant @scalar_fallback_first_slice
// PIPE-SAME: fallback_role = "conservative"
// PIPE-SAME: origin = "scalar-plugin"
// PIPE-SAME: requires = [@frontend_rvv_scalar_profile]
// PIPE-NOT: tcrv_scalar.element_count
// PIPE-NOT: tcrv_scalar.lowering_descriptor
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: reason = "variant-selected"
// PIPE-SAME: selection_kind = "static-variant"
// PIPE-SAME: target = @rvv_first_slice
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: required_capabilities = [@frontend_rvv_scalar_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_lmul = "m2"
// PIPE-SAME: selected_vector_sew = 32 : i64
// PIPE-SAME: selected_vector_shape = "i32m2"
// PIPE-SAME: selected_vector_suffix = "i32m2"
// PIPE-SAME: selected_vector_type = "vint32m2_t"
// PIPE-SAME: source_kernel = "frontend_i32_vsub"
// PIPE: tcrv_rvv.i32_vsub_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@frontend_rvv_scalar_profile]
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_lmul = "m2"
// PIPE-SAME: selected_vector_shape = "i32m2"
// PIPE-SAME: selected_vector_type = "vint32m2_t"
// PIPE-SAME: source_kernel = "frontend_i32_vsub"
// PIPE: tcrv_rvv.setvl
// PIPE-SAME: lmul = "m2"
// PIPE: tcrv_rvv.with_vl
// PIPE-SAME: lmul = "m2"
// PIPE: tcrv_rvv.i32_load
// PIPE-SAME: buffer_role = "lhs-input-buffer"
// PIPE-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i32m2
// PIPE: tcrv_rvv.i32_load
// PIPE-SAME: buffer_role = "rhs-input-buffer"
// PIPE-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i32m2
// PIPE: tcrv_rvv.i32_sub
// PIPE-SAME: !tcrv_rvv.i32m2, !tcrv_rvv.i32m2, !tcrv_rvv.vl -> !tcrv_rvv.i32m2
// PIPE: tcrv_rvv.i32_store
// PIPE-SAME: !tcrv_rvv.i32m2, !tcrv_rvv.vl
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-i32-vsub-microkernel-c"
// PIPE-SAME: reason = "emission_plan"
// PIPE-SAME: runtime_abi = "rvv-i32-vsub-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// PIPE-SAME: runtime_abi_name = "rvv-i32-vsub-runtime-callable-c-function.v1"
// PIPE-SAME: runtime_glue_role = "runtime-callable-i32-vsub-function"
// PIPE-SAME: selected_plan_metadata = [{name = "tcrv_rvv.selected_vector_shape"
// PIPE-SAME: value = "i32m2"}
// PIPE-SAME: {name = "tcrv_rvv.selected_setvl_suffix"
// PIPE-SAME: value = "e32m2"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_avl_source"
// PIPE-SAME: value = "runtime-element-count-abi-parameter"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_avl_role"
// PIPE-SAME: value = "runtime-element-count"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_vl_source"
// PIPE-SAME: value = "tcrv_rvv.setvl"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_vl_scope"
// PIPE-SAME: value = "tcrv_rvv.with_vl"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_dtype"
// PIPE-SAME: value = "i32"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_family"
// PIPE-SAME: value = "i32-vsub"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_operator"
// PIPE-SAME: value = "subtract"}
// PIPE-SAME: {name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: value = "tcrv_rvv.i32_sub"}
// PIPE-SAME: {name = "tcrv_rvv.emitc_lowerable_op_interface"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_element_count_c_name"
// PIPE-SAME: value = "n"}
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @rvv_first_slice

// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// SOURCE: /* selected_binary_config: dtype=i32, family=i32-vsub
// SOURCE-SAME: runtime_element_count_c_name=n
// SOURCE-SAME: selected_role=direct variant */
// SOURCE: /* selected_runtime_vl_boundary: runtime_element_count_c_name=n
// SOURCE-SAME: runtime_avl_source=runtime-element-count-abi-parameter
// SOURCE-SAME: runtime_vl_source=tcrv_rvv.setvl
// SOURCE-SAME: runtime_vl_scope=tcrv_rvv.with_vl
// SOURCE: /* control_plane_runtime_avl: body index argument maps to target/export-owned runtime n ABI parameter */
// SOURCE: /* control_plane_vl: !tcrv_rvv.vl value consumed by tcrv_rvv.with_vl */
// SOURCE: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_sub -> tcrv_rvv.i32_store */
// SOURCE: /* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
// SOURCE: /* dataflow_emission_step[1]: op=tcrv_rvv.i32_load, role=rhs-input-buffer, result=rhs_vec */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* selected_vector_shape_config: shape=i32m2, sew=32, lmul=m2, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i32_m2.sew32 rvv.i32_m2.lmul_m2 rvv.i32_m2.tail_policy.agnostic rvv.i32_m2.mask_policy.agnostic */
// SOURCE: /* control_plane_config: sew=32, lmul=m2, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
// SOURCE: /* intrinsic_config: vector_type=vint32m2_t, vector_suffix=i32m2, setvl_suffix=e32m2, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: /* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
// SOURCE: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: /* runtime_callable_abi: void tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n) */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vsetvl_e32m2
// SOURCE: __riscv_vle32_v_i32m2
// SOURCE: __riscv_vsub_vv_i32m2
// SOURCE: __riscv_vse32_v_i32m2
// SOURCE: void tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice

// HEADER: /* selected_body_authority: tcrv_rvv.i32_vsub_microkernel */
// HEADER: /* selected_binary_config: dtype=i32, family=i32-vsub
// HEADER-SAME: runtime_element_count_c_name=n
// HEADER-SAME: selected_role=direct variant */
// HEADER: /* selected_runtime_vl_boundary: runtime_element_count_c_name=n
// HEADER-SAME: runtime_avl_source=runtime-element-count-abi-parameter
// HEADER-SAME: runtime_vl_source=tcrv_rvv.setvl
// HEADER-SAME: runtime_vl_scope=tcrv_rvv.with_vl
// HEADER: /* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
// HEADER: /* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
// HEADER: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// HEADER: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// HEADER: /* runtime_callable_abi: void tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n) */
// HEADER: #ifndef TIANCHENRV_RVV_I32_VSUB_MICROKERNEL_FRONTEND_I32_VSUB_RVV_FIRST_SLICE_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: void tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #endif /* TIANCHENRV_RVV_I32_VSUB_MICROKERNEL_FRONTEND_I32_VSUB_RVV_FIRST_SLICE_H */

// OBJECT: Format: elf64-littleriscv
// OBJECT: Type: Relocatable
// OBJECT: Machine: EM_RISCV
// OBJECT: Name: tcrv_rvv_i32_vsub_microkernel_frontend_i32_vsub_rvv_first_slice

// STALE-SHAPE: selected RVV variant @rvv_first_slice
// STALE-SHAPE: selected vector-shape id must be 'i32m2'

// STALE-ABI: TianChen-RV execution plan coherence check failed for kernel @frontend_i32_vsub
// STALE-ABI-SAME: duplicate runtime ABI parameter role 'lhs-input-buffer'
