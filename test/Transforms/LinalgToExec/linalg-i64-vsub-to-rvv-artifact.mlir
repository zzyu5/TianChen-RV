// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=linalg.generic --implicit-check-not=func.func --implicit-check-not=tcrv_rvv.i64_vadd_microkernel --implicit-check-not=tcrv_rvv.i64_vmul_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i64m1 --implicit-check-not=__riscv_vmul_vv_i64m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

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

  func.func @source_i64_vsub(%lhs: memref<?xi64>, %rhs: memref<?xi64>, %out: memref<?xi64>)
      attributes {
        tcrv_frontend_kernel = "frontend_i64_vsub",
        tcrv_frontend_target = @frontend_rvv_i64_profile
      } {
    linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"],
        tcrv_frontend_lowering = "i64-vsub"
      }
      ins(%lhs, %rhs : memref<?xi64>, memref<?xi64>)
      outs(%out : memref<?xi64>) {
    ^bb0(%a: i64, %b: i64, %old: i64):
      %diff = arith.subi %a, %b : i64
      linalg.yield %diff : i64
    }
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_i64_vsub
// LOWER-SAME: target = @frontend_rvv_i64_profile
// LOWER-SAME: tcrv_frontend_lowering = "i64-vsub"
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

// PIPE-LABEL: tcrv.exec.kernel @frontend_i64_vsub
// PIPE-SAME: target = @frontend_rvv_i64_profile
// PIPE-SAME: tcrv_frontend_lowering = "i64-vsub"
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: requires = [@frontend_rvv_i64_profile]
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-NOT: tcrv_rvv.lowering_descriptor
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
// PIPE-SAME: source_kernel = "frontend_i64_vsub"
// PIPE: tcrv_rvv.i64_vsub_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@frontend_rvv_i64_profile]
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_suffix = "i64m1"
// PIPE-SAME: source_kernel = "frontend_i64_vsub"
// PIPE: tcrv_rvv.setvl
// PIPE-SAME: lmul = "m1"
// PIPE-SAME: sew = 64 : i64
// PIPE: tcrv_rvv.i64_load
// PIPE-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_sub
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_store
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.vl
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "rvv-explicit-i64-vsub-microkernel-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-i64-vsub-microkernel-c"
// PIPE-SAME: runtime_abi = "rvv-i64-vsub-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// PIPE-SAME: runtime_abi_name = "rvv-i64-vsub-runtime-callable-c-function.v1"
// PIPE-SAME: runtime_abi_parameters = [{c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"}
// PIPE-SAME: {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"}
// PIPE-SAME: {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"}
// PIPE-SAME: {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}]
// PIPE-SAME: runtime_glue_role = "runtime-callable-i64-vsub-function"
// PIPE-SAME: selected_plan_metadata = [{name = "tcrv_rvv.selected_vector_shape"
// PIPE-SAME: value = "i64m1"}
// PIPE-SAME: {name = "tcrv_rvv.selected_vector_sew"
// PIPE-SAME: value = "64"}
// PIPE-SAME: {name = "tcrv_rvv.selected_vector_lmul"
// PIPE-SAME: value = "m1"}
// PIPE-SAME: {name = "tcrv_rvv.selected_setvl_suffix"
// PIPE-SAME: value = "e64m1"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_avl_source"
// PIPE-SAME: value = "runtime-element-count-abi-parameter"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_avl_role"
// PIPE-SAME: value = "runtime-element-count"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_vl_source"
// PIPE-SAME: value = "tcrv_rvv.setvl"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_vl_scope"
// PIPE-SAME: value = "tcrv_rvv.with_vl"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_dtype"
// PIPE-SAME: role = "typed-rvv-binary-source"
// PIPE-SAME: value = "i64"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_family"
// PIPE-SAME: value = "i64-vsub"}
// PIPE-SAME: {name = "tcrv_rvv.selected_binary_operator"
// PIPE-SAME: value = "subtract"}
// PIPE-SAME: {name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: value = "tcrv_rvv.i64_sub"}
// PIPE-SAME: {name = "tcrv_rvv.emitc_lowerable_op_interface"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"}
// PIPE-SAME: {name = "tcrv_rvv.runtime_element_count_c_name"
// PIPE-SAME: value = "n"}
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @rvv_first_slice

// SOURCE: /* executable_microkernel: tcrv_rvv.i64_vsub_microkernel */
// SOURCE: /* arithmetic_family: i64-vsub */
// SOURCE: /* dtype: i64 */
// SOURCE: /* active_route: tcrv-export-rvv-i64-vsub-microkernel-c */
// SOURCE: /* control_plane_runtime_avl: body index argument maps to target/export-owned runtime n ABI parameter */
// SOURCE: /* control_plane_vl: !tcrv_rvv.vl value consumed by tcrv_rvv.with_vl */
// SOURCE: /* dataflow_body: tcrv_rvv.i64_load -> tcrv_rvv.i64_load -> tcrv_rvv.i64_sub -> tcrv_rvv.i64_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i64_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* selected_vector_shape_config: dtype=i64, shape=i64m1, sew=64, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i64_m1.sew64 rvv.i64_m1.lmul_m1 rvv.i64_m1.tail_policy.agnostic rvv.i64_m1.mask_policy.agnostic */
// SOURCE: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int64_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[2]: c_name=out, c_type=int64_t *, role=output-buffer, ownership=target-export-abi-owned */
// SOURCE: void tcrv_rvv_i64_vsub_microkernel_frontend_i64_vsub_rvv_first_slice(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n)
// SOURCE: __riscv_vsetvl_e64m1(n - offset)
// SOURCE: __riscv_vle64_v_i64m1
// SOURCE: __riscv_vsub_vv_i64m1
// SOURCE: __riscv_vse64_v_i64m1
