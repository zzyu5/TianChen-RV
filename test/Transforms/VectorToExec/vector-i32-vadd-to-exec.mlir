// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=lowering_descriptor --implicit-check-not=int64_t --implicit-check-not=__riscv_vsub_vv_i32m1 --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module {
  // LOWER-LABEL: tcrv.exec.target @vector_frontend_rvv_scalar_profile
  // PIPE-LABEL: tcrv.exec.target @vector_frontend_rvv_scalar_profile
  tcrv.exec.target @vector_frontend_rvv_scalar_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.frontend",
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

  func.func @source_vector_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_i32_vadd",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_frontend_rvv_scalar_profile
      } {
    %c0 = arith.constant 0 : index
    %pad = arith.constant 0 : i32
    %lhs_vec = vector.transfer_read %lhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %rhs_vec = vector.transfer_read %rhs[%c0], %pad {in_bounds = [true]} : memref<?xi32>, vector<16xi32>
    %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
    vector.transfer_write %sum, %out[%c0] {in_bounds = [true]} : vector<16xi32>, memref<?xi32>
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_vector_i32_vadd
// LOWER-SAME: target = @vector_frontend_rvv_scalar_profile
// LOWER-SAME: tcrv_frontend_lowering = "i32-vadd"
// LOWER: tcrv.exec.mem_window @abi_lhs_input_buffer
// LOWER-SAME: abi_role = "lhs-input-buffer"
// LOWER-SAME: access = "read"
// LOWER-SAME: c_type = "const int32_t *"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER-SAME: purpose = "runtime-abi-buffer"
// LOWER: tcrv.exec.mem_window @abi_rhs_input_buffer
// LOWER-SAME: abi_role = "rhs-input-buffer"
// LOWER-SAME: access = "read"
// LOWER-SAME: c_type = "const int32_t *"
// LOWER: tcrv.exec.mem_window @abi_output_buffer
// LOWER-SAME: abi_role = "output-buffer"
// LOWER-SAME: access = "write"
// LOWER-SAME: c_type = "int32_t *"
// LOWER: tcrv.exec.runtime_param @abi_runtime_element_count
// LOWER-SAME: abi_role = "runtime-element-count"
// LOWER-SAME: c_name = "n"
// LOWER-SAME: c_type = "size_t"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER-SAME: purpose = "runtime-abi-scalar"

// PIPE-LABEL: tcrv.exec.kernel @frontend_vector_i32_vadd
// PIPE-SAME: target = @vector_frontend_rvv_scalar_profile
// PIPE-SAME: tcrv_frontend_lowering = "i32-vadd"
// PIPE: tcrv.exec.mem_window @abi_lhs_input_buffer
// PIPE: tcrv.exec.runtime_param @abi_runtime_element_count
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: requires = [@vector_frontend_rvv_scalar_profile]
// PIPE-SAME: tcrv_rvv.base_i32_m1_lanes = 4 : i64
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-NOT: tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"
// PIPE-SAME: tcrv_rvv.required_march = "rv64gcv"
// PIPE-SAME: tcrv_rvv.selected_setvl_suffix = "e32m1"
// PIPE-SAME: tcrv_rvv.selected_vector_lmul = "m1"
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
// PIPE-SAME: tcrv_rvv.selected_vector_type = "vint32m1_t"
// PIPE-SAME: tcrv_rvv.vlenb_bytes = 16 : i64
// PIPE: tcrv.exec.variant @scalar_fallback_first_slice
// PIPE-SAME: origin = "scalar-plugin"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: reason = "variant-selected"
// PIPE-SAME: selection_kind = "static-variant"
// PIPE-SAME: target = @rvv_first_slice
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: origin = "rvv-plugin"
// PIPE-SAME: required_capabilities = [@vector_frontend_rvv_scalar_profile]
// PIPE-SAME: role = "direct variant"
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: source_kernel = "frontend_vector_i32_vadd"
// PIPE: tcrv_rvv.i32_vadd_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: required_capabilities = [@vector_frontend_rvv_scalar_profile]
// PIPE-SAME: selected_variant = @rvv_first_slice
// PIPE-SAME: selected_vector_shape = "i32m1"
// PIPE-SAME: source_kernel = "frontend_vector_i32_vadd"
// PIPE: ^bb0(%[[N:.*]]: index):
// PIPE: %[[VL:.*]] = tcrv_rvv.setvl %[[N]]
// PIPE-SAME: lmul = "m1"
// PIPE-SAME: policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
// PIPE-SAME: sew = 32 : i64
// PIPE: tcrv_rvv.with_vl %[[VL]]
// PIPE: %[[LHS_VEC:.*]] = tcrv_rvv.i32_load %[[VL]]
// PIPE-SAME: buffer_role = "lhs-input-buffer"
// PIPE: %[[RHS_VEC:.*]] = tcrv_rvv.i32_load %[[VL]]
// PIPE-SAME: buffer_role = "rhs-input-buffer"
// PIPE: %[[SUM_VEC:.*]] = tcrv_rvv.i32_add %[[LHS_VEC]], %[[RHS_VEC]], %[[VL]]
// PIPE: tcrv_rvv.i32_store %[[SUM_VEC]], %[[VL]]
// PIPE-SAME: buffer_role = "output-buffer"
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
// PIPE-SAME: reason = "emission_plan"
// PIPE-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
// PIPE-SAME: status = "supported"
// PIPE-SAME: target = @rvv_first_slice

// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vadd_microkernel */
// SOURCE: /* arithmetic_family: i32-vadd */
// SOURCE: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_add -> tcrv_rvv.i32_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++ */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_add tcrv_rvv.i32_store */
// SOURCE: /* emitc.call_opaque[3]: __riscv_vadd_vv_i32m1 from tcrv_rvv.i32_add */
// SOURCE: /* selected_vector_shape_config: shape=i32m1, sew=32, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1 */
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i32_vadd_microkernel_frontend_vector_i32_vadd_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_frontend_vector_i32_vadd_rvv_first_slice
