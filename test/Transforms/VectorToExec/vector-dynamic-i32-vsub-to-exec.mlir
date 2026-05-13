// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=i32_vadd --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not="__builtin_trap" --implicit-check-not="__riscv_vadd" --implicit-check-not=i32_vadd --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/tcrv_rvv.element_count = 16 : i64/s//tcrv_rvv.element_count = 16 : i64, tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DESC --implicit-check-not="TianChen-RV RVV runtime-callable microkernel C export." --implicit-check-not=__riscv_vsub_vv_i32m1
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/name = "tcrv_rvv.selected_vector_sew"/s//name = "tcrv_rvv.stale_selected_vector_sew"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-SEW --implicit-check-not="void tcrv_rvv_i32_vsub"
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/value = "tcrv_rvv.with_vl"/s//value = "descriptor-element-count"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-VL-SCOPE --implicit-check-not="void tcrv_rvv_i32_vsub"

module {
  tcrv.exec.target @vector_dynamic_vsub_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.dynamic.vsub.frontend",
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

  func.func @source_vector_dynamic_vsub(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_i32_vsub",
        tcrv_frontend_lowering = "i32-vsub",
        tcrv_frontend_target = @vector_dynamic_vsub_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %diff = arith.subi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %diff, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_vector_dynamic_i32_vsub
// LOWER-SAME: target = @vector_dynamic_vsub_frontend_profile
// LOWER-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// LOWER-SAME: tcrv_frontend_lowering = "i32-vsub"
// LOWER-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// LOWER-SAME: tcrv_frontend_runtime_extent_arg = "n"
// LOWER-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// LOWER-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vsub.v1"
// LOWER-SAME: tcrv_frontend_source_loop_step = 16 : i64
// LOWER-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// LOWER-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64
// LOWER: tcrv.exec.runtime_param @abi_runtime_element_count
// LOWER-SAME: abi_role = "runtime-element-count"
// LOWER-SAME: c_name = "n"
// LOWER-SAME: c_type = "size_t"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER-SAME: purpose = "runtime-abi-scalar"
// LOWER-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// LOWER-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// LOWER-SAME: tcrv_frontend_runtime_extent_arg = "n"
// LOWER-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// LOWER-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vsub.v1"
// LOWER-SAME: tcrv_frontend_source_loop_step = 16 : i64
// LOWER-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// LOWER-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64

// PIPE-LABEL: tcrv.exec.kernel @frontend_vector_dynamic_i32_vsub
// PIPE-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vsub.v1"
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-SAME: tcrv_rvv.selected_binary_family = "i32-vsub"
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
// PIPE: tcrv_rvv.i32_vsub_microkernel
// PIPE-SAME: element_count = 16 : i64
// PIPE: ^bb0(%[[N:.*]]: index):
// PIPE: %[[VL:.*]] = tcrv_rvv.setvl %[[N]]
// PIPE: tcrv_rvv.with_vl %[[VL]]
// PIPE: tcrv_rvv.i32_sub
// PIPE: name = "tcrv_rvv.runtime_avl_source"
// PIPE-SAME: role = "rvv-runtime-vl-avl-boundary"
// PIPE-SAME: value = "runtime-element-count-abi-parameter"
// PIPE: name = "tcrv_rvv.runtime_vl_source"
// PIPE-SAME: role = "rvv-runtime-vl-avl-boundary"
// PIPE-SAME: value = "tcrv_rvv.setvl"
// PIPE: name = "tcrv_frontend.source_kind"
// PIPE-SAME: role = "source-frontdoor-runtime-avl-authority"
// PIPE-SAME: value = "mlir-vector-scf-runtime-i32-vsub.v1"
// PIPE: name = "tcrv_frontend.source_authority"
// PIPE-SAME: value = "source-scf-for-runtime-upper-bound"
// PIPE: name = "tcrv_frontend.active_lane_authority"
// PIPE-SAME: value = "mlir-vector-transfer-tail-active-lanes"
// PIPE: name = "tcrv_frontend.source_tail_policy"
// PIPE-SAME: value = "runtime-n-bounded-transfer-tail-padding-and-store"
// PIPE: name = "tcrv_frontend.runtime_element_count_constraint"
// PIPE-SAME: value = "source-runtime-extent"
// PIPE: name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: role = "typed-rvv-emitc-source-op"
// PIPE-SAME: value = "tcrv_rvv.i32_sub"
// PIPE: name = "tcrv_rvv.emitc_lowerable_op_interface"
// PIPE-SAME: role = "typed-rvv-emitc-source-op"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"
// PIPE: name = "tcrv_rvv.descriptor_element_count"
// PIPE-SAME: role = "rvv-descriptor-local-component-capacity"
// PIPE-SAME: value = "16"

// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vsub_microkernel */
// SOURCE: /* arithmetic_family: i32-vsub */
// SOURCE: /* selected_binary_config: {{.*}}family=i32-vsub
// SOURCE-SAME: runtime_extent_arg=n
// SOURCE-SAME: source_loop_step=16
// SOURCE-SAME: source_vector_chunk_extent=16
// SOURCE-SAME: active_lane_authority=mlir-vector-transfer-tail-active-lanes
// SOURCE-SAME: source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store
// SOURCE-SAME: runtime_element_count_constraint=source-runtime-extent
// SOURCE: /* selected_runtime_vl_boundary: {{.*}}runtime_avl_source=runtime-element-count-abi-parameter{{.*}}runtime_vl_source=tcrv_rvv.setvl{{.*}}runtime_vl_scope=tcrv_rvv.with_vl{{.*}}runtime_extent_arg=n
// SOURCE: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vsub.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// SOURCE: /* runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted for this dynamic vector route */
// SOURCE: /* arithmetic_source: typed op tcrv_rvv.i32_sub via generated EmitC route and IR-backed callable ABI */
// SOURCE: /* descriptor_mirror_status: optional legacy descriptor metadata is compatibility/diagnostic only after typed RVV body authority; it cannot select emitted compute semantics */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++ */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_sub tcrv_rvv.i32_store */
// SOURCE: /* emitc.call_opaque[3]: __riscv_vsub_vv_i32m1 from tcrv_rvv.i32_sub */
// SOURCE: void tcrv_rvv_i32_vsub_microkernel_frontend_vector_dynamic_i32_vsub_rvv_first_slice

// MISSING-SEW: requires selected_plan_metadata 'tcrv_rvv.selected_vector_sew'
// STALE-VL-SCOPE: selected_plan_metadata 'tcrv_rvv.runtime_vl_scope'
// STALE-VL-SCOPE-SAME: must use value 'tcrv_rvv.with_vl'
// STALE-DESC: legacy RVV binary descriptor mirror 'i32-vadd-microkernel.v1'
// STALE-DESC-SAME: typed RVV authority from direct-typed-microkernel-body names family 'i32-vsub'
// STALE-DESC-SAME: descriptor metadata is non-authoritative mirror metadata
