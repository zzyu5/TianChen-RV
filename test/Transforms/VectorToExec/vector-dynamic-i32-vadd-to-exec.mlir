// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not="__builtin_trap" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/tcrv_rvv.element_count = 16 : i64/s//tcrv_rvv.element_count = 16 : i64, tcrv_rvv.lowering_descriptor = "i32-vsub-microkernel.v1"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DESC --implicit-check-not="TianChen-RV RVV runtime-callable microkernel C export." --implicit-check-not="void tcrv_rvv_i32_vadd"
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_frontend\.runtime_extent_arg\"[^}]*value = )\"n\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"len\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-PLAN-METADATA --implicit-check-not="TianChen-RV RVV runtime-callable microkernel C export."
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import sys; text=sys.stdin.read(); marker="tcrv.exec.runtime_param @abi_runtime_element_count {"; i=text.index(marker); needle=", tcrv_frontend_source_loop_step = 16 : i64"; j=text.index(needle, i); sys.stdout.write(text[:j]+text[j+len(needle):])' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-PARAM-STEP --implicit-check-not="TianChen-RV RVV runtime-callable microkernel C export."
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_frontend\.active_lane_authority\"[^}]*value = )\"mlir-vector-transfer-tail-active-lanes\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"stale-active-lanes\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-ACTIVE-LANES --implicit-check-not="TianChen-RV RVV runtime-callable microkernel C export."
// RUN: tcrv-opt %s --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import sys; text=sys.stdin.read(); marker="tcrv.exec.runtime_param @abi_runtime_element_count {"; i=text.index(marker); needle=", tcrv_frontend_active_lane_authority = \"mlir-vector-transfer-tail-active-lanes\""; j=text.index(needle, i); sys.stdout.write(text[:j]+text[j+len(needle):])' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-ACTIVE-LANES --implicit-check-not="TianChen-RV RVV runtime-callable microkernel C export."

module {
  // LOWER-LABEL: tcrv.exec.target @vector_dynamic_frontend_profile
  // PIPE-LABEL: tcrv.exec.target @vector_dynamic_frontend_profile
  tcrv.exec.target @vector_dynamic_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.dynamic.frontend",
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

  func.func @source_vector_dynamic_vadd(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_i32_vadd",
        tcrv_frontend_lowering = "i32-vadd",
        tcrv_frontend_target = @vector_dynamic_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %sum = arith.addi %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %sum, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_vector_dynamic_i32_vadd
// LOWER-SAME: target = @vector_dynamic_frontend_profile
// LOWER-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// LOWER-SAME: tcrv_frontend_lowering = "i32-vadd"
// LOWER-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// LOWER-SAME: tcrv_frontend_runtime_extent_arg = "n"
// LOWER-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// LOWER-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vadd.v1"
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
// LOWER-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vadd.v1"
// LOWER-SAME: tcrv_frontend_source_loop_step = 16 : i64
// LOWER-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// LOWER-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64

// PIPE-LABEL: tcrv.exec.kernel @frontend_vector_dynamic_i32_vadd
// PIPE-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// PIPE-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// PIPE-SAME: tcrv_frontend_runtime_extent_arg = "n"
// PIPE-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// PIPE-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vadd.v1"
// PIPE-SAME: tcrv_frontend_source_loop_step = 16 : i64
// PIPE-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// PIPE-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
// PIPE: tcrv_rvv.i32_vadd_microkernel attributes
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface"
// PIPE-SAME: emitc_source_op = "tcrv_rvv.i32_add"
// PIPE-SAME: selected_binary_dtype = "i32"
// PIPE-SAME: selected_binary_family = "i32-vadd"
// PIPE-SAME: selected_binary_microkernel_op = "tcrv_rvv.i32_vadd_microkernel"
// PIPE-SAME: selected_binary_operator = "add"
// PIPE-SAME: selected_binary_source_kind = "frontend-lowering"
// PIPE: ^bb0(%[[N:.*]]: index):
// PIPE: %[[VL:.*]] = tcrv_rvv.setvl %[[N]]
// PIPE: tcrv_rvv.with_vl %[[VL]]
// PIPE: name = "tcrv_frontend.source_kind"
// PIPE-SAME: role = "source-frontdoor-runtime-avl-authority"
// PIPE-SAME: value = "mlir-vector-scf-runtime-i32-vadd.v1"
// PIPE: name = "tcrv_frontend.source_authority"
// PIPE-SAME: value = "source-scf-for-runtime-upper-bound"
// PIPE: name = "tcrv_frontend.runtime_extent_arg"
// PIPE-SAME: value = "n"
// PIPE: name = "tcrv_frontend.source_loop_step"
// PIPE-SAME: value = "16"
// PIPE: name = "tcrv_frontend.source_vector_chunk_extent"
// PIPE-SAME: value = "16"
// PIPE: name = "tcrv_frontend.active_lane_authority"
// PIPE-SAME: value = "mlir-vector-transfer-tail-active-lanes"
// PIPE: name = "tcrv_frontend.source_tail_policy"
// PIPE-SAME: value = "runtime-n-bounded-transfer-tail-padding-and-store"
// PIPE: name = "tcrv_frontend.runtime_element_count_constraint"
// PIPE-SAME: value = "source-runtime-extent"
// PIPE: name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: role = "typed-rvv-emitc-source-op"
// PIPE-SAME: value = "tcrv_rvv.i32_add"
// PIPE: name = "tcrv_rvv.emitc_lowerable_op_interface"
// PIPE-SAME: role = "typed-rvv-emitc-source-op"
// PIPE-SAME: value = "TCRVEmitCLowerableOpInterface"
// PIPE: name = "tcrv_rvv.descriptor_element_count"
// PIPE-SAME: role = "rvv-descriptor-local-component-capacity"
// PIPE-SAME: value = "16"

// SOURCE: /* selected_binary_config: {{.*}}descriptor_element_count=16, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent
// SOURCE: /* selected_runtime_vl_boundary: {{.*}}runtime_avl_source=runtime-element-count-abi-parameter{{.*}}runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent
// SOURCE: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vadd.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// SOURCE: /* runtime_element_count_source: n is the source scf.for upper bound and runtime AVL; no fixed source-extent trap is emitted for this dynamic vector route */
// SOURCE: /* arithmetic_source: typed op tcrv_rvv.i32_add via generated EmitC route and IR-backed callable ABI */
// SOURCE: /* descriptor_mirror_status: optional legacy descriptor metadata is compatibility/diagnostic only after typed RVV body authority; it cannot select emitted compute semantics */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++ */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_add tcrv_rvv.i32_store */
// SOURCE: /* emitc.call_opaque[0]: __riscv_vsetvl_e32m1 from tcrv_rvv.setvl */
// SOURCE: /* emitc.call_opaque[3]: __riscv_vadd_vv_i32m1 from tcrv_rvv.i32_add */
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_frontend_vector_dynamic_i32_vadd_rvv_first_slice

// STALE-DESC: legacy RVV binary descriptor mirror 'i32-vsub-microkernel.v1'
// STALE-DESC-SAME: typed RVV authority from direct-typed-microkernel-body names family 'i32-vadd'
// STALE-DESC-SAME: descriptor metadata is non-authoritative mirror metadata

// STALE-PLAN-METADATA: selected_plan_metadata 'tcrv_frontend.runtime_extent_arg' frontend runtime extent arg must be 'n'

// MISSING-PARAM-STEP: dynamic vector runtime extent authority validation failed for kernel @frontend_vector_dynamic_i32_vadd: runtime_param requires integer attribute 'tcrv_frontend_source_loop_step' for source scf.for step

// STALE-ACTIVE-LANES: selected_plan_metadata 'tcrv_frontend.active_lane_authority' frontend active-lane authority must be 'mlir-vector-transfer-tail-active-lanes'

// MISSING-ACTIVE-LANES: dynamic vector runtime extent authority validation failed for kernel @frontend_vector_dynamic_i32_vadd: dynamic vector runtime extent authority requires string attribute 'tcrv_frontend_active_lane_authority' on both kernel and runtime_param
