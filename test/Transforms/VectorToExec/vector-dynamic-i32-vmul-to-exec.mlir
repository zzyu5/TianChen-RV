// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec | FileCheck %s --check-prefix=LOWER --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE --implicit-check-not=func.func --implicit-check-not=vector.transfer_read --implicit-check-not=vector.transfer_write --implicit-check-not=tcrv_frontend_source_vector_extent --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_rvv.i32_vsub_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=must-equal-fixed-source-vector-extent --implicit-check-not="__builtin_trap" --implicit-check-not="__riscv_vadd" --implicit-check-not="__riscv_vsub" --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not="lhs[index] + rhs[index]" --implicit-check-not="lhs[index] - rhs[index]" --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | sed '0,/tcrv_rvv.element_count = 16 : i64/s//tcrv_rvv.element_count = 16 : i64, tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-DESC --implicit-check-not="TianChen-RV RVV runtime-callable microkernel C export." --implicit-check-not="void tcrv_rvv_i32_vmul"
// RUN: tcrv-opt %s --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline | python3 -c 'import re, sys; text=sys.stdin.read(); pattern=r"(\{name = \"tcrv_rvv\.emitc_arithmetic_intrinsic\"[^}]*value = )\"__riscv_vmul_vv_i32m1\""; text,count=re.subn(pattern, lambda m: m.group(1) + "\"__riscv_vadd_vv_i32m1\"", text, count=1); assert count == 1; sys.stdout.write(text)' | not tcrv-translate --tcrv-export-rvv-microkernel-self-check-c 2>&1 | FileCheck %s --check-prefix=STALE-EMITC-BODY-MAPPING --implicit-check-not="#include <riscv_vector.h>"

module {
  tcrv.exec.target @vector_dynamic_vmul_frontend_profile {
    architecture = "riscv64",
    count = 64 : i64,
    id = "rvv.profile.vector.dynamic.vmul.frontend",
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

  func.func @source_vector_dynamic_vmul(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)
      attributes {
        tcrv_frontend_kernel = "frontend_vector_dynamic_i32_vmul",
        tcrv_frontend_lowering = "i32-vmul",
        tcrv_frontend_target = @vector_dynamic_vmul_frontend_profile
      } {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %n step %c16 {
      %pad = arith.constant 0 : i32
      %lhs_vec = vector.transfer_read %lhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %rhs_vec = vector.transfer_read %rhs[%i], %pad : memref<?xi32>, vector<16xi32>
      %product = arith.muli %lhs_vec, %rhs_vec : vector<16xi32>
      vector.transfer_write %product, %out[%i] : vector<16xi32>, memref<?xi32>
    }
    return
  }
}

// LOWER-LABEL: tcrv.exec.kernel @frontend_vector_dynamic_i32_vmul
// LOWER-SAME: target = @vector_dynamic_vmul_frontend_profile
// LOWER-SAME: tcrv_frontend_active_lane_authority = "mlir-vector-transfer-tail-active-lanes"
// LOWER-SAME: tcrv_frontend_lowering = "i32-vmul"
// LOWER-SAME: tcrv_frontend_runtime_element_count_constraint = "source-runtime-extent"
// LOWER-SAME: tcrv_frontend_runtime_extent_arg = "n"
// LOWER-SAME: tcrv_frontend_source_authority = "source-scf-for-runtime-upper-bound"
// LOWER-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vmul.v1"
// LOWER-SAME: tcrv_frontend_source_loop_step = 16 : i64
// LOWER-SAME: tcrv_frontend_source_tail_policy = "runtime-n-bounded-transfer-tail-padding-and-store"
// LOWER-SAME: tcrv_frontend_source_vector_chunk_extent = 16 : i64
// LOWER: tcrv.exec.runtime_param @abi_runtime_element_count
// LOWER-SAME: abi_role = "runtime-element-count"
// LOWER-SAME: c_name = "n"
// LOWER-SAME: c_type = "size_t"
// LOWER-SAME: ownership = "target-export-abi-owned"
// LOWER-SAME: purpose = "runtime-abi-scalar"
// LOWER-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vmul.v1"

// PIPE-LABEL: tcrv.exec.kernel @frontend_vector_dynamic_i32_vmul
// PIPE-SAME: tcrv_frontend_source_kind = "mlir-vector-scf-runtime-i32-vmul.v1"
// PIPE: tcrv.exec.variant @rvv_first_slice
// PIPE-SAME: tcrv_rvv.element_count = 16 : i64
// PIPE-SAME: tcrv_rvv.selected_binary_family = "i32-vmul"
// PIPE-SAME: tcrv_rvv.selected_vector_shape = "i32m1"
// PIPE: tcrv_rvv.i32_vmul_microkernel attributes
// PIPE-SAME: element_count = 16 : i64
// PIPE-SAME: emitc_lowerable_op_interface = "TCRVEmitCLowerableOpInterface"
// PIPE-SAME: emitc_source_op = "tcrv_rvv.i32_mul"
// PIPE-SAME: selected_binary_dtype = "i32"
// PIPE-SAME: selected_binary_family = "i32-vmul"
// PIPE-SAME: selected_binary_microkernel_op = "tcrv_rvv.i32_vmul_microkernel"
// PIPE-SAME: selected_binary_operator = "multiply"
// PIPE-SAME: selected_binary_source_kind = "frontend-lowering"
// PIPE: tcrv_rvv.i32_mul
// PIPE: name = "tcrv_frontend.source_kind"
// PIPE-SAME: role = "source-frontdoor-runtime-avl-authority"
// PIPE-SAME: value = "mlir-vector-scf-runtime-i32-vmul.v1"
// PIPE: name = "tcrv_rvv.emitc_source_op"
// PIPE-SAME: role = "typed-rvv-emitc-source-op"
// PIPE-SAME: value = "tcrv_rvv.i32_mul"
// PIPE: name = "tcrv_rvv.emitc_arithmetic_intrinsic"
// PIPE-SAME: value = "__riscv_vmul_vv_i32m1"

// SOURCE: /* executable_microkernel: tcrv_rvv.i32_vmul_microkernel */
// SOURCE: /* arithmetic_family: i32-vmul */
// SOURCE: /* selected_binary_config: {{.*}}family=i32-vmul
// SOURCE-SAME: runtime_extent_arg=n
// SOURCE-SAME: source_loop_step=16
// SOURCE-SAME: source_vector_chunk_extent=16
// SOURCE-SAME: active_lane_authority=mlir-vector-transfer-tail-active-lanes
// SOURCE-SAME: source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store
// SOURCE-SAME: runtime_element_count_constraint=source-runtime-extent
// SOURCE: /* source_frontend_runtime_avl_authority: source_kind=mlir-vector-scf-runtime-i32-vmul.v1, source_authority=source-scf-for-runtime-upper-bound, runtime_extent_arg=n, source_loop_step=16, source_vector_chunk_extent=16, active_lane_authority=mlir-vector-transfer-tail-active-lanes, source_tail_policy=runtime-n-bounded-transfer-tail-padding-and-store, runtime_element_count_constraint=source-runtime-extent */
// SOURCE: /* arithmetic_source: typed op tcrv_rvv.i32_mul via generated EmitC route and IR-backed callable ABI */
// SOURCE: /* emitc_body_mapping: route_kind=extension-family-ops-to-emitc-call-opaque, source_authority=mlir-emitc-cpp-emitter, required_header=riscv_vector.h, arithmetic_intrinsic=__riscv_vmul_vv_i32m1 */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_mul, lhs=lhs_vec, rhs=rhs_vec, result=product_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_mul tcrv_rvv.i32_store */
// SOURCE: /* emitc.call_opaque[3]: __riscv_vmul_vv_i32m1 from tcrv_rvv.i32_mul */
// SOURCE: void tcrv_rvv_i32_vmul_microkernel_frontend_vector_dynamic_i32_vmul_rvv_first_slice

// STALE-DESC: legacy RVV binary descriptor mirror 'i32-vadd-microkernel.v1'
// STALE-DESC-SAME: typed RVV authority from direct-typed-microkernel-body names family 'i32-vmul'
// STALE-DESC-SAME: descriptor metadata is non-authoritative mirror metadata
// STALE-EMITC-BODY-MAPPING: selected RVV EmitC body mapping
// STALE-EMITC-BODY-MAPPING-SAME: selected_plan_metadata 'tcrv_rvv.emitc_arithmetic_intrinsic'
// STALE-EMITC-BODY-MAPPING-SAME: EmitC arithmetic intrinsic must be '__riscv_vmul_vv_i32m1'
