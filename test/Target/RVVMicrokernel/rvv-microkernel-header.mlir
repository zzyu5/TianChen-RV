// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-microkernel-header > %t.direct.h
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not="out[" --implicit-check-not="int main" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not=riscv_vector --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.direct.h
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-header-artifact > %t.generic.h
// RUN: cmp %t.direct.h %t.generic.h
// RUN: FileCheck %s --check-prefix=GENERIC-HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not="out[" --implicit-check-not="int main" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not=riscv_vector --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.generic.h
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not="#ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H"
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP

module @rvv_microkernel_header_input {
  tcrv.exec.kernel @header_i32_vadd {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
  }
}

// HEADER: /* TianChen-RV RVV runtime-callable microkernel C header. */
// HEADER: /* selected_body_authority: tcrv_rvv.i32_vadd_microkernel */
// HEADER: /* selected_binary_config: dtype=i32, family=i32-vadd
// HEADER-SAME: runtime_element_count_c_name=n
// HEADER-SAME: selected_role=direct variant */
// HEADER: /* selected_runtime_vl_boundary: runtime_element_count_c_name=n
// HEADER-SAME: runtime_avl_source=runtime-element-count-abi-parameter
// HEADER-SAME: runtime_vl_source=tcrv_rvv.setvl
// HEADER-SAME: runtime_vl_scope=tcrv_rvv.with_vl
// HEADER: /* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
// HEADER: /* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
// HEADER: /* callable_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer
// HEADER-SAME: c_type=const int32_t *
// HEADER: /* callable_runtime_param[0]: symbol=@abi_runtime_element_count, abi_role=runtime-element-count, c_name=n, c_type=size_t, ownership=target-export-abi-owned */
// HEADER: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// HEADER: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// HEADER: /* runtime_callable_abi: void tcrv_rvv_i32_vadd_microkernel_header_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n) */
// HEADER: #ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H
// HEADER-NEXT: #define TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H
// HEADER: #include <stddef.h>
// HEADER-NEXT: #include <stdint.h>
// HEADER: #ifdef __cplusplus
// HEADER-NEXT: extern "C"
// HEADER-NEXT: #endif
// HEADER: void tcrv_rvv_i32_vadd_microkernel_header_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// HEADER: #ifdef __cplusplus
// HEADER-NEXT: }
// HEADER-NEXT: #endif
// HEADER: #endif /* TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H */

// GENERIC-HEADER: /* selected_body_authority: tcrv_rvv.i32_vadd_microkernel */
// GENERIC-HEADER: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// GENERIC-HEADER: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// GENERIC-HEADER: #ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H
// GENERIC-HEADER: #include <stddef.h>
// GENERIC-HEADER: #include <stdint.h>
// GENERIC-HEADER: void tcrv_rvv_i32_vadd_microkernel_header_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// GENERIC-HEADER: #endif /* TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H */

// SOURCE: /* artifact_kind: runtime-callable-c-source */
// SOURCE: #include <riscv_vector.h>
// SOURCE: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// SOURCE: static void tcrv_rvv_i32_vadd_microkernel_header_i32_vadd_rvv_first_slice__tcrv_emitc_body
// SOURCE: __riscv_vadd_vv_i32m1
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_header_i32_vadd_rvv_first_slice

// HELP-DAG: --tcrv-export-rvv-microkernel-header
// HELP-DAG: --tcrv-export-rvv-i32-vsub-microkernel-c
// HELP-DAG: --tcrv-export-rvv-i32-vsub-microkernel-header
// HELP-DAG: --tcrv-export-rvv-i32-vmul-microkernel-c
// HELP-DAG: --tcrv-export-rvv-i32-vmul-microkernel-header
// HELP-DAG: --tcrv-export-target-header-artifact
