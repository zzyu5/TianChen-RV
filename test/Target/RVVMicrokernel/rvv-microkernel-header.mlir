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

// GENERIC-HEADER: #ifndef TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H
// GENERIC-HEADER: #include <stddef.h>
// GENERIC-HEADER: #include <stdint.h>
// GENERIC-HEADER: void tcrv_rvv_i32_vadd_microkernel_header_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n);
// GENERIC-HEADER: #endif /* TIANCHENRV_RVV_I32_VADD_MICROKERNEL_HEADER_I32_VADD_RVV_FIRST_SLICE_H */

// SOURCE: #include <riscv_vector.h>
// SOURCE: /* artifact_kind: runtime-callable-c-source */
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_header_i32_vadd_rvv_first_slice
// SOURCE: __riscv_vadd_vv_i32m1

// HELP-DAG: --tcrv-export-rvv-microkernel-header
// HELP-DAG: --tcrv-export-rvv-i32-vsub-microkernel-c
// HELP-DAG: --tcrv-export-rvv-i32-vsub-microkernel-header
// HELP-DAG: --tcrv-export-rvv-i32-vmul-microkernel-c
// HELP-DAG: --tcrv-export-rvv-i32-vmul-microkernel-header
// HELP-DAG: --tcrv-export-target-header-artifact
