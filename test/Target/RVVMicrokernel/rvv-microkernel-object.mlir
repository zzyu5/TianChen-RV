// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-microkernel-object > %t.direct.o
// RUN: test -s %t.direct.o
// RUN: llvm-readobj --file-headers --symbols %t.direct.o | FileCheck %s --check-prefixes=OBJ,DIRECTOBJ --implicit-check-not="Name: main"
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-artifact > %t.generic.o
// RUN: test -s %t.generic.o
// RUN: llvm-readobj --file-headers --symbols %t.generic.o | FileCheck %s --check-prefixes=OBJ,GENERICOBJ --implicit-check-not="Name: main"
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP

module @rvv_microkernel_object_input {
  tcrv.exec.kernel @static_object_i32_vadd {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
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

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable

// DIRECTOBJ: Name: tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice
// GENERICOBJ: Name: tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice

// SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// SOURCE: /* selected_kernel: @static_object_i32_vadd */
// SOURCE: /* selected_role: direct variant */
// SOURCE: /* selected_march: rv64gcv */
// SOURCE: /* selected_mabi: lp64d */
// SOURCE: /* artifact_kind: runtime-callable-c-source */
// SOURCE: void tcrv_rvv_i32_vadd_microkernel_static_object_i32_vadd_rvv_first_slice
// SOURCE: __riscv_vadd_vv_i32m1

// HELP: --tcrv-export-rvv-microkernel-object
