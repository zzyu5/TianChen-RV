// REQUIRES: tianchenrv-local-rvv-object-clang

// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-i32-vsub-microkernel-object > %t.i32m2-vsub.o
// RUN: test -s %t.i32m2-vsub.o
// RUN: llvm-readobj --file-headers --symbols %t.i32m2-vsub.o | FileCheck %s --check-prefixes=OBJ,VSUBOBJ --implicit-check-not="Name: main"

module @rvv_microkernel_i32m2_object_input {
  func.func @high_level_placeholder() {
    return
  }

  tcrv.exec.kernel @object_i32m2_vsub attributes {
    tcrv_frontend_lowering = "i32-vsub"
  } {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m2.sew32", "rvv.i32_m2.lmul_m2", "rvv.i32_m2.tail_policy.agnostic", "rvv.i32_m2.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m2",
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

// OBJ: Format: elf
// OBJ: Arch: riscv
// OBJ: Type: Relocatable

// VSUBOBJ: Name: tcrv_rvv_i32_vsub_microkernel_object_i32m2_vsub_rvv_first_slice

