// RUN: not tcrv-translate --tcrv-rvv-emitc-to-cpp %s 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>"

module {
  tcrv.exec.kernel @rvv_i32_add_kernel {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}

// CHECK: TianChen-RV target artifact export failed for kernel @rvv_i32_add_kernel
// CHECK-SAME: requires a selected path surface before exporting a target artifact
