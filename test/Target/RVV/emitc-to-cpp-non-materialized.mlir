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

// CHECK: TianChen-RV RVV EmitC C/C++ translate route failed:
// CHECK-SAME: requires an already materialized EmitC module
// CHECK-SAME: found non-EmitC op 'tcrv.exec.
