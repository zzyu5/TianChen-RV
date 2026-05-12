// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | not tcrv-translate --tcrv-export-rvv-microkernel-c 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>"

module {
  tcrv.exec.kernel @scalar_only_microkernel_export {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: TianChen-RV target source artifact export failed
// CHECK-SAME: exact target artifact route 'tcrv-export-rvv-microkernel-c'
// CHECK-SAME: requires exactly one selected emission-plan candidate; found none
