// RUN: not tcrv-translate --tcrv-export-rvv-smoke-probe-c %s 2>&1 | FileCheck %s --implicit-check-not="#include <riscv_vector.h>" --implicit-check-not="__riscv_"

// CHECK: Unknown command line argument
// CHECK-SAME: tcrv-export-rvv-smoke-probe-c

module {
}
