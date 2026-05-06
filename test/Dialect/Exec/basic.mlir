// RUN: tcrv-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: tcrv.exec.kernel @saxpy
tcrv.exec.kernel @saxpy attributes {} {
  // CHECK: tcrv.exec.target @rvv_main
  tcrv.exec.target @rvv_main {arch = "riscv64", cores = 64 : i64, isa = ["rv64", "rvv"], vlen = 256 : i64}

  // CHECK: tcrv.exec.capability @rvv
  // CHECK-SAME: id = "rvv"
  // CHECK-SAME: kind = "isa-vector"
  tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", version = "1.0"}

  // CHECK: tcrv.exec.variant @rvv_variant
  // CHECK-SAME: origin = "rvv-plugin"
  // CHECK-SAME: requires = [@rvv]
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }

  // CHECK: tcrv.exec.dispatch
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.fallback @rvv_variant
    tcrv.exec.fallback @rvv_variant
  }
}
