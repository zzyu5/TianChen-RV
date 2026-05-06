// RUN: tcrv-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: tcrv.exec.kernel @saxpy
tcrv.exec.kernel @saxpy attributes {} {
  // CHECK: tcrv.exec.target @rvv_main
  tcrv.exec.target @rvv_main {cores = 64 : i64, isa = ["rv64", "rvv"], vlen = 256 : i64}

  // CHECK: tcrv.exec.capability @rvv
  tcrv.exec.capability @rvv {kind = "isa", version = "1.0"}

  // CHECK: tcrv.exec.variant @rvv_variant
  tcrv.exec.variant @rvv_variant attributes {origin = "rvv-plugin", requires = [@rvv]} {
  }

  // CHECK: tcrv.exec.dispatch
  tcrv.exec.dispatch attributes {} {
    // CHECK: tcrv.exec.fallback @rvv_variant
    tcrv.exec.fallback @rvv_variant
  }
}
