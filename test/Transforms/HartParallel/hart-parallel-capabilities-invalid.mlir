// RUN: not weft-opt %s --weft-check-hart-parallel-capabilities 2>&1 | FileCheck %s

weft.exec.kernel @too_many_harts attributes {} {
  weft.exec.capability @rvv_hart_count {
    id = "rvv.hart_count",
    kind = "uarch",
    relations = #weft.capability_relations<provides = ["target.hart_count"]>,
    count = 64 : i64,
    status = "available"
  }
  weft.exec.variant @rvv_parallel_path attributes {
    origin = "rvv-plugin",
    requires = [@rvv_hart_count]
  } {
    // CHECK: error: Weft-RV hart_parallel capability check failed: requested 128 harts
    // CHECK-SAME: kernel @too_many_harts
    // CHECK-SAME: exceeds capability @rvv_hart_count
    // CHECK-SAME: provides = "target.hart_count"
    // CHECK-SAME: count = 64
    weft.exec.hart_parallel attributes {harts = 128 : i64, policy = "static"} {
    }
  }
}
