// RUN: weft-opt %s --weft-check-hart-parallel-capabilities -split-input-file | FileCheck %s

// CHECK-LABEL: weft.exec.kernel @bounded_hart_parallel
weft.exec.kernel @bounded_hart_parallel attributes {} {
  // CHECK: weft.exec.capability @rvv_hart_count
  // CHECK-SAME: count = 64
  // CHECK-SAME: relations = #weft.capability_relations<provides = ["target.hart_count"]>
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
    // CHECK: weft.exec.hart_parallel
    // CHECK-SAME: harts = 32
    weft.exec.hart_parallel attributes {harts = 32 : i64, policy = "static"} {
    }
  }
}

// -----

// CHECK-LABEL: weft.exec.kernel @unspecified_hart_count_request
weft.exec.kernel @unspecified_hart_count_request attributes {} {
  weft.exec.variant @policy_only_parallel_path attributes {
    origin = "generic-plugin",
    requires = []
  } {
    // CHECK: weft.exec.hart_parallel
    // CHECK-SAME: policy = "runtime-owned"
    // CHECK-NOT: harts
    weft.exec.hart_parallel attributes {policy = "runtime-owned"} {
    }
  }
}
