// RUN: tcrv-opt %s --tcrv-check-hart-parallel-capabilities -split-input-file | FileCheck %s

// CHECK-LABEL: tcrv.exec.kernel @bounded_hart_parallel
tcrv.exec.kernel @bounded_hart_parallel attributes {} {
  // CHECK: tcrv.exec.capability @rvv_hart_count
  // CHECK-SAME: count = 64
  // CHECK-SAME: relations = #tcrv.capability_relations<provides = ["target.hart_count"]>
  tcrv.exec.capability @rvv_hart_count {
    id = "rvv.hart_count",
    kind = "uarch",
    relations = #tcrv.capability_relations<provides = ["target.hart_count"]>,
    count = 64 : i64,
    status = "available"
  }
  tcrv.exec.variant @rvv_parallel_path attributes {
    origin = "rvv-plugin",
    requires = [@rvv_hart_count]
  } {
    // CHECK: tcrv.exec.hart_parallel
    // CHECK-SAME: harts = 32
    tcrv.exec.hart_parallel attributes {harts = 32 : i64, policy = "static"} {
    }
  }
}

// -----

// CHECK-LABEL: tcrv.exec.kernel @unspecified_hart_count_request
tcrv.exec.kernel @unspecified_hart_count_request attributes {} {
  tcrv.exec.variant @policy_only_parallel_path attributes {
    origin = "generic-plugin",
    requires = []
  } {
    // CHECK: tcrv.exec.hart_parallel
    // CHECK-SAME: policy = "runtime-owned"
    // CHECK-NOT: harts
    tcrv.exec.hart_parallel attributes {policy = "runtime-owned"} {
    }
  }
}
