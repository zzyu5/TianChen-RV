// RUN: not tcrv-opt %s --tcrv-execution-planning-pipeline 2>&1 | FileCheck %s

module {
  tcrv.exec.kernel @pipeline_no_viable_after_rvv_decline {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}

// CHECK: error: TianChen-RV plugin variant materialization for kernel @pipeline_no_viable_after_rvv_decline collected no viable plugin proposals; recoverable plugin declines in registration order: rvv-plugin:
// CHECK-SAME: explicit typed tcrv_rvv extension-family IR
