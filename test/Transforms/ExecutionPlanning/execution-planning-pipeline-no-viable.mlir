// RUN: not weft-opt %s --weft-execution-planning-pipeline 2>&1 | FileCheck %s

module {
  weft.exec.kernel @pipeline_no_viable_after_rvv_decline {
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}

// CHECK: error: Weft-RV plugin variant materialization for kernel @pipeline_no_viable_after_rvv_decline collected no viable plugin proposals; recoverable plugin declines in registration order: rvv-plugin:
// CHECK-SAME: explicit typed weft_rvv extension-family IR
