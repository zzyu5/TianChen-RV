// RUN: not weft-opt %s --weft-execution-planning-pipeline 2>&1 | FileCheck %s

module {
  weft.exec.kernel @pipeline_no_viable_after_rvv_decline attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.int8_mac_problem @canonical_problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      status = "available"
    }
  }
}

// CHECK: error: Weft-RV plugin variant materialization for kernel @pipeline_no_viable_after_rvv_decline collected no viable plugin proposals; recoverable plugin declines in registration order: rvv-plugin:
// CHECK-SAME: explicit typed weft_rvv extension-family IR
