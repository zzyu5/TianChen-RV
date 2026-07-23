// RUN: not weft-opt %s --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
  weft.exec.kernel @rvv_capability_no_body attributes {construction_domain = "riscv-execution", problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
    weft.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    weft.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    weft.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
  }
}

// CHECK: collected no viable plugin proposals
// CHECK-SAME: rvv-plugin:
// CHECK-SAME: explicit typed weft_rvv extension-family IR
