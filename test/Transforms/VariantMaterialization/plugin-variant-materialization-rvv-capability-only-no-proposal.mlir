// RUN: not weft-opt %s --weft-materialize-plugin-variants 2>&1 | FileCheck %s

module {
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
  weft.exec.target @rvv_only_profile {id = "rvv.only.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@rvv, @rvv_hart_count, @rvv_probe_compile_run]}
  weft.exec.kernel @rvv_capability_no_body attributes {target = @rvv_only_profile, problem = @problem} {
    weft.exec.int8_mac_problem @problem {lhs_signedness = #weft<integer_signedness signed>, rhs_signedness = #weft<integer_signedness signed>, m = 4 : i64, n = 4 : i64, k = 8 : i64}
  }
}

// CHECK: collected no viable plugin proposals
// CHECK-SAME: no enabled extension plugin produced a viable proposal
