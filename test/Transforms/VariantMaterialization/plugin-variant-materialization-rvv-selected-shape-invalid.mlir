// RUN: not tcrv-opt %s --tcrv-materialize-plugin-variants 2>&1 | FileCheck %s --implicit-check-not=tcrv_rvv.i32_vsub_microkernel --implicit-check-not="emission_plan" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module {
  tcrv.exec.kernel @selector_i32m2_missing_lmul attributes {
    tcrv_frontend_lowering = "i32-vsub"
  } {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_binary_selected_shape {
      id = "rvv.i32_binary.selected_vector_shape",
      kind = "isa-vector-config",
      shape = "i32m2",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_sew32 {
      id = "rvv.i32_m1.sew32",
      kind = "isa-vector-config",
      sew_bits = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_lmul_m1 {
      id = "rvv.i32_m1.lmul_m1",
      kind = "isa-vector-config",
      lmul = "m1",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_tail_agnostic {
      id = "rvv.i32_m1.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m1_mask_agnostic {
      id = "rvv.i32_m1.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m2_sew32 {
      id = "rvv.i32_m2.sew32",
      kind = "isa-vector-config",
      sew_bits = 32 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m2_tail_agnostic {
      id = "rvv.i32_m2.tail_policy.agnostic",
      kind = "isa-vector-config",
      tail_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_i32_m2_mask_agnostic {
      id = "rvv.i32_m2.mask_policy.agnostic",
      kind = "isa-vector-config",
      mask_policy = "agnostic",
      status = "available"
    }
    tcrv.exec.capability @rvv_hart_count {
      id = "rvv.hart_count",
      kind = "uarch",
      count = 64 : i64,
      status = "available"
    }
    tcrv.exec.capability @rvv_probe_compile_run {
      id = "rvv.probe.compile_run",
      kind = "toolchain",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      value = "rv64gcv",
      status = "available"
    }
  }
}

// CHECK: no viable plugin proposals
// CHECK: rvv-plugin:
// CHECK: RVV property decision requires capability id 'rvv.i32_m2.lmul_m2'
