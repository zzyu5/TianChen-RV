// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-smoke-probe-c | FileCheck %s --implicit-check-not=emission_manifest --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module @rvv_smoke_probe_inputs {
  tcrv.exec.kernel @smoke_b {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
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
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }

  tcrv.exec.kernel @smoke_a {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i32_m1.sew32", "rvv.i32_m1.lmul_m1", "rvv.i32_m1.tail_policy.agnostic", "rvv.i32_m1.mask_policy.agnostic"],
      sew_bits = 32 : i64,
      lmul = "m1",
      tail_policy = "agnostic",
      mask_policy = "agnostic",
      architecture = "riscv64",
      isa_vector_hints = "rv64gcv_zvl128b",
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
      selected_mabi = "lp64d",
      selected_march = "rv64gcv",
      status = "available"
    }
    tcrv.exec.capability @rvv_toolchain_march {
      id = "rvv.toolchain.march",
      kind = "toolchain",
      status = "available",
      value = "rv64gcv"
    }
    tcrv.exec.capability @rvv_toolchain_mabi {
      id = "rvv.toolchain.mabi",
      kind = "toolchain",
      status = "available",
      value = "lp64d"
    }
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }
  }
}

// CHECK: /* TianChen-RV RVV smoke-probe C export. */
// CHECK: /* Scope: hardware/toolchain smoke probe derived from selected RVV metadata. */
// CHECK: /* This is not TianChen-RV kernel lowering, runtime support, kernel correctness evidence, or performance evidence. */
// CHECK: /* Deterministic probe count: 2 */
// CHECK: #include <riscv_vector.h>

// CHECK-LABEL: /* probe[0] function: tcrv_rvv_smoke_probe_0_smoke_a_rvv_first_slice */
// CHECK: /* probe[0] selected_kernel: @smoke_a */
// CHECK: /* probe[0] selected_variant: @rvv_first_slice */
// CHECK: /* probe[0] selected_role: direct variant */
// CHECK: /* probe[0] selected_march: rv64gcv */
// CHECK: /* probe[0] selected_mabi: lp64d */
// CHECK: /* probe[0] lowering_boundary: tcrv_rvv.lowering_boundary */
// CHECK: /* probe[0] required_capabilities: @rvv */
// CHECK-LABEL: static int tcrv_rvv_smoke_probe_0_smoke_a_rvv_first_slice(void)
// CHECK: __riscv_vsetvl_e32m1
// CHECK: __riscv_vle32_v_i32m1
// CHECK: __riscv_vadd_vv_i32m1
// CHECK: __riscv_vse32_v_i32m1

// CHECK-LABEL: /* probe[1] function: tcrv_rvv_smoke_probe_1_smoke_b_rvv_first_slice */
// CHECK: /* probe[1] selected_kernel: @smoke_b */
// CHECK: /* probe[1] selected_variant: @rvv_first_slice */
// CHECK: /* probe[1] selected_role: direct variant */
// CHECK: /* probe[1] selected_march: rv64gcv */
// CHECK: /* probe[1] selected_mabi: lp64d */
// CHECK-LABEL: static int tcrv_rvv_smoke_probe_1_smoke_b_rvv_first_slice(void)
// CHECK: int main(void)
// CHECK: tcrv_rvv_smoke_probe_0_smoke_a_rvv_first_slice();
// CHECK: tcrv_rvv_smoke_probe_1_smoke_b_rvv_first_slice();
// CHECK: printf("tcrv_rvv_smoke_probe_ok probes=%zu\n", (size_t)2);
