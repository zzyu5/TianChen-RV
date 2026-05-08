// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-smoke-probe-c | FileCheck %s --check-prefix=SMOKE --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=secret
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | sed '0,/tcrv.exec.kernel @profile_backed_smoke/{s/selected_march = "rv64gcv"/selected_march = "rv64gcv_bad"/; s/value = "rv64gcv"/value = "rv64gcv_bad"/;}' | not tcrv-translate --tcrv-export-rvv-smoke-probe-c 2>&1 | FileCheck %s --check-prefix=MISMATCH --implicit-check-not="#include <riscv_vector.h>" --implicit-check-not="requires available preserved selected_march metadata" --implicit-check-not=password --implicit-check-not=token --implicit-check-not=secret

module {
  tcrv.exec.target @smoke_rvv_profile {
    id = "rvv.profile.smoke",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.probe.compile_run", "rvv.toolchain.march"],
    architecture = "riscv64",
    count = 64 : i64,
    isa_vector_hints = "rv64gcv_zvl128b",
    selected_mabi = "lp64d",
    selected_march = "rv64gcv",
    status = "available",
    value = "rv64gcv"
  }

  tcrv.exec.kernel @profile_backed_smoke attributes {target = @smoke_rvv_profile} {
  }
}

// SMOKE: /* TianChen-RV RVV smoke-probe C export. */
// SMOKE: /* Scope: hardware/toolchain smoke probe derived from selected RVV metadata. */
// SMOKE: /* This is not TianChen-RV kernel lowering, runtime support, kernel correctness evidence, or performance evidence. */
// SMOKE: /* Deterministic probe count: 1 */
// SMOKE: #include <riscv_vector.h>
// SMOKE-LABEL: /* probe[0] function: tcrv_rvv_smoke_probe_0_profile_backed_smoke_rvv_first_slice */
// SMOKE: /* probe[0] selected_kernel: @profile_backed_smoke */
// SMOKE: /* probe[0] selected_variant: @rvv_first_slice */
// SMOKE: /* probe[0] selected_role: direct variant */
// SMOKE: /* probe[0] selected_march: rv64gcv */
// SMOKE: /* probe[0] selected_mabi: lp64d */
// SMOKE: /* probe[0] lowering_boundary: tcrv_rvv.lowering_boundary */
// SMOKE: /* probe[0] required_capabilities: @smoke_rvv_profile */
// SMOKE-LABEL: static int tcrv_rvv_smoke_probe_0_profile_backed_smoke_rvv_first_slice(void)
// SMOKE: __riscv_vsetvl_e32m1
// SMOKE: __riscv_vadd_vv_i32m1
// SMOKE: printf("tcrv_rvv_smoke_probe_ok probes=%zu\n", (size_t)1);

// MISMATCH: RVV smoke-probe C export failed
// MISMATCH: selected RVV variant @rvv_first_slice 'tcrv_rvv.required_march' metadata is not satisfied by preserved selected_march capability metadata
