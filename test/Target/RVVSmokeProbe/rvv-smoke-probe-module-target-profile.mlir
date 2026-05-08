// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=secret
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-smoke-probe-c | FileCheck %s --check-prefix=SMOKE --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=secret
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=GENERIC --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token --implicit-check-not=secret
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | sed '0,/tcrv.exec.kernel @profile_backed_smoke/{s/selected_march = "rv64gcv"/selected_march = "rv64gcv_bad"/; s/value = "rv64gcv"/value = "rv64gcv_bad"/;}' | not tcrv-translate --tcrv-export-rvv-smoke-probe-c 2>&1 | FileCheck %s --check-prefix=MISMATCH --implicit-check-not="#include <riscv_vector.h>" --implicit-check-not="requires available preserved selected_march metadata" --implicit-check-not=password --implicit-check-not=token --implicit-check-not=secret

module {
  tcrv.exec.target @smoke_rvv_profile {
    id = "rvv.profile.smoke",
    kind = "profile",
    provides = ["rvv", "rvv.hart_count", "rvv.probe.compile_run", "rvv.toolchain.march", "rvv.smoke_probe"],
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

// IR: tcrv.exec.variant @rvv_first_slice
// IR-SAME: requires = [@smoke_rvv_profile]
// IR-SAME: tcrv_rvv.required_march = "rv64gcv"
// IR-SAME: tcrv_rvv.smoke_probe_descriptor = "standalone-c-toolchain-smoke-probe.v1"
// IR-NOT: tcrv_rvv.i32_vadd_microkernel
// IR: tcrv.exec.diagnostic {artifact_kind = "standalone-c-source"
// IR-SAME: emission_kind = "rvv-smoke-probe-standalone-c-source"
// IR-SAME: lowering_pipeline = "tcrv-export-rvv-smoke-probe-c"
// IR-SAME: required_capabilities = [@smoke_rvv_profile]
// IR-SAME: runtime_abi = "rvv-smoke-probe-standalone-c-main.v1"
// IR-SAME: runtime_abi_kind = "rvv-smoke-probe-standalone-c-main"
// IR-SAME: runtime_abi_name = "rvv-smoke-probe-standalone-c-main.v1"
// IR-SAME: runtime_glue_role = "rvv-smoke-probe-standalone-main"
// IR-SAME: status = "supported"

// SMOKE: /* TianChen-RV RVV smoke-probe C export. */
// SMOKE: /* Scope: hardware/toolchain smoke probe derived from selected RVV metadata. */
// SMOKE: /* This is not TianChen-RV kernel lowering, runtime support, kernel correctness evidence, or performance evidence. */
// SMOKE: /* Deterministic probe count: 1 */
// SMOKE: /* origin_plugin: rvv-plugin */
// SMOKE: /* emission_kind: rvv-smoke-probe-standalone-c-source */
// SMOKE: /* route: tcrv-export-rvv-smoke-probe-c */
// SMOKE: /* artifact_kind: standalone-c-source */
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

// GENERIC: /* TianChen-RV RVV smoke-probe C export. */
// GENERIC: /* Scope: hardware/toolchain smoke probe derived from selected RVV metadata. */
// GENERIC: /* This is not TianChen-RV kernel lowering, runtime support, kernel correctness evidence, or performance evidence. */
// GENERIC: /* Deterministic probe count: 1 */
// GENERIC: /* origin_plugin: rvv-plugin */
// GENERIC: /* emission_kind: rvv-smoke-probe-standalone-c-source */
// GENERIC: /* route: tcrv-export-rvv-smoke-probe-c */
// GENERIC: /* artifact_kind: standalone-c-source */
// GENERIC: #include <riscv_vector.h>
// GENERIC-LABEL: /* probe[0] function: tcrv_rvv_smoke_probe_0_profile_backed_smoke_rvv_first_slice */
// GENERIC: /* probe[0] selected_march: rv64gcv */
// GENERIC: /* probe[0] selected_mabi: lp64d */
// GENERIC: /* probe[0] required_capabilities: @smoke_rvv_profile */
// GENERIC-LABEL: static int tcrv_rvv_smoke_probe_0_profile_backed_smoke_rvv_first_slice(void)
// GENERIC: __riscv_vsetvl_e32m1
// GENERIC: __riscv_vadd_vv_i32m1
// GENERIC: printf("tcrv_rvv_smoke_probe_ok probes=%zu\n", (size_t)1);

// MISMATCH: RVV smoke-probe C export failed
// MISMATCH: selected RVV variant @rvv_first_slice 'tcrv_rvv.required_march' metadata is not satisfied by preserved selected_march capability metadata
