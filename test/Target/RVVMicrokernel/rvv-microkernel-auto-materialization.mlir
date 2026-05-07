// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline --tcrv-check-emission-paths | FileCheck %s --check-prefix=READY
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=EXPORT --implicit-check-not=emission_manifest --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module @rvv_auto_microkernel_input {
  // The input intentionally has capabilities only: no hand-authored
  // tcrv_rvv.i32_vadd_microkernel op appears here.
  tcrv.exec.kernel @auto_i32_vadd {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
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

// IR-LABEL: tcrv.exec.kernel @auto_i32_vadd
// READY-LABEL: tcrv.exec.kernel @auto_i32_vadd
// READY: tcrv_rvv.i32_vadd_microkernel
// READY-SAME: element_count = 16 : i64
// READY-SAME: required_march = "rv64gcv"
// READY: tcrv.exec.diagnostic
// READY-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
// READY-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
// READY-SAME: status = "supported"
// IR: tcrv.exec.variant @rvv_first_slice
// IR-SAME: origin = "rvv-plugin"
// IR-SAME: requires = [@rvv]
// IR-SAME: tcrv_rvv.element_count = 16 : i64
// IR-SAME: tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"
// IR-SAME: tcrv_rvv.required_march = "rv64gcv"
// IR: tcrv.exec.case @rvv_first_slice
// IR: tcrv_rvv.lowering_boundary
// IR-SAME: required_capabilities = [@rvv]
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: source_kernel = "auto_i32_vadd"
// IR-SAME: status = "unsupported"
// IR: tcrv_rvv.i32_vadd_microkernel
// IR-SAME: element_count = 16 : i64
// IR-SAME: origin = "rvv-plugin"
// IR-SAME: required_capabilities = [@rvv]
// IR-SAME: required_march = "rv64gcv"
// IR-SAME: role = "dispatch case"
// IR-SAME: selected_mabi = "lp64d"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: source_kernel = "auto_i32_vadd"
// IR: tcrv.exec.diagnostic
// IR-SAME: artifact_kind = "standalone-c-source"
// IR-SAME: emission_kind = "rvv-explicit-i32-vadd-microkernel-c-source"
// IR-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
// IR-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
// IR-SAME: reason = "emission_plan"
// IR-SAME: required_capabilities = [@rvv]
// IR-SAME: role = "dispatch case"
// IR-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
// IR-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// IR-SAME: runtime_abi_name = "rvv-i32-vadd-runtime-callable-c-function.v1"
// IR-SAME: runtime_glue_role = "runtime-callable-i32-vadd-function"
// IR-SAME: status = "supported"
// IR-SAME: target = @rvv_first_slice

// EXPORT: /* TianChen-RV RVV explicit microkernel C export. */
// EXPORT: /* Scope: executable C for exactly one tcrv_rvv.i32_vadd_microkernel. */
// EXPORT: #include <riscv_vector.h>
// EXPORT-LABEL: /* microkernel function: tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice */
// EXPORT: /* selected_kernel: @auto_i32_vadd */
// EXPORT: /* selected_variant: @rvv_first_slice */
// EXPORT: /* selected_role: dispatch case */
// EXPORT: /* selected_march: rv64gcv */
// EXPORT: /* selected_mabi: lp64d */
// EXPORT: /* lowering_boundary: tcrv_rvv.lowering_boundary */
// EXPORT: /* executable_microkernel: tcrv_rvv.i32_vadd_microkernel */
// EXPORT: /* element_count: 16 */
// EXPORT: /* required_capabilities: @rvv */
// EXPORT-LABEL: void tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)
// EXPORT: while (offset < n)
// EXPORT: __riscv_vsetvl_e32m1
// EXPORT: __riscv_vle32_v_i32m1
// EXPORT: __riscv_vadd_vv_i32m1
// EXPORT: __riscv_vse32_v_i32m1
// EXPORT-LABEL: static int tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice_self_check(void)
// EXPORT: enum { kTCRVMicrokernelElements = 16 };
// EXPORT: tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice(lhs, rhs, out, (size_t)kTCRVMicrokernelElements);
// EXPORT-LABEL: int main(void)
// EXPORT: tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice_self_check();
// EXPORT: printf("tcrv_rvv_microkernel_ok elements=%zu\n", (size_t)16);
