// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-microkernel-c | FileCheck %s --check-prefix=LIB --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=emission_manifest --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-microkernel-self-check-c | FileCheck %s --check-prefix=HARNESS --implicit-check-not=emission_manifest --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_input {
  tcrv.exec.kernel @micro_a {
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
    tcrv.exec.variant @rvv_first_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv"
    } {
    }
    tcrv.exec.variant @scalar_fallback_first_slice attributes {
      fallback_role = "conservative",
      origin = "scalar-plugin",
      policy = "portable_scalar_fallback_first_slice",
      requires = [@scalar_fallback]
    } {
    }
    tcrv.exec.dispatch {
      tcrv.exec.case @rvv_first_slice {
        condition = "rvv_capability_properties_available",
        guard = "plugin_local_rvv_property_evidence",
        origin = "rvv-plugin",
        policy = "metadata_only_first_slice"
      }
      tcrv.exec.fallback @scalar_fallback_first_slice {
        fallback_role = "conservative",
        origin = "scalar-plugin"
      }
    }
    tcrv_rvv.i32_vadd_microkernel attributes {
      element_count = 16 : i64,
      origin = "rvv-plugin",
      required_capabilities = [@rvv],
      required_march = "rv64gcv",
      role = "dispatch case",
      selected_mabi = "lp64d",
      selected_variant = @rvv_first_slice,
      source_kernel = "micro_a"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        tcrv_rvv.i32_vadd_dataflow {lhs_role = "lhs-input-buffer", out_role = "output-buffer", rhs_role = "rhs-input-buffer", runtime_n_role = "runtime-element-count"}
      } : !tcrv_rvv.vl
    }
  }
}

// LIB: /* TianChen-RV RVV runtime-callable microkernel C export. */
// LIB: /* Scope: library-style C source for exactly one tcrv_rvv.i32_vadd_microkernel. */
// LIB: /* Default artifact shape: runtime-callable C ABI function with no embedded main or self-check harness. */
// LIB: #include <riscv_vector.h>
// LIB-LABEL: /* microkernel function: tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice */
// LIB: /* selected_kernel: @micro_a */
// LIB: /* selected_variant: @rvv_first_slice */
// LIB: /* selected_role: dispatch case */
// LIB: /* selected_march: rv64gcv */
// LIB: /* selected_mabi: lp64d */
// LIB: /* lowering_boundary: tcrv_rvv.lowering_boundary */
// LIB: /* executable_microkernel: tcrv_rvv.i32_vadd_microkernel */
// LIB: /* control_plane_body: tcrv_rvv.setvl -> tcrv_rvv.with_vl */
// LIB: /* control_plane_runtime_avl: body index argument maps to target/export-owned runtime n ABI parameter */
// LIB: /* control_plane_vl: !tcrv_rvv.vl value consumed by tcrv_rvv.with_vl */
// LIB: /* dataflow_body: tcrv_rvv.i32_vadd_dataflow runtime ABI role references */
// LIB: /* dataflow_abi_roles: lhs_role=lhs-input-buffer, rhs_role=rhs-input-buffer, out_role=output-buffer, runtime_n_role=runtime-element-count */
// LIB: /* control_plane_config: sew=32, lmul=m1, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
// LIB: /* artifact_kind: runtime-callable-c-source */
// LIB: /* element_count: 16 */
// LIB: /* required_capabilities: @rvv */
// LIB: /* callable_abi_source: tcrv.exec.mem_window + tcrv.exec.runtime_param */
// LIB: /* callable_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int32_t * */
// LIB: /* callable_runtime_param[0]: symbol=@abi_runtime_element_count, abi_role=runtime-element-count, c_name=n, c_type=size_t, ownership=target-export-abi-owned */
// LIB: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// LIB: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// LIB: /* runtime_callable_abi: void tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice
// LIB: void tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice
// LIB: while (offset < n)
// LIB: __riscv_vsetvl_e32m1
// LIB: __riscv_vle32_v_i32m1
// LIB: __riscv_vadd_vv_i32m1
// LIB: __riscv_vse32_v_i32m1

// HARNESS: /* Harness mode: adds a bounded self-check main for explicit ssh rvv evidence only. */
// HARNESS: #include <stdio.h>
// HARNESS: void tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice
// HARNESS: __riscv_vadd_vv_i32m1
// HARNESS-LABEL: static int tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice_self_check(void)
// HARNESS: enum { kTCRVMicrokernelElements = 16 };
// HARNESS: tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice(lhs, rhs, out, (size_t)kTCRVMicrokernelElements);
// HARNESS-LABEL: int main(void)
// HARNESS: tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice_self_check();
// HARNESS: printf("tcrv_rvv_microkernel_ok elements=%zu\n", (size_t)16);
