// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-microkernel-c | FileCheck %s --check-prefix=LIB --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=emission_manifest --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-microkernel-self-check-c | FileCheck %s --check-prefix=HARNESS --implicit-check-not=emission_manifest --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: sed '/%lhs = tcrv_rvv.i32_load/s/buffer_role = "lhs-input-buffer"/buffer_role = "output-buffer"/' %s | not tcrv-opt - --tcrv-materialize-selected-lowering-boundaries 2>&1 | FileCheck %s --check-prefix=BAD-DATAFLOW-ROLE --implicit-check-not="__riscv_vle32_v_i32m1"

module @rvv_microkernel_input {
  tcrv.exec.kernel @micro_a {
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
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
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
// LIB: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_add -> tcrv_rvv.i32_store */
// LIB: /* dataflow_emission_source: derived from verified tcrv_rvv.with_vl body order, SSA chain, and buffer_role attributes */
// LIB: /* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
// LIB: /* dataflow_emission_step[0]: op=tcrv_rvv.i32_load, role=lhs-input-buffer, result=lhs_vec */
// LIB: /* dataflow_emission_step[1]: op=tcrv_rvv.i32_load, role=rhs-input-buffer, result=rhs_vec */
// LIB: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// LIB: /* dataflow_emission_step[3]: op=tcrv_rvv.i32_store, role=output-buffer, value=sum_vec */
// LIB: /* emitc_lowerable_interface: TCRVEmitCLowerableInterface */
// LIB: /* emitc_materialization_boundary: verified MLIR EmitC module with emitc.include, emitc.func, and emitc.call_opaque before bounded legacy C source output */
// LIB: /* emitc_materialization_function: @tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice */
// LIB: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// LIB: /* emitc_route_id: tcrv-export-rvv-microkernel-c, route_kind=extension-family-ops-to-emitc-call-opaque */
// LIB: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_add tcrv_rvv.i32_store */
// LIB: /* emitc.call_opaque[3]: __riscv_vadd_vv_i32m1 from tcrv_rvv.i32_add */
// LIB: /* emitc.call_opaque_boundary[3]: source_role=compute, operands=3, result=sum_vec:vint32m1_t, op_interface=TCRVEmitCLowerableOpInterface */
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
// HARNESS: /* Harness capacity comes from descriptor-local element_count; each call still supplies runtime n through the generated C ABI. */
// HARNESS-LABEL: static int tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice_self_check_one(size_t runtime_n)
// HARNESS: enum { kTCRVMicrokernelCapacity = 16 };
// HARNESS: tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice(lhs, rhs, out, runtime_n);
// HARNESS: for (size_t index = runtime_n; index < (size_t)kTCRVMicrokernelCapacity; ++index)
// HARNESS-LABEL: int main(void)
// HARNESS: enum { kTCRVMicrokernelShortRuntimeN = kTCRVMicrokernelCapacity >= 7 ? 7 : kTCRVMicrokernelCapacity };
// HARNESS: tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice_self_check_one((size_t)kTCRVMicrokernelShortRuntimeN);
// HARNESS: tcrv_rvv_i32_vadd_microkernel_micro_a_rvv_first_slice_self_check_one((size_t)kTCRVMicrokernelCapacity);
// HARNESS: printf("tcrv_rvv_microkernel_ok runtime_counts=%zu,%zu\n", (size_t)kTCRVMicrokernelShortRuntimeN, (size_t)kTCRVMicrokernelCapacity);

// BAD-DATAFLOW-ROLE: tcrv_rvv.i32_vadd_microkernel
// BAD-DATAFLOW-ROLE-SAME: requires first tcrv_rvv.i32_load to reference runtime ABI role 'lhs-input-buffer'
