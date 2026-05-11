// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=IR
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline --tcrv-check-emission-paths | FileCheck %s --check-prefix=READY
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=EXPORT --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=tcrv_rvv_microkernel_ok --implicit-check-not=emission_manifest --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module @rvv_auto_microkernel_input {
  // The input intentionally has capabilities only: no hand-authored
  // tcrv_rvv.i32_vadd_microkernel op appears here.
  tcrv.exec.kernel @auto_i32_vadd {
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
// IR-NOT: tcrv.exec.dispatch
// IR: tcrv.exec.diagnostic
// IR-SAME: selection_kind = "static-variant"
// IR-SAME: target = @rvv_first_slice
// IR: tcrv_rvv.lowering_boundary
// IR-SAME: required_capabilities = [@rvv]
// IR-SAME: role = "direct variant"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: source_kernel = "auto_i32_vadd"
// IR-SAME: status = "unsupported"
// IR: tcrv_rvv.i32_vadd_microkernel
// IR-SAME: element_count = 16 : i64
// IR-SAME: origin = "rvv-plugin"
// IR-SAME: required_capabilities = [@rvv]
// IR-SAME: required_march = "rv64gcv"
// IR-SAME: role = "direct variant"
// IR-SAME: selected_mabi = "lp64d"
// IR-SAME: selected_variant = @rvv_first_slice
// IR-SAME: source_kernel = "auto_i32_vadd"
// IR: ^bb0(%[[N:.*]]: index):
// IR: %[[VL:.*]] = tcrv_rvv.setvl %[[N]]
// IR-SAME: lmul = "m1"
// IR-SAME: policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
// IR-SAME: sew = 32 : i64
// IR: tcrv_rvv.with_vl %[[VL]] attributes
// IR-SAME: lmul = "m1"
// IR-SAME: policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
// IR-SAME: sew = 32 : i64
// IR: %[[LHS:.*]] = tcrv_rvv.i32_load %[[VL]]
// IR-SAME: buffer_role = "lhs-input-buffer"
// IR: %[[RHS:.*]] = tcrv_rvv.i32_load %[[VL]]
// IR-SAME: buffer_role = "rhs-input-buffer"
// IR: %[[SUM:.*]] = tcrv_rvv.i32_add %[[LHS]], %[[RHS]], %[[VL]]
// IR: tcrv_rvv.i32_store %[[SUM]], %[[VL]]
// IR-SAME: buffer_role = "output-buffer"
// IR: tcrv.exec.diagnostic
// IR-SAME: artifact_kind = "runtime-callable-c-source"
// IR-SAME: emission_kind = "rvv-explicit-i32-vadd-microkernel-c-source"
// IR-SAME: lowering_boundary = "tcrv_rvv.lowering_boundary"
// IR-SAME: lowering_pipeline = "tcrv-export-rvv-microkernel-c"
// IR-SAME: reason = "emission_plan"
// IR-SAME: required_capabilities = [@rvv]
// IR-SAME: role = "direct variant"
// IR-SAME: runtime_abi = "rvv-i32-vadd-runtime-callable-c-abi.v1"
// IR-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// IR-SAME: runtime_abi_name = "rvv-i32-vadd-runtime-callable-c-function.v1"
// IR-SAME: runtime_glue_role = "runtime-callable-i32-vadd-function"
// IR-SAME: status = "supported"
// IR-SAME: target = @rvv_first_slice

// EXPORT: /* TianChen-RV RVV runtime-callable microkernel C export. */
// EXPORT: /* Scope: library-style C source for exactly one tcrv_rvv.i32_vadd_microkernel. */
// EXPORT: /* Route: verified RVV family ops lower through the plugin-local EmitC intrinsic route before C/C++ emission. */
// EXPORT: /* Default artifact shape: runtime-callable C ABI function with no embedded main or self-check harness. */
// EXPORT: #include <riscv_vector.h>
// EXPORT-LABEL: /* microkernel function: tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice */
// EXPORT: /* selected_kernel: @auto_i32_vadd */
// EXPORT: /* selected_variant: @rvv_first_slice */
// EXPORT: /* selected_role: direct variant */
// EXPORT: /* selected_march: rv64gcv */
// EXPORT: /* selected_mabi: lp64d */
// EXPORT: /* lowering_boundary: tcrv_rvv.lowering_boundary */
// EXPORT: /* executable_microkernel: tcrv_rvv.i32_vadd_microkernel */
// EXPORT: /* control_plane_body: tcrv_rvv.setvl -> tcrv_rvv.with_vl */
// EXPORT: /* control_plane_runtime_avl: body index argument maps to target/export-owned runtime n ABI parameter */
// EXPORT: /* control_plane_vl: !tcrv_rvv.vl value consumed by tcrv_rvv.with_vl */
// EXPORT: /* dataflow_body: tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_add -> tcrv_rvv.i32_store */
// EXPORT: /* dataflow_abi_roles: lhs_load.buffer_role=lhs-input-buffer, rhs_load.buffer_role=rhs-input-buffer, store.buffer_role=output-buffer; runtime n remains the target/export-owned runtime element-count ABI parameter */
// EXPORT: /* dataflow_emission_step[2]: op=tcrv_rvv.i32_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// EXPORT: /* emitc_route: tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++ */
// EXPORT: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// EXPORT: /* emitc_route_headers: <stddef.h> <stdint.h> <riscv_vector.h> */
// EXPORT: /* emitc_route_source_ops: tcrv_rvv.setvl tcrv_rvv.with_vl tcrv_rvv.i32_load tcrv_rvv.i32_load tcrv_rvv.i32_add tcrv_rvv.i32_store */
// EXPORT: /* emitc.call_opaque[0]: __riscv_vsetvl_e32m1 from tcrv_rvv.setvl */
// EXPORT: /* emitc.call_opaque[3]: __riscv_vadd_vv_i32m1 from tcrv_rvv.i32_add */
// EXPORT: /* control_plane_config: sew=32, lmul=m1, policy=#tcrv_rvv.policy<tail = agnostic, mask = agnostic> */
// EXPORT: /* intrinsic_config_source: validated tcrv_rvv.setvl and tcrv_rvv.with_vl SEW/LMUL/policy metadata */
// EXPORT: /* intrinsic_config: vector_type=vint32m1_t, vector_suffix=i32m1, setvl_suffix=e32m1, tail_policy=agnostic, mask_policy=agnostic */
// EXPORT: /* artifact_kind: runtime-callable-c-source */
// EXPORT: /* element_count: 16 */
// EXPORT: /* required_capabilities: @rvv */
// EXPORT: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// EXPORT: /* runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// EXPORT: /* runtime_callable_abi: void tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice
// EXPORT: void tcrv_rvv_i32_vadd_microkernel_auto_i32_vadd_rvv_first_slice
// EXPORT: while (offset < n)
// EXPORT: __riscv_vsetvl_e32m1
// EXPORT: __riscv_vle32_v_i32m1
// EXPORT: __riscv_vadd_vv_i32m1
// EXPORT: __riscv_vse32_v_i32m1
