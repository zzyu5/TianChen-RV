// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c > %t.dispatch.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.c
// RUN: FileCheck %s --check-prefix=BODY --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.c
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c > %t.dispatch-self-check.c
// RUN: FileCheck %s --check-prefix=HARNESS --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch-self-check.c
// RUN: tcrv-opt %S/../EmissionManifest/emission-manifest-pipeline.mlir --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c | FileCheck %s --check-prefix=AUTO --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %S/../EmissionManifest/emission-manifest-pipeline.mlir --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c | FileCheck %s --check-prefix=AUTO-HARNESS --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object %s 2>&1 | FileCheck %s --check-prefix=OBJECT-NO-PLAN --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "malformed-runtime-element-count"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object 2>&1 | FileCheck %s --check-prefix=OBJECT-BAD-ABI --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."

module @rvv_scalar_dispatch_input {
  tcrv.exec.kernel @dispatch_vadd {
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
      source_kernel = "dispatch_vadd"
    } {
    ^bb0(%runtime_n: index):
      %vl = tcrv_rvv.setvl %runtime_n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        tcrv_rvv.i32_vadd_dataflow {lhs_role = "lhs-input-buffer", out_role = "output-buffer", rhs_role = "rhs-input-buffer", runtime_n_role = "runtime-element-count"}
      } : !tcrv_rvv.vl
    }
    tcrv_scalar.i32_vadd_microkernel {
      element_count = 16 : i64,
      origin = "scalar-plugin",
      required_capabilities = [@scalar_fallback],
      role = "dispatch fallback",
      selected_variant = @scalar_fallback_first_slice,
      source_kernel = "dispatch_vadd"
    }
  }
}

// HEADER: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// HEADER: /* Runtime guard: explicit host-provided rvv_available parameter; no automatic hardware probe is generated. */
// HEADER: /* selected_kernel: @dispatch_vadd */
// HEADER: /* rvv_selected_variant: @rvv_first_slice */
// HEADER: /* rvv_selected_role: dispatch case */
// HEADER: /* rvv_artifact_kind: runtime-callable-c-source */
// HEADER: /* rvv_artifact_route_id: tcrv-export-rvv-microkernel-c */
// HEADER: /* rvv_runtime_abi: rvv-i32-vadd-runtime-callable-c-abi.v1 */
// HEADER: /* rvv_runtime_abi_kind: rvv-runtime-callable-c-abi */
// HEADER: /* rvv_runtime_abi_name: rvv-i32-vadd-runtime-callable-c-function.v1 */
// HEADER: /* rvv_runtime_glue_role: runtime-callable-i32-vadd-function */
// HEADER: /* rvv_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// HEADER: /* rvv_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// HEADER: /* rvv_required_capabilities: @rvv */
// HEADER: /* scalar_selected_variant: @scalar_fallback_first_slice */
// HEADER: /* scalar_selected_role: dispatch fallback */
// HEADER: /* scalar_artifact_kind: runtime-callable-c-source */
// HEADER: /* scalar_artifact_route_id: tcrv-export-scalar-microkernel-c */
// HEADER: /* scalar_runtime_abi: scalar-i32-vadd-runtime-callable-c-abi.v1 */
// HEADER: /* scalar_runtime_abi_kind: scalar-runtime-callable-c-abi */
// HEADER: /* scalar_runtime_abi_name: scalar-i32-vadd-runtime-callable-c-function.v1 */
// HEADER: /* scalar_runtime_glue_role: runtime-callable-i32-vadd-fallback-function */
// HEADER: /* scalar_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// HEADER: /* scalar_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// HEADER: /* scalar_required_capabilities: @scalar_fallback */
// HEADER: /* rvv_callable_symbol: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice */
// HEADER: /* scalar_callable_symbol: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice */
// HEADER: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// HEADER: dispatch_runtime_callable_abi

// BODY: riscv_vector.h
// BODY: void tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice
// BODY: __riscv_vsetvl_e32m1
// BODY: __riscv_vle32_v_i32m1
// BODY: __riscv_vadd_vv_i32m1
// BODY: __riscv_vse32_v_i32m1
// BODY: void tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice
// BODY: for (size_t index = 0; index < n; ++index)
// BODY: out[index] = lhs[index] + rhs[index];
// BODY-LABEL: {{^}}void tcrv_dispatch_i32_vadd_dispatch_vadd
// BODY: if (rvv_available)
// BODY: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice(lhs, rhs, out, n);
// BODY: return;
// BODY: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice(lhs, rhs, out, n);

// AUTO: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// AUTO: /* selected_kernel: @pipeline_manifest */
// AUTO: /* rvv_selected_variant: @rvv_first_slice */
// AUTO: /* scalar_selected_variant: @scalar_fallback_first_slice */
// AUTO: /* scalar_artifact_route_id: tcrv-export-scalar-microkernel-c */
// AUTO: void tcrv_rvv_i32_vadd_microkernel_pipeline_manifest_rvv_first_slice
// AUTO: __riscv_vadd_vv_i32m1
// AUTO: void tcrv_scalar_i32_vadd_microkernel_pipeline_manifest_scalar_fallback_first_slice
// AUTO: out[index] = lhs[index] + rhs[index];
// AUTO-LABEL: {{^}}void tcrv_dispatch_i32_vadd_pipeline_manifest
// AUTO: if (rvv_available)
// AUTO: tcrv_rvv_i32_vadd_microkernel_pipeline_manifest_rvv_first_slice(lhs, rhs, out, n);
// AUTO: tcrv_scalar_i32_vadd_microkernel_pipeline_manifest_scalar_fallback_first_slice(lhs, rhs, out, n);

// HARNESS: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// HARNESS: /* selected_kernel: @dispatch_vadd */
// HARNESS: void tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice
// HARNESS: void tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice
// HARNESS-LABEL: {{^}}void tcrv_dispatch_i32_vadd_dispatch_vadd
// HARNESS: if (rvv_available)
// HARNESS: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice(lhs, rhs, out, n);
// HARNESS: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice(lhs, rhs, out, n);
// HARNESS: /* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
// HARNESS: /* Harness scope: calls the generated dispatcher once with rvv_available = 0 and once with rvv_available = 1. */
// HARNESS: Runtime n is a target/export-owned ABI parameter
// HARNESS: descriptor-local element_count remains metadata only
// HARNESS: #include <stdio.h>
// HARNESS-LABEL: {{^}}static int tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(int rvv_available)
// HARNESS: int32_t out[16] = {0};
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd(lhs, rhs, out, 16, rvv_available);
// HARNESS: if (out[index] != lhs[index] + rhs[index])
// HARNESS-LABEL: {{^}}int main(void)
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(0)
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(1)
// HARNESS: puts("tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok");

// AUTO-HARNESS: /* selected_kernel: @pipeline_manifest */
// AUTO-HARNESS: void tcrv_dispatch_i32_vadd_pipeline_manifest
// AUTO-HARNESS: static int tcrv_dispatch_i32_vadd_pipeline_manifest_self_check_one(int rvv_available)
// AUTO-HARNESS: tcrv_dispatch_i32_vadd_pipeline_manifest(lhs, rhs, out, 16, rvv_available);
// AUTO-HARNESS: tcrv_dispatch_i32_vadd_pipeline_manifest_self_check_one(0)
// AUTO-HARNESS: tcrv_dispatch_i32_vadd_pipeline_manifest_self_check_one(1)
// AUTO-HARNESS: tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok

// HELP: tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object

// OBJECT-NO-PLAN: RVV+scalar i32-vadd dispatch self-check object export failed
// OBJECT-NO-PLAN: selected path @rvv_first_slice as dispatch case requires exactly one emission-plan diagnostic before target artifact export

// OBJECT-BAD-ABI: unsupported runtime ABI parameter role 'malformed-runtime-element-count'
