// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c > %t.dispatch.c
// RUN: FileCheck %s --check-prefix=HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.c
// RUN: FileCheck %s --check-prefix=BODY --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.c
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header > %t.dispatch.h
// RUN: FileCheck %s --check-prefix=ABI-HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not="out[index]" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch.h
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact > %t.generic-dispatch.h
// RUN: FileCheck %s --check-prefix=GENERIC-ABI-HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not="out[index]" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.generic-dispatch.h
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c > %t.dispatch-self-check.c
// RUN: FileCheck %s --check-prefix=HARNESS --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token < %t.dispatch-self-check.c
// RUN: tcrv-opt %S/../EmissionManifest/emission-manifest-pipeline.mlir --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c | FileCheck %s --check-prefix=AUTO --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %S/../EmissionManifest/emission-manifest-pipeline.mlir --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c | FileCheck %s --check-prefix=AUTO-HARNESS --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c %s | FileCheck %s --check-prefix=NO-PLAN-SOURCE --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header %s | FileCheck %s --check-prefix=NO-PLAN-HEADER --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not="__riscv" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-c 2>&1 | FileCheck %s --check-prefix=ROUTE-MISMATCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-header 2>&1 | FileCheck %s --check-prefix=HEADER-ROUTE-MISMATCH --implicit-check-not="#ifndef TIANCHENRV_RVV_SCALAR_I32_VSUB_DISPATCH"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-object 2>&1 | FileCheck %s --check-prefix=OBJECT-ROUTE-MISMATCH --implicit-check-not="generated object file must have an ELF"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c 2>&1 | FileCheck %s --check-prefix=SELFCHECK-ROUTE-MISMATCH --implicit-check-not="tcrv_rvv_scalar_i32_vsub_dispatch_self_check_ok"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | not tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-object 2>&1 | FileCheck %s --check-prefix=SELFCHECK-OBJECT-ROUTE-MISMATCH --implicit-check-not="generated object file must have an ELF"
// RUN: sed '/tcrv.exec.capability @rvv_hart_count/,+5d' %s | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object 2>&1 | FileCheck %s --check-prefix=OBJECT-NO-PLAN --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object %s 2>&1 | FileCheck %s --check-prefix=SELFCHECK-OBJECT-NO-PLAN --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"/c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", role = "malformed-runtime-element-count"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object 2>&1 | FileCheck %s --check-prefix=OBJECT-BAD-ABI --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/architecture = "riscv64"/architecture = ""/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object 2>&1 | FileCheck %s --check-prefix=OBJECT-MISSING-ARCH --implicit-check-not="TianChen-RV RVV+scalar host runtime dispatch C export."
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '/^    tcrv.exec.mem_window @abi_lhs_input_buffer {/d' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=MISSING-MEM-WINDOW --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/abi_role = "lhs-input-buffer", access = "read"/s//abi_role = "lhs-input-buffer", access = "write"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=BAD-MEM-WINDOW --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: sed '/^    tcrv.exec.mem_window @abi_rhs_input_buffer/i\    tcrv.exec.mem_window @abi_lhs_input_buffer_dup {abi_role = "lhs-input-buffer", access = "read", binding = "kernel-argument", c_type = "const int32_t *", memory_space = "host", ownership = "target-export-abi-owned", purpose = "runtime-abi-buffer"}' %s | not tcrv-opt - 2>&1 | FileCheck %s --check-prefix=DUPLICATE-MEM-WINDOW --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '/^    tcrv.exec.runtime_param @abi_runtime_element_count {/d' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=MISSING-RUNTIME-N --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '/^    tcrv.exec.runtime_param @abi_dispatch_availability_guard {/d' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=MISSING-RUNTIME-GUARD --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/, runtime_guard = @abi_dispatch_availability_guard//' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=MISSING-CASE-RUNTIME-GUARD --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/, runtime_guard = @abi_dispatch_availability_guard//' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header 2>&1 | FileCheck %s --check-prefix=HEADER-MISSING-CASE-RUNTIME-GUARD --implicit-check-not="#ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/, runtime_guard = @abi_dispatch_availability_guard//' | not tcrv-translate --tcrv-export-target-header-artifact 2>&1 | FileCheck %s --check-prefix=GENERIC-HEADER-MISSING-CASE-RUNTIME-GUARD --implicit-check-not="#ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH"
// RUN: sed '/^    tcrv.exec.runtime_param @abi_dispatch_availability_guard/i\    tcrv.exec.runtime_param @abi_runtime_element_count_dup {abi_role = "runtime-element-count", c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}' %s | not tcrv-opt - 2>&1 | FileCheck %s --check-prefix=DUPLICATE-RUNTIME-N --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: sed '/^    tcrv.exec.variant @rvv_first_slice/i\    tcrv.exec.runtime_param @abi_dispatch_availability_guard_dup {abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned", purpose = "runtime-abi-scalar"}' %s | not tcrv-opt - 2>&1 | FileCheck %s --check-prefix=DUPLICATE-RUNTIME-GUARD --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int"/abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "bool"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=BAD-RUNTIME-GUARD-TYPE --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed 's/abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "target-export-abi-owned"/abi_role = "dispatch-availability-guard", c_name = "rvv_available", c_type = "int", ownership = "ir-modeled"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=BAD-RUNTIME-GUARD-OWNERSHIP --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lowering_pipeline = "tcrv-export-rvv-microkernel-c"/s//lowering_pipeline = "tcrv-export-rvv-i32-vsub-microkernel-c"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=STALE-RVV-COMPONENT-ROUTE --implicit-check-not="void tcrv_dispatch_i32_vadd"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lowering_pipeline = "tcrv-export-scalar-microkernel-c"/s//lowering_pipeline = "tcrv-export-scalar-i32-vsub-microkernel-c"/' | not tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c 2>&1 | FileCheck %s --check-prefix=STALE-SCALAR-COMPONENT-ROUTE --implicit-check-not="void tcrv_dispatch_i32_vadd"

module @rvv_scalar_dispatch_input {
  tcrv.exec.kernel @dispatch_vadd {
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
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int32_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "n",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
    tcrv.exec.runtime_param @abi_dispatch_availability_guard {
      abi_role = "dispatch-availability-guard",
      c_name = "rvv_available",
      c_type = "int",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
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
        runtime_guard = @abi_dispatch_availability_guard,
        runtime_guard_required = true,
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
        %lhs = tcrv_rvv.i32_load %vl {buffer_role = "lhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %rhs = tcrv_rvv.i32_load %vl {buffer_role = "rhs-input-buffer"} : !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        %sum = tcrv_rvv.i32_add %lhs, %rhs, %vl : !tcrv_rvv.i32m1, !tcrv_rvv.i32m1, !tcrv_rvv.vl -> !tcrv_rvv.i32m1

        tcrv_rvv.i32_store %sum, %vl {buffer_role = "output-buffer"} : !tcrv_rvv.i32m1, !tcrv_rvv.vl
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
// HEADER: /* dispatch_mem_window[0]: symbol=@abi_lhs_input_buffer, abi_role=lhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// HEADER: /* dispatch_mem_window[1]: symbol=@abi_rhs_input_buffer, abi_role=rhs-input-buffer, access=read, ownership=target-export-abi-owned, c_type=const int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// HEADER: /* dispatch_mem_window[2]: symbol=@abi_output_buffer, abi_role=output-buffer, access=write, ownership=target-export-abi-owned, c_type=int32_t *, purpose=runtime-abi-buffer, binding=kernel-argument, memory_space=host */
// HEADER: /* dispatch_runtime_param[0]: symbol=@abi_runtime_element_count, abi_role=runtime-element-count, c_name=n, c_type=size_t, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// HEADER: /* dispatch_runtime_param[1]: symbol=@abi_dispatch_availability_guard, abi_role=dispatch-availability-guard, c_name=rvv_available, c_type=int, ownership=target-export-abi-owned, purpose=runtime-abi-scalar */
// HEADER: /* dispatch_runtime_guard_link: case=@rvv_first_slice, runtime_guard=@abi_dispatch_availability_guard */
// HEADER: /* dispatch_fallback_link: target=@scalar_fallback_first_slice, selected_scalar_callable=@scalar_fallback_first_slice */
// HEADER: /* rvv_callable_symbol: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice */
// HEADER: /* scalar_callable_symbol: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice */
// HEADER: /* dispatch_runtime_abi_parameter[0]: c_name=lhs, c_type=const int32_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// HEADER: /* dispatch_runtime_abi_parameter[3]: c_name=n, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// HEADER: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// HEADER: dispatch_runtime_callable_abi
// HEADER: /* dispatch_runtime_abi_invocation_contract: source=RVVScalarDispatch.cpp, callable_symbol=tcrv_dispatch_i32_vadd_dispatch_vadd
// HEADER-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard
// HEADER-SAME: runtime_element_count_c_name=n
// HEADER-SAME: dispatch_guard_c_name=rvv_available
// HEADER-SAME: production_owner=rvv-scalar-dispatch-target

// ABI-HEADER: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// ABI-HEADER: /* dispatch_runtime_abi_invocation_contract: source=RVVScalarDispatch.cpp, callable_symbol=tcrv_dispatch_i32_vadd_dispatch_vadd
// ABI-HEADER-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard
// ABI-HEADER-SAME: production_owner=rvv-scalar-dispatch-target
// ABI-HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H
// ABI-HEADER-NEXT: #define TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H
// ABI-HEADER: #include <stddef.h>
// ABI-HEADER-NEXT: #include <stdint.h>
// ABI-HEADER: #ifdef __cplusplus
// ABI-HEADER-NEXT: extern "C" {
// ABI-HEADER-NEXT: #endif
// ABI-HEADER: void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// ABI-HEADER: #ifdef __cplusplus
// ABI-HEADER-NEXT: }
// ABI-HEADER-NEXT: #endif
// ABI-HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H */

// GENERIC-ABI-HEADER: /* dispatch_runtime_abi_parameter[4]: c_name=rvv_available, c_type=int, role=dispatch-availability-guard, ownership=target-export-abi-owned */
// GENERIC-ABI-HEADER: /* dispatch_runtime_abi_invocation_contract: source=RVVScalarDispatch.cpp, callable_symbol=tcrv_dispatch_i32_vadd_dispatch_vadd
// GENERIC-ABI-HEADER-SAME: ordered_roles=lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard
// GENERIC-ABI-HEADER-SAME: production_owner=rvv-scalar-dispatch-target
// GENERIC-ABI-HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H
// GENERIC-ABI-HEADER-NEXT: #define TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H
// GENERIC-ABI-HEADER: #include <stddef.h>
// GENERIC-ABI-HEADER-NEXT: #include <stdint.h>
// GENERIC-ABI-HEADER: #ifdef __cplusplus
// GENERIC-ABI-HEADER-NEXT: extern "C" {
// GENERIC-ABI-HEADER-NEXT: #endif
// GENERIC-ABI-HEADER: void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// GENERIC-ABI-HEADER: #ifdef __cplusplus
// GENERIC-ABI-HEADER-NEXT: }
// GENERIC-ABI-HEADER-NEXT: #endif
// GENERIC-ABI-HEADER: #endif /* TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H */

// BODY: riscv_vector.h
// BODY: void tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice
// BODY: __riscv_vsetvl_e32m1
// BODY: __riscv_vle32_v_i32m1
// BODY: __riscv_vadd_vv_i32m1
// BODY: __riscv_vse32_v_i32m1
// BODY: void tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice
// BODY: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// BODY: // tcrv_emitc.source_op=tcrv_scalar.i32_vadd_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_add
// BODY: tcrv_scalar_i32_add
// BODY: /* dispatch_emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// BODY: /* dispatch_emitc_c_source_authority: MLIR EmitC module translated by mlir::emitc::translateToCpp */
// BODY: /* dispatch_emitc_runtime_guard_value: rvv_available */
// BODY: /* dispatch_emitc.call_opaque[0]: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice from tcrv.exec.case, source_role=dispatch-case-call, operands=4 */
// BODY: /* dispatch_emitc.call_opaque[1]: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice from tcrv.exec.fallback, source_role=dispatch-fallback-call, operands=4 */
// BODY: // tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch
// BODY: // tcrv_emitc.dispatch_guard_value=rvv_available
// BODY-LABEL: {{^}}void tcrv_dispatch_i32_vadd_dispatch_vadd
// BODY: bool [[BODY_GUARD:v[0-9]+]] = {{v[0-9]+}} != 0;
// BODY: if ([[BODY_GUARD]])
// BODY: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});
// BODY: return;
// BODY: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});

// AUTO: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// AUTO: /* selected_kernel: @pipeline_manifest */
// AUTO: /* rvv_selected_variant: @rvv_first_slice */
// AUTO: /* scalar_selected_variant: @scalar_fallback_first_slice */
// AUTO: /* scalar_artifact_route_id: tcrv-export-scalar-microkernel-c */
// AUTO: /* dispatch_runtime_guard_link: case=@rvv_first_slice, runtime_guard=@abi_dispatch_availability_guard */
// AUTO: /* dispatch_fallback_link: target=@scalar_fallback_first_slice, selected_scalar_callable=@scalar_fallback_first_slice */
// AUTO: // tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
// AUTO: static void tcrv_rvv_i32_vadd_microkernel_pipeline_manifest_rvv_first_slice__tcrv_emitc_body
// AUTO: __riscv_vadd_vv_i32m1
// AUTO: void tcrv_rvv_i32_vadd_microkernel_pipeline_manifest_rvv_first_slice
// AUTO: void tcrv_scalar_i32_vadd_microkernel_pipeline_manifest_scalar_fallback_first_slice
// AUTO: // tcrv_emitc.source_op=tcrv_scalar.i32_vadd_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_add
// AUTO: tcrv_scalar_i32_add
// AUTO: /* dispatch_emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// AUTO: // tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch
// AUTO: // tcrv_emitc.dispatch_guard_value=rvv_available
// AUTO-LABEL: {{^}}void tcrv_dispatch_i32_vadd_pipeline_manifest
// AUTO: bool [[AUTO_GUARD:v[0-9]+]] = {{v[0-9]+}} != 0;
// AUTO: if ([[AUTO_GUARD]])
// AUTO: tcrv_rvv_i32_vadd_microkernel_pipeline_manifest_rvv_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});
// AUTO: tcrv_scalar_i32_vadd_microkernel_pipeline_manifest_scalar_fallback_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});

// HARNESS: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// HARNESS: /* selected_kernel: @dispatch_vadd */
// HARNESS: static void tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice__tcrv_emitc_body
// HARNESS: __riscv_vadd_vv_i32m1
// HARNESS: void tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice
// HARNESS: void tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice
// HARNESS: /* dispatch_emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// HARNESS: // tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch
// HARNESS: // tcrv_emitc.dispatch_guard_value=rvv_available
// HARNESS-LABEL: {{^}}void tcrv_dispatch_i32_vadd_dispatch_vadd
// HARNESS: bool [[HARNESS_GUARD:v[0-9]+]] = {{v[0-9]+}} != 0;
// HARNESS: if ([[HARNESS_GUARD]])
// HARNESS: tcrv_rvv_i32_vadd_microkernel_dispatch_vadd_rvv_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});
// HARNESS: tcrv_scalar_i32_vadd_microkernel_dispatch_vadd_scalar_fallback_first_slice({{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}});
// HARNESS: /* Explicit bounded self-check harness for RVV+scalar dispatch runtime invocation evidence. */
// HARNESS: /* Harness scope: calls the generated dispatcher with explicit n values 7 and 16 for rvv_available = 0 and rvv_available = 1. */
// HARNESS: Runtime element count is a target/export-owned ABI parameter
// HARNESS: artifact-local component capacity remains metadata only
// HARNESS: self_check_expectation_source: validated RVV dispatch-case component + validated scalar fallback component + IR-backed dispatch ABI; legacy descriptor mirrors cannot select expected arithmetic or scalar element type.
// HARNESS: int puts(const char *);
// HARNESS-LABEL: {{^}}static int tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(size_t runtime_n, int rvv_available)
// HARNESS: enum { kCapacity = 32 };
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd(lhs, rhs, out, runtime_n, rvv_available);
// HARNESS: for (size_t index = 0; index < runtime_n; ++index)
// HARNESS: if (out[index] != lhs[index] + rhs[index])
// HARNESS: for (size_t index = runtime_n; index < (size_t)kCapacity; ++index)
// HARNESS-LABEL: {{^}}int main(void)
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(7, 0)
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(16, 0)
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(7, 1)
// HARNESS: tcrv_dispatch_i32_vadd_dispatch_vadd_self_check_one(16, 1)
// HARNESS: puts("tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv");

// AUTO-HARNESS: /* selected_kernel: @pipeline_manifest */
// AUTO-HARNESS: /* dispatch_emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority */
// AUTO-HARNESS: // tcrv_emitc.dispatch_guard_value=rvv_available
// AUTO-HARNESS: void tcrv_dispatch_i32_vadd_pipeline_manifest
// AUTO-HARNESS: static int tcrv_dispatch_i32_vadd_pipeline_manifest_self_check_one(size_t runtime_n, int rvv_available)
// AUTO-HARNESS: tcrv_dispatch_i32_vadd_pipeline_manifest(lhs, rhs, out, runtime_n, rvv_available);
// AUTO-HARNESS: tcrv_dispatch_i32_vadd_pipeline_manifest_self_check_one(7, 0)
// AUTO-HARNESS: tcrv_dispatch_i32_vadd_pipeline_manifest_self_check_one(16, 1)
// AUTO-HARNESS: tcrv_rvv_scalar_i32_vadd_dispatch_self_check_ok runtime_counts=7,16 branches=scalar_and_rvv

// NO-PLAN-SOURCE: /* TianChen-RV RVV+scalar host runtime dispatch C export. */
// NO-PLAN-SOURCE: /* dispatch_manifest_route_id: tcrv-export-rvv-scalar-i32-vadd-dispatch-c */
// NO-PLAN-SOURCE: void tcrv_dispatch_i32_vadd_dispatch_vadd
// NO-PLAN-SOURCE: __riscv_vadd_vv_i32m1

// NO-PLAN-HEADER: #ifndef TIANCHENRV_RVV_SCALAR_I32_VADD_DISPATCH_DISPATCH_VADD_H
// NO-PLAN-HEADER: void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);
// NO-PLAN-HEADER: #endif

// HELP-DAG: tcrv-export-rvv-scalar-i32-vadd-dispatch-c
// HELP-DAG: tcrv-export-rvv-scalar-i32-vadd-dispatch-header
// HELP-DAG: tcrv-export-rvv-scalar-i32-vadd-dispatch-object
// HELP-DAG: tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c
// HELP-DAG: tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object
// HELP-DAG: tcrv-export-rvv-scalar-i32-vsub-dispatch-c
// HELP-DAG: tcrv-export-rvv-scalar-i32-vsub-dispatch-header
// HELP-DAG: tcrv-export-rvv-scalar-i32-vsub-dispatch-object
// HELP-DAG: tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c
// HELP-DAG: tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-object
// HELP-DAG: tcrv-export-rvv-scalar-i32-vmul-dispatch-c
// HELP-DAG: tcrv-export-rvv-scalar-i32-vmul-dispatch-header
// HELP-DAG: tcrv-export-rvv-scalar-i32-vmul-dispatch-object
// HELP-DAG: tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-c
// HELP-DAG: tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-object
// HELP: tcrv-export-target-header-artifact

// ROUTE-MISMATCH: TianChen-RV target source artifact export failed
// ROUTE-MISMATCH-SAME: exact composite target artifact route 'tcrv-export-rvv-scalar-i32-vsub-dispatch-c'
// ROUTE-MISMATCH-SAME: requires exactly one selected emission-plan candidate group; found none
// HEADER-ROUTE-MISMATCH: TianChen-RV target source artifact export failed
// HEADER-ROUTE-MISMATCH-SAME: exact composite target artifact route 'tcrv-export-rvv-scalar-i32-vsub-dispatch-header'
// HEADER-ROUTE-MISMATCH-SAME: requires exactly one selected emission-plan candidate group; found none
// OBJECT-ROUTE-MISMATCH: TianChen-RV target source artifact export failed
// OBJECT-ROUTE-MISMATCH-SAME: exact composite target artifact route 'tcrv-export-rvv-scalar-i32-vsub-dispatch-object'
// OBJECT-ROUTE-MISMATCH-SAME: requires exactly one selected emission-plan candidate group; found none
// SELFCHECK-ROUTE-MISMATCH: self-check export route expected i32-vsub dispatch artifacts, got i32-vadd
// SELFCHECK-OBJECT-ROUTE-MISMATCH: self-check object export route expected i32-vsub dispatch artifacts, got i32-vadd

// OBJECT-NO-PLAN: TianChen-RV selected lowering-boundary materialization failed
// OBJECT-NO-PLAN-SAME: RVV property decision requires capability id 'rvv.hart_count'
// OBJECT-NO-PLAN: artifact-backed direct translate route 'tcrv-export-rvv-scalar-i32-vadd-dispatch-object' failed during execution planning before exact target artifact export
// SELFCHECK-OBJECT-NO-PLAN: RVV+scalar binary dispatch object export failed
// SELFCHECK-OBJECT-NO-PLAN-SAME: selected path @rvv_first_slice as dispatch case requires exactly one emission-plan diagnostic before target artifact export

// OBJECT-BAD-ABI: unsupported runtime ABI parameter role 'malformed-runtime-element-count'

// OBJECT-MISSING-ARCH: TianChen-RV selected lowering-boundary materialization failed
// OBJECT-MISSING-ARCH-SAME: capability id 'rvv' requires preserved property 'architecture'

// MISSING-MEM-WINDOW: RVV+scalar binary dispatch C export failed
// MISSING-MEM-WINDOW-SAME: runtime ABI mem_window validation failed
// MISSING-MEM-WINDOW-SAME: requires exactly one tcrv.exec.mem_window with ABI role 'lhs-input-buffer'

// BAD-MEM-WINDOW: RVV+scalar binary dispatch C export failed
// BAD-MEM-WINDOW-SAME: runtime ABI mem_window validation failed
// BAD-MEM-WINDOW-SAME: tcrv.exec.mem_window @abi_lhs_input_buffer requires attribute 'access' = "read"

// DUPLICATE-MEM-WINDOW: duplicates mem_window ABI role 'lhs-input-buffer' in enclosing tcrv.exec.kernel

// MISSING-RUNTIME-N: RVV+scalar binary dispatch C export failed
// MISSING-RUNTIME-N-SAME: runtime ABI runtime_param validation failed
// MISSING-RUNTIME-N-SAME: requires exactly one tcrv.exec.runtime_param with ABI role 'runtime-element-count'

// MISSING-RUNTIME-GUARD: runtime_guard references unknown runtime_param @abi_dispatch_availability_guard

// MISSING-CASE-RUNTIME-GUARD: execution plan coherence check failed
// MISSING-CASE-RUNTIME-GUARD-SAME: dispatch case @rvv_first_slice carries typed runtime_guard_required = true
// MISSING-CASE-RUNTIME-GUARD-SAME: missing runtime_guard linkage

// HEADER-MISSING-CASE-RUNTIME-GUARD: execution plan coherence check failed
// HEADER-MISSING-CASE-RUNTIME-GUARD-SAME: dispatch case @rvv_first_slice carries typed runtime_guard_required = true
// HEADER-MISSING-CASE-RUNTIME-GUARD-SAME: missing runtime_guard linkage

// GENERIC-HEADER-MISSING-CASE-RUNTIME-GUARD: execution plan coherence check failed
// GENERIC-HEADER-MISSING-CASE-RUNTIME-GUARD-SAME: dispatch case @rvv_first_slice carries typed runtime_guard_required = true
// GENERIC-HEADER-MISSING-CASE-RUNTIME-GUARD-SAME: missing runtime_guard linkage

// DUPLICATE-RUNTIME-N: duplicates runtime_param ABI role 'runtime-element-count' in enclosing tcrv.exec.kernel

// DUPLICATE-RUNTIME-GUARD: duplicates runtime_param ABI role 'dispatch-availability-guard' in enclosing tcrv.exec.kernel

// BAD-RUNTIME-GUARD-TYPE: RVV+scalar binary dispatch C export failed
// BAD-RUNTIME-GUARD-TYPE-SAME: runtime ABI runtime_param validation failed
// BAD-RUNTIME-GUARD-TYPE-SAME: tcrv.exec.runtime_param @abi_dispatch_availability_guard requires attribute 'c_type' = "int"

// BAD-RUNTIME-GUARD-OWNERSHIP: RVV+scalar binary dispatch C export failed
// BAD-RUNTIME-GUARD-OWNERSHIP-SAME: runtime ABI runtime_param validation failed
// BAD-RUNTIME-GUARD-OWNERSHIP-SAME: tcrv.exec.runtime_param @abi_dispatch_availability_guard requires attribute 'ownership' = "target-export-abi-owned"

// STALE-RVV-COMPONENT-ROUTE: RVV+scalar binary dispatch C export failed
// STALE-RVV-COMPONENT-ROUTE-SAME: selected RVV dispatch case callable route 'tcrv-export-rvv-i32-vsub-microkernel-c' for i32-vadd has stale route id
// STALE-RVV-COMPONENT-ROUTE-SAME: expected 'tcrv-export-rvv-microkernel-c'

// STALE-SCALAR-COMPONENT-ROUTE: RVV+scalar binary dispatch C export failed
// STALE-SCALAR-COMPONENT-ROUTE-SAME: selected scalar dispatch fallback callable route 'tcrv-export-scalar-i32-vsub-microkernel-c' for i32-vadd has stale route id
// STALE-SCALAR-COMPONENT-ROUTE-SAME: expected 'tcrv-export-scalar-microkernel-c'
