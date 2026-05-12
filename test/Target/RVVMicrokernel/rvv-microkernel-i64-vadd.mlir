// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PIPE --implicit-check-not=tcrv_rvv.i32_vadd_microkernel --implicit-check-not=tcrv_rvv.i32_vsub_microkernel --implicit-check-not=tcrv_rvv.i32_vmul_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i32 --implicit-check-not=__riscv_vsub --implicit-check-not=__riscv_vmul --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans > %t.i64-vadd-post-planning.mlir
// RUN: sed '/tcrv.exec.mem_window @abi_lhs_input_buffer/d' %t.i64-vadd-post-planning.mlir | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=MISSING-I64-LHS --implicit-check-not="#include <riscv_vector.h>"
// RUN: sed '/tcrv.exec.runtime_param @abi_runtime_element_count/ s/c_type = "size_t"/c_type = "uint64_t"/' %t.i64-vadd-post-planning.mlir | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=STALE-I64-RUNTIME --implicit-check-not="#include <riscv_vector.h>"
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '/tcrv_rvv.i64_vadd_microkernel/,+20 s/element_count = 8 : i64/element_count = 16 : i64/' | not tcrv-translate --tcrv-export-target-source-artifact 2>&1 | FileCheck %s --check-prefix=BAD-ELEMENT-COUNT --implicit-check-not="#include <riscv_vector.h>" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-i64-vadd-microkernel-c | FileCheck %s --check-prefix=SOURCE --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i32 --implicit-check-not=__riscv_vsub --implicit-check-not=__riscv_vmul --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries | tcrv-translate --tcrv-export-rvv-i64-vadd-microkernel-header | FileCheck %s --check-prefix=HEADER --implicit-check-not=") {" --implicit-check-not="while (" --implicit-check-not="__riscv" --implicit-check-not=riscv_vector --implicit-check-not=int32_t --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP

module @rvv_microkernel_i64_vadd_export_input {
  tcrv.exec.kernel @export_i64_vadd attributes {
    tcrv_frontend_lowering = "i64-vadd"
  } {
    tcrv.exec.capability @rvv {
      id = "rvv",
      kind = "isa-vector",
      provides = ["rvv.i64_m1.sew64", "rvv.i64_m1.lmul_m1", "rvv.i64_m1.tail_policy.agnostic", "rvv.i64_m1.mask_policy.agnostic"],
      sew_bits = 64 : i64,
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
    tcrv.exec.mem_window @abi_lhs_input_buffer {
      abi_role = "lhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_rhs_input_buffer {
      abi_role = "rhs-input-buffer",
      access = "read",
      binding = "kernel-argument",
      c_type = "const int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.mem_window @abi_output_buffer {
      abi_role = "output-buffer",
      access = "write",
      binding = "kernel-argument",
      c_type = "int64_t *",
      memory_space = "host",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-buffer"
    }
    tcrv.exec.runtime_param @abi_runtime_element_count {
      abi_role = "runtime-element-count",
      c_name = "len64",
      c_type = "size_t",
      ownership = "target-export-abi-owned",
      purpose = "runtime-abi-scalar"
    }
    tcrv.exec.variant @rvv_i64_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.element_count = 8 : i64,
      tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>,
      tcrv_rvv.required_march = "rv64gcv",
      tcrv_rvv.selected_mask_policy = "agnostic",
      tcrv_rvv.selected_setvl_suffix = "e64m1",
      tcrv_rvv.selected_tail_policy = "agnostic",
      tcrv_rvv.selected_vector_lmul = "m1",
      tcrv_rvv.selected_vector_sew = 64 : i64,
      tcrv_rvv.selected_vector_shape = "i64m1",
      tcrv_rvv.selected_vector_suffix = "i64m1",
      tcrv_rvv.selected_vector_type = "vint64m1_t"
    } {
    }
    tcrv.exec.diagnostic {
      message = "static RVV i64 vadd microkernel path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_i64_slice
    }
  }
}

// PIPE-LABEL: tcrv.exec.kernel @export_i64_vadd
// PIPE: tcrv.exec.mem_window @abi_lhs_input_buffer
// PIPE-SAME: c_type = "const int64_t *"
// PIPE: tcrv.exec.mem_window @abi_rhs_input_buffer
// PIPE-SAME: c_type = "const int64_t *"
// PIPE: tcrv.exec.mem_window @abi_output_buffer
// PIPE-SAME: c_type = "int64_t *"
// PIPE: tcrv_rvv.lowering_boundary
// PIPE-SAME: selected_vector_lmul = "m1"
// PIPE-SAME: selected_vector_sew = 64 : i64
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_type = "vint64m1_t"
// PIPE: tcrv_rvv.i64_vadd_microkernel
// PIPE-SAME: element_count = 8 : i64
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_suffix = "i64m1"
// PIPE: tcrv_rvv.setvl
// PIPE-SAME: lmul = "m1"
// PIPE-SAME: sew = 64 : i64
// PIPE: tcrv_rvv.i64_load
// PIPE-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_add
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_store
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.vl
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "rvv-explicit-i64-vadd-microkernel-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-i64-vadd-microkernel-c"
// PIPE-SAME: runtime_abi = "rvv-i64-vadd-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_kind = "rvv-runtime-callable-c-abi"
// PIPE-SAME: runtime_abi_name = "rvv-i64-vadd-runtime-callable-c-function.v1"
// PIPE-SAME: runtime_abi_parameters = [{c_name = "lhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "lhs-input-buffer"}
// PIPE-SAME: {c_name = "rhs", c_type = "const int64_t *", ownership = "target-export-abi-owned", role = "rhs-input-buffer"}
// PIPE-SAME: {c_name = "out", c_type = "int64_t *", ownership = "target-export-abi-owned", role = "output-buffer"}
// PIPE-SAME: {c_name = "len64", c_type = "size_t", ownership = "target-export-abi-owned", role = "runtime-element-count"}]
// PIPE-SAME: runtime_glue_role = "runtime-callable-i64-vadd-function"
// PIPE-SAME: status = "supported"

// MISSING-I64-LHS: runtime ABI mem_window validation failed
// MISSING-I64-LHS-SAME: requires exactly one tcrv.exec.mem_window with ABI role 'lhs-input-buffer'
// STALE-I64-RUNTIME: runtime ABI runtime_param validation failed
// STALE-I64-RUNTIME-SAME: requires attribute 'c_type' = "size_t"

// BAD-ELEMENT-COUNT: TianChen-RV RVV microkernel body verifier failed
// BAD-ELEMENT-COUNT-SAME: family 'tcrv_rvv.i64_vadd_microkernel'
// BAD-ELEMENT-COUNT-SAME: descriptor-local element_count layer is stale
// BAD-ELEMENT-COUNT-SAME: body element_count=16
// BAD-ELEMENT-COUNT-SAME: tcrv_rvv.element_count=8

// SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// SOURCE: /* Scope: library-style C source for exactly one tcrv_rvv.i64_vadd_microkernel. */
// SOURCE: #include <stdint.h>
// SOURCE: #include <riscv_vector.h>
// SOURCE-LABEL: /* microkernel function: tcrv_rvv_i64_vadd_microkernel_export_i64_vadd_rvv_i64_slice */
// SOURCE: /* selected_kernel: @export_i64_vadd */
// SOURCE: /* selected_variant: @rvv_i64_slice */
// SOURCE: /* executable_microkernel: tcrv_rvv.i64_vadd_microkernel */
// SOURCE: /* arithmetic_family: i64-vadd */
// SOURCE: /* dtype: i64 */
// SOURCE: /* arithmetic_source: typed op tcrv_rvv.i64_add via generated EmitC route and IR-backed callable ABI */
// SOURCE: /* active_route: tcrv-export-rvv-i64-vadd-microkernel-c */
// SOURCE: /* dataflow_body: tcrv_rvv.i64_load -> tcrv_rvv.i64_load -> tcrv_rvv.i64_add -> tcrv_rvv.i64_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i64_add, lhs=lhs_vec, rhs=rhs_vec, result=sum_vec, interface=TCRVEmitCLowerableOpInterface, source_role=compute */
// SOURCE: /* emitc_lowerable_op_interface: TCRVEmitCLowerableOpInterface */
// SOURCE: /* selected_vector_shape_config: dtype=i64, shape=i64m1, sew=64, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i64_m1.sew64 rvv.i64_m1.lmul_m1 rvv.i64_m1.tail_policy.agnostic rvv.i64_m1.mask_policy.agnostic */
// SOURCE: /* intrinsic_config: vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int64_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[2]: c_name=out, c_type=int64_t *, role=output-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[3]: c_name=len64, c_type=size_t, role=runtime-element-count, ownership=target-export-abi-owned */
// SOURCE: void tcrv_rvv_i64_vadd_microkernel_export_i64_vadd_rvv_i64_slice(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t len64)
// SOURCE: while (offset < len64)
// SOURCE: __riscv_vsetvl_e64m1
// SOURCE: __riscv_vle64_v_i64m1
// SOURCE: __riscv_vadd_vv_i64m1
// SOURCE: __riscv_vse64_v_i64m1

// HEADER: #ifndef TIANCHENRV_RVV_I64_VADD_MICROKERNEL_EXPORT_I64_VADD_RVV_I64_SLICE_H
// HEADER: #include <stddef.h>
// HEADER: #include <stdint.h>
// HEADER: void tcrv_rvv_i64_vadd_microkernel_export_i64_vadd_rvv_i64_slice(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t len64);
// HEADER: #endif /* TIANCHENRV_RVV_I64_VADD_MICROKERNEL_EXPORT_I64_VADD_RVV_I64_SLICE_H */

// HELP-DAG: --tcrv-export-rvv-i64-vadd-microkernel-c
// HELP-DAG: --tcrv-export-rvv-i64-vadd-microkernel-header
// HELP-DAG: --tcrv-export-rvv-i64-vadd-microkernel-object
// HELP-DAG: --tcrv-export-rvv-i64-vsub-microkernel-c
// HELP-DAG: --tcrv-export-rvv-i64-vsub-microkernel-header
// HELP-DAG: --tcrv-export-rvv-i64-vsub-microkernel-object
// HELP-DAG: --tcrv-export-rvv-i64-vmul-microkernel-c
// HELP-DAG: --tcrv-export-rvv-i64-vmul-microkernel-header
// HELP-DAG: --tcrv-export-rvv-i64-vmul-microkernel-object
