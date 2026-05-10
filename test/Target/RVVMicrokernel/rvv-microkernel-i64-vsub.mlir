// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck %s --check-prefix=PIPE --implicit-check-not=tcrv_rvv.i64_vadd_microkernel --implicit-check-not=tcrv_rvv.i64_vmul_microkernel --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token
// RUN: tcrv-opt %s --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=SOURCE --implicit-check-not=int32_t --implicit-check-not=__riscv_vadd_vv_i64m1 --implicit-check-not=__riscv_vmul_vv_i64m1 --implicit-check-not=i32_vadd --implicit-check-not=i32_vsub --implicit-check-not=i32_vmul --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=password --implicit-check-not=token

module @rvv_microkernel_i64_vsub_export_input {
  tcrv.exec.kernel @export_i64_vsub {
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
    tcrv.exec.variant @rvv_i64_slice attributes {
      condition = "rvv_capability_properties_available",
      guard = "plugin_local_rvv_property_evidence",
      origin = "rvv-plugin",
      policy = "metadata_only_first_slice",
      requires = [@rvv],
      tcrv_rvv.element_count = 8 : i64,
      tcrv_rvv.lowering_descriptor = "i64-vsub-microkernel.v1",
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
      message = "static RVV i64 vsub microkernel path selected by test fixture",
      origin = "rvv-plugin",
      reason = "variant-selected",
      selection_kind = "static-variant",
      severity = "note",
      status = "selected",
      target = @rvv_i64_slice
    }
  }
}

// PIPE-LABEL: tcrv.exec.kernel @export_i64_vsub
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
// PIPE: tcrv_rvv.i64_vsub_microkernel
// PIPE-SAME: element_count = 8 : i64
// PIPE-SAME: selected_vector_shape = "i64m1"
// PIPE-SAME: selected_vector_suffix = "i64m1"
// PIPE: tcrv_rvv.setvl
// PIPE-SAME: lmul = "m1"
// PIPE-SAME: sew = 64 : i64
// PIPE: tcrv_rvv.i64_load
// PIPE-SAME: !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_sub
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.i64m1, !tcrv_rvv.vl -> !tcrv_rvv.i64m1
// PIPE: tcrv_rvv.i64_store
// PIPE-SAME: !tcrv_rvv.i64m1, !tcrv_rvv.vl
// PIPE: tcrv.exec.diagnostic
// PIPE-SAME: artifact_kind = "runtime-callable-c-source"
// PIPE-SAME: emission_kind = "rvv-explicit-i64-vsub-microkernel-c-source"
// PIPE-SAME: lowering_pipeline = "tcrv-export-rvv-i64-vsub-microkernel-c"
// PIPE-SAME: runtime_abi = "rvv-i64-vsub-runtime-callable-c-abi.v1"
// PIPE-SAME: runtime_abi_name = "rvv-i64-vsub-runtime-callable-c-function.v1"
// PIPE-SAME: status = "supported"

// SOURCE: /* TianChen-RV RVV runtime-callable microkernel C export. */
// SOURCE: /* Scope: library-style C source for exactly one tcrv_rvv.i64_vsub_microkernel. */
// SOURCE: #include <stdint.h>
// SOURCE: #include <riscv_vector.h>
// SOURCE-LABEL: /* microkernel function: tcrv_rvv_i64_vsub_microkernel_export_i64_vsub_rvv_i64_slice */
// SOURCE: /* selected_kernel: @export_i64_vsub */
// SOURCE: /* selected_variant: @rvv_i64_slice */
// SOURCE: /* executable_microkernel: tcrv_rvv.i64_vsub_microkernel */
// SOURCE: /* arithmetic_family: i64-vsub */
// SOURCE: /* dtype: i64 */
// SOURCE: /* arithmetic_c_operator: - */
// SOURCE: /* active_route: tcrv-export-rvv-i64-vsub-microkernel-c */
// SOURCE: /* dataflow_body: tcrv_rvv.i64_load -> tcrv_rvv.i64_load -> tcrv_rvv.i64_sub -> tcrv_rvv.i64_store */
// SOURCE: /* dataflow_emission_step[2]: op=tcrv_rvv.i64_sub, lhs=lhs_vec, rhs=rhs_vec, result=difference_vec */
// SOURCE: /* selected_vector_shape_config: dtype=i64, shape=i64m1, sew=64, lmul=m1, tail_policy=agnostic, mask_policy=agnostic, vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1 */
// SOURCE: /* selected_vector_shape_capabilities: rvv.i64_m1.sew64 rvv.i64_m1.lmul_m1 rvv.i64_m1.tail_policy.agnostic rvv.i64_m1.mask_policy.agnostic */
// SOURCE: /* intrinsic_config: vector_type=vint64m1_t, vector_suffix=i64m1, setvl_suffix=e64m1, tail_policy=agnostic, mask_policy=agnostic */
// SOURCE: /* runtime_abi_parameter[0]: c_name=lhs, c_type=const int64_t *, role=lhs-input-buffer, ownership=target-export-abi-owned */
// SOURCE: /* runtime_abi_parameter[2]: c_name=out, c_type=int64_t *, role=output-buffer, ownership=target-export-abi-owned */
// SOURCE: void tcrv_rvv_i64_vsub_microkernel_export_i64_vsub_rvv_i64_slice(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n)
// SOURCE: while (offset < n)
// SOURCE: __riscv_vsetvl_e64m1
// SOURCE: __riscv_vle64_v_i64m1
// SOURCE: __riscv_vsub_vv_i64m1
// SOURCE: __riscv_vse64_v_i64m1
