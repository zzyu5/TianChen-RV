// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | FileCheck %s --check-prefix=PIPE
// RUN: tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact | FileCheck %s --check-prefix=EXPORT --implicit-check-not=riscv_vector --implicit-check-not=__riscv --implicit-check-not="int main(void)" --implicit-check-not="_self_check" --implicit-check-not=runtime_success --implicit-check-not=throughput --implicit-check-not=latency --implicit-check-not=artifacts/tmp --implicit-check-not=password --implicit-check-not=token

module {
  // PIPE-LABEL: tcrv.exec.kernel @pipeline_coherence_scalar
  tcrv.exec.kernel @pipeline_coherence_scalar {
    tcrv.exec.capability @scalar_fallback {
      id = "scalar.fallback",
      kind = "fallback",
      status = "available"
    }

    // PIPE: tcrv.exec.variant @scalar_fallback_first_slice
    // PIPE-SAME: origin = "scalar-plugin"
    // PIPE-SAME: requires = [@scalar_fallback]
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: reason = "variant-selected"
    // PIPE-SAME: selection_kind = "fallback-only"
    // PIPE-SAME: target = @scalar_fallback_first_slice
    // PIPE: tcrv_scalar.lowering_boundary
    // PIPE-SAME: selected_variant = @scalar_fallback_first_slice
    // PIPE-SAME: source_kernel = "pipeline_coherence_scalar"
    // PIPE: tcrv.exec.diagnostic
    // PIPE-SAME: artifact_kind = "runtime-callable-c-source"
    // PIPE-SAME: lowering_pipeline = "tcrv-export-scalar-microkernel-c"
    // PIPE-SAME: reason = "emission_plan"
    // PIPE-SAME: status = "supported"
    // PIPE-SAME: target = @scalar_fallback_first_slice

    // EXPORT: /* TianChen-RV scalar runtime-callable microkernel C export. */
    // EXPORT: /* selected_kernel: @pipeline_coherence_scalar */
    // EXPORT: /* selected_variant: @scalar_fallback_first_slice */
    // EXPORT: /* selected_role: direct variant */
    // EXPORT: /* artifact_kind: runtime-callable-c-source */
    // EXPORT: void tcrv_scalar_i32_vadd_microkernel_pipeline_coherence_scalar_scalar_fallback_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)
    // EXPORT: // tcrv_emitc.source_op=tcrv_scalar.i32_vadd_microkernel role=compute op_interface=TCRVEmitCLowerableOpInterface callee=tcrv_scalar_i32_add
    // EXPORT: tcrv_scalar_i32_add
  }
}
