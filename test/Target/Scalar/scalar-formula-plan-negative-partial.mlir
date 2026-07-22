// RUN: not weft-translate --weft-scalar-emitc-to-cpp %s 2>&1 | FileCheck %s

// A forged/partial plan cannot become an emitter-side default.  The family
// constructor compares it with its typed g-derived complete result and fails
// the direct route closed.
module {
  weft.exec.kernel @partial_scalar_plan {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft_scalar.dequantize_row_q4_0 {
      source_kernel = "partial_scalar_plan",
      selected_variant = @scalar_fallback_first_slice,
      qk = 32 : i64,
      weight_block_stride = 18 : i64,
      weight_d_byte_offset = 0 : i64,
      weight_quant_byte_offset = 2 : i64,
      "weft.scalar.final_plan" = {formula_id = "weft.scalar.q4-0.dequantize-row.construct"}
    }
  }
}

// CHECK: no registered backend emission driver fully legalizes the selected portable-scalar body to EmitC
