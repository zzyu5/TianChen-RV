// RUN: not weft-translate --weft-scalar-emitc-to-cpp %s 2>&1 | FileCheck %s

// Even a field-complete input plan is not an authority.  Scalar construction
// independently rebuilds the final plan from typed g/c and rejects this forged
// half_width instead of letting the emitter consume it.
module {
  weft.exec.kernel @conflicting_scalar_plan {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft_scalar.dequantize_row_q4_0 {
      source_kernel = "conflicting_scalar_plan",
      selected_variant = @scalar_fallback_first_slice,
      qk = 32 : i64,
      weight_block_stride = 18 : i64,
      weight_d_byte_offset = 0 : i64,
      weight_quant_byte_offset = 2 : i64,
      "weft.scalar.final_plan" = {
        formula_id = "weft.scalar.q4-0.dequantize-row.construct",
        qk = 32 : i64,
        weight_block_stride = 18 : i64,
        weight_d_byte_offset = 0 : i64,
        weight_quant_byte_offset = 2 : i64,
        half_width = 17 : i64,
        field_bits = 4 : i64,
        field_mask = 15 : i64,
        decode_zero_point = 8 : i64
      }
    }
  }
}

// CHECK: scalar construction plan is partial, stale, or conflicts with typed geometry for weft.scalar.q4-0.dequantize-row.construct
