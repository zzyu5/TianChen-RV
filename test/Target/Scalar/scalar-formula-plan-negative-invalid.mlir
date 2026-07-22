// RUN: not weft-translate --weft-scalar-emitc-to-cpp %s 2>&1 | FileCheck %s

// A direct typed body is an input to the scalar construction formula, not a
// license to reinterpret arbitrary layout facts in the emitter.
module {
  weft.exec.kernel @bad_scalar_geometry {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft_scalar.dequantize_row_q4_0 {
      source_kernel = "bad_scalar_geometry",
      selected_variant = @scalar_fallback_first_slice,
      qk = 64 : i64,
      weight_block_stride = 18 : i64,
      weight_d_byte_offset = 0 : i64,
      weight_quant_byte_offset = 2 : i64
    }
  }
}

// CHECK: q4_0 dequant construction only admits the canonical qk/stride/offset geometry
