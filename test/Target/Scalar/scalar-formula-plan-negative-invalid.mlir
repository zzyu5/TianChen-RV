// RUN: not weft-translate --weft-scalar-emitc-to-cpp %s 2>&1 | FileCheck %s

// An exact canonical problem is the source input to Scalar construction, not a
// license to reinterpret invalid layout facts in the emitter.
module {
  weft.exec.kernel @bad_scalar_geometry attributes {construction_domain = "riscv-execution", problem = @invalid_problem} {
    weft.exec.dequantize_row_q4_0_problem @invalid_problem {
      qk = 64 : i64,
      weight_block_stride = 18 : i64,
      weight_d_byte_offset = 0 : i64,
      weight_quant_byte_offset = 2 : i64
    }
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
  }
}

// CHECK: requires canonical q4_0 geometry qk=32
