// REQUIRES: weft-local-native-clangxx
// RUN: weft-translate --weft-scalar-emitc-to-cpp %s > %t.cpp
// RUN: clang++ -std=c++17 -O0 %t.cpp %S/dequantize-row-q4-0-scalar-runtime-harness.cpp -o %t.exe
// RUN: %t.exe

// Numeric reconstruction oracle: the public Scalar route must construct the
// exact packed_affine_dequant_body from this canonical exec problem before projecting
// it to C++. Two blocks exercise both nibbles, signed zero-point subtraction,
// distinct fp16 scales, block stride, and the paired output index map.
module {
  weft.exec.kernel @q4_0_dequant_kernel attributes {construction_domain = "riscv-execution", problem = @canonical_problem} {
    weft.exec.dequantize_row_q4_0_problem @canonical_problem {
      qk = 32 : i64,
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
