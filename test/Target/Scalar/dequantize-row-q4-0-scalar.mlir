// RUN: weft-translate --help | FileCheck %s --check-prefix=HELP
// RUN: weft-translate --weft-scalar-emitc-to-cpp %s | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="popcount" --implicit-check-not="weft_rvv"
// RUN: weft-translate --weft-scalar-emitc-to-cpp %s | diff %S/dequantize-row-q4-0-scalar.golden.c -
// RUN: sed 's/id = "scalar.fallback"/id = "scalar.other"/' %s | not weft-translate --weft-scalar-emitc-to-cpp 2>&1 | FileCheck %s --check-prefix=MISSING-CAPABILITY

// X-SCALAR family #4: a REAL 4-bit nibble dequantize scalar fallback kernel. A
// portable-scalar weft_scalar.dequantize_row_q4_0 source problem is bound to an
// explicit family variant; Scalar formula construction consumes it and creates
// a distinct packed_affine_dequant_body. The scalar backend emission driver
// lowers only that final body to a standalone EmitC module that the
// --weft-scalar-emitc-to-cpp route renders as PURE SCALAR C/C++: the ggml
// `dequantize_row_q4_0` block expansion as nested C loops. NO __riscv_
// intrinsics, NO XOR-popcount codebook, NO vector machinery -- each packed byte
// decodes its low nibble `(q & 0x0F) - 8` and high nibble `(q >> 4) - 8`, and
// scatters `x0*d` / `x1*d` into the two halves of the float output row.
//
// The construction is typed-input-driven, NOT vacuous: the family formula
// consumes source_kernel + selected_variant and the canonical q4_0 block facts
// (qk=32, stride 18, offsets 0/2), then constructs the typed mechanism body,
// including qk/2=16 and the nibble decode constants. The emitter cannot match
// the source op and only projects the final body's defined semantics.
//
// The last RUN is a strict byte-exact gate versus the captured golden C (the
// ggml scalar q4_0 reference), sibling dequantize-row-q4-0-scalar.golden.c.
//
// NOTE: --weft-materialize-emission-plans is intentionally NOT in the pipe (the
// scalar plugin's emission-readiness still fail-closes as Unsupported, locked by
// test/Plugin/ScalarExtensionPluginTest.cpp); the translate route lowers the
// selected typed body directly through the shared backend-emission registry.

// HELP: --weft-scalar-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter
// MISSING-CAPABILITY: materialized scalar fallback variant requires an available capability id 'scalar.fallback'

module {
  weft.exec.kernel @q4_0_dequant_kernel {
    weft.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    weft.exec.variant @scalar_fallback_first_slice attributes {
      origin = "scalar-plugin",
      requires = [@scalar_fallback]
    } {
    }
    weft_scalar.dequantize_row_q4_0 {
      source_kernel = "q4_0_dequant_kernel",
      selected_variant = @scalar_fallback_first_slice,
      qk = 32 : i64,
      weight_block_stride = 18 : i64,
      weight_d_byte_offset = 0 : i64,
      weight_quant_byte_offset = 2 : i64
    }
  }
}

// SOURCE: #include <stdint.h>
// SOURCE: extern "C" void weft_emitc_q4_0_dequant_kernel_scalar_fallback_first_slice(int v{{[0-9]+}}, float* v{{[0-9]+}}, const uint8_t* v{{[0-9]+}})
// SOURCE: weft_emitc.route_source_op=weft_scalar.dequantize_row_q4_0 role=compute op_interface=WEFTEmitCLowerableOpInterface

// block count nb = (size_t)n / 32  (qk attribute).
// SOURCE: size_t v[[N:[0-9]+]] = (size_t) v{{[0-9]+}};
// SOURCE: size_t v{{[0-9]+}} = v[[N]] / 32;

// block loop; weight base vx + ib*18, fp16 scale at +0, nibbles at +2.
// SOURCE: for (size_t v[[IB:[0-9]+]] = 0; v[[IB]] < v{{[0-9]+}}; v[[IB]] += 1) {
// SOURCE: size_t v{{[0-9]+}} = v[[IB]] * 18;
// SOURCE: const uint8_t* v{{[0-9]+}} = v{{[0-9]+}} + v{{[0-9]+}};
// SOURCE: float v{{[0-9]+}} = (float)*(const _Float16 *)(v{{[0-9]+}});
// SOURCE: const uint8_t* v{{[0-9]+}} = v{{[0-9]+}} + 2;
// SOURCE: size_t v{{[0-9]+}} = v[[IB]] * 32;

// the qk/2 nibble loop.
// SOURCE: for (size_t v{{[0-9]+}} = 0; v{{[0-9]+}} < 16; v{{[0-9]+}} += 1) {

// the NIBBLE decode: (q & 15) - 8 and (q >> 4) - 8, pure integer arithmetic.
// SOURCE: const uint8_t v[[QB:[0-9]+]] = v{{[0-9]+}}[v{{[0-9]+}}];
// SOURCE: int v[[QI:[0-9]+]] = (int) v[[QB]];
// SOURCE: int v{{[0-9]+}} = v[[QI]] & 15;
// SOURCE: int v{{[0-9]+}} = v{{[0-9]+}} - 8;
// SOURCE: int v{{[0-9]+}} = v[[QI]] >> 4;
// SOURCE: int v{{[0-9]+}} = v{{[0-9]+}} - 8;

// low nibble scatters to y[yb + j], high nibble to y[yb + j + 16].
// SOURCE: size_t v[[O0:[0-9]+]] = v{{[0-9]+}} + v{{[0-9]+}};
// SOURCE: float v{{[0-9]+}} = (float) v{{[0-9]+}};
// SOURCE: float v{{[0-9]+}} = v{{[0-9]+}} * v{{[0-9]+}};
// SOURCE: v{{[0-9]+}}[v[[O0]]] = v{{[0-9]+}};
// SOURCE: size_t v{{[0-9]+}} = v[[O0]] + 16;
// SOURCE: float v{{[0-9]+}} = (float) v{{[0-9]+}};
// SOURCE: float v{{[0-9]+}} = v{{[0-9]+}} * v{{[0-9]+}};
// SOURCE: v{{[0-9]+}}[v{{[0-9]+}}] = v{{[0-9]+}};
