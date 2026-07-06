// RUN: tcrv-translate --help | FileCheck %s --check-prefix=HELP
// RUN: tcrv-opt %s --tcrv-check-capability-requires --tcrv-materialize-plugin-variants --tcrv-verify-plugin-variant-legality --tcrv-select-variants | tcrv-translate --tcrv-scalar-emitc-to-cpp | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="popcount" --implicit-check-not="tcrv_rvv"
// RUN: tcrv-translate --tcrv-scalar-emitc-to-cpp %s | FileCheck %s --check-prefix=SOURCE --implicit-check-not="__riscv_" --implicit-check-not="popcount" --implicit-check-not="tcrv_rvv"
// RUN: tcrv-translate --tcrv-scalar-emitc-to-cpp %s | diff %S/tq2-0-q8-k-ternary-vec-dot.golden.c -

// X-SCALAR family #3: a REAL ternary 2-bit vec_dot scalar kernel (replaces the
// tracer-bullet trivial compute op). A hand-written portable-scalar
// tcrv_scalar.tq2_0_q8_k_vec_dot boundary flows through the generic
// capability/variant planning passes and then the scalar backend emission
// driver lowers it to a standalone EmitC module that the
// --tcrv-scalar-emitc-to-cpp route renders as PURE SCALAR C/C++: the ggml
// `ggml_vec_dot_tq2_0_q8_K` contraction as nested C loops. NO __riscv_
// intrinsics, NO XOR-popcount codebook, NO vector machinery -- the ternary
// weight is decoded `(((qs >> shift) & 3) - 1)` and multiply-accumulated
// against the int8 q8_K activation into a scalar int32 `sumi`, then folded
// `sumf += (float) sumi * (y.d * fp16(x.d))`.
//
// The emission is operand-driven, NOT vacuous: the exported function name is
// derived from source_kernel + selected_variant, and the block-format facts
// (qk=256 -> `/ 256`, weight_block_stride=66 -> `* 66`, activation_block_stride
// =292 -> `* 292`, activation_quant_byte_offset=4 -> `+ 4`, weight_d_byte_offset
// =64 -> `+ 64`) become the emitted loop bounds and address arithmetic --
// changing any attribute changes the emitted C.
//
// The last RUN is a strict byte-exact gate versus the captured golden C
// (the ggml scalar ternary reference), sibling tq2-0-q8-k-ternary-vec-dot.golden.c.
//
// NOTE: --tcrv-materialize-emission-plans is intentionally NOT in the pipe (the
// scalar plugin's emission-readiness still fail-closes as Unsupported, locked by
// test/Plugin/ScalarExtensionPluginTest.cpp); the translate route lowers the
// selected typed body directly through the shared backend-emission registry.

// HELP: --tcrv-scalar-emitc-to-cpp
// HELP-SAME: MLIR EmitC C/C++ emitter

module {
  tcrv.exec.kernel @tq2_0_kernel {
    tcrv.exec.capability @scalar_fallback {id = "scalar.fallback", kind = "fallback", status = "available"}
    tcrv_scalar.tq2_0_q8_k_vec_dot {
      source_kernel = "tq2_0_kernel",
      selected_variant = @scalar_fallback_first_slice,
      qk = 256 : i64,
      weight_block_stride = 66 : i64,
      activation_block_stride = 292 : i64,
      weight_d_byte_offset = 64 : i64,
      activation_d_byte_offset = 0 : i64,
      activation_quant_byte_offset = 4 : i64
    }
  }
}

// SOURCE: #include <stdint.h>
// SOURCE: extern "C" void tcrv_emitc_tq2_0_kernel_scalar_fallback_first_slice(int v{{[0-9]+}}, float* v{{[0-9]+}}, const uint8_t* v{{[0-9]+}}, const int8_t* v{{[0-9]+}})
// SOURCE: tcrv_emitc.route_source_op=tcrv_scalar.tq2_0_q8_k_vec_dot role=compute op_interface=TCRVEmitCLowerableOpInterface

// super-block count nb = (size_t)n / 256  (qk attribute).
// SOURCE: size_t v[[N:[0-9]+]] = (size_t) v{{[0-9]+}};
// SOURCE: size_t v{{[0-9]+}} = v[[N]] / 256;
// SOURCE: v{{[0-9]+}} = 0.0f;

// super-block loop; weight base vx + ib*66, q8 base vy + ib*292 + 4.
// SOURCE: for (size_t v[[IB:[0-9]+]] = 0; v[[IB]] < v{{[0-9]+}}; v[[IB]] += 1) {
// SOURCE: size_t v{{[0-9]+}} = v[[IB]] * 66;
// SOURCE: const uint8_t* v{{[0-9]+}} = v{{[0-9]+}} + v{{[0-9]+}};
// SOURCE: size_t v{{[0-9]+}} = v[[IB]] * 292;
// SOURCE: const int8_t* v{{[0-9]+}} = v{{[0-9]+}} + v{{[0-9]+}};
// SOURCE: const int8_t* v{{[0-9]+}} = v{{[0-9]+}} + 4;
// SOURCE: v{{[0-9]+}} = 0;

// the ggml TERNARY nested loops: 32-byte plane groups, 4 planes, 32 lanes.
// SOURCE: for (size_t v{{[0-9]+}} = 0; v{{[0-9]+}} < 64; v{{[0-9]+}} += 32) {
// SOURCE: for (size_t v{{[0-9]+}} = 0; v{{[0-9]+}} < 4; v{{[0-9]+}} += 1) {
// SOURCE: int v{{[0-9]+}} = (int) v{{[0-9]+}};
// SOURCE: for (size_t v{{[0-9]+}} = 0; v{{[0-9]+}} < 32; v{{[0-9]+}} += 1) {

// the ternary DECODE: (((int)qs[j+k] >> shift) & 3) - 1, pure int8xint8 MAC.
// SOURCE: const uint8_t v[[QB:[0-9]+]] = v{{[0-9]+}}[v{{[0-9]+}}];
// SOURCE: int v{{[0-9]+}} = (int) v[[QB]];
// SOURCE: int v{{[0-9]+}} = v{{[0-9]+}} >> v{{[0-9]+}};
// SOURCE: int v{{[0-9]+}} = v{{[0-9]+}} & 3;
// SOURCE: int v{{[0-9]+}} = v{{[0-9]+}} - 1;
// SOURCE: const int8_t v{{[0-9]+}} = v{{[0-9]+}}[v{{[0-9]+}}];
// SOURCE: int v[[W:[0-9]+]] = v{{[0-9]+}} * v{{[0-9]+}};
// SOURCE: int v{{[0-9]+}} = v{{[0-9]+}} + v[[W]];

// per-super-block scale fold: sumf += (float)sumi * (y.d * fp16(x.d)).
// SOURCE: const float* v{{[0-9]+}} = (const float*) v{{[0-9]+}};
// SOURCE: const float v{{[0-9]+}} = v{{[0-9]+}}[0];
// SOURCE: const uint8_t* v{{[0-9]+}} = v{{[0-9]+}} + 64;
// SOURCE: float v{{[0-9]+}} = (float)*(const _Float16 *)(v{{[0-9]+}});
// SOURCE: float v[[D:[0-9]+]] = v{{[0-9]+}} * v{{[0-9]+}};
// SOURCE: v{{[0-9]+}} = v{{[0-9]+}} + (float) v{{[0-9]+}} * v[[D]];

// *s = sumf.
// SOURCE: v{{[0-9]+}}[0] = v{{[0-9]+}};
