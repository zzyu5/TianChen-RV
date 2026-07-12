// G4 M1a: the FORMAT-KEYED q4_0 IME GEMM tile typed-region front door.
//
// A kernel carrying ONLY the spacemit.ime capability FACT + a whole-matrix SHAPE
// fact (ime_matmul_shape) + a WEIGHT-FORMAT fact (ime_weight_format = "q4_0") --
// no high-level op, no family-name branch -- drives the generic
// proposal/selection/boundary pipeline to CONSTRUCT the typed-region
// weft_ime.q4_0_matmul_tile op (the RVV lowerToRepackGemm front-door precedent
// applied to the IME matrix paradigm). The weight-format fact is pure data flow of
// the capability, keyed ON TOP of the whole-matrix GEMM prior.
//
// RUN 1 (STRUCTURE): stop at boundary materialization and assert the CONSTRUCTED
// typed region is the three DECOMPOSED bricks (q4_0_dequant_core + vmadot_mac_leaf
// + yield), NOT an opaque body -- the certified-shape checker.
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REGION
//
// RUN 2 (EMISSION): the full pipeline lowers the region to the q4_0-decode +
// vmadot MAC EmitC kernel (op-identity driven, I5).
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy"

module {
  weft.exec.kernel @ime_q4_0_matmul_kernel {
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_matmul_shape = "256x256x256",
      ime_weight_format = "q4_0"
    }
  }
}

// The prior routes GEMM ^ ime ^ q4_0 to the format-keyed q4_0 whole-matrix variant
// (still cost 0.5, GEMM takeover) and CONSTRUCTS the typed region.
// REGION: weft.exec.variant @ime_vmadot_matmul_slice
// REGION-SAME: ime.weight_format = "q4_0"
// REGION: weft_ime.q4_0_matmul_tile
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: mat_k = 256
// REGION-SAME: weight_format = "q4_0"
// The typed region is the three DECOMPOSED bricks (block_index + int8 activation
// fragment + int32 accumulator entry args), NOT an opaque helper.
// REGION: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>):
// REGION: weft_ime.q4_0_dequant_core %{{.*}} {decode_model = "q4_0_offset_binary_nibble"
// REGION-SAME: -> vector<32xi8>
// REGION: weft_ime.vmadot_mac_leaf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {accum_bits = 32
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: -> vector<16xi32>
// REGION: weft_ime.q4_0_matmul_tile_yield %{{.*}} : vector<16xi32>

// The emitted kernel: the register-resident BATCHED vmadot MAC leaf (single
// vsetvli, store once) + the q4_0 decode helper + the tiled q4_0 kernel
// (decode-to-scratch then one batched MAC per tile) + the structured extern "C"
// wrapper. int32-EXACT ([GAP-IME-LEAF-PIPELINE] closed in the emitter).
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: register_resident_accumulate=1
// EMITC-SAME: static inline void weft_ime_vmadot_mac_kloop
// EMITC-SAME: vmadot    v2, v0, v1
// EMITC: emitc.verbatim
// EMITC-SAME: decode_model=q4_0_offset_binary_nibble
// EMITC-SAME: static inline void weft_ime_q4_0_dequant_fragment
// EMITC-SAME: (int8_t)((int)(qs[j] & 0x0F) - 8)
// EMITC: emitc.verbatim
// EMITC-SAME: int32_exact=1
// EMITC-SAME: static void weft_ime_q4_0_vmadot_matmul
// EMITC-SAME: weft_ime_q4_0_dequant_fragment(Bcol + kf * q40_block_bytes, Bdec + kf * 32)
// EMITC-SAME: weft_ime_vmadot_mac_kloop(Arow, Bdec, kt, frag)
// EMITC: emitc.func @weft_emitc_ime_q4_0_matmul_kernel_ime_vmadot_matmul_slice
// EMITC: weft_emitc.route_source_op=weft_ime.q4_0_matmul_tile role=compute
// EMITC: call_opaque "weft_ime_q4_0_vmadot_matmul"
// G5-M3 forward bridge: the deferred per-block d_a*d_w SCALE-FOLD epilogue kernel
// (f32) is emitted AFTER the int32 kernel + wrapper (declared-before-use), then a
// SECOND extern "C" wrapper (<name>_f32) the ggml forward hook calls. The int32
// core stays the seal object; the fold introduces fp16 rounding and is
// board-validated against a canonical q4_0xq8_0 ZERO-MODEL reference
// (test/Target/IME/q4-0-matmul-tile-scalefold-{oracle,k1seal}.c).
// EMITC: emitc.verbatim
// EMITC-SAME: scale_fold_epilogue=weft_ime_q4_0_vmadot_matmul_f32
// EMITC-SAME: fold_model=per_block_da_dw
// EMITC-SAME: static void weft_ime_q4_0_vmadot_matmul_f32
// EMITC-SAME: Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[r * 4 + c]
// EMITC: emitc.func @weft_emitc_ime_q4_0_matmul_kernel_ime_vmadot_matmul_slice_f32
// EMITC: weft_emitc.route_source_op=weft_ime.q4_0_matmul_tile role=compute
// EMITC: call_opaque "weft_ime_q4_0_vmadot_matmul_f32"
