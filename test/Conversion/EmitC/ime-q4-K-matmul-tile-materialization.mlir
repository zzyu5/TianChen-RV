// G4 M2b: the FORMAT-KEYED q4_K IME GEMM tile typed-region front door (the
// SUPER-BLOCK K-quant tile; the DEDICATED effort with the two-level 6-bit
// scale/min fold).
//
// A kernel carrying ONLY the spacemit.ime capability FACT + a whole-matrix SHAPE
// fact (ime_matmul_shape) + a WEIGHT-FORMAT fact (ime_weight_format = "q4_K") --
// no high-level op, no family-name branch -- drives the generic
// proposal/selection/boundary pipeline to CONSTRUCT the typed-region
// tcrv_ime.q4_K_matmul_tile op. The weight-format fact is pure data flow of the
// capability, keyed ON TOP of the whole-matrix GEMM prior.
//
// RUN 1 (STRUCTURE): stop at boundary materialization and assert the CONSTRUCTED
// typed region is the SIX DECOMPOSED bricks (q4_K_dequant_core +
// q4_K_scale_min_unpack_core + vmadot_mac_leaf + q4_K_scale_weighted_accum +
// q4_K_min_bias_accum + the two-tile yield), NOT an opaque body.
// RUN: tcrv-opt %s --tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REGION
//
// RUN 2 (EMISSION): the full pipeline lowers the region to the q4_K raw-nibble
// decode + 6-bit scale/min unpack + two-level-fold vmadot MAC EmitC kernel
// (op-identity driven, I5).
// RUN: tcrv-opt %s --tcrv-materialize-plugin-variants --tcrv-select-variants --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="tcrv_rvv" --implicit-check-not="tcrv_toy"

module {
  tcrv.exec.kernel @ime_q4_K_matmul_kernel {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_matmul_shape = "256x256x256",
      ime_weight_format = "q4_K"
    }
  }
}

// The prior routes GEMM ^ ime ^ q4_K to the format-keyed q4_K whole-matrix variant
// (still cost 0.5, GEMM takeover) and CONSTRUCTS the typed region.
// REGION: tcrv.exec.variant @ime_vmadot_matmul_slice
// REGION-SAME: ime.weight_format = "q4_K"
// REGION: tcrv_ime.q4_K_matmul_tile
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: mat_k = 256
// REGION-SAME: weight_format = "q4_K"
// The typed region is the SIX DECOMPOSED bricks (block_index + int8 activation
// fragment + TWO int32 accumulators entry args), NOT an opaque helper.
// REGION: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>, %{{.*}}: vector<16xi32>):
// REGION: tcrv_ime.q4_K_dequant_core %{{.*}} {decode_model = "q4_K_raw_nibble"
// REGION-SAME: -> vector<32xi8>
// REGION: tcrv_ime.q4_K_scale_min_unpack_core %{{.*}} {{{.*}}scale_min_model = "get_scale_min_k4"
// REGION: tcrv_ime.vmadot_mac_leaf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {accum_bits = 32
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: -> vector<16xi32>
// REGION: tcrv_ime.q4_K_scale_weighted_accum %{{.*}}, %{{.*}}, %{{.*}} {accum_model = "scale_weighted_sum"}
// REGION: tcrv_ime.q4_K_min_bias_accum %{{.*}}, %{{.*}}, %{{.*}} {bias_model = "activation_sum_min_bias"}
// REGION: tcrv_ime.q4_K_matmul_tile_yield %{{.*}}, %{{.*}} : vector<16xi32>, vector<16xi32>

// The emitted kernel: the validated vmadot MAC leaf + the fp16 epilogue helpers +
// the q4_K raw-nibble decode + the 6-bit get_scale_min_k4 unpack + the tiled
// two-level-fold q4_K kernel (S_scale AND S_min) + the structured extern "C"
// wrapper. The int32 core is int32-EXACT; the d/dmin fold is the deferred float
// epilogue.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// EMITC-SAME: static inline void tcrv_ime_vmadot_mma_4x4x8
// EMITC-SAME: vmadot    v2, v0, v1
// EMITC: emitc.verbatim
// EMITC-SAME: tcrv_ime.fp16_epilogue=tcrv_ime_fp16_to_f32
// EMITC-SAME: static inline float tcrv_ime_fp16_to_f32
// EMITC: emitc.verbatim
// EMITC-SAME: decode_model=q4_K_raw_nibble
// EMITC-SAME: static inline void tcrv_ime_q4_K_dequant_fragment
// EMITC: emitc.verbatim
// EMITC-SAME: scale_min_model=get_scale_min_k4
// EMITC-SAME: static inline void tcrv_ime_q4_K_get_scale_min
// EMITC: emitc.verbatim
// EMITC-SAME: two_level_fold=kquant_dmin_bsums_min
// EMITC-SAME: static void tcrv_ime_q4_K_vmadot_matmul
// EMITC-SAME: tcrv_ime_q4_K_dequant_fragment(blk[nl], b, kf, Bframe + nl * 8)
// EMITC-SAME: tcrv_ime_vmadot_mma_4x4x8(Aframe, Bframe, frag)
// EMITC-SAME: Sc[ml * 4 + nl] += (int32_t)sc[b][nl] * sumi[ml * 4 + nl]
// EMITC-SAME: Sm[ml * 4 + nl] += (int32_t)mm[b][nl] * asum[ml]
// EMITC: emitc.func @tcrv_emitc_ime_q4_K_matmul_kernel_ime_vmadot_matmul_slice
// EMITC: tcrv_emitc.route_source_op=tcrv_ime.q4_K_matmul_tile role=compute
// EMITC: call_opaque "tcrv_ime_q4_K_vmadot_matmul"
