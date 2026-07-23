// G4 M2b: the FORMAT-KEYED q4_K IME GEMM tile typed-region front door (the
// SUPER-BLOCK K-quant tile; the DEDICATED effort with the two-level 6-bit
// scale/min fold).
//
// A canonical q4_K contraction problem plus the spacemit.ime target capability
// -- with no family-name branch -- drive the generic
// proposal/selection/boundary pipeline to CONSTRUCT the typed-region
// weft_ime.q4_K_matmul_tile op. Geometry and super-block layout come from exact P.
//
// RUN 1 (STRUCTURE): stop at boundary materialization and assert the CONSTRUCTED
// typed region is the SIX DECOMPOSED bricks (q4_K_dequant_core +
// q4_K_scale_min_unpack_core + vmadot_mac_leaf + q4_K_scale_weighted_accum +
// q4_K_min_bias_accum + the two-tile yield), NOT an opaque body.
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REGION
//
// RUN 2 (EMISSION): the full pipeline lowers the region to the q4_K raw-nibble
// decode + 6-bit scale/min unpack + two-level-fold vmadot MAC EmitC kernel
// (op-identity driven, I5).
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy" --implicit-check-not="mac_kloop_w2"

module {
  weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
  }
  weft.exec.target @ime_q4_K_profile {id = "ime.q4_K.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@spacemit_ime]}
  weft.exec.kernel @ime_q4_K_matmul_kernel attributes {target = @ime_q4_K_profile, problem = @canonical_problem} {
    weft.exec.block_q4_K_contraction_problem @canonical_problem {activation_signedness = #weft<integer_signedness signed>, m = 256 : i64, n = 256 : i64, k = 256 : i64, qk = 256 : i64, weight_block_stride = 144 : i64, weight_scale_byte_offset = 4 : i64, weight_quant_byte_offset = 16 : i64, subblock_length = 32 : i64, num_subblocks = 8 : i64, scale_bits = 6 : i64, scale_table_bytes = 12 : i64}
  }
}

// The prior routes GEMM ^ ime ^ q4_K to the format-keyed q4_K whole-matrix variant
// (still cost 0.5, GEMM takeover) and CONSTRUCTS the typed region.
// REGION: weft.exec.variant @ime_vmadot_matmul_slice
// REGION: weft_ime.q4_K_matmul_tile
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: mac_batched = 1
// REGION-SAME: mat_k = 256
// REGION-SAME: weight_format = "q4_K"
// REGION-SAME: wide_njw = 1
// REGION-SAME: wide_vlen_bits = 256
// The typed region is the SIX DECOMPOSED bricks (block_index + int8 activation
// fragment + TWO int32 accumulators entry args), NOT an opaque helper.
// REGION: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>, %{{.*}}: vector<16xi32>):
// REGION: weft_ime.q4_K_dequant_core %{{.*}} {decode_model = "q4_K_raw_nibble"
// REGION-SAME: -> vector<32xi8>
// REGION: weft_ime.q4_K_scale_min_unpack_core %{{.*}} {{{.*}}scale_min_model = "get_scale_min_k4"
// REGION: weft_ime.vmadot_mac_leaf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {accum_bits = 32
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: -> vector<16xi32>
// REGION: weft_ime.q4_K_scale_weighted_accum %{{.*}}, %{{.*}}, %{{.*}} {accum_model = "scale_weighted_sum"}
// REGION: weft_ime.q4_K_min_bias_accum %{{.*}}, %{{.*}}, %{{.*}} {bias_model = "activation_sum_min_bias"}
// REGION: weft_ime.q4_K_matmul_tile_yield %{{.*}}, %{{.*}} : vector<16xi32>, vector<16xi32>

// The emitted kernel: the validated vmadot MAC leaf + the fp16 epilogue helpers +
// the q4_K raw-nibble decode + the 6-bit get_scale_min_k4 unpack + the tiled
// two-level-fold q4_K kernel (S_scale AND S_min) + the structured extern "C"
// wrapper. The int32 core is int32-EXACT; the d/dmin fold is the deferred float
// epilogue.
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// [档 C#7] leaf-batching selection provenance (sel.reason wired, no longer dead):
// EMITC-SAME: weft_ime.mac_leaf_batching helper=weft_ime_vmadot_mac_kloop batched=1
// EMITC-SAME: register_resident_accumulate=1
// EMITC-SAME: static inline void weft_ime_vmadot_mac_kloop
// EMITC-SAME: vmadot    v2, v0, v1
// q4_K's current two-accumulator scale/min topology exposes an honest-null
// wide-reuse axis. The exact typed body carries NJW=1 and no wide leaf is emitted.
// EMITC: emitc.verbatim
// EMITC-SAME: weft_ime.pat1_tiling=decline njw=1
// EMITC-SAME: exact typed body carries the narrow computation schedule
// EMITC: emitc.verbatim
// EMITC-SAME: weft_ime.fp16_epilogue=weft_ime_fp16_to_f32
// EMITC-SAME: static inline float weft_ime_fp16_to_f32
// EMITC: emitc.verbatim
// EMITC-SAME: decode_model=q4_K_raw_nibble
// EMITC-SAME: static inline void weft_ime_q4_K_dequant_fragment
// EMITC: emitc.verbatim
// EMITC-SAME: scale_min_model=get_scale_min_k4
// EMITC-SAME: static inline void weft_ime_q4_K_get_scale_min
// EMITC: emitc.verbatim
// EMITC-SAME: two_level_fold=kquant_dmin_bsums_min
// EMITC-SAME: static void weft_ime_q4_K_vmadot_matmul
// EMITC-SAME: weft_ime_q4_K_dequant_fragment(blk[nl], b, kf, Bdec + kf * 32 + nl * 8)
// EMITC-SAME: weft_ime_vmadot_mac_kloop(Ablk, Bdec, 4, sumi)
// EMITC-SAME: Sc[ml * 4 + nl] += (int32_t)sc[b][nl] * sumi[ml * 4 + nl]
// EMITC-SAME: Sm[ml * 4 + nl] += (int32_t)mm[b][nl] * asum[ml]
// EMITC: emitc.func @weft_emitc_ime_q4_K_matmul_kernel_ime_vmadot_matmul_slice
// EMITC: weft_emitc.route_source_op=weft_ime.q4_K_matmul_tile role=compute
// EMITC: call_opaque "weft_ime_q4_K_vmadot_matmul"
