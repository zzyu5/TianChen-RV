// G4 M1a: the FORMAT-KEYED q4_0 IME GEMM tile typed-region front door.
//
// A canonical q4_0 contraction problem plus the spacemit.ime target capability
// -- with no family-name branch -- drive the generic
// proposal/selection/boundary pipeline to CONSTRUCT the typed-region
// weft_ime.q4_0_matmul_tile op (the RVV lowerToRepackGemm front-door precedent
// applied to the IME matrix paradigm). Geometry and block layout come from exact P.
//
// RUN 1 (STRUCTURE): stop at boundary materialization and assert the CONSTRUCTED
// typed region is the three DECOMPOSED bricks (q4_0_dequant_core + vmadot_mac_leaf
// + yield), NOT an opaque body -- the certified-shape checker.
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REGION
//
// RUN 2 (EMISSION): the full pipeline lowers the region to the q4_0-decode +
// vmadot MAC EmitC kernel (op-identity driven, I5).
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy"
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries > %t.constructed
// RUN: sed 's/, wide_vreg_floor = 7 : i64//' %t.constructed | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: sed 's/wide_njw = 2 : i64/wide_njw = 1 : i64/' %t.constructed | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=CONFLICT

module {
  weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3"
  }
  weft.exec.target @ime_q4_0_profile {id = "ime.q4_0.profile", target_kind = "profile", construction_domain = "riscv-execution", capability_providers = [@spacemit_ime]}
  weft.exec.kernel @ime_q4_0_matmul_kernel attributes {target = @ime_q4_0_profile, problem = @canonical_problem} {
    weft.exec.block_q4_0_contraction_problem @canonical_problem {activation_signedness = #weft<integer_signedness signed>, m = 256 : i64, n = 256 : i64, k = 256 : i64, qk = 32 : i64, weight_block_stride = 18 : i64, weight_scale_byte_offset = 0 : i64, weight_quant_byte_offset = 2 : i64}
  }
}

// The prior routes GEMM ^ ime ^ q4_0 to the format-keyed q4_0 whole-matrix variant
// (still cost 0.5, GEMM takeover) and CONSTRUCTS the typed region.
// REGION: weft.exec.variant @ime_vmadot_matmul_slice
// REGION: weft_ime.q4_0_matmul_tile
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: mac_batched = 1
// REGION-SAME: mat_k = 256
// REGION-SAME: weight_format = "q4_0"
// REGION-SAME: wide_njw = 2
// REGION-SAME: wide_vlen_bits = 256
// The typed region is the three DECOMPOSED bricks (block_index + int8 activation
// fragment + int32 accumulator entry args), NOT an opaque helper.
// REGION: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>):
// REGION: weft_ime.q4_0_dequant_core %{{.*}} {decode_model = "q4_0_offset_binary_nibble"
// REGION-SAME: -> vector<32xi8>
// REGION: weft_ime.vmadot_mac_leaf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {accum_bits = 32
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: -> vector<16xi32>
// REGION: weft_ime.q4_0_matmul_tile_yield %{{.*}} : vector<16xi32>

// PARTIAL: typed IME quant schedule must be absent before construction or carry all schedule fields atomically
// CONFLICT: IME typed schedule field 'wide_njw' conflicts with family construction

// G8 applied=>deployed: the emitted kernel is the register-resident BATCHED vmadot
// MAC leaf (single vsetvli, store once) + the capability-keyed DEPLOYED WIDE leaf
// (_w2, emitted in the PROLOGUE declared-before-use) + the q4_0 decode helper + the
// tiled q4_0 kernel whose BULK column-tiles run through the wide leaf and whose
// leftover (<njw) tiles fall back to the narrow leaf. int32-EXACT; wide sub-tile ==
// narrow leaf (K1-sealed f5e77482).
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// [档 C#7] leaf-batching selection provenance (sel.reason wired, no longer dead):
// EMITC-SAME: weft_ime.mac_leaf_batching helper=weft_ime_vmadot_mac_kloop batched=1
// EMITC-SAME: register_resident_accumulate=1
// EMITC-SAME: static inline void weft_ime_vmadot_mac_kloop
// EMITC-SAME: vmadot    v2, v0, v1
// The exact typed body's constructed schedule is projected directly.
// EMITC: emitc.verbatim
// EMITC-SAME: weft_ime.constructed_wide_schedule njw=2 deployed=1
// EMITC-SAME: rpfIn=1 rpfAcc=2
// The DEPLOYED wide leaf: ONE `vle8` of A (v0) across TWO independent `vmadot`s (v2
// and v4) into two 4x4 int32 accumulator pairs.
// EMITC: emitc.verbatim
// EMITC-SAME: weft_ime_vmadot_mac_kloop_w2
// EMITC-SAME: tile_width_njw=2 a_fragment_reuse=1
// EMITC-SAME: vmadot    v2, v0, v1
// EMITC-SAME: vmadot    v4, v0, v6
// EMITC: emitc.verbatim
// EMITC-SAME: decode_model=q4_0_offset_binary_nibble
// EMITC-SAME: static inline void weft_ime_q4_0_dequant_fragment
// EMITC-SAME: (int8_t)((int)(qs[j] & 0x0F) - 8)
// The int32 seal kernel now CALLS the deployed wide leaf over the bulk column-tiles
// (frag[w * 16 ...]) and the narrow leaf over the remainder.
// EMITC: emitc.verbatim
// EMITC-SAME: int32_exact=1
// EMITC-SAME: wide_deployed_njw=2
// EMITC-SAME: static void weft_ime_q4_0_vmadot_matmul
// EMITC-SAME: weft_ime_vmadot_mac_kloop_w2(Arow, Bdec, kt * 32, kt, frag)
// EMITC-SAME: frag[w * 16 + r * 4 + c]
// EMITC-SAME: weft_ime_vmadot_mac_kloop(Arow, Bdec, kt, frag)
// EMITC: emitc.func @weft_emitc_ime_q4_0_matmul_kernel_ime_vmadot_matmul_slice
// EMITC: weft_emitc.route_source_op=weft_ime.q4_0_matmul_tile role=compute
// EMITC: call_opaque "weft_ime_q4_0_vmadot_matmul"
// G5-M3 forward bridge + G8: the deferred per-block d_a*d_w SCALE-FOLD epilogue
// kernel (f32, the ggml-called FORWARD path) ALSO deploys the wide leaf over the
// bulk column-tiles (byte-exact int32 core + order-preserved fp16 fold), closing
// applied!=deployed on the forward path; narrow remainder + fold stays intact.
// EMITC: emitc.verbatim
// EMITC-SAME: scale_fold_epilogue=weft_ime_q4_0_vmadot_matmul_f32
// EMITC-SAME: fold_model=per_block_da_dw
// EMITC-SAME: wide_deployed_njw=2
// EMITC-SAME: static void weft_ime_q4_0_vmadot_matmul_f32
// EMITC-SAME: weft_ime_vmadot_mac_kloop_w2(Arow + b * frags_per_block * 32, Bdec, frags_per_block * 32, frags_per_block, frag)
// EMITC-SAME: Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[w * 16 + r * 4 + c]
// EMITC-SAME: Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[r * 4 + c]
// EMITC: emitc.func @weft_emitc_ime_q4_0_matmul_kernel_ime_vmadot_matmul_slice_f32
// EMITC: weft_emitc.route_source_op=weft_ime.q4_0_matmul_tile role=compute
// EMITC: call_opaque "weft_ime_q4_0_vmadot_matmul_f32"
