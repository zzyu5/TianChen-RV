// G4 M2: the FORMAT-KEYED q8_0 IME GEMM tile typed-region front door (the FLAT-int8
// copy-adapt sibling of q4_0).
//
// A kernel carrying ONLY the spacemit.ime capability FACT + a whole-matrix SHAPE
// fact (ime_matmul_shape) + a WEIGHT-FORMAT fact (ime_weight_format = "q8_0") --
// no high-level op, no family-name branch -- drives the generic
// proposal/selection/boundary pipeline to CONSTRUCT the typed-region
// weft_ime.q8_0_matmul_tile op. The weight-format fact is pure data flow of the
// capability, keyed ON TOP of the whole-matrix GEMM prior.
//
// RUN 1 (STRUCTURE): stop at boundary materialization and assert the CONSTRUCTED
// typed region is the three DECOMPOSED bricks (q8_0_dequant_core + vmadot_mac_leaf
// + yield), NOT an opaque body.
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries | FileCheck %s --check-prefix=REGION
//
// RUN 2 (EMISSION): the full pipeline lowers the region to the q8_0-decode +
// vmadot MAC EmitC kernel (op-identity driven, I5).
// RUN: weft-opt %s --weft-materialize-plugin-variants --weft-select-variants --weft-materialize-selected-lowering-boundaries --weft-materialize-emitc-lowerable-routes | FileCheck %s --check-prefix=EMITC --implicit-check-not="weft_rvv" --implicit-check-not="weft_toy"

module {
  weft.exec.kernel @ime_q8_0_matmul_kernel {
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available",
      march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii",
      vlen_bits = "256",
      available_harts = "0-3",
      ime_matmul_shape = "256x256x256",
      ime_weight_format = "q8_0"
    }
  }
}

// The prior routes GEMM ^ ime ^ q8_0 to the format-keyed q8_0 whole-matrix variant
// (still cost 0.5, GEMM takeover) and CONSTRUCTS the typed region.
// REGION: weft.exec.variant @ime_vmadot_matmul_slice
// REGION-SAME: ime.weight_format = "q8_0"
// REGION: weft_ime.q8_0_matmul_tile
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: mat_k = 256
// REGION-SAME: weight_format = "q8_0"
// The typed region is the three DECOMPOSED bricks (block_index + int8 activation
// fragment + int32 accumulator entry args), NOT an opaque helper.
// REGION: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>):
// REGION: weft_ime.q8_0_dequant_core %{{.*}} {decode_model = "q8_0_direct_int8"
// REGION-SAME: -> vector<32xi8>
// REGION: weft_ime.vmadot_mac_leaf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {accum_bits = 32
// REGION-SAME: ime_op = "vmadot"
// REGION-SAME: -> vector<16xi32>
// REGION: weft_ime.q8_0_matmul_tile_yield %{{.*}} : vector<16xi32>

// G8 applied=>deployed: the emitted kernel is the register-resident BATCHED vmadot
// MAC leaf + the capability-keyed DEPLOYED WIDE leaf (_w2, emitted in the PROLOGUE
// declared-before-use; q8_0 is a flat MAC-BOUND format so the bottleneck-shape leg
// admits the wide tiling) + the q8_0 decode helper + the tiled q8_0 kernel whose
// BULK column-tiles run through the wide leaf and whose leftover tiles fall back to
// the narrow leaf. int32-EXACT; wide sub-tile == narrow leaf (K1-sealed f5e77482).
// EMITC: emitc.include <"stdint.h">
// EMITC: emitc.verbatim
// [档 C#7] leaf-batching selection provenance (sel.reason wired, no longer dead):
// EMITC-SAME: weft_ime.mac_leaf_batching helper=weft_ime_vmadot_mac_kloop batched=1
// EMITC-SAME: register_resident_accumulate=1
// EMITC-SAME: static inline void weft_ime_vmadot_mac_kloop
// EMITC-SAME: vmadot    v2, v0, v1
// EMITC: emitc.verbatim
// EMITC-SAME: weft_ime.pat1_tiling=IME-VMADOT-TILE-W2-Areuse status=mechanized njw=2
// EMITC-SAME: discriminant=vreg_budget deployed=1
// EMITC-SAME: rpfIn=1 rpfAcc=2
// EMITC: emitc.verbatim
// EMITC-SAME: weft_ime_vmadot_mac_kloop_w2
// EMITC-SAME: tile_width_njw=2 a_fragment_reuse=1
// EMITC-SAME: vmadot    v4, v0, v6
// EMITC: emitc.verbatim
// EMITC-SAME: decode_model=q8_0_direct_int8
// EMITC-SAME: static inline void weft_ime_q8_0_dequant_fragment
// EMITC-SAME: out[j] = qs[j]
// EMITC: emitc.verbatim
// EMITC-SAME: int32_exact=1
// EMITC-SAME: wide_deployed_njw=2
// EMITC-SAME: static void weft_ime_q8_0_vmadot_matmul
// EMITC-SAME: weft_ime_vmadot_mac_kloop_w2(Arow, Bdec, kt * 32, kt, frag)
// EMITC-SAME: frag[w * 16 + r * 4 + c]
// EMITC-SAME: weft_ime_vmadot_mac_kloop(Arow, Bdec, kt, frag)
// EMITC: emitc.func @weft_emitc_ime_q8_0_matmul_kernel_ime_vmadot_matmul_slice
// EMITC: weft_emitc.route_source_op=weft_ime.q8_0_matmul_tile role=compute
// EMITC: call_opaque "weft_ime_q8_0_vmadot_matmul"
