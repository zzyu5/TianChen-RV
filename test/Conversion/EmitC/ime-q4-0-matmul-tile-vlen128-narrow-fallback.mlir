// G8 wide-vmadot Leg1 (VLEN-mismatch capability decline) COVERAGE — the "VLEN-
// invariant / dual-board" seller-point fixture that every other IME fixture (all
// vlen_bits=256) leaves uncovered.
//
// The IME front door (IMEExtensionPlugin) fails closed for VLEN != 256 (the only
// real-K1-validated 4x4x8 MAC-fragment envelope), so a VLEN128 capability cannot be
// driven through --weft-materialize-plugin-variants. This fixture therefore exercises
// EMISSION ONLY on a PRE-CONSTRUCTED q4_0 tile whose sibling spacemit.ime capability
// provider carries vlen_bits = "128" — exactly what readDeployedVlenBits reads at
// emit time. It asserts the wide-vmadot capability leg (Leg1) DECLINES: the 4x8 int8
// fragment (macM*macK*elem_in_bits = 4*8*8 = 256 bits) needs rpfIn = ceil(256/128) =
// 2 vregs at VLEN128, so the single-`vle8 e8,m1` wide leaf is invalid -> narrow
// fallback. The honest decline is MATERIALIZED (not silent) and NO `_w2` wide leaf is
// emitted.
//
// RUN: weft-opt %s --weft-materialize-emitc-lowerable-routes | FileCheck %s --implicit-check-not="mac_kloop_w2" --implicit-check-not="wide_deployed_njw"

module {
  weft.exec.kernel @ime_q4_0_matmul_kernel {
    weft.exec.capability @spacemit_ime {available_harts = "0-3", id = "spacemit.ime", ime_matmul_shape = "256x256x256", ime_weight_format = "q4_0", kind = "isa-matrix-vector-backed", march = "rv64gcv_zfh_zvfh_zba_zicbop_xsmtvdotii", status = "available", vlen_bits = "128"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {condition = "spacemit_ime_capability_available", guard = "plugin_local_ime_vmadot_boundary", ime.signedness = "signed", ime.weight_format = "q4_0", origin = "ime-plugin", policy = "ime_int8_matmul_vmadot_mac", requires = [@spacemit_ime]} {
    }
    weft_ime.q4_0_matmul_tile attributes {accum_bits = 32 : i64, available_harts = "0-3", elem_in_bits = 8 : i64, ime_op = "vmadot", mac_k = 8 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mat_k = 256 : i64, mat_m = 256 : i64, mat_n = 256 : i64, origin = "ime-plugin", qk = 32 : i64, required_capabilities = [@spacemit_ime], role = "direct variant", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_0_matmul_kernel", status = "role-op-boundary", weight_block_stride = 18 : i64, weight_format = "q4_0", weight_quant_byte_offset = 2 : i64} {
    ^bb0(%arg0: index, %arg1: vector<32xi8>, %arg2: vector<16xi32>):
      %0 = weft_ime.q4_0_dequant_core %arg0 {decode_model = "q4_0_offset_binary_nibble", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %1 = weft_ime.vmadot_mac_leaf %arg1, %0, %arg0, %arg2 {accum_bits = 32 : i64, elem_in_bits = 8 : i64, ime_op = "vmadot", mac_k = 8 : i64, mac_m = 4 : i64, mac_n = 4 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_0_matmul_tile_yield %1 : vector<16xi32>
    }
  }
}

// The DEPLOYED narrow batched vmadot MAC leaf is still emitted (register-resident,
// single vsetvli, store once) — the wide leaf declines, so the narrow leaf stays the
// deployed leaf (auto fallback).
// CHECK: emitc.include <"stdint.h">
// CHECK: emitc.verbatim
// CHECK-SAME: register_resident_accumulate=1
// CHECK-SAME: static inline void weft_ime_vmadot_mac_kloop(
// CHECK-SAME: vmadot    v2, v0, v1

// Leg1 VLEN-mismatch capability DECLINE, MATERIALIZED as honest provenance (not
// silent): rpfIn = ceil(256/128) = 2 > 1, so the single-`vle8 e8,m1` wide leaf is
// invalid at VLEN128 -> narrow fallback. deployed=0, njw=1.
// CHECK: emitc.verbatim
// CHECK-SAME: weft_ime.pat1_tiling=decline njw=1 deployed=0
// CHECK-SAME: capability=vlen-fragment-mismatch decline
// CHECK-SAME: macM*macK*elem_in_bits=256 does not fit one 128-bit vreg (rpfIn=2)
// CHECK-SAME: narrow fallback

// The q4_0 decode helper + the tiled q4_0 int32 kernel. The kernel runs NARROW-ONLY
// (no wide column-tile loop, no wide_deployed_njw tag): the bulk loop calls the
// narrow batched leaf directly.
// CHECK: emitc.verbatim
// CHECK-SAME: decode_model=q4_0_offset_binary_nibble
// CHECK-SAME: static inline void weft_ime_q4_0_dequant_fragment
// CHECK: emitc.verbatim
// CHECK-SAME: int32_exact=1
// CHECK-SAME: static void weft_ime_q4_0_vmadot_matmul
// CHECK-SAME: weft_ime_vmadot_mac_kloop(Arow, Bdec, kt, frag)
// CHECK: emitc.func @weft_emitc_ime_q4_0_matmul_kernel_ime_vmadot_matmul_slice
// CHECK: call_opaque "weft_ime_q4_0_vmadot_matmul"
