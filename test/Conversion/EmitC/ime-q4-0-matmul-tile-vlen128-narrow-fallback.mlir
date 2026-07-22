// Direct IME emission must use the same family legality domain as proposal and
// selected-body construction. VLEN=128 is outside the only real-K1-validated
// IME1 4x4x8 envelope, so a hand-written tile cannot bypass that boundary and
// reach an emitter-side "narrow fallback". Missing VLEN is rejected as well.
//
// RUN: not weft-opt %s --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-VLEN
// RUN: sed 's/, vlen_bits = "128"//' %s | not weft-opt --weft-materialize-emitc-lowerable-routes 2>&1 | FileCheck %s --check-prefix=MISSING-VLEN

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

// UNSUPPORTED-VLEN: bound family construction for origin 'ime-plugin' rejected selected variant legality
// MISSING-VLEN: bound family construction for origin 'ime-plugin' rejected selected variant legality
