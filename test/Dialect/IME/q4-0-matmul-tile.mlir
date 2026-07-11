// RUN: tcrv-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G4 M1a: the FORMAT-KEYED q4_0 IME GEMM tile typed-region boundary. It carries the
// SAME int8->int32 vmadot MAC envelope + whole-matrix problem dims as tcrv.ime.matmul
// PLUS the q4_0 weight-format FACTS, and OWNS a typed region: the innermost
// contraction-block tile body whose entry args are block_index (index), the int8
// activation fragment (vector<32xi8>), and the carried-in int32 accumulator
// (vector<16xi32>). The region carries exactly the DECOMPOSED bricks -- the q4_0
// weight decode, the vmadot MAC leaf, and the int32 yield -- and NO opaque body.
module {
  // CHECK-LABEL: tcrv.exec.kernel @ime_q4_0_tile_valid
  tcrv.exec.kernel @ime_q4_0_tile_valid {
    tcrv.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: tcrv_ime.q4_0_matmul_tile
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK-SAME: mat_k = 256 : i64
    // CHECK-SAME: weight_format = "q4_0"
    // CHECK: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>):
    // CHECK: tcrv_ime.q4_0_dequant_core %{{.*}} {decode_model = "q4_0_offset_binary_nibble"
    // CHECK: tcrv_ime.vmadot_mac_leaf
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK: tcrv_ime.q4_0_matmul_tile_yield %{{.*}} : vector<16xi32>
    tcrv_ime.q4_0_matmul_tile attributes {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadot_matmul_slice,
      source_kernel = "ime_q4_0_tile_valid",
      ime_op = "vmadot",
      elem_in_bits = 8 : i64,
      accum_bits = 32 : i64,
      mac_m = 4 : i64,
      mac_n = 4 : i64,
      mac_k = 8 : i64,
      mat_m = 256 : i64,
      mat_n = 256 : i64,
      mat_k = 256 : i64,
      weight_format = "q4_0",
      qk = 32 : i64,
      weight_block_stride = 18 : i64,
      weight_quant_byte_offset = 2 : i64,
      available_harts = "0-3"
    } {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = tcrv_ime.q4_0_dequant_core %bi {
        decode_model = "q4_0_offset_binary_nibble",
        qk = 32 : i64,
        weight_block_stride = 18 : i64,
        weight_quant_byte_offset = 2 : i64,
        weight_scale_byte_offset = 0 : i64
      } : index -> vector<32xi8>
      %next = tcrv_ime.vmadot_mac_leaf %a, %b, %bi, %acc {
        ime_op = "vmadot",
        elem_in_bits = 8 : i64,
        accum_bits = 32 : i64,
        mac_m = 4 : i64,
        mac_n = 4 : i64,
        mac_k = 8 : i64
      } : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      tcrv_ime.q4_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (anti-opaque): the typed region admits ONLY the three decomposed
// bricks. A foreign op in the region is rejected fail-closed (the M-FLAT
// construction discipline: no opaque body ever emitted).
module {
  tcrv.exec.kernel @ime_q4_0_tile_opaque_body {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{typed region admits ONLY the decomposed q4_0 dequant / vmadot-leaf / yield bricks}}
    tcrv_ime.q4_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_0_tile_opaque_body", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_0", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = tcrv_ime.q4_0_dequant_core %bi {decode_model = "q4_0_offset_binary_nibble", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %opaque = builtin.unrealized_conversion_cast %b : vector<32xi8> to vector<16xi32>
      %next = tcrv_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      tcrv_ime.q4_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed decode): the dequant core admits ONLY the q4_0
// offset-binary nibble decode model.
module {
  tcrv.exec.kernel @ime_q4_0_tile_bad_decode {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    tcrv_ime.q4_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_0_tile_bad_decode", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_0", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      // expected-error@+1 {{decode_model must be 'q4_0_offset_binary_nibble'}}
      %b = tcrv_ime.q4_0_dequant_core %bi {decode_model = "q8_0_passthrough", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %next = tcrv_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      tcrv_ime.q4_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed format): only q4_0 is modeled in M1.
module {
  tcrv.exec.kernel @ime_q4_0_tile_bad_format {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{weight_format must be 'q4_0'}}
    tcrv_ime.q4_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_0_tile_bad_format", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_K", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = tcrv_ime.q4_0_dequant_core %bi {decode_model = "q4_0_offset_binary_nibble", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %next = tcrv_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      tcrv_ime.q4_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed remainder): mat_k=16 is a whole multiple of mac_k=8 but
// NOT of qk=32, so K does not partition into whole q4_0 blocks => rejected.
module {
  tcrv.exec.kernel @ime_q4_0_tile_partial_block {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{must be a whole multiple of qk=32}}
    tcrv_ime.q4_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_0_tile_partial_block", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 16 : i64, weight_format = "q4_0", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = tcrv_ime.q4_0_dequant_core %bi {decode_model = "q4_0_offset_binary_nibble", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %next = tcrv_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      tcrv_ime.q4_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed MAC): the vmadot leaf admits ONLY the int32-exact
// accumulate. A non-int32 accumulator tile is rejected.
module {
  tcrv.exec.kernel @ime_q4_0_tile_bad_acc {
    tcrv.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    tcrv.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    tcrv_ime.q4_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_0_tile_bad_acc", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_0", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi16>):
      %b = tcrv_ime.q4_0_dequant_core %bi {decode_model = "q4_0_offset_binary_nibble", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      // expected-error@+1 {{acc_in/acc_out must be vector<16xi32>}}
      %next = tcrv_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi16>) -> vector<16xi16>
      tcrv_ime.q4_0_matmul_tile_yield %next : vector<16xi16>
    }
  }
}
