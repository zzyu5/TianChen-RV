// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G4 M2: the FORMAT-KEYED q8_0 IME GEMM tile typed-region boundary (the FLAT-int8
// copy-adapt sibling of q4_0). It carries the SAME int8->int32 vmadot MAC envelope
// + whole-matrix problem dims as weft.ime.matmul PLUS the q8_0 weight-format FACTS
// (34-byte block stride, direct int8 decode), and OWNS a typed region: the
// innermost contraction-block tile body whose entry args are block_index (index),
// the int8 activation fragment (vector<32xi8>), and the carried-in int32
// accumulator (vector<16xi32>). The region carries exactly the DECOMPOSED bricks --
// the q8_0 weight decode, the vmadot MAC leaf, and the int32 yield -- and NO opaque
// body.
module {
  // CHECK-LABEL: weft.exec.kernel @ime_q8_0_tile_valid
  weft.exec.kernel @ime_q8_0_tile_valid {
    weft.exec.capability @spacemit_ime {
      id = "spacemit.ime",
      kind = "isa-matrix-vector-backed",
      status = "available"
    }
    weft.exec.variant @ime_vmadot_matmul_slice attributes {
      origin = "ime-plugin",
      requires = [@spacemit_ime]
    } {
    }
    // CHECK: weft_ime.q8_0_matmul_tile
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK-SAME: mat_k = 256 : i64
    // CHECK-SAME: weight_format = "q8_0"
    // CHECK: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>):
    // CHECK: weft_ime.q8_0_dequant_core %{{.*}} {decode_model = "q8_0_direct_int8"
    // CHECK: weft_ime.vmadot_mac_leaf
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK: weft_ime.q8_0_matmul_tile_yield %{{.*}} : vector<16xi32>
    weft_ime.q8_0_matmul_tile attributes {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadot_matmul_slice,
      source_kernel = "ime_q8_0_tile_valid",
      ime_op = "vmadot",
      elem_in_bits = 8 : i64,
      accum_bits = 32 : i64,
      mac_m = 4 : i64,
      mac_n = 4 : i64,
      mac_k = 8 : i64,
      mat_m = 256 : i64,
      mat_n = 256 : i64,
      mat_k = 256 : i64,
      weight_format = "q8_0",
      qk = 32 : i64,
      weight_block_stride = 34 : i64,
      weight_quant_byte_offset = 2 : i64,
      available_harts = "0-3"
    } {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = weft_ime.q8_0_dequant_core %bi {
        decode_model = "q8_0_direct_int8",
        qk = 32 : i64,
        weight_block_stride = 34 : i64,
        weight_quant_byte_offset = 2 : i64,
        weight_scale_byte_offset = 0 : i64
      } : index -> vector<32xi8>
      %next = weft_ime.vmadot_mac_leaf %a, %b, %bi, %acc {
        ime_op = "vmadot",
        elem_in_bits = 8 : i64,
        accum_bits = 32 : i64,
        mac_m = 4 : i64,
        mac_n = 4 : i64,
        mac_k = 8 : i64
      } : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      weft_ime.q8_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (anti-opaque): the typed region admits ONLY the three decomposed
// bricks. A foreign op in the region is rejected fail-closed.
module {
  weft.exec.kernel @ime_q8_0_tile_opaque_body {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{typed region admits ONLY the decomposed q8_0 dequant / vmadot-leaf / yield bricks}}
    weft_ime.q8_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q8_0_tile_opaque_body", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q8_0", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = weft_ime.q8_0_dequant_core %bi {decode_model = "q8_0_direct_int8", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %opaque = builtin.unrealized_conversion_cast %b : vector<32xi8> to vector<16xi32>
      %next = weft_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      weft_ime.q8_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed decode): the dequant core admits ONLY the q8_0 direct
// int8 decode model.
module {
  weft.exec.kernel @ime_q8_0_tile_bad_decode {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    weft_ime.q8_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q8_0_tile_bad_decode", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q8_0", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      // expected-error@+1 {{decode_model must be 'q8_0_direct_int8'}}
      %b = weft_ime.q8_0_dequant_core %bi {decode_model = "q4_0_offset_binary_nibble", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %next = weft_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      weft_ime.q8_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed format): the q8_0 tile op admits ONLY weight_format=q8_0.
module {
  weft.exec.kernel @ime_q8_0_tile_bad_format {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{weight_format must be 'q8_0'}}
    weft_ime.q8_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q8_0_tile_bad_format", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_0", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = weft_ime.q8_0_dequant_core %bi {decode_model = "q8_0_direct_int8", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %next = weft_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      weft_ime.q8_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed stride): the q8_0 block stride is 34 bytes (fp16 d + 32
// int8 quants); the q4_0 stride 18 is rejected.
module {
  weft.exec.kernel @ime_q8_0_tile_bad_stride {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{weight_block_stride must be 34}}
    weft_ime.q8_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q8_0_tile_bad_stride", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q8_0", qk = 32 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi32>):
      %b = weft_ime.q8_0_dequant_core %bi {decode_model = "q8_0_direct_int8", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      %next = weft_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      weft_ime.q8_0_matmul_tile_yield %next : vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed MAC): the vmadot leaf admits ONLY the int32-exact
// accumulate. A non-int32 accumulator tile is rejected.
module {
  weft.exec.kernel @ime_q8_0_tile_bad_acc {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    weft_ime.q8_0_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q8_0_tile_bad_acc", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q8_0", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %acc: vector<16xi16>):
      %b = weft_ime.q8_0_dequant_core %bi {decode_model = "q8_0_direct_int8", qk = 32 : i64, weight_block_stride = 34 : i64, weight_quant_byte_offset = 2 : i64, weight_scale_byte_offset = 0 : i64} : index -> vector<32xi8>
      // expected-error@+1 {{acc_in/acc_out must be vector<16xi32>}}
      %next = weft_ime.vmadot_mac_leaf %a, %b, %bi, %acc {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi16>) -> vector<16xi16>
      weft_ime.q8_0_matmul_tile_yield %next : vector<16xi16>
    }
  }
}
