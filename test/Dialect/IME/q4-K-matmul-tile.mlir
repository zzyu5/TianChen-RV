// RUN: weft-opt %s --split-input-file --verify-diagnostics | FileCheck %s

// G4 M2b: the FORMAT-KEYED q4_K IME GEMM tile typed-region boundary (the
// SUPER-BLOCK K-quant tile; the DEDICATED effort beyond the q4_0/q8_0 flat tiles).
// It carries the SAME int8->int32 vmadot MAC envelope + whole-matrix problem dims
// as weft.ime.matmul PLUS the q4_K weight-format FACTS (144-byte super-block stride,
// raw-nibble decode, qk=256), and OWNS a typed region: the innermost super-block
// tile body whose entry args are block_index (index), the int8 activation fragment
// (vector<32xi8>), and the carried-in TWO int32 accumulators (S_scale + S_min,
// vector<16xi32>). The region carries exactly the SIX DECOMPOSED bricks -- the
// raw-nibble decode, the 6-bit scale/min unpack, the vmadot MAC leaf, the
// scale-weighted accum (S_scale), the min-bias accum (S_min), and the two-tile
// yield -- and NO opaque body. Dropping the scale-weighted or min-bias accum brick
// (the HOLLOW bare-MAC q4_K shape) is fail-closed rejected.
module {
  // CHECK-LABEL: weft.exec.kernel @ime_q4_K_tile_valid
  weft.exec.kernel @ime_q4_K_tile_valid {
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
    // CHECK: weft_ime.q4_K_matmul_tile
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK-SAME: mat_k = 256 : i64
    // CHECK-SAME: weight_format = "q4_K"
    // CHECK: ^bb0(%{{.*}}: index, %{{.*}}: vector<32xi8>, %{{.*}}: vector<16xi32>, %{{.*}}: vector<16xi32>):
    // CHECK: weft_ime.q4_K_dequant_core %{{.*}} {decode_model = "q4_K_raw_nibble"
    // CHECK: weft_ime.q4_K_scale_min_unpack_core %{{.*}} {{{.*}}scale_min_model = "get_scale_min_k4"
    // CHECK: weft_ime.vmadot_mac_leaf
    // CHECK-SAME: ime_op = "vmadot"
    // CHECK: weft_ime.q4_K_scale_weighted_accum %{{.*}} {accum_model = "scale_weighted_sum"}
    // CHECK: weft_ime.q4_K_min_bias_accum %{{.*}} {bias_model = "activation_sum_min_bias"}
    // CHECK: weft_ime.q4_K_matmul_tile_yield %{{.*}}, %{{.*}} : vector<16xi32>, vector<16xi32>
    weft_ime.q4_K_matmul_tile attributes {
      origin = "ime-plugin",
      required_capabilities = [@spacemit_ime],
      role = "direct variant",
      status = "role-op-boundary",
      selected_variant = @ime_vmadot_matmul_slice,
      source_kernel = "ime_q4_K_tile_valid",
      ime_op = "vmadot",
      elem_in_bits = 8 : i64,
      accum_bits = 32 : i64,
      mac_m = 4 : i64,
      mac_n = 4 : i64,
      mac_k = 8 : i64,
      mat_m = 256 : i64,
      mat_n = 256 : i64,
      mat_k = 256 : i64,
      weight_format = "q4_K",
      qk = 256 : i64,
      weight_block_stride = 144 : i64,
      weight_quant_byte_offset = 16 : i64,
      available_harts = "0-3"
    } {
    ^bb0(%bi: index, %a: vector<32xi8>, %accS: vector<16xi32>, %accM: vector<16xi32>):
      %b = weft_ime.q4_K_dequant_core %bi {
        decode_model = "q4_K_raw_nibble",
        qk = 256 : i64,
        weight_block_stride = 144 : i64,
        weight_quant_byte_offset = 16 : i64,
        weight_scale_byte_offset = 4 : i64
      } : index -> vector<32xi8>
      %sc, %m = weft_ime.q4_K_scale_min_unpack_core %bi {
        scale_min_model = "get_scale_min_k4",
        num_sub_blocks = 8 : i64,
        scale_bits = 6 : i64,
        k_scale_size = 12 : i64,
        weight_scale_byte_offset = 4 : i64
      } : index -> vector<16xi32>, vector<16xi32>
      %sumi = weft_ime.vmadot_mac_leaf %a, %b, %bi, %accS {
        ime_op = "vmadot",
        elem_in_bits = 8 : i64,
        accum_bits = 32 : i64,
        mac_m = 4 : i64,
        mac_n = 4 : i64,
        mac_k = 8 : i64
      } : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      %nextS = weft_ime.q4_K_scale_weighted_accum %sumi, %sc, %accS {
        accum_model = "scale_weighted_sum"
      } : (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %nextM = weft_ime.q4_K_min_bias_accum %a, %m, %accM {
        bias_model = "activation_sum_min_bias"
      } : (vector<32xi8>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_K_matmul_tile_yield %nextS, %nextM : vector<16xi32>, vector<16xi32>
    }
  }
}

// -----

// Falsifier (ANTI-HOLLOW, the q4_K-specific rule): the typed region MUST carry the
// scale-weighted accum brick (S_scale). Dropping it -- the bare-nibble-MAC shape
// that would be a HOLLOW q4_K representation (no per-sub-block scale weighting) --
// is rejected fail-closed.
module {
  weft.exec.kernel @ime_q4_K_tile_hollow_no_scale {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{typed region must contain exactly one weft.ime.q4_K_scale_weighted_accum brick}}
    weft_ime.q4_K_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_K_tile_hollow_no_scale", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_K", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %accS: vector<16xi32>, %accM: vector<16xi32>):
      %b = weft_ime.q4_K_dequant_core %bi {decode_model = "q4_K_raw_nibble", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<32xi8>
      %sc, %m = weft_ime.q4_K_scale_min_unpack_core %bi {scale_min_model = "get_scale_min_k4", num_sub_blocks = 8 : i64, scale_bits = 6 : i64, k_scale_size = 12 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<16xi32>, vector<16xi32>
      %sumi = weft_ime.vmadot_mac_leaf %a, %b, %bi, %accS {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      %nextM = weft_ime.q4_K_min_bias_accum %a, %m, %accM {bias_model = "activation_sum_min_bias"} : (vector<32xi8>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_K_matmul_tile_yield %sumi, %nextM : vector<16xi32>, vector<16xi32>
    }
  }
}

// -----

// Falsifier (ANTI-HOLLOW): the typed region MUST carry the min-bias accum brick
// (S_min). Dropping it -- the shape that would omit the q4_K min bias -- is
// rejected fail-closed.
module {
  weft.exec.kernel @ime_q4_K_tile_hollow_no_min {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{typed region must contain exactly one weft.ime.q4_K_min_bias_accum brick}}
    weft_ime.q4_K_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_K_tile_hollow_no_min", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_K", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %accS: vector<16xi32>, %accM: vector<16xi32>):
      %b = weft_ime.q4_K_dequant_core %bi {decode_model = "q4_K_raw_nibble", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<32xi8>
      %sc, %m = weft_ime.q4_K_scale_min_unpack_core %bi {scale_min_model = "get_scale_min_k4", num_sub_blocks = 8 : i64, scale_bits = 6 : i64, k_scale_size = 12 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<16xi32>, vector<16xi32>
      %sumi = weft_ime.vmadot_mac_leaf %a, %b, %bi, %accS {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      %nextS = weft_ime.q4_K_scale_weighted_accum %sumi, %sc, %accS {accum_model = "scale_weighted_sum"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_K_matmul_tile_yield %nextS, %nextS : vector<16xi32>, vector<16xi32>
    }
  }
}

// -----

// Falsifier (anti-opaque): the typed region admits ONLY the six decomposed bricks.
// A foreign op in the region is rejected fail-closed.
module {
  weft.exec.kernel @ime_q4_K_tile_opaque_body {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{typed region admits ONLY the decomposed q4_K}}
    weft_ime.q4_K_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_K_tile_opaque_body", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_K", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %accS: vector<16xi32>, %accM: vector<16xi32>):
      %b = weft_ime.q4_K_dequant_core %bi {decode_model = "q4_K_raw_nibble", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<32xi8>
      %sc, %m = weft_ime.q4_K_scale_min_unpack_core %bi {scale_min_model = "get_scale_min_k4", num_sub_blocks = 8 : i64, scale_bits = 6 : i64, k_scale_size = 12 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<16xi32>, vector<16xi32>
      %opaque = builtin.unrealized_conversion_cast %b : vector<32xi8> to vector<16xi32>
      %sumi = weft_ime.vmadot_mac_leaf %a, %b, %bi, %accS {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      %nextS = weft_ime.q4_K_scale_weighted_accum %sumi, %sc, %accS {accum_model = "scale_weighted_sum"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %nextM = weft_ime.q4_K_min_bias_accum %a, %m, %accM {bias_model = "activation_sum_min_bias"} : (vector<32xi8>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_K_matmul_tile_yield %nextS, %nextM : vector<16xi32>, vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed decode): the q4_K dequant core admits ONLY the raw-nibble
// decode model (q4_0 offset-binary is rejected).
module {
  weft.exec.kernel @ime_q4_K_tile_bad_decode {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    weft_ime.q4_K_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_K_tile_bad_decode", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_K", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %accS: vector<16xi32>, %accM: vector<16xi32>):
      // expected-error@+1 {{decode_model must be 'q4_K_raw_nibble'}}
      %b = weft_ime.q4_K_dequant_core %bi {decode_model = "q4_0_offset_binary_nibble", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<32xi8>
      %sc, %m = weft_ime.q4_K_scale_min_unpack_core %bi {scale_min_model = "get_scale_min_k4", num_sub_blocks = 8 : i64, scale_bits = 6 : i64, k_scale_size = 12 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<16xi32>, vector<16xi32>
      %sumi = weft_ime.vmadot_mac_leaf %a, %b, %bi, %accS {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      %nextS = weft_ime.q4_K_scale_weighted_accum %sumi, %sc, %accS {accum_model = "scale_weighted_sum"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %nextM = weft_ime.q4_K_min_bias_accum %a, %m, %accM {bias_model = "activation_sum_min_bias"} : (vector<32xi8>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_K_matmul_tile_yield %nextS, %nextM : vector<16xi32>, vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed stride): the q4_K super-block stride is 144 bytes; the
// q4_0 stride 18 is rejected.
module {
  weft.exec.kernel @ime_q4_K_tile_bad_stride {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    // expected-error@+1 {{weight_block_stride must be 144}}
    weft_ime.q4_K_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_K_tile_bad_stride", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_K", qk = 256 : i64, weight_block_stride = 18 : i64, weight_quant_byte_offset = 16 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %accS: vector<16xi32>, %accM: vector<16xi32>):
      %b = weft_ime.q4_K_dequant_core %bi {decode_model = "q4_K_raw_nibble", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<32xi8>
      %sc, %m = weft_ime.q4_K_scale_min_unpack_core %bi {scale_min_model = "get_scale_min_k4", num_sub_blocks = 8 : i64, scale_bits = 6 : i64, k_scale_size = 12 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<16xi32>, vector<16xi32>
      %sumi = weft_ime.vmadot_mac_leaf %a, %b, %bi, %accS {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      %nextS = weft_ime.q4_K_scale_weighted_accum %sumi, %sc, %accS {accum_model = "scale_weighted_sum"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %nextM = weft_ime.q4_K_min_bias_accum %a, %m, %accM {bias_model = "activation_sum_min_bias"} : (vector<32xi8>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_K_matmul_tile_yield %nextS, %nextM : vector<16xi32>, vector<16xi32>
    }
  }
}

// -----

// Falsifier (fail-closed scale/min model): the scale/min unpack admits ONLY the
// canonical get_scale_min_k4.
module {
  weft.exec.kernel @ime_q4_K_tile_bad_scalemin {
    weft.exec.capability @spacemit_ime {id = "spacemit.ime", kind = "isa-matrix-vector-backed", status = "available"}
    weft.exec.variant @ime_vmadot_matmul_slice attributes {origin = "ime-plugin", requires = [@spacemit_ime]} {}
    weft_ime.q4_K_matmul_tile attributes {origin = "ime-plugin", required_capabilities = [@spacemit_ime], role = "direct variant", status = "role-op-boundary", selected_variant = @ime_vmadot_matmul_slice, source_kernel = "ime_q4_K_tile_bad_scalemin", ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64, mat_m = 256 : i64, mat_n = 256 : i64, mat_k = 256 : i64, weight_format = "q4_K", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, available_harts = "0-3"} {
    ^bb0(%bi: index, %a: vector<32xi8>, %accS: vector<16xi32>, %accM: vector<16xi32>):
      %b = weft_ime.q4_K_dequant_core %bi {decode_model = "q4_K_raw_nibble", qk = 256 : i64, weight_block_stride = 144 : i64, weight_quant_byte_offset = 16 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<32xi8>
      // expected-error@+1 {{scale_min_model must be 'get_scale_min_k4'}}
      %sc, %m = weft_ime.q4_K_scale_min_unpack_core %bi {scale_min_model = "wrong_unpack", num_sub_blocks = 8 : i64, scale_bits = 6 : i64, k_scale_size = 12 : i64, weight_scale_byte_offset = 4 : i64} : index -> vector<16xi32>, vector<16xi32>
      %sumi = weft_ime.vmadot_mac_leaf %a, %b, %bi, %accS {ime_op = "vmadot", elem_in_bits = 8 : i64, accum_bits = 32 : i64, mac_m = 4 : i64, mac_n = 4 : i64, mac_k = 8 : i64} : (vector<32xi8>, vector<32xi8>, index, vector<16xi32>) -> vector<16xi32>
      %nextS = weft_ime.q4_K_scale_weighted_accum %sumi, %sc, %accS {accum_model = "scale_weighted_sum"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      %nextM = weft_ime.q4_K_min_bias_accum %a, %m, %accM {bias_model = "activation_sum_min_bias"} : (vector<32xi8>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
      weft_ime.q4_K_matmul_tile_yield %nextS, %nextM : vector<16xi32>, vector<16xi32>
    }
  }
}
