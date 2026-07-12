// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M2-后 iq4_xs codebook CONSTRUCTION -- the SECOND codebook decode family, the SUPER-BLOCK
// sibling of iq4_nl, COMPLETING the iq4 codebook pair. The abstract, algorithm-UNCOMMITTED
// weft_rvv.quant_contraction op (iq4_xs codebook / decode) is AUTO-LOWERED by
// --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection reads the
// STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived capability VLEN
// (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the committed codebook
// scale_model WHAT, so the C1 bridge lowerToRepackGemvCodebook CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (fold_model
// "codebook_superblock_signed6_no_min") carrying the SHARED weft_rvv.repack_gemv_codebook_core
// integer-core brick (decode_model "iq4_xs"), reconstructing the block_iq4_xsx16 x16 weight
// facts 2176/128 + the scales_l LOW pair @64 + scales_h HIGH 2-bit @32 + n_subblocks 8 + the
// plain block_q8_K activation facts 292/4, half_lanes=8 => mf2, numHalves==2, AND -- crucially
// -- RECONSTRUCTS the 16-entry non-linear int8 codebook (kvalues_iq4nl, SHARED with iq4_nl)
// the abstract request DOES NOT carry (a true CONSTRUCT-from-abstract).
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the codebook SUPER-BLOCK repack
// region from a REAL quant_contraction request. The constructed region lowers to the
// byte-exact iq4_xs GEVM kernel the retired monolithic emitRepackGemvIq4XsQ8K produced.
// NO perf/e2e claim -- lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_repack_gemv_iq4_xs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq4_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_iq4_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_iq4_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT codebook super-block request: PLAIN block_iq4_xs (stride 136, nibbles
        // @8) x PLAIN block_q8_K (stride 292), qk 256, scale_model = the codebook iq4_xs WHAT,
        // block_dot_compute_heavy = true (routes REPACK). It carries NO codebook and NO scales
        // offsets -- the compiler RECONSTRUCTS kvalues_iq4nl + the signed-6 scale facts. The
        // result is dead (the repacked lane-wise codebook GEVM sinks through the output).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq4_xs", scale_model = "superblock.fp16-signed6-scale-codebook-nomin", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 8 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the codebook SUPER-BLOCK typed_repack
// GEVM region (fold_model codebook_superblock_signed6_no_min) carrying the SHARED codebook
// core brick (decode_model iq4_xs), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track that.
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: fold_model = "codebook_superblock_signed6_no_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: n_subblocks = 8
// CONSTRUCT-SAME: scale_model = "superblock.fp16-signed6-scale-codebook-nomin"
// CONSTRUCT-SAME: weft_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 2176
// CONSTRUCT-SAME: weight_scales_byte_offset = 64
// CONSTRUCT-SAME: weight_scales_high_byte_offset = 32
// The in-region block_index-tied codebook integer-core BRICK, decode_model iq4_xs + the
// RECONSTRUCTED 16-entry codebook (the abstract request carried none).
// CONSTRUCT: weft_rvv.repack_gemv_codebook_core
// CONSTRUCT-SAME: codebook = array<i8: -127
// CONSTRUCT-SAME: decode_model = "iq4_xs"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the iq4_xs GEVM kernel (byte-exact to the retired direct
// emitter -- the SAME emit as rvv-to-emitc-repack-gemv-iq4-xs-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_codebook_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq4_xs_q8_K_kernel_ggml_repack_gemv_iq4_xs_q8_K(
// The 16-entry NON-LINEAR int8 codebook decl (RECONSTRUCTED by the compiler).
// CHECK: verbatim "static const int8_t weft_iq4_xs_repack_kvalues[16] = {-127, -104
// Per-group weight base vx + x*nb*2176 (block_iq4_xsx16 stride 2176); plain q8_K stride 292.
// CHECK: literal "2176"
// CHECK: literal "292"
// The activation super-block delta d_y is a fp32 scalar read.
// CHECK: call_opaque "*(const float *)"
// The SIGNED 6-bit scale bit-dance (q4_K vand/vsrl/vsll/vor) + -32 bias + vsext to i32.
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// The REAL memory codebook GATHER (vzext -> vluxei16), NOT a register vrgather.
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The i32 sub-block dot (vwmul + vwadd_wv) then the SIGNED-scale vmacc fold.
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The end-of-block fp16*fp32 fold (NO min) + the lane-wise vse32 store.
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The codebook decode is a MEMORY gather, NOT a register vrgather; iq4_xs is scale-ONLY
// (NO min: no vfnmsac), and the block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
