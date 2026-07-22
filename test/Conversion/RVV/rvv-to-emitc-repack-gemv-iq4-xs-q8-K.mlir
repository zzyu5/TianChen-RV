// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M2-后 iq4_xs codebook front-door: the ggml iq4_xs x q8_K 16x1-REPACKED GEVM (decode)
// hot kernel -- the SECOND CODEBOOK decode family, the SUPER-BLOCK sibling of iq4_nl -- is
// now CONSTRUCTED through the typed-region FRONT DOOR (the q4_0 / ternary / K-quant / iq4_nl
// typed_repack precedent), NOT the retired monolithic emitRepackGemvIq4XsQ8K direct emitter.
// The weft_rvv.typed_repack_gemv_loop_body region (fold_model
// "codebook_superblock_signed6_no_min") carries the SHARED weft_rvv.repack_gemv_codebook_core
// integer-core BRICK (decode_model "iq4_xs" + the SAME 16-entry non-linear int8 codebook
// iq4_nl uses), block_index-tied (anti-bypass) and named off the loop-body's own weight /
// activation ABI bases. The lowering GATES the emit on that brick's anti-bypass ties, then
// RE-EMITS the byte-exact iq4_xs GEVM body via emitTypedRepackGemvLoopBody's codebook
// super-block branch -> emitRepackCodebookGemvBodyIq4Xs (byte-identical to the retired direct
// emitter). The 4-bit nibble is an INDEX into the codebook (REAL memory vluxei16 GATHER); the
// K-quant 6-bit SIGNED per-sub-block scale is assembled LANE-WISE via the q4_K
// vand/vsrl/vsll/vor bit-dance from the scales_l LOW pair (@64) + the scales_h HIGH 2-bit
// (@32) regions, biased -32, sign-extended (vsext_vf4) and vmacc-weighted onto the i32
// sub-block dot; single fp16 super-block scale, NO min. block_iq4_xsx16 stride 2176;
// activation is one plain block_q8_K (stride 292, qs at +4). VLEN=128 => TWO strips.

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
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock.fp16-signed6-scale-codebook-nomin", qk = 256 : i64, weight_block_stride = 2176 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 64 : i64, weight_scales_high_byte_offset = 32 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "codebook_superblock_signed6_no_min"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied CODEBOOK integer-core BRICK (SHARED with iq4_nl): per-block
          // lane-wise iq4_xs memory-gather dot -> the numHalves (2) per-strip i32 sumi. The
          // typed emitter re-emits the whole byte-exact iq4_xs body (codebook gather + i32 dot
          // + 6-bit signed sub-block-scale vmacc fold + single fp16 fold) from this brick's
          // identity + its 16-entry codebook + the loop body's scales_l/scales_h/n_subblocks;
          // the yield passes through the carried-in accs.
          %sumi:2 = weft_rvv.repack_gemv_codebook_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_codebook_core", decode_model = "iq4_xs", weight_quant_byte_offset = 128 : i64, activation_quant_byte_offset = 4 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_codebook_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq4_xs_q8_K_kernel_ggml_repack_gemv_iq4_xs_q8_K(
// The 16-entry NON-LINEAR int8 codebook decl (the SAME kvalues_iq4nl table).
// CHECK: verbatim "static const int8_t weft_iq4_xs_repack_kvalues[16] = {-127, -104
// The block count nb = n/256 (QK_K), column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The per-super-block weight base vx + x*nb*2176; plain q8_K base al = a + l*292; d_y a
// fp32 scalar.
// CHECK: literal "2176"
// CHECK: literal "292"
// CHECK: call_opaque "*(const float *)"

// ===== The SIGNED 6-bit per-sub-block scale unpack (q4_K bit-dance + -32 bias). =
// SCALE: call_opaque "__riscv_vand_vx_u8mf2"
// SCALE: call_opaque "__riscv_vsrl_vx_u8mf2"
// SCALE: call_opaque "__riscv_vsll_vx_u8mf2"
// SCALE: call_opaque "__riscv_vor_vv_u8mf2"
// SCALE: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// The -32 K-quant bias -> SIGNED scale, sign-extended to i32 for the vmacc weight.
// SCALE: call_opaque "__riscv_vsub_vx_i8mf2"
// SCALE: literal "32"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"

// ===== The REAL codebook GATHER (memory vluxei16), then i32 sub-block dot. =====
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "weft_iq4_xs_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"
// The SIGNED scale weights the i32 sub-block dot into the block accumulator.
// GATHER: call_opaque "__riscv_vmacc_vv_i32m2"

// The end-of-block fold: vfwcvt the fp16 d strip, vfmul by fp32 d_y, vfcvt sumi,
// vfmacc (NO min term).
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Codebook decode is a MEMORY gather (vluxei16), NOT a register vrgather. iq4_xs is
// scale-ONLY: NO min term (no vfnmsac), NO activation bsums read, NO cross-lane
// reduction wall. Also NO trailing dead result-token vmv (the region is result-less,
// unlike the retired direct emitter).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
