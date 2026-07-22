// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMIN
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=RETIRED

// G3 主线A T3 format3: the ggml q3_K x q8_K 16x1-REPACKED GEVM (decode) hot kernel --
// the LAST (and most intricate) K-quant repack sibling -- is now CONSTRUCTED through the
// typed-region FRONT DOOR (the q4_0 / ternary / q4_K / q6_K typed_repack precedent), NOT
// the retired monolithic emitRepackGemvQ3KQ8K direct emitter. The
// weft_rvv.typed_repack_gemv_loop_body region (fold_model "kquant_single_scale_no_min",
// SHARED with q6_K) carries the weft_rvv.repack_gemv_kquant_core integer-core BRICK
// (decode_model "q3_K"), block_index-tied (anti-bypass) and named off the loop-body's own
// weight / activation ABI bases. The lowering GATES the emit on that brick's anti-bypass
// ties, then RE-EMITS the byte-exact q3_K GEVM body via emitTypedRepackGemvLoopBody's
// K-quant no-min branch -> emitRepackKQuantGemvBodyQ3K (byte-identical to the retired
// direct emitter). q3_K REUSES q6_K's no-min single-accumulator + signed-scale scaffold,
// with ONE delta: the 3-bit SUBTRACTIVE-HMASK weight VALUE is `((qs >> shift) & 3) -
// ((hmask & (1<<p)) ? 0 : 4)`, assembled NATIVE-MASK (REDESIGN-B KNEST): a 2-bit qs low
// plane (vand 0x03) reinterpreted to signed i8 is the base, and the SINGLE hmask high bit
// (ONE bit, NOT q6_K's two-bit 0x03 qh) is tested IN PLACE by vand(1<<p) + vmseq==0, with
// the -4 SUBTRACTIVE bias FUSED into ONE masked op vadd_vx_i8mf2_mu(mask0, base, base, -4)
// -- byte-exact to the retired vsll 2 | vor | vsub 4 expand chain, the q3_K analogue of
// q5_0/q5_1's native-mask qh decode. The 16 SIGNED 6-bit scales are
// PRE-UNPACKED + -32-biased at repack time, loaded vle8_v_i8 + SIGN-extended vsext.
// block_q3_Kx16 stride 1824, qs at +800, hmask at +288, signed scales at +32. VLEN=128 =>
// TWO disjoint 8-lane strips.

module {
  weft.exec.kernel @ggml_repack_gemv_q3_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q3_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_single_scale_no_min", main_term_form = "unrolled"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied q3_K integer-core BRICK: per-block lane-wise q3_K
          // dot -> the numHalves (2) per-strip i32 sumi. The typed emitter re-emits the
          // whole byte-exact q3_K body (3-bit subtractive qs|hmask assembly + signed scale
          // + single no-min fold) from this brick's identity; the yield passes accs through.
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q3_K", weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*1824 (block_q3_Kx16 stride 1824).
// CHECK: literal "1824"
// The two 8-lane f32 accumulators (cols 0..7, 8..15): two vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block plain q8_K activation base al = a + l*292 (stride 292).
// CHECK: literal "292"
// The activation super-block delta d_y is a single fp32 SCALAR (NOT fp16).
// CHECK: call_opaque "*(const float *)"
// The SINGLE per-strip i32 accumulator seed vmv_v_x_i32m2(0, 8) (NO bsums accumulator).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The q3_K SIGNED int8 per-sub-block scale: vle8_v_i8 + vsext_vf2_i16 (SIGN-extend, the
// same signed scale side as q6_K -- NOT q4_K/q5_K's vzext_vf2_u16 unsigned scale).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The q3_K 3-bit subtractive weight assembly (NATIVE-MASK KNEST): the qs plane strip
// load, the 2-bit low plane (vand 0x03) at shift {0,2,4,6} reinterpreted to signed i8,
// the hmask plane strip load, then the SINGLE hmask high bit is tested IN PLACE by
// vand(1<<p) + vmseq==0 (a per-lane bool, ONE bit -- NOT q6_K's two-bit 0x03), and the
// -4 SUBTRACTIVE bias is FUSED with the high-bit select into ONE masked op
// vadd_vx_i8mf2_mu(mask0, base, base, -4) -- byte-exact to the retired vsll 2 | vor |
// vsub 4 chain (be_q3k 0/8), the q3_K analogue of q5_0/q5_1's native-mask qh decode.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vmseq_vx_u8mf2_b16"
// CHECK: literal "-4"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2_mu"
// The lane-wise integer dot against the single-column q8 quants (NO vredsum), each
// sub-block partial promoted to i32 weighted by the SIGNED scale (vwmacc_vv).
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block SINGLE fold: vle16 the super-block d strip, vfwcvt + vfmul_vf the
// activation d, vfcvt the i32 sumi, vfmacc the ONE main term (NO vfnmsac min term).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2 (s+x*16+0, s+x*16+8).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the dot
// accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum appears. Also NO trailing
// dead result-token vmv (the region is result-less, unlike the retired direct emitter).
// NOWALL-NOT: redsum

// q3_K is a SINGLE-accumulator no-min fold: the -4 subtractive bias lives inside each
// weight lane, so NO min-term subtract (vfnmsac) and NO activation bsums read (int16_t)
// appear -- the q4_K/q5_K min-fold signature is ABSENT.
// NOMIN-NOT: vfnmsac
// NOMIN-NOT: *(const int16_t *)

// The OLD per-lane hmask expand chain (vsll<<2 | vor | vsub 4) is fully RETIRED by the
// native-mask KNEST recon -- none of those three ops survive anywhere in the q3_K GEVM.
// RETIRED-NOT: __riscv_vsll_vx_u8mf2
// RETIRED-NOT: __riscv_vor_vv_u8mf2
// RETIRED-NOT: __riscv_vsub_vx_i8mf2
