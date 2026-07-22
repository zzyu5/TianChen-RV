// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMIN

// G3 主线A T3: the ggml q6_K x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the
// LARGEST K-quant structural delta -- is now CONSTRUCTED through the typed-region
// FRONT DOOR (the q4_0 / ternary / q4_K typed_repack precedent), NOT the retired
// monolithic emitRepackGemvQ6KQ8K direct emitter. The weft_rvv.typed_repack_gemv_loop_body
// region (fold_model "kquant_single_scale_no_min") carries the
// weft_rvv.repack_gemv_kquant_core integer-core BRICK (decode_model "q6_K"),
// block_index-tied (anti-bypass) and named off the loop-body's own weight /
// activation ABI bases. The lowering GATES the emit on that brick's anti-bypass ties,
// then RE-EMITS the byte-exact q6_K GEVM body via emitTypedRepackGemvLoopBody's K-quant
// no-min branch -> emitRepackKQuantGemvBodyQ6K (byte-identical to the retired direct
// emitter). Three q6_K-specific facts vs q4_K/q5_K: (1) the 6-bit weight VALUE is
// assembled from a low-4-bit ql plane + a high-2-bit qh plane -- ((ql & 0xF) |
// (((qh >> shift) & 3) << 4)) - 32, SIGNED in [-32,31] (vand 0x03 masks TWO bits;
// vsub 32 folds the -32 offset-binary bias into each weight lane); (2) 16 SIGNED int8
// per-sub-block scales (vle8_v_i8 + vsext, NOT q4_K's vzext); (3) a SINGLE accumulator,
// NO per-sub-block min term (NO bsums, NO vfnmsac). block_q6_Kx16 stride 3360, ql at
// +1312, qh at +288, signed scales at +32. VLEN=128 => TWO disjoint 8-lane strips.

module {
  weft.exec.kernel @ggml_repack_gemv_q6_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q6_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q6-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin", qk = 256 : i64, weight_block_stride = 3360 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 1312 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "kquant_single_scale_no_min", main_term_form = "unrolled"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied q6_K integer-core BRICK: per-block lane-wise q6_K
          // dot -> the numHalves (2) per-strip i32 sumi. The typed emitter re-emits the
          // whole byte-exact q6_K body (6-bit ql|qh assembly + signed scale + single
          // no-min fold) from this brick's identity; the yield passes the accs through.
          %sumi:2 = weft_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_kquant_core", decode_model = "q6_K", weight_quant_byte_offset = 1312 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*3360 (block_q6_Kx16 stride 3360).
// CHECK: literal "3360"
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
// The q6_K SIGNED int8 per-sub-block scale: vle8_v_i8 + vsext_vf2_i16 (SIGN-extend, the
// sole scale-side q6_K delta vs q4_K/q5_K's vzext_vf2_u16 unsigned scale).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The q6_K 6-bit two-plane weight assembly: the ql plane strip load, the low nibble
// (vand 0x0F) / high nibble (vsrl 4), the qh plane strip load, the TWO-bit qh mask
// (vand 0x03 -- q5_K masks a SINGLE bit 0x01), vsll 4 lifts it to bits 4-5, vor merges
// onto the nibble -> a raw value in [0,63], then vsub 32 folds the -32 offset-binary
// bias straight into each SIGNED weight lane (the sole q6_K min replacement).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
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

// q6_K is a SINGLE-accumulator no-min fold: the -32 offset-binary bias lives inside
// each weight lane, so NO min-term subtract (vfnmsac) and NO activation bsums read
// (int16_t) appear -- the q4_K/q5_K min-fold signature is ABSENT.
// NOMIN-NOT: vfnmsac
// NOMIN-NOT: *(const int16_t *)
