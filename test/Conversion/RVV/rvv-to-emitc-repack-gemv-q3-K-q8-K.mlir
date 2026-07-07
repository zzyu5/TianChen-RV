// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMIN

// The ggml q3_K x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the LAST (and most
// intricate) K-quant repack sibling -- as STRUCTURED emitc IR (I5; ZERO raw()
// strings). The single typed op tcrv_rvv.repack_gemv_q3_K_q8_K lowers to the
// BLOCK-AS-LANE single-output-column matmul: the 16 interleaved weight columns of a
// group occupy 16 vector lanes, the dot accumulates LANE-WISE via vwmacc (NO cross-lane
// vredsum), and at VLEN=128 each 16-block group is processed as TWO disjoint 8-lane
// halves inline. The activation is ONE plain block_q8_K stream (stride 292, fp32 d at
// +0, int8 quants at +4; q3_K reads NO bsums). q3_K REUSES q6_K's no-min single-
// accumulator + signed-scale scaffold, with ONE delta: the 3-bit SUBTRACTIVE-HMASK
// weight VALUE is `((qs >> shift) & 3) - ((hmask & (1<<p)) ? 0 : 4)`, assembled from a
// 2-bit qs low plane (vand 0x03) | a SINGLE hmask high bit (vand 0x01 -- ONE bit, NOT
// q6_K's two-bit 0x03 qh) lifted to bit-2 (vsll 2), then the -4 SUBTRACTIVE bias
// (vsub 4 folds bit-CLEAR -> -4 straight into each SIGNED weight lane, the q3_K
// analogue of q6_K's -32). The 16 SIGNED 6-bit scales are PRE-UNPACKED + -32-biased at
// repack time, loaded vle8_v_i8 + SIGN-extended vsext. block_q3_Kx16 stride 1824, qs at
// +800, hmask at +288, signed scales at +32.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q3_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q3_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q3_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q3_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q3_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q3_K_q8_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_scales_byte_offset = 32 : i64, weight_hmask_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q3_K_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*1824 (block_q3_Kx16 stride 1824).
// CHECK: mul %[[X]], %{{.*}}
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
// The q3_K SIGNED 6-bit per-sub-block scale (pre-unpacked + -32-biased at repack):
// vle8_v_i8 + vsext_vf2_i16 (SIGN-extend, byte-identical to q6_K's scale side).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The i16 sub-block partial seed vmv_v_x_i16m1 (ONE 16-position partial, NO 2x8 k-chunk
// split -- q3_K's [-4,3] weight keeps 16*4*127 = 8128 < 32767).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The activation quant scalar read a.qs[k] (int8, offset +4).
// CHECK: call_opaque "*(const int8_t *)"
// The q3_K 3-bit SUBTRACTIVE two-plane weight assembly: the qs low-2-bit plane strip
// load (offset +800) then the hmask high-bit plane strip load (offset +288).
// CHECK: literal "800"
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "288"
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// low2 = (qs >> shift) & 3 (vand 0x03), then the SINGLE hmask bit ((hm >> p) & 1)
// (vand 0x01 -- ONE bit, NOT q6_K's two-bit 0x03 qh), vsll 2 lifts it to bit-2, vor
// merges onto the 2-bit qs -> a raw value in [0,7], then vsub 4 folds the -4
// SUBTRACTIVE bias (bit CLEAR -> -4) straight into each SIGNED weight lane.
// CHECK: literal "0x03"
// CHECK: literal "0x01"
// CHECK: call_opaque "__riscv_vsll_vx_u8mf2"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: literal "4"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The lane-wise integer dot against the single-column q8 quants (NO vredsum), each
// sub-block partial promoted to i32 weighted by the SIGNED scale (vwmacc_vv).
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
// accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum appears.
// NOWALL-NOT: redsum

// q3_K is a SINGLE-accumulator no-min fold: the -4 SUBTRACTIVE bias lives inside each
// weight lane, so NO min-term subtract (vfnmsac) and NO activation bsums read
// (int16_t) appear -- the q4_K/q5_K min-fold signature is ABSENT (as for q6_K).
// NOMIN-NOT: vfnmsac
// NOMIN-NOT: *(const int16_t *)
