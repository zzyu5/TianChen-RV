// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml q5_K x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- q5_K == q4_K PLUS
// the qh 5th (high) weight bit -- as STRUCTURED emitc IR (I5; ZERO raw() strings).
// The single typed op tcrv_rvv.repack_gemv_q5_K_q8_K lowers to the BLOCK-AS-LANE
// single-output-column matmul: the 16 interleaved weight columns of a group occupy
// 16 vector lanes, the dot accumulates LANE-WISE via vwmacc (NO cross-lane vredsum
// wall), and at VLEN=128 each 16-block group is processed as TWO disjoint 8-lane
// halves (cols 0..7 then 8..15) inline. The activation is ONE plain block_q8_K
// stream (stride 292, fp32 d at +0, int8 quants at +4, int16 bsums at +260 --
// byte-identical to q4_K; the qh plane is a WEIGHT-side fact only). The K-quant
// super-block decode is IDENTICAL to q4_K (QK_K=256, 8 sub-blocks of 32, get_scale_
// min_k4 6-bit scale/min unpack, bsums-min fold, dual d/dmin fold), with ONE added
// step: each 4-bit nibble is lifted to a 5-bit value in [0,31] by OR-ing qh[i]'s
// per-sub-block high bit (vand 0x01 / vsll 4 / vor) loaded from the block_q5_Kx16
// qh region (stride 2816, qh at +256, qs at +768). This is K-quant scale+min, NOT
// the q5_0 offset-binary (no -16) form.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q5_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q5_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q5_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q5_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q5_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5", qk = 256 : i64, weight_block_stride = 2816 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 768 : i64, activation_quant_byte_offset = 4 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, weight_qh_byte_offset = 256 : i64, activation_bsums_byte_offset = 260 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q5_K_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*2816 (block_q5_Kx16 stride 2816 = q4_K 2304 + 512
// qh high-bit bytes).
// CHECK: mul %[[X]], %{{.*}}
// CHECK: literal "2816"
// The two 8-lane f32 accumulators (cols 0..7, 8..15): two vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block plain q8_K activation base al = a + l*292 (stride 292).
// CHECK: literal "292"
// The activation super-block delta d_y is a single fp32 SCALAR (NOT fp16).
// CHECK: call_opaque "*(const float *)"
// The per-strip super-block dmin fp16 strips widened to f32 and scaled by d_y.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// The per-strip i32 main + bsums accumulators seed vmv_v_x_i32m2(0, 8).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The per-sub-block 6-bit scale/min unpack, LANE-WISE (IDENTICAL to q4_K): vle8 the
// low byte + high byte, vand 0x0F / vsrl 4, the j-dependent vand/vsll/vsrl high
// bits, vor, then vzext to i16 / reinterpret.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_u8mf2"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"
// The MIN term: read the activation int16 bsums and fold LANE-WISE via vwmacc_vx.
// CHECK: call_opaque "*(const int16_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The MAIN integer dot WITH the q5_K qh 5th-bit inject (the sole q5_K-vs-q4_K
// delta at the core): the nibble strip load, then a SECOND vle8 loads the qh
// strip, vand 0x01 masks the selected per-sub-block bit (a mask q4_K NEVER emits),
// vsll 4 lifts it to bit 4, vor merges it onto the 4-bit nibble -> a 5-bit value in
// [0,31], then vwmacc_vx against the single-column q8 quants (NO vredsum), each
// chunk promoted to i32 weighted by the 6-bit scale (vwmacc_vv).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x01"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block fold: vle16 the super-block d strip, vfwcvt + vfmul_vf, vfcvt the
// i32 sumi, vfmacc the main term, then vfnmsac the MIN term.
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2 (s+x*16+0, s+x*16+8).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the
// dot accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum appears.
// NOWALL-NOT: redsum
