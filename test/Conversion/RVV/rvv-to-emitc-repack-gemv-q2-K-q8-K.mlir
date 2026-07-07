// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml q2_K x q8_K 16x1-REPACKED GEMV (decode) hot kernel -- the LOWEST-bit
// K-quant (super-block) block-as-lane sibling of the oracle-verified q4_K/q5_K
// repacked GEMVs -- as STRUCTURED emitc IR (I5; ZERO raw() strings). The single
// typed op tcrv_rvv.repack_gemv_q2_K_q8_K lowers to the BLOCK-AS-LANE single-output-
// column matmul: the 16 interleaved weight columns of a group occupy 16 vector
// lanes, the dot accumulates LANE-WISE via vwmacc (NO cross-lane vredsum wall), and
// at VLEN=128 each 16-block group is processed as TWO disjoint 8-lane halves. The
// activation is ONE plain block_q8_K stream (stride 292, fp32 d at +0, int8 quants
// at +4, int16 bsums at +260). q2_K is the MIN-TERM regression -- it REUSES the
// q4_K/q5_K DUAL super-block d/dmin + bsums-min fold (NOT q6_K's single no-min
// accumulator) -- with THREE q2_K deltas: (1) the weight is 2-BIT, FOUR lanes per
// byte peeled by vsrl {0,2,4,6} + vand 0x03 (UNSIGNED [0,3], NO offset-binary bias);
// (2) the per-sub-block scale/min is a SINGLE 4-bit-packed byte (vand 0x0F scale /
// vsrl 4 min, NOT q4_K's 6-bit two-byte bit-dance); (3) 16 sub-blocks of 16, so the
// MIN term reads a SINGLE bsum per sub-block. block_q2_Kx16 stride 1344, qs at +320,
// dmin at +32, packed scale/min at +64.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q2_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q2_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q2_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q2_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q2_K_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q2_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit", qk = 256 : i64, weight_block_stride = 1344 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 320 : i64, activation_quant_byte_offset = 4 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, activation_bsums_byte_offset = 260 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q2_K_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*1344 (block_q2_Kx16 stride 1344).
// CHECK: mul %[[X]], %{{.*}}
// CHECK: literal "1344"
// The two 8-lane f32 accumulators (cols 0..7, 8..15): two vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block plain q8_K activation base al = a + l*292 (stride 292).
// CHECK: literal "292"
// The activation super-block delta d_y is a single fp32 SCALAR (NOT fp16).
// CHECK: call_opaque "*(const float *)"
// The per-strip super-block dmin fp16 strips widened to f32 and scaled by d_y
// (the MIN-term scale) -- vle16 dmin strip, vfwcvt, vfmul_vf.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// The per-strip i32 main + bsums accumulators seed vmv_v_x_i32m2(0, 8).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The q2_K per-sub-block 4-bit packed scale/min unpack, LANE-WISE across the strip:
// vle8 the SINGLE packed byte, vand 0x0F (the low-nibble 4-bit scale) / vsrl 4 (the
// high-nibble 4-bit min), then vzext to i16 / reinterpret. This is q2_K-specific: a
// SINGLE byte, NOT q4_K's 6-bit two-byte get_scale_min_k4 vand/vsll/vsrl/vor dance.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"
// The MIN term: read the SINGLE activation int16 bsum per 16-element sub-block and
// fold it LANE-WISE via vwmacc_vx (bsum * min strip -> i32), the q2_K bsums-min
// correction (each q2_K sub-block is EXACTLY one q8_K bsums group, no bs0+bs1 pair).
// CHECK: call_opaque "*(const int16_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The q2_K 2-bit weight assembly: the qs byte strip is loaded ONCE per column strip
// then peeled into FOUR 2-bit lanes -- vand 0x03 (shift 0), then vsrl {2,4,6} + vand
// 0x03 -- each reinterpreted to a SIGNED i8 lane (value-identity for 0..3). The
// 0x03 TWO-bit mask distinguishes q2_K from q4_K's 0x0F nibble / q6_K's ql|qh.
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// The lane-wise integer dot against the single-column q8 quants (NO vredsum), each
// 16-element sub-block partial promoted to i32 weighted by the 4-bit scale.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block fold: vfcvt the i32 sumi, vfmacc the main d term, then vfnmsac
// the MIN term (sumf -= dmins_d * cvt(bsums)) -- the q4_K/q5_K dual d/dmin signature.
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
