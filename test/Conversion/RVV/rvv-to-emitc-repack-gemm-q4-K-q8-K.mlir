// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml q4_K x q8_K 16x1-REPACKED PREFILL GEMM hot kernel -- the K-QUANT
// (super-block) block-as-lane sibling of the validated q4_1 repacked GEMM and
// the PREFILL sibling of the oracle-verified q4_K repacked GEVM -- as STRUCTURED
// emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.repack_gemm_q4_K_q8_K lowers to the BLOCK-AS-LANE multi-output-column
// matmul: the 16 interleaved weight columns of a group occupy 16 vector lanes,
// the dot accumulates LANE-WISE via vwmacc (NO cross-lane vredsum wall), and at
// VLEN=128 each 16-block group is processed as TWO disjoint contiguous 8-lane
// halves (cols 0..7 then 8..15). The activation is the INTERLEAVED block_q8_Kx4
// stream (stride 1168, 4 fp32 d at +0, 4-column-interleaved int8 quants at +16,
// 64 int16 bsums at +1040). The K-quant super-block (QK_K=256, 8 sub-blocks of
// 32) carries the SAME three structural extensions as the q4_K GEVM, with the
// 6-bit scale/min unpack + nibble decode amortized ONCE per 16-weight group and
// REUSED across the 4 interleaved activation columns (the prefill e2e-win point):
//   (1) per-sub-block 6-bit scale/min unpacked LANE-WISE (vand 0x0F / vsrl / vsll
//       bit dance, vzext to i16), SHARED across columns;
//   (2) the MIN correction folds the per-COLUMN int16 bsums LANE-WISE via
//       vwmacc_vx weighted by the 6-bit mins (bsums group16-major/column-minor:
//       a.bsums[gsub*8 + m] + a.bsums[gsub*8 + m + 4]);
//   (3) the 32-element sub-block dot split into 2x16 i16 chunks (i16 overflow
//       guard) promoted to i32 weighted by the 6-bit scale (vwmacc_vv).
// The super-block (d, dmin) are fp16 VECTOR strips; the activation d is 4 fp32
// SCALARS (one per interleaved column).
//
// NUMERIC STATUS: this lit checks the LOWERED STRUCTURE only (8-sub-block unpack,
// multi-column fold, NO vredsum); it does NOT prove numeric correctness. The
// byte-exact rvv oracle vs ggml_gemm_q4_K_16x1_q8_K is a deferred follow-up.

module {
  tcrv.exec.kernel @ggml_repack_gemm_q4_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_q4_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q4_K_q8_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-4col", qk = 256 : i64, weight_block_stride = 2304 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 256 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_q4_K_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The row-group count nr/4 (%arg4 = nr) and column-group count nc/16 (%arg5 = nc).
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop over nr/4.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// Per-group activation base vy + y*nb*1168 (block_q8_Kx4 stride 1168).
// CHECK: literal "1168"
// The weight-column-GROUP loop over nc/16; per-group weight base vx + x*nb*2304.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "2304"
// The per-column f32 accumulators: vfmv_v_f_f32m2(0.0f, 8) (4 cols x 2 strips).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block interleaved q8_Kx4 activation base al = a + l*1168.
// CHECK: literal "1168"
// The per-COLUMN activation super-block delta d_y_c is a fp32 SCALAR (NOT fp16),
// read at al + c*4.
// CHECK: call_opaque "*(const float *)"
// The SHARED per-strip super-block dmin/d fp16 strips widened to f32 ONCE per
// 16-weight group (vle16, vfwcvt), reused across the 4 columns.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// The per-column per-strip i32 main + bsums accumulators seed vmv_v_x_i32m2(0, 8).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The per-sub-block 6-bit scale/min unpack, LANE-WISE across the strip, SHARED
// across the columns: vle8 low+high bytes, vand 0x0F / vsrl 4, the j-dependent
// vand/vsll/vsrl high bits, vor, vzext to i16 / reinterpret.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsll_vx_u8mf2"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"
// The MIN term: read the per-COLUMN interleaved int16 bsums and fold them
// LANE-WISE via vwmacc_vx (bsum_pair * min strip -> i32).
// CHECK: call_opaque "*(const int16_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// The MAIN integer dot, multi-column: the SHARED weight nibble decode reused
// across columns, the 32-element sub-block split into 2x16 i16 chunks
// (vwmacc_vx_i16m1 against the per-column interleaved q8_Kx4 quants, NO cross-lane
// vredsum), each chunk promoted to i32 weighted by the 6-bit scale (vwmacc_vv).
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block per-column fold: vfmul_vf (d_x*d_y_c), vfcvt the i32 sumi,
// vfmacc the main term, then vfnmsac the MIN term (sumf -= dmin_x*d_y_c*bsums).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// The per-column per-strip vector store vse32_v_f32m2 to s + (y*4+c)*bs + x*16 + h*8.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the
// dot accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum appears.
// NOWALL-NOT: redsum
