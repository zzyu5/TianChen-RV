// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOBIAS

// The ggml q5_1 16x1-REPACKED GEMV (decode) hot kernel as STRUCTURED emitc IR
// (I5; ZERO raw() strings). q5_1 is the UNION of two proven repacked-GEVM
// siblings: the FIVE-BIT transposed-qh weight decode of the q5_0 repacked GEVM
// fused with the FAMILY-B (scale+MIN, asymmetric) dual fold of the q4_1 repacked
// GEVM. The single typed op tcrv_rvv.repack_gemv_q5_1_q8_1 lowers to the
// BLOCK-AS-LANE single-output-column matmul: the 16 interleaved weight rows of a
// group occupy 16 vector lanes, the dot accumulates LANE-WISE via vwmacc (NO
// cross-lane vredsum wall), and at VLEN=128 each 16-block group is processed as
// TWO disjoint contiguous 8-lane halves. The activation is ONE plain block_q8_1
// stream (stride 36, quants at +4, scaled-sum s at +2). The block_q5_1x16 repack
// carries 16 fp16 d @0, 16 fp16 m @32, RAW nibbles at +64, and a 64-byte
// TRANSPOSED bit-packed qh region at +320: per nibble step the lane decode reads
// a 16-bit qh mask (one per element), expands it lane-wise
// (vid/vsrl/vand/vsll/vncvt), ORs the {0,16} term into the unsigned nibble, then
// reinterprets u8->i8 -- WITHOUT the q5_0 `-16` bias (q5_1 is asymmetric; the
// bias lives in the MIN scale). The fold carries the per-block MIN correction
// LANE-WISE (sumf += (d_x*d_y)*sumi + m_x*s_y). Byte-exact vs canonical ggml
// q5_1 by construction.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q5_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q5_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q5_1_q8_1, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q5_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_q5_1_q8_1 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_q5_1_q8_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", qk = 32 : i64, weight_block_stride = 384 : i64, activation_block_stride = 36 : i64, weight_quant_byte_offset = 64 : i64, weight_qh_byte_offset = 320 : i64, activation_quant_byte_offset = 4 : i64, weight_min_byte_offset = 32 : i64, activation_sum_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_q5_1_q8_1 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q5_1_q8_1_kernel_ggml_repack_gemv_q5_1_q8_1(
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*384 (block_q5_1x16 stride 384).
// CHECK: literal "384"
// The two 8-lane f32 accumulators (rows 0..7, 8..15).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb; plain q8_1 activation base (stride 36).
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "36"
// The nibble-step loop over the 16 weight bytes.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// Two disjoint repacked UNSIGNED nibble sub-loads (rows 0..7, 8..15).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// The TRANSPOSED bit-packed qh: two 16-bit mask scalar reads (low element i, high
// element i+16), each expanded lane-wise to the {0,16} 5th-bit term.
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// CHECK: call_opaque "(uint16_t)*(const uint16_t *)"
// The lane-wise qh expansion: vid/vsrl/vand/vsll over u16, narrow u16->u8.
// CHECK: call_opaque "__riscv_vid_v_u16m1"
// CHECK: call_opaque "__riscv_vsrl_vv_u16m1"
// CHECK: call_opaque "__riscv_vand_vx_u16m1"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf2"
// The 5-bit assemble: OR the {0,16} term into the unsigned nibble, reinterpret
// u8->i8. UNLIKE q5_0 there is NO `-16` bias subtract (q5_1 is asymmetric).
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// The lane-wise vwmacc accumulate against the SINGLE-column scalar q8 quants (NO
// cross-lane vredsum) -- low qs[i] and high qs[16+i], for both 8-lane halves.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// The lo/hi combine vwadd_vv_i32m2 for both halves.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// The scale fold: vle16 the two 8-lane delta d strips AND the two 8-lane MIN m
// strips; read the SINGLE raw _Float16 activation delta d_y once and scaled-sum
// s_y once. Per half the scale term (vfwmul d_x*d_y / vfcvt / vfmacc) THEN the
// LANE-WISE MIN term (vfwmul m_x*s_y / vfadd) -- the Family-B distinction.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// The single-column two-half store vse32_v_f32m2.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: NO
// vredsum / vwredsum (any SEW or signedness) appears anywhere in the emitted GEVM.
// NOWALL-NOT: redsum

// q5_1 is asymmetric: the 5-bit weight stays UNSIGNED [0,31] (bias in the MIN
// scale), so the q5_0 offset-binary `-16` reconstruct MUST NOT appear.
// NOBIAS-NOT: vsub
