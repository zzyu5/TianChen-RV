// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s

// The ggml q4_1 16x1-REPACKED GEMM (prefill) hot kernel as STRUCTURED emitc IR
// (I5; ZERO raw() strings). The single typed op tcrv_rvv.repack_gemm_q4_1_q8_1
// lowers to the FAMILY-B (scale+MIN, asymmetric) BLOCK-AS-LANE matmul: the 16
// interleaved weight rows of a group occupy 16 vector lanes, the dot accumulates
// LANE-WISE via vwmacc (NO cross-lane vredsum wall), and at VLEN=128 each 16-block
// group is processed as TWO disjoint contiguous 8-lane halves (rows 0..7 then
// 8..15). Two q4_1 specifics distinguish it from the q4_0 repacked GEMM:
//   (a) UNSIGNED nibble decode (vand 0x0F / vsrl 0x04 -> vreinterpret to i8), NO
//       offset-binary vsll/vsra sign-extend (the repacked nibbles are stored RAW);
//   (b) the per-column fold carries the per-block MIN correction LANE-WISE:
//       sumf_c += (d_x*d_y_c)*sumi_c (vfmacc) + m_x*s_y_c (vfwmul then vfadd).
//
// Every emitted value is a NODE in the IR graph -- no emitc.verbatim with C
// control flow, no raw string blob.

module {
  tcrv.exec.kernel @ggml_repack_gemm_q4_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_q4_1_q8_1 %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q4_1_q8_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", qk = 32 : i64, weight_block_stride = 320 : i64, activation_block_stride = 144 : i64, weight_quant_byte_offset = 64 : i64, activation_quant_byte_offset = 16 : i64, weight_min_byte_offset = 32 : i64, activation_sum_byte_offset = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_q4_1_q8_1 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_q4_1_q8_1_kernel_ggml_repack_gemm_q4_1_q8_1(
// The block count nb = n / 32.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The row-group count nr/4 (%arg4 = nr) and column-group count nc/16 (%arg5 = nc).
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop over nr/4.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// Per-group activation base vy + y*nb*144 (block_q8_1x4 stride 144).
// CHECK: mul %[[Y]], %{{.*}}
// CHECK: literal "144"
// The weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*320 (block_q4_1x16 stride 320).
// CHECK: literal "320"
// The two-halves loop over the 8-lane halves (rows 0..7, 8..15).
// CHECK: for %[[H:.*]] = %{{.*}} to %{{.*}} step
// The 4x8 f32 accumulator set: four vfmv_v_f_f32m2(0.0f, 8) seeds.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The 4x{lo,hi} i16 lane accumulator seeds vmv_v_x_i16m1(0, 8).
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The nibble-step loop over the 16 weight bytes.
// CHECK: for %[[I:.*]] = %{{.*}} to %{{.*}} step
// FAMILY-B UNSIGNED nibble decode: vle8_u8 then vand 0x0F (lo) / vsrl 0x04 (hi)
// then vreinterpret to i8 -- NO vsll/vsra sign-extend, NO vxor.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// The lane-wise vwmacc accumulate against the scalar q8 quants (NO cross-lane
// vredsum) -- the low-half then high-half products per activation column.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// The lo/hi combine vwadd_vv_i32m2.
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// The per-column scale+MIN fold: vle16 the 8 weight d AND m strips, vfwmul by the
// raw _Float16 activation d_y_c (scale) and s_y_c (min), vfcvt the i32 acc,
// vfmacc the scale term, vfadd the MIN term.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "*(const _Float16 *)"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// The 4x8 vector store vse32_v_f32m2 (NO per-block scalar *s store).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
