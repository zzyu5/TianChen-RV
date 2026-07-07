// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml tq2_0 x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) hot kernel -- the tq2_0
// prefill sibling of tcrv_rvv.repack_gemv_tq2_0_q8_K -- as STRUCTURED emitc IR (I5;
// ZERO raw() strings). The single typed op tcrv_rvv.repack_gemm_tq2_0_q8_K lowers to
// the BLOCK-AS-LANE multi-output-column matmul: the 16 interleaved weight columns of
// a group occupy 16 vector lanes (LANE-WISE vwmacc, NO cross-lane vredsum), and the
// activation is the interleaved block_q8_Kx4 stream (4 activation ROWS per group,
// stride 1168). The 2-bit TERNARY weight decode (((byte>>shift)&3)-1, the vand 0x03
// peel + vsub_vx_i8 by 1) is AMORTIZED once per 16-weight group and REUSED across the
// 4 interleaved activation columns: qs@16 (element e of column c at qs[e*4+c]), d[4]@0
// are 4 fp32 scalars. LINEAR single fp16 super-block scale, NO min term.
// block_tq2_0x16 stride 1056.

module {
  tcrv.exec.kernel @ggml_repack_gemm_tq2_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_tq2_0_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_tq2_0_q8_K", scale_model = "superblock-d.fp16-single-scale-2bit-ternary-4col-nomin", qk = 256 : i64, weight_block_stride = 1056 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_tq2_0_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_tq2_0_q8_K_kernel_ggml_repack_gemm_tq2_0_q8_K(
// The block count nb = n / 256, row-group count nr/4 (%arg4), column-group nc/16 (%arg5).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop over nr/4; interleaved base vy + y*nb*1168.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "1168"
// The weight-column-GROUP loop; weight base vx + x*nb*1056.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "1056"
// The per-column-per-strip f32 accumulators.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The 4 per-column activation super-block deltas d_y_c are fp32 SCALARS.
// CHECK: call_opaque "*(const float *)"
// The SHARED per-strip fp16 d strip (unpacked ONCE per group, reused across columns).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// The per-column i32 accumulators + per-super-half i16 partials.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The tq2_0 2-bit TERNARY weight assembly SHARED across columns: vand 0x03,
// reinterpret to i8, then vsub_vx_i8 by 1 (the `-1` ternary bias).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The per-column integer dot against the interleaved q8_Kx4 quants (NO vredsum),
// folded into i32 via vwadd_wv.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The per-column end-of-block SINGLE-scale fold: vfmul, vfcvt, vfmacc (NO vfnmsac).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The per-column per-strip store s + (y*4+c)*bs + x*16 + h*8.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
