// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMIN

// The ggml q6_K x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) hot kernel as STRUCTURED
// emitc IR (I5; ZERO raw() strings) -- the q6_K prefill sibling of the decode GEVM.
// The single typed op tcrv_rvv.repack_gemm_q6_K_q8_K lowers to the BLOCK-AS-LANE
// multi-output-column matmul nest: row-group (nr/4) x weight-column-group (nc/16) x
// contraction-block, over the interleaved block_q8_Kx4 activation (stride 1168, qs at
// +16). The 6-bit ql|qh weight decode (and its -32 offset-binary bias) is AMORTIZED
// once per 16-weight group and REUSED across the 4 interleaved activation columns --
// each column reads its own q8_Kx4 quant. Same q6_K facts as the GEVM: 6-bit two-plane
// weight (vand 0x03 / vsub 32), SIGNED int8 scales (vle8_v_i8 / vsext), SINGLE
// accumulator NO-min fold (NO bsums, NO vfnmsac). block_q6_Kx16 stride 3360.

module {
  tcrv.exec.kernel @ggml_repack_gemm_q6_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_q6_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q6-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q6_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q6_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_q6_K_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_q6_K_q8_K", scale_model = "superblock-d.fp16-signed8-scale-16-subblocks-6bit-4col-nomin", qk = 256 : i64, weight_block_stride = 3360 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 1312 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_q6_K_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The row-group count nr/4 (%arg4 = nr) and the column-group count nc/16 (%arg5 = nc).
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop over nr/4.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// Per-group interleaved q8_Kx4 base vy + y*nb*1168 (block_q8_Kx4 stride 1168).
// CHECK: literal "1168"
// The weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*3360 (block_q6_Kx16 stride 3360).
// CHECK: literal "3360"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// Per-column activation super-block delta d_y_c is an fp32 SCALAR (4 per q8_Kx4 block).
// CHECK: call_opaque "*(const float *)"
// The SINGLE per-column per-strip i32 accumulator seed (NO bsums accumulator).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The q6_K SIGNED int8 per-sub-block scale (SHARED across columns): vle8_v_i8 + vsext.
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The q6_K 6-bit two-plane weight assembly (SHARED across the 4 columns): ql/qh strip
// loads, vand 0x0F / vsrl 4 nibble, the TWO-bit qh mask vand 0x03, vsll 4, vor, then
// vsub 32 (the -32 offset-binary bias folded into each SIGNED weight lane).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The per-column interleaved q8_Kx4 quant read + lane-wise integer dot (NO vredsum),
// scale-weighted i32 promote (vwmacc_vv).
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block SINGLE fold per column: vfcvt the i32 sumi, vfmacc the ONE main
// term (NO vfnmsac min term), then the per-column per-strip vse32 store.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall.
// NOWALL-NOT: redsum

// q6_K is a SINGLE-accumulator no-min fold: NO min-term subtract (vfnmsac) and NO
// activation bsums read (int16_t) appear.
// NOMIN-NOT: vfnmsac
// NOMIN-NOT: *(const int16_t *)
