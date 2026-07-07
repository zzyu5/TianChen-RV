// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml tq1_0 x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the BASE-3 TERNARY
// (BitNet-class, 1.6 bits/weight) block-as-lane sibling of
// tcrv_rvv.repack_gemv_tq2_0_q8_K -- as STRUCTURED emitc IR (I5; ZERO raw() strings).
// It REUSES tq2_0's single-scale no-min block-as-lane scaffold (LANE-WISE vwmacc, ONE
// i32 accumulator per strip, single fp16 super-block scale, NO dmin / bsums / min
// term) and differs ONLY in the WEIGHT DECODE: a BASE-3 unpack of the 48-byte qs
// plane (5 trits/byte) + the 4-byte qh plane (4 trits/byte). Each trit is recovered
// by ggml's exact pipeline: q = (uint8_t)(byte * pow3[l]) (vmul_vx_u8, the mandatory
// 8-bit wrap), xi = ((uint16_t)q * 3) >> 8 (vwmulu_vx_u16 * 3, vsrl_vx_u16 >> 8, in
// {0,1,2}), narrow (vncvt), reinterpret to i8, then xi - 1 (vadd_vx_i8 by -1, the
// trit in {-1,0,1}). block_tq1_0x16 stride 864, d at +0, qs at +32, qh at +800.

module {
  tcrv.exec.kernel @ggml_repack_gemv_tq1_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_tq1_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq1-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_tq1_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_tq1_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_tq1_0_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_tq1_0_q8_K", scale_model = "superblock-d.fp16-single-scale-base3-ternary-nomin", qk = 256 : i64, weight_block_stride = 864 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 32 : i64, weight_qh_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_tq1_0_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_tq1_0_q8_K_kernel_ggml_repack_gemv_tq1_0_q8_K(
// The block count nb = n / 256 and the column-group count nc/16 (%arg4 = nc).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop; per-group weight base vx + x*nb*864.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "864"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb; plain q8_K activation stride 292.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "292"
// The activation super-block delta d_y is a single fp32 SCALAR.
// CHECK: call_opaque "*(const float *)"
// The per-strip i32 accumulator + per-region i16 partial seeds.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The tq1_0 BASE-3 TERNARY weight decode: vle8 the qs strip, then q = byte*pow3[l]
// (vmul_vx_u8), xi = (q*3)>>8 (vwmulu_vx_u16 * 3, vsrl_vx_u16 >> 8), narrow (vncvt),
// reinterpret to i8, then xi - 1 (vadd_vx_i8 by -1) -- the trit in {-1,0,1}.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vmul_vx_u8mf2"
// CHECK: call_opaque "__riscv_vwmulu_vx_u16m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u16m1"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: literal "-1"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2"
// The lane-wise integer dot (NO vredsum) + the region fold into i32 (vwadd_wv).
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The qh base-3 high plane at repacked byte offset +800.
// CHECK: literal "800"
// The end-of-block SINGLE-scale fold: vle16 fp16 d, vfwcvt, vfmul, vfcvt, vfmacc
// (NO vfnmsac MIN term).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the reduction wall; the LINEAR fold has no min term.
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
