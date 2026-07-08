// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=E8M0
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml mxfp4 x q8_0 16x1-REPACKED GEVM (decode) hot kernel -- the FP4-CODEBOOK +
// E8M0 (micro-exponent) block-as-lane sibling of the iq4_nl repacked GEVM -- as
// STRUCTURED emitc IR (I5; ZERO raw() strings). The single typed op
// tcrv_rvv.repack_gemv_mxfp4_q8_0 lowers to the BLOCK-AS-LANE single-output-column
// matmul: the 16 interleaved weight columns of a group occupy 16 vector lanes, and
// the 4-bit weight nibble is an INDEX into the 16-entry doubled-E2M1 int8 fp4
// codebook (kvalues_mxfp4), decoded by a REAL MEMORY codebook GATHER (vzext the
// nibble to a u16 byte offset, then vluxei16_v_i8 through the codebook array -- NOT a
// register vrgather, NOT a fake-linear value). The per-column weight scale is ONE
// E8M0 shared-exponent BYTE reconstructed to 2^(e-128) by ggml's EXACT bit
// construction (vzext_vf4 -> vsll/vmerge -> vreinterpret to f32, NOT an fp16 delta).
// The dot accumulates LANE-WISE in an i32 accumulator (no min). block_mxfp4x16
// stride 272, E8M0 e at +0, qs at +16; activation is one plain block_q8_0 (stride 34,
// qs at +2).

module {
  tcrv.exec.kernel @ggml_repack_gemv_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_mxfp4_q8_0 %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_mxfp4_q8_0", scale_model = "flat.e8m0-single-scale-codebook-nomin", qk = 32 : i64, weight_block_stride = 272 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 16 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_mxfp4_q8_0 %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_mxfp4_q8_0_kernel_ggml_repack_gemv_mxfp4_q8_0(
// The 16-entry doubled-E2M1 int8 fp4 codebook decl (the kvalues_mxfp4 table).
// CHECK: verbatim "static const int8_t tcrv_mxfp4_repack_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1
// The block count nb = n / 32 (QK_MXFP4) and the column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop; per-group weight base vx + x*nb*272.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "272"
// The two 8-lane f32 accumulators: vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop; per-block plain q8_0 base al = a + l*34.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "34"
// The per-strip i32 accumulator seed.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"

// ===== The REAL fp4-codebook GATHER (memory vluxei16, NOT register vrgather). =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// The low nibble index (vand 0x0F) is zero-extended to a u16 byte offset and
// gathered THROUGH the codebook table by vluxei16_v_i8 (kvalues_mxfp4[nibble]).
// GATHER: call_opaque "__riscv_vand_vx_u8mf2"
// GATHER: literal "0x0F"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "tcrv_mxfp4_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// The high nibble index (vsrl 0x04) is gathered the SAME way.
// GATHER: call_opaque "__riscv_vsrl_vx_u8mf2"
// GATHER: literal "0x04"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// The lane-wise i8xi8->i16 product widened into the i32 accumulator.
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL E8M0 shared-exponent scale reconstruction (2^(e-128)). =====
// The 16 E8M0 exponent bytes are loaded as a lane vector and widened u8 -> u32.
// E8M0: call_opaque "__riscv_vzext_vf4_u32m2"
// bits = (e < 2) ? (0x00200000u << (e & 0x1F)) : ((e - 1) << 23). Each scalar
// immediate is a SEPARATE literal emitted just before its consuming intrinsic.
// E8M0: literal "0x1F"
// E8M0: call_opaque "__riscv_vand_vx_u32m2"
// E8M0: literal "0x00200000"
// E8M0: call_opaque "__riscv_vmv_v_x_u32m2"
// E8M0: call_opaque "__riscv_vsll_vv_u32m2"
// E8M0: call_opaque "__riscv_vsub_vx_u32m2"
// E8M0: literal "23"
// E8M0: call_opaque "__riscv_vsll_vx_u32m2"
// E8M0: call_opaque "__riscv_vmsltu_vx_u32m2_b16"
// E8M0: call_opaque "__riscv_vmerge_vvm_u32m2"
// scale_x = *(float *)&bits, then dC = scale_x * (float)y.d (vfmul_vf, NOT vfwmul).
// E8M0: call_opaque "__riscv_vreinterpret_v_u32m2_f32m2"
// E8M0: call_opaque "__riscv_vfmul_vf_f32m2"

// The end-of-block fold: vfcvt the i32 sumi, then vfmacc (NO min term, NO dmin).
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The codebook decode is a MEMORY gather (vluxei16), NOT a register vrgather; the
// E8M0 scale is reconstructed by bit-arithmetic, NOT read as an fp16 delta (no
// vle16/vfwmul weight-scale load). The block-as-lane repack erases the cross-lane
// reduction wall; the flat codebook fold has NO min (no vfnmsac) and NO per-sub-block
// scale (no vwmacc_vv_i32).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
// NOWALL-NOT: vfwmul
// NOWALL-NOT: vle16
