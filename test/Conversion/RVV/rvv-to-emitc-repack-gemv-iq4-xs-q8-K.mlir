// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq4_xs x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the SUPER-BLOCK
// CODEBOOK sibling of iq4_nl and the codebook analogue of the q4_K K-quant repack.
// iq4_xs is a QK_K=256 super-block of 8 sub-blocks, each carrying its OWN 6-bit
// SIGNED per-sub-block scale (biased -32, NO min). The 4-bit nibble indexes the SAME
// 16-entry non-linear int8 codebook (memory vluxei16 GATHER). The 6-bit scale is
// assembled LANE-WISE via the q4_K vand/vsrl/vsll/vor bit-dance from a scales_l pair
// byte + a split sh_lo/sh_hi high byte, then vsub 32 -> a signed i8 scale, sign-
// extended to i32 (vsext_vf4) and vmacc-weighted onto the i32 sub-block dot.
// block_iq4_xsx16 stride 2176; activation is one plain block_q8_K (stride 292).

module {
  tcrv.exec.kernel @ggml_repack_gemv_iq4_xs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_iq4_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_iq4_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_iq4_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemv_iq4_xs_q8_K %vx, %vy, %s, %n, %nc, %vl {kind = "ggml_repack_gemv_iq4_xs_q8_K", scale_model = "superblock-d.fp16-codebook-6bit-signed-scale-nomin", qk = 256 : i64, weight_block_stride = 2176 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 128 : i64, weight_scales_low_byte_offset = 64 : i64, weight_scales_high_byte_offset = 32 : i64, activation_quant_byte_offset = 4 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemv_iq4_xs_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_iq4_xs_q8_K_kernel_ggml_repack_gemv_iq4_xs_q8_K(
// CHECK: verbatim "static const int8_t tcrv_iq4_xs_repack_kvalues[16] = {-127, -104
// The block count nb = n/256 (QK_K), column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The per-super-block plain q8_K base al = a + l*292 (stride 292); d_y a fp32 scalar.
// CHECK: literal "2176"
// CHECK: literal "292"
// CHECK: call_opaque "*(const float *)"

// ===== The SIGNED 6-bit per-sub-block scale unpack (q4_K bit-dance + -32 bias). =
// SCALE: call_opaque "__riscv_vand_vx_u8mf2"
// SCALE: call_opaque "__riscv_vsrl_vx_u8mf2"
// SCALE: call_opaque "__riscv_vsll_vx_u8mf2"
// SCALE: call_opaque "__riscv_vor_vv_u8mf2"
// SCALE: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// The -32 K-quant bias -> SIGNED scale, sign-extended to i32 for the vmacc weight.
// SCALE: call_opaque "__riscv_vsub_vx_i8mf2"
// SCALE: literal "32"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"

// ===== The REAL codebook GATHER (memory vluxei16), then i32 sub-block dot. =====
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "tcrv_iq4_xs_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"
// The SIGNED scale weights the i32 sub-block dot into the block accumulator.
// GATHER: call_opaque "__riscv_vmacc_vv_i32m2"

// The end-of-block fold: vfwcvt the fp16 d strip, vfmul by fp32 d_y, vfcvt sumi,
// vfmacc (NO min term).
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Codebook decode is a MEMORY gather (vluxei16), NOT a register vrgather. iq4_xs is
// scale-ONLY: NO min term (no vfnmsac), NO activation bsums read, NO cross-lane
// reduction wall.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
