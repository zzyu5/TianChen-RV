// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq4_xs x q8_K 16x1-REPACKED GEMM (prefill) hot kernel -- the SUPER-BLOCK
// CODEBOOK prefill sibling of tcrv_rvv.repack_gemv_iq4_xs_q8_K. The codebook memory
// GATHER (vluxei16) + 6-bit SIGNED per-sub-block scale (vsub 32, vsext_vf4, vmacc,
// NO min) are AMORTIZED once per 16-weight group and reused across the 4 interleaved
// block_q8_Kx4 activation columns (stride 1168, qs at +16, bsums UNREAD).
// block_iq4_xsx16 stride 2176.

module {
  tcrv.exec.kernel @ggml_repack_gemm_iq4_xs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_iq4_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq4_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq4_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_iq4_xs_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_iq4_xs_q8_K", scale_model = "superblock-d.fp16-codebook-6bit-signed-scale-4col-nomin", qk = 256 : i64, weight_block_stride = 2176 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 128 : i64, weight_scales_low_byte_offset = 64 : i64, weight_scales_high_byte_offset = 32 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_iq4_xs_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_iq4_xs_q8_K_kernel_ggml_repack_gemm_iq4_xs_q8_K(
// CHECK: verbatim "static const int8_t tcrv_iq4_xs_repack_kvalues[16] = {-127, -104
// The block count nb, row-group nr/4, column-group nc/16.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The interleaved q8_Kx4 base (a + y*nb*1168, stride 1168); per-column fp32 d_y_c.
// CHECK: literal "1168"
// CHECK: call_opaque "*(const float *)"

// ===== The SHARED signed-scale unpack + memory codebook GATHER (vluxei16). =====
// GATHER: call_opaque "__riscv_vsub_vx_i8mf2"
// GATHER: call_opaque "__riscv_vsext_vf4_i32m2"
// GATHER: literal "tcrv_iq4_xs_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"
// GATHER: call_opaque "__riscv_vmacc_vv_i32m2"

// The per-column fold + store through the runtime output row stride bs.
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
