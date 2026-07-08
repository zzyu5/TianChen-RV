// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=SIGN
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq2_xxs x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) -- the prefill sibling of
// the iq2_xxs GEVM. The SAME real grid GATHER + sign-plane GATHER + per-sub-block ls
// scale (i32 accumulator, no min) + trailing 0.125 factor; the DISTINGUISHING fact is
// the interleaved block_q8_Kx4 activation (4 rows, 4 fp32 d at +0, interleaved int8
// quants at +16 as pos*4+c). The grid+sign weight decode is AMORTIZED once per
// 16-weight group and reused across the 4 activation columns. block_iq2_xxsx16 stride
// 1184; activation block_q8_Kx4 stride 1168.

module {
  tcrv.exec.kernel @ggml_repack_gemm_iq2_xxs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_iq2_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq2_xxs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq2_xxs_q8_K_kernel", status = "selected-lowering-boundary"} {
        %g = tcrv_rvv.repack_gemm_iq2_xxs_q8_K %vx, %vy, %s, %n, %nr, %nc, %bs, %vl {kind = "ggml_repack_gemm_iq2_xxs_q8_K", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-4col-nomin-eighth", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 160 : i64, weight_scale_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// CHECK-NOT: tcrv_rvv.repack_gemm_iq2_xxs_q8_K %
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_iq2_xxs_q8_K_kernel_ggml_repack_gemm_iq2_xxs_q8_K(
// The FIXED 256-entry GRID-of-8 table + the DERIVED signs64 +-1 plane (static decls).
// CHECK: verbatim "static const int64_t tcrv_iq2xxs_grid[256] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t tcrv_iq2xxs_signs64[1024] = {1, 1, 1
// The interleaved block_q8_Kx4 activation stride 1168; the weight stride 1184.
// CHECK: literal "1184"
// CHECK: literal "1168"
// Per-column activation super-block d (fp32) reads.
// CHECK: call_opaque "*(const float *)"

// ===== The REAL per-lane GRID + SIGN memory GATHER (AMORTIZED across columns). =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: call_opaque "__riscv_vsll_vx_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vmul_vv_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL SIGN-PLANE application: signs64[sel*8+j] gathered, vmul onto grid. =
// SIGN: literal "tcrv_iq2xxs_signs64"
// SIGN: call_opaque "__riscv_vluxei16_v_i8mf2"
// SIGN: call_opaque "__riscv_vmul_vv_i8mf2"

// ===== The per-sub-block ls scale (int8 [1,31], widened to i32, vmacc-weighted). ==
// SCALE: call_opaque "__riscv_vle8_v_i8mf2"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"

// The per-column end-of-block fold + the iq2_xxs 0.125 (1/8) factor + vse32.
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// Grid + sign decode are MEMORY gathers (vluxei16), NOT a register vrgather. iq2_xxs
// is scale-ONLY: NO min term (no vfnmsac), NO cross-lane reduction wall (no vredsum).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
