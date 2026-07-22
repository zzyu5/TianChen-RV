// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SIGN
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq2_s x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) -- the prefill sibling of the
// iq2_s GEVM. The 1024-entry GRID CODEBOOK gather + EXPLICIT-SIGN plane + DUAL per-sub-block
// ls scale + trailing 0.125 are IDENTICAL to the GEVM; the DISTINGUISHING fact is the
// interleaved block_q8_Kx4 activation (4 fp32 d at +0, int8 quants at +16 as pos*4+c,
// stride 1168). The grid + sign weight decode is AMORTIZED once per 16-weight group and
// REUSED across the 4 activation columns. block_iq2_sx16 stride 1824.

module {
  weft.exec.kernel @ggml_repack_gemm_iq2_s_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq2_s_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-grid-explicitsign-dualscale-4col-nomin-eighth", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "grid_sign_dualscale_eighth", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied dual-ls GRID GEMM core BRICK
          // (decode_model "iq2_s"): the 1024-grid + explicit signs256 decode AMORTIZED across
          // the 4 interleaved block_q8_Kx4 activation columns -> the columnsPerPass (4)
          // per-column i32 sumi. The typed emitter re-emits the whole byte-exact iq2_s GEMM
          // dual-ls body (PLAIN untiled) from this brick's identity; accumulators pass through.
          %sumi:4 = weft_rvv.repack_gemm_grid_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_grid_core", decode_model = "iq2_s", weight_quant_byte_offset = 288 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 1312 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemm_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq2_s_q8_K_kernel_ggml_repack_gemm_iq2_s_q8_K(
// The FIXED 1024-entry GRID + UNIVERSAL signs256 plane emitted ONCE.
// CHECK: verbatim "static const int64_t weft_iq2s_grid[1024] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t weft_iq2s_signs256[2048] = {1
// CHECK: literal "1824"
// CHECK: literal "1168"
// The 4 interleaved block_q8_Kx4 fp32 d; per-column int8 quant at pos*4+c.
// CHECK: call_opaque "*(const float *)"

// ===== The REAL per-lane GRID memory GATHER (u16 assembled index strip, NO vzext). ==
// GATHER: call_opaque "__riscv_vle16_v_u16m1"
// GATHER: call_opaque "__riscv_vsll_vx_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vmul_vv_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL EXPLICIT-SIGN-PLANE application (raw 8-bit sign byte vzext + gather). ==
// SIGN: literal "weft_iq2s_signs256"
// SIGN: call_opaque "__riscv_vzext_vf2_u16m1"
// SIGN: call_opaque "__riscv_vluxei16_v_i8mf2"
// SIGN: call_opaque "__riscv_vmul_vv_i8mf2"

// ===== The DUAL per-sub-block ls scale (ls1/ls2, two vsext + two vmacc per sub-block). ==
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"
// SCALE: call_opaque "__riscv_vsext_vf4_i32m2"
// SCALE: call_opaque "__riscv_vmacc_vv_i32m2"

// The end-of-block per-column fold + 0.125 store.
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// MEMORY gathers (vluxei16), NOT register vrgather; scale-ONLY (no min/vfnmsac),
// block-as-lane (no cross-lane vredsum wall).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
