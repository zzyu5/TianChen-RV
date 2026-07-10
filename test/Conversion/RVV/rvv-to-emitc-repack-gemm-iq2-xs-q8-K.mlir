// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=SIGN
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// The ggml iq2_xs x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) -- the prefill sibling of the
// iq2_xs GEVM. The 512-entry GRID CODEBOOK gather + ksigns SIGN plane + DUAL per-sub-block
// ls scale + trailing 0.125 are IDENTICAL to the GEVM; the DISTINGUISHING fact is the
// interleaved block_q8_Kx4 activation (4 fp32 d at +0, int8 quants at +16 as pos*4+c,
// stride 1168). The grid + sign weight decode is AMORTIZED once per 16-weight group and
// REUSED across the 4 activation columns. block_iq2_xsx16 stride 1824.

module {
  tcrv.exec.kernel @ggml_repack_gemm_iq2_xs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_iq2_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq2_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq2_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-grid-sign-dualscale-4col-nomin-eighth", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 288 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "grid_sign_dualscale_eighth"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied dual-ls GRID GEMM core BRICK
          // (decode_model "iq2_xs"): the grid + signs64 decode AMORTIZED across the 4
          // interleaved block_q8_Kx4 activation columns -> the columnsPerPass (4) per-column
          // i32 sumi. The typed emitter re-emits the whole byte-exact iq2_xs GEMM dual-ls
          // body (PLAIN untiled) from this brick's identity; the accumulators pass through.
          %sumi:4 = tcrv_rvv.repack_gemm_grid_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_grid_core", decode_model = "iq2_xs", weight_quant_byte_offset = 288 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 1312 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: tcrv_rvv.repack_gemm_grid_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_iq2_xs_q8_K_kernel_ggml_repack_gemm_iq2_xs_q8_K(
// The FIXED 512-entry GRID + DERIVED signs64 plane emitted ONCE.
// CHECK: verbatim "static const int64_t tcrv_iq2xs_grid[512] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t tcrv_iq2xs_signs64[1024] = {1
// block count nb = n/256, row-group count nr/4, column-group count nc/16.
// CHECK: literal "1824"
// CHECK: literal "1168"
// The 4 interleaved block_q8_Kx4 fp32 d (at 0,4,8,12); per-column int8 quant at pos*4+c.
// CHECK: call_opaque "*(const float *)"

// ===== The REAL per-lane GRID + SIGN memory GATHER (u16 grid index strip, NO vzext). ==
// GATHER: call_opaque "__riscv_vle16_v_u16m1"
// GATHER: call_opaque "__riscv_vsll_vx_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// GATHER: call_opaque "__riscv_vmul_vv_i8mf2"
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// ===== The REAL ksigns SIGN-PLANE application (7-bit selector vzext + gather). ==
// SIGN: literal "tcrv_iq2xs_signs64"
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
