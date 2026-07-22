// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SIGN
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=SCALE
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M4 iq2-grid front-door (first cell iq2_xxs): the ggml iq2_xxs x q8_K 16x1-REPACKED
// PREFILL GEMM (M>>1) -- the prefill sibling of the iq2_xxs GEVM -- is now CONSTRUCTED
// through the typed-region FRONT DOOR, NOT the retired monolithic emitRepackGemmIq2XxsQ8K
// direct emitter. The weft_rvv.typed_repack_gemm_loop_body region (fold_model
// "grid_sign_single_scale_eighth") carries the NEW weft_rvv.repack_gemm_grid_core BRICK
// (decode_model "iq2_xxs"), block_index + strip_row_offset tied (double anti-bypass) and
// named off the loop-body's own ABI bases. The lowering RE-EMITS the byte-exact iq2_xxs
// GEMM body via emitTypedRepackGemmLoopBody's grid branch -> emitRepackGridGemmBodyIq2Xxs
// (byte-identical to the retired direct emitter). The SAME real grid GATHER + sign-plane
// GATHER + per-sub-block ls scale (i32 accumulator, no min) + trailing 0.125 factor; the
// DISTINGUISHING fact is the interleaved block_q8_Kx4 activation (4 rows, 4 fp32 d at +0,
// interleaved int8 quants at +16 as pos*4+c). The grid+sign weight decode is AMORTIZED
// once per 16-weight group and reused across the 4 activation columns. block_iq2_xxsx16
// stride 1184; activation block_q8_Kx4 stride 1168. Ships PLAIN (untiled): iq2_xxs sits at
// the <=32-vreg cliff. VLEN=128 => TWO strips, columnsPerPass 4.

module {
  weft.exec.kernel @ggml_repack_gemm_iq2_xxs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq2_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq2_xxs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq2_xxs_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth-4col", qk = 256 : i64, weight_block_stride = 1184 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 160 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "grid_sign_single_scale_eighth", loop_order = "col_outer"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied GRID GEMM integer-core BRICK: per-block
          // lane-wise iq2_xxs grid + sign memory-gather ls-scaled dot across the 4
          // interleaved columns -> the columnsPerPass (4) per-column i32 sumi. The typed
          // emitter re-emits the whole byte-exact iq2_xxs GEMM body from this brick's
          // identity + its grid/ls/sign offsets; the yield passes through the carried-in
          // per-column accs.
          %sumi:4 = weft_rvv.repack_gemm_grid_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_grid_core", decode_model = "iq2_xxs", weight_quant_byte_offset = 160 : i64, weight_ls_byte_offset = 32 : i64, weight_sign_byte_offset = 672 : i64, activation_quant_byte_offset = 16 : i64, n_subblocks = 8 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq2_xxs_q8_K_kernel_ggml_repack_gemm_iq2_xxs_q8_K(
// The FIXED 256-entry GRID-of-8 table + the DERIVED signs64 +-1 plane (static decls).
// CHECK: verbatim "static const int64_t weft_iq2xxs_grid[256] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1
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
// SIGN: literal "weft_iq2xxs_signs64"
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
