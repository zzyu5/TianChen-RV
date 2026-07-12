// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 format2: the ggml q2_K x q8_K 16x1-REPACKED PREFILL GEMM hot kernel is now
// CONSTRUCTED through the typed-region FRONT DOOR (the q4_0 / ternary / q4_K / q6_K
// typed_repack precedent), NOT the retired monolithic emitRepackGemmQ2KQ8K direct
// emitter. The weft_rvv.typed_repack_gemm_loop_body region (fold_model
// "kquant_dmin_bsums_min", SHARED with q4_K) carries the weft_rvv.repack_gemm_kquant_core
// integer-core BRICK (decode_model "q2_K"), block_index + strip_row_offset tied
// (anti-bypass). The lowering GATES the emit on those anti-bypass ties, then RE-EMITS the
// byte-exact S6-TILED q2_K GEMM body via emitTypedRepackGemmLoopBody's K-quant branch ->
// emitRepackKQuantGemmBodyQ2K. Because q2_K SHARES the q4_K min fold, the FAMILY S6 tiling
// transfers WHOLE: S1 h-strip output tile + decoded 4-bit scale/min stack-panel + idle
// i32 MIN accumulator stack-panel + on-demand d/dmin widen (SSA-register hot accs). The
// activation is the INTERLEAVED block_q8_Kx4 stream (stride 1168, 4 fp32 d at +0,
// 4-column-interleaved int8 quants at +16, 64 int16 bsums at +1040); the 2-bit weight
// decode + 4-bit scale/min unpack are AMORTIZED once per 16-weight group across the 4
// columns. VLEN=128 mf2 => columnsPerPass=4, TWO 8-lane strips (h-tiled).
//
// NUMERIC STATUS: this lit checks the LOWERED STRUCTURE only. Numeric correctness is
// proven SEPARATELY by the board oracle (independent scalar q2_K dequant-matmul
// reference, byte-exact-integer isum + summs). The S6 tiling is integer stack-panels +
// unchanged fold order = bit-identical to the plain untiled emit.

module {
  weft.exec.kernel @ggml_repack_gemm_q2_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q2_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q2_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q2_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-16-subblocks-2bit-4col", qk = 256 : i64, weight_block_stride = 1344 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 320 : i64, activation_quant_byte_offset = 16 : i64, weight_dmin_byte_offset = 32 : i64, weight_scales_byte_offset = 64 : i64, activation_bsums_byte_offset = 1040 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "kquant_dmin_bsums_min"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied q2_K GEMM integer-core BRICK:
          // per-block lane-wise q2_K dot across the 4 interleaved columns -> the
          // columnsPerPass (4) per-column i32 sumi. The typed emitter re-emits the whole
          // byte-exact S6-tiled q2_K GEMM body from this brick's identity; the yield
          // passes through the carried-in per-column accs.
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q2_K", weight_quant_byte_offset = 320 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemm_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The row-group count nr/4 (%arg4 = nr) and column-group count nc/16 (%arg5 = nc).
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop over nr/4.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// Per-group activation base vy + y*nb*1168 (block_q8_Kx4 stride 1168).
// CHECK: literal "1168"
// The weight-column-GROUP loop over nc/16; per-group weight base vx + x*nb*1344.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "1344"
// The per-column f32 accumulators: vfmv_v_f_f32m2(0.0f, 8) (4 cols, S1 h-strip tiled).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb (per h strip).
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-COLUMN activation super-block delta d_y_c is a fp32 SCALAR (NOT fp16),
// read at al + c*4.
// CHECK: call_opaque "*(const float *)"
// The per-column per-strip i32 main accumulator seed vmv_v_x_i32m2(0, 8).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The q2_K per-sub-block 4-bit packed scale/min unpack (SHARED across the 4 columns),
// STAGED to the int16_t scale/min stack panel (vse16/vle16): vle8 the SINGLE packed
// byte, vand 0x0F (scale) / vsrl 4 (min), vzext to i16 / reinterpret. This is
// q2_K-specific: a SINGLE byte, NOT q4_K's 6-bit two-byte vand/vsll/vsrl/vor dance.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vreinterpret_v_u16m1_i16m1"
// CHECK: call_opaque "__riscv_vse16_v_i16m1"
// The MIN term: read the per-COLUMN interleaved SINGLE int16 bsum and fold it LANE-WISE
// via vwmacc_vx (bsum * min strip -> i32), the idle i32 accumulator staged in the
// int32_t bsums stack panel (vse32/vle32).
// CHECK: call_opaque "*(const int16_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// CHECK: call_opaque "__riscv_vse32_v_i32m2"
// The q2_K 2-bit weight assembly (SHARED across the 4 columns): the qs byte strip is
// loaded ONCE then peeled into FOUR 2-bit lanes -- vand 0x03 (shift 0), vsrl {2,4,6} +
// vand 0x03 -- each reinterpreted to a SIGNED i8 lane. The 0x03 TWO-bit mask
// distinguishes q2_K from q4_K's 0x0F nibble.
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// The MAIN integer dot, multi-column: the SHARED weight 2-bit decode reused across the
// columns (vwmacc_vx_i16m1 against the per-column interleaved q8_Kx4 quants, NO
// cross-lane vredsum), each 16-element partial promoted to i32 weighted by the 4-bit
// scale reloaded from the panel (vwmacc_vv).
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block per-column fold: the ON-DEMAND d/dmin widen (vle16, vfwcvt, vfmul_vf
// d_x*d_y_c), vfcvt the i32 sumi, vfmacc the main term, then vfnmsac the MIN term
// (sumf -= dmin_x*d_y_c*bsums reloaded from the panel).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// The per-column per-strip vector store vse32_v_f32m2 to s + (y*4+c)*bs + x*16 + h*8.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall: the
// dot accumulates LANE-WISE via vwmacc, so NO vredsum / vwredsum appears.
// NOWALL-NOT: redsum
