// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMIN

// G3 主线A T3 format3: the ggml q3_K x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) hot kernel
// is now CONSTRUCTED through the typed-region FRONT DOOR (the q4_0 / ternary / q4_K / q6_K
// typed_repack precedent), NOT the retired monolithic emitRepackGemmQ3KQ8K direct emitter.
// The weft_rvv.typed_repack_gemm_loop_body region (fold_model "kquant_single_scale_no_min",
// SHARED with q6_K) carries the weft_rvv.repack_gemm_kquant_core integer-core BRICK
// (decode_model "q3_K"), block_index + strip_row_offset tied (anti-bypass) and named off
// the loop-body's own weight / activation ABI bases. The lowering GATES the emit on those
// anti-bypass ties, then RE-EMITS the byte-exact PLAIN/UNTILED q3_K GEMM body (S6 output
// tiling is a family-lever NULL for weight-reconstruction-bound q3_K -- like q6_K, the
// q4_K register-cliff lever does NOT transfer -- so the GEMM ships plain per 更简单者胜)
// via emitTypedRepackGemmLoopBody's K-quant no-min branch -> emitRepackKQuantGemmBodyQ3K
// (byte-identical to the retired direct emitter; SSA-register accumulators, no inline
// min-fold as q3_K has no min). The activation is the INTERLEAVED block_q8_Kx4 stream
// (stride 1168, 4-column-interleaved int8 quants at +16; q3_K reads NO bsums); the 3-bit
// qs|hmask decode is AMORTIZED once per 16-weight group across the 4 columns. block_q3_Kx16
// stride 1824, qs at +800, hmask at +288, signed scales at +32. VLEN=128 mf2 =>
// columnsPerPass=4, TWO 8-lane strips.
//
// NUMERIC STATUS: this lit checks the LOWERED STRUCTURE only. Numeric correctness is
// proven SEPARATELY by the board oracle (independent scalar q3_K dequant-matmul reference,
// byte-exact-integer GREEN).

module {
  weft.exec.kernel @ggml_repack_gemm_q3_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q3_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_q3_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_q3_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-4col-nomin", qk = 256 : i64, weight_block_stride = 1824 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_scales_byte_offset = 32 : i64, weight_qh_byte_offset = 288 : i64, n_subblocks = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "kquant_single_scale_no_min"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied q3_K GEMM integer-core BRICK:
          // per-block lane-wise q3_K dot across the 4 interleaved columns -> the
          // columnsPerPass (4) per-column i32 sumi. The typed emitter re-emits the whole
          // byte-exact PLAIN q3_K GEMM body from this brick's identity; the yield passes
          // through the carried-in per-column accs.
          %sumi:4 = weft_rvv.repack_gemm_kquant_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_kquant_core", decode_model = "q3_K", weight_quant_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_q3_K_q8_K_kernel_ggml_repack_gemm_q3_K_q8_K(
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
// Per-group weight base vx + x*nb*1824 (block_q3_Kx16 stride 1824).
// CHECK: literal "1824"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// Per-column activation super-block delta d_y_c is an fp32 SCALAR (4 per q8_Kx4 block).
// CHECK: call_opaque "*(const float *)"
// The SINGLE per-column per-strip i32 accumulator seed (NO bsums accumulator).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// The q3_K SIGNED int8 per-sub-block scale (SHARED across columns): vle8_v_i8 + vsext,
// kept in the plain per-block scale form (S6 stack-panel NULL for weight-bound q3_K).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The q3_K 3-bit subtractive weight assembly (SHARED across the 4 columns): qs/hmask
// strip loads, the 2-bit low plane vand 0x03, the SINGLE hmask bit vand 0x01, vsll 2,
// vor, then vsub 4 (the -4 subtractive bias folded into each SIGNED weight lane).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: literal "0x01"
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

// q3_K is a SINGLE-accumulator no-min fold: NO min-term subtract (vfnmsac) and NO
// activation bsums read (int16_t) appear -- the q4_K/q5_K min-fold signature is ABSENT.
// NOMIN-NOT: vfnmsac
// NOMIN-NOT: *(const int16_t *)
