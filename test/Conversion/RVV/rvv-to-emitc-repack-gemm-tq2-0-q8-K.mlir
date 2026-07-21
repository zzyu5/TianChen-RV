// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线B batch1: the ggml tq2_0 x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) hot
// kernel is now CONSTRUCTED through the typed-region FRONT DOOR (the q4_0
// typed_repack precedent), NOT the retired monolithic emitRepackGemmTQ20Q8K direct
// emitter. The weft_rvv.typed_repack_gemm_loop_body region (fold_model
// "ternary_single_fp16_scale") carries the weft_rvv.repack_gemm_ternary_core
// integer-core BRICK (decode_model "tq2_0"), block_index + strip_row_offset tied
// (anti-bypass). The lowering GATES on that brick's ties then RE-EMITS the byte-exact
// ternary GEMM body via emitTypedRepackGemmLoopBody's ternary branch ->
// emitRepackTernaryGemmBodyTQ20 (byte-identical to the retired direct emitter). The
// BLOCK-AS-LANE multi-output-column matmul: the 16 interleaved weight columns of a
// group occupy 16 vector lanes (LANE-WISE vwmacc, NO cross-lane vredsum); the
// activation is the interleaved block_q8_Kx4 stream (4 activation ROWS per group,
// stride 1168). The 2-bit TERNARY weight decode (((byte>>shift)&3)-1) is AMORTIZED
// once per 16-weight group and REUSED across the 4 interleaved activation columns.
// LINEAR single fp16 super-block scale, NO min term. block_tq2_0x16 stride 1056.

module {
  weft.exec.kernel @ggml_repack_gemm_tq2_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nr = weft_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-single-scale-2bit-ternary-4col-nomin", qk = 256 : i64, weight_block_stride = 1056 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "ternary_single_fp16_scale", weft_rvv.loop_order = "row_outer", weft_rvv.loop_order_selection_reason = "prior"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">, %acc2: !weft_rvv.vector<f32, "m2">, %acc3: !weft_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied ternary GEMM integer-core BRICK:
          // per-block lane-wise trit dot across the 4 interleaved columns -> the
          // columnsPerPass (4) per-column i32 sumi. The typed emitter re-emits the whole
          // byte-exact ternary GEMM body from this brick's identity; the yield passes
          // through the carried-in per-column accs.
          %sumi:4 = weft_rvv.repack_gemm_ternary_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_ternary_core", decode_model = "tq2_0", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 16 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, index, index
      } : !weft_rvv.vl
    }
  }
}

// CHECK-NOT: weft_rvv.repack_gemm_ternary_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_tq2_0_q8_K_kernel_ggml_repack_gemm_tq2_0_q8_K(
// The block count nb = n / 256, row-group count nr/4 (%arg4), column-group nc/16 (%arg5).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop over nr/4; interleaved base vy + y*nb*1168.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "1168"
// The weight-column-GROUP loop; weight base vx + x*nb*1056.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "1056"
// The per-column-per-strip f32 accumulators.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The 4 per-column activation super-block deltas d_y_c are fp32 SCALARS.
// CHECK: call_opaque "*(const float *)"
// The SHARED per-strip fp16 d strip (unpacked ONCE per group, reused across columns).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// The per-column i32 accumulators + per-super-half i16 partials.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The tq2_0 2-bit TERNARY weight assembly SHARED across columns: vand 0x03,
// reinterpret to i8, then vsub_vx_i8 by 1 (the `-1` ternary bias).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The per-column integer dot against the interleaved q8_Kx4 quants (NO vredsum),
// folded into i32 via vwadd_wv.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The per-column end-of-block SINGLE-scale fold: vfmul, vfcvt, vfmacc (NO vfnmsac).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The per-column per-strip store s + (y*4+c)*bs + x*16 + h*8.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
