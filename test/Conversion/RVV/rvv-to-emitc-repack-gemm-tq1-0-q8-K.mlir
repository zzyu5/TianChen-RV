// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线B batch2: the ggml tq1_0 x q8_K 16x1-REPACKED PREFILL GEMM (M>>1) hot kernel --
// the tq1_0 prefill sibling of the tq1_0 GEVM -- is now CONSTRUCTED through the
// typed-region FRONT DOOR (the q4_0 / tq2_0 typed_repack precedent), NOT the retired
// monolithic emitRepackGemmTQ10Q8K direct emitter. The
// tcrv_rvv.typed_repack_gemm_loop_body region (fold_model "ternary_single_fp16_scale")
// carries the tcrv_rvv.repack_gemm_ternary_core integer-core BRICK (decode_model
// "tq1_0"), block_index + strip_row_offset tied (anti-bypass). The lowering RE-EMITS the
// byte-exact ternary GEMM body via emitTypedRepackGemmLoopBody's ternary branch ->
// emitRepackTernaryGemmBodyTQ10 (byte-identical to the retired direct emitter). The
// BASE-3 TERNARY weight decode (q = byte*pow3[l], xi = (q*3)>>8, xi-1) is AMORTIZED once
// per 16-weight group and REUSED across the 4 interleaved block_q8_Kx4 activation columns
// (qs@16, element e of column c at qs[e*4+c]). LINEAR single fp16 super-block scale, NO
// min term. block_tq1_0x16 stride 864, qh at +800 (the SECOND weight plane tq2_0 lacks,
// carried on the loop body op's OPTIONAL weight_qh_byte_offset attr); interleaved
// activation stride 1168.

module {
  tcrv.exec.kernel @ggml_repack_gemm_tq1_0_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_tq1_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq1-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nr = tcrv_rvv.runtime_abi_value {c_name = "nr", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nr", role = "source-byte-stride"} : index
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_tq1_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_tq1_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", scale_model = "superblock-d.fp16-single-scale-base3-ternary-4col-nomin", qk = 256 : i64, weight_block_stride = 864 : i64, activation_block_stride = 1168 : i64, weight_quant_byte_offset = 32 : i64, weight_qh_byte_offset = 800 : i64, activation_quant_byte_offset = 16 : i64, weight_interleave = 16 : i64, activation_interleave = 4 : i64, half_lanes = 8 : i64, fold_model = "ternary_single_fp16_scale"} {
        ^bb0(%block_index: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          // The block_index + strip_row_offset tied ternary GEMM integer-core BRICK:
          // per-block lane-wise base-3 trit dot across the 4 interleaved columns -> the
          // columnsPerPass (4) per-column i32 sumi. The typed emitter re-emits the whole
          // byte-exact ternary GEMM body from this brick's identity; the yield passes
          // through the carried-in per-column accs.
          %sumi:4 = tcrv_rvv.repack_gemm_ternary_core %vx, %vy, %vl block %block_index strip %roff : index, index {kind = "repack_gemm_ternary_core", decode_model = "tq1_0", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %acc0, %acc1, %acc2, %acc3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: tcrv_rvv.repack_gemm_ternary_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_tq1_0_q8_K_kernel_ggml_repack_gemm_tq1_0_q8_K(
// The block count nb, row-group count nr/4 (%arg4), column-group nc/16 (%arg5).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg5, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER activation-row-GROUP loop; interleaved base vy + y*nb*1168.
// CHECK: for %[[Y:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "1168"
// The weight-column-GROUP loop; weight base vx + x*nb*864.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "864"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: call_opaque "*(const float *)"
// The SHARED per-strip fp16 d strip.
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// The per-column i32 accumulators + per-region i16 partials.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The tq1_0 BASE-3 TERNARY weight decode SHARED across columns: vle8, vmul_vx_u8
// (byte*pow3[l]), vwmulu_vx_u16 (*3), vsrl_vx_u16 (>>8), vncvt, reinterpret, vadd
// (-1).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vmul_vx_u8mf2"
// CHECK: call_opaque "__riscv_vwmulu_vx_u16m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u16m1"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2"
// The per-column integer dot against the interleaved q8_Kx4 quants; region fold i32.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The qh base-3 high plane at repacked byte offset +800.
// CHECK: literal "800"
// The per-column end-of-block SINGLE-scale fold: vfmul, vfcvt, vfmacc (NO vfnmsac).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// NO cross-lane reduction wall; the LINEAR ternary fold has NO min term (no vfnmsac)
// and NO per-sub-block scale (no vwmacc_vv_i32). Also NO trailing dead result-token vmv
// (the region is result-less, unlike the retired direct emitter).
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
