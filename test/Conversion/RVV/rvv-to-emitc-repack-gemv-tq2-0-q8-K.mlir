// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线B batch1: the ggml tq2_0 x q8_K 16x1-REPACKED GEVM (decode) hot kernel is
// now CONSTRUCTED through the typed-region FRONT DOOR (the q4_0 typed_repack
// precedent), NOT the retired monolithic emitRepackGemvTQ20Q8K direct emitter. The
// weft_rvv.typed_repack_gemv_loop_body region (fold_model "ternary_single_fp16_scale")
// carries the weft_rvv.repack_gemv_ternary_core integer-core BRICK (decode_model
// "tq2_0"), block_index-tied (anti-bypass) and named off the loop-body's own weight /
// activation ABI bases. The lowering GATES the emit on that brick's anti-bypass ties,
// then RE-EMITS the byte-exact ternary GEVM body via emitTypedRepackGemvLoopBody's
// ternary branch -> emitRepackTernaryGemvBodyTQ20 (byte-identical to the retired
// direct emitter; the git diff shows ZERO emitOpaqueCall body churn). tq2_0 is LINEAR:
// the weight is a 2-BIT TERNARY trit, ((byte >> {0,2,4,6}) & 3) - 1 in {-1,0,1} (the
// vand 0x03 peel + the vsub_vx_i8 by 1 ternary bias), ONE fp16 super-block scale, the
// dot folds LANE-WISE into ONE i32 accumulator per strip (a per-super-half i16 partial
// widened via vwadd_wv) -- NO per-sub-block scale, NO dmin, NO bsums, NO min term.
// block_tq2_0x16 stride 1056 (d at +0, qs at +32); plain block_q8_K activation stride
// 292 (fp32 d at +0, int8 quants at +4). VLEN=128 => TWO disjoint 8-lane strips.

module {
  weft.exec.kernel @ggml_repack_gemv_tq2_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_tq2_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_tq2_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_tq2_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-single-scale-2bit-ternary-nomin", qk = 256 : i64, weight_block_stride = 1056 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "ternary_single_fp16_scale"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied ternary integer-core BRICK: per-block lane-wise trit
          // dot -> the numHalves (2) per-strip i32 sumi. The typed emitter re-emits the
          // whole byte-exact ternary body (integer core + single-scale fold) from this
          // brick's identity; the yield passes through the carried-in per-strip accs.
          %sumi:2 = weft_rvv.repack_gemv_ternary_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_ternary_core", decode_model = "tq2_0", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_ternary_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_tq2_0_q8_K_kernel_ggml_repack_gemv_tq2_0_q8_K(
// The block count nb = n / 256 (QK_K).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The column-group count nc/16 (%arg4 = nc). There is NO nr/4 row loop.
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop over nc/16.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// Per-group weight base vx + x*nb*1056 (block_tq2_0x16 stride 1056).
// CHECK: literal "1056"
// The two 8-lane f32 accumulators (cols 0..7, 8..15): vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// The per-block plain q8_K activation base al = a + l*292 (stride 292).
// CHECK: literal "292"
// The activation super-block delta d_y is a single fp32 SCALAR (NOT fp16).
// CHECK: call_opaque "*(const float *)"
// The per-strip i32 accumulator (single, no min) seed vmv_v_x_i32m2, and the
// per-super-half i16 partial seed vmv_v_x_i16m1.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The tq2_0 2-bit TERNARY weight assembly: vle8 the qs byte strip ONCE, vand 0x03
// (the low 2-bit lane), reinterpret to SIGNED i8, then vsub_vx_i8 by 1 -- the `-1`
// TERNARY BIAS (trit in {-1,0,1}) the q2_K repack (unsigned [0,3]) LACKS.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The lane-wise integer dot against the single-column q8 quants (NO vredsum), a
// 16-bit partial vwmacc_vx, and the higher 2-bit lanes peeled by vsrl {2,4,6}.
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u8mf2"
// The per-super-half fold widens the i16 partial into the i32 accumulator: vwadd_wv.
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The end-of-block SINGLE-scale fold: vle16 the fp16 d strip, vfwcvt, vfmul by d_y,
// vfcvt the i32 sumi, then vfmacc the single d term (NO vfnmsac MIN term).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2 (s+x*16+0, s+x*16+8).
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the per-block cross-lane reduction wall, and the
// LINEAR ternary fold has NO min term (no vfnmsac) and NO per-sub-block scale
// (no vwmacc_vv_i32). Also NO trailing dead result-token vmv (the region is
// result-less, unlike the retired direct emitter).
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
