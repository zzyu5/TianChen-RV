// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线B batch2: the ggml tq1_0 x q8_K 16x1-REPACKED GEVM (decode) hot kernel -- the
// BASE-3 TERNARY (BitNet-class, 1.6 bits/weight) block-as-lane sibling of tq2_0 -- is
// now CONSTRUCTED through the typed-region FRONT DOOR (the q4_0 / tq2_0 typed_repack
// precedent), NOT the retired monolithic emitRepackGemvTQ10Q8K direct emitter. The
// weft_rvv.typed_repack_gemv_loop_body region (fold_model "ternary_single_fp16_scale")
// carries the weft_rvv.repack_gemv_ternary_core integer-core BRICK (decode_model
// "tq1_0"), block_index-tied (anti-bypass) and named off the loop-body's own weight /
// activation ABI bases. The lowering GATES the emit on that brick's anti-bypass ties,
// then RE-EMITS the byte-exact ternary GEVM body via emitTypedRepackGemvLoopBody's
// ternary branch -> emitRepackTernaryGemvBodyTQ10 (byte-identical to the retired direct
// emitter). tq1_0 REUSES tq2_0's single-scale no-min block-as-lane scaffold and differs
// ONLY in the WEIGHT DECODE: a BASE-3 unpack of the 48-byte qs plane (5 trits/byte) +
// the 4-byte qh plane (4 trits/byte). Each trit: q = (uint8_t)(byte * pow3[l])
// (vmul_vx_u8, the mandatory 8-bit wrap), xi = ((uint16_t)q * 3) >> 8 (vwmulu_vx_u16 *
// 3, vsrl_vx_u16 >> 8, in {0,1,2}), narrow (vncvt), reinterpret to i8, then xi - 1
// (vadd_vx_i8 by -1, the trit in {-1,0,1}). block_tq1_0x16 stride 864, d at +0, qs at
// +32, qh at +800 (the SECOND weight plane tq2_0 lacks, carried on the loop body op's
// OPTIONAL weight_qh_byte_offset attr). VLEN=128 => TWO disjoint 8-lane strips.

module {
  weft.exec.kernel @ggml_repack_gemv_tq1_0_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_tq1_0_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "tq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_tq1_0_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_tq1_0_q8_K_kernel", status = "selected-lowering-boundary"} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "superblock-d.fp16-single-scale-base3-ternary-nomin", qk = 256 : i64, weight_block_stride = 864 : i64, activation_block_stride = 292 : i64, weight_quant_byte_offset = 32 : i64, weight_qh_byte_offset = 800 : i64, activation_quant_byte_offset = 4 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "ternary_single_fp16_scale"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied ternary integer-core BRICK: per-block lane-wise base-3
          // trit dot -> the numHalves (2) per-strip i32 sumi. The typed emitter re-emits
          // the whole byte-exact ternary body (base-3 core + single-scale fold) from this
          // brick's identity; the yield passes through the carried-in per-strip accs.
          %sumi:2 = weft_rvv.repack_gemv_ternary_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_ternary_core", decode_model = "tq1_0", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 4 : i64} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
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
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_tq1_0_q8_K_kernel_ggml_repack_gemv_tq1_0_q8_K(
// The block count nb = n / 256 and the column-group count nc/16 (%arg4 = nc).
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop; per-group weight base vx + x*nb*864.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "864"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop over nb; plain q8_K activation stride 292.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "292"
// The activation super-block delta d_y is a single fp32 SCALAR.
// CHECK: call_opaque "*(const float *)"
// The per-strip i32 accumulator + per-region i16 partial seeds.
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"
// CHECK: call_opaque "__riscv_vmv_v_x_i16m1"
// The tq1_0 BASE-3 TERNARY weight decode: vle8 the qs strip, then q = byte*pow3[l]
// (vmul_vx_u8), xi = (q*3)>>8 (vwmulu_vx_u16 * 3, vsrl_vx_u16 >> 8), narrow (vncvt),
// reinterpret to i8, then xi - 1 (vadd_vx_i8 by -1) -- the trit in {-1,0,1}.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vmul_vx_u8mf2"
// CHECK: call_opaque "__riscv_vwmulu_vx_u16m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u16m1"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: literal "-1"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2"
// The lane-wise integer dot (NO vredsum) + the region fold into i32 (vwadd_wv).
// CHECK: call_opaque "*(const int8_t *)"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The qh base-3 high plane at repacked byte offset +800.
// CHECK: literal "800"
// The end-of-block SINGLE-scale fold: vle16 fp16 d, vfwcvt, vfmul, vfcvt, vfmacc
// (NO vfnmsac MIN term).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the reduction wall; the LINEAR fold has no min term.
// Also NO trailing dead result-token vmv (the region is result-less, unlike the retired
// direct emitter).
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
