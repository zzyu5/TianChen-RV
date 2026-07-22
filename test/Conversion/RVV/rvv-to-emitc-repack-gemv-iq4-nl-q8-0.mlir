// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=GATHER
// RUN: weft-opt %s --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M2 iq4_nl codebook front-door: the ggml iq4_nl x q8_0 16x1-REPACKED GEVM (decode) hot
// kernel -- the FIRST CODEBOOK (non-linear decode) decode family -- is now CONSTRUCTED
// through the typed-region FRONT DOOR (the q4_0 / ternary / K-quant typed_repack precedent),
// NOT the retired monolithic emitRepackGemvIq4NlQ80 direct emitter. The
// weft_rvv.typed_repack_gemv_loop_body region (fold_model "codebook_flat_single_scale")
// carries the weft_rvv.repack_gemv_codebook_core integer-core BRICK (decode_model "iq4_nl" +
// the 16-entry non-linear int8 codebook DenseI8ArrayAttr), block_index-tied (anti-bypass)
// and named off the loop-body's own weight / activation ABI bases. The lowering GATES the
// emit on that brick's anti-bypass ties, then RE-EMITS the byte-exact iq4_nl GEVM body via
// emitTypedRepackGemvLoopBody's codebook branch -> emitRepackCodebookGemvBodyIq4Nl
// (byte-identical to the retired direct emitter). The 4-bit weight nibble is an INDEX into
// the 16-entry codebook, decoded by a REAL MEMORY codebook GATHER (vzext the nibble to a u16
// byte offset, then vluxei16_v_i8 through the codebook array -- NOT a register vrgather, NOT
// a fake-linear value); the dot accumulates LANE-WISE in an i32 accumulator (codebook
// products overflow i16), single fp16 scale, NO min. block_iq4_nlx16 stride 288 (d @0,
// qs @32); activation is one plain block_q8_0 (stride 34, qs at +2). VLEN=128 => TWO strips.

module {
  weft.exec.kernel @ggml_repack_gemv_iq4_nl_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        weft_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", scale_model = "flat.fp16-single-scale-codebook-nomin", qk = 32 : i64, weight_block_stride = 288 : i64, activation_block_stride = 34 : i64, weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, weight_interleave = 16 : i64, half_lanes = 8 : i64, integer_core_lmul = "mf2", fold_model = "codebook_flat_single_scale"} {
        ^bb0(%block_index: index, %acc0: !weft_rvv.vector<f32, "m2">, %acc1: !weft_rvv.vector<f32, "m2">):
          // The block_index-tied CODEBOOK integer-core BRICK: per-block lane-wise iq4_nl
          // memory-gather dot -> the numHalves (2) per-strip i32 sumi. The typed emitter
          // re-emits the whole byte-exact iq4_nl body (codebook gather + i32 dot + single
          // fp16 scale fold) from this brick's identity + its 16-entry codebook; the yield
          // passes through the carried-in accs.
          %sumi:2 = weft_rvv.repack_gemv_codebook_core %vx, %vy, %vl block %block_index : index {kind = "repack_gemv_codebook_core", decode_model = "iq4_nl", weight_quant_byte_offset = 32 : i64, activation_quant_byte_offset = 2 : i64, codebook = array<i8: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113>} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.vl -> !weft_rvv.vector<i32, "m2">, !weft_rvv.vector<i32, "m2">
          weft_rvv.typed_repack_gemv_loop_yield %acc0, %acc1 : !weft_rvv.vector<f32, "m2">, !weft_rvv.vector<f32, "m2">
        } : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index
      } : !weft_rvv.vl
    }
  }
}

// The front door leaves NO typed op behind (fully lowered to emitc).
// CHECK-NOT: weft_rvv.repack_gemv_codebook_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq4_nl_q8_0_kernel_ggml_repack_gemv_iq4_nl_q8_0(
// The 16-entry NON-LINEAR int8 codebook decl (the kvalues_iq4nl table).
// CHECK: verbatim "static const int8_t weft_iq4_nl_repack_kvalues[16] = {-127, -104
// The block count nb = n / 32 (QK4_NL) and the column-group count nc/16.
// CHECK: div %arg0, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// CHECK: div %arg4, %{{.*}} : (!emitc.opaque<"size_t">, !emitc.opaque<"size_t">) -> !emitc.opaque<"size_t">
// The OUTER weight-column-GROUP loop; per-group weight base vx + x*nb*288.
// CHECK: for %[[X:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "288"
// The two 8-lane f32 accumulators: vfmv_v_f_f32m2(0.0f, 8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The inner contraction-block loop; per-block plain q8_0 base al = a + l*34.
// CHECK: for %[[L:.*]] = %{{.*}} to %{{.*}} step
// CHECK: literal "34"
// The per-strip i32 accumulator seed (codebook products overflow i16).
// CHECK: call_opaque "__riscv_vmv_v_x_i32m2"

// ===== The REAL codebook GATHER (memory vluxei16, NOT register vrgather). =====
// GATHER: call_opaque "__riscv_vle8_v_u8mf2"
// The low nibble index (vand 0x0F) is zero-extended to a u16 byte offset and
// gathered THROUGH the codebook table by vluxei16_v_i8 (codebook[nibble]).
// GATHER: call_opaque "__riscv_vand_vx_u8mf2"
// GATHER: literal "0x0F"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: literal "weft_iq4_nl_repack_kvalues"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// The high nibble index (vsrl 0x04) is gathered the SAME way.
// GATHER: call_opaque "__riscv_vsrl_vx_u8mf2"
// GATHER: literal "0x04"
// GATHER: call_opaque "__riscv_vzext_vf2_u16m1"
// GATHER: call_opaque "__riscv_vluxei16_v_i8mf2"
// The lane-wise i8xi8->i16 product widened into the i32 accumulator.
// GATHER: call_opaque "__riscv_vwmul_vx_i16m1"
// GATHER: call_opaque "__riscv_vwadd_wv_i32m2"

// The end-of-block SINGLE-scale fold: vle16 the fp16 d strip, vfwmul by d_y,
// vfcvt the i32 sumi, then vfmacc (NO min term, NO dmin).
// CHECK: call_opaque "__riscv_vle16_v_f16m1"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// The single-column two-half vector store vse32_v_f32m2.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The codebook decode is a MEMORY gather (vluxei16), NOT a register vrgather (the
// fractional mf2 anchor's VLMAX < 16 forbids indexing all 16 entries in a vreg).
// The block-as-lane repack erases the cross-lane reduction wall; the flat codebook
// fold has NO min (no vfnmsac) and NO per-sub-block scale (no vwmacc_vv_i32). Also NO
// trailing dead result-token vmv (the region is result-less, unlike the retired direct
// emitter).
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
