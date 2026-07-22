// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线B batch2 CONSTRUCTION -- the ternary tq1_0 (BASE-3) sibling of the tq2_0
// rvv-emit-identity-quant-contraction-tq2-0-repack-vlen128 proof. The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (ternary tq1_0 / decode) is
// AUTO-LOWERED by --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler
// selection reads the STRUCTURED opponent facts (block_dot_compute_heavy = true, no
// opponent_vlen_native_floor) + the derived capability VLEN (rv64gcv => 128) and picks
// REPACK; the decode FAMILY is keyed off the committed ternary tq1_0 scale_model WHAT,
// so the C1 bridge lowerToRepackGemvTernary (shared with tq2_0, parameterized by the
// per-family TernaryDecodeFacts) CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (fold_model "ternary_single_fp16_scale")
// carrying the SINGLE weft_rvv.repack_gemv_ternary_core integer-core brick (decode_model
// "tq1_0"), reconstructing the block_tq1_0x16 x16 weight facts 864/16/32 + the base-3 qh
// SECOND-plane offset 800 + the plain block_q8_K activation facts 292/4, half_lanes=8 =>
// mf2, numHalves==2, NO integer_core_lmul.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the ternary tq1_0 repack region
// from a REAL quant_contraction request (previously the typed region was authored by
// hand in the input IR of rvv-to-emitc-repack-gemv-tq1-0-q8-K.mlir). The constructed
// region lowers to the byte-exact base-3 ternary GEVM kernel the retired monolithic
// emitRepackGemvTQ10Q8K produced. NO perf/e2e claim -- lit-emitted, NOT run.

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
        // The ABSTRACT ternary request: PLAIN block_tq1_0 (stride 54, base-3 qs plane at
        // +0, the fp16 d trailing) x PLAIN block_q8_K (stride 292), qk 256, scale_model =
        // the ternary tq1_0 base-3 WHAT, block_dot_compute_heavy = true (routes REPACK).
        // The result is dead (the repacked lane-wise ternary GEVM sinks through *s).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "tq1_0", scale_model = "superblock-d.fp16-single-scale-base3-ternary-nomin", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 54 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the ternary tq1_0 typed_repack GEVM
// region (fold_model ternary_single_fp16_scale) carrying the ternary core brick
// (decode_model tq1_0), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track that
// order: activation_block_stride < fold_model < half_lanes < scale_model < weft_rvv.*
// audit attrs < weight_block_stride < weight_qh_byte_offset.
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: activation_block_stride = 292
// CONSTRUCT-SAME: fold_model = "ternary_single_fp16_scale"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-single-scale-base3-ternary-nomin"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 864
// CONSTRUCT-SAME: weight_qh_byte_offset = 800
// The in-region block_index-tied ternary integer-core BRICK, decode_model tq1_0.
// CONSTRUCT: weft_rvv.repack_gemv_ternary_core
// CONSTRUCT-SAME: decode_model = "tq1_0"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the base-3 ternary GEVM kernel (byte-exact to the
// retired direct emitter -- the SAME emit as rvv-to-emitc-repack-gemv-tq1-0-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_ternary_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_tq1_0_q8_K_kernel_ggml_repack_gemv_tq1_0_q8_K(
// Per-group weight base vx + x*nb*864 (block_tq1_0x16 stride 864).
// CHECK: literal "864"
// The two 8-lane f32m2 accumulators (mf2 / half_lanes=8), seeded before the block loop.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The plain q8_K activation base al = a + l*292 (stride 292), inside the block loop.
// CHECK: literal "292"
// The tq1_0 BASE-3 TERNARY weight decode: vmul_vx_u8 (byte*pow3[l]), vwmulu_vx_u16 (*3),
// vsrl_vx_u16 (>>8), vncvt, reinterpret to i8, vadd_vx_i8 by -1 -- the trit in {-1,0,1}.
// CHECK: call_opaque "__riscv_vmul_vx_u8mf2"
// CHECK: call_opaque "__riscv_vwmulu_vx_u16m1"
// CHECK: call_opaque "__riscv_vsrl_vx_u16m1"
// CHECK: call_opaque "__riscv_vncvt_x_x_w_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2"
// The lane-wise integer dot + per-super-half i16->i32 widening fold, and the qh base-3
// SECOND plane at repacked byte offset +800.
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: literal "800"
// The end-of-block SINGLE-scale fold (NO min term).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall; the LINEAR ternary fold
// has NO min term (no vfnmsac) and NO per-sub-block scale (no vwmacc_vv_i32).
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
