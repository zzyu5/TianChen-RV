// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 format4 CONSTRUCTION -- the K-quant q5_K sibling of the q4_K
// rvv-emit-quant-contraction proof, the THIRD MIN-fold K-quant (after q4_K/q2_K) and the
// LAST K-quant repack sibling, COMPLETING the K-quant repack family. The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (K-quant q5_K / decode) is
// AUTO-LOWERED by --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection
// reads the STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived
// capability VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the
// committed K-quant q5_K scale_model WHAT, so the C1 bridge lowerToRepackGemvKQuant
// (parameterized by kQ5KDecodeFacts, hasMin=true -- REUSING the q4_K dual d/dmin +
// bsums-min fold WHOLE, ZERO framework re-pay) CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (fold_model "kquant_dmin_bsums_min", SHARED
// with q4_K) carrying the SINGLE weft_rvv.repack_gemv_kquant_core integer-core brick
// (decode_model "q5_K"), reconstructing the block_q5_Kx16 x16 weight facts 2816/768 (dmin
// @32, 6-bit scales/mins @64, the qh 5th-bit plane @256 [the SHARED qh slot, here on a MIN
// fold]) + the plain block_q8_K activation facts 292/4 (bsums @260), n_subblocks 8,
// half_lanes=8 => mf2, numHalves==2. The q5_K delta over q4_K is the qh SECOND weight
// plane: it rides the OPTIONAL loop-body weight_qh_byte_offset attr (the same slot q6_K/q3_K
// use on the no-min fold), stamped ALONGSIDE the q4_K min structure (dmin + bsums).
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the q5_K repack region from a REAL
// quant_contraction request. The constructed region lowers to the byte-exact q5_K GEVM
// kernel the retired monolithic emitRepackGemvQ5KQ8K produced. NO perf/e2e claim -- lit-
// emitted, NOT run.

module {
  weft.exec.kernel @ggml_repack_gemv_q5_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q5_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q5-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q5_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q5_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT K-quant q5_K request: PLAIN block_q5_K (stride 176, qs @48, qh @16) x
        // PLAIN block_q8_K (stride 292), qk 256, scale_model = the K-quant q5_K WHAT,
        // block_dot_compute_heavy = true (routes REPACK). The result is dead (the repacked
        // lane-wise q5_K GEVM sinks through the output pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q5_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 176 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 48 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the K-quant q5_K typed_repack GEVM
// region (fold_model kquant_dmin_bsums_min, SHARED with q4_K) carrying the K-quant core
// brick (decode_model q5_K), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track that
// order. The q4_K MIN structure (dmin + bsums) AND the q5_K qh 5th-bit plane are BOTH
// stamped.
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: activation_bsums_byte_offset = 260
// CONSTRUCT-SAME: activation_quant_byte_offset = 4
// CONSTRUCT-SAME: fold_model = "kquant_dmin_bsums_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: main_term_form = "unrolled"
// CONSTRUCT-SAME: n_subblocks = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks-qh5"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 2816
// CONSTRUCT-SAME: weight_dmin_byte_offset = 32
// CONSTRUCT-SAME: weight_qh_byte_offset = 256
// CONSTRUCT-SAME: weight_quant_byte_offset = 768
// CONSTRUCT-SAME: weight_scales_byte_offset = 64
// The in-region block_index-tied K-quant integer-core BRICK, decode_model q5_K.
// CONSTRUCT: weft_rvv.repack_gemv_kquant_core
// CONSTRUCT-SAME: decode_model = "q5_K"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the q5_K GEVM kernel (byte-exact to the retired direct
// emitter -- the SAME emit as rvv-to-emitc-repack-gemv-q5-K-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q5_K_q8_K_kernel_ggml_repack_gemv_q5_K_q8_K(
// Per-group weight base vx + x*nb*2816 (block_q5_Kx16 stride 2816).
// CHECK: literal "2816"
// The two 8-lane f32m2 accumulators (mf2 / half_lanes=8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The plain q8_K activation base al = a + l*292 (stride 292), inside the block loop.
// CHECK: literal "292"
// The q4_K-shared 6-bit scale/min unpack (vand 0x0F / vsrl 4 / vzext to i16).
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// The q5_K 5-bit weight assembly: the q4_K nibble (vand 0x0F / vsrl 4) PLUS a SECOND vle8
// loading the qh strip, vand 0x01 masking the selected per-sub-block bit (a mask q4_K
// NEVER emits), vsll 4 lifting it, vor merging it onto the nibble -> a 5-bit value in
// [0,31].
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x01"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// The lane-wise integer dot + scale-weighted i32 promote (NO vredsum).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The MIN term: read the per-block int16 bsums, fold via vwmacc, subtract via vfnmsac.
// CHECK: call_opaque "*(const int16_t *)"
// The end-of-block dual fold: vfmacc the main term, vfnmsac the MIN term.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: redsum
