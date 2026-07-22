// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 format3 CONSTRUCTION -- the K-quant q3_K sibling of the q6_K
// rvv-emit-identity-quant-contraction-q6-K-repack proof, the SECOND NO-MIN K-quant fold
// (after q6_K) and the LAST K-quant repack sibling. The abstract, algorithm-UNCOMMITTED
// weft_rvv.quant_contraction op (K-quant q3_K / decode) is AUTO-LOWERED by
// --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection reads the
// STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived capability
// VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the committed
// K-quant q3_K scale_model WHAT, so the C1 bridge lowerToRepackGemvKQuant (parameterized
// by kQ3KDecodeFacts, hasMin=false -- REUSING the q6_K no-min fold, ZERO framework re-pay)
// CONSTRUCTS the typed weft_rvv.typed_repack_gemv_loop_body REGION (fold_model
// "kquant_single_scale_no_min") carrying the SINGLE weft_rvv.repack_gemv_kquant_core
// integer-core brick (decode_model "q3_K"), reconstructing the block_q3_Kx16 x16 weight
// facts 1824/800 (hmask @288 [the SHARED qh slot], SIGNED scales @32) + the plain
// block_q8_K activation facts 292/4 (NO bsums), n_subblocks 16, half_lanes=8 => mf2,
// numHalves==2. NO dmin / NO bsums attrs are stamped (single-accumulator no-min), and the
// hmask SECOND weight plane rides the OPTIONAL loop-body weight_qh_byte_offset attr.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the q3_K repack region from a
// REAL quant_contraction request (previously the typed region was authored by hand). The
// constructed region lowers to the byte-exact q3_K GEVM kernel the retired monolithic
// emitRepackGemvQ3KQ8K produced. NO perf/e2e claim -- lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_repack_gemv_q3_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q3_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q3-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q3_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q3_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT K-quant q3_K request: PLAIN block_q3_K (stride 110, hmask @0) x
        // PLAIN block_q8_K (stride 292), qk 256, scale_model = the K-quant q3_K WHAT,
        // block_dot_compute_heavy = true (routes REPACK). The result is dead (the
        // repacked lane-wise q3_K GEVM sinks through the output pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q3_K", scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 110 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the K-quant q3_K typed_repack GEVM
// region (fold_model kquant_single_scale_no_min) carrying the K-quant core brick
// (decode_model q3_K), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track
// that order. NO dmin / NO bsums attrs (single-accumulator no-min).
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: activation_quant_byte_offset = 4
// CONSTRUCT-SAME: fold_model = "kquant_single_scale_no_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: main_term_form = "unrolled"
// CONSTRUCT-SAME: n_subblocks = 16
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-signed6-scale-16-subblocks-3bit-subtractive-hmask-nomin"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1824
// CONSTRUCT-SAME: weight_qh_byte_offset = 288
// CONSTRUCT-SAME: weight_scales_byte_offset = 32
// CONSTRUCT-NOT: weight_dmin_byte_offset
// The in-region block_index-tied K-quant integer-core BRICK, decode_model q3_K.
// CONSTRUCT: weft_rvv.repack_gemv_kquant_core
// CONSTRUCT-SAME: decode_model = "q3_K"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the q3_K GEVM kernel (byte-exact to the retired
// direct emitter -- the SAME emit as rvv-to-emitc-repack-gemv-q3-K-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q3_K_q8_K_kernel_ggml_repack_gemv_q3_K_q8_K(
// Per-group weight base vx + x*nb*1824 (block_q3_Kx16 stride 1824).
// CHECK: literal "1824"
// The two 8-lane f32m2 accumulators (mf2 / half_lanes=8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The plain q8_K activation base al = a + l*292 (stride 292), inside the block loop.
// CHECK: literal "292"
// The q3_K SIGNED int8 per-sub-block scale: vle8_v_i8 + vsext_vf2_i16 (SIGN-extend).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The q3_K 3-bit subtractive weight assembly (NATIVE-MASK KNEST): vle8 qs/hmask, 2-bit
// low plane vand 0x03 reinterpreted to signed i8, then the SINGLE hmask high bit tested
// IN PLACE by vand(1<<p) + vmseq==0 with the -4 subtractive bias FUSED via masked add
// vadd_vx_i8mf2_mu (byte-exact to the retired vsll 2 | vor | vsub 4 chain).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vreinterpret_v_u8mf2_i8mf2"
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vmseq_vx_u8mf2_b16"
// CHECK: literal "-4"
// CHECK: call_opaque "__riscv_vadd_vx_i8mf2_mu"
// The lane-wise integer dot + scale-weighted i32 promote (NO vredsum).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block SINGLE fold: vfcvt the i32 sumi, vfmacc the ONE main term (NO min).
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
// q3_K has NO min term: NO vfnmsac, NO int16 bsums scalar read.
// CHECK-NOT: vfnmsac

// The block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: redsum
