// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T3 CONSTRUCTION -- the K-quant q6_K sibling of the q4_K
// rvv-emit-identity-quant-contraction-q4-K-repack proof, GENERALIZING the front-door
// framework's K-quant fold to the NO-MIN variant. The abstract, algorithm-UNCOMMITTED
// weft_rvv.quant_contraction op (K-quant q6_K / decode) is AUTO-LOWERED by
// --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection reads the
// STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived capability
// VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the committed
// K-quant q6_K scale_model WHAT, so the C1 bridge lowerToRepackGemvKQuant (parameterized
// by kQ6KDecodeFacts, hasMin=false) CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (fold_model "kquant_single_scale_no_min")
// carrying the SINGLE weft_rvv.repack_gemv_kquant_core integer-core brick (decode_model
// "q6_K"), reconstructing the block_q6_Kx16 x16 weight facts 3360/1312 (qh @288, SIGNED
// scales @32) + the plain block_q8_K activation facts 292/4 (NO bsums), n_subblocks 16,
// half_lanes=8 => mf2, numHalves==2. NO dmin / NO bsums attrs are stamped (single-
// accumulator no-min), and the qh SECOND weight plane rides the OPTIONAL loop-body attr.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the q6_K repack region from a
// REAL quant_contraction request (previously the typed region was authored by hand). The
// constructed region lowers to the byte-exact q6_K GEVM kernel the retired monolithic
// emitRepackGemvQ6KQ8K produced. NO perf/e2e claim -- lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_repack_gemv_q6_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_q6_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q6-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q6_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q6_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT K-quant q6_K request: PLAIN block_q6_K (stride 210, ql @0) x
        // PLAIN block_q8_K (stride 292), qk 256, scale_model = the K-quant q6_K WHAT,
        // block_dot_compute_heavy = true (routes REPACK). The result is dead (the
        // repacked lane-wise q6_K GEVM sinks through the output pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q6_K", scale_model = "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 210 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the K-quant q6_K typed_repack GEVM
// region (fold_model kquant_single_scale_no_min) carrying the K-quant core brick
// (decode_model q6_K), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track
// that order. NO dmin / NO bsums attrs (single-accumulator no-min).
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: activation_quant_byte_offset = 4
// CONSTRUCT-SAME: fold_model = "kquant_single_scale_no_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: n_subblocks = 16
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-signed8-scale-16-subblocks-6bit-nomin"
// CONSTRUCT-SAME: weft_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 3360
// CONSTRUCT-SAME: weight_qh_byte_offset = 288
// CONSTRUCT-SAME: weight_scales_byte_offset = 32
// CONSTRUCT-NOT: weight_dmin_byte_offset
// The in-region block_index-tied K-quant integer-core BRICK, decode_model q6_K.
// CONSTRUCT: weft_rvv.repack_gemv_kquant_core
// CONSTRUCT-SAME: decode_model = "q6_K"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the q6_K GEVM kernel (byte-exact to the retired
// direct emitter -- the SAME emit as rvv-to-emitc-repack-gemv-q6-K-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_kquant_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K(
// Per-group weight base vx + x*nb*3360 (block_q6_Kx16 stride 3360).
// CHECK: literal "3360"
// The two 8-lane f32m2 accumulators (mf2 / half_lanes=8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The plain q8_K activation base al = a + l*292 (stride 292), inside the block loop.
// CHECK: literal "292"
// The q6_K SIGNED int8 per-sub-block scale: vle8_v_i8 + vsext_vf2_i16 (SIGN-extend).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf2_i16m1"
// The q6_K 6-bit two-plane weight assembly: vle8 ql/qh, TWO-bit qh mask 0x03, vor,
// vsub 32 (the -32 offset-binary bias folded into each SIGNED weight lane).
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: literal "0x03"
// CHECK: call_opaque "__riscv_vor_vv_u8mf2"
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// The lane-wise integer dot + scale-weighted i32 promote (NO vredsum).
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block SINGLE fold: vfcvt the i32 sumi, vfmacc the ONE main term (NO min).
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return
// q6_K has NO min term: NO vfnmsac, NO int16 bsums scalar read.
// CHECK-NOT: vfnmsac

// The block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: redsum
