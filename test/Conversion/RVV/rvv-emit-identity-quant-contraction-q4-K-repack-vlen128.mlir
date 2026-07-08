// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 主线A T2-construct CONSTRUCTION -- the K-quant q4_K sibling of the q4_0 / ternary
// rvv-emit-identity-quant-contraction-*-repack proofs. The abstract, algorithm-
// UNCOMMITTED tcrv_rvv.quant_contraction op (K-quant q4_K / decode) is AUTO-LOWERED by
// --tcrv-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection reads the
// STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived capability
// VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the committed
// K-quant scale_model WHAT, so the C1 bridge lowerToRepackGemvKQuant CONSTRUCTS the typed
// tcrv_rvv.typed_repack_gemv_loop_body REGION (fold_model "kquant_dmin_bsums_min")
// carrying the SINGLE tcrv_rvv.repack_gemv_kquant_core integer-core brick (decode_model
// "q4_K"), reconstructing the block_q4_Kx16 x16 weight facts 2304/16/256 (dmin @32,
// 6-bit scales @64) + the plain block_q8_K activation facts 292/4 (bsums @260),
// n_subblocks 8, half_lanes=8 => mf2, numHalves==2, PLUS the 4 emitter-INERT audit attrs.
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the K-quant repack region from a
// REAL quant_contraction request (previously the typed region was authored by hand in the
// input IR of rvv-to-emitc-repack-gemv-q4-K-q8-K.mlir). The constructed region lowers to
// the byte-exact q4_K GEVM kernel the retired monolithic emitRepackGemvQ4KQ8K produced.
// NO perf/e2e claim -- lit-emitted, NOT run.

module {
  tcrv.exec.kernel @ggml_repack_gemv_q4_K_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_q4_K_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_q4_K_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT K-quant request: PLAIN block_q4_K (stride 144, nibbles @16) x
        // PLAIN block_q8_K (stride 292), qk 256, scale_model = the K-quant q4_K WHAT,
        // block_dot_compute_heavy = true (routes REPACK). The result is dead (the
        // repacked lane-wise K-quant GEVM sinks through the output pointer).
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the K-quant typed_repack GEVM
// region (fold_model kquant_dmin_bsums_min) carrying the K-quant core brick
// (decode_model q4_K), NOT a hand-authored region.
// CONSTRUCT-NOT: tcrv_rvv.quant_contraction
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track
// that order: fold_model < half_lanes < n_subblocks < scale_model < weight_block_stride.
// CONSTRUCT: tcrv_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: activation_bsums_byte_offset = 260
// CONSTRUCT-SAME: fold_model = "kquant_dmin_bsums_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: n_subblocks = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks"
// CONSTRUCT-SAME: tcrv_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: tcrv_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 2304
// CONSTRUCT-SAME: weight_dmin_byte_offset = 32
// CONSTRUCT-SAME: weight_scales_byte_offset = 64
// The in-region block_index-tied K-quant integer-core BRICK, decode_model q4_K.
// CONSTRUCT: tcrv_rvv.repack_gemv_kquant_core
// CONSTRUCT-SAME: decode_model = "q4_K"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the q4_K GEVM kernel (byte-exact to the retired
// direct emitter -- the SAME emit as rvv-to-emitc-repack-gemv-q4-K-q8-K).
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemv_kquant_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
// Per-group weight base vx + x*nb*2304 (block_q4_Kx16 stride 2304).
// CHECK: literal "2304"
// The two 8-lane f32m2 accumulators (mf2 / half_lanes=8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The plain q8_K activation base al = a + l*292 (stride 292), inside the block loop.
// CHECK: literal "292"
// The 6-bit scale/min lane-wise unpack (vand 0x0F / vsrl / vsll / vor / vzext).
// CHECK: call_opaque "__riscv_vand_vx_u8mf2"
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// The MIN term (bsums) + the main split-32 dot.
// CHECK: call_opaque "__riscv_vwmacc_vx_i32m2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwmacc_vv_i32m2"
// The end-of-block dual d/dmin fold: vfmacc main + vfnmsac MIN.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vfnmsac_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall.
// NOWALL-NOT: redsum
