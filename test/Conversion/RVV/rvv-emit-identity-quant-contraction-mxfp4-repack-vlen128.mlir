// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=E8M0
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 retirement_batch 4 mxfp4 codebook CONSTRUCTION -- the THIRD codebook decode family
// (the E8M0 sibling of iq4_nl), the codebook sibling of the q4_0 / ternary / K-quant /
// iq4_nl rvv-emit-identity-quant-contraction-*-repack proofs. The abstract,
// algorithm-UNCOMMITTED tcrv_rvv.quant_contraction op (mxfp4 codebook / decode) is
// AUTO-LOWERED by --tcrv-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection
// reads the STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived
// capability VLEN (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the
// committed codebook scale_model WHAT, so the C1 bridge lowerToRepackGemvCodebook CONSTRUCTS
// the typed tcrv_rvv.typed_repack_gemv_loop_body REGION (fold_model "codebook_flat_e8m0_scale")
// carrying the SINGLE tcrv_rvv.repack_gemv_codebook_core integer-core brick (decode_model
// "mxfp4"), reconstructing the block_mxfp4x16 x16 weight facts 272/16 (E8M0 strip @0) + the
// plain block_q8_0 activation facts 34/2, half_lanes=8 => mf2, numHalves==2, AND -- crucially
// -- RECONSTRUCTS the 16-entry DOUBLED-E2M1 int8 codebook (kvalues_mxfp4) the abstract request
// DOES NOT carry (a true CONSTRUCT-from-abstract).
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the codebook repack region from a
// REAL quant_contraction request (previously the typed region was authored by the monolithic
// tcrv_rvv.repack_gemv_mxfp4_q8_0 op in rvv-to-emitc-repack-gemv-mxfp4-q8-0.mlir). The
// constructed region lowers to the byte-exact mxfp4 GEVM kernel the retired monolithic
// emitRepackGemvMxfp4Q8 direct emitter produced. NO perf/e2e claim -- lit-emitted, NOT run.

module {
  tcrv.exec.kernel @ggml_repack_gemv_mxfp4_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemv_mxfp4_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "mxfp4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_mxfp4_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemv_mxfp4_q8_0_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT codebook request: PLAIN block_mxfp4 (stride 17, nibbles @1 after the
        // E8M0 exponent byte) x PLAIN block_q8_0 (stride 34), qk 32, scale_model = the mxfp4
        // codebook WHAT, block_dot_compute_heavy = true (routes REPACK). It carries NO
        // codebook -- the compiler RECONSTRUCTS kvalues_mxfp4. The result is dead (the
        // repacked lane-wise codebook GEVM sinks through the output pointer).
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "mxfp4", scale_model = "flat.e8m0-single-scale-codebook-nomin", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 17 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 1 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the codebook typed_repack GEVM region
// (fold_model codebook_flat_e8m0_scale) carrying the codebook core brick (decode_model
// mxfp4), NOT a hand-authored region.
// CONSTRUCT-NOT: tcrv_rvv.quant_contraction
// CONSTRUCT: tcrv_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: fold_model = "codebook_flat_e8m0_scale"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "flat.e8m0-single-scale-codebook-nomin"
// CONSTRUCT-SAME: tcrv_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: tcrv_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 272
// The in-region block_index-tied codebook integer-core BRICK, decode_model mxfp4 + the
// RECONSTRUCTED 16-entry doubled-e2m1 codebook (the abstract request carried none).
// CONSTRUCT: tcrv_rvv.repack_gemv_codebook_core
// CONSTRUCT-SAME: codebook = array<i8: 0, 1, 2, 3, 4, 6, 8, 12, 0, -1
// CONSTRUCT-SAME: decode_model = "mxfp4"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the mxfp4 GEVM kernel (byte-exact to the retired direct
// emitter -- the SAME emit as rvv-to-emitc-repack-gemv-mxfp4-q8-0).
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemv_codebook_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemv_mxfp4_q8_0_kernel_ggml_repack_gemv_mxfp4_q8_0(
// The 16-entry doubled-E2M1 int8 fp4 codebook decl (RECONSTRUCTED by the compiler).
// CHECK: verbatim "static const int8_t tcrv_mxfp4_repack_kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1
// Per-group weight base vx + x*nb*272 (block_mxfp4x16 stride 272).
// CHECK: literal "272"
// The two 8-lane f32m2 accumulators (mf2 / half_lanes=8).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The plain q8_0 activation base al = a + l*34 (stride 34), inside the block loop.
// CHECK: literal "34"
// The REAL memory codebook GATHER (vzext -> vluxei16), NOT a register vrgather.
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The i32 in-block dot (vwmul + vwadd_wv).
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// The E8M0 scale reconstruction 2^(e-128), then dC = scale_x * (float)y.d (vfmul_vf).
// E8M0: call_opaque "__riscv_vzext_vf4_u32m2"
// E8M0: literal "0x1F"
// E8M0: call_opaque "__riscv_vand_vx_u32m2"
// E8M0: literal "0x00200000"
// E8M0: call_opaque "__riscv_vmv_v_x_u32m2"
// E8M0: call_opaque "__riscv_vsll_vv_u32m2"
// E8M0: call_opaque "__riscv_vsub_vx_u32m2"
// E8M0: literal "23"
// E8M0: call_opaque "__riscv_vsll_vx_u32m2"
// E8M0: call_opaque "__riscv_vmsltu_vx_u32m2_b16"
// E8M0: call_opaque "__riscv_vmerge_vvm_u32m2"
// E8M0: call_opaque "__riscv_vreinterpret_v_u32m2_f32m2"
// E8M0: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfcvt_f_x_v_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The codebook decode is a MEMORY gather, NOT a register vrgather; the E8M0 scale is
// reconstructed by bit-arithmetic, NOT an fp16 delta (no vle16/vfwmul weight-scale load);
// the block-as-lane repack erases the cross-lane reduction wall; the flat codebook fold has
// NO min.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
// NOWALL-NOT: vfwmul
// NOWALL-NOT: vle16
