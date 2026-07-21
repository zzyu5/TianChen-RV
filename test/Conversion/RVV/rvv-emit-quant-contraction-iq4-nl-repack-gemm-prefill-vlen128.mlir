// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M2 iq4_nl codebook CONSTRUCTION (PREFILL/GEMM) -- the codebook sibling of the q4_0 /
// ternary / K-quant GEMM construction proofs. The abstract weft_rvv.quant_contraction op
// (iq4_nl codebook / prefill) auto-lowers at rv64gcv: block_dot_compute_heavy=true + VLEN 128
// route REPACK, m_regime "prefill" selects the GEMM granularity, and the committed codebook
// scale_model keys the C1 bridge lowerToRepackGemmCodebook, which CONSTRUCTS the typed
// weft_rvv.typed_repack_gemm_loop_body REGION (fold_model "codebook_flat_single_scale")
// carrying the SINGLE weft_rvv.repack_gemm_codebook_core brick (decode_model "iq4_nl"),
// MATERIALIZES the nr/bs GEMM ABI values the abstract op does not carry, reconstructs the
// block_iq4_nlx16 weight facts (288/32) + the INTERLEAVED block_q8_0x4 activation facts
// (136/8), AND RECONSTRUCTS the 16-entry non-linear codebook (kvalues_iq4nl) the abstract
// request omits. The GEMM body ships PLAIN/UNTILED (iq4_nl already at the <=32-vreg cliff, S6
// a structural no-op). Byte-exact to the retired monolithic emitRepackGemmIq4NlQ80. NO
// perf/e2e claim -- lit-emitted, NOT run.

module {
  weft.exec.kernel @ggml_repack_gemm_iq4_nl_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq4_nl_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq4_nl_q8_0, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq4_nl_q8_0_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT codebook PREFILL request: PLAIN block_iq4_nl (stride 18, nibbles @2) x
        // PLAIN block_q8_0 (stride 34), qk 32, m_regime "prefill". It carries NO codebook and
        // NO nr/bs -- the compiler RECONSTRUCTS kvalues_iq4nl AND materializes nr/bs.
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq4_nl", scale_model = "flat.fp16-single-scale-codebook-nomin", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the codebook typed_repack GEMM region
// (fold_model codebook_flat_single_scale) carrying the codebook GEMM core brick (decode_model
// iq4_nl), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "nr"
// CONSTRUCT: weft_rvv.runtime_abi_value {c_name = "bs"
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track that
// order.
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 136
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "codebook_flat_single_scale"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "flat.fp16-single-scale-codebook-4col-nomin"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 288
// The in-region block_index + strip_row_offset tied codebook GEMM integer-core BRICK,
// decode_model iq4_nl + the RECONSTRUCTED codebook.
// CONSTRUCT: weft_rvv.repack_gemm_codebook_core
// CONSTRUCT-SAME: codebook = array<i8: -127
// CONSTRUCT-SAME: decode_model = "iq4_nl"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the iq4_nl GEMM kernel (byte-exact to the retired direct
// emitter -- the SAME emit as rvv-to-emitc-repack-gemm-iq4-nl-q8-0; PLAIN untiled).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_codebook_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(
// The RECONSTRUCTED 16-entry non-linear int8 codebook decl.
// CHECK: verbatim "static const int8_t weft_iq4_nl_repack_kvalues[16] = {-127, -104
// The selected col_outer schedule forms the weight-group base first, then the
// activation-row-group base inside it.
// CHECK: literal "288"
// CHECK: literal "136"
// The per-column f32m2 accumulators (columnsPerPass == 4 folded in one pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The SHARED (amortized) memory codebook GATHER reused across the 4 columns.
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The i32 in-block dot (vwmul + vwadd_wv) + the per-column single fp16 scale fold.
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vfwmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall; the flat codebook GEMM
// fold has NO min; the decode is a MEMORY gather, NOT a register vrgather.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// NOWALL-NOT: vwmacc_vv_i32
