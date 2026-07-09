// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M2-后 iq4_xs codebook CONSTRUCTION (PREFILL/GEMM) -- the SUPER-BLOCK codebook sibling of
// the iq4_nl GEMM construction proof, COMPLETING the iq4 codebook pair. The abstract
// tcrv_rvv.quant_contraction op (iq4_xs codebook / prefill) auto-lowers at rv64gcv:
// block_dot_compute_heavy=true + VLEN 128 route REPACK, m_regime "prefill" selects the GEMM
// granularity, and the committed codebook scale_model keys the C1 bridge
// lowerToRepackGemmCodebook, which CONSTRUCTS the typed tcrv_rvv.typed_repack_gemm_loop_body
// REGION (fold_model "codebook_superblock_signed6_no_min") carrying the SINGLE SHARED
// tcrv_rvv.repack_gemm_codebook_core brick (decode_model "iq4_xs"), MATERIALIZES the nr/bs
// GEMM ABI values the abstract op does not carry, reconstructs the block_iq4_xsx16 weight
// facts (2176/128) + the scales_l @64 + scales_h @32 + n_subblocks 8 + the INTERLEAVED
// block_q8_Kx4 activation facts (1168/16), AND RECONSTRUCTS the 16-entry non-linear codebook
// (kvalues_iq4nl) the abstract request omits. The GEMM body ships PLAIN/UNTILED (iq4_xs at the
// <=32-vreg cliff, S6 a structural no-op). Byte-exact to the retired monolithic
// emitRepackGemmIq4XsQ8K. NO perf/e2e claim -- lit-emitted, NOT run.

module {
  tcrv.exec.kernel @ggml_repack_gemm_iq4_xs_q8_K_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_repack_gemm_iq4_xs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq4_xs_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq4_xs_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT codebook super-block PREFILL request: PLAIN block_iq4_xs (stride 136,
        // nibbles @8) x PLAIN block_q8_K (stride 292), qk 256, m_regime "prefill". It carries
        // NO codebook, NO scales offsets, and NO nr/bs -- the compiler RECONSTRUCTS
        // kvalues_iq4nl + the signed-6 scale facts AND materializes nr/bs.
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq4_xs", scale_model = "superblock.fp16-signed6-scale-codebook-nomin", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 136 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 8 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the codebook SUPER-BLOCK typed_repack GEMM
// region (fold_model codebook_superblock_signed6_no_min) carrying the SHARED codebook GEMM core
// brick (decode_model iq4_xs), and MATERIALIZED the nr/bs ABI values.
// CONSTRUCT-NOT: tcrv_rvv.quant_contraction
// The materialized runtime ABI values the internalized M-tiling nest needs.
// CONSTRUCT: tcrv_rvv.runtime_abi_value {c_name = "nr"
// CONSTRUCT: tcrv_rvv.runtime_abi_value {c_name = "bs"
// The loop-body attrs print alphabetically; the ordered CONSTRUCT-SAME checks track that.
// CONSTRUCT: tcrv_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "codebook_superblock_signed6_no_min"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: n_subblocks = 8
// CONSTRUCT-SAME: scale_model = "superblock.fp16-signed6-scale-codebook-4col-nomin"
// CONSTRUCT-SAME: tcrv_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 2176
// CONSTRUCT-SAME: weight_scales_byte_offset = 64
// CONSTRUCT-SAME: weight_scales_high_byte_offset = 32
// The in-region block_index + strip_row_offset tied codebook GEMM integer-core BRICK,
// decode_model iq4_xs + the RECONSTRUCTED codebook.
// CONSTRUCT: tcrv_rvv.repack_gemm_codebook_core
// CONSTRUCT-SAME: codebook = array<i8: -127
// CONSTRUCT-SAME: decode_model = "iq4_xs"

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the iq4_xs GEMM kernel (byte-exact to the retired direct
// emitter -- the SAME emit as rvv-to-emitc-repack-gemm-iq4-xs-q8-K; PLAIN untiled).
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemm_codebook_core %
// CHECK-NOT: tcrv_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @tcrv_emitc_ggml_repack_gemm_iq4_xs_q8_K_kernel_ggml_repack_gemm_iq4_xs_q8_K(
// The RECONSTRUCTED 16-entry non-linear int8 codebook decl.
// CHECK: verbatim "static const int8_t tcrv_iq4_xs_repack_kvalues[16] = {-127, -104
// The interleaved activation base vy + y*nb*1168 (block_q8_Kx4) and weight base vx +
// x*nb*2176 (block_iq4_xsx16).
// CHECK: literal "1168"
// CHECK: literal "2176"
// The per-column f32m2 accumulators (columnsPerPass == 4 folded in one pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The SHARED weight d strip is widened ONCE (vfwcvt) at the top of the block loop, reused
// across the 4 columns (amortized, UNLIKE the GEVM's per-block end-of-block widen).
// CHECK: call_opaque "__riscv_vfwcvt_f_f_v_f32m2"
// The SHARED (amortized) signed-6 scale unpack + memory codebook GATHER reused across cols.
// CHECK: call_opaque "__riscv_vsub_vx_i8mf2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The i32 sub-block dot (vwmul + vwadd_wv) + the SIGNED-scale vmacc fold.
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The per-column end-of-block fold (vfmul the widened weight d by d_y, vfmacc, NO min).
// CHECK: call_opaque "__riscv_vfmul_vf_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The block-as-lane repack erases the cross-lane reduction wall; iq4_xs is scale-ONLY (NO
// min: no vfnmsac); the decode is a MEMORY gather, NOT a register vrgather.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
