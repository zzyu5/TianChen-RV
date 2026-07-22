// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M4 iq2_xxs GRID CONSTRUCTION -- the FIRST grid-codebook + sign-plane decode family,
// the grid sibling of the q4_0 / ternary / K-quant / iq4 codebook
// rvv-emit-identity-quant-contraction-*-repack proofs. The abstract, algorithm-UNCOMMITTED
// weft_rvv.quant_contraction op (iq2_xxs grid / decode) is AUTO-LOWERED by
// --weft-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection reads the
// STRUCTURED opponent facts (block_dot_compute_heavy = true) + the derived capability VLEN
// (rv64gcv => 128) and picks REPACK; the decode FAMILY is keyed off the committed grid
// scale_model WHAT, so the C1 bridge lowerToRepackGemvGrid CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (fold_model "grid_sign_single_scale_eighth")
// carrying the SINGLE weft_rvv.repack_gemv_grid_core integer-core brick (decode_model
// "iq2_xxs"), reconstructing the block_iq2_xxsx16 x16 weight facts 1184/160/32/672 + the
// plain block_q8_K activation facts 292/4, half_lanes=8 => mf2, numHalves==2. The FIXED
// 256-entry grid + DERIVED signs64 plane the abstract request DOES NOT carry are
// RECONSTRUCTED at emit as static const decls (a true CONSTRUCT-from-abstract).
//
// THIS proves the COMPILER now AUTO-SELECTS + CONSTRUCTS the grid repack region from a REAL
// quant_contraction request. The constructed region lowers to the byte-exact iq2_xxs GEVM
// kernel the retired monolithic emitRepackGemvIq2XxsQ8K produced. NO perf/e2e claim.

module {
  weft.exec.kernel @ggml_repack_gemv_iq2_xxs_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq2_xxs_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // The ABSTRACT grid request: PLAIN block_iq2_xxs (stride 66, qs @2) x PLAIN
        // block_q8_K (stride 292), qk 256, scale_model = the grid iq2_xxs WHAT,
        // block_dot_compute_heavy = true (routes REPACK). It carries NO grid / signs plane --
        // the compiler RECONSTRUCTS them. The result is dead (the repacked lane-wise grid
        // GEVM sinks through the output pointer).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq2_xxs", scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 66 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the grid typed_repack GEVM region
// (fold_model grid_sign_single_scale_eighth) carrying the grid core brick (decode_model
// iq2_xxs), NOT a hand-authored region.
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: fold_model = "grid_sign_single_scale_eighth"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-grid-sign-4bit-scale-nomin-eighth"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1184
// The in-region block_index-tied grid integer-core BRICK, decode_model iq2_xxs + the
// RECONSTRUCTED grid / ls / sign byte offsets (the abstract request carried plain facts).
// CONSTRUCT: weft_rvv.repack_gemv_grid_core
// CONSTRUCT-SAME: decode_model = "iq2_xxs"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 672

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the iq2_xxs GEVM kernel (byte-exact to the retired
// direct emitter -- the SAME emit as rvv-to-emitc-repack-gemv-iq2-xxs-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq2_xxs_q8_K_kernel_ggml_repack_gemv_iq2_xxs_q8_K(
// The FIXED 256-entry grid + DERIVED signs64 plane decls (RECONSTRUCTED by the compiler).
// CHECK: verbatim "static const int64_t weft_iq2xxs_grid[256] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t weft_iq2xxs_signs64[1024] = {1, 1, 1
// Per-group weight base vx + x*nb*1184 (block_iq2_xxsx16 stride 1184).
// CHECK: literal "1184"
// The plain q8_K activation stride 292.
// CHECK: literal "292"
// The REAL memory grid + sign GATHER (vzext -> vsll -> vluxei16), NOT a register vrgather.
// CHECK: call_opaque "__riscv_vzext_vf2_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The sign fold onto grid + the i32 in-block dot + the per-sub-block ls-scale vmacc.
// CHECK: call_opaque "__riscv_vmul_vv_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The end-of-block fold + the iq2_xxs 0.125 factor + vse32.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The grid + sign decode are MEMORY gathers, NOT a register vrgather; the block-as-lane
// repack erases the cross-lane reduction wall; the grid fold has NO min.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
