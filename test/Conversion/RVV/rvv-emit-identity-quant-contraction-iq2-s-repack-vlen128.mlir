// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// G3 M4 iq2_s GRID CONSTRUCTION -- the DUAL-ls grid-codebook + EXPLICIT signs256 sign-plane
// sibling of iq2_xxs / iq2_xs, the LAST cell of the iq2 grid family. The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (iq2_s grid / decode) is AUTO-LOWERED
// by --weft-rvv-lower-quant-contraction at rv64gcv: REPACK is selected
// (block_dot_compute_heavy + VLEN128), and the C1 bridge lowerToRepackGemvGrid CONSTRUCTS the
// typed weft_rvv.typed_repack_gemv_loop_body REGION (fold_model "grid_sign_dualscale_eighth")
// carrying the SINGLE weft_rvv.repack_gemv_grid_core integer-core brick (decode_model
// "iq2_s"), reconstructing the block_iq2_sx16 x16 weight facts 1824/288/32/1312 + the plain
// block_q8_K activation facts 292/4, half_lanes=8 => mf2, numHalves==2. The FIXED 1024-entry
// iq2s_grid + DIRECT signs256 plane the abstract request DOES NOT carry are RECONSTRUCTED at
// emit as static const decls (a true CONSTRUCT-from-abstract). NO perf/e2e claim.

module {
  weft.exec.kernel @ggml_repack_gemv_iq2_s_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemv_iq2_s_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq2-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemv_iq2_s_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemv_iq2_s_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT dual grid request: PLAIN block_iq2_s (stride 82, qs @2) x PLAIN
        // block_q8_K (stride 292), qk 256, scale_model = the explicit-sign dual grid iq2_s
        // WHAT, block_dot_compute_heavy = true (routes REPACK). It carries NO grid / signs
        // plane -- the compiler RECONSTRUCTS them. The result is dead.
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq2_s", scale_model = "superblock-d.fp16-grid-explicitsign-dualscale-nomin-eighth", m_regime = "decode", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 82 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// The abstract op is GONE; the compiler CONSTRUCTED the dual grid typed_repack GEVM region
// (fold_model grid_sign_dualscale_eighth) carrying the grid core brick (decode_model iq2_s).
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemv_loop_body
// CONSTRUCT-SAME: fold_model = "grid_sign_dualscale_eighth"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: scale_model = "superblock-d.fp16-grid-explicitsign-dualscale-nomin-eighth"
// CONSTRUCT-SAME: weft_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1824
// The in-region block_index-tied grid integer-core BRICK, decode_model iq2_s + the
// RECONSTRUCTED grid / ls / sign byte offsets (the abstract request carried plain facts).
// CONSTRUCT: weft_rvv.repack_gemv_grid_core
// CONSTRUCT-SAME: decode_model = "iq2_s"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 1312

// ============================ EMISSION (byte-exact shape) ===================
// The constructed region lowers to the iq2_s GEVM kernel (byte-exact to the retired direct
// emitter -- the SAME emit as rvv-to-emitc-repack-gemv-iq2-s-q8-K).
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemv_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemv_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemv_iq2_s_q8_K_kernel_ggml_repack_gemv_iq2_s_q8_K(
// The FIXED 1024-entry grid + DIRECT signs256 plane decls (RECONSTRUCTED by the compiler).
// CHECK: verbatim "static const int64_t weft_iq2s_grid[1024] = {0x0808080808080808ULL
// CHECK: verbatim "static const int8_t weft_iq2s_signs256[2048] = {1
// Per-group weight base vx + x*nb*1824 (block_iq2_sx16 stride 1824).
// CHECK: literal "1824"
// The plain q8_K activation stride 292.
// CHECK: literal "292"
// The REAL memory grid GATHER: a u16 assembled index strip (vle16, NO vzext) -> vsll ->
// vluxei16.
// CHECK: call_opaque "__riscv_vle16_v_u16m1"
// CHECK: call_opaque "__riscv_vsll_vx_u16m1"
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// The sign fold onto grid + the i32 in-block dot + the DUAL per-sub-block ls-scale vmacc.
// CHECK: call_opaque "__riscv_vmul_vv_i8mf2"
// CHECK: call_opaque "__riscv_vwmul_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vsext_vf4_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The end-of-block fold + the iq2 0.125 factor + vse32.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The grid + sign decode are MEMORY gathers, NOT a register vrgather; the block-as-lane
// repack erases the cross-lane reduction wall; the grid fold has NO min.
// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
