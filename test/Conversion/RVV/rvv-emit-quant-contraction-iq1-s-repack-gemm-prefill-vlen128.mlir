// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-2 iq1_s PREFILL GEMM CONSTRUCTION -- the prefill half of the iq1_s landing. The
// abstract weft_rvv.quant_contraction (iq1_s grid / m_regime = "prefill") is AUTO-LOWERED at
// rv64gcv: selection picks REPACK, and lowerToRepackGemmGrid CONSTRUCTS the typed
// weft_rvv.typed_repack_gemm_loop_body REGION (fold_model "grid_ternary_delta_eighth")
// carrying the weft_rvv.repack_gemm_grid_core brick (decode_model "iq1_s"), reconstructing
// the block_iq1_sx16 x16 weight facts 1312/288/32/160 AND the INTERLEAVED block_q8_Kx4
// activation facts (stride 1168, quants @16 as pos*4+c, bsums @1040 as g16*4+c), plus
// MATERIALIZING the two GEMM ABI values (nr, bs) the abstract op does not carry.
//
// This is the "front door can construct a repack GEMM op" half of the C4a-2 acceptance:
// registry row + front door + emitter leaf + byte-exact oracle, together. Without the
// emitter leaf this construction would be exactly the "verifier accepts but nobody can
// lower" state C4a refused to ship.
//
// NO perf/e2e claim: C4a-2 never touched a board. Numeric correctness is gated byte-exactly
// (GEMM interleaved addressing included) by tools/oracle-repack/oracle_repack_iq1_s.cpp.

module {
  weft.exec.kernel @ggml_repack_gemm_iq1_s_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq1_s_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq1_s_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq1_s_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT dual grid PREFILL request: PLAIN block_iq1_s (stride 74, qs @2) x
        // PLAIN block_q8_K (stride 292), qk 256, m_regime = "prefill",
        // block_dot_compute_heavy = true (routes REPACK). It carries NO grid / signs plane and
        // NO nr/bs -- the compiler RECONSTRUCTS the planes + MATERIALIZES the GEMM ABI values.
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq1_s", scale_model = "superblock-d.fp16-grid-ternary-delta-singlescale-nomin-eighth", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 50 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// The INTERLEAVED block_q8_Kx4 activation + its bsums plane @1040 (group16-major /
// column-minor). iq1_s is the ONLY grid row that reads bsums at all.
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_bsums_byte_offset = 1040
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "grid_ternary_delta_eighth"
// CONSTRUCT-SAME: weft_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1312
// The in-region grid GEMM core BRICK. The weight_sign_byte_offset SLOT carries the +-1
// DELTA strip @160 (a TernaryDelta row has no sign plane).
// CONSTRUCT: weft_rvv.repack_gemm_grid_core
// CONSTRUCT-SAME: decode_model = "iq1_s"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_sign_byte_offset = 160

// ============================ EMISSION ======================================
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq1_s_q8_K_kernel_ggml_repack_gemm_iq1_s_q8_K(
// The RECONSTRUCTED 2048-entry ternary grid decl + the body's registry-DATA table pointer
// (the trailing quote makes this exact: a corrupted gridArrayName cannot satisfy it).
// CHECK: verbatim "static const uint64_t weft_iq1s_grid[2048] = {0xffffffffffffffffULL
// CHECK: literal "weft_iq1s_grid"
// The repacked weight stride 1312 + the INTERLEAVED q8_Kx4 activation stride 1168.
// CHECK: literal "1312"
// CHECK: literal "1168"
// The REAL ternary-grid GATHER (u16 index strip -> vsll -> vluxei16), NO sign gather.
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The DELTA term over the interleaved bsums: ls*delta then vmacc_vx by the bsum pair.
// CHECK: call_opaque "__riscv_vmul_vv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vx_i32m2"
// The DeltaGrid fold: cvt(sumi) + 0.125f*cvt(sumi1), vfmacc by d_x*d_y, vse32.
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// iq1_s has NO sign plane: neither an iq2-style signs table nor the vmul-onto-grid sign
// fold may appear. If either did, the emitter would be silently running an iq2 leaf.
// NOWALL-NOT: signs64
// NOWALL-NOT: signs256
// NOWALL-NOT: __riscv_vmul_vv_i8mf2
