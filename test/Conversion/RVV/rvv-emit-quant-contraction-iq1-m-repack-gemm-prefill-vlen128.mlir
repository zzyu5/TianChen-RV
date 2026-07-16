// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=CONSTRUCT
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWALL

// C4a-3 iq1_m GRID PREFILL GEMM -- the prefill half of the fifth registry row. The
// abstract weft_rvv.quant_contraction (iq1_m grid / m_regime = "prefill") is AUTO-LOWERED at
// rv64gcv into the typed weft_rvv.typed_repack_gemm_loop_body REGION (fold_model
// "grid_ternary_delta_groupsum_eighth") carrying the weft_rvv.repack_gemm_grid_core brick
// (decode_model "iq1_m"), with the weight decode AMORTIZED across the 4 interleaved
// block_q8_Kx4 activation columns (stride 1168, quants @16 as pos*4+c).
//
// Unlike the iq1_s GEMM sibling there is NO bsums plane on either side: iq1_m's delta term
// needs per-GROUP-of-8 activation sums, and block_q8_K's bsums are sums over groups of
// SIXTEEN, so a bsums entry cannot express it (the two 8-groups inside one entry carry
// INDEPENDENT delta signs). The group sums are accumulated IN-KERNEL, per column, from the
// same interleaved quants the grid dot already reads.
//
// Ships PLAIN (untiled): C4a-3 built no tiled iq1_m variant, and the fold_model is
// deliberately absent from classifyTilingBottleneckShape's list so it falls to nullopt
// rather than being stamped with an unmeasured AlreadyLean prior.
//
// NO perf/e2e claim of any kind: C4a-3 is a CONSTRUCTION line and never touched a board.
// Numeric correctness is gated separately + byte-exactly (GEMM interleaved addressing
// included) by tools/oracle-repack/oracle_repack_iq1_m.cpp.

module {
  weft.exec.kernel @ggml_repack_gemm_iq1_m_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_iq1_m_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "iq1-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_repack_gemm_iq1_m_q8_K, sew = 32 : i64, source_kernel = "ggml_repack_gemm_iq1_m_q8_K_kernel", status = "selected-lowering-boundary"} {
        // The ABSTRACT ternary-delta-groupsum grid PREFILL request: PLAIN block_iq1_m
        // (stride 56, qs @0 -- NO inline d) x PLAIN block_q8_K (stride 292), qk 256,
        // m_regime = "prefill", block_dot_compute_heavy = true (routes REPACK).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "iq1_m", scale_model = "superblock-d.fp16-grid-ternary-delta-groupsum-dualscale-nomin-eighth", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 56 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 0 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// ============================ CONSTRUCTION proof ============================
// CONSTRUCT-NOT: weft_rvv.quant_contraction
// CONSTRUCT: weft_rvv.typed_repack_gemm_loop_body
// CONSTRUCT-SAME: activation_block_stride = 1168
// CONSTRUCT-SAME: activation_interleave = 4
// CONSTRUCT-SAME: fold_model = "grid_ternary_delta_groupsum_eighth"
// CONSTRUCT-SAME: half_lanes = 8
// CONSTRUCT-SAME: weft_rvv.contraction_algorithm = "repack"
// CONSTRUCT-SAME: weft_rvv.weight_layout_contract = "x16"
// CONSTRUCT-SAME: weight_block_stride = 1824
// The in-region grid GEMM core BRICK. The weight_sign_byte_offset SLOT carries the PER-GROUP
// +-1 DELTA strip @288 (a TernaryDelta row has no sign plane).
// CONSTRUCT: weft_rvv.repack_gemm_grid_core
// CONSTRUCT-SAME: decode_model = "iq1_m"
// CONSTRUCT-SAME: weight_ls_byte_offset = 32
// CONSTRUCT-SAME: weight_quant_byte_offset = 800
// CONSTRUCT-SAME: weight_sign_byte_offset = 288
// NO bsums plane, unlike the iq1_s GEMM sibling (which CHECKs
// activation_bsums_byte_offset = 1040). Per-16 bsums cannot express a per-8 group sum.
// CONSTRUCT-NOT: activation_bsums_byte_offset

// ============================ EMISSION ======================================
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_grid_core %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK-NOT: unrealized_conversion_cast
// CHECK: emitc.func @weft_emitc_ggml_repack_gemm_iq1_m_q8_K_kernel_ggml_repack_gemm_iq1_m_q8_K(
// The RECONSTRUCTED 2048-entry ternary grid decl + the body's registry-DATA table pointer
// (the trailing quote makes this exact: a corrupted gridArrayName cannot satisfy it).
// CHECK: verbatim "static const uint64_t weft_iq1m_grid[2048] = {0xffffffffffffffffULL
// CHECK: literal "weft_iq1m_grid"
// The repacked weight stride 1824 + the INTERLEAVED q8_Kx4 activation stride 1168.
// CHECK: literal "1824"
// CHECK: literal "1168"
// The REAL ternary-grid GATHER (u16 index strip -> vsll -> vluxei16), NO sign gather.
// CHECK: call_opaque "__riscv_vluxei16_v_i8mf2"
// CHECK: call_opaque "__riscv_vwadd_wv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vv_i32m2"
// The DELTA term over the IN-KERNEL per-column group-of-8 sums: ls*delta then vmacc_vx by
// the group sum.
// CHECK: call_opaque "__riscv_vmul_vv_i32m2"
// CHECK: call_opaque "__riscv_vmacc_vx_i32m2"
// The DeltaGridGroupSum fold: cvt(sumi1) + 0.125f*cvt(sumi2), vfmacc by d_x*d_y, vse32.
// CHECK: literal "0.125f"
// CHECK: call_opaque "__riscv_vfadd_vv_f32m2"
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// NOWALL-NOT: vrgather
// NOWALL-NOT: redsum
// NOWALL-NOT: vfnmsac
// iq1_m has NO sign plane: neither an iq2-style signs table nor the vmul-onto-grid sign
// fold may appear. If either did, the emitter would be silently running the iq2 DUAL-ls
// leaf -- a live hazard here, since iq1_m's ls arity really is Dual and only the
// fold-arith-first dispatch order keeps it off that leaf.
// NOWALL-NOT: signs64
// NOWALL-NOT: signs256
// NOWALL-NOT: __riscv_vmul_vv_i8mf2
// iq1_m reads NO bsums: no int16 scalar activation read anywhere in the body.
// NOWALL-NOT: const int16_t
