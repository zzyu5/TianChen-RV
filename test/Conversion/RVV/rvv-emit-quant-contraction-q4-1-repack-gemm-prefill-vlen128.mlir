// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MIN
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWHOLE

// G3-lode-flat 曳光弹 -- q4_1 gemm_tile (PREFILL) FRONT-DOOR construction on the
// SUPPORTED RVV1.0 regime (march=rv64gcv => Zvl128b => VLEN128). The abstract,
// algorithm-UNCOMMITTED tcrv_rvv.quant_contraction op (q4_1 / PREFILL) is
// AUTO-LOWERED: the in-compiler selection picks REPACK and -- because m_regime ==
// "prefill" -- the C1 bridge CONSTRUCTS the typed tcrv_rvv.typed_repack_gemm_loop_body
// REGION carrying the SHARED q4_0 GEMM bricks -- the one-strip N-column integer CORE
// (repack_gemm_lane_wise_q4_x_i8_dot, stamping weight_nibble_unsigned = the q4_1 RAW
// nibble decode) + the four per-column dual-fp16 scale FOLDs
// (repack_gemm_dual_fp16_scale_fold, stamping the per-column MIN-fold offset pair
// weight_min_byte_offset=32 / activation_sum_byte_offset=8). It reconstructs the
// block_q4_1x16 x16 weight facts (stride 320, nibbles @64, m strip @32) AND the
// block_q8_1x4 INTERLEAVED activation facts (stride 144, quants @16, per-column s_y
// @8), MATERIALIZES the two GEMM ABI values (nr, bs). NO perf/e2e claim.

module {
  tcrv.exec.kernel @ggml_gemm_q4_1_q8_1_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_gemm_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_gemm_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_gemm_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 20 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected q4_1 repack-GEMM is lowered through the
// SAME front door as q4_0.
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemm_q4_1_q8_1 %
// CHECK-NOT: tcrv_rvv.typed_repack_gemm_loop_body
// CHECK: emitc.func @tcrv_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1(
// The per-group activation base vy + y*nb*144 (block_q8_1x4 stride 144) and the
// per-group weight base vx + x*nb*320 (block_q4_1x16 stride 320).
// CHECK: literal "144"
// CHECK: literal "320"
// The 4x8 f32m2 accumulator set (columnsPerPass == 4 columns folded in ONE pass).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The q4_1 UNSIGNED-nibble load + lane-wise vwmacc + lo/hi combine.
// CHECK: call_opaque "__riscv_vle8_v_u8mf2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m2"
// The dual-fp16 scale fold, the MIN fold, and the 4x8 vector store.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The q4_1 min term: the SECOND vfwmul (m_x * s_y[c]) folded by vfadd per column.
// MIN: call_opaque "__riscv_vand_vx_u8mf2"
// MIN: call_opaque "__riscv_vfadd_vv_f32m2"
// MIN-NOT: __riscv_vsll_vx_i8mf2

// The SUPPORTED RVV1.0 mf2 form -- the dropped RVV0.7 whole-LMUL m1/f32m4 spellings
// must NOT appear, and NO cross-lane vredsum reduction wall.
// NOWHOLE-NOT: __riscv_vfmv_v_f_f32m4
// NOWHOLE-NOT: __riscv_vle8_v_i8m1
// NOWHOLE-NOT: redsum
