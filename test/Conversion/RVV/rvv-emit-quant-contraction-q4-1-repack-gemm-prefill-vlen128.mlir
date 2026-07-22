// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=MIN
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOMF2

// G3-lode-flat 曳光弹 -- q4_1 gemm_tile (PREFILL) FRONT-DOOR construction on the
// SUPPORTED RVV1.0 regime (march=rv64gcv => Zvl128b => VLEN128). The abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction op (q4_1 / PREFILL) is
// AUTO-LOWERED: the in-compiler selection picks REPACK and -- because m_regime ==
// "prefill" -- the C1 bridge CONSTRUCTS the typed weft_rvv.typed_repack_gemm_loop_body
// REGION carrying the SHARED q4_0 GEMM bricks -- the one-strip N-column integer CORE
// (repack_gemm_lane_wise_q4_x_i8_dot, stamping weight_nibble_unsigned = the q4_1 RAW
// nibble decode) + the four per-column dual-fp16 scale FOLDs
// (repack_gemm_dual_fp16_scale_fold, stamping the per-column MIN-fold offset pair
// weight_min_byte_offset=32 / activation_sum_byte_offset=8). It reconstructs the
// block_q4_1x16 x16 weight facts (stride 320, nibbles @64, m strip @32) AND the
// block_q8_1x4 INTERLEAVED activation facts (stride 144, quants @16, per-column s_y
// @8), MATERIALIZES the two GEMM ABI values (nr, bs). NO perf/e2e claim.

module {
  weft.exec.kernel @ggml_gemm_q4_1_q8_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_gemm_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 20 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected q4_1 repack-GEMM is lowered through the
// SAME front door as q4_0.
// CHECK-NOT: weft_rvv.quant_contraction
// CHECK-NOT: weft_rvv.repack_gemm_q4_1_q8_1 %
// CHECK-NOT: weft_rvv.typed_repack_gemm_loop_body
// CHECK: emitc.func @weft_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1(
// The per-group activation base vy + y*nb*144 (block_q8_1x4 stride 144) and the
// per-group weight base vx + x*nb*320 (block_q4_1x16 stride 320).
// CHECK: literal "144"
// CHECK: literal "320"
// The 16-lane f32m4 accumulator set (r51g m1 flip; mf2 default was 4x8 f32m2, columnsPerPass 4).
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m4"
// The q4_1 UNSIGNED-nibble load + lane-wise vwmacc + lo/hi combine.
// CHECK: call_opaque "__riscv_vle8_v_u8m1"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m2"
// CHECK: call_opaque "__riscv_vwadd_vv_i32m4"
// The dual-fp16 scale fold, the MIN fold, and the 4x8 vector store.
// CHECK: call_opaque "__riscv_vfmacc_vv_f32m4"
// CHECK: call_opaque "__riscv_vse32_v_f32m4"
// CHECK: return

// The q4_1 min term: the SECOND vfwmul (m_x * s_y[c]) folded by vfadd per column.
// MIN: call_opaque "__riscv_vand_vx_u8m1"
// MIN: call_opaque "__riscv_vfadd_vv_f32m4"
// MIN-NOT: __riscv_vsll_vx_i8m1

// The board-MEASURED r51g m1 whole-LMUL form (q4_1 flipped) -- the mf2 half_lanes=8
// spellings must NOT appear, and NO cross-lane vredsum reduction wall.
// NOMF2-NOT: __riscv_vfmv_v_f_f32m2
// NOMF2-NOT: __riscv_vle8_v_u8mf2
// NOMF2-NOT: redsum
