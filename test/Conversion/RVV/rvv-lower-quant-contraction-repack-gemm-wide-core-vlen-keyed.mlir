// GAP-P1 -- the WITHIN-repack wide-vs-fractional integer-core LAYOUT variant is
// keyed on the VLEN CAPABILITY FACT (never a board name). The abstract q4_0 PREFILL
// tcrv_rvv.quant_contraction op auto-selects Repack at every VLEN tier (fact 3
// keeps Repack for ANY prefill), so the SAME module reaches the repack-GEMM bridge
// on each tier and the ONLY thing that moves is the core LMUL rung:
//
//   * VLEN128 (rv64gcv => Zvl128b => 128): the fractional strip is 8 lanes
//     (deriveRepackHalfLanes(128)=8 < weight_interleave 16), so the fractional mf2
//     default is the RIGHT fit here (two 8-lane strips cover one 16-block group; the
//     whole-LMUL register holds no more, so widening buys nothing). Stays on the
//     byte-identical mf2 form: half_lanes 8, NO integer_core_lmul, columnsPerPass 4.
//     RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=MF2IR
//     RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MF2EMIT
//
//   * VLEN256 (rv64gcv_zvl256b => 256): the ONLY delta from the rv64gcv line above
//     is the `_zvl256b` VLEN capability token -- NOT a board name -- yet it flips to
//     the WHOLE-LMUL m1 wide core: at VLEN256 the fractional mf2 register already
//     holds a whole 16-block group in HALF the register file, so the fractional
//     core is UNDER-fed and the whole-LMUL m1 register (twice the lanes) is the
//     fully-fed wide variant. half_lanes 16, integer_core_lmul = "m1",
//     columnsPerPass 1 (one f32m4 fold). This is the GAP-P1 fix: the wide variant
//     already existed + emits, but was keyed only on the RVV0.7 generation, never
//     on VLEN, so VLEN256+RVV1.0 fell to the under-fed mf2 core.
//     RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=M1IR
//     RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=M1EMIT
//
//   * RVV0.7 (rv64gc_xtheadvector, VLEN128 floor): the GENERATION capability fact
//     ALSO pins the same whole-LMUL m1 wide core (0.7.1 has no fractional LMUL), so
//     the wide variant is reached by EITHER capability fact (VLEN >= 256 OR RVV0.7)
//     -- proving the axis is a capability-fact key, not a board/march-name switch.
//     Byte-identical emit to the VLEN256 m1 form (same kernel body, checked by
//     M1EMIT reuse).
//     RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gc_xtheadvector | FileCheck %s --check-prefix=M1IR
//     RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=M1EMIT
//
// The variant is NUMERICALLY EQUIVALENT by construction: the m1 whole-LMUL core
// reads BYTE-IDENTICAL 16-way-interleaved repacked data and produces the SAME
// kernel body the RVV0.7 m1 arm already emits (rvv-to-emitc-typed-repack-gemm-loop-
// body-rvv07-full-body.mlir), differing from the mf2 form ONLY in the LMUL rung.
// This is an L2 within-repack schedule choice, NOT the repack-vs-block-dot ALGORITHM
// selection. NO e2e/perf claim -- the true VLEN256 read-wall utilization awaits ssh
// rvv / k1 board measurement; this fixture lit-VERIFIES the capability-keyed
// SELECTION + emit, it does NOT run the kernel.

module {
  tcrv.exec.kernel @ggml_gemm_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_gemm_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %nc = tcrv_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_gemm_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_gemm_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "prefill", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// ===== VLEN128: the fractional mf2 wide-NOT core (half_lanes 8, columnsPerPass 4).
// The loop body carries the capability-derived half_lanes 8; the four per-column
// dual-fp16 folds are the columnsPerPass-4 fractional shape.
// MF2IR: tcrv_rvv.typed_repack_gemm_loop_body
// MF2IR-SAME: half_lanes = 8 : i64
// MF2IR: tcrv_rvv.repack_gemm_dual_fp16_scale_fold
// MF2IR: tcrv_rvv.repack_gemm_dual_fp16_scale_fold
// MF2IR: tcrv_rvv.repack_gemm_dual_fp16_scale_fold
// MF2IR: tcrv_rvv.repack_gemm_dual_fp16_scale_fold
// MF2IR: tcrv_rvv.typed_repack_gemm_loop_yield

// The emitted C rides the FRACTIONAL chain (i8mf2 -> i16m1 -> i32m2 -> f32m2); the
// whole-LMUL m1/f32m4 spellings must NOT appear at VLEN128 RVV1.0. (The f32m2
// accumulator seed vfmv_v_f is emitted at the column-group head, ahead of the inner
// block loop's i8mf2 sub-load, so it is checked first.)
// MF2EMIT: call_opaque "__riscv_vfmv_v_f_f32m2"
// MF2EMIT: call_opaque "__riscv_vle8_v_i8mf2"
// MF2EMIT: call_opaque "__riscv_vwmacc_vx_i16m1"
// MF2EMIT: call_opaque "__riscv_vfmacc_vv_f32m2"
// MF2EMIT: call_opaque "__riscv_vse32_v_f32m2"
// MF2EMIT-NOT: __riscv_vle8_v_i8m1
// MF2EMIT-NOT: f32m4

// ===== VLEN256 / RVV0.7: the WHOLE-LMUL m1 wide core (half_lanes 16, one column).
// The loop body pins integer_core_lmul = "m1" and half_lanes 16; the single
// dual-fp16 fold is the columnsPerPass-1 whole-LMUL shape.
// M1IR: tcrv_rvv.typed_repack_gemm_loop_body
// M1IR-SAME: half_lanes = 16 : i64
// M1IR-SAME: integer_core_lmul = "m1"
// M1IR: tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot
// M1IR-SAME: integer_core_lmul = "m1"
// M1IR: tcrv_rvv.repack_gemm_dual_fp16_scale_fold
// M1IR-SAME: integer_core_lmul = "m1"
// M1IR: tcrv_rvv.typed_repack_gemm_loop_yield

// The emitted C rides the WHOLE-LMUL chain (i8m1 -> i16m2 -> i32m4 -> f32m4); NO
// fractional-LMUL spelling anywhere -- the fully-fed wide variant.
// M1EMIT: call_opaque "__riscv_vle8_v_i8m1"
// M1EMIT: call_opaque "__riscv_vwmacc_vx_i16m2"
// M1EMIT: call_opaque "__riscv_vwadd_vv_i32m4"
// M1EMIT: call_opaque "__riscv_vfwmul_vf_f32m4"
// M1EMIT: call_opaque "__riscv_vfmacc_vv_f32m4"
// M1EMIT: call_opaque "__riscv_vse32_v_f32m4"
// M1EMIT-NOT: mf2
