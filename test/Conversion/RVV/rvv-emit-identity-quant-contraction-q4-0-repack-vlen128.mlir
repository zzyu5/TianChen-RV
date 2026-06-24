// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=MF2
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv --tcrv-rvv-lower-to-emitc | FileCheck %s --check-prefix=NOWHOLE

// OPTION-2 M1 -- EMIT-IDENTITY on the SUPPORTED RVV1.0 regime (march=rv64gcv =>
// Zvl128b => VLEN128). The abstract, algorithm-UNCOMMITTED
// tcrv_rvv.quant_contraction op (q4_0 / decode) is AUTO-LOWERED by
// --tcrv-rvv-lower-quant-contraction at rv64gcv: the in-compiler selection picks
// REPACK (deriveMinimumVLEN(rv64gcv)=128 => the q4_0-vlen128-decode keep) and the
// C1 bridge realizes the real tcrv_rvv.repack_gemv_q4_0_q8_0 op (carrying the
// block_q4_0x16 x16 facts 288/16/32, half_lanes=8 => mf2, NO integer_core_lmul,
// PLUS the 4 emitter-INERT audit attrs contraction_algorithm/path_materialization/
// path_selection_reason/weight_layout_contract). --tcrv-rvv-lower-to-emitc then
// lowers it to the mf2/half_lanes=8 two-8-lane-halves repack-GEMV kernel.
//
// THIS proves the COMPILER now AUTO-SELECTS the repack (previously hand-chosen by
// authoring the concrete repack op directly in the input IR) and auto-emits the
// repack kernel SHAPE. The companion EMIT-IDENTITY byte-diff (this auto-selected
// emit vs the DIRECT repack-op emit carrying the SAME x16 facts but WITHOUT the
// audit attrs) is byte-identical -- proving the audit attrs are emitter-inert AND
// the auto-selected path emits EXACTLY the kernel hand-authoring the repack op
// produces. That byte-diff is run as a host harness step in
// option2-M1-emit-identity-FINDING.md (kernelA.cpp vs kernelB.cpp, same sha256).
//
// SCOPE: EMIT-IDENTITY ONLY. This is the SUPPORTED RVV1.0 mf2 form, NOT the
// dropped RVV0.7 (xtheadvector / whole-LMUL m1) regime. NO perf/e2e claim -- the
// kernel is lit-emitted, NOT run.

module {
  tcrv.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    tcrv.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    tcrv.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = tcrv_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !tcrv_rvv.runtime_abi_value
      %bs = tcrv_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %bx = tcrv_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !tcrv_rvv.runtime_abi_value
      %by = tcrv_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = tcrv_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = tcrv_rvv.setvl %n {lmul = "m1", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_0_q8_0, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_0_q8_0_kernel", status = "selected-lowering-boundary"} {
        %dot = tcrv_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}

// The abstract op is GONE; the auto-selected repack-GEMV is lowered to the SAME
// repack kernel the direct repack op produces (the audit attrs are emitter-inert).
// CHECK-NOT: tcrv_rvv.quant_contraction
// CHECK-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0 %
// CHECK-NOT: tcrv_rvv.q4_0_q8_0_block_dot
// CHECK-NOT: unrealized_conversion_cast
// The repack-GEMV export signature (the SAME 8-value ABI wrapper the abstract op
// carried: n,s,bs,vx,bx,vy,by,nrc -- unused values stay as params).
// CHECK: emitc.func @tcrv_emitc_ggml_vec_dot_q4_0_q8_0_kernel_ggml_vec_dot_q4_0_q8_0(
// The per-group weight base vx + x*nb*288 (block_q4_0x16 stride 288).
// CHECK: literal "288"
// The two 8-lane f32m2 accumulators (rows 0..7, 8..15) -- the mf2/half_lanes=8 form.
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// CHECK: call_opaque "__riscv_vfmv_v_f_f32m2"
// The two disjoint repacked i8mf2 sub-loads then lane-wise vwmacc accumulate (NO
// cross-lane reduction wall).
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vle8_v_i8mf2"
// CHECK: call_opaque "__riscv_vwmacc_vx_i16m1"
// The single-column two-half vector store vse32_v_f32m2.
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: call_opaque "__riscv_vse32_v_f32m2"
// CHECK: return

// The SUPPORTED RVV1.0 fractional form: the i8mf2 strip is PRESENT (NOT the
// dropped RVV0.7 whole-LMUL i8m1/f32m4 form).
// MF2: call_opaque "__riscv_vle8_v_i8mf2"

// This is the RVV1.0 mf2 (two 8-lane halves) arm -- the whole-LMUL m1/f32m4
// spellings (the dropped RVV0.7 / xtheadvector regime) must NOT appear.
// NOWHOLE-NOT: __riscv_vfmv_v_f_f32m4
// NOWHOLE-NOT: __riscv_vle8_v_i8m1
// NOWHOLE-NOT: redsum
