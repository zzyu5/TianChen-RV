// P2 decode-debt -- the q4_1 DECODE (M=1 GEVM) front-door construction. The q4_1 sibling of
// the q5_1 decode cell (rvv-lower-quant-contraction-q5-1-vlen256-decode-measured-repack.mlir)
// and the DECODE regime of the q4_1 PREFILL front door
// (rvv-emit-quant-contraction-q4-1-repack-gemm-prefill-vlen128.mlir): the SAME abstract,
// algorithm-UNCOMMITTED weft_rvv.quant_contraction request, only m_regime prefill -> decode.
//
// This CLOSES the A2-batch4 §4 "q4_1 decode = BLOCKED-ON-CONSTRUCTION (no clean weft-opt leaf
// source; dataflow uses the RETIRED monolithic repack_gemv_q4_1_q8_1; lower-to-emitc rejects
// exec.variant; front-door input missing)" entry: the front door carries kNibbleQ41ScaleModel
// ("dual-fp16-per-block-d_x.d_y-plus-min"), so the decode request CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body region through the SHARED q4_0 core + fold bricks --
// the core stamping weight_nibble_unsigned (RAW [0,15] nibble, NO offset-binary -8; the
// asymmetric bias lives in the separate MIN scale) and the fold stamping the single MIN offset
// pair (m_x @ +32 weight strip, s_y @ +2 activation). NO retired emitter, NO archived .inc.
// NO perf/e2e claim -- lit-emitted, not run.
//
// VLEN128 (rv64gcv => 128): repack via the capability/regime rule (half_lanes 8), the
// q4_0-vlen128 audit token -- the GENUINE front-door decode path on the rvv board.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=VLEN128
//
// VLEN256 (rv64gcv_zvl256b => 256): the format-blind selector DECLINES repack (the board-seeded
// kRepackVlen256DecodeMeasurements registry has NO q4_1 decode measurement -- the same decline
// q4_0/iq4_nl take; q5_0/q5_1 are kept only because they measured BENEFICIAL). But the decline
// branch is q4_0-ONLY: it builds a weft_rvv.q4_0_q8_0_block_dot still carrying q4_1's
// "...-plus-min" scale_model, and the q4_0 block-dot VERIFIER correctly REJECTS it (accepting it
// would silently DROP the m_x*s_y MIN term = a q4_1 miscompile). So the q4_1 VLEN256 decode
// decline FAILS CLOSED with a verifier diagnostic rather than miscompiling. This is the
// STRUCTURAL reason the k1 q4_1 decode cell is not front-door constructible: "decline to
// block-dot" is not expressible for a non-q4_0 flat format. Pinned as a fail-closed contract:
// RUN: not weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b 2>&1 | FileCheck %s --check-prefix=VLEN256
//
// Full front-door export to C (the measured leaf) is the two-pass form:
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

module {
  weft.exec.kernel @ggml_vec_dot_q4_1_q8_1_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q4_1_q8_1 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", origin = "rvv-plugin", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, required_capabilities = [@rvv], rvv_construction_protocol = "extension-family-construction-protocol.v1", selected_path_role = "dispatch case", selected_variant = @ggml_vec_dot_q4_1_q8_1, sew = 32 : i64, source_kernel = "ggml_vec_dot_q4_1_q8_1_kernel", status = "selected-lowering-boundary"} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_1", scale_model = "dual-fp16-per-block-d_x.d_y-plus-min", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 20 : i64, activation_block_stride = 36 : i64, quant_byte_offset = 4 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// (VLEN128 tier) the abstract op is GONE; the typed repack-GEVM region is realized (x16).
// VLEN128-NOT: weft_rvv.quant_contraction
// VLEN128: weft_rvv.typed_repack_gemv_loop_body
// VLEN128-SAME: half_lanes = 8 : i64
// VLEN128-SAME: weft_rvv.contraction_algorithm = "repack"
// VLEN128-SAME: weft_rvv.path_materialization = "realized"
// VLEN128-SAME: weft_rvv.path_selection_reason = "repack-kept-q4_0-vlen128-decode"
// VLEN128-SAME: weft_rvv.weight_layout_contract = "x16"

// (VLEN256 tier) FAIL-CLOSED: the q4_0-only decline route refuses the q4_1 plus-min scale_model
// instead of dropping the MIN term.
// VLEN256: error: 'weft_rvv.q4_0_q8_0_block_dot' op requires scale_model "dual-fp16-per-block-d_x.d_y" for the ggml Q4_0 x Q8_0 block dot-product route

// (EMITC) the exported C leaf: block_q4_1x16 stride 320 (d[16]@0, m[16]@32, nibbles @64) over
// a PLAIN block_q8_1 activation (stride 36); RAW-nibble decode (vand 0x0F / vsrl 0x04, NO -8),
// lane-wise vwmacc, dual-fp16 d_x*d_y fold + the vfadd MIN term.
// EMITC: emitc.func @weft_emitc_ggml_vec_dot_q4_1_q8_1_kernel_ggml_vec_dot_q4_1_q8_1
// EMITC: literal "320"
// EMITC: call_opaque "__riscv_vle8_v_u8mf2"
// EMITC: call_opaque "__riscv_vand_vx_u8mf2"
// EMITC: call_opaque "__riscv_vsrl_vx_u8mf2"
// EMITC: call_opaque "__riscv_vwmacc_vx_i16m1"
// EMITC: call_opaque "__riscv_vfwmul_vf_f32m2"
// EMITC: call_opaque "__riscv_vfmacc_vv_f32m2"
// EMITC: call_opaque "__riscv_vfadd_vv_f32m2"
// EMITC: call_opaque "__riscv_vse32_v_f32m2"
