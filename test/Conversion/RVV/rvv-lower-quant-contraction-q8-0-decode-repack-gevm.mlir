// P2 decode-debt -- the q8_0 DECODE (M=1 GEVM) front-door construction. The DECODE regime of
// the q8_0 PREFILL front door (rvv-emit-quant-contraction-q8-0-repack-gemm-prefill-vlen128.mlir):
// the SAME abstract, algorithm-UNCOMMITTED weft_rvv.quant_contraction request, only
// m_regime prefill -> decode.
//
// This CLOSES the A2-batch4 §4 "q8_0 decode = BLOCKED-ON-CONSTRUCTION (no clean GEVM leaf;
// q8_0 repack-gemv is dataflow-only/multi-module; gevm_q8_timing_driver uses the RETIRED
// monolithic repack_gemv_q8_0_q8_0; needs a q8_0 typed-region single-module source)" entry for
// the VLEN128 board: the front door carries kNibbleQ80ScaleModel
// ("dual-fp16-per-block-d_x.d_y-full-i8") and the decode request CONSTRUCTS the typed
// weft_rvv.typed_repack_gemv_loop_body region through the SHARED dual-fp16 fold brick + the
// q8_0 FULL-int8 core brick variant (weight_full_i8: vle8 i8 + per-position vwmul/vwadd_wv i32
// in-block accumulation, NO nibble decode, NO qh, NO offset). NO retired emitter, NO archived
// .inc. NO perf/e2e claim -- lit-emitted, not run.
//
// VLEN128 (rv64gcv => 128): the memory-bound q8_0 decode keeps repack -- its OWN selection
// reason (distinct from the q4_0 regime token), the GENUINE front-door decode path on rvv.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=VLEN128
//
// VLEN256 (rv64gcv_zvl256b => 256): FAILS CLOSED. The analytic prior declines repack,
// but there is NO q8_0
// block-dot decline path at all -- the block-dot identity lowering is q4_0-nibble-only, which
// would MISCOMPILE a full-int8 q8_0 weight as nibbles. So the pass emits a capability
// diagnostic rather than a wrong kernel. This is the STRUCTURAL reason the k1 q8_0 decode cell
// is not front-door constructible (a force-constructed what-if leaf is the only route, as
// A2-batch4 did for q4_0@k1). Pinned as a fail-closed contract:
// RUN: not weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b 2>&1 | FileCheck %s --check-prefix=VLEN256
//
// Full front-door export to C (the analytic leaf):
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC

module {
  weft.exec.kernel @ggml_vec_dot_q8_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q8_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q8_0", scale_model = "dual-fp16-per-block-d_x.d_y-full-i8", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 34 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_memory_bound = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// (VLEN128 tier) the abstract op is GONE; the typed repack-GEVM region is realized (x16), with
// the q8_0 memory-bound decode selection reason. The analytic resource prior
// supplies the mf2 two-strip accumulator schedule.
// VLEN128-NOT: weft_rvv.quant_contraction
// VLEN128: weft_rvv.typed_repack_gemv_loop_body
// VLEN128-SAME: half_lanes = 8 : i64
// VLEN128-SAME: weft_rvv.weight_layout_contract = "x16"

// (VLEN256 tier) FAIL-CLOSED: no q8_0 block-dot decline path exists; refuse rather than
// miscompile full-int8 weights as nibbles.
// VLEN256: error: {{.*}}quant_contraction requires a repack-affording capability

// (EMITC) the exported C leaf: block_q8_0x16 stride 544 (d[16]@0, int8 quants @32) over a PLAIN
// block_q8_0 activation (stride 34, qs @2); FULL-int8 load (NO vand/vsrl nibble decode),
// per-position vwmul_vx + vwadd_wv i32 accumulation, dual-fp16 d_x*d_y fold, NO min.
// EMITC: emitc.func @weft_emitc_ggml_vec_dot_q8_0_q8_0_kernel_ggml_vec_dot_q8_0_q8_0
// EMITC: literal "544"
// EMITC: call_opaque "__riscv_vle8_v_i8mf2"
// EMITC: call_opaque "__riscv_vwmul_vx_i16m1"
// EMITC: call_opaque "__riscv_vwadd_wv_i32m2"
// EMITC: call_opaque "__riscv_vfwmul_vf_f32m2"
// EMITC: call_opaque "__riscv_vfmacc_vv_f32m2"
// EMITC: call_opaque "__riscv_vse32_v_f32m2"
// EMITC-NOT: call_opaque "__riscv_vand_vx_u8mf2"
