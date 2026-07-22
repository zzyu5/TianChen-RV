// A-line c-axis debake (裁决7·阶段0) 坐实: the integer-core LMUL board value is a
// CAPABILITY FACT, never a baked default. The front door ALWAYS stamps
// integer_core_lmul on every wired repack leaf (RVVLowerQuantContraction.cpp builds it
// non-null = isM1 ? "m1" : "mf2" and unconditionally addAttribute's it), and the EmitC
// integer-core lowering now READS it FAIL-CLOSED (RVVToEmitCBlockQuantLinear.cpp: the 18
// former getIntegerCoreLmul().value_or("mf2") board-baked defaults are dead code and were
// deleted). This test pins BOTH halves: (a) the front door stamps the anchor on both
// boards (VLEN flip: no wired leaf drops it), and (b) an absent anchor fails closed --
// no silent "mf2" fractional-chain board value is synthesized.
//
// (1) Front-door stamp 坐实 @ VLEN128 (rv64gcv => 128): the CONSTRUCTED typed_repack GEMM
//     loop body carries integer_core_lmul = "mf2" (the RVV1.0 fractional-chain anchor) at
//     the capability-derived half_lanes 8.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=FD128
//
// (2) VLEN flip 坐实 @ VLEN256 (rv64gcv_zvl256b => 256): the SAME source stamps the SAME
//     integer_core_lmul anchor, now at half_lanes 16 -- the anchor is present on BOTH
//     boards (only half_lanes moves with the capability; the anchor never goes absent).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=FD256
//
// (3) With the anchor present the full front-door -> EmitC pipeline lowers cleanly (the
//     fail-closed read consumes the stamped capability fact).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc | FileCheck %s --check-prefix=EMITC
//
// (4) FAIL-CLOSED: strip ONLY the front-door integer_core_lmul stamp and re-run the SAME
//     EmitC pipeline -- it now REFUSES to legalize (the deleted value_or default no longer
//     silently synthesizes "mf2"). The verifier accepts the absent optional anchor, so the
//     refusal comes from the EmitC integer-core read, not the op verifier -- fail-closed by
//     construction. (The controlled delta vs run (3) is exactly the removed stamp.)
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | sed 's/integer_core_lmul = "mf2", //g' | not weft-opt --weft-rvv-lower-to-emitc 2>&1 | FileCheck %s --check-prefix=FAILCLOSED

module {
  weft.exec.kernel @ggml_repack_gemm_q4_K_q8_K_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_repack_gemm_q4_K_q8_K attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %nc = weft_rvv.runtime_abi_value {c_name = "nc", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "nc", role = "destination-byte-stride"} : index
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        // Abstract, algorithm-UNCOMMITTED K-quant q4_K PREFILL request; the front door
        // reads block_dot_compute_heavy + the derived VLEN and CONSTRUCTS the typed repack
        // GEMM region, stamping the capability facts (half_lanes + integer_core_lmul).
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %nc, %vl {quant = "q4_K", scale_model = "superblock-d.dmin-fp16-plus-bsums-min-8-subblocks", m_regime = "prefill", qk = 256 : i64, weight_layout = "plain", weight_block_stride = 144 : i64, activation_block_stride = 292 : i64, quant_byte_offset = 16 : i64, activation_high_byte_offset = 0 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
    }
  }
}

// (1) VLEN128: the front door stamps the fractional-chain anchor "mf2" at half_lanes 8.
// The loop-body attrs print alphabetically, so half_lanes immediately precedes the anchor.
// FD128: weft_rvv.typed_repack_gemm_loop_body
// FD128-SAME: half_lanes = 8 : i64
// FD128-SAME: integer_core_lmul = "mf2"

// (2) VLEN256: the SAME anchor is stamped, now at half_lanes 16 -- present on BOTH boards.
// FD256: weft_rvv.typed_repack_gemm_loop_body
// FD256-SAME: half_lanes = 16 : i64
// FD256-SAME: integer_core_lmul = "mf2"

// (3) The stamped anchor is READ (not defaulted): the pipeline lowers to a real EmitC leaf.
// EMITC: emitc.func @weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel
// EMITC: call_opaque

// (4) Stripping the anchor makes the SAME pipeline fail closed (no silent mf2 default).
// FAILCLOSED: error: failed to legalize operation 'weft.exec.variant'
