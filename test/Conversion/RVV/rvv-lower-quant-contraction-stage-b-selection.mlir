// OPTION-2 STAGE B + C1 -- the IN-COMPILER capability/resource-aware
// contraction-PATH SELECTION (B) and the in-IR BRIDGE that LOWERS a
// repack-SELECTED request to a REAL repack op + DECLARES its weight-layout
// OUTPUT CONTRACT (C1). The --tcrv-rvv-lower-quant-contraction pass DERIVES the
// target VLEN from the selected -march (deriveMinimumVLEN -- the SAME capability
// authority every other capability-gated pass uses; the op's advisory min_vlen
// attr is NOT the source) and calls the pure, branch-free, capability-fact-driven
// selectContractionAlgorithm. The decision is stamped as INERT audit attrs.
//
// THE C1 CHANGE (was Option (i): byte-identical block-dot on every path): the
// abstract op carries PLAIN stride-18 weights; the concrete repack target needs
// pre-interleaved block_q4_0x16 weights (stride 288) that are a stage-C/system
// materialization. C1 LOWERS the repack-SELECTED cell to the REAL
// tcrv_rvv.repack_gemv_q4_0_q8_0 op carrying the x16 facts (288/16/32, half_lanes)
// AND the DECLARED OUTPUT CONTRACT tcrv_rvv.weight_layout_contract = "x16" -- the
// ASSERTION the weight bytes are block_q4_0x16 (which some later layer, C3-C4,
// must make true). The BlockDot-SELECTED (decline) branch is UNCHANGED + emits
// the byte-identical block-dot body (proven by rvv-to-emitc-quant-contraction-
// q4-0-block-dot.mlir staying byte-identical). The repack op carries the SAME SSA
// weight pointer (the IR cannot tell a plain base from an x16 base); the contract
// is the bridge's declared requirement, NOT an e2e-correct run on plain weights
// (no producer authors this abstract op outside lit -- fixture-only, no
// miscompile). This fixture lit-VERIFIES the repack op + the contract appear; it
// does NOT run them. NO perf/e2e claim.
//
// The SAME q4_0-decode module is lowered TWICE at two -march tiers to prove the
// capability-fact-3 discrimination (VLEN128 -> repack op realized, VLEN256/K1 ->
// block-dot declined):
//
// VLEN128 (rv64gcv => Zvl128b => 128): q4_0 decode -> REPACK SELECTED + REALIZED
// as the real repack op carrying weight_layout_contract = "x16" (half_lanes 8).
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=VLEN128
//
// VLEN256 (rv64gcv_zvl256b => 256, the K1 decode cell that measured a 0.74x
// LOSS): q4_0 decode -> BLOCK-DOT SELECTED (declined), fully realized here.
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=VLEN256
//
// DEFAULT -march "" => deriveMinimumVLEN 0 => no capability => fact 3 false =>
// BLOCK-DOT (the honest no-capability behavior; this is what keeps the existing
// rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir emitted C unchanged).
// RUN: tcrv-opt %s --tcrv-rvv-lower-quant-contraction | FileCheck %s --check-prefix=DEFAULT

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

// The abstract quant_contraction op is GONE (lowered) on every tier. C1 flips the
// VLEN128 (repack-SELECTED) cell to a REAL repack op carrying the x16 contract;
// the VLEN256/DEFAULT (block-dot-SELECTED, decline) cells stay the byte-identical
// block-dot op + only the audit attrs differ.
//
// (Per-tier prose lives in the RUN-line block above so it cannot be mistaken for
// a CHECK directive.)

// (VLEN128 tier) the repack-SELECTED cell is REALIZED as the real repack-GEMV op
// (C1), carrying the block_q4_0x16 x16 facts the verifier pins (288/16/32,
// half_lanes 8 at VLEN128) AND the DECLARED OUTPUT CONTRACT weight_layout_contract
// = "x16". NO block-dot op, NO leftover abstract op.
// VLEN128-NOT: tcrv_rvv.quant_contraction
// VLEN128-NOT: tcrv_rvv.q4_0_q8_0_block_dot
// VLEN128: tcrv_rvv.repack_gemv_q4_0_q8_0
// VLEN128-SAME: half_lanes = 8 : i64
// VLEN128-SAME: tcrv_rvv.contraction_algorithm = "repack"
// VLEN128-SAME: tcrv_rvv.path_materialization = "realized"
// VLEN128-SAME: tcrv_rvv.path_selection_reason = "repack-kept-q4_0-vlen128-decode"
// VLEN128-SAME: tcrv_rvv.weight_layout_contract = "x16"
// VLEN128-SAME: weight_block_stride = 288 : i64
// VLEN128-SAME: weight_interleave = 16 : i64
// VLEN128-SAME: weight_quant_byte_offset = 32 : i64

// VLEN256-NOT: tcrv_rvv.quant_contraction
// VLEN256-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0
// VLEN256: tcrv_rvv.q4_0_q8_0_block_dot
// VLEN256-SAME: tcrv_rvv.contraction_algorithm = "block-dot"
// VLEN256-SAME: tcrv_rvv.path_materialization = "realized"
// VLEN256-SAME: tcrv_rvv.path_selection_reason = "block-dot-decline-q4_0-vlen256-decode-k1-loss"

// DEFAULT-NOT: tcrv_rvv.quant_contraction
// DEFAULT-NOT: tcrv_rvv.repack_gemv_q4_0_q8_0
// DEFAULT: tcrv_rvv.q4_0_q8_0_block_dot
// DEFAULT-SAME: tcrv_rvv.contraction_algorithm = "block-dot"
// DEFAULT-SAME: tcrv_rvv.path_materialization = "realized"
