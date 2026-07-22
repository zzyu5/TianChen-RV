// OPTION-2 STAGE B + C1 -- the IN-COMPILER capability/resource-aware
// contraction-PATH SELECTION (B) and the in-IR BRIDGE that LOWERS a
// repack-SELECTED request to a REAL repack op + DECLARES its weight-layout
// OUTPUT CONTRACT (C1). The --weft-rvv-lower-quant-contraction pass DERIVES the
// target VLEN from the selected -march (deriveMinimumVLEN -- the SAME capability
// authority every other capability-gated pass uses; the op's advisory min_vlen
// attr is NOT the source) and calls the pure, branch-free, capability-fact-driven
// selectContractionAlgorithm. The decision is stamped as INERT audit attrs.
//
// THE C1 CHANGE (was Option (i): byte-identical block-dot on every path): the
// abstract op carries PLAIN stride-18 weights; the concrete repack target needs
// pre-interleaved block_q4_0x16 weights (stride 288) that are a stage-C/system
// materialization. C1 CONSTRUCTS the repack-SELECTED cell as the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (the monolithic
// weft_rvv.repack_gemv_q4_0_q8_0 op is retired) carrying the x16 facts (288/16/32,
// half_lanes) AND the DECLARED OUTPUT CONTRACT weft_rvv.weight_layout_contract = "x16" -- the
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
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv | FileCheck %s --check-prefix=VLEN128
//
// VLEN256 (rv64gcv_zvl256b => 256, the K1 decode cell): q4_0 decode -> BLOCK-DOT
// SELECTED (declined), fully realized here. [G8 六.3] the decline is now PER-FORMAT
// MEASURED (kRepackVlen256DecodeMeasurements): q4_0 carries a board-measured-NEGATIVE
// row (0.74x LOSS), distinct from q5_0/q5_1 which measured BENEFICIAL at this SAME cell
// (1.190x/1.306x) and SELECT repack -- see
// rvv-lower-quant-contraction-vlen256-decode-per-format-measured.mlir.
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction=march=rv64gcv_zvl256b | FileCheck %s --check-prefix=VLEN256
//
// DEFAULT -march "" => deriveMinimumVLEN 0 => no capability => fact 3 false =>
// BLOCK-DOT (the honest no-capability behavior; this is what keeps the existing
// rvv-to-emitc-quant-contraction-q4-0-block-dot.mlir emitted C unchanged).
// RUN: weft-opt %s --weft-rvv-lower-quant-contraction | FileCheck %s --check-prefix=DEFAULT

module {
  weft.exec.kernel @ggml_vec_dot_q4_0_q8_0_kernel {
    weft.exec.capability @rvv {id = "rvv", kind = "isa-vector", status = "available"}
    weft.exec.variant @ggml_vec_dot_q4_0_q8_0 attributes {origin = "rvv-plugin", requires = [@rvv], weft_rvv.policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>} {
      %n = weft_rvv.runtime_abi_value {c_name = "n", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "n", role = "runtime-element-count"} : index
      %s = weft_rvv.runtime_abi_value {c_name = "s", c_type = "float *", ownership = "target-export-abi-owned", purpose = "out", role = "output-buffer"} : !weft_rvv.runtime_abi_value
      %bs = weft_rvv.runtime_abi_value {c_name = "bs", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bs", role = "output-stride"} : index
      %vx = weft_rvv.runtime_abi_value {c_name = "vx", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q4-weight", role = "lhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %bx = weft_rvv.runtime_abi_value {c_name = "bx", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "bx", role = "lhs-input-stride"} : index
      %vy = weft_rvv.runtime_abi_value {c_name = "vy", c_type = "const uint8_t *", ownership = "target-export-abi-owned", purpose = "q8-act", role = "rhs-input-buffer"} : !weft_rvv.runtime_abi_value
      %by = weft_rvv.runtime_abi_value {c_name = "by", c_type = "size_t", ownership = "target-export-abi-owned", purpose = "by", role = "rhs-input-stride"} : index
      %nrc = weft_rvv.runtime_abi_value {c_name = "nrc", c_type = "int32_t", ownership = "target-export-abi-owned", purpose = "nrc", role = "rhs-scalar-value"} : i32
      %vl = weft_rvv.setvl %n {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} : index -> !weft_rvv.vl
      weft_rvv.with_vl %vl attributes {lmul = "m1", policy = #weft_rvv.policy<tail = agnostic, mask = agnostic>, sew = 32 : i64} {
        %dot = weft_rvv.quant_contraction %vx, %vy, %s, %n, %bs, %vl {quant = "q4_0", scale_model = "dual-fp16-per-block-d_x.d_y", m_regime = "decode", qk = 32 : i64, weight_layout = "plain", weight_block_stride = 18 : i64, activation_block_stride = 34 : i64, quant_byte_offset = 2 : i64, activation_high_byte_offset = 16 : i64, block_dot_compute_heavy = true} : !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, !weft_rvv.runtime_abi_value, index, index, !weft_rvv.vl -> !weft_rvv.vector<i32, "m1">
      } : !weft_rvv.vl
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

// (VLEN128 tier) the repack-SELECTED cell is REALIZED as the typed
// weft_rvv.typed_repack_gemv_loop_body REGION (C1, the SOLE representation now the
// monolithic weft_rvv.repack_gemv_q4_0_q8_0 op is retired), carrying the
// block_q4_0x16 x16 facts the verifier pins (288/16/32, half_lanes 8 at VLEN128)
// AND the DECLARED OUTPUT CONTRACT weight_layout_contract = "x16", with the two
// decomposed inner bricks (the per-block lane-wise integer CORE + the numHalves==2
// per-strip dual-fp16 scale FOLDs) + the loop yield constructed in-region. NO
// block-dot op, NO monolithic repack op, NO leftover abstract op.
// VLEN128-NOT: weft_rvv.quant_contraction
// VLEN128-NOT: weft_rvv.q4_0_q8_0_block_dot
// VLEN128-NOT: weft_rvv.repack_gemv_q4_0_q8_0
// VLEN128: weft_rvv.typed_repack_gemv_loop_body
// VLEN128-SAME: half_lanes = 16 : i64
// [档 C#9 full-LMUL[B]] the accumulator-LMUL selection reason (m1/mf2 provenance,
// previously discarded): r51g [GAP-P1]-loosen board-MEASURED q4_0 => WIDE m1 chain
// (GEVM 2.3-2.5x / GEMM 1.24x faster, spill-free, byte-exact; see FINDING).
// VLEN128-SAME: weft_rvv.weight_layout_contract = "x16"
// VLEN128-SAME: weight_block_stride = 288 : i64
// VLEN128-SAME: weight_interleave = 16 : i64
// VLEN128-SAME: weight_quant_byte_offset = 32 : i64
// The decomposed inner region bricks are CONSTRUCTED (not test-authored): the
// per-block lane-wise integer CORE, the dual-fp16 scale FOLD, and the loop yield.
// r51g board-measured m1 => ONE 16-lane strip (half_lanes 16) => a SINGLE fold
// (the mf2 default carried two 8-lane-strip folds).
// VLEN128: weft_rvv.repack_lane_wise_q4_x_i8_dot
// VLEN128: weft_rvv.repack_dual_fp16_scale_fold
// VLEN128: weft_rvv.typed_repack_gemv_loop_yield

// VLEN256-NOT: weft_rvv.quant_contraction
// VLEN256-NOT: weft_rvv.repack_gemv_q4_0_q8_0
// VLEN256: weft_rvv.q4_0_q8_0_block_dot

// DEFAULT-NOT: weft_rvv.quant_contraction
// DEFAULT-NOT: weft_rvv.repack_gemv_q4_0_q8_0
// DEFAULT: weft_rvv.q4_0_q8_0_block_dot
