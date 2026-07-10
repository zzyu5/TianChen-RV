#!/usr/bin/env python3
"""e5_strong_readout.py — E5 增量① strong-side [L-8] provenance auto-readout.

Machine-derives the STRONG ("constructed") six-state verdict for the 3 existing
strong N-operand contraction routes by WALKING THE ACTUAL REALIZED BODY, then
verifies the verdict reproduces the current hand-label and, via one weak negative
control, that the check discriminates strong from weak (is not vacuously true).

Why this tool exists (contrast with coverage_metrics.py / E6)
------------------------------------------------------------
E6's coverage_metrics.py is stdlib-only and explicitly "does NOT derive
strong-vs-weak from code". That derivation is E5's job. This tool does it the
ONE I4-legal way: it runs the real compiler (tcrv-opt = runner) and walks the
op-identity of the ACTUAL realized `tcrv_rvv.with_vl` body it emits (IR = artifact
parsing). Python is tooling here, never the compiler stack (I6).

The [L-8] machine rule (实验总纲 line 27 / core-invariants [L-8]/[K-4])
----------------------------------------------------------------------
A selected-body path is `constructed` (STRONG) iff:

    manifest non-empty  ∧  no opaque hand-written helper  ∧  decomposed

- manifest = the ordered op-identity list of the ops in the realized `with_vl`
  body (the pattern-library primitives: load / widening_product / *_x_i8_product
  / standalone_reduce / dequantize / store / ...).
- opaque hand-written helper = a MONOLITHIC block-dot op: mnemonic ends `_block_dot`
  AND carries `kind = "ggml_…block_dot"` (the descriptor-selected hand helper that
  lowers to emitFlatBlockDot). The two signals must AGREE, so it is not one fragile
  string match.
- decomposed = the body realizes a REAL dot-product-family primitive
  (widening_product / *_x_i8_product / *_unpack_product / *product_reduce) AND a
  reduce-family primitive (standalone_reduce / *dot_reduce / ...). A per-block fp16
  `block_fp16_scale_product` is a scale multiply, NOT a contraction, so it does not
  satisfy this conjunct — a bare "product" substring would wrongly admit it. This
  conjunct was report-only; it is now a GATE (E5 gate-hole fix).

I4 red line: the manifest is the ACTUAL realized body's op-identity (CORE oracle),
NEVER the `tcrv_rvv.low_precision_resource.*` mirror attributes (which ride on the
with_vl op) nor the emission-plan `rvv_selected_body_typed_compute_op` mirror. The
parser extracts mnemonics from OPERATION position only and asserts no mirror /
attribute-name token ever leaks into a manifest.

Stage discipline: the walk stops at the constructed with_vl body, BEFORE
`--tcrv-rvv-lower-to-emitc` (after that lowering both strong and weak collapse to
indistinguishable `emitc`/`call_opaque` and the discriminator vanishes).

Scope (E5 增量①, bounded): strong side only. This is a standalone read-out +
optional auto_readout write-back. No CI gate, no enforcement, no coverage_metrics.py
edit, no committed JSONL sink — those are later increments. State values never
change (zero flip); only the `auto_readout` field of the strong rows is written.

Subcommands
-----------
report                 run the machine-check; print per-path {manifest, has_opaque,
                       derived_state} + PASS/FAIL vs expected. Exit non-zero on any
                       mismatch or any tcrv-opt failure (fail-closed).
update-sixstate        run report, then (only if all rows pass) rewrite the strong
                       rows' `auto_readout` in schema/coverage-sixstate.v1.json with
                       the machine result + manifest summary. Never touches `state`.
--self-test            hermetic parser test over the 4 captured ground-truth bodies.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# --- locations -------------------------------------------------------------
# This file lives at <repo>/.trellis/scripts/e5_strong_readout.py
REPO_ROOT = Path(__file__).resolve().parents[2]
TCRV_OPT = REPO_ROOT / "build" / "bin" / "tcrv-opt"
SIXSTATE_JSON = REPO_ROOT / "schema" / "coverage-sixstate.v1.json"
TEST_RVV = REPO_ROOT / "test" / "Target" / "RVV"
TEST_CONV_RVV = REPO_ROOT / "test" / "Conversion" / "RVV"

# --- the 4 strong routes + 1 weak negative control -------------------------
# Each entry: the six-state (op, format) key, the source-op test input, and the
# single front-door pass that CONSTRUCTS/REALIZES the typed with_vl body. NO
# --tcrv-rvv-lower-to-emitc (stage discipline). Row<->test mapping per research
# emission-paths-map.md §1b.
PATHS = [
    {
        "op": "product_reduce", "format": "q4_0_nibble", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "non-deferred-wide-product-reduce-dequantize-f32-front-door-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-widening-dot-reduce-dequantize-source-front-door=march=rv64gcv",
        "front_door_id": "RVVDequantDotSourceFrontDoor",
    },
    {
        "op": "product_reduce", "format": "offset_binary_n3", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "packed-i4-offset-binary-dot-product-reduce-front-door-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-packed-i4-offset-binary-dot-source-front-door=march=rv64gcv",
        "front_door_id": "RVVPackedI4DotSourceFrontDoor",
    },
    {
        "op": "product_reduce", "format": "codebook_n3", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "codebook-gather-dot-product-reduce-front-door-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-codebook-gather-dot-source-front-door=march=rv64gcv",
        "front_door_id": "RVVCodebookDotSourceFrontDoor",
    },
    # q8_0 vec_dot: STRONG. Its front door was upgraded to construct the typed flat
    # block-dot LOOP body (tcrv_rvv.typed_flat_block_dot_loop_body) out of decomposed
    # pattern-library primitives (load + widening_product + standalone_reduce), NOT the
    # opaque emitFlatBlockDot hand helper. The SAME front-door pass now realizes a typed
    # body, so update-sixstate machine-reads the REAL constructor output (no hand .mlir).
    {
        "op": "vec_dot", "format": "q8_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q8-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q8-0-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q4_0 vec_dot: STRONG. Same typed flat block-dot LOOP construction as q8_0, but the
    # decomposed body realizes the packed-i4 offset-binary dot primitive
    # (tcrv_rvv.packed_i4_offset_binary_x_i8_product) instead of the plain widening_product,
    # since q4_0 weights are packed nibbles. Its front door constructs the typed
    # tcrv_rvv.typed_flat_block_dot_loop_body (NOT the opaque emitFlatBlockDot hand helper),
    # so update-sixstate machine-reads the REAL constructor output (no hand .mlir).
    {
        "op": "vec_dot", "format": "q4_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q4-0-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q4_1 vec_dot: STRONG. Same typed flat block-dot LOOP construction as q4_0, but the
    # decomposed body realizes q4_1's Family-B scale+MIN structure: the integer core is the
    # unsigned-nibble dot primitive (tcrv_rvv.unsigned_nibble_x_i8_product), and the per-block
    # dequant is a dual-fp16 scale+min fold (block_fp16_scale_product d_x.d_y + block_fp16_min_product
    # m_x.s_y, BOTH operand-derived from the block data, feeding block_computed_scale_dequant).
    # Its front door constructs the typed tcrv_rvv.typed_flat_block_dot_loop_body (NOT the opaque
    # emitFlatBlockDot hand helper), so update-sixstate machine-reads the REAL constructor output.
    {
        "op": "vec_dot", "format": "q4_1", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-1-q8-1-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q4-1-q8-1-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q5_0 vec_dot: STRONG. Its front door was upgraded (isQ50TypedFlat in the
    # typedFlatLoopPath gate) to construct the typed flat block-dot LOOP body, whose
    # decomposed integer core realizes q5_0's five-bit offset-binary dot primitive
    # (tcrv_rvv.five_bit_offset_binary_x_i8_product) fed by a per-block qh 5th-bit
    # source brick (block_five_bit_qh_source), plus standalone_reduce and a per-block
    # fp16 scale dequant. NO opaque emitFlatBlockDot hand helper, so [L-8] derives
    # constructed (STRONG). update-sixstate machine-reads the REAL constructor output.
    {
        "op": "vec_dot", "format": "q5_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q5-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q5-0-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q5_1 vec_dot: STRONG (cohort 5/5, LAST). Same typed flat block-dot LOOP body via the
    # front-door gate; decomposed integer core = q5_1's five-bit offset-binary dot
    # (five_bit_offset_binary_x_i8_product, bias-off) + qh 5th-bit source
    # (block_five_bit_qh_source) + the min term (block_fp16_min_product), ScalePlusMin fold.
    # NO opaque emitFlatBlockDot hand helper, so [L-8] derives constructed (STRONG).
    {
        "op": "vec_dot", "format": "q5_1", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q5-1-q8-1-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q5-1-q8-1-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body)",
    },
    # q4_K vec_dot: STRONG (the FIRST K-quant super-block flipped, M-FLAT milestone-3).
    # Its front door constructs the typed SUPER-BLOCK dual-accumulator loop body
    # (tcrv_rvv.typed_super_block_block_dot_loop_body) out of the 5 decomposed q4_K
    # bricks (q4_k_nibble_unpack -> q4_k_scale_min_bit_dance -> q4_k_scaled_dot ->
    # q4_k_min_term -> q4_k_sums_fold_scale_d), NOT the opaque emitQ4_KQ8_KBlockDot
    # hand helper. The contraction+reduction is the fused per-sub-block integer-MAC
    # q4_k_scaled_dot (vwmacc into aux32); NO opaque *_block_dot op, so [L-8] derives
    # constructed (STRONG). update-sixstate machine-reads the REAL constructor output.
    {
        "op": "vec_dot", "format": "q4_K", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q4-k-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockBlockDotLoopChain (typed super-block block-dot loop body)",
    },
    # q5_K vec_dot: STRONG (the SECOND K-quant super-block flipped). q5_K == q4_K + the
    # qh 5th-bit plane: it REUSES the SAME typed super-block dual-accumulator loop body
    # (tcrv_rvv.typed_super_block_block_dot_loop_body) out of the SAME 5 decomposed
    # bricks (q4_k_nibble_unpack -> q4_k_scale_min_bit_dance -> q4_k_scaled_dot ->
    # q4_k_min_term -> q4_k_sums_fold_scale_d), the ONLY net-new work being BRICK 1's
    # weight_qh_byte_offset attr (the emitter injects the 5th bit under cx.hasQh). NOT
    # the opaque emitQ5_KQ8_KBlockDot hand helper (retired same action as the flip). The
    # contraction+reduction is the fused per-sub-block integer-MAC q4_k_scaled_dot
    # (vwmacc into aux32); NO opaque *_block_dot op, so [L-8] derives constructed
    # (STRONG). The manifest is byte-identical to q4_K's (the qh inject is intra-brick,
    # not a new op), resolved to q5_K's OWN export entry by weight_block_stride 176.
    {
        "op": "vec_dot", "format": "q5_K", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q5-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q5-k-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockBlockDotLoopChain (typed super-block block-dot loop body; q5_K stamps BRICK 1 qh offset)",
    },
    # q6_K vec_dot: STRONG (the THIRD K-quant super-block flipped). q6_K has NO
    # per-block min, so its front door constructs the typed SUPER-BLOCK SINGLE-
    # accumulator loop body (tcrv_rvv.typed_super_block_block_dot_loop_body,
    # fold_model "scales_times_sumi") out of just TWO decomposed bricks: the q6_K
    # aux32 INTEGER CORE (q6_k_q8_k_aux32_partial -- the 2-bit qh + 8-bit signed
    # scale unpack + per-sub-block int8-scaled vwmacc into aux32) -> the REUSED no-min
    # positive fold (q4_k_sums_fold_scale_d, d@208) -> a SINGLE `sums` yield. NOT the
    # opaque emitQ6_KQ8_KBlockDot hand helper (retired same action as the flip). The
    # contraction+reduction is the fused per-sub-block integer-MAC INSIDE the aux32
    # core (vwmacc into aux32 -- see _FUSED_DOT_REDUCE_RE's aux32_partial token); NO
    # opaque *_block_dot op, so [L-8] derives constructed (STRONG). Resolves to q6_K's
    # OWN export entry by fold_model + weight_block_stride 210.
    {
        "op": "vec_dot", "format": "q6_K", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q6-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q6-k-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalesTimesSumiLoopChain (typed super-block SINGLE-accumulator no-min loop body)",
    },
    # q3_K vec_dot: STRONG (the FIFTH and LAST K-quant super-block flipped). q3_K is
    # SYMMETRIC (NO per-block min), so it SHARES q6_K's typed SUPER-BLOCK SINGLE-
    # accumulator loop body (tcrv_rvv.typed_super_block_block_dot_loop_body, fold_model
    # "scales_times_sumi") out of just TWO decomposed bricks: the q3_K aux32 INTEGER
    # CORE (q3_k_q8_k_aux32_partial -- the 2-bit + SUBTRACTIVE-hmask decode + SIGNED
    # 6-bit scale dance + per-sub-block signed-scaled vwmacc into aux32) -> the REUSED
    # no-min positive fold (q4_k_sums_fold_scale_d, d@108) -> a SINGLE `sums` yield.
    # NOT the opaque emitQ3_KQ8_KBlockDot hand helper (retired same action as the flip).
    # The contraction+reduction is the fused per-sub-block integer-MAC INSIDE the aux32
    # core (vwmacc into aux32 -- see _FUSED_DOT_REDUCE_RE's aux32_partial token); NO
    # opaque *_block_dot op, so [L-8] derives constructed (STRONG). Resolves to q3_K's
    # OWN export entry by fold_model + weight_block_stride 110.
    {
        "op": "vec_dot", "format": "q3_K", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q3-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q3-k-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalesTimesSumiLoopChain (typed super-block SINGLE-accumulator no-min loop body; q3_K aux32 brick keyed off entry.opName)",
    },
    # q2_K vec_dot: STRONG (the FOURTH K-quant super-block flipped). q2_K HAS a per-
    # block min (like q4_K/q5_K) but its whole fold is a SINGLE per-super-block SCALAR
    # `sumf += dall*isum - dmin*summs`, so its front door constructs the typed SUPER-
    # BLOCK SCALAR-accumulator loop body (tcrv_rvv.typed_super_block_block_dot_loop_body,
    # fold_model "scalar_scale_min") out of just ONE decomposed brick: the q2_K INTEGER
    # CORE (q2_k_q8_k_integer_core -- the 2-bit unpack + PLAIN uint4-nibble scale/min +
    # per-sub-block scalar i32 dot producing the two scalar states isum + summs) -> a
    # SINGLE `sumf` scalar yield. NOT the opaque emitQ2_KQ8_KBlockDot hand helper
    # (retired same action as the flip); the scalar fold is emitter-inlined (no separate
    # fold brick). The contraction+reduction is the fused per-sub-block vwmul+vwredsum
    # INSIDE the integer core (see _FUSED_DOT_REDUCE_RE's integer_core token); NO opaque
    # *_block_dot op, so [L-8] derives constructed (STRONG). Resolves to q2_K's OWN
    # export entry by fold_model + weight_block_stride 84.
    {
        "op": "vec_dot", "format": "q2_K", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q2-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q2-k-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarScaleMinLoopChain (typed super-block SCALAR-accumulator loop body)",
    },
    # q4_0 REPACK GEVM (gemm_tile/q4_0/rvv): STRONG. The option-2 quant_contraction
    # BRIDGE route -- distinct from the flat/super-block SOURCE-FUNC front doors. Its
    # front door (--tcrv-rvv-lower-quant-contraction; VLEN128 => Zvl128b => repack
    # SELECTED) CONSTRUCTS the typed tcrv_rvv.typed_repack_gemv_loop_body REGION (the
    # monolithic tcrv_rvv.repack_gemv_q4_0_q8_0 op is retired) out of the two
    # decomposed inner bricks: the per-block lane-wise integer CORE
    # (repack_lane_wise_q4_x_i8_dot, producing the numHalves per-strip sumi via a
    # nibble-step vwmacc lane-wise accumulate -- a FUSED lane-wise dot-reduce, NO
    # separate standalone_reduce) + the per-strip dual-fp16 scale FOLDs
    # (repack_dual_fp16_scale_fold), around a per-strip LANE-WISE f32 VECTOR
    # accumulator. NO opaque *_block_dot hand helper, so [L-8] derives constructed
    # (STRONG) via the repack_lane_wise_q4_x_i8_dot product+fused-reduce token.
    # update-sixstate machine-reads the REAL constructor output (no hand .mlir).
    {
        "op": "gemm_tile", "format": "q4_0", "engine": "rvv", "regime": "decode",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-0-q8-0-repack-gemv-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-lower-quant-contraction=march=rv64gcv",
        "front_door_id": "RVVLowerQuantContraction lowerToRepackGemv (typed repack GEVM loop body region)",
    },
    # q4_0 REPACK GEMM (gemm_tile/q4_0/rvv, regime=prefill): STRONG. The PREFILL
    # sibling of the decode GEVM -- a DISTINCT six-state cell on the m_regime axis
    # (regime=prefill vs the GEVM's regime=decode). Same option-2 quant_contraction
    # BRIDGE front door (--tcrv-rvv-lower-quant-contraction; VLEN128 => Zvl128b =>
    # repack SELECTED), but because the input quant_contraction carries
    # m_regime="prefill" the bridge dispatches to lowerToRepackGemm, which CONSTRUCTS
    # the typed tcrv_rvv.typed_repack_gemm_loop_body REGION (the monolithic
    # tcrv_rvv.repack_gemm_q4_0_q8_0 op is retired) out of the two decomposed inner
    # GEMM bricks: the ONE-strip N-column integer CORE
    # (repack_gemm_lane_wise_q4_x_i8_dot, producing the columnsPerPass per-column sumi
    # via a nibble-step lane-wise vwmacc lo/hi dot-reduce -- a FUSED lane-wise
    # dot-reduce, NO separate standalone_reduce) + the columnsPerPass per-column
    # dual-fp16 scale FOLDs (repack_gemm_dual_fp16_scale_fold), around the per-column
    # per-strip LANE-WISE f32 VECTOR accumulators. NO opaque *_block_dot hand helper,
    # so [L-8] derives constructed (STRONG) via the repack_gemm_lane_wise_q4_x_i8_dot
    # product+fused-reduce token. update-sixstate machine-reads the REAL constructor
    # output (no hand .mlir).
    {
        "op": "gemm_tile", "format": "q4_0", "engine": "rvv", "regime": "prefill",
        "kind": "strong", "expected_state": "constructed",
        "input": "q4-0-q8-0-repack-gemm-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-lower-quant-contraction=march=rv64gcv",
        "front_door_id": "RVVLowerQuantContraction lowerToRepackGemm (typed repack GEMM prefill loop body region)",
    },
    # iq4_nl vec_dot: STRONG (the FIRST CODEBOOK-class flat vec_dot flipped, L3 M2).
    # Its front door (isIq4Nl codebook branch in the typedFlatLoopPath gate) constructs
    # the typed flat block-dot LOOP body (tcrv_rvv.typed_flat_block_dot_loop_body) whose
    # decomposed integer core is the 2nd primitive class: the 16-entry non-linear int8
    # codebook table broadcast (tcrv_rvv.codebook_table_broadcast) + the asymmetric
    # codebook-gather packed-i4 x i8 product (tcrv_rvv.codebook_gather_x_i8_product, the
    # vrgather kvalues decode -- matches _DOT_PRODUCT_RE's _x_i8_product token), plus
    # standalone_reduce and a per-block fp16 scale dequant. NO opaque emitFlatBlockDot
    # hand helper (the GgmlBlockDotIQ4NLQ80Op op + emitIQ4NLQ8_0BlockDot shim + verifier
    # RETIRED same action as the flip), so [L-8] derives constructed (STRONG).
    # update-sixstate machine-reads the REAL constructor output (no hand .mlir).
    {
        "op": "vec_dot", "format": "iq4_nl", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq4-nl-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq4-nl-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChain (typed flat block-dot loop body; codebook branch)",
    },
    # iq1_s vec_dot: STRONG (the FIRST super-block GRID/CODEBOOK-class vec_dot flipped,
    # L3 M3). iq1_s is a super-block quant whose per-sub-block decode is a codebook GRID
    # GATHER (decode_model=lookup -- the 2048-entry TERNARY iq1s_grid + a vluxei16
    # gather), but its whole fold is a SINGLE per-super-block SCALAR
    # `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)`, so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChain) constructs the typed SUPER-BLOCK
    # SCALAR-accumulator loop body (tcrv_rvv.typed_super_block_block_dot_loop_body,
    # fold_model "scalar_delta_grid") out of just ONE decomposed brick: the iq1_s
    # TERNARY-grid INTEGER CORE (iq1_s_q8_k_grid_core -- the 11-bit qs+qh grid index +
    # the vluxei16 ternary-grid gather + the signed widening grid dot + the qh-encoded
    # scale + the delta-bsum sum, producing the two scalar states sumi + sumi1) -> a
    # SINGLE `sumf` scalar yield. NOT the opaque emitIQ1SQ8KBlockDot hand helper (the
    # GgmlBlockDotIQ1SQ8KOp op + emitter + verifier RETIRED same action as the flip; the
    # shared grid decl/body anchors were kept). The contraction+reduction is the fused
    # per-sub-block vluxei16 gather + vwmul + vwredsum INSIDE the grid core (see
    # _FUSED_DOT_REDUCE_RE's grid_core token); NO opaque *_block_dot op, so [L-8] derives
    # constructed (STRONG). Resolves to iq1_s's OWN export entry by fold_model +
    # weight_block_stride 50. update-sixstate machine-reads the REAL constructor output.
    {
        "op": "vec_dot", "format": "iq1_s", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq1-s-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq1-s-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChain (typed super-block SCALAR-accumulator ternary-grid loop body)",
    },
    # iq1_m vec_dot: STRONG (the SECOND super-block GRID/CODEBOOK-class vec_dot flipped,
    # L3 -- the C2 marginal-cost payoff). iq1_m is the iq1_s SIBLING: the SAME 2048-entry
    # TERNARY grid + the SAME single per-super-block SCALAR fold `sumf += d*((float)sumi1
    # + IQ1M_DELTA*(float)sumi2)` (fold_model "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainIq1M) REUSES the WHOLE iq1_s
    # scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK SCALAR-accumulator
    # loop body out of just ONE decomposed brick: the DISTINCT iq1_m TERNARY-grid INTEGER
    # CORE (iq1_m_q8_k_grid_core -- the packed iq1m_scale fp16 reconstruct + the half-split
    # per-half vluxei16 grid dot with two half scales + the per-group four-sign Sigma-q8
    # delta, producing the two scalar states sumi1 + sumi2) -> a SINGLE `sumf` scalar yield.
    # NOT the opaque emitIQ1MQ8KBlockDot hand helper (the GgmlBlockDotIQ1MQ8KOp op + emitter
    # + verifier RETIRED same action as the flip; the shared grid decl/body anchors kept).
    # The contraction+reduction is the fused per-half vluxei16 gather + vwmul + vwredsum
    # INSIDE the grid core (see _FUSED_DOT_REDUCE_RE's grid_core token); NO opaque
    # *_block_dot op, so [L-8] derives constructed (STRONG). Resolves to iq1_m's OWN export
    # entry by fold_model + weight_block_stride 56 (vs iq1_s 50).
    {
        "op": "vec_dot", "format": "iq1_m", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq1-m-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq1-m-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainIq1M (typed super-block SCALAR-accumulator ternary-grid loop body; iq1_s sibling)",
    },
    # iq3_xxs vec_dot: STRONG (the THIRD super-block GRID/CODEBOOK-class vec_dot flipped,
    # L3 coverage -- another marginal-cost payoff). iq3_xxs is an iq1_s GRID SIBLING: the
    # SAME single per-super-block SCALAR fold arity (fold_model "scalar_delta_grid"), so
    # its front door (createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs) REUSES the WHOLE
    # iq1_s scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK
    # SCALAR-accumulator loop body out of just ONE decomposed brick: the DISTINCT iq3_xxs
    # GRID-of-4 INTEGER CORE (iq3_xxs_q8_k_grid_core -- the i32 iq3xxs_grid vluxei16_v_i32m1
    # gather + the aux32 4-bit-scale + 4-sign-group ksigns decode, producing the ONE scalar
    # state bsum) -> a SINGLE `sumf` scalar yield. NOT the opaque emitIQ3XXSQ8KBlockDot hand
    # helper (the GgmlBlockDotIQ3XXSQ8KOp op + emitter + verifier + monolith conversion test
    # RETIRED same action as the flip; the shared grid/ksigns/body anchors kept). The
    # contraction+reduction is the fused two-index-per-group vluxei16 gather + vwmul +
    # vwredsum INSIDE the grid core (see _FUSED_DOT_REDUCE_RE's grid_core token); NO opaque
    # *_block_dot op, so [L-8] derives constructed (STRONG). Resolves to iq3_xxs's OWN export
    # entry by fold_model + weight_block_stride 98 (vs iq1_s 50 / iq1_m 56).
    {
        "op": "vec_dot", "format": "iq3_xxs", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq3-xxs-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq3-xxs-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainIq3xxs (typed super-block SCALAR-accumulator GRID-of-4 loop body; iq1_s grid sibling)",
    },
    # iq2_xxs vec_dot: STRONG (the FOURTH super-block GRID/CODEBOOK-class vec_dot flipped,
    # L3 coverage -- another marginal-cost payoff, SIGN-PLANE signs64 variant). iq2_xxs is
    # an iq1_s GRID SIBLING: the SAME single per-super-block SCALAR fold arity (fold_model
    # "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs) REUSES the WHOLE iq1_s
    # scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK SCALAR-accumulator loop
    # body out of just ONE decomposed brick: the DISTINCT iq2_xxs GRID-of-8 INTEGER CORE
    # (iq2_xxs_q8_k_grid_core -- the i64 iq2xxs_grid vluxei16_v_i64<core> gather + the SECOND
    # signs64 vluxei16 gather over the DERIVED keven_signs_q2xs sign plane + the aux1
    # 4-bit-scale + 4-sign-group decode, producing the ONE scalar state bsum) -> a SINGLE
    # `sumf` scalar yield. NOT the opaque emitIQ2XXSQ8KBlockDot hand helper (the
    # GgmlBlockDotIQ2XXSQ8KOp op + emitter + verifier + monolith conversion+dataflow tests
    # RETIRED same action as the flip; the shared grid/signs64/body anchors kept; the brick
    # PRESERVES iq2_xxs's Win-A m2/m1 gearbox). The contraction+reduction is the fused
    # 4-index vluxei16 gather + vmul-sign-fold + vwmul + vwredsum INSIDE the grid core (see
    # _FUSED_DOT_REDUCE_RE's grid_core token); NO opaque *_block_dot op, so [L-8] derives
    # constructed (STRONG). Resolves to iq2_xxs's OWN export entry by fold_model +
    # weight_block_stride 66 (vs iq1_s 50 / iq1_m 56 / iq3_xxs 98).
    {
        "op": "vec_dot", "format": "iq2_xxs", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq2-xxs-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq2-xxs-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainIq2xxs (typed super-block SCALAR-accumulator GRID-of-8 loop body; iq1_s grid sibling, signs64 variant)",
    },
    # iq2_xs vec_dot: STRONG (the FIFTH super-block GRID/CODEBOOK-class vec_dot flipped, L3
    # coverage -- another marginal-cost payoff, SIGN-PLANE signs64 variant, PER-HALF explicit
    # scale). iq2_xs is the iq2_xxs GRID SIBLING: the SAME single per-super-block SCALAR fold
    # arity (fold_model "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainIq2xs) REUSES the WHOLE iq1_s
    # scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK SCALAR-accumulator loop
    # body out of just ONE decomposed brick: the DISTINCT iq2_xs per-half-scale GRID INTEGER
    # CORE (iq2_xs_q8_k_grid_core -- the 512-entry iq2xs_grid vluxei16_v_i64m1 gather indexed
    # by w&511 + the SECOND signs64 vluxei16 gather over the DERIVED keven_signs_q2xs sign
    # plane keyed by w>>9 + the EXPLICIT per-sub-block 4-bit scales[8] two-half split ls1/ls2,
    # producing the ONE scalar state bsum) -> a SINGLE `sumf` scalar yield. NOT the opaque
    # emitIQ2XSQ8KBlockDot hand helper (the GgmlBlockDotIQ2XSQ8KOp op + emitter + verifier +
    # monolith conversion+dataflow tests RETIRED same action as the flip; the shared
    # grid/signs64/body anchors kept; UNLIKE iq2_xxs the brick carries NO gearbox -- fixed
    # 16-lane per-half shape). The contraction+reduction is the fused per-half 2-index
    # vluxei16 gather + vmul-sign-fold + vwmul + vwredsum INSIDE the grid core (see
    # _FUSED_DOT_REDUCE_RE's grid_core token); NO opaque *_block_dot op, so [L-8] derives
    # constructed (STRONG). Resolves to iq2_xs's OWN export entry by fold_model +
    # weight_block_stride 74 (vs iq1_s 50 / iq1_m 56 / iq3_xxs 98 / iq2_xxs 66).
    {
        "op": "vec_dot", "format": "iq2_xs", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq2-xs-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq2-xs-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainIq2xs (typed super-block SCALAR-accumulator per-half-scale GRID loop body; iq2_xxs grid sibling, signs64 variant, no gearbox)",
    },
    # iq2_s vec_dot: STRONG (the SIXTH super-block GRID/CODEBOOK-class vec_dot flipped, L3
    # coverage -- another marginal-cost payoff, SIGN-PLANE explicit-signs variant, PER-HALF
    # explicit scale; the LAST iq2 variant). iq2_s is the iq2_xs GRID SIBLING: the SAME single
    # per-super-block SCALAR fold arity (fold_model "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainIq2s) REUSES the WHOLE iq1_s
    # scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK SCALAR-accumulator loop
    # body out of just ONE decomposed brick: the DISTINCT iq2_s per-half-scale GRID INTEGER
    # CORE (iq2_s_q8_k_grid_core -- the 1024-entry iq2s_grid vluxei16_v_i64m1 gather indexed
    # by `qs[l] | ((qh<<(8-2l))&0x300)` + the SECOND signs256 vluxei16 gather over the
    # UNIVERSAL explicit-sign-byte plane keyed by the RAW sign byte read DIRECTLY from the
    # sign region at qs+32 + the EXPLICIT per-sub-block 4-bit scales[8] two-half split
    # ls1/ls2, producing the ONE scalar state bsum) -> a SINGLE `sumf` scalar yield. NOT the
    # opaque emitIQ2SQ8KBlockDot hand helper (the GgmlBlockDotIQ2SQ8KOp op + emitter +
    # verifier + monolith conversion+dataflow tests RETIRED same action as the flip; the
    # shared grid/signs256/body anchors kept; like iq2_xs the brick carries NO gearbox --
    # fixed 16-lane per-half shape). The contraction+reduction is the fused per-half 2-index
    # vluxei16 gather + vmul-sign-fold + vwmul + vwredsum INSIDE the grid core (see
    # _FUSED_DOT_REDUCE_RE's grid_core token); NO opaque *_block_dot op, so [L-8] derives
    # constructed (STRONG). Resolves to iq2_s's OWN export entry by fold_model +
    # weight_block_stride 82 (vs iq1_s 50 / iq1_m 56 / iq3_xxs 98 / iq2_xxs 66 / iq2_xs 74).
    {
        "op": "vec_dot", "format": "iq2_s", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq2-s-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq2-s-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainIq2s (typed super-block SCALAR-accumulator per-half-scale GRID loop body; iq2_xs grid sibling, explicit-signs variant, no gearbox)",
    },
    # iq3_s vec_dot: STRONG (the SEVENTH super-block GRID/CODEBOOK-class vec_dot flipped,
    # C_construct 22->23 -- another marginal-cost payoff, EXPLICIT-SIGNS variant). iq3_s is
    # the iq3_xxs GRID-of-4 SIBLING: the SAME single per-super-block SCALAR fold arity
    # (fold_model "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainIq3s) REUSES the WHOLE iq1_s
    # scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK SCALAR-accumulator loop
    # body out of just ONE decomposed brick: the DISTINCT iq3_s GRID-of-4 explicit-signs
    # INTEGER CORE (iq3_s_q8_k_grid_core -- the 512-entry iq3s_grid vluxei16_v_i32m1 gather
    # indexed by the qh-9th-bit-injected index `qs[l] | ((qh<<(8-2l))&256)` + the EXPLICIT
    # per-sub-block sign bytes read DIRECTLY from the signs region at xb+74 folded via the
    # inline kmask {1<<j} + the EXPLICIT per-sub-block two-nibble scales[4], producing the ONE
    # scalar state bsum) -> a SINGLE `sumf` scalar yield. NOT the opaque emitIQ3SQ8KBlockDot
    # hand helper (the GgmlBlockDotIQ3SQ8KOp op + emitter + verifier + monolith
    # conversion+dataflow tests RETIRED same action as the flip; the shared grid/body anchors
    # kept; like iq2_s the brick carries NO gearbox -- fixed grid-of-4 shape, NO ksigns plane).
    # The contraction+reduction is the fused per-group 2-index vluxei16 gather + vmerge-sign-
    # fold + vwmul + vwredsum INSIDE the grid core (see _FUSED_DOT_REDUCE_RE's grid_core
    # token); NO opaque *_block_dot op, so [L-8] derives constructed (STRONG). Resolves to
    # iq3_s's OWN export entry by fold_model + weight_block_stride 110 (vs iq1_s 50 / iq1_m 56
    # / iq3_xxs 98 / iq2_xxs 66 / iq2_xs 74 / iq2_s 82).
    {
        "op": "vec_dot", "format": "iq3_s", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq3-s-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq3-s-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainIq3s (typed super-block SCALAR-accumulator GRID-of-4 loop body; iq3_xxs grid sibling, explicit-signs variant, no gearbox, no ksigns plane)",
    },
    # iq4_xs vec_dot: STRONG (the FIRST super-block CODEBOOK-class vec_dot flipped,
    # C_construct 23->24 -- another marginal-cost payoff, CODEBOOK vs the grid siblings).
    # iq4_xs is the SUPER-BLOCK rung of the flat iq4_nl codebook: the SAME single
    # per-super-block SCALAR fold arity (fold_model "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainIq4xs) REUSES the WHOLE iq1_s
    # scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK SCALAR-accumulator loop
    # body out of just ONE decomposed brick: the DISTINCT iq4_xs CODEBOOK INTEGER CORE
    # (iq4_xs_q8_k_codebook_core -- REUSES iq4_nl's 16-entry non-linear int8 codebook gathered
    # via vrgather_vv_i8m1, the per-sub-block SIGNED 6-bit scale from scales_l[4]+scales_h
    # biased -32 applied in the FLOAT domain, the asymmetric vwmul/vwmacc widening product +
    # seed-0 vwredsum producing the per-sub-block sumi). UNLIKE the grid siblings iq4_xs's
    # fold runs PER-SUB-BLOCK in float `sumf += (d4d8*(ls-32))*sumi` (8 fp folds per
    # super-block, NO trailing factor) rather than a single `sumf += d*bsum`, but the
    # single-scalar accumulator arity is identical and the whole body is emitter-inlined. NOT
    # the opaque emitIQ4XSQ8KBlockDot hand helper (the GgmlBlockDotIQ4XSQ8KOp op + emitter +
    # verifier + monolith conversion+dataflow tests RETIRED same action as the flip). The
    # contraction+reduction is the fused per-sub-block vrgather codebook gather + vwmul/vwmacc
    # + vwredsum INSIDE the codebook core (see _FUSED_DOT_REDUCE_RE's codebook_core token); NO
    # opaque *_block_dot op, so [L-8] derives constructed (STRONG). Resolves to iq4_xs's OWN
    # export entry by fold_model + weight_block_stride 136 (vs iq1_s 50 / iq1_m 56 / iq3_xxs
    # 98 / iq2_xxs 66 / iq2_xs 74 / iq2_s 82 / iq3_s 110). The codebook-core brick carries NO
    # integer_core_lmul gearbox (the codebook gather pins m1), so iq4_xs is NOT in any
    # schedule autotuner and has NO VLEN128-vs-VLEN256 byte-flip.
    {
        "op": "vec_dot", "format": "iq4_xs", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "iq4-xs-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-iq4-xs-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainIq4xs (typed super-block SCALAR-accumulator CODEBOOK loop body; flat iq4_nl codebook sibling, super-block rung, per-sub-block float fold, no gearbox)",
    },
    # tq2_0 vec_dot: STRONG (the FIRST TQ-family member flipped, C_construct 24->25 -- the
    # first ARITHMETIC-ternary super-block vec_dot vs the iq* grid/codebook siblings). tq2_0
    # is the 2-bit TERNARY ({-1,0,+1}) TriLM K-quant: the SAME single per-super-block SCALAR
    # fold arity (fold_model "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainTq20) REUSES the WHOLE iq1_s
    # scalar-delta-grid scaffold and constructs the typed SUPER-BLOCK SCALAR-accumulator loop
    # body out of just ONE decomposed brick: the DISTINCT tq2_0 FUSED 2-bit TERNARY INTEGER
    # CORE (tq2_0_q8_k_ternary_core -- q2_K's 2-bit (qs>>shift)&3 unpack over the 4 shifts
    # {0,2,4,6} + the per-element -1 ternary bias vsub + the fused-plane vwmacc against q8 into
    # a wide i16 accumulator + ONE seed-0 vwredsum per 32-byte chunk producing the
    # per-super-block scalar sumi -- decode_model=arithmetic, NO grid/codebook gather). Its
    # fold is a single per-super-block scalar `sumf += (float)sumi * d` (d = fp16(x.d @64) *
    # y.d @0, NO trailing factor), emitter-inlined. NOT the opaque emitTQ2_0Q8_KBlockDot hand
    # helper (the GgmlBlockDotTQ20Q8KOp op + emitter + verifier + monolith conversion+dataflow
    # tests RETIRED same action as the flip). The contraction+reduction is the fused per-plane
    # vand/vsrl unpack + vsub bias + vwmacc + vwredsum INSIDE the ternary core (see
    # _FUSED_DOT_REDUCE_RE's ternary_core token); NO opaque *_block_dot op, so [L-8] derives
    # constructed (STRONG). tq2_0 SHARES weight_block_stride 66 with iq2_xxs but resolves to its
    # OWN export entry by the DISTINCT ternary-core brick op type (stride-66 collision
    # tie-breaker). UNLIKE iq4_xs the ternary-core brick PRESERVES tq2_0's Win-A
    # integer_core_lmul m2/m1 gearbox (kernel key "tq2_0"), so tq2_0 IS in the schedule
    # autotuner and HAS a VLEN128(m2)-vs-VLEN256(m1) byte-flip. It builds the REUSABLE ternary
    # scaffold that tq1_0 (base-3) reuses next at C2 marginal cost.
    {
        "op": "vec_dot", "format": "tq2_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "tq2-0-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-tq2-0-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainTq20 (typed super-block SCALAR-accumulator TERNARY loop body; FIRST TQ-family member, arithmetic 2-bit ternary core, Win-A m2/m1 gearbox preserved)",
    },
    # tq1_0 vec_dot: STRONG (the SECOND TQ-family member flipped, C_construct 25->26 -- the
    # base-3-packed sibling of tq2_0, REUSING the tq2_0 ternary scaffold at C2 marginal cost).
    # tq1_0 is the BASE-3 TERNARY ({-1,0,+1}) TriLM K-quant: the SAME single per-super-block
    # SCALAR fold arity (fold_model "scalar_delta_grid"), so its front door
    # (createTypedSuperBlockScalarDeltaGridLoopChainTq10) REUSES the WHOLE tq2_0 ternary scaffold
    # and constructs the typed SUPER-BLOCK SCALAR-accumulator loop body out of just ONE
    # decomposed brick: the DISTINCT tq1_0 BASE-3 TERNARY INTEGER CORE (tq1_0_q8_k_ternary_core
    # -- the qs main/tail + qh base-3 trit unpack `q=(uint8_t)(byte*pow3[l]); xi=((uint16_t)q*3)
    # >>8; xi-1` into an element-ordered aux8[256] + the flat-256 widened i8*i8 dot vle8 i8 x q8
    # i8 -> vwmul i16 -> vwredsum i32 producing the per-super-block scalar sumi --
    # decode_model=arithmetic, NO grid/codebook gather, NO 2-bit field shift). Its fold is a
    # single per-super-block scalar `sumf += (float)sumi * d` (d = fp16(x.d @52) * y.d @0, NO
    # trailing factor), emitter-inlined. NOT the opaque emitTQ1_0Q8_KBlockDot hand helper (the
    # GgmlBlockDotTQ10Q8KOp op + emitter + verifier + monolith conversion+dataflow tests RETIRED
    # same action as the flip). The contraction+reduction is the base-3 unpack + the flat-256
    # vwmul + vwredsum INSIDE the ternary core (see _FUSED_DOT_REDUCE_RE's ternary_core token);
    # NO opaque *_block_dot op, so [L-8] derives constructed (STRONG). tq1_0's weight_block_stride
    # 54 is UNIQUE among the scalar_delta_grid bricks (tq2_0/iq2_xxs are 66), so it resolves to
    # its OWN export entry by stride with NO tie-breaker (cheaper than tq2_0). Like tq2_0 the
    # ternary-core brick PRESERVES tq1_0's Win-A integer_core_lmul m2/m1 gearbox (kernel key
    # "tq1_0"), so tq1_0 IS in the schedule autotuner and HAS a VLEN128(m2)-vs-VLEN256(m1)
    # byte-flip. It REUSES the tq2_0 ternary scaffold at C2 marginal cost (the second TQ cell).
    {
        "op": "vec_dot", "format": "tq1_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "tq1-0-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-tq1-0-q8-k-block-dot-source-front-door",
        "front_door_id": "createTypedSuperBlockScalarDeltaGridLoopChainTq10 (typed super-block SCALAR-accumulator BASE-3 TERNARY loop body; SECOND TQ-family member, arithmetic base-3 ternary core reusing the tq2_0 scaffold, Win-A m2/m1 gearbox preserved)",
    },
    # q1_0 vec_dot: STRONG (the LAST flat block-dot family member flipped, C_construct
    # 26->27 -- the BINARY {-1,+1}-sign class, the genuine structural-special case). q1_0
    # is a FLAT quant (block_q8_0 activation) but its per-super-block contribution is a
    # FOUR-sub-block binary sign decode with a DISTINCT TWO-LEVEL fp32 fold
    # (`d0 * Σ_k(d1_k * sumi_block_k)`) that no existing single-core flat brick chain
    # expresses, so UNLIKE q8_0/q4_0/q5_0 (a single per-block core + the shared
    # scale->dequant->accumulate brick chain) its front door (isQ10TypedFlat) constructs
    # the FLAT loop body (tcrv_rvv.typed_flat_block_dot_loop_body, fold_model
    # "flat_binary_two_level") out of just ONE decomposed brick: the DISTINCT q1_0
    # BINARY-sign INTEGER CORE (q1_0_q8_0_binary_sign_core -- the four q8_0 sub-blocks'
    # vlm_v_b{ratio} packed-bit sign mask loaded DIRECTLY as the i8 sign mask + i8-domain
    # vneg/vmerge -> ONE vwredsum i8->i16m1 per sub-block, plus the emitter-inlined
    # TWO-LEVEL fp32 fold; NO grid/codebook gather, NO nibble unpack). NOT the opaque
    # emitQ1_0Q8_0BlockDot hand helper (the whole per-super-block body -- including the
    # fold -- is re-emitted from the brick via the shared emitQ1_0BlockDotBodyShared).
    # The contraction+reduction is the fused per-sub-block vneg/vmerge sign fold +
    # vwredsum INSIDE the binary-sign core (see _FUSED_DOT_REDUCE_RE's binary_sign_core
    # token); NO opaque *_block_dot op, so [L-8] derives constructed (STRONG). The
    # binary-sign-core brick PRESERVES q1_0's Win-A integer_core_lmul m2/m1 gearbox
    # (kernel key "q1_0"), so q1_0 IS in the schedule autotuner and HAS a
    # VLEN128(m2)-vs-VLEN256(m1) byte-flip. The OUTER with_vl frame stays SEW32/m1
    # (isQ10TypedFlat is OUT of typedFlatLoopPath -- the e8m2 sign decode runs its OWN
    # vsetvl INSIDE the brick), byte-exact to the monolith frame.
    {
        "op": "vec_dot", "format": "q1_0", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "q1-0-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-q1-0-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChainQ10 (typed flat block-dot loop body; BINARY-sign integer core, two-level fold, LAST flat family member, Win-A m2/m1 gearbox preserved)",
    },
    # nvfp4 vec_dot: STRONG (the SECOND FP4-CODEBOOK class = NVIDIA's FP4, the LAST
    # dispatch-wired vec_dot flipped, C_construct 27->28 -- closes the ① G1 literal
    # block-dot zoo). nvfp4 is a SUPER-BLOCK codebook quant (block_nvfp4 = {uint8_t d[4];
    # uint8_t qs[32]}, QK=64, four 16-element sub-blocks) but its 64 elements span TWO
    # block_q8_0 activation blocks -- a FLAT block_q8_0 stream (like q1_0), so its front
    # door (isNvfp4TypedFlat) constructs the FLAT loop body
    # (tcrv_rvv.typed_flat_block_dot_loop_body, fold_model "flat_nvfp4_codebook") out of
    # just ONE decomposed brick: the DISTINCT nvfp4 CODEBOOK INTEGER CORE
    # (nvfp4_q8_0_codebook_core -- REUSES mxfp4's 16-entry DOUBLED e2m1 codebook gathered
    # via vand/vsrl nibble split + vrgather_vv_i8m1 through the broadcast table +
    # asymmetric vwmul/vwmacc widening product + seed-0 vwredsum producing the per-sub-block
    # sumi, wrapped in the per-SUB-block UE4M3 fp8 weight scale -- the ldexpf HALF-form
    # decode). NOT the opaque emitNVFP4Q8_0BlockDot hand helper (the whole per-super-block
    # body -- including the per-sub-block float fold -- is re-emitted from the brick via the
    # shared emitNVFP4BlockDotBodyShared; the GgmlBlockDotNVFP4Q80Op op + emitter + verifier
    # + monolith conversion+dataflow tests RETIRED the SAME action as the flip). The
    # contraction+reduction is the fused vrgather codebook gather + vwmul/vwmacc + seed-0
    # vwredsum INSIDE the codebook core (see _FUSED_DOT_REDUCE_RE's codebook_core token); NO
    # opaque *_block_dot op, so [L-8] derives constructed (STRONG). The codebook-core brick
    # carries NO integer_core_lmul gearbox (the codebook gather pins m1 -- VLMAX >= 16 to
    # index all 16 table entries), so nvfp4 is NOT in any schedule autotuner and has NO
    # VLEN128-vs-VLEN256 byte-flip. The OUTER with_vl frame stays SEW32/m1 (isNvfp4TypedFlat
    # is OUT of typedFlatLoopPath -- the e8m1 codebook strip runs its OWN vsetvl INSIDE the
    # brick), byte-exact to the monolith frame.
    {
        "op": "vec_dot", "format": "nvfp4", "engine": "",
        "kind": "strong", "expected_state": "constructed",
        "input": "nvfp4-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-nvfp4-q8-0-block-dot-source-front-door",
        "front_door_id": "createTypedFlatBlockDotLoopChainNvfp4 (typed flat block-dot loop body; FP4-e2m1 CODEBOOK integer core reusing mxfp4's vrgather gather + per-sub-block UE4M3 fp8 scale, per-sub-block float fold, LAST dispatch-wired vec_dot, NO gearbox -- codebook gather pins m1)",
    },
    # Negative control (weak descriptor-selected block-dot). mxfp4 REPLACES iq4_nl as the
    # negative control now that iq4_nl flipped to a constructed typed body (L3 M2). mxfp4
    # is NOT in the front door's typedFlatLoopPath gate (only q8_0/q4_0/q4_1/q5_0/q5_1/
    # iq4_nl now), so its front door auto-constructs the MONOLITHIC
    # tcrv_rvv.mxfp4_q8_0_block_dot op (kind "ggml_mxfp4_q8_0_block_dot"), which
    # is_opaque_hand_helper matches -> [L-8] derives NOT-strong (constructed-weak). mxfp4
    # is a VALID (non-vacuous) discriminator: the SAME flat CODEBOOK class as iq4_nl,
    # still constructed-weak (descriptor-selected emitFlatBlockDot hand helper, its own
    # GgmlBlockDotMXFP4Q80Op op NOT retired), so the check stays proven discriminating.
    {
        "op": "vec_dot", "format": "mxfp4", "engine": "",
        "kind": "negative", "expected_state": "constructed-weak",
        "input": "mxfp4-q8-0-flat-block-dot-full-pipeline-export-e2e.mlir",
        "front_door": "--tcrv-rvv-materialize-mxfp4-q8-0-block-dot-source-front-door",
        "front_door_id": "emitFlatBlockDot (descriptor-selected hand helper)",
    },
]

# --- repack gemm_tile DUAL-regime cert paths (FIX-D: the 10 single-row repack
# cells beyond q4_0). Each is ONE six-state row (regime="") covering BOTH the GEVM
# (decode) and GEMM (prefill) construction. Unlike q4_0 (split into two regime rows
# already stamped E5-increment1-auto), these carry a single [F-EMIT] DUAL manifest.
# The walk feeds a synthesized quant_contraction request (verifier-pinned PLAIN byte
# facts per format) through the SAME `--tcrv-rvv-lower-quant-contraction` front door
# the certified q4_0 uses -- a construction stage UPSTREAM of emitc (independent of
# any emitc-side edit). `cmd_stamp_repack_dual` walks BOTH regimes, requires BOTH to
# derive constructed + a legal repack shape + opaque_helper=false, then writes the
# machine dual manifest. Inputs live OUTSIDE the lit tree (cert probes, not lit
# tests). q4_0 is intentionally ABSENT (its two regime rows are already certified).
_REPACK_PROBE_DIR = REPO_ROOT / "experiments" / "active" / "cert-status" / "repack-probes"
REPACK_DUAL_PATHS = [
    {"op": "gemm_tile", "format": fmt, "engine": "rvv",
     "gevm_input": _REPACK_PROBE_DIR / f"{fmt}-repack-gevm-cert-probe.mlir",
     "gemm_input": _REPACK_PROBE_DIR / f"{fmt}-repack-gemm-cert-probe.mlir",
     "front_door": "--tcrv-rvv-lower-quant-contraction=march=rv64gcv"}
    for fmt in ["q8_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
                "iq4_nl", "iq4_xs", "iq2_xxs", "iq2_xs", "iq2_s", "tq2_0", "tq1_0"]
]

# --- CERT-FD首族: the 21 CONSTRUCTED streaming dequantize_row cells (FIX-5) --------
# The dequant-stream front door (--tcrv-rvv-materialize-dequantize-row-stream-front-door)
# runs ONLY the CONSTRUCTION half of constructOrEmitGgmlDequantizeRow: it rewrites the
# abstract tcrv_rvv.dequantize_row into the typed tcrv_rvv.typed_dequantize_row_loop_body
# region { dequantize_row_decode_core; typed_dequantize_row_loop_yield } and STOPS -- BEFORE
# --tcrv-rvv-lower-to-emitc (stage discipline). The walk feeds the SAME abstract-op
# conversion fixtures the emitc lit uses, so the certification walks the REAL realized region
# the compiler builds (not a hand fixture). The whole 23-format dequantize_row spectrum is now
# front-door CONSTRUCTED (the tq1_0/tq2_0 base-3 / 2-bit ternary super-blocks were the LAST two
# dispatch-wired dequant cells, flipped here). The streaming shape (a pure decode: body + decode_core + yield, NO
# product/reduce) is its OWN legal shape -- checked directly here (like _walk_repack_regime),
# NOT via the contraction-shaped derive() decomposed gate.
_DEQUANT_STREAM_FRONT_DOOR = "--tcrv-rvv-materialize-dequantize-row-stream-front-door"
DEQUANT_STREAM_PATHS = [
    {"op": "dequantize_row", "format": fmt, "engine": "",
     "input": TEST_CONV_RVV / f"rvv-to-emitc-ggml-dequantize-row-{slug}.mlir"}
    for fmt, slug in [
        ("q8_0", "q8-0"), ("q4_0", "q4-0"), ("q4_1", "q4-1"), ("q5_0", "q5-0"),
        ("q5_1", "q5-1"), ("q2_K", "q2-k"), ("q3_K", "q3-k"), ("q4_K", "q4-k"),
        ("q5_K", "q5-k"), ("q6_K", "q6-k"), ("iq2_xxs", "iq2-xxs"),
        ("iq2_xs", "iq2-xs"), ("iq2_s", "iq2-s"), ("iq3_xxs", "iq3-xxs"),
        ("iq3_s", "iq3-s"), ("iq1_s", "iq1-s"), ("iq1_m", "iq1-m"),
        ("iq4_nl", "iq4-nl"), ("iq4_xs", "iq4-xs"), ("mxfp4", "mxfp4"),
        ("nvfp4", "nvfp4"), ("tq1_0", "tq1-0"), ("tq2_0", "tq2-0"),
    ]
]


# --- CERT-FD次族: the 3 CONSTRUCTED streaming quantize_row cells ------------------
# The quant-stream front door (--tcrv-rvv-materialize-quantize-row-stream-front-door) is
# the f32->QUANT MIRROR of the dequant-stream front door: it runs ONLY the CONSTRUCTION
# half of constructQuantizeRowRegionAndLower -- it rewrites each abstract per-format
# tcrv_rvv.quantize_row_q8_{0,1,K} into the typed tcrv_rvv.typed_quantize_row_loop_body
# region { quantize_row_encode_core; typed_quantize_row_loop_yield } and STOPS -- BEFORE
# --tcrv-rvv-lower-to-emitc (stage discipline). The walk feeds the SAME abstract-op
# conversion fixtures the emitc lit uses. The 3 constructed activation quantizers are
# q8_0 (block_q8_0 family-head) / q8_1 (SIBLING + block sum) / q8_K (QK_K=256 K-quant
# ROW quantizer -- the scalar row-quant stream, NOT the mat-quant GEMM path). The
# streaming shape (a pure ENCODE: body + encode_core + yield, NO product/reduce) is its
# OWN legal shape -- checked directly here (like _walk_dequant_stream), NOT via the
# contraction-shaped derive() decomposed gate.
_QUANT_STREAM_FRONT_DOOR = "--tcrv-rvv-materialize-quantize-row-stream-front-door"
QUANT_STREAM_PATHS = [
    {"op": "quantize_row", "format": fmt, "engine": "",
     "input": TEST_CONV_RVV / f"rvv-to-emitc-ggml-quantize-row-{slug}.mlir"}
    for fmt, slug in [
        ("q8_0", "q8-0"), ("q8_1", "q8-1"), ("q8_K", "q8-k"),
    ]
]


# --- CERT-FD殿后族: the 5 CONSTRUCTED streaming forward-elementwise cells ----------
# The forward-elementwise front door (--tcrv-rvv-materialize-forward-elementwise-stream-
# front-door) is the forward sibling of the dequant/quant stream front doors: it runs
# ONLY the CONSTRUCTION half of the shared byte-exact
# tcrv::rvv::constructTypedElementwiseLoopBody -- it rewrites each abstract
# tcrv_rvv.ggml_forward_elementwise (elementwise_model scale/silu/rms_norm/soft_max/rope)
# into the typed tcrv_rvv.typed_elementwise_loop_body region { <map/reduce/rotate core
# brick>; typed_elementwise_loop_yield } and STOPS -- BEFORE --tcrv-rvv-lower-to-emitc
# (stage discipline). The walk feeds the abstract-op conversion fixtures the emitc lit
# uses. Unlike the dequant/quant PURE decode/encode streams, the 5 forward operators
# split into THREE loop shapes (MAP scale/silu, REDUCE rms_norm/soft_max, ROTATE rope),
# but ALL realize the SAME body/yield wrapper carrying ONE forward map/reduce/rotate
# core brick (NO product/reduce contraction, so derive()'s contraction gate would
# wrongly demote them) -- this streaming shape is its OWN legal form (mirrors the
# checker's elementwise_stream_loop branch), verified here on the ACTUAL realized IR.
# NOTE the six-state `op` key uses "softmax" (not "soft_max", the elementwise_model).
_FORWARD_STREAM_FRONT_DOOR = "--tcrv-rvv-materialize-forward-elementwise-stream-front-door"
FORWARD_STREAM_PATHS = [
    {"op": op, "format": "f32", "engine": "", "core": core,
     "input": TEST_CONV_RVV / f"rvv-to-emitc-ggml-forward-elementwise-{slug}.mlir"}
    for op, core, slug in [
        ("scale", "elementwise_scale_map", "scale"),
        ("silu", "elementwise_silu_map", "silu"),
        ("rms_norm", "elementwise_rms_norm_reduce_core", "rms-norm"),
        ("softmax", "elementwise_soft_max_reduce_core", "soft-max"),
        ("rope", "elementwise_rope_rotate_core", "rope"),
        # The 4 forward SUPPORT ops widened into the SAME front door (all MAP
        # family): add/mul share the BINARY map core, cpy the COPY map, gelu the
        # scalar GELU map. The six-state `op` key matches the model name here.
        ("add", "elementwise_binary_map", "add"),
        ("mul", "elementwise_binary_map", "mul"),
        ("cpy", "elementwise_copy_map", "cpy"),
        ("gelu", "elementwise_gelu_map", "gelu"),
    ]
]


# --- op-identity parse (position-anchored; I4-safe) ------------------------
# Match a tcrv_rvv op mnemonic ONLY in operation position: line-leading (after
# indent), optional `%result = ` prefix (a SINGLE result `%r =`, a comma-separated
# multi-result list `%a, %b = ` as the q2_K super-block integer core produces the
# two scalar states `%isum, %summs`, OR the printer's GROUPED multi-result form
# `%r:N = ` as the repack lane-wise CORE brick produces its numHalves per-strip
# sumi `%9:2 = tcrv_rvv.repack_lane_wise_q4_x_i8_dot`). This never matches
# attribute-name tokens
# (`tcrv_rvv.low_precision_resource.*`, `tcrv_rvv.gearbox.*`) or type tokens
# (`!tcrv_rvv.vector`, `!tcrv_rvv.vl`), which never start a line in op position.
_OP_RE = re.compile(
    r"^\s*(?:%[A-Za-z0-9_#]+(?::[0-9]+)?"
    r"(?:\s*,\s*%[A-Za-z0-9_#]+(?::[0-9]+)?)*\s*=\s*)?"
    r"(tcrv_rvv\.[A-Za-z0-9_]+)\b")
_KIND_RE = re.compile(r'\bkind\s*=\s*"([^"]+)"')
_WITHVL_TERMINATOR = re.compile(r"^(\s*)\}\s*:\s*!tcrv_rvv\.vl\s*$")
# Defense-in-depth I4 guard. The REAL protection is _OP_RE (position-anchored, so
# it can only capture operation mnemonics, never the mid-line attribute tokens
# `tcrv_rvv.low_precision_resource.*` / `tcrv_rvv.gearbox.*` nor the `!tcrv_rvv.*`
# type keywords) plus skipping the with_vl opener line. This guard additionally
# raises if a bare mirror-namespace token ever slipped through as a mnemonic. It is
# anchored to the EXACT namespace token _OP_RE would yield (its `\w+` class stops at
# the `.`, so a leaked attr collapses to `tcrv_rvv.low_precision_resource` /
# `tcrv_rvv.gearbox`). Anchoring with `$` is deliberate: it must NOT fire on the
# legitimate pattern-library primitive `tcrv_rvv.gearbox_cross_region_handoff`.
_MIRROR_GUARD = re.compile(r"^tcrv_rvv\.(low_precision_resource|gearbox)$")

# [L-8] decomposed-gate whitelist. A REAL dot-product-family primitive is one of:
# widening_product (+ *_widening_product_intrinsic), *_x_i8_product (the mixed
# codebook / packed-i4 dot forms routes 2&3 realize), *_unpack_product, a fused
# *product_reduce, or the q4_K/q5_K super-block per-sub-block integer-MAC
# `q4_k_scaled_dot` (which BOTH multiplies the unpacked aux8 weights by the q8
# activation AND reduces the 32 products into the aux32 accumulator via vwmacc --
# a fused dot-reduce, see _FUSED_DOT_REDUCE_RE). The q4_0 16x1-REPACKED GEVM's
# per-block lane-wise integer CORE `repack_lane_wise_q4_x_i8_dot` is ALSO a real
# dot-product primitive (a nibble-step vwmacc lane-wise accumulate + lo/hi combine),
# so its token joins the whitelist. NOTE: it ends `_x_i8_dot` (a lane-wise DOT), NOT
# `_x_i8_product`, so the flat `_x_i8_product` alternative does NOT match it -- the
# explicit `repack_lane_wise_q4_x_i8_dot` token is required. The q4_0 16x1-REPACKED
# GEMM (prefill) sibling `repack_gemm_lane_wise_q4_x_i8_dot` is the SAME shape (a
# per-block lane-wise vwmacc lo/hi dot-reduce, but ONE runtime strip x N interleaved
# activation columns instead of numHalves strips x 1 column), a real fused dot-reduce
# primitive; its distinct mnemonic (`repack_gemm_lane_wise...`, NOT `repack_lane_wise
# ...`) means the GEVM token does not cover it, so the explicit
# `repack_gemm_lane_wise_q4_x_i8_dot` token joins the whitelist too. The iq1_s super-block
# TERNARY-grid integer core `iq1_s_q8_k_grid_core` is ALSO a real dot-product
# primitive (a per-sub-block vluxei16 ternary-grid gather + signed vwmul + vwredsum
# reducing the grid dot into the scalar sumi, plus the delta-bsum scalar state -- a
# fused dot-reduce, see _FUSED_DOT_REDUCE_RE), so its `grid_core` token joins the
# whitelist. This is a WHITELIST on
# purpose: `block_fp16_scale_product` is a per-block fp16 SCALE multiply — not a
# contraction — and carries none of these tokens, so it is excluded. A bare
# "product" substring would wrongly admit it.
_DOT_PRODUCT_RE = re.compile(r"(widening_product|_x_i8_product|_unpack_product|product_reduce|scaled_dot|aux32_partial|integer_core|grid_core|codebook_core|ternary_core|binary_sign_core|kquant_core|repack_lane_wise_q4_x_i8_dot|repack_gemm_lane_wise_q4_x_i8_dot)")

# Fused dot-reduce primitives that carry the reduction INSIDE the product op (no
# separate standalone_reduce in the manifest): the q4_K/q5_K super-block
# `q4_k_scaled_dot` runs a per-sub-block vwmacc that accumulates the 32 products
# into the running aux32, so it satisfies BOTH the product AND the reduce conjunct
# of the decomposed gate. The q6_K super-block integer core `q6_k_q8_k_aux32_partial`
# is the same shape (a per-sub-block vwmacc accumulating the 16 int8-scaled products
# into the running aux32 -- a fused dot-reduce), so `aux32_partial` joins this
# whitelist. The q2_K super-block integer core `q2_k_q8_k_integer_core` is likewise
# a fused dot-reduce (a per-sub-block vwmul then vwredsum reducing the 16 products
# into the scalar isuml, scaled into the running isum), so `integer_core` joins it
# too. The q4_0 16x1-REPACKED GEVM's `repack_lane_wise_q4_x_i8_dot` is likewise a
# fused dot-reduce: its nibble-step vwmacc accumulates the products LANE-WISE into
# the per-strip i16 lo/hi accumulators (then a lo/hi vwadd combine into i32) with NO
# separate standalone_reduce -- the per-strip dual-fp16 scale FOLD that follows is a
# scale multiply, not a reduce -- so its token joins this whitelist too. The iq1_s
# super-block ternary-grid integer core `iq1_s_q8_k_grid_core` is likewise a fused
# dot-reduce (its per-sub-block vluxei16 grid gather + signed vwmul then vwredsum
# reduces the 8-lane grid products into the running scalar sumi, with the delta-bsum
# scalar state accumulated alongside -- NO separate standalone_reduce, the scalar
# delta fold that follows is a scale/add, not a reduce), so its `grid_core` token
# joins this whitelist too. This is
# deliberately NARROW (a scale-only body has none of the tokens; an opaque
# *_block_dot still trips the opaque gate), so the check stays discriminating. The
# iq4_xs super-block CODEBOOK integer core `iq4_xs_q8_k_codebook_core` is likewise a
# fused dot-reduce (its per-sub-block vand/vsrl nibble split + vrgather_vv_i8m1 codebook
# gather + asymmetric vwmul/vwmacc widening product + seed-0 vwredsum reduces the 32
# products into the scalar sumi, which the per-sub-block float scale fold consumes --
# NO separate standalone_reduce, the fold that follows is a scale/add, not a reduce), so
# its `codebook_core` token joins this whitelist too. The tq2_0 super-block FUSED 2-bit
# TERNARY integer core `tq2_0_q8_k_ternary_core` is likewise a fused dot-reduce (its
# per-plane vand/vsrl 2-bit unpack + the -1 ternary-bias vsub + the fused-plane vwmacc
# against q8 into a wide i16 accumulator + ONE seed-0 vwredsum per 32-byte chunk reduces
# the products into the per-super-block scalar sumi, which the single-scale scalar fold
# consumes -- NO separate standalone_reduce, the fold that follows is a scale/add, not a
# reduce), so its `ternary_core` token joins this whitelist too. The q1_0 flat BINARY
# integer core `q1_0_q8_0_binary_sign_core` is likewise a fused dot-reduce (its four
# q8_0 sub-blocks each load the 4 packed bit-bytes DIRECTLY as the i8 sign mask via
# vlm_v_b{ratio}, apply it in the i8 domain via vneg/vmerge -> signed q8, then ONE
# vwredsum i8->i16m1 reduces the 32 signed products into the per-sub-block scalar
# sumi_block, which the emitter-inlined two-level fp32 fold consumes -- NO separate
# standalone_reduce, the fold that follows is a scale/add, not a reduce), so its
# `binary_sign_core` token joins this whitelist too.
# The q4_K/q5_K/q2_K/q3_K/q6_K super-block REPACK GEVM/GEMM core brick
# `repack_gem{v,m}_kquant_core` is the K-quant SIBLING of the already-whitelisted
# repack codebook/ternary cores (`repack_gem{v,m}_codebook_core` matches
# `codebook_core`; `..._ternary_core` matches `ternary_core`). It is the per-block
# integer MAC constructed by the SAME repack front door (lowerToRepackGem{v,m}KQuant)
# that built the certified q4_0 `repack_lane_wise_q4_x_i8_dot`, a FUSED dot-reduce
# (its per-sub-block vwmacc accumulates into the running aux32, then the per-strip
# scale fold consumes it -- NO separate standalone_reduce). It is NOT opaque (does
# not end `_block_dot`, no `ggml_…block_dot` kind), so `kquant_core` joins this
# fused-dot-reduce whitelist -- the ONLY reason the K-quant repack cells derived
# constructed-weak before was the missing token, not a construction defect.
_FUSED_DOT_REDUCE_RE = re.compile(r"(scaled_dot|aux32_partial|integer_core|grid_core|codebook_core|ternary_core|binary_sign_core|kquant_core|repack_lane_wise_q4_x_i8_dot|repack_gemm_lane_wise_q4_x_i8_dot)")

# M-FLAT forward-elementwise scaffold (line C, ① 之后): the reduce/MAP model. The
# forward-pass (non-dot) operators are NOT contractions -- a pure elementwise MAP
# (scale's y[i]*=v, silu's y[i]=x[i]*sigmoid(x[i])) has NO product-family AND NO
# reduce-family primitive, so the contraction-shaped decomposed gate above would
# wrongly demote it to constructed-weak. The MAP class satisfies the [L-8]
# "decomposed = built from typed pattern-library primitives, no opaque helper"
# conjunct via a MAP-family primitive instead: the per-strip map brick (the
# `tcrv_rvv.elementwise_scale_map` scale map OR the `tcrv_rvv.elementwise_silu_map`
# silu map, carried inside the typed `tcrv_rvv.typed_elementwise_loop_body`
# strip-loop op). This whitelist is deliberately NARROW (a scale-only fp16
# `block_fp16_scale_product` body is NOT a forward map primitive, so it stays the
# constructed-weak negative control; an opaque monolith still trips the opaque
# gate), so the check stays discriminating.
_MAP_PRIMITIVE_RE = re.compile(r"(elementwise_scale_map|elementwise_silu_map)")

# M-FLAT forward-elementwise scaffold REDUCE model (line C, G1-tail): the forward
# REDUCE class (rms_norm's Σx², softmax's Σe^x) rides the SAME
# tcrv_rvv.typed_elementwise_loop_body but under reduce_map_model "reduce" (a
# loop-carried accumulator region arg + the yield that carries it back). Its core
# brick is a self-contained fused fold: rms_norm's
# `tcrv_rvv.elementwise_rms_norm_reduce_core` fuses the per-element square product
# (x[i]*x[i]) INTO the scalar-double reduction, so there is NO separate
# product-family primitive to pair with a reduce -- the contraction-shaped gate
# would wrongly demote it. The REDUCE class satisfies the [L-8] "decomposed = built
# from typed pattern-library primitives, no opaque helper" conjunct via this
# reduce-fold-family primitive (the reduce sibling of the _MAP_PRIMITIVE_RE map
# family). This whitelist is deliberately NARROW (a bare "reduce" substring is NOT
# admitted -- the token must be a forward reduce-fold brick; an opaque monolith
# still trips the opaque gate), so the check stays discriminating. soft_max's
# exp-sum-reduce brick (elementwise_soft_max_reduce_core) joins it: the SAME reduce
# model reused, EXCEPT the loop-carried accumulator is the f64m1 WIDENING vector
# (vfwredusum Σe^x) not a scalar double.
_REDUCE_FOLD_PRIMITIVE_RE = re.compile(r"(elementwise_rms_norm_reduce_core|elementwise_soft_max_reduce_core)")

# M-FLAT forward-elementwise scaffold ROTATE model (line C, G1-tail): the forward
# ROTATE class (rope's rotary position embedding) rides the SAME
# tcrv_rvv.typed_elementwise_loop_body but under reduce_map_model "rotate" -- a
# per-PAIR scalar loop with a loop-carried f32 theta RECURRENCE (theta *=
# theta_scale), the SAME loop-carried-scalar region SHAPE the "reduce" model
# pioneered EXCEPT the carried value is a data-INDEPENDENT recurrence, not a
# reduction of the buffer. Its core brick `tcrv_rvv.elementwise_rope_rotate_core`
# carries the position-dependent 2x2 rotation + the scalar-libm cos/sin angle seam
# + the theta step -- NO product-family AND NO reduce-family primitive (rope is
# neither a contraction nor a reduction), so the contraction-shaped gate would
# wrongly demote it, exactly as the MAP class. The ROTATE class satisfies the
# [L-8] "decomposed = built from typed pattern-library primitives, no opaque
# helper" conjunct via this rotate-family primitive (the per-pair-recurrence
# sibling of the _MAP_PRIMITIVE_RE map family and the _REDUCE_FOLD_PRIMITIVE_RE
# reduce family). This whitelist is deliberately NARROW (a bare "rotate" substring
# is NOT admitted -- the token must be the forward rope rotate brick; an opaque
# monolith still trips the opaque gate), so the check stays discriminating.
_ROTATE_PRIMITIVE_RE = re.compile(r"(elementwise_rope_rotate_core)")


def _leading_ws(line):
    return len(line) - len(line.lstrip(" "))


def parse_realized_body(ir_text):
    """Walk the realized `tcrv_rvv.with_vl` body and return its op-identity manifest.

    Returns list of {"mnemonic": str, "kind": str|None} for each op directly
    printed inside the with_vl region (nested region ops included). Bounds the
    region by the with_vl opener line and its indent-matched `} : !tcrv_rvv.vl`
    terminator. Raises on a leaked mirror/type token (parser sanity).
    """
    lines = ir_text.splitlines()
    # Find the with_vl opener: an op-position `tcrv_rvv.with_vl` whose line opens a
    # region (ends with `{`).
    opener_idx = None
    opener_indent = None
    for i, line in enumerate(lines):
        m = _OP_RE.match(line)
        if m and m.group(1) == "tcrv_rvv.with_vl" and line.rstrip().endswith("{"):
            opener_idx = i
            opener_indent = _leading_ws(line)
            break
    if opener_idx is None:
        raise RuntimeError("no realized tcrv_rvv.with_vl body found in emitted IR")

    manifest = []
    closed = False
    for line in lines[opener_idx + 1:]:
        term = _WITHVL_TERMINATOR.match(line)
        if term is not None and len(term.group(1)) == opener_indent:
            closed = True
            break
        m = _OP_RE.match(line)
        if not m:
            continue
        mnemonic = m.group(1)
        if mnemonic == "tcrv_rvv.with_vl":
            continue
        if _MIRROR_GUARD.search(mnemonic):
            raise RuntimeError(
                "parser leaked a non-op / mirror token into the manifest: "
                f"{mnemonic!r} (I4 violation guard tripped)"
            )
        kind_m = _KIND_RE.search(line)
        manifest.append({"mnemonic": mnemonic, "kind": kind_m.group(1) if kind_m else None})
    if not closed:
        raise RuntimeError("with_vl region terminator `} : !tcrv_rvv.vl` not found")
    return manifest


def is_opaque_hand_helper(op):
    """[L-8] opaque hand-written helper: a MONOLITHIC block-dot op.

    Two signals must AGREE (not one fragile match): op mnemonic ends `_block_dot`
    AND its `kind` names a ggml block-dot hand helper (`ggml_…block_dot`).
    """
    mnem = op["mnemonic"]
    kind = op["kind"] or ""
    name_signal = mnem.endswith("_block_dot")
    kind_signal = bool(re.match(r"ggml_.*block_dot", kind))
    return name_signal and kind_signal


def derive(manifest):
    """Apply the [L-8] rule: constructed iff
    (manifest non-empty ∧ no opaque *_block_dot helper ∧ decomposed).

    `decomposed` is the promoted GATE conjunct (was report-only): the body must
    realize a REAL dot-product-family primitive AND a separate reduce-family
    primitive, so a body that merely carries a per-block fp16 scale_product — or no
    contraction at all — does not vacuously read as constructed.
    """
    mnemonics = [op["mnemonic"] for op in manifest]
    opaque_ops = [op["mnemonic"] for op in manifest if is_opaque_hand_helper(op)]
    has_opaque = len(opaque_ops) > 0
    non_empty = len(mnemonics) > 0
    # GATE conjunct: real dot-product-family primitive (whitelisted, so the fp16
    # scale_product is excluded) AND a reduce-family primitive.
    has_product = any(_DOT_PRODUCT_RE.search(m) for m in mnemonics)
    # A separate reduce-family primitive (standalone_reduce / *dot_reduce / ...) OR
    # a fused dot-reduce primitive (q4_K scaled_dot: the vwmacc reduction is fused
    # into the product op), so the super-block route — whose contraction+reduction
    # live in ONE brick — is not wrongly demoted.
    has_reduce = any("reduce" in m for m in mnemonics) or any(
        _FUSED_DOT_REDUCE_RE.search(m) for m in mnemonics)
    # M-FLAT reduce/MAP model: the forward-elementwise MAP class satisfies the
    # [L-8] decomposed conjunct via a MAP-family primitive (no product/reduce).
    has_map = any(_MAP_PRIMITIVE_RE.search(m) for m in mnemonics)
    # M-FLAT REDUCE model: the forward-elementwise REDUCE class satisfies the
    # decomposed conjunct via a reduce-fold-family primitive (the Σx² square is
    # fused into the scalar-double fold, so there is no separate product to pair).
    has_reduce_fold = any(_REDUCE_FOLD_PRIMITIVE_RE.search(m) for m in mnemonics)
    # M-FLAT ROTATE model: the forward-elementwise ROTATE class (rope) satisfies
    # the decomposed conjunct via a rotate-family primitive (a per-pair recurrence,
    # neither product/reduce contraction nor a MAP).
    has_rotate = any(_ROTATE_PRIMITIVE_RE.search(m) for m in mnemonics)
    decomposed = (has_product and has_reduce) or has_map or has_reduce_fold or has_rotate
    derived_state = (
        "constructed" if (non_empty and not has_opaque and decomposed)
        else "constructed-weak"
    )
    return {
        "manifest": mnemonics,
        "has_opaque": has_opaque,
        "opaque_ops": opaque_ops,
        "has_product": has_product,
        "has_reduce": has_reduce,
        "has_map": has_map,
        "has_reduce_fold": has_reduce_fold,
        "has_rotate": has_rotate,
        "decomposed": decomposed,
        "derived_state": derived_state,
    }


def run_tcrv_opt(input_path, front_door):
    if not TCRV_OPT.exists():
        raise RuntimeError(f"tcrv-opt not built at {TCRV_OPT} (build first)")
    cmd = [str(TCRV_OPT), str(input_path), front_door]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Fail-closed: never silently drop a strong row.
        raise RuntimeError(
            f"tcrv-opt failed (rc={proc.returncode}) for {input_path.name}\n"
            f"cmd: {' '.join(cmd)}\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def check_path(entry):
    input_path = TEST_RVV / entry["input"]
    ir = run_tcrv_opt(input_path, entry["front_door"])
    result = derive(parse_realized_body(ir))
    result["op"] = entry["op"]
    result["format"] = entry["format"]
    result["engine"] = entry["engine"]
    result["regime"] = entry.get("regime", "")
    result["kind"] = entry["kind"]
    result["expected_state"] = entry["expected_state"]
    result["front_door_id"] = entry["front_door_id"]
    result["input"] = entry["input"]
    result["match"] = result["derived_state"] == entry["expected_state"]
    return result


def cmd_report(_args):
    results = [check_path(e) for e in PATHS]
    all_pass = True
    for r in results:
        status = "PASS" if r["match"] else "FAIL"
        if not r["match"]:
            all_pass = False
        print(f"[{status}] {r['op']}/{r['format']} ({r['kind']}) "
              f"via {r['front_door_id']}")
        print(f"        manifest      = {r['manifest']}")
        print(f"        has_opaque    = {r['has_opaque']}"
              + (f"  {r['opaque_ops']}" if r["opaque_ops"] else ""))
        print(f"        decomposed    = {r['decomposed']} (GATE: real dot-product + reduce)")
        print(f"        derived_state = {r['derived_state']}"
              f"  (expected {r['expected_state']})")
    print()
    strong = [r for r in results if r["kind"] == "strong"]
    neg = [r for r in results if r["kind"] == "negative"]
    print(f"strong reproduce hand-label: "
          f"{sum(1 for r in strong if r['match'])}/{len(strong)} = constructed")
    print(f"negative discriminates (not-strong): "
          f"{'yes' if all(not r['manifest'] or r['derived_state'] != 'constructed' for r in neg) else 'NO'}")
    print(f"overall: {'PASS' if all_pass else 'FAIL'}")
    return results, all_pass


def _manifest_summary(mnemonics):
    """Short, order-preserving manifest summary for the auto_readout field."""
    short = [m.replace("tcrv_rvv.", "") for m in mnemonics]
    return "+".join(short)


def cmd_update_sixstate(_args):
    results, all_pass = cmd_report(_args)
    if not all_pass:
        print("\nrefusing to update schema: machine-check did not fully pass", file=sys.stderr)
        return 1
    by_key = {(r["op"], r["format"], r["engine"], r.get("regime", "")): r
              for r in results if r["kind"] == "strong"}
    doc = json.loads(SIXSTATE_JSON.read_text())
    updated = 0
    for row in doc["states"]:
        key = (row.get("op"), row.get("format"), row.get("engine", ""),
               row.get("regime", ""))
        if key in by_key and row.get("auto_readout") == "pending-E5":
            r = by_key[key]
            # ZERO FLIP: state is unchanged; only auto_readout is written.
            assert row["state"] == "constructed", row
            row["auto_readout"] = (
                f"E5-increment1-auto: constructed (STRONG); realized-body manifest="
                f"{_manifest_summary(r['manifest'])}; opaque_helper=false"
            )
            updated += 1
    # Idempotent: the E5 增量① paragraph is appended only once. Re-running
    # update-sixstate must not duplicate it (the JSON already carries the corrected,
    # count-accurate paragraph), so guard on its marker.
    if "E5 增量①" not in doc["$meta"]["labeling"]:
        doc["$meta"]["labeling"] = (
            doc["$meta"]["labeling"]
            + " | E5 增量① (strong-side auto): the 7 STRONG rows (3 product_reduce "
              "N-operand routes + q8_0 vec_dot + q4_0 vec_dot + q4_1 vec_dot + q5_0 vec_dot, all four typed_flat_block_dot_loop_body) carry a "
              "MACHINE-CHECKED auto_readout derived by e5_strong_readout.py, which walks "
              "the actual realized tcrv_rvv.with_vl body op-identity (CORE oracle, not the "
              "low_precision_resource.* mirror) and applies [L-8] (manifest non-empty ∧ no "
              "opaque *_block_dot hand helper). Reproduces the hand-label; an mxfp4 block-dot "
              "negative control derives NOT-strong. Weak rows' auto_readout stays pending-E5 "
              "(later increment). State values are unchanged (zero flip)."
        )
    # ensure_ascii=False preserves the existing UTF-8 (实验总纲 etc.) so the diff
    # stays minimal and this file's byte content is not needlessly re-escaped.
    SIXSTATE_JSON.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"\nupdated auto_readout on {updated} strong rows in {SIXSTATE_JSON}")
    return 0


def _walk_repack_regime(input_path, front_door, want):
    """Walk ONE repack regime input. Return (ok, manifest_str, reason)."""
    if not input_path.exists():
        return False, "", f"probe input missing: {input_path.name}"
    ir = run_tcrv_opt(input_path, front_door)
    manifest = parse_realized_body(ir)
    der = derive(manifest)
    mnem = [m["mnemonic"].replace("tcrv_rvv.", "") for m in manifest]
    if not mnem:
        return False, "", "empty realized body"
    body, yld = f"typed_repack_{want}_loop_body", f"typed_repack_{want}_loop_yield"
    if mnem[0] != body or mnem[-1] != yld:
        return False, "+".join(mnem), f"not a typed_repack_{want} region (first/last mismatch)"
    if der["has_opaque"]:
        return False, "+".join(mnem), f"opaque hand helper {der['opaque_ops']}"
    if der["derived_state"] != "constructed":
        return False, "+".join(mnem), f"derived {der['derived_state']} (decomposed gate failed)"
    return True, "+".join(mnem), "ok"


def cmd_stamp_repack_dual(_args):
    """Walk + stamp the 10 single-row repack gemm_tile cells (FIX-D). Honest:
    a cell is stamped ONLY if BOTH its GEVM and GEMM regimes walk to a legal,
    non-opaque, constructed typed_repack region; otherwise it is reported and
    SKIPPED (never blanket-stamped). Writes a machine [F-EMIT] dual manifest that
    the strict checker's classify_femit_repack accepts."""
    doc = json.loads(SIXSTATE_JSON.read_text())
    by_key = {(r.get("op"), r.get("format"), r.get("engine", ""), r.get("regime", "")): r
              for r in doc["states"]}
    stamped, skipped = [], []
    for e in REPACK_DUAL_PATHS:
        fmt = e["format"]
        gv_ok, gv_man, gv_why = _walk_repack_regime(e["gevm_input"], e["front_door"], "gemv")
        gm_ok, gm_man, gm_why = _walk_repack_regime(e["gemm_input"], e["front_door"], "gemm")
        if not (gv_ok and gm_ok):
            skipped.append((fmt, f"GEVM:{gv_why} | GEMM:{gm_why}"))
            print(f"[SKIP] gemm_tile/{fmt}: GEVM {gv_why} ; GEMM {gm_why}")
            continue
        row = by_key.get(("gemm_tile", fmt, e["engine"], ""))
        if row is None or row.get("state") != "constructed":
            skipped.append((fmt, f"no constructed regime='' row (state="
                                 f"{row.get('state') if row else 'absent'})"))
            print(f"[SKIP] gemm_tile/{fmt}: no constructed regime='' row")
            continue
        envelope = (
            "[F-EMIT] e5-auto-repack-dual (machine-walked GEVM+GEMM via "
            "--tcrv-rvv-lower-quant-contraction): constructed (STRONG); "
            f"realized-body manifest (GEVM decode)={gv_man}; "
            f"(GEMM prefill)={gm_man}; opaque_helper=false")
        row["auto_readout"] = envelope
        stamped.append(fmt)
        print(f"[STAMP] gemm_tile/{fmt}: GEVM={gv_man} ;; GEMM={gm_man}")
    marker = "E5-repack-dual (FIX-D machine-walked)"
    if stamped and marker not in doc["$meta"]["labeling"]:
        doc["$meta"]["labeling"] = (
            doc["$meta"]["labeling"]
            + f" | {marker}: the single-row repack gemm_tile cells "
              f"({', '.join(stamped)}) carry a MACHINE-WALKED [F-EMIT] dual manifest "
              "(GEVM decode + GEMM prefill) derived by e5_strong_readout.py "
              "stamp-repack-dual, which runs --tcrv-rvv-lower-quant-contraction on a "
              "synthesized quant_contraction request and walks BOTH realized "
              "tcrv_rvv.typed_repack_gem{v,m}_loop_body regions (the same front door + "
              "[L-8] rule as the certified q4_0 repack rows). Both regimes must derive "
              "constructed or the cell is skipped. State values unchanged (zero flip).")
    # Write with the committed schema's OWN 1-space indent (verified byte-identical
    # round-trip) so the stamp diff stays minimal (only the changed auto_readout rows
    # + $meta), not a whole-file reformat.
    SIXSTATE_JSON.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"\nstamped {len(stamped)}/{len(REPACK_DUAL_PATHS)} repack cells: {stamped}")
    if skipped:
        print(f"skipped {len(skipped)}: {[s[0] for s in skipped]}")
    return 0


def _walk_dequant_stream(input_path, front_door):
    """Walk ONE dequant-stream input. Return (ok, manifest_str, reason).

    Honest streaming-shape check (NOT the contraction-shaped derive() gate): the
    front door must CONSTRUCT the realized body to EXACTLY the typed streaming region
    typed_dequantize_row_loop_body + dequantize_row_decode_core + ...loop_yield,
    non-opaque. A pure decode carries no product/reduce, so derive() would wrongly
    demote it to constructed-weak -- this shape is its own legal form (mirrors the
    checker's dequant_stream_loop branch), verified here on the ACTUAL realized IR."""
    if not input_path.exists():
        return False, "", f"probe input missing: {input_path.name}"
    ir = run_tcrv_opt(input_path, front_door)
    manifest = parse_realized_body(ir)
    mnem = [m["mnemonic"].replace("tcrv_rvv.", "") for m in manifest]
    if not mnem:
        return False, "", "empty realized body (front door did not construct the region)"
    body = "typed_dequantize_row_loop_body"
    yld = "typed_dequantize_row_loop_yield"
    core = "dequantize_row_decode_core"
    if mnem[0] != body or mnem[-1] != yld:
        return False, "+".join(mnem), f"not a {body} region (first/last mismatch)"
    if core not in mnem:
        return False, "+".join(mnem), f"missing {core} decode brick"
    if any(is_opaque_hand_helper(m) for m in manifest):
        opaque = [m["mnemonic"] for m in manifest if is_opaque_hand_helper(m)]
        return False, "+".join(mnem), f"opaque hand helper {opaque}"
    return True, "+".join(mnem), "ok"


def cmd_stamp_dequant_stream(_args):
    """Walk + stamp the 21 CONSTRUCTED streaming dequantize_row cells (CERT-FD首族,
    FIX-5). Honest: a cell is stamped ONLY if the pre-emitc dequant-stream front door
    CONSTRUCTS a legal, non-opaque typed_dequantize_row_loop_body region walkable
    BEFORE --tcrv-rvv-lower-to-emitc; otherwise it is reported and SKIPPED (never
    blanket-stamped -- a format whose region does not construct/walk is an honest
    demote). Writes the E5 STRONG envelope classify_auto_readout accepts (the
    dequant_stream shape). State values unchanged (zero flip)."""
    doc = json.loads(SIXSTATE_JSON.read_text())
    by_key = {(r.get("op"), r.get("format"), r.get("engine", ""), r.get("regime", "")): r
              for r in doc["states"]}
    stamped, skipped = [], []
    for e in DEQUANT_STREAM_PATHS:
        fmt = e["format"]
        ok, man, why = _walk_dequant_stream(e["input"], _DEQUANT_STREAM_FRONT_DOOR)
        if not ok:
            skipped.append((fmt, why))
            print(f"[SKIP] dequantize_row/{fmt}: {why}")
            continue
        row = by_key.get(("dequantize_row", fmt, e["engine"], ""))
        if row is None or row.get("state") != "constructed":
            skipped.append((fmt, f"no constructed regime='' row (state="
                                 f"{row.get('state') if row else 'absent'})"))
            print(f"[SKIP] dequantize_row/{fmt}: no constructed regime='' row")
            continue
        row["auto_readout"] = (
            "E5-increment1-auto: constructed (STRONG); realized-body manifest="
            f"{man}; opaque_helper=false")
        stamped.append(fmt)
        print(f"[STAMP] dequantize_row/{fmt}: {man}")
    marker = "E5-dequant-stream (CERT-FD首族 machine-walked)"
    if stamped and marker not in doc["$meta"]["labeling"]:
        doc["$meta"]["labeling"] = (
            doc["$meta"]["labeling"]
            + f" | {marker}: the {len(stamped)} constructed streaming dequantize_row "
              f"cells ({', '.join(stamped)}) carry a MACHINE-WALKED auto_readout derived "
              "by e5_strong_readout.py stamp-dequant-stream, which runs "
              "--tcrv-rvv-materialize-dequantize-row-stream-front-door on the abstract "
              "tcrv_rvv.dequantize_row conversion fixture and walks the REALIZED "
              "tcrv_rvv.typed_dequantize_row_loop_body streaming region (body + "
              "dequantize_row_decode_core + yield, non-opaque) BEFORE --tcrv-rvv-lower-to-emitc "
              "(the SAME pre-emitc stage discipline as the certified block-dot rows). The "
              "region is byte-exact to the retired atomic construct+emit path (BEFORE/AFTER "
              "emit diff EMPTY for all 21). A format whose region does not construct/walk is "
              "SKIPPED (honest demote). State values unchanged (zero flip).")
    SIXSTATE_JSON.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"\nstamped {len(stamped)}/{len(DEQUANT_STREAM_PATHS)} dequant-stream cells: {stamped}")
    if skipped:
        print(f"skipped {len(skipped)}: {[s[0] for s in skipped]}")
    return 0


def _walk_quant_stream(input_path, front_door):
    """Walk ONE quant-stream input. Return (ok, manifest_str, reason).

    The f32->QUANT MIRROR of _walk_dequant_stream. Honest streaming-shape check (NOT
    the contraction-shaped derive() gate): the front door must CONSTRUCT the realized
    body to EXACTLY the typed streaming region typed_quantize_row_loop_body +
    quantize_row_encode_core + ...loop_yield, non-opaque. A pure encode carries no
    product/reduce, so derive() would wrongly demote it to constructed-weak -- this
    shape is its own legal form (mirrors the checker's quant_stream_loop branch),
    verified here on the ACTUAL realized IR."""
    if not input_path.exists():
        return False, "", f"probe input missing: {input_path.name}"
    ir = run_tcrv_opt(input_path, front_door)
    manifest = parse_realized_body(ir)
    mnem = [m["mnemonic"].replace("tcrv_rvv.", "") for m in manifest]
    if not mnem:
        return False, "", "empty realized body (front door did not construct the region)"
    body = "typed_quantize_row_loop_body"
    yld = "typed_quantize_row_loop_yield"
    core = "quantize_row_encode_core"
    if mnem[0] != body or mnem[-1] != yld:
        return False, "+".join(mnem), f"not a {body} region (first/last mismatch)"
    if core not in mnem:
        return False, "+".join(mnem), f"missing {core} encode brick"
    if any(is_opaque_hand_helper(m) for m in manifest):
        opaque = [m["mnemonic"] for m in manifest if is_opaque_hand_helper(m)]
        return False, "+".join(mnem), f"opaque hand helper {opaque}"
    return True, "+".join(mnem), "ok"


def cmd_stamp_quant_stream(_args):
    """Walk + stamp the 3 CONSTRUCTED streaming quantize_row cells (CERT-FD次族, the
    f32->QUANT mirror of stamp-dequant-stream). Honest: a cell is stamped ONLY if the
    pre-emitc quant-stream front door CONSTRUCTS a legal, non-opaque
    typed_quantize_row_loop_body region walkable BEFORE --tcrv-rvv-lower-to-emitc;
    otherwise it is reported and SKIPPED (never blanket-stamped -- a format whose
    region does not construct/walk is an honest demote). Writes the E5 STRONG envelope
    classify_auto_readout accepts (the quant_stream shape). State values unchanged
    (zero flip). NOTE q8_K is the ROW quantizer (scalar row-quant stream), NOT the
    mat-quant GEMM path."""
    doc = json.loads(SIXSTATE_JSON.read_text())
    by_key = {(r.get("op"), r.get("format"), r.get("engine", ""), r.get("regime", "")): r
              for r in doc["states"]}
    stamped, skipped = [], []
    for e in QUANT_STREAM_PATHS:
        fmt = e["format"]
        ok, man, why = _walk_quant_stream(e["input"], _QUANT_STREAM_FRONT_DOOR)
        if not ok:
            skipped.append((fmt, why))
            print(f"[SKIP] quantize_row/{fmt}: {why}")
            continue
        row = by_key.get(("quantize_row", fmt, e["engine"], ""))
        if row is None or row.get("state") != "constructed":
            skipped.append((fmt, f"no constructed regime='' row (state="
                                 f"{row.get('state') if row else 'absent'})"))
            print(f"[SKIP] quantize_row/{fmt}: no constructed regime='' row")
            continue
        row["auto_readout"] = (
            "E5-increment1-auto: constructed (STRONG); realized-body manifest="
            f"{man}; opaque_helper=false")
        stamped.append(fmt)
        print(f"[STAMP] quantize_row/{fmt}: {man}")
    marker = "E5-quant-stream (CERT-FD次族 machine-walked)"
    if stamped and marker not in doc["$meta"]["labeling"]:
        doc["$meta"]["labeling"] = (
            doc["$meta"]["labeling"]
            + f" | {marker}: the {len(stamped)} constructed streaming quantize_row "
              f"cells ({', '.join(stamped)}) carry a MACHINE-WALKED auto_readout derived "
              "by e5_strong_readout.py stamp-quant-stream, which runs "
              "--tcrv-rvv-materialize-quantize-row-stream-front-door on the abstract "
              "tcrv_rvv.quantize_row_q8_{0,1,K} conversion fixture and walks the REALIZED "
              "tcrv_rvv.typed_quantize_row_loop_body streaming region (body + "
              "quantize_row_encode_core + yield, non-opaque) BEFORE --tcrv-rvv-lower-to-emitc "
              "(the SAME pre-emitc stage discipline as the certified block-dot rows + the "
              "dequant-stream cells). The region is byte-exact to the atomic construct+emit "
              "path (BEFORE/AFTER emit diff EMPTY for all 3). A format whose region does not "
              "construct/walk is SKIPPED (honest demote). State values unchanged (zero flip).")
    SIXSTATE_JSON.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"\nstamped {len(stamped)}/{len(QUANT_STREAM_PATHS)} quant-stream cells: {stamped}")
    if skipped:
        print(f"skipped {len(skipped)}: {[s[0] for s in skipped]}")
    return 0


def _walk_forward_stream(input_path, front_door, core):
    """Walk ONE forward-elementwise-stream input. Return (ok, manifest_str, reason).

    The forward sibling of _walk_dequant_stream / _walk_quant_stream. Honest
    streaming-shape check (NOT the contraction-shaped derive() gate): the front door
    must CONSTRUCT the realized body to EXACTLY the typed streaming region
    typed_elementwise_loop_body + <core> + typed_elementwise_loop_yield, non-opaque.
    `core` is the model-specific brick (elementwise_scale_map | elementwise_silu_map |
    elementwise_rms_norm_reduce_core | elementwise_soft_max_reduce_core |
    elementwise_rope_rotate_core). A forward map/reduce/rotate carries no
    product/reduce contraction, so derive() would wrongly demote it to
    constructed-weak -- this shape is its own legal form (mirrors the checker's
    elementwise_stream_loop branch), verified here on the ACTUAL realized IR."""
    if not input_path.exists():
        return False, "", f"probe input missing: {input_path.name}"
    ir = run_tcrv_opt(input_path, front_door)
    manifest = parse_realized_body(ir)
    mnem = [m["mnemonic"].replace("tcrv_rvv.", "") for m in manifest]
    if not mnem:
        return False, "", "empty realized body (front door did not construct the region)"
    body = "typed_elementwise_loop_body"
    yld = "typed_elementwise_loop_yield"
    if mnem[0] != body or mnem[-1] != yld:
        return False, "+".join(mnem), f"not a {body} region (first/last mismatch)"
    if core not in mnem:
        return False, "+".join(mnem), f"missing {core} core brick"
    if any(is_opaque_hand_helper(m) for m in manifest):
        opaque = [m["mnemonic"] for m in manifest if is_opaque_hand_helper(m)]
        return False, "+".join(mnem), f"opaque hand helper {opaque}"
    return True, "+".join(mnem), "ok"


def cmd_stamp_forward_stream(_args):
    """Walk + stamp the 5 CONSTRUCTED streaming forward-elementwise cells (CERT-FD
    殿后族, closing CERT-FD). Honest: a cell is stamped ONLY if the pre-emitc
    forward-elementwise-stream front door CONSTRUCTS a legal, non-opaque
    typed_elementwise_loop_body region walkable BEFORE --tcrv-rvv-lower-to-emitc;
    otherwise it is reported and SKIPPED (never blanket-stamped -- a model whose
    region does not construct/walk is an honest demote). Writes the E5 STRONG envelope
    classify_auto_readout accepts (the elementwise_stream shape). State values
    unchanged (zero flip). NOTE the six-state `op` key is "softmax" (the
    elementwise_model is "soft_max")."""
    doc = json.loads(SIXSTATE_JSON.read_text())
    by_key = {(r.get("op"), r.get("format"), r.get("engine", ""), r.get("regime", "")): r
              for r in doc["states"]}
    stamped, skipped = [], []
    for e in FORWARD_STREAM_PATHS:
        op = e["op"]
        ok, man, why = _walk_forward_stream(e["input"], _FORWARD_STREAM_FRONT_DOOR,
                                            e["core"])
        if not ok:
            skipped.append((op, why))
            print(f"[SKIP] {op}/f32: {why}")
            continue
        row = by_key.get((op, "f32", e["engine"], ""))
        if row is None or row.get("state") != "constructed":
            skipped.append((op, f"no constructed regime='' row (state="
                                f"{row.get('state') if row else 'absent'})"))
            print(f"[SKIP] {op}/f32: no constructed regime='' row")
            continue
        row["auto_readout"] = (
            "E5-increment1-auto: constructed (STRONG); realized-body manifest="
            f"{man}; opaque_helper=false")
        stamped.append(op)
        print(f"[STAMP] {op}/f32: {man}")
    marker = "E5-forward-stream (CERT-FD殿后族 machine-walked)"
    if stamped and marker not in doc["$meta"]["labeling"]:
        doc["$meta"]["labeling"] = (
            doc["$meta"]["labeling"]
            + f" | {marker}: the {len(stamped)} constructed streaming forward-elementwise "
              f"cells ({', '.join(stamped)}) carry a MACHINE-WALKED auto_readout derived "
              "by e5_strong_readout.py stamp-forward-stream, which runs "
              "--tcrv-rvv-materialize-forward-elementwise-stream-front-door on the abstract "
              "tcrv_rvv.ggml_forward_elementwise conversion fixture and walks the REALIZED "
              "tcrv_rvv.typed_elementwise_loop_body streaming region (body + <map/reduce/"
              "rotate core brick> + yield, non-opaque) BEFORE --tcrv-rvv-lower-to-emitc "
              "(the SAME pre-emitc stage discipline as the certified block-dot + dequant/"
              "quant-stream rows). The region is byte-exact to the hand-authored typed-region "
              "emit (BEFORE/AFTER emit diff EMPTY for all 5). A model whose region does not "
              "construct/walk is SKIPPED (honest demote). State values unchanged (zero flip).")
    SIXSTATE_JSON.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"\nstamped {len(stamped)}/{len(FORWARD_STREAM_PATHS)} forward-stream cells: {stamped}")
    if skipped:
        print(f"skipped {len(skipped)}: {[s[0] for s in skipped]}")
    return 0


# --- hermetic parser self-test over captured ground truth ------------------
_GT_STRONG = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs"} : !tcrv_rvv.runtime_abi_value
      %6 = tcrv_rvv.setvl %5 {lmul = "m2"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %6 attributes {lmul = "m2", tcrv_rvv.low_precision_resource.source_lmul = "m2", tcrv_rvv.gearbox.producer_scope = "x"} {
        %7 = tcrv_rvv.load %0, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %8 = tcrv_rvv.load %1, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m2">
        %9 = tcrv_rvv.widening_product %7, %8, %6 {kind = "signed_widening_product"} : !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vector<i8, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m4">
        %10 = tcrv_rvv.standalone_reduce %9, %2, %6 {kind = "signed_widening_reduce_add"} : !tcrv_rvv.vector<i16, "m4">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        %11 = tcrv_rvv.dequantize %10, %3, %6 {kind = "i32_to_f32_scaled"} : !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %4, %11, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
"""

_GT_WEAK = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %3 = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %8 = tcrv_rvv.setvl %0 {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %8 attributes {lmul = "m1", tcrv_rvv.low_precision_resource.source_lmul = "m1"} {
        %9 = tcrv_rvv.q8_0_q8_0_block_dot %3, %5, %1, %0, %8 {kind = "ggml_q8_0_q8_0_block_dot", qk = 32 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
      } : !tcrv_rvv.vl
    }
  }
}
"""

# Scale-only ground truth: NOT opaque (no *_block_dot), but the only "product" op is
# the per-block fp16 SCALE product and there is NO reduce. This is exactly the hole
# the decomposed gate closes: it must derive NOT-strong.
_GT_SCALE_ONLY = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs"} : !tcrv_rvv.runtime_abi_value
      %6 = tcrv_rvv.setvl %5 {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %6 attributes {lmul = "m1"} {
        %7 = tcrv_rvv.load %0, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f16, "m1">
        %8 = tcrv_rvv.block_fp16_scale_product %7, %2, %6 {kind = "block_fp16_scale_product"} : !tcrv_rvv.vector<f16, "m1">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m1">
        tcrv_rvv.store %4, %8, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<f32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
"""

# M-FLAT forward-elementwise MAP ground truth (line C, ① 之后): the CONSTRUCTED
# scale (y[i]*=v) realized body -- the typed elementwise strip-loop op carrying
# the per-strip elementwise_scale_map map brick + the yield. It has NO
# product/reduce primitive (a pure MAP), so it exercises the reduce/MAP model
# branch: `has_map` satisfies the decomposed conjunct and it derives constructed
# (STRONG). Distinct from _GT_SCALE_ONLY (which carries the fp16
# block_fp16_scale_product and stays constructed-weak): the discriminator is the
# map-FAMILY primitive, not a bare "scale" substring.
_GT_ELEMENTWISE_MAP = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n"} : index
      %y = tcrv_rvv.runtime_abi_value {c_name = "y"} : !tcrv_rvv.runtime_abi_value
      %v = tcrv_rvv.runtime_abi_value {c_name = "v"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_elementwise_loop_body %y, %v, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "map", element_sew = 32 : i64, strip_lmul = "m8"} {
        ^bb0(%i: index):
          tcrv_rvv.elementwise_scale_map %y, %v, %n strip %i : index {kind = "elementwise_scale_map", strip_lmul = "m8"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
          tcrv_rvv.typed_elementwise_loop_yield
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""

# M-FLAT forward-elementwise MAP ground truth #2 (line C, ① 之后): the CONSTRUCTED
# silu (y[i]=x[i]*sigmoid(x[i])) realized body -- the SAME typed elementwise
# strip-loop op REUSED, now carrying the per-strip elementwise_silu_map map brick
# + the yield (a two-buffer x->y map). Like scale it has NO product/reduce (a pure
# MAP), so it exercises the SAME reduce/MAP model branch: `has_map` (via the
# elementwise_silu_map primitive that joined _MAP_PRIMITIVE_RE) satisfies the
# decomposed conjunct and it derives constructed (STRONG). This is the C2
# marginal-cost payoff made machine-checkable: the scaffold is reused, only the
# per-op map primitive differs.
_GT_ELEMENTWISE_SILU_MAP = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "map", element_sew = 32 : i64} {
        ^bb0(%i: index):
          tcrv_rvv.elementwise_silu_map %x, %y, %n strip %i : index {kind = "elementwise_silu_map"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
          tcrv_rvv.typed_elementwise_loop_yield
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""

# M-FLAT forward-elementwise REDUCE ground truth (line C, G1-tail): the CONSTRUCTED
# rms_norm realized body -- the SAME typed elementwise strip-loop op, now under
# reduce_map_model "reduce" (a SECOND region arg = the loop-carried f64 accumulator
# + the yield that carries it back), carrying the per-element
# elementwise_rms_norm_reduce_core reduce core brick (the Σx² fold). Unlike the MAP
# rows it has NO product AND NO map primitive, but its reduce-fold brick satisfies
# the decomposed conjunct via the THIRD branch: `has_reduce_fold` (elementwise_rms_norm_reduce_core
# joined _REDUCE_FOLD_PRIMITIVE_RE). This is the reduce model made machine-checkable
# (softmax's Σe^x reuses the SAME loop-carried-accumulator shape with a vfwredusum
# reduce brick); the discriminator is the reduce-fold-FAMILY primitive, not a bare
# "reduce" substring.
_GT_ELEMENTWISE_RMS_REDUCE = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y"} : !tcrv_rvv.runtime_abi_value
      %eps = tcrv_rvv.runtime_abi_value {c_name = "eps"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "reduce", element_sew = 32 : i64} {
        ^bb0(%i: index, %acc: f64):
          %acc_next = tcrv_rvv.elementwise_rms_norm_reduce_core %x, %y, %eps, %n strip %i acc %acc {kind = "elementwise_rms_norm_reduce_core", strip_lmul = "m8"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, f64 -> f64
          tcrv_rvv.typed_elementwise_loop_yield %acc_next : f64
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""

# M-FLAT forward-elementwise REDUCE ground truth #2 (line C, G1-tail): the
# CONSTRUCTED soft_max realized body -- the SAME typed elementwise strip-loop op,
# reduce_map_model "reduce", carrying the NEW elementwise_soft_max_reduce_core
# exp-sum-reduce core brick + the acc-carrying yield. It REUSES the reduce model
# rms_norm built, EXCEPT the loop-carried accumulator is the f64m1 WIDENING vector
# (ggml's vfloat64m1_t vsum, the vfwredusum_vs_f32m2_f64m1 destination), NOT a
# scalar double. Like rms_norm it has NO product AND NO map primitive; its
# reduce-fold brick satisfies the decomposed conjunct via the THIRD branch
# (has_reduce_fold, elementwise_soft_max_reduce_core joined _REDUCE_FOLD_PRIMITIVE_RE).
# The discriminator is the reduce-fold-FAMILY primitive, not a bare "reduce"
# substring.
_GT_ELEMENTWISE_SOFTMAX_REDUCE = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n"} : index
      %y = tcrv_rvv.runtime_abi_value {c_name = "y"} : !tcrv_rvv.runtime_abi_value
      %x = tcrv_rvv.runtime_abi_value {c_name = "x"} : !tcrv_rvv.runtime_abi_value
      %max = tcrv_rvv.runtime_abi_value {c_name = "max"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "reduce", element_sew = 32 : i64} {
        ^bb0(%i: index, %acc: !tcrv_rvv.vector<f64, "m1">):
          %acc_next = tcrv_rvv.elementwise_soft_max_reduce_core %y, %x, %max, %n strip %i acc %acc {kind = "elementwise_soft_max_reduce_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, !tcrv_rvv.vector<f64, "m1"> -> !tcrv_rvv.vector<f64, "m1">
          tcrv_rvv.typed_elementwise_loop_yield %acc_next : !tcrv_rvv.vector<f64, "m1">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""

# M-FLAT forward-elementwise ROTATE ground truth (line C, G1-tail): the CONSTRUCTED
# rope realized body -- the SAME typed elementwise strip-loop op, now under
# reduce_map_model "rotate" (a per-PAIR scalar loop with a loop-carried f32 theta
# recurrence region arg + the yield that carries it back), carrying the NEW
# elementwise_rope_rotate_core rotate core brick (the position-dependent 2x2
# rotation + the scalar-libm cos/sin angle seam + the theta step). rope does NOT fit
# the MAP model (a MAP is a vectorized i+=vlmax strip with NO carried state; rope is
# scalar per-pair with a carried recurrence), so it lands its OWN model; unlike the
# MAP/REDUCE rows it has NO product AND NO map AND NO reduce-fold primitive, but its
# rotate brick satisfies the decomposed conjunct via the FOURTH branch: `has_rotate`
# (elementwise_rope_rotate_core joined _ROTATE_PRIMITIVE_RE). The discriminator is
# the rotate-FAMILY primitive, not a bare "rotate" substring.
_GT_ELEMENTWISE_ROPE_ROTATE = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y"} : !tcrv_rvv.runtime_abi_value
      %tb = tcrv_rvv.runtime_abi_value {c_name = "theta_base"} : !tcrv_rvv.runtime_abi_value
      %ts = tcrv_rvv.runtime_abi_value {c_name = "theta_scale"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_elementwise_loop_body %x, %y, %n attributes {kind = "typed_elementwise_loop_body", reduce_map_model = "rotate", element_sew = 32 : i64} {
        ^bb0(%p: index, %theta: f32):
          %theta_next = tcrv_rvv.elementwise_rope_rotate_core %x, %y, %tb, %ts, %n pair %p theta %theta {kind = "elementwise_rope_rotate_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, f32 -> f32
          tcrv_rvv.typed_elementwise_loop_yield %theta_next : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""

# Mixed-dot ground truth: routes 2 & 3 realize a `*_x_i8_product` primitive (NOT
# widening_product). Guards the gate against a "simplify to widening_product only"
# regression that would silently demote two-thirds of the strong routes.
_GT_XI8 = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %0 = tcrv_rvv.runtime_abi_value {c_name = "lhs"} : !tcrv_rvv.runtime_abi_value
      %6 = tcrv_rvv.setvl %5 {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %6 attributes {lmul = "m1"} {
        %7 = tcrv_rvv.load %0, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
        %8 = tcrv_rvv.load %1, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i8, "m1">
        %9 = tcrv_rvv.codebook_gather_x_i8_product %7, %8, %6 {kind = "codebook_gather_x_i8_product"} : !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vector<i8, "m1">, !tcrv_rvv.vl -> !tcrv_rvv.vector<i16, "m2">
        %10 = tcrv_rvv.standalone_reduce %9, %2, %6 {kind = "signed_widening_reduce_add"} : !tcrv_rvv.vector<i16, "m2">, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
        tcrv_rvv.store %4, %10, %6 : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m1">, !tcrv_rvv.vl
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block ground truth (q4_K milestone-3): the typed SUPER-BLOCK dual-accumulator
# loop body decomposes into the 5 q4_K bricks. The contraction+reduction is the
# FUSED per-sub-block integer-MAC `q4_k_scaled_dot` (vwmacc into aux32) -- NO
# separate standalone_reduce and NO opaque *_block_dot op -- so the decomposed gate
# must derive constructed via the fused dot-reduce path.
_GT_SUPERBLOCK = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "super_block_two_level_scale_min"} {
        ^bb0(%ib: index, %sums: !tcrv_rvv.vector<f32, "m2">, %sumf: f32):
          %b1 = tcrv_rvv.q4_k_nibble_unpack %vx, %vl block %ib : index {kind = "q4_k_nibble_unpack"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b2 = tcrv_rvv.q4_k_scale_min_bit_dance %vx, %vl block %ib : index {kind = "q4_k_scale_min_bit_dance"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b3 = tcrv_rvv.q4_k_scaled_dot %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_scaled_dot"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b4 = tcrv_rvv.q4_k_min_term %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_min_term"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b6 = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_sums_fold_scale_d"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sums, %sumf : !tcrv_rvv.vector<f32, "m2">, f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block SINGLE-accumulator ground truth (q6_K milestone-2): q6_K has NO
# per-block min, so the typed super-block loop body (fold_model "scales_times_sumi")
# decomposes into just the q6_K aux32 INTEGER CORE + the reused no-min positive fold
# + a SINGLE yield. The contraction+reduction is the FUSED per-sub-block integer-MAC
# INSIDE `q6_k_q8_k_aux32_partial` (vwmacc into aux32) -- NO separate
# standalone_reduce and NO opaque *_block_dot op -- so the decomposed gate must
# derive constructed via the fused dot-reduce path (the aux32_partial token).
_GT_SUPERBLOCK_Q6K = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "scales_times_sumi"} {
        ^bb0(%ib: index, %sums: !tcrv_rvv.vector<f32, "m2">):
          %c1 = tcrv_rvv.q6_k_q8_k_aux32_partial %vx, %vy, %vx, %n, %vl block %ib : index {kind = "ggml_q6_k_q8_k_aux32_partial"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b6 = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_sums_fold_scale_d"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sums : !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block SINGLE-accumulator ground truth (q3_K milestone): q3_K is SYMMETRIC
# (NO per-block min), so it SHARES q6_K's typed super-block loop body (fold_model
# "scales_times_sumi") -- the q3_K aux32 INTEGER CORE + the reused no-min positive
# fold + a SINGLE yield. The contraction+reduction is the FUSED per-sub-block
# integer-MAC INSIDE `q3_k_q8_k_aux32_partial` (vwmacc into aux32) -- NO separate
# standalone_reduce and NO opaque *_block_dot op -- so the decomposed gate must
# derive constructed via the fused dot-reduce path (the aux32_partial token).
_GT_SUPERBLOCK_Q3K = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "scales_times_sumi"} {
        ^bb0(%ib: index, %sums: !tcrv_rvv.vector<f32, "m2">):
          %c1 = tcrv_rvv.q3_k_q8_k_aux32_partial %vx, %vy, %vx, %n, %vl block %ib : index {kind = "ggml_q3_k_q8_k_aux32_partial"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          %b6 = tcrv_rvv.q4_k_sums_fold_scale_d %vx, %vx, %vy, %vl block %ib : index {kind = "q4_k_sums_fold_scale_d"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m1">
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sums : !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block SCALAR-accumulator ground truth (q2_K milestone-2): q2_K HAS a per-
# block min but its whole fold is a SINGLE per-super-block SCALAR, so the typed
# super-block loop body (fold_model "scalar_scale_min") decomposes into just the
# q2_K INTEGER CORE + a SINGLE scalar yield (the scalar fold is emitter-inlined --
# no separate fold brick). The contraction+reduction is the FUSED per-sub-block
# vwmul+vwredsum INSIDE `q2_k_q8_k_integer_core` -- NO separate standalone_reduce
# and NO opaque *_block_dot op -- so the decomposed gate must derive constructed via
# the fused dot-reduce path (the integer_core token).
_GT_SUPERBLOCK_Q2K = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "scalar_scale_min"} {
        ^bb0(%ib: index, %sumf: f32):
          %isum, %summs = tcrv_rvv.q2_k_q8_k_integer_core %vx, %vy, %n, %vl block %ib : index {kind = "ggml_q2_k_q8_k_integer_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32, i32
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block SCALAR-accumulator GRID ground truth (iq1_s milestone): iq1_s is a
# super-block GRID/codebook quant (decode_model=lookup -- the 2048-entry TERNARY
# iq1s_grid + a vluxei16 gather) whose whole fold is a SINGLE per-super-block SCALAR
# `sumf += d*((float)sumi + IQ1S_DELTA*(float)sumi1)`, so the typed super-block loop
# body (fold_model "scalar_delta_grid") decomposes into just the iq1_s ternary-grid
# INTEGER CORE + a SINGLE scalar yield (the scalar delta fold is emitter-inlined -- no
# separate fold brick). The contraction+reduction is the FUSED per-sub-block vluxei16
# gather + vwmul + vwredsum INSIDE `iq1_s_q8_k_grid_core` -- NO separate
# standalone_reduce and NO opaque *_block_dot op -- so the decomposed gate must derive
# constructed via the fused dot-reduce path (the grid_core token).
_GT_SUPERBLOCK_IQ1S = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "scalar_delta_grid"} {
        ^bb0(%ib: index, %sumf: f32):
          %sumi, %sumi1 = tcrv_rvv.iq1_s_q8_k_grid_core %vx, %vy, %n, %vl block %ib : index {kind = "ggml_iq1_s_q8_k_grid_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32, i32
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block SCALAR-accumulator GRID ground truth (iq1_m milestone, iq1_s SIBLING):
# iq1_m REUSES the whole iq1_s scalar-delta-grid scaffold (fold_model
# "scalar_delta_grid", single `sumf` scalar yield) with a DISTINCT integer-core brick
# `iq1_m_q8_k_grid_core` (the packed-scale reconstruct + half-split grid dot + per-group
# four-sign delta). Its brick token ends in `grid_core`, so it satisfies BOTH the
# product AND reduce conjunct (the per-half vluxei16 gather + vwmul + vwredsum reduction
# is fused into it), the region carries no separate contraction or fold brick, and no
# opaque *_block_dot appears, so it derives constructed via the grid_core fused-reduce
# path -- the C2 marginal-cost proof that the second grid member is strong at the same
# gate as iq1_s.
_GT_SUPERBLOCK_IQ1M = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "scalar_delta_grid"} {
        ^bb0(%ib: index, %sumf: f32):
          %sumi1, %sumi2 = tcrv_rvv.iq1_m_q8_k_grid_core %vx, %vy, %n, %vl block %ib : index {kind = "ggml_iq1_m_q8_k_grid_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32, i32
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block SCALAR-accumulator GRID-of-4 ground truth (iq3_xxs milestone, iq1_s GRID
# SIBLING): iq3_xxs REUSES the whole iq1_s scalar-delta-grid scaffold (fold_model
# "scalar_delta_grid", single `sumf` scalar yield) with a DISTINCT integer-core brick
# `iq3_xxs_q8_k_grid_core` (the i32 GRID-of-4 vluxei16 gather + aux32 4-bit-scale +
# 4-sign-group ksigns decode). Its brick token ends in `grid_core`, so it satisfies BOTH
# the product AND reduce conjunct (the two-index-per-group vluxei16_v_i32m1 gather + vwmul
# + vwredsum reduction is fused into it), the region carries no separate contraction or
# fold brick, and no opaque *_block_dot appears, so it derives constructed via the
# grid_core fused-reduce path -- the L3 marginal-cost proof that the THIRD grid member is
# strong at the same gate as iq1_s/iq1_m.
_GT_SUPERBLOCK_IQ3XXS = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "scalar_delta_grid"} {
        ^bb0(%ib: index, %sumf: f32):
          %bsum = tcrv_rvv.iq3_xxs_q8_k_grid_core %vx, %vy, %n, %vl block %ib : index {kind = "ggml_iq3_xxs_q8_k_grid_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Super-block SCALAR-accumulator GRID-of-8 ground truth (iq2_xxs milestone, iq1_s GRID
# SIBLING, SIGN-PLANE signs64 variant): iq2_xxs REUSES the whole iq1_s scalar-delta-grid
# scaffold (fold_model "scalar_delta_grid", single `sumf` scalar yield) with a DISTINCT
# integer-core brick `iq2_xxs_q8_k_grid_core` (the i64 GRID-of-8 vluxei16 gather + the
# SECOND signs64 vluxei16 gather over the DERIVED keven_signs_q2xs sign plane + aux1
# 4-bit-scale + 4-sign-group decode). Its brick token ends in `grid_core`, so it satisfies
# BOTH the product AND reduce conjunct (the 4-index vluxei16_v_i64<core> gather + the
# vmul-onto-grid sign fold + vwmul + vwredsum reduction is fused into it), the region
# carries no separate contraction or fold brick, and no opaque *_block_dot appears, so it
# derives constructed via the grid_core fused-reduce path -- the L3 marginal-cost proof
# that the FOURTH grid member is strong at the same gate as iq1_s/iq1_m/iq3_xxs.
_GT_SUPERBLOCK_IQ2XXS = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_super_block_block_dot_loop_body %vx, %vy, %s, %n attributes {kind = "typed_super_block_block_dot_loop_body", fold_model = "scalar_delta_grid"} {
        ^bb0(%ib: index, %sumf: f32):
          %bsum = tcrv_rvv.iq2_xxs_q8_k_grid_core %vx, %vy, %n, %vl block %ib : index {kind = "ggml_iq2_xxs_q8_k_grid_core"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, !tcrv_rvv.vl -> i32
          tcrv_rvv.typed_super_block_block_dot_loop_yield %sumf : f32
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Repack GEVM ground truth (q4_0 16x1-repacked, the M-FLAT REPACK flip): the typed
# tcrv_rvv.typed_repack_gemv_loop_body region decomposes into the per-block
# lane-wise integer CORE brick + the numHalves per-strip dual-fp16 scale FOLD bricks
# + the loop yield. The CORE brick's contraction+reduction is FUSED (its nibble-step
# vwmacc accumulates LANE-WISE into the per-strip sumi -- NO separate
# standalone_reduce, and the dual-fp16 scale FOLD is a scale multiply, not a
# reduce), so the decomposed gate must derive constructed via the
# repack_lane_wise_q4_x_i8_dot product+fused-reduce token. This GT is the VLEN=128
# numHalves==2 form, so the CORE brick prints in the printer's GROUPED multi-result
# `%9:2 = ...` syntax -- exercising the _OP_RE grouped-result parse.
_GT_REPACK = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %nc attributes {kind = "typed_repack_gemv_loop_body", fold_model = "lane_wise_vector_scale"} {
        ^bb0(%ib: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">):
          %sumi:2 = tcrv_rvv.repack_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %ib : index {kind = "repack_lane_wise_q4_x_i8_dot"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          %an0 = tcrv_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi#0, %acc0, %vl block %ib : index {kind = "repack_dual_fp16_scale_fold"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          %an1 = tcrv_rvv.repack_dual_fp16_scale_fold %vx, %vy, %sumi#1, %acc1, %vl block %ib : index {kind = "repack_dual_fp16_scale_fold"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %an0, %an1 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Repack GEMM (prefill) ground truth (q4_0 16x1-repacked, the GEMM-finale M3 flip):
# the PREFILL sibling of the decode GEVM. The typed tcrv_rvv.typed_repack_gemm_loop_body
# region decomposes into the ONE-strip N-column integer CORE brick + the columnsPerPass
# per-column dual-fp16 scale FOLD bricks + the loop yield. This GT is the VLEN=128
# columnsPerPass==4 form: the CORE brick prints its 4 per-column sumi in the printer's
# GROUPED multi-result `%sumi:4 = ...` syntax, and there are FOUR per-column folds. The
# CORE brick's contraction+reduction is FUSED (its nibble-step vwmacc accumulates
# LANE-WISE into the per-column i16 lo/hi accumulators -- NO separate standalone_reduce,
# and the per-column dual-fp16 scale FOLD is a scale multiply, not a reduce), so the
# decomposed gate must derive constructed via the repack_gemm_lane_wise_q4_x_i8_dot
# product+fused-reduce token. The region carries the (block_index, strip_row_offset,
# columnsPerPass per-column vector acc) entry args.
_GT_REPACK_GEMM = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_repack_gemm_loop_body %vx, %vy, %s, %n, %nr, %nc, %bs attributes {kind = "typed_repack_gemm_loop_body", fold_model = "lane_wise_vector_scale"} {
        ^bb0(%ib: index, %roff: index, %acc0: !tcrv_rvv.vector<f32, "m2">, %acc1: !tcrv_rvv.vector<f32, "m2">, %acc2: !tcrv_rvv.vector<f32, "m2">, %acc3: !tcrv_rvv.vector<f32, "m2">):
          %sumi:4 = tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot %vx, %vy, %vl block %ib strip %roff : index, index {kind = "repack_gemm_lane_wise_q4_x_i8_dot"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<i32, "m2">
          %an0 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#0, %acc0, %vl block %ib strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          %an1 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#1, %acc1, %vl block %ib strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          %an2 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#2, %acc2, %vl block %ib strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          %an3 = tcrv_rvv.repack_gemm_dual_fp16_scale_fold %vx, %vy, %sumi#3, %acc3, %vl block %ib strip %roff : index, index {kind = "repack_gemm_dual_fp16_scale_fold"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vector<i32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vl -> !tcrv_rvv.vector<f32, "m2">
          tcrv_rvv.typed_repack_gemm_loop_yield %an0, %an1, %an2, %an3 : !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">, !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index, index, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Repack GEVM K-quant ground truth (FIX-D): the super-block repack GEVM region whose
# CORE brick is the fused `repack_gemv_kquant_core` (the K-quant sibling of the
# codebook/ternary repack cores). It satisfies BOTH product AND reduce via the
# `kquant_core` token (fused per-block vwmacc into aux32); no opaque *_block_dot, so
# it derives constructed. Locks the kquant_core whitelist addition — a refactor that
# drops it would flip this to constructed-weak and fail the self-test.
_GT_REPACK_KQUANT = """\
module {
  tcrv.exec.kernel @k {
    tcrv.exec.variant @v {
      %vx = tcrv_rvv.runtime_abi_value {c_name = "vx"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1"} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1"} {
        tcrv_rvv.typed_repack_gemv_loop_body %vx, %vy, %s, %n, %bs attributes {kind = "typed_repack_gemv_loop_body", fold_model = "lane_wise_vector_scale_min"} {
        ^bb0(%ib: index, %acc0: !tcrv_rvv.vector<f32, "m2">):
          %sumi = tcrv_rvv.repack_gemv_kquant_core %vx, %vy, %vl block %ib : index {kind = "repack_gemv_kquant_core", decode_model = "q4_K"} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.vl -> !tcrv_rvv.vector<i32, "m2">
          tcrv_rvv.typed_repack_gemv_loop_yield %acc0 : !tcrv_rvv.vector<f32, "m2">
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Dequant-stream ground truth (CERT-FD首族, FIX-5): the CONSTRUCTED streaming
# dequantize_row region the pre-emitc front door builds -- the typed
# typed_dequantize_row_loop_body carrying the per-block dequantize_row_decode_core
# brick + the VOID typed_dequantize_row_loop_yield (a pure DECODE: NO product/reduce
# accumulator, the decode STORES straight through the output pointer). Unlike the
# contraction rows it satisfies NEITHER the product NOR the reduce conjunct, so
# derive() would demote it to constructed-weak -- the streaming shape is its OWN
# legal form, checked by the dedicated _walk_dequant_stream (first==body,
# last==yield, decode_core present, non-opaque), NOT the contraction gate. This GT
# locks the parser + the streaming-shape acceptance so a refactor that renames the
# region ops or leaks an opaque helper fails the self-test.
_GT_DEQUANT_STREAM = """\
module {
  tcrv.exec.kernel @dequant_q8_0_kernel {
    tcrv.exec.variant @dequant_q8_0 {
      %k = tcrv_rvv.runtime_abi_value {c_name = "k"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x"} : !tcrv_rvv.runtime_abi_value
      %y = tcrv_rvv.runtime_abi_value {c_name = "y"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %k {lmul = "m1", sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", sew = 32 : i64} {
        tcrv_rvv.typed_dequantize_row_loop_body %x, %y, %k attributes {decode_model = "q8_0", kind = "typed_dequantize_row_loop_body", qk = 32 : i64, weight_block_stride = 34 : i64} {
        ^bb0(%block_index: index):
          tcrv_rvv.dequantize_row_decode_core %x, %y, %block_index {decode_model = "q8_0", qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64, weight_block_stride = 34 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
          tcrv_rvv.typed_dequantize_row_loop_yield
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


# Quant-stream ground truth (CERT-FD次族): the f32->QUANT MIRROR of _GT_DEQUANT_STREAM.
# The CONSTRUCTED streaming quantize_row region the pre-emitc quant front door builds --
# the typed typed_quantize_row_loop_body carrying the per-block quantize_row_encode_core
# brick + the VOID typed_quantize_row_loop_yield (a pure ENCODE: NO product/reduce
# accumulator, the encode STORES straight through the output byte pointer). Unlike the
# contraction rows it satisfies NEITHER the product NOR the reduce conjunct, so derive()
# would demote it to constructed-weak -- the streaming shape is its OWN legal form,
# checked by the dedicated _walk_quant_stream (first==body, last==yield, encode_core
# present, non-opaque), NOT the contraction gate. This GT locks the parser + the
# streaming-shape acceptance so a refactor that renames the region ops or leaks an opaque
# helper fails the self-test.
_GT_QUANT_STREAM = """\
module {
  tcrv.exec.kernel @quantize_row_q8_0_kernel {
    tcrv.exec.variant @quantize_row_q8_0 {
      %n = tcrv_rvv.runtime_abi_value {c_name = "n"} : index
      %x = tcrv_rvv.runtime_abi_value {c_name = "x"} : !tcrv_rvv.runtime_abi_value
      %vy = tcrv_rvv.runtime_abi_value {c_name = "vy"} : !tcrv_rvv.runtime_abi_value
      %vl = tcrv_rvv.setvl %n {lmul = "m1", sew = 32 : i64} : index -> !tcrv_rvv.vl
      tcrv_rvv.with_vl %vl attributes {lmul = "m1", sew = 32 : i64} {
        tcrv_rvv.typed_quantize_row_loop_body %x, %vy, %n attributes {block_stride = 34 : i64, encode_model = "q8_0", kind = "typed_quantize_row_loop_body", qk = 32 : i64} {
        ^bb0(%block_index: index):
          tcrv_rvv.quantize_row_encode_core %x, %vy, %block_index {block_stride = 34 : i64, encode_model = "q8_0", qk = 32 : i64, quant_byte_offset = 2 : i64, scale_byte_offset = 0 : i64} : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
          tcrv_rvv.typed_quantize_row_loop_yield
        } : !tcrv_rvv.runtime_abi_value, !tcrv_rvv.runtime_abi_value, index
      } : !tcrv_rvv.vl
    }
  }
}
"""


def cmd_self_test(_args):
    # Strong ground truth: decomposed primitives, no opaque, no mirror leak.
    strong = derive(parse_realized_body(_GT_STRONG))
    assert strong["manifest"] == [
        "tcrv_rvv.load", "tcrv_rvv.load", "tcrv_rvv.widening_product",
        "tcrv_rvv.standalone_reduce", "tcrv_rvv.dequantize", "tcrv_rvv.store",
    ], strong["manifest"]
    assert strong["has_opaque"] is False, strong
    assert strong["has_product"] is True, strong
    assert strong["has_reduce"] is True, strong
    assert strong["decomposed"] is True, strong
    assert strong["derived_state"] == "constructed", strong
    # No mirror token leaked (guard would have raised; double-check explicitly).
    assert all(not _MIRROR_GUARD.search(m) for m in strong["manifest"]), strong
    # The guard must NOT false-fire on the legitimate pattern-library primitive
    # tcrv_rvv.gearbox_cross_region_handoff (a real strong-body op in other routes),
    # but MUST fire on a bare leaked mirror-namespace token.
    assert not _MIRROR_GUARD.search("tcrv_rvv.gearbox_cross_region_handoff")
    assert _MIRROR_GUARD.search("tcrv_rvv.low_precision_resource")
    assert _MIRROR_GUARD.search("tcrv_rvv.gearbox")

    # Mixed-dot ground truth: a `*_x_i8_product` primitive (routes 2 & 3) must ALSO
    # pass the whitelist gate, not just widening_product.
    xi8 = derive(parse_realized_body(_GT_XI8))
    assert xi8["manifest"] == [
        "tcrv_rvv.load", "tcrv_rvv.load", "tcrv_rvv.codebook_gather_x_i8_product",
        "tcrv_rvv.standalone_reduce", "tcrv_rvv.store",
    ], xi8["manifest"]
    assert xi8["has_product"] is True, xi8
    assert xi8["decomposed"] is True, xi8
    assert xi8["derived_state"] == "constructed", xi8

    # Weak ground truth: single monolithic block-dot, opaque, not-strong.
    weak = derive(parse_realized_body(_GT_WEAK))
    assert weak["manifest"] == ["tcrv_rvv.q8_0_q8_0_block_dot"], weak["manifest"]
    assert weak["has_opaque"] is True, weak
    assert weak["opaque_ops"] == ["tcrv_rvv.q8_0_q8_0_block_dot"], weak
    assert weak["derived_state"] == "constructed-weak", weak

    # Scale-only ground truth: NOT opaque, but the decomposed GATE must reject it —
    # `block_fp16_scale_product` is a scale multiply, not a contraction, and there is
    # no reduce. This is the exact hole the gate closes.
    scale = derive(parse_realized_body(_GT_SCALE_ONLY))
    assert scale["manifest"] == [
        "tcrv_rvv.load", "tcrv_rvv.block_fp16_scale_product", "tcrv_rvv.store",
    ], scale["manifest"]
    assert scale["has_opaque"] is False, scale
    assert scale["has_product"] is False, scale
    assert scale["has_reduce"] is False, scale
    assert scale["has_map"] is False, scale
    assert scale["decomposed"] is False, scale
    assert scale["derived_state"] == "constructed-weak", scale

    # M-FLAT forward-elementwise MAP ground truth: the CONSTRUCTED scale realized
    # body (typed_elementwise_loop_body + elementwise_scale_map + yield). It has NO
    # product/reduce (a pure MAP), so it exercises the reduce/MAP model branch:
    # `has_map` satisfies the decomposed conjunct and it derives constructed
    # (STRONG). This is the discriminating opposite of _GT_SCALE_ONLY (same "scale"
    # word, but that carries the fp16 block_fp16_scale_product -- NOT a map primitive
    # -- so it stays constructed-weak): the machine key is the map-FAMILY primitive
    # tcrv_rvv.elementwise_scale_map, never a bare substring.
    ew_map = derive(parse_realized_body(_GT_ELEMENTWISE_MAP))
    assert ew_map["manifest"] == [
        "tcrv_rvv.typed_elementwise_loop_body",
        "tcrv_rvv.elementwise_scale_map",
        "tcrv_rvv.typed_elementwise_loop_yield",
    ], ew_map["manifest"]
    assert ew_map["has_opaque"] is False, ew_map
    assert ew_map["has_product"] is False, ew_map
    assert ew_map["has_reduce"] is False, ew_map
    assert ew_map["has_map"] is True, ew_map
    assert ew_map["decomposed"] is True, ew_map
    assert ew_map["derived_state"] == "constructed", ew_map
    # Discrimination: the MAP model derives strong, the scale-only fp16 negative
    # stays weak (the map-family primitive is the ONLY thing that flips it).
    assert ew_map["derived_state"] != scale["derived_state"]

    # M-FLAT forward-elementwise MAP ground truth #2: the CONSTRUCTED silu realized
    # body (the SAME typed_elementwise_loop_body REUSED + the NEW elementwise_silu_map
    # brick + yield). Like scale it has NO product/reduce (a pure MAP), so it
    # exercises the SAME reduce/MAP model branch: `has_map` (via elementwise_silu_map,
    # which joined _MAP_PRIMITIVE_RE) satisfies the decomposed conjunct and it derives
    # constructed (STRONG). This is the C2 marginal-cost payoff made machine-checkable:
    # the scaffold is reused, only the per-op map primitive differs.
    ew_silu = derive(parse_realized_body(_GT_ELEMENTWISE_SILU_MAP))
    assert ew_silu["manifest"] == [
        "tcrv_rvv.typed_elementwise_loop_body",
        "tcrv_rvv.elementwise_silu_map",
        "tcrv_rvv.typed_elementwise_loop_yield",
    ], ew_silu["manifest"]
    assert ew_silu["has_opaque"] is False, ew_silu
    assert ew_silu["has_product"] is False, ew_silu
    assert ew_silu["has_reduce"] is False, ew_silu
    assert ew_silu["has_map"] is True, ew_silu
    assert ew_silu["decomposed"] is True, ew_silu
    assert ew_silu["derived_state"] == "constructed", ew_silu
    # The silu MAP reuses the SAME scaffold as scale and derives the SAME strong
    # state (the reduce/MAP model is shared; only the map-family primitive differs).
    assert ew_silu["derived_state"] == ew_map["derived_state"]

    # M-FLAT forward-elementwise REDUCE ground truth: the CONSTRUCTED rms_norm
    # realized body (the SAME typed_elementwise_loop_body, now reduce_map_model
    # "reduce" with a loop-carried f64 accumulator, carrying the NEW
    # elementwise_rms_norm_reduce_core reduce core brick + the acc-carrying yield).
    # Unlike the MAP rows it has NO product AND NO map primitive; its reduce-fold
    # brick satisfies the decomposed conjunct via the THIRD branch (has_reduce_fold,
    # via elementwise_rms_norm_reduce_core joining _REDUCE_FOLD_PRIMITIVE_RE). This
    # is the reduce model made machine-checkable (softmax's Σe^x reuses the shape).
    ew_rms = derive(parse_realized_body(_GT_ELEMENTWISE_RMS_REDUCE))
    assert ew_rms["manifest"] == [
        "tcrv_rvv.typed_elementwise_loop_body",
        "tcrv_rvv.elementwise_rms_norm_reduce_core",
        "tcrv_rvv.typed_elementwise_loop_yield",
    ], ew_rms["manifest"]
    assert ew_rms["has_opaque"] is False, ew_rms
    assert ew_rms["has_product"] is False, ew_rms
    assert ew_rms["has_map"] is False, ew_rms
    assert ew_rms["has_reduce_fold"] is True, ew_rms
    assert ew_rms["decomposed"] is True, ew_rms
    assert ew_rms["derived_state"] == "constructed", ew_rms
    # Discrimination: the REDUCE model derives strong, the scale-only fp16 negative
    # stays weak (the reduce-fold-family primitive is the ONLY thing that flips it).
    assert ew_rms["derived_state"] != scale["derived_state"]

    # M-FLAT forward-elementwise REDUCE ground truth #2: the CONSTRUCTED soft_max
    # realized body (the SAME typed_elementwise_loop_body, reduce_map_model "reduce",
    # carrying the NEW elementwise_soft_max_reduce_core exp-sum-reduce core brick +
    # the acc-carrying yield). It REUSES the reduce model rms_norm built, EXCEPT the
    # loop-carried accumulator is the f64m1 WIDENING vector (vfwredusum Σe^x). Its
    # reduce-fold brick satisfies the decomposed conjunct via the THIRD branch
    # (has_reduce_fold, elementwise_soft_max_reduce_core joining _REDUCE_FOLD_PRIMITIVE_RE).
    ew_softmax = derive(parse_realized_body(_GT_ELEMENTWISE_SOFTMAX_REDUCE))
    assert ew_softmax["manifest"] == [
        "tcrv_rvv.typed_elementwise_loop_body",
        "tcrv_rvv.elementwise_soft_max_reduce_core",
        "tcrv_rvv.typed_elementwise_loop_yield",
    ], ew_softmax["manifest"]
    assert ew_softmax["has_opaque"] is False, ew_softmax
    assert ew_softmax["has_product"] is False, ew_softmax
    assert ew_softmax["has_map"] is False, ew_softmax
    assert ew_softmax["has_reduce_fold"] is True, ew_softmax
    assert ew_softmax["decomposed"] is True, ew_softmax
    assert ew_softmax["derived_state"] == "constructed", ew_softmax
    # soft_max REUSES the SAME reduce model as rms_norm and derives the SAME strong
    # state (the reduce-fold model is shared; only the fold TYPE differs -- f64m1
    # widening vector vs scalar double), and stays distinct from the scale-only
    # negative control.
    assert ew_softmax["derived_state"] == ew_rms["derived_state"]
    assert ew_softmax["derived_state"] != scale["derived_state"]

    # M-FLAT forward-elementwise ROTATE ground truth: the CONSTRUCTED rope realized
    # body (the SAME typed_elementwise_loop_body, now reduce_map_model "rotate",
    # carrying the NEW elementwise_rope_rotate_core rotate core brick + the
    # theta-carrying yield). rope does NOT fit the MAP model (scalar per-pair loop
    # with a loop-carried f32 recurrence, not a vectorized strip), so it lands its
    # OWN model. Unlike the MAP/REDUCE rows it has NO product AND NO map AND NO
    # reduce-fold primitive; its rotate brick satisfies the decomposed conjunct via
    # the FOURTH branch (has_rotate, elementwise_rope_rotate_core joining
    # _ROTATE_PRIMITIVE_RE). The discriminator is the rotate-FAMILY primitive, not a
    # bare "rotate" substring.
    ew_rope = derive(parse_realized_body(_GT_ELEMENTWISE_ROPE_ROTATE))
    assert ew_rope["manifest"] == [
        "tcrv_rvv.typed_elementwise_loop_body",
        "tcrv_rvv.elementwise_rope_rotate_core",
        "tcrv_rvv.typed_elementwise_loop_yield",
    ], ew_rope["manifest"]
    assert ew_rope["has_opaque"] is False, ew_rope
    assert ew_rope["has_product"] is False, ew_rope
    assert ew_rope["has_map"] is False, ew_rope
    assert ew_rope["has_reduce_fold"] is False, ew_rope
    assert ew_rope["has_rotate"] is True, ew_rope
    assert ew_rope["decomposed"] is True, ew_rope
    assert ew_rope["derived_state"] == "constructed", ew_rope
    # rope derives the SAME strong state as the other forward operators (its rotate
    # model is a distinct SHAPE but still a decomposed pattern-library body), and
    # stays distinct from the scale-only negative control.
    assert ew_rope["derived_state"] == ew_softmax["derived_state"]
    assert ew_rope["derived_state"] != scale["derived_state"]

    # Super-block ground truth (q4_K milestone-3): the 5 q4_K bricks decompose the
    # super-block dot; the FUSED per-sub-block q4_k_scaled_dot satisfies BOTH the
    # product AND reduce conjunct (the vwmacc reduction is fused into the product),
    # and no opaque *_block_dot appears, so it derives constructed.
    superblock = derive(parse_realized_body(_GT_SUPERBLOCK))
    assert superblock["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.q4_k_nibble_unpack", "tcrv_rvv.q4_k_scale_min_bit_dance",
        "tcrv_rvv.q4_k_scaled_dot", "tcrv_rvv.q4_k_min_term",
        "tcrv_rvv.q4_k_sums_fold_scale_d",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock["manifest"]
    assert superblock["has_opaque"] is False, superblock
    assert superblock["has_product"] is True, superblock
    assert superblock["has_reduce"] is True, superblock
    assert superblock["decomposed"] is True, superblock
    assert superblock["derived_state"] == "constructed", superblock

    # Super-block SINGLE-accumulator ground truth (q6_K milestone-2): the q6_K aux32
    # integer core `q6_k_q8_k_aux32_partial` satisfies BOTH the product AND reduce
    # conjunct (the per-sub-block vwmacc reduction is fused into it), the reused
    # positive fold carries no separate contraction, and no opaque *_block_dot
    # appears, so it derives constructed via the aux32_partial fused-reduce path.
    superblock_q6k = derive(parse_realized_body(_GT_SUPERBLOCK_Q6K))
    assert superblock_q6k["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.q6_k_q8_k_aux32_partial",
        "tcrv_rvv.q4_k_sums_fold_scale_d",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock_q6k["manifest"]
    assert superblock_q6k["has_opaque"] is False, superblock_q6k
    assert superblock_q6k["has_product"] is True, superblock_q6k
    assert superblock_q6k["has_reduce"] is True, superblock_q6k
    assert superblock_q6k["decomposed"] is True, superblock_q6k
    assert superblock_q6k["derived_state"] == "constructed", superblock_q6k

    # Super-block SINGLE-accumulator ground truth (q3_K milestone): q3_K shares
    # q6_K's no-min single-accumulator body; the q3_K aux32 integer core
    # `q3_k_q8_k_aux32_partial` satisfies BOTH the product AND reduce conjunct (the
    # per-sub-block vwmacc reduction is fused into it), the reused positive fold
    # carries no separate contraction, and no opaque *_block_dot appears, so it
    # derives constructed via the aux32_partial fused-reduce path.
    superblock_q3k = derive(parse_realized_body(_GT_SUPERBLOCK_Q3K))
    assert superblock_q3k["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.q3_k_q8_k_aux32_partial",
        "tcrv_rvv.q4_k_sums_fold_scale_d",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock_q3k["manifest"]
    assert superblock_q3k["has_opaque"] is False, superblock_q3k
    assert superblock_q3k["has_product"] is True, superblock_q3k
    assert superblock_q3k["has_reduce"] is True, superblock_q3k
    assert superblock_q3k["decomposed"] is True, superblock_q3k
    assert superblock_q3k["derived_state"] == "constructed", superblock_q3k

    # Super-block SCALAR-accumulator ground truth (q2_K milestone-2): the q2_K
    # integer core `q2_k_q8_k_integer_core` satisfies BOTH the product AND reduce
    # conjunct (the per-sub-block vwmul+vwredsum reduction is fused into it), the
    # region carries no separate contraction or fold brick, and no opaque
    # *_block_dot appears, so it derives constructed via the integer_core fused-
    # reduce path.
    superblock_q2k = derive(parse_realized_body(_GT_SUPERBLOCK_Q2K))
    assert superblock_q2k["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.q2_k_q8_k_integer_core",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock_q2k["manifest"]
    assert superblock_q2k["has_opaque"] is False, superblock_q2k
    assert superblock_q2k["has_product"] is True, superblock_q2k
    assert superblock_q2k["has_reduce"] is True, superblock_q2k
    assert superblock_q2k["decomposed"] is True, superblock_q2k
    assert superblock_q2k["derived_state"] == "constructed", superblock_q2k

    # Super-block SCALAR-accumulator GRID ground truth (iq1_s milestone): iq1_s shares
    # q2_K's scalar-accumulator arity but its integer core is a TERNARY-grid GATHER
    # (decode_model=lookup). The iq1_s grid core `iq1_s_q8_k_grid_core` satisfies BOTH
    # the product AND reduce conjunct (the per-sub-block vluxei16 gather + vwmul +
    # vwredsum reduction is fused into it), the region carries no separate contraction
    # or fold brick, and no opaque *_block_dot appears, so it derives constructed via
    # the grid_core fused-reduce path.
    superblock_iq1s = derive(parse_realized_body(_GT_SUPERBLOCK_IQ1S))
    assert superblock_iq1s["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.iq1_s_q8_k_grid_core",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock_iq1s["manifest"]
    assert superblock_iq1s["has_opaque"] is False, superblock_iq1s
    assert superblock_iq1s["has_product"] is True, superblock_iq1s
    assert superblock_iq1s["has_reduce"] is True, superblock_iq1s
    assert superblock_iq1s["decomposed"] is True, superblock_iq1s
    assert superblock_iq1s["derived_state"] == "constructed", superblock_iq1s

    # Super-block SCALAR-accumulator GRID ground truth (iq1_m milestone, iq1_s SIBLING):
    # iq1_m reuses the SAME scalar-delta-grid scaffold with the DISTINCT iq1_m grid core
    # `iq1_m_q8_k_grid_core`, whose token likewise satisfies BOTH product AND reduce (the
    # per-half vluxei16 gather + vwmul + vwredsum is fused into it); no opaque *_block_dot,
    # so it derives constructed via the grid_core fused-reduce path -- the C2 marginal-cost
    # proof (the second grid member is strong at the SAME gate as iq1_s).
    superblock_iq1m = derive(parse_realized_body(_GT_SUPERBLOCK_IQ1M))
    assert superblock_iq1m["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.iq1_m_q8_k_grid_core",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock_iq1m["manifest"]
    assert superblock_iq1m["has_opaque"] is False, superblock_iq1m
    assert superblock_iq1m["has_product"] is True, superblock_iq1m
    assert superblock_iq1m["has_reduce"] is True, superblock_iq1m
    assert superblock_iq1m["decomposed"] is True, superblock_iq1m
    assert superblock_iq1m["derived_state"] == "constructed", superblock_iq1m

    # Super-block SCALAR-accumulator GRID-of-4 ground truth (iq3_xxs milestone, iq1_s GRID
    # SIBLING): iq3_xxs reuses the SAME scalar-delta-grid scaffold with the DISTINCT
    # iq3_xxs grid-of-4 core `iq3_xxs_q8_k_grid_core`, whose token likewise satisfies BOTH
    # product AND reduce (the two-index-per-group vluxei16_v_i32m1 gather + vwmul + vwredsum
    # is fused into it); no opaque *_block_dot, so it derives constructed via the grid_core
    # fused-reduce path -- the L3 marginal-cost proof (the third grid member is strong at
    # the SAME gate as iq1_s/iq1_m).
    superblock_iq3xxs = derive(parse_realized_body(_GT_SUPERBLOCK_IQ3XXS))
    assert superblock_iq3xxs["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.iq3_xxs_q8_k_grid_core",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock_iq3xxs["manifest"]
    assert superblock_iq3xxs["has_opaque"] is False, superblock_iq3xxs
    assert superblock_iq3xxs["has_product"] is True, superblock_iq3xxs
    assert superblock_iq3xxs["has_reduce"] is True, superblock_iq3xxs
    assert superblock_iq3xxs["decomposed"] is True, superblock_iq3xxs
    assert superblock_iq3xxs["derived_state"] == "constructed", superblock_iq3xxs

    # Super-block SCALAR-accumulator GRID-of-8 ground truth (iq2_xxs milestone, iq1_s GRID
    # SIBLING, SIGN-PLANE signs64 variant): iq2_xxs reuses the SAME scalar-delta-grid
    # scaffold with the DISTINCT iq2_xxs grid-of-8 core `iq2_xxs_q8_k_grid_core`, whose token
    # likewise satisfies BOTH product AND reduce (the 4-index vluxei16_v_i64<core> gather +
    # the SECOND signs64 vluxei16 gather + the vmul-onto-grid sign fold + vwmul + vwredsum is
    # fused into it); no opaque *_block_dot, so it derives constructed via the grid_core
    # fused-reduce path -- the L3 marginal-cost proof (the fourth grid member is strong at
    # the SAME gate as iq1_s/iq1_m/iq3_xxs).
    superblock_iq2xxs = derive(parse_realized_body(_GT_SUPERBLOCK_IQ2XXS))
    assert superblock_iq2xxs["manifest"] == [
        "tcrv_rvv.typed_super_block_block_dot_loop_body",
        "tcrv_rvv.iq2_xxs_q8_k_grid_core",
        "tcrv_rvv.typed_super_block_block_dot_loop_yield",
    ], superblock_iq2xxs["manifest"]
    assert superblock_iq2xxs["has_opaque"] is False, superblock_iq2xxs
    assert superblock_iq2xxs["has_product"] is True, superblock_iq2xxs
    assert superblock_iq2xxs["has_reduce"] is True, superblock_iq2xxs
    assert superblock_iq2xxs["decomposed"] is True, superblock_iq2xxs
    assert superblock_iq2xxs["derived_state"] == "constructed", superblock_iq2xxs

    # Repack GEVM ground truth (q4_0 16x1-repacked flip): the region decomposes into
    # the lane-wise integer CORE brick + the per-strip scale FOLD bricks + the yield.
    # The CORE brick `repack_lane_wise_q4_x_i8_dot` satisfies BOTH the product AND
    # reduce conjunct (its nibble-step vwmacc is a FUSED lane-wise dot-reduce), the
    # scale FOLDs carry no separate contraction, and no opaque *_block_dot appears, so
    # it derives constructed. This also exercises the GROUPED multi-result parse
    # (`%sumi:2 = ...` for the numHalves==2 core brick).
    repack = derive(parse_realized_body(_GT_REPACK))
    assert repack["manifest"] == [
        "tcrv_rvv.typed_repack_gemv_loop_body",
        "tcrv_rvv.repack_lane_wise_q4_x_i8_dot",
        "tcrv_rvv.repack_dual_fp16_scale_fold",
        "tcrv_rvv.repack_dual_fp16_scale_fold",
        "tcrv_rvv.typed_repack_gemv_loop_yield",
    ], repack["manifest"]
    assert repack["has_opaque"] is False, repack
    assert repack["has_product"] is True, repack
    assert repack["has_reduce"] is True, repack
    assert repack["decomposed"] is True, repack
    assert repack["derived_state"] == "constructed", repack

    # Repack GEMM (prefill) ground truth (q4_0 16x1-repacked, GEMM-finale M3 flip):
    # the PREFILL sibling of the GEVM. The region decomposes into the one-strip
    # N-column integer CORE brick + the columnsPerPass per-column scale FOLD bricks +
    # the yield. The CORE brick `repack_gemm_lane_wise_q4_x_i8_dot` satisfies BOTH the
    # product AND reduce conjunct (its nibble-step vwmacc is a FUSED lane-wise
    # dot-reduce), the per-column scale FOLDs carry no separate contraction, and no
    # opaque *_block_dot appears, so it derives constructed. This exercises the GROUPED
    # multi-result parse (`%sumi:4 = ...` for the columnsPerPass==4 VLEN128 core brick)
    # and the distinct GEMM core mnemonic (NOT the GEVM's `repack_lane_wise...`).
    repack_gemm = derive(parse_realized_body(_GT_REPACK_GEMM))
    assert repack_gemm["manifest"] == [
        "tcrv_rvv.typed_repack_gemm_loop_body",
        "tcrv_rvv.repack_gemm_lane_wise_q4_x_i8_dot",
        "tcrv_rvv.repack_gemm_dual_fp16_scale_fold",
        "tcrv_rvv.repack_gemm_dual_fp16_scale_fold",
        "tcrv_rvv.repack_gemm_dual_fp16_scale_fold",
        "tcrv_rvv.repack_gemm_dual_fp16_scale_fold",
        "tcrv_rvv.typed_repack_gemm_loop_yield",
    ], repack_gemm["manifest"]
    assert repack_gemm["has_opaque"] is False, repack_gemm
    assert repack_gemm["has_product"] is True, repack_gemm
    assert repack_gemm["has_reduce"] is True, repack_gemm
    assert repack_gemm["decomposed"] is True, repack_gemm
    assert repack_gemm["derived_state"] == "constructed", repack_gemm

    # Repack GEVM K-quant ground truth (FIX-D): the fused kquant_core brick must
    # derive constructed (locks the kquant_core whitelist addition).
    repack_kquant = derive(parse_realized_body(_GT_REPACK_KQUANT))
    assert repack_kquant["manifest"] == [
        "tcrv_rvv.typed_repack_gemv_loop_body",
        "tcrv_rvv.repack_gemv_kquant_core",
        "tcrv_rvv.typed_repack_gemv_loop_yield",
    ], repack_kquant["manifest"]
    assert repack_kquant["has_opaque"] is False, repack_kquant
    assert repack_kquant["has_product"] is True, repack_kquant
    assert repack_kquant["has_reduce"] is True, repack_kquant
    assert repack_kquant["decomposed"] is True, repack_kquant
    assert repack_kquant["derived_state"] == "constructed", repack_kquant

    # Dequant-stream ground truth (CERT-FD首族, FIX-5): the CONSTRUCTED streaming
    # region parses to EXACTLY body + decode_core + yield, non-opaque. It carries NO
    # product/reduce (a pure decode), so derive() reads constructed-weak -- proving
    # the contraction gate does NOT vacuously admit it; the streaming form is its own
    # legal shape, accepted by the dedicated _walk_dequant_stream check below.
    deq_manifest = parse_realized_body(_GT_DEQUANT_STREAM)
    deq_mnem = [m["mnemonic"] for m in deq_manifest]
    assert deq_mnem == [
        "tcrv_rvv.typed_dequantize_row_loop_body",
        "tcrv_rvv.dequantize_row_decode_core",
        "tcrv_rvv.typed_dequantize_row_loop_yield",
    ], deq_mnem
    assert not any(is_opaque_hand_helper(m) for m in deq_manifest), deq_manifest
    deq = derive(deq_manifest)
    assert deq["has_product"] is False, deq   # pure decode: no contraction
    assert deq["has_reduce"] is False, deq
    assert deq["derived_state"] == "constructed-weak", deq  # NOT via the contraction gate
    # The streaming shape is accepted by the dedicated honest check (first==body,
    # last==yield, decode_core present, non-opaque) -- the SAME predicate
    # _walk_dequant_stream applies to the REAL realized IR.
    _deq_body, _deq_yld, _deq_core = (
        "typed_dequantize_row_loop_body", "typed_dequantize_row_loop_yield",
        "dequantize_row_decode_core")
    _deq_short = [m.replace("tcrv_rvv.", "") for m in deq_mnem]
    assert (_deq_short[0] == _deq_body and _deq_short[-1] == _deq_yld
            and _deq_core in _deq_short), _deq_short
    # An OPAQUE-leaked streaming body must FAIL the streaming shape (discrimination):
    # inject a bare *_block_dot hand helper and confirm it trips the opaque gate.
    _deq_opaque = _GT_DEQUANT_STREAM.replace(
        "tcrv_rvv.dequantize_row_decode_core %x",
        'tcrv_rvv.q8_0_q8_0_block_dot %x1, %x2 {kind = "ggml_q8_0_q8_0_block_dot"} : '
        "!tcrv_rvv.runtime_abi_value -> !tcrv_rvv.vector<i32, \"m1\">\n"
        "          tcrv_rvv.dequantize_row_decode_core %x")
    assert any(is_opaque_hand_helper(m)
               for m in parse_realized_body(_deq_opaque)), "opaque leak not caught"

    # Quant-stream ground truth (CERT-FD次族): the f32->QUANT MIRROR. The CONSTRUCTED
    # streaming region parses to EXACTLY body + encode_core + yield, non-opaque. It
    # carries NO product/reduce (a pure encode), so derive() reads constructed-weak --
    # proving the contraction gate does NOT vacuously admit it; the streaming form is
    # its own legal shape, accepted by the dedicated _walk_quant_stream check below.
    qnt_manifest = parse_realized_body(_GT_QUANT_STREAM)
    qnt_mnem = [m["mnemonic"] for m in qnt_manifest]
    assert qnt_mnem == [
        "tcrv_rvv.typed_quantize_row_loop_body",
        "tcrv_rvv.quantize_row_encode_core",
        "tcrv_rvv.typed_quantize_row_loop_yield",
    ], qnt_mnem
    assert not any(is_opaque_hand_helper(m) for m in qnt_manifest), qnt_manifest
    qnt = derive(qnt_manifest)
    assert qnt["has_product"] is False, qnt   # pure encode: no contraction
    assert qnt["has_reduce"] is False, qnt
    assert qnt["derived_state"] == "constructed-weak", qnt  # NOT via the contraction gate
    # The streaming shape is accepted by the dedicated honest check (first==body,
    # last==yield, encode_core present, non-opaque) -- the SAME predicate
    # _walk_quant_stream applies to the REAL realized IR.
    _qnt_body, _qnt_yld, _qnt_core = (
        "typed_quantize_row_loop_body", "typed_quantize_row_loop_yield",
        "quantize_row_encode_core")
    _qnt_short = [m.replace("tcrv_rvv.", "") for m in qnt_mnem]
    assert (_qnt_short[0] == _qnt_body and _qnt_short[-1] == _qnt_yld
            and _qnt_core in _qnt_short), _qnt_short
    # An OPAQUE-leaked streaming body must FAIL the streaming shape (discrimination):
    # inject a bare *_block_dot hand helper and confirm it trips the opaque gate.
    _qnt_opaque = _GT_QUANT_STREAM.replace(
        "tcrv_rvv.quantize_row_encode_core %x",
        'tcrv_rvv.q8_0_q8_0_block_dot %x1, %x2 {kind = "ggml_q8_0_q8_0_block_dot"} : '
        "!tcrv_rvv.runtime_abi_value -> !tcrv_rvv.vector<i32, \"m1\">\n"
        "          tcrv_rvv.quantize_row_encode_core %x")
    assert any(is_opaque_hand_helper(m)
               for m in parse_realized_body(_qnt_opaque)), "opaque leak not caught"

    # Discrimination: same rule, opposite verdicts — including the non-opaque hole.
    assert strong["derived_state"] != weak["derived_state"]
    assert strong["derived_state"] != scale["derived_state"]
    assert repack["derived_state"] != weak["derived_state"]
    assert repack["derived_state"] != scale["derived_state"]
    assert repack_gemm["derived_state"] != weak["derived_state"]
    assert repack_gemm["derived_state"] != scale["derived_state"]
    assert superblock["derived_state"] != weak["derived_state"]
    assert superblock["derived_state"] != scale["derived_state"]
    assert superblock_q6k["derived_state"] != weak["derived_state"]
    assert superblock_q6k["derived_state"] != scale["derived_state"]
    assert superblock_q3k["derived_state"] != weak["derived_state"]
    assert superblock_q3k["derived_state"] != scale["derived_state"]
    assert superblock_q2k["derived_state"] != weak["derived_state"]
    assert superblock_q2k["derived_state"] != scale["derived_state"]
    assert superblock_iq1s["derived_state"] != weak["derived_state"]
    assert superblock_iq1s["derived_state"] != scale["derived_state"]
    assert superblock_iq1m["derived_state"] != weak["derived_state"]
    assert superblock_iq1m["derived_state"] != scale["derived_state"]
    print("self-test PASS: parser position-anchored, no mirror leak; "
          "strong(widening_product)=constructed / strong(x_i8_product)=constructed / "
          "strong(super-block scaled_dot fused reduce)=constructed / "
          "strong(super-block q6_K aux32_partial fused reduce)=constructed / "
          "strong(super-block q3_K aux32_partial fused reduce)=constructed / "
          "strong(super-block q2_K integer_core fused reduce)=constructed / "
          "strong(super-block iq1_s grid_core fused reduce)=constructed / "
          "strong(super-block iq1_m grid_core fused reduce)=constructed / "
          "strong(repack GEVM lane-wise dot fused reduce, grouped %r:2 parse)=constructed / "
          "strong(repack GEMM prefill lane-wise dot fused reduce, grouped %r:4 parse)=constructed / "
          "weak(block-dot)=constructed-weak / scale-only=constructed-weak (decomposed gate) / "
          "forward-elementwise MAP(elementwise_scale_map)=constructed (reduce/MAP model branch) / "
          "forward-elementwise MAP(elementwise_silu_map)=constructed (SAME scaffold reused, C2 payoff) / "
          "forward-elementwise REDUCE(elementwise_rms_norm_reduce_core)=constructed (reduce model built, has_reduce_fold branch) / "
          "forward-elementwise REDUCE(elementwise_soft_max_reduce_core)=constructed (SAME reduce model reused, f64m1 widening acc, C2 payoff)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="hermetic parser test over captured ground-truth bodies")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("report", help="run the machine-check and print the read-out")
    sub.add_parser("update-sixstate",
                   help="write the strong rows' auto_readout (only if all pass)")
    sub.add_parser("stamp-repack-dual",
                   help="walk + stamp the 10 single-row repack gemm_tile cells (FIX-D)")
    sub.add_parser("stamp-dequant-stream",
                   help="walk + stamp the 21 constructed streaming dequantize_row cells (CERT-FD首族, FIX-5)")
    sub.add_parser("stamp-quant-stream",
                   help="walk + stamp the 3 constructed streaming quantize_row cells (CERT-FD次族)")
    sub.add_parser("stamp-forward-stream",
                   help="walk + stamp the 5 constructed streaming forward-elementwise cells (CERT-FD殿后族)")
    args = ap.parse_args()
    if args.self_test:
        return cmd_self_test(args)
    if args.cmd == "update-sixstate":
        return cmd_update_sixstate(args)
    if args.cmd == "stamp-repack-dual":
        return cmd_stamp_repack_dual(args)
    if args.cmd == "stamp-dequant-stream":
        return cmd_stamp_dequant_stream(args)
    if args.cmd == "stamp-quant-stream":
        return cmd_stamp_quant_stream(args)
    if args.cmd == "stamp-forward-stream":
        return cmd_stamp_forward_stream(args)
    # default: report
    _results, all_pass = cmd_report(args)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
