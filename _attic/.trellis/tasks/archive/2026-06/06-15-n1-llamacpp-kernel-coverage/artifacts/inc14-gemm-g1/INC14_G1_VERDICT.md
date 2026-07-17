# INC-14 G1 — Q4_0 x Q8_0 GEMM tile (weight-decode reuse) — VERDICT

**Date**: 2026-06-16 · **Board**: `ssh rvv` (VLEN=128, rv64gcv, clang 18.1.3)

## What G1 is

A STRUCTURED, byte-exact, compiler-emitted Q4_0 GEMM tile: ONE weight-row block
array × M activation columns, decoding each q4_0 weight block ONCE and reusing the
decoded nibble lanes (v0/v1) across all M columns. This is the `w_hoist` shape the
research recommended (research/gemv-perf-scoping.md §2b/§3): it eliminates the
redundant weight re-decode that ggml's per-(row,col) `vec_dot` fallback does at
VLEN=128 (where ggml's repack GEMM is disabled, `case 128: break`).

## The op + lowering (every value an emitc node; raw()=0)

- New op `tcrv_rvv.q4_0_q8_0_gemm_tile` (RVVOps.td) — ABI operands: weight base
  (vx), activation-columns base (vy), per-column byte stride (by), output (s, M
  contiguous), n, vl. `activation_cols` (M) is a bounded compile-time attr [1,16].
  Fail-closed verifier (RVVDialectWideningOps.cpp, I7): pins kind / scale model /
  the ggml block-format facts (qk=32, strides 18/34, offsets +2/+16) / M bounds /
  operand C types.
- Lowering (RVVToEmitC.cpp `emitQ4_0Q8_0GemmTile`): outer `emitc.for` block loop;
  per block the weight fp16 scale + the HOISTED `vsetvl_e8m1(16)` weight load +
  offset-binary decode into v0/v1 (m1 whole-half-block, correct ∀VLEN≥128); an
  inner `emitc.for` over M columns each loading column j's q8 halves, replaying the
  SAME vwmul/vwmacc product against the hoisted v0/v1, a per-column vwredsum →
  sumi_j, and the per-column ascending-block-order fp32 fold into an
  `!emitc.array<Mxf32>`; then the M-output store. The decode is HOISTED above the
  inner loop — the v0/v1 SSA values defined above `for j` and used inside ARE the
  weight-decode reuse.
- The decode chain reuses INC-1's `emitOffsetBinaryDecodeProductValue`, REFACTORED
  into `emitOffsetBinaryDecodeValue` (vxor/vsll/vsra → v0/v1) +
  `emitOffsetBinaryProductFromDecodedValue` (vwmul/vwmacc). The combined function
  composes them back-to-back, so the per-row block-dot + q4_1 + q8_0 + packed-i4
  siblings emit BYTE-IDENTICAL nodes (all 7 sibling translate lit tests pass).

## Byte-exact result (ssh rvv, -ffp-contract=off)

**The oracle the task names is M× `ggml_vec_dot_q4_0_q8_0` (the REAL ggml RVV
kernel — what actually runs at VLEN=128). Ours == that oracle bit-for-bit under
BOTH -ffp-contract=off AND =fast** (`real_eq=0` count = 0 in every case). G1's
acceptance gate is this byte-exactness, NOT a win bar (the win bar is G2/G3).

- **2000 tiles checked (M∈{4,6} × n∈{32,64,256,1024,4096} × 200 reps), 0 failures.**
  Every one of the M outputs is BITWISE equal to an independent
  `ggml_vec_dot_q4_0_q8_0(weight_row, column_j)` AND to ggml `_generic` — i.e. byte-
  exact vs M× vec_dot, the real ggml-at-VLEN128 prefill behavior. The positive test
  itself validates the per-column addressing + low/high q8 pairing (the independent
  oracle reads `vy + j*by` and the +2 / +2+16 halves; a wrong stride or swapped
  pairing would mismatch all M outputs).
- **Negative control discriminates**: a 1-bit flip of column-2's scale changes ONLY
  s[2] (independent accumulators), s[2] still tracks the perturbed oracle, and the
  bitwise comparator REPORTS MISMATCH on divergent inputs (non-vacuous).
- **-ffp-contract=fast**: ZERO `real_eq=0` failures — our kernel stays byte-exact vs
  ggml's REAL RVV kernel in EVERY case. The ~719 reported `=fast` failures are all
  `gen_eq=0` (the scalar `_generic` doubly-sure cross-check contracting its
  `sumf += sumi*dx*dy` differently under `=fast`) — the SAME documented `=fast`
  reference artifact INC-2a recorded. The byte-exact claim vs the actual
  ggml-at-VLEN128 path (the vector kernel) holds under both modes.

## Measured speedup (G1 tile vs M× per-row vec_dot, min-of-reps, taskset -c 3)

| M | K=4096 speedup (range over runs) |
|---|---|
| 4 | 1.22–1.25× |
| 6 | 1.31–1.33× |

**Honest read: a DIRECTION-robust modest win (every run > 1.2×), magnitude
consistent-with-to-slightly-above the research's ~1.10–1.13×.** The absolute ns/out
swings ~2× run-to-run on this shared 64-core board (e.g. M=4 baseline 1634→3229 ns),
so the ratio's two-digit magnitude is NOT trustworthy — only the direction (>1.2×
every run) and the mechanism are. The win is genuinely from hoisting the decode:
our tile's decode and ggml's vec_dot decode are both ~4 vector ops, so the gap is
the M-fold amortization of the weight decode, not a decode-sequence artifact. **G1's
gate is byte-exactness, not a win bar** — the honest, modest, noisy magnitude is
expected here and already satisfies G1; locking the win is G2 (mechanism at a real
K) and G3 (tuner selects M_opt by measurement).

## raw()=0 + structured proof

`grep -c raw(` on both emitted C files = 0. Every `emitc.verbatim` in the lowered
IR is a `// tcrv_emitc...` comment marker (no C control-flow blob). The loop nest,
M-wide accumulator array, decode, products, reduce, and folds are all emitc nodes.

## lit + reds

Full clean rebuild green. `check-tianchenrv`: 620/623 pass. The 3 failing are the
PRE-EXISTING documented reds (`computed-masked-strided-input-widening-dot-reduce-add`
e2e ×2 + self-test) — unrelated to q4_0. The 2 NEW G1 tests are discovered IN-SUITE
(`lit --filter=gemm-tile` reports 623 total, 2 matched, both PASS) and pass:
`test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-gemm-tile.mlir` (structured lowering),
`test/Dialect/RVV/q4-0-q8-0-gemm-tile-dataflow.mlir` (verifier accept + 4 fail-closed).

## Left for G2 / G3

- **G2 (full ABI)**: wrap G1 in the weight-ROW loop + the K block loop; emit ggml's
  gemm ABI `void gemm(int n, float *s, size_t bs, const void *vx, const void *vy,
  int nr, int nc)` with the `*s + bs` output stride (G1 stores s[0..M-1] contiguous);
  gate the WHOLE kernel bitwise on board + a perf assert at K=4096.
- **G3 (autotuner tunes M)**: add `activation_cols` (the M-block) as a bounded
  enumerate→prune→MEASURE→select knob (M_opt is cache-bound/∝1/K and analytically
  unpredictable per the research, so it must be measurement-selected, NOT vreg-pruned).
