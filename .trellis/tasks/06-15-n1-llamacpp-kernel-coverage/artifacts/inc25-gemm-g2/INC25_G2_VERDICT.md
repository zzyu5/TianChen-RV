# INC-25 GEMM G2 — Q4_0 x Q8_0 full GEMM (NR rows x nc cols) — VERDICT

**Date**: 2026-06-16 · **Board**: `ssh rvv` (VLEN=128, rv64gcv, clang 18.1.3, 64-core shared)

## What G2 is

A STRUCTURED, byte-exact, compiler-emitted **full** Q4_0 x Q8_0 GEMM: NR weight
rows x nc activation columns -> the NR x nc fp32 output matrix s[row][col] =
dot(weight_row, activation_col). It wraps G1's weight-decode-reuse TILE (decode
each q4_0 weight block ONCE, reuse the decoded nibble lanes across M columns) in
an outer weight-ROW loop x an inner M-wide column-strip loop. Each output is
byte-exact vs the per-(row,col) `ggml_vec_dot_q4_0_q8_0` ggml actually runs at
VLEN=128 (the repack GEMM is disabled at `case 128`, so prefill falls back to
per-(row,col) vec_dot, redundantly re-decoding the weight every column — exactly
what this amortizes M-fold).

## The op + lowering (every value an emitc node; raw()=0)

- New op `tcrv_rvv.q4_0_q8_0_gemm` (RVVOps.td) — the full ggml-gemm-like ABI:
  weight base (vx), activation-columns base (vy), per-column byte stride (by),
  output (s), n, **nr, nc, bx (weight-row byte stride), bs (output-row float
  stride)**, vl. `activation_cols` (M) is a bounded compile-time attr [1,16].
  Fail-closed verifier (RVVDialectWideningOps.cpp, I7): pins kind / scale model /
  the ggml block-format facts (qk=32, strides 18/34, offsets +2/+16) / M bounds /
  the 6 operand C types & 4 index operands / the 10-operand+1-result arity.
- Lowering (RVVToEmitC.cpp `emitQ4_0Q8_0Gemm`): outer `emitc.for` weight-ROW loop
  (`xr = vx + ir*bx`, `sr = s + ir*bs`); the nc columns processed in M-wide
  strips split into **(a) a FULL-strip loop** over `ncFull = (nc/M)*M` with a
  COMPILE-TIME-CONSTANT M inner column loop (the C compiler fully unrolls it,
  recovering G1's tile shape), each strip running G1's body (the HOISTED
  `vsetvl_e8m1(16)` weight load + offset-binary decode into v0/v1, the inner
  M-column product/reduce against the hoisted lanes, the M INDEPENDENT
  ascending-block-order fp32 accumulators, the M-output store `sr[cb+j]`), and
  **(b) ONE runtime tail strip** (the final `nc % M` columns) guarded by an
  `emitc.if (ncFull < nc)`. The decode chain reuses INC-1's
  `emitOffsetBinaryDecodeValue` + `...ProductFromDecodedValue` — the SAME nodes
  the per-row block dot and the GEMM tile emit.
- **Additive (747 insertions, 0 deletions across the 3 files).** G1's gemm_tile,
  the block dot, q8_0, q4_1, the shared decode/product helpers are BYTE-UNTOUCHED;
  all sibling conversion + dialect tests pass.

## Byte-exact result (ssh rvv)

**The oracle is the per-(row,col) `ggml_vec_dot_q4_0_q8_0` (the REAL ggml RVV
kernel — what actually runs at VLEN=128). Every s[row][col] == that oracle
bit-for-bit under BOTH -ffp-contract=off AND =fast.**

- **60 shape cases (M=4 and M=6 kernels x 6 shapes x 5 contraction n), 0
  mismatches** under `-ffp-contract=off` AND `=fast`. Shapes include **tail nc**
  (nc=7/13/5/1, NOT multiples of M — exercises the runtime tail-strip clamp),
  rectangular (nr!=nc), and a single-column nc=1. n ∈ {32,64,256,1024,4096}.
- **2 negative controls discriminate**: perturb ONE weight-row scale's low bit ->
  ONLY that row's nc outputs diverge, every other row stays bit-exact
  (perturbed-row-diffs=6, other-row-mismatches=0). Proves the check is not
  trivially passing.
- raw()=0 (no actual raw() calls); the lowering is fully structured emitc (no
  emitc.verbatim C-control, no string blob).

## Perf (ssh rvv, taskset -c 3, min-of-reps ns/output, K=4096; 5 clean runs)

Full GEMM vs NR*nc per-row vec_dot (ggml's used VLEN=128 prefill path):

| kernel | shape | full-GEMM ns/out | NR*nc vec_dot ns/out | speedup |
|---|---|---|---|---|
| **M4** | nr=8 nc=4  | ~5555 | ~5760 | **1.036x** (stable across 5 runs) |
| **M4** | nr=8 nc=8  | ~5567 | ~5770 | **1.036x** |
| M6 | nr=8 nc=6  | ~6720 | ~5765 | 0.857x (regression) |
| M6 | nr=8 nc=12 | ~6740 | ~5772 | 0.857x |

**Honest read (modest, M-shape sensitive, VLEN=128-capped):**
- **M4 wins ~1.04x, reproducibly** (5 runs, ±0.001x). This is the real, honest
  full-GEMM number. It is BELOW G1's reported tile 1.27–1.33x: those G1 numbers
  were contention-flattered (their GEMM baseline measured slower on the shared
  64-core board); the within-run adjacent pair here (GEMM 5555 vs vec_dot 5760)
  is the fair measure. **Do not cite 1.13/1.27x for G2.** It is also below the
  research probe's 1.135x (gemv-perf-scoping.md §2b) — the probe was a tight
  single-row tile; the emitted full GEMM carries the extra row-loop + strip-loop +
  full/tail-split loop nest, which dilutes the per-output amortization.
- **M6 REGRESSES ~0.857x — a repeatable register-pressure effect, NOT noise, and
  it is in TENSION with the research (worth stating honestly).** The M6 GEMM
  absolute time (~6720 ns) is repeatably ~20% WORSE than M4 (~5555 ns), inverted
  from the decode-amortization mechanism (M6 amortizes the decode over MORE
  columns, so per-output it should be cheaper). objdump
  (`m6_register_ceiling_objdump.txt`): the 6-wide full unroll emits MORE
  whole-register vector stores (10 vs M4's 7; touching v8/v9/v10/v12/v14 vs
  v8/v9/v10) — consistent with register pressure from the wider unroll.
  - **The tension**: the research (§2b/§4) found M is a CACHE-capacity resource
    and states "vector-register use is FLAT in M" and "a pure vreg-budget prune
    would MISPREDICT M." Our emitted kernel shows the OPPOSITE — vreg pressure
    GROWS with M. The resolution: the research probe did NOT unroll the M loop
    (its 2 decoded-weight vregs were the only persistent ones), so its vreg use
    was flat. Our emitter's CONSTANT-M unroll (the very thing that delivers the
    M4 win) materializes M parallel vwredsum/vmv_x_s/fp32 reduction chains, so
    OUR kernel's vreg pressure scales with M. **This means G3's prune needs BOTH
    a cache term (the research's L1/column-block axis) AND a vreg/unroll term
    (this finding) — a second resource axis the research probe did not surface.**
    Honest: the two findings do not contradict; they apply to different code
    shapes (un-unrolled probe vs unrolled emission), and the unrolled emission is
    the one that wins.
- **K=11008 (FFN down-proj) is parity** (M4 ~1.003x), as the research §2b
  predicts (the win fades at the large FFN K). G2 proves the MECHANISM at the
  dominant K=4096; an everywhere-win is not claimed.

Why the split-strip matters (the fix): the first G2 cut used a RUNTIME-bounded
inner column loop (`for j < active`), which serialized the per-column
vwredsum/vmv_x_s/fp32 chains -> 0.75x regression even at M4. Splitting into
constant-M full strips (unrollable) + a runtime tail recovered M4 to the 1.04x
win, byte-exactness preserved.

## Deployment

The kernels under test are the UNMODIFIED, compiler-emitted
`tcrv_emitted_gemm_m{4,6}.cpp` (`tcrv-opt … --tcrv-rvv-lower-to-emitc |
mlir-translate-20 --mlir-to-cpp` — the same sanctioned recipe INC-10 used; every
line tagged source_op=tcrv_rvv.q4_0_q8_0_gemm). ABI:
`void <name>(size_t n, float *s, const uint8_t *vx, const uint8_t *vy,
size_t by, size_t nr, size_t nc, size_t bx, size_t bs)` — a drop-in for the
per-(row,col) mul_mat path (vx = NR q4_0 weight rows at stride bx; vy = nc q8_0
columns at stride by; s = NR x nc, row stride bs floats).

**Two deployment caveats (fine for THIS artifact; a future G3/integrator must
know):**
1. **The real `--tcrv-rvv-emitc-to-cpp` bundle exporter was NOT exercised for
   G2.** That path needs a `tcrv.exec.dispatch` selected-path surface (it does
   the lowering + cpp emission internally); the hand-authored G2 typed body only
   carries the `with_vl` `status="selected-lowering-boundary"`, not a dispatch.
   The validated C is from the `--tcrv-rvv-lower-to-emitc | mlir-translate-20`
   recipe, which is byte-identical to what the exporter would render (the lowering
   is the same `emitQ4_0Q8_0Gemm`). A proper deployable `.o/.h` bundle through the
   full dispatch->exporter path is follow-up (as it was for INC-2's bundle).
2. **The `nr`/`nc` runtime ABI operands are bound to placeholder roles**
   (`source-byte-stride`/`destination-byte-stride`) because the
   `RuntimeABIParameterRole` enum has no "row-count"/"col-count" role. They are
   semantically counts, not strides. This is INVISIBLE to the validated path
   because `emitQ4_0Q8_0Gemm` reads operands BY POSITION (not by role), so the
   emitted C is correct; but the dispatch->bundle exporter interprets roles, so a
   future dispatch-driven deployment must either add the count roles to the enum
   or re-map. `bx`/`bs` (`lhs-input-stride`/`output-stride`) are role-correct.

## Tests / build

- New lit tests: `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-gemm.mlir` (the
  structured lowering: row loop, ncFull span, constant-bound full-strip inner
  loop, hoisted decode, tail if) + `test/Dialect/RVV/q4-0-q8-0-gemm-dataflow.mlir`
  (verifier accept + 4 fail-closed rejects). Both PASS.
- Full clean rebuild green; **lit 641 tests, 638 pass, exactly the 3 documented
  environmental reds** (the `rvv-generated-bundle-abi-e2e` widening-dot-reduce
  self-tests — harness-env, unrelated to this op). G1 + all siblings
  byte-identical (additive; 0 deletions).

## Left for G3 (do NOT build here)

The autotuner TUNES M — enumerate -> prune -> **measure** -> select ->
attr-stamp `activation_cols`. G2 makes M a bounded compile-time attr [1,16] so G3
can sweep it. **G2 surfaces TWO orthogonal resource axes G3's prune must model,
one from the research and one new from this work:**
- the **cache/L1 column-block** axis (research §2b: M_opt ∝ 1/K, the win shrinks
  toward the large FFN K; analytically unpredictable -> must be MEASURED), and
- the **vreg/unroll** axis (THIS work: because the emitter unrolls the constant-M
  inner loop, each extra column adds a parallel reduction chain -> vreg pressure
  grows with M -> M6 already regresses on register pressure at K=4096). The
  research probe did not unroll and so reported vreg use "flat in M"; the
  emission's unroll changes that, so G3 must bound M by the register budget too.
Net: the prune is two-axis (cache AND vreg), and because both are noisy /
shape-dependent the final M must be MEASUREMENT-selected, not analytically
predicted — exactly the measurement-backed-autotuner thesis. G2 builds none of
this (M is a fixed attr); it only makes the knob and the evidence available.
