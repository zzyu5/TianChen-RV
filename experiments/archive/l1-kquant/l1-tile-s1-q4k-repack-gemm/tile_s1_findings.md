# [KQUANT-L1 / T2-tile S1] q4_K repack GEMM — does the S1 h-strip TILE convert the opening?

**Board:** rvv / VLEN128, core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (the third lever):** the q4_K repack GEMM's full-unroll register-pressure "opening" survived
two cheaper probes — 曳光弹 #1 (byte-exact *reorder*, cell `l1-pipeline-q4k-repack-gemm`) = **NULL**
(+0.65%, vsetvli/spill unchanged); 曳光弹 #2 (naive k/ii *re-roll*, cell `l1-reroll-q4k-repack-gemm`) =
**−11% REGRESSION** (spill ROSE 84→118). The reroll cell named the real lever explicitly: *"only a
register-tiled column blocking would shrink the [col][half]=16-accumulator fan-out."* This probe builds
exactly that: the **S1 STRUCTURAL TILE** — split the single fully-unrolled block loop into **TWO
per-h-strip `emitc.for` block loops** (mr=4 / hs=1), each carrying only its OWN per-strip
sumf/sumi/bsums accumulators, so the scheduling-level peak live-vreg per contraction pass halves.
**Answer: YES — it CONVERTS.** spill DROPS 84→23 (−73%), board throughput **+53%**, and vs-opponent
parity **FLIPS 0.962× → 1.471×** (ours now leads the block-dot). **S1 HOLDS** (pre-registered judgment
met on all three conditions).

> ⚠ **This CONTRADICTS the build-phase static prediction.** The build report (driver-export basis, a
> ~100KB `.o`, objdump counting `vs1r+vl1r=2240`) concluded S1 was a **structural NULL** — *"spill ≈ FLAT
> (−6%), frame unchanged, S1 didn't convert to spill↓; two contraction passes cancel."* That was the WRONG
> basis. On the **campaign-canonical basis** (the SAME test fixture + `vs*r.v`/`vl*r.v` seal method the
> golden PRE and the reroll/pipeline siblings use), spill drops 84→23 and silicon shows +53%. The
> "two-pass cancel" reasoning was wrong because the two h-strip passes are **SEQUENTIAL**, so the PEAK
> live-set HALVES (not sums). The board is the ground truth; the static estimate is corrected.

## What was compared (real binaries, not extrapolation)
- **PRE**  = full-unroll golden emitter lowering of `rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir`
  (`/tmp/q4k_tile_s1/pre_q4K.c`, 14113 lines) — **md5 b0b5beac == the cached golden baseline**
  `gemm_q4_K_q8_K.kernel.c` (oracle-GREEN @039133ea; SAME baseline the reroll/pipeline siblings measured).
- **POST** = the SAME fixture re-lowered through the working-tree **S1-tile** emitter delta on
  `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (`/tmp/q4k_tile_s1/post_q4K.c`, 19043 lines):
  the single block loop `for(v36 < nb)` became **two** loops `for(v28 < nb)` + `for(v5719 < nb)`
  (one per h-strip); each strip carries its own sumf/sumi/bsums; per-accumulator op order preserved
  ⇒ **byte-exact-identical** (compute-op multiset identical: vfmacc=16, vfnmsac=16, **vwmacc=4480 C-intrinsics / 2240 asm**).
- Both compiled clang-17.0.6 -O2 (SAME flags as `kquant_gemm_paired.sh`) into **separate** board
  binaries, each linked against the board's own `libggml-cpu.so` (opponent = real dispatched block-dot).
- Harness: `tools/e2e-harness/board/kquant_gemm_tile_s1_ab.sh` (+ the shared `kquant_gemm_paired_driver.c`).
  Cold N rounds; each round runs pre then post as fresh processes; K=2048, nc=512, iters=20.

## 0. Byte-exact identity gate (correctness of the timed binary) → GREEN
PRE vs POST raw fp32 output `cmp`, seed 0xC0FFEE, K=2048 nc=512:
- **int** regime (d_w=dmin_w=d_a=1.0 → output IS the exact integer isum−summs core): **IDENTICAL, 0 mismatch.**
- **norm** regime (small finite fp16/fp32 scales, real fold path): **IDENTICAL, 0 mismatch.**
- driver output checksum `sink` identical PRE vs POST at both shapes (nr64 49115.7 / nr16 29769.9).
The S1 tile preserved per-accumulator op order; the timed POST binary computes the golden result.

## 1. objdump vsetvli + spill PRE → POST (the "confirm it drops" gate) → IT DROPS
| form | vsetvli | spill(vs*r.v) | reload(vl*r.v) | vwmacc | textB |
|---|---:|---:|---:|---:|---:|
| PRE  -O2 (measured binary) | 53 | **84** | **88** | 2240 | 23486 |
| POST -O2 (measured binary) | 73 | **23** | **34** | 2240 | 25358 |
| PRE  -O3 | 52 | 84 | 87 | 2240 | 23540 |
| POST -O3 | 73 | 23 | 34 | 2240 | 25330 |

spill **−73%** (84→23), reload **−61%** (88→34) at BOTH -O2 and -O3 (deterministic). vsetvli rose
53→73 (+38% — cheap scalar vtype re-config from the per-strip preamble), text +8% (per-strip preamble +
scalar-activation re-read ×2). **vwmacc UNCHANGED (2240)** ⇒ byte-exact-preserving (same MAC multiset,
re-tiled). **★ the PRE -O2 seal reproduces the `l1-reroll` cell's PRE -O2 seal BYTE-FOR-BYTE**
(vsetvli=53 spill=84 reload=88 vwmacc=2240 textB=23486) → the measurement basis is validated as the
campaign-canonical one.

**Why it works (where reroll failed):** the full-unroll PRE keeps ALL 16 [col][half] fp32 accumulators
(+ sumi/bsums) live across the SINGLE block loop → 84 whole-register spills. The S1 tile splits into TWO
SEQUENTIAL per-strip loops, each carrying only ~half the accumulators; because the passes are sequential
the peak live-set HALVES → clang keeps each strip's chain in registers → spill 84→23. Unlike the reroll
(which rolled the k/ii dims INSIDE the fan-out, multiplying vsetvli and spilling the iter-arg-less
loop-carried accumulators), the tile shrinks the *fan-out itself* and does NOT forfeit the vector
vwmacc decode amortization (count preserved) — it only re-reads the scalar activation broadcasts ×2.

## 2. A/B (ours-pre-unroll vs ours-post-tile) — S1 isolated speedup
Paired per-round `ours_post / ours_pre` (cold, fresh procs):
- **nr=64, N=12:** ours_pre 4.2338 → ours_post 6.4799 GMAC/s ⇒ **1.527 (median; +53.1%)** — mean-of-means 1.5305.
- **nr=16, N=10:** ours_pre 4.2488 → ours_post 6.4569 GMAC/s ⇒ **1.519 (+52.0%)** — SHAPE-ROBUST.

Both sides tight (ours cv ~0.08–0.29%; nr64 pre cv 0.96% inflated by one R7 hiccup, median clean), so
the +52–53% is **~55–179× our own run-to-run noise floor** — a large, REAL win, decisively
non-overlapping. **T-N PASS.** (Contrast #1 +0.65% [noise floor] and #2 −11% [regression].)

## 3. vs-opponent (post vs opponent block-dot) — did parity move? did it flip?
Clean anchor = **nr=64** (opponent block-dot cv ~0.5%):
- **PRE parity  = 0.962×**  (reproduces the precedent's 0.94–0.97× band → harness validated)
- **POST parity = 1.471×**

The S1 tile moved parity the RIGHT way and FLIPPED it: **0.962× → 1.471×**, i.e. from slightly UNDER
parity to a **~47% LEAD** over the opponent's real dispatched block-dot. Opponent GMAC/s is stable across
the pre/post binaries (24 samples 4.34–4.45, cv ~0.54%), so the entire parity jump is our side speeding
up, not opponent variance. At nr=16 the opponent block-dot is noise-dominated (cv ~3.2%) so it is not a
usable parity anchor (indicative POST 1.43×; the ours-side +52% A/B there is the load-bearing shard).

## Pre-registered judgment → S1 HOLDS
The board pre-registration was: **spill significantly down ∧ MAC/cycle significantly up ∧ wall passes
T-N ⇒ S1 holds.** All three met:
- spill **84→23 (−73%)** — significantly down ✓
- MAC/cycle (∝ GMAC/s at fixed 2.6 GHz) **4.23→6.48 (+53%)** — significantly up ✓
- wall passes T-N **(+53% is 55–179× the noise floor, non-overlapping IQR)** ✓

**S1 SUCCEEDS.** It is the FIRST of the three q4_K full-unroll register-pressure levers to convert the
opening into throughput: reorder (#1) = NULL, re-roll (#2) = −11%, **S1 tile (#3) = +53%**.

## [NG-4] discipline
This is an **L1 kernel datapoint, NOT a beat.** The eight [PERF-1] gates are not walked; single core;
opponent = single-thread block-dot loop (kernel-axis proxy, not threaded `mul_mat`); correctness of OUR
kernel is the construction oracle (039133ea) + the byte-exact identity gate above. The 1.471× vs-opponent
is a **KERNEL-axis parity flip**, NOT a sealed e2e beat; per `kernel-wins-dont-transplant-to-e2e`, a
compute-side micro win of this shape does not automatically transduce to a memory-bound decode e2e — that
transduction (+ the remaining S2–S6 tile steps toward ≤32 vreg) is out of scope for this S1-only cell.

## Bottom line (structural opening → throughput?)
**Yes — S1 converts it, +53%.** The register-pressure opening was genuinely structural in the
[col][half]=16-accumulator fan-out (as the reroll cell diagnosed); the S1 h-strip tile is the
register-tiled blocking that shrinks that fan-out, halving the sequential peak live-set (spill 84→23)
while preserving the vwmacc decode amortization (byte-exact). This is logged as **S1 alone** (its own
table row); S2–S6 (further tiling toward the ≤32-vreg spill-cliff and an e2e transduction check) are
separate, not merged into this datapoint.
