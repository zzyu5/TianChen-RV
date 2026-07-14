# [KQUANT-L1 / RE-ROLL 曳光弹 #2] q4_K repack GEMM — does the RE-ROLL convert the opening?

**Board:** rvv / VLEN128, core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (the tracer bullet):** 曳光弹 #1 (the byte-exact *reorder*, cell `l1-pipeline-q4k-repack-gemm`)
did **not** convert the q4_K repack GEMM's full-unroll register-pressure "opening" into throughput
(+0.65%, noise floor; vsetvli/spill unchanged). This probe asks the second, harder lever: does the
**non-byte-exact structural RE-ROLL** — turning the fully-unrolled 32-elem sub-block's k-chunk[2] +
ii-nibble[16] dims into runtime `emitc.for` loops, so clang can hoist the loop-invariant vsetvli and
reuse accumulator registers — convert it?
**Answer: NO — it REGRESSES.** vsetvli RISES, spill RISES, throughput drops ~11%, and vs-opponent
parity gets WORSE (0.962x -> 0.860x, still < 1.0). The diagnosis model is falsified.

## What was compared (real binaries, not extrapolation)
- **PRE**  = full-unroll golden emitter lowering (`/tmp/q4k_reroll/pre_q4K.c`, 14113 lines) —
  **md5 b0b5beac == the cached golden baseline** `gemm_q4_K_q8_K.kernel.c` (oracle-GREEN @039133ea,
  the same baseline the `kquant-l1-q4k-q5k-repack-prefill` precedent measured).
- **POST** = k/ii RE-ROLLED lowering (`/tmp/q4k_reroll/post_q4K.c`, 2525 lines, 11 runtime for-loops):
  the two 2x16 i16 k-chunks and the 16 ii nibbles became runtime `emitc.for`; `sLo/sHi` became
  loop-carried locals; per-accumulator op order over (k, ii) preserved => **byte-exact identical**.
- Both compiled clang-17 -O2 (SAME flags as `kquant_gemm_paired.sh`) into **separate** board binaries,
  each linked against the board's own `libggml-cpu.so` (opponent = real dispatched block-dot).
- Harness: `tools/e2e-harness/board/kquant_gemm_reroll_ab.sh` (+ the precedent's paired driver).
  Cold N rounds; each round runs pre then post as fresh processes; K=2048, nc=512, iters=20.

## 0. Byte-exact identity gate (correctness of the timed binary) -> GREEN
PRE vs POST raw fp32 output `cmp`, seed 0xC0FFEE, K=2048 nr=64 nc=512:
- **int** regime (d_w=dmin_w=d_a=1.0 -> output IS the exact integer isum-summs core): **IDENTICAL, 0 mismatch.**
- **norm** regime (small finite fp16/fp32 scales, real fold path): **IDENTICAL, 0 mismatch.**
The RE-ROLL preserved op order; the timed POST binary computes the golden result.

## 1. objdump vsetvli + spill PRE -> POST (the "confirm it drops" gate) -> IT RISES
| form | vsetvli | spill(vs*r.v) | reload(vl*r.v) | vwmacc | textB |
|---|---:|---:|---:|---:|---:|
| PRE  -O2 (measured binary) | **53** | 84 | 88 | 2240 | 23486 |
| POST -O2 (measured binary) | **76** | 118 | 127 | 320 | 10220 |
| PRE  -O3 | 52 | 84 | 87 | 2240 | 23540 |
| POST -O3 | 75 | 143 | 158 | 320 | 11394 |

vsetvli **+43%** (53->76), spill **+40%** (84->118), reload **+44%** at the measured -O2 form (worse at
-O3: spill +70%, reload +82%). Text HALVED (23486->10220, static vwmacc 2240->320, 7x fewer) — but the
register-pressure metric moved the WRONG way. Reproduces the build-report diagnosis exactly.

**Why the model was wrong:** (1) vsetvli is NOT loop-invariant — the sub-block body genuinely
alternates SEW (e8/mf2 nibble decode <-> e16/m1 i16 accumulate), so clang MUST re-issue vtype; rolling
multiplies the toggles instead of hoisting them. (2) `emitc.for` has no iter-args, so the 16
loop-carried [col][half] accumulators are materialized as C locals (memory) and spilled/reloaded every
iteration, whereas the fully-unrolled PRE keeps those chains in SSA registers as straight-line code.

## 2. A/B (ours-pre-unroll vs ours-post-reroll) — RE-ROLL isolated speedup
Paired per-round `ours_post / ours_pre` (cold, fresh procs):
- **nr=64, N=12:** ours_pre 4.2398 -> ours_post 3.7878 GMAC/s => **0.893 (a consistent ~-10.7% REGRESSION)**.
- **nr=16, N=10:** ours_pre 4.2460 -> ours_post 3.7853 GMAC/s => **0.892 (~-10.8%)** — SHAPE-ROBUST.

Both sides are very tight (ours cv ~0.1-0.2%), so the ~-11% is ~50x our own run-to-run noise: a REAL
regression, NOT a wash. (Contrast 曳光弹 #1's +0.65%, which sat AT the noise floor.)

## 3. vs-opponent (post vs opponent block-dot) — did parity move? did it win?
Clean anchor = **nr=64** (opponent noise cv ~0.7%):
- **PRE parity  = 0.962x**  (reproduces the precedent's 0.94-0.97x band -> harness validated)
- **POST parity = 0.860x**

The RE-ROLL moved parity the WRONG way: **0.962x -> 0.860x**, i.e. ~10.6% FURTHER from parity — a
DEEPER loss, **not a win**. Opponent GMAC/s is stable across the pre/post binaries (24 samples,
4.35-4.49, cv ~0.7%), so the entire parity delta is our side slowing down, not opponent variance.
At nr=16 the opponent block-dot is noise-dominated (cv ~3.7%) so it is not a usable parity anchor.

## [NG-4] discipline
This is an **L1 datapoint, NOT a beat.** The eight [PERF-1] gates are not walked; single core;
opponent = single-thread block-dot loop (kernel-axis proxy, not threaded mul_mat); correctness of OUR
kernel is the construction oracle (039133ea + the byte-exact identity gate above). There is no win
over the opponent (both parities < 1.0, and POST is WORSE than PRE). No beat is claimed.

## Bottom line (structural opening -> throughput?)
**No — and worse, it regresses.** Both cheap levers on the q4_K repack GEMM's full-unroll
register-pressure opening are now falsified on silicon: byte-exact reorder (#1) = NULL (+0.65%,
noise floor); naive structural RE-ROLL (#2) = **-11% throughput, vsetvli +43%, spill +40%, parity
0.962x -> 0.860x**. The opening is genuinely STRUCTURAL in the [col][half]=16-accumulator fan-out;
rolling k/ii does not shrink that fan-out (only rolling the *column* dim would, but that forfeits the
ONCE-per-16-weight decode amortization that is the GEMM's entire advantage over the GEVM). The opening
is a real tension, not a free win — closing it needs a genuinely different schedule (iter-arg-carried
vector accumulators or a register-tiled column blocking), out of scope for a naive re-roll.
