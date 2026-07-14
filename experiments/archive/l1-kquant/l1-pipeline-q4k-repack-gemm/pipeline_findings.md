# [KQUANT-L1 / pipeline] q4_K repack GEMM — does the structural opening convert to throughput?

**Board:** rvv / VLEN128, core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (the tracer bullet):** the L1-maturity probe asked whether the byte-exact *schedule
rearrangement* ("pipelining" = decode-hoist regrouping of the 16-ii nibble decode) converts the
q4_K repack GEMM's structural register-pressure opening into throughput on VLEN128.
**Answer: NO.** vsetvli does not drop, spill does not drop, and the board A/B is at the noise floor.

## What was compared (real binaries, not extrapolation)
- **PRE**  = golden emitter lowering (`/tmp/q4k_emitc_golden.mlir`) — **bit-identical** to the cached
  baseline `gemm_q4_K_q8_K.kernel.c` that the `kquant-l1-q4k-q5k-repack-prefill` precedent measured.
- **POST** = the decode-hoist regrouped lowering (`/tmp/q4k_emitc_after.mlir`), op-multiset identical
  (vwmacc=2240 both), differing only in emit order + code layout.
- Both compiled clang-17 -O2 (same flags as `kquant_gemm_paired.sh`) into **separate** board binaries,
  each linked against the board's own `libggml-cpu.so` (opponent = real dispatched block-dot).
- Harness: `tools/e2e-harness/board/kquant_gemm_pipeline_ab.sh` (+ the precedent's paired driver).
  Cold N rounds; each round runs pre then post as fresh processes; K=2048, nc=512, iters=20.

## 1. objdump vsetvli PRE -> POST (the "confirm it drops" gate)  ->  IT DOES NOT DROP
| form | vsetvli | spill (vs1r) | reload (vl1r) | text B |
|---|---:|---:|---:|---:|
| board clang-17 -O2 (measured binary) | **52 -> 52** | (rolled) | (rolled) | — |
| clang-20 -O3 (build-report diagnosis) | **363 -> 363** | 591 -> 591 | 1597 -> 1601 | 99134 -> 96334 |

Reproduces the build report exactly. clang-17 -O2 keeps the block loops **rolled** (52 vsetvli), so the
board binary never even enters the full-unroll register-pressure regime the -O3 diagnosis describes.
Either way the reorder changes **no** vsetvli and **no** spill — peak liveness (~80 vregs > 32) is
structural and a byte-exact permutation cannot move it.

## 2. A/B  (ours-pre vs ours-post) — pipelining isolated speedup
Paired per-round `ours_post / ours_pre`:
- **nr=64, N=12:** median **+0.67%** (11/12 rounds +0.55..+0.78%; one cold-start outlier -0.4%).
- **nr=16, N=10:** median **+0.63%** (10/10 rounds positive).

Consistent but **negligible ~+0.65%**, and it tracks the **-2.8% smaller code layout** (post text
96334 vs 99134), NOT any vsetvli/spill reduction. It sits at/under the board noise floor.

## 3. vs-opponent (post vs opponent block-dot) — did parity move? did it win?
Clean anchor = **nr=64** (opponent noise cv 0.71%, range 2.98%):
- **PRE parity  = 0.962x**  (reproduces the precedent's 0.94-0.97x band -> harness validated)
- **POST parity = 0.970x**

Parity nudged from ~0.96x to ~0.97x — the same ~+0.7% the ours-side A/B shows — but **remains < 1.0
(parity / slight loss vs the opponent's hand-tuned _vl128 block-dot). NOT a win.** At nr=16 the
opponent block-dot is noise-dominated (cv 3.75%, 4.26-4.82 GMAC/s swings) so it is not a usable
parity anchor; the ours-side A/B still reproduces there.

## T-N noise floor
Opponent (identical linked code in both binaries), 24 samples @ nr=64: cv 0.71%, range 2.98%. Our
kernel is far tighter (pre cv 0.05%, post cv 0.28%). The +0.67% A/B is comparable to the opponent's
own run-to-run cv — i.e. at the floor of significance.

## [NG-4] discipline
This is an **L1 pipeline datapoint, NOT a beat.** The eight [PERF-1] gates are not walked; single
core; opponent = single-thread block-dot loop (kernel-axis proxy, not threaded mul_mat); correctness
of OUR kernel is the construction oracle (039133ea bounded-norm), not re-verified here. Nothing here
is a win over the opponent (both parities < 1.0). No beat is claimed.

## Bottom line (structural opening -> throughput?)
**No.** The byte-exact "pipelining" reorder leaves vsetvli (52/363) and spill (591) untouched and moves
board throughput by ~+0.65% (noise-floor). The register-pressure opening is only reachable by a
**non-byte-exact RE-ROLL** (turning the fully-unrolled superblock into runtime C for-loops so clang
hoists the loop-invariant vsetvli and reuses accumulator registers across iterations) — out of scope
for this byte-exact tracer, logged as the real lever for a follow-up.
