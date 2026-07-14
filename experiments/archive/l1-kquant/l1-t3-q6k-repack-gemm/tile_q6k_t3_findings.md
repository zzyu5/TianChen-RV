# [KQUANT-L1 / T3-tile] q6_K repack GEMM — does the q4_K S6 stack-panel lever transfer to q6_K?

**Board:** rvv / VLEN128 (openEuler), core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (the tiling-lever transfer):** the q4_K S6 register-cliff step (cell `l1-tile-s6-q4k-repack-gemm`)
converted the full-unroll opening into throughput by staging the idle-across-the-hot-dot i32 MIN
accumulator + the 6-bit decode strips to STACK PANELS, dropping spill 84→23→3, peak vreg v31→v30, and
gaining +27.9% over S1 — **byte-exactly**. T3 asks whether the SAME lever, adapted no-min for q6_K
(decoded signed-scale int16 stack-panel via `vse16/vle16` + on-demand `d` f16 widen deferred to the
end-of-block fold; **no min-fold panel — q6_K has no min**), reaches the ≤32-vreg cliff on q6_K's
much heavier fully-unrolled dual-plane (6-bit `ql`+`qh`, 16 sub-blocks) body.
**Answer: NO — the lever does NOT transfer. q6_K T3 is a NULL (small regression).** spill did NOT
reach ~0 (**913 → 949**, it rose), peak vreg did NOT reach ≤32 (**v31 → v31**, no cliff), throughput
did NOT rise (ours **0.6724 → 0.6625 GMAC/s, −1.5%**), and vs-opponent parity did NOT improve
(**0.184 → 0.181**, still a ~5.4× LOSS). The tiling is **byte-exact** (correctness holds) but achieves
nothing on the register cliff, because q6_K's ~900-spill pressure lives in the dual-plane weight
materialization, not in the decode-scale strips the panel stages.

## What was compared (real binaries, not extrapolation)
- **UNTILED** = HEAD 924dc31f front-door export of `rvv-to-emitc-repack-gemm-q6-K-q8-K.mlir`
  (`untiled_q6K_gemm.c`, **md5 0f14791e**; source op `tcrv_rvv.repack_gemm_q6_K_q8_K`; == the
  `kquant-l1-q6q2q3-repack` cell's q6_K GEMM = the **board-harvest2 q6_K prefill 0.18× LOSS** baseline).
  Full-unroll direct-emit; the cached cold export (no rebuild).
- **TILED** = SAME fixture (working-tree region form) re-lowered through the working-tree S6 emitter
  delta on `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (`emitRepackKQuantGemmBodyQ6K`;
  `tiled_q6K_gemm.c`, **md5 28cbeebd**), regenerated via **read-only `build/bin/tcrv-opt`** (already
  built at the parent's forced-clean rebuild; ZERO opt/translate errors) + `mlir-translate-20 --mlir-to-cpp`.
- Both export the SAME extern-C symbol `…repack_gemm_q6_K_q8_K…` (the parent kept the func name in the
  region fixture), so a single q6_K-only paired driver (`tools/e2e-harness/board/kquant_gemm_paired_q6k_driver.c`)
  links TWICE (untiled.o / tiled.o) into two board binaries differing ONLY in the q6_K GEMM object → a
  clean tiling-isolated A/B. Each linked against the board's OWN `libggml-cpu.so` (opponent = real
  dispatched `ggml_vec_dot_q6_K_q8_K` block-dot; q6_K's repack trait is NULL@VLEN128 → ggml falls back
  to exactly this per-(row,col) block-dot).
- Harness: `tools/e2e-harness/board/kquant_gemm_tile_q6k_t3_ab.sh`. Cold N rounds; each round runs
  untiled then tiled as fresh processes; K=2048, nc=512, iters=20. **No git stash** — untiled is the
  cached HEAD export; main tree + build/ read-only (no rebuild).

## 0. Byte-exact gates (correctness of the timed TILED binary) → GREEN (both)
- **Silicon verify** (`kquant_repack_verify_q6K`, independent reference decoded from the ORIGINAL
  pre-repack block), linked with UNTILED then TILED objects, seed 0xC0FFEE:
  - UNTILED: `INT_mismatch_total=0`, NORM `worst_ulp=220 worst_norm=4.579e-07` → BYTE-EXACT-INTEGER + BOUNDED-NORM.
  - TILED:   **identical** — `INT_mismatch_total=0`, NORM `worst_ulp=220 worst_norm=4.579e-07`.
- **Identity dump** (full nr·nc=32768 fp32 output, small-finite fp16-scale real fold path, seed 0xC0FFEE):
  `o_untiled.bin == o_tiled.bin` → **BYTE-EXACT, 0 mismatch**.
- Corroborated by the objdump `vwmacc 2304` multiset being UNCHANGED untiled→tiled (§1). The S6
  scale-panel staging preserved the end-of-block fold order; the timed TILED computes bit-for-bit the
  untiled/oracle result. (q6_K construction is oracle-GREEN @924dc31f ⇒ TILED transitively oracle-GREEN.)

## 1. objdump seal UNTILED → TILED (the "spill→~0, peak ≤32" gate) → ★NOT REACHED (NULL)
| form | vsetvli | spill(vs*r.v) | reload(vl*r.v) | maxVreg | vwmacc | textB |
|---|---:|---:|---:|---:|---:|---:|
| UNTILED -O2 | 21 | **913** | 953 | v31 | 2304 | 72378 |
| TILED   -O2 | 21 | **949** | 957 | v31 | 2304 | 73100 |
| UNTILED -O3 | 20 | 927 | 965 | v31 | 2304 | 73200 |
| TILED   -O3 | 20 | 953 | 960 | v31 | 2304 | 73626 |

- **spill 913 → 949 (−O2) / 927 → 953 (−O3)**: the stack-panel did NOT drive spill toward ~0 — it rose
  slightly (+36 / +26). The q6_K fully-unrolled body carries **~900 whole-register spills**, an ORDER OF
  MAGNITUDE more than q4_K golden's 84 / S1's 23.
- **maxVreg v31 → v31** (both −O2/−O3): the peak live-set did NOT drop under 32; **the ≤32-vreg cliff
  was NEVER reached** (q4_K S6 hit v30). No spare register.
- **vwmacc 2304 UNCHANGED across all four** ⇒ byte-exact multiset preserved (confirmed by §0). vsetvli 21
  unchanged (the no-min single-scale decode band is one vtype). Text +722 B (the added panel store/load).
- **Hot-loop classification (TILED −O2):** the `vs*r.v` are NOT confined to the prologue (as they were
  for q4_K S6's benign residual-3) — they begin at 0x92 and continue densely through the block body (949
  total). These are genuine hot-loop allocator live-value spills, **saturated in BOTH untiled and tiled**.

**Why it does NOT work (the transfer boundary):** q4_K's residual pressure at the peak WAS the idle i32
MIN accumulator (`bsums`) + the 6-bit decode strips — EXACTLY what the stack-panel targets — so staging
them collapsed spill 23→3 / v31→v30. q6_K's dominant pressure is the **dual-plane 6-bit weight
reconstruction** (`ql`@1312 + `qh`@288, 16 sub-blocks) plus the f32 accumulator fan, which the
decode-scale panel does not touch. Staging only the scale strips is a drop in the ~900-spill bucket and
adds panel store/load traffic ⇒ spill nudges UP, throughput nudges DOWN. **The pattern does not migrate
q4_K → q6_K without a different lever** (e.g. a block-loop re-roll to shrink the unroll, or a dedicated
`ql`/`qh` materialization panel).

## 2. A/B (tiled vs untiled) — tiling-isolated increment → small consistent REGRESSION
Paired per-round `ours_tiled / ours_untiled` (cold, fresh procs):
- **nr=64, N=12:** untiled 0.6724 → tiled 0.6625 GMAC/s ⇒ **0.985 (−1.5%)**; per-round A/B max 0.992 < 1, min 0.980.
- **nr=16, N=10:** untiled 0.6524 → tiled 0.6374 GMAC/s ⇒ **0.977 (−2.3%)** — **SHAPE-ROBUST** (same sign, similar magnitude).

Both sides tight (ours cv 0.15–0.35%), so the ~1.5–2.3% delta is real and non-overlapping (max A/B
0.992 < 1) — a small, decisive REGRESSION, not noise. **T-N: the "MAC/cycle up" gate FAILS** (direction
is down). The panel store/load traffic is not amortized because the unroll body is not register-relieved.

## 3. vs-opponent (untiled/tiled vs opponent block-dot) — parity did NOT climb
Clean anchor = **nr=64** (opponent block-dot cv 0.575% across all 24 samples):
- **UNTILED parity = 0.184×**  (reproduces the board-harvest2 q6_K prefill 0.18× LOSS → basis/harness validated)
- **TILED parity   = 0.181×**  (slightly WORSE)

The opponent's real dispatched `ggml_vec_dot_q6_K_q8_K` block-dot is **~5.4–5.5× faster** than our q6_K
repack GEMM, both untiled and tiled; opponent GMAC/s is stable across binaries (mean 3.66, cv 0.575%), so
the small parity drop is entirely our side. At nr=16 the opponent is noise-dominated (cv 3.5–5.3%) so it
is not a usable parity anchor (indicative untiled 0.192 / tiled 0.189; the ours-side A/B −2.3% is the
load-bearing shard). **q6_K is a genuine repack LOSS regime at VLEN128, and tiling does not rescue it.**

## Pre-registered judgment → q6_K S6-scheme does NOT hold (NULL)
The board pre-registration was: **spill → ~0 ∧ MAC/cycle significantly up ∧ wall passes T-N ⇒ holds.**
NONE met:
- spill **913 → 949** (rose; hot-loop-saturated), peak vreg **v31** (NOT ≤32) ✗
- MAC/cycle (∝ GMAC/s at fixed 2.6 GHz) **0.6724 → 0.6625 (−1.5%)** — down, not up ✗
- wall: a consistent **−1.5%/−2.3% REGRESSION** (non-overlapping, shape-robust) — fails the "improvement" direction ✗

**q6_K T3 FAILS the tiling gate.** This is the honest boundary of the q4_K S6 lever: it wins where the
peak pressure IS the decode/min strips (q4_K), and is a null-to-slight-regression where the peak pressure
is the dual-plane weight materialization (q6_K). The construction itself (front-door region + no-min
stack-panel) is sound and **byte-exact**; it simply does not buy a register cliff on this format.

## Loadavg / contention honesty
nr64: loadavg pre 2.33 / mid 2.90 / end 3.21. nr16: pre 2.51 / end 3.26. Core 8 (our taskset pin) at
2.6 GHz gov=performance. The A/B is **paired** (untiled then tiled back-to-back each round) so shared
memory/scheduler contention is common-mode and cancels in the ratio; the −1.5% (nr64) and −2.3% (nr16)
reproduced across two shapes and stable loadavg bands, so the regression is not a contention artifact.

## [NG-4] discipline
This is an **L1 kernel datapoint, NOT a beat** (and not an anti-beat claim about the opponent — it is our
own negative result). The eight [PERF-1] gates are not walked; single core; opponent = single-thread
block-dot loop (kernel-axis proxy, not threaded `mul_mat`); correctness of OUR kernel is the construction
oracle (924dc31f) + the byte-exact verify + identity gates above. The 0.18× vs-opponent is a **KERNEL-axis
parity number**, and the tiling delta is a **KERNEL-axis structural datapoint**, NOT a sealed e2e result.

## Bottom line (does the tiling lever transfer to q6_K?)
**No.** The q4_K S6 register-cliff win does not generalize to q6_K: staging the decode-scale strips to a
stack panel leaves q6_K's ~900-spill dual-plane fully-unrolled body untouched (spill 913→949, maxVreg
v31, no ≤32 cliff), and throughput regresses ~1.5–2.3% while vs-opponent stays a ~5.4× LOSS (0.184→0.181).
Byte-exact throughout (verify INT_mismatch=0 both + identity cmp 0 mismatch; vwmacc 2304 unchanged).
Logged as the **q6_K T3 negative datapoint** — it maps the transfer boundary of the tiling pattern
(C3′ pattern-library: this lever is q4_K/q5_K-shaped, not q6_K-shaped) and flags that the parent's
"pending-hardware spill→0/≤32" projection for q6_K is **refuted on silicon**.
