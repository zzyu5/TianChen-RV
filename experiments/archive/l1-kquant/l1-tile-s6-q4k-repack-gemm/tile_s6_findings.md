# [KQUANT-L1 / T2-tile S6] q4_K repack GEMM — does the S6 min-fold + stack-panel reach the register cliff?

**Board:** rvv / VLEN128, core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (the register-cliff step):** the S1 h-strip tile (#3 lever, cell `l1-tile-s1-q4k-repack-gemm`)
already converted the full-unroll opening into throughput (spill 84→23, +53%, parity 0.962→1.471). S6 is
step 6 of the tile ladder: drive the residual register pressure to the **≤32-vreg cliff** so allocator
live-value spills reach **~0** — *byte-exactly*. The two levers: (1) **inline-min-fold as a stack panel** —
the per-column i32 MIN accumulator (`bsums`, idle across the entire hot main-term dot) is staged to
`int32_t bsums_panel[]`, accumulated in the cold MIN band and reloaded once at the end-of-block fold for
the UNCHANGED `vfnmsac`; (2) **decoded-weight stack-panel** — the 6-bit scale/min i16 decode strips are
staged to `int16_t scale_panel/min_panel[]` (+ d/dmin f16 widen deferred on-demand) so the decode band
never sits live across the ii dot. The load-bearing insight: `sumi`/`bsums` are INTEGER accumulators with
a SINGLE f32 convert+fold at end-of-block, so the i32 accumulation can be reordered/staged (exact +
associative) as long as the final `vfmacc`(main)→`vfnmsac`(min) f32 sequence is kept IDENTICAL — this hits
≤32 **without** the T1-doc's per-sub-block f32 inline-fold (which would NOT be bit-exact).
**Answer: YES — S6 HOLDS.** spill (hot-loop live-value) **23→0**, peak vreg **v31→v30 (≤32)**, byte-exact
bit-identical, and A/B over S1 **+27.9%**, with vs-opponent parity climbing further **1.473→1.884**.

## What was compared (real binaries, not extrapolation)
- **golden** = full-unroll golden lowering of `rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir` (`golden_q4K.c`,
  **md5 b0b5beac** == the cached baseline the S1/reroll/pipeline siblings measured; oracle-GREEN @039133ea).
  Compiled for the objdump 84-spill base anchor ONLY (not timed).
- **PRE (S1)** = SAME fixture lowered through the committed **d5a28efc S1-tile** emitter (`s1_q4K.c`,
  **md5 9a757c02**; == cached S1 baseline; == build-phase `s1_q4K.c`, cross-checked identical).
- **POST (S6)** = SAME fixture re-lowered through the working-tree **S6** emitter delta on
  `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (`emitRepackKQuantGemmBodyQ4K`, +122/−30; `s6_q4K.c`,
  **md5 90d454da**; freshly regenerated via read-only `build/bin/tcrv-opt` and cross-checked **byte-identical**
  to the build-phase `s6_final.c`). ZERO opt/translate errors.
- All three compiled clang-17.0.6 -O2 (SAME flags as `kquant_gemm_tile_s1_ab.sh`) into SEPARATE board
  binaries, S1/S6 each linked against the board's own `libggml-cpu.so` (opponent = real dispatched block-dot).
- Harness: `tools/e2e-harness/board/kquant_gemm_tile_s6_ab.sh` (+ shared `kquant_gemm_paired_driver.c`).
  Cold N rounds; each round runs pre(S1) then post(S6) as fresh processes; K=2048, nc=512, iters=20.
  **No git stash** — the S1 baseline is the cached export; main tree + build/ read-only (no rebuild).

## 0. Byte-exact identity gate (correctness of the timed S6 binary) → GREEN
PRE(S1) vs POST(S6) raw fp32 output `cmp`, seed 0xC0FFEE, K=2048:
- **int** regime (unit scales → output IS the exact integer isum−summs core): **IDENTICAL, 0 mismatch** (nr64 & nr16).
- **norm** regime (small finite fp16/fp32 scales, real fold path): **IDENTICAL, 0 mismatch** (nr64 & nr16).
- driver `sink` identical PRE(S1) vs POST(S6) both shapes (nr64 49115.7 / nr16 29769.9).
The S6 min-fold staging preserved the end-of-block `vfmacc→vfnmsac` fold order; the timed S6 binary
computes bit-for-bit the S1/golden result. (S1 is oracle-GREEN @039133ea, WORST_NORM ~7e-7 ⇒ S6 is
transitively oracle-GREEN by bit-identity.)

## 1. objdump seal golden → S1 → S6 (the "spill→~0, peak ≤32" gate) → REACHED
| form | vsetvli | spill(vs*r.v) | reload(vl*r.v) | maxVreg | vwmacc | textB |
|---|---:|---:|---:|---:|---:|---:|
| golden -O2 (84-base anchor) | 53 | **84** | 88 | v31 | 2240 | 23486 |
| S1 -O2 (committed d5a28efc)  | 73 | **23** | 34 | v31 | 2240 | 25358 |
| S6 -O2 (measured binary)     | 71 | **3**  | 8  | **v30** | 2240 | 25358 |
| golden -O3 | 52 | 84 | 87 | v31 | 2240 | 23540 |
| S1 -O3     | 73 | 23 | 34 | v31 | 2240 | 25330 |
| S6 -O3     | 71 | **3** | 8 | **v30** | 2240 | 25366 |

- **maxVreg v31 → v30 ⇒ the peak live-set fits in ≤32 vreg** (v30 = 31 registers, 1 spare) at BOTH -O2/-O3.
- **vwmacc 2240 UNCHANGED across golden→S1→S6** ⇒ the widening-MAC multiset is preserved ⇒ byte-exact
  (confirmed by §0). vsetvli 73→71 (slightly fewer — the min-term collapses to one cold panel band).
- **★ RESIDUAL-3 IS NOT HOT-LOOP SPILLING (hardware-verified, `objdump -d s6_O2.o`).** The seal regex
  `vs[1248]r.v` conflates allocator spills with S6's deliberate stack panels (which use `vle/vse` ELEMENT
  ops, not whole-register `vs*r.v`). The 3 residual `vs2r.v v8,(a0)` are at prologue offsets **0x00aa /
  0x00be / 0x00f2** — all BEFORE the hot-dot loops (`.LBB0_9`@0x320e, `.LBB0_10`@0x327e) — storing the same
  zeroed `v8` = the one-time bsums-panel inits. The 8 `vl2r.v` (3 setup + 5 at the fold 0x3200–0x6230) are
  panel READS with no matching spill-STORE. So there are **zero store→reload live-value pairs in the hot
  dot ⇒ allocator hot-loop live-value spills went 23 → 0** (raw seal spill 84→23→3, reload 88→34→8).

**Why it works:** S1's two sequential h-strip loops still keep, per strip, the full sumf/sumi/bsums +
decode band live across the ii dot → 23 residual whole-register spills at the peak. S6 pushes the
idle-across-the-dot `bsums` MIN accumulator + the decode strips + d/dmin OFF the hot vreg set (stack panel
/ on-demand), leaving only the hot main-term accumulators (sumf v20/22/24/26, sumi v47/49/51/53) as
SSA-register locals → the hot dot's peak live-vreg drops under 32 → spill 23→0. The i32 min-accumulation is
reordered but exact+associative, and the f32 fold order is untouched → byte-exact (unlike the T1-doc's
per-sub-block f32 inline-fold, which would have added ~32 f32 op/block and broken bit-exactness — S6 is a
strictly stronger, byte-exact route to the same ≤32 goal).

## 2. A/B (ours-S1 vs ours-S6) — S6 isolated incremental speedup over S1
Paired per-round `ours_post(S6) / ours_pre(S1)` (cold, fresh procs):
- **nr=64, N=12:** S1 6.4769 → S6 8.2817 GMAC/s ⇒ **1.279 (+27.9%)**; per-round A/B min 1.269 / max 1.289.
- **nr=16, N=10:** S1 6.4568 → S6 8.2536 GMAC/s ⇒ **1.279 (+27.9%)** — **SHAPE-ROBUST** (identical to nr64).

Both sides tight (ours cv 0.25–0.49%), so the +27.9% is **~57–67× our own run-to-run noise floor** — a
large, REAL win, decisively non-overlapping (min A/B 1.269 > 1). **T-N PASS.** This is the S6 increment
*on top of* S1's +53%; the golden→S6 stack is ~2.0× (6.48/4.23 vs golden 4.23 in the S1 cell → S6 ≈ 1.96×
the full-unroll golden) but logged HERE as S6-over-S1 alone.

## 3. vs-opponent (S6 vs opponent block-dot) — did parity climb further?
Clean anchor = **nr=64** (opponent block-dot cv 0.80% across all 24 samples):
- **S1 parity  = 1.473×**  (reproduces cell `l1-tile-s1`'s 1.471× → basis/harness validated)
- **S6 parity  = 1.884×**

S6 pushes parity from S1's ~47% lead to a **~88% lead** over the opponent's real dispatched block-dot.
Opponent GMAC/s is stable across the S1/S6 binaries (mean 4.392, cv 0.80%), so the entire jump is our side
speeding up, not opponent variance. At nr=16 the opponent block-dot is noise-dominated (cv 3.2%) so it is
not a usable parity anchor (indicative S6 1.808×; the ours-side +27.9% A/B there is the load-bearing shard).

## Pre-registered judgment → S6 HOLDS
The board pre-registration was: **spill → ~0 ∧ MAC/cycle significantly up ∧ wall passes T-N ⇒ S6 holds.** All met:
- spill **23 → ~0** (raw 23→3; hot-loop live-value spills 23→0, hardware-classified), peak vreg **≤32** (v30) ✓
- MAC/cycle (∝ GMAC/s at fixed 2.6 GHz) **6.48 → 8.28 (+27.9%)** — significantly up ✓
- wall passes T-N **(+27.9% is ~57–67× the noise floor, non-overlapping, shape-robust)** ✓

**S6 SUCCEEDS.** It is the register-cliff step: the S1 fan-out relief (spill 84→23) is completed to the
≤32-vreg cliff (hot-loop spills →0) **byte-exactly**, and the throughput follows (+27.9% over S1;
vs-opponent 1.473→1.884). Logged as **S6 alone** (its own table row); not merged with S1.

## Loadavg / contention honesty
nr64: loadavg pre 3.24 / mid 3.76 / end 3.97. nr16: pre 3.59 / mid 4.49 / end 3.98. A SPEC `bzip2` co-tenant
(~1 core at 99.7%) was present throughout and is captured in the loadavg; core 8 (our taskset pin) was
otherwise quiescent at 2.6 GHz gov=performance. Because the A/B is **paired** (S1 then S6 back-to-back each
round), any shared memory/scheduler contention is common-mode and cancels in the ratio — and this is a
compute-bound prefill GEMM (K=2048, dense vwmacc) so it is less memory-sensitive than the decode cells that
the memory-contention caveat targets. The +27.9% A/B and 1.884 parity reproduced across two shapes and two
loadavg bands, so the result is not a contention artifact.

## [NG-4] discipline
This is an **L1 kernel datapoint, NOT a beat.** The eight [PERF-1] gates are not walked; single core;
opponent = single-thread block-dot loop (kernel-axis proxy, not threaded `mul_mat`); correctness of OUR
kernel is the construction oracle (039133ea) + the byte-exact identity gate above. The 1.884× vs-opponent
is a **KERNEL-axis parity number**, NOT a sealed e2e beat; per `kernel-wins-dont-transplant-to-e2e`, a
compute-side micro win of this shape does not automatically transduce to a memory-bound decode e2e.

## Bottom line (register cliff → throughput, byte-exactly?)
**Yes — S6 reaches it.** The S1 h-strip tile's residual 23-spill peak is driven to the ≤32-vreg cliff
(hot-loop live-value spills → 0, maxVreg v30) by staging the idle MIN accumulator + decode strips to stack
panels while keeping the single end-of-block f32 fold order identical — **byte-exact**, verified
bit-for-bit on hardware. Throughput follows: **+27.9% over S1**, vs-opponent parity **1.473 → 1.884**.
Logged as **S6 alone**; the doc's per-sub-block-fold ≤32 route (non-byte-exact) is superseded by this
strictly stronger byte-exact one.
