# [KQUANT-L1 / T3-tile] q2_K repack GEMM — does the q4_K S6 stack-panel lever transfer to q2_K?

**Board:** rvv / VLEN128 (openEuler), core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (the tiling-lever transfer, ★register-cliff prediction):** the q4_K S6 register-cliff step
(cell `l1-tile-s6-q4k-repack-gemm`) converted the full-unroll opening into throughput by staging the
idle-across-the-hot-dot i32 MIN accumulator + the 4-bit decode strips to STACK PANELS, dropping spill
84→23→3, peak vreg v31→v30, +27.9% — **byte-exactly**. q6_K T3 (cell `l1-t3-q6k-repack-gemm`) then
showed the lever is a **NULL** where the peak pressure is dual-plane weight materialization. T3-fmt2 asks
q2_K: does the SAME lever transfer? q2_K **SHARES the q4_K min fold** (`kquant_dmin_bsums_min` — dual
d/dmin + bsums-min, 16 sub-blocks), so the pre-registration was that q2_K would eat the register cliff
(like q4_K) and NOT be weight-bound (unlike q6_K).
**Answer: YES — the lever transfers. q2_K T3 is a decisive HOLDS/WIN.** spill collapsed to ~0
(**619 → 7**, −98.9%), peak vreg reached the cliff (**v31 → v30**, ≤32), throughput rose **+565%**
(ours **0.9546 → 6.3513 GMAC/s**, A/B **6.65×**), and vs-opponent flipped from a **0.21× LOSS to a
1.41× WIN**. The tiling is **byte-exact** (verify INT_mismatch=0 both + IDENTITY cmp 0 mismatch;
vwmacc 2304 unchanged). This is the mirror image of q6_K: **the register-cliff prediction is CONFIRMED.**

## What was compared (real binaries, not extrapolation)
- **UNTILED** = HEAD c3cf7301 export of the q2_K GEMM (`untiled_q2K_gemm.c`, **md5 6c9322f6**;
  source op `tcrv_rvv.repack_gemm_q2_K_q8_K` full-unroll direct-emit). Pulled from the cached 924dc31f
  export in the `kquant-l1-q6q2q3-repack` cell's export dir (== the **board-harvest2 q2_K prefill ~0.20×
  LOSS** baseline). Provenance-clean: the `emitRepackGemmQ2KQ8K` direct-emit body is **byte-identical
  between 924dc31f and HEAD c3cf7301** — the only q2_K token in that commit range is a ternary comment,
  not the q2_K emitter — so the cached export == HEAD's retired-direct-emit UNTILED baseline.
- **TILED** = the campaign-canonical fixture `rvv-to-emitc-repack-gemm-q2-K-q8-K.mlir` (working-tree
  FRONT-DOOR region form) re-lowered through the working-tree S6 emitter delta on
  `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (`emitRepackKQuantGemmBodyQ2K`; `tiled_q2K_gemm.c`,
  **md5 9c5bac28**), regenerated via **read-only `build/bin/tcrv-opt`** (already built at the parent's
  forced-clean rebuild; ZERO opt/translate errors) + `mlir-translate-20 --mlir-to-cpp`.
- Both export the SAME extern-C symbol `…repack_gemm_q2_K_q8_K…`, so a single q2_K-only paired driver
  (`tools/e2e-harness/board/kquant_gemm_paired_q2k_driver.c`) links TWICE (untiled.o / tiled.o) into two
  board binaries differing ONLY in the q2_K GEMM object → a clean tiling-isolated A/B. Each linked against
  the board's OWN `libggml-cpu.so` (opponent = real dispatched `ggml_vec_dot_q2_K_q8_K` block-dot; q2_K's
  repack case-128 is a TODO in ggml → falls back to exactly this per-(row,col) block-dot).
- Harness: `tools/e2e-harness/board/kquant_gemm_tile_q2k_t3_ab.sh`. Cold N rounds; each round runs
  untiled then tiled as fresh processes; K=2048, nc=512, iters=20. **No git stash** — untiled is the
  cached HEAD export; main tree + build/ read-only (no rebuild).

## 0. Byte-exact gates (correctness of the timed TILED binary) → GREEN (both)
- **Silicon verify** (`kquant_repack_verify_q2K`, independent reference decoded from the ORIGINAL
  pre-repack block, canonical `vec_dot_q2_K_q8_K` isum − summs fold), linked with UNTILED then TILED
  objects, seed 0xC0FFEE:
  - UNTILED: `INT_mismatch_total=0`, NORM `worst_ulp=3402 worst_norm=6.618e-07` → BYTE-EXACT-INTEGER + BOUNDED-NORM.
  - TILED:   **identical** — `INT_mismatch_total=0`, NORM `worst_ulp=3402 worst_norm=6.618e-07`.
  - (The `worst_ulp=3402` is on the adversarial-NORM path only — INT path is byte-exact. q2_K's ULP is
    higher than q6_K's 220 because q2_K folds TWO fp16 scales (d AND dmin), so the fp divergence from the
    f64 reference is larger; still bounded-norm 6.6e-07 ⇒ the fp16 fold is faithful.)
- **Identity dump** (full nr·nc fp32 output, small-finite dual-fp16 d/dmin fold path, seed 0xC0FFEE):
  `o_untiled.bin == o_tiled.bin` → **BYTE-EXACT, 0 mismatch** (nr64 32768 floats; nr16 8192 floats).
- Corroborated by the objdump `vwmacc 2304` multiset being UNCHANGED untiled→tiled (§1). The S6
  scale/min-panel staging preserved the end-of-block fold order; the timed TILED computes bit-for-bit the
  untiled/oracle result.

## 1. objdump seal UNTILED → TILED (the "spill→~0, peak ≤32" gate) → ★REACHED (HOLDS)
| form | vsetvli | spill(vs*r.v) | reload(vl*r.v) | maxVreg | vwmacc | textB |
|---|---:|---:|---:|---:|---:|---:|
| UNTILED -O2 | 77 | **619** | 747 | v31 | 2304 | 51014 |
| TILED   -O2 | 87 | **7**   | 12  | v30 | 2304 | 26968 |
| UNTILED -O3 | 76 | 603 | 730 | v31 | 2304 | 47796 |
| TILED   -O3 | 87 | 7   | 12  | v30 | 2304 | 26974 |

- **spill 619 → 7 (−O2) / 603 → 7 (−O3)**: the stack-panel DROVE spill to ~0 (a −98.9% collapse). The
  q2_K untiled full-unroll body carried ~600 whole-register spills — NOT the q4_K golden 84, but a
  q6_K-magnitude ~600 — YET, unlike q6_K, that pressure is EXACTLY the decode-4bit-scale/min strips + the
  idle-across-the-hot-dot i32 MIN (bsums) accumulator that the S6 int16/i32 panels stage. So staging them
  collapses the WHOLE spill set.
- **maxVreg v31 → v30** (both −O2/−O3): the peak live-set dropped to v30 — **the ≤32-vreg register cliff
  was REACHED** (the same v31→v30 step q4_K S6 hit). q6_K never left v31; q2_K does.
- **vwmacc 2304 UNCHANGED across all four** ⇒ byte-exact multiset preserved (confirmed by §0). textB
  **HALVED** 51014→26968 (the fully-unrolled spill-and-reload sequence collapses into a compact panel
  store/load + a register-resident hot dot). vsetvli 77→87 (a few extra vtype toggles for the panel
  store/load bands; the hot dot itself is register-resident).
- **Hot-loop classification (TILED −O2):** there are only **7 `vs*r.v` total**, so all are shown: 5
  cluster in the prologue (0xd2/0xe6/0x10e/0x20a/0x21a) and 2 at the far epilogue (0x35d8/0x35e8). These
  are benign function-boundary whole-register saves — the same pattern as q4_K S6's benign residual — NOT
  hot-loop allocator live-value spills.

**Why it DOES work (the transfer boundary, mirror of q6_K):** q4_K's residual pressure at the peak WAS the
idle i32 MIN accumulator (`bsums`) + the 4-bit decode strips — EXACTLY what the stack-panel targets. q2_K
**shares that fold** (`kquant_dmin_bsums_min`): 2-bit weight decode is cheap, and the peak pressure is the
same dual-d/dmin scale/min decode + MIN accumulator. So the SAME lever collapses spill 619→7 / v31→v30.
q6_K's dominant pressure was the **dual-plane 6-bit weight reconstruction** (`ql`@1312 + `qh`@288), which
the decode-scale panel does not touch — hence the q6_K NULL. **The pattern migrates q4_K → q2_K (and, by
the shared fold, q5_K) but not q4_K → q6_K.**

## 2. A/B (tiled vs untiled) — tiling-isolated increment → massive consistent WIN
Paired per-round `ours_tiled / ours_untiled` (cold, fresh procs):
- **nr=64, N=12:** untiled 0.9546 → tiled 6.3513 GMAC/s ⇒ **6.653× (+565%)**; per-round A/B min 6.595, max 6.713.
- **nr=16, N=10:** untiled 0.9235 → tiled 5.2160 GMAC/s ⇒ **5.648× (+465%)** — **SHAPE-ROBUST** (same sign, similar magnitude).

Both sides tight on the ours axis (untiled cv 0.12–0.19%, tiled cv 0.49–1.40%), so the ×5.6–6.7 delta is
massively non-overlapping (per-round A/B min 5.525 > 1) — a decisive WIN, not noise. **T-N: the "MAC/cycle
up" gate PASSES** (direction is up, huge margin). The untiled 0.955 GMAC/s is spill-bound (619 whole-reg
spills, half its code is spill/reload); the tiled 6.35 GMAC/s is register-resident.

## 3. vs-opponent (untiled/tiled vs opponent block-dot) — parity FLIPPED to a WIN
Clean anchor = **nr=64** (opponent block-dot cv 1.14% across all 24 samples, mean 4.4955 GMAC/s):
- **UNTILED parity = 0.2124×**  (reproduces the board-harvest2 q2_K prefill ~0.20× LOSS → basis/harness validated)
- **TILED parity   = 1.4128×**  (**WIN** — our tiled q2_K repack GEMM is 1.41× FASTER than ggml's block-dot)

The opponent's real dispatched `ggml_vec_dot_q2_K_q8_K` block-dot is stable across binaries (mean 4.50, cv
1.14%), so the parity flip is entirely our side (0.955 → 6.351 GMAC/s). At nr16 the opponent is
noise-dominated (cv 5.3%) so it is not a usable parity anchor (indicative tiled parity 1.23×; the ours-side
A/B 5.65× is the load-bearing shard there). **q2_K, un-tiled a repack LOSS regime at VLEN128, becomes a
repack WIN once the register cliff is taken.**

## Pre-registered judgment → q2_K S6-scheme HOLDS (ship tiled)
The board pre-registration was: **spill → ~0 ∧ MAC/cycle significantly up ∧ wall passes T-N ⇒ holds.**
ALL met:
- spill **619 → 7** (→~0, prologue-only residual), peak vreg **v31 → v30** (≤32 cliff REACHED) ✓
- MAC/cycle (∝ GMAC/s at fixed 2.6 GHz) **0.9546 → 6.3513 (+565%)** — up, huge margin ✓
- wall: a decisive **6.65× / 5.65× WIN** (non-overlapping, shape-robust) — passes the "improvement" direction ✓

**q2_K T3 PASSES the tiling gate.** This CONFIRMS the register-cliff prediction and maps the tiling-lever
transfer boundary from the other side: the q4_K S6 lever wins where the peak pressure IS the decode/min
strips (q4_K / q5_K / q2_K — the shared `kquant_dmin_bsums_min` fold) and is a null-to-regression where the
peak pressure is dual-plane weight materialization (q6_K). The construction itself (front-door region +
scale/min + inline-min-fold stack-panel) is sound and **byte-exact**; here it also buys the register cliff.

## Loadavg / contention honesty
nr64: loadavg pre 2.13 / mid 2.69 / end 3.05. nr16: pre 2.36 / mid 2.83 / end 2.98. Core 8 (our taskset
pin) at 2.6 GHz gov=performance. The A/B is **paired** (untiled then tiled back-to-back each round) so
shared memory/scheduler contention is common-mode and cancels in the ratio; the ×6.65 (nr64) and ×5.65
(nr16) reproduced across two shapes and stable loadavg bands, so the win is not a contention artifact
(and it is far too large to be one — the untiled cv is 0.12%).

## [NG-4] discipline
This is an **L1 kernel datapoint, NOT a beat**. The eight [PERF-1] gates are not walked; single core;
opponent = single-thread block-dot loop (kernel-axis proxy, not threaded `mul_mat`); correctness of OUR
kernel is the construction oracle + the byte-exact verify + identity gates above. The 1.41× vs-opponent is
a **KERNEL-axis parity number**, and the ×6.65 tiling delta is a **KERNEL-axis structural datapoint**, NOT
a sealed e2e result.

## Bottom line (does the tiling lever transfer to q2_K?)
**Yes — decisively.** The q4_K S6 register-cliff win generalizes to q2_K (shared min fold): staging the
decode-scale/min strips + the idle MIN accumulator to stack panels collapses q2_K's ~600-spill full-unroll
body (spill 619→7, maxVreg v30, ≤32 cliff reached, text halved), throughput rises 5.6–6.7×, and vs-opponent
flips 0.21× LOSS → 1.41× WIN. Byte-exact throughout (verify INT_mismatch=0 both + identity cmp 0 mismatch;
vwmacc 2304 unchanged). Logged as the **q2_K T3 positive datapoint** — it CONFIRMS the register-cliff
prediction and, together with the q6_K T3 NULL, delineates the transfer boundary of the tiling pattern
(C3′ pattern-library: q4_K/q5_K/q2_K-shaped = the `kquant_dmin_bsums_min` min-fold family, not q6_K-shaped).
