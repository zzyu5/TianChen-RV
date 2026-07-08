# [KQUANT-L1 / T3-tile] q3_K repack GEMM — does the q4_K S6 stack-panel lever transfer to q3_K?

**Board:** rvv / VLEN128 (openEuler), core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (the tiling-lever transfer, ★weight-bound prediction):** the q4_K S6 register-cliff step
(cell `l1-tile-s6-q4k-repack-gemm`) converted the full-unroll opening into throughput by staging the
idle-across-the-hot-dot i32 MIN accumulator + the 4-bit decode strips to STACK PANELS, dropping spill
84→23→3, peak vreg v31→v30, +27.9% — **byte-exactly**. q2_K T3 (cell `l1-t3-q2k`) then CONFIRMED the
lever transfers where the peak pressure IS the decode/min strips (shared `kquant_dmin_bsums_min` fold:
spill 619→7, v30, +565% WIN). q6_K T3 (cell `l1-t3-q6k`) showed the lever is a **NULL** where the peak
pressure is dual-plane weight materialization (spill 913→949 rose, v31, −1.5%). T3-fmt3 asks q3_K:
q3_K **SHARES the q6_K no-min single-scale fold** (`kquant_single_scale_no_min` — signed 6-bit scales,
NO dmin/bsums; the parent constructor reuses the q6_K `KQuantDecodeFacts` WHOLE), so the pre-registration
was that q3_K would be **weight-reconstruction-bound like q6_K** — the two-plane 3-bit `qs|hmask`
reconstruction dominates the spill, not the decode-scale strips the panel stages — and the lever would
NOT reach the register cliff.
**Answer: NO — the lever does NOT reach the register cliff. q3_K T3 is a NULL (weight-bound like q6_K).**
spill did NOT reach ~0 (**978 → 894, −8.6%**, still ~900 weight-bound magnitude), peak vreg did NOT reach
≤32 (**v31 → v31**, no cliff — q2_K/q4_K hit v30), and vs-opponent stayed a deep LOSS
(**0.176× → 0.191×**, still ~5.3× behind ggml's block-dot). The tiling IS **byte-exact** (verify
INT_mismatch=0 both + IDENTITY cmp 0 mismatch; vwmacc 2176 unchanged) and does buy a **modest +8.0% / +8.4%**
throughput (the clean scale-panel shaves the ~84 scale-strip spills) — but that is an order of magnitude
short of the q2_K register-cliff win (+565%) and does NOT flip vs-opponent, so it does **not** engage the
tiling lever. **This is the mirror of q2_K: the register-cliff prediction is REFUTED for q3_K — it is
q6_K-shaped, not q4_K/q2_K(min-fold)-shaped.**

## What was compared (real binaries, not extrapolation)
- **UNTILED** = the working-tree HEAD 1b367c1f front-door export of the campaign-canonical fixture
  `rvv-to-emitc-repack-gemm-q3-K-q8-K.mlir` (source op `tcrv_rvv.typed_repack_gemm_loop_body`,
  `decode_model "q3_K"`, `fold_model "kquant_single_scale_no_min"`), lowered via **read-only
  `build/bin/tcrv-opt`** (built at the parent's forced-clean rebuild) + `mlir-translate-20 --mlir-to-cpp`,
  ZERO opt/translate errors (`untiled_q3K_gemm.c`, **md5 813e5851**). The fixture explicitly ships
  PLAIN/UNTILED ("S6 output tiling is a family-lever NULL for weight-reconstruction-bound q3_K"), and the
  front-door PLAIN body is byte-identical to the retired direct emitter = the **board-harvest2 q3_K prefill
  0.176× LOSS** baseline (reproduced below).
- **TILED** = a deterministic S6 stack-panel transform of the SAME plain export
  (`tools/e2e-harness/board/s6_tile_scales.py` → `tiled_q3K_gemm.c`, **md5 31bdf294**): it stages ALL 32
  decoded signed-6bit-scale `vint16m1_t` vectors (`vsext_vf2_i16m1`) through a function-scope int16 stack
  panel — `vse16` right after each decode, `vle16` reload deferred to the scale's FIRST use (the
  `vwmacc_vv_i32m2` scale × weight-partial accumulation, ~4400 lines downstream) — cutting the scale live
  range so the allocator need not keep the 32 scale vectors resident across the fully-unrolled body. This
  is the q3_K analogue of the q6_K S6 no-min emitter (decode-scale int16 panel; NO min fold — q3_K has no
  min). Byte-exact by construction (reloaded == stored == original); proven byte-exact on silicon (§0).
- Both export the SAME extern-C symbol `…repack_gemm_q3_K_q8_K…`, so a single q3_K-only paired driver
  (`kquant_gemm_paired_q3k_driver.c`) links TWICE (untiled.o / tiled.o) into two board binaries differing
  ONLY in the q3_K GEMM object → a clean tiling-isolated A/B. Each linked against the board's OWN
  `libggml-cpu.so` (opponent = real dispatched `ggml_vec_dot_q3_K_q8_K` block-dot).
- Harness: `tools/e2e-harness/board/kquant_gemm_tile_q3k_t3_ab.sh`. Cold N rounds; each round runs untiled
  then tiled as fresh processes; K=2048, nc=512, iters=20. **No git stash** — UNTILED is the working-tree
  read-only export; main tree + build/ read-only (no rebuild). The TILED variant is a /tmp scratch
  transform (nothing in the source tree changed — the shipped emitter stays PLAIN, this cell only *tests*
  whether a tile *would* have helped).

## 0. Byte-exact gates (correctness of the timed TILED binary) → GREEN (both)
- **Silicon verify** (`kquant_repack_verify_q3K`, independent scalar q3_K dequant-matmul reference, seed
  0xC0FFEE), linked with UNTILED then TILED objects:
  - UNTILED: `INT_mismatch_total=0`, NORM `worst_ulp=760 worst_norm=4.973e-07` → BYTE-EXACT-INTEGER + BOUNDED-NORM.
  - TILED:   **identical** — `INT_mismatch_total=0`, NORM `worst_ulp=760 worst_norm=4.973e-07`.
- **Identity dump** (full nr·nc fp32 output, small-finite fp16-scale fold path, seed 0xC0FFEE):
  `o_untiled.bin == o_tiled.bin` → **BYTE-EXACT, 0 mismatch** (nr64 32768 floats; nr16 8192 floats).
- Corroborated by the objdump `vwmacc 2176` multiset being UNCHANGED untiled→tiled (§1).

## 1. objdump seal UNTILED → TILED (the "spill→~0, peak ≤32" gate) → ★NOT REACHED (NULL, weight-bound)
| form | vsetvli | spill(vs*r.v) | reload(vl*r.v) | maxVreg | vwmacc | textB |
|---|---:|---:|---:|---:|---:|---:|
| UNTILED -O2 | 13 | **978** | 1095 | v31 | 2176 | 75686 |
| TILED   -O2 | 13 | **894** | 1016 | v31 | 2176 | 72464 |
| UNTILED -O3 | 12 | 983 | 1099 | v31 | 2176 | 75900 |
| TILED   -O3 | 12 | 910 | 1031 | v31 | 2176 | 73158 |

- **spill 978 → 894 (−O2) / 983 → 910 (−O3)**: the stack-panel shaved only **~84 spills** — EXACTLY the
  32 staged scale strips + their reloads — leaving a **~894 residual**. spill did NOT reach ~0 (q2_K hit
  **7**, q4_K hit **3**). The q3_K fully-unrolled body carries **~900 whole-register spills**, an ORDER OF
  MAGNITUDE more than q4_K golden's 84, and slightly MORE than q6_K's 913 — a definitive weight-bound
  signature.
- **maxVreg v31 → v31** (both −O2/−O3): the peak live-set did NOT drop under 32; **the ≤32-vreg cliff was
  NEVER reached** (q4_K S6 / q2_K S6 both hit v30). No spare register freed.
- **vwmacc 2176 UNCHANGED across all four** ⇒ byte-exact multiset preserved (confirmed by §0). text shrank
  75686→72464 (the staged scale spill/reload collapse), the one place the panel demonstrably works.

**Why it does NOT reach the cliff (the transfer boundary, same as q6_K):** q4_K/q2_K's residual pressure
at the peak WAS the idle i32 MIN accumulator + the decode-scale/min strips — EXACTLY what the panel
targets — so staging them collapsed the WHOLE spill set (→7/→3). q3_K **has no min** and its dominant
pressure is the **dual-plane 3-bit weight reconstruction** (`qs`@800 low-2-bit + subtractive `hmask`@288
high-bit, 16 sub-blocks → the i16 weight-partial-sums that are the OTHER operand of the 256 `vwmacc_vv`),
plus the f32 accumulator fan. Staging only the 32 scale strips is a drop in the ~900-spill bucket. **The
pattern does not migrate q4_K → q3_K without a different lever** (a `qs`/`hmask` materialization panel or a
block-loop re-roll to shrink the unroll) — identical conclusion to q6_K, which q3_K shares the fold with.

## 2. A/B (tiled vs untiled) — tiling-isolated increment → small consistent GAIN (not the cliff)
Paired per-round `ours_tiled / ours_untiled` (cold, fresh procs):
- **nr=64, N=12:** untiled 0.6153 → tiled 0.6643 GMAC/s ⇒ **1.080× (+8.0%)** (ours cv 0.04%; per-round non-overlapping).
- **nr=16, N=10:** untiled 0.5951 → tiled 0.6454 GMAC/s ⇒ **1.084× (+8.4%)** — **SHAPE-ROBUST** (same sign, similar magnitude; ours cv 0.17–0.21%).

Both sides tight (ours cv 0.04–0.21%), so the +8% is real and non-overlapping — **T-N: the "MAC/cycle up"
gate PASSES in direction**, but the magnitude is a minor overhead-shave (the clean panel removes the
scale-strip spill traffic), **NOT** the register-cliff lever engaging (q2_K was +565%, q4_K +27.9% from
reaching ≤32). Interpreted with the objdump: this is the ~8% of runtime that WAS scale-spill overhead;
the ~91% weight-reconstruction spill body is untouched. **It does not clear the pre-registered
spill→~0 ∧ ≤32-cliff bar**, so it is logged as a NULL on the lever (a small weight-bound-limited gain),
not a HOLDS.

## 3. vs-opponent (untiled/tiled vs opponent block-dot) — parity did NOT flip (deep LOSS both)
- **nr=64** (opponent block-dot cv 1.26% across 24 samples, mean 3.484 GMAC/s):
  - **UNTILED parity = 0.176×** (reproduces the board-harvest2 q3_K prefill 0.176× LOSS → basis/harness validated)
  - **TILED parity   = 0.191×** (nudged up by the +8%, still a ~5.2× LOSS)
- **nr=16** (opponent cv 1.41%): UNTILED 0.173× → TILED 0.190× (same LOSS regime, shape-robust).

The opponent's real dispatched `ggml_vec_dot_q3_K_q8_K` block-dot is **~5.2–5.8× faster** than our q3_K
repack GEMM, both untiled and tiled; opponent GMAC/s is stable across binaries, so the small parity climb
is entirely our +8%. **q3_K is a genuine repack LOSS regime at VLEN128, and tiling does NOT rescue it**
(unlike q2_K, which the SAME lever flipped 0.21× LOSS → 1.41× WIN because q2_K shares the min fold).

## Pre-registered judgment → q3_K S6-scheme does NOT hold on the cliff (NULL, weight-bound)
The board pre-registration was: **spill → ~0 ∧ peak vreg ≤32 ∧ MAC/cycle significantly up ∧ wall passes T-N ⇒ holds.**
NOT met:
- spill **978 → 894** (−8.6%, still ~900 weight-bound; NOT ~0) ✗
- peak vreg **v31 → v31** (the ≤32 cliff NOT reached) ✗
- MAC/cycle **+8.0% / +8.4%** — up, but a minor overhead-shave, not the cliff (q2_K +565%) — direction PASSES, magnitude does not clear the lever bar
- wall: a consistent **+8%** (non-overlapping, shape-robust) — real but small

**q3_K T3 FAILS the register-cliff gate ⇒ tiling_holds = FALSE (NULL, weight-bound like q6_K).** This is
the honest boundary of the q4_K S6 lever: it wins where the peak pressure IS the decode/min strips
(q4_K / q5_K / q2_K — the `kquant_dmin_bsums_min` min-fold family) and is a null-to-small-gain where the
peak pressure is dual-plane weight materialization (q6_K / q3_K — the `kquant_single_scale_no_min` fold).
The construction itself (front-door region + no-min scale-panel) is sound and **byte-exact**; it simply
does not buy a register cliff on this format. This **CONFIRMS the parent constructor's "weight-bound like
q6_K, ship PLAIN" ruling on silicon** (and quantifies the residual +8% overhead-shave the plain form
leaves on the table — an order of magnitude short of a rescue).

## Loadavg / contention honesty
nr64: loadavg pre 2.43 / mid 3.13 / end 3.14. nr16: pre 2.62 / mid 3.01 / end 3.07. Core 8 (our taskset
pin) at 2.6 GHz gov=performance. The A/B is **paired** (untiled then tiled back-to-back each round) so
shared memory/scheduler contention is common-mode and cancels in the ratio; the +8.0% (nr64) and +8.4%
(nr16) reproduced across two shapes and stable loadavg bands (ours cv 0.04–0.21%), so the small gain is
not a contention artifact — and the load-bearing verdict (spill 978→894, v31, NULL-on-cliff) is an
objdump fact independent of timing.

## [NG-4] discipline
This is an **L1 kernel datapoint, NOT a beat** (and not an anti-beat claim — our own negative result on
the tiling lever). The eight [PERF-1] gates are not walked; single core; opponent = single-thread
block-dot loop (kernel-axis proxy, not threaded `mul_mat`); correctness of OUR kernel is the construction
oracle + the byte-exact verify + identity gates above. The 0.176× vs-opponent is a **KERNEL-axis parity
number**, and the tiling delta is a **KERNEL-axis structural datapoint**, NOT a sealed e2e result.

## Bottom line (does the tiling lever transfer to q3_K?)
**No — it does not reach the register cliff (NULL, weight-bound like q6_K).** Staging the decode-scale
strips to a stack panel leaves q3_K's ~900-spill dual-plane fully-unrolled body largely untouched (spill
978→894, maxVreg v31, no ≤32 cliff), buying only a modest **+8.0%/+8.4%** overhead-shave (byte-exact:
verify INT_mismatch=0 both + identity 0 mismatch; vwmacc 2176 unchanged) while vs-opponent stays a ~5.3×
LOSS (0.176×→0.191×). Logged as the **q3_K T3 negative datapoint** — with the q6_K T3 NULL and the
q2_K/q4_K T3 WINs it completes the tiling-lever transfer-boundary map (C3′ pattern-library: the lever is
`kquant_dmin_bsums_min` min-fold-shaped = q4_K/q5_K/q2_K, NOT `kquant_single_scale_no_min`-shaped =
q6_K/q3_K), and **confirms on silicon** the parent's "weight-bound, ship PLAIN" q3_K ruling.
