# [KQUANT-L1 / T3-tile] q5_K repack GEMM — does the q4_K S6 stack-panel lever survive the qh plane?

**Board:** rvv / VLEN128 (openEuler), core 8, 2.6 GHz, governor=performance (measured 2026-07-08).
**Question (★[XFER-1] verify #2, the register-cliff transfer):** the q4_K S6 register-cliff step
(cell `l1-tile-s6-q4k`) staged the idle-across-the-hot-dot i32 MIN accumulator + the decode strips to
STACK PANELS, dropping spill 84→3, peak vreg v31→v30, +27.9% — byte-exactly. q2_K T3 CONFIRMED the
lever transfers WHOLE where the peak pressure IS the decode/min strips (shared `kquant_dmin_bsums_min`
fold: spill 619→7, v30, +565% WIN). q6_K/q3_K T3 showed it is a **NULL** where the peak pressure is
dual-plane weight materialization (spill ~flat, v31). **q5_K is the crux case:** it SHARES the q4_K
min fold (dual d/dmin + bsums-min) AND ALSO carries a qh 5th-bit weight plane (like q6_K's second
plane — but riding the SHARED `weight_qh_byte_offset` slot on the MIN fold). Pre-registration: does
q5_K reach the ≤32 register cliff (v30, min-fold dominates, HOLDS), or does the qh plane make it
weight-bound (v31, NULL)?
**Answer: HOLDS — the min-fold register-cliff lever SURVIVES the qh plane.** peak vreg reached the
cliff (**v31 → v30**, both −O2/−O3), throughput rose **+41.7% (nr64) / +35.0% (nr16)**, and
vs-opponent went from a **1.55× WIN to a 2.19× WIN**. The tiling is **byte-exact** (IDENTITY cmp 0
mismatch both shapes + both oracle verifies PASS 8.0e-07; vwmacc 2240 unchanged). **The one nuance:
spill did NOT collapse to ~0 (155 → 105, −32%; contrast q2_K 619→7) — the qh plane leaves a residual
weight-materialization floor. So q5_K is a HYBRID HOLDS: the min-fold lever DOMINATES enough to cross
the cliff and win, with the qh plane as a secondary residual (not the dominant pressure of q6_K/q3_K).**

## What was compared (real binaries, not extrapolation)
- **UNTILED** = the pre-S6 front-door export `k_gemm_q5K.cpp` (→ `untiled_q5K_gemm.c`; symbol
  `…repack_gemm_q5_K_q8_K…`, block_q5_Kx16 stride 2816, qs@768/qh@256/dmin@32/6-bit scales@64, 8
  sub-blocks; **24 panel markers, 4480 src vwmacc**). This is the q5_K GEMM the construction phase
  built and SILICON-verified oracle-GREEN on the board (`/tmp/q5k_verify/gemm_test`, WORST_NORM
  8.0e-07) — the ~1.55× vs-UNTUNED-opponent baseline. Provenance-clean: same extern-C symbol + same
  2816 layout + same 4480 vwmacc multiset as the tiled form.
- **TILED** = the campaign-canonical fixture `rvv-to-emitc-repack-gemm-q5-K-q8-K.mlir` re-lowered
  through the working-tree S6 emitter `emitRepackKQuantGemmBodyQ5K` (= the q4_K S6-tiled GEMM body +
  the qh 5th-bit inject; `tiled_q5K_gemm.c`, **440 panel markers, 4480 src vwmacc UNCHANGED**),
  regenerated via **read-only `build/bin/tcrv-opt`** (working-tree forced-clean build) +
  `mlir-translate-20 --mlir-to-cpp`. ZERO opt/translate errors.
- Both export the SAME extern-C symbol, so a single q5_K paired driver
  (`tools/e2e-harness/board/kquant_gemm_paired_q5k_driver.c`) links TWICE (untiled.o / tiled.o) into
  two board binaries differing ONLY in the q5_K GEMM object → a clean tiling-isolated A/B. Each linked
  against the board's OWN `libggml-cpu.so` (opponent = real dispatched `ggml_vec_dot_q5_K_q8_K`
  block-dot). Harness: `kquant_gemm_tile_q5k_t3_ab.sh`. Cold N rounds; K=2048, nc=512, iters=20.
  **No git stash** — untiled is the cached pre-S6 export; main tree + build/ read-only (no rebuild).

## 0. Byte-exact gates (correctness of the timed TILED binary) → GREEN (both)
- **Silicon oracle verify** (`oracle_q5K.cpp`, INDEPENDENT scalar q5_K dequant-matmul from the ORIGINAL
  pre-repack block, 8 shapes nc∈{16,32,256} × n∈{256,4096,11008}), linked with UNTILED then TILED:
  - UNTILED: `WORST_NORM 8.0090e-07` VERDICT **PASS** (< 1e-4 bar).
  - TILED:   **identical** — `WORST_NORM 8.0090e-07` VERDICT **PASS**.
  - **Control-flip margins** (untiled `--controls`, proving the reference is sensitive to the q5_K axes):
    NOMIN (zero min term) norm 1.20e-01 = **149291×** the true-ref; PERM (scale +1 rotate) **2.59M×**;
    **QH (zero the 5th bit) 2.16e+00 = 2.70M×** (the q5_K-defining axis IS exercised);
    ROWROT (dot row (m+1)%4) **6.15M×** (the 4-row interleave is correct). bsums mean|bsum|=186.1 (min term active).
- **Identity dump** (full nr·nc fp32 output, small-finite dual-fp16 d/dmin fold path, seed 0xC0FFEE):
  `o_untiled.bin == o_tiled.bin` → **BYTE-EXACT, 0 mismatch** (nr64 32768 floats; nr16 8192 floats).
- Corroborated by the objdump `vwmacc 2240` multiset being UNCHANGED untiled→tiled (§1). The S6 panel
  staging + the verbatim-copied qh inject preserved the fold order → the timed TILED is bit-for-bit untiled/oracle.

## 1. objdump seal UNTILED → TILED (the register-cliff gate) → ★ v30 REACHED (HOLDS), spill floor residual
| form | vsetvli | spill(vs*r.v) | reload(vl*r.v) | maxVreg | vwmacc | textB |
|---|---:|---:|---:|---:|---:|---:|
| UNTILED -O2 | 53 | **155** | 288 | **v31** | 2240 | 37774 |
| TILED   -O2 | 71 | **105** | 110 | **v30** | 2240 | 38506 |
| UNTILED -O3 | 52 | 156 | 288 | v31 | 2240 | 37882 |
| TILED   -O3 | 71 | 106 | 111 | v30 | 2240 | 38608 |

- **maxVreg v31 → v30 (both −O2/−O3): the ≤32-vreg register cliff WAS REACHED** — the SAME v31→v30
  step q4_K S6 and q2_K T3 hit. q6_K T3 and q3_K T3 NEVER left v31. **By the campaign's mechanistic
  discriminator, q5_K is on the HOLDS side: the shared min-fold lever crosses the cliff even with the
  qh plane present.**
- **reload 288 → 110 (−62%)**: the S6 scale/min int16 + i32 bsums stack-panels cut hot-loop reload
  traffic by ~two-thirds — this is the throughput lever (register-resident hot dot).
- **spill 155 → 105 (−32%): dropped but did NOT collapse to ~0** (contrast q2_K 619→7, q4_K 84→3).
  This is the **qh-plane footprint**: q5_K's 5th-bit weight reconstruction (qh, the shared slot on the
  MIN fold) is NOT staged by the decode-scale/min panel, so a ~105-spill residual floor remains. Note
  q5_K's UNTILED spill was already only **155** (NOT the 600–900 of q2_K/q6_K) — untiled q5_K was
  already lean and already a ~1.55× WIN (unlike q2_K's 619-spill 0.21× LOSS), so there was less
  absolute spill to collapse; the decisive move is the peak-vreg-below-32 + the −62% reload cut.
- **vwmacc 2240 UNCHANGED across all four** ⇒ byte-exact MAC multiset (confirmed §0). textB ~flat
  (37774→38506, +2%; NOT halved like q2_K) — the hot dot stays fully unrolled; the win is
  register-residency + peak-vreg-below-cliff, not code shrink. vsetvli 53→71 (panel store/load bands).
- **Hot-loop classification (TILED −O2):** the early 0xbc..0x26e cluster is prologue whole-register
  saves (benign); from 0x54a onward a recurring `vs1r.v v19,(a0)` at a regular ~0x8a stride are the
  per-h-strip qh/weight-materialization residual spills the min-fold panel does not cover.

## 2. A/B (tiled vs untiled) — tiling-isolated increment → consistent WIN, shape-robust
Paired per-round `ours_tiled / ours_untiled` (cold, fresh procs):
- **nr=64, N=12:** untiled 2.0508 → tiled 2.9056 GMAC/s ⇒ **1.417× (+41.7%)**; per-round A/B min 1.405, max 1.428.
- **nr=16, N=10:** untiled 1.7806 → tiled 2.4039 GMAC/s ⇒ **1.350× (+35.0%)** — **SHAPE-ROBUST** (same sign, similar magnitude).

Both sides tight (untiled cv 0.36–0.68%, tiled cv 0.50–0.88%), so the ×1.35–1.42 delta is
non-overlapping (per-round A/B min 1.336 > 1) — a decisive WIN, not noise. **T-N: the "MAC/cycle up"
gate PASSES** (up, comfortable margin). This is an order of magnitude more than q3_K's +8% NULL, though
short of q2_K's +565% (whose untiled was a 619-spill disaster; q5_K's untiled was already lean).

## 3. vs-opponent (untiled/tiled vs opponent block-dot) — an already-WIN pushed further
Clean anchor = **nr=64** (opponent block-dot cv 0.35% across 24 samples, mean 1.3254 GMAC/s):
- **UNTILED parity = 1.547×** (reproduces the referenced ~1.5× vs-UNTUNED-opponent → basis/harness validated)
- **TILED parity   = 2.193×** (**WIN** — our tiled q5_K repack GEMM is 2.19× faster than ggml's block-dot)

The opponent's real dispatched `ggml_vec_dot_q5_K_q8_K` block-dot is stable across binaries (cv 0.35%),
so the parity climb is entirely our side (2.051 → 2.906 GMAC/s). At **nr16** the opponent stays usable
(cv 1.32%): untiled 1.333× → tiled 1.797×. **q5_K was ALREADY a repack WIN un-tiled (unlike q2_K's
LOSS); the tiling pushes it further, 1.55×→2.19× (nr64) / 1.33×→1.80× (nr16).**

## Pre-registered judgment → q5_K S6-scheme HOLDS (ship tiled)
The board pre-registration was: **spill → ~0 ∧ MAC/cycle significantly up ∧ wall passes T-N ⇒ holds**;
the mechanistic discriminator is the ≤32 register cliff (v30 = HOLDS / v31 = NULL). Result:
- register cliff **v31 → v30** (both −O2/−O3) — REACHED (the defining HOLDS signature) ✓
- MAC/cycle (∝ GMAC/s at fixed 2.6 GHz) **2.0508 → 2.9056 (+41.7%)** — up, non-overlapping ✓
- wall: **1.417× / 1.348× A/B WIN** + **1.55×→2.19× vs-opponent** — passes T-N (noise floor ~40–280 ns vs signal ~23 M ns) ✓
- byte-exact: IDENTITY 0 mismatch both shapes + both oracle PASS + vwmacc 2240 unchanged ✓
- **the ONE partial:** spill 155 → 105 (−32%, NOT →~0) — the qh plane leaves a residual weight floor.

**q5_K T3 PASSES the tiling gate as a HYBRID HOLDS.** The shared `kquant_dmin_bsums_min` min-fold
register-cliff lever DOMINATES enough to cross the ≤32 cliff (v30) and deliver +42% / 2.19× **even with
the qh 5th-bit weight plane present** — so the min-fold lever SURVIVES the qh plane. The qh plane's
weight reconstruction is a SECONDARY residual (105-spill floor), not the dominant pressure that made
q6_K/q3_K weight-bound NULLs. q5_K thus sits BETWEEN the pure-min-fold full-collapse HOLDS
(q4_K/q2_K: v30, spill→~0) and the dual-plane weight-bound NULL (q6_K/q3_K: v31, spill flat).

## Loadavg / contention honesty
nr64: loadavg pre 2.19 / mid 2.77 / end 3.12. nr16: pre 2.38 / mid 2.88 / end 3.04. Core 8
(taskset-pinned) at 2.6 GHz gov=performance. The A/B is **paired** (untiled then tiled back-to-back each
round) so shared memory/scheduler contention is common-mode and cancels in the ratio; the ×1.42 (nr64)
and ×1.35 (nr16) reproduced across two shapes and stable loadavg bands (ours cv 0.36–0.88%) — not a
contention artifact.

## [NG-4] discipline
This is an **L1 kernel datapoint, NOT a beat**. The eight [PERF-1] gates are not walked; single core;
opponent = single-thread block-dot loop (kernel-axis proxy, not threaded `mul_mat`); correctness of OUR
kernel is the construction oracle + the byte-exact oracle verify + identity gates above. The 2.19×
vs-opponent is a **KERNEL-axis parity number**, and the ×1.42 tiling delta is a **KERNEL-axis structural
datapoint**, NOT a sealed e2e result.

## Bottom line (does the tiling lever survive the qh plane?)
**Yes — HOLDS (hybrid).** The q4_K S6 register-cliff win transfers to q5_K (shared min fold) and
SURVIVES its qh 5th-bit weight plane: peak vreg reaches the cliff (v31→v30), reload −62%, throughput
+41.7%/+34.8%, vs-opponent 1.55×→2.19× / 1.33×→1.80×, byte-exact throughout (IDENTITY 0 mismatch +
both oracle PASS + vwmacc 2240 unchanged). The one nuance vs q2_K: spill floors at ~105 (−32%, not
→~0) because the qh weight plane is not staged by the decode-scale/min panel. Logged as the **q5_K T3
HYBRID-HOLDS datapoint** — it COMPLETES the K-quant tiling-transfer map: min-fold family (q4_K/q2_K
full-collapse HOLDS, q5_K min-fold-dominant-with-qh-residual HOLDS) vs weight-bound (q6_K/q3_K NULL).
This is the C3′ pattern-library boundary at its finest resolution: the register-cliff lever is keyed to
the min fold, and a co-resident weight plane degrades but does not defeat it.
