# cell MANIFEST — l1-tile-s1-q4k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (T2-tile S1 — q4_K repack GEMM STRUCTURAL h-strip TILE A/B on
  rvv/VLEN128: does register-tiled column/h-strip blocking convert the full-unroll register-pressure
  "opening" into throughput? Third lever after `l1-pipeline-q4k-repack-gemm` (#1 byte-exact reorder =
  NULL) and `l1-reroll-q4k-repack-gemm` (#2 naive k/ii re-roll = −11% REGRESSION). S1 = step 1 of the
  6-step tile ladder; logged ALONE, not merged with S2–S6.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ CRUX: the S1 STRUCTURAL TILE (split the single fully-unrolled block loop into TWO per-h-strip
  `emitc.for` block loops (mr=4/hs=1), each carrying its OWN per-strip sumf/sumi/bsums accumulators;
  byte-exact-identical, oracle GREEN) **CONVERTS the opening — S1 HOLDS.** (1) objdump at the measured
  clang-17 -O2 board form: spill **84 → 23 (−73%)**, reload **88 → 34 (−61%)** at BOTH -O2 and -O3
  (vsetvli rose 53→73 (+38%, cheap scalar re-config), text +8%; **vwmacc 2240→2240 UNCHANGED** =
  byte-exact multiset). The PRE -O2 seal reproduces the `l1-reroll` cell's PRE seal BYTE-FOR-BYTE →
  basis validated as campaign-canonical. (2) A/B ours_post/ours_pre paired cold (nr=64 N=12 / nr=16
  N=10): **1.527 / 1.519 = a consistent +53% / +52% SPEEDUP**, ~55–179× our own run-to-run cv → T-N
  PASS (real, decisively non-overlapping; contrast #1 +0.65% wash, #2 −11% regression). (3) vs-opponent
  block-dot (clean anchor nr=64): PRE parity **0.962×** (reproduces the precedent's 0.94–0.97× band →
  harness validated) → POST parity **1.471×** — parity **FLIPS from under-parity to a ~47% LEAD** over
  the opponent's real dispatched block-dot; opponent GMAC/s stable across pre/post (24 samples, cv
  ~0.54%) so the flip is 100% ours-side. **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked; single
  core; opponent = single-thread block-dot proxy; our correctness = construction oracle (039133ea) + the
  byte-exact identity gate in this cell. ⚠ This CONTRADICTS the build-phase static prediction (which,
  on a driver-export basis / different objdump counting, called S1 a "spill ≈ FLAT" structural NULL);
  the campaign-canonical fixture basis + silicon correct it — the two h-strip passes are SEQUENTIAL so
  the peak live-set HALVES (not sums). The remaining S2–S6 tile steps (toward the ≤32-vreg spill-cliff)
  + an e2e transduction check are separate and out of scope for this S1-only datapoint.
- **role**: L1-maturity T2-tile S1 A/B evidence. Isolates the S1 h-strip STRUCTURAL TILE from the
  full-unroll golden by measuring PRE (full-unroll golden, == cached baseline md5 b0b5beac) vs POST
  (S1-tiled, two per-h-strip block loops) as separate board binaries against the opponent's REAL linked
  block-dot, with a PRE-vs-POST byte-exact `cmp` gate proving the timed POST is the golden result.
  Harness/driver live under `tools/e2e-harness/board/` (`kquant_gemm_tile_s1_ab.sh` +
  `kquant_gemm_paired_driver.c`). This cell holds data/evidence only; the exported .c are NOT stored
  (deterministically regenerable — see the objdump seal + the sibling
  `kquant-l1-q4k-q5k-repack-prefill/EXPORT_RECIPE.md`; the S1 tile is the working-tree emitter delta on
  `RVVToEmitCBlockQuantLinear.cpp` (`emitRepackKQuantGemmBodyQ4K`), uncommitted).

## durable files

- `tile_s1_findings.md`           — writeup: byte-exact gate, spill 84→23 DROP, A/B +53%/+52%,
  vs-opponent parity 0.962→1.471 flip, S1-HOLDS verdict, build-report-discrepancy note, [NG-4] discipline.
- `tile_s1_ab_nr64.txt`           — raw board A/B, cold N=12, nr=64 (clean opponent anchor) + objdump + oracle.
- `tile_s1_ab_nr16.txt`           — raw board A/B, cold N=10, nr=16 (ours-side A/B reproduced; opponent noise-dominated).
- `tile_s1_ab.csv`                — parsed summary (nr, variant, ours/opp GMAC/s, cv, parity, A/B, vsetvli, spill).
- `objdump_tile_s1_seal.objdump`  — PRE vs POST vsetvli/spill/reload/vwmacc/text seal at clang-17 -O2 AND -O3 + mechanism.

## note — data-only cell
Data/evidence only. The load-bearing result is a **WIN (S1 HOLDS)**: the S1 h-strip structural tile
relieves the register-pressure opening — spill drops 84→23, board throughput +53%, and vs-opponent
parity flips 0.962× → 1.471× (a ~47% lead, KERNEL-axis). [NG-4] — L1 datapoint, not a beat. Nothing here
was committed / git add-ed; main tree + build/ untouched (no rebuild — PRE is the cached golden baseline;
POST is the working-tree S1-tile export via read-only `tcrv-opt`). Logged as S1 ALONE (not merged S2–S6).
