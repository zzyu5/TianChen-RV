# cell MANIFEST — l1-tile-s6-q4k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (T2-tile S6 — q4_K repack GEMM register-cliff step on rvv/VLEN128:
  does the S6 inline-min-fold + stack-panel drive the S1 tile's residual register pressure to the
  ≤32-vreg cliff — hot-loop live-value spills → ~0 — *byte-exactly*, and does the throughput follow?
  Step 6 of the 6-step tile ladder, built on committed S1 (d5a28efc, cell `l1-tile-s1-q4k-repack-gemm`,
  spill 84→23 / +53%). Logged ALONE, not merged with S1.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ CRUX: the S6 levers (stage the idle-across-the-hot-dot i32 MIN accumulator `bsums` + the 6-bit
  scale/min i16 decode strips to STACK PANELS via opaque `vle/vse`, and defer the d/dmin f16 widen
  on-demand, keeping the SINGLE end-of-block `vfmacc`(main)→`vfnmsac`(min) f32 fold order IDENTICAL)
  **REACH THE CLIFF byte-exactly — S6 HOLDS.** (1) objdump at the measured clang-17 -O2 board form
  (campaign-canonical fixture + `vs*r.v`/`vl*r.v` seal, golden PRE=84 base validated): spill golden
  **84 → S1 23 → S6 3**, reload **88 → 34 → 8**, maxVreg **v31 → v31 → v30 (peak ≤32)** at BOTH -O2/-O3;
  **vwmacc 2240 UNCHANGED across all three** = byte-exact multiset. The residual raw spill=3 is NOT
  hot-loop spilling — hardware-verified (`objdump -d s6_O2.o`): the 3 `vs2r.v` are prologue bsums-panel
  zero-inits (offsets 0x00aa/0x00be/0x00f2, before the hot loops `.LBB0_9`@0x320e/`.LBB0_10`@0x327e) and
  the 8 `vl2r.v` are panel READS with no matching spill-STORE ⇒ **allocator hot-loop live-value spills
  23 → 0.** (2) byte-exact identity gate PRE(S1) vs POST(S6) (int + norm, both shapes): **IDENTICAL, 0
  mismatch** — the timed S6 computes bit-for-bit the S1/golden result. (3) A/B ours_post(S6)/ours_pre(S1)
  paired cold (nr=64 N=12 / nr=16 N=10): **1.279 / 1.279 = a consistent +27.9% SPEEDUP**, ~57–67× our own
  run-to-run cv, min A/B 1.269 > 1 → T-N PASS, SHAPE-ROBUST. (4) vs-opponent block-dot (clean anchor
  nr=64): S1 parity **1.473×** (reproduces cell `l1-tile-s1`'s 1.471× → basis/harness validated) → S6
  parity **1.884×** — the ~47% lead climbs to a **~88% lead**; opponent GMAC/s stable across S1/S6 (24
  samples, cv 0.80%) so the jump is 100% ours-side. This is a strictly STRONGER, byte-exact route to the
  ≤32 goal than the T1 design-doc's per-sub-block f32 inline-fold (which would have added ~32 f32 op/block
  and broken bit-exactness). The S6 increment is ON TOP of S1's +53% (golden→S6 ≈ 1.96×), logged HERE as
  S6-over-S1 alone. **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked; single core; opponent =
  single-thread block-dot proxy; our correctness = construction oracle (039133ea) + the byte-exact
  identity gate in this cell.
- **role**: L1-maturity T2-tile S6 A/B evidence. Isolates the S6 register-cliff increment from the S1
  tile by measuring PRE (S1 h-strip tile, == committed d5a28efc / cached baseline md5 9a757c02) vs POST
  (S6 min-fold+stack-panel, working-tree emitter delta md5 90d454da == build-phase s6_final.c) as separate
  board binaries against the opponent's REAL linked block-dot, with a PRE(S1)-vs-POST(S6) byte-exact `cmp`
  gate proving the timed S6 is the golden result. Harness/driver live under `tools/e2e-harness/board/`
  (`kquant_gemm_tile_s6_ab.sh` + the shared `kquant_gemm_paired_driver.c`). This cell holds data/evidence
  only; the exported .c are NOT stored (deterministically regenerable — see the objdump seal + the sibling
  `kquant-l1-q4k-q5k-repack-prefill/EXPORT_RECIPE.md`; the S6 tile is the working-tree emitter delta on
  `RVVToEmitCBlockQuantLinear.cpp` (`emitRepackKQuantGemmBodyQ4K`), uncommitted).

## durable files

- `tile_s6_findings.md`           — writeup: byte-exact gate, spill 84→23→3 (hot-loop live 23→0), peak
  vreg v30, A/B +27.9% (nr64+nr16), vs-opponent parity 1.473→1.884, S6-HOLDS verdict, residual-3
  classification, loadavg honesty, [NG-4] discipline.
- `tile_s6_ab_nr64.txt`           — raw board A/B, cold N=12, nr=64 (clean opponent anchor) + objdump seal
  + oracle + hardware residual-spill verification (objdump -d addresses).
- `tile_s6_ab_nr16.txt`           — raw board A/B, cold N=10, nr=16 (shape-robust ours-side +27.9%; opponent noise-dominated).
- `tile_s6_ab.csv`                — parsed summary (nr, variant, ours/opp GMAC/s, cv, parity, A/B, vsetvli, spill, reload, maxVreg, vwmacc).
- `objdump_tile_s6_seal.objdump`  — golden→S1→S6 vsetvli/spill/reload/maxVreg/vwmacc/text seal at clang-17 -O2 AND -O3 + residual-3 classification + mechanism.

## note — data-only cell
Data/evidence only. The load-bearing result is a **WIN (S6 HOLDS)**: the S6 min-fold + stack-panel drives
the S1 tile's residual register pressure to the ≤32-vreg cliff (hot-loop live-value spills 23→0, maxVreg
v30) **byte-exactly** (bit-identical to S1/golden on hardware), and throughput follows (+27.9% over S1;
vs-opponent parity 1.473→1.884). [NG-4] — L1 kernel datapoint, NOT a beat. Nothing here was committed /
git add-ed; main tree + build/ untouched (no rebuild — S1 is the cached baseline; S6 is the working-tree
emitter export via read-only `tcrv-opt`). Logged as S6 ALONE (not merged S1).
