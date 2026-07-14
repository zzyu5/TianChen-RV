# cell MANIFEST — l1-reroll-q4k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (RE-ROLL 曳光弹 #2 — q4_K repack GEMM structural RE-ROLL
  A/B on rvv/VLEN128: does turning the full-unroll k/ii dims into runtime for-loops convert the
  register-pressure "opening" into throughput? Sibling to `l1-pipeline-q4k-repack-gemm`, which
  already falsified the byte-exact reorder lever.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ CRUX: the RE-ROLL (k-chunk[2] + ii-nibble[16] dims of the 32-elem sub-block dot -> runtime
  `emitc.for`; byte-exact-identical, oracle GREEN) **REGRESSES**, refuting the diagnosis model.
  (1) objdump at the measured clang-17 -O2 board form: vsetvli **53 -> 76 (+43%)**, spill **84 -> 118
  (+40%)**, reload **88 -> 127** — vsetvli+spill did NOT drop, they ROSE (text halved 23486->10220,
  static vwmacc 2240->320, but the register-pressure metric moved the WRONG way). (2) A/B
  ours_post/ours_pre paired cold (nr=64 N=12 / nr=16 N=10): **0.893 / 0.892 = a consistent ~-11%
  REGRESSION**, ~50x our own run-to-run cv (real, not noise; contrast #1's +0.65% wash). (3)
  vs-opponent block-dot (clean anchor nr=64): PRE parity **0.962x** (reproduces the precedent's
  0.94-0.97x band -> harness validated) -> POST parity **0.860x** — parity got ~10.6% WORSE, still
  < 1.0, **NOT a win** (a DEEPER loss). **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked; single
  core; opponent = single-thread block-dot proxy; our correctness = construction oracle (039133ea) +
  the byte-exact identity gate in this cell. The full-unroll register-pressure opening is STRUCTURAL
  in the [col][half]=16-accumulator fan-out, which rolling k/ii does not shrink; only column blocking
  would, but that forfeits the once-per-16-weight decode amortization. Both cheap levers (#1 reorder,
  #2 re-roll) are now falsified on silicon.
- **role**: L1-maturity RE-ROLL A/B evidence. Isolates the non-byte-exact structural re-roll from a
  genuine schedule win by measuring PRE (full-unroll golden, == cached baseline md5 b0b5beac) vs POST
  (k/ii re-rolled) as separate board binaries against the opponent's REAL linked block-dot, with a
  PRE-vs-POST byte-exact `cmp` gate proving the timed POST is the golden result. Harness/driver live
  under `tools/e2e-harness/board/` (`kquant_gemm_reroll_ab.sh` + the precedent's paired driver). This
  cell holds data/evidence only; the exported .c / re-rolled .c are NOT stored (deterministically
  regenerable — see the objdump seal + the sibling `kquant-l1-q4k-q5k-repack-prefill/EXPORT_RECIPE.md`;
  the re-roll is the working-tree emitter delta on RVVToEmitCBlockQuantLinear.cpp, uncommitted).

## durable files

- `reroll_findings.md`           — writeup: byte-exact gate, vsetvli+spill-RISE, A/B -11%, vs-opponent
  parity 0.962->0.860, structural-opening verdict (regresses), [NG-4] discipline.
- `reroll_ab_nr64.txt`           — raw board A/B, cold N=12, nr=64 (clean opponent anchor) + objdump + oracle.
- `reroll_ab_nr16.txt`           — raw board A/B, cold N=10, nr=16 (ours-side A/B reproduced; opponent noise-dominated).
- `reroll_ab.csv`                — parsed summary (nr, variant, ours/opp GMAC/s, cv, parity, A/B, vsetvli, spill).
- `objdump_reroll_seal.objdump`  — PRE vs POST vsetvli/spill/reload/vwmacc/text seal at clang-17 -O2 AND -O3 + mechanism.

## note — data-only cell
Data/evidence only. The load-bearing result is a **NEGATIVE**: the structural RE-ROLL does not relieve
the register-pressure opening — vsetvli rises 53->76, spill rises 84->118, board throughput regresses
~11%, and vs-opponent parity worsens 0.962x -> 0.860x (still < 1.0, NOT a win). [NG-4] — L1 datapoint,
not a beat. Nothing here was committed / git add-ed; main tree + build/ untouched (no rebuild — PRE is
the cached golden baseline, POST is the pre-existing exported re-roll .c).
