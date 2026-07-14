# cell MANIFEST — l1-pipeline-q4k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (pipeline — q4_K repack GEMM byte-exact schedule-rearrangement
  A/B on rvv/VLEN128: does the full-unroll register-pressure "opening" convert to throughput?)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ CRUX: the "pipelining" tracer bullet (byte-exact decode-hoist reorder of the 16-ii nibble decode)
  does **NOT** convert the structural register-pressure opening into throughput. (1) objdump: vsetvli
  **52 -> 52** at the actual board form (clang-17 -O2, loops rolled) and **363 -> 363** with spill
  **591 -> 591** at the build-report -O3 full-unroll form — NO reduction at any opt level; the reorder
  only shrinks code layout (-O3 text 99134 -> 96334, -2.8%). (2) A/B ours_post/ours_pre paired cold
  (nr=64 N=12 / nr=16 N=10): **+0.67% / +0.63%** — consistent but negligible, at the board noise floor
  (opponent cv 0.71%), tracking the smaller layout, NOT any vsetvli/spill win. (3) vs-opponent
  block-dot (clean anchor nr=64): PRE parity **0.962x** (reproduces the precedent 0.94-0.97x band ->
  harness validated) -> POST parity **0.970x** — nudged ~+0.8% but **still < 1.0 (parity/loss), NOT a
  win**; the pipelining did not flip parity. **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked;
  single core; opponent = single-thread block-dot proxy; our correctness = construction oracle
  (039133ea), not re-verified here. The register-pressure opening is only reachable by a NON-byte-exact
  RE-ROLL (unrolled superblock -> runtime for-loops so clang hoists the invariant vsetvli + reuses
  accumulators), logged as the real follow-up lever.
- **role**: L1-maturity pipeline A/B evidence. Isolates the "pipelining" (byte-exact reorder) from a
  genuine schedule win by measuring PRE (golden lowering, == cached baseline) vs POST (decode-hoist
  lowering) as separate board binaries against the opponent's REAL linked block-dot. Both emitc MLIRs
  saved from the build phase; harness/driver live under `tools/e2e-harness/board/`
  (`kquant_gemm_pipeline_ab.sh` + `kquant_gemm_paired_driver.c`). This cell holds data/evidence only;
  the 1.3 MB exported .kernel.c are NOT stored (deterministically regenerable — see the objdump seal +
  the sibling `kquant-l1-q4k-q5k-repack-prefill/EXPORT_RECIPE.md`).

## durable files

- `pipeline_findings.md`          — writeup: vsetvli-no-drop, A/B, vs-opponent parity, structural-opening
  verdict, T-N noise floor, [NG-4] discipline, RE-ROLL follow-up lever.
- `pipeline_ab_nr64.txt`          — raw board A/B, cold N=12, nr=64 (clean opponent anchor) + board objdump lines.
- `pipeline_ab_nr16.txt`          — raw board A/B, cold N=10, nr=16 (ours-side A/B reproduced; opponent noise-dominated).
- `pipeline_ab.csv`               — parsed summary (nr, variant, ours/opp GMAC/s, parity, paired A/B, vsetvli).
- `objdump_pipeline_seal.objdump` — PRE vs POST vsetvli/spill seal at board clang-17 -O2 AND clang-20 -O3.

## note — data-only cell
Data/evidence only. The load-bearing result is a **NULL**: the byte-exact "pipelining" does not move
vsetvli/spill and moves board throughput only ~+0.65% (noise floor); parity vs opponent stays < 1.0.
[NG-4] — L1 pipeline datapoint, not a beat. Nothing here was committed / git add-ed.
