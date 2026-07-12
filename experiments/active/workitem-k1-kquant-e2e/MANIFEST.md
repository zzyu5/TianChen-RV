# cell MANIFEST — workitem-k1-kquant-e2e

- **campaign**: 测量总攻 / Line C reopen — **[WORK-ITEM-K1-KQUANT-E2E]** (k1-clang q4_K e2e: gcc-codegen 候选因素[冠名待二.2 出口A·裁二.1 锁定] vs
  weight-reconstruction-bound). Reopen trigger of `experiments/active/line-c-k1-strike/q5_K_X0_amdahl_verdict.md`.
- **status**: DELIVERED. Board = `ssh k1` (SpacemiT X60 / VLEN256 / 8 harts / stock clang-18 / DVFS perf-gov 1.6GHz).
- **role**: decisive thesis measurement — q4_K **repack (stock as-shipped)** vs **vec_dot (forced fallback)** e2e on
  **k1-clang**, the compiler-symmetric analog of the rvv-gcc M2-q4_K 0.42× (repack-vs-vecdot). Resolves whether the
  rvv K-quant e2e LOSS was **gcc-742-spill-death** (rvv-specific) or **intrinsic weight-reconstruction overhead**.
- **discipline**: NO git · board reversible (shared source md5 3cac40aa preserved · live stock lib untouched ·
  dedicated build `/data/build-k1-workitem`) · correctness-first · kernel==system (clang-18 compiler-symmetric).

## 双账本 disclosure
k1 kernel-axis compiler = clang-18; system/deploy = clang-18 ⇒ CONVERGE (both clang-18; no asymmetry artifact).
Material difference from rvv (stock = gcc-15, the [CASE-COMPILER-ASYMMETRY] axis).

## durable files (this cell)
- `evidence.md`
- `MANIFEST.md`
- `.gitignore`
- `raw_correctness.txt`
- `raw_phase_split.txt`
- `phase_split_analysis.txt`
- `raw_seal.txt`

## board harness (touch-set) — `tools/e2e-harness/board/workitem-k1-kquant-e2e/`
- `build_repack_vs_vecdot.sh`
- `correctness_ab.sh`
- `phase_split_ab.sh`
- `analyze_phase_split.py`
- `seal_and_manifest.sh`

## board-resident artifacts (gitignored; md5 recorded)
- `libggml-cpu.so.REPACK`
- `libggml-cpu.so.VECDOT`
