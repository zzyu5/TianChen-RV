# cell MANIFEST — rvv-vlen128-q4_0-gemm-constructed-sealed

> ★M2c min-term-bug 隔离确认(2026-07-09 G3-minterm-fix 裁决一.2):本 cell kernel(q4_0 flat repack GEMM)
> grep 无 `kquant_dmin_bsums_min` / dmin·bsums-min fold —— q4_0 是 flat 格、无 K-quant min fold。故与 M2c
> q4_K/q5_K/q2_K repack-GEMM min-term VLEN128 bug(commit 53666846)**无共享路径、不受影响**;5.9× 数值不动
> (成色仍按既有 routing 归因,见 `docs/reports/2026-07-09-q4_0-5.9x-routing-attribution-correction.md`)。

- **campaign**: repack
- **status**: SEALED (D-board return finale)
- **role**: q4_0 REPACK GEMM prefill ~5.9x SEALED + decode P1; step0 restore + noisefloor + constructed + objdump fingerprint + bandwidth analysis.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `bandwidth_analysis.txt`
- `constructed_evidence.json`
- `constructed_raw.txt`
- `noisefloor_evidence.json`
- `noisefloor_raw.txt`
- `NOTES.md`
- `objdump_fingerprint.txt`
- `restore_verify.txt`
- `target_profile.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.

## 5.92× ↔ 5.08× reconciliation (org STAGE2 §6.1)

Cross-read of this cell vs `experiments/sealed/repack/rvv-SEALED-q4_0-vlen128-fullmarch/`
(the SEALED **5.077×** row) on the four comparability axes:

| axis | this cell (5.92×) | rvv-SEALED (5.08×) | verdict |
|---|---|---|---|
| board / target_profile | rvv-openeuler-vlen128, `ssh rvv`, kernel 6.12.66, VLEN128, cores 8–11 ×4t, 2.6 GHz, DVFS 0.00% | identical | **SAME** |
| march (A==B symmetric) | `rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause`, gcc-15.2.0 | identical string + compiler | **SAME** |
| opponent (B_stock) | unpatched upstream ggml **block-dot** (git f3e1828) | unpatched upstream ggml block-dot | **SAME** |
| paired protocol | interleaved same-session A/B, PASSES=2 REPS=5, greedy 3/3, model `tinyllama-q4_0.gguf` sha256 `da3087fb14aede55` | same paired 2-pass protocol + same model sha | **SAME** |

**SAME spine → the `同` branch applies:** `headline = 5.92× (constructed era); 5.08× = the
routing / hand-op era row; +Δ attribution needs disassembly.` The two are quotable each on this
one comparability spine but **NOT additively**, and the delta is not a clean codegen win:

- **The A_ours artifact differs by era.** 5.08× = capability-**routing** engages ggml's *own*
  tiled repack GEMM (adversarial-verify: the prefill kernel is byte-identical in A and B — the win
  is dispatch, decode GEVM is ours). 5.92× = our compiler **front-door-CONSTRUCTED** GEMM
  (`be66c917 typed_repack_gemm_loop_body`), a genuinely ours-emitted prefill kernel.
- **The absolute baselines shifted across sessions** (stock 3.903 → 5.237 t/s; ours 19.819 → 31.0
  t/s), and this session ran under a **SPEC CPU2006 bzip2 co-tenant** (see `target_profile.txt`).
  The paired interleaved protocol protects the *ratio* but not the absolute t/s, so the ratio
  delta 5.08 → 5.92 is session/co-tenant-confounded, **not** a demonstrated pure-codegen gain.
- **Kernel level, the two eras agree:** this cell's own byte-exact kernel estimate is **5.045×**
  (NOTES), essentially equal to rvv-SEALED's e2e **5.077×**; the cell attributes the +Δ up to
  5.92× to **e2e memory-locality amplification**. Confirming that +Δ is denser-MAC codegen rather
  than co-tenant/session drift is exactly the disassembly step — instrument = this cell's
  `objdump_fingerprint.txt` (mattr-decoded), with the stock-baseline drift netted out.
