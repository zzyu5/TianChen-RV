# cell MANIFEST — silicon-validation-gemm

- **campaign**: silicon
- **status**: SEALED (GEMM debt 1->0)
- **role**: constructed q4_0 REPACK GEMM (PREFILL, M>1) FMA-fold bounded-ULP; board rvv/VLEN128 + f64 ref.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `kernels/q4_0_repack_gemm.kernel.c`
- `NOTES.md`
- `results/q4_0_repack_gemm/evidence.json`
- `results/q4_0_repack_gemm/run_rvv.txt`
- `target_profile.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
