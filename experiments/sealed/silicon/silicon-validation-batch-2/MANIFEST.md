# cell MANIFEST — silicon-validation-batch-2

- **campaign**: silicon
- **status**: SEALED (silicon debt 2->0)
- **role**: iq3_xxs + iq2_xxs super-block-grid vec_dot BIT_EXACT ULP=0; board rvv/VLEN128.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `kernels/iq2_xxs.kernel.c`
- `kernels/iq3_xxs.kernel.c`
- `NOTES.md`
- `results/iq2_xxs/evidence.json`
- `results/iq2_xxs/run_rvv.txt`
- `results/iq3_xxs/evidence.json`
- `results/iq3_xxs/run_rvv.txt`
- `target_profile.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
