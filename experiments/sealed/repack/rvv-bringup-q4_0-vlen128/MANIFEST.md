# cell MANIFEST — rvv-bringup-q4_0-vlen128

- **campaign**: repack
- **status**: SEALED (F1 bring-up)
- **role**: rvv/VLEN128 q4_0 e2e bring-up (preflight/correctness/phase-split/aggregate).
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `aggregate.txt`
- `correctness.txt`
- `evidence.json`
- `NOTES.md`
- `phase_split_raw.txt`
- `preflight.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
