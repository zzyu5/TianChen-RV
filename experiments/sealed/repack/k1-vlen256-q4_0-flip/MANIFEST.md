# cell MANIFEST — k1-vlen256-q4_0-flip

- **campaign**: repack
- **status**: SEALED (F5)
- **role**: k1/VLEN256 q4_0 flip e2e; decode 0.857x PARK-P1 reversal source.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `aggregate.txt`
- `correctness.txt`
- `evidence.json`
- `phase_split_raw.txt`
- `preflight.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
