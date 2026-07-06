# cell MANIFEST — quant-label-proof

- **campaign**: c1-cleanliness
- **status**: SEALED (quant-fix F22)
- **role**: string-deletion byte-exact proof (fact-drives-routing, label-inert); NOTES sha256 a457e1b9 + machine evidence.json.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `evidence.json`
- `NOTES.md`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
