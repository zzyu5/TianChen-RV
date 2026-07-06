# cell MANIFEST — e2e-harness (index)

- **campaign**: e2e-harness
- **status**: ACTIVE (harness data-side index)
- **role**: README (result-cell schema + Win ladder) + models.manifest.csv. Protocol/driver scripts live in tools/e2e-harness/; sealed result cells moved to experiments/sealed/repack/.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `models.manifest.csv`
- `README.md`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
