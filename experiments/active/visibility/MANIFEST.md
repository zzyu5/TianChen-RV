# cell MANIFEST — visibility

- **campaign**: visibility
- **status**: ACTIVE (CI-regenerated)
- **role**: T0 sixstate / T7 burndown / T2 ledger-anchor auto-generated pack. Generators + drift-check live in tools/visibility/ (hardcode this path — see MOVES.md STAGE2 flag).
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `T0-sixstate.md`
- `T2-ledger-anchor.md`
- `T7-burndown.md`
- `T7-three-curve-G3-closure.md`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
