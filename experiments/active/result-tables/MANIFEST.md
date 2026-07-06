# cell MANIFEST — result-tables

- **campaign**: cross-cutting result tables
- **status**: ACTIVE (filled measurement tables; CI-refreshed)
- **role**: T-N noise floor + T3_A/T3_B dual-board measurements + T8 win/loss gap ledger. Empty zero-value template CSVs live in experiments/_templates/.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `T3_A_board_A_rvv1.0_vlen128.csv`
- `T3_B_board_B_rvv1.0_vlen256.csv`
- `T8_winloss_gap_ledger.csv`
- `T-N_noise_floor.csv`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
