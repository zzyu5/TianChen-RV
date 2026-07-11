# cell MANIFEST — ondevice-q8_0

- **campaign**: perf-historical
- **status**: ARCHIVE
- **role**: q8_0 flat block-dot dual-board bit-exact + micro (host_k1 / host_rvv).
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `evidence.json`
- `host_k1/run_rv64gc.txt`
- `host_k1/run_rv64gcv.txt`
- `host_k1/target_profile.txt`
- `host_rvv/run_rv64gc.txt`
- `host_rvv/run_rv64gcv.txt`
- `host_rvv/target_profile.txt`
- `results_summary.csv`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
