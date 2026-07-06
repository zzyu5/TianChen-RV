# cell MANIFEST — T3_step3

- **campaign**: perf-historical
- **status**: ARCHIVE
- **role**: q8_0 m1/m2 dual-board evidence bundle (T3_A/T3_B q8_0 rows): emitc + factory + raw + objdump seals. Local *.o ignored via in-cell .gitignore.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `aggregate_summary.txt`
- `fold_isolation_k1.txt`
- `.gitignore`
- `k1_ab_raw.txt`
- `k1_rdir.txt`
- `kernel_factory.c`
- `kernel_m1.cpp`
- `kernel_m1.emitc.mlir`
- `kernel_m2.cpp`
- `kernel_m2.emitc.mlir`
- `objdump_seals_local.txt`
- `rvv_ab_raw.txt`
- `rvv_rdir.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
