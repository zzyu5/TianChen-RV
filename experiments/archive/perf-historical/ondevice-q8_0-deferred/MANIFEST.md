# cell MANIFEST — ondevice-q8_0-deferred

- **campaign**: perf-historical
- **status**: ARCHIVE (some STALE)
- **role**: q8_0 deferred-fold P2c fair perf (perf_rvv_vlen128_FAIR.csv, referenced by schema/pattern-registry.v1.json) + A_deferred.o vsetvli-diagnosis seal.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `A_deferred.rv64gcv_zvfhmin.objdump`
- `fair/perf_rvv_vlen128_FAIR.csv`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
