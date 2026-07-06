# cell MANIFEST — ondevice-q5_K

- **campaign**: perf-historical
- **status**: ARCHIVE (banked board-split; some STALE)
- **role**: q5_K deferred-reduce vs aux8-roundtrip dual-board split (k1/VLEN256 win / rvv/VLEN128 loss), bit-exact; perf-char #4 decode-width evidence. Referenced by schema/pattern-registry.v1.json (dangling — MOVES.md STAGE2 flag).
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `k1_ab_p3_regres_raw.txt`
- `k1_ab_raw.txt`
- `kernel_factory.c`
- `kernel_ours.emitc.mlir`
- `rvv_ab_p3_regres_raw.txt`
- `rvv_ab_raw.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
