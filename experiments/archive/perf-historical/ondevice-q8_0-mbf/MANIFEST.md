# cell MANIFEST — ondevice-q8_0-mbf

- **campaign**: perf-historical
- **status**: ARCHIVE
- **role**: q8_0 mbf1/mbf2 (e8m1) fold-segment objdump seal + kernel_core_mbf2.o / kernel_q8_mbf1.o / kernel_ggml_factory.o vsetvli-diagnosis .o.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `kernel_core_mbf2.o`
- `kernel_ggml_factory.o`
- `kernel_q8_mbf1.o`
- `seal/fold_segment_objdump.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
