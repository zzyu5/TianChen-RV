# cell MANIFEST — silicon-validation-batch-1

- **campaign**: silicon
- **status**: SEALED (4 constructed格 silicon_validated)
- **role**: iq4_nl/iq1_s/iq1_m BIT_EXACT ULP=0 + q4_0_repack FMA-fold bounded-ULP; board rvv/VLEN128, no-FMA left-assoc ggml scalar oracle.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `evidence.json`
- `kernels/iq1_m.kernel.c`
- `kernels/iq1_s.kernel.c`
- `kernels/iq4_nl.kernel.c`
- `kernels/q4_0_repack.kernel.c`
- `NOTES.md`
- `results/iq1_m/evidence.json`
- `results/iq1_m/run_rvv.txt`
- `results/iq1_s/evidence.json`
- `results/iq1_s/run_rvv.txt`
- `results/iq4_nl/evidence.json`
- `results/iq4_nl/run_rvv.txt`
- `results/q4_0_repack/evidence.json`
- `results/q4_0_repack/run_rvv.txt`
- `target_profile.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
