# cell MANIFEST — rvv-vlen128-q4_0-repack-genroute

- **campaign**: repack
- **status**: SEALED
- **role**: q4_0 repack generation-vs-routing phase split + constructed-GEVM redeploy (baseline + redeploy evidence).
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `aggregate_baseline.txt`
- `evidence_baseline.json`
- `evidence.json`
- `evidence_redeploy.json`
- `NOTES.md`
- `phase_split_raw.txt`
- `redeploy_aggregate.txt`
- `redeploy_correctness.txt`
- `redeploy_phase_split_raw.txt`
- `target_profile.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
