# cell MANIFEST — rvv-SEALED-q4_0-vlen128-fullmarch

- **campaign**: repack
- **status**: SEALED (immutable evidence; F4)
- **role**: rvv/VLEN128 q4_0 SEALED fullmarch e2e — decode light/medium/heavy + phase-split + objdump mechanism seal.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `correctness.txt`
- `decode_heavy.txt`
- `decode_light.txt`
- `decode_medium.txt`
- `evidence_decode_heavy.json`
- `evidence_decode_light.json`
- `evidence_decode_medium.json`
- `evidence.json`
- `evidence_phasesplit.json`
- `NOTES.md`
- `objdump_mechanism_fixed.txt`
- `objdump_seal.txt`
- `phase_split_raw.txt`
- `preflight.txt`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
