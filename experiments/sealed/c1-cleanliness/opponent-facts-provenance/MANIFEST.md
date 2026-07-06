# cell MANIFEST — opponent-facts-provenance

- **campaign**: c1-cleanliness
- **status**: SEALED (CI-consumed pin)
- **role**: opponent-fact provenance -> pinned ggml line anchors. opponent-facts.pin.json is consumed by tools/lint/check_opponent_facts_pin.sh (pin change -> STALE).
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `NOTES.md`
- `opponent-facts.pin.json`

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer (data cell may hold registered evidence-code; harness/protocol scripts live under `tools/`). Gitignored scratch (build `.o`/`.cpp`/`.err`/`.sh`) rides with the cell but is not durable.
