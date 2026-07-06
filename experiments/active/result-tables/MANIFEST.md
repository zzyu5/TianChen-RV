# cell MANIFEST — result-tables

- **campaign**: cross-cutting result tables
- **status**: ACTIVE (filled measurement tables; CI-refreshed)
- **role**: T-N noise floor + T3_A/T3_B dual-board measurements + T8 win/loss gap ledger + T1 C1 structural-conjunction rows + T2 C2 marginal-cost curve. T1/T2 are HEAD-derived (six-state schema read via `git show HEAD:…` — never the working tree — + docs/method/C2 ledger + git-log diffstat); the F-1 machine-check is `tools/lint/check_construction_manifest_regex.py`. Empty zero-value template CSVs live in experiments/_templates/.
- **layout**: org STAGE1 (2026-07-06). See experiments/archive/MOVES.md for old→new path map.

## durable files (git-tracked + untracked-not-ignored in this cell)

- `T3_A_board_A_rvv1.0_vlen128.csv`
- `T3_B_board_B_rvv1.0_vlen256.csv`
- `T8_winloss_gap_ledger.csv`
- `T-N_noise_floor.csv`
- `T1_C1_structural_conjunction.csv`
- `T2_C2_ledger_marginal_cost.csv`

## notes — T1/T2 口径

- **T1** (C1 结构合取): each row = one `constructed` (STRONG) instance × the four conjuncts {lives-in-IR (dialect op) ∧ verifier-can-reject ([L-8] gate) ∧ pass-can-transform (typed lowering) ∧ provenance-traceable (anchor + machine auto_readout)}. All facts derive from the HEAD six-state schema; `conjunction_holds=TRUE` means all four co-exist for that instance. Rows #1/#2 = the flat and super-block family *primitive founders* (q8_0, q4_K).
- **T2** (C2 边际成本): seq-0 is the recomputable `[LED-1]` family anchor (IME raw wc-l 2484, reproduced by `tools/visibility/recompute_ledger_anchor.sh`). Rows 1..N are per-flip marginal cost. Two LOC columns: `ledger_delta_hand_LOC` = the docs/method/C2 ledger's authoritative per-flip Δ (a **milestone-chain sum** for multi-commit flips like q4_K/q6_K/q2_K); `gitcheck_flip_commit_net_LOC` = my single-flip-commit `git show --numstat -- lib/ include/` net (reproducible cross-check; diverges from the ledger when the flip spans a milestone chain). **`cost_tier` is the capability judgment (net-new vs reuse), NOT the LOC sign** — e.g. iq1_m gitchecks +327 (a body code-move) yet is LOW-reuse (the iq1_s grid scaffold was reused). The flat retire (-1785 gitcheck vs -1797 ledger) differs by test/CMake lines outside lib+include.

> Evidence-pointer code (`*.kernel.c` / `*.emitc.mlir` / sealed `*.o` / `*.cpp`) stays in-cell as an evidence pointer; harness/protocol scripts live under `tools/`. Gitignored scratch rides with the cell but is not durable.
