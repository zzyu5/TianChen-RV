# cell MANIFEST — frontdoor-framework

- **campaign**: frontdoor-framework (C2 layer-B — per-family front-door-ization cost)
- **status**: ACTIVE (C2 layer-B ledger; first point tq2_0/ternary landed, recompute-anchored to commit 7a4250c5)
- **role**: NEW C2 evidence line ("layer B"): the LOC/labor cost to lift a decode family from a DIRECT-EMIT bypass to a front-door typed-region CONSTRUCTION, split into reusable-framework LOC (paid once) vs family-specific LOC (paid per family, shrinks for close siblings). Distinct from the T2 `net_new_vs_reuse` construction ledger (which measures monolith→constructed inside the block-dot / super-block / grid primitives); this ledger measures the front-door-CONSTRUCTION apparatus generalizing to accept a NEW decode family. Machine-anchored: `git show 7a4250c5 --numstat -- lib/ include/`; component split derived in NOTES.md.

## why a separate ledger (not a T2 row)

- T2 (`T2_C2_ledger_marginal_cost.csv`) records **monolith/dispatch-wired → constructed** marginal cost *inside* a primitive family (flat block-dot, super-block K-quant, IQ grid, repack loop-shape). Its rows are keyed to a `ΔC_construct` step on the six-state lattice.
- This cell records a DIFFERENT axis (**C2 layer B**): the cost to make the shared **front-door construction apparatus** (`RVVLowerQuantContraction.cpp` `lowerToRepackGem{v,m}*` + `GgmlQuantContractionOp::verify` + the typed loop-body verifiers) accept a **second decode family** (q4_0 nibble → ternary tq2_0 trit). The load-bearing C2 claim is the **framework/family split**: the reusable-framework LOC is paid ONCE and the next decode family (tq1_0) re-pays only its family-specific decode leaf.
- The two ledgers cross-reference the SAME flip commits but answer different C2 questions (primitive-internal marginal cost vs front-door-apparatus generalization cost).

## durable files

- `frontdoor_framework_ledger.csv` — the per-family front-door-ization ledger (one row per family; first row = tq2_0/ternary).
- `NOTES.md` — method (the framework-vs-family-specific decomposition rule), the tq2_0 first-point component breakdown, the tq1_0 C2 prediction, and the cross-reference to the adversarial-verify interception discipline note (result-tables T8 口径).

## scope / honesty

- **LOC only; labor-hours NOT instrumented.** A `labor_proxy_passes` column records the number of authoring passes as a coarse effort proxy (tq2_0 = 2: an emission-only first pass was rejected by adversarial-verify, then a real construction pass landed — see the T8 口径 discipline note + NOTES.md).
- Data-only cell (CSV + MD). No code artifacts, no harness. Untracked (not committed) — rides the durable lens for dir-lint.
