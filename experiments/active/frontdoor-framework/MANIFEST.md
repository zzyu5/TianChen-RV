# cell MANIFEST — frontdoor-framework

> **贡献归属(2026-07-10 正名)**: this ledger is **C3′ (capability-keyed pattern library) construction economics / mature-compiler coverage axis**, measuring the cost to front-door-ize a **decode FORMAT** (q4_K/ternary/iq*…). It is **NOT C2** — C2 = the per-**extension-family** (RVV/IME/scalar/zvfh) onboarding cost, anchored at IME ≈2484 LOC (`experiments/active/visibility/T2-ledger-anchor.md`), curve still MISSING (pending X-SCALAR). Decode formats are `∩ rvv.* ≠ ∅` so by core-invariants [F-6] they are NOT independent families and do NOT enter the C2 denominator. **Numbers unchanged; only the contribution label + "family" wording re-named (decode FORMAT vs extension FAMILY).**

- **campaign**: frontdoor-framework (C3′ front-door-ization construction-cost ledger — per-decode-format)
- **status**: ACTIVE (C3′ front-door-ization construction ledger; first point tq2_0/ternary landed, recompute-anchored to commit 7a4250c5)
- **role**: NEW C3′ construction-cost evidence line: the LOC/labor cost to lift a decode FORMAT from a DIRECT-EMIT bypass to a front-door typed-region CONSTRUCTION, split into reusable-framework LOC (paid once) vs format-specific LOC (paid per format, shrinks for close siblings). Distinct from the T2 `net_new_vs_reuse` construction ledger (which measures monolith→constructed inside the block-dot / super-block / grid primitives); this ledger measures the front-door-CONSTRUCTION apparatus generalizing to accept a NEW decode FORMAT. Machine-anchored: `git show 7a4250c5 --numstat -- lib/ include/`; component split derived in NOTES.md.

## why a separate ledger (not a T2 row)

- T2 (`T2_C2_ledger_marginal_cost.csv`) records **monolith/dispatch-wired → constructed** marginal cost *inside* a primitive family (flat block-dot, super-block K-quant, IQ grid, repack loop-shape). Its rows are keyed to a `ΔC_construct` step on the six-state lattice.
- This cell records a DIFFERENT axis (**C3′ construction cost**): the cost to make the shared **front-door construction apparatus** (`RVVLowerQuantContraction.cpp` `lowerToRepackGem{v,m}*` + `GgmlQuantContractionOp::verify` + the typed loop-body verifiers) accept a **second decode FORMAT** (q4_0 nibble → ternary tq2_0 trit). The load-bearing C3′ claim is the **framework/format split**: the reusable-framework LOC is paid ONCE and the next decode FORMAT (tq1_0) re-pays only its format-specific decode leaf.
- The two ledgers cross-reference the SAME flip commits but answer different C3′-construction questions (primitive-internal construction cost vs front-door-apparatus generalization cost).

## durable files

- `frontdoor_framework_ledger.csv` — the per-family front-door-ization ledger (one row per family; first row = tq2_0/ternary).
- `NOTES.md` — method (the framework-vs-format-specific decomposition rule), the tq2_0 first-point component breakdown, the tq1_0 C3′ construction-cost prediction, and the cross-reference to the adversarial-verify interception discipline note (result-tables T8 口径).

## scope / honesty

- **LOC only; labor-hours NOT instrumented.** A `labor_proxy_passes` column records the number of authoring passes as a coarse effort proxy (tq2_0 = 2: an emission-only first pass was rejected by adversarial-verify, then a real construction pass landed — see the T8 口径 discipline note + NOTES.md).
- Data-only cell (CSV + MD). No code artifacts, no harness. Untracked (not committed) — rides the durable lens for dir-lint.
