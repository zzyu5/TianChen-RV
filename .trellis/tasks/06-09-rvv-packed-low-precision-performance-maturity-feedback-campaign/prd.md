# RVV packed low-precision performance-maturity feedback campaign

## Goal

Make packed low-precision RVV performance evidence truthful across the
production compiler boundary. Packed-i4 low-precision routes may remain
route-supported and executable for correctness, but no-win/regression
same-target evidence must prevent the route from being mirrored or selected as
performance-ready until later same-target measurement proves maturity.

This is a macro campaign. The current round completes Gate 1 and Gate 2 by
introducing the provider/target validation contract for performance maturity
and wiring the current packed-i4 regression into production route/artifact
validation mirrors.

## What I Already Know

- Commit `5ffa4741` repaired packed-i4 statement planning to product-pair sum
  plus single reduction.
- Gate 6 evidence under
  `artifacts/tmp/gate6-same-target-measurement/gate6_packed_i4_product_pair_sum_same_target_measure_ssh`
  classifies the repaired packed-i4 artifact as `regression` with
  `best_speedup_range = 0.688427..0.705724`.
- Existing production code already carries provider-owned packed-i4 feedback
  facts:
  `performance_feedback`, `performance_baseline`,
  `performance_best_speedup_range`, and `performance_action`.
- The missing contract is a production maturity/selection boundary: executable
  correctness support must stay true, while performance-ready selection and
  dispatch preference must be blocked under no-win/regression feedback.

## Requirements

- Add provider-owned performance-maturity fields for packed low-precision RVV
  resource selection.
- Keep the fields owned by the RVV provider/resource and target artifact
  validators. Common EmitC must only carry provider-built route metadata.
- Preserve executable route support and target artifact generation when the
  route is correct but not performance-mature.
- Make the current packed-i4 regression visible as explicit no-win/regression
  production mirrors.
- Fail closed if target artifact metadata claims performance-ready or
  dispatch-preferred status while the provider contract says no-win/regression.
- Keep q4/q8/llama.cpp as motivation only; do not introduce q4/q8/llama named
  route ids, artifact names, or helper semantics.

## Macro Campaign Gates

- [x] Gate 1: define the production performance-maturity contract and mirror
  names for executable-but-not-performance-mature packed low-precision RVV
  routes.
- [x] Gate 2: wire the contract through RVV provider/resource facts and target
  artifact validation without moving semantics into common EmitC or scripts.
- [ ] Gate 3: connect same-target measurement outcome fields to the contract as
  evidence input, not route authority.
- [ ] Gate 4: add focused fail-closed tests proving regression/no-win cannot
  become performance-selected/claimed while route-supported executable
  correctness remains allowed.
- [ ] Gate 5: rerun focused artifact/correctness and same-target measurement
  checks and record whether the policy reports win, no-win, or regression
  truthfully.

## Current Round Slice

Complete Gate 1 and Gate 2 as one coherent production slice:

- Introduce the provider-owned maturity fields and constants for packed-i4
  no-win/regression evidence.
- Populate and verify those fields through Gearbox resource materialization,
  selected-body realization/resource planning, statement-plan owner checks,
  route metadata emission, target support bundle mapping, and target artifact
  validation.
- Update focused C++/MLIR tests so the executable route remains accepted with
  performance maturity blocked, and stale performance-ready mirrors fail
  closed.
- Leave Gate 3+ as the continuation point unless they become unavoidable for
  the production contract.

## Acceptance Criteria

- [x] Provider resource selection carries explicit performance maturity,
  maturity evidence, maturity outcome, performance selection eligibility, and
  dispatch preference fields for the packed-i4 candidate.
- [x] Target artifact validation requires those mirrors and rejects stale
  `performance_selection_eligible = "true"` /
  `dispatch_preference = "performance-preferred"` claims under the current
  no-win/regression evidence.
- [x] Existing packed-i4 executable artifact tests still pass with route support
  preserved.
- [x] Focused tests distinguish executable support from performance maturity.
- [x] No Common EmitC or script-only semantic authority is introduced.
- [x] Focused build/tests pass:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`, and relevant dry-run or
  self-test coverage if scripts change.
- [x] Bounded old-authority scan over touched diff lines shows no new positive
  legacy i32/source-front-door/descriptor authority.

## Out of Scope

- No new q4/q8/llama.cpp route ids, artifact names, helper semantics, or
  wrapper owners.
- No disabling executable correctness solely because performance maturity is
  false.
- No broad benchmark dashboard, tuning database, or report-only work.
- No Common EmitC invention of RVV dtype, schedule, performance, or dispatch
  semantics.
- No high-level Linalg/Vector/StableHLO frontend work.
- No new same-target measurement run unless Gate 1/2 implementation changes
  executable code or the tests require it.

## Technical Approach

Use the existing low-precision resource selection flow as the contract owner:

```text
Gearbox/resource attrs
  -> selected-body realization / route-family plan
  -> RVVLowPrecisionContractionResourceSelection
  -> statement-plan owner validation
  -> provider-built route metadata
  -> target support bundle mirrors
  -> target route-family artifact validation
```

The maturity contract should use explicit mirror names under
`tcrv_rvv.low_precision_resource.*`, for example:

- `performance_maturity`
- `performance_maturity_evidence`
- `performance_maturity_outcome`
- `performance_selection_eligible`
- `dispatch_preference`

For the current packed-i4 evidence, the expected values are no-win/regression
and not eligible for performance selection or dispatch preference. The route
continues to be `legal` and provider-supported for executable correctness.

## Decision (ADR-lite)

Context: The previous packed-i4 repair changed production code but same-target
measurement still showed a regression. Treating that route as performance-ready
would overstate the production compiler result.

Decision: Represent performance maturity as RVV provider-owned resource facts
and target artifact mirrors, separate from executable route legality.

Consequences: The route can remain correct and exportable, while performance
selection and dispatch preference fail closed until later same-target evidence
changes the provider-owned maturity contract.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous archive:
  `.trellis/tasks/archive/2026-06/06-09-rvv-production-kernel-packed-low-precision-performance-repair-campaign/`.
- Evidence input:
  `artifacts/tmp/gate6-same-target-measurement/gate6_packed_i4_product_pair_sum_same_target_measure_ssh`.

## Continuation Point

Gate 1/2 are complete for this slice. The macro task remains active. Continue
with Gate 3: connect same-target measurement outcome fields to the maturity
contract as evidence input, not route authority, then finish Gate 4/5 with
focused stale-evidence and same-target policy checks. Keep this macro task
active unless all gates are complete or human steering redirects the campaign.
