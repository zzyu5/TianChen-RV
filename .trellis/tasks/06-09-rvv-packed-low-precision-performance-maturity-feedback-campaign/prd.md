# RVV packed low-precision performance-maturity feedback campaign

## Goal

Make packed low-precision RVV performance evidence truthful across the
production compiler boundary. Packed-i4 low-precision routes may remain
route-supported and executable for correctness, but no-win/regression
same-target evidence must prevent the route from being mirrored or selected as
performance-ready until later same-target measurement proves maturity.

This is a macro campaign. Gate 1 and Gate 2 are already complete in commit
`723d7eb7`; Gate 3 is complete in commit `d2af0b8f`. The current round
continues with Gate 4 by adding focused fail-closed stale/no-win/regression
evidence coverage around the same-target measurement policy and provider-owned
packed-i4 maturity mirrors. Measurement output may explain or deny performance
preference, but it must not become route support, RVV compute semantics, Common
EmitC authority, or artifact-name authority.

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
- [x] Gate 3: connect same-target measurement outcome fields to the contract as
  evidence input, not route authority.
- [x] Gate 4: add focused fail-closed tests proving regression/no-win cannot
  become performance-selected/claimed while route-supported executable
  correctness remains allowed.
- [ ] Gate 5: rerun focused artifact/correctness and same-target measurement
  checks and record whether the policy reports win, no-win, or regression
  truthfully.

## Current Round Slice

Complete Gate 4 as one coherent fail-closed coverage slice:

- Add focused coverage proving stale packed-i4 maturity mirrors cannot turn
  regression/no-win/not-measured same-target evidence into performance
  selection eligibility, dispatch preference, or a performance-win claim.
- Cover stale or contradictory measurement evidence id/classification/outcome
  family/speedup range, provider maturity outcome, selection eligibility,
  dispatch preference, claim allowance, and mirror/evidence disagreement.
- Preserve the boundary that measurement facts can deny performance preference
  or claims without denying route support, artifact generation, or correctness
  execution when provider route/artifact correctness remains valid.
- Repair the production provider/target validation seam if the new coverage
  exposes a way for artifact metadata or mirrors to overclaim performance
  maturity. If production validation is already fail-closed, keep source changes
  focused on tests and evidence policy self-tests.
- Leave Gate 5 as the continuation point unless focused artifact/correctness
  plus same-target policy rerun can be completed truthfully in this same slice.

## Completed Gate 1/2 Acceptance Criteria

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

## Gate 3 Acceptance Criteria

- [x] Same-target measurement evidence records a structured maturity-contract
  evidence-input object for packed-i4 runs.
- [x] The object carries measured classification, outcome family, speedup range,
  measurement evidence id, provider maturity evidence/outcome,
  performance-selection eligibility, dispatch preference, and performance claim
  allowance.
- [x] Regression/no-win/not-measured evidence denies performance preference or
  performance win claims without denying executable correctness support.
- [x] Reporting mirrors distinguish provider-owned contract fields from
  measurement interpretation fields.
- [x] Focused script self-tests and dry-run FileCheck coverage prove the bridge.
- [x] No Common EmitC, route id, artifact name, q4/q8/llama label, or fixture
  path becomes performance maturity authority.

## Gate 4 Acceptance Criteria

- [x] Stale provider maturity outcome cannot agree with packed-i4 regression
  evidence unless it mirrors the provider-owned regression contract exactly.
- [x] Stale performance selection eligibility cannot become `true` under the
  current regression/no-win/not-measured evidence.
- [x] Stale dispatch preference cannot become `performance-preferred` under the
  current regression/no-win/not-measured evidence.
- [x] Script self-tests prove regression, no-win, not-measured, and
  measurement-win/provider-no-win conflict states all deny performance-win
  claims while preserving `correctness_execution_allowed = true`.
- [x] Target/provider negative coverage proves stale maturity/selection/dispatch
  mirrors fail closed before artifact acceptance or provider route acceptance.
- [x] Focused dry-run/FileCheck or script evidence shows the measurement policy
  still reports provider-owned contract fields and does not move authority into
  scripts, route ids, artifact names, or Common EmitC.
- [x] Route-supported artifact generation and executable correctness allowance
  remain true for the valid packed-i4 route.

## Gate 4 Result

Completed:

- Measurement policy self-tests now validate the generated
  `performance_maturity_contract_evidence_input` against the parsed
  classification and provider-owned packed-i4 mirrors. Stale evidence id,
  classification, outcome family, speedup range, provider maturity outcome,
  selection eligibility, dispatch preference, and win-claim allowance fail
  closed in the script policy path.
- Provider C++ negative coverage now mutates packed-i4 maturity outcome and
  dispatch preference in addition to the existing feedback/selection checks.
- Target artifact C++ and packed-i4 lit negative coverage now prove stale
  performance maturity outcome and stale dispatch preference mirrors fail
  before artifact acceptance.
- A focused packed-i4 same-target dry-run under
  `artifacts/tmp/packed-i4-gate4-fail-closed-dry-run/gate4-packed-i4-fail-closed`
  preserves `performance_win_claim_allowed = false`,
  `performance_preference_denied = true`, and
  `correctness_execution_allowed = true`.

Production seam decision:

- No production provider/target semantic repair was required in this slice. The
  existing provider/resource and target artifact validators already fail closed
  on stale packed-i4 maturity mirrors; this round added the missing Gate 4
  coverage and script evidence-input integrity checks around that seam.

Remaining:

- Gate 5 remains open: rerun focused artifact/correctness plus same-target
  policy checks, and record whether the policy reports win, no-win, or
  regression truthfully.

## Out of Scope

- No new q4/q8/llama.cpp route ids, artifact names, helper semantics, or
  wrapper owners.
- No disabling executable correctness solely because performance maturity is
  false.
- No broad benchmark dashboard, tuning database, or report-only work.
- No Common EmitC invention of RVV dtype, schedule, performance, or dispatch
  semantics.
- No high-level Linalg/Vector/StableHLO frontend work.
- No real same-target measurement rerun unless Gate 3 implementation changes
  executable code or the tests require it. A dry-run or self-test evidence path
  is enough for the bridge unless a performance/runtime claim is added.

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
  -> same-target measurement script validates generated object/header mirrors
  -> maturity-contract evidence-input reporting object
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

Gate 3 does not make measurement scripts the source of truth. The script reports
whether measured classification aligns with or conflicts with the provider-owned
contract, and whether the provider-owned contract allows a performance claim.
Changing maturity fields still requires a provider/resource contract update and
new same-target evidence.

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

Gate 1/2/3/4 are complete. Continue with Gate 5: rerun focused
artifact/correctness and same-target policy checks, record whether the policy
reports win, no-win, or regression truthfully, and keep correctness/executable
support separate from performance maturity. Keep this macro task active unless
all gates are complete or human steering redirects the campaign.
