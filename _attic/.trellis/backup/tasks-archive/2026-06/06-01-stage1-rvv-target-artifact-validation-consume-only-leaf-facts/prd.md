# Stage1 RVV target artifact validation consume-only leaf facts

## Goal

Make RVV target artifact route-family validation a consume-only boundary for
provider-derived route/type/leaf facts. Target/export validation may check that
the rebuilt `TCRVEmitCLowerableRoute`, target artifact metadata mirrors, and
selected route description agree, but it must not locally choose RVV intrinsic
or vector type spellings from SEW, LMUL, operation kind, route ids, artifact
names, or legacy i32 helper surfaces.

## What I Already Know

* The Hermes Direction Brief is the task source.
* The repository started this round on `main` with a clean worktree at
  `508ab110 rvv: derive shared route profile leaves`.
* No active `.trellis/.current-task` existed; this task was created as
  `.trellis/tasks/06-01-stage1-rvv-target-artifact-validation-consume-only-leaf-facts`.
* Specs require selected RVV routes to flow through typed low-level
  `tcrv_rvv` body/config facts, RVV plugin-owned route/provider planning,
  common EmitC materialization, then target artifact export as a consumer of
  materialized EmitC plus provider route mirrors.
* The previous archived task moved shared RVV route-profile leaves to typed
  provider derivation. Target artifact validation is now the remaining boundary
  that can still locally compute exact RVV leaves.
* Focused inspection of
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` shows active
  exact leaf helpers/constants in the widening-dot contraction artifact path
  and standalone/runtime-scalar standalone reduction artifact paths.
* Existing target validation already has a useful shape for consume-only
  checks: it validates route headers, type mappings, ABI mappings, route
  statement steps, source provenance, and candidate metadata mirrors against
  `RVVSelectedBodyEmitCRouteDescription`.

## Requirements

* Preserve the production authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body/config ->
  RVV provider-derived `RVVSelectedBodyEmitCRouteDescription` and
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact validation.
* Remove target/export-local exact intrinsic/type expectation helpers for the
  bounded Stage 1 slice: widening-dot contraction and
  standalone/runtime-scalar standalone reduction validation.
* Validate provider leaf facts by requiring them to be present, coherent with
  route payload/type mappings/statement plan/candidate mirrors, and free of
  stale non-family facts.
* Keep structural route-family facts fail-closed: wrong route family, missing
  provider mirror, missing route-family plan, missing ABI order, missing
  header/type mappings, missing mask/reduction/accumulation facts, and stale
  incompatible family facts must produce targeted diagnostics before artifact
  export.
* Exact `__riscv_*`, `vint*`, `vbool*`, and typed vector spellings may remain
  only as provider-derived artifact output or focused test expectation tied to
  provider facts. They must not be target/export route selection logic.
* Keep changes bounded to
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, directly
  required provider/route-description interfaces if inspection proves they are
  missing a provider fact, and focused target/RVV validation tests.

## Acceptance Criteria

* [x] Target artifact validation no longer has local helper functions or
  constants that compute exact widening-dot or standalone reduction RVV
  intrinsic/type leaves from SEW/LMUL/operation as validation authority.
* [x] Widening-dot artifact validation consumes provider-derived leaf facts
  already present in `RVVSelectedBodyEmitCRouteDescription`; it fails closed
  for missing or inconsistent provider facts by comparing them to the rebuilt
  route payload and artifact mirrors, not to target-local intrinsic tables.
* [x] Standalone and runtime-scalar standalone reduction artifact validation
  consumes provider-derived vector, scalar-result, mask, splat, compare, merge,
  reduction, and store facts; missing or stale provider facts fail closed with
  targeted diagnostics.
* [x] Positive focused tests continue to cover already-supported artifact cases
  where concrete i32/i64 RVV leaves appear as provider-derived outputs.
* [x] Negative focused tests cover missing or mismatched provider leaf facts
  for the affected target validation paths.
* [x] A bounded scan over the touched target validation file, directly touched
  provider interfaces/tests, and relevant specs classifies any remaining
  legacy-string hits as provider-derived artifact output, negative test,
  parseable/deprecated inventory, or spec text rather than active target/export
  authority.
* [x] Focused build/test commands for touched RVV target/plugin paths pass and
  `git diff --check` passes.

## Definition of Done

* `implement.jsonl` and `check.jsonl` contain only relevant spec/context
  entries and validate through the Trellis task workflow.
* The focused target artifact and RVV plugin tests for the changed paths pass.
* No runtime, correctness, or performance claim is made without real `ssh rvv`
  evidence.
* Trellis task status, evidence, and workspace journal truthfully record the
  change and any exact continuation point.
* One coherent commit is created if the bounded task is complete.

## Out of Scope

* New RVV route coverage, reductions, contractions, dtype/LMUL clone batches,
  source-front-door routes, broad smoke matrices, common EmitC semantic
  branches, descriptor-driven export, or Stage 2 expansion.
* Moving intrinsic derivation out of the RVV plugin into target/export code.
* Compatibility wrappers that keep `RVVI32M1*`, `rvv-i32m1`, finite
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, route ids, artifact names, or exact
  intrinsic spelling as executable route authority.
* Report-only, prompt-only, helper-only, or guardrail-only changes as the main
  achievement.

## Technical Approach

Reuse the route-consumer shape already present in this target validation file:
require provider facts to be populated, validate that the rebuilt
`TCRVEmitCLowerableRoute` uses the provider-supplied callees/types/ABI/order,
and validate target artifact candidate metadata mirrors directly against the
selected provider description. For the affected widening-dot and standalone
reduction paths, delete or shrink target-local exact leaf tables, replacing
them with presence checks and route/candidate mirror coherence checks. If a
provider fact required for artifact validation is genuinely absent from
`RVVSelectedBodyEmitCRouteDescription`, add only that description field at the
provider boundary and populate it from the already plugin-owned route plan.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Read: `.trellis/spec/guides/capability-first-design-guide.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-stage1-typed-rvv-shared-route-profile-leaf-derivation/prd.md`.
* Inspected: `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`.
* Inspected: `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* Inspected: `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Relevant journal context: previous June 1 Stage 1 route-profile and MAcc /
  elementwise typed leaf derivation entries in
  `.trellis/workspace/codex/journal-19.md`.
