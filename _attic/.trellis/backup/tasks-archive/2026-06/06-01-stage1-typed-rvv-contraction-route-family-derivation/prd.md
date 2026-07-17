# Stage1 typed RVV contraction route-family derivation

## Goal

Replace fixed `i16mf2`/`i32m1` intrinsic and type spelling authority in the
active RVV contraction EmitC route-family plan owner with typed selected-body,
config, capability, and runtime fact derivation. Existing contraction routes
may still emit today's `i16mf2` and `i32m1` RVV intrinsic names, but only as
validated derived output from selected `tcrv_rvv` body/config structure:
source element type, source SEW/LMUL, result element type, result SEW/LMUL,
policy, memory form, operation kind, relation, accumulator/result layout, mask
facts, and runtime AVL/VL.

## What I Already Know

* The Hermes Direction Brief is the task source.
* The repository started this round on `main` with a clean worktree at
  `d9f0a81d rvv: derive macc route leaves from typed facts`.
* There was no active `.trellis/.current-task`; this task was created as
  `.trellis/tasks/06-01-stage1-typed-rvv-contraction-route-family-derivation`.
* Specs require RVV route support to start from a selected typed low-level
  `tcrv_rvv` body and plugin-owned route provider output. Route ids, artifact
  names, ABI strings, emission metadata, exact intrinsic spellings, and legacy
  `i32m1` names are mirrors or derived outputs only.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-01-stage1-typed-rvv-macc-route-family-derivation/`
  completed the local MAcc pattern: route-family plans snapshot typed config
  facts and validate leaf mirrors against those facts before provider
  materialization.
* The active owner is
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, with
  directly required shared planning fields/helpers in
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* A focused scan shows contraction still has owner-local constants and mirror
  checks for `__riscv_*_i32m1`, `vint16mf2_t`, `vint32m1_t`, `vbool32_t`,
  `!tcrv_rvv.vector<i16, "mf2">`, `!tcrv_rvv.vector<i32, "m1">`, and fixed
  target/profile strings.

## Requirements

* Keep the production route as selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body/config -> RVV plugin contraction route-family plan
  -> direct contraction provider plan -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materialization.
* Derive contraction result-side leaves from selected typed config facts:
  result vector type, result vector C type, mask type, mask C type, VL C type,
  setvl leaf, store leaf, and policy/config mirrors.
* Derive contraction source-side facts from typed selected-body operands and
  source config: source element type, source SEW/LMUL, source vector type,
  source vector C type, unit-stride source load, and strided source load.
* Derive widening MAcc, widening product, masked widening product, reduction,
  compare, masked merge, scalar seed splat, accumulator layout, result layout,
  relation, memory-form, and runtime ABI leaves from the verified contraction
  body/config facts.
* Treat concrete `__riscv_*_i32m1`, `vint*`, `vbool*`, and typed vector/mask
  spelling as derived output only. Do not use owner-local fixed spelling
  constants or validation templates as route authority.
* Unsupported or incomplete source/result SEW/LMUL, relation, policy, memory
  form, accumulator/result layout, mask role/source/form, or runtime AVL/VL
  facts must fail closed with targeted diagnostics before route construction.
* Keep changes bounded to contraction route-family ownership, directly needed
  shared typed-source planning plumbing, focused tests, and Trellis evidence.

## Acceptance Criteria

* [ ] `RVVEmitCContractionRouteFamilyPlanOwners.cpp` no longer treats complete
  `__riscv_*_i32m1`, `vint16mf2_t`, `vint32m1_t`, `vbool32_t`,
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, or `!tcrv_rvv.i32m*` spellings
  as owner-local route authority for supported contraction routes.
* [ ] Existing widening MAcc contraction support derives compute, result,
  source, accumulator, runtime, and store leaves from typed facts and fails
  closed for stale or unsupported typed combinations.
* [ ] Existing widening dot-reduction support derives widening product,
  reduction, scalar seed splat, source/result layout, and runtime leaves from
  typed facts and fails closed for stale or unsupported typed combinations.
* [ ] Existing computed-mask and strided dot-reduction variants derive compare,
  mask, masked product, merge, strided source-load, memory layout, and stride
  facts from typed owner facts or this round records the exact continuation
  point after a coherent widening MAcc plus dot-reduction proof point.
* [ ] Positive focused tests prove typed config-derived contraction leaf
  planning on already supported cases without adding new contraction coverage.
* [ ] Negative focused tests mutate missing or stale source/result/mask/
  intrinsic facts and assert fail-closed diagnostics before provider route
  construction.
* [ ] A bounded scan over touched contraction/MAcc/route-planning files,
  relevant tests, and specs classifies remaining legacy-string hits as derived
  output, parseable-only/deprecated, negative/spec text, or unrelated remaining
  Stage 1 debt rather than active contraction authority.
* [ ] Focused build/test commands for touched RVV plugin paths pass and
  `git diff --check` passes.

## Definition of Done

* `git diff --check` passes.
* Focused RVV plugin C++ tests for changed contraction planning behavior pass.
* The smallest relevant lit/export tests for touched contraction routes pass if
  route metadata or artifact expectations change.
* No runtime correctness or performance claim is made without `ssh rvv`
  evidence.
* Trellis task status, evidence, and workspace journal truthfully record what
  changed and the exact continuation point if unfinished.
* One coherent commit is created if the task is complete.

## Out of Scope

* New contraction coverage, matmul, new reductions beyond existing contraction
  validation, dtype/LMUL clone batches, source-front-door/source-artifact
  positive routes, common EmitC semantic branches, artifact/report-only work,
  or Stage 2 coverage expansion.
* Descriptor-driven C/source export.
* Compatibility wrappers that preserve old i32 route authority.
* Prompt/report/helper-only changes as the main achievement.

## Technical Approach

Reuse the completed MAcc typed derivation pattern. Add explicit contraction
typed fact snapshots to the contraction route-family plan where missing, and
derive source-side facts from selected body operand types/config instead of
hard-coded source `i16mf2` leaves. Plan validation should compare accepted
fields against owner-derived facts, not fixed owner-local spelling constants.
Description mirror verification should compare against the verified
contraction plan. Shared route planning may keep neutral helper functions for
deriving typed source/config leaves, but common EmitC/export must not choose
RVV semantics.

If the full contraction owner proves too large, finish one coherent widening
MAcc plus widening dot-reduction derivation proof point and record the exact
computed-mask/strided continuation point.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/testing/index.md`, and `.trellis/spec/variant-pipeline/index.md`.
* Read previous task PRD/evidence:
  `.trellis/tasks/archive/2026-06/06-01-stage1-typed-rvv-macc-route-family-derivation/`.
* Read recent workspace journal entries in
  `.trellis/workspace/codex/journal-19.md`.
