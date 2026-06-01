# Stage1 typed RVV MAcc route-family derivation

## Goal

Replace bounded `i32m1`/`i32m2` exact-spelling authority in the active RVV
MAcc EmitC route-family plan owner with typed selected-body/config derivation.
Existing plain MAcc, scalar-broadcast MAcc, and computed-mask MAcc routes may
still emit today's `i32m1` or `i32m2` RVV intrinsic names, but only as validated
outputs derived from selected `tcrv_rvv` body/config/capability/runtime facts:
element type, SEW, LMUL, policy, memory form, operation kind, accumulator
layout, mask/result type, and runtime AVL/VL.

## What I Already Know

* The Hermes direction brief is the source for this task.
* The repository started this round on `main` with a clean worktree at
  `64201712 rvv: derive elementwise leaves from typed facts`.
* There was no active `.trellis/.current-task`; this task was created as
  `.trellis/tasks/06-01-stage1-typed-rvv-macc-route-family-derivation`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-01-stage1-typed-rvv-elementwise-intrinsic-derivation/`
  moved elementwise and scalar-broadcast elementwise leaves to typed-fact
  derivation and provides the local implementation pattern.
* Specs require the RVV route provider to consume typed low-level
  `tcrv_rvv` body/config facts. `tcrv.exec`, route ids, C ABI strings,
  artifact names, emission-plan metadata, and exact intrinsic spellings are
  mirrors only and must not become RVV compute/type authority.
* The active owner under inspection is
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`, with connected
  planning facts in `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.

## Requirements

* Keep the production route as selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body/config -> RVV plugin MAcc route-family planning ->
  provider-built `TCRVEmitCLowerableRoute` -> common EmitC materialization.
* For existing plain MAcc, derive vector C types, load/store leaves, MAcc
  intrinsic leaf, accumulator/result type facts, and validation expectations
  from typed element/SEW/LMUL/policy/memory/runtime facts.
* For existing scalar-broadcast MAcc, derive scalar element C type, scalar
  splat leaf, MAcc intrinsic leaf, vector/result C types, load/store leaves,
  accumulator facts, and target/profile mirrors from typed facts.
* For existing computed-mask MAcc, derive compare/mask leaves, mask/result C
  types, MAcc leaf, memory leaves, accumulator facts, and policy validation
  from typed facts.
* Treat concrete `i32m1` and `i32m2` RVV intrinsic/type spellings as derived
  output only. Do not use route ids, artifact names, helper op names, ABI
  strings, owner-local legacy constants, or exact intrinsic spellings as
  selected-route authority.
* Unsupported or incomplete element type, SEW, LMUL, policy, memory form,
  operation kind, accumulator layout, mask/result type, scalar type, or runtime
  AVL/VL facts must fail closed with targeted diagnostics at the MAcc
  route-family boundary.
* Keep changes bounded to the RVV plugin-owned MAcc route-family planning
  surfaces, directly connected typed-fact helpers, focused tests, and Trellis
  task evidence.

## Acceptance Criteria

* [x] `RVVEmitCMAccRouteFamilyPlanOwners.cpp` no longer treats complete
  `__riscv_*_i32m1`, `__riscv_*_i32m2`, `vint32m1_t`, `vint32m2_t`,
  `vbool32_t`, `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` spellings as owner-local route authority for supported
  MAcc routes.
* [x] Existing plain MAcc and scalar-broadcast MAcc produce intrinsic/type
  leaves only through typed element/SEW/LMUL plus operation/memory/accumulator
  derivation and fail closed for unsupported typed combinations.
* [x] Existing computed-mask MAcc either consumes typed facts for compare,
  mask, MAcc, memory, result, and policy leaves, or this round records the exact
  continuation point after completing the coherent plain plus scalar-broadcast
  submodule.
* [x] Positive focused tests prove typed config-derived MAcc leaf planning for
  already supported cases without adding new MAcc coverage.
* [x] Negative focused tests mutate missing or unsupported typed config,
  accumulator, scalar, or mask facts and assert fail-closed diagnostics before
  provider route construction.
* [x] A bounded scan over touched MAcc, elementwise, base-memory planning files,
  relevant tests, and specs classifies remaining legacy-string hits as derived
  output, parseable-only/deprecated, negative/spec text, or unrelated remaining
  Stage 1 debt rather than active MAcc authority.
* [x] Focused build/test commands for touched RVV plugin paths pass.

## Definition of Done

* `git diff --check` passes.
* Focused RVV plugin tests for changed MAcc planning behavior pass.
* Broader `check-tianchenrv` or the smallest relevant target is run if touched
  behavior is visible through lit tests.
* No runtime correctness or performance claim is made without `ssh rvv`
  evidence.
* Trellis task status, evidence, and workspace journal truthfully record what
  changed and the exact continuation point if unfinished.
* One coherent commit is created if the task is complete.

## Out of Scope

* New MAcc coverage, reductions, contractions, compare/select expansion,
  dtype/LMUL clone growth, source-front-door/source-artifact positive routes,
  artifact packaging, broad smoke matrices, or Stage 2 coverage expansion.
* Descriptor-driven C/source export.
* Common EmitC semantic branches.
* Compatibility wrappers that preserve old i32 route authority.
* Prompt/report/helper-only changes as the main achievement.

## Technical Approach

Use the completed elementwise typed derivation pattern as the local model.
Reuse or extend typed route-planning helpers so the MAcc owner asks for
operation, vector type, mask type, scalar splat, memory, accumulator, and
intrinsic leaves through `RVVSelectedBodyTypedConfigFacts` plus MAcc-specific
facts. Plan validation should compare accepted plan fields against typed
config-derived expected values, not against owner-local `i32m1`/`i32m2`
templates.

If the full MAcc owner is too large for one round, finish a coherent plain
MAcc plus scalar-broadcast MAcc derivation submodule, keep computed-mask MAcc
fail-closed or unchanged only where not touched, and record the exact
computed-mask continuation point.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/testing/index.md`, and `.trellis/spec/variant-pipeline/index.md`.
* Read previous task PRD/evidence:
  `.trellis/tasks/archive/2026-06/06-01-stage1-typed-rvv-elementwise-intrinsic-derivation/`.
* Read current workspace journal entries for recent elementwise/MAcc route
  ownership work in `.trellis/workspace/codex/journal-19.md`.
