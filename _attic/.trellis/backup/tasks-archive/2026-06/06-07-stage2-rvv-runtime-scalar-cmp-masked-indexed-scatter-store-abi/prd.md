# Stage2 RVV runtime-scalar-cmp masked indexed scatter-store executable artifact ABI boundary

## Goal

Complete one bounded Stage 2 RVV workflow submodule: selected or
pre-realized runtime-scalar-cmp masked indexed scatter-store bodies must carry
runtime scalar comparison, scalar-splat compare mask facts, masked active-lane
indexed scatter-store behavior, inactive-lane no-write/preserve-destination
policy, scaled index vector, payload value, destination memory role,
dtype/SEW/LMUL/config/policy, runtime AVL/VL, per-operand ABI/header bindings,
RVV plugin-owned route validation, EmitC materialization, target artifact
export, generated bundle ABI, and truthful `ssh rvv` correctness evidence
through the provider-owned route boundary.

## What I Already Know

* Session-start repository state was clean on `main`; recent HEAD is
  `dedd53a4 rvv: harden computed mask indexed scatter ABI evidence`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief before source edits.
* The previous archived computed-mask indexed scatter-store task completed the
  vector-compare scatter seam and added fail-closed evidence against stale
  masked-indexed-load residue, ordinary unit-store residue, unit-stride
  destination fallback, and unscaled index-vector residue.
* The archived runtime-scalar indexed gather task completed the read-side
  runtime-scalar compare/mask indexed memory seam but explicitly left indexed
  scatter out of scope.
* The archived runtime-scalar segment2 store task is the bounded reference for
  runtime scalar ABI binding, scalar splat/compare producer facts, mask-tail
  plan/owner evidence, and fail-closed target validation on a write-side memory
  route.
* Specs require this route to start from typed or realized `tcrv_rvv` body
  facts and RVV provider-owned validation. Route ids, helper names, test names,
  artifact metadata, ABI strings, descriptor residue, and common EmitC behavior
  are mirrors or neutral carriers only.
* The computed-mask memory statement-plan boundary already includes
  runtime-scalar computed-mask store/load-store and computed-mask indexed
  scatter/unit-load as production-active subfamilies. This task must prove or
  repair the combined runtime-scalar compare plus indexed scatter-store seam.

## Requirements

* Scope is one route family only:
  runtime-scalar-cmp masked indexed scatter-store with unit-stride payload
  load, scaled index vector, and masked indexed destination store.
* Add or repair the production path:
  selected/pre-realized RVV body -> RVV selected-body realization owner ->
  realized typed `tcrv_rvv` setvl/runtime-scalar-splat/compare/index-load/
  payload-load/mask/masked-indexed-store body -> RVV route-family/provider
  validation -> `TCRVEmitCLowerableRoute` -> common EmitC materialization ->
  target artifact export -> generated bundle ABI -> `ssh rvv` correctness
  when executable behavior is claimed.
* If no positive runtime-scalar scatter fixture exists yet, create the minimal
  explicit and pre-realized selected-body fixture pair needed for this seam
  rather than broadening to unrelated indexed-memory routes.
* The selected/pre-realized body must structurally carry runtime scalar compare
  operand binding, compare predicate, scalar splat/compare mask facts,
  payload/source role, index role, destination scatter role, ABI order,
  runtime AVL/VL, dtype/SEW/LMUL/config, index EEW/offset unit/source/
  uniqueness, scaled index vector use, mask/tail policy, and inactive-lane
  destination preservation/no-write behavior.
* The RVV plugin owner must derive or validate route/type/header/intrinsic
  facts from typed body/config/capability/runtime facts, and fail closed with
  targeted diagnostics if runtime scalar binding, compare operand role,
  computed mask binding, inactive-lane/store policy, scaled index vector,
  payload/destination role, ABI/header binding, runtime AVL/VL, or statement
  facts are missing or stale.
* Target artifact validation must consume the provider-owned validation
  contract before accepting route payloads, statements, ABI mappings,
  headers/types, or metadata mirrors.
* Common EmitC/export must remain neutral: they may consume provider-built
  routes and explicit mirror fields, but must not invent RVV semantics,
  intrinsic spelling, dtype, memory form, index roles, inactive-lane policy,
  ABI order, or output ordering.
* Preserve existing computed-mask indexed scatter, runtime-scalar indexed
  gather, runtime-scalar segment2, strided memory, segment2 memory, and
  unrelated mask-route behavior unless live code proves a directly shared
  validation repair is required.

## Acceptance Criteria

* [ ] Positive explicit and pre-realized runtime-scalar-cmp masked indexed
      scatter-store evidence reaches materialized selected boundary, emission
      plan, target artifact export, generated bundle compile, and `ssh rvv`
      correctness when executable behavior is claimed.
* [ ] Positive generated evidence exposes runtime scalar ABI/header binding,
      scalar splat/compare producer source, compare predicate construction,
      compare-produced mask facts, mask and inactive-lane policy, indexed
      destination role, payload/source role, index source/EEW/offset unit/
      uniqueness, scaled index vector, runtime AVL/VL, dtype/config,
      header/prototype binding, ABI order, and explicit provider-supported
      mirrors.
* [ ] Scatter facts carry a masked indexed store leaf and reject stale masked
      indexed load, ordinary unit-store, unit-stride destination, stale gather,
      stale header/prototype, wrong generated C type, or wrong ABI value
      mapping residue before target artifact acceptance.
* [ ] Focused fail-closed evidence rejects at least one stale or missing
      runtime-scalar/scatter executable-boundary fact such as runtime scalar
      binding, compare operand role, computed mask binding, inactive-lane/
      store policy, scaled index vector, payload/destination ABI role,
      route-family validation contract, unsupported executable route claim, or
      stale gather/unit-store residue.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
* [ ] Relevant focused lit/generated-bundle dry-run tests pass.
* [ ] `scripts/rvv_generated_bundle_abi_e2e.py` self-test or focused dry-run
      path passes if the script changes.
* [ ] Runtime RVV correctness is claimed only after real `ssh rvv` evidence for
      the generated bundle, including representative counts/patterns and
      destination preservation on inactive lanes.
* [ ] Bounded old-authority scan over touched files and added diff lines finds
      no new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, source-artifact,
      or source-front-door route authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are checked.

## Definition of Done

* The runtime-scalar-cmp masked indexed scatter-store executable artifact/ABI
  seam is implemented end to end, or the task remains open with a precise
  production blocker and exact next continuation point.
* Trellis task status/context and the codex workspace journal record the
  outcome truthfully.
* One coherent commit is created only after the bounded module is complete and
  verified.

## Out of Scope

* No broad indexed-memory matrix.
* No dtype/LMUL clone batch.
* No computed-mask scatter/gather rework except as bounded reference or shared
  validation repair.
* No segment2 expansion.
* No MAcc, reduction, compare-select, conversion, strided-memory, base-memory,
  or unrelated mask-route expansion.
* No high-level Linalg, Vector, StableHLO, per-Linalg route authority, or
  source-front-door positive route.
* No dashboard, index, report-only, helper-only, or broad smoke-test work as
  the main achievement.
* No common EmitC invention of RVV semantics.
* No compatibility path that lets stale gather/load/unit-store residue
  authorize scatter output.
* No descriptor-driven, direct-C, source-export, or legacy i32 route authority.

## Technical Approach

Inspect the current computed-mask indexed scatter-store, runtime-scalar indexed
gather, and runtime-scalar segment2 store implementations. Determine whether a
runtime-scalar-cmp masked indexed scatter-store selected/pre-realized path is
absent, dry-run-only, under-validated, or already present but lacking boundary
evidence. Then implement or harden only the missing production seam: typed body
or realization surface, RVV route-family facts, computed-mask memory statement
plan, target artifact validation, generated-bundle script expectations,
focused fixtures/tests, and `ssh rvv` evidence.

## Decision (ADR-lite)

**Context**: The computed-mask indexed scatter-store seam is complete for
vector compare operands, and the runtime-scalar indexed gather seam is complete
for read-side indexed memory. The next bounded write-side indexed memory seam
is runtime scalar compare driving a masked indexed scatter-store.

**Decision**: Implement or harden only the runtime-scalar-cmp masked indexed
scatter-store executable artifact/ABI boundary, combining the existing
runtime-scalar mask producer rules with the indexed scatter-store route facts
through RVV-owned selected-body realization, provider validation, target
validation, and generated-bundle evidence.

**Consequences**: The work remains Stage 2 typed RVV route coverage, avoids a
broad indexed-memory matrix, and keeps common EmitC/export neutral. Other
memory routes remain explicit future tasks.

## Technical Notes

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/validation/index.md`
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-computed-masked-indexed-scatter-store-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-load-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-07-rvv-runtime-scalar-cmp-segment2-store-abi/prd.md`

Likely implementation and evidence files from the direction brief and repo
inventory:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* Current computed-masked scatter fixtures under `test/Scripts/` and
  `test/Target/RVV/`
* Runtime-scalar indexed gather fixtures only as bounded ABI/mask reference
