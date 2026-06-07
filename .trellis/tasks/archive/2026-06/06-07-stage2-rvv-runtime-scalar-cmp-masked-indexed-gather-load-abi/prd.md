# Stage2 RVV runtime-scalar-cmp masked indexed gather-load executable artifact ABI boundary

## Goal

Complete one bounded Stage 2 RVV workflow submodule: selected or pre-realized
runtime-scalar-cmp masked indexed gather-load bodies must carry runtime scalar
comparison, scalar-splat compare mask facts, indexed source gather-read facts,
inactive-lane output preservation through passthrough/no-write behavior,
ordinary unit-store output roles, dtype/SEW/LMUL/config/policy, runtime AVL/VL,
per-operand ABI/header bindings, RVV plugin-owned route validation, EmitC
materialization, target artifact export, generated bundle ABI, and truthful
`ssh rvv` correctness evidence through the provider-owned route boundary.

## What I Already Know

* Session-start repository state was clean on `main`; recent HEAD is
  `f00dcaca rvv: add runtime scalar indexed scatter ABI path`.
* No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
* The previous archived runtime-scalar indexed scatter-store task completed the
  write-side indexed-memory counterpart with typed/pre-realized body,
  verifier, plugin-local realization, route planning, statement/operand
  binding, target validation, generated bundle ABI, and `ssh rvv` evidence.
* Specs require computed-mask indexed gather-load to use typed compute op
  `tcrv_rvv.masked_indexed_load`, memory form
  `computed-mask-indexed-gather-load-unit-store`, indexed data memory form
  `masked-indexed-load`, empty scatter uniqueness, non-empty masked indexed
  load leaf, empty masked indexed store leaf, and ordinary masked/unit-store
  output leaf.
* Gather and scatter both use runtime ABI order
  `cmp_lhs,cmp_rhs,src,index,dst,n`, but gather treats `src` plus `index` as
  the indexed source read and `dst` as the unit-stride output preserving
  inactive lanes.
* Specs require route facts to be structural and provider-owned. Route ids,
  helper names, test names, artifact metadata, ABI strings, descriptor residue,
  and common EmitC behavior are mirrors or neutral carriers only.
* The computed-mask memory statement-plan boundary already includes computed
  mask indexed gather/unit-store and indexed scatter/unit-load as
  production-active subfamilies. This task must prove or repair the combined
  runtime-scalar compare plus indexed gather-load executable seam.

## Requirements

* Scope is one route family only:
  runtime-scalar-cmp masked indexed gather-load with indexed source reads,
  compare-produced mask, inactive-lane output preservation, and unit-stride
  destination store.
* Add or repair the production path:
  selected/pre-realized RVV body -> RVV selected-body realization owner ->
  realized typed `tcrv_rvv` setvl/runtime-scalar-splat/compare/index-load/
  masked-indexed-load/passthrough/unit-store body -> RVV route-family/provider
  validation -> `TCRVEmitCLowerableRoute` -> common EmitC materialization ->
  target artifact export -> generated bundle ABI -> `ssh rvv` correctness
  when executable behavior is claimed.
* If positive runtime-scalar gather fixture coverage is dry-run-only,
  under-validated, or missing runtime evidence, repair the minimal explicit and
  pre-realized selected-body fixture pair needed for this seam rather than
  broadening to unrelated indexed-memory routes.
* The selected/pre-realized body must structurally carry runtime scalar compare
  operand binding, compare predicate, scalar splat/compare mask facts, indexed
  source role, index role, output/destination role, ABI order, runtime AVL/VL,
  dtype/SEW/LMUL/config, index EEW/offset unit/source, masked indexed load
  fact, ordinary store fact, mask/tail policy, and inactive-lane output
  preservation/no-write behavior.
* The RVV plugin owner must derive or validate route/type/header/intrinsic
  facts from typed body/config/capability/runtime facts, and fail closed with
  targeted diagnostics if runtime scalar binding, compare operand role,
  computed mask binding, indexed source/index role, inactive-lane policy,
  output role, ABI/header binding, runtime AVL/VL, statement facts, or C type
  facts are missing or stale.
* Target artifact validation must consume the provider-owned indexed-memory
  validation contract before accepting route payloads, statements, ABI
  mappings, headers/types, or metadata mirrors.
* Common EmitC/export must remain neutral: they may consume provider-built
  routes and explicit mirror fields, but must not invent RVV semantics,
  intrinsic spelling, dtype, memory form, index roles, inactive-lane policy,
  ABI order, or output ordering.
* Preserve existing computed-mask indexed scatter-store, runtime-scalar
  scatter-store, strided memory, segment2 memory, MAcc/reduction, and unrelated
  mask-route behavior unless live code proves a directly shared validation
  repair is required.

## Acceptance Criteria

* [ ] Positive explicit and pre-realized runtime-scalar-cmp masked indexed
      gather-load evidence reaches materialized selected boundary, emission
      plan, target artifact export, generated bundle compile, and `ssh rvv`
      correctness when executable behavior is claimed.
* [ ] Positive generated evidence exposes runtime scalar ABI/header binding,
      scalar splat/compare producer source, compare predicate construction,
      compare-produced mask facts, mask and inactive-lane policy, indexed
      source role, output role, index source/EEW/offset unit, runtime AVL/VL,
      dtype/config, header/prototype binding, ABI order, masked indexed load
      leaf, ordinary store leaf, and explicit provider-supported mirrors.
* [ ] Gather facts carry a masked indexed load leaf and ordinary output store
      leaf, and reject stale scatter indexed-store, destination uniqueness,
      wrong indexed data/destination form, stale header/prototype, wrong C
      type, wrong ABI value mapping, or unsupported executable route residue
      before target artifact acceptance.
* [ ] Focused fail-closed evidence rejects at least one stale or missing
      executable-boundary fact such as runtime scalar binding, compare operand
      role, index/source role, inactive-lane policy, output role, route-family
      validation contract, header/prototype binding, wrong generated C type,
      wrong ABI value mapping, stale scatter/store residue, or stale producer
      claim.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
* [ ] Relevant focused lit/generated-bundle dry-run tests pass.
* [ ] `scripts/rvv_generated_bundle_abi_e2e.py` self-test or focused dry-run
      path passes if the script changes.
* [ ] Runtime RVV correctness is claimed only after real `ssh rvv` evidence for
      the generated bundle, including representative counts/scalar values and
      destination preservation on inactive lanes.
* [ ] Bounded old-authority scan over touched files and added diff lines finds
      no new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, source-artifact,
      or source-front-door route authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are checked.

## Definition of Done

* The runtime-scalar-cmp masked indexed gather-load executable artifact/ABI
  seam is implemented end to end, or the task remains open with a precise
  production blocker and exact next continuation point.
* Trellis task status/context and the codex workspace journal record the
  outcome truthfully.
* One coherent commit is created only after the bounded module is complete and
  verified.

## Out of Scope

* No broad indexed-memory matrix.
* No dtype/LMUL clone batch.
* No scatter-store rework except as bounded reference or shared validation
  repair.
* No unit-load/unit-store expansion outside this gather-load seam.
* No segment2 expansion.
* No MAcc, reduction, compare-select, conversion, strided-memory,
  base-memory, or unrelated mask-route expansion.
* No high-level Linalg, Vector, StableHLO, per-Linalg route authority, or
  source-front-door positive route.
* No dashboard, index, report-only, helper-only, or broad smoke-test work as
  the main achievement.
* No common EmitC invention of RVV semantics.
* No compatibility path that lets stale scatter/store/indexed-destination
  residue authorize gather-load output.
* No descriptor-driven, direct-C, source-export, or legacy i32 route authority.

## Technical Approach

Inspect the current computed-mask indexed gather-load, runtime-scalar indexed
gather, and last completed runtime-scalar scatter-store implementation.
Determine whether this runtime-scalar-cmp masked indexed gather-load
selected/pre-realized path is absent, dry-run-only, under-validated, or already
present but lacking boundary evidence. Then implement or harden only the
missing production seam: typed body or realization surface, RVV route-family
facts, computed-mask memory statement plan, target artifact validation,
generated-bundle script expectations, focused fixtures/tests, and `ssh rvv`
evidence.

## Decision (ADR-lite)

**Context**: The runtime-scalar indexed scatter-store seam is complete and
proved the write-side indexed memory ABI boundary. The next bounded read-side
indexed memory seam is runtime scalar compare driving a masked indexed
gather-load with inactive-lane output preservation.

**Decision**: Implement or harden only the runtime-scalar-cmp masked indexed
gather-load executable artifact/ABI boundary, combining existing runtime-scalar
mask producer rules with indexed gather-load route facts through RVV-owned
selected-body realization, provider validation, target validation, generated
bundle evidence, and `ssh rvv` correctness when claimed.

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
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-masked-indexed-scatter-store-abi/prd.md`

Likely implementation and evidence files from the direction brief and repo
inventory:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-load-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-indexed-gather-load-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-indexed-gather-load-fail-closed.test`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-load.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-load.mlir`
