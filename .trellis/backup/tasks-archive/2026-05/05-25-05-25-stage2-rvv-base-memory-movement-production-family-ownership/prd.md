# Stage2 RVV base memory-movement production family ownership closure

## Goal

Close one bounded production-code submodule of the RVV plugin-local base
memory-movement route-family owner. The production C++ planning/provider path
must own base-memory legality, runtime/ABI role validation, ordered statement
planning, provider materialization facts, and mirror payload validation before
`RVVEmitCRouteProvider` builds `TCRVEmitCLowerableRoute`. Generated-bundle
scripts, target artifacts, FileCheck fixtures, route ids, ABI strings,
metadata, and harness constants may only consume provider-built facts and
mirror fields after route construction.

## Direction Source

- Direction title: `Stage2 RVV base memory-movement production family
  ownership closure`.
- Module owner: RVV plugin-local base memory-movement route-family production
  boundary in C++ planning/provider code.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `8653efda rvv: expose base memory movement boundary evidence`.
- `.trellis/.current-task` was absent, so this task was created from the
  provided Direction Brief before source edits.
- Prior archived task:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-base-memory-movement-route-family-boundary/prd.md`.

## Current Repository Facts

- The required authority chain is selected `tcrv.exec` RVV variant -> typed or
  realized `tcrv_rvv` body -> RVV plugin-owned legality / selected-body
  realization / family planning / provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC -> target artifact -> `ssh rvv` evidence when runtime
  correctness is claimed.
- The previous base-memory task proved generated-bundle and `ssh rvv`
  evidence for `strided_load_unit_store`, but its implementation mix was
  dominated by script, fixture, spec, and Trellis metadata changes.
- Current C++ already contains base-memory route-family plan types, a memory
  family owner registry, materialization facts, memory operand-binding facts,
  base-memory statement plans, migrated statement-plan aggregate consumption,
  and broad C++ tests.
- `RVVEmitCRouteProvider.cpp` already calls
  `verifyRVVSelectedBodyRouteFamilyProviderPlans(...)`, route
  materialization facts, memory operand-binding facts, and
  `getRVVSelectedBodyMigratedRouteStatementPlan(...)` before falling back to
  older generic provider-local route assembly.
- This round must tighten or factor the active production owner/provider
  boundary instead of adding another evidence-only assertion layer.

## Scope

1. Work only on the already route-supported base memory-movement family:
   `strided_load_unit_store`, `unit_load_strided_store`,
   `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
   `masked_unit_load_store`, and `masked_unit_store`.
2. Keep computed-mask memory and segment2 memory under their existing family
   owners.
3. Prefer C++ planning/provider or target mirror-consumption changes over
   script-only, fixture-only, or report-only changes.
4. If the current owner path is already mostly present, finish one bounded
   production submodule by making the provider consume a stronger owner-owned
   artifact and adding fail-closed tests for that path.
5. Preserve common EmitC neutrality: common EmitC materializes
   provider-built routes and must not infer RVV memory semantics, dtype,
   schedule, ABI order, intrinsic spelling, or support state.

## Requirements

1. The base memory family plan must remain the source of truth for memory form,
   source/destination/index/mask/stride roles, runtime AVL/VL control,
   SEW/LMUL/policy, target leaf profile, required headers, type mappings,
   intrinsic leaves, and mirror labels.
2. Provider construction must consume validated owner-built facts before
   route statement construction. It must fail closed if the base memory family
   plan, materialization facts, operand-binding facts, statement plan, or
   mirror payloads are missing or stale.
3. Invalid memory form, wrong pointer/value/index/mask/stride roles, missing
   runtime `n`, incompatible typed config/capability facts, missing source
   operation provenance, missing ordered statement dependencies, and mirror
   mismatches must fail before common EmitC or target artifact authority.
4. Scripts and target artifacts must only consume provider-built and
   provider-validated facts; they must not become memory-route authority.
5. No new operation coverage, dtype/LMUL batch, high-level frontend route,
   source-front-door positive path, descriptor/direct-C/source-export path, or
   legacy i32 route authority may be introduced.

## Acceptance Criteria

- [x] Task context files reference the RVV plugin and EmitC route specs plus
      the archived base-memory evidence task.
- [x] Current base-memory production C++ is inventoried before implementation,
      distinguishing already-closed C++ owner paths from evidence-only gaps.
- [x] At least one production C++ submodule is tightened or factored so that
      base memory-movement provider route construction consumes an
      RVV-owned owner boundary instead of central ad hoc, name-derived,
      metadata-derived, ABI-string-derived, script-derived, artifact-derived,
      descriptor-derived, common-EmitC-derived, source-front-door-derived, or
      legacy-i32 authority.
- [x] Focused C++ tests prove positive provider consumption for at least one
      representative base memory route and fail-closed behavior for missing or
      stale owner facts before route statement construction.
- [x] Negative diagnostics include at least one stale/missing family plan or
      statement-plan dependency and at least one invalid memory fact or mirror
      mismatch relevant to base memory.
- [x] Existing explicit and pre-realized base-memory lit/FileCheck evidence
      still passes where touched and continues to show mirror labels as
      mirrors, not authority.
- [x] Generated-bundle dry-run and one `ssh rvv` correctness run are reused
      only if emitted target sequence, ABI, or executable behavior changes or
      runtime correctness is claimed.
- [x] Bounded authority scans over touched plugin/provider/target/script/test
      files find no new script-, metadata-, name-, descriptor-, ABI-string-,
      common-EmitC-, artifact-, harness-, source-front-door-, or legacy-i32-
      derived route authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Task status, journal, archive, clean final git status, and one coherent
      commit complete if the task finishes.

## Completion Evidence

This round closed a production C++ owner-to-provider submodule rather than
adding another generated-bundle evidence layer.

- Added `RVVSelectedBodyBaseMemoryMovementRouteProviderPlan`, an RVV-local
  provider input that joins the verified base memory movement family plan, the
  same-analysis route operand-binding plan, validated mirror payloads, and the
  ordered base memory statement plan.
- Added
  `getRVVSelectedBodyBaseMemoryMovementRouteProviderPlan(...)`, which fails
  closed before provider route construction when the base family plan is
  missing, materialization facts come from a stale selected-route analysis,
  memory operand-binding facts are missing/stale, or the statement plan does
  not match the verified family plan.
- Updated `getRVVSelectedBodyMigratedRouteStatementPlan(...)` so base memory
  routes flow through the provider-plan owner boundary before being attached to
  the migrated statement-plan aggregate consumed by `RVVEmitCRouteProvider`.
- Extended `test/Plugin/RVVExtensionPluginTest.cpp` to prove positive
  provider-plan consumption for representative base memory routes, validated
  mirror payload propagation, ordered statement-plan ownership, stale
  materialization rejection, and stale binding rejection.
- No EmitC statement sequence, target ABI, generated-bundle script behavior,
  or runtime executable behavior changed. Therefore no new `ssh rvv`
  correctness claim was made in this round.

Checks run:

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-base-memory-movement-production-family-ownership`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `git diff --check`
- Diff-only authority scan over added C++/test lines for legacy i32,
  source-front-door, descriptor, script/harness, artifact/metadata authority,
  ABI-string, and common-EmitC-derived authority terms: no hits.
- `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

## Non-Goals

- No new RVV operation coverage or memory semantics.
- No dtype, SEW, LMUL, or intrinsic coverage batch.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No positive source-front-door route.
- No dashboards, broad smoke matrices, or harness-only summaries.
- No migration of RVV memory semantics into common EmitC, target artifact
  plumbing, `tcrv.exec`, scripts, route ids, metadata fields, manifests,
  artifact names, ABI strings, descriptors, or test names.
- No performance claim.
- No subagents or parallel-agent workflow.

## Validation Plan

1. Build the focused RVV plugin test target if needed.
2. Run the focused RVV extension plugin C++ test.
3. Run focused lit/FileCheck tests for any touched RVV target/script fixtures.
4. Run generated-bundle dry-run and `ssh rvv` correctness only if the emitted
   executable path changes or the final report claims runtime behavior.
5. Run bounded authority scans over touched files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-base-memory-movement-route-family-boundary/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-15.md`, including the previous
  base-memory route-family boundary session.
- Initial production surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
