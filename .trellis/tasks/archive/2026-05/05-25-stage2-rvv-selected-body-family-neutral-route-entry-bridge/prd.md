# Stage2 RVV selected-body realization family-neutral route-entry bridge

## Goal

Generalize the current pre-realized selected-body route-entry materialization
from an elementwise/compare-select-only entry point into an RVV plugin-owned
family-neutral bridge for already statement-plan-backed selected-body families.
A selected `tcrv.exec` RVV variant that still contains a supported pre-realized
typed `tcrv_rvv` body should be realized by RVV plugin code before route facts
are collected, then feed the existing RVV-owned route analysis,
materialization facts, operand-binding facts, statement plans, provider-built
`TCRVEmitCLowerableRoute`, and common EmitC materializer path.

This is a route-entry ownership and production-path rewire. It is not new RVV
operation coverage.

## Direction Source

- Direction title: `Stage2 RVV selected-body realization family-neutral route-entry bridge`.
- Module owner: RVV plugin-owned selected-body realization route-entry bridge
  for already statement-plan-backed selected-body families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `a41263d6 rvv: materialize pre-realized route entries`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the RVV-first chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires pre-realized body
  hints/config/profile that affect generated code to be consumed into real
  `tcrv_rvv` structure before route planning/provider construction.
- The same spec already defines RVV-owned statement-plan boundaries for
  compare/select and base memory movement, and states that provider code must
  consume those plans instead of rebuilding statement sequences from operation
  names, ABI strings, or memory-form branches.
- `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  common code consumes provider-built routes and must not choose RVV semantics,
  intrinsics, schedules, body shapes, or route authority from metadata.
- The archived task
  `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-selected-body-realization-route-materialization/prd.md`
  wired production route/emission entries through
  `getOrRealizeRVVElementwiseCompareSelectRouteBoundary(...)`, proving
  pre-realized elementwise add and compare/select can reach EmitC route
  materialization without a prior explicit selected-boundary pass.
- Current code evidence:
  - `RVVExtensionPlugin::materializeSelectedLoweringBoundary(...)` already
    uses the generic `realizePreRealizedRVVSelectedBody(...)` path for all
    supported pre-realized selected-body families.
  - `RVVExtensionPlugin::buildVariantEmissionPlan(...)` and
    `RVVExtensionPlugin::buildVariantEmitCLowerableRoute(...)` currently use
    an elementwise/compare-select-specific route-entry helper.
  - `RVVSelectedBodyRealization.cpp` already contains realization logic for
    compare/select and multiple memory families, including base memory forms
    such as `typed_strided_memory_pre_realized_body`.
  - Existing target fixtures for pre-realized base memory run the explicit
    `--tcrv-materialize-selected-lowering-boundaries` pass before emission
    planning/export.
  - Existing direct EmitC route-entry fixture covers pre-realized add and
    compare/select only.

## Requirements

1. Add or refactor to a bounded RVV plugin-local route-entry bridge that can
   either return the existing unique realized `setvl`/`with_vl` boundary or
   realize a unique supported pre-realized selected body before route facts are
   collected.
2. Wire RVV `buildVariantEmissionPlan(...)` and
   `buildVariantEmitCLowerableRoute(...)` through that family-neutral bridge
   instead of an elementwise/compare-select-only helper.
3. Preserve mixed realized + pre-realized fail-closed behavior.
4. Preserve the route-analysis guard: provider/common route analysis must still
   reject pre-realized bodies if the RVV plugin realization bridge did not run
   or could not complete.
5. Prove with focused coverage that pre-realized compare/select and one
   base-memory form realize before route fact collection and then consume the
   existing materialization facts, operand-binding facts, statement plans, and
   provider-built route path.
6. Keep semantic realization and family decisions in RVV plugin code. Common
   EmitC/export may consume provider-built routes but must not infer RVV dtype,
   operation kind, policy, schedule, ABI roles, memory form, body shape, or
   intrinsic names.
7. Preserve explicit already-realized selected-body behavior and the explicit
   `--tcrv-materialize-selected-lowering-boundaries` path.
8. Do not add route coverage, new operation families, reductions,
   contractions, dtype/LMUL clone batches, frontend lowering, source-front-door
   positive routes, legacy i32 authority, descriptor/direct-C/source-export
   paths, runtime ABI changes, dispatch/fallback behavior changes, or runtime
   correctness/performance claims.

## Acceptance Criteria

- [x] PRD, `implement.jsonl`, and `check.jsonl` reference the relevant RVV
      plugin, EmitC route, plugin-protocol, testing specs, and previous
      route-materialization PRD.
- [x] RVV route/emission production entries use a reusable
      selected-body route-entry bridge rather than an
      elementwise/compare-select-specific helper.
- [x] A pre-realized compare/select body is realized before route analysis and
      reaches the existing compare/select materialization facts,
      operand-binding facts, statement-plan, provider route, and common EmitC
      materializer path.
- [x] A pre-realized base-memory body, using one already route-supported form,
      is realized before route analysis and reaches the existing base-memory
      materialization facts, memory operand-binding facts, statement-plan,
      provider route, and common EmitC materializer path.
- [x] Unsupported, malformed, or incomplete route-entry realization fails
      closed with targeted diagnostics before provider/common semantic
      invention.
- [x] Explicit already-realized selected-body fixtures and explicit
      selected-boundary materialization fixtures remain unchanged.
- [x] Representative lit/FileCheck coverage proves direct pre-realized
      selected-body EmitC route materialization for compare/select and one
      base-memory form without first running
      `--tcrv-materialize-selected-lowering-boundaries`.
- [x] Bounded scan over touched realization/planning/provider/common files
      shows provider/common EmitC code does not invent selected-body
      realization semantics.
- [x] Active-authority scan over touched files finds no newly introduced
      legacy `RVVI32M1`, `rvv-i32m1`, positive finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/source-front-door/direct-C/source-export,
      or mirror-only route authority drift.
- [x] `git diff --check` passes.
- [x] Focused build/tests and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Out Of Scope

- No new RVV operation coverage, new route families, memory/reduction/
  contraction expansion, dtype/LMUL matrix expansion, or Stage 2 coverage
  expansion.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive routes, descriptor-driven computation,
  direct-C/source-export route, or legacy i32 route authority.
- No runtime ABI, dispatch/fallback, emitted target sequence, correctness,
  runtime, or performance claim changes.
- No movement of RVV semantics into common EmitC/export.
- No `ssh rvv` evidence unless emitted target sequence, runtime correctness,
  ABI behavior, or performance is changed or claimed.

## Technical Approach

Replace the local route-entry helper in `RVVExtensionPlugin.cpp` with a
family-neutral bridge that:

- first accepts an existing unique realized `tcrv_rvv.setvl` /
  `tcrv_rvv.with_vl` boundary;
- rejects mixed realized and pre-realized selected bodies;
- when no realized boundary exists, checks for any supported pre-realized RVV
  selected body and invokes the existing generic RVV plugin-owned
  `realizePreRealizedRVVSelectedBody(...)` path;
- lets unsupported or malformed bodies fail through the owning realization
  diagnostics;
- returns the realized `with_vl` boundary to the existing route-analysis and
  provider path.

The provider should still consume only realized route analysis,
materialization facts, operand-binding facts, and statement plans. Common
EmitC/export should remain untouched unless a neutral interface adjustment is
unavoidably required.

## Validation Plan

1. Validate and start this Trellis task.
2. Build the focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if needed.
5. Run focused lit/FileCheck coverage for:
   - direct pre-realized compare/select EmitC route materialization;
   - direct pre-realized base-memory EmitC route materialization;
   - existing pre-realized and explicit selected-body target artifact fixtures
     for compare/select and the chosen base-memory form.
6. Run bounded semantic-realization scan over touched RVV and common files.
7. Run active-authority scan over touched files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Implementation Result

- Added public RVV selected-body route-entry predicates/helpers:
  `variantContainsPreRealizedRVVRouteEntrySelectedBody(...)` and
  `realizePreRealizedRVVRouteEntrySelectedBody(...)`.
- Added a route-entry family predicate that accepts the existing
  elementwise/compare-select cluster plus base memory movement pre-realized
  bodies: strided load/unit store, unit load/strided store, indexed gather,
  indexed scatter, and static masked base memory.
- Replaced `RVVExtensionPlugin.cpp`'s
  `getOrRealizeRVVElementwiseCompareSelectRouteBoundary(...)` helper with
  `getOrRealizeRVVSelectedBodyRouteBoundary(...)`.
- Rewired `RVVExtensionPlugin::buildVariantEmissionPlan(...)` and
  `RVVExtensionPlugin::buildVariantEmitCLowerableRoute(...)` through the new
  family-neutral route-entry bridge.
- Kept route-entry realization bounded: unrelated pre-realized families such
  as reduction fail closed with a targeted route-entry diagnostic instead of
  silently falling back to provider/common semantic invention.
- Added C++ plugin coverage showing direct route/emission production entries
  realize pre-realized add, compare/select, and strided load/unit-store bodies
  before route facts are collected, then reach the expected RVV-owned provider
  plan IDs.
- Extended direct EmitC lit/FileCheck coverage so a pre-realized strided
  load/unit-store base-memory body reaches EmitC route materialization without
  first running `--tcrv-materialize-selected-lowering-boundaries`.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  selected-body route-entry realization bridge contract, including signatures,
  validation/error behavior, good/base/bad cases, and tests required.
- Common EmitC, target export, RVV EmitC route provider, and RVV EmitC route
  planning files were not changed in this round.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-selected-body-family-neutral-route-entry-bridge`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Focused direct EmitC route-entry FileCheck passed:
  `build/bin/tcrv-opt test/Conversion/EmitC/rvv-pre-realized-elementwise-route-materialization.mlir --split-input-file --tcrv-materialize-emitc-lowerable-routes | /usr/lib/llvm-20/bin/FileCheck test/Conversion/EmitC/rvv-pre-realized-elementwise-route-materialization.mlir`.
- `cmake --build build --target check-tianchenrv -j2` passed with 364/364
  lit tests.
- Added-line active-authority scan over touched implementation and test files
  found no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, positive finite
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/source-front-door/direct-C,
  source-export, provider-supported mirror, or mirror-only route authority
  drift.
- A broader added-line scan did find one ordinary capability fixture attribute
  `status = "available"` in the new lit input. This is capability availability
  test input, not route acceptance or route authority.
- Bounded provider/common diff scan found no changes under common EmitC/export,
  target export, RVV EmitC route provider, or RVV EmitC route planning files.
- `git diff --check` passed.
- No `ssh rvv` evidence was collected because this task did not change or
  claim emitted target sequence, runtime correctness, ABI behavior, or
  performance.

## Definition Of Done

The RVV plugin production route/emission entries materialize supported
pre-realized selected-body families through a reusable RVV-owned route-entry
bridge before collecting route facts; compare/select and one base-memory form
prove the family-neutral path; realized bodies feed existing statement-plan/
provider/common EmitC machinery; unsupported or malformed cases fail closed;
focused and full checks pass; Trellis task metadata is truthful; and the
completed work is committed as one coherent change.
