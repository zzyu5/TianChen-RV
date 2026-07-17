# Stage2 RVV MAcc route-family provider-plan owner boundary

## Goal

Move production-active MAcc/accumulation route-family provider-plan ownership
for plain MAcc, scalar-broadcast MAcc, and computed-mask MAcc out of central
`RVVEmitCRoutePlanning.cpp` into an explicit RVV-owned MAcc owner boundary.
Central route planning should remain responsible for shared route analysis,
typed config/capability facts, materialization facts, operand-binding facts,
route descriptions, mirror metadata, and top-level neutral orchestration; it
must not keep MAcc-specific provider-plan authority disguised as shared
planning.

## Direction Source

- Direction title: `Switch: Stage2 RVV MAcc route-family provider-plan owner boundary`.
- Module owner: RVV plugin-local MAcc/accumulation route-family provider-plan
  owner boundary for plain MAcc, scalar-broadcast MAcc, and computed-mask MAcc.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b2a500e1 rvv: move segment2 route plans to owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The predecessor task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-segment2-route-family-provider-plan-owner-boundary/prd.md`
  moved segment2 route-family provider-plan construction into
  `RVVEmitCSegment2RouteFamilyPlanOwners.h/cpp`, kept shared plan data in
  `RVVEmitCRoutePlanning.h`, and passed `check-tianchenrv` 464/464.
- The segment2 owner boundary now provides the local extraction pattern:
  explicit owner header/source, registry entries, exact-one selection,
  consumer predicates, provider-plan builder, fail-closed diagnostics, CMake
  wiring, and focused C++ owner tests.
- `.trellis/spec/index.md` and `.trellis/spec/extension-plugins/index.md`
  require Stage2 RVV work to start from typed `tcrv_rvv` bodies and plugin route
  providers, not route ids, legacy i32 helpers, descriptors, or source
  front-door metadata.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines MAcc and
  computed-mask accumulation as selected-body typed-route work and requires
  provider/statement-plan boundaries to consume typed body/config/runtime
  facts, route-family plans, route-control facts, and operand-binding facts
  before common EmitC.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires
  `TCRVEmitCLowerableRoute` to be provider-built and common EmitC to remain
  neutral; common EmitC must not infer dtype, SEW, LMUL, policy, operation
  kind, ABI order, intrinsic spelling, or route support.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` still exposes
  `RVVSelectedBodyMAccRouteFamilyOwner`,
  `getRVVSelectedBodyMAccRouteFamilyOwners`,
  `isRVVSelectedBodyMAccRouteFamilyConsumer`, and the plain/scalar/computed
  MAcc provider-plan verifier declarations as central planning APIs.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` still defines the plain
  MAcc, scalar-broadcast MAcc, and computed-mask MAcc consumer predicates,
  provider-plan verification, MAcc owner registry, aggregate MAcc selector, and
  exact/multiple owner diagnostics.
- The computed-mask accumulation verifier currently serves both computed-mask
  MAcc and computed-mask standalone accumulation. Moving it must preserve
  standalone reduction/accumulation top-level orchestration while removing the
  MAcc-specific authority from central route planning.
- `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp` uses the plain and
  scalar-broadcast MAcc consumer predicates and must include the new owner
  boundary after the move.
- `test/Plugin/RVVExtensionPluginTest.cpp` already has focused MAcc owner
  registry, missing-plan, stale-plan, computed-mask accumulation, plain MAcc,
  scalar-broadcast MAcc, and top-level owner tests that can be rewired to the
  new owner header and extended if the extraction exposes a coverage gap.

## Requirements

1. Add an explicit RVV-owned MAcc route-family provider-plan owner header and
   implementation module, or an equivalent owner-local component matching the
   established segment2 owner pattern.
2. Move `RVVSelectedBodyMAccRouteFamilyOwner`, MAcc owner registry declarations
   and definitions, MAcc aggregate consumer selection, exact/multiple owner
   diagnostics, and MAcc provider-plan verification authority out of
   `RVVEmitCRoutePlanning.h/cpp`.
3. Move the plain MAcc, scalar-broadcast MAcc, and computed-mask MAcc consumer
   predicates plus provider-plan verification functions into the MAcc owner
   boundary.
4. Preserve the existing shared route-family plan structs, route analysis,
   route materialization facts, operand-binding facts, statement-plan structs,
   route descriptions, and mirror metadata in the shared route-planning layer
   when they are used by multiple owners or consumers.
5. Preserve the existing selected typed `tcrv_rvv` route path, provider facts,
   mirrors, artifact ABI, generated-bundle behavior, and statement-plan
   consumption behavior for plain MAcc, scalar-broadcast MAcc, computed-mask
   MAcc, and unrelated standalone accumulation routes.
6. Keep central `RVVEmitCRoutePlanning.cpp` limited to neutral orchestration:
   top-level owner registries, shared analysis/fact helpers, and calls into
   explicit owner boundaries. It must not retain MAcc-specific registry entries
   or provider-plan validation bodies.
7. Preserve fail-closed diagnostics for missing MAcc route-family plans, stale
   non-consumer MAcc plans, mismatched route-family plan mirrors, stale
   runtime/control/type/intrinsic/layout facts, missing or wrong
   operand-binding closure, computed-mask MAcc suffix/provenance mismatches,
   and multiple selected MAcc owners.
8. Wire all production consumers and focused tests through the new MAcc owner
   header, including `RVVEmitCControlPolicyPlanOwners.cpp`,
   `RVVEmitCRoutePlanning.cpp` top-level orchestration, and
   `RVVExtensionPluginTest.cpp`.
9. Update CMake so the new owner source is compiled into the RVV EmitC route
   provider library.
10. Do not add new MAcc variants, reductions, widening-dot coverage, segment2
    behavior, source-front-door routes, high-level frontend/Linalg lowering,
    one-intrinsic wrapper dialects, dtype/LMUL clone batches, dashboards,
    performance tuning, or evidence-only work.

## Acceptance Criteria

- [x] New RVV-owned MAcc owner module/header owns MAcc provider-plan
      verification and owner selection for plain, scalar-broadcast, and
      computed-mask MAcc.
- [x] `RVVEmitCRoutePlanning.cpp` no longer defines the moved MAcc owner
      registry, MAcc owner struct, MAcc-specific consumer predicates, or
      MAcc-specific provider-plan verifier bodies.
- [x] `RVVEmitCRoutePlanning.h` no longer exposes the MAcc owner registry as a
      central planning-owned API; consumers use the explicit MAcc owner header.
- [x] Central route planning retains only shared structs/facts and neutral
      top-level orchestration, including any calls into the new MAcc owner
      boundary needed by aggregate math-family verification.
- [x] Focused C++ tests prove owner registry membership, owner order/names,
      non-null hooks, exact classification for plain/scalar/computed-mask MAcc,
      empty/non-consumer behavior, and fail-closed missing/stale/mismatched
      provider-plan diagnostics.
- [x] Representative generated-bundle dry-runs pass for plain MAcc,
      scalar-broadcast MAcc, and computed-mask MAcc selected-body paths.
- [x] Segment2 owner boundary non-regression passes after adding the MAcc owner
      module.
- [x] Bounded symbol/ref scan shows moved MAcc owner symbols concentrated in the
      new owner module and central `RVVEmitCRoutePlanning.cpp` retaining only
      neutral calls/declarations where necessary.
- [x] Bounded authority scan over touched production/test files finds no new
      legacy-i32, source-front-door/source-artifact, descriptor-derived,
      direct-route-entry-only, route-id, artifact-name, exact-intrinsic,
      bare status/supported, script-derived, ABI-string, common-EmitC, or
      mirror-only authority drift.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin C++ build/test passes.
- [x] `tcrv-opt` and `tcrv-translate` build.
- [x] `check-tianchenrv` passes, or the exact blocker is recorded.
- [x] `ssh rvv` is not required unless this round changes executable behavior
      or makes a new runtime, correctness, or performance claim.

## Technical Approach

The intended production flow after this task is:

```text
selected typed RVV MAcc body
  -> route analysis facts
  -> route-family provider-plan verification
  -> explicit RVV-owned MAcc owner boundary
  -> validated MAcc provider facts and mirrors
  -> route materialization and operand-binding facts
  -> migrated MAcc statement-plan owner
  -> RVVEmitCRouteProvider builds TCRVEmitCLowerableRoute
  -> common EmitC materialization
```

Implementation will add `RVVEmitCMAccRouteFamilyPlanOwners.h/cpp` under the
RVV plugin include/source tree. The header will expose the MAcc owner struct,
registry getter, aggregate consumer predicate, and provider-plan verifier
entry points. The source file will contain the moved plain/scalar/computed
MAcc consumer predicates and verification logic plus the aggregate exact-one
owner selection. Central route planning will include this boundary only for
top-level neutral orchestration and materialization helper predicates that
need to classify an MAcc-selected route.

## Validation Plan

1. Validate task context with
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-stage2-rvv-macc-route-family-provider-plan-owner-boundary`.
2. Build focused plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused plugin C++ coverage:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
5. Run representative generated-bundle dry-runs for plain MAcc,
   scalar-broadcast MAcc, and computed-mask MAcc selected-body paths.
6. Run a segment2 generated-bundle or C++ non-regression covering the existing
   segment2 owner boundary.
7. Run bounded central/owner symbol scans and authority-drift scans.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Out of Scope

- New MAcc/accumulation operations, new coverage classes, dtype/SEW/LMUL
  expansion, or runtime behavior changes.
- Reworking selected-body realization semantics, statement-plan construction,
  target artifact validation, common EmitC materialization, or runtime
  executable behavior unless a compile failure exposes a direct ownership
  dependency.
- Moving shared route-analysis structs, neutral fact containers, or top-level
  aggregate orchestration into the MAcc owner just because MAcc uses them.
- IME, Offload, TensorExt, Toy, Template, future plugin work, dashboards,
  artifact indexes, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Added `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`
  and `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp` as the
  explicit RVV-owned MAcc route-family provider-plan owner boundary.
- Moved `RVVSelectedBodyMAccRouteFamilyOwner`,
  `getRVVSelectedBodyMAccRouteFamilyOwners`,
  `isRVVSelectedBodyMAccRouteFamilyConsumer`, plain/scalar/computed-mask MAcc
  consumer predicates, and MAcc provider-plan verifier bodies out of central
  `RVVEmitCRoutePlanning.h/cpp`.
- Kept shared MAcc/accumulation route-family plan structs, route analysis,
  route materialization facts, operand-binding facts, statement-plan structs,
  route descriptions, and mirror metadata in the central shared route-planning
  layer.
- Central `RVVEmitCRoutePlanning.cpp` now includes the MAcc owner boundary for
  neutral aggregate orchestration and has only thin structural validation
  wrappers for plan validators that remain part of route-family plan derivation.
- Updated `RVVEmitCControlPolicyPlanOwners.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `lib/Plugin/RVV/EmitC/CMakeLists.txt` to consume/build the new MAcc owner
  module.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with a durable
  `MAcc Route-Family Provider-Plan Owner Boundary` contract covering scope,
  signatures, contracts, validation/error matrix, tests, and wrong/correct
  cases.
- Checks run:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-stage2-rvv-macc-route-family-provider-plan-owner-boundary`;
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tcrv-opt tcrv-translate -j2`;
  focused lit generated-bundle dry-runs for plain MAcc, scalar-broadcast MAcc,
  computed-mask MAcc, runtime-scalar computed-mask MAcc, and computed-mask
  segment2 update non-regression; bounded symbol/authority scans;
  `git diff --check`; and
  `cmake --build build --target check-tianchenrv -j2`.
- Representative generated-bundle dry-run result: 5/5 passed for
  `macc_add`, `scalar_broadcast_macc_add`, `computed_masked_macc_add`,
  `runtime_scalar_cmp_masked_macc_add`, and explicit
  `computed_masked_segment2_update_unit_load`.
- Final `check-tianchenrv` result: 464/464 passed.
- `clang-format` was not available in this environment, but `git diff --check`
  passed.
- `ssh rvv` was not run because this task changed owner placement and spec
  documentation only, with no new runtime, correctness, or performance claim.
