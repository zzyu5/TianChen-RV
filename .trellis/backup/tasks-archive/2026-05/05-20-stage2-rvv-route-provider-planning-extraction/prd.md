# Stage2 RVV route-provider planning extraction

## Goal

Extract the active realized typed `tcrv_rvv` body -> RVV route plan boundary
out of the monolithic `RVVEmitCRouteProvider.cpp` implementation into a
plugin-local provider planning module consumed by the RVV EmitC route provider.
The provider remains the RVV route authority and the common
`TCRVEmitCLowerableRoute` builder, but route classification, typed fact
collection, fail-closed planning diagnostics, metadata/header/intrinsic
planning, and construction-route description support for the existing realized
generic, strided, masked, macc, and reduce paths should live behind an explicit
RVV-owned planning API.

This is a bounded architecture/ownership extraction over existing Stage2 route
behavior. It is not a new RVV operation coverage task and not a helper-only
cleanup.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `90497af1 rvv: extract selected body realization`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as
  `.trellis/tasks/05-20-stage2-rvv-route-provider-planning-extraction`.
- `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md` require the RVV
  authority chain:
  selected `tcrv.exec` RVV variant -> typed/realized low-level `tcrv_rvv` body
  -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- The previous selected-body extraction task added
  `RVVSelectedBodyRealization` and left route authority in
  `RVVEmitCRouteProvider.cpp`.
- Current `RVVEmitCRouteProvider.cpp` is still monolithic: it owns typed body
  route-slice collection, runtime ABI binding extraction, operation/memory-form
  classification, typed config validation, construction route lookup, route
  description verification, RVV header/type/intrinsic planning, and final
  `TCRVEmitCLowerableRoute` materialization.
- Current supported realized route families include generic add/sub/mul and
  compare/select, RHS broadcast arithmetic, strided add, masked add, macc add,
  and reduce add. The current task must preserve those behaviors rather than
  expanding coverage.
- Common EmitC/materialization and target export must remain neutral consumers
  of provider-built routes and mirrors.
- Stage1 guardrails remain active: do not reintroduce positive `RVVI32M1`,
  `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor, artifact-name, status, or exact
  intrinsic spelling as route authority.

## Requirements

1. Add a plugin-local RVV EmitC route-planning owner with a narrow API consumed
   by `RVVEmitCRouteProvider.cpp`.
2. Move realized typed body route planning out of the monolithic provider:
   selected-body slice collection, runtime ABI fact extraction, operation and
   memory-form classification, typed config checks, construction route lookup,
   route description planning, and fail-closed diagnostics.
3. Keep `RVVEmitCRouteProvider.cpp` as the public route authority and the code
   that turns a verified plan into `TCRVEmitCLowerableRoute`.
4. Preserve positive behavior for existing realized routes: unit-stride
   binary, RHS broadcast arithmetic, strided add, masked add, macc add, reduce
   add, and compare/select where already supported.
5. Preserve negative fail-closed behavior for missing/malformed selected RVV
   variants, missing explicit typed RVV body, unsupported body ops, stale
   legacy `tcrv_rvv.i32_*` body ops, runtime ABI role mismatches,
   inconsistent typed config facts, unsupported memory forms, and stale
   route-description mirrors.
6. Ensure route planning consumes only typed `tcrv_rvv` body/config/runtime
   facts plus RVV capability/construction contracts. Do not make route ids,
   artifact names, source-front-door metadata, descriptors, or common EmitC
   code into route authority.
7. Keep common EmitC/export neutral; do not move RVV semantic planning into
   `lib/Conversion/EmitC` or target artifact metadata.
8. Do not add new RVV operation classes, dtype/LMUL coverage, high-level
   frontend lowering, source-front-door positive routes, Template/Toy/future
   plugin work, dashboards, report-only inventory, runtime claims, correctness
   claims, or performance claims.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata truthfully describe a
      route-provider planning extraction and do not drift into coverage
      expansion.
- [x] A consumed RVV plugin-local planning module owns route slice collection,
      typed fact validation, fail-closed planning diagnostics, route
      description/header/type/intrinsic planning, and construction-route
      description verification for the current supported realized routes.
- [x] `RVVEmitCRouteProvider.cpp` consumes the extracted plan and no longer
      owns the large body-specific planning tables/analysis inline.
- [x] Provider authority remains RVV-local: public functions such as
      `describeRVVSelectedBodyEmitCRoute` and
      `buildRVVSelectedBodyEmitCLowerableRoute` still return provider-built
      descriptions/routes.
- [x] Common EmitC/export remains unchanged or only consumes provider payloads
      neutrally; no common RVV semantic derivation is added.
- [x] Existing positive provider/materialization behavior remains covered by
      focused evidence for at least unit-stride binary plus one nontrivial
      strided/masked/macc/reduce route.
- [x] Selected-body realization positive fixtures still reach provider routes
      after extraction.
- [x] Existing negative fail-closed provider/selected-body tests still pass.
- [x] Active-authority scan confirms no active `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door /
      source-seed, descriptor/direct-C/source-export, or common/export RVV
      semantic authority is reintroduced.
- [x] Focused build, C++ unit, lit, script self-test or dry-run where relevant,
      `git diff --check`, task validation, and `check-tianchenrv` pass if
      shared provider behavior changes.

## Non-Goals

- No new RVV operation coverage, dtype/LMUL expansion, or operation-family
  growth.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV route or source-seed evidence mode.
- No Template/Toy/TensorExtLite/IME/Offload/future-plugin work.
- No descriptor-driven computation or descriptor-driven C/source export.
- No compatibility wrapper preserving legacy i32 route authority.
- No runtime, correctness, or performance claim without fresh `ssh rvv`
  evidence. This extraction should preserve existing behavior and does not
  require fresh hardware evidence unless executable behavior changes.

## Validation Plan

1. Validate task context with
   `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-route-provider-planning-extraction`.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused C++ tests:
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
4. Run focused lit for representative provider/materialization and
   selected-body realization handoff coverage: unit-stride binary plus at least
   one strided, masked, macc, or reduce route, and selected-body/provider
   negative fixtures.
5. Run `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`; run
   focused generated-bundle dry-runs only if script-facing or artifact fixture
   paths are touched.
6. Run `rtk git diff --check`.
7. Run an active-authority scan over active RVV include/lib/script/test paths.
8. Run `rtk cmake --build build --target check-tianchenrv -j2` if focused
   validation shows shared provider behavior changed, or if the extraction is
   broad enough that the full project lit gate is the cleanest evidence.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`

Prior context read:

- `.trellis/workspace/codex/journal-11.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-selected-body-realization-extraction/prd.md`
- Current memory note for TianchenRV RVV Stage1/Stage2 authority boundaries.

Initial code surface inspected:

- `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/EmitC/CMakeLists.txt`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`

## Implementation Results

- Added `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` as the RVV plugin-local
  planning owner for realized typed selected-body routes.
- Moved route slice collection, runtime ABI binding extraction, operation and
  memory-form classification, typed config validation, construction route
  lookup, route profile/header/type/intrinsic planning, route description
  verification, config artifact metadata construction, and fail-closed planning
  diagnostics out of `RVVEmitCRouteProvider.cpp`.
- Kept `RVVEmitCRouteProvider.cpp` as the consumer that turns a verified
  `RVVSelectedBodyRouteAnalysis` into `TCRVEmitCLowerableRoute` payload:
  headers, ABI mappings, source provenance, call-opaque steps, and loop
  materialization remain provider-owned.
- Preserved the current supported route families: generic add/sub/mul,
  compare/select, RHS broadcast arithmetic, strided add, masked add, macc add,
  and reduce add.
- Left common EmitC/export code unchanged; no RVV semantic planning moved into
  `lib/Conversion/EmitC` or target artifact metadata.
- Did not add operation coverage, dtype/LMUL expansion, source-front-door
  positive routing, descriptor/direct-C/source-export behavior, runtime
  behavior, correctness claims, or performance claims.

## Validation Results

- [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-route-provider-planning-extraction`
- [x] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [x] `rtk cmake --build build --target tianchenrv-rvv-dialect-test -j2`
- [x] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `rtk build/bin/tianchenrv-construction-protocol-common-test`
- [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [x] `rtk build/bin/tianchenrv-rvv-dialect-test`
- [x] Focused provider/materialization/selected-body lit from `build/test`:
      21/21 passed for unit-stride binary, strided, masked, macc, reduce,
      pre-realized selected-body positive fixtures, and negative
      provider/selected-body fail-closed fixtures.
- [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused generated-bundle dry-run lit from `build/test`: 6/6 passed for
      explicit selected-body and pre-realized unit-stride/strided/masked/macc/
      reduce paths.
- [x] `rtk git diff --check`
- [x] Active-authority scan over the changed RVV planning/provider/task files:
      only provider-derived intrinsic leaves, an existing fail-closed
      `tcrv_rvv.i32_` rejection branch, and PRD guardrail text matched; no new
      positive legacy/source-front-door/descriptor/common-export route
      authority was introduced.
- [x] `rtk cmake --build build --target check-tianchenrv -j2`: 176/176 lit
      tests passed.
- [x] `clang-format` was not available in the environment; formatting was
      checked with `git diff --check` and a focused rebuild.

## Definition Of Done

- [x] Extracted planning module is consumed by the production/default RVV
      provider path.
- [x] Focused checks pass and failures are repaired.
- [x] Task status and journal notes are truthful.
- [x] Task is finished/archived if complete.
- [x] One coherent commit records the PRD, extraction, validation, and task
      closeout, or the task remains open with an exact continuation point.
