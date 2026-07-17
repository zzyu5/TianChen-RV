# Stage2 RVV runtime AVL/VL and policy provider-plan boundary

## Goal

Introduce one bounded production C++ owner boundary for RVV route-control facts
before migrated route construction. Existing route-supported base memory
movement and standalone reduction families must consume validated runtime
AVL/VL, SEW, LMUL, tail policy, mask policy, config contract, runtime ABI, and
selected target capability facts through an RVV-local provider plan instead of
rechecking or mirroring those facts ad hoc in each family/provider path.

## Direction Source

- Direction title: `Stage2 RVV runtime AVL/VL and policy provider-plan
  boundary`.
- Module owner: RVV plugin-local route-control provider plan for runtime
  AVL/VL, SEW/LMUL, tail policy, and mask-policy facts consumed by migrated
  route families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `74048fde rvv: close base memory production owner boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  provided Direction Brief before source edits.

## Current Repository Facts

- Specs require selected `tcrv.exec` RVV variant -> typed or realized
  `tcrv_rvv` body -> RVV plugin-owned legality / selected-body realization /
  route provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target
  artifact, with target artifacts and metadata as mirrors only.
- `RVVRuntimeAVLVLControlPlan` already derives and verifies runtime `n`/AVL,
  `setvl`, `with_vl`, SEW/LMUL, tail/mask policy, config contract,
  runtime ABI order, and loop-control names.
- Family plans for base memory movement and standalone reduction already carry
  `runtimeControlPlan`, family mirrors, runtime ABI parameters, typed config
  leaves, and statement-plan dependencies.
- Recent production closures added materialization facts, math/memory
  operand-binding facts, a base memory provider plan, standalone reduction
  statement-plan reuse, and migrated statement-plan aggregate consumption.
- The current gap is shared control-fact ownership across families: the
  provider has materialization and binding owners, but no route-control
  provider plan that explicitly joins typed config facts, runtime AVL/VL
  control, selected target capability facts, and family control consumers.

## Scope

1. Add or tighten a production C++ route-control provider-plan boundary in RVV
   planning/provider code.
2. Apply the boundary to base memory movement plus one math family:
   standalone reduction.
3. Keep other migrated families on their existing provider boundaries unless a
   tiny neutral hook is required for compile consistency.
4. Store or pass selected target capability facts structurally in route
   analysis so the route-control plan validates capability/config/policy from
   provider-owned facts, not target metadata mirrors.
5. Preserve common EmitC neutrality and target artifact mirror-only behavior.

## Requirements

1. The route-control provider plan must be built from the same selected route
   analysis as route materialization facts.
2. The plan must validate typed config facts, selected target capability facts,
   and the family runtime-control plan before provider route construction.
3. Base memory movement provider-plan construction must consume the
   route-control plan before constructing or attaching its ordered statement
   plan.
4. Standalone reduction statement-plan construction must consume the
   route-control plan before constructing or attaching its ordered statement
   plan.
5. Missing runtime AVL binding, invalid/stale runtime control plan,
   typed-config mismatch, stale materialization facts, stale target capability
   facts, policy mismatch, or unsupported family/control pairing must fail
   closed with diagnostics naming the route-control provider plan.
6. Route descriptions, target artifacts, generated headers, scripts, and tests
   may mirror provider-built control facts only after provider construction.
7. Do not add new RVV operation coverage, dtype/LMUL clone batches, frontend
   lowering, source-front-door positive routes, broad smoke dashboards,
   descriptor/direct-C/source-export paths, or legacy i32 route authority.

## Acceptance Criteria

- [x] Task context references the RVV plugin spec, EmitC route spec, and
      relevant archived base memory, standalone reduction, typed-config, and
      mask/tail policy tasks.
- [x] `RVVSelectedBodyRouteAnalysis` or equivalent provider input carries
      selected target capability facts structurally, not only description
      mirrors.
- [x] A route-control provider plan validates typed config facts,
      target-capability facts, runtime AVL/VL control, SEW/LMUL, tail policy,
      mask policy, config contract, runtime ABI order, and setvl/with_vl
      control names for base memory and standalone reduction consumers.
- [x] Base memory movement provider-plan construction consumes the
      route-control provider plan before statement-plan ownership is returned.
- [x] Standalone reduction statement-plan construction consumes the
      route-control provider plan before statement-plan ownership is returned.
- [x] Focused C++ tests prove positive route-control consumption for
      `strided_load_unit_store` or `masked_unit_store`, and for
      `standalone_reduce_add`.
- [x] Focused C++ tests prove fail-closed diagnostics for missing/stale route
      control facts, stale selected target capability facts, and at least one
      policy/control mismatch before route statement construction.
- [x] Representative lit/FileCheck or generated-header checks continue to show
      provider-built control facts and explicit mirror labels; no target/script
      path becomes authority.
- [x] No new `ssh rvv` runtime claim is made unless emitted executable
      behavior changes. If no emitted behavior changes, the final report states
      why historical runtime evidence remains sufficient.
- [x] Bounded scans over touched files find no new name-, route-id-,
      metadata-, descriptor-, ABI-string-, script-, artifact-, common-EmitC-,
      source-front-door-, or legacy-i32-derived AVL/VL or policy authority.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin tests and `check-tianchenrv` pass, or an exact
      blocker is recorded.
- [x] Task status, journal, archive, and one coherent commit complete if this
      task finishes.

## Non-Goals

- No new RVV operation family or Stage 2 coverage expansion.
- No new frontend lowering or source-front-door positive route.
- No descriptor-driven, direct-C, source-export, common-EmitC, script, target
  artifact, route-id, ABI-string, metadata, or test-name authority.
- No performance claim.
- No subagents, spawned agents, or parallel-agent workflow.

## Validation Plan

1. Validate and start the Trellis task.
2. Build and run `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck only if route metadata/header fixtures change.
4. Run `git diff --check`.
5. Run bounded authority scans over touched RVV planning/provider/test/target
   files.
6. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-base-memory-movement-production-family-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-standalone-reduction-route-family-boundary-reuse/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-typed-config-artifact-executable-closure/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-mask-tail-policy-route-closure/prd.md`.
- Workspace journal read: `.trellis/workspace/codex/journal-15.md` sessions
  for standalone reduction and base memory movement closure.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Implementation Result

Completed for the bounded base memory movement plus standalone reduction
consumer set.

- Added structural `selectedTargetCapabilityFacts` to
  `RVVSelectedBodyRouteAnalysis`, preserving selected RVV capability/provider
  facts beside typed config facts instead of relying only on route description
  mirror strings.
- Added `RVVSelectedBodyRouteControlProviderPlan` and
  `getRVVSelectedBodyRouteControlProviderPlan(...)`, which joins the
  same-analysis typed config facts, selected target capability facts, and the
  owning family `RVVRuntimeAVLVLControlPlan`.
- The route-control provider plan validates SEW, LMUL, tail policy, mask
  policy, config contract, runtime VL contract, runtime AVL source, runtime ABI
  order, `setvl`/`with_vl` control names, loop-control names, remaining AVL,
  pointer advance, bounded-slice, multi-VL, and selected target capability
  legality before provider route construction.
- Base memory movement provider-plan construction now requires the
  route-control provider plan before constructing/returning the ordered base
  memory statement plan.
- Standalone reduction statement-plan construction now requires the
  route-control provider plan before constructing/returning the ordered
  standalone reduction statement plan.
- Updated the RVV plugin spec with the durable Route-Control Provider-Plan
  Boundary, including signatures, contracts, validation matrix, cases, tests,
  and wrong-vs-correct examples.
- No EmitC statement sequence, target ABI, generated-bundle script behavior, or
  runtime executable behavior changed. No new `ssh rvv` correctness claim was
  made; historical runtime evidence for the unchanged executable paths remains
  sufficient.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-runtime-vl-policy-provider-plan`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] Focused lit/FileCheck from `build/test`:
      `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='pre-realized-selected-body-artifact-(strided-load-unit-store|standalone-reduce-add)'`
- [x] Bounded authority scan over touched planning/provider/test/spec files.
      The only matches were existing spec prohibition text and existing
      negative/fail-closed legacy tests, not new AVL/VL or policy authority.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.
