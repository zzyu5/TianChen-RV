# Stage2 RVV computed-mask select route-control provider-plan integration

## Goal

Make the existing computed-mask select RVV route family consume the shared
`RVVSelectedBodyRouteControlProviderPlan` before provider/statement-plan
construction. The already-supported `computed_mask_select` family path must use
the same RVV-owned structural boundary for runtime AVL/VL, SEW/LMUL, tail
policy, mask policy, runtime ABI order, materialization facts, operand binding,
producer-source facts, selected capability facts, and same-analysis ownership
that is now used by ordinary elementwise, scalar-broadcast elementwise, base
memory, standalone reduction, scalar MAcc, and plain compare/select.

This is a production-path migration for an existing route family. It must not
add new compare/select operation kinds, new dtype/LMUL clone batches, computed
mask memory, conversion, contraction, frontend lowering, source-front-door
positive routes, dashboards, broad smoke matrices, or evidence-only fixture
copying.

## Source Direction

- Direction title: `Stage2 RVV computed-mask select route-control provider-plan
  integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by
  the existing computed-mask select route-family/provider path.
- Initial HEAD: `d31b9e48 rvv: consume route control plan in plain compare
  select`.
- Initial worktree: clean.
- No `.trellis/.current-task` existed at session start, so this task records
  the Hermes/User brief as the Trellis work item before source edits.

## What Is Already Known

- The shared route-control provider-plan boundary is RVV plugin-local provider
  input after route-family verification and materialization facts, and before
  statement-plan construction.
- Recent commits completed route-control consumption for ordinary elementwise,
  scalar-broadcast elementwise, base memory, standalone reduction, scalar MAcc,
  and plain compare/select.
- `RVVSelectedBodyComputedMaskSelectRouteFamilyPlan` already exists and carries
  runtime-control, producer-source, runtime ABI order, operand-binding, and
  provider mirror fields.
- The compare/select statement plan already has a family-owned provider-ready
  statement sequence surface covering plain compare/select and computed-mask
  select variants. The current gap is that computed-mask select has not yet
  joined the shared route-control provider-plan before that statement plan is
  returned.
- Specs require route ids, metadata, ABI strings, artifact names, descriptors,
  scripts, common EmitC, source-front-door markers, and legacy i32 spellings to
  remain non-authoritative mirrors or invalid inputs for AVL/VL, policy, dtype,
  predicate, and compute semantics.

## Requirements

1. Production C++ planning/provider behavior must make the existing
   computed-mask select route-family path consume
   `RVVSelectedBodyRouteControlProviderPlan` before compare/select
   statement-plan construction.
2. The computed-mask select route-control plan must join and validate:
   - same-analysis typed config facts;
   - same-analysis selected target capability facts;
   - the owning runtime AVL/VL control plan;
   - runtime ABI order and runtime `n`/AVL binding;
   - SEW, LMUL, tail policy, and mask policy;
   - computed-mask select family and materialization facts;
   - producer-source facts for the computed compare mask;
   - elementwise/select operand-binding facts;
   - route description and provider mirror consistency.
3. The statement-plan boundary must fail closed before route construction when
   computed-mask select requires route-control facts but any control plan,
   materialization facts, family facts, producer-source facts, operand-binding
   facts, runtime ABI facts, policy facts, selected capability facts, or mirrors
   are stale, missing, or mismatched.
4. If the whole computed-mask select family is too large for one round, finish
   the vector-compare-producer `computed_mask_select` subpath as the coherent
   owner and leave an exact continuation point for runtime-scalar and dual
   compare-mask variants.
5. The provider route must still attach provider-built compare/select statement
   plans to `TCRVEmitCLowerableRoute`; common EmitC and target artifact code
   must remain neutral materialization/mirror consumers only.
6. Any emitted object/header/artifact metadata touched by this task must use
   explicit mirror labels. Metadata must remain mirror-only after provider
   route construction.

## Acceptance Criteria

- [x] `RVVSelectedBodyRouteControlProviderPlan` includes a computed-mask select
      consumer flag or equivalent structural marker.
- [x] Computed-mask select route-control plan construction joins the verified
      computed-mask select family plan, same-analysis materialization facts,
      typed config facts, selected target capability facts, producer-source
      facts, operand-binding facts, runtime AVL/VL control plan, and route
      description mirrors.
- [x] `getRVVSelectedBodyCompareSelectRouteStatementPlan(...)` consumes the
      route-control provider plan for computed-mask select before building the
      compare/mask/select/store statement sequence.
- [x] Plain compare/select behavior remains on its existing route-control path;
      runtime-scalar and dual-compare variants are migrated only if code
      inspection proves they are inseparable from the coherent computed-mask
      select owner.
- [x] Focused C++ tests prove positive computed-mask select route-control
      consumption and provider/statement-plan attachment.
- [x] Focused negative C++ tests fail closed for representative stale or
      missing dependencies, including missing/stale route control facts, missing
      runtime AVL binding or wrong AVL role, runtime ABI mirror mismatch,
      policy mismatch, unsupported selected capability/config, missing
      computed-mask select family or materialization plan, stale producer-source
      marker, stale operand binding, and invalid masked/unmasked policy use.
- [x] Focused lit/FileCheck or generated-header checks cover explicit
      mirror-only labels if route metadata/header mirrors change.
- [x] Generated-bundle dry-run evidence is rerun if emitted output changes.
      Real `ssh rvv` correctness is rerun for affected computed-mask select
      behavior if executable behavior or ABI evidence changes; otherwise the
      final report states why historical runtime evidence is sufficient.
- [x] Run `git diff --check`.
- [x] Run focused RVV plugin tests and focused lit/FileCheck for touched
      computed-mask select route-control behavior.
- [x] Run `check-tianchenrv`, or report the exact blocker.
- [x] Run a bounded authority scan over touched planning/provider/test/spec or
      target/script files for name-, route-id-, metadata-, descriptor-,
      ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or
      legacy-i32-derived AVL/VL/mask/policy authority.

## Out Of Scope

- New compare/select operation kinds or predicates.
- Computed-mask memory, conversion, contraction, reduction, new dtype/LMUL clone
  batches, high-level frontend lowering, source-front-door positive routes,
  dashboards, broad smoke matrices, or evidence-only fixture copying.
- Treating runtime counts, route ids, metadata fields, manifests, artifact
  names, ABI strings, descriptors, scripts, tests, common EmitC, target artifact
  code, or legacy i32 spellings as AVL/VL, mask, policy, dtype, predicate, or
  compute authority.
- Changing computation semantics, dispatch/fallback behavior, or emitted
  statement order unless required by the control boundary and explicitly
  evidenced.

## Evidence Plan

1. Inspect current planning/provider/test code:
   - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
   - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
   - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
   - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
   - `test/Plugin/RVVExtensionPluginTest.cpp`
   - existing computed-mask select fixtures.
2. Implement the smallest production-path migration that makes computed-mask
   select a route-control consumer.
3. Add focused positive and fail-closed tests in the existing RVV plugin test
   style.
4. Update the RVV plugin spec after the production path and tests establish the
   new consumer.
5. Run focused build/tests, generated evidence if needed, authority scan,
   `git diff --check`, and `check-tianchenrv`.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-scalar-broadcast-route-control/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-scalar-macc-route-control-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-elementwise-route-control-plan/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-plain-compare-select-route-control-plan/prd.md`.
- Historical memory note used only to orient the current Stage2 RVV surface:
  prior restart context had already recorded computed-mask select as a Stage2
  executable slice, while warning against legacy i32/source-front-door drift.

## Implementation Result

Completed for the full active computed-mask select route family: vector
`computed_mask_select`, runtime-scalar `runtime_scalar_cmp_select`, and
runtime-scalar dual `runtime_scalar_dual_cmp_mask_and_select`.

- Added `controlsComputedMaskSelect` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Added computed-mask select to the route-control provider-plan consumer set.
- `getRVVSelectedBodyRouteControlProviderPlan(...)` now requires the verified
  computed-mask select route-family plan from the same selected route analysis
  before exposing runtime AVL/VL, typed config, selected target capability,
  policy, runtime ABI, and mirror facts for computed-mask select consumers.
- `getRVVSelectedBodyCompareSelectRouteStatementPlan(...)` now requires the
  computed-mask select route-control provider plan before constructing the
  compare/mask/select/store statement sequence for computed-mask select family
  routes.
- Existing plain compare/select route-control behavior remains isolated on
  `controlsPlainCompareSelect`; no new compare/select operation kinds or
  emitted statement-order changes were introduced.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so route-control and
  compare/select statement-plan specs list computed-mask select as an adopted
  route-control consumer.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-computed-mask-select-route-control-provider-plan`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Focused lit/FileCheck from `build/test` passed 16/16 after rerun:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='computed-mask-select|cmp-select|compare-select'`.
  The first attempt was invalid because it ran concurrently with `tcrv-opt`
  relinking and hit `Permission denied`; rerunning after the tools linked
  passed.
- Generated-bundle dry-run passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_computed_mask_select_route_control --run-id pre-realized-computed-mask-select-route-control-dry --overwrite --op-kind computed_mask_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`.
- `git diff --check` passed.
- Added-line authority scan over touched planning/provider/test/spec files
  found only explicit mirror-only/spec-negative wording and the intentional
  `metadata_n` negative test value; no new positive name-, route-id-,
  metadata-, descriptor-, ABI-string-, script-, artifact-, common-EmitC-,
  source-front-door-, or legacy-i32-derived AVL/VL/mask/policy authority was
  added.
- `cmake --build build --target check-tianchenrv -j2` passed 379/379.
- No new `ssh rvv` correctness claim was made because the emitted statement
  sequence and ABI/output materialization did not change. Historical
  computed-mask select `ssh rvv` evidence remains the runtime evidence for the
  unchanged executable behavior; this round only adds a provider-plan
  validation gate before statement construction.
