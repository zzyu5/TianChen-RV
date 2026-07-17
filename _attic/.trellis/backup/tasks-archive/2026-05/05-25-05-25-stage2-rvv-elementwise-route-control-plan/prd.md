# Stage2 RVV elementwise arithmetic route-control provider-plan integration

## Goal

Make the existing plain RVV elementwise arithmetic route path consume the shared
RVV route-control provider plan before elementwise statement-plan construction.
The bounded production behavior in this round is ordinary vector
`Add`/`Sub`/`Mul` with the existing vector-RHS-load body surface, so basic
elementwise arithmetic uses the same RVV-owned AVL/VL, SEW/LMUL, tail-policy,
mask-policy, runtime ABI, selected capability, typed config, materialization,
and same-analysis boundary already used by base memory movement, standalone
reduction, and scalar-broadcast MAcc.

This task does not add new operation coverage. It migrates the already
supported plain elementwise arithmetic path through the route-control owner
before provider/statement construction.

## Direction Source

- Direction title: `Stage2 RVV elementwise arithmetic route-control
  provider-plan integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by the
  existing plain elementwise arithmetic route-family/provider path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `46afb939 rvv: consume route control plan in scalar macc`.
- `.trellis/.current-task` was absent, so this task was created from the
  provided Direction Brief before source edits.

## Current Repository Facts

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  or realized `tcrv_rvv` body -> RVV plugin-owned legality / selected-body
  realization / route provider -> `TCRVEmitCLowerableRoute` -> common EmitC ->
  target artifact mirrors.
- `RVVSelectedBodyRouteControlProviderPlan` already validates typed config
  facts, selected target capability facts, runtime AVL/VL control, SEW/LMUL,
  tail policy, mask policy, runtime ABI order, and route-description mirrors.
- The route-control plan currently has adopted consumers for base memory
  movement, standalone reduction, and scalar-broadcast MAcc.
- `RVVSelectedBodyElementwiseArithmeticRouteStatementPlan` already owns the
  ordinary Add/Sub/Mul, scalar-broadcast elementwise, masked elementwise, and
  strided-add statement sequences after materialization and operand-binding
  facts.
- The current gap is that ordinary plain Add/Sub/Mul elementwise arithmetic can
  build its statement plan without first consuming the shared route-control
  provider plan.

## Scope

1. Extend the RVV route-control provider-plan consumer set to include existing
   ordinary plain elementwise arithmetic routes.
2. Make the elementwise arithmetic statement-plan boundary require the shared
   route-control plan for ordinary Add/Sub/Mul before statement construction.
3. Keep scalar-broadcast, masked, strided, compare/select, memory, reduction,
   MAcc, conversion, and residual routes on their current boundaries unless a
   compile-local neutral hook is required.
4. Preserve current emitted statement order, route ids, ABI order, runtime
   counts, target artifacts, and common EmitC behavior for valid inputs.
5. Keep target artifacts and scripts as mirror-only consumers after provider
   construction.

## Requirements

- Route-control provider-plan construction must recognize ordinary plain
  elementwise arithmetic as an adopted consumer only when the existing
  elementwise arithmetic family plan and materialization facts come from the
  same selected route analysis.
- The plan must validate AVL/VL source, runtime ABI order, SEW/LMUL, tail
  policy, mask policy, selected target capability facts, typed config contract,
  runtime VL contract, materialization facts, operand bindings, and
  same-analysis ownership before the elementwise statement plan builds calls.
- Stale analysis, missing runtime AVL binding, wrong AVL/runtime role,
  policy mismatch, unsupported config/capability, missing control plan,
  missing elementwise family/materialization plan, stale operand bindings, or
  mirror mismatch must fail closed before route/artifact authority.
- Route ids, metadata fields, manifests, artifact names, ABI strings,
  descriptors, scripts, tests, common EmitC, source-front-door markers, and
  legacy i32 spellings must not become AVL/VL, policy, dtype, or compute
  authority.
- No runtime/correctness/performance claim is made unless emitted executable
  behavior changes.

## Acceptance Criteria

- [x] Production C++ route planning/provider code makes ordinary Add/Sub/Mul
      elementwise arithmetic consume `RVVSelectedBodyRouteControlProviderPlan`
      before elementwise arithmetic statement-plan construction.
- [x] Focused C++ tests prove positive ordinary elementwise route-control-plan
      consumption joins typed config, selected capability, runtime AVL/VL
      control, materialization facts, and elementwise operand-binding facts.
- [x] Focused negative diagnostics cover stale/mismatched analysis, missing
      elementwise family/materialization facts, missing or wrong runtime AVL
      facts, policy mismatch, selected capability/config mismatch, runtime ABI
      mirror mismatch, and stale operand bindings before route construction
      where the current harness can express them.
- [x] Existing explicit/pre-realized ordinary elementwise artifact FileCheck
      coverage still passes; emitted metadata remains explicit mirror labels.
- [x] No touched code treats names, route ids, metadata, descriptors, ABI
      strings, scripts, artifacts, common EmitC, source-front-door markers, or
      legacy i32 spellings as AVL/VL, policy, dtype, or compute authority.
- [x] Focused build/tests, `git diff --check`, bounded authority scan, and
      `check-tianchenrv` pass, or an exact blocker is recorded.
- [x] Trellis task status, journal, archive, and one coherent commit are
      completed if all acceptance criteria are met.

## Non-Goals

- No new RVV operation kind, dtype/LMUL clone batch, high-level frontend
  lowering, source-front-door positive route, descriptor/direct-C/source-export
  path, dashboard, broad smoke matrix, or evidence-only fixture copy.
- No migration of masked or strided elementwise route-control ownership unless
  the plain path requires a neutral compile hook.
- No change to computation semantics, dispatch/fallback behavior, emitted
  statement order, or target artifact authority.
- No subagents, spawned agents, parallel agents, or multi-agent workflow.

## Validation Plan

1. Validate and start the Trellis task.
2. Build and run `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck for ordinary elementwise selected-body artifacts
   if metadata/header mirrors change or to prove unchanged artifact flow.
4. Run a bounded authority scan over touched RVV planning/provider/test/spec
   files.
5. Run `git diff --check`.
6. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/index.md`.
- Relevant previous tasks read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-scalar-macc-route-control-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-elementwise-arithmetic-route-family-ownership/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-24-05-24-stage2-rvv-elementwise-arithmetic-statement-plan-ownership/prd.md`.
- Primary implementation surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Implementation Result

Completed for the bounded ordinary Add/Sub/Mul elementwise arithmetic route
consumer set.

- Added `controlsOrdinaryElementwiseArithmetic` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Extended route-control provider-plan consumer detection to include ordinary
  plain elementwise arithmetic only when the selected route is Add/Sub/Mul with
  the existing `vector-rhs-load` memory form.
- `getRVVSelectedBodyRouteControlProviderPlan(...)` now requires the verified
  elementwise arithmetic route-family plan and same-analysis materialization
  facts before it exposes the ordinary elementwise runtime-control plan.
- `getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan(...)` now consumes
  the shared route-control provider plan for ordinary Add/Sub/Mul before
  building setvl/load/binary/store statement steps.
- Added focused C++ checks proving positive ordinary elementwise control-plan
  consumption and fail-closed diagnostics for stale runtime AVL role, stale
  policy, stale selected capability facts, stale same-analysis materialization,
  missing materialization family plan, and stale operand-binding facts.
- Updated the RVV plugin spec to record ordinary elementwise arithmetic as a
  required route-control consumer and to document the statement-plan ordering.
- No EmitC statement sequence, route id, target ABI, target artifact metadata,
  generated-bundle script behavior, or executable runtime behavior changed.
  No new `ssh rvv` correctness claim was made.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-elementwise-route-control-plan`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [x] Focused lit/FileCheck from `build/test`:
      `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='(explicit-selected-body-artifact-(add|sub|mul)|pre-realized-selected-body-artifact-(add|sub|mul)|generic-selected-body-artifact-arithmetic)'`
      passed 7/7 selected ordinary elementwise tests.
- [x] Bounded changed-line authority scan over touched RVV planning/provider,
      test, and spec files found no new name-, route-id-, metadata-,
      descriptor-, ABI-string-, script-, artifact-, common-EmitC-,
      source-front-door-, or legacy-i32-derived AVL/VL or policy authority.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

## Spec Update Judgment

`.trellis/spec/extension-plugins/rvv-plugin.md` was updated because this task
changed a durable planning/provider contract: ordinary elementwise arithmetic
is now a required route-control provider-plan consumer before statement-plan
construction. The update records scope, consumer flags, fail-closed behavior,
good/base/bad route ownership, and test requirements.
