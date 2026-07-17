# Stage2 RVV scalar-broadcast elementwise route-control provider-plan integration

## Goal

Make the existing scalar-broadcast RVV elementwise route family consume the
shared RVV route-control provider plan before scalar-broadcast elementwise
statement/provider construction. Supported `scalar_broadcast_add`,
`scalar_broadcast_sub`, and `scalar_broadcast_mul` routes should use the same
RVV-owned AVL/VL, SEW/LMUL, tail-policy, mask-policy, runtime ABI, selected
capability, typed config, materialization, operand-binding, and same-analysis
boundary already used by ordinary elementwise arithmetic, base memory movement,
standalone reduction, and scalar-broadcast MAcc.

This is an integration of an already-supported scalar-broadcast elementwise
path. It does not add a new RVV operation family, dtype/LMUL matrix, frontend
lowering path, source-front-door positive route, descriptor/direct-C path, or
new runtime/correctness claim.

## Direction Source

- Direction title: `Stage2 RVV scalar-broadcast elementwise route-control
  provider-plan integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by the
  existing scalar-broadcast elementwise route-family/provider path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `33963fb1 rvv: consume route control plan in elementwise arithmetic`.
- `.trellis/.current-task` was absent, so this task was created from the
  provided Direction Brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflow.

## Current Repository Facts

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  or realized `tcrv_rvv` body -> RVV plugin-owned legality / selected-body
  realization / route provider -> `TCRVEmitCLowerableRoute` -> common EmitC ->
  target artifact mirrors.
- `RVVSelectedBodyRouteControlProviderPlan` already validates typed config
  facts, selected target capability facts, runtime AVL/VL control, SEW/LMUL,
  tail policy, mask policy, runtime ABI order, config/runtime VL mirrors, and
  same-analysis materialization facts.
- The route-control plan currently has adopted consumers for ordinary
  Add/Sub/Mul elementwise arithmetic, base memory movement, standalone
  reduction, and scalar-broadcast MAcc.
- `RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan` already carries a
  runtime control plan, scalar splat leaf, arithmetic leaf, runtime ABI order,
  typed vector facts, provider mirror constants, and route-family plan mirror.
- `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts` already owns
  scalar-broadcast elementwise logical operand bindings for `lhs`,
  `rhs_scalar`, `out`, and runtime `n`.
- `RVVSelectedBodyElementwiseArithmeticRouteStatementPlan` already owns
  scalar-broadcast Add/Sub/Mul statement construction after family plan,
  materialization facts, and operand-binding facts, but currently requires the
  shared route-control provider plan only for ordinary vector-RHS-load
  Add/Sub/Mul.

## Scope

1. Extend the RVV route-control provider-plan consumer set to include existing
   scalar-broadcast elementwise Add/Sub/Mul routes.
2. Make the elementwise arithmetic statement-plan boundary require the shared
   route-control plan for scalar-broadcast Add/Sub/Mul before statement
   construction.
3. Keep ordinary, masked, strided, compare/select, memory, reduction, MAcc,
   conversion, and residual routes on their current behavior except for shared
   compile-local declarations or checks needed by the scalar-broadcast
   integration.
4. Preserve current emitted statement order, route ids, ABI order, runtime
   counts, target artifacts, generated-bundle scripts, and common EmitC
   behavior for valid inputs.
5. Keep target artifacts, generated headers, route descriptions, scripts, and
   metadata as mirrors only after provider route construction.

## Requirements

- Route-control provider-plan construction must recognize scalar-broadcast
  elementwise routes only when the selected operation is
  `ScalarBroadcastAdd`, `ScalarBroadcastSub`, or `ScalarBroadcastMul` with
  `RHSScalarBroadcast` memory form.
- The route-control plan must require the verified
  `scalarBroadcastElementwiseRouteFamilyPlan` and materialization facts from
  the same selected route analysis before it exposes scalar-broadcast control
  facts.
- The elementwise statement-plan boundary must consume the route-control plan
  for scalar-broadcast Add/Sub/Mul before building setvl/load/splat/binary/store
  steps.
- The positive path must join typed config facts, selected target capability
  facts, runtime AVL/VL control facts, scalar-broadcast route-family facts,
  materialization facts, elementwise/select operand-binding facts, and
  same-analysis ownership.
- Missing scalar-broadcast family/materialization facts, stale analysis,
  missing runtime AVL binding, wrong runtime AVL role/source, policy mismatch,
  unsupported selected config/capability, missing control plan, stale operand
  bindings, runtime ABI mirror mismatch, or target capability mirror mismatch
  must fail closed before route/artifact authority.
- Route ids, metadata fields, manifests, artifact names, ABI strings,
  descriptors, scripts, tests, common EmitC, source-front-door markers, and
  legacy i32 spellings must not become AVL/VL, policy, dtype, scalar-broadcast,
  or compute authority.
- No runtime/correctness/performance claim is made unless emitted executable
  behavior changes.

## Acceptance Criteria

- [x] Production C++ route planning/provider code makes scalar-broadcast
      elementwise Add/Sub/Mul consume `RVVSelectedBodyRouteControlProviderPlan`
      before elementwise arithmetic statement-plan construction.
- [x] Focused C++ tests prove positive scalar-broadcast elementwise
      route-control-plan consumption joins typed config, selected capability,
      runtime AVL/VL control, materialization facts, and elementwise/select
      operand-binding facts.
- [x] Focused negative diagnostics cover missing scalar-broadcast
      family/materialization facts, stale/mismatched analysis, missing or wrong
      runtime AVL facts, policy mismatch, selected capability/config mismatch,
      runtime ABI mirror mismatch, and stale operand bindings before route
      construction where the current harness can express them.
- [x] Existing explicit/pre-realized scalar-broadcast elementwise artifact
      FileCheck coverage still passes; emitted metadata remains explicit mirror
      labels.
- [x] No touched code treats names, route ids, metadata, descriptors, ABI
      strings, scripts, artifacts, common EmitC, source-front-door markers, or
      legacy i32 spellings as AVL/VL, policy, dtype, scalar-broadcast, or
      compute authority.
- [x] Focused build/tests, `git diff --check`, bounded authority scan, and
      `check-tianchenrv` pass, or an exact blocker is recorded.
- [x] Trellis task status, journal, archive, and one coherent commit are
      completed if all acceptance criteria are met.

## Non-Goals

- No new RVV operation kind, dtype/LMUL clone batch, high-level frontend
  lowering, compare/select expansion, conversion, reduction, contraction,
  source-front-door positive route, descriptor/direct-C/source-export path,
  dashboard, broad smoke matrix, or evidence-only fixture copy.
- No migration of masked or strided elementwise route-control ownership unless
  scalar-broadcast integration requires a neutral compile hook.
- No change to computation semantics, dispatch/fallback behavior, emitted
  statement order, or target artifact authority.
- No subagents, spawned agents, parallel agents, or multi-agent workflow.

## Validation Plan

1. Validate and start the Trellis task.
2. Build and run `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck for scalar-broadcast selected-body artifacts if
   metadata/header mirrors change or to prove unchanged artifact flow.
4. Run a bounded authority scan over touched RVV planning/provider/test/spec
   files.
5. Run `git diff --check`.
6. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant previous tasks read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-scalar-macc-route-control-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-elementwise-route-control-plan/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-elementwise-broadcast-route-closure/prd.md`.
- Primary implementation surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Implementation Result

- Added `controlsScalarBroadcastElementwise` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Added scalar-broadcast elementwise Add/Sub/Mul with
  `RHSScalarBroadcast` memory form to the route-control consumer set.
- Required scalar-broadcast route-control construction to use the verified
  `scalarBroadcastElementwiseRouteFamilyPlan` and scalar-broadcast
  materialization facts from the same selected route analysis before exposing
  runtime control facts.
- Required the scalar-broadcast elementwise statement-plan boundary to consume
  the RVV-owned route-control provider plan before setvl/load/splat/binary/store
  statement construction.
- Extended the RVV plugin spec so scalar-broadcast elementwise arithmetic is an
  explicit route-control consumer alongside ordinary elementwise arithmetic,
  base memory movement, standalone reduction, and scalar-broadcast MAcc.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-scalar-broadcast-route-control`
- `git diff --check`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- Focused lit/FileCheck from `build/test` with filter
  `(explicit-selected-body-artifact-scalar-broadcast-(add|sub|mul)|pre-realized-selected-body-artifact-scalar-broadcast-(add|sub|mul)|rvv-generic-stage2-scalar-broadcast-sub)`,
  passing 7 selected tests.
- Bounded changed-line scan over the touched spec/planning/test files found no
  new descriptor, source-front-door, common-EmitC, route-id, artifact, ABI
  string, script, or legacy-i32 authority. Hits were expected mirror
  comparisons, selected-boundary fixture status text, provider-derived intrinsic
  leaves in test expectations, and a negative stale runtime ABI mirror.
- `cmake --build build --target check-tianchenrv -j2` passed.

No `ssh rvv` correctness/performance rerun was required because this round did
not change emitted statement order, common EmitC lowering, generated target
artifact code, ABI order for valid routes, or executable semantics. The change
adds fail-closed provider-plan ownership checks and tests around an existing
scalar-broadcast elementwise route family.
