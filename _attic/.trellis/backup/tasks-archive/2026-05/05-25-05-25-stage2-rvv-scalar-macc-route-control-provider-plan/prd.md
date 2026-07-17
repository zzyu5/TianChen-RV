# Stage2 RVV scalar MAcc route-control provider-plan integration

## Goal

Make the existing executable `scalar_broadcast_macc_add` RVV route family consume the shared RVV route-control provider plan before scalar MAcc family/provider statement construction, matching the route-control boundary already used by base memory movement and standalone reduction.

This task closes the remaining route-control-plan adoption gap for the current executable scalar MAcc math family without adding new RVV operation coverage, new dtype/LMUL clones, frontend lowering, source-front-door routes, dashboards, or evidence-only fixture copying.

## Requirements

* Keep the production path RVV-local: selected `tcrv.exec` RVV variant and typed/pre-realized `tcrv_rvv` body/config facts feed RVV route-family/provider planning, then `TCRVEmitCLowerableRoute`, then common EmitC.
* Extend scalar-broadcast MAcc route planning so `scalar_broadcast_macc_add` joins same-analysis typed config facts, selected target capability facts, runtime AVL/VL control facts, materialization facts, and math operand-binding facts before plain/scalar-broadcast MAcc statement-plan construction.
* Route-control provider-plan consumption must validate AVL/VL source, runtime ABI order, SEW/LMUL, tail policy, mask policy, selected target capability facts, typed config contract, runtime VL contract, and same-analysis ownership.
* Stale analysis, missing runtime AVL binding, wrong AVL/runtime role, policy mismatch, unsupported config/capability, missing control plan, or mirror mismatch must fail closed before route/artifact authority.
* Target artifacts, generated headers, scripts, route ids, metadata, manifests, ABI strings, descriptors, source-front-door markers, and runtime counts remain mirror-only consumers after provider route construction.
* If scalar MAcc has multiple subpaths, finish the executable `scalar_broadcast_macc_add` subpath and leave an explicit continuation point for non-covered MAcc subpaths.

## Acceptance Criteria

* [x] Production C++ route planning/provider changes make `scalar_broadcast_macc_add` consume `RVVSelectedBodyRouteControlProviderPlan` before scalar MAcc statement-plan construction.
* [x] Focused C++ provider/unit tests prove positive scalar-broadcast MAcc route-control-plan consumption joins typed config, selected capability, runtime AVL/VL control, materialization facts, and math operand-binding facts.
* [x] Focused negative diagnostics cover stale/mismatched analysis, missing runtime AVL/VL control facts, wrong runtime AVL role/source, policy mismatch, unsupported SEW/LMUL or selected capability/config, missing control plan, and route-control mirror mismatch where the existing test harness can express them.
* [x] Existing explicit and pre-realized `scalar_broadcast_macc_add` fixture/lit or generated-bundle evidence still routes through provider-built statements; emitted metadata did not change, so existing explicit mirror labels remained unchanged.
* [x] No changed code treats names, route ids, metadata, descriptors, ABI strings, scripts, artifacts, common EmitC, source-front-door markers, or legacy i32 spellings as AVL/VL, policy, dtype, or compute authority.
* [x] Run focused build/tests for the touched RVV plugin path, `git diff --check`, bounded authority scan, and `check-tianchenrv` or report the exact blocker.
* [x] Reused prior `ssh rvv` scalar-broadcast MAcc evidence because emitted executable behavior, statement order, and target artifact metadata did not change.

## Definition of Done

* Trellis task context is valid and active.
* Source changes are scoped to RVV plugin planning/provider, target mirror validation, fixtures/tests, and spec only if the implementation reveals a durable rule gap.
* Focused tests pass after any self-repair.
* Task status is finished/archived if acceptance criteria are complete.
* One coherent commit records the completed round.

## Technical Approach

Follow the existing route-control provider-plan implementation from commit `6f7d052a` for base memory movement and standalone reduction:

1. Inspect `RVVSelectedBodyRouteControlProviderPlan` construction and consumer flags in `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` and `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
2. Identify where `getRVVSelectedBodyPlainMAccRouteStatementPlan(...)` currently receives materialization and math operand-binding facts for `scalar_broadcast_macc_add`.
3. Thread the route-control provider plan into the scalar-broadcast MAcc family/provider statement-plan boundary, requiring a non-empty scalar MAcc control-plan consumer only for the executable scalar-broadcast path.
4. Add fail-closed validation before statement construction and before `TCRVEmitCLowerableRoute` attachment.
5. Update focused C++ tests and any FileCheck/generate-header expectations only where the provider-built control-plan mirror surface changes.

## Decision (ADR-lite)

**Context**: Base memory movement and standalone reduction already consume the shared route-control provider plan. `scalar_broadcast_macc_add` has family-plan and statement-plan ownership but still needs to join the same route-control owner before statement construction.

**Decision**: Adopt the existing shared `RVVSelectedBodyRouteControlProviderPlan` boundary in the scalar-broadcast MAcc provider-plan/statement-plan path instead of creating scalar-MAcc-specific AVL/VL or policy checks.

**Consequences**: The scalar-broadcast MAcc route has the same structural control-fact owner as current executable base memory and reduction families. Non-adopted MAcc variants should keep empty/default route-control behavior until explicitly migrated.

## Out of Scope

* New RVV operation coverage, new MAcc variants, dtype/LMUL clone batches, or high-level frontend lowering.
* Source-front-door positive routes, descriptor-driven computation, direct-C/source export, dashboards, broad smoke matrices, or evidence-only fixture duplication.
* Changing computation semantics, dispatch/fallback behavior, or emitted statement order unless required by the route-control boundary and explicitly evidenced.
* Moving route semantics into common EmitC, target artifact code, scripts, manifests, ABI strings, metadata fields, or legacy i32 route names.

## Technical Notes

* Current repo state before task creation: `main`, clean worktree, HEAD `6f7d052a rvv: add route control provider plan boundary`.
* Relevant specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/index.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/testing/index.md`
  * `.trellis/spec/guides/index.md`
  * `.trellis/spec/guides/capability-first-design-guide.md`
  * `.trellis/spec/guides/plugin-locality-review-guide.md`
  * `.trellis/spec/guides/compute-boundary-review-guide.md`
* Relevant journal context:
  * Session 212 closed `scalar_broadcast_macc_add` through a route-family module boundary and scalar-broadcast MAcc statement-plan path.
  * Session 214 introduced `RVVSelectedBodyRouteControlProviderPlan` for base memory movement and standalone reduction.
* Primary code to inspect before editing:
  * `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  * `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  * `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  * `test/Plugin/RVV/RVVExtensionPluginTest.cpp`
