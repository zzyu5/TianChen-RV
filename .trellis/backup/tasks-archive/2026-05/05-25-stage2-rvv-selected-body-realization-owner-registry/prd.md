# Stage2 RVV Pre-Realized Selected-Body Realization Owner Registry

## Goal

Make RVV pre-realized selected-body realization owner-driven instead of a loose central sequence of string predicates and family-specific branches. The production boundary should classify each supported pre-realized `tcrv_rvv` body family through one RVV plugin-local owner registry, validate the selected typed/config/runtime/memory/policy facts before materialization, and leave realized `setvl` / `with_vl` / typed `tcrv_rvv` body structure for the existing route-control registry and route provider to consume.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created from the Hermes Direction Brief.
* Current HEAD is `bad3524b rvv: consolidate route-control owner registry`, and `git status --short` was clean before this task was created.
* The previous route-control registry task consolidated downstream route-control owner selection for route-supported families without changing emitted semantics.
* `.trellis/spec/index.md` defines the RVV-first authority chain: selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned selected-body realization / legality / route provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires selected-body realization to consume pre-realized facts into real `tcrv_rvv` structure before route/provider construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to stay neutral and never infer RVV dtype/config/operation/schedule from metadata or route ids.
* `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h` currently exposes broad realization helpers but no owner registry surface.
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` currently contains many central `isPreRealized...` string predicates, a central pre-realized body discovery list, and a long `realizePreRealizedRVVSelectedBody(...)` branch sequence.
* Existing route-control registry APIs live in `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` and `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* `test/Plugin/RVVExtensionPluginTest.cpp` is the focused C++ test surface for RVV plugin boundaries; `test/Target/RVV/pre-realized-selected-body-artifact-*` and `explicit-selected-body-artifact-*` are the representative lit artifact surfaces.

## Requirements

* Add an RVV plugin-local selected-body realization owner registry or equivalent owner table.
* Registry entries must have explicit owner names, a consumer predicate, and a realization hook.
* `variantContainsPreRealizedRVVSelectedBody(...)`, route-entry selected-body detection, and `realizePreRealizedRVVSelectedBody(...)` must use registry-backed owner selection rather than only a central hard-coded ad hoc branch sequence.
* Owner selection must fail closed if no owner supports a recognized pre-realized body, or if more than one owner matches the same body.
* Supported owner predicates must be structural over typed pre-realized body op classes and RVV-owned attrs such as `op_kind` and `memory_form`; downstream route authority must not come from route ids, artifact names, descriptors, ABI strings, test names, common EmitC, scripts, or legacy i32 names.
* Owner realization hooks must continue to call the existing family validators and realization builders so runtime AVL/VL source, ABI roles/order, mem_window/imported values, typed config, SEW/LMUL, policy, operation kind, memory form, mask/passthrough facts, accumulator/result layout, and selected capability facts fail closed before realized ops are created.
* Preserve explicit already-realized selected-body behavior.
* Preserve existing emitted semantics, statement order, runtime ABI, dispatch/fallback behavior, and target artifact semantics.
* Keep this round bounded if the file is too large: complete the registry and owner-selection boundary plus at least one representative route-entry family and one representative elementwise/compare-select family, leaving exact continuation points for any remaining family-local owner migration.

## Acceptance Criteria

* [x] Production C++ exposes a selected-body realization owner registry with explicit owner names and non-null consumer/realization hooks.
* [x] Every pre-realized body op class currently recognized by `variantContainsPreRealizedRVVSelectedBody(...)` is either represented in the registry exactly once or explicitly rejected with a targeted unsupported diagnostic.
* [x] `realizePreRealizedRVVSelectedBody(...)` dispatches through the owner registry and fails closed for missing or ambiguous owners before route-control/provider construction.
* [x] Route-entry realization selection uses the same owner registry instead of a separate ad hoc allowlist as route authority.
* [x] Existing validation for unsupported/mismatched `op_kind`, `memory_form`, runtime AVL role, ABI role/order, mem_window/imported value role, typed config, policy, mask/passthrough, accumulator/result layout, selected capability, and stale family facts remains active before realized ops are created.
* [x] At least one representative elementwise/compare-select pre-realized body and one representative route-entry pre-realized body realize through the owner boundary and still reach the route-control registry/provider path.
* [x] Explicit selected-body artifacts remain unaffected.
* [x] Focused C++ or lit/FileCheck tests cover registry membership/exact owner matching, non-consumer/unsupported diagnostics, representative elementwise/compare-select realization, representative route-entry realization, and explicit-body non-regression.
* [x] Bounded scan over touched files finds no new name-, metadata-, descriptor-, ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or legacy-i32-derived route authority.
* [x] `git diff --check` passes.
* [x] Focused RVV plugin tests and representative lit/generated-bundle dry-runs pass; run `check-tianchenrv` if feasible, otherwise record the exact blocker.

## Completion Notes

* Added `RVVSelectedBodyRealizationOwner`, `getRVVSelectedBodyRealizationOwners()`, and `getRVVSelectedBodyRealizationOwnerForBody(...)`.
* Replaced selected-body discovery, route-entry realization selection, and the public `realizePreRealizedRVVSelectedBody(...)` entrypoint with registry-owned owner selection.
* Preserved existing family validators and materializers; the non-elementwise owners currently dispatch through the existing family branch implementation after registry selection. The exact continuation point for future cleanup is to split each non-elementwise owner hook into a family-local implementation instead of the shared existing-family branch helper.
* Added C++ coverage for owner count/names/hooks, route-entry eligibility, representative owner matching, route-entry positive paths, and reduction route-entry rejection.
* Added RVV plugin spec coverage for the selected-body realization owner registry API and validation/error matrix.
* Verified representative pre-realized compare/select, pre-realized base-memory route-entry, and explicit add artifact paths.

## Out of Scope

* New RVV route coverage, new dtypes/LMUL clones, source-front-door positive routes, Linalg/Vector frontend lowering, common EmitC semantics, target artifact semantics, dashboards, reports, or broad smoke matrices.
* Changing computation semantics, runtime ABI, emitted statement order, dispatch/fallback behavior, or performance claims.
* Preserving old ad hoc string predicates as downstream route authority behind compatibility glue.
* Moving RVV realization, legality, dtype/config, or intrinsic authority into common EmitC/export/scripts/artifacts/metadata.

## Technical Notes

* Relevant specs:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Relevant prior task:
  * `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-route-control-registry-consolidation/prd.md`
* Relevant production surfaces:
  * `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
  * `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* Relevant tests:
  * `test/Plugin/RVVExtensionPluginTest.cpp`
  * `test/Target/RVV/pre-realized-selected-body-artifact-*.mlir`
  * `test/Target/RVV/explicit-selected-body-artifact-*.mlir`

## Definition of Done

* Trellis context is curated and task status is truthful.
* Production selected-body realization dispatch is registry-owned.
* Focused tests and artifact evidence demonstrate unchanged representative routes.
* Quality checks pass or exact blockers are recorded.
* Workspace journal is updated and one coherent commit is created if the task is complete.
