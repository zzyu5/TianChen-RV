# Stage2 RVV runtime-scalar-cmp masked segment2 load executable artifact ABI boundary

## Goal

Make the existing RVV runtime-scalar-cmp masked segment2 load selected-body route family truthful at the executable artifact ABI boundary, or fail closed at the exact missing boundary fact. The production chain under review is:

```text
selected/pre-realized tcrv.exec RVV variant
  -> typed tcrv_rvv segment2 load body facts
  -> RVV-owned runtime scalar compare and mask facts
  -> old destination passthrough and segment field order facts
  -> RVV provider route validation
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact bundle
  -> generated bundle compile and ssh rvv correctness evidence
```

## Why Now

The previous completed route-family task hardened computed-masked segment2 update and proved stale source binding rejection plus ssh rvv correctness. Runtime-scalar-cmp masked segment2 load is the next distinct boundary because it carries a runtime scalar through ABI binding into compare-produced mask construction for segmented memory movement.

## What I Already Know

* The repository currently has no active Trellis task, so this task was created from the Hermes Direction Brief before source edits.
* HEAD is `a77bf8d8 rvv: harden segment2 update artifact ABI`; the working tree was clean at task creation.
* The relevant long-term invariant is extension-family body -> RVV plugin route provider -> common EmitC materializer -> target artifact; common EmitC/export must not choose RVV semantics.
* `tcrv.exec` owns ABI/runtime role binding only. Runtime scalar value, source/output windows, and AVL/VL must be consumed by the typed selected RVV body and route facts before artifact export claims executability.
* Route ids, artifact names, metadata fields, helper names, and test names are mirrors only. They cannot be authority for dtype/config/mask/segment/pass-through/header semantics.
* The Direction Brief explicitly excludes broad segment2 matrices, runtime-scalar-cmp segment2 store, computed-mask update rework except as reference, dtype/LMUL clone batches, high-level frontend work, dashboards, and source-front-door positive routes.

## Requirements

* Inspect the current selected/pre-realized runtime-scalar-cmp masked segment2 load production path before editing.
* If the path is dry-run-only or under-validated, make the route executable through generated RVV artifact ABI evidence, using existing typed body and RVV-owned route/provider boundaries.
* If an executable claim is missing a boundary fact, harden the production seam to fail closed with targeted diagnostics instead of silently generating stale or misleading artifacts.
* Validate, at minimum, one stale or missing executable-boundary fact related to runtime scalar binding, compare operand/mask source, inactive passthrough field, segment field order, header/prototype binding, ABI value mapping, or unsupported executable route claim.
* Preserve common EmitC neutrality; any RVV semantic validation or route construction belongs in RVV-owned code.
* Keep old-authority drift out of touched code: no new `RVVI32M1*`, `rvv-i32m1` route ids, finite `tcrv_rvv.i32_*` executable surface, descriptor-driven compute, source-front-door positive route, or exact intrinsic spelling as route authority.

## Acceptance Criteria

* [x] Selected-body explicit and pre-realized runtime-scalar-cmp masked segment2 load tests materialize the selected boundary, emission plan, target artifact export, and generated bundle ABI path.
* [x] The generated bundle ABI evidence includes executable compile/run where runtime correctness is claimed, including `ssh rvv` correctness for focused counts/patterns.
* [x] At least one focused fail-closed case rejects stale or missing artifact/ABI facts for this route family.
* [x] Relevant production checks pass: `build/bin/tianchenrv-rvv-extension-plugin-test`, `build/bin/tianchenrv-target-artifact-export-test`, relevant generated-bundle dry-run tests, and any focused lit/FileCheck tests touched by the implementation. Local `FileCheck` / `llvm-lit` were not available in PATH or `build/bin`, so the new RUN behavior was validated with equivalent `tcrv-opt` / `tcrv-translate` / `rg` commands.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Final worktree is clean after one coherent commit.

## Completion Notes

* Inspection found the explicit and pre-realized runtime-scalar-cmp masked segment2 load production paths are already wired through typed `tcrv_rvv` selected bodies, RVV plugin-local pre-realized body materialization, provider-owned runtime-scalar/segment2 route facts, common EmitC materialization, target artifact validation, generated bundle export, and executable runtime evidence.
* No production C++/Python compiler path changed. The existing RVV route-family owner derives `rhs_scalar` splat binding, compare-produced mask facts, old field passthroughs, field order, runtime ABI order, header declarations, dtype/config, and segment2 intrinsic facts from typed body/config/runtime facts. The existing target artifact validator rebuilds the provider route and compares candidate mirrors before accepting the bundle.
* Added explicit selected-body target artifact negative coverage for stale `rhs_scalar` operand binding, stale runtime ABI order, stale required header declarations, stale inactive-lane contract, and stale field0 role, in addition to the existing stale mask producer and stale out1 binding coverage.
* Added matching pre-realized selected-body target artifact negative coverage for the same stale boundary facts after selected-body materialization, proving stale pre-realized metadata cannot replace provider-built route facts.
* Positive artifact/header export evidence covers the provider-derived ABI order `lhs,rhs_scalar,src,out0,out1,n`, route operand binding summary, `rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_load_unit_store.v1`, runtime scalar splat producer `runtime-scalar-splat-compare-rhs`, computed mask facts, inactive-lane preservation policy, field0/field1 output roles, segment2 load/extract intrinsics, and generated C prototype.
* Generated-bundle dry-run passed for explicit and pre-realized selected-body inputs with counts `0,1,16,17,257`, rhs scalars `-37,91`, and patterns `0,1`. This local environment has no `llvm-readobj`, so dry-run and runtime evidence used `--llvm-readobj ''`; bundle/header/harness validation still ran.
* Non-dry-run `ssh rvv` generated-bundle correctness passed for explicit and pre-realized selected-body inputs with counts `0,1,16,17,257` and patterns `0,1`, including active lanes, inactive old field preservation, field-distinguishing lanes, source preservation, and tail preservation.
* Spec update judgment: no `.trellis/spec/` change was needed. Existing RVV plugin, unified EmitC route, provider operand binding summary, and segment2 target artifact validation contracts already require plugin-owned route construction and fail-closed candidate mirror validation for exactly these boundary facts.

## Out Of Scope

* No runtime-scalar-cmp segment2 store work in this task.
* No broad segment2 matrix or dtype/LMUL clone batch.
* No computed-mask segment2 update rework except as bounded reference.
* No high-level Linalg/Vector/StableHLO frontend generalization.
* No performance tuning database, dashboard/index/report-only work, or source-front-door positive route.
* No mass rewrite of MAcc, reduction, compare/select, or unrelated memory routes.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* Archived reference task `.trellis/tasks/archive/2026-06/06-08-06-08-stage2-rvv-computed-masked-segment2-update-executable-abi/`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* Segment2/memory route-owner headers under `include/TianChenRV/Plugin/RVV`
* `lib/Plugin/RVV` selected-body realization owners for segment2/runtime-scalar masks
* `lib/Plugin/RVV/EmitC` route/statement owners around `runtime_scalar_cmp_masked_segment2_load_unit_store`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* Relevant dry-run and target artifact MLIR tests named in the Direction Brief

## Definition Of Done

* Task context and PRD are truthful and current.
* Source changes, if any, are scoped to this route family and its artifact ABI validation.
* Focused tests and artifact evidence have been run and recorded.
* Task is finished/archived through Trellis convention.
* One coherent git commit is created, or the exact blocker and continuation point are documented.
