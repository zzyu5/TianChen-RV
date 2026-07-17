# Stage2 RVV strided memory production validation boundary

## Goal

Complete one bounded Stage 2 RVV strided-memory compiler boundary: the RVV
provider/planning layer owns the canonical fact surface for existing plain and
computed-mask strided memory movement routes, and target artifact validation
consumes those provider facts instead of reconstructing strided semantics
locally.

The four in-scope selected-body routes are:

* `strided_load_unit_store`
* `unit_load_strided_store`
* `computed_masked_strided_load_unit_store`
* `computed_masked_strided_store`

## Direction Source

Hermes Direction Brief:

`Stage2 RVV strided memory production validation boundary`

## Continuation Note

This continuation round resumed the same in-progress task from a dirty
worktree whose diff already contained provider/planning, target validation,
generated-bundle, and strided fixture changes. The dirty diff was kept where
it matched this PRD, repaired where validation coverage was incomplete, and
bounded to the four strided-memory owner routes.

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `ea496f75 rvv: complete indexed memory provider facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The immediately preceding archived task
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-memory-provider-owned-fact-surface-completion/`
  completed provider-owned indexed facts for plain and computed-mask indexed
  gather/scatter and rewired target validation to consume those facts.
* `.trellis/spec/index.md` requires the active RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness/performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` places strided load/unit
  store and unit load/strided store under the base memory movement
  statement-plan boundary, and computed-mask strided load/store under the
  computed-mask memory statement-plan boundary.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-owned fact
  surfaces whenever both provider planning and target validation consume the
  same route-family facts.
* Current code already has `RVVBaseMemoryMovementRouteFacts`, but that surface
  does not fully expose typed config, VL/vector C types, strided load/store
  leaves, stride C type, or byte-stride contract for the two plain strided
  routes.
* Current code has provider-local computed-mask memory constants and plans for
  computed-mask strided routes, but target artifact validation still
  reconstructs several computed-mask strided facts through target-local
  `getRVVCompareSelectMaskExpected*` helpers.

## Requirements

* Keep the implementation owner bounded to strided memory. Do not switch to
  new indexed route families, segment2, compare/select, conversion, reduction,
  MAcc, strided arithmetic such as `strided_add`, high-level frontend
  lowering, source-front-door routes, global tuning, or evidence-only
  packaging.
* Production RVV provider/planning code must expose canonical strided-memory
  facts for the four in-scope routes.
* Plain strided base-memory facts must cover:
  runtime ABI order and parameter roles; data element type, SEW, LMUL,
  tail/mask policy; VL/vector C type facts; source/destination memory forms;
  stride source; stride parameter role; stride C type; byte-vs-element stride
  contract; strided memory layout; strided load/store intrinsic leaves;
  ordinary unit load/store leaves; route operand binding plan/summary with
  header participation; required headers; C type mapping; target leaf profile;
  and explicit `provider_supported_mirror`.
* Computed-mask strided memory facts must additionally cover:
  typed compute op; compare predicate and compare leaf; mask producer source;
  mask role/source/memory form; mask type/C type; mask-tail route-family plan
  and owner; inactive-lane contract; masked passthrough layout; masked load or
  strided store leaves; source/destination memory forms; stride source; stride
  parameter role; stride C type; byte-vs-element stride contract; strided and
  masked memory layout; runtime ABI order and parameter roles; header/type
  facts; target leaf profile; provider mirror; and exact route operand binding
  summary.
* Provider-side validation must fail closed when the selected strided memory
  plan cannot obtain canonical route facts, or when typed config/runtime/route
  family/mask/stride/header/type/binding facts disagree with those canonical
  facts.
* Target artifact validation must consume provider-owned strided facts and
  fail closed on stale or missing provider/candidate facts, including:
  indexed, segment2, compare/select, conversion, reduction, MAcc,
  scalar-splat, unit-only, plain-vs-computed-mask, load-vs-store, stale stride
  layout/type/source, memory form, mask, operand binding, header/type, target
  profile, provider mirror, and route-family mirror facts.
* Common EmitC/export remains neutral. It may carry provider-built payloads and
  mirror metadata unchanged, but must not infer RVV strided memory semantics.

## Acceptance Criteria

* [x] Production provider/planning code exposes canonical strided-memory facts
      for all four in-scope routes.
* [x] Target artifact validation consumes the provider strided facts instead of
      duplicating strided route-family truth locally for the same fields.
* [x] Plain strided validation checks provider-derived runtime ABI order,
      runtime ABI parameters, data type/config policy, stride source, stride
      role/type/byte contract, memory forms, strided layout, strided leaves,
      operand binding, header/type mapping, target profile, and provider
      mirror.
* [x] Computed-mask strided validation checks provider-derived compare/mask
      facts, mask-tail owner/plan, masked passthrough/inactive-lane facts,
      stride source/role/type/byte contract, strided/masked layouts, leaves,
      operand binding, header/type mapping, target profile, and provider
      mirror.
* [x] Target validation rejects plain strided facts on computed-mask strided
      routes and computed-mask strided facts on plain base-memory routes.
* [x] Target validation rejects load/store cross-contamination between
      `strided_load_unit_store` and `unit_load_strided_store`, and between
      `computed_masked_strided_load_unit_store` and
      `computed_masked_strided_store`.
* [x] Focused C++ target artifact tests prove stale or missing strided
      provider facts and stale candidate metadata mirrors fail closed.
* [x] Existing explicit and pre-realized lit/script dry-runs for the four
      strided routes still pass.
* [x] No descriptor-driven computation, common EmitC semantic inference,
      source-front-door positive route, route-id authority, artifact-name
      authority, mirror-only authority, or legacy `i32m1` authority is
      introduced.
* [x] If route emission, generated bundle behavior, runtime ABI order, stride
      semantics, mask/passthrough behavior, or runtime behavior changes, real
      `ssh rvv` correctness is run for counts `0,1,16,17,257` with stride
      patterns that distinguish load versus store, byte/element stride
      interpretation, masked inactive lanes, tail preservation, and
      destination preservation. If this round only tightens provider/target
      validation facts, archived runtime evidence is reused with an explicit
      rationale.

## Technical Approach

1. Extend the provider-owned plain base-memory fact surface so the two plain
   strided routes carry typed config, stride, leaf, header/type, binding, and
   target-profile facts in one canonical surface.
2. Add a provider-owned computed-mask strided memory fact surface, using the
   existing computed-mask memory provider constants and operand-binding plan
   construction instead of target-local constants.
3. Rewire provider validation to compare strided plan/description facts with
   the canonical strided surfaces before materialization.
4. Rewire target validation to consume the shared strided surfaces for
   provider route payload facts and candidate metadata mirrors.
5. Add focused C++ mutations for stale stride/source/layout/mask/binding
   facts and stale candidate mirrors only where they prove the provider-owned
   boundary.
6. Run the smallest build/test set that exercises changed behavior, then
   finish/archive and commit if complete.

## Out Of Scope

* No new strided route families.
* No indexed redo, segment2 changes, strided arithmetic such as
  `strided_add`, dot-reduce changes, compare/select changes, conversion
  changes, reduction changes, MAcc changes, frontend lowering,
  source-front-door, global tuning, dashboard, or broad smoke matrix work.
* No movement of RVV semantics into common EmitC/export.
* No runtime correctness/performance claim unless generated runtime behavior
  changes and real `ssh rvv` evidence is collected.

## Evidence Plan

* Validate Trellis task context.
* Build focused targets as needed:
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for:
  * `strided-load-unit-store`
  * `unit-load-strided-store`
  * `computed-masked-strided-load`
  * `computed-masked-strided-store`
* Run generated-bundle dry-runs for explicit and pre-realized strided forms.
* Run direct fail-closed checks for stale or missing strided provider facts and
  stale candidate mirrors.
* Run a bounded old-authority scan over touched files for legacy `i32m1`,
  descriptor/direct-C/source-front-door/source-export, route-id/artifact-name
  authority, and mirror-only authority.
* Run `rtk git diff --check`.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived task read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-memory-provider-owned-fact-surface-completion/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-memory-provider-owned-fact-surface-completion/task.json`

Initial code focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
