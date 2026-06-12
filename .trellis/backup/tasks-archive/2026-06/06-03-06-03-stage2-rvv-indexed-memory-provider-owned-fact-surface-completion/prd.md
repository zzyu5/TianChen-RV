# Stage2 RVV indexed memory provider-owned fact surface completion

## Goal

Complete one bounded Stage 2 RVV indexed-memory compiler boundary: the RVV
provider/planning layer owns the canonical fact surface for the existing plain
and computed-mask indexed gather/scatter routes, and target artifact validation
consumes those provider facts instead of reconstructing indexed semantics
locally.

The four in-scope route operations are:

* `indexed_gather_unit_store`
* `indexed_scatter_unit_load`
* `computed_masked_indexed_gather_load_unit_store`
* `computed_masked_indexed_scatter_store_unit_load`

## Direction Source

Hermes Direction Brief:

`Stage2 RVV indexed memory provider-owned fact surface completion`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `3de849a2 rvv: validate indexed memory stale route facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The immediately preceding archived task
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-memory-production-validation-boundary/`
  tightened stale plain base-memory rejection for computed-mask indexed routes,
  but mostly changed target validation/tests/spec and did not complete the
  production provider/planning owner requested here.
* `.trellis/spec/index.md` requires the active RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness/performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` makes RVV plugin ownership
  explicit for legality, selected-body realization, route support, intrinsic
  mapping, C/RVV type mapping, ABI mapping, and fail-closed diagnostics.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires shared
  route-family fact surfaces where provider and target validation both consume
  route-family facts.

## Requirements

* Keep the implementation owner bounded to indexed memory. Do not switch to
  segment2, strided, conversion, reduction, MAcc, compare/select, scalar
  broadcast, high-level frontend lowering, source-front-door routes, global
  tuning, or evidence-only packaging.
* Production RVV provider/planning code must expose canonical indexed-memory
  facts for the four in-scope routes. This cannot be completed by adding only
  target-side stale-field tests.
* Provider facts must cover the shared selected route surface:
  runtime ABI order and parameter roles; data element type, SEW, LMUL, tail
  policy, mask policy; route operand binding plan/summary with header
  participation; required headers; C type mapping; target leaf profile; and
  explicit `provider_supported_mirror`.
* Plain indexed gather/scatter facts must cover:
  typed compute op, indexed route-family plan, gather-vs-scatter role,
  source/destination memory forms, index source, index type/C type, index
  EEW, index scale/offset unit, indexed memory layout, indexed load/store
  intrinsic class, and scatter uniqueness assumptions where required.
* Computed-mask indexed gather/scatter facts must additionally cover:
  compare predicate, mask producer source, mask role/source/memory form,
  mask type/C type, mask-tail policy owner/plan, inactive-lane contract,
  masked passthrough/no-write layout, indexed data/destination form, and
  plain-vs-computed-mask separation facts.
* Target artifact validation must consume provider-owned facts and fail closed
  on stale or missing provider/candidate facts, including:
  compare/select, conversion, reduction, MAcc, segment, scalar-splat,
  strided, plain-vs-computed-mask, gather-vs-scatter, stale index
  layout/scale/type/source, stale memory form, stale mask/passthrough,
  stale operand binding, stale header/type mapping, stale target profile, and
  stale provider mirror facts.
* Common EmitC/export remains neutral. It may carry provider-built payloads and
  mirror metadata unchanged, but must not infer RVV indexed memory semantics.

## Acceptance Criteria

* [ ] Production provider/planning code has a focused diff exposing canonical
      indexed-memory facts for all four in-scope routes.
* [ ] Target artifact validation consumes those provider facts instead of
      duplicating indexed route-family truth locally for the same fields.
* [ ] Provider-side validation fails closed when a supported indexed-memory
      route cannot obtain its canonical facts or carries cross-family stale
      facts.
* [ ] Target validation rejects gather/scatter cross-contamination for both
      plain and computed-mask indexed routes.
* [ ] Target validation rejects stale plain facts on computed-mask indexed
      routes and stale computed-mask facts on plain indexed routes.
* [ ] Target validation checks provider-derived runtime ABI order, ABI
      parameters, typed compute op, memory forms, index facts, mask facts where
      applicable, route-family plan, operand binding plan/summary, header/type
      mapping, target leaf profile, and explicit provider mirror.
* [ ] Focused C++ target artifact tests prove fail-closed behavior for stale or
      missing indexed provider facts and stale candidate metadata mirrors.
* [ ] Existing explicit and pre-realized lit/script dry-runs for the four
      indexed routes still pass.
* [ ] No descriptor-driven computation, common EmitC semantic inference,
      source-front-door positive route, route-id authority, artifact-name
      authority, mirror-only authority, or legacy `i32m1` authority is
      introduced.
* [ ] If runtime emission, generated bundle behavior, runtime ABI order, index
      semantics, or mask/passthrough behavior changes, real `ssh rvv`
      correctness is run for counts `0,1,16,17,257` with index/mask patterns
      that distinguish gather versus scatter, index scaling, inactive lanes,
      tail preservation, and destination preservation. If this round only
      tightens provider/validation facts, archived runtime evidence is reused
      with an explicit rationale.

## Technical Approach

1. Inspect current indexed-memory fact structs/accessors, provider route
   planning, control-policy plans, statement plans, route construction, target
   validation, and indexed tests.
2. Identify provider/planning facts that are still missing, target-local, or
   only mirrored in route descriptions.
3. Extend the provider-owned fact surface and provider-side validation so
   plain and computed-mask indexed gather/scatter carry canonical route facts.
4. Rewire target validation to consume that shared provider surface and reject
   stale cross-family or stale mirror metadata before artifact acceptance.
5. Add focused C++ mutations and lit/script checks only where they prove the
   provider-owned indexed fact boundary.
6. Run the smallest build/test set that exercises changed behavior, then
   finish/archive and commit if complete.

## Out Of Scope

* No new indexed route families.
* No segment2, strided memory, unit load/store expansion, compare/select,
  conversion, reduction, MAcc, scalar broadcast, frontend lowering,
  source-front-door, global tuning, dashboard, or broad smoke matrix work.
* No movement of RVV semantics into common EmitC/export.
* No runtime correctness/performance claim unless generated runtime behavior
  changes and real `ssh rvv` evidence is collected.

## Evidence Plan

* Validate Trellis task context.
* Build the focused target artifact test binary and affected tools as needed:
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for:
  * `indexed-gather-unit-store`
  * `indexed-scatter-unit-load`
  * `computed-masked-indexed-gather-load`
  * `computed-masked-indexed-scatter-store`
* Run generated-bundle dry-runs for explicit and pre-realized indexed forms.
* Run direct fail-closed checks for stale or missing indexed provider facts and
  stale candidate mirrors.
* Run a bounded old-authority scan over touched files for legacy `i32m1`,
  descriptor/direct-C/source-front-door/source-export, route-id/artifact-name
  authority, and mirror-only authority.
* Run `rtk git diff --check`.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived task read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-memory-production-validation-boundary/prd.md`

Initial code focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
