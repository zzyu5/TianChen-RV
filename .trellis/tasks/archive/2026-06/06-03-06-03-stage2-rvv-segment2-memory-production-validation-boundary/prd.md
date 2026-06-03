# Stage2 RVV segment2 memory production validation boundary

## Goal

Complete one bounded Stage 2 RVV segment2-memory compiler boundary: the RVV
provider/planning layer owns the canonical fact surface for existing segment2
memory movement routes, and target artifact validation consumes those provider
facts instead of reconstructing segment2, mask, update, ABI, header/type, or
statement semantics locally.

The five in-scope selected-body routes are:

* `segment2_interleave_unit_load`
* `segment2_deinterleave_unit_store`
* `computed_masked_segment2_load_unit_store`
* `computed_masked_segment2_store_unit_load`
* `computed_masked_segment2_update_unit_load`

## Direction Source

Hermes Direction Brief:

`Stage2 RVV segment2 memory production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `430f2887 rvv: validate strided memory provider facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* The immediately preceding archived task
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-strided-memory-production-validation-boundary/`
  completed provider-owned strided memory facts and rewired target validation
  to consume them.
* `.trellis/spec/index.md` requires the active RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness/performance is claimed.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines dedicated plain and
  computed-mask segment2 provider-owned route fact surfaces.
* `.trellis/spec/extension-plugins/rvv-plugin.md` places segment2 memory under
  plugin-local selected-body route-family planning and statement-plan
  boundaries, not target-local route reconstruction.

## Requirements

* Keep the implementation owner bounded to segment2 memory. Do not redo
  strided or indexed memory as the owner, and do not switch to compare/select,
  conversion, reduction, MAcc, scalar-splat, high-level frontend lowering,
  source-front-door routes, global tuning, or evidence-only packaging.
* Production RVV provider/planning code must expose canonical segment2 facts
  for the five in-scope routes.
* Plain segment2 facts must cover:
  runtime ABI order and parameter roles; data element type, SEW, LMUL,
  tail/mask policy; runtime control plan; source/destination memory forms;
  segment count; segment tuple C type; segment memory layout; segment load,
  store, tuple, and field-extract leaves as applicable; interleave versus
  deinterleave direction and field0/field1 roles; route-family plan; route
  operand binding plan/summary with header participation; required headers; C
  type mapping; target leaf profile; and explicit
  `provider_supported_mirror`.
* Computed-mask segment2 facts must additionally cover:
  typed compute op; compare predicate and compare mask producer facts; mask
  role/source/memory form; mask-tail policy plan and owner; inactive-lane
  contract; masked passthrough/no-write layout; segment load/store/tuple and
  field-extract leaves as applicable; source/destination and field memory
  forms; update arithmetic kind/intrinsic for the update route; runtime ABI
  order and parameter roles; header/type facts; target leaf profile; provider
  mirror; and exact route operand binding summary.
* Provider-side validation must fail closed when a supported segment2 route
  cannot obtain canonical route facts, or when typed config/runtime/route
  family/mask/segment/field/update/header/type/binding facts disagree with
  those canonical facts.
* Target artifact validation must consume provider-owned segment2 facts and
  fail closed on stale or missing provider/candidate facts, including stale
  indexed, strided, compare/select, conversion, reduction, MAcc, scalar-splat,
  unit-only, plain-vs-computed-mask, load-vs-store, interleave-vs-deinterleave,
  segment count/tuple/layout/leaf/field/update facts, memory form, mask,
  operand binding, header/type, target profile, and provider mirror facts.
* Common EmitC/export remains neutral. It may carry provider-built payloads and
  mirror metadata unchanged, but must not infer RVV segment2 semantics.

## Acceptance Criteria

* [x] Production provider/planning code exposes canonical plain segment2 facts
      for `segment2_interleave_unit_load` and
      `segment2_deinterleave_unit_store`.
* [x] Production provider/planning code exposes canonical computed-mask
      segment2 facts for `computed_masked_segment2_load_unit_store`,
      `computed_masked_segment2_store_unit_load`, and
      `computed_masked_segment2_update_unit_load`.
* [x] Target artifact validation consumes provider-owned segment2 facts instead
      of duplicating segment2 route-family truth locally for the same fields.
* [x] Plain segment2 validation checks provider-derived runtime ABI order,
      runtime ABI parameters, typed config policy, segment direction, segment
      count, tuple/layout/leaves, field roles/forms, operand binding,
      header/type mapping, target profile, and provider mirror.
* [x] Computed-mask segment2 validation checks provider-derived compare/mask
      facts, mask-tail owner/plan, inactive-lane and passthrough/no-write
      facts, segment tuple/layout/leaves, field roles/forms, update arithmetic
      facts, operand binding, header/type mapping, target profile, and provider
      mirror.
* [x] Target validation rejects plain segment2 facts on computed-mask segment2
      routes and computed-mask segment2 facts on plain segment2 routes.
* [x] Target validation rejects load/store/update and
      interleave/deinterleave cross-contamination among the five in-scope
      routes.
* [x] Focused C++ target artifact tests prove stale or missing segment2
      provider facts and stale candidate metadata mirrors fail closed.
* [x] Existing explicit and pre-realized lit/script dry-runs for the five
      segment2 routes still pass.
* [x] No descriptor-driven computation, common EmitC semantic inference,
      source-front-door positive route, route-id authority, artifact-name
      authority, mirror-only authority, or legacy `i32m1` authority is
      introduced.
* [x] If route emission, generated bundle behavior, runtime ABI order,
      segment tuple/field semantics, mask/passthrough behavior, update
      arithmetic behavior, or runtime behavior changes, real `ssh rvv`
      correctness is run for counts `0,1,16,17,257` with data patterns that
      distinguish interleave versus deinterleave, field0 versus field1,
      segment load/store leaves, masked inactive lanes, update behavior, tail
      preservation, and destination preservation. If this round only tightens
      provider/target validation facts, archived runtime evidence is reused
      with an explicit rationale.

## Completion Evidence

This round kept and repaired the existing dirty production diff. The repair
classifies computed-mask segment2 operations as segment2 memory routes for
canonical fact verification, while excluding them from the generic non-segment
computed-mask memory target/profile expectation path. Target artifact candidate
mirror validation now consumes provider-owned computed-mask segment2 facts for
computed-mask plan, mask, tuple/leaf, field, update arithmetic, header/type,
target profile, and provider mirror checks.

Checks run:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 8`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j 8`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv` on the five
  explicit selected-body segment2 target artifact tests from `build/test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv` on the five
  pre-realized selected-body segment2 target artifact tests from `build/test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv` on the thirteen
  segment2 generated-bundle dry-run/fail-closed script tests from `build/test`
* `rtk git diff --check`
* Bounded old-authority scan over added lines and touched production files for
  legacy `RVVI32M1` / `rvv-i32m1`, descriptor, direct-C, source-front-door,
  source-export, route-id, artifact-name, status, or supported-as-authority drift

No new `ssh rvv` run was required because this round only tightened provider
and target validation fact routing. It did not change route emission, generated
bundle behavior, runtime ABI order, segment tuple/field semantics,
mask/passthrough behavior, update arithmetic behavior, or runtime correctness
claims.

## Technical Approach

1. Inspect the existing segment2 provider/planning fact structs, route
   descriptions, statement plans, target validation helpers, generated-bundle
   script facts, and segment2 fixtures.
2. Extend or repair provider-owned plain and computed-mask segment2 fact
   surfaces so the canonical fields required by the specs are exposed from the
   RVV provider/planning layer.
3. Rewire provider validation to compare segment2 route plans/descriptions
   with the canonical segment2 surfaces before materialization.
4. Rewire target validation to consume the shared segment2 surfaces for
   provider route payload facts and candidate metadata mirrors.
5. Add focused C++ mutations for stale segment/mask/update/source/field/
   binding/header/profile/provider facts and stale candidate mirrors only where
   they prove the provider-owned boundary.
6. Run the smallest build/test set that exercises changed behavior, then
   finish/archive and commit if complete.

## Out Of Scope

* No new segment arities beyond existing segment2.
* No new route families.
* No strided or indexed redo, strided arithmetic, dot-reduce changes,
  compare/select changes, conversion changes, reduction changes, MAcc changes,
  frontend lowering, source-front-door, global tuning, dashboard, or broad
  smoke matrix work.
* No movement of RVV semantics into common EmitC/export.
* No runtime correctness/performance claim unless generated runtime behavior
  changes and real `ssh rvv` evidence is collected.

## Evidence Plan

* Validate Trellis task context.
* Build focused targets as needed:
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for:
  * `segment2-interleave-unit-load`
  * `segment2-deinterleave-unit-store`
  * `computed-masked-segment2-load`
  * `computed-masked-segment2-store`
  * `computed-masked-segment2-update`
* Run generated-bundle dry-runs for explicit and pre-realized segment2 forms
  that are script-supported.
* Run direct fail-closed checks for stale or missing segment2 provider facts
  and stale candidate mirrors.
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
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived task read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-strided-memory-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-strided-memory-production-validation-boundary/implement.jsonl`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-strided-memory-production-validation-boundary/check.jsonl`

Initial code focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/*segment2*.mlir`
* `test/Scripts/*segment2*.test`
* `test/Target/TargetArtifactExportTest.cpp`
