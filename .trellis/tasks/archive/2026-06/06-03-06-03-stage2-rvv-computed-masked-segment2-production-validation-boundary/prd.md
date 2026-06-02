# Stage2 RVV computed masked segment2 production validation boundary

## Goal

Close the production provider-to-target validation boundary for computed
masked segment2 RVV memory routes:
`computed_masked_segment2_load_unit_store`,
`computed_masked_segment2_store_unit_load`, and
`computed_masked_segment2_update_unit_load`.

The target artifact validator must consume provider-owned typed `tcrv_rvv`
body/config/runtime facts for segment2 lane roles, compare-produced masks,
inactive-lane policy, source/destination memory roles, route-family plan,
operand binding, headers, C type mapping, target profile, and explicit
provider support mirrors. It must not reconstruct those facts from route ids,
artifact names, fixture names, script constants, candidate metadata mirrors,
descriptors, common EmitC/export code, or exact intrinsic spellings.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed masked segment2 production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned a clean short-status through RTK.
* Initial `git status --porcelain=v1 -b` showed `## main...origin/main
  [ahead 551]`.
* Initial `git log --oneline -8` started at
  `b05967d9 rvv: validate computed masked indexed memory facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The immediately previous indexed task
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-indexed-memory-validation/`
  added provider-owned computed-mask indexed route facts and rewired target
  validation to consume those facts instead of target-local duplicate truth.
* Archived computed-mask segment2 load/store/update ABI tasks already proved
  executable paths, generated-bundle dry-runs, and `ssh rvv` correctness for
  their individual routes. This round should not redo runtime semantics unless
  route emission or runtime ABI behavior changes.

## Current Repository Evidence To Confirm

Live inspection must confirm the current production shape before editing:

* Provider/planning code already has segment2 route-family planning owners,
  computed-mask memory family plans, segment2 statement plans, and
  `RVVSelectedBodyEmitCRouteDescription` fields for segment2/mask/field/header
  facts.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already contains
  segment2 target validators, but may still encode route-family expectations
  as target-local constants instead of consuming one provider-owned fact
  surface shared with provider/planning.
* `test/Target/TargetArtifactExportTest.cpp` already contains targeted
  segment2 fail-closed coverage from the earlier individual ABI tasks; this
  task should add only the focused mutations needed to prove the production
  owner boundary.
* Explicit and pre-realized computed-mask segment2 MLIR fixtures and
  generated-bundle dry-run tests already exist under `test/Target/RVV/` and
  `test/Scripts/`.

## Requirements

* Keep scope to the three computed-mask segment2 route forms:
  load/unit-store, store/unit-load, and update/unit-load.
* Provide or reuse a production provider-owned computed-mask segment2 fact
  surface that records, for each route:
  * operation and memory form;
  * runtime ABI order and runtime ABI parameters;
  * SEW, LMUL, tail policy, and mask policy;
  * typed compute op and segment2 lane/field roles;
  * compare predicate and compare-produced mask producer/source facts;
  * `mask_role`, `mask_source`, `mask_memory_form`, mask type, and C type
    mapping;
  * inactive-lane contract and passthrough/no-write behavior;
  * source and destination memory forms;
  * segment count and segment2 memory layout;
  * computed-mask memory route-family plan;
  * route operand binding plan and exact summary with `abi` and `hdr`
    participation for every exported runtime ABI parameter;
  * required headers and C type mapping summary;
  * target leaf profile and explicit `provider_supported_mirror` label;
  * route-specific statement facts such as masked segment load, tuple create,
    field extraction, field store, masked segment store, and update arithmetic.
* Make target artifact validation compare rebuilt provider descriptions and
  candidate metadata mirrors against the provider-owned facts.
* Reject stale load facts on store/update, stale store/update facts on load,
  missing mask facts, stale segment lane facts, stale typed-compute facts,
  stale route-family plan, stale header/type mapping, stale target profile,
  stale provider mirror, stale binding plan/summary, and accidental scalar,
  unit, strided, indexed, descriptor, direct-C, source-front-door, or legacy
  `i32m1` fallback.
* Keep common EmitC/export neutral. It may carry provider-built payloads and
  mirrors unchanged, but must not infer segment2 or computed-mask semantics.
* Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

* [ ] Production RVV provider/planning and/or target validation has a focused
      diff that makes computed masked segment2 validation consume
      provider-owned facts instead of target-local switch constants.
* [ ] Target validation checks provider-derived runtime ABI order/parameters,
      SEW/LMUL/policy, typed compute op, segment lane/field roles, compare
      predicate, mask role/source/form, inactive-lane contract, source and
      destination memory forms, segment2 memory layout, route-family plan,
      binding plan/summary, header/type mapping, target profile, provider
      mirror, and route-specific statement facts for load, store, and update.
* [ ] C++ target validation tests prove fail-closed behavior for stale or
      missing computed masked segment2 facts, including load-vs-store/update
      cross-contamination and stale candidate metadata mirrors.
* [ ] Existing generated-bundle dry-run tests for explicit and pre-realized
      computed masked segment2 load/store/update still pass.
* [ ] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy `i32m1` route
      authority is introduced.
* [ ] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [ ] Trellis finish/archive and one coherent commit complete this round if
      the task is finished.

## Technical Approach

1. Inspect the computed-mask indexed validation commit pattern from `b05967d9`
   and the current segment2 target validator/provider plan code.
2. Add a provider-owned `RVVComputedMaskSegment2MemoryRouteFacts` surface or
   equivalent accessors if the current code lacks one. Keep it in the RVV
   plugin/provider layer, not target-local validation.
3. Rewire target segment2 validation to consume the provider facts. Target
   code remains responsible for verifying rebuilt routes and candidate
   mirrors, not for inventing route semantics.
4. Add focused C++ mutation tests for provider descriptions and candidate
   mirrors, emphasizing stale load/store/update cross-contamination, stale
   mask/segment/header/type/binding/profile/provider facts, and accidental
   fallback forms.
5. Run the smallest lit/script/build checks that exercise the changed surface
   plus generated-bundle dry-runs for explicit and pre-realized segment2
   load/store/update.

## Out Of Scope

* Segment width greater than 2.
* Broad masked route matrices.
* Indexed or strided route expansion.
* Reductions, MAcc, dot/contraction, standalone compare/select, dtype/LMUL
  clone batches, high-level frontend/source-front-door positive routes, or
  global tuning/database/dashboard work.
* Reworking the previously completed computed masked indexed validation task.
* Moving RVV semantics into common EmitC/export.
* Evidence-only packaging as the main deliverable.

## Evidence Plan

* Build the focused target artifact export test target and any needed RVV
  provider test target.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for computed-masked-segment2 load/store/update target
  fixtures.
* Run generated-bundle dry-run lit checks for explicit and pre-realized
  computed-masked-segment2 load/store/update forms.
* Run direct fail-closed checks through C++ mutation tests for stale or
  missing masked/segment/provider facts.
* Run a bounded old-authority scan over touched files for legacy `i32m1`,
  descriptor, source-front-door/source-artifact, direct-C/source-export,
  route-id, artifact-name, and mirror-only route authority drift.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior, or
  runtime ABI semantics change. If this round only tightens production
  validation, reuse archived RVV correctness evidence from the individual
  computed-masked segment2 load/store/update tasks and state that no new
  runtime/correctness claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-indexed-memory-validation/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-load-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-segment2-store-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-update-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-load-unit-store-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-segment2-store-unit-load-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-segment2-update-unit-load-abi/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/RVV/*computed-masked-segment2*.mlir`
* `test/Scripts/*computed-masked-segment2*.test`
* `test/Target/TargetArtifactExportTest.cpp`
