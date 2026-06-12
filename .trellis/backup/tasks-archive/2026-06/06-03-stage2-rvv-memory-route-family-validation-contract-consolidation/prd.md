# Stage2 RVV memory route-family validation contract consolidation

## Goal

Consolidate the production RVV memory route-family target validation boundary
so existing memory movement families are validated through one provider-owned
memory route validation contract shape. The target artifact validator must
consume facts rebuilt from the selected typed `tcrv_rvv` body through the RVV
provider and must not reconstruct operation, dtype/config, policy, intrinsic,
runtime ABI, mask, passthrough, stride/index/segment, header/type, target
profile, or route-family support authority from route ids, artifact names,
metadata mirrors, descriptors, scripts, fixture names, C strings, or common
EmitC/export code.

The bounded route families are existing base/unit-stride, masked unit-stride,
strided, indexed, plain segment2, and computed-mask segment2 memory movement
routes. This task does not add route coverage.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV memory route-family validation contract consolidation`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `213fdf24 rvv: validate masked unit memory provider facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires the active RVV chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact, with real
  `ssh rvv` evidence only for runtime/correctness/performance claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines target artifact
  route-family validators as consumers of rebuilt provider descriptions and
  lowerable routes. Metadata remains mirror evidence after route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines provider-owned fact
  surfaces for base memory movement, computed-mask strided/indexed memory,
  plain segment2 memory, and computed-mask segment2 memory.
* Archived production-validation tasks show that base indexed, computed-mask
  indexed, strided, plain/computed-mask segment2, and masked unit-stride memory
  families were each individually wired to provider-owned facts. The repeated
  target-side validation shape is now the drift risk.

## Requirements

* Keep this owner bounded to the RVV plugin-owned memory route validation
  contract consumed by target artifact validation.
* Reuse existing provider fact surfaces. Add only the missing common contract
  view/helper needed to make target artifact validation consume a canonical
  memory contract across existing families.
* Preserve family-specific provider authority. Base/unit/masked/strided/indexed
  facts remain provider/planning facts; plain segment2 and computed-mask
  segment2 keep their dedicated segment fact surfaces and statement checks.
* Target validation must compare rebuilt provider route descriptions,
  lowerable route payloads, and candidate metadata mirrors against the
  provider-owned contract. Candidate metadata may only mirror provider facts.
* The contract must cover, where applicable:
  operation, memory form, typed compute op, load/store/update roles, source and
  destination memory forms, stride source/type/unit, index source/EEW/unit/
  uniqueness, segment lane/field/update axes, mask source/role/form, mask-tail
  policy, inactive-lane/passthrough behavior, runtime ABI order and parameter
  roles, route operand binding plan/summary with header participation,
  required headers, C type mapping, type/header/intrinsic leaves, target leaf
  profile, `provider_supported_mirror`, and route-family mirrors.
* Fail closed when provider facts are missing, stale, cross-family,
  cross-route, or mismatched for header/type/intrinsic/profile/mirror/runtime
  ABI/mask/passthrough/stride/index/segment fields.
* Keep common EmitC/export neutral. Do not move RVV route semantics into common
  EmitC or target metadata.
* Do not change route emission, generated runtime semantics, runtime ABI order,
  mask/tail behavior, passthrough/destination preservation, intrinsic behavior,
  or performance behavior. If implementation requires any such change, real
  `ssh rvv` evidence becomes required before completion.

## Acceptance Criteria

* [x] Production target artifact validation has a focused consolidation diff
      that consumes a canonical provider-owned memory validation contract for
      the in-scope existing memory route families.
* [x] The contract consumer covers operation, memory form, runtime ABI,
      binding, header/type, intrinsic/profile, provider mirror, route-family
      mirror, and family-specific mask/passthrough/stride/index/segment axes
      without target-side semantic reconstruction.
* [x] Base/unit, masked unit, strided, indexed, plain segment2, and
      computed-mask segment2 routes still dispatch through provider
      descriptions and family-specific validators, but shared repeated field
      checks are expressed through the common memory contract helper/view.
* [x] Target validation rejects missing, stale, cross-family, and cross-route
      provider facts and candidate metadata mirrors before bundle acceptance.
* [x] Focused C++ target artifact tests prove the contract consumer rejects at
      least representative stale provider/candidate facts across non-segment
      memory and segment2 memory. If the full segment2 contract is too large
      for one round, complete unit+strided+indexed consumption and leave a
      truthful segment2 continuation point.
* [x] Existing explicit/pre-realized lit/script checks for the touched memory
      families still pass.
* [x] No new source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, mirror-only authority, exact intrinsic spelling authority, or
      legacy `i32m1` route authority is introduced.
* [x] Focused builds/tests, bounded old-authority scan, `rtk git diff --check`,
      Trellis finish/archive, clean worktree, and one coherent commit complete
      this round if the module behavior is complete.

## Technical Approach

1. Inspect the live provider fact structs/accessors and target route-family
   validation helpers for memory routes.
2. Identify the repeated target-side comparison shape across base/masked,
   strided, indexed, plain segment2, and computed-mask segment2 validators.
3. Add a small target-side contract view/helper that receives provider-owned
   facts and expected mirror keys, then checks provider description fields and
   candidate mirrors exactly. The helper must not define RVV semantics; it
   only packages comparisons against provider facts.
4. Rewire the existing memory validators to feed their provider-owned fact
   surfaces into that helper. Keep family-specific statement checks and
   route-family dispatch local to their existing validators.
5. Add focused C++ fail-closed mutations proving stale or cross-family memory
   contract facts cannot be accepted through provider descriptions or
   candidate mirrors.
6. Run the smallest build/lit/script set that exercises the touched validators,
   then finish/archive and commit if complete.

## Out Of Scope

* No new RVV routes, dtype or LMUL clone batches, source-front-door routes,
  high-level frontend lowering, arithmetic/reduction/compare/conversion
  rewrites, dashboards, broad smoke matrices, or evidence-only packaging.
* No movement of RVV semantics into common EmitC/export.
* No resurrection of legacy `i32m1` authority or descriptor/source-export
  route authority.
* No runtime correctness/performance claim unless generated runtime behavior
  changes and real `ssh rvv` evidence is collected.

## Evidence Plan

* Build and run `tianchenrv-target-artifact-export-test`.
* Build and run `tianchenrv-rvv-extension-plugin-test` if provider closure
  changes.
* Build `tcrv-opt` and `tcrv-translate` if route fixtures or scripts are
  touched.
* Run focused lit filters for the memory families whose validation consumers
  change.
* Run generated-bundle dry-runs/direct fail-closed script tests only for
  touched memory fixtures.
* Run a bounded old-authority scan over touched files.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` unless route emission, runtime ABI order, mask/tail,
  load/store, passthrough, destination preservation, emitted intrinsic
  behavior, runtime correctness, or performance behavior changes.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-masked-unit-memory-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-base-memory-route-family-production-validation-closeout/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-strided-memory-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-memory-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-indexed-memory-validation/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-segment2-memory-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-segment2-production-validation-boundary/prd.md`

Likely live files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/TargetArtifactExportTest.cpp`
* focused memory fixtures under `test/Target/RVV` and `test/Scripts`

## Completion Evidence

* Added a local `RVVMemoryRouteMetadataMirrorContract` target-side helper that
  packages provider-derived memory metadata mirror checks and empty stale-mirror
  rejection without defining RVV route semantics in target validation.
* Rewired base-memory candidate mirror validation for strided load/unit store,
  unit load/strided store, indexed gather/unit store, indexed scatter/unit
  load, masked unit load/store, and masked unit store through the shared memory
  mirror contract. Expected values come from `RVVBaseMemoryMovementRouteFacts`
  and, for masked unit typed compute, `RVVUnitStrideMaskedMemoryRouteFacts`.
* Rewired plain segment2 lane/field candidate mirror validation and segment2
  common/stale mirror validation through the same memory mirror contract while
  preserving existing segment2 family dispatch, provider-fact validation, and
  route statement-plan checks.
* Did not change provider fact accessors, route planning, common EmitC,
  generated route payloads, runtime ABI order, emitted intrinsics, mask/tail
  behavior, passthrough/destination preservation, or runtime semantics.
* Existing C++ target artifact mutation coverage already exercises stale
  provider and candidate mirrors for base memory and segment2 memory; no new
  duplicate mutation test was needed for this refactor-shaped consolidation.
* `ssh rvv` was not run because this round only consolidated target-side
  metadata mirror validation shape and made no runtime/correctness/performance
  claim.

Checks run:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter '(strided-load-unit-store|unit-load-strided-store|indexed-gather-unit-store|indexed-scatter-unit-load|masked-unit-(load-store|store)|segment2-(deinterleave|interleave)|computed-masked-segment2-(load|store|update))'` from `build/test`, selecting 60 tests and passing 60.
* `rtk git diff --check`
* Bounded old-authority scan over added lines. The only `supported` hit was the
  required `provider_supported_mirror` contract key; no new positive legacy
  `i32m1`, source-front-door, source-export, descriptor/direct-C,
  route-id/artifact-name, or mirror-only route authority was introduced.

## Spec Update Judgment

No `.trellis/spec/` update was needed. This round did not add a new
cross-layer API, provider fact surface, route family, executable behavior, or
validation rule. It implemented the already documented contract that RVV target
artifact validation consumes provider-owned facts and treats metadata as
mirrors only.
