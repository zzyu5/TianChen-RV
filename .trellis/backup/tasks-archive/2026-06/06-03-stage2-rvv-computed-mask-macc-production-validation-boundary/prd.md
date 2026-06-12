# Stage2 RVV computed-mask MAcc production validation boundary

## Goal

Close the production provider-to-target validation boundary for the existing
computed-mask multiply-accumulate routes, bounded to
`computed_masked_macc_add` and the closely related
`runtime_scalar_cmp_masked_macc_add` path where it shares the same provider
fact surface.

Target artifact validation must consume provider-owned MAcc facts for runtime
ABI order and parameter roles, SEW/LMUL/policy, typed compute op, compare
predicate, mask producer/source/form, payload/accumulator/output roles,
multiply-add arithmetic kind, inactive-lane and passthrough contracts, route
operand binding/header participation, required headers, C type mapping, target
leaf profile, and explicit `provider_supported_mirror` labels. It must not
reconstruct these facts from route ids, artifact names, fixtures, candidate
metadata mirrors, descriptors, common EmitC/export code, scripts, or exact
intrinsic spelling.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask MAcc production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned no file entries through RTK.
* Initial `git log --oneline -8` started at
  `8c819ae0 rvv: validate unit-stride macc route facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The immediately previous task added provider-owned facts and target
  validation for `macc_add` and `scalar_broadcast_macc_add`.
* Archived computed-mask and runtime-scalar computed-mask MAcc ABI tasks show
  existing realization, dry-run, and `ssh rvv` correctness evidence. This
  round should tighten production validation unless live evidence requires a
  route emission or runtime ABI behavior change.

## Current Repository Evidence

Live inspection before implementation showed:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` already exposes
  `RVVComputedMaskMAccRouteFacts` and
  `RVVRuntimeScalarComputedMaskMAccRouteFacts`, plus provider accessors for
  both supported computed-mask MAcc operations.
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp` already owns
  canonical runtime ABI order, target leaf profile, provider mirror, header,
  C type mapping, predicate, mask, accumulation, layout, and binding-summary
  facts for both computed-mask MAcc variants.
* The computed-mask MAcc fact structs do not yet expose SEW/LMUL/policy,
  runtime control plan, arithmetic kind, explicit operand roles, or runtime
  ABI parameter lists like the unit-stride MAcc fact surface does.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` consumes many
  computed-mask facts from provider accessors, but runtime ABI parameter roles
  are checked only through route statement shape, not directly against a
  provider-owned parameter list.
* `test/Target/TargetArtifactExportTest.cpp` has stronger positive mirror and
  stale-fact coverage for the runtime-scalar computed-mask MAcc variant than
  for the vector computed-mask MAcc variant.

## Requirements

* Keep scope to `computed_masked_macc_add` and
  `runtime_scalar_cmp_masked_macc_add`.
* Extend the provider-owned computed-mask MAcc fact surface, or an equivalent
  accessor, so both variants expose:
  * operation and memory form;
  * SEW, LMUL, tail policy, mask policy, and runtime control plan;
  * runtime ABI order and exported runtime ABI parameters;
  * typed compute op name and multiply-add arithmetic kind;
  * compare operand roles or runtime-scalar threshold role;
  * payload lhs/rhs, accumulator, output, and runtime-count roles;
  * compare predicate kind, mask role/source/memory form, and mask producer
    source;
  * computed-mask accumulation route-family plan, compute suffix,
    accumulator/result contracts, inactive-lane contract, passthrough layout,
    source/destination memory forms, and memory layout;
  * route operand binding plan and exact summary with `abi` and `hdr`
    participation for every exported runtime ABI parameter;
  * required headers, C type mapping summary, target leaf profile, and
    explicit `provider_supported_mirror` label.
* Rewire or tighten target validation so both variants compare rebuilt
  provider descriptions and candidate metadata mirrors against provider-owned
  facts rather than target-local constants or mirror-only metadata.
* Reject stale vector computed-mask facts on runtime-scalar computed-mask MAcc,
  stale runtime-scalar facts on vector computed-mask MAcc, missing accumulator
  or mask facts, stale arithmetic kind, stale runtime ABI roles, stale operand
  binding, stale header/type mapping, stale target profile, stale provider
  mirror, stale candidate mirrors, and accidental segment/indexed/widening/
  reduction fallback.
* Keep common EmitC/export neutral. It may carry provider-built payloads and
  mirrors unchanged, but must not infer MAcc or mask semantics.
* Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

* [ ] Production RVV provider exposes provider-owned computed-mask MAcc route
      facts for `computed_masked_macc_add` and
      `runtime_scalar_cmp_masked_macc_add`.
* [ ] Target validation consumes provider-owned facts for runtime ABI order,
      parameter roles, SEW/LMUL/policy, typed operation, arithmetic kind,
      compare predicate, mask producer/source/form, payload and accumulator
      roles, inactive-lane and passthrough contracts, route-family plan,
      binding plan/summary, header/type mapping, target profile, and provider
      mirror.
* [ ] Target validation rejects vector/runtime-scalar cross-contamination,
      missing accumulator or mask facts, stale arithmetic kind, stale binding
      summary, stale header/type facts, stale target profile, stale provider
      mirror, stale candidate metadata, and accidental segment/indexed/
      widening/reduction fallback.
* [ ] Focused C++ target validation tests prove the new fail-closed boundary.
* [ ] Existing lit/script dry-runs for explicit and pre-realized
      `computed_masked_macc_add` and any touched
      `runtime_scalar_cmp_masked_macc_add` form still pass.
* [ ] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy `i32m1` route
      authority is introduced.
* [ ] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [ ] Trellis finish/archive and one coherent commit complete this round if
      the module behavior is complete.

## Technical Approach

1. Extend `RVVComputedMaskMAccRouteFacts` and
   `RVVRuntimeScalarComputedMaskMAccRouteFacts` with the missing provider-owned
   typed config, runtime ABI parameter, arithmetic, and logical role facts.
2. Populate those fields from the same MAcc owner-local constants and typed
   route-family values used by route planning.
3. Rewire MAcc target validation to check computed-mask runtime ABI
   parameters, SEW/LMUL/policy/runtime-control facts, arithmetic kind, and
   logical roles directly against provider facts.
4. Add focused C++ positive checks proving both computed-mask MAcc route
   descriptions mirror provider facts, with special emphasis on the vector
   computed-mask variant.
5. Add focused C++ negative checks for vector/runtime-scalar cross
   contamination, stale runtime ABI roles, stale mask/accumulator facts, stale
   arithmetic, stale header/type/profile/provider facts, and candidate mirror
   drift.
6. Run only the focused build, C++ target test, lit filters,
   generated-bundle dry-runs, bounded authority scan, and diff check required
   by this validation boundary.

## Out Of Scope

* Widening MAcc, dot/reduction routes, standalone compare/select expansion,
  masked memory expansion, segment/indexed routes, dtype/LMUL clone batches,
  source-front-door positive routes, high-level frontend lowering, and
  evidence-only packaging.
* Re-implementing selected-body realization or changing generated bundle
  runtime ABI semantics unless a focused check exposes a live defect.
* Reopening unit-stride MAcc validation as the owner.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, descriptors, C strings, test
  names, spec prose, or mirror metadata as route authority.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `computed-masked-macc-add` and any touched
  `runtime-scalar-cmp-masked-macc-add` target/script tests.
* Run generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_macc_add`, and for runtime-scalar computed-mask MAcc if
  touched.
* Run direct fail-closed checks for stale or missing MAcc/mask/provider facts
  through focused C++ mutations.
* Run a bounded old-authority scan over touched files for legacy `i32m1`,
  descriptor, source-front-door/source-artifact, direct-C/source-export,
  route-id, artifact-name, and mirror-only route authority drift.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior, or
  runtime ABI semantics change. If this round only tightens production
  validation, reuse archived RVV correctness evidence from the existing
  computed-mask and runtime-scalar computed-mask MAcc ABI tasks and state that
  no new runtime claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-macc-add-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-macc-add-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-computed-mask-macc-realization-boundary/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*computed-masked-macc-add*.mlir`
* `test/Target/RVV/*runtime-scalar-cmp-masked-macc-add*.mlir`
* `test/Scripts/*computed-masked-macc-add*.test`
* `test/Scripts/*runtime-scalar-cmp-masked-macc-add*.test`
