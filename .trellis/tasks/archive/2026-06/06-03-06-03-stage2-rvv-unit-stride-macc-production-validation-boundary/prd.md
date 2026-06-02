# Stage2 RVV unit-stride MAcc production validation boundary

## Goal

Close the production provider-to-target validation boundary for the existing
unit-stride RVV multiply-accumulate routes:
`macc_add` and `scalar_broadcast_macc_add`.

Target artifact validation must consume provider-owned MAcc facts for runtime
ABI order and parameter roles, SEW/LMUL/policy, typed compute op, vector-RHS
versus scalar-broadcast RHS form, multiply-add arithmetic kind, accumulator
and result layout, source/RHS/accumulator/output memory forms, route-family
plan, operand binding/header participation, required headers, C type mapping,
target leaf profile, and explicit `provider_supported_mirror` labels. It must
not reconstruct these facts from route ids, artifact names, fixtures,
candidate metadata mirrors, descriptors, common EmitC/export code, or exact
intrinsic spelling.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV unit-stride MAcc production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned clean short status through RTK.
* Initial `git log --oneline -8` started at
  `591fa371 rvv: validate plain segment2 route facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The immediately previous committed task closed plain segment2 production
  validation with provider-owned facts and target fail-closed checks.
* Archived MAcc realization, plain `macc_add`, scalar-broadcast MAcc, and
  computed-mask MAcc ABI tasks show that realization and runtime ABI behavior
  already exist. This round should tighten production validation unless live
  evidence shows route emission or runtime ABI behavior must change.

## Current Repository Evidence

Live inspection before implementation showed:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes provider
  fact structs/accessors for computed-mask MAcc and widening MAcc, but not for
  the bounded plain/scalar-broadcast unit-stride MAcc pair.
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp` already owns
  the plain/scalar-broadcast MAcc constants, route-family plan validation,
  runtime ABI order, provider mirror labels, header/type summaries, and
  computed-mask MAcc fact accessors.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` currently keeps
  duplicate target-local `kRVVPlainMAcc*` and
  `kRVVScalarBroadcastMAcc*` facts for runtime ABI order, binding summary,
  route-family plan, target leaf profile, header/type mapping, and layouts.
* `test/Target/TargetArtifactExportTest.cpp` already has broad MAcc positive
  and negative coverage; it should be extended to prove the new provider-owned
  fact surface and stale cross-contamination failures.

## Requirements

* Keep scope to `macc_add` and `scalar_broadcast_macc_add`.
* Add a production provider-owned unit-stride MAcc route fact surface or
  equivalent accessor in the RVV provider layer.
* The fact surface must record, for each route:
  * operation and memory form;
  * runtime ABI order and exported runtime ABI parameters;
  * SEW, LMUL, tail policy, and mask policy;
  * typed compute op name;
  * vector-RHS load versus scalar-broadcast RHS source form;
  * multiply-add arithmetic kind;
  * lhs/source, RHS, accumulator, output, and runtime-count roles;
  * accumulator/result layout;
  * source, RHS, accumulator, and destination memory forms;
  * plain or scalar-broadcast MAcc route-family plan;
  * route operand binding plan and exact summary with `abi` and `hdr`
    participation for every exported runtime ABI parameter;
  * required headers and C type mapping summary;
  * target leaf profile and explicit `provider_supported_mirror` label.
* Rewire MAcc target artifact validation so `macc_add` and
  `scalar_broadcast_macc_add` compare rebuilt provider descriptions and
  candidate metadata mirrors against provider-owned facts rather than
  target-local constants.
* Reject stale plain-MAcc facts on scalar-broadcast MAcc, stale
  scalar-broadcast facts on plain MAcc, missing accumulator facts, stale
  arithmetic kind, stale operand binding, stale header/type mapping, stale
  target profile, stale provider mirror, stale candidate mirrors, and
  accidental segment/indexed/masked/widening/reduction fallback.
* Keep common EmitC/export neutral. It may carry provider-built payloads and
  mirrors unchanged, but must not infer MAcc semantics.
* Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

* [ ] Production RVV provider exposes provider-owned unit-stride MAcc route
      facts for `macc_add` and `scalar_broadcast_macc_add`.
* [ ] Target validation consumes provider-owned facts for runtime ABI order,
      parameter roles, SEW/LMUL/policy, typed operation, vector-RHS versus
      scalar-broadcast RHS form, arithmetic kind, accumulator/result layout,
      memory forms, route-family plan, binding plan/summary, header/type
      mapping, target profile, and provider mirror.
* [ ] Target validation rejects stale plain/scalar-broadcast cross
      contamination, missing accumulator facts, stale binding summary, stale
      header/type facts, stale target profile, stale provider mirror, stale
      candidate metadata, and accidental computed-mask/widening/reduction/
      segment/indexed fallback.
* [ ] Focused C++ target validation tests prove the new fail-closed boundary.
* [ ] Existing lit/script dry-runs for explicit and pre-realized `macc_add`
      and `scalar_broadcast_macc_add` still pass.
* [ ] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy `i32m1` route
      authority is introduced.
* [ ] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [ ] Trellis finish/archive and one coherent commit complete this round if
      the module behavior is complete.

## Technical Approach

1. Add `RVVUnitStrideMAccRouteFacts` or equivalent provider-owned accessor in
   the RVV route provider public header and MAcc route-family owner
   implementation.
2. Populate facts from the same owner-local constants and typed route-family
   values used by plain/scalar-broadcast MAcc route planning.
3. Build the exact route operand binding summary from provider facts, including
   `abi` and `hdr` tokens for every exported ABI parameter.
4. Rewire target MAcc validation helpers to query provider facts for
   `macc_add` and `scalar_broadcast_macc_add`; keep computed-mask MAcc facts
   on their existing provider accessors.
5. Add focused C++ tests that compare live route descriptions to provider
   facts and mutate plain/scalar-broadcast facts, mirrors, and stale
   cross-family residue to prove fail-closed behavior.
6. Run only the focused build, C++ target test, lit filters, generated-bundle
   dry-runs, bounded authority scan, and diff check required by this boundary.

## Out Of Scope

* Computed-mask MAcc, runtime-scalar computed-mask MAcc, widening MAcc, dot or
  reduction routes, segment/indexed routes, dtype/LMUL clone batches,
  source-front-door positive routes, high-level frontend lowering, and
  evidence-only packaging.
* Re-implementing selected-body realization or changing generated bundle
  runtime ABI semantics unless a focused check exposes a live defect.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, descriptors, C strings, test
  names, spec prose, or mirror metadata as route authority.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `macc-add` and `scalar-broadcast-macc-add` target
  fixtures.
* Run generated-bundle dry-runs for explicit and pre-realized `macc_add` and
  `scalar_broadcast_macc_add` counts `0,1,16,17,257`.
* Run direct fail-closed checks for stale or missing unit-stride MAcc provider
  facts and candidate mirrors through focused C++ mutations.
* Run a bounded old-authority scan over touched files for legacy `i32m1`,
  descriptor, source-front-door/source-artifact, direct-C/source-export,
  route-id, artifact-name, and mirror-only route authority drift.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior, or
  runtime ABI semantics change. If this round only tightens production
  validation, reuse archived RVV correctness evidence from the existing plain
  and scalar-broadcast MAcc ABI tasks and state that no new runtime claim
  changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/validation/index.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-macc-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-plain-macc-add-vector-vector-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-scalar-broadcast-macc-add-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-macc-add-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-plain-segment2-production-validation-boundary/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*macc-add*.mlir`
* `test/Scripts/*macc-add*.test`

## Final Outcome

Completed.

Production changes:

* Added provider-owned `RVVUnitStrideMAccRouteFacts` for `macc_add` and
  `scalar_broadcast_macc_add`.
* Rewired target MAcc validation to consume provider facts for operation,
  memory form, SEW/LMUL/policy, runtime ABI, typed compute op, RHS form,
  route-family plan, binding summary, header/type summaries, target profile,
  provider mirror, and accumulator/result layouts.
* Added fail-closed target validation coverage for stale plain/scalar-broadcast
  cross-contamination, stale binding/header/type/profile/provider facts,
  missing accumulator role facts, and stale computed-mask or other route-family
  residue on unit-stride MAcc routes.
* Updated one adjacent computed-mask MAcc dry-run FileCheck assertion to check
  the current `tcrv_rvv.source_memory_form` /
  `tcrv_rvv.destination_memory_form` route metadata mirror fields.

Checks run:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter macc-add`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter scalar-broadcast-macc-add`
* Focused lit run over explicit/pre-realized `macc_add` and
  `scalar_broadcast_macc_add` target/script tests.
* `rtk git diff --check`
* Bounded added-line authority scan over touched files for `RVVI32M1`,
  `rvv-i32m1`, dtype-prefixed `tcrv_rvv.i32_*`, descriptor,
  source-front-door/source-artifact, direct-C/source-export, and exact
  `__riscv_*_i32m1` authority residue.

Runtime evidence:

* No new `ssh rvv` run was required because this round tightened production
  provider-to-target validation and did not change route emission, generated
  bundle runtime behavior, or runtime ABI semantics.
* Existing archived plain and scalar-broadcast MAcc runtime evidence remains
  the executable correctness evidence for those unchanged runtime paths.
