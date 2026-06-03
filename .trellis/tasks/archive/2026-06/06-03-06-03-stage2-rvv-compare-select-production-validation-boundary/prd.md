# Stage2 RVV compare/select production validation boundary

## Goal

Close the production provider-to-target validation boundary for existing RVV
compare/select routes, bounded to `cmp_select` and
`runtime_scalar_cmp_select`. Include `runtime_scalar_dual_cmp_mask_and_select`
only if live inspection shows it already shares the same coherent
compare/select fact surface without expanding this round into a separate route
owner.

The RVV provider must expose canonical compare/select route facts derived from
typed `tcrv_rvv` body/config/runtime facts. Target artifact validation must
consume those provider-owned facts for predicate, compare intrinsic, select
layout, mask type/composition, true/false operand roles, runtime scalar
threshold roles, runtime ABI roles/order, SEW/LMUL/policy, type/header
participation, target leaf profile, route operand binding, and explicit
provider mirrors instead of rebuilding or trusting route ids, artifact names,
fixture names, C strings, intrinsic spellings, descriptor residue, or metadata
mirrors.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `06bf5324 rvv: validate widening conversion route facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization/provider facts -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness/performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires RVV route support to
  start from typed `tcrv_rvv` body/config/runtime facts and plugin-owned
  legality, realization, route provider output, intrinsic mapping, C/RVV vector
  type mapping, ABI mapping, and fail-closed diagnostics.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the shared route
  family fact surface contract: provider facts are canonical when consumed by
  both provider planning and target artifact validation; target validation must
  not define a duplicate route-family truth source.
* Archived compare/select artifact tasks already established executable paths
  for `cmp_select`, `runtime_scalar_cmp_select`, and
  `runtime_scalar_dual_cmp_mask_and_select` through selected-body realization,
  generated-bundle dry-runs, and `ssh rvv` correctness evidence.
* The immediately preceding completed task
  `06bf5324 rvv: validate widening conversion route facts` provides the closest
  current implementation pattern for a production validation boundary:
  complete the provider-owned fact struct/accessor, propagate the facts through
  route planning/description, make target validation compare against the same
  accessor, add fail-closed C++ mutations, and avoid rerunning runtime evidence
  unless emitted behavior or ABI semantics change.

## Requirements

* Keep the implementation owner bounded to the production provider-to-target
  validation surface for `cmp_select` and `runtime_scalar_cmp_select`.
* Preserve the production chain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized compare/select body ->
  RVV plugin-local realization -> realized `tcrv_rvv` body -> RVV
  route/provider facts -> `TCRVEmitCLowerableRoute` -> target artifact
  validation.
* The provider-owned compare/select fact surface must expose and provider
  planning / target validation must consume:
  * operation and typed compute op;
  * source/result element type, SEW, LMUL, tail policy, mask policy, memory
    forms, mask type, mask C type, and VL/control participation;
  * predicate kind, compare intrinsic, and select intrinsic/layout;
  * true/false operand roles and passthrough/select result layout;
  * runtime ABI order and runtime ABI parameter roles/C types;
  * runtime scalar threshold role and C type for runtime-scalar compare/select;
  * required headers, C type mapping, route operand binding plan/summary, target
    leaf profile, and explicit `provider_supported_mirror`;
  * stale fact separation between plain vector/vector compare/select and
    runtime-scalar compare/select.
* Target artifact validation must reject stale or missing provider-owned facts:
  stale elementwise, conversion, reduction, MAcc, memory-movement,
  scalar-splat, segment/indexed, or source-front-door facts on compare/select
  routes; stale plain facts on runtime-scalar routes; stale runtime-scalar facts
  on plain routes; stale predicate, compare intrinsic, select layout, SEW/LMUL,
  policy, runtime ABI, operand binding, header/type, target profile, provider
  mirror, and candidate metadata mirrors.
* If `runtime_scalar_dual_cmp_mask_and_select` is already wired through the
  same compare/select provider fact surface, include it; otherwise leave a
  truthful continuation point without expanding this task into the dual-mask
  route owner.
* Common EmitC/export remains neutral. It may carry provider-built payloads and
  metadata mirrors, but must not infer RVV compare/select semantics.

## Acceptance Criteria

* [ ] Provider-owned compare/select route facts expose the complete validation
      surface for `cmp_select` and `runtime_scalar_cmp_select`.
* [ ] Provider route-family plan derivation/validation copies or validates
      predicate, compare/select intrinsic/layout, type/config/policy, memory
      form, mask type/C type, runtime ABI, binding, header/type, target profile,
      and provider mirror facts from canonical accessors.
* [ ] Target artifact validation consumes the same provider accessors and
      rejects stale cross-family, plain-vs-runtime-scalar, predicate/layout,
      type/config/policy, runtime ABI, binding, header/type, target profile, and
      provider mirror facts before accepting a bundle.
* [ ] Candidate metadata mirror validation includes provider-derived
      compare/select facts and remains fail-closed for non-compare/select
      route-family mirrors.
* [ ] Focused C++ target artifact tests prove stale or missing
      compare/select provider facts fail closed for both owned route variants.
* [ ] Existing lit/script dry-runs for explicit and pre-realized
      compare/select and runtime-scalar compare/select still pass.
* [ ] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy i32 route
      authority is introduced.
* [ ] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [ ] Trellis finish/archive and one coherent commit complete this round if the
      module behavior is complete.

## Technical Approach

1. Inspect existing compare/select route facts, route-family plan structs,
   statement plan owners, and target artifact validation to identify which facts
   are still target-local or mirror-derived.
2. Complete the provider-owned fact surface for `cmp_select` and
   `runtime_scalar_cmp_select`, using parameterized operation/config facts when
   SEW or LMUL affects types, intrinsic spelling, ABI C types, or binding
   summaries.
3. Rewire route planning/description propagation so target validation can
   compare compare/select structural facts directly instead of inferring them
   from C strings or artifact metadata.
4. Rewire target artifact validation and candidate mirror checks to compare the
   added provider facts and reject stale plain-vs-runtime-scalar or cross-family
   payloads.
5. Add focused C++ mutations in `TargetArtifactExportTest.cpp` for the added
   validation boundary.
6. Update durable specs only if implementation establishes a stronger contract
   not already captured by `.trellis/spec/lowering-runtime/emitc-route.md`.

## Out Of Scope

* New compare predicates beyond the existing selected-body surface, broad
  mask/tail expansion, new dtype/LMUL clone batches, conversion changes,
  reduction changes, MAcc changes, segment/indexed routes, high-level frontend
  lowering, source-front-door routes, global tuning, dashboards, or
  evidence-only packaging.
* Redoing widening conversion, standalone reduction, or MAcc production
  validation as this round's owner.
* Changing generated bundle runtime ABI order, predicate semantics, select
  semantics, or emitted route behavior unless required by a focused validation
  defect.
* Moving RVV compare/select semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, descriptors, C strings, test
  names, spec prose, intrinsic strings, or mirror metadata as route authority.

## Evidence Plan

* Validate the Trellis task context.
* Build `tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `cmp-select`, `runtime-scalar-cmp-select`, and
  `computed-mask-select` only if that surface is touched.
* Run generated-bundle dry-runs for explicit and pre-realized compare/select
  forms when provider metadata mirrors or route descriptions change.
* Run direct fail-closed C++ checks for stale or missing compare/select
  provider facts and candidate mirrors.
* Run bounded old-authority scans over touched source/test/spec/task files for
  legacy i32 route authority, descriptor/source-front-door/source-artifact/
  direct-C residue, route-id/artifact-name authority, and mirror-only
  authority.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior,
  runtime ABI order, predicate semantics, select/mask behavior, or runtime
  correctness claim changes. If this round only tightens production
  provider-to-target fact ownership and validation, reuse archived compare/select
  runtime evidence and state that no new runtime claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/variant-pipeline/index.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-widening-conversion-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-compare-select-selected-body-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-runtime-scalar-cmp-select-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-dual-cmp-mask-and-select-artifact-abi-boundary/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/RVV/*cmp-select*.mlir`
* `test/Scripts/*cmp-select*.test`
* `test/Target/TargetArtifactExportTest.cpp`
