# Stage2 RVV standalone reduction production validation boundary

## Goal

Close the production provider-to-target validation boundary for existing RVV
standalone reduction routes: `standalone_reduce_add`,
`standalone_reduce_min`, `standalone_reduce_max`, and the existing
computed-mask plus runtime-scalar computed-mask standalone reduction variants
that share the same route-family contract.

The provider must expose operation-specific standalone reduction facts derived
from the typed `tcrv_rvv` body/config/runtime surface. Target artifact
validation must consume that provider-owned surface instead of rebuilding
runtime ABI, binding, header/type, profile, inactive-lane, scalar-result, or
mirror facts from route ids, artifact names, candidate metadata, C strings, or
test fixture names.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV standalone reduction production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `f70b5a7b rvv: validate widening dot-reduce route facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* Specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization/provider facts -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> RVV target
  artifact validation.
* Archived standalone reduction tasks already proved selected-body
  realization and executable generated-bundle scalar-result ABI behavior for
  plain `standalone_reduce_add`, including real `ssh rvv` counts
  `0,1,16,17,257`.
* Archived runtime-scalar standalone reduction canonicalization already added
  `RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts` for runtime
  scalar computed-mask add/min/max and rewired provider/target consumers for
  that subfamily.
* Live code inspection shows plain and vector computed-mask standalone
  reduction facts are still rebuilt in provider planning and target artifact
  validation through local constants/helpers, while runtime-scalar
  computed-mask variants already consume a provider-owned facts accessor.

## Requirements

* Keep scope to existing standalone reduction add/min/max route operations:
  plain unit-stride, computed-mask unit-stride, and runtime-scalar
  computed-mask unit-stride variants.
* Preserve the production chain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized standalone reduction
  body -> RVV plugin-local realization -> provider-built
  `TCRVEmitCLowerableRoute` -> target artifact validation.
* Production provider facts for each included route must expose and target
  validation must consume:
  * operation kind and typed compute op;
  * source element/type facts and scalar result type facts;
  * SEW, LMUL, tail policy, mask policy, runtime AVL/VL control, and
    setvl/with-vl participation;
  * runtime ABI order and exported parameter roles;
  * scalar seed/neutral literal by operation and SEW;
  * accumulation carry contract, scalar-result runtime boundary, accumulator
    layout, result layout, and reduction store VL;
  * source and destination memory forms;
  * mask producer/source/form, predicate kind, mask type/C type,
    inactive-lane contract, and masked passthrough layout when present;
  * route operand binding plan and exact summary with `abi` and `hdr`
    participation for every exported runtime ABI parameter;
  * required header declarations and C type mapping;
  * target leaf profile and explicit `provider_supported_mirror`.
* Provider-side standalone reduction route-family plan validation must fail
  closed when canonical provider facts are missing, stale, or inconsistent
  with the selected typed body/config/runtime facts.
* Target artifact validation must reject stale dot-reduce or MAcc facts on
  standalone reductions, stale plain facts on masked reductions, stale masked
  facts on plain reductions, stale add facts on min/max or min/max facts on
  add, missing scalar-result boundary facts, stale neutral literal or
  inactive-lane facts, stale operand binding, stale header/type mapping, stale
  target profile, stale provider mirror, and accidental contraction/segment/
  indexed/source-front-door fallback.
* Common EmitC/export must remain neutral. It may carry provider-built payloads
  and metadata mirrors unchanged, but must not infer RVV standalone reduction
  semantics.
* Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

* [x] Production RVV provider exposes one standalone reduction route facts
      surface covering plain, computed-mask, and runtime-scalar computed-mask
      standalone reduction add/min/max variants.
* [x] Plain and vector computed-mask standalone reduction provider planning
      and validation consume the provider facts instead of local duplicate
      ABI/binding/header/type/profile constants.
* [x] Target validation consumes the same provider facts for standalone
      reduction ABI order, runtime ABI parameters, route payload fields,
      target profile, header/type facts, binding summary, scalar-result
      boundary, inactive-lane contract, and candidate mirrors.
* [x] Target validation rejects stale cross-family facts, cross-mask facts,
      cross-operation add/min/max facts, stale scalar-result/inactive-lane
      facts, stale binding/header/type/profile/provider mirrors, and stale
      candidate metadata mirrors.
* [x] Focused C++ target artifact tests prove the new fail-closed boundary for
      provider facts and candidate mirrors.
* [x] Existing lit/script dry-runs for explicit and pre-realized standalone
      reduction routes still pass for touched variants.
* [x] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy i32 route
      authority is introduced.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [x] Trellis finish/archive and one coherent commit complete this round if
      the module behavior is complete.

## Technical Approach

1. Generalize the existing runtime-scalar standalone reduction canonical facts
   surface into a provider-owned standalone reduction facts accessor for plain,
   computed-mask, and runtime-scalar computed-mask add/min/max.
2. Populate canonical facts from the existing typed body/config/runtime
   constants already used by standalone reduction route-family planning,
   including operation-specific plan IDs, binding summaries, inactive-lane
   policy, scalar-result boundary, header/type mapping, target leaf profile,
   and provider mirror labels.
3. Rewire provider-side standalone reduction plan derivation and validation to
   use those facts for all included subfamilies.
4. Rewire RVV target artifact validation and candidate mirror checks to use
   the provider-owned facts instead of target-local standalone reduction
   constants/helpers.
5. Add focused C++ mutations in `TargetArtifactExportTest.cpp` for stale or
   missing standalone reduction provider facts and candidate mirrors.
6. Run the smallest focused build, C++ target test, standalone-reduce lit/script
   filters, bounded old-authority scan, and diff check required by the changed
   validation boundary.

## Out Of Scope

* New reduction families beyond existing standalone add/min/max.
* Dot-product reductions, widening dot-reduce changes, MAcc changes,
  segment/indexed routes, dtype/LMUL clone batches, high-level frontend
  lowering, source-front-door routes, global performance tuning, dashboards, or
  evidence-only packaging.
* Redoing widening dot-reduce validation as this round's owner.
* Changing generated bundle runtime ABI semantics or scalar-result semantics
  unless required by a focused validation defect.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, descriptors, C strings, test
  names, spec prose, or mirror metadata as route authority.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `standalone-reduce`.
* Run generated-bundle dry-runs for explicit and pre-realized standalone
  reduction forms touched by the validation surface.
* Run direct fail-closed C++ checks for stale or missing standalone
  reduction/provider facts and candidate mirrors.
* Run bounded old-authority scans over touched source/test files for legacy i32
  route authority, descriptor/source-front-door/source-artifact/direct-C
  residue, route-id/artifact-name authority, and mirror-only authority.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior,
  runtime ABI order, or scalar-result semantics change. If this round only
  tightens production provider-to-target validation, reuse archived standalone
  reduction runtime evidence and state that no new runtime claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-widening-dot-reduce-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-standalone-reduction-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-standalone-reduction-route-fact-canonicalization/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/RVV/*standalone-reduce*.mlir`
* `test/Scripts/*standalone-reduce*.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Implementation Outcome

* Added `RVVStandaloneReductionRouteFacts` as the provider-owned canonical
  fact surface for plain, computed-mask, and runtime-scalar computed-mask
  standalone reduction add/min/max routes.
* Kept `RVVRuntimeScalarComputedMaskStandaloneReductionRouteFacts` as a
  compatibility alias and rewired the old runtime-scalar accessor to the new
  standalone facts surface.
* Populated canonical facts with operation, typed compute op, memory form,
  runtime ABI order and SEW-derived ABI parameters, target leaf profile,
  provider mirror, required headers, C type mapping, route operand binding
  plan/summary, scalar-result runtime boundary, accumulator/result layout,
  reduction store VL, mask facts, inactive-lane policy, neutral literals, and
  accumulation carry facts where applicable.
* Rewired standalone reduction route operand binding, route-family plan
  derivation, provider validation, initial route description ABI selection, and
  route-description mirror verification to consume the provider facts.
* Rewired RVV target artifact validation so plain, computed-mask, and
  runtime-scalar standalone reduction payload/runtime-ABI checks consume the
  same provider facts. Removed target-local standalone reduction ABI/binding
  helper constants.
* Strengthened `TargetArtifactExportTest.cpp` to assert canonical facts for
  all nine standalone reduction variants and preserve runtime-scalar
  compatibility accessors.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the durable
  standalone reduction route-family fact contract.

## Evidence Results

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  passed.
* `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter standalone-reduce`
  from `build/test` passed: 42 selected tests passed, 435 excluded.
* `rtk git diff --check` passed.
* Diff-only old-authority scan over touched source/test files found no newly
  introduced `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor,
  direct-C/source-export, route-id/artifact-name authority, or exact
  `__riscv_*_i32m1` authority.

No `ssh rvv` path was rerun in this round because generated bundle behavior,
runtime ABI order, route emission, and scalar-result runtime semantics did not
change. The diff tightens production provider-to-target fact ownership and
validation. Existing archived standalone reduction runtime evidence remains
the runtime correctness basis for this validation-only round.
