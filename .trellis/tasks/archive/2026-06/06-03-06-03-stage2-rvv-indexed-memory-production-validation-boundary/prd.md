# Stage2 RVV indexed memory production validation boundary

## Goal

Close one bounded Stage 2 RVV production provider-to-target validation boundary
for the existing indexed memory movement routes:
`indexed_gather_unit_store`, `indexed_scatter_unit_load`,
`computed_masked_indexed_gather_load_unit_store`, and
`computed_masked_indexed_scatter_store_unit_load`.

The RVV provider must expose indexed-memory facts derived from typed
`tcrv_rvv` body/config/runtime structure. Target artifact validation must
consume those provider-owned facts and reject stale or cross-family facts before
accepting generated object/header/bundle artifacts. Route ids, artifact names,
test fixture names, C strings, descriptors, common EmitC/export code, and
metadata mirrors must not become route authority.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV indexed memory production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `d04ff1f2 rvv: validate compare select route facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires the active RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization/provider facts -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness/performance is claimed.
* `.trellis/spec/lowering-runtime/emitc-route.md` already defines provider
  fact surfaces for base indexed memory and computed-mask indexed memory.
* Archived task
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-base-memory-route-family-production-validation-closeout/`
  completed provider-owned production validation for plain indexed
  gather/scatter only, and explicitly left masked indexed paths outside that
  owner.
* Archived task
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-indexed-memory-validation/`
  completed provider-owned production validation for computed-mask indexed
  gather/scatter only.
* The immediately preceding completed task
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-compare-select-production-validation-boundary/`
  is the closest pattern for this round: add/consume provider-owned route facts
  in production target validation and add fail-closed C++ mutations without
  changing runtime semantics.

## Requirements

* Keep the implementation owner bounded to the production provider-to-target
  validation surface for the four existing indexed memory routes named above.
* Preserve the production chain:
  selected `tcrv.exec` RVV variant -> typed indexed memory body -> RVV
  plugin-local realization -> provider-built route facts ->
  `TCRVEmitCLowerableRoute` -> target artifact validation.
* The validation surface must expose and consume provider-owned facts for:
  * runtime ABI order and parameter roles;
  * data element type, SEW, LMUL, tail policy, and mask policy;
  * index source, index EEW/C type, offset unit, index layout/scale, and
    scatter uniqueness where required;
  * gather-vs-scatter source/destination polarity and memory forms;
  * plain indexed route-family plan, typed compute op, indexed data/destination
    memory forms, headers, C type mapping, target leaf profile, binding plan,
    binding summary, and explicit `provider_supported_mirror`;
  * computed-mask indexed route-family plan, compare predicate, mask producer
    source, mask role/source/memory form, mask type/C type, inactive-lane
    contract, masked passthrough/no-write layout, typed compute op, headers,
    C type mapping, target leaf profile, binding plan, binding summary, and
    explicit `provider_supported_mirror`.
* Target artifact validation must reject stale facts before bundle acceptance:
  * stale compare/select, conversion, reduction, MAcc, segment, scalar-splat,
    strided, unit, descriptor/direct-C/source-front-door, or legacy route facts
    on indexed routes;
  * stale gather facts on scatter and stale scatter facts on gather;
  * stale plain indexed facts on computed-mask indexed routes and stale
    computed-mask facts on plain indexed routes;
  * stale index layout/scale/type/source, memory form, mask, inactive-lane,
    operand binding, header/type, target profile, provider mirror, and
    candidate metadata mirror facts.
* Common EmitC/export remains neutral. It may carry provider-built payloads and
  metadata mirrors unchanged, but must not infer RVV indexed memory semantics.
* Existing explicit and pre-realized generated-bundle support must remain
  intact for all four routes.

## Acceptance Criteria

* [x] Production provider/planning and/or target validation has a focused diff
      making all four indexed memory routes consume provider-owned facts through
      the existing base-memory and computed-mask-indexed fact surfaces.
* [x] Target validation rejects stale plain facts on computed-mask indexed
      routes and stale computed-mask facts on plain indexed routes.
* [x] Target validation rejects gather/scatter cross-contamination for both
      plain and computed-mask indexed routes.
* [x] Target validation checks provider-derived runtime ABI order, runtime ABI
      parameters, typed compute op, memory forms, index facts, route-family
      plan, operand binding plan/summary, header/type mapping, target leaf
      profile, and explicit provider mirror for all four routes.
* [x] Target validation additionally checks computed-mask indexed predicate,
      mask producer/source/form, mask type/C type, inactive-lane contract, and
      masked passthrough/no-write layout for the computed-mask routes.
* [x] Focused C++ target artifact tests prove fail-closed behavior for stale or
      missing provider facts and stale candidate metadata mirrors across the
      four-route boundary, including plain/masked stale-fact separation.
* [x] Existing lit/script dry-runs for explicit and pre-realized indexed gather,
      indexed scatter, computed-mask indexed gather, and computed-mask indexed
      scatter still pass.
* [x] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, mirror-only authority, exact intrinsic spelling authority, or
      legacy `i32m1` route authority is introduced.
* [x] Focused checks, `git diff --check`, bounded old-authority scan, Trellis
      finish/archive, clean worktree, and one coherent commit complete this
      round if the module behavior is complete.

## Technical Approach

1. Inspect live provider fact accessors, route planning propagation, and target
   artifact route-family validation for plain indexed memory and
   computed-mask indexed memory.
2. Identify any remaining target-local indexed-memory truth or missing
   plain-vs-computed-mask stale-fact rejection.
3. Rewire target validation to compare route descriptions and candidate
   metadata mirrors against provider-owned fact surfaces for the selected
   family. Keep target code responsible for validation only, not semantic
   derivation.
4. Add focused C++ mutations in `TargetArtifactExportTest.cpp` that mutate
   provider descriptions or candidate mirrors to prove stale cross-family,
   stale gather/scatter, stale mask/plain, stale index, stale binding,
   stale header/type, stale target profile, and stale provider mirror facts
   fail closed.
5. Run the smallest build/lit/script set that exercises the changed boundary.

## Out Of Scope

* No new indexed route families beyond the four existing plain and
  computed-mask indexed gather/scatter routes.
* No segment2 changes, strided memory changes, unit load/store expansion,
  compare/select changes, conversion changes, reduction changes, MAcc changes,
  high-level frontend lowering, source-front-door routes, global tuning, or
  evidence-only packaging.
* No rewrite of generated runtime semantics, route emission behavior, runtime
  ABI order, index semantics, mask semantics, or passthrough semantics unless
  live inspection exposes a focused validation blocker.
* No RVV semantics moved into common EmitC/export.

## Evidence Plan

* Validate task context.
* Build `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and
  `tcrv-translate` if route validation or lit dry-runs require them.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for:
  * `indexed-gather-unit-store`;
  * `indexed-scatter-unit-load`;
  * `computed-masked-indexed-gather-load`;
  * `computed-masked-indexed-scatter-store`.
* Run generated-bundle dry-runs for explicit and pre-realized plain indexed and
  computed-mask indexed forms.
* Run direct fail-closed C++ checks for stale or missing indexed-memory
  provider facts and stale candidate mirrors.
* Run a bounded old-authority scan over touched files for legacy `i32m1`,
  descriptor/direct-C/source-front-door/source-export, route-id/artifact-name
  authority, and mirror-only authority.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior,
  runtime ABI order, index semantics, mask/passthrough behavior, or runtime
  correctness claims change. If this round only tightens production validation,
  reuse archived runtime evidence from the completed plain indexed and
  computed-mask indexed artifact ABI tasks and state that no new runtime claim
  changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-compare-select-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-base-memory-route-family-production-validation-closeout/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-indexed-memory-validation/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-indexed-gather-unit-store-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-scatter-unit-load-artifact-abi-boundary/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/RVV/*indexed*.mlir`
* `test/Scripts/*indexed*.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Completion Evidence

* Added production target validation that rejects stale plain
  `baseMemoryMovementRouteFamilyPlanID` provider facts on compare-produced
  computed-mask memory routes before artifact export.
* Added candidate mirror validation that rejects
  `tcrv_rvv.base_memory_movement_route_family_plan` on computed-mask memory
  routes, including computed-mask indexed gather/scatter.
* Added C++ regression coverage for stale plain base-memory provider facts and
  stale plain base-memory artifact metadata on
  `computed_masked_indexed_gather_load_unit_store`.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` so the computed-mask
  indexed memory fact-surface contract explicitly rejects stale plain
  base-memory route-family provider facts and candidate mirrors.
* Passed focused build and C++ validation:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16`
  followed by `rtk build/bin/tianchenrv-target-artifact-export-test`.
  The build emitted only pre-existing switch coverage warnings in the test file.
* Passed tool build:
  `rtk cmake --build build --target tcrv-opt tcrv-translate -j 16`.
* Passed lit from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'indexed-gather-unit-store'`
  selected 5 tests and passed 5 tests.
* Passed lit from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'indexed-scatter-unit-load'`
  selected 5 tests and passed 5 tests.
* Passed lit from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-computed-masked-indexed-(gather-load|scatter-store)'`
  selected 4 tests and passed 4 tests.
* Passed lit from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?computed-masked-indexed-gather-load-dry-run'`
  selected 2 tests and passed 2 tests.
* Passed lit from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?computed-masked-indexed-scatter-store-dry-run'`
  selected 2 tests and passed 2 tests.
* Passed generated-bundle dry-run:
  `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --overwrite --run-id stage2-indexed-memory-validation-explicit --op-kind indexed_gather_unit_store --op-kind indexed_scatter_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`.
* Passed generated-bundle dry-run:
  `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --overwrite --run-id stage2-indexed-memory-validation-pre-realized --op-kind indexed_gather_unit_store --op-kind indexed_scatter_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`.
* Passed `rtk git diff --check`.
* Bounded `git diff -U0` old-authority scan over added lines had no positive
  legacy `i32m1`, source-front-door, source-export, descriptor, direct-C,
  route-id authority, or artifact-name authority hits.
* No new `ssh rvv` runtime claim was made. This round only tightened
  production provider/target validation and did not change route emission,
  generated runtime semantics, runtime ABI order, index semantics, mask
  semantics, passthrough semantics, or performance behavior. It reuses archived
  real RVV correctness evidence from the completed plain indexed and
  computed-mask indexed artifact ABI tasks.
