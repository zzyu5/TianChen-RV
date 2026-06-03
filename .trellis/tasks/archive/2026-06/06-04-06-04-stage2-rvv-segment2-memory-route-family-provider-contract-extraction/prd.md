# Stage2 RVV segment2 memory route-family provider contract extraction

## Goal

Extract a provider-owned RVV segment2 memory route validation contract for the
existing production segment2 route families, then rewire target artifact
validation to consume that contract as a route-payload/check client. This round
must keep generated segment2 behavior unchanged and must not add new segment2
route coverage.

The intended chain is:

```text
selected typed tcrv_rvv segment2 memory body
  -> RVV plugin-owned segment2 route facts and validation contract
  -> TCRVEmitCLowerableRoute
  -> target artifact validation consumes the contract for payload and mirrors
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV segment2 memory route-family provider contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `705b961b rvv: extract base-memory route validation contract`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires current RVV execution to stay on the
  selected `tcrv.exec` -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned provider -> common EmitC -> target artifact chain.
* `.trellis/spec/extension-plugins/rvv-plugin.md` says segment2 planning,
  statement plans, route support, ABI, dtype/config, memory form, mask/tail,
  and statement sequence facts are RVV provider-owned and must not be inferred
  by target validation from route ids, artifact names, metadata mirrors,
  descriptors, generated C strings, scripts, test names, or exact intrinsic
  spelling.
* `.trellis/spec/lowering-runtime/emitc-route.md` already defines plain and
  computed-mask segment2 fact surfaces plus the memory metadata mirror contract.
  It also defines the newer route validation contract pattern used by base
  memory, elementwise, conversion, compare/select, standalone reduction, MAcc,
  and widening dot-reduce.
* The archived base-memory task completed the closest template:
  `RVVBaseMemoryMovementRouteValidationContract` in the provider interface,
  provider construction from route facts plus rebuilt route description fields,
  and target validation as a consume-only client.
* Live `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes
  `RVVPlainSegment2MemoryRouteFacts`,
  `RVVComputedMaskSegment2MemoryRouteFacts`, and
  `getRVVSegment2MemoryRouteMetadataMirrorContract(...)`, but it does not
  expose a segment2 route validation contract.
* Live `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` already has canonical
  facts for existing segment2 routes: plain interleave/deinterleave,
  computed-mask load/store/update, runtime ABI order and parameters, header
  and C type summaries, provider support mirror, route-family plans,
  binding summaries, SEW/LMUL/policy, mask facts, segment/field layout,
  update arithmetic facts, intrinsic leaves, and metadata mirror contracts.

## Requirements

* Keep scope to existing production segment2 memory families:
  `segment2_deinterleave_unit_store`, `segment2_interleave_unit_load`,
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load`.
* Add a provider-owned `RVVSegment2MemoryRouteValidationContract` in the RVV
  provider interface, derived from existing plain/computed-mask segment2 route
  facts plus selected rebuilt route description fields that reflect route
  payload shape and statement-plan names.
* The contract must cover provider-supported mirror, route payload identity,
  operation kind, memory form, route-family plan, header/type facts, runtime
  ABI order and parameters, route operand binding plan/summary, element dtype,
  SEW/LMUL/tail/mask policy, runtime control, source/destination layout,
  segment memory layout, lane/field roles and names, unit/segment load/store
  direction, computed-mask facts, update arithmetic facts, AVL/VL names, and
  statement-plan expectations.
* Rewire target artifact segment2 provider-fact validation to require and
  consume the provider-owned contract before accepting route id/payload,
  headers, type mappings, ABI mappings, runtime ABI order, route-family plans,
  segment/field facts, source/destination layout, mask/tail facts, update
  arithmetic facts, AVL/VL names, and statement plan.
* Keep target artifact validation as a consume-only client. It may compare the
  rebuilt route and candidate metadata to provider contract fields; it must not
  choose segment2 semantics from route ids, artifact metadata, descriptors,
  generated C strings, exact intrinsic spellings, or test names.
* Preserve the already provider-owned segment2 metadata mirror contract and
  keep candidate mirror validation as a mirror check after provider route
  reconstruction.
* Preserve existing generated segment2 route behavior, emitted C/C++,
  runtime ABI order, load/store/update behavior, lane/field layout,
  source/destination layout, dtype/config relation, mask/tail behavior, and
  generated-bundle evidence shape.

## Acceptance Criteria

* [x] `RVVEmitCRouteProvider.h` exposes a segment2 memory
      `RouteValidationContract` accessor analogous to the existing
      base-memory movement route validation contract.
* [x] Provider implementation builds the contract from existing
      `RVVPlainSegment2MemoryRouteFacts` or
      `RVVComputedMaskSegment2MemoryRouteFacts` plus selected route
      description fields; unsupported or cross-family operations return no
      contract.
* [x] Target segment2 provider-fact validation requires the contract before
      accepting route id/payload, headers, type mappings, ABI mappings,
      operation kind, memory form, route-family plan, runtime ABI order,
      route operand binding, typed config, source/destination layout,
      segment/field facts, mask/tail facts, update arithmetic facts,
      intrinsic leaves, and statement plan.
* [x] Candidate mirror validation continues to consume
      `getRVVSegment2MemoryRouteMetadataMirrorContract(...)` and remains
      fail-closed for stale/missing/cross-family mirrors.
* [x] Focused C++ target artifact tests cover positive contract access for
      representative plain and computed-mask segment2 routes and fail-closed
      stale or mismatched contract fields.
* [x] Focused segment2 lit or generated-bundle dry-run filters still pass for
      touched segment2 fixture families.
* [x] Build and run `tianchenrv-target-artifact-export-test`.
* [x] Build and run `tianchenrv-rvv-extension-plugin-test` because provider
      headers and plugin implementation code change.
* [x] Bounded old-authority scans over touched files find no new positive
      dependency on legacy `i32m1`, descriptor, source-front-door,
      source-artifact, route-id, artifact-name, exact-intrinsic,
      common-EmitC, or mirror-only route authority.
* [x] `git diff --check` passes.
* [x] No `ssh rvv` run is required unless emitted runtime ABI order,
      segment2 memory behavior, field/lane layout, source/destination layout,
      dtype/config relation, mask/tail behavior, correctness, or performance
      claims change.

## Out Of Scope

* No new segment2 route coverage, new memory forms, segment widths greater
  than 2, base-memory ownership, runtime-scalar splat-store ownership,
  dtype/LMUL clone batches, source-front-door routes, common EmitC RVV
  semantics, dashboards, broad smoke matrices, or artifact-only evidence.
* No new elementwise, conversion, compare/select, reduction, MAcc, widening
  dot, IME, Offload, TensorExt, frontend, or Stage 3 work.
* No movement of segment2 memory semantics into target validation, common
  EmitC/export, route ids, artifact metadata, descriptor residue, generated C
  strings, scripts, exact intrinsic spelling, or test names.
* No resurrection of old `i32m1` authority or new dtype-prefixed helper ops.
* No runtime correctness/performance reruns unless generated ABI or emitted
  segment2 memory sequence changes.

## Technical Approach

1. Define `RVVSegment2MemoryRouteValidationContract` in
   `RVVEmitCRouteProvider.h` with route-family fields and statement-plan
   expectations needed by target validation.
2. Build `getRVVSegment2MemoryRouteValidationContract(...)` in
   `RVVEmitCRoutePlanning.cpp` from `RVVPlainSegment2MemoryRouteFacts` or
   `RVVComputedMaskSegment2MemoryRouteFacts` plus selected route description
   fields such as route id, typed config contract, EmitC loop/VL names, field
   names, mask names, and result names.
3. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` so segment2
   provider-fact, header/type, ABI, layout, runtime ABI, mask/update, and
   statement-plan validators accept the provider contract instead of fetching
   segment2 facts or reconstructing route truth locally.
4. Keep the existing provider-owned segment2 metadata mirror contract path for
   candidate metadata validation.
5. Extend focused C++ tests only where the new contract accessor or target
   consumption lacks direct evidence.
6. Run focused build/tests, lit or generated-bundle filters, old-authority
   scan, and diff check.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Archived task context read:

* `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-base-memory-movement-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-segment2-memory-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-segment2-memory-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-plain-segment2-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-segment2-production-validation-boundary/prd.md`

Live files inspected before implementation:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`

## Definition Of Done

* The bounded segment2 route validation contract is provider-owned and
  consumed by target artifact validation.
* Focused checks pass and the task records truthful evidence.
* No runtime semantic claim is made without real `ssh rvv` evidence.
* Trellis task state, journal, archive, and one coherent commit complete the
  round if all acceptance criteria are met.

## Completion Evidence

Implemented:

* Added `RVVSegment2MemoryRouteValidationContract` and
  `getRVVSegment2MemoryRouteValidationContract(...)` to the RVV provider
  interface.
* Built the contract in `RVVEmitCRoutePlanning.cpp` from existing plain and
  computed-mask segment2 facts plus rebuilt route description fields for
  route id, config, EmitC AVL/VL names, field/mask names, statement counts,
  headers, type mappings, ABI parameters, segment/field layout, mask/tail
  facts, and update arithmetic facts.
* Rewired segment2 target artifact validation so provider-fact, header/type,
  ABI, route payload, runtime ABI, route-family, field/lane, mask/update, and
  statement-plan checks consume the provider-owned contract.
* Preserved the existing provider-owned segment2 metadata mirror contract and
  candidate mirror validation path.
* Populated computed-mask segment2 rebuilt route descriptions with the
  provider-owned mask/tail route-family plan and owner so the contract can
  validate those facts before artifact acceptance.
* Extended `TargetArtifactExportTest.cpp` with positive route validation
  contract coverage for all five existing segment2 routes and updated stale
  cross-family diagnostics to the new contract surface.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the concrete
  segment2 memory route validation contract signature, validation matrix, and
  test requirements.

Checks:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-(interleave-unit-load|deinterleave-unit-store)|computed-masked-segment2-(load-unit-store|store-unit-load|update-unit-load)'`
  from `build/test`, passing 8 focused tests.
* `rtk git diff --check`
* Bounded old-authority scan over touched files found no added-line hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  `__riscv_*_i32m1`, source-front-door, source-artifact, descriptor,
  direct-C/source-export, common-EmitC semantic inference, route-id authority,
  artifact-name authority, or mirror-only authority.

Self-repair:

* Fixed an accidental pre-loop statement-count edit that had matched
  compare/select and widening-dot statement-plan code instead of segment2.
* Tightened computed-mask segment2 route descriptions to carry mask/tail
  provider facts, then updated target tests for the new fail-closed diagnostic
  surface.
* Marked retired direct segment2 target helper functions as `[[maybe_unused]]`
  after the active path moved to the route validation contract.

Runtime evidence:

* `ssh rvv` was not run because this round changes provider validation
  contract ownership, target-consumer checks, and metadata mirror availability
  for existing computed-mask segment2 facts only. It does not change emitted
  runtime ABI order, segment2 load/store/update statement sequence,
  field/lane layout semantics, source/destination layout semantics,
  dtype/config relation, mask/tail behavior, correctness claims, or
  performance claims.
