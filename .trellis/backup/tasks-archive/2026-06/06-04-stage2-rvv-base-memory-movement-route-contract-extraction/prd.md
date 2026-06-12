# Stage2 RVV base-memory movement route-family provider contract extraction

## Goal

Extract a provider-owned RVV base-memory movement route validation contract for
the existing production base-memory route families, then rewire target artifact
validation to consume that contract as a route-payload/check client. This round
must keep generated memory movement behavior unchanged and must not add new
memory route coverage.

The intended chain is:

```text
selected typed tcrv_rvv base-memory body
  -> RVV plugin-owned base-memory route facts and validation contract
  -> TCRVEmitCLowerableRoute
  -> target artifact validation consumes the contract for payload and mirrors
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV base-memory movement route-family provider contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `e5257565 rvv: extract elementwise route validation contract`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires current RVV execution to stay on the
  selected `tcrv.exec` -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned provider -> common EmitC -> target artifact chain.
* `.trellis/spec/extension-plugins/rvv-plugin.md` says RVV dtype/config,
  memory form, policy, ABI use, and intrinsic mapping must be derived by the
  RVV plugin/provider from typed body/config/runtime facts, not from route ids,
  artifact names, metadata, descriptors, generated C strings, or test names.
* `.trellis/spec/lowering-runtime/emitc-route.md` documents the
  provider-owned route-family fact and validation-contract pattern already used
  by MAcc, widening dot-reduce, standalone reduction, compare/select,
  conversion, elementwise, and memory mirror contracts.
* The archived provider-owned memory route task already moved the normalized
  memory metadata mirror contract out of target validation, but live code still
  shows base-memory route payload/provider-fact validation directly consuming
  `RVVBaseMemoryMovementRouteFacts` and rebuilding statement expectations in
  `RVVTargetArtifactRouteFamilyValidation.cpp`.
* Live `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes
  `RVVBaseMemoryMovementRouteFacts` and memory metadata mirror contracts, but
  it does not expose a base-memory movement route validation contract.
* Live `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
  already has the canonical facts needed for the contract: memory form,
  runtime ABI order/parameters, header/type summaries, provider-supported
  mirror, route-family plan, route operand binding plan/summary,
  SEW/LMUL/policy, vector and index/mask types, intrinsic leaves, source and
  destination forms, stride/index facts, mask/tail facts, and result/mask
  names.

## Requirements

* Keep scope to the existing production base-memory movement route families:
  `StridedLoadUnitStore`, `UnitLoadStridedStore`,
  `IndexedGatherUnitStore`, `IndexedScatterUnitLoad`,
  `MaskedUnitLoadStore`, and `MaskedUnitStore`.
* Add a provider-owned `RVVBaseMemoryMovementRouteValidationContract` in the
  RVV provider interface, derived from `RVVBaseMemoryMovementRouteFacts` plus
  selected route description fields that reflect runtime AVL/VL statement
  names and route payload shape.
* The contract must cover provider-supported mirror, route payload identity,
  memory operation kind/form, route-family plan, header/type facts, runtime ABI
  order and parameters, route operand binding plan/summary, element dtype,
  SEW/LMUL/tail/mask policy, VL/control names, source/destination layout,
  stride/index facts, mask/tail facts, intrinsic leaves, result/mask names,
  and statement-plan expectations.
* Rewire target artifact base-memory provider-fact validation to require and
  consume the provider-owned contract before accepting route id/payload,
  headers, type mappings, ABI mappings, runtime ABI order, memory form,
  source/destination layout, stride/index facts, mask/passthrough facts,
  intrinsic leaves, AVL/VL names, and statement plan.
* Keep target artifact validation as a consume-only client. It may compare the
  rebuilt route and candidate metadata to provider contract fields; it must not
  choose base-memory semantics from route ids, artifact metadata, descriptors,
  generated C strings, exact intrinsic spellings, or test names.
* Preserve the already provider-owned memory metadata mirror contract and keep
  candidate mirror validation as a mirror check after provider route
  reconstruction.
* Preserve existing generated base-memory route behavior, emitted C/C++,
  runtime ABI order, load/store behavior, mask/tail behavior, passthrough and
  destination preservation behavior, and generated-bundle evidence shape.

## Acceptance Criteria

* [x] `RVVEmitCRouteProvider.h` exposes a base-memory movement
      `RouteValidationContract` accessor analogous to existing elementwise,
      conversion, compare/select, standalone reduction, MAcc, and widening dot
      validation contracts.
* [x] Provider implementation builds the contract from existing
      `RVVBaseMemoryMovementRouteFacts` and selected route description fields;
      unsupported or cross-family operations return no contract.
* [x] Target base-memory provider-fact validation requires the contract before
      accepting route id/payload, headers, type mappings, ABI mappings,
      operation kind, memory form, route-family plan, runtime ABI order,
      route operand binding, typed config, source/destination layout,
      stride/index facts, mask/tail facts, intrinsic leaves, and statement
      plan.
* [x] Candidate mirror validation continues to consume
      `getRVVBaseMemoryRouteMetadataMirrorContract(...)` and remains
      fail-closed for stale/missing/cross-family mirrors.
* [x] Focused C++ target artifact tests cover positive contract access for
      representative strided, indexed, and masked base-memory routes and
      fail-closed stale or mismatched contract fields.
* [x] Focused base-memory lit or generated-bundle dry-run filters still pass
      for touched strided, indexed, and masked fixture families.
* [x] Build and run `tianchenrv-target-artifact-export-test`.
* [x] Build and run `tianchenrv-rvv-extension-plugin-test` because provider
      headers and plugin implementation code change.
* [x] Bounded old-authority scans over touched files find no new positive
      dependency on legacy `i32m1`, descriptor, source-front-door,
      source-artifact, route-id, artifact-name, exact-intrinsic,
      common-EmitC, or mirror-only route authority.
* [x] `git diff --check` passes.
* [x] No `ssh rvv` run is required unless emitted runtime ABI order,
      memory operation behavior, source/destination layout, dtype/config
      relation, mask/tail behavior, correctness, or performance claims change.

## Out Of Scope

* No new memory route coverage, new memory forms, segment2 ownership,
  computed-mask memory ownership, runtime-scalar splat-store ownership,
  dtype/LMUL clone batches, source-front-door routes, common EmitC RVV
  semantics, dashboards, broad smoke matrices, or artifact-only evidence.
* No new elementwise, conversion, compare/select, reduction, MAcc, widening
  dot, IME, Offload, TensorExt, frontend, or Stage 3 work.
* No movement of memory semantics into target validation, common EmitC/export,
  route ids, artifact metadata, descriptor residue, generated C strings,
  scripts, exact intrinsic spelling, or test names.
* No resurrection of old `i32m1` authority or new dtype-prefixed helper ops.
* No runtime correctness/performance reruns unless generated ABI or emitted
  memory movement sequence changes.

## Technical Approach

1. Define `RVVBaseMemoryMovementRouteValidationContract` in
   `RVVEmitCRouteProvider.h` with route-family fields and statement-plan
   expectations needed by target validation.
2. Build `getRVVBaseMemoryMovementRouteValidationContract(...)` in
   `RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` from
   `RVVBaseMemoryMovementRouteFacts` plus selected route description fields
   such as route id, typed config contract, EmitC loop/VL names, and result
   names.
3. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` so base-memory
   provider-fact, header/type, ABI, layout, runtime ABI, and statement-plan
   validators accept the provider contract instead of fetching base-memory
   facts or reconstructing route truth locally.
4. Keep the existing provider-owned base-memory metadata mirror contract path
   for candidate metadata validation.
5. Extend focused C++ tests only where the new contract accessor or target
   consumption lacks direct evidence.
6. Run focused build/tests, lit filters, old-authority scan, and diff check.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Archived task context read:

* `.trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-elementwise-broadcast-arithmetic-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-provider-owned-memory-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-base-memory-route-family-production-validation-closeout/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-base-memory-movement-selected-body-realization-boundary/prd.md`

Live files inspected before implementation:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

## Definition Of Done

* The bounded base-memory route validation contract is provider-owned and
  consumed by target artifact validation.
* Focused checks pass and the task records truthful evidence.
* No runtime semantic claim is made without real `ssh rvv` evidence.
* Trellis task state, journal, archive, and one coherent commit complete the
  round if all acceptance criteria are met.

## Completion Evidence

Implemented:

* Added `RVVBaseMemoryMovementRouteValidationContract` and
  `getRVVBaseMemoryMovementRouteValidationContract(...)` to the RVV provider
  interface.
* Populated the contract in
  `RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` for existing base-memory
  movement families, including strided load/store, indexed gather/scatter, and
  masked unit load/store variants.
* Rewired `RVVTargetArtifactRouteFamilyValidation.cpp` so base-memory
  payload, header/type, ABI, runtime ABI, memory-layout, mask/tail, AVL/VL, and
  statement-plan validation consumes the provider contract.
* Extended `TargetArtifactExportTest.cpp` with positive contract accessor
  coverage and stale-field rejection evidence.

Checks:

* `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter '(strided-load-unit-store|unit-load-strided-store|indexed-gather-unit-store|indexed-scatter-unit-load|masked-unit-(load-store|store))'`
  from `build/test`, passing 32 focused tests.
* `git diff --check`
* Bounded old-authority scan over touched files found no added
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  `__riscv_*_i32m1`, source-front-door, descriptor, or common-EmitC authority
  hits.

Self-repair:

* Fixed the masked base-memory contract to derive mask type and mask C type
  from provider-owned masked route facts rather than mutable target
  descriptions.
* Updated the stale masked unit binding assertion to match the new
  provider-contract diagnostic surface.
* Removed two target-local helper functions that became unused after contract
  rewiring.

Runtime evidence:

* `ssh rvv` was not run because this round changes provider validation
  contract ownership and target-consumer checks only. It does not change
  emitted runtime ABI order, generated memory operation behavior,
  source/destination layout semantics, dtype/config relation, mask/tail
  behavior, correctness claims, or performance claims.
