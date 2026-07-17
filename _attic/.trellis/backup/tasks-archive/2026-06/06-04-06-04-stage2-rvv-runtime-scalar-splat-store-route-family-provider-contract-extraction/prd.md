# Stage2 RVV runtime-scalar splat-store route-family provider contract extraction

## Goal

Extract a provider-owned runtime-scalar splat-store route validation contract
for the existing `runtime_scalar_splat_store` route family, then rewire target
artifact validation to consume that contract as a route-payload and mirror
client. This round must keep generated runtime-scalar splat-store behavior
unchanged and must not add new runtime-scalar route coverage.

The intended chain is:

```text
selected typed tcrv_rvv runtime-scalar splat-store body
  -> RVV plugin-owned runtime scalar, splat, store, dtype/config,
     ABI binding, policy, and statement-plan facts
  -> provider-owned route validation contract
  -> TCRVEmitCLowerableRoute
  -> target artifact validation consumes the contract for payload and mirrors
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar splat-store route-family provider contract extraction`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `77d0d40e rvv: extract segment2 route validation contract`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain to stay:
  selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/provider route -> common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` places runtime ABI use,
  scalar/vector body facts, dtype/config, policy, intrinsic/backend spelling,
  ABI mapping, route support, and fail-closed diagnostics in the RVV plugin.
  Target validation must not infer those facts from route ids, artifact names,
  ABI strings, metadata mirrors, test names, descriptors, common EmitC, or
  exact intrinsic spellings.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines provider-owned
  route facts and validation contract patterns, plus the rule that target
  artifact validation compares provider payloads and metadata mirrors after
  provider route construction.
* The archived segment2 task completed the closest current template:
  provider interface contract type plus accessor, provider construction from
  existing route facts and rebuilt route description fields, then target
  artifact validation as a consume-only client.
* The archived runtime-scalar Stage1 task completed the generic typed
  `runtime_scalar_splat_store` replacement and records that executable support
  remains the typed SEW32/LMUL m1 instance routed through generic typed body
  and config facts, not old i32-shaped authority.
* The archived runtime-scalar ABI preflight task already closed the provider
  preflight before `TCRVEmitCLowerableRoute` construction. This task should
  not redo that preflight; it should expose a provider-owned route validation
  contract for target artifact validation to consume.

## Requirements

* Keep scope to the existing runtime-scalar splat-store route family. Do not
  add new scalar operations, dtype/LMUL clone batches, new memory families, or
  new runtime-scalar route coverage.
* Add an RVV provider-owned runtime-scalar splat-store
  `RouteValidationContract` in the provider interface, analogous to the
  recent base-memory, segment2, elementwise, compare/select, conversion,
  reduction, MAcc, and widening dot-reduce contracts.
* Build the contract from existing RVV route-family facts, materialization
  facts, route-control facts, residual operand-binding facts, runtime ABI
  order, and runtime-scalar statement-plan facts, plus selected rebuilt route
  description fields needed to validate route payload shape and statement
  names.
* The contract must cover provider support mirrors or an explicit provider
  mirror contract, route payload identity, memory form, operation kind,
  header/type/intrinsic/profile facts, runtime ABI roles/order, scalar runtime
  parameter binding, scalar input and destination output binding, splat vector
  value facts, destination store layout, dtype/config relation, tail/mask
  policy, AVL/VL names, and statement-plan expectations.
* Rewire target artifact runtime-scalar splat-store validation to require and
  consume the provider-owned contract before accepting route id/payload,
  header/type summaries, intrinsic/profile summaries, runtime ABI mappings,
  runtime ABI order, scalar input binding, splat vector facts, destination
  store layout, dtype/config relation, tail/mask policy, AVL/VL names, and
  statement-plan metadata mirrors.
* Keep target artifact validation as a consume-only client. It may compare the
  rebuilt route and candidate metadata to provider contract fields; it must not
  choose runtime-scalar splat/store semantics from target-local route names,
  artifact metadata, C strings, exact intrinsic spellings, descriptors,
  scripts, or test names.
* Preserve generated runtime-scalar splat-store behavior, emitted C/C++,
  runtime ABI order, scalar splat/store statement sequence, destination
  layout, dtype/config relation, tail/mask behavior, and generated-bundle
  evidence shape.

## Acceptance Criteria

* [x] `RVVEmitCRouteProvider.h` exposes a runtime-scalar splat-store route
      validation contract type and accessor.
* [x] Provider implementation returns the contract for the supported existing
      runtime-scalar splat-store route family and returns no contract for
      non-runtime-splat or unsupported operations. Missing, stale, cross-family,
      or cross-route description/payload fields fail closed when the target
      consumer compares them to the contract.
* [x] The provider contract is derived from provider-owned runtime scalar,
      splat, store, dtype/config, ABI, policy, and statement-plan facts rather
      than route ids, artifact names, metadata mirrors, descriptors, common
      EmitC, test names, or exact intrinsic spelling.
* [x] Target artifact runtime-scalar splat-store validation requires the
      provider contract before accepting route payload, headers, type mappings,
      intrinsic/profile fields, ABI mappings, runtime ABI order, scalar input
      binding, output binding, destination layout, dtype/config facts,
      tail/mask policy, AVL/VL names, and statement-plan expectations.
* [x] Focused C++ target artifact tests cover positive contract consumption and
      fail-closed missing, stale, cross-family, cross-route, and mismatched
      contract fields.
* [x] Focused provider/plugin C++ tests cover the provider contract accessor
      when provider headers or provider code change.
* [x] Focused runtime-scalar splat-store lit filters or generated-bundle
      dry-runs still pass for touched fixtures.
* [x] Build and run `tianchenrv-target-artifact-export-test`.
* [x] Build and run `tianchenrv-rvv-extension-plugin-test` if provider headers
      or plugin implementation code change.
* [x] Bounded old-authority scans over touched files find no added positive
      dependency on legacy `i32m1`, descriptor, source-front-door,
      source-artifact, route-id, artifact-name, exact-intrinsic,
      common-EmitC, or mirror-only route authority.
* [x] `git diff --check` passes.
* [x] No `ssh rvv` run is required unless emitted runtime ABI order, scalar
      splat/store behavior, destination layout, dtype/config relation,
      mask/tail behavior, correctness, or performance claims change.

## Validation Evidence

* `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2` passed after self-repairing the provider accessor linkage from an anonymous-namespace implementation to an external wrapper.
* `./build/bin/tianchenrv-target-artifact-export-test` passed.
* `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime.*scalar.*splat|splat.*store|runtime_i32_splat_store|runtime_scalar_splat_store'` passed from `build/test`: 6 selected tests passed, 471 excluded.
* `git diff --check` passed.
* Bounded source-only old-authority scan over touched production files found no added `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, descriptor, source-front-door, source-artifact, direct-C, common EmitC, exact-intrinsic, or `i32m1` authority. The full touched-file scan only reported exact RVV intrinsic strings in focused tests, where they assert provider-derived leaf facts rather than route authority.
* No `ssh rvv` run: this round changed validation-contract ownership and target consume-only checks. It did not change generated runtime ABI order, emitted scalar splat/store sequence, destination layout, dtype/config relation, mask/tail behavior, correctness, or performance behavior.

## Out Of Scope

* No new runtime-scalar route coverage, new scalar operations, new memory or
  segment2 work, elementwise/conversion/compare/reduction/MAcc/widening-dot
  work, dtype/LMUL clone batches, source-front-door routes, common EmitC RVV
  semantics, dashboards, broad smoke matrices, artifact-only evidence, IME,
  Offload, TensorExt, frontend, or Stage 3 work.
* No movement of runtime scalar, splat, store, dtype, ABI, policy, or
  statement semantics into target validation, common export, route ids,
  artifact metadata, descriptor residue, generated C strings, scripts, exact
  intrinsic spelling, or test names.
* No resurrection of old i32m1 authority or new dtype-prefixed helper ops.
* No runtime correctness or performance reruns unless generated ABI or emitted
  runtime-scalar splat-store behavior changes.

## Technical Approach

1. Inspect the current runtime-scalar splat-store fact surfaces in
   `RVVEmitCRouteProvider.h`, `RVVEmitCRoutePlanning.h`,
   `RVVEmitCRoutePlanning.cpp`,
   `RVVEmitCResidualStatementPlanOwners.cpp`,
   `RVVEmitCControlPolicyPlanOwners.cpp`, and
   `RVVEmitCRouteProvider.cpp`.
2. Define a provider-owned
   `RVVRuntimeScalarSplatStoreRouteValidationContract` with only the fields
   needed for target artifact validation and fail-closed diagnostics.
3. Implement a contract accessor in the RVV provider/planning layer from the
   existing runtime-scalar route facts and route description fields.
4. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` so runtime-scalar
   splat-store provider-fact, route payload, header/type, intrinsic/profile,
   ABI, binding, config, policy, and statement-plan validators consume the
   provider-owned contract.
5. Extend focused C++ tests in the existing target artifact and RVV extension
   plugin test suites only where the contract accessor or target consumption
   lacks direct evidence.
6. Run focused build/tests, runtime-scalar fixture filters or dry-runs,
   bounded old-authority scan, and diff check.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Archived task context read before implementation:

* `.trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-segment2-memory-route-family-provider-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage1-generic-typed-rvv-runtime-scalar-splat-store-replacement/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-runtime-scalar-splat-store-provider-abi-preflight/prd.md`

Live files to inspect before source edits:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* focused runtime-scalar splat-store fixtures under `test/Target/RVV` and
  `test/Scripts`

## Definition Of Done

* The bounded runtime-scalar splat-store route validation contract is
  provider-owned and consumed by target artifact validation.
* Focused checks pass and the task records truthful evidence.
* No runtime semantic claim is made without real `ssh rvv` evidence.
* Trellis task state, journal, archive, and one coherent commit complete the
  round if all acceptance criteria are met.
