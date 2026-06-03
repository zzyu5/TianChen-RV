# Stage2 RVV vector-reduction route-family provider contract extraction

## Goal

Extract a provider-owned vector-reduction route validation contract for the
existing vector RHS-load reduction route family, then rewire target artifact
validation to consume that contract as a route-payload and mirror client. This
round must keep generated vector-reduction behavior unchanged and must not add
new reduction coverage.

The intended ownership chain is:

```text
selected typed tcrv_rvv.reduce vector-reduction body
  -> RVV plugin-owned vector-reduction route/control/statement/binding facts
  -> provider-owned vector-reduction route validation contract
  -> TCRVEmitCLowerableRoute
  -> target artifact validation consumes the contract for payload and mirrors
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV vector-reduction route-family provider contract extraction`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `77c22e7b rvv: extract runtime splat-store route validation contract`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain to stay:
  selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/provider route -> common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` places runtime ABI use,
  vector body facts, dtype/config, policy, intrinsic/backend spelling, ABI
  mapping, route support, and fail-closed diagnostics in the RVV plugin.
  Target validation must not infer those facts from route ids, artifact names,
  ABI strings, metadata mirrors, test names, descriptors, common EmitC, or
  exact intrinsic spellings.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines provider-owned route
  validation contract patterns for MAcc, widening dot-reduce, standalone
  reduction, segment2 memory, runtime-scalar splat-store, and related RVV
  route families. Target artifact validation should compare rebuilt routes
  and candidate mirrors against provider contract fields after provider route
  construction.
* The archived standalone reduction contract task is the closest reduction
  template: provider contract from canonical reduction facts plus rebuilt
  route description, target validation consumes the contract, metadata mirrors
  remain separate mirror-only checks.
* The archived runtime-scalar splat-store task is the closest most recent
  template: provider interface contract type plus accessor, provider
  implementation from existing route/control/statement/binding facts, and
  target validation rewired as consume-only client.
* Live search confirms `validateRVVVectorReductionRoutePayloadFacts` still
  exists in `RVVTargetArtifactRouteFamilyValidation.cpp`, so the current
  vector-reduction target path still has target-local validation logic to
  migrate behind a provider-owned contract.

## Requirements

* Keep scope to the existing vector RHS-load reduction route family. Do not
  add new reduction coverage, accumulator layouts, dtype/LMUL clone batches,
  source-front-door routes, runtime-scalar splat-store ownership, standalone
  reduction expansion, memory/segment2/elementwise/conversion/compare/MAcc/
  widening-dot work, common EmitC RVV semantics, dashboards, or broad smoke
  matrices.
* Add an RVV provider-owned vector-reduction
  `RouteValidationContract` in the provider interface, analogous to the recent
  route-family contracts.
* Build the contract from existing RVV reduction route facts, route-control
  facts, statement-plan facts, operand-binding facts, runtime ABI order, and
  rebuilt route description fields needed to validate route payload shape and
  statement names.
* The contract must cover provider-supported mirrors or an explicit mirror
  contract, route payload identity, typed compute op, vector RHS-load memory
  form, runtime ABI roles/order, operand binding summary, lhs/rhs/out/n roles,
  rhs seed or accumulator facts, scalar output slot, runtime `n` binding,
  accumulator/result layout, reduction store VL, header/type/intrinsic/profile
  facts, dtype/config relation, AVL/VL loop facts, and statement-plan
  expectations.
* Rewire target artifact vector-reduction validation to require and consume
  the provider-owned contract before accepting rebuilt route payload,
  header/type summaries, intrinsic/profile fields, runtime ABI mappings,
  runtime ABI order, lhs input, rhs vector load or seed/accumulator binding,
  scalar output slot, runtime `n` binding, accumulator/result layout,
  reduction store VL, dtype/config relation, AVL/VL names, and statement-plan
  metadata mirrors.
* Preserve fail-closed diagnostics for missing, stale, cross-family,
  cross-route, or mismatched contract fields. This includes stale standalone
  reduction, MAcc, widening-dot, memory, segment2, elementwise, conversion,
  compare/select, runtime-scalar splat-store, descriptor, route-id,
  artifact-name, or mirror-only authority residue.
* Keep target artifact validation as a consume-only client. It may compare the
  rebuilt route and candidate metadata to provider contract fields; it must not
  choose vector-reduction semantics from target-local route names, artifact
  metadata, C strings, exact intrinsic spellings, descriptors, scripts, or
  test names.
* Preserve generated vector-reduction C/C++, runtime ABI order, loop/control
  structure, accumulator/result layout, reduction store behavior, dtype/config
  relation, mask/tail behavior, correctness, and performance behavior.

## Acceptance Criteria

* [ ] `RVVEmitCRouteProvider.h` exposes a vector-reduction route validation
      contract type and accessor for the existing vector RHS-load reduction
      family.
* [ ] Provider implementation returns the contract for the supported existing
      vector-reduction route family and returns no contract for unrelated
      operations or unsupported descriptions.
* [ ] The provider contract is derived from provider-owned reduction route,
      control, statement, binding, dtype/config, ABI, policy, and rebuilt-route
      facts rather than route ids, artifact names, metadata mirrors,
      descriptors, common EmitC, test names, or exact intrinsic spelling.
* [ ] Target artifact vector-reduction validation requires the provider
      contract before accepting route payload, headers, type mappings,
      intrinsic/profile fields, ABI mappings, runtime ABI order,
      lhs/rhs/out/n roles, accumulator/result layout, reduction store VL,
      dtype/config facts, AVL/VL names, and statement-plan expectations.
* [ ] Candidate metadata mirror validation for the vector-reduction family
      consumes provider contract fields or an explicit provider mirror contract
      and still rejects stale/missing mirrors separately from rebuilt route
      payload validation.
* [ ] Focused C++ target artifact tests cover positive contract consumption
      and fail-closed missing, stale, cross-family, cross-route, and mismatched
      contract fields.
* [ ] Focused provider/plugin C++ tests cover the provider contract accessor
      when provider headers or provider implementation code change.
* [ ] Focused vector-reduction lit filters or generated-bundle dry-runs still
      pass for touched fixtures.
* [ ] Build and run `tianchenrv-target-artifact-export-test`.
* [ ] Build and run `tianchenrv-rvv-extension-plugin-test` because provider
      headers/plugin code are expected to change.
* [ ] Bounded old-authority scans over touched files find no added positive
      dependency on legacy `i32m1`, descriptor, source-front-door,
      source-artifact, route-id, artifact-name, exact-intrinsic,
      common-EmitC, or mirror-only route authority.
* [ ] `git diff --check` passes.
* [ ] No `ssh rvv` run is required unless emitted runtime ABI order, emitted
      vector-reduction code, reduction loop/statement behavior,
      accumulator/result layout, dtype/config relation, mask/tail behavior,
      correctness, or performance claims change.

## Technical Approach

1. Inspect existing vector-reduction fact surfaces and recent route-contract
   patterns in the provider headers, provider implementation, statement-plan
   owners, and target artifact route-family validation.
2. Define a provider-owned vector-reduction route validation contract with
   only the fields needed for target artifact payload and mirror validation.
3. Implement a contract accessor in the RVV provider/planning layer from
   existing vector-reduction route facts and rebuilt route description fields.
4. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` so vector-reduction
   provider-fact, route payload, header/type, intrinsic/profile, ABI, binding,
   config, policy, accumulator/result, AVL/VL, statement-plan, and candidate
   mirror validators consume the provider-owned contract.
5. Extend focused C++ tests in the existing target artifact and RVV extension
   plugin test suites only where direct contract evidence is missing.
6. Run focused build/tests, vector-reduction fixture filters or dry-runs,
   bounded old-authority scan, and diff check.

## Evidence Plan

* `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused lit or generated-bundle dry-run filters for touched vector-reduction
  fixtures under `test/Target/RVV` and `test/Scripts`.
* Bounded old-authority scan over touched production/test files and changed
  lines.
* `git diff --check`
* No `ssh rvv` unless the final diff changes generated runtime ABI, emitted
  vector-reduction behavior, accumulator/result layout, dtype/config relation,
  mask/tail behavior, correctness, or performance claims.

## Out Of Scope

* No new reduction coverage, new accumulator layouts, dtype/LMUL clone
  batches, source-front-door routes, runtime-scalar splat-store ownership,
  standalone-reduction expansion beyond shared helper reuse,
  memory/segment2/elementwise/conversion/compare/MAcc/widening-dot work,
  common EmitC RVV semantics, dashboards, broad smoke matrices, artifact-only
  evidence, IME, Offload, TensorExt, frontend, or Stage 3 work.
* No movement of vector-reduction semantics into target validation, common
  export, route ids, artifact metadata, descriptor residue, generated C
  strings, scripts, exact intrinsic spelling, or test names.
* No resurrection of old i32m1 authority or new dtype-prefixed helper ops.
* No runtime correctness or performance reruns unless generated ABI or emitted
  vector-reduction behavior changes.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/validation/index.md`

Archived task context read before implementation:

* `.trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-runtime-scalar-splat-store-route-family-provider-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-standalone-reduction-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-standalone-reduction-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary/prd.md`

Live files to inspect before source edits:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* focused vector-reduction fixtures under `test/Target/RVV` and
  `test/Scripts`

## Definition Of Done

* The bounded vector-reduction route validation contract is provider-owned and
  consumed by target artifact validation.
* Focused checks pass and the task records truthful evidence.
* No runtime semantic claim is made without real `ssh rvv` evidence.
