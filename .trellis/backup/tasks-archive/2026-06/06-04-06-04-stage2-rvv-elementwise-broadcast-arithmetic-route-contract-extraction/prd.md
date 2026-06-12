# Stage2 RVV elementwise/broadcast arithmetic route-family provider contract extraction

## Goal

Extract a provider-owned RVV elementwise arithmetic route validation contract
for the existing ordinary vector/vector, masked vector/vector, and
scalar-broadcast elementwise arithmetic route families, then rewire target
artifact validation to consume that contract as a mirror/check client. This
task must not add new arithmetic coverage or change emitted arithmetic runtime
behavior.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV elementwise/broadcast arithmetic route-family provider contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `ac9c5b31 rvv: extract conversion route validation contract`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` require the current RVV path
  to stay: selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body ->
  RVV plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* `.trellis/spec/lowering-runtime/emitc-route.md` already contains the
  provider-owned route-family fact/validation-contract pattern used by MAcc,
  widening dot-reduce, standalone reduction, compare/select, conversion, and
  memory route families.
* The archived conversion contract task completed the closest implementation
  pattern: expose a validation contract and metadata mirror contract in
  `RVVEmitCRouteProvider.h`, build them in the RVV provider layer, and make
  `RVVTargetArtifactRouteFamilyValidation.cpp` consume them.
* The archived elementwise/broadcast production validation task made target
  validation fail closed on operation-specific elementwise facts, but live
  inspection still shows `RVVEmitCRouteProvider.h` has no elementwise
  arithmetic route validation contract.
* Existing elementwise route-family plans already carry the provider facts that
  need to become contract fields: operation kind, memory form, typed config,
  route-family plan, provider-supported mirror, header/type summaries,
  runtime ABI order/parameters, route operand binding plan/summary, vector or
  scalar RHS source form, mask/tail facts, intrinsic leaves, and statement-plan
  expectations.

## Requirements

* Keep scope to existing production-active elementwise arithmetic routes:
  ordinary `Add`/`Sub`/`Mul`, masked `MaskedAdd`/`MaskedSub`/`MaskedMul`, and
  scalar-broadcast `ScalarBroadcastAdd`/`ScalarBroadcastSub`/
  `ScalarBroadcastMul`.
* Add a provider-owned elementwise arithmetic route validation contract in the
  RVV provider interface, derived from the selected route description and the
  existing canonical elementwise/scalar-broadcast route-family facts.
* The contract must cover provider-supported mirrors, route payload identity,
  header/type facts, runtime ABI order and parameters, route operand binding
  plan/summary, operation kind, vector/vector versus scalar-broadcast source
  form, source/result memory layout, dtype/config facts, SEW/LMUL/tail/mask
  facts, mask/passthrough facts for masked arithmetic, intrinsic leaves, result
  naming, and statement-plan expectations.
* Add a provider-owned metadata mirror contract for elementwise arithmetic
  target candidate validation, including stale cross-family and cross-route
  mirror rejection.
* Rewire elementwise/broadcast target artifact route validation to require and
  consume the provider-owned validation contract before accepting rebuilt route
  payload, header/type mappings, ABI mappings, operand binding, operation
  kind, memory/source form, mask/tail facts, intrinsic leaves, and statement
  plan.
* Keep target artifact validation as a consume-only client. It may compare the
  rebuilt route and candidate metadata to provider contract fields; it must not
  choose elementwise semantics from route ids, artifact names, metadata,
  descriptor residue, generated C strings, scripts, or test names.
* Preserve existing generated elementwise route behavior, runtime ABI order,
  arithmetic computation, mask/tail semantics, destination behavior, and
  dry-run/runtime evidence shape.

## Acceptance Criteria

* [x] `RVVEmitCRouteProvider.h` exposes an elementwise arithmetic
      `RouteValidationContract` accessor and metadata mirror contract accessor
      analogous to the existing conversion, compare/select, standalone
      reduction, MAcc, widening dot, and memory contracts.
* [x] Provider implementation builds the contract from existing
      elementwise/scalar-broadcast provider facts plus selected route
      description fields; unsupported/non-elementwise operations return no
      contract.
* [x] Target elementwise provider-fact validation requires the contract before
      accepting route id/payload, required headers, type mappings, runtime ABI
      mappings, operation kind, source form, memory form, typed config,
      mask/tail policy, intrinsic leaves, route operand binding, and statement
      plan.
* [x] Target candidate mirror validation consumes the provider-owned metadata
      mirror contract and rejects stale non-elementwise, cross-operation,
      vector-RHS versus scalar-broadcast, plain versus masked, mask, binding,
      header/type, target-profile, and provider-supported mirrors.
* [x] Focused C++ target artifact tests cover positive contract access for
      representative ordinary, masked, and scalar-broadcast arithmetic routes
      and fail-closed stale/missing/mismatched contract fields.
* [x] Focused elementwise/broadcast lit or generated-bundle dry-run filters
      still pass for touched existing add/sub/mul, masked add/sub/mul, and
      scalar-broadcast add/sub/mul fixture families.
* [x] Build and run `tianchenrv-target-artifact-export-test`.
* [x] Build and run `tianchenrv-rvv-extension-plugin-test` if provider headers
      or plugin implementation code change.
* [x] Bounded old-authority scans over touched files find no new positive
      dependency on legacy `i32m1`, descriptor, source-front-door,
      source-artifact, route-id, artifact-name, exact intrinsic spelling,
      common-EmitC, or mirror-only route authority.
* [x] `git diff --check` passes.
* [x] No `ssh rvv` run is required unless emitted runtime ABI order,
      arithmetic computation, scalar-broadcast behavior, source/result layout,
      dtype/config relation, mask/tail behavior, correctness, or performance
      claims change.

## Out Of Scope

* New elementwise arithmetic ops, new route coverage, dtype/LMUL clone batches,
  source-front-door routes, standalone runtime-scalar splat-store ownership,
  new conversion/compare/reduction/MAcc/widening-dot work, memory expansion,
  common EmitC RVV semantics, dashboards, broad smoke matrices, or
  artifact-only evidence.
* Moving computation, dtype, config, operation kind, mask/tail, or source-form
  semantics into target validation, common EmitC/export, route ids, artifact
  metadata, descriptor residue, generated C strings, scripts, or test names.
* Resurrecting old `i32m1` authority or adding new dtype-prefixed helper ops.
* Runtime correctness/performance reruns unless generated ABI or emitted
  arithmetic sequence changes.

## Technical Approach

Use the just-completed conversion dtype-policy contract as the implementation
shape:

1. Define `RVVElementwiseArithmeticRouteValidationContract` plus a metadata
   mirror contract set in `RVVEmitCRouteProvider.h`.
2. Build the contracts in the RVV provider/planning layer from the already
   derived elementwise and scalar-broadcast route-family plans, route operand
   binding facts, typed config facts, and selected route description.
3. Replace target-local elementwise route truth in
   `RVVTargetArtifactRouteFamilyValidation.cpp` with consumption of the
   provider contract while preserving local generic comparison helpers.
4. Extend focused C++ tests for provider contract presence and fail-closed
   stale/mismatched route payload and metadata mirrors.
5. Run focused build/tests/dry-run filters and bounded old-authority scans.

If the whole family becomes too large for one round, complete ordinary
`Add`/`Sub`/`Mul` plus directly connected scalar-broadcast
`ScalarBroadcastAdd`/`ScalarBroadcastSub`/`ScalarBroadcastMul`, leave masked
arithmetic as the exact continuation point, and keep the task open.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`

Archived task context read:

* `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-conversion-dtype-policy-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-elementwise-broadcast-arithmetic-production-validation/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage1-typed-rvv-elementwise-intrinsic-derivation/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-elementwise-selected-body-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-scalar-broadcast-add-artifact-abi-boundary/prd.md`

Live files inspected before implementation:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

## Implementation Result

Completed in this round:

* Added `RVVElementwiseArithmeticRouteValidationContract` and
  `RVVElementwiseArithmeticRouteMetadataMirrorContractSet` to the RVV provider
  interface.
* Built provider-owned contracts for existing plain, masked, scalar-broadcast,
  and already-connected strided elementwise arithmetic route descriptions.
  Contract facts include route payload identity, operation kind, typed compute
  op, typed config, SEW/LMUL/policy, source/destination/mask/strided facts,
  runtime ABI order and expected role order, route operand binding facts,
  headers, type mappings, intrinsic leaves, EmitC runtime AVL/VL statement
  names, and statement-count expectations.
* Rewired target artifact elementwise provider-fact validation to require and
  consume the provider contract. Target validation now compares rebuilt route
  payload, headers, type mappings, ABI mappings, source/memory/mask facts,
  intrinsic leaves, binding facts, and statement plan against contract fields.
* Rewired target candidate mirror validation to consume the provider-owned
  metadata mirror contract and reject stale cross-family mirrors.
* Added focused C++ tests for contract presence on plain multiply, masked
  subtract, and scalar-broadcast subtract; non-elementwise rejection; stale
  binding facts; stale metadata mirrors; and stale scalar-broadcast ABI role
  order.
* No `ssh rvv` run was required because this round changed validation contract
  ownership and target consumption only; it did not change emitted C/C++,
  runtime ABI order, arithmetic computation, scalar-broadcast behavior,
  source/result layout, dtype/config relation, mask/tail behavior,
  correctness, or performance claims.

Checks completed:

* `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-(add|sub|mul|masked-add|masked-sub|masked-mul|scalar-broadcast-add|scalar-broadcast-sub|scalar-broadcast-mul)'`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(selected-body|pre-realized|masked-add|pre-realized-masked-add|scalar-broadcast-add|pre-realized-scalar-broadcast-add|pre-realized-scalar-broadcast-sub)-dry-run'`
* `git diff --check`
* Diff-only old-authority scan over touched source/test files; no added
  positive `i32m1`, descriptor, source-front-door, source-artifact, route-id,
  artifact-name, exact-intrinsic, common-EmitC, or mirror-only route authority
  was found.
