# Stage2 RVV standalone reduction route-family provider contract extraction

## Goal

Extract a provider-owned standalone reduction route validation contract from
the existing RVV standalone reduction route facts and rebuilt route
descriptions, then rewire target artifact validation to consume that contract
for the existing standalone reduction route families.

The intended ownership chain is:

```text
selected typed tcrv_rvv standalone reduction body
  -> RVV plugin-owned standalone reduction route facts
  -> provider-owned standalone reduction validation contract
  -> target artifact validation consumes the contract plus rebuilt route
```

Target artifact validation remains a consume-only client. It must not choose
standalone reduction semantics from route names, artifact metadata, C strings,
test names, descriptors, exact intrinsic spellings, candidate mirrors, or
target-local tables.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV standalone reduction route-family provider contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `d412b97f rvv: extract widening dot route validation contract`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact -> `ssh rvv` evidence when
  runtime/correctness/performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires standalone
  reduction source/work and scalar-result channels to remain distinct provider
  facts, and requires inactive-neutral literals to come from RVV provider
  helpers/facts rather than target-local reduction-kind tables.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the RVV
  route-family fact surface contract. Standalone reduction facts must include
  plain, computed-mask, and runtime-scalar computed-mask standalone
  add/min/max routes, with provider-owned ABI, binding, header/type,
  scalar-result, inactive-lane, mask, and mirror facts.
* The archived standalone reduction production validation task already added
  `RVVStandaloneReductionRouteFacts` as the provider-owned canonical fact
  surface for existing plain, computed-mask, and runtime-scalar computed-mask
  standalone reduction add/min/max routes.
* The immediately previous widening dot-reduce task added a provider-owned
  `RVVWideningDotReduceRouteValidationContract` and rewired target validation
  to consume it for route payload, headers, type mappings, ABI mappings,
  layout, mask/tail, and statement-plan expectations.
* Live code inspection shows standalone reduction still has target-local
  provider-fact and candidate-mirror validation functions that directly read
  `RVVStandaloneReductionRouteFacts` and description fields. The missing
  boundary is a provider-owned validation contract analogous to MAcc and
  widening dot-reduce.

## Requirements

* Add a provider-owned standalone reduction route validation contract API in
  the RVV provider layer for existing active operations:
  `standalone_reduce_add`, `standalone_reduce_min`,
  `standalone_reduce_max`, `computed_mask_standalone_reduce_add`,
  `computed_mask_standalone_reduce_min`,
  `computed_mask_standalone_reduce_max`,
  `runtime_scalar_cmp_masked_standalone_reduce_add`,
  `runtime_scalar_cmp_masked_standalone_reduce_min`, and
  `runtime_scalar_cmp_masked_standalone_reduce_max`.
* Build the contract from existing `RVVStandaloneReductionRouteFacts` plus the
  rebuilt `RVVSelectedBodyEmitCRouteDescription`. Do not add new route
  coverage, dtype/LMUL clone batches, new standalone reduction variants, or
  artifact-only evidence.
* Include common route expectations:
  route id, typed compute op, memory form, selected SEW/LMUL, tail/mask
  policy, runtime-control plan, runtime ABI order and parameters, target leaf
  profile, `provider_supported_mirror`, required headers, C type summary,
  route operand binding plan/summary, route payload header/type requirements,
  and dynamic statement payload names.
* Include standalone reduction family expectations:
  source vector type/C type, scalar C type, scalar-result vector type/C type,
  source and scalar-result channel split, scalar seed/result layouts,
  scalar-result runtime boundary, reduction store VL, source splat, scalar seed
  splat, reduction intrinsic, scalar-result store intrinsic, and statement-plan
  shape.
* Include computed-mask standalone reduction expectations:
  compare predicate, mask role/source/form, mask type/C type, compare
  intrinsic, inactive neutral source splat, inactive-lane requirement,
  operation-specific inactive neutral literal by SEW, inactive merge leaf,
  accumulation route-family plan, accumulation compute suffix, mask producer
  source, accumulator/result/scalar-carry contracts, and optional
  runtime-scalar RHS splat.
* Rewire target artifact standalone reduction provider-fact validation to
  consume the provider validation contract for rebuilt route payload,
  header/type mappings, runtime ABI mappings, source/scalar-result layout,
  reduction kind, mask/tail, passthrough/initial value policy, inactive neutral
  policy, and statement-plan expectations.
* Add a provider-owned standalone reduction metadata mirror contract set and
  rewire target candidate mirror validation to consume it. Candidate metadata
  remains a mirror after provider route construction, not route authority.
* Preserve fail-closed behavior for missing, stale, cross-family, cross-route,
  or mismatched contract fields, including stale dot-reduce/MAcc facts, stale
  plain facts on masked routes, stale masked/runtime-scalar facts on plain
  routes, cross-operation add/min/max neutral literals or bindings, stale
  scalar-result boundary facts, stale route operand binding, stale header/type
  mapping, stale target profile, stale provider mirror, stale candidate
  metadata mirrors, and statement-plan mismatches.
* Keep common EmitC/export neutral. Do not move RVV standalone reduction
  semantics into common EmitC, descriptors, target artifact metadata, route
  ids, artifact names, test names, scripts, or exact intrinsic spelling.
* Do not change route emission, generated C/C++, runtime ABI order,
  standalone reduction computation, source/scalar-result layout, mask/tail
  policy, inactive neutral behavior, correctness, or performance behavior. If
  any of those change, real `ssh rvv` evidence is required.

## Acceptance Criteria

* [x] Production code exposes a provider-owned standalone reduction route
      validation contract API beyond metadata mirrors and beyond raw canonical
      facts.
* [x] The contract covers existing plain, computed-mask, and runtime-scalar
      computed-mask standalone reduction add/min/max route families.
* [x] Target artifact validation consumes the provider contract for standalone
      reduction route payload, ABI order/parameters, header/type mappings,
      source/scalar-result channel facts, scalar-result boundary, inactive
      neutral policy, mask/accumulation facts, and statement-plan expectations.
* [x] Target standalone reduction validation no longer owns duplicate expected
      route payload/header/type/runtime ABI/statement-plan truth for fields
      represented in the provider contract.
* [x] Candidate metadata mirror validation consumes a provider-owned standalone
      reduction mirror contract and still rejects stale/mismatched mirrors as
      a separate consume-only check.
* [x] Focused target/provider tests prove contract construction and
      fail-closed rejection of stale/missing/cross-family/cross-route
      standalone reduction contract fields.
* [x] Focused lit/script filters for touched standalone reduction
      fixture/script families pass.
* [x] Bounded old-authority scan over touched files finds no new legacy
      `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-artifact, descriptor/direct-C/source-export,
      route-id/artifact-name authority, mirror-only authority, or exact
      `i32m1` intrinsic authority additions.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Technical Approach

1. Mirror the existing MAcc and widening dot-reduce contract shapes only where
   they match standalone reduction needs.
2. Add standalone reduction validation contract declarations to
   `RVVEmitCRouteProvider.h`.
3. Implement the contract builder in the RVV plugin/provider-owned code that
   already owns standalone reduction facts and route-family validation.
4. Add a standalone reduction metadata mirror contract set in the provider
   layer so target validation consumes expected mirrors and stale mirror keys
   from provider-owned contract data.
5. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` to validate standalone
   reduction rebuilt route payloads and candidate mirrors against provider
   contracts instead of target-local expected facts.
6. Add or adjust focused C++ checks in `TargetArtifactExportTest.cpp` for
   positive contract construction and fail-closed stale/missing/cross-family
   contract fields.
7. Run focused builds, C++ tests, lit/script filters, bounded old-authority
   scan, and diff check.

## Out Of Scope

* No new standalone reduction coverage, dtype/LMUL clone batch, new
  add/min/max variants, source-front-door positive route, widening dot-reduce
  work, MAcc work, compare/select/conversion/memory expansion, high-level
  frontend lowering, dashboards, broad smoke matrix, or artifact-only
  evidence.
* No generated runtime behavior change.
* No movement of RVV semantics into common EmitC/export or target artifact
  metadata.
* No `ssh rvv` run unless this task changes generated code, runtime ABI order,
  emitted intrinsics, standalone reduction computation, source/scalar-result
  layout, mask/tail policy, inactive neutral behavior, correctness, or
  performance claims.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Build `tianchenrv-rvv-extension-plugin-test` if provider headers or RVV
  plugin code change.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-rvv-extension-plugin-test` if built.
* Run focused lit filters for standalone reduction fixture/script families,
  especially `standalone-reduce` and runtime-scalar computed-mask standalone
  reduction filters touched by this change.
* Run bounded old-authority scans over touched source/test files and changed
  lines.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if the final diff remains a provider/target validation
  ownership extraction with no generated runtime semantic change; state the
  no-runtime-change rationale and point to archived runtime evidence.

## Completion Evidence

Completed in this round:

* Added `RVVStandaloneReductionRouteValidationContract` and
  `RVVStandaloneReductionRouteMetadataMirrorContractSet` provider APIs.
* Built provider contracts from `RVVStandaloneReductionRouteFacts` plus rebuilt
  route descriptions for plain, computed-mask, and runtime-scalar
  computed-mask standalone add/min/max routes.
* Rewired target artifact standalone reduction provider-fact and candidate
  mirror validation to consume provider contracts.
* Removed obsolete target-local standalone reduction payload and unreachable
  candidate mirror helper bodies for fields now represented by provider
  contracts.
* Added focused C++ coverage for positive contract construction and stale
  provider/candidate mirror failures.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the standalone
  reduction route validation contract.

Checks passed:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter standalone-reduce`
  from `build/test` selected 42 tests and all passed.
* Bounded old-authority diff scan found no matches.
* `rtk git diff --check`

No `ssh rvv` was run because this task changed provider/target validation
ownership and metadata/contract checking only. It did not change emitted
runtime code, runtime ABI order, standalone reduction computation,
source/scalar-result layout, mask/tail behavior, inactive neutral behavior,
correctness claims, or performance claims.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-widening-dot-reduce-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-standalone-reduction-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-standalone-reduction-route-fact-canonicalization/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-standalone-reduction-production-validation-boundary/prd.md`

Likely live files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*standalone-reduce*.mlir`
* `test/Scripts/*standalone-reduce*.test`
