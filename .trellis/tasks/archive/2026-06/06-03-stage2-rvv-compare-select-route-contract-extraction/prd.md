# Stage2 RVV compare/select route-family provider contract extraction

## Goal

Extract a provider-owned compare/select route validation contract from the
existing RVV compare/select route facts and rebuilt route descriptions, then
rewire target artifact validation to consume that contract for the existing
compare/select mask route families.

The intended ownership chain is:

```text
selected typed tcrv_rvv compare/select body
  -> RVV plugin-owned compare/select route facts
  -> provider-owned compare/select validation contract
  -> target artifact validation consumes the contract plus rebuilt route
```

Target artifact validation remains a consume-only client. It must not choose
compare/select semantics from target-local tables, route names, artifact
metadata, C strings, test names, descriptors, exact intrinsic spellings,
candidate mirrors, or stale route-family metadata.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select route-family provider contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `99bbec13 rvv: extract standalone reduction route validation contract`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact -> `ssh rvv` evidence when
  runtime/correctness/performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select
  selected-body realization and provider route facts to remain RVV
  plugin-owned, with common EmitC/export carrying only provider-built payloads.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the RVV
  route-family fact surface contract. Provider-owned facts are canonical when
  consumed by both provider planning and target artifact validation.
* The archived compare/select selected-body realization task established that
  plain compare/select, computed-mask select, runtime-scalar compare/select,
  and runtime-scalar dual compare-mask-and-select are realized before route
  facts are collected.
* The archived compare/select production validation task already established a
  provider fact surface for `cmp_select`, `runtime_scalar_cmp_select`, and the
  already-coherent dual compare/select route, but target validation still owns
  several compare/select provider-fact and mirror helper functions.
* The immediately previous standalone reduction commit provides the closest
  implementation pattern for this round: add a provider-owned route validation
  contract and metadata mirror contract, then make target validation iterate
  those contracts instead of reconstructing route-family truth locally.

## Requirements

* Add a provider-owned compare/select route validation contract API in the RVV
  provider layer for existing active compare/select mask route families:
  `cmp_select`, `runtime_scalar_cmp_select`, and
  `runtime_scalar_dual_cmp_mask_and_select` when it already shares the same
  compare/select facts surface.
* Build the contract from existing `RVVCompareSelectRouteFacts` plus the
  rebuilt `RVVSelectedBodyEmitCRouteDescription`. Do not add new predicates,
  new compare/select route coverage, dtype/LMUL clone batches, source-front-door
  routes, or artifact-only evidence.
* Include common route expectations:
  route id, operation, memory form, selected element type, SEW/LMUL, tail/mask
  policy, runtime-control plan, runtime ABI order and parameters, target leaf
  profile, `provider_supported_mirror`, required headers, C type summary,
  route operand binding plan/summary, route payload header/type requirements,
  and dynamic statement-plan names.
* Include compare/select family expectations:
  typed compute op, predicate kind, secondary predicate for dual compare,
  compare/select/mask-and intrinsics, mask type/C type, mask role/source/form,
  mask composition, result/mask names, true/false operand roles, selected
  result role, select layout, source/destination memory forms, inactive lane
  and passthrough/destination policy, runtime-scalar threshold role/C type,
  and statement-plan step counts.
* Rewire target artifact compare/select provider-fact validation to consume the
  provider validation contract for rebuilt route payload, route headers, type
  mappings, runtime ABI mappings, predicate/select semantics, mask/result
  layout, passthrough policy, mask/tail facts, and statement-plan
  expectations.
* Add a provider-owned compare/select metadata mirror contract set and rewire
  target candidate mirror validation to consume it. Candidate metadata remains
  a mirror after provider route construction, not route authority.
* Preserve fail-closed behavior for missing, stale, cross-family, cross-route,
  or mismatched contract fields, including stale plain facts on runtime-scalar
  routes, stale runtime-scalar facts on plain routes, stale dual facts on
  single-compare routes, stale non-compare route-family facts, stale predicate,
  stale mask/source/layout, stale runtime ABI order/parameters, stale binding,
  stale header/type mapping, stale target profile, stale provider mirror,
  stale candidate metadata mirrors, and statement-plan mismatches.
* Keep common EmitC/export neutral. Do not move RVV compare/select semantics
  into common EmitC, descriptors, target artifact metadata, route ids, artifact
  names, test names, scripts, or exact intrinsic spelling.
* Do not change route emission, generated C/C++, runtime ABI order,
  compare/select computation, mask/result layout, passthrough policy,
  mask/tail behavior, correctness, or performance behavior. If any of those
  change, real `ssh rvv` evidence is required.

## Acceptance Criteria

* [x] Production code exposes a provider-owned compare/select route validation
      contract API beyond raw canonical facts and beyond metadata mirrors.
* [x] The contract covers existing `cmp_select`, `runtime_scalar_cmp_select`,
      and already-coherent `runtime_scalar_dual_cmp_mask_and_select` routes, or
      leaves an exact continuation point if the dual route cannot be included
      without broadening the owner.
* [x] Target artifact validation consumes the provider contract for
      compare/select route payload, ABI order/parameters, header/type
      mappings, predicate/select semantics, mask/result layout,
      passthrough/destination policy, mask/tail facts, and statement-plan
      expectations.
* [x] Target compare/select validation no longer owns duplicate expected route
      payload/header/type/runtime ABI/statement-plan truth for fields
      represented in the provider contract.
* [x] Candidate metadata mirror validation consumes a provider-owned
      compare/select mirror contract and still rejects stale/mismatched mirrors
      as a separate consume-only check.
* [x] Focused target/provider tests prove contract construction and
      fail-closed rejection of stale/missing/cross-family/cross-route
      compare/select contract fields.
* [x] Focused lit/script filters for touched compare/select fixture/script
      families pass.
* [x] Bounded old-authority scan over touched files finds no new legacy
      `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-artifact, descriptor/direct-C/source-export,
      route-id/artifact-name authority, mirror-only authority, or exact
      `i32m1` intrinsic authority additions.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Technical Approach

1. Mirror the existing MAcc, widening dot-reduce, and standalone reduction
   contract shapes only where they match compare/select needs.
2. Add compare/select validation contract and metadata mirror contract
   declarations to `RVVEmitCRouteProvider.h`.
3. Implement the contract builder in the RVV plugin/provider-owned route
   planning file that already owns `RVVCompareSelectRouteFacts`.
4. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` to validate
   compare/select rebuilt route payloads and candidate mirrors against provider
   contracts instead of target-local expected facts.
5. Add or adjust focused C++ checks in `TargetArtifactExportTest.cpp` for
   positive contract construction and fail-closed stale/missing/cross-family
   contract fields.
6. Run focused builds, C++ tests, lit/script filters, bounded old-authority
   scan, and diff check.

## Out Of Scope

* No new compare/select predicates, new dtype/LMUL clone batch, new
  compare/select route variants, source-front-door positive route, reduction
  work, MAcc work, widening-dot work, conversion/memory/elementwise expansion,
  high-level frontend lowering, dashboards, broad smoke matrix, or artifact-only
  evidence.
* No generated runtime behavior change.
* No movement of RVV semantics into common EmitC/export or target artifact
  metadata.
* No `ssh rvv` run unless this task changes generated code, runtime ABI order,
  emitted intrinsics, compare/select computation, mask/result layout,
  passthrough policy, mask/tail behavior, correctness, or performance claims.

## Evidence Plan

* Validate the Trellis task context.
* Build `tianchenrv-target-artifact-export-test`.
* Build `tianchenrv-rvv-extension-plugin-test` because provider headers and
  RVV plugin planning code are expected to change.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
* Run focused lit/script filters for `cmp-select`, `runtime-scalar-cmp-select`,
  and dual compare/select script fixtures if touched.
* Run bounded old-authority scans over touched source/test/task files and
  changed lines.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if the final diff remains a provider/target validation
  ownership extraction with no generated runtime semantic change; state the
  no-runtime-change rationale and point to archived compare/select runtime
  evidence.

## Definition Of Done

* `implement.jsonl` and `check.jsonl` contain relevant spec/context entries and
  validate through the Trellis workflow.
* No runtime, correctness, or performance claim is made without real `ssh rvv`
  evidence.
* Trellis task status, evidence, workspace journal, and archive state
  truthfully record the completed bounded change or the exact continuation
  point.
* One coherent commit is created if the bounded task is complete.

## Completion Evidence

Implementation completed the bounded contract extraction:

* Added provider-owned compare/select route validation contract and metadata
  mirror contract APIs in `RVVEmitCRouteProvider.h`.
* Implemented provider contract construction in `RVVEmitCRoutePlanning.cpp`
  from existing compare/select route facts plus rebuilt route descriptions.
* Rewired compare/select target artifact route-family validation to require
  and consume provider route contracts and provider mirror contracts for
  compare/select producer route families.
* Added focused target artifact export checks proving positive contract
  construction for plain compare/select, runtime-scalar compare/select, and
  runtime-scalar dual compare-mask-and-select, while preserving fail-closed
  stale/mismatched mutation coverage.

Checks run:

* `rtk python3 ./.trellis/scripts/task.py validate 06-03-stage2-rvv-compare-select-route-contract-extraction`
* `rtk cmake --build build --target tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test`
* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused compare/select generated-bundle script dry-runs for `cmp_select`,
  pre-realized `cmp_select`, pre-realized `computed_mask_select`,
  pre-realized `runtime_scalar_cmp_select`, and pre-realized
  `runtime_scalar_dual_cmp_mask_and_select`.
* Bounded changed-line old-authority scan over touched source/test/task files.
* `rtk git diff --check`

Self-repair performed:

* The first target artifact export test run caught a changed stale
  non-dual-mask diagnostic. The provider contract validation now preserves the
  targeted `non-dual` diagnostic while keeping dual mask composition facts
  provider-owned.
* A concurrent build attempt hit a transient Ninja/static-library race. The
  affected target was rebuilt serially and passed.

No `ssh rvv` run was required: this round changed provider/target validation
contract ownership only. It did not change generated C/C++, runtime ABI order,
emitted compare/select intrinsics, computation, mask/result layout,
passthrough policy, mask/tail behavior, correctness, or performance behavior.
