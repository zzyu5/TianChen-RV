# Stage2 RVV compare/select route-contract target consume-only closeout

## Goal

Rewire the compare/select mask producer target artifact validation path so it
consumes the existing provider-owned `RVVCompareSelectRouteValidationContract`
and metadata mirror contract as the route-validation authority.

The bounded production path is:

```text
selected typed tcrv_rvv compare/select body
  -> RVV plugin-owned compare/select route facts
  -> provider-owned compare/select route validation contract
  -> target artifact validation as a consume-only client
```

This task is a target-consumer closeout. It does not add compare/select route
coverage, change emitted C/C++, change runtime ABI order, or alter
compare/select execution semantics.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select route-contract target consume-only closeout`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` started at
  `84e3a48a rvv: extract unit-stride masked memory route contract`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select
  operation kind, predicate, dtype/config, runtime ABI, mask/tail policy,
  intrinsic/header/type, and fail-closed route support to stay RVV
  plugin/provider-owned.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-owned
  facts and contracts to be the canonical shared surface when both provider and
  target validation consume route-family fields. Target artifact validation may
  compare provider contracts and mirrors, but must not reconstruct RVV route
  semantics from target-local constants, route ids, artifact names, C strings,
  descriptor residue, test names, or mirror metadata.
* The archived compare/select contract extraction task already added
  `RVVCompareSelectRouteValidationContract`,
  `getRVVCompareSelectRouteValidationContract(...)`, and
  `getRVVCompareSelectRouteMetadataMirrorContract(...)` for the active
  compare/select producer family.
* The immediately previous unit-stride masked-memory task provides the closest
  consume-only pattern: target validation consumes provider contracts and a
  bounded scan proves direct raw fact consumption was removed from target
  validation.
* Live inspection shows `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  already exposes `RVVCompareSelectRouteValidationContract` and the metadata
  mirror contract.
* Live inspection shows
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` builds the compare/select
  validation contract from provider-owned `RVVCompareSelectRouteFacts` plus the
  rebuilt route description.
* Live inspection shows
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` still directly
  references `RVVCompareSelectRouteFacts`,
  `RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts`, and
  `getRVVCompareSelectRouteFacts(...)` through target-local helper paths such
  as `validateRVVCompareSelectCanonicalProviderFacts(...)`,
  `getRVVCompareSelectFactsForDescription(...)`, and runtime-scalar dual helper
  reconstruction.

## Requirements

* Rewire target artifact validation for the compare/select producer family
  (`CmpSelect`, `ComputedMaskSelect`, `RuntimeScalarCompareSelect`, and
  `RuntimeScalarDualCompareMaskAndSelect`) to consume
  `RVVCompareSelectRouteValidationContract` and the provider-owned metadata
  mirror contract.
* Remove direct target-side consumption of
  `RVVCompareSelectRouteFacts` and
  `RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts` from
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Preserve the existing computed-mask memory contract consumers for
  unit-stride, strided, and indexed memory routes. Do not weaken their
  provider-owned validation contracts or mirror checks.
* Preserve fail-closed diagnostics for missing, stale, cross-family, or
  mismatched compare/select contract fields covering operation kind, memory
  form, route id, primary and secondary predicates, dtype/config, mask/tail
  policy, runtime `n`/AVL/VL binding, runtime ABI order and parameter roles,
  true/false value roles, operand binding summary, header/type/intrinsic/profile
  facts, statement-plan expectations, `provider_supported_mirror`, and stale
  mirror rejection.
* Keep common EmitC/export neutral. Target validation must not choose
  compare/select semantics, runtime ABI order, runtime parameter roles,
  predicate structure, intrinsic leaves, statement-plan shape, or artifact
  support from target-local reconstruction, route ids, metadata mirrors, C
  strings, descriptors, or fixture names.

## Acceptance Criteria

* [x] `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` no longer
      directly consumes `RVVCompareSelectRouteFacts`,
      `RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts`, or
      `getRVVCompareSelectRouteFacts(...)`.
* [x] Compare/select producer payload validation obtains and validates
      `RVVCompareSelectRouteValidationContract` before accepting rebuilt
      route payloads.
* [x] Compare/select producer candidate mirror validation obtains and validates
      `RVVCompareSelectRouteMetadataMirrorContractSet` before accepting
      artifact metadata mirrors.
* [x] The shared compare/select mask helper path no longer falls back to
      target-local compare/select expected facts for producer routes.
* [x] Existing computed-mask memory target contract consumers continue to pass
      and remain provider-contract-backed.
* [x] Focused C++ coverage proves stale or mismatched compare/select contract
      fields still fail closed, including at least runtime-scalar dual
      secondary predicate or mask-composition mismatch and stale/cross-family
      mirror rejection.
* [x] Focused generated-bundle/lit dry-run filters for compare/select and
      runtime-scalar dual compare/mask-select fixtures pass.
* [x] Bounded direct-consumption scan over
      `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` finds no
      direct `RVVCompareSelectRouteFacts`,
      `RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts`, or
      `getRVVCompareSelectRouteFacts` target validation usage.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
      descriptor/direct-C/source-export, route-id/artifact-name authority,
      mirror-only authority, or exact `__riscv_*_i32m1` intrinsic authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Technical Approach

1. Keep provider headers and provider contract construction mostly unchanged
   unless target closeout reveals a missing contract field.
2. In target validation, treat compare/select producers as the sole consumers
   of `RVVCompareSelectRouteValidationContract`; avoid invoking raw facts or
   target-local compare/select expected helper tables for those operations.
3. Remove obsolete target helper functions that reconstruct compare/select
   facts or runtime-scalar dual facts.
4. Where shared helper code still needs route expectations for memory-family
   routes, keep those checks behind the existing memory contracts rather than
   compare/select raw facts.
5. Add or update focused C++ mutations in `TargetArtifactExportTest.cpp` only
   if existing coverage does not prove the closeout behavior.
6. Run focused build/test/lit checks, direct-consumption scan, old-authority
   scan, `git diff --check`, Trellis validation, archive, and commit if
   complete.

## Evidence Plan

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused lit/generated-bundle dry-run filters for compare/select producer
  fixtures, including `cmp_select`, `computed_mask_select`,
  `runtime_scalar_cmp_select`, and
  `runtime_scalar_dual_cmp_mask_and_select`.
* Direct-consumption scan:
  `rtk rg -n "RVVCompareSelectRouteFacts|RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts|getRVVCompareSelectRouteFacts" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* Bounded old-authority scan over touched code/test/task files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-compare-select-route-contract-target-consume-only-closeout`
* Do not run `ssh rvv` if this remains validation-contract ownership only and
  emitted C, runtime ABI, compare/select behavior, mask/tail behavior,
  correctness behavior, and performance behavior do not change.

## Out Of Scope

* No new compare/select operations, predicates, memory operations, dtype/LMUL
  clone batches, segment2 work, MAcc/reduction work, conversion expansion,
  source-front-door routes, high-level frontend lowering, dashboards, broad
  smoke matrices, or artifact-only evidence.
* No weakening of unit-stride masked-memory, computed-mask strided memory, or
  computed-mask indexed memory contracts.
* No movement of RVV semantics into common EmitC/export or target-local
  reconstruction.
* No `ssh rvv` runtime claim unless emitted C, runtime ABI, compare/select
  behavior, mask/tail behavior, correctness, or performance changes.

## Definition Of Done

* Target compare/select producer validation is provider-contract-backed and no
  longer consumes raw compare/select facts directly.
* Focused local checks, lit/generated-bundle dry-runs, bounded scans, and
  `git diff --check` are recorded.
* No runtime/correctness/performance claim is made without real `ssh rvv`
  evidence; if this remains validation-only, record the no-runtime-change
  rationale.
* The Trellis task is finished/archived and one coherent commit is created if
  all acceptance criteria pass.

## Implementation Results

* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` no longer
  defines or calls target-local helpers that rebuild compare/select producer
  route truth from `RVVCompareSelectRouteFacts` or
  `RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts`.
* Compare/select producer payload validation now obtains
  `RVVCompareSelectRouteValidationContract`, checks the rebuilt route id
  against `contract.emitCRouteID`, validates route-description fields against
  the contract, and passes the provider-owned statement-plan step counts into
  the shared statement-plan checker.
* The shared runtime ABI helper now selects a provider contract first:
  compare/select producers use `RVVCompareSelectRouteValidationContract`, while
  computed-mask indexed, strided, and unit-stride memory routes continue to use
  their existing memory validation contracts.
* The default statement-plan step-count helper was narrowed to memory routes;
  compare/select producers no longer get statement-plan counts from target-local
  operation-name switches.
* Candidate metadata mirror validation for compare/select producers already
  consumed `RVVCompareSelectRouteMetadataMirrorContractSet`; this round kept
  that path intact and removed the dead raw-facts fallback.

## Evidence Results

* Build passed:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`.
  The build emitted one pre-existing unused-function warning in
  `RVVTargetArtifactRouteFamilyValidation.cpp` for
  `validateRVVTargetOwnedRouteABIMappings`, but no errors.
* C++ target artifact test passed:
  `rtk build/bin/tianchenrv-target-artifact-export-test`.
* C++ RVV plugin test passed:
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`.
* Focused lit/generated-bundle dry-run passed from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select'`.
  It selected 36 of 477 tests; all 36 passed.
* Direct compare/select raw-fact consumption scan passed with no matches:
  `rtk rg -n "RVVCompareSelectRouteFacts|RVVRuntimeScalarDualCompareMaskAndSelectRouteFacts|getRVVCompareSelectRouteFacts|getRVVRuntimeScalarDualCompareMaskAndSelectRouteFacts|getRVVCompareSelectFactsForDescription|getRVVRuntimeScalarDualCompareSelectFactsForDescription" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Old-authority scan over the touched source file found no added positive
  legacy-authority dependency. A broader scan over the task directory only
  found negative PRD boundary text.
* `rtk git diff --check` passed.
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-compare-select-route-contract-target-consume-only-closeout`
  passed.
* `ssh rvv` was not run because this round changed only target validation
  ownership. It did not change emitted C/C++, runtime ABI order, emitted
  compare/select intrinsics, compare/select computation, mask/result layout,
  passthrough policy, mask/tail behavior, correctness behavior, or performance
  behavior.
