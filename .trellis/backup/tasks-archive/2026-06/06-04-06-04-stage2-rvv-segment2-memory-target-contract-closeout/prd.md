# Stage2 RVV segment2 memory target raw-fact residue closeout

## Goal

Remove the remaining target-side raw route-fact reconstruction for plain and
computed-mask segment2 memory target artifact validation. The target consumer
must validate rebuilt route payloads and artifact metadata by consuming the
provider-owned `RVVSegment2MemoryRouteValidationContract`.

The bounded production path is:

```text
selected typed tcrv_rvv segment2 memory body
  -> RVV plugin-owned segment2 family facts and statement-plan facts
  -> RVVSegment2MemoryRouteValidationContract
  -> target artifact validation as a consume-only client
```

This task is a target-consumer closeout. It does not add segment widths,
segment3+, new memory operations, dtype/LMUL clones, emitted C behavior,
runtime ABI behavior, or runtime/correctness/performance claims.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV segment2 memory target raw-fact residue closeout`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` started at
  `f9269159 rvv: close compare-select target contract consumer`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` has a Segment2 Target Export
  Consumer Contract requiring target export to rebuild the provider route and
  consume rebuilt route/provider description authority before accepting
  generated artifact/header claims.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-owned
  facts and contracts to be the canonical shared surface. Target artifact
  validation may compare provider contracts and mirrors, but must not
  reconstruct RVV route semantics from target-local constants, route ids,
  artifact names, C strings, descriptor residue, test names, or mirror
  metadata.
* The archived compare/select closeout task provides the closest consume-only
  pattern: target validation consumes a provider validation contract and a
  bounded scan proves raw route-fact consumption was removed from target
  validation.
* Live inspection shows
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` already exposes
  `RVVSegment2MemoryRouteValidationContract` and
  `getRVVSegment2MemoryRouteValidationContract(...)`.
* Live inspection shows `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  still references `RVVPlainSegment2MemoryRouteFacts`,
  `RVVComputedMaskSegment2MemoryRouteFacts`,
  `getRVVPlainSegment2MemoryRouteFacts(...)`, and
  `getRVVComputedMaskSegment2MemoryRouteFacts(...)`.
* Provider-side raw fact structs/functions may remain as RVV plugin/provider
  internals. The target-side closeout requirement applies to
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` and target
  validation consumers.

## Requirements

* Rewire segment2 target artifact validation for
  `segment2_deinterleave_unit_store`, `segment2_interleave_unit_load`,
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load` to consume
  `RVVSegment2MemoryRouteValidationContract`.
* Remove or make unreachable target-local consumption of
  `RVVComputedMaskSegment2MemoryRouteFacts`,
  `RVVPlainSegment2MemoryRouteFacts`,
  `getRVVComputedMaskSegment2MemoryRouteFacts(...)`, and
  `getRVVPlainSegment2MemoryRouteFacts(...)` from
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Remove segment2 expected-runtime-ABI reconstruction from target validation.
  Runtime ABI order and parameter roles must come from the segment2 validation
  contract.
* Preserve fail-closed diagnostics for missing, stale, cross-family, or
  mismatched segment2 contract fields covering operation kind, plain versus
  computed-mask family, interleave/deinterleave/update/load/store direction,
  memory form, segment count/layout, field roles, mask source and mask/tail
  policy, runtime `n`/AVL/VL binding, ABI order and parameter roles, route
  operand binding summary, header/type/intrinsic/profile facts,
  statement-plan expectations, `provider_supported_mirror`, and stale mirror
  rejection.
* Keep common EmitC/export neutral. Target validation must not choose segment2
  semantics, interleave/deinterleave direction, computed-mask behavior,
  runtime ABI order, parameter roles, segment intrinsic leaves, field roles, or
  statement-plan shape from target-local route facts, route ids, metadata
  mirrors, C strings, descriptor residue, or test names.
* Do not weaken the already closed compare/select, unit-stride masked-memory,
  computed-mask strided, computed-mask indexed, base-memory, or reduction
  contracts.

## Acceptance Criteria

* [x] `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` no longer
      directly consumes `RVVComputedMaskSegment2MemoryRouteFacts`,
      `RVVPlainSegment2MemoryRouteFacts`,
      `getRVVComputedMaskSegment2MemoryRouteFacts(...)`, or
      `getRVVPlainSegment2MemoryRouteFacts(...)`.
* [x] Segment2 payload validation obtains and validates
      `RVVSegment2MemoryRouteValidationContract` before accepting rebuilt
      route payloads.
* [x] Segment2 runtime ABI validation uses the contract's runtime ABI order
      and parameter roles rather than target-local expected ABI helpers.
* [x] Segment2 candidate metadata mirror validation remains fail-closed for
      stale, missing, cross-family, or mismatched provider mirrors.
* [x] Focused C++ coverage proves stale or mismatched segment2 contract fields
      still fail closed where production-boundary changes require new
      coverage.
* [x] Focused generated-bundle/lit dry-run filters for segment2, interleave,
      deinterleave, computed-mask segment2, and segment2 update/load/store
      fixtures pass.
* [x] Bounded direct-consumption scan over
      `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` finds no
      direct segment2 raw route-fact consumption.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
      descriptor/direct-C/source-export, route-id/artifact-name authority,
      mirror-only authority, or exact `__riscv_*_i32m1` intrinsic authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Technical Approach

1. Treat `RVVSegment2MemoryRouteValidationContract` as the target-side segment2
   validation authority.
2. Replace any target helper that switches from operation kind to plain or
   computed-mask segment2 raw facts with contract lookup from the rebuilt route
   description.
3. Pass contract-derived runtime ABI order, ABI roles, statement-plan counts,
   segment layout, field roles, mask facts, headers, type mappings, profile
   mirrors, and intrinsic facts into existing target checks.
4. Remove dead target helper functions that only reconstruct segment2 raw facts
   or expected ABI facts.
5. Add or update focused C++ mutations only if the production rewiring removes
   coverage or exposes a missing stale/cross-family failure.
6. Run focused build/test/lit checks, direct-consumption scan,
   old-authority scan, `git diff --check`, Trellis validation, archive, and
   commit if complete.

## Evidence Plan

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused lit/generated-bundle dry-run filters for `segment2`, `interleave`,
  `deinterleave`, `computed-mask-segment2`, and segment2 update/load/store
  fixtures.
* Direct-consumption scan:
  `rtk rg -n "RVVComputedMaskSegment2MemoryRouteFacts|RVVPlainSegment2MemoryRouteFacts|getRVVComputedMaskSegment2MemoryRouteFacts|getRVVPlainSegment2MemoryRouteFacts" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* Bounded old-authority scan over touched code/test/task files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-segment2-memory-target-contract-closeout`
* Do not run `ssh rvv` if this remains validation-contract ownership only and
  emitted C, runtime ABI, segment2 behavior, correctness behavior, and
  performance behavior do not change.

## Out Of Scope

* No new segment widths, segment3+, new memory operations, dtype/LMUL clone
  batches, compare/select expansion, reduction/MAcc work, source-front-door
  routes, high-level frontend lowering, dashboards, broad smoke matrices, or
  artifact-only evidence.
* No weakening of compare/select, unit-stride masked-memory, computed-mask
  strided, computed-mask indexed, base-memory, or reduction contracts.
* No movement of RVV semantics into common EmitC/export or target-local
  reconstruction.
* No `ssh rvv` runtime claim unless emitted C, runtime ABI, segment2 behavior,
  correctness, or performance changes.

## Definition Of Done

* Target segment2 memory validation is provider-contract-backed and no longer
  consumes raw plain/computed-mask segment2 facts directly.
* Focused local checks, lit/generated-bundle dry-runs, bounded scans, and
  `git diff --check` are recorded.
* No runtime/correctness/performance claim is made without real `ssh rvv`
  evidence; if this remains validation-only, record the no-runtime-change
  rationale.
* The Trellis task is finished/archived and one coherent commit is created if
  all acceptance criteria pass.

## Implementation Results

* Removed target-local segment2 raw fact consumers from
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`:
  `validateRVVComputedMaskSegment2MemoryProviderFacts(...)`,
  `validateRVVPlainSegment2MemoryProviderFacts(...)`,
  `getRVVComputedMaskSegment2MemoryFactsForDescription(...)`,
  `getRVVPlainSegment2MemoryFactsForDescription(...)`, the plain/computed-mask
  segment2 classifier helpers used only by those raw paths, and the
  target-local expected runtime ABI reconstruction helpers.
* Kept the active segment2 payload path on
  `RVVSegment2MemoryRouteValidationContract`:
  `validateRVVSegment2MemoryRoutePayloadFacts(...)` obtains the contract,
  checks route id/description fields, validates runtime ABI facts from the
  contract, validates provider fields from the contract, checks route
  headers/type mappings/ABI mappings, and validates the statement plan with
  contract-owned step counts and ABI parameters.
* Kept candidate mirror validation on
  `getRVVSegment2MemoryRouteMetadataMirrorContract(...)`; no target-local
  plain/computed-mask segment2 raw facts remain as fallback authority.
* Updated two computed-mask segment2 update negative fixtures so stale mirror
  FileCheck strings match the now-active provider mirror contract diagnostics
  (`computed-mask segment2` rather than the old generic `segment2-memory`
  wording).

## Evidence Results

* Build passed:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`.
  The build emitted one pre-existing unused-function warning for
  `validateRVVTargetOwnedRouteABIMappings`, but no errors.
* C++ target artifact test passed:
  `rtk build/bin/tianchenrv-target-artifact-export-test`.
* C++ RVV plugin test passed:
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`.
* Focused lit/generated-bundle dry-run passed from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter '(segment2|interleave|deinterleave|computed-masked-segment2)'`.
  It selected 30 of 477 tests; all 30 passed.
* Self-repair performed: the first lit run exposed stale FileCheck text in
  computed-mask segment2 update negative fixtures. The production diagnostics
  were more specific because candidate mirror validation now consumes the
  provider mirror contract. The fixtures were updated, and the same lit filter
  passed.
* Direct segment2 raw-fact consumption scan passed with no matches:
  `rtk rg -n "RVVComputedMaskSegment2MemoryRouteFacts|RVVPlainSegment2MemoryRouteFacts|getRVVComputedMaskSegment2MemoryRouteFacts|getRVVPlainSegment2MemoryRouteFacts|getRVVComputedMaskSegment2MemoryFactsForDescription|getRVVPlainSegment2MemoryFactsForDescription|getRVVSegment2MemoryExpectedRuntimeABI" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Old-authority scan over the touched source, test, and task files found no
  added positive legacy-authority dependency.
* `rtk git diff --check` passed.
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-segment2-memory-target-contract-closeout`
  passed.
* Spec update judgment: no `.trellis/spec/` update was needed. This round
  implemented the existing Segment2 Target Export Consumer Contract rather than
  creating a new contract or convention.
* `ssh rvv` was not run because this round changed only target artifact
  validation ownership and stale diagnostic expectations. It did not change
  emitted C/C++, runtime ABI order, segment2 intrinsics, segment2 computation,
  mask/tail behavior, correctness behavior, or performance behavior.
