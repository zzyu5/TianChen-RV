# Stage2 RVV provider-owned memory route contract extraction

## Goal

Extract the normalized RVV memory route metadata mirror contract from the
target-local helper into the RVV plugin/provider-owned interface. Target
artifact validation must consume provider-built memory contract entries for
the already-supported memory route families and must not assemble operation,
ABI, header/type, intrinsic, mask/tail, stride/index/segment, passthrough, or
route-family mirror semantics locally from metadata, route names, fixture
names, descriptors, or exact intrinsic spellings.

The intended chain is:

```text
selected typed tcrv_rvv memory body
  -> RVV plugin route facts
  -> provider-owned normalized memory validation contract
  -> target artifact validation consumes that contract
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV provider-owned memory route contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `8ae576aa rvv: consolidate memory route validation mirrors`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-memory-route-family-validation-contract-consolidation/`
  consolidated memory mirror checks only inside
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Live target code still defines `RVVMemoryRouteMetadataMirrorContract` and
  `validateRVVMemoryRouteMetadataMirrorContract` in the target validator
  anonymous namespace.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` already exposes
  provider-owned fact surfaces for base memory, unit-stride masked memory,
  computed-mask indexed/strided memory, plain segment2 memory, and
  computed-mask segment2 memory.
* The provider implementations populate those facts in
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` and
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires target artifact
  validation to dispatch from the rebuilt provider description and validate
  metadata only as mirrors of provider facts.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the relevant
  provider-owned memory fact surfaces and fail-closed target acceptance
  behavior.

## Requirements

* Add a small RVV plugin/provider-owned normalized memory route contract
  surface in C++.
* The contract surface must package mirror entries as provider-owned
  `key/expected/label` facts and stale mirror keys as provider-owned rejection
  lists.
* Build contract entries from existing provider facts and provider route
  descriptions for the already-supported memory families:
  base/unit-stride, masked unit-stride, strided, indexed, plain segment2, and
  computed-mask segment2 memory.
* Rewire target artifact validation to consume provider-owned contract entries
  instead of defining `RVVMemoryRouteMetadataMirrorContract` locally.
* Keep target-side behavior limited to artifact candidate metadata lookup,
  error formatting, and invocation of family validators.
* Preserve family-specific provider authority and existing provider fact
  validators. Do not weaken segment2 statement-plan checks or computed-mask
  memory checks.
* Fail closed for missing, stale, cross-family, cross-route, or mismatched
  provider contract fields, including runtime ABI order, operand binding,
  header/type summaries, provider support mirrors, route-family mirrors,
  memory form, stride/index/mask/segment fields, passthrough/inactive-lane
  fields, mask-tail plan/owner fields, update arithmetic fields, and stale
  non-memory route-family mirrors.
* Keep common EmitC/export neutral. Do not move RVV route semantics into common
  EmitC or target metadata.
* Do not change route emission, runtime ABI order, generated code, load/store
  behavior, mask/tail policy, passthrough behavior, destination preservation,
  intrinsic spelling, runtime correctness, or performance behavior. If any of
  those change, real `ssh rvv` evidence becomes required.

## Acceptance Criteria

* [ ] Production code exposes a provider-owned normalized memory metadata
      mirror contract from RVV plugin/provider headers and implementation.
* [ ] The target validator no longer owns the memory mirror contract struct or
      target-local memory contract construction logic for the in-scope memory
      route families.
* [ ] Base/unit, masked unit, strided, indexed, plain segment2, and
      computed-mask segment2 candidate mirror validation consumes
      provider-owned contract entries. If full-family extraction is too large
      for one round, complete one coherent submodule such as base + masked
      unit + strided memory and document the exact continuation point.
* [ ] Candidate metadata is still validated only after provider route
      reconstruction and only as a mirror of provider facts.
* [ ] Fail-closed behavior remains in place for missing/stale/cross-family/
      cross-route provider facts and candidate mirrors.
* [ ] Existing focused target artifact tests continue to prove representative
      stale provider and candidate facts fail closed.
* [ ] No new route coverage, dtype/LMUL clones, source-front-door positive
      routes, descriptor-driven computation, common EmitC RVV semantics,
      mirror-only authority, exact intrinsic spelling authority, or legacy
      `i32m1` route authority is introduced.
* [ ] Focused build/tests, bounded old-authority scan, `rtk git diff --check`,
      Trellis finish/archive, clean worktree, and one coherent commit complete
      the round if the module behavior is complete.

## Technical Approach

1. Add provider-owned mirror contract data structures and accessors in
   `RVVEmitCRouteProvider.h`.
2. Implement base-memory contract construction close to
   `getRVVBaseMemoryMovementRouteFacts(...)`, because those entries are built
   directly from `RVVBaseMemoryMovementRouteFacts` and unit-stride masked
   typed-compute facts.
3. Implement segment2 contract construction close to the segment2 facts in
   `RVVEmitCRoutePlanning.cpp`, because it needs plain and computed-mask
   segment2 fact surfaces plus common provider description fields.
4. Rewire
   `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` to iterate
   provider contract entries and stale-key arrays while preserving target
   error messages and existing provider/statement validations.
5. Keep current C++ target tests as the primary fail-closed coverage unless
   extraction exposes a gap; add narrowly focused tests only for new uncovered
   failure modes.
6. Run focused provider/target builds and memory lit filters, then archive and
   commit if complete.

## Out Of Scope

* No new RVV memory route coverage.
* No dtype, LMUL, arithmetic, reduction, compare, conversion, frontend,
  dashboard, smoke-matrix, or evidence-only expansion.
* No source-front-door positive route or descriptor/direct-C/source-export
  route authority.
* No movement of RVV semantics into common EmitC/export or target artifact
  metadata.
* No `ssh rvv` runtime run unless generated runtime behavior, ABI order,
  mask/tail behavior, load/store behavior, passthrough/destination
  preservation, correctness, or performance claims change.

## Evidence Plan

* Build and run `tianchenrv-target-artifact-export-test`.
* Build and run `tianchenrv-rvv-extension-plugin-test`, because provider
  headers or plugin implementation code will change.
* Run focused lit filters for the memory families whose target consumers
  change:
  `(strided-load-unit-store|unit-load-strided-store|indexed-gather-unit-store|indexed-scatter-unit-load|masked-unit-(load-store|store)|segment2-(deinterleave|interleave)|computed-masked-segment2-(load|store|update))`.
* Run a bounded old-authority scan over touched files.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if the change remains a provider/target validation
  contract extraction with no generated runtime semantic change.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Previous task read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-memory-route-family-validation-contract-consolidation/task.json`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-memory-route-family-validation-contract-consolidation/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-memory-route-family-validation-contract-consolidation/implement.jsonl`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-memory-route-family-validation-contract-consolidation/check.jsonl`

Relevant journal:

* `.trellis/workspace/codex/journal-21.md`, Session 415.

Likely live files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* focused memory fixtures under `test/Target/RVV` and `test/Scripts`

## Completion Evidence

* Added provider-owned `RVVMemoryRouteMetadataMirrorContract` and
  `RVVMemoryRouteMetadataMirrorContractSet` in
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
* Added `getRVVBaseMemoryRouteMetadataMirrorContract(...)` in the RVV
  base-memory route-family provider implementation. The contract is built from
  `RVVBaseMemoryMovementRouteFacts` plus unit-stride masked memory facts where
  masked typed-compute mirrors are required.
* Added `getRVVSegment2MemoryRouteMetadataMirrorContract(...)` in the RVV
  route planning provider implementation. The contract is built from plain
  segment2 and computed-mask segment2 provider facts plus the rebuilt provider
  route description for route-description-only mirror fields that are not
  currently materialized as artifact metadata.
* Rewired target artifact validation to consume provider contract sets for
  base/unit, masked unit, strided, indexed, plain segment2, and computed-mask
  segment2 memory candidate mirrors.
* Deleted the target-local `RVVMemoryRouteMetadataMirrorContract` struct and
  target-local base/segment2 memory mirror table construction.
* Preserved existing family-specific provider fact validation, route payload
  validation, header/type validation, runtime ABI validation, and segment2
  statement-plan validation.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  provider-owned memory metadata mirror contract API and validation matrix.
* Did not change route emission, runtime ABI order, generated C/C++, load/store
  behavior, mask/tail behavior, passthrough/destination preservation,
  intrinsic behavior, runtime correctness, or performance behavior.
* `ssh rvv` was not run because this round only moved target metadata mirror
  contract ownership into the provider and made no runtime/correctness/
  performance claim.

Checks run:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter '(strided-load-unit-store|unit-load-strided-store|indexed-gather-unit-store|indexed-scatter-unit-load|masked-unit-(load-store|store)|segment2-(deinterleave|interleave)|computed-masked-segment2-(load|store|update))'` from `build/test`, selecting 60 tests and passing 60.
* `rtk git diff --check`
* Bounded changed-line old-authority scan over touched production files found
  no new legacy `i32m1`, `RVVI32M1`, `rvv-i32m1`, source-front-door,
  source-export, descriptor/direct-C, route-id/artifact-name, bare status, or
  mirror-only authority additions.

Self-repair:

* Initial provider segment2 contract required mask/tail metadata from the
  segment2 fact accessor, but current segment2 artifact metadata mirrors the
  rebuilt provider route description for those fields. The target C++ test
  failed on a missing `tcrv_rvv.mask_tail_policy_route_family_plan` mirror.
  The contract now uses the rebuilt provider route description for those
  route-description-only fields, preserving existing artifact behavior while
  keeping contract construction provider-owned.
