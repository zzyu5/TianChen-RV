# Stage2 RVV unit-stride masked-memory route-family provider contract extraction

## Goal

Introduce a provider-owned route validation contract for the existing
unit-stride masked-memory route family and rewire target artifact validation
for `masked_unit_load_store` and `masked_unit_store` to consume that contract.

The bounded production path is:

```text
selected typed tcrv_rvv unit-stride masked-memory body
  -> RVV plugin-owned mask/memory/provider route facts
  -> provider-owned unit-stride masked-memory route validation contract
  -> target artifact validation as a consume-only client
```

This task does not add a new memory route or change emitted runtime behavior.
It tightens the validation boundary so target code no longer reconstructs the
static-mask unit-stride masked load/store payload directly from
`RVVUnitStrideMaskedMemoryRouteFacts`, target-local helper truth, route ids,
C strings, fixture names, descriptor residue, or mirror metadata.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV unit-stride masked-memory route-family provider contract extraction`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `d7f49bee rvv: extract computed-mask indexed memory route contract`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md` require RVV route authority
  to remain selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body
  -> RVV plugin-owned provider route -> common EmitC -> target artifact.
* `.trellis/spec/testing/mlir-testing-contract.md` and
  `.trellis/spec/validation/index.md` require real `ssh rvv` evidence only
  when runtime, correctness, or performance behavior is claimed or changed.
* The previous completed tasks extracted provider-owned route validation
  contracts for computed-mask strided memory and computed-mask indexed memory.
  Their target validators now consume contracts rather than raw route facts.
* The older masked unit-stride production validation task introduced
  `RVVUnitStrideMaskedMemoryRouteFacts` for static-mask, computed-mask, and
  runtime-scalar masked memory routes and wired target artifact validation to
  consume those facts.
* Live inspection shows `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  exposes `RVVUnitStrideMaskedMemoryRouteFacts` and
  `getRVVUnitStrideMaskedMemoryRouteFacts(...)`, but does not expose
  `RVVUnitStrideMaskedMemoryRouteValidationContract`.
* Live inspection shows
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` still has
  direct target-side consumption of
  `RVVUnitStrideMaskedMemoryRouteFacts` /
  `getRVVUnitStrideMaskedMemoryRouteFacts(...)` for the unit-stride masked
  route-family validation path.
* The primary in-scope contract family is the existing static-mask
  unit-stride `MaskedUnitLoadStore` and `MaskedUnitStore` pair. During
  implementation, the shared
  `RVVUnitStrideMaskedMemoryRouteFacts` accessor boundary was converted as one
  coherent family so target validation no longer directly consumes raw
  unit-stride masked route facts for `ComputedMaskUnitLoadStore`,
  `RuntimeScalarComputedMaskStore`, or
  `RuntimeScalarComputedMaskLoadStore` either.

## Requirements

* Add a provider-owned
  `RVVUnitStrideMaskedMemoryRouteValidationContract` for
  `MaskedUnitLoadStore` and `MaskedUnitStore`, with the same accessor covering
  the already-existing unit-stride computed-mask/runtime-scalar consumers that
  share `RVVUnitStrideMaskedMemoryRouteFacts`.
* The contract must cover operation kind, memory form, route id, static mask
  source, mask/tail policy, inactive-lane and passthrough behavior,
  input/output buffer roles, runtime `n`/AVL/VL binding, runtime ABI parameter
  order and roles, route operand binding plan and summary, header/type facts,
  intrinsic/profile facts, dtype/config relation, statement-plan
  expectations, and `provider_supported_mirror`.
* The provider must build the contract from the canonical unit-stride masked
  memory facts and rebuilt provider route description. Target validation may
  consume the contract but must not choose masked load/store semantics from
  target-local fact reconstruction.
* Target artifact validation for `MaskedUnitLoadStore`, `MaskedUnitStore`, and
  existing unit-stride computed-mask/runtime-scalar consumers must consume the
  provider contract before validating rebuilt route headers, type mappings,
  ABI mappings, statement plans, and candidate artifact metadata mirrors.
* Candidate metadata mirror validation for static-mask unit-stride masked
  routes must reject stale, missing, cross-family, or cross-load/store
  mirrors through provider-owned expected fields wherever this route family
  has provider mirror contract coverage.
* Fail closed for stale or mismatched contract fields, including stale plain
  base-memory, strided, indexed, segment2, computed-mask, runtime-scalar,
  arithmetic, descriptor, source-export/direct-C, legacy i32, or mirror-only
  residue.
* Keep common EmitC/export neutral. It may carry provider-built payloads and
  mirrors unchanged, but must not infer mask, load/store direction,
  passthrough, header/type, intrinsic, policy, support, dtype, or ABI facts.

## Acceptance Criteria

* [x] `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes a
      `RVVUnitStrideMaskedMemoryRouteValidationContract` and accessor whose
      shape follows the existing provider contract idiom.
* [x] RVV provider implementation builds the contract from
      `RVVUnitStrideMaskedMemoryRouteFacts` and the rebuilt route description,
      including runtime ABI parameters, required headers, type mappings,
      route operand binding summaries, statement-plan counts, and mask/policy
      facts.
* [x] Target artifact provider-fact validation for `MaskedUnitLoadStore` and
      `MaskedUnitStore`, plus existing unit-stride computed-mask/runtime-scalar
      consumers, consumes the provider validation contract rather than directly
      consuming `RVVUnitStrideMaskedMemoryRouteFacts` in target-local checks.
* [x] Target candidate mirror validation for these unit-stride masked-memory
      routes consumes provider-owned expected mirror fields and rejects stale
      or cross-family mirrors.
* [x] Focused target C++ coverage proves stale unit-stride masked-memory
      contract fields fail closed, including provider mirror, target profile,
      header/type, route operand binding, runtime ABI, static mask source,
      passthrough/inactive-lane behavior, statement-plan counts, and
      load-store/store cross-contamination.
* [x] Focused generated-bundle dry-run/lit filters for
      `masked_unit_load_store` and `masked_unit_store` still pass.
* [x] Focused C++ targets
      `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` pass because provider headers and
      plugin code change.
* [x] A bounded scan over touched files and
      `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` shows direct
      unit-stride masked route-fact consumption is removed from target
      validation.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on legacy i32, route-id, artifact-name, descriptor,
      source-front-door/source-export/direct-C, common-EmitC, exact-intrinsic,
      or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task status, context, journal, archive state, and commit are
      truthful.

## Technical Approach

1. Mirror the computed-mask indexed/strided route validation contract pattern
   for the unit-stride masked memory route family in the RVV provider header
   and route planning implementation.
2. Populate the contract from the canonical
   `RVVUnitStrideMaskedMemoryRouteFacts` accessor plus rebuilt
   `RVVSelectedBodyEmitCRouteDescription` payload details that are provider
   route details rather than canonical route facts.
3. Reuse or extend provider-owned memory metadata mirror contract helpers for
   static-mask unit-stride masked-memory expected mirrors and stale mirror
   rejection.
4. Replace target-local unit-stride masked provider-fact validation with a
   contract consumer that validates operation, memory form, route id, runtime
   ABI parameters, headers, type mappings, ABI mappings, route operand
   binding, mask/policy facts, passthrough/inactive-lane behavior, statement
   plan shape, target profile, and stale cross-family residue.
5. Add focused C++ mutations in the existing target artifact test where
   production validator paths are already exercised.
6. Run focused build/test/lit checks, direct-consumption scan,
   old-authority scan, `git diff --check`, validate task context, archive, and
   commit if complete.

## Evidence Plan

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* From `build/test`, focused lit filters for static-mask unit-stride masked
  memory, including explicit/pre-realized selected-body target artifact
  fixtures and generated-bundle dry-runs for:
  * `masked-unit-load-store`
  * `masked-unit-store`
* Bounded target-validator scan proving direct unit-stride masked route-fact
  consumption is removed from target validation.
* Bounded old-authority scan over touched implementation/test/spec/task files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-unit-stride-masked-memory-provider-contract`
* Do not run `ssh rvv` if this remains validation-contract ownership only and
  emitted C, runtime ABI, mask behavior, correctness behavior, and performance
  behavior do not change.

## Implementation Results

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` now exposes
  `RVVUnitStrideMaskedMemoryRouteValidationKind`,
  `RVVUnitStrideMaskedMemoryRouteValidationContract`, the type-mapping
  contract, and provider accessors for route validation plus metadata mirror
  validation.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` builds the contract from
  `RVVUnitStrideMaskedMemoryRouteFacts` and the rebuilt route description,
  including operation/memory form, dtype/config, runtime ABI order and roles,
  binding summary, mask source/role/form, base vs computed route-family facts,
  header/type/intrinsic facts, result/mask names, loop/VL names, and expected
  statement counts.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` no longer has a
  target-local `RVVUnitStrideMaskedMemoryRouteFacts` helper. The target
  validator consumes `RVVUnitStrideMaskedMemoryRouteValidationContract` before
  validating provider payloads and consumes
  `getRVVUnitStrideMaskedMemoryRouteMetadataMirrorContract(...)` before
  accepting candidate mirrors.
* `test/Target/TargetArtifactExportTest.cpp` adds focused positive and stale
  contract coverage for the static masked unit-load/store pair and the
  existing unit-stride computed-mask/runtime-scalar consumers covered by the
  shared accessor.
* `.trellis/spec/lowering-runtime/emitc-route.md` documents the new
  unit-stride masked memory route validation contract using the same 7-section
  code-spec shape as the computed-mask strided/indexed contracts.

## Evidence Results

* Build: `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
  passed.
* C++ provider/plugin test: `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
  passed.
* C++ target consumer test:
  `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
* Focused lit/generated-bundle dry-run:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'masked-unit-(load-store|store)|computed-masked-unit-load-store|runtime-scalar-cmp-masked-(memory|store|load-store)'`
  from `build/test` selected 24 of 477 tests and all 24 passed.
* Direct fact-consumption scan:
  `rtk rg -n "RVVUnitStrideMaskedMemoryRouteFacts|getRVVUnitStrideMaskedMemoryRouteFacts|getRVVUnitStrideMaskedMemoryFactsForDescription" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  returned no matches.
* Old-authority scan over code/test added lines returned no matches for
  legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
  descriptor/direct-C/source-export, or exact `__riscv_*_i32m1` authority.
  The only added-line hit across task/spec docs is a negative fail-closed
  wording in the spec.
* `rtk git diff --check` passed.
* `ssh rvv` was not run because this round only changes provider-owned
  validation contracts and target artifact validation consumption. It does not
  change emitted C, runtime ABI, mask behavior, correctness behavior, or
  performance behavior.

## Out Of Scope

* No new memory operations.
* No computed-mask strided/indexed rewrites beyond incidental shared helper
  reuse.
* No segment2, base-memory expansion, compare/select expansion, MAcc,
  reduction, conversion, dtype/LMUL clone batches, high-level Linalg/Vector
  frontend lowering, source-front-door routes, dashboards, broad smoke
  matrices, or artifact-only evidence task.
* No weakening of computed-mask strided/indexed contracts or existing artifact
  evidence.
* No movement of RVV semantics into common EmitC/export or target-local
  reconstruction.
* No new `ssh rvv` runtime claim unless emitted C, runtime ABI, mask/load-store
  behavior, correctness, or performance changes.

## Definition Of Done

* The two static-mask unit-stride masked-memory routes use a provider-owned
  route validation contract and target artifact validation consumes it as a
  client.
* Focused local checks, lit/generated-bundle dry-runs, bounded scans, and
  `git diff --check` are recorded.
* No runtime/correctness/performance claim is made without real `ssh rvv`
  evidence; if this remains validation-only, record the no-runtime-change
  rationale.
* The Trellis task is finished/archived and one coherent commit is created if
  all acceptance criteria pass.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/validation/index.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-computed-mask-strided-memory-provider-contract/prd.md`
* `.trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-computed-mask-indexed-memory-provider-contract/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-masked-unit-memory-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-provider-owned-memory-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-runtime-scalar-masked-memory-artifact-abi/prd.md`

Relevant journal entries read:

* `.trellis/workspace/codex/journal-22.md`, Session 426.
* `.trellis/workspace/codex/journal-21.md`, Session 414 and Session 416.

Relevant live files inspected before PRD:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* focused masked-memory fixtures under `test/Target/RVV` and `test/Scripts`
