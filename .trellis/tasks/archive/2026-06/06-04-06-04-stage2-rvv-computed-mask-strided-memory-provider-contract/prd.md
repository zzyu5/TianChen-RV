# Stage2 RVV computed-mask strided-memory route-family provider contract extraction

## Goal

Extract the existing computed-mask strided-memory route facts into a
provider-owned route validation contract and rewire target artifact validation
for the current `computed_masked_strided_store` and
`computed_masked_strided_load_unit_store` family to consume that contract.

The bounded production path is:

```text
selected typed tcrv_rvv computed-mask strided-memory body
  -> RVV plugin-owned computed-mask memory family plan
  -> RVV-owned materialization, memory operand-binding, route-control, and statement-plan facts
  -> provider-owned computed-mask strided-memory route validation contract
  -> target artifact validation as a consume-only client
```

This task does not add a new memory route or change emitted runtime semantics.
It tightens the validation boundary so target code no longer reconstructs the
computed-mask strided store/load-unit-store route payload from scattered
target-local helpers, route ids, C strings, fixture names, descriptor residue,
or mirror metadata.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask strided-memory route-family provider contract extraction`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `5a9c6489 chore(task): archive 06-04-stage2-rvv-vector-reduction-executable-closeout`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` require the RVV authority
  chain to remain selected `tcrv.exec` envelope -> typed low-level
  `tcrv_rvv` body -> RVV plugin-owned realization/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* `.trellis/spec/lowering-runtime/emitc-route.md` already defines the
  computed-mask strided memory fact surface for
  `computed_masked_strided_store` and
  `computed_masked_strided_load_unit_store`; this task implements the
  next contract layer around that existing surface.
* The previous strided-memory archive completed provider-owned fact surfaces
  for plain and computed-mask strided routes and target validation consumption
  of those facts.
* Current code has `RVVComputedMaskStridedMemoryRouteFacts` and
  `getRVVComputedMaskStridedMemoryRouteFacts(...)` in the RVV provider
  surface.
* Current code has adjacent validation contract patterns for base memory,
  segment2 memory, compare/select, MAcc, standalone reduction, and vector
  reduction.
* Current target artifact validation still contains a large
  `validateRVVComputedMaskStridedMemoryCanonicalProviderFacts(...)` path and
  computed-mask memory candidate mirror checks that compare directly against
  `RVVComputedMaskStridedMemoryRouteFacts` or `description` fields instead of
  a single provider-owned validation/mirror contract.

## Requirements

* Add a provider-owned computed-mask strided-memory route validation contract
  for `ComputedMaskStridedStore` and `ComputedMaskStridedLoadUnitStore`.
* The contract must cover operation, memory form, element/config facts,
  runtime AVL/VL control, runtime ABI order and parameter roles, input/output
  buffer roles, compare-produced mask facts, mask/tail policy, inactive-lane
  and passthrough contracts, source/destination memory forms, source or
  destination stride binding, stride C type and byte-stride unit, route
  operand binding plan and summary, header/type facts, intrinsic leaves,
  statement-plan step expectations, target leaf profile, and
  `provider_supported_mirror`.
* The contract must fail closed when a rebuilt provider route description has
  missing, stale, cross-family, or cross-load/store facts, including stale
  indexed, segment2, unit-only, scalar-splat, arithmetic, descriptor,
  source-export/direct-C, or legacy i32 route-authority residue.
* Target artifact validation must consume the computed-mask strided-memory
  contract before validating rebuilt route headers, type mappings, ABI
  mappings, statement plans, and candidate metadata mirrors.
* Candidate metadata mirror validation for computed-mask strided routes must
  consume a provider-owned mirror contract instead of reusing target-local
  expected field helpers.
* Common EmitC/export remains neutral. It may carry provider-built payloads
  and mirror metadata unchanged, but must not infer mask, stride, load/store
  direction, header/type, intrinsic, policy, or support facts.

## Acceptance Criteria

* [x] `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes a
      `RVVComputedMaskStridedMemoryRouteValidationContract` and accessor whose
      shape matches the existing provider contract idiom.
* [x] RVV provider implementation builds the contract from
      `RVVComputedMaskStridedMemoryRouteFacts` and the rebuilt route
      description, including runtime ABI parameters, required headers, type
      mappings, and statement-plan count expectations.
* [x] Target artifact provider-fact validation for computed-mask strided
      store/load-unit-store consumes the provider validation contract rather
      than directly consuming `RVVComputedMaskStridedMemoryRouteFacts` in
      target-local checks.
* [x] Target candidate mirror validation for computed-mask strided
      store/load-unit-store consumes a provider-owned metadata mirror contract
      and rejects stale/cross-family mirrors.
* [x] Focused target C++ coverage proves stale computed-mask strided contract
      fields fail closed, including provider mirror, target profile,
      header/type, route operand binding, runtime ABI, mask facts,
      stride source/unit, and load/store cross-contamination.
* [x] Focused generated-bundle dry-run/lit filters for computed-mask strided
      store and computed-mask strided load-unit-store still pass.
* [x] Focused C++ targets
      `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` pass if provider/plugin code is
      touched.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on legacy i32, route-id, artifact-name, descriptor,
      source-front-door/source-export/direct-C, common-EmitC, exact-intrinsic,
      or mirror-only authority.
* [x] `git diff --check` passes.
* [x] Trellis task status, context, journal, archive state, and commit are
      truthful.

## Technical Approach

1. Mirror the existing base-memory/segment2 route validation contract pattern
   for computed-mask strided memory in the RVV provider header and planning
   implementation.
2. Populate the contract from the canonical
   `RVVComputedMaskStridedMemoryRouteFacts` accessor and the rebuilt
   `RVVSelectedBodyEmitCRouteDescription` fields that are provider payload
   details rather than canonical route facts.
3. Add a provider-owned computed-mask strided metadata mirror contract set
   covering non-empty expected mirrors and stale empty mirrors.
4. Replace target-local computed-mask strided provider-fact validation with a
   contract consumer that validates operation, route id, payload facts,
   runtime ABI parameters, headers, type mappings, ABI mappings, route
   statement plan counts, byte-stride contract, and stale indexed/segment2
   residue.
5. Rewire candidate mirror validation for these two routes through the
   provider-owned metadata mirror contract while preserving existing
   compare/select and unit/indexed computed-mask behavior.
6. Add focused C++ mutations in the existing target artifact test where the
   production validator is already exercised.
7. Run focused build/test/lit checks, old-authority scan, `git diff --check`,
   validate task context, archive, and commit.

## Evidence Plan

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* From `build/test`, focused lit filters for computed-mask strided memory,
  including explicit/pre-realized selected-body target artifact fixtures and
  generated-bundle dry-runs for:
  * `computed-masked-strided-store`
  * `computed-masked-strided-load`
  * `pre-realized-computed-masked-strided-store`
  * `pre-realized-computed-masked-strided-load`
* Bounded old-authority scan over touched implementation/test/task files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-strided-memory-provider-contract`

## Out Of Scope

* No new memory operations.
* No computed-mask indexed expansion except incidental shared helper reuse.
* No segment2, base-memory expansion, compare/select expansion, MAcc,
  reduction, conversion, dtype/LMUL clone batches, high-level Linalg/Vector
  frontend lowering, source-front-door routes, dashboards, broad smoke
  matrices, or artifact-only evidence task.
* No weakening of vector-reduction/runtime-scalar/base-memory contracts.
* No movement of RVV semantics into common EmitC/export or target-local
  reconstruction.
* No new `ssh rvv` runtime claim unless emitted C, runtime ABI, mask/stride
  behavior, correctness, or performance changes.

## Definition Of Done

* The two computed-mask strided-memory routes use a provider-owned route
  validation contract and target artifact validation consumes it as a
  client.
* Focused local checks, lit/generated-bundle dry-runs, bounded old-authority
  scan, and `git diff --check` are recorded.
* No runtime/correctness/performance claim is made without real `ssh rvv`
  evidence; if this remains validation-only, record the no-runtime-change
  rationale.
* The Trellis task is finished/archived and one coherent commit is created if
  all acceptance criteria pass.

## Implementation Results

* Added `RVVComputedMaskStridedMemoryRouteValidationContract` and
  `getRVVComputedMaskStridedMemoryRouteValidationContract(...)` to the RVV
  provider surface.
* Added provider-owned computed-mask strided memory metadata mirror contract
  construction through
  `getRVVComputedMaskStridedMemoryRouteMetadataMirrorContract(...)`.
* Rewired target provider-fact validation for `computed_masked_strided_store`
  and `computed_masked_strided_load_unit_store` to consume the validation
  contract for route payload, headers, type mappings, ABI mappings, runtime
  byte-stride roles, mask/tail facts, binding summaries, and statement-plan
  shape.
* Rewired target candidate mirror validation for the two computed-mask strided
  routes to consume the provider-owned mirror contract before accepting
  metadata mirrors.
* Added focused C++ negative coverage for stale provider mirror, binding
  summary, header facts, runtime stride ABI role, mask source, stale candidate
  mirror values, and stale base-memory mirror residue.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the durable
  computed-mask strided memory route validation contract and test/error
  matrix.

## Evidence Results

* Passed:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j 16`.
* Passed: `rtk build/bin/tianchenrv-target-artifact-export-test`.
* Passed: `rtk build/bin/tianchenrv-rvv-extension-plugin-test`.
* Passed focused lit:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-computed-masked-strided-(store|load)|rvv-generated-bundle-abi-e2e-(pre-realized-)?computed-masked-strided-(store|load)-dry-run'`
  from `build/test`: 5 passed, 472 excluded.
* Passed direct load/unit-store generated-bundle dry-run:
  `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-04-computed-mask-strided-provider-contract/load-dry --run-id pre-realized-computed-mask-strided-load --overwrite --op-kind computed_masked_strided_load_unit_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --stride-bytes 4 --stride-bytes 8 --stride-bytes 12 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`.
* Passed: `rtk git diff --check`.
* Passed:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-strided-memory-provider-contract`.
* Bounded added-line old-authority scan over touched code/spec/task files found
  only contract route-id validation fields, provider mirror fields, and
  negative guardrail text; it found no newly added positive legacy i32,
  source-front-door/source-export/direct-C, descriptor, common-EmitC, or
  mirror-only authority path.
* `ssh rvv` was not run. This task changed provider/target validation ownership
  only; it did not change emitted C, runtime ABI shape, mask/stride behavior,
  correctness behavior, or performance behavior.
