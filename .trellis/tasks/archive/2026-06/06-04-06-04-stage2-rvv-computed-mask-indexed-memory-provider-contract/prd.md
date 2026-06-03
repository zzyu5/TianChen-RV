# Stage2 RVV computed-mask indexed-memory route-family provider contract extraction

## Goal

Extract the existing computed-mask indexed-memory route facts into a
provider-owned route validation contract and rewire target artifact validation
for the current `computed_masked_indexed_gather_load_unit_store` and
`computed_masked_indexed_scatter_store_unit_load` family to consume that
contract.

The bounded production path is:

```text
selected typed tcrv_rvv computed-mask indexed-memory body
  -> RVV plugin-owned computed-mask indexed memory family facts
  -> RVV-owned materialization, mask/index operand binding, route-control, and statement-plan facts
  -> provider-owned computed-mask indexed-memory route validation contract
  -> target artifact validation as a consume-only client
```

This task does not add a new memory route or change emitted runtime semantics.
It tightens the validation boundary so target code no longer reconstructs the
computed-mask indexed gather/scatter route payload from scattered target-local
helpers, route ids, C strings, fixture names, descriptor residue, or mirror
metadata.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask indexed-memory route-family provider contract extraction`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean before creating this Trellis task.
* Initial `git log --oneline -8` starts at
  `d84d397d rvv: extract computed-mask strided memory route contract`.
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
  computed-mask indexed memory fact surface for
  `computed_masked_indexed_gather_load_unit_store` and
  `computed_masked_indexed_scatter_store_unit_load`.
* The immediately preceding commit `d84d397d` completed the same provider
  validation contract extraction pattern for computed-mask strided store/load
  routes.
* The archived computed-mask indexed tasks completed provider-owned fact
  surface consumption in target validation, but this brief requires the next
  contract layer: target artifact validation must consume a provider-built
  validation contract rather than directly rebuilding expectations from
  `RVVComputedMaskIndexedMemoryRouteFacts`.
* The in-scope computed-mask indexed facts must cover compare-produced mask,
  index buffer, gather/scatter memory polarity, runtime `n`/AVL/VL binding,
  route operand bindings, header/type/intrinsic/profile facts, dtype/config
  relation, statement-plan expectations, `provider_supported_mirror`, and
  stale mirror rejection.

## Requirements

* Add a provider-owned computed-mask indexed-memory route validation contract
  for `ComputedMaskIndexedGatherLoadUnitStore` and
  `ComputedMaskIndexedScatterStoreUnitLoad`.
* The contract must cover operation, memory form, element/config facts,
  runtime AVL/VL control, runtime ABI order and parameter roles, input/output
  buffer roles, compare-produced mask facts, mask/tail policy,
  inactive-lane and passthrough/no-write contracts, index source/EEW/offset
  unit/uniqueness, source/destination memory forms, indexed data/destination
  memory forms, route operand binding plan and summary, header/type facts,
  intrinsic leaves, statement-plan step expectations, target leaf profile,
  and `provider_supported_mirror`.
* The contract must fail closed when a rebuilt provider route description has
  missing, stale, cross-family, or cross-gather/scatter facts, including stale
  plain indexed/base-memory, computed-mask unit-only, computed-mask strided,
  segment2, scalar-splat, arithmetic, descriptor, source-export/direct-C, or
  legacy i32 route-authority residue.
* Target artifact validation must consume the computed-mask indexed-memory
  contract before validating rebuilt route headers, type mappings, ABI
  mappings, statement plans, and candidate metadata mirrors.
* Candidate metadata mirror validation for computed-mask indexed routes must
  consume a provider-owned mirror contract after route payload validation
  succeeds and must reject stale/cross-family mirrors.
* Common EmitC/export remains neutral. It may carry provider-built payloads
  and mirror metadata unchanged, but must not infer mask, index, load/store
  direction, header/type, intrinsic, policy, or support facts.

## Acceptance Criteria

* [x] `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes a
      `RVVComputedMaskIndexedMemoryRouteValidationContract` and accessor whose
      shape matches the existing provider contract idiom.
* [x] RVV provider implementation builds the contract from
      `RVVComputedMaskIndexedMemoryRouteFacts` and the rebuilt route
      description, including runtime ABI parameters, required headers, type
      mappings, and statement-plan count expectations.
* [x] Target artifact provider-fact validation for computed-mask indexed
      gather/scatter consumes the provider validation contract rather than
      directly consuming `RVVComputedMaskIndexedMemoryRouteFacts` in
      target-local checks.
* [x] Target candidate mirror validation for computed-mask indexed
      gather/scatter consumes a provider-owned metadata mirror contract and
      rejects stale/cross-family mirrors.
* [x] Focused target C++ coverage proves stale computed-mask indexed contract
      fields fail closed, including provider mirror, target profile,
      header/type, route operand binding, runtime ABI, mask facts, index
      source/EEW/offset/uniqueness, intrinsic leaves, statement counts, and
      gather/scatter cross-contamination.
* [x] Focused generated-bundle dry-run/lit filters for computed-mask indexed
      gather/scatter explicit and pre-realized fixtures still pass.
* [x] Focused C++ targets `tianchenrv-target-artifact-export-test` and
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

1. Mirror the computed-mask strided route validation contract pattern for
   computed-mask indexed memory in the RVV provider header and planning
   implementation.
2. Populate the contract from the canonical
   `RVVComputedMaskIndexedMemoryRouteFacts` accessor and rebuilt
   `RVVSelectedBodyEmitCRouteDescription` payload details that are provider
   route details rather than canonical route facts.
3. Add or reuse a provider-owned computed-mask indexed metadata mirror
   contract set covering non-empty expected mirrors and stale mirror keys.
4. Replace target-local computed-mask indexed provider-fact validation with a
   contract consumer that validates operation, route id, payload facts,
   runtime ABI parameters, headers, type mappings, ABI mappings, route
   statement plan counts, index/mask contracts, intrinsic leaves, and stale
   base/strided/segment residue.
5. Rewire candidate mirror validation for these two routes through the
   provider-owned metadata mirror contract while preserving existing plain
   indexed and other memory route behavior.
6. Add focused C++ mutations in the existing target artifact test where the
   production validator is already exercised.
7. Run focused build/test/lit checks, old-authority scan, `git diff --check`,
   validate task context, archive, and commit.

## Evidence Plan

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* From `build/test`, focused lit filters for computed-mask indexed memory,
  including explicit/pre-realized selected-body target artifact fixtures and
  generated-bundle dry-runs for:
  * `computed-masked-indexed-gather-load`
  * `computed-masked-indexed-scatter-store`
  * `pre-realized-computed-masked-indexed-gather-load`
  * `pre-realized-computed-masked-indexed-scatter-store`
* Bounded old-authority scan over touched implementation/test/task files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-indexed-memory-provider-contract`

## Out Of Scope

* No new memory operations.
* No strided-memory rewrites beyond shared helper reuse.
* No segment2, base-memory expansion, compare/select expansion, MAcc,
  reduction, conversion, dtype/LMUL clone batches, high-level Linalg/Vector
  frontend lowering, source-front-door routes, dashboards, broad smoke
  matrices, or artifact-only evidence task.
* No weakening of computed-mask strided, vector-reduction, runtime-scalar, or
  base-memory contracts.
* No movement of RVV semantics into common EmitC/export or target-local
  reconstruction.
* No new `ssh rvv` runtime claim unless emitted C, runtime ABI,
  mask/index behavior, correctness, or performance changes.

## Definition Of Done

* The two computed-mask indexed-memory routes use a provider-owned route
  validation contract and target artifact validation consumes it as a client.
* Focused local checks, lit/generated-bundle dry-runs, bounded old-authority
  scan, and `git diff --check` are recorded.
* No runtime/correctness/performance claim is made without real `ssh rvv`
  evidence; if this remains validation-only, record the no-runtime-change
  rationale.
* The Trellis task is finished/archived and one coherent commit is created if
  all acceptance criteria pass.

## Implementation Results

* Added `RVVComputedMaskIndexedMemoryRouteValidationContract` and
  `getRVVComputedMaskIndexedMemoryRouteValidationContract(...)` to the RVV
  provider surface.
* Added provider-owned computed-mask indexed memory metadata mirror contract
  construction through
  `getRVVComputedMaskIndexedMemoryRouteMetadataMirrorContract(...)`.
* Rewired target provider-fact validation for
  `computed_masked_indexed_gather_load_unit_store` and
  `computed_masked_indexed_scatter_store_unit_load` to consume the validation
  contract for route payload, headers, type mappings, ABI mappings, runtime
  ABI roles, mask/tail facts, binding summaries, index source/EEW/offset and
  uniqueness facts, intrinsic leaves, and statement-plan shape.
* Rewired target candidate mirror validation for the two computed-mask indexed
  routes to consume the provider-owned mirror contract before accepting
  metadata mirrors.
* Removed direct `RVVComputedMaskIndexedMemoryRouteFacts` consumption from
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`; target
  validation now reaches computed-mask indexed route facts only through the
  provider-owned validation/mirror contracts.
* Added focused C++ contract self-checks for computed-mask indexed gather and
  scatter, plus updated stale metadata expectations for the new provider
  mirror labels.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the durable
  computed-mask indexed memory route validation contract and test/error
  matrix.

## Evidence Results

* Passed:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`.
  The build emitted only pre-existing switch coverage warnings in
  `test/Target/TargetArtifactExportTest.cpp`.
* Passed:
  `rtk cmake --build build --target tcrv-opt tcrv-translate -j 16`.
* Passed: `rtk build/bin/tianchenrv-target-artifact-export-test`.
* Passed: `rtk build/bin/tianchenrv-rvv-extension-plugin-test`.
* Passed focused lit from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-computed-masked-indexed-(gather-load|scatter-store)|rvv-generated-bundle-abi-e2e-(pre-realized-)?computed-masked-indexed-(gather-load|scatter-store)-dry-run'`;
  8 passed, 469 excluded.
* Passed target-validator boundary scan:
  `rtk rg -n "getRVVComputedMaskIndexedMemoryRouteFacts|getRVVComputedMaskIndexedMemoryFactsForDescription|RVVComputedMaskIndexedMemoryRouteFacts" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`;
  no matches.
* Passed: `rtk git diff --check`.
* Passed:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-computed-mask-indexed-memory-provider-contract`.
* Bounded added-line old-authority scan over touched code/spec/task files found
  only negative guardrail/spec text; it found no newly added positive legacy
  i32, source-front-door/source-export/direct-C, descriptor, common-EmitC, or
  mirror-only authority path.
* `ssh rvv` was not run. This task changed provider/target validation
  ownership only; it did not change emitted C, runtime ABI shape, mask/index
  behavior, correctness behavior, or performance behavior.
