# Stage2 RVV standalone-reduction runtime AVL/VL contract migration

## Goal

Migrate the selected-body RVV standalone reduction / accumulation route-family
validation contract to embed the provider-owned
`RVVRuntimeAVLVLSelectedBoundaryContract`, then require target artifact
validation to consume that contract before accepting standalone reduction route
payloads, runtime ABI facts, header/type facts, statement plans, accumulator
layout, scalar-result boundary facts, computed-mask/runtime-scalar variants,
and metadata mirrors.

This round's bounded consumer slice is the production-active standalone
reduction route family: plain `standalone_reduce_{add,min,max}`,
computed-mask `computed_mask_standalone_reduce_{add,min,max}`, and
runtime-scalar computed-mask
`runtime_scalar_cmp_masked_standalone_reduce_{add,min,max}`.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV standalone-reduction runtime AVL/VL contract migration`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: `ok`, meaning RTK reported no short
  status entries.
* Initial `rtk git log --oneline -8` started at
  `615c5b0a rvv: consume runtime AVL VL contract for compare select`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / realization / provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` says runtime ABI values must
  be explicitly imported into the selected `tcrv_rvv` body and consumed by
  `setvl` / `with_vl` / route provider facts. Standalone reduction is an RVV
  selected-body route-family provider consumer, not a common EmitC semantic
  branch.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` and requires route-family
  validation contracts that consume runtime AVL/VL facts to embed it rather
  than duplicate target-side runtime checks.
* The archived runtime AVL/VL selected-boundary, base-memory, and
  compare/select tasks already established the migration pattern:
  route-family validation contract embeds the runtime AVL/VL contract, then
  target artifact validation consumes it before payload/header/type/ABI/
  statement/mirror acceptance.
* The archived standalone reduction route-contract extraction task already
  introduced `RVVStandaloneReductionRouteValidationContract` and rewired
  standalone reduction target validation to consume provider-owned facts for
  route payload, ABI, header/type, statement-plan, scalar-result, mask, and
  mirror facts.
* Live code inspection shows `RVVStandaloneReductionRouteValidationContract`
  still lacks `RVVRuntimeAVLVLSelectedBoundaryContract`, while the rebuilt
  `RVVSelectedBodyEmitCRouteDescription` already carries `configContractID`,
  `runtimeVLContractID`, runtime AVL source, selected `with_vl` / VL op facts,
  full-chunk VL, loop VL, loop induction, remaining AVL metadata, pointer
  advancement metadata, bounded-slice/multi-VL facts, and runtime ABI
  parameters.
* Live target validation already has
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`, used by promoted
  base-memory, segment2, and compare/select consumers.
* Live standalone reduction target statement-plan validation still reconstructs
  runtime `n`, setvl, full-chunk VL, loop VL, loop induction, and pointer
  expressions from standalone-local contract fields instead of first consuming
  the named runtime AVL/VL selected-boundary contract.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVStandaloneReductionRouteValidationContract`.
* Build the embedded runtime contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` from provider-owned
  standalone reduction facts plus rebuilt selected-body description fields:
  SEW, LMUL, tail/mask policy, config contract id, runtime-VL contract id,
  selected `with_vl` boundary, runtime AVL source, setvl intrinsic, VL C type,
  full-chunk VL, loop VL, loop induction, remaining AVL metadata, pointer
  advancement metadata, bounded-slice/multi-VL support, runtime ABI order, and
  the runtime `n` ABI parameter.
* Update standalone reduction target validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before accepting
  route id, payload, ABI parameter facts, header declarations, type mappings,
  pre-loop statements, loop statements, accumulator/result layout,
  scalar-result runtime boundary, reduction store VL, mask/accumulation facts,
  and candidate metadata mirrors.
* Make standalone reduction statement-plan validation consume the embedded
  runtime contract for provider-expected runtime `n`, pre-loop setvl, loop
  setvl, full-chunk VL, loop VL, loop induction, loop upper bound, remaining
  AVL expression, selected-body provenance, and pointer advancement metadata
  checks.
* Fail closed for missing, stale, or mismatched runtime AVL/VL facts:
  runtime AVL source, runtime VL contract id, selected `with_vl` boundary /
  scope, setvl intrinsic, VL C type, full-chunk VL, loop VL, loop induction,
  runtime `n` ABI role/order/ownership, remaining AVL metadata, pointer
  advancement metadata, and reduction statement-plan facts.
* Keep common EmitC/export neutral. Do not infer runtime control from route
  ids, artifact names, manifests, descriptor residue, C strings, test names,
  status fields, exact intrinsic spelling, or metadata mirrors.
* Do not change generated C/C++, runtime ABI order, runtime counts, statement
  ordering, accumulator behavior, scalar-result behavior, correctness, or
  performance behavior. If any of those change, run real `ssh rvv` evidence.

## Acceptance Criteria

* [x] `RVVStandaloneReductionRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The standalone reduction contract builder derives the embedded runtime
      AVL/VL contract through the shared provider-owned helper rather than
      target-local reconstruction.
* [x] Standalone reduction target artifact validation consumes the runtime
      AVL/VL contract before accepting route payloads, headers, type mappings,
      ABI mappings, statement plans, accumulator/reduction/scalar-result facts,
      mask/accumulation facts, and metadata mirrors.
* [x] Standalone reduction statement-plan validation uses the embedded runtime
      contract for pre-loop setvl, loop setvl, full-chunk VL, loop VL,
      induction, runtime n/AVL, remaining AVL, and pointer advancement checks.
* [x] Negative C++ target coverage rejects stale or missing runtime AVL/VL
      contract facts, including runtime AVL source, runtime VL contract id,
      selected `with_vl` scope, setvl callee, VL C type, full-chunk VL, loop
      VL, loop induction, runtime `n` ABI facts, and pointer advancement
      metadata.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test`.
* [x] Focused lit or generated-bundle dry-run coverage for standalone
      reduction selected-body artifacts remains green.
* [x] Added-line old-authority scan over touched production/test files finds
      no new positive dependency on `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/
      source-front-door authority, mirror-only authority, route-id/artifact-name
      authority, or exact `__riscv_*_i32m1` route authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Evidence Plan

* Build focused C++ tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused lit/generated-bundle dry-run tests for standalone reduction
  selected-body artifacts after locating the exact live filter.
* Run added-line old-authority scan over touched source/test files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-standalone-reduction-runtime-avl-vl-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order, runtime counts, statement ordering, accumulator/scalar-result behavior,
harness behavior, or correctness semantics. The intended diff is
contract/validation/test-only. If that remains true, state the
no-runtime-change rationale and cite existing standalone reduction executable
evidence:

* plain `standalone_reduce_add` evidence from
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-standalone-reduce-add-scalar-artifact-abi/prd.md`;
* computed-mask `computed_mask_standalone_reduce_add` evidence from
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-mask-standalone-reduce-add-artifact-abi/prd.md`;
* runtime-scalar computed-mask `runtime_scalar_cmp_masked_standalone_reduce_add`
  evidence from
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi/prd.md`.

## Out Of Scope

* No new reduction ops, min/max runtime evidence expansion, dtype/LMUL clone
  batch, MAcc, widening-dot, conversion, segment2, compare/select,
  source-front-door routes, high-level frontend lowering, dashboards, broad
  smoke matrices, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, tests, exact
  intrinsic spelling, or metadata mirrors.
* No cross-family migration beyond standalone reduction in this round.

## Definition Of Done

Standalone reduction target route validation accepts only provider-built
standalone reduction routes whose embedded runtime AVL/VL selected-boundary
contract matches the rebuilt route description and statement payload. Focused
C++/lit checks pass, old-authority scan is clean, the task is finished and
archived, and a coherent commit records the production diff plus PRD evidence.

## Implementation Results

* Added `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVStandaloneReductionRouteValidationContract`.
* Rewired
  `populateRVVStandaloneReductionValidationContract(...)` to fill the embedded
  runtime AVL/VL contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after standalone reduction
  runtime ABI parameters are populated.
* Made the standalone reduction contract use provider-derived canonical runtime
  facts for selected SEW/LMUL config id, setvl intrinsic, and VL C type before
  constructing the embedded runtime contract. This prevents stale
  route-description mirrors from bypassing the shared runtime validator.
* Rewired standalone reduction target route validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before rebuilt
  route id, route payload, header/type mapping, ABI mapping, statement-plan,
  accumulator/scalar-result, mask/accumulation, and metadata mirror acceptance.
* Rewired standalone reduction pre-loop and loop statement-plan validation to
  consume the embedded runtime contract for runtime `n`, setvl callee, VL C
  type, full-chunk VL, loop VL, loop induction, loop upper bound, and
  induction-based pointer expressions.
* Added standalone reduction target C++ coverage asserting the embedded runtime
  AVL/VL selected-boundary contract mirrors rebuilt route facts.
* Added fail-closed standalone reduction target coverage for stale runtime AVL
  source, missing runtime VL contract, stale selected `with_vl` scope, stale
  setvl callee, stale VL C type, stale full-chunk VL, stale loop VL, stale
  loop induction, stale runtime `n` ABI role, stale remaining AVL metadata, and
  stale pointer advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the standalone
  reduction runtime contract embedding rule and the provider-derived
  setvl/VL-type/config gotcha.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16
rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings remained in target/test code; no final build
errors remain.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: initially failed the new stale-setvl negative expectation because
standalone reduction copied `description.setVLIntrinsic` / `description.vlCType`
into the embedded runtime contract. The builder was fixed to use canonical
provider-derived runtime facts, then the target test passed.

```text
rtk build/bin/tianchenrv-rvv-extension-plugin-test
```

Result: passed with:

```text
RVV extension plugin smoke test passed
```

### Focused Lit / Generated-Bundle Dry-Run

From `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'standalone-reduce'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 435 (91.19%)
  Passed  :  42 (8.81%)
```

### Old-Authority Scan

Added-line scan over touched production/test files:

```text
rtk bash -lc 'git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)"'
```

Result: no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### Trellis Validation

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-standalone-reduction-runtime-avl-vl-contract
```

Result: passed for `implement.jsonl` and `check.jsonl`.

### SSH RVV

Not rerun. This round changed provider/target validation contracts, target C++
tests, Trellis task files, and one spec note only. It did not change emitted
C/C++, runtime ABI order, runtime counts, statement ordering, accumulator or
scalar-result behavior, harness behavior, correctness, or performance behavior.
Existing executable evidence for unchanged standalone reduction paths remains
the archived plain, computed-mask, and runtime-scalar computed-mask standalone
reduce-add `ssh rvv` evidence cited above.
