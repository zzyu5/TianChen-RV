# Stage2 RVV runtime-scalar splat-store runtime AVL/VL contract migration

## Direction

Migrate the RVV runtime-scalar splat-store route-family validation contract so
target artifact validation consumes the provider-owned
`RVVRuntimeAVLVLSelectedBoundaryContract` before accepting runtime scalar ABI,
setvl/VL control, splat/store statement-plan facts, headers, type mappings,
ABI mappings, route payload mirrors, and stale mirror rejection.

## Module Goal

Make runtime `n` / AVL / VL authority flow through the RVV plugin-owned
selected-boundary contract for runtime scalar splat-store:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv runtime-scalar splat-store body
  -> RVVRuntimeAVLVLSelectedBoundaryContract embedded in
     RVVRuntimeScalarSplatStoreRouteValidationContract
  -> provider-built TCRVEmitCLowerableRoute setvl/splat/store statements
  -> target artifact validation consumes that contract
  -> common EmitC materialization remains neutral
```

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean.
* Initial `git log --oneline -8` started at
  `98472af2 rvv: consume runtime AVL VL contract for conversion`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain to flow from
  selected `tcrv.exec` RVV variant to typed low-level `tcrv_rvv` body, RVV
  plugin-owned legality/realization/provider output,
  `TCRVEmitCLowerableRoute`, common EmitC materialization, and target artifact
  validation. Route ids, artifact names, ABI strings, C snippets, tests,
  descriptors, status fields, and mirrors cannot authorize runtime control.
* `.trellis/spec/extension-plugins/rvv-plugin.md` keeps runtime ABI values in
  `tcrv.exec` as role declarations only; the selected typed `tcrv_rvv` body
  must import and consume runtime values through `setvl`, `with_vl`, splat,
  store, and provider route facts.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` and requires promoted route-family
  validation contracts to embed it instead of duplicating target-side runtime
  checks.
* Archived segment2, base-memory, compare/select, standalone reduction, MAcc,
  widening-dot, and conversion runtime AVL/VL tasks show the established
  migration pattern: add the embedded runtime contract, populate it through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`, validate it before family
  payload checks, and make statement-plan validation read runtime names and
  the runtime `n` ABI parameter from the embedded contract.
* Live inspection shows `RVVRuntimeScalarSplatStoreRouteValidationContract`
  carries `configContractID`, runtime control id, ABI order, setvl/VL names,
  scalar/vector type facts, splat/store leaves, and statement counts, but does
  not embed `RVVRuntimeAVLVLSelectedBoundaryContract`.
* Live target validation already provides
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`, but runtime-scalar
  splat-store validation currently checks runtime ABI, runtime control plan,
  setvl intrinsic, VL C type, full-chunk VL, loop VL, loop induction, and
  statement-plan facts through splat-store-local fields.
* Existing runtime-scalar splat-store tests are concentrated in
  `test/Target/TargetArtifactExportTest.cpp`, plus focused target/export,
  generated-bundle dry-run, EmitC, and lowering-boundary lit fixtures.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVRuntimeScalarSplatStoreRouteValidationContract`.
* Populate the embedded runtime contract in the splat-store contract builder
  through `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` using
  provider-derived SEW/LMUL, tail/mask policy, selected config contract id,
  canonical setvl intrinsic, VL C type, runtime ABI order,
  provider-owned `rhs_scalar,out,n` ABI parameters, and the splat-store
  consumer label.
* Update runtime-scalar splat-store target provider-fact validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before accepting
  runtime ABI facts, runtime/control mirrors, scalar ABI facts, vector/type
  mappings, setvl/splat/store leaves, headers, route operand bindings,
  statement-plan facts, and metadata mirrors.
* Make runtime-scalar splat-store statement-plan validation consume the
  embedded runtime contract for runtime `n`, pre-loop setvl, loop setvl,
  full-chunk VL, loop VL, loop induction, loop upper bound, remaining AVL,
  selected-body provenance, pointer advancement, and runtime ABI role/order
  checks.
* Fail closed for missing, stale, or mismatched runtime AVL/VL facts:
  runtime AVL source, runtime VL contract id, selected `with_vl` scope,
  setvl callee, VL C type, full-chunk VL, loop VL, loop induction, runtime
  `n` ABI role/order/ownership, remaining AVL metadata, pointer advancement
  metadata, and splat-store statement-plan facts.
* Preserve splat-store-specific validation for runtime scalar ABI, output
  store ABI, element type, SEW/LMUL, policy, scalar/vector C types, vector
  type, splat intrinsic, store intrinsic, result name, route operand binding
  plan/summary, target leaf profile, provider support mirror, required
  headers, C type mappings, and stale non-splat-store family facts.
* Keep common EmitC/export neutral. Do not infer runtime control or
  splat-store semantics from route ids, artifact names, manifests, descriptor
  residue, C strings, tests, status fields, exact intrinsic spelling, or
  metadata mirrors.

## Acceptance Criteria

* [x] `RVVRuntimeScalarSplatStoreRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The splat-store contract builder derives the embedded runtime contract
      through `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`.
* [x] Runtime-scalar splat-store target artifact validation consumes the
      embedded runtime AVL/VL contract before accepting runtime ABI,
      setvl/pre-loop/loop facts, scalar ABI facts, vector/type mappings,
      splat/store intrinsics, headers, statement-plan facts, and mirrors.
* [x] Runtime-scalar splat-store statement-plan validation reads setvl,
      VL C type, full-chunk VL, loop VL, loop induction, remaining AVL,
      pointer advancement, loop bounds/steps, selected-body provenance, and
      runtime `n` ABI checks from the embedded runtime contract.
* [x] Positive target C++ coverage asserts the embedded runtime AVL/VL
      contract mirrors the rebuilt runtime-scalar splat-store route facts.
* [x] Negative target C++ coverage rejects stale runtime AVL source, missing
      runtime VL contract, stale selected `with_vl` scope, stale setvl callee,
      stale VL C type, stale full-chunk VL, stale loop VL, stale loop
      induction, stale runtime `n` ABI role, stale remaining AVL metadata,
      and stale pointer advancement metadata.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test`.
* [x] Focused runtime-scalar splat-store generated-bundle/lit dry-run
      coverage remains green.
* [x] Added-line old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door
      authority, mirror-only authority, route-id/artifact-name authority, or
      exact `__riscv_*_i32m1` route authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Evidence Plan

* Build focused C++ tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused runtime-scalar splat-store lit/generated-bundle coverage after
  locating the live filter.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-runtime-scalar-splat-store-avl-vl-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order, runtime counts, statement ordering, splat/store behavior, harness
behavior, or correctness semantics. The intended diff is contract/validation/
test/spec-only. If that remains true, state the no-runtime-change rationale and
cite existing executable evidence for unchanged runtime-scalar splat-store
paths when present in archived tasks.

## Out Of Scope

* No new splat/store operations, vector reduction work, elementwise arithmetic
  work, masked/indexed/strided memory work, conversion work, MAcc or
  widening-dot work, source-front-door routes, high-level frontend lowering,
  dashboards, broad smoke matrices, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, tests, exact
  intrinsic spelling, or metadata mirrors.
* No cross-family migration beyond runtime-scalar splat-store in this round.

## Definition Of Done

Runtime-scalar splat-store target route validation accepts only provider-built
routes whose embedded runtime AVL/VL selected-boundary contract matches the
rebuilt route description and statement payload. Focused C++/lit checks pass,
old-authority scan is clean, the task is finished and archived, and a coherent
commit records the production diff plus PRD evidence.

## Implementation Results

* Added `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVRuntimeScalarSplatStoreRouteValidationContract`.
* Rewired
  `buildRVVRuntimeScalarSplatStoreRouteValidationContract(...)` to build the
  embedded runtime AVL/VL contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` after the provider-owned
  `rhs_scalar,out,n` ABI parameters are known.
* The embedded runtime contract is derived from splat-store provider facts:
  SEW32, LMUL m1, tail/mask policy, config contract id, canonical setvl
  intrinsic, `size_t` VL C type, runtime ABI order, and provider-owned runtime
  ABI parameters.
* Rewired target runtime-scalar splat-store provider-fact validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before local
  runtime control, ABI, type, intrinsic, statement-plan, and mirror acceptance.
* Rewired runtime-scalar splat-store statement-plan validation to consume the
  embedded runtime contract for runtime `n`, pre-loop setvl, loop setvl,
  full-chunk VL, loop VL, loop induction, loop bounds/step, remaining AVL,
  pointer advancement, selected-body provenance, and runtime AVL ABI facts.
* Added positive target C++ coverage asserting the embedded runtime AVL/VL
  selected-boundary contract mirrors the rebuilt runtime-scalar splat-store
  route facts.
* Added fail-closed target C++ coverage for stale runtime AVL source, missing
  runtime VL contract, stale selected `with_vl` scope, stale setvl callee,
  stale VL C type, stale full-chunk VL, stale loop VL, stale loop induction,
  stale runtime `n` ABI role, stale remaining AVL metadata, and stale pointer
  advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` so the
  Runtime Scalar Splat-Store Route Validation Contract section records the
  embedded runtime AVL/VL selected-boundary contract, target consumer order,
  and fail-closed diagnostics.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings were emitted in target/test code, but no
build errors occurred.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: passed with no output.

```text
rtk build/bin/tianchenrv-rvv-extension-plugin-test
```

Result: passed with `RVV extension plugin smoke test passed`.

### Focused Lit / Generated-Bundle Dry-Run

From `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter runtime-scalar-splat-store
```

Result: passed 6 focused tests. Lit discovered 477 tests, excluded 471 via the
filter, and passed all 6 selected runtime-scalar splat-store tests.

### Old-Authority Scan

Added-line scan over touched production, test, spec, and task files:

```text
rtk bash -lc 'git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp .trellis/spec/lowering-runtime/emitc-route.md .trellis/tasks/06-04-stage2-rvv-runtime-scalar-splat-store-avl-vl-contract | rg -n "^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)" || true'
```

Result: no matches.

### Whitespace And Trellis

```text
rtk git diff --check
```

Result: passed.

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-runtime-scalar-splat-store-avl-vl-contract
```

Result: passed.

## SSH RVV

Not rerun. This round changed provider/target validation contracts, target C++
tests, Trellis records, and spec text only. It did not change emitted C/C++,
runtime ABI order, runtime counts, statement ordering, splat/store behavior,
harness behavior, or correctness/performance semantics.

## Spec Update

Updated `.trellis/spec/lowering-runtime/emitc-route.md` because this task
changed a concrete cross-layer route validation contract. The runtime scalar
splat-store section now includes the embedded
`RVVRuntimeAVLVLSelectedBoundaryContract`, target consumer order,
fail-closed runtime AVL/VL diagnostics, and statement-plan ownership rule.

## Self-Repair

No failed verification step required code repair in this round. The focused
build, C++ tests, lit filter, Trellis context validation, whitespace check, and
old-authority scan all passed on first run after the implementation patch.
