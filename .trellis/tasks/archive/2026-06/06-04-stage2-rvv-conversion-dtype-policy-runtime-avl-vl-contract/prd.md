# Stage2 RVV conversion dtype-policy runtime AVL/VL contract migration

## Direction

Migrate the selected-body RVV conversion dtype-policy route-family validation
contract so target artifact validation consumes the provider-owned
`RVVRuntimeAVLVLSelectedBoundaryContract` before accepting widening-conversion
route payloads, runtime/control facts, source/result dtype policy, headers,
type mappings, ABI mappings, statement-plan facts, and metadata mirrors.

## Module Goal

Make runtime `n` / AVL / VL authority flow through the RVV plugin-owned
selected-boundary contract for widening conversion:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv widening-conversion body
  -> RVVRuntimeAVLVLSelectedBoundaryContract embedded in
     RVVConversionDtypePolicyRouteValidationContract
  -> provider-built TCRVEmitCLowerableRoute setvl/loop/conversion statements
  -> target artifact validation consumes that contract
  -> common EmitC materialization remains neutral
```

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean.
* Initial `git log --oneline -8` started at
  `2981157c rvv: consume runtime AVL VL contract for widening dot`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires RVV route authority to flow from selected
  `tcrv.exec` RVV variant to typed low-level `tcrv_rvv` body, RVV plugin-owned
  legality/realization/provider output, common `TCRVEmitCLowerableRoute`, and
  then target artifact validation. Route ids, artifact metadata, ABI strings,
  C snippets, tests, descriptors, or mirrors cannot authorize runtime control
  or dtype/config semantics.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines conversion
  dtype-policy as a route-family owner boundary. Widening conversion owns
  source/result dtype, source/result SEW/LMUL, conversion relation, source
  load, conversion intrinsic, store, runtime ABI, route operand bindings, and
  statement-plan facts.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` and requires promoted route-family
  validation contracts to embed it instead of duplicating target-side runtime
  checks.
* Archived segment2, base-memory, compare/select, standalone reduction, MAcc,
  and widening-dot runtime AVL/VL tasks show the established migration pattern:
  add the embedded runtime contract, populate it through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`, validate it before
  family payload checks, and make statement-plan validation read runtime
  names and the runtime `n` ABI parameter from the embedded contract.
* Live inspection shows `RVVConversionDtypePolicyRouteValidationContract`
  already has `configContractID`, but the conversion builder does not populate
  it and the contract does not embed `RVVRuntimeAVLVLSelectedBoundaryContract`.
* Live target validation already has
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`, but conversion
  dtype-policy validation currently checks runtime ABI, setvl, VL type,
  full-chunk VL, loop VL, loop induction, remaining AVL expression, and
  pointer advancement through conversion-local fields and route statements.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVConversionDtypePolicyRouteValidationContract`.
* Populate `configContractID` and the embedded runtime contract in the
  conversion dtype-policy contract builder. The runtime contract must be built
  with provider-derived selected result SEW/LMUL, tail/mask policy,
  selected config contract id, canonical conversion setvl intrinsic, VL C type,
  runtime ABI order, provider-owned runtime ABI parameters, and the conversion
  consumer label.
* Update conversion target provider-fact validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before accepting
  runtime ABI facts, source/result dtype policy, route payload, headers,
  type mappings, ABI mappings, conversion statement-plan facts, and mirrors.
* Make conversion statement-plan validation consume the embedded runtime
  contract for runtime `n`, pre-loop setvl, loop setvl, full-chunk VL, loop VL,
  loop induction, loop upper bound, remaining AVL expression, selected-body
  provenance, pointer advancement, and runtime ABI role/order checks.
* Fail closed for missing, stale, or mismatched runtime AVL/VL facts:
  runtime AVL source, runtime VL contract id, selected `with_vl` scope,
  setvl callee, VL C type, full-chunk VL, loop VL, loop induction, runtime
  `n` ABI role/order/ownership, remaining AVL metadata, pointer advancement
  metadata, and conversion statement-plan facts.
* Preserve conversion-specific validation for source/result element types,
  source/result SEW/LMUL, conversion kind/relation, source/destination memory
  forms, source load, conversion intrinsic, store intrinsic, headers, type
  mappings, route operand binding plan/summary, target leaf profile, provider
  support mirror, and stale non-conversion family facts.
* Keep common EmitC/export neutral. Do not infer runtime control or conversion
  semantics from route ids, artifact names, manifests, descriptor residue,
  C strings, tests, status fields, exact intrinsic spelling, or metadata
  mirrors.

## Acceptance Criteria

* [x] `RVVConversionDtypePolicyRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The conversion contract builder populates `configContractID` and derives
      the embedded runtime contract through
      `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`.
* [x] Conversion target artifact validation consumes the embedded runtime
      AVL/VL contract before accepting runtime ABI, source/result dtype policy,
      route payload, header/type mapping, statement-plan, conversion relation,
      memory-form, and mirror facts.
* [x] Conversion statement-plan validation reads setvl, VL C type,
      full-chunk VL, loop VL, loop induction, remaining AVL, pointer
      advancement, loop bounds/steps, and runtime `n` ABI checks from the
      embedded runtime contract.
* [x] Positive target C++ coverage asserts the embedded runtime AVL/VL
      contract mirrors rebuilt `widen_i16_to_i32` and `widen_i32_to_i64`
      route facts.
* [x] Negative target C++ coverage rejects stale runtime AVL source, missing
      runtime VL contract, stale selected `with_vl` scope, stale setvl callee,
      stale VL C type, stale full-chunk VL, stale loop VL, stale loop
      induction, stale runtime `n` ABI role, stale remaining AVL metadata, and
      stale pointer advancement metadata.
* [x] Focused provider/target C++ checks pass:
      `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test`.
* [x] Focused conversion generated-bundle or lit dry-run coverage remains
      green.
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
* Run focused conversion generated-bundle/lit dry-run after locating the live
  filter.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-conversion-dtype-policy-runtime-avl-vl-contract`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order, runtime counts, statement ordering, conversion result behavior, harness
behavior, or correctness semantics. The intended diff is contract/validation/
test/spec-only. If that remains true, state the no-runtime-change rationale and
cite existing executable evidence for unchanged conversion paths when present
in archived tasks.

## Out Of Scope

* No new conversion operations, dtype/LMUL clone batches, widening-dot work,
  MAcc work, standalone reduction coverage, segment2, compare/select,
  source-front-door routes, high-level frontend lowering, dashboards, broad
  smoke matrices, or report-only evidence.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, manifests,
  emission-plan mirrors, descriptor residue, C strings, scripts, tests, exact
  intrinsic spelling, or metadata mirrors.
* No cross-family migration beyond conversion dtype-policy in this round.

## Definition Of Done

Conversion dtype-policy target route validation accepts only provider-built
widening-conversion routes whose embedded runtime AVL/VL selected-boundary
contract matches the rebuilt route description and statement payload. Focused
C++/lit checks pass, old-authority scan is clean, the task is finished and
archived, and a coherent commit records the production diff plus PRD evidence.

## Implementation Results

* Added `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVConversionDtypePolicyRouteValidationContract`.
* Rewired
  `populateRVVConversionDtypePolicyValidationContract(...)` to populate
  `configContractID` from the rebuilt selected-body route description and to
  build the embedded runtime AVL/VL contract through
  `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`.
* The embedded runtime contract is built with provider-owned conversion result
  SEW/LMUL, tail/mask policy, selected config contract id, canonical conversion
  setvl intrinsic, `size_t` VL C type, runtime ABI order, and provider-owned
  `lhs,out,n` ABI parameters.
* Rewired conversion target provider-fact validation so
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` runs before runtime
  ABI, route payload, source/result dtype-policy, conversion relation,
  memory-form, headers/type mappings, statement-plan, and mirror acceptance.
* Rewired conversion statement-plan validation to consume the embedded runtime
  contract for runtime `n`, pre-loop setvl, loop setvl, full-chunk VL, loop VL,
  loop induction, loop upper bound, remaining AVL expression, selected-body
  provenance, and pointer advancement.
* Added positive target C++ coverage asserting embedded runtime AVL/VL
  contract fields for both `widen_i16_to_i32` and `widen_i32_to_i64`.
* Added fail-closed conversion target coverage for stale runtime AVL source,
  missing runtime VL contract, stale selected `with_vl` scope, stale setvl
  callee, stale VL C type, stale full-chunk VL, stale loop VL, stale loop
  induction, stale runtime `n` ABI role, stale remaining AVL metadata, and
  stale pointer advancement metadata.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` so the Widening
  Conversion Route Validation Contract section records the embedded runtime
  AVL/VL selected-boundary contract and target consumer order.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16
```

Result: passed. Existing warnings about unrelated unused functions and
switch-enum coverage remained, but no build errors remained.

After updating target test expectations for the new first rejection point:

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16
```

Result: passed.

### C++ Tests

```text
rtk build/bin/tianchenrv-target-artifact-export-test
```

Result: failed twice during self-repair because the embedded runtime AVL/VL
contract now rejects stale destination LMUL / tail policy and cross-variant
result SEW before the later conversion dtype-policy diagnostics. The affected
expectations were updated to the new first rejection point.

Rerun result: passed.

```text
rtk build/bin/tianchenrv-rvv-extension-plugin-test
```

Result: passed with `RVV extension plugin smoke test passed`.

### Focused Lit / Generated-Bundle Dry-Run

From `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -av . --filter 'widen.*conversion|widen_i16|widen_i32'
```

Result: passed 2 focused tests:

* `Dialect/RVV/generic-widening-conversion-dataflow.mlir`
* `Transforms/LoweringBoundary/rvv-pre-realized-widening-conversion-negative.mlir`

Broader widening route dry-run/regression filter:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -av . --filter 'widen'
```

Result: passed 39 focused widening tests, including generated-bundle
fail-closed coverage for direct pre-realized `widen_i16_to_i32` and
`widen_i32_to_i64` routes plus target artifact conversion fixtures.

### Old-Authority Scan

Added-line scan:

```text
rtk git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp .trellis/spec/lowering-runtime/emitc-route.md | rtk rg -n '^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only)'
```

Result: no matches.

### Whitespace And Trellis

```text
rtk git diff --check
```

Result: passed.

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-conversion-dtype-policy-runtime-avl-vl-contract
```

Result: passed.

## SSH RVV

Not rerun. This round changed provider/target validation contracts, target C++
tests, Trellis records, and spec text only. It did not change emitted C/C++,
runtime ABI order, runtime counts, statement ordering, conversion load/convert/
store behavior, harness behavior, or correctness/performance semantics.

## Spec Update

Updated `.trellis/spec/lowering-runtime/emitc-route.md` because this task
changed a concrete cross-layer contract. The Widening Conversion Route
Validation Contract section now includes the embedded
`RVVRuntimeAVLVLSelectedBoundaryContract`, target consumer order, fail-closed
runtime AVL/VL diagnostics, and required statement-plan ownership rule.

## Self-Repair

* Initial target C++ test run failed because stale destination LMUL and stale
  tail-policy tests still expected later conversion dtype-policy diagnostics.
  The embedded runtime AVL/VL contract now correctly rejects those fields first,
  so the expected fragments were updated.
* A second target C++ test run failed because cross-variant stale-fact tests
  still expected conversion-kind diagnostics. The embedded runtime contract now
  rejects the stale selected result SEW first, so those expected fragments were
  updated.
