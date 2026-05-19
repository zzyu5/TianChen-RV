# Stage2 generic RVV reduction and accumulation route skeleton

## Goal

Add one bounded Stage 2 reduction/accumulation submodule on the corrected
generic typed `tcrv_rvv` body surface. The positive route starts with a single
sum/add reduction instance as an ordinary typed-body case:

```text
tcrv.exec selected RVV variant
  -> explicit tcrv_rvv.runtime_abi_value imports
  -> typed tcrv_rvv vector/config/control/dataflow body
  -> typed reduction/accumulation structure
  -> RVV plugin-owned legality / selected-body realization
  -> RVV provider-derived TCRVEmitCLowerableRoute
  -> neutral common EmitC / target artifact mechanics
```

The module goal is route-supported generic reduction structure. Executable
`ssh rvv` correctness evidence is optional in this round and must be reported
only if actually collected.

## Current Facts

- Git was clean before task creation and HEAD was
  `7f9bfa0 rvv: close generic stage2 executable artifacts`.
- No `.trellis/.current-task` existed before this task was created.
- Archived Stage 1 work retired legacy `RVVI32M1*`, `rvv-i32m1-*`, finite
  `tcrv_rvv.i32_*`, source-front-door, and descriptor route authority.
- Archived Stage 2 route coverage completed generic typed arithmetic,
  RHS-broadcast memory form, and compare/select predicate dataflow without
  adding reductions.
- Archived executable artifact closure proved the existing generic typed
  arithmetic, RHS-broadcast add, and compare/select selected-body routes can
  generate bundles and pass real `ssh rvv` correctness runs.
- Long-term specs require RVV operation, dtype, SEW, LMUL, policy, memory
  form, runtime ABI use, and intrinsic/materialization mapping to be structural
  in typed body/config/runtime facts or derived by the RVV plugin.
- Common EmitC/export must remain neutral and must not infer RVV semantics from
  route ids, artifact names, ABI strings, fixture strings, status fields, or
  mirror metadata.

## Requirements

1. Extend the generic typed RVV body surface so a selected body can carry a
   bounded reduction/accumulation structure with:
   - typed vector input;
   - explicit reduction kind, initially add/sum only;
   - accumulator or scalar/result role;
   - VL, policy, SEW, LMUL, and runtime ABI facts already present in the body.
2. Keep dtype/config/operation authority structural in typed values, config,
   body ops, runtime ABI imports, and capability facts.
3. Update RVV dialect verification/parser/printer and supporting contracts so
   supported reduction structure verifies and unsupported malformed forms fail
   closed with targeted diagnostics.
4. Update RVV construction protocol recognition so generic executable role
   structure includes the reduction/accumulation case without treating
   manifests, template names, route ids, or fixture names as executable
   authority.
5. Update the RVV provider so it builds or rejects a route from typed
   body/config/runtime facts. The provider owns the route/type/header/intrinsic
   or loop/materialization derivation.
6. Keep common EmitC/export neutral. Common code may materialize a
   provider-built route payload, but it must not choose reduction semantics,
   dtype, SEW/LMUL, policy, accumulator layout, or intrinsic spelling.
7. Add at least one positive sum/add reduction fixture that reaches
   provider-built EmitC or target artifact dry-run output if feasible in this
   round.
8. Add negative fail-closed coverage for unsupported reduction kinds or stale
   legacy/source-front-door/metadata-only forms.
9. Preserve Stage 1 guardrails: do not add new `tcrv_rvv.i32_*` helpers,
   `!tcrv_rvv.i32m*` positive types, `RVVI32M1*` route tables/slices,
   `rvv-i32m1-*` route authority, source-front-door positive routes,
   descriptor-driven computation, or common/export RVV semantic branches.

## Acceptance Criteria

- Positive lit/FileCheck or C++ coverage proves a selected generic typed RVV
  sum/add reduction body verifies and reaches provider-built route metadata and
  materialized EmitC or target artifact output.
- The RVV provider derives supported reduction route facts from typed body,
  config, policy, VL/runtime ABI, element type, LMUL, memory/result roles, and
  operation kind, or fails closed with a targeted diagnostic.
- Negative coverage proves unsupported reduction forms, stale legacy helper
  paths, metadata-only paths, or source-front-door inputs fail closed before
  target artifact construction.
- Common EmitC/export remains free of new RVV semantic inference or
  family-specific route selection.
- An active-authority scan over `include lib test scripts .trellis/spec`
  classifies remaining `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m*`, `__riscv_*_i32m1`, source-front-door/source-artifact,
  descriptor, and emission-plan matches as negative, deprecated,
  derived-intrinsic leaves, mirrors, or specs.
- Focused verifier/lit/unit checks for changed RVV dialect/plugin/provider,
  EmitC route, and target artifact paths pass.
- `ssh rvv` correctness evidence is reported only if the route is actually
  made executable and run remotely in this round.

## Non-Goals

- No high-level Linalg/Vector/StableHLO frontend reduction lowering.
- No matrix multiply, contraction, broad accumulation families, dtype/LMUL
  clone batches, conversion coverage, global tuning, dashboards, or readiness
  state machines.
- No source-front-door positive RVV route restoration.
- No one-intrinsic wrapper dialect or dtype-prefixed helper namespace growth.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission.
- No RVV runtime, correctness, or performance claim unless generated artifacts
  are actually run on `ssh rvv` and evidenced in this task.

## Validation Plan

1. Validate Trellis task context.
2. Build focused touched targets, expected to include `tcrv-opt`,
   `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, or
   `tianchenrv-target-artifact-export-test` depending on changed files.
3. Run focused lit tests for RVV dialect/provider/EmitC/target artifact
   positives and negatives changed in this task.
4. Run generated artifact dry-run for the positive reduction fixture if the
   target artifact path is reached.
5. Run `git diff --check`.
6. Run an active-authority scan over `include lib test scripts .trellis/spec`.
7. Run broader `check-tianchenrv` only if changes broaden shared
   route/provider/export behavior enough to justify it.

## Implementation Results

- Added generic `tcrv_rvv.reduce` as a typed vector reduction/accumulation
  dataflow op with explicit `kind`, input vector, accumulator vector, VL, and
  same-typed vector result. The bounded supported kind is `add`.
- Added dialect verifier coverage for `tcrv_rvv.reduce`: only `kind` is
  accepted, unsupported kinds fail closed, the op must be directly under the
  matching `tcrv_rvv.with_vl`, input/accumulator/result must be the same
  generic vector type, and VL must be `!tcrv_rvv.vl`.
- Extended RVV construction protocol role recognition and route mapping with
  `reduce_add` using generic `tcrv_rvv.reduce` and a non-legacy
  `rvv-generic-reduce-add-*` route/ABI identity.
- Extended the RVV EmitC route provider so a selected body containing
  `load, load, reduce(add), store` derives the provider-owned reduction leaf
  from typed body/config/runtime facts. RHS broadcast reduction is explicitly
  rejected in this bounded slice.
- Added positive reduction EmitC materialization and target header/artifact
  dry-run fixtures, plus negative verifier/provider cases.
- Did not collect `ssh rvv` correctness evidence and therefore this task makes
  no runtime correctness or performance claim.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-05-19-stage2-generic-rvv-reduction-accumulation-route`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused reduction lit set: `generic-stage2-dataflow.mlir`,
  `rvv-generic-stage2-reduction-materialization.mlir`,
  `rvv-generic-stage2-reduction-negative.mlir`,
  `rvv-generic-stage2-materialization.mlir`, and
  `explicit-selected-body-artifact-reduce-add.mlir` passed 5/5.
- [OK] focused legacy-message regression lit set for existing first-slice
  EmitC materialization fixtures passed 5/5.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 154/154.
- [OK] `git diff --check`
- [OK] active-authority scan over `include lib test scripts .trellis/spec`
  found broad pre-existing/spec/negative/mirror hits. The diff-only scan found
  one new legacy-shaped token: provider-derived
  `__riscv_vredsum_vs_i32m1_i32m1`, classified as an intrinsic leaf after
  generic typed-body validation, not route authority. No new
  `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or
  `tcrv_rvv.i32_macc` helper surface was added.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- focused RVV tests under `test/Conversion/EmitC`, `test/Target/RVV`,
  `test/Dialect/RVV`, and related negative fixture directories discovered by
  inspection.

## Definition Of Done

- One coherent Stage 2 submodule is complete: a generic typed RVV sum/add
  reduction/accumulation route skeleton is represented, validated, and wired
  through the production RVV provider path with focused tests.
- The task context, PRD, implementation notes, and final report truthfully
  distinguish route-supported evidence from executable `ssh rvv` evidence.
- No legacy i32 route authority, descriptor-driven computation, or common
  EmitC/export semantic inference is reintroduced.
- The task is finished/archived and one coherent commit is created if the
  module is complete.
- If only part of the module is completed, the task remains open with a precise
  continuation point and no misleading Stage 2 completion claim.
