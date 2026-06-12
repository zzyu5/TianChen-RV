# Stage2 generic typed RVV elementwise and predicate route coverage

## Goal

Add bounded Stage 2 route-supported RVV coverage on the corrected generic
typed `tcrv_rvv` body surface for:

- broadcast / scalar-load elementwise memory form; and
- compare/select predicate dataflow.

The route authority must remain:

```text
tcrv.exec selected RVV variant
  -> explicit tcrv_rvv.runtime_abi_value imports
  -> typed tcrv_rvv vector/config/control/dataflow body
  -> RVV plugin-owned legality / selected-body realization
  -> RVV provider-derived TCRVEmitCLowerableRoute
  -> neutral common EmitC / target artifact mechanics
```

This task starts from the archived Stage 1 result after commit `03535f3`,
where legacy `RVVI32M1*`, `rvv-i32m1-*`, finite `tcrv_rvv.i32_*`,
source-front-door, and descriptor route authority were retired or
fail-closed. Stage 2 coverage may reintroduce broadcast and compare/select
only through generic typed `tcrv_rvv` body structure.

## Current Facts

- Git was clean before task creation and HEAD was `03535f3 rvv: retire legacy
  stage1 route authority`.
- No `.trellis/.current-task` existed before this task was created.
- The Stage 1 archived PRD says positive arithmetic route support now uses
  generic typed `tcrv_rvv.load` -> `tcrv_rvv.binary {kind}` ->
  `tcrv_rvv.store` bodies.
- Stage 1 intentionally did not add broadcast, compare/select, reductions,
  conversions, dtype/LMUL clone batches, high-level frontends, or
  source-front-door positive routes.
- Long-term specs require RVV operation, dtype, SEW, LMUL, policy, memory
  form, runtime ABI use, and intrinsic/type/header mapping to be structural in
  typed body/config or derived by the RVV plugin. Common EmitC/export must
  remain neutral.

## Requirements

1. Extend the generic typed RVV body surface, verifier/parser/printer, or
   supporting contracts as needed so selected RVV bodies can structurally carry
   broadcast/scalar-load memory form and compare/select predicate dataflow.
2. Keep dtype/config authority structural in typed values, config, body ops,
   runtime ABI imports, and capability facts. Do not infer route support from
   route ids, artifact names, ABI strings, test names, exact intrinsic
   spellings, status fields, or mirrors.
3. Update RVV plugin/provider logic so route/type/header/intrinsic choices for
   the new supported forms are validated or derived from generic typed body
   facts. Unsupported forms must fail closed with targeted diagnostics.
4. Use RVV plugin-local selected-body realization only to materialize legal
   generic `tcrv_rvv` structure. It must not change compute semantics,
   parameter roles, dispatch/fallback behavior, variant origin, or runtime AVL.
5. Keep common EmitC/export route materialization neutral. It may consume
   provider-built payloads, but must not choose RVV semantics, dtype, memory
   form, mask policy, select behavior, or intrinsic names itself.
6. Migrate directly related fail-closed broadcast and compare/select fixtures
   into positive coverage only if they use generic typed `tcrv_rvv` bodies.
   Keep stale legacy/source-front-door fixtures negative or deprecated.
7. Preserve Stage 1 guardrails: do not add new `tcrv_rvv.i32_*` helpers,
   `!tcrv_rvv.i32m*` positive types, `RVVI32M1*` route tables/slices,
   `rvv-i32m1-*` route authority, source-front-door positive routes, or
   descriptor-driven computation.

## Acceptance Criteria

- Positive lit/FileCheck or C++ coverage proves selected RVV broadcast or
  scalar-load memory-form bodies reach provider-built route metadata and
  materialized target/EmitC output through generic typed `tcrv_rvv` structure.
- Positive lit/FileCheck or C++ coverage proves selected RVV compare/select
  predicate dataflow reaches provider-built route metadata and materialized
  target/EmitC output through generic typed `tcrv_rvv` structure.
- Negative coverage proves unsupported broadcast/scalar forms, compare/select
  predicate mismatches, stale legacy RVV helper paths, or source-front-door
  inputs fail closed before target artifact construction.
- Provider validation checks typed body-derived operation kind, element type,
  SEW, LMUL, policy, VL/runtime facts, memory form, predicate/select facts,
  and intrinsic/type/header leaves.
- Common EmitC/export remains free of new RVV semantic inference or
  family-specific route selection.
- An active-authority scan over `include lib test .trellis/spec` classifies
  remaining `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m*`, `__riscv_*_i32m1`, source-front-door/source-artifact,
  and emission-plan matches as negative, deprecated, intrinsic leaves derived
  after generic validation, mirrors, or specs.
- Focused verifier/lit/unit checks for changed RVV dialect/plugin/provider,
  EmitC route, and target artifact paths pass.

## Non-Goals

- No reductions, contractions, conversions, dtype clone batches, LMUL clone
  batches, global tuning, dashboards, readiness state machines, or report-only
  progress.
- No high-level Linalg/Vector/StableHLO frontend lowering in this task.
- No source-front-door positive RVV route restoration.
- No one-intrinsic wrapper dialect or dtype-prefixed helper namespace growth.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.
- No RVV runtime, correctness, or performance claim unless a generated
  artifact is actually run on `ssh rvv` and evidenced in this task.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission.

## Validation Plan

- Validate Trellis task context.
- Build focused touched targets, expected to include `tcrv-opt`,
  `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, or
  `tianchenrv-target-artifact-export-test` depending on changed files.
- Run focused lit tests for RVV dialect/provider/EmitC/target artifact
  positives and negatives changed in this task.
- Run `git diff --check`.
- Run an active-authority scan over `include lib test .trellis/spec`.
- Run broader `check-tianchenrv` only if changes broaden fixtures or shared
  route/export behavior enough to justify it.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- focused RVV tests under `test/Conversion/EmitC`, `test/Target/RVV`, and
  related negative fixture directories discovered by inspection.

## Definition Of Done

- One coherent Stage 2 submodule is complete: generic typed RVV
  broadcast/scalar-load and compare/select predicate route support is wired
  through the production RVV provider path with focused tests.
- The task context, PRD, implementation notes, and final report truthfully
  describe the bounded behavior completed.
- The task is finished/archived and one coherent commit is created if the
  module is complete.
- If only part of the module is completed, the task remains open with a precise
  continuation point and no misleading Stage 2 completion claim.
