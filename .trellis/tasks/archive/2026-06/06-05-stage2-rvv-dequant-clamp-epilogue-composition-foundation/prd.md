# Stage2 RVV dequant-clamp epilogue composition foundation

## Goal

Add one bounded production route-supported RVV Stage 2 dequant-clamp epilogue
composition foundation:

```text
selected/pre-realized typed tcrv_rvv body
  -> RVV plugin-local realization
  -> i32 source load
  -> runtime f32 scale dequantization
  -> runtime f32 lower/upper clamp-select
  -> f32 output store/runtime boundary
  -> provider-derived route planning and target artifact validation
```

This task composes the already proven i32-to-f32 runtime-scale dequantization
and f32 runtime-bound clamp/select surfaces in one low-level typed RVV selected
body. It is route-supported and target-validation work, not another executable
ABI evidence round.

## What I Already Know

* The session began in `/home/kingdom/phdworks/TianchenRV` on `main` with a
  clean worktree at `b40cc3cb rvv: prove f32 clamp select executable abi`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
* `.trellis/spec/index.md` keeps the RVV-first authority chain at selected
  `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV plugin-owned
  realization/provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target
  artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires selected-body
  realization to run before route planning and requires route support to derive
  from typed/config/runtime facts rather than route ids, artifact names, ABI
  strings, mirrors, descriptors, scripts, or legacy i32 helper names.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires Common EmitC to
  remain neutral and target validation to consume provider-derived facts and
  reject stale mirrors.
* The archived i32-to-f32 dequant route foundation added typed
  `tcrv_rvv.dequantize`, provider-owned dequantization facts, operand binding,
  statement planning, and target validation for `lhs,scale,out,n`.
* The archived f32 clamp/select route foundation added the pre-realized f32
  runtime-bound clamp/select body, selected-body realization, provider-owned
  compare/select facts, operand binding, and target validation for
  `input,lower_bound,upper_bound,out,n`.
* Current code search shows `RVVSelectedBodyRouteSlice` can already carry a
  dequantize op, lower/upper bound splats, compare/select ops, and store, but
  route operation/memory-form/facts currently treat dequantization and f32
  clamp/select as separate route kinds.

## Requirements

* Add a bounded production composition surface. The selected typed or
  pre-realized body must structurally carry:
  * i32 source or accumulator/input value role;
  * runtime f32 scale role `dequant-scale-value`;
  * f32 dequant result type/config;
  * runtime f32 lower and upper bound roles;
  * lower-before-upper order and clamp-lower-then-upper select layout;
  * SEW, LMUL, tail/mask policy, AVL/VL, memory form, and f32 output boundary.
* RVV plugin-local realization must materialize dequantization followed by
  lower/upper clamp-select without changing computation semantics, dtype
  semantics, parameter roles, dispatch/fallback behavior, or runtime n/AVL.
* Provider route planning must validate the cross-op dataflow boundary:
  `i32 load -> dequantize -> f32 lower clamp -> f32 upper clamp -> f32 store`.
* Provider diagnostics must fail closed for missing scale, missing lower or
  upper bound, ambiguous/swapped bound roles, dtype-chain mismatch, stale
  mirrors, unsupported config/policy, and route-string or artifact-name
  authority.
* Target artifact validation must mirror only provider-derived composition
  facts after rebuilding and validating the route.
* Common EmitC/export must not infer RVV compute, dtype, scale semantics, bound
  semantics, schedule, policy, or intrinsic spelling.

## Acceptance Criteria

* [x] Production C++/MLIR/TableGen code changes are made; this is not a
      harness-only, report-only, metadata-only, or standalone evidence task.
* [x] A typed pre-realized composition body or equivalent selected-body surface
      accepts only the bounded i32-to-f32 runtime-scale dequant plus f32
      runtime-bound clamp/select chain when all structural facts are present.
* [x] RVV selected-body realization emits a realized typed body containing the
      canonical dequant then clamp/select sequence before provider planning.
* [x] Provider route planning builds a provider-owned route/fact surface for
      the composition and derives headers, type mapping, operand binding,
      runtime ABI order, statement plan, and mirror facts from typed body facts.
* [x] Target route-family validation rejects stale/missing composition mirrors
      and stale dequant/clamp sub-facts before accepting a generated bundle.
* [x] Focused positive lit and/or C++ tests prove route-supported composition
      behavior through provider route construction and target artifact export.
* [x] Focused negative tests cover missing scale, missing lower bound, missing
      upper bound, swapped/ambiguous bound roles, dtype-chain mismatch,
      unsupported config/policy, stale metadata, and route-string/artifact-name
      authority.
* [x] Run `tianchenrv-rvv-extension-plugin-test` for provider/plugin changes and
      `tianchenrv-target-artifact-export-test` for target validation changes.
* [x] Include a generated artifact or lit check for the route-supported
      composition fixture. Do not claim executable correctness unless real
      `ssh rvv` evidence is run and recorded.
* [x] Run a bounded old-authority and q-name-authority scan over touched files.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis metadata is truthful, the task is finished/archived when complete,
      one coherent commit is created, and final worktree status is clean.

## Definition Of Done

* One bounded dequant-clamp epilogue composition is route-supported by the
  production RVV typed-body, realization, provider, statement-plan, and target
  validation path.
* Unsupported, stale, or metadata-derived forms fail closed before Common
  EmitC/export can authorize the route.
* The final report states route-supported versus executable capability. This
  task defaults to route-supported only unless real `ssh rvv` evidence is added.

## Out Of Scope

* q8/q4/llama benchmark routes.
* Zero-point or additional clamp fusion beyond i32 scale-to-f32 then f32
  lower/upper clamp/select.
* Contraction/matmul fusion in this round.
* High-level Linalg, Vector, StableHLO, or frontend lowering.
* dtype/LMUL clone batches.
* One-intrinsic wrapper dialects.
* Compatibility wrappers preserving legacy i32m1 authority.
* Broad smoke matrices, dashboards, report-only work, or repeated standalone
  dequant/clamp executable evidence as the primary deliverable.

## Technical Notes

Specs and prior task context read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/variant-pipeline/index.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-i32-to-f32-dequant-route-foundation/prd.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-i32-to-f32-dequant-executable-abi-closure/prd.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-f32-clamp-select-route-foundation/prd.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-f32-clamp-select-executable-abi-closure/prd.md`

Initial production/test surfaces to inspect or change:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Support/RuntimeABI.h`
* `lib/Support/RuntimeABIContract.cpp`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* Focused RVV dialect and target tests under `test/Dialect/RVV` and
  `test/Target/RVV`.
