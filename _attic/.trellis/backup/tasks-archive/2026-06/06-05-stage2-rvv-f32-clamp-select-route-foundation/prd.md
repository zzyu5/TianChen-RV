# Stage2 RVV f32 Clamp/Select Route Foundation

## Goal

Add one bounded production route-supported RVV Stage 2 f32 clamp/select slice:

```text
selected tcrv.exec RVV variant
  -> typed f32 tcrv_rvv body with runtime lower/upper bounds
  -> RVV plugin-owned compare/select or min/max realization and route planning
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact validation mirrors provider-derived facts only
```

The owner is the adjacent compare/select class needed after f32 dequant output.
This is not another contraction-dequant evidence task and not q8/q4/llama or
high-level activation/frontend lowering work.

## What I Already Know

* Commit `17fd2765` closed executable `ssh rvv` evidence for the pre-realized
  contraction-dequant path. That module is complete for this handoff.
* `.trellis/spec/index.md` and `.trellis/spec/extension-plugins/rvv-plugin.md`
  require the RVV path to start from selected typed `tcrv_rvv` body structure,
  not route ids, artifact names, ABI strings, descriptors, metadata mirrors, or
  legacy i32 helper names.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines compare/select target
  validation through `RVVCompareSelectRouteValidationContract`, including an
  embedded `RVVRuntimeAVLVLSelectedBoundaryContract` that target validation
  consumes before route payload, predicate/select facts, statement plans, or
  artifact mirrors.
* Existing compare/select implementation surfaces include
  `RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `RVVEmitCControlPolicyPlanOwners.cpp`,
  `RVVEmitCStatementPlanOwners.cpp`, `RVVEmitCRoutePlanning.cpp`,
  `RVVEmitCRouteProvider.cpp`, and target route-family validation.
* Runtime lower and upper clamp bounds must be imported and consumed as runtime
  scalar facts in the selected typed body/provider path; their roles and order
  cannot be inferred from names, fixture paths, route strings, or artifact
  metadata.

## Requirements

* Implement a production route-supported f32 clamp/select foundation on the
  corrected typed RVV surface.
* The selected typed body must structurally carry:
  * f32 vector input/result types;
  * runtime f32 lower and upper bound values;
  * unambiguous lower-vs-upper bound order and roles;
  * SEW, LMUL, tail/mask policy, AVL/VL, and memory roles;
  * f32 output store/runtime boundary.
* RVV plugin/provider code must derive compare/select or min/max statement
  facts from typed body/config/runtime facts and fail closed for missing,
  mismatched, stale, or unsupported facts.
* Target artifact validation may mirror provider-derived facts only after
  rebuilding and validating the provider route. Mirror fields must not become
  route authority.
* Common EmitC/export must remain neutral and must not choose RVV compute,
  dtype, bound semantics, schedule, policy, or intrinsic spelling.
* If executable closure becomes too large, stop at a coherent route-supported
  plus target-validation module and leave `ssh rvv` correctness as an explicit
  continuation point.

## Acceptance Criteria

* [ ] Production C++/MLIR/TableGen code changes are made; this is not a
      harness-only, report-only, or metadata-only task.
* [ ] A positive focused dialect/plugin/target fixture proves typed f32
      clamp/select route support reaches provider route construction and
      common EmitC/target artifact validation.
* [ ] Provider validation rejects missing lower bound, missing upper bound,
      ambiguous or inverted bound roles/order, dtype mismatch, unsupported
      SEW/LMUL/policy/config, stale mirrors, and route-string/artifact-name
      authority.
* [ ] Target artifact validation mirrors only provider-derived f32 clamp/select
      facts and rejects stale metadata, stale runtime AVL/VL facts, stale
      compare/select predicate/layout facts, and stale operand binding/header
      summaries.
* [ ] Focused positive and negative tests cover the changed provider and target
      behavior.
* [ ] Run `tianchenrv-rvv-extension-plugin-test` when provider/plugin C++
      changes and `tianchenrv-target-artifact-export-test` when target artifact
      validation changes.
* [ ] Include a generated artifact or lit check for the route-supported
      fixture. Do not claim runtime correctness unless real `ssh rvv` evidence
      is run and recorded.
* [ ] A bounded old-authority and q-name-authority scan over touched files finds
      no positive legacy `i32m1`, `RVVI32M1`, `rvv-i32m1`, descriptor,
      source-front-door, q8/q4, or llama route drift.
* [ ] `git diff --check` and `git diff --cached --check` pass.
* [ ] Trellis metadata is truthful, the task is finished/archived when complete,
      one coherent commit is created, and final worktree status is clean.

## Definition Of Done

* The route-supported f32 clamp/select behavior is implemented in production
  owner code, not only fixtures or evidence scripts.
* Typed body/config/runtime facts remain the authority for dtype, bounds,
  policy, AVL/VL, memory roles, statement planning, and target validation.
* The changed behavior has focused positive and negative tests.
* No executable correctness claim is made without `ssh rvv` evidence.

## Out Of Scope

* q8/q4/llama benchmark routes.
* High-level activation, Linalg, Vector, StableHLO, or frontend lowering.
* Full fused quantized-kernel closure.
* Zero-point expansion.
* Repeating contraction-dequant evidence as the main achievement.
* dtype/LMUL clone batches.
* One-intrinsic wrapper dialects or compatibility wrappers preserving legacy
  i32m1 authority.
* Broad smoke matrices, dashboards, reports-only work, or harness-only tasks.
* Any common EmitC/export semantic inference for RVV compute, dtype, bound
  semantics, schedule, or policy.

## Technical Notes

Specs and prior context read for this PRD:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-pre-realized-contraction-dequant-executable-closure/prd.md`

Initial production/test surfaces to inspect:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* Existing compare/select, dequant, and target RVV tests under
  `test/Dialect/RVV`, `test/Plugin`, and `test/Target/RVV`.
