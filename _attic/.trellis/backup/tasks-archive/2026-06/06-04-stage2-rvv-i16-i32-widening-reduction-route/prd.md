# Stage2 RVV i16-to-i32 widening reduction route foundation

## Goal

Implement one production route-supported RVV low-precision reduction
foundation: selected typed `tcrv_rvv` i16 vector reduction input with an i32
accumulator/result boundary, lowered through an RVV plugin-owned
`vwredsum`-style widening reduction route. The route must be derived from typed
body/config/runtime facts, validated by target artifact logic, and covered by
focused positive and fail-closed tests. This is not a q8/q4/llama-specific
route and not an evidence-recording task.

## What I Already Know

- The repository had no current Trellis task at session start; this task was
  created from the Hermes Direction Brief.
- The previous completed task added route-supported signed i8 unit loads plus
  i8 x i8 widening product to an i16 result, then explicitly stopped before
  i16-to-i32 widening reduction and dequant/full contraction closure.
- Current RVV mainline is selected typed `tcrv_rvv` body -> RVV plugin
  legality/realization/provider route -> common EmitC materializer -> target
  artifact.
- Specs require dtype, SEW, LMUL, policy, runtime AVL/VL, operation kind, and
  accumulator/result facts to be structural typed body/config/runtime facts,
  not inferred from route ids, artifact names, test names, metadata mirrors,
  descriptors, q names, or legacy i32 helper authority.
- Existing nearby surfaces include RVV config contracts, contraction and
  standalone-reduction route planning, statement-plan owners, provider route
  construction, and target artifact route-family validation.

## Scope

- Add or repair the production RVV route-supported slice for standalone
  i16-to-i32 widening reduction.
- Keep the surface generic and typed. If a new op/surface is required, it must
  be a typed vector-level RVV operation, not benchmark-named and not a
  dtype-prefixed helper family.
- Derive provider route facts from typed body/config/runtime facts:
  source element type i16, source SEW, source LMUL, tail/mask policy, runtime
  AVL/VL, reduction operation kind, i32 accumulator/result element type,
  accumulator/result layout, reduction intrinsic facts, and result boundary.
- Validate target artifact mirrors only after provider route construction and
  fail closed on stale or mismatched source/result dtype, accumulator layout,
  reduction-store VL, or metadata mirrors.

## Requirements

- Production code changes are required before task closure.
- Typed body/config validation accepts the selected widening reduction only
  when source element type, result/accumulator type, source/result SEW, LMUL,
  policy, runtime AVL/VL, reduction operation kind, accumulator/result layout,
  and result boundary are structurally present.
- RVV plugin route planning derives C types, vector types, required headers,
  accumulator/result layout, reduction intrinsic facts, route operand binding,
  and metadata mirrors from typed body/config/runtime facts.
- Common EmitC/export may carry provider-built payloads but must not choose RVV
  dtype, vector type, intrinsic spelling, reduction semantics, or accumulator
  layout.
- Target validation rejects stale or mismatched source/result dtype,
  accumulator layout, reduction-store VL, provider fact summary, route payload,
  or candidate metadata mirror.
- Focused positive and negative tests prove route-supported behavior and
  fail-closed unsupported or stale cases.

## Acceptance Criteria

- [x] A production RVV dialect/plugin/target route or validation surface is
  changed to support the bounded typed i16-to-i32 widening reduction route
  foundation.
- [x] Positive focused test proves route-supported behavior for i16 vector
  reduction input with i32 accumulator/result boundary.
- [x] Negative focused tests prove unsupported source/result dtype,
  accumulator/result layout, reduction-store VL, or stale provider/metadata
  mirrors fail closed where those paths exist.
- [x] `tianchenrv-rvv-extension-plugin-test` passes if provider/plugin code
  changes.
- [x] `tianchenrv-target-artifact-export-test` passes if target validation code
  changes.
- [x] Focused lit/unit tests for the changed route pass.
- [x] `git diff --check` passes.
- [x] Bounded scan over touched files finds no new q8/q4/llama/name authority
  or legacy i32m1 positive route authority.
- [x] Worktree is clean after the final commit.

## Definition of Done

- The task status, PRD, and notes truthfully reflect route-supported versus
  executable/runtime claims.
- No `ssh rvv` evidence is claimed unless generated artifacts are executed on
  the RVV host.
- A coherent commit records the production and test changes if the task is
  complete.

## Out Of Scope

- q8/q4/llama benchmark-specific route names, authority, or harnesses.
- Dequantization or full q-like contraction closure unless the standalone
  widening reduction foundation is already complete and the extension is
  demonstrably small.
- Runtime/correctness/performance claims without real `ssh rvv` evidence.
- High-level Linalg/frontend generalization.
- Descriptor-driven computation, source-front-door positive routes, or
  compatibility wrappers preserving legacy i32m1 authority.
- Broad smoke matrices, dashboards, report-only tasks, or Trellis archive-only
  updates.

## Technical Approach

Implement the smallest production route-supported standalone i16-to-i32
widening reduction submodule that fits the existing RVV provider structure:

- Reuse the corrected typed RVV route surface and existing standalone
  reduction/contraction planning patterns where possible.
- Add only the generic typed operation/config facts needed for i16 source and
  i32 accumulator/result validation.
- Extend RVV provider-owned route facts and statement/route planning so
  headers, C/vector types, `vwredsum`-style intrinsic facts, runtime AVL/VL
  boundary, operand bindings, and mirrors are derived by the provider.
- Extend target route-family validation to compare provider route facts and
  candidate metadata mirrors fail-closed.
- Add focused lit and C++ tests for the positive route and representative
  stale/mismatched negative cases.

## Decision (ADR-lite)

Context: The previous low-precision route task stopped after i8 widening
product to i16, leaving the reduction/result-dtype half as the next bottleneck.

Decision: Implement the standalone i16-to-i32 widening reduction foundation
first, rather than trying to close a full product-plus-reduction/dequant chain
in the same round.

Consequences: This gives the RVV plugin a real low-precision reduction/result
boundary while keeping the task bounded. Full q-like contraction closure remains
explicit continuation work.

## Technical Notes

- Specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task reference:
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-low-precision-widening-product-route/`.
- First code inspection targets:
  `test/Dialect/RVV/generic-widening-product-dataflow.mlir`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  and existing reduction/widening tests under `test/Dialect/RVV` and
  `test/Target/RVV`.
