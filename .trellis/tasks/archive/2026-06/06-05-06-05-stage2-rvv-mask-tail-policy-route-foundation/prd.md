# Stage2 RVV explicit mask/tail-policy route foundation

## Goal

Implement one bounded Stage 2 RVV route-supported foundation where explicit
mask and tail policy facts are carried by the selected typed `tcrv_rvv`
body/config, validated by the RVV plugin, used to derive the EmitC statement
plan, and mirrored only after provider construction into target artifact
validation. The module owner for this round is the existing
elementwise/compare-select route family, with computed-mask select as the
primary mask-producing/consuming subcase.

## What I Already Know

- The repository started clean on `main`; the latest commit is
  `c51bdef4 rvv: close contraction dequant clamp executable abi`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
- The previous archived task closed executable ABI evidence for
  `widening_product_reduce_dequant_clamp_f32`; it is not this round's primary
  deliverable.
- `.trellis/spec/index.md` requires the current RVV authority chain:
  `tcrv.exec` envelope, selected typed low-level `tcrv_rvv` body, RVV
  plugin-owned realization/provider, common EmitC, target artifact, and
  `ssh rvv` only when runtime/correctness/performance is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` already defines a
  compare/select provider preflight requiring realized typed config facts,
  materialization facts, operand-binding facts, and statement-plan mask/tail
  provider facts before `TCRVEmitCLowerableRoute` construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` with explicit `tailPolicy` and
  `maskPolicy`, and requires target validation to consume provider-derived
  runtime AVL/VL facts before accepting mirrors.
- Common EmitC/export must stay neutral and must not choose RVV mask semantics,
  tail behavior, intrinsics, dtype, schedule, or ABI role semantics.

## Requirements

- Use selected typed `tcrv_rvv` body/config/provider facts as authority for
  mask production or consumption, tail policy, mask policy, AVL/VL placement,
  element type, SEW/LMUL, memory roles, and store behavior.
- Reuse the existing elementwise/compare-select route family and statement-plan
  boundary where possible instead of adding a new high-level frontend, one-op
  intrinsic wrapper, q-name route, or compatibility i32 route.
- The RVV provider must validate legal mask/tail policy combinations and fail
  closed for missing, stale, unsupported, or cross-family policy facts before
  building a `TCRVEmitCLowerableRoute`.
- Target artifact validation must reject stale policy mirrors and must compare
  candidate metadata only against provider-derived facts.
- Focused positive fixtures must prove route-supported explicit mask/tail
  policy flow through dialect/plugin/provider/target validation.
- Focused negative coverage must prove fail-closed behavior for missing mask
  role when required, unsupported mask/tail policy combinations, stale
  provider policy mirrors, dtype/config mismatch, missing AVL/VL facts, and
  route-string/artifact-name/intrinsic-spelling authority.

## Acceptance Criteria

- [x] Production C++ changes implement a route-supported explicit mask/tail
      policy foundation for the chosen compare/select computed-mask submodule.
- [x] Typed body/config structurally carries the policy facts; no route id,
      intrinsic spelling, artifact name, ABI string, FileCheck label, status,
      or harness expectation is used as route authority.
- [x] Provider-side validation derives or verifies the EmitC statement plan and
      rejects stale/missing mask/tail policy facts before route construction.
- [x] Target artifact validation rejects stale candidate policy mirrors before
      accepting the bundle.
- [x] Focused positive and negative tests cover the route-supported fixture and
      fail-closed diagnostics.
- [x] `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test` run if provider or target C++
      code changes.
- [x] A generated artifact or lit/FileCheck route-supported check covers the
      positive fixture.
- [x] `ssh rvv` evidence is not required unless this round makes a runtime,
      correctness, or performance claim.
- [x] Bounded scans over touched files show no new legacy i32, q-name,
      route-id, artifact-name, ABI-string, descriptor, source-front-door, or
      intrinsic-spelling authority.
- [x] `git diff --check` and `git diff --cached --check` pass before commit.

## Technical Approach

Start from current compare/select and computed-mask select route planning:

1. Inspect `RVVConfigContract`, dialect verification, compare/select statement
   plan owners, route planning/provider preflight, target route-family
   validation, and existing positive/negative fixtures.
2. Find the smallest production boundary where mask/tail policy facts already
   exist or should be surfaced, then make that boundary explicit and
   provider-derived.
3. Add or strengthen provider preflight and target validation so stale mirrors
   fail before route construction or artifact acceptance.
4. Add focused tests for one route-supported positive fixture and the requested
   negative diagnostics.

## Decision (ADR-lite)

Context: The brief asks for a Stage 2 explicit mask/tail-policy route
foundation, not executable evidence or new route coverage unrelated to policy.

Decision: Use the existing elementwise/compare-select route family, centered on
computed-mask select, as the bounded MVP because it already has mask-producing
and mask-consuming structure plus a spec-defined provider preflight.

Consequences: This keeps the implementation inside RVV plugin/provider/target
ownership and avoids moving semantics into common EmitC. It may leave broader
memory, reduction, contraction, and runtime-scalar policy promotion for later
Hermes rounds.

## Out Of Scope

- q8/q4/llama benchmark routes.
- ProviderSpec/model-name route authority.
- New contraction/dequant/clamp executable evidence as the primary result.
- High-level Linalg, Vector, StableHLO, or source-front-door frontend work.
- dtype/LMUL clone batches.
- One-intrinsic wrapper dialects.
- Compatibility wrappers preserving legacy i32m1 authority.
- Broad smoke matrices, dashboards, or report-only work.
- Common EmitC/export semantic selection for RVV masks or tail behavior.

## Technical Notes

- Direction brief read first list:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  archived task
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-contraction-dequant-clamp-executable-abi-closure/`,
  `test/Target/RVV` compare/select and clamp/dequant fixtures,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Memory-derived historical guardrail: do not overfit Stage 2 work to q8/q4
  examples, and do not move mask/tail realization into EmitC, route strings,
  artifacts, dashboards, or status metadata.

## Evidence Recorded

- Production provider preflight now verifies computed-mask select
  `RVVSelectedBodyMaskTailPolicyProviderPlan` mirrors for plan id, owner,
  tail policy, mask policy, route operand binding plan, selected target
  provider mirror, and selected target legality mirror before creating
  `TCRVEmitCLowerableRoute`.
- Target artifact candidate validation now rejects stale
  `tcrv_rvv.tail_policy` and `tcrv_rvv.mask_policy` mirrors for compare/select
  mask routes before accepting the artifact.
- Dialect negative coverage now rejects unsupported computed-mask select
  `#tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>` before route
  construction.
- `tianchenrv-rvv-extension-plugin-test` passed after rebuilding from
  `build/`.
- `tianchenrv-target-artifact-export-test` passed after rebuilding from
  `build/`.
- Focused lit filter passed from `build/test`:
  `computed-mask-select-dataflow|pre-realized-selected-body-artifact-computed-mask-select`.
- `git diff --check` passed.
- Bounded new-diff authority scan passed with no matches for legacy i32,
  q-name, descriptor, source-front-door, route-id, artifact-name, ABI-string,
  or intrinsic-spelling authority.
- `ssh rvv` was not applicable because this task claims route-supported
  compiler/target validation, not executable runtime correctness or
  performance.

## Spec Update Decision

No `.trellis/spec/` update was needed. The existing RVV plugin spec already
requires compare/select provider preflight to reject stale mask/tail policy
provider facts before route construction, and the existing EmitC route spec
already requires target validation to reject stale provider mirrors. This round
implements that existing contract rather than introducing a new API or
architecture rule.
